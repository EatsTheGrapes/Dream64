use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{
    EventSubject, InitializationEvent, InitializationExecutionError, LifecycleIndex, LifecycleKind,
    build_initialization_plan, execute_initialization_plan,
};
use dm_map::parse;
use dm_runtime::RuntimeImage;
use dm_semantics::ProcedureRegistry;
use dm_value::{FieldName, Value};
use dm_world::{allocate_world, build_plan};

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

struct TestProject {
    root: PathBuf,
}

impl TestProject {
    fn compile(types: &str, map: &str) -> (Self, Compilation, String) {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-headless-boot-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project directory should be created");
        fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        fs::write(root.join("types.dm"), types).expect("types should be written");
        fs::write(root.join("boot.dmm"), map).expect("map should be written");
        let compilation = CompilerDatabase::new()
            .compile(root.join("world.dme"))
            .expect("fixture should compile");
        (Self { root }, compilation, map.to_owned())
    }
}

impl Drop for TestProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn field(name: &str) -> FieldName {
    FieldName::parse(name).expect("fixture field should be valid")
}

#[test]
fn headless_boot_executes_deterministic_map_lifecycle_and_preserves_state() {
    let types = concat!(
        "/world/New()\n\tsrc.stage = 7\n",
        "/atom/proc/New()\n\tsrc.stage = 10\n",
        "/atom/proc/Initialize()\n\tsrc.stage += 1\n",
        "/atom/proc/LateInitialize()\n\tsrc.stage += 100\n",
        "/area/boot\n/turf/boot\n/obj/boot\n",
    );
    let map = concat!(
        "\"a\" = (/obj/boot, /turf/boot, /area/boot)\n",
        "(1,1,1) = {\"\naa\n\"}\n",
    );
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let map = parse(&map_source).expect("map should parse");
    let world = build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

    let execution = execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .expect("headless boot should execute");

    assert_eq!(execution.events.len(), 16);
    assert_eq!(
        execution.duplicate_map_events, 3,
        "shared area runs once per phase"
    );
    let phases: Vec<_> = execution
        .events
        .iter()
        .map(|event| match event.event {
            InitializationEvent::Lifecycle { kind, .. } => kind,
            InitializationEvent::Globals => panic!("globals are not lifecycle hooks"),
        })
        .collect();
    assert_eq!(
        phases,
        [
            LifecycleKind::New,
            LifecycleKind::New,
            LifecycleKind::New,
            LifecycleKind::New,
            LifecycleKind::New,
            LifecycleKind::New,
            LifecycleKind::Initialize,
            LifecycleKind::Initialize,
            LifecycleKind::Initialize,
            LifecycleKind::Initialize,
            LifecycleKind::Initialize,
            LifecycleKind::LateInitialize,
            LifecycleKind::LateInitialize,
            LifecycleKind::LateInitialize,
            LifecycleKind::LateInitialize,
            LifecycleKind::LateInitialize,
        ]
    );
    assert!(matches!(
        execution.events[0].event,
        InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::New,
            ..
        }
    ));

    let world_id = execution.world.expect("world datum should be allocated");
    assert_eq!(
        runtime.heap().datum_field(world_id, &field("stage")),
        Ok(&Value::number(7.0))
    );
    for datum in allocation.allocation_order() {
        assert_eq!(
            runtime.heap().datum_field(*datum, &field("stage")),
            Ok(&Value::number(111.0))
        );
    }
}

#[test]
fn headless_boot_keeps_runtime_errors_source_mapped() {
    let types = concat!(
        "/obj/broken/New()\n\treturn \"text\" + 1\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/obj/broken, /turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let map = parse(&map_source).expect("map should parse");
    let world = build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

    let error = execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .expect_err("invalid arithmetic should fail at runtime");
    let InitializationExecutionError::Runtime { target, error, .. } = error else {
        panic!("expected a source-aware runtime failure");
    };
    assert_eq!(target.procedure_path, "/obj/broken/proc/New");
    assert!(
        error
            .message
            .contains("addition requires compatible DM values")
    );
    assert!(error.source_span.is_some());
    assert_eq!(error.call_stack.len(), 1);
    assert!(
        error.call_stack[0]
            .procedure
            .starts_with("/obj/broken/proc/New@")
    );
    assert!(error.call_stack[0].source_span.is_some());
}
