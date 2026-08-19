use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{
    EventSubject, HeadlessReadinessProbe, InitializationEvent, InitializationExecutionError,
    LifecycleIndex, LifecycleKind, SchedulerDrainLimits, SchedulerDrainTermination,
    build_initialization_plan, execute_initialization_plan,
    execute_initialization_plan_with_scheduler_limits,
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
    let world_new = execution
        .events
        .iter()
        .position(|event| {
            matches!(
                event.event,
                InitializationEvent::Lifecycle {
                    subject: EventSubject::World,
                    kind: LifecycleKind::New,
                    ..
                }
            )
        })
        .expect("world New should execute");
    assert!(execution.events[..world_new].iter().all(|event| matches!(
        event.event,
        InitializationEvent::Lifecycle {
            subject: EventSubject::MapAtom(_),
            kind: LifecycleKind::New,
            ..
        }
    )));

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

#[test]
fn headless_boot_drains_spawned_startup_work_to_stable_idle() {
    let types = concat!(
        "/world/New()\n",
        "\tsrc.stage = 1\n",
        "\tspawn(3)\n",
        "\t\tsrc.stage = 9\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(&map_source).expect("map should parse"), &compilation);
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
    .expect("scheduled startup should drain");

    assert_eq!(
        execution.scheduler.termination,
        SchedulerDrainTermination::StableIdle
    );
    assert_eq!(execution.scheduler.final_tick, 3);
    assert_eq!(execution.scheduler.rounds, 1);
    assert_eq!(execution.scheduler.completed_tasks, 1);
    assert_eq!(execution.scheduler.pending_tasks, 0);
    assert_eq!(
        runtime
            .heap()
            .datum_field(execution.world.unwrap(), &field("stage")),
        Ok(&Value::number(9.0))
    );
}

#[test]
fn headless_boot_continues_on_startup_scheduler_failure() {
    let types = concat!(
        "var/global/trace = 0\n",
        "/proc/fail_startup()\n",
        "\tCRASH(\"isolated\")\n",
        "/proc/finish_startup()\n",
        "\tglobal.trace = 7\n",
        "/world/New()\n",
        "\tspawn(1)\n",
        "\t\tfail_startup()\n",
        "\tspawn(2)\n",
        "\t\tfinish_startup()\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(&map_source).expect("map should parse"), &compilation);
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
    .expect("startup task failure should not abort boot");

    assert_eq!(execution.scheduler.failed_tasks, 1);
    assert_eq!(
        execution.scheduler.termination,
        SchedulerDrainTermination::StableIdle
    );
    assert_eq!(execution.scheduler.pending_tasks, 0);
    assert_eq!(execution.scheduler.completed_tasks, 1);
    let trace = runtime
        .variables()
        .iter()
        .find(|variable| variable.path.ends_with("/trace"))
        .unwrap_or_else(|| panic!("trace global should exist"))
        .value
        .clone();
    assert_eq!(trace, Value::number(7.0));
}

#[test]
fn headless_boot_reports_pending_work_at_scheduler_tick_limit() {
    let types = concat!(
        "/world/New()\n",
        "\tspawn(10)\n",
        "\t\tsrc.stage = 9\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(&map_source).expect("map should parse"), &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

    let execution = execute_initialization_plan_with_scheduler_limits(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
        SchedulerDrainLimits {
            max_ticks: 2,
            max_rounds: 10,
        },
    )
    .expect("bounded scheduler drain should be a successful partial boot");

    assert_eq!(
        execution.scheduler.termination,
        SchedulerDrainTermination::TickLimit
    );
    assert_eq!(execution.scheduler.final_tick, 0);
    assert_eq!(execution.scheduler.rounds, 0);
    assert_eq!(execution.scheduler.completed_tasks, 0);
    assert_eq!(execution.scheduler.pending_tasks, 1);
}

#[test]
fn headless_boot_reports_codebase_readiness_with_persistent_work() {
    let types = concat!(
        "var/global/startup_ready = 0\n",
        "/proc/heartbeat()\n",
        "\tspawn(1)\n",
        "\t\theartbeat()\n",
        "/world/New()\n",
        "\tspawn(0)\n",
        "\t\theartbeat()\n",
        "\tspawn(2)\n",
        "\t\tstartup_ready = 1\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(&map_source).expect("map should parse"), &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
    let probe = HeadlessReadinessProbe {
        qualified_storage: None,
        global: field("startup_ready"),
        fields: Vec::new(),
        expected: Value::number(1.0),
    };

    let execution = dm_lifecycle::execute_initialization_plan_with_scheduler_policy(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
        SchedulerDrainLimits {
            max_ticks: 20,
            max_rounds: 20,
        },
        Some(&probe),
    )
    .expect("the explicit readiness marker should complete persistent startup");

    assert_eq!(
        execution.scheduler.termination,
        SchedulerDrainTermination::HeadlessReady
    );
    assert_eq!(execution.scheduler.final_tick, 2);
    assert!(execution.scheduler.pending_tasks > 0);
}

#[test]
fn genesis_persists_logger_and_scheduled_state_into_world_new() {
    let types = concat!(
        "var/global/datum/logger/logger = null\n",
        "var/global/seen = 0\n",
        "var/global/genesis_scheduled = 0\n",
        "/datum/logger/proc/mark()\n\treturn 9\n",
        "/world/Genesis()\n",
        "\tglobal.logger = new /datum/logger\n",
        "\tspawn(2)\n",
        "\t\tglobal.genesis_scheduled = 1\n",
        "/world/New()\n\tglobal.seen = call(global.logger, \"mark\")()\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let world = build_plan(&parse(&map_source).expect("map should parse"), &compilation);
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
    .expect("Genesis and world/New should share runtime state");

    assert!(matches!(
        execution.events[0].event,
        InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::Genesis,
            ..
        }
    ));
    assert!(matches!(
        execution.events[1].event,
        InitializationEvent::Lifecycle {
            subject: EventSubject::World,
            kind: LifecycleKind::New,
            ..
        }
    ));
    let global = |name: &str| {
        runtime
            .variables()
            .iter()
            .find(|variable| variable.path.ends_with(&format!("/{name}")))
            .unwrap_or_else(|| panic!("global {name} should exist"))
            .value
            .clone()
    };
    assert!(matches!(global("logger"), Value::Datum(_)));
    assert_eq!(global("seen"), Value::number(9.0));
    assert_eq!(global("genesis_scheduled"), Value::number(1.0));
    assert_eq!(execution.scheduler.final_tick, 2);
    assert_eq!(
        execution.scheduler.termination,
        SchedulerDrainTermination::StableIdle
    );
}
