use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::CompilerDatabase;
use dm_lifecycle::{LifecycleIndex, build_initialization_plan, execute_initialization_plan};
use dm_map::parse;
use dm_runtime::RuntimeImage;
use dm_semantics::ProcedureRegistry;
use dm_value::{FieldName, Value};
use dm_world::{WorldCoordinate, allocate_world, build_plan};

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

struct TestProject {
    root: PathBuf,
}

impl TestProject {
    fn new(types: &str) -> Self {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-world-global-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project directory should be created");
        fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        fs::write(root.join("types.dm"), types).expect("types should be written");
        Self { root }
    }
}

impl Drop for TestProject {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn built_in_world_global_resolves_to_the_singleton_world_during_atom_lifecycle() {
    let project = TestProject::new(concat!(
        "/world\n",
        "/obj/test\n\tvar/seen_world = 0\n",
        "/obj/test/New()\n\tif(world)\n\t\tseen_world = 1\n",
        "/turf/test\n",
        "/area/test\n",
    ));
    let compilation = CompilerDatabase::new()
        .compile(project.root.join("world.dme"))
        .expect("fixture should compile");
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let map = parse(
        "\"a\" = (/obj/test, /turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n",
    )
    .expect("map should parse");
    let world_plan = build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world_plan, "boot.dmm");
    let allocation = allocate_world(&world_plan, &mut runtime).expect("world should allocate");

    let execution = execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .expect("world global should be available during lifecycle execution");

    assert!(
        execution.world.is_some(),
        "the /world singleton should be allocated"
    );
    let object = allocation
        .coordinate(WorldCoordinate { x: 1, y: 1, z: 1 })
        .expect("map coordinate should exist")
        .movables[0];
    let seen_world = FieldName::parse("seen_world").expect("field name should be valid");
    assert_eq!(
        runtime.heap().datum_field(object, &seen_world),
        Ok(&Value::number(1.0))
    );
}
