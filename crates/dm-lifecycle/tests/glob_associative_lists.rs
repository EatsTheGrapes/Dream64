use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{LifecycleIndex, build_initialization_plan, execute_initialization_plan};
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
            "dream64-glob-associative-lists-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project directory should be created");
        fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        fs::write(root.join("types.dm"), types).expect("types should be written");
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
fn lifecycle_procedures_read_and_write_glob_associative_lists_with_inherited_bare_keys() {
    let types = concat!(
        "/datum/controller/global_vars\n\tvar/global/list/loading_bays = list()\n",
        "var/global/datum/controller/global_vars/GLOB = new /datum/controller/global_vars\n",
        "/obj/loading_bay\n\tvar/loading_id = \"cargo\"\n\tvar/observed = 0\n",
        "/obj/loading_bay/active/New()\n",
        "\tGLOB.loading_bays[loading_id] = 41\n",
        "\tobserved = GLOB.loading_bays[loading_id] + 1\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/obj/loading_bay/active, /turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (_project, compilation, map_source) = TestProject::compile(types, map);
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let map = parse(&map_source).expect("map should parse");
    let world = build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world, "boot.dmm");
    let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

    execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .expect("GLOB associative access should execute during lifecycle initialization");

    let object = allocation
        .coordinate(dm_world::WorldCoordinate { x: 1, y: 1, z: 1 })
        .expect("map coordinate should be allocated")
        .movables
        .first()
        .copied()
        .expect("object should be allocated");
    assert_eq!(
        runtime.heap().datum_field(object, &field("observed")),
        Ok(&Value::number(42.0)),
        "the inherited bare loading_id must select the associative GLOB entry"
    );

    let Value::List(loading_bays) = runtime
        .variables()
        .iter()
        .find(|variable| variable.path.ends_with("/loading_bays"))
        .expect("GLOB field should remain materialized")
        .value
    else {
        panic!("GLOB.loading_bays should be a list");
    };
    assert_eq!(
        runtime
            .heap()
            .list(loading_bays)
            .expect("global list should remain live")
            .get_key(&Value::text("cargo")),
        Ok(&Value::number(41.0)),
        "the lifecycle write must persist in the global associative list"
    );
}
