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
            "dream64-map-expression-overrides-{}-{ordinal}",
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
fn evaluates_dynamic_map_overrides_before_new_with_global_and_src_bindings() {
    let types = concat!(
        "var/global/offset = 3\n",
        "/obj/boot\n\tvar/base = 4\n\tvar/value = 0\n\tvar/seen_by_new = 0\n",
        "/obj/boot/New()\n\tsrc.seen_by_new = src.value\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = concat!(
        "\"a\" = (/obj/boot{value = global.offset + src.base}, /turf/boot, /area/boot)\n",
        "(1,1,1) = {\"\na\n\"}\n",
    );
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
    .expect("dynamic map overrides should evaluate before lifecycle execution");

    let object = allocation
        .coordinate(dm_world::WorldCoordinate { x: 1, y: 1, z: 1 })
        .expect("map coordinate should be allocated")
        .movables
        .first()
        .copied()
        .expect("object should be allocated");
    assert_eq!(
        runtime.heap().datum_field(object, &field("value")),
        Ok(&Value::number(7.0)),
        "the override must bind global and src fields"
    );
    assert_eq!(
        runtime.heap().datum_field(object, &field("seen_by_new")),
        Ok(&Value::number(7.0)),
        "New() must observe the evaluated map override"
    );
}
