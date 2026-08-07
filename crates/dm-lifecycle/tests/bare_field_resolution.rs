use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_lifecycle::{
    InitializationExecutionError, LifecycleIndex, build_initialization_plan,
    execute_initialization_plan,
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
            "dream64-bare-field-resolution-{}-{ordinal}",
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

fn boot(types: &str, map: &str) -> (dm_world::WorldAllocation, RuntimeImage) {
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
    .expect("bare fields should execute during lifecycle initialization");
    (allocation, runtime)
}

fn movable_at(allocation: &dm_world::WorldAllocation, x: i32) -> dm_value::DatumId {
    allocation
        .coordinate(dm_world::WorldCoordinate { x, y: 1, z: 1 })
        .expect("map coordinate should be allocated")
        .movables
        .first()
        .copied()
        .expect("object should be allocated")
}

#[test]
fn lifecycle_procedures_read_and_write_inherited_fields_without_src() {
    let types = concat!(
        "/obj/base\n\tvar/inherited = 3\n\tvar/seen = 0\n",
        "/obj/base/child/New()\n\tseen = inherited\n\tinherited += 2\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = "\"a\" = (/obj/base/child, /turf/boot, /area/boot)\n(1,1,1) = {\"\na\n\"}\n";
    let (allocation, runtime) = boot(types, map);
    let datum = movable_at(&allocation, 1);

    assert_eq!(
        runtime.heap().datum_field(datum, &field("seen")),
        Ok(&Value::number(3.0))
    );
    assert_eq!(
        runtime.heap().datum_field(datum, &field("inherited")),
        Ok(&Value::number(5.0))
    );
}

#[test]
fn parameters_and_locals_shadow_same_named_instance_fields() {
    let types = concat!(
        "/obj/parameter\n\tvar/value = 3\n\tvar/seen = 0\n",
        "/obj/parameter/New(value = 8)\n\tseen = value\n\tvalue += 1\n",
        "/obj/local\n\tvar/value = 3\n\tvar/seen = 0\n",
        "/obj/local/New()\n\tvar/value = 5\n\tvalue += 1\n\tseen = value\n",
        "/turf/boot\n/area/boot\n",
    );
    let map = concat!(
        "\"a\" = (/obj/parameter, /turf/boot, /area/boot)\n",
        "\"b\" = (/obj/local, /turf/boot, /area/boot)\n",
        "(1,1,1) = {\"\nab\n\"}\n",
    );
    let (allocation, runtime) = boot(types, map);
    let parameter = movable_at(&allocation, 1);
    let local = movable_at(&allocation, 2);

    assert_eq!(
        runtime.heap().datum_field(parameter, &field("value")),
        Ok(&Value::number(3.0))
    );
    assert_eq!(
        runtime.heap().datum_field(parameter, &field("seen")),
        Ok(&Value::number(8.0))
    );
    assert_eq!(
        runtime.heap().datum_field(local, &field("value")),
        Ok(&Value::number(3.0))
    );
    assert_eq!(
        runtime.heap().datum_field(local, &field("seen")),
        Ok(&Value::number(6.0))
    );
}

#[test]
fn unknown_bare_names_remain_compile_diagnostics() {
    let types = concat!(
        "/obj/broken/New()\n\tmissing += 1\n",
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
    .expect_err("an undeclared bare name must not bind to src");
    let InitializationExecutionError::Compile(error) = error else {
        panic!("expected a lifecycle compilation diagnostic");
    };
    assert!(
    error.message.contains("unknown local \"missing\""),
    "unexpected diagnostic: {}",
    error.message
);
}
