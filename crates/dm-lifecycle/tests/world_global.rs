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
    let map = parse("\"a\" = (/obj/test, /turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
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

#[test]
fn compiled_map_traits_initialize_typed_z_level_tables_from_world_maxz() {
    let project = TestProject::new(concat!(
        "#define TRUE 1\n",
        "#define DL_NAME \"name\"\n",
        "#define DL_TRAITS \"traits\"\n",
        "#define ZTRAITS_CENTCOM list(\"CentCom\" = TRUE)\n",
        "#define DECLARE_LEVEL(NAME, TRAITS) list(DL_NAME = NAME, DL_TRAITS = TRAITS)\n",
        "#define DEFAULT_MAP_TRAITS list(DECLARE_LEVEL(\"CentCom\", ZTRAITS_CENTCOM))\n",
        "var/global/z_probe = 0\n",
        "/datum/space_level\n\tvar/z_value\n\tNew(z)\n\t\tz_value = z\n",
        "/datum/mapping\n",
        "\tvar/list/datum/space_level/z_list\n",
        "\tvar/list/gravity_by_z_level = list()\n",
        "\tvar/list/z_level_to_stack = list()\n",
        "\tproc/initialize_default_z_levels()\n",
        "\t\tif(z_list)\n\t\t\treturn\n",
        "\t\tz_list = list()\n",
        "\t\tvar/list/default_map_traits = DEFAULT_MAP_TRAITS\n",
        "\t\tif(default_map_traits.len > world.maxz)\n\t\t\tdefault_map_traits.Cut(world.maxz + 1)\n",
        "\t\tfor(var/i in 1 to default_map_traits.len)\n",
        "\t\t\tvar/datum/space_level/level = new(i)\n",
        "\t\t\tz_list += level\n",
        "\t\t\tgravity_by_z_level.len += 1\n",
        "\t\t\tz_level_to_stack.len += 1\n",
        "\t\t\tz_level_to_stack[i] = list(i)\n",
        "/world\n",
        "\tNew()\n",
        "\t\tvar/datum/mapping/mapping = new\n",
        "\t\tmapping.initialize_default_z_levels()\n",
        "\t\tz_probe = world.maxz * 1000 + mapping.z_list.len * 100 + mapping.gravity_by_z_level.len * 10 + mapping.z_level_to_stack.len\n",
        "/turf/test\n/area/test\n",
    ));
    let compilation = CompilerDatabase::new()
        .compile(project.root.join("world.dme"))
        .expect("fixture should compile");
    let procedures = ProcedureRegistry::build(&compilation);
    let mut runtime = RuntimeImage::from_compilation(&compilation).expect("runtime should build");
    let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
    let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
        .expect("map should parse");
    let world_plan = build_plan(&map, &compilation);
    let plan = build_initialization_plan(&runtime, &index, &world_plan, "boot.dmm");
    let allocation = allocate_world(&world_plan, &mut runtime).expect("world should allocate");
    execute_initialization_plan(
        &compilation,
        &procedures,
        &index,
        &plan,
        &allocation,
        &mut runtime,
    )
    .expect("mapping-shaped world initialization should execute");

    assert_eq!(
        runtime
            .variable("/var/z_probe")
            .map(|variable| &variable.value),
        Some(&Value::number(1111.0)),
        "the compiled-in z level must populate every mapping lookup"
    );
}
