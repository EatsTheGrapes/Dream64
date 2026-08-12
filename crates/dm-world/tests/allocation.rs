use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use dm_compiler::{Compilation, CompilerDatabase};
use dm_globals::UnsupportedCategory;
use dm_map::parse;
use dm_runtime::RuntimeImage;
use dm_value::{FieldName, Value};
use dm_world::{
    WorldAllocationWorkKind, WorldCoordinate, allocate_world, build_plan,
    materialize_world_map_state,
};

static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

struct TestProject {
    root: PathBuf,
}

impl TestProject {
    fn compile(source: &str) -> (Self, Compilation) {
        let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "dream64-world-allocation-{}-{ordinal}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("test project directory should be created");
        fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
            .expect("environment should be written");
        fs::write(root.join("types.dm"), source).expect("types should be written");
        let project = Self { root };
        let compilation = CompilerDatabase::new()
            .compile(project.root.join("world.dme"))
            .expect("test project should compile");
        (project, compilation)
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
fn allocates_defaults_and_constants_but_defers_dynamic_map_values() {
    let (_project, compilation) = TestProject::compile(concat!(
        "/area/test\n\tvar/name = \"base\"\n",
        "/turf/test\n\tvar/density = 0\n",
        "/obj/test\n\tvar/value = 1\n\tvar/runtime = 9\n",
        "\tNew()\n\t\tvalue = 100\n",
    ));
    let map_source = concat!(
        "\"a\" = (/obj/test{value = 2 + 3; runtime = build_value()}, /turf/test{density = 1}, /area/test{name = \"shared\"})\n",
        "\"b\" = (/turf/test, /area/test{name = \"shared\"})\n",
        "(3,4,1) = {\"\nab\n\"}\n",
    );
    let map = parse(map_source).expect("map should parse");
    let plan = build_plan(&map, &compilation);
    let mut image = RuntimeImage::from_compilation(&compilation).expect("image should materialize");
    let allocation = allocate_world(&plan, &mut image).expect("world should allocate");
    let first = allocation
        .coordinate(WorldCoordinate { x: 3, y: 4, z: 1 })
        .expect("first coordinate should exist");
    let second = allocation
        .coordinate(WorldCoordinate { x: 4, y: 4, z: 1 })
        .expect("second coordinate should exist");

    assert_eq!(
        first.area, second.area,
        "equivalent area initializers share"
    );
    assert_ne!(first.turf, second.turf, "each coordinate owns a turf");
    assert_eq!(first.source_order.len(), 3);
    assert_eq!(first.movables.len(), 1);
    let object = image
        .heap()
        .datum(first.movables[0])
        .expect("object should be live");
    assert_eq!(
        object.field(&field("value")).unwrap().as_number(),
        Some(5.0)
    );
    assert_eq!(
        object.field(&field("runtime")).unwrap().as_number(),
        Some(9.0),
        "deferred override leaves the inherited default and New() is not called"
    );
    assert_eq!(object.field(&field("x")).unwrap().as_number(), Some(3.0));
    assert_eq!(object.field(&field("y")).unwrap().as_number(), Some(4.0));
    assert_eq!(object.field(&field("z")).unwrap().as_number(), Some(1.0));
    let area = image
        .heap()
        .datum(first.area.expect("area should exist"))
        .expect("area should be live");
    assert_eq!(area.field(&field("name")), Ok(&Value::text("shared")));
    let first_turf = image
        .heap()
        .datum(first.turf.expect("turf should exist"))
        .expect("turf should be live");
    assert_eq!(
        first_turf.field(&field("density")).unwrap().as_number(),
        Some(1.0)
    );
    assert_eq!(
        first_turf.field(&field("loc")),
        Ok(&Value::Datum(first.area.expect("area should exist"))),
        "a map turf's loc is its effective area before lifecycle execution"
    );
    assert_eq!(
        object.field(&field("loc")),
        Ok(&Value::Datum(first.turf.expect("turf should exist"))),
        "a map movable's loc is its coordinate turf before lifecycle execution"
    );

    assert_eq!(allocation.stats().cells, 2);
    assert_eq!(allocation.stats().datums_allocated, 4);
    assert_eq!(allocation.stats().unique_areas, 1);
    assert_eq!(allocation.stats().turfs, 2);
    assert_eq!(allocation.stats().movables, 1);
    assert_eq!(allocation.stats().constant_overrides, 3);
    assert_eq!(allocation.stats().unsupported_overrides, 1);
    assert_eq!(
        allocation.stats().execution_state_transfers,
        1,
        "bulk world allocation must reuse one VM execution state"
    );
    assert_eq!(image.stats().stateful_datums_allocated, 4);
    assert_eq!(allocation.work_items().len(), 1);
    assert_eq!(
        allocation.work_items()[0].kind,
        WorldAllocationWorkKind::DynamicOverride(UnsupportedCategory::Call)
    );
    let blocker = allocation.work_items()[0]
        .blocker_span
        .expect("dynamic override should have a blocker span");
    assert_eq!(&map_source[blocker.start..blocker.end], "build_value");
}

#[test]
fn different_ordered_area_overrides_create_distinct_instances() {
    let (_project, compilation) =
        TestProject::compile("/area/test\n\tvar/name = \"base\"\n/turf/test\n");
    let map = parse(concat!(
        "\"a\" = (/turf/test, /area/test{name = \"one\"})\n",
        "\"b\" = (/turf/test, /area/test{name = \"two\"})\n",
        "(1,1,1) = {\"\nab\n\"}\n",
    ))
    .expect("map should parse");
    let plan = build_plan(&map, &compilation);
    let mut image = RuntimeImage::from_compilation(&compilation).expect("image should materialize");
    let allocation = allocate_world(&plan, &mut image).expect("world should allocate");

    assert_ne!(
        allocation.snapshots()[0].area,
        allocation.snapshots()[1].area
    );
    assert_eq!(allocation.stats().unique_areas, 2);
    assert_eq!(allocation.stats().datums_allocated, 4);
    assert!(allocation.work_items().is_empty());
}

#[test]
fn materializes_map_dimensions_and_initial_world_contents() {
    let (_project, compilation) = TestProject::compile(concat!(
        "/world\n\tmaxx = 9\n",
        "/area/test\n/turf/test\n/obj/test\n",
    ));
    let map = parse(concat!(
        "\"a\" = (/obj/test, /turf/test, /area/test)\n",
        "(4,7,2) = {\"\naa\n\"}\n",
    ))
    .expect("map should parse");
    let plan = build_plan(&map, &compilation);
    let mut image = RuntimeImage::from_compilation(&compilation).expect("image should materialize");
    let allocation = allocate_world(&plan, &mut image).expect("world should allocate");
    let world = image
        .allocate_datum(&dm_value::TypePath::parse("/world").unwrap())
        .expect("world datum should allocate");

    materialize_world_map_state(&allocation, &mut image, world)
        .expect("map-derived world fields should materialize");

    assert_eq!(
        image.heap().datum_field(world, &field("maxx")),
        Ok(&Value::number(9.0)),
        "the compile-time lower bound must win over the map's maximum x"
    );
    assert_eq!(
        image.heap().datum_field(world, &field("maxy")),
        Ok(&Value::number(7.0))
    );
    assert_eq!(
        image.heap().datum_field(world, &field("maxz")),
        Ok(&Value::number(2.0))
    );
    let Value::List(contents) = image
        .heap()
        .datum_field(world, &field("contents"))
        .expect("world contents should exist")
    else {
        panic!("world contents should be a live list")
    };
    assert_eq!(
        image.heap().list(*contents).unwrap().len(),
        allocation.allocation_order().len()
    );
    let first = &allocation.snapshots()[0];
    let Value::List(turf_contents) = image
        .heap()
        .datum_field(first.turf.unwrap(), &field("contents"))
        .unwrap()
    else {
        panic!("turf contents should be a list")
    };
    assert_eq!(image.heap().list(*turf_contents).unwrap().len(), 1);
    let Value::List(area_contents) = image
        .heap()
        .datum_field(first.area.unwrap(), &field("contents"))
        .unwrap()
    else {
        panic!("area contents should be a list")
    };
    assert_eq!(
        image.heap().list(*area_contents).unwrap().len(),
        4,
        "a shared area's contents include both turfs and both mapped movables"
    );
}

#[test]
fn large_inert_map_keeps_turf_defaults_sparse_and_dynamic_lists_per_cell() {
    const SIDE: usize = 64;
    const SCALARS: usize = 96;
    let mut source = String::from("/area/scaled\n/turf/scaled\n");
    for index in 0..SCALARS {
        source.push_str(&format!("\tvar/scalar_{index} = {index}\n"));
    }
    source.push_str("\tvar/list/per_cell[2]\n");
    let (_project, compilation) = TestProject::compile(&source);

    let row = "a".repeat(SIDE);
    let rows = std::iter::repeat_n(row, SIDE)
        .collect::<Vec<_>>()
        .join("\n");
    let map = parse(&format!(
        "\"a\" = (/turf/scaled, /area/scaled)\n(1,1,1) = {{\"\n{rows}\n\"}}\n"
    ))
    .expect("scaled map should parse");
    let plan = build_plan(&map, &compilation);
    let mut image = RuntimeImage::from_compilation(&compilation).expect("image should materialize");
    let allocation = allocate_world(&plan, &mut image).expect("scaled world should allocate");

    let turf_count = SIDE * SIDE;
    assert_eq!(allocation.stats().turfs, turf_count);
    let materialized_fields = allocation
        .snapshots()
        .iter()
        .map(|snapshot| {
            image
                .heap()
                .datum(snapshot.turf.expect("every scaled cell has a turf"))
                .unwrap()
                .field_len()
        })
        .sum::<usize>();
    assert!(
        materialized_fields <= turf_count * 6,
        "compact turfs should own only per-cell list/contents and x/y/z/loc; owned {materialized_fields} fields"
    );
    let eager_scalar_fields = turf_count * SCALARS;
    assert!(
        eager_scalar_fields >= materialized_fields * 16,
        "the scaling gate must prove sparse allocation, not a small constant reduction"
    );

    let first = allocation.snapshots()[0].turf.unwrap();
    let second = allocation.snapshots()[1].turf.unwrap();
    assert!(
        image
            .heap()
            .datum_field(first, &field("scalar_95"))
            .is_err(),
        "untouched scalar defaults must not be copied into each turf"
    );
    let first_list = image
        .heap()
        .datum_field(first, &field("per_cell"))
        .expect("first per-cell suffix array");
    let second_list = image
        .heap()
        .datum_field(second, &field("per_cell"))
        .expect("second per-cell suffix array");
    assert_ne!(first_list, second_list, "dynamic lists must not alias");
    for value in [first_list, second_list] {
        let Value::List(list) = value else {
            panic!("per-cell suffix array should be a list");
        };
        assert_eq!(image.heap().list(*list).unwrap().len(), 2);
    }

    let field_slot_bytes = std::mem::size_of::<(FieldName, Value)>();
    eprintln!(
        "compact-map-scale turfs={turf_count} materialized_fields={materialized_fields} field_slot_bytes={field_slot_bytes} compact_field_bytes={} eager_scalar_bytes={}",
        materialized_fields * field_slot_bytes,
        eager_scalar_fields * field_slot_bytes,
    );
}
