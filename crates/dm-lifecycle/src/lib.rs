//! Deterministic, non-executing lifecycle resolution and initialization plans.

#![cfg_attr(not(test), deny(missing_docs))]

/// Versioned, self-validating storage for compiled Dream64 payloads.
pub mod artifact;
/// Deterministic, non-executing initialization planning.
pub mod initialization_plan;
/// Precompilation pipeline for lifecycle bytecode.
pub mod precompile;

pub use precompile::{
    PrecompiledLifecycle, precompile_lifecycle_for_world,
    precompile_lifecycle_for_world_with_executable, precompile_portable_lifecycle_for_world,
};
/// Lifecycle execution, datum construction/deletion, and compatibility sweeping.
pub mod execute;
/// Loopback-only client IPC with scheduler-boundary command application.
pub mod ipc;
/// Lifecycle resolution and indexing.
pub mod lifecycle_index;
/// Artifact-time DMM catalog products.
pub mod map_catalog;
/// Procedure semantic-identity directory.
pub mod procedure_semantics;
/// Headless boot readiness probe.
pub mod readiness;
/// Deterministic scheduler draining for post-initialization and persistent
/// host slices.
pub mod scheduler;

pub use initialization_plan::{
    EventSubject, GlobalInitialization, InitializationEvent, InitializationPlan,
    MapPlacementContext, PlannedAtom, build_initialization_plan,
};

pub use lifecycle_index::{
    LifecycleCompatibilityIssue, LifecycleCompatibilityLocation, LifecycleCompatibilitySweep,
    LifecycleDiagnostic, LifecycleDiagnosticKind, LifecycleIndex, LifecycleKind,
    LifecycleResolution, LifecycleSource, LifecycleTarget, LifecycleTargetIssue,
    LifecycleTargetIssueKind, LifecycleTargets, TypeLifecycle,
};

pub use map_catalog::{
    PortableDmmGrid, PortableDmmMeasurement, PortableParsedDmm, build_dmm_measurements,
    build_parsed_dmm_cache, decode_dmm_measurements, decode_parsed_dmm_cache,
    dmm_measurements_from_parsed, encode_dmm_measurements, encode_parsed_dmm_cache,
    measure_dmm_source,
};

pub use readiness::{HeadlessReadinessProbe, derive_lobby_readiness, readiness_probe_matches};

pub use scheduler::{
    HostSliceBudget, SchedulerDrain, SchedulerDrainLimits, SchedulerDrainTermination,
    advance_persistent_scheduler, advance_persistent_scheduler_responsive,
};

pub use execute::{
    ConstructionError, DeletionError, ExecutedLifecycleEvent, InitializationExecution,
    InitializationExecutionError, audit_initialization_plan_with_precompiled, construct_datum,
    delete_datum, execute_boot_initialization_plan_with_precompiled,
    execute_boot_initialization_plan_with_precompiled_and_startup_service,
    execute_initialization_plan, execute_initialization_plan_with_precompiled,
    execute_initialization_plan_with_scheduler_limits,
    execute_initialization_plan_with_scheduler_policy, sweep_lifecycle_compatibility,
    sweep_lifecycle_compatibility_with_closures,
};

use dm_vm::Module;

const PROCEDURE_SEMANTICS_MAGIC: &[u8; 8] = b"D64PSEM\0";
const PROCEDURE_SEMANTICS_VERSION: u16 = 1;
const MAX_PROCEDURE_SEMANTICS_BYTES: u64 = 256 * 1024 * 1024;

/// Builds a portable semantic-identity directory for every eager procedure.
pub fn encode_procedure_semantics(module: &Module) -> Result<Vec<u8>, String> {
    if module.deferred_procedure_count() != 0 || module.procedure_count() > 1_000_000 {
        return Err(
            "procedure semantic directory requires a bounded fully eager module".to_owned(),
        );
    }
    let mut payload = Vec::new();
    payload.extend_from_slice(&(module.procedure_count() as u32).to_le_bytes());
    let digests = module.compute_all_procedure_semantic_digests()?;
    for (path, digest) in module.procedure_paths().zip(digests) {
        if path.len() > 64 * 1024 * 1024 {
            return Err("procedure semantic path exceeds its limit".to_owned());
        }
        payload.extend_from_slice(&(path.len() as u32).to_le_bytes());
        payload.extend_from_slice(path.as_bytes());
        payload.extend_from_slice(&digest);
    }
    if payload.len() as u64 > MAX_PROCEDURE_SEMANTICS_BYTES {
        return Err("procedure semantic directory exceeds its limit".to_owned());
    }
    let mut encoded = Vec::with_capacity(22 + payload.len());
    encoded.extend_from_slice(PROCEDURE_SEMANTICS_MAGIC);
    encoded.extend_from_slice(&PROCEDURE_SEMANTICS_VERSION.to_le_bytes());
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

/// Validates and attaches an artifact-emitted semantic directory to a module.
pub fn decode_and_attach_procedure_semantics(
    bytes: &[u8],
    module: &mut Module,
) -> Result<(), String> {
    if bytes.len() < 22 || &bytes[..8] != PROCEDURE_SEMANTICS_MAGIC {
        return Err("invalid procedure semantic directory header".to_owned());
    }
    if u16::from_le_bytes([bytes[8], bytes[9]]) != PROCEDURE_SEMANTICS_VERSION {
        return Err("unsupported procedure semantic directory version".to_owned());
    }
    let length = u64::from_le_bytes(bytes[10..18].try_into().unwrap());
    if length > MAX_PROCEDURE_SEMANTICS_BYTES || length as usize != bytes.len() - 22 {
        return Err("invalid procedure semantic directory length".to_owned());
    }
    let payload = &bytes[22..];
    if crc32fast::hash(payload) != u32::from_le_bytes(bytes[18..22].try_into().unwrap()) {
        return Err("procedure semantic directory checksum mismatch".to_owned());
    }
    let mut cursor = 0usize;
    let take = |cursor: &mut usize, count: usize| -> Result<&[u8], String> {
        let end = cursor
            .checked_add(count)
            .ok_or("procedure semantic offset overflow")?;
        let value = payload
            .get(*cursor..end)
            .ok_or("truncated procedure semantic directory")?;
        *cursor = end;
        Ok(value)
    };
    let count = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
    if count != module.procedure_count() || count > 1_000_000 {
        return Err("procedure semantic count does not match module".to_owned());
    }
    let expected_paths = module.procedure_paths().collect::<Vec<_>>();
    let mut digests = Vec::with_capacity(count);
    for expected in expected_paths {
        let path_len = u32::from_le_bytes(take(&mut cursor, 4)?.try_into().unwrap()) as usize;
        let path = std::str::from_utf8(take(&mut cursor, path_len)?)
            .map_err(|_| "procedure semantic path is not UTF-8")?;
        if path != expected {
            return Err("procedure semantic path table does not match module".to_owned());
        }
        digests.push(take(&mut cursor, 32)?.try_into().unwrap());
    }
    if cursor != payload.len() {
        return Err("trailing procedure semantic directory bytes".to_owned());
    }
    module.attach_procedure_semantic_digests(digests)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_map::parse;
    use dm_runtime::RuntimeImage;
    use dm_semantics::{ExecutableProcedures, ProcedureRegistry};
    use dm_syntax::parse as parse_dm;
    use dm_value::{FieldName, TypePath, Value};
    use dm_vm::{
        ExecutionContext, ExecutionState, compile_module, execute_module_in_context,
        execute_module_in_state,
    };
    use dm_world::{WorldCoordinate, allocate_world, build_plan};

    use super::{
        EventSubject, HeadlessReadinessProbe, HostSliceBudget, InitializationEvent,
        InitializationExecutionError, LifecycleIndex, LifecycleKind, LifecycleResolution,
        PortableDmmMeasurement, SchedulerDrainLimits, SchedulerDrainTermination,
        advance_persistent_scheduler, audit_initialization_plan_with_precompiled,
        build_dmm_measurements, build_initialization_plan, build_parsed_dmm_cache, construct_datum,
        decode_and_attach_procedure_semantics, decode_dmm_measurements, decode_parsed_dmm_cache,
        delete_datum, encode_dmm_measurements, encode_parsed_dmm_cache, encode_procedure_semantics,
        execute_boot_initialization_plan_with_precompiled,
        execute_boot_initialization_plan_with_precompiled_and_startup_service,
        execute_initialization_plan, execute_initialization_plan_with_precompiled,
        measure_dmm_source, precompile_lifecycle_for_world, sweep_lifecycle_compatibility,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn procedure_semantic_directory_is_stable_and_rejects_body_changes_and_corruption() {
        let build = |source: &str| compile_module(&parse_dm(source).unwrap().definitions).unwrap();
        let first = build("/proc/example(value)\n\treturn value + 1\n");
        let equivalent = build("/proc/example(value)\n\n\treturn value + 1\n");
        let mut changed = build("/proc/example(value)\n\treturn value + 2\n");
        let first_digest = first.compute_all_procedure_semantic_digests().unwrap()[0];
        assert_eq!(
            first_digest,
            equivalent.compute_all_procedure_semantic_digests().unwrap()[0]
        );
        assert_ne!(
            first_digest,
            changed.compute_all_procedure_semantic_digests().unwrap()[0]
        );

        let encoded = encode_procedure_semantics(&first).unwrap();
        let mut restored = first.clone();
        decode_and_attach_procedure_semantics(&encoded, &mut restored).unwrap();
        let procedure = restored.procedure_id("/proc/example").unwrap();
        assert_eq!(
            restored.procedure_semantic_digest(procedure),
            Some(first_digest)
        );
        assert!(decode_and_attach_procedure_semantics(&encoded, &mut changed).is_err());
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode_and_attach_procedure_semantics(&corrupt, &mut restored).is_err());
    }

    #[test]
    fn portable_dmm_measurements_match_dmm_and_tgm_bounds() {
        let dmm = "\"a\" = (/turf)\n(3,5,2) = {\"\naa\naa\n\"}\n";
        let tgm = "\"aa\" = (\n/turf,\n/area)\n(7,9,4) = {\"\naaaa\naaaa\naaaa\n\"}\n";
        assert_eq!(measure_dmm_source(dmm).unwrap().bounds, [3, 5, 2, 4, 6, 2]);
        assert_eq!(measure_dmm_source(tgm).unwrap().bounds, [7, 9, 4, 8, 11, 4]);

        let mut catalog = BTreeMap::new();
        catalog.insert(
            "_maps/example.dmm".to_owned(),
            PortableDmmMeasurement {
                digest: md5::compute(dmm).0,
                measurement: measure_dmm_source(dmm).unwrap(),
            },
        );
        let encoded = encode_dmm_measurements(&catalog).unwrap();
        assert_eq!(decode_dmm_measurements(&encoded).unwrap(), catalog);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode_dmm_measurements(&corrupt).is_err());
    }

    #[test]
    fn dmm_measurement_discovery_includes_unincluded_nested_resources() {
        let (fixture, compilation) = Fixture::compile("/proc/run()\n\treturn 1\n");
        let nested = fixture.0.join("_maps").join("nested");
        fs::create_dir_all(&nested).unwrap();
        fs::write(
            nested.join("Unincluded.DMM"),
            "\"a\" = (/turf)\n(2,3,4) = {\"\naa\naa\n\"}\n",
        )
        .unwrap();
        let measurements = build_dmm_measurements(&compilation).unwrap();
        assert_eq!(
            measurements
                .get("_maps/nested/unincluded.dmm")
                .unwrap()
                .measurement
                .bounds,
            [2, 3, 4, 3, 4, 4]
        );
        let parsed = build_parsed_dmm_cache(&compilation).unwrap();
        let entry = parsed.get("_maps/nested/unincluded.dmm").unwrap();
        assert!(!entry.tgm);
        assert_eq!(entry.models, vec![("a".to_owned(), "/turf".to_owned())]);
        assert_eq!(entry.grids[0].lines, vec!["aa", "aa"]);
        let encoded = encode_parsed_dmm_cache(&parsed).unwrap();
        assert_eq!(decode_parsed_dmm_cache(&encoded).unwrap(), parsed);
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(decode_parsed_dmm_cache(&corrupt).is_err());
    }

    #[test]
    fn parsed_tgm_grid_y_is_the_top_source_row() {
        let (fixture, compilation) = Fixture::compile("/proc/run()\n\treturn 1\n");
        let maps = fixture.0.join("_maps");
        fs::create_dir_all(&maps).unwrap();
        fs::write(
            maps.join("column.dmm"),
            "\"aa\" = (\n/turf,\n/area)\n(7,9,4) = {\"\naaaa\naaaa\naaaa\n\"}\n",
        )
        .unwrap();

        let parsed = build_parsed_dmm_cache(&compilation).unwrap();
        let entry = parsed.get("_maps/column.dmm").unwrap();
        assert!(entry.tgm);
        assert_eq!(entry.grids[0].lines, vec!["aaaa", "aaaa", "aaaa"]);
        assert_eq!(entry.grids[0].y, 11, "reader.dm advances y by len - 1");
        assert_eq!(entry.bounds, [7, 9, 4, 8, 11, 4]);
    }

    struct Fixture(PathBuf);

    impl Fixture {
        fn compile(source: &str) -> (Self, Compilation) {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-lifecycle-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("fixture directory should be created");
            fs::write(root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(root.join("types.dm"), source).expect("source should be written");
            let compilation = CompilerDatabase::new()
                .compile(root.join("world.dme"))
                .expect("fixture should compile");
            (Self(root), compilation)
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("fixture should be removed");
        }
    }

    /// Drives the tgstation timsort port through the whole frontend + semantics
    /// + lowering + VM + `RuntimeImage` pipeline (real `#define fetchElement`,
    /// `CREATE_SORT_INSTANCE`, `call(cmp)()`, a reused `GLOB.sortInstance`, and
    /// interleaved associative sorts) and asserts the sorted list keeps its
    /// length and order for every size class.
    #[test]
    fn timsort_full_pipeline_holds_bounds() {
        let source = include_str!("../../../fixtures/runtime/timsort_repro/repro.dm");
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let executable = procedures
            .compile_vm(&compilation)
            .expect("timsort fixture should lower");
        let resolve = |name: &str| {
            let target = procedures
                .procedures()
                .iter()
                .find(|procedure| procedure.path.to_string() == name)
                .and_then(|procedure| procedure.effective_target)
                .unwrap_or_else(|| panic!("{name} should have an effective implementation"));
            executable
                .implementation(target)
                .unwrap_or_else(|| panic!("{name} should be in the VM module"))
        };
        let run_repro = resolve("/proc/run_repro");
        let assoc_then_plain = resolve("/proc/run_repro_assoc_then_plain");
        let repeated = resolve("/proc/run_repro_repeated");
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("runtime image should materialize defaults");
        let mut run = |entry, args: &[Value], label: String| {
            let mut state = runtime.take_execution_state();
            let got = execute_module_in_state(executable.module(), entry, args, &mut state);
            eprintln!("{label} => {got:?}");
            assert_eq!(got, Ok(Value::text("OK")), "{label}");
            runtime.restore_execution_state(state);
        };
        for mode in [0.0, 1.0, 2.0] {
            for count in [8.0, 33.0, 40.0, 64.0, 100.0] {
                run(
                    run_repro,
                    &[Value::number(count), Value::number(mode)],
                    format!("run_repro mode={mode} count={count}"),
                );
            }
        }
        for count in [33.0, 40.0, 64.0, 100.0] {
            run(
                assoc_then_plain,
                &[Value::number(count)],
                format!("assoc_then_plain count={count}"),
            );
            run(
                repeated,
                &[Value::number(count)],
                format!("repeated count={count}"),
            );
        }

        // Inherited instance-var list initializer feeds the merge path (from-compilation).
        run(
            resolve("/proc/run_repro_inherited"),
            &[],
            "run_repro_inherited (from_compilation)".to_owned(),
        );
    }

    #[test]
    fn timsort_merge_path_over_inherited_linked_artifact_list_initializer() {
        // Regression for the reported "DM list position N exceeds length M" fatal
        // inside gallopRight while booting Monkestation: a merge-path-sized list
        // literal declared on a parent type, inherited unchanged by a child type,
        // and sorted from the child's constructor after a `.d64` linked-artifact
        // restore (the boot path for map atoms such as the chem dispensers).
        let source = include_str!("../../../fixtures/runtime/timsort_repro/repro.dm");
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let executable = procedures
            .compile_vm(&compilation)
            .expect("timsort fixture should lower");
        let target = procedures
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == "/proc/run_repro_inherited")
            .and_then(|procedure| procedure.effective_target)
            .expect("run_repro_inherited should have an effective implementation");
        let entry = executable
            .implementation(target)
            .expect("run_repro_inherited should be in the VM module");

        let mut image =
            RuntimeImage::from_compilation(&compilation).expect("image should materialize");
        image
            .materialize_linked_artifact_initializers(8)
            .expect("deferred initializer module should become portable");
        let encoded = image
            .encode_linked_artifact(executable.module())
            .expect("linked artifact should encode");
        let mut restored = RuntimeImage::decode_linked_artifact(&encoded, executable.module())
            .expect("linked artifact should decode");

        // Direct restored allocation: the child must receive the parent's
        // `list(...)` initializer through the restored VM catalog
        // (`instance_initializer_plan` -> `linked_catalog_plan`), the exact path
        // 64f577d repaired. Without it `dispensable` is null and the sort throws.
        let child = TypePath::parse("/obj/dispenser/fullupgrade").unwrap();
        let datum = restored
            .allocate_datum(&child)
            .expect("child should allocate from the linked artifact");
        let dispensable = restored
            .heap()
            .datum_field(datum, &FieldName::parse("dispensable").unwrap())
            .expect("child must expose the inherited dispensable field")
            .clone();
        let Value::List(list) = dispensable else {
            panic!("inherited dispensable must be a list, got {dispensable:?}");
        };
        assert_eq!(
            restored.heap().list(list).unwrap().len(),
            45,
            "the child must inherit the full parent list initializer, not a truncated one"
        );

        // End-to-end: construct the child through the VM and sort the inherited list.
        let mut state = restored.take_execution_state();
        let got = execute_module_in_state(executable.module(), entry, &[], &mut state);
        assert_eq!(
            got,
            Ok(Value::text("OK")),
            "sorting a child-inherited 45-element list initializer must not corrupt the list"
        );
    }

    #[test]
    fn cached_lobby_lifecycle_index_does_not_force_procedure_dependencies() {
        let (_fixture, compilation) = Fixture::compile(concat!(
            "/world/New()\n\treturn ..()\n",
            "/client/New()\n\treturn ..()\n",
            "/datum/example/proc/run()\n\treturn 1\n",
        ));
        let procedures = ProcedureRegistry::build_lazy(&compilation);
        assert!(!procedures.dependencies_initialized());
        let index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        assert!(index.find_path("/world").is_some());
        assert!(!procedures.dependencies_initialized());
    }

    #[test]
    fn portable_lifecycle_directory_roundtrips_without_compiler_node_ids() {
        let (_fixture, compilation) = Fixture::compile(concat!(
            "/world/New()\n\treturn ..()\n",
            "/obj/example\n\tInitialize()\n\t\treturn 7\n",
        ));
        let procedures = ProcedureRegistry::build(&compilation);
        let index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let bytes = index.encode_portable().expect("directory should encode");
        let restored = LifecycleIndex::decode_portable(&bytes).expect("directory should decode");
        assert_eq!(
            restored
                .types()
                .iter()
                .map(|ty| ty.path.as_str())
                .collect::<Vec<_>>(),
            index
                .types()
                .iter()
                .map(|ty| ty.path.as_str())
                .collect::<Vec<_>>()
        );
        let original = index
            .find_path("/obj/example")
            .unwrap()
            .targets
            .get(LifecycleKind::Initialize);
        let decoded = restored
            .find_path("/obj/example")
            .unwrap()
            .targets
            .get(LifecycleKind::Initialize);
        let (LifecycleResolution::Resolved(original), LifecycleResolution::Resolved(decoded)) =
            (original, decoded)
        else {
            panic!("Initialize should remain resolved")
        };
        assert_eq!(decoded.procedure.index(), original.procedure.index());
        assert_eq!(
            decoded.implementation.index(),
            original.implementation.index()
        );
        assert_eq!(decoded.procedure_path, original.procedure_path);
    }

    #[test]
    fn portable_boot_manifest_roundtrips_and_rejects_corruption() {
        let probe = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("SSticker").unwrap(),
            fields: vec![FieldName::parse("current_state").unwrap()],
            expected: Value::number(2.0),
        };
        let bytes = probe.encode_portable_manifest().unwrap();
        assert_eq!(
            HeadlessReadinessProbe::decode_portable_manifest(&bytes).unwrap(),
            probe
        );
        let mut corrupt = bytes;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(HeadlessReadinessProbe::decode_portable_manifest(&corrupt).is_err());
    }

    #[test]
    fn artifact_backed_precompile_does_not_rebuild_procedure_dependencies() {
        let (_fixture, compilation) = Fixture::compile(concat!(
            "/world/New()\n\treturn ..()\n",
            "/area/test\n/turf/test\n/obj/test\n\tNew()\n\t\treturn ..()\n",
        ));
        let eager = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let eager_index = LifecycleIndex::build_compile_only(&compilation, &eager);
        let roots = crate::precompile::lifecycle_targets_for_world(&eager_index, &world);
        let executable = eager
            .compile_vm_all_symbolic_with_eager_roots(&compilation, roots.iter().copied())
            .expect("fixture executable should link");

        let lazy = ProcedureRegistry::build_lazy(&compilation);
        let lazy_index = LifecycleIndex::build_compile_only(&compilation, &lazy);
        assert!(!lazy.dependencies_initialized());
        let precompiled = crate::precompile_lifecycle_for_world_with_executable(
            &compilation,
            &lazy,
            &lazy_index,
            &world,
            executable,
        );
        assert!(!lazy.dependencies_initialized());
        assert_eq!(
            precompiled.reachable_bodies(),
            precompiled.module_procedures()
        );
    }

    fn index(source: &str) -> (Fixture, Compilation, RuntimeImage, LifecycleIndex) {
        let (fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        (fixture, compilation, runtime, index)
    }

    #[test]
    fn resolves_inherited_and_overridden_effective_targets() {
        assert!(
            std::mem::size_of::<LifecycleResolution>() <= 2 * std::mem::size_of::<usize>(),
            "every type retains five lifecycle resolution slots, so payloads must remain indirect"
        );
        let (_fixture, _compilation, _runtime, index) = index(
            "/datum/base\n\tproc/New()\n\tproc/Initialize()\n\tproc/Destroy()\n/datum/base/child\n\tInitialize()\n\tproc/LateInitialize()\n/datum/base/sibling\n",
        );
        let child = index
            .find_path("/datum/base/child")
            .expect("child lifecycle should exist");
        let sibling = index
            .find_path("/datum/base/sibling")
            .expect("sibling lifecycle should exist");

        let LifecycleResolution::Resolved(new_target) = &child.targets.new_target else {
            panic!("New should resolve");
        };
        assert!(new_target.inherited);
        assert_eq!(new_target.declaring_type, "/datum/base");
        let LifecycleResolution::Resolved(sibling_new) = &sibling.targets.new_target else {
            panic!("sibling New should resolve");
        };
        assert!(Arc::ptr_eq(new_target, sibling_new));
        let LifecycleResolution::Resolved(initialize) = &child.targets.initialize else {
            panic!("Initialize should resolve");
        };
        assert!(!initialize.inherited);
        assert_eq!(initialize.declaring_type, "/datum/base/child");
        assert!(matches!(
            child.targets.late_initialize,
            LifecycleResolution::Resolved(_)
        ));
        assert!(matches!(
            child.targets.destroy,
            LifecycleResolution::Resolved(_)
        ));
        assert!(index.diagnostics().is_empty());
    }

    #[test]
    fn runtime_upcast_dereference_uses_derived_initial_fields() {
        let source = "/datum/later\n\tvar/datum/pointless_base/a\n/datum/pointless_base/derived/var/x = 7\n/proc/RunTest()\n\tvar/datum/later/L = new\n\tL.a = new /datum/pointless_base/derived()\n\treturn L.a:x\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let target = procedures
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == "/proc/RunTest")
            .and_then(|procedure| procedure.effective_target)
            .expect("RunTest should have an effective implementation");
        let executable = procedures
            .compile_vm_implementations(&compilation, [target])
            .expect("RunTest should lower");
        let entry = executable
            .implementation(target)
            .expect("RunTest should be in the VM module");
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("runtime image should materialize defaults");
        let mut state = runtime.take_execution_state();
        assert_eq!(
            execute_module_in_state(executable.module(), entry, &[], &mut state),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn construction_orders_defaults_parent_new_and_arguments_and_cleans_failures() {
        let source = "/datum/base\n\tvar/value = 1\n\tvar/stage = 1\n\tvar/seen_default = 0\n\tvar/list/waiting_calls\n\tNew(arg)\n\t\tseen_default = value\n\t\tstage = stage * 10 + arg\n/datum/base/sub\n\tvalue = 7\n\tNew(arg)\n\t\t..()\n\t\tstage = stage * 10 + 2\n/datum/fail/New()\n\tvar/list/L = null\n\treturn L[1]\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("runtime image should materialize defaults");
        let subtype = TypePath::parse("/datum/base/sub").unwrap();
        let datum = construct_datum(
            &compilation,
            &procedures,
            &mut runtime,
            &subtype,
            &[Value::number(3.0)],
        )
        .expect("subtype constructor should run");
        let record = runtime.heap().datum(datum).unwrap();
        assert_eq!(
            record.field(&FieldName::parse("seen_default").unwrap()),
            Ok(&Value::number(7.0))
        );
        assert_eq!(
            record.field(&FieldName::parse("stage").unwrap()),
            Ok(&Value::number(132.0))
        );
        assert_eq!(
            record.field(&FieldName::parse("waiting_calls").unwrap()),
            Ok(&Value::Null),
            "plain inherited declarations must exist before New runs"
        );

        let before = runtime.heap().datums().count();
        let failure = TypePath::parse("/datum/fail").unwrap();
        assert!(construct_datum(&compilation, &procedures, &mut runtime, &failure, &[]).is_err());
        assert_eq!(
            runtime.heap().datums().count(),
            before,
            "failed constructor allocation must be destroyed"
        );
    }

    #[test]
    fn deletion_runs_parent_cleanup_once_and_invalidates_on_failure() {
        let source = "var/global/events = 0\n/datum/base/Destroy()\n\tevents = events * 10 + 1\n/datum/base/sub/Destroy()\n\t..()\n\tevents = events * 10 + 2\n/datum/fail/Destroy()\n\tvar/list/L = null\n\treturn L[1]\n/datum/reentrant/Destroy()\n\tqdel(src)\n";
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let mut runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let subtype = TypePath::parse("/datum/base/sub").unwrap();
        let datum = construct_datum(&compilation, &procedures, &mut runtime, &subtype, &[])
            .expect("datum should construct");
        delete_datum(&compilation, &procedures, &mut runtime, datum)
            .expect("cleanup chain should succeed");
        assert!(
            runtime.heap().datum(datum).is_err(),
            "deleted handle must be stale"
        );
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/events"))
                .map(|variable| &variable.value),
            Some(&Value::number(12.0)),
            "parent Destroy must run before subtype cleanup exactly once"
        );

        let failure = TypePath::parse("/datum/fail").unwrap();
        let failing = construct_datum(&compilation, &procedures, &mut runtime, &failure, &[])
            .expect("failing-cleanup datum should construct");
        assert!(delete_datum(&compilation, &procedures, &mut runtime, failing).is_err());
        assert!(
            runtime.heap().datum(failing).is_err(),
            "cleanup failure must still invalidate the datum"
        );

        let reentrant = TypePath::parse("/datum/reentrant").unwrap();
        let reentrant_datum =
            construct_datum(&compilation, &procedures, &mut runtime, &reentrant, &[])
                .expect("reentrant-cleanup datum should construct");
        delete_datum(&compilation, &procedures, &mut runtime, reentrant_datum)
            .expect("qdel(src) during cleanup should count as already deleted");
        assert!(runtime.heap().datum(reentrant_datum).is_err());
    }

    #[test]
    fn plans_globals_world_and_map_lifecycles_without_execution() {
        let source = concat!(
            "/world/New()\n",
            "/atom/proc/New()\n",
            "/atom/proc/Initialize()\n",
            "/atom/proc/LateInitialize()\n",
            "/atom/proc/Destroy()\n",
            "/area/test\n",
            "/turf/test\n",
            "/obj/test\n\tInitialize()\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test{name = \"crate\"; dir = 4}, /turf/test, /area/test)\n",
            "(5,7,2) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");

        assert_eq!(plan.map_atoms.len(), 3);
        assert_eq!(
            plan.map_atoms[0].placement.coordinate,
            WorldCoordinate { x: 5, y: 7, z: 2 }
        );
        assert_eq!(plan.map_atoms[0].placement.map_path.as_ref(), "test.dmm");
        assert_eq!(plan.map_atoms[0].variables.len(), 2);
        assert_eq!(plan.map_atoms[0].variables[0].name, "name");
        assert_eq!(plan.map_atoms[0].variables[0].value.raw, "\"crate\"");
        assert_eq!(plan.map_atoms[0].variables[1].name, "dir");
        assert_eq!(plan.map_atoms[0].variables[1].raw, "dir = 4");
        assert!(
            plan.map_atoms[0].variables[0].name_span.start
                < plan.map_atoms[0].variables[0].span.end
        );
        assert_eq!(plan.events[0], InitializationEvent::Globals);
        assert!(matches!(
            plan.events[1],
            InitializationEvent::Lifecycle {
                subject: EventSubject::MapAtom(_),
                kind: LifecycleKind::New,
                ..
            }
        ));
        let lifecycle_events: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { kind, .. } => Some(*kind),
                InitializationEvent::Globals => None,
            })
            .collect();
        assert!(lifecycle_events.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(
            plan.map_lifecycle_counts(&index)[&LifecycleKind::Destroy],
            3
        );
        assert!(plan.diagnostics.is_empty());
    }

    #[test]
    fn repeated_map_template_rows_share_immutable_planning_metadata() {
        let source = concat!(
            "/world/New()\n",
            "/area/test\n",
            "/turf/test\n",
            "/obj/test\n\tInitialize()\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test{name = \"crate\"; dir = 4}, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\naaa\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "shared-map-path.dmm");

        assert_eq!(plan.map_atoms.len(), 9);
        let first = &plan.map_atoms[0];
        let second = &plan.map_atoms[3];
        let third = &plan.map_atoms[6];
        assert_ne!(first.placement.coordinate, second.placement.coordinate);
        assert_ne!(second.placement.coordinate, third.placement.coordinate);
        assert!(Arc::ptr_eq(
            &first.placement.map_path,
            &second.placement.map_path,
        ));
        assert!(Arc::ptr_eq(
            &second.placement.map_path,
            &third.placement.map_path,
        ));
        assert!(Arc::ptr_eq(&first.placement.key, &second.placement.key));
        assert!(Arc::ptr_eq(&second.placement.key, &third.placement.key));
        assert!(Arc::ptr_eq(&first.type_path, &second.type_path));
        assert!(Arc::ptr_eq(&second.type_path, &third.type_path));
        assert!(Arc::ptr_eq(&first.variables, &second.variables));
        assert!(Arc::ptr_eq(&second.variables, &third.variables));
        assert_eq!(first.variables.len(), 2);
        assert_eq!(first.variables[0].span, second.variables[0].span);
        assert_eq!(
            second.variables[1].value.span,
            third.variables[1].value.span
        );
        assert_eq!(first, &plan.clone().map_atoms[0]);
    }

    #[test]
    fn monk_pipeline_constructs_compiled_atoms_before_world_new_and_defers_init_to_ssatoms() {
        let source = concat!(
            "/world/Genesis()\n/world/New()\n",
            "/atom/New(loc)\n/atom/Initialize()\n/atom/LateInitialize()\n",
            "/datum/controller/subsystem/atoms/Initialize()\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell subsystem-managed map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "managed.dmm");
        let lifecycle: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { subject, kind, .. } => Some((*subject, *kind)),
                InitializationEvent::Globals => None,
            })
            .collect();

        assert_eq!(lifecycle[0], (EventSubject::World, LifecycleKind::Genesis));
        let world_new = lifecycle
            .iter()
            .position(|event| *event == (EventSubject::World, LifecycleKind::New))
            .expect("world New should be planned");
        assert!(
            lifecycle[1..world_new]
                .iter()
                .all(
                    |(subject, kind)| matches!(subject, EventSubject::MapAtom(_))
                        && *kind == LifecycleKind::New
                )
        );
        assert!(lifecycle[world_new + 1..].is_empty());
        assert!(!lifecycle.iter().any(|(_, kind)| matches!(
            kind,
            LifecycleKind::Initialize | LifecycleKind::LateInitialize
        )));
    }

    #[test]
    fn monk_pipeline_defers_lateinitialize_to_ssatoms_even_without_atoms_initialize_override() {
        let source = concat!(
            "/world/Genesis()\n/world/New()\n",
            "/atom/New(loc)\n/atom/LateInitialize()\n",
            "/datum/controller/subsystem/atoms/LateInitialize()\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell subsystem-managed map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "managed-lateonly.dmm");
        let lifecycle: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { subject, kind, .. } => Some((*subject, *kind)),
                InitializationEvent::Globals => None,
            })
            .collect();

        assert_eq!(lifecycle[0], (EventSubject::World, LifecycleKind::Genesis));
        let world_new = lifecycle
            .iter()
            .position(|event| *event == (EventSubject::World, LifecycleKind::New))
            .expect("world New should be planned");
        assert!(
            lifecycle[1..world_new]
                .iter()
                .all(
                    |(subject, kind)| matches!(subject, EventSubject::MapAtom(_))
                        && *kind == LifecycleKind::New
                )
        );
        assert!(lifecycle[world_new + 1..].is_empty());
        assert!(!lifecycle.iter().any(|(_, kind)| matches!(
            kind,
            LifecycleKind::Initialize | LifecycleKind::LateInitialize
        )));
    }

    #[test]
    fn monk_pipeline_defers_atom_lifecycle_to_atoms_descendant() {
        let source = concat!(
            "/world/Genesis()\n/world/New()\n",
            "/atom/New(loc)\n/atom/Initialize()\n",
            "/atom/LateInitialize()\n",
            "/datum/controller/subsystem/atoms\n",
            "/datum/controller/subsystem/atoms/descendant/Initialize()\n",
            "/datum/controller/subsystem/atoms/descendant/LateInitialize()\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell subsystem-managed descendant map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "managed-derivative.dmm");
        let lifecycle: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { subject, kind, .. } => Some((*subject, *kind)),
                InitializationEvent::Globals => None,
            })
            .collect();

        assert_eq!(lifecycle[0], (EventSubject::World, LifecycleKind::Genesis));
        let world_new = lifecycle
            .iter()
            .position(|event| *event == (EventSubject::World, LifecycleKind::New))
            .expect("world New should be planned");
        assert!(
            lifecycle[1..world_new]
                .iter()
                .all(
                    |(subject, kind)| matches!(subject, EventSubject::MapAtom(_))
                        && *kind == LifecycleKind::New
                )
        );
        assert!(lifecycle[world_new + 1..].is_empty());
        assert!(!lifecycle.iter().any(|(_, kind)| matches!(
            kind,
            LifecycleKind::Initialize | LifecycleKind::LateInitialize
        )));
    }

    #[test]
    fn monk_pipeline_defers_atom_lifecycle_to_atoms_granddescendant() {
        let source = concat!(
            "/world/Genesis()\n/world/New()\n",
            "/atom/New(loc)\n/atom/Initialize()\n",
            "/atom/LateInitialize()\n",
            "/datum/controller/subsystem/atoms\n",
            "/datum/controller/subsystem/atoms/branch\n",
            "/datum/controller/subsystem/atoms/branch/leaf/Initialize()\n",
            "/datum/controller/subsystem/atoms/branch/leaf/LateInitialize()\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell subsystem-managed grand-descendant map should parse");
        let world = build_plan(&map, &compilation);
        let plan =
            build_initialization_plan(&runtime, &index, &world, "managed-granddescendant.dmm");
        let lifecycle: Vec<_> = plan
            .events
            .iter()
            .filter_map(|event| match event {
                InitializationEvent::Lifecycle { subject, kind, .. } => Some((*subject, *kind)),
                InitializationEvent::Globals => None,
            })
            .collect();

        assert_eq!(lifecycle[0], (EventSubject::World, LifecycleKind::Genesis));
        let world_new = lifecycle
            .iter()
            .position(|event| *event == (EventSubject::World, LifecycleKind::New))
            .expect("world New should be planned");
        assert!(
            lifecycle[1..world_new]
                .iter()
                .all(
                    |(subject, kind)| matches!(subject, EventSubject::MapAtom(_))
                        && *kind == LifecycleKind::New
                )
        );
        assert!(lifecycle[world_new + 1..].is_empty());
        assert!(!lifecycle.iter().any(|(_, kind)| matches!(
            kind,
            LifecycleKind::Initialize | LifecycleKind::LateInitialize
        )));
    }

    #[test]
    fn executes_map_lifecycles_in_phase_order_without_compiling_unrelated_procs() {
        let source = concat!(
            "var/global/lifecycle_count = 1\n",
            "/world/New()\n\tsrc.stage = 5\n\tglobal.lifecycle_count += 1\n",
            "/atom/proc/New(loc)\n\tsrc.stage = (args.len * 10) + (args[1] == src.loc)\n",
            "/atom/proc/Initialize()\n\tsrc.stage += 1\n\tglobal.lifecycle_count += 1\n",
            "/atom/proc/LateInitialize()\n\tsrc.stage += 100\n",
            "/area/test\n/turf/test\n/obj/test\n",
            "/proc/not_a_lifecycle_proc()\n\tspawn(1) return 0\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

        let execution = execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("lifecycle execution should succeed");

        assert_eq!(execution.events.len(), 10);
        assert_eq!(execution.duplicate_map_events, 0);
        let kinds: Vec<_> = execution
            .events
            .iter()
            .map(|event| match event.event {
                InitializationEvent::Lifecycle { kind, .. } => kind,
                InitializationEvent::Globals => panic!("globals are not executed as a hook"),
            })
            .collect();
        assert_eq!(
            kinds,
            [
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::New,
                LifecycleKind::Initialize,
                LifecycleKind::Initialize,
                LifecycleKind::Initialize,
                LifecycleKind::LateInitialize,
                LifecycleKind::LateInitialize,
                LifecycleKind::LateInitialize,
            ]
        );
        let stage = FieldName::parse("stage").expect("stage should be a field name");
        let world_id = execution.world.expect("world should be allocated");
        assert_eq!(
            runtime.heap().datum_field(world_id, &stage),
            Ok(&Value::number(5.0))
        );
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/lifecycle_count"))
                .expect("global should remain materialized")
                .value,
            Value::number(5.0)
        );
        for datum in allocation.allocation_order() {
            assert_eq!(
                runtime.heap().datum_field(*datum, &stage),
                Ok(&Value::number(112.0))
            );
        }
    }

    #[test]
    fn map_new_that_initializes_through_ssatoms_is_not_initialized_twice() {
        let source = concat!(
            "var/global/initialize_count = 0\n",
            "/world/New()\n",
            "/atom\n\tvar/flags_1 = 0\n\tvar/stage = 0\n",
            "/atom/proc/New(loc)\n\tsrc.Initialize(1)\n",
            "/atom/proc/Initialize(mapload)\n\tif(flags_1 & 128)\n\t\treturn -1\n\tflags_1 |= 128\n\tstage += 1\n\tglobal.initialize_count += 1\n\treturn 0\n",
            "/atom/proc/LateInitialize()\n\tstage += 100\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

        let execution = execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("immediate initialization should not be repeated");

        assert_eq!(execution.events.len(), 4, "world New plus three atom News");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/initialize_count"))
                .expect("counter global")
                .value,
            Value::number(3.0),
        );
        for datum in allocation.allocation_order() {
            assert_eq!(
                runtime
                    .heap()
                    .datum_field(*datum, &FieldName::parse("stage").expect("stage field"),),
                Ok(&Value::number(1.0)),
                "synthetic Initialize/LateInitialize must be skipped after New initialized it",
            );
        }
    }

    #[test]
    fn precompiled_lifecycle_links_dynamic_map_expressions_to_project_procs() {
        let source = concat!(
            "/proc/map_value()\n\treturn 37\n",
            "/area/test\n/turf/test\n/obj/test\n\tvar/value = 0\n\tNew()\n\t\tmap_value()\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test{value = map_value()}, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("lifecycle should precompile without a runtime image");

        let mut runtime =
            RuntimeImage::from_compilation(&compilation).expect("runtime image should materialize");
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            None,
            &mut precompiled,
        )
        .expect("precompiled lifecycle and linked map expression should execute");

        let object = allocation
            .allocation_order()
            .iter()
            .copied()
            .find(|datum| {
                runtime
                    .heap()
                    .datum(*datum)
                    .is_ok_and(|record| record.type_path().as_str() == "/obj/test")
            })
            .expect("mapped object should exist");
        assert_eq!(
            runtime
                .heap()
                .datum_field(object, &FieldName::parse("value").unwrap()),
            Ok(&Value::number(37.0))
        );
    }

    #[test]
    fn precompiled_global_initializer_family_smoke_executes_transitive_file_constructor() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB\n",
            "var/global/smoke = \"\"\n",
            "/proc/trim(value)\n\treturn trimtext(value)\n",
            "/proc/file2list(filename, separator = \"\\n\", trim_file = TRUE)\n",
            "\tif(trim_file)\n",
            "\t\treturn splittext(trim(file2text(filename)), separator)\n",
            "\treturn splittext(file2text(filename), separator)\n",
            "/datum/controller/global_vars\n",
            "\tvar/datum/advertisements/advertisements\n",
            "\tproc/InitGlobaladvertisements()\n",
            "\t\tadvertisements = new\n",
            "\tproc/Initialize()\n",
            "\t\tfor(var/global_init in typesof(/datum/controller/global_vars/proc))\n",
            "\t\t\tif(global_init == /datum/controller/global_vars/proc/Initialize)\n",
            "\t\t\t\tcontinue\n",
            "\t\t\tcall(src, global_init)()\n",
            "/datum/advertisements\n",
            "\tvar/result = \"\"\n",
            "\tNew()\n",
            "\t\tresult = load_file(\"advertisements.txt\")\n",
            "\tproc/load_file(filename)\n",
            "\t\tvar/list/lines = file2list(filename)\n",
            "\t\tvar/output = \"\"\n",
            "\t\tfor(var/line in lines)\n",
            "\t\t\toutput += line[1]\n",
            "\t\treturn output\n",
            "/world/Genesis()\n",
            "\tGLOB = new\n",
            "\tGLOB.Initialize()\n",
            "\tglobal.smoke = GLOB.advertisements.result\n",
            "/area/test\n/turf/test\n",
        );
        let (fixture, compilation) = Fixture::compile(source);
        fs::write(
            fixture.0.join("advertisements.txt"),
            "- advertisement separator\n",
        )
        .expect("advertisement fixture should be written");
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("smoke selector map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let genesis = compile_index
            .find_path("/world")
            .and_then(
                |lifecycle| match lifecycle.targets.get(LifecycleKind::Genesis) {
                    LifecycleResolution::Resolved(target) => Some(target.implementation),
                    _ => None,
                },
            )
            .expect("world Genesis should resolve");
        let precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("global initializer family should precompile");
        let entry = precompiled
            .executable
            .implementation(genesis)
            .expect("Genesis should be retained by precompile");
        let mut runtime = RuntimeImage::from_compilation(&compilation)
            .expect("tiny smoke runtime should materialize");
        let world_datum = runtime
            .canonical_world()
            .expect("canonical world should exist");
        let mut state = runtime.take_execution_state();
        state.set_global(
            FieldName::parse("world").expect("world field name"),
            Value::Datum(world_datum),
        );
        execute_module_in_context(
            precompiled.executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(world_datum), Value::Null),
        )
        .expect("transitive generated global initializer should execute before map allocation");
        assert_eq!(
            state.global(&FieldName::parse("smoke").unwrap()),
            Some(&Value::text("-")),
        );
    }

    #[test]
    fn generated_global_qdel_executes_full_item_destroy_chain_before_map_allocation() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB\n",
            "var/global/destroy_trace = \"\"\n",
            "/proc/qdel(datum/to_delete)\n\treturn to_delete.Destroy()\n",
            "/datum/controller/global_vars\n",
            "\tproc/InitGlobalcleanup()\n",
            "\t\tvar/obj/item/temporary = new\n",
            "\t\tqdel(temporary)\n",
            "\tproc/Initialize()\n",
            "\t\tfor(var/global_init in typesof(/datum/controller/global_vars/proc))\n",
            "\t\t\tif(global_init == /datum/controller/global_vars/proc/Initialize)\n",
            "\t\t\t\tcontinue\n",
            "\t\t\tcall(src, global_init)()\n",
            "/datum/Destroy()\n\tglobal.destroy_trace += \"D\"\n\treturn 1\n",
            "/atom/Destroy()\n\tglobal.destroy_trace += \"A\"\n\treturn ..()\n",
            "/atom/movable/Destroy()\n\tglobal.destroy_trace += \"M\"\n\treturn ..()\n",
            "/obj/Destroy()\n",
            "\tvis_locs = null\n",
            "\tglobal.destroy_trace += \"O\"\n",
            "\treturn ..()\n",
            "/obj/item/Destroy()\n\tglobal.destroy_trace += \"I\"\n\treturn ..()\n",
            "/world/Genesis()\n",
            "\tGLOB = new /datum/controller/global_vars\n",
            "\tGLOB.Initialize()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let genesis = compile_index
            .find_path("/world")
            .and_then(
                |lifecycle| match lifecycle.targets.get(LifecycleKind::Genesis) {
                    LifecycleResolution::Resolved(target) => Some(target.implementation),
                    _ => None,
                },
            )
            .unwrap();
        let precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("generated qdel/Destroy family should precompile");
        let entry = precompiled.executable.implementation(genesis).unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let world_datum = runtime.canonical_world().unwrap();
        let mut state = runtime.take_execution_state();
        state.set_global(
            FieldName::parse("world").unwrap(),
            Value::Datum(world_datum),
        );
        execute_module_in_context(
            precompiled.executable.module(),
            entry,
            &[],
            &mut state,
            &ExecutionContext::new(Value::Datum(world_datum), Value::Null),
        )
        .expect("generated initializer qdel should execute every inherited Destroy body");
        assert_eq!(
            state.global(&FieldName::parse("destroy_trace").unwrap()),
            Some(&Value::text("IOMAD")),
        );
    }

    #[test]
    fn lifecycle_drains_waitfor_false_world_continuations() {
        let source = concat!(
            "var/global/finished = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\t. = 7\n\tsleep(1)\n\tglobal.finished = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");

        let execution = execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("detached world continuation should drain");

        assert_eq!(execution.scheduler.pending_tasks, 0);
        assert!(execution.scheduler.rounds >= 1);
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/finished"))
                .map(|variable| &variable.value),
            Some(&Value::number(1.0))
        );
    }

    #[test]
    fn readiness_and_persistent_slices_preserve_delayed_server_work() {
        let source = concat!(
            "var/global/ready = 0\nvar/global/pulses = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\tsleep(2)\n\tglobal.ready = 1\n\tsleep(20)\n\tglobal.pulses = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 10,
            },
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 1);
        assert_eq!(
            precompiled.persistent_tick_duration(),
            std::time::Duration::from_millis(100)
        );

        let first = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 5,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(
            first.final_tick, 7,
            "idle slices must advance toward future work"
        );
        assert_eq!(first.pending_tasks, 1);
        for _ in 0..3 {
            advance_persistent_scheduler(
                &mut precompiled,
                &mut runtime,
                SchedulerDrainLimits {
                    max_ticks: 5,
                    max_rounds: 10,
                },
            )
            .unwrap();
        }
        let final_slice = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(final_slice.pending_tasks, 0);
        assert_eq!(
            final_slice.termination,
            SchedulerDrainTermination::StableIdle
        );
    }

    #[test]
    fn production_boot_without_readiness_retains_pending_scheduler_state() {
        let source = concat!(
            "var/global/finished = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\tsleep(2)\n\tglobal.finished = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let execution = execute_boot_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 0,
            },
            None,
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::RoundLimit
        );
        assert_eq!(execution.scheduler.pending_tasks, 1);
        assert!(precompiled.persistent_state.is_some());

        let resumed = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 5,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(resumed.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(resumed.pending_tasks, 0);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .unwrap()
                .global(&FieldName::parse("finished").unwrap()),
            Some(&Value::number(1.0)),
        );
    }

    #[test]
    fn startup_service_attaches_client_before_readiness_and_preserves_session() {
        let source = concat!(
            "var/global/ready = 0\nvar/global/client_started = 0\n",
            "/world/New()\n\tset waitfor = FALSE\n\tsleep(2)\n\tglobal.ready = 1\n",
            "/client/New()\n\tglobal.client_started = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let mut attached = None;
        let mut service = |executable: &ExecutableProcedures, state: &mut ExecutionState| {
            if attached.is_none() {
                attached = Some(state.connect_local_guest(executable.module()).unwrap());
            }
        };
        let execution = execute_boot_initialization_plan_with_precompiled_and_startup_service(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 20,
            },
            Some(&readiness),
            &mut precompiled,
            &mut service,
        )
        .unwrap();
        drop(service);

        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert!(execution.executed_events > 0);
        assert_eq!(
            execution.executed_event_counts.values().sum::<usize>(),
            execution.executed_events,
        );
        assert!(
            execution.events.is_empty(),
            "production boot retains aggregate lifecycle counts, not per-event audit records",
        );
        let attached = attached.expect("startup service attached a client");
        let state = precompiled
            .persistent_state
            .as_ref()
            .expect("ready boot preserves persistent state");
        assert_eq!(
            state.global(&FieldName::parse("client_started").unwrap()),
            Some(&Value::number(1.0))
        );
        assert!(state.local_client_state(attached.client).is_ok());
    }

    #[test]
    fn persistent_idle_slices_keep_the_server_clock_advancing() {
        let source = concat!(
            "var/global/ready = 0\n",
            "/world/New()\n\tglobal.ready = 1\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 0);

        let first = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(first.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(first.final_tick, 1);
        assert_eq!(first.pending_tasks, 0);

        let second = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 3,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(second.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(second.final_tick, 4);
        let state = precompiled.persistent_state.as_ref().unwrap();
        let Value::Datum(world) = state.global(&FieldName::parse("world").unwrap()).unwrap() else {
            panic!("persistent state should retain the world singleton");
        };
        assert_eq!(
            state
                .heap()
                .datum_field(*world, &FieldName::parse("time").unwrap()),
            Ok(&Value::number(4.0)),
        );
    }

    #[test]
    fn infinite_native_walk_does_not_block_readiness_or_persistent_idle_slices() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/walker\n",
            "/world/New()\n",
            "\tglobal.walker = new /obj/walker\n",
            "\twalk(global.walker, EAST, 1)\n",
            "\tglobal.ready = 1\n",
            "/obj/walker\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .expect("a perpetual engine walk must not prevent startup readiness");
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady,
        );
        assert_eq!(execution.scheduler.pending_tasks, 0);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .unwrap()
                .next_scheduled_tick(),
            Some(1),
        );

        for expected_tick in 1..=3 {
            let slice = advance_persistent_scheduler(
                &mut precompiled,
                &mut runtime,
                SchedulerDrainLimits {
                    max_ticks: 1,
                    max_rounds: 10,
                },
            )
            .expect("persistent walk ticks should remain bounded and non-blocking");
            assert_eq!(slice.termination, SchedulerDrainTermination::StableIdle);
            assert_eq!(slice.pending_tasks, 0);
            assert_eq!(slice.final_tick, expected_tick);
            assert_eq!(
                precompiled
                    .persistent_state
                    .as_ref()
                    .unwrap()
                    .next_scheduled_tick(),
                Some(expected_tick + 1),
            );
        }
    }

    #[test]
    fn persistent_scheduler_isolates_a_failed_thread_and_runs_later_due_work() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/trace = \"\"\n",
            "/proc/fail_later()\n\tCRASH(\"isolated\")\n",
            "/proc/finish_later()\n\tglobal.trace += \"L\"\n",
            "/world/New()\n",
            "\tglobal.ready = 1\n",
            "\tspawn(1) fail_later()\n",
            "\tspawn(1) finish_later()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady
        );
        assert_eq!(execution.scheduler.pending_tasks, 2);

        let slice = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .expect("one failed scheduled thread must not stop the server");
        assert_eq!(slice.failed_tasks, 1);
        assert_eq!(slice.completed_tasks, 1);
        assert_eq!(slice.pending_tasks, 0);
        assert_eq!(slice.final_tick, 1);
        assert_eq!(slice.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("trace").unwrap())),
            Some(&Value::text("L")),
        );
    }

    #[test]
    fn pre_readiness_scheduler_drain_is_wall_bounded_and_resumable() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/progress = 0\n",
            "/proc/finish_startup()\n",
            "\tvar/local_progress = 0\n",
            "\twhile(local_progress < 200000)\n",
            "\t\tlocal_progress += 1\n",
            "\tglobal.progress = local_progress\n",
            "\tglobal.ready = 1\n",
            "/world/New()\n",
            "\tspawn(0) finish_startup()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        let started = Instant::now();
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::RoundLimit
        );
        assert!(execution.scheduler.pending_tasks > 0);

        let resumed = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 10,
            },
        )
        .unwrap();
        assert_eq!(resumed.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .unwrap()
                .global(&FieldName::parse("progress").unwrap()),
            Some(&Value::number(200000.0)),
        );
    }

    #[test]
    fn persistent_round_limit_preserves_same_tick_work_for_the_next_slice() {
        let source = concat!(
            "var/global/ready = 0\n",
            "var/global/runs = 0\n",
            "/proc/run_again()\n",
            "\tglobal.runs += 1\n",
            "\tif(global.runs < 3)\n",
            "\t\tspawn(0) run_again()\n",
            "/world/New()\n",
            "\tglobal.ready = 1\n",
            "\tspawn(0) run_again()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .unwrap();
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .unwrap();
        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(1.0),
        };
        execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            Some(&readiness),
            &mut precompiled,
        )
        .unwrap();

        let bounded = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 2,
            },
        )
        .unwrap();
        assert_eq!(bounded.termination, SchedulerDrainTermination::RoundLimit);
        assert_eq!(bounded.final_tick, 0);
        assert_eq!(bounded.rounds, 2);
        assert_eq!(bounded.completed_tasks, 2);
        assert_eq!(bounded.pending_tasks, 1);

        let resumed = advance_persistent_scheduler(
            &mut precompiled,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 2,
            },
        )
        .unwrap();
        assert_eq!(resumed.termination, SchedulerDrainTermination::StableIdle);
        assert_eq!(resumed.final_tick, 1);
        assert_eq!(resumed.completed_tasks, 1);
        assert_eq!(resumed.pending_tasks, 0);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("runs").unwrap())),
            Some(&Value::number(3.0)),
        );
    }

    #[test]
    fn monk_like_master_stages_reach_readiness_through_waitfor_scheduler() {
        let source = concat!(
            "var/global/datum/controller/master/Master\n",
            "var/global/trace = \"\"\n",
            "var/global/ready = 0\n",
            "/proc/dispatch_initialize(target)\n",
            "\tcall(target, \"Initialize\")()\n",
            "/proc/finish_subsystem_stage()\n",
            "\tsleep(2)\n",
            "\tglobal.trace += \"S\"\n",
            "\tglobal.ready = 2\n",
            "/datum/controller/master\n",
            "\tNew()\n",
            "\t\tglobal.Master = src\n",
            "\t\tglobal.trace += \"N\"\n",
            "\tproc/Initialize()\n",
            "\t\tset waitfor = FALSE\n",
            "\t\tglobal.trace += \"I\"\n",
            "\t\tfinish_subsystem_stage()\n",
            "/world/Genesis()\n",
            "\tMaster = new /datum/controller/master\n",
            "/world/New()\n",
            "\tglobal.trace += \"W\"\n",
            "\tConfigLoaded()\n",
            "\tdispatch_initialize(Master)\n",
            "/world/proc/ConfigLoaded()\n",
            "\tglobal.trace += \"C\"\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation) = Fixture::compile(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("one-cell staged smoke map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("Master stages should link lazily before runtime allocation");
        assert!(precompiled.deferred_procedures() > 0);

        let mut runtime = RuntimeImage::from_compilation(&compilation).unwrap();
        let index = LifecycleIndex::build(&compilation, &procedures, &runtime);
        let plan = build_initialization_plan(&runtime, &index, &world, "staged-smoke.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        let readiness = HeadlessReadinessProbe {
            qualified_storage: None,
            global: FieldName::parse("ready").unwrap(),
            fields: vec![],
            expected: Value::number(2.0),
        };
        let execution = execute_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits {
                max_ticks: 10,
                max_rounds: 20,
            },
            Some(&readiness),
            &mut precompiled,
        )
        .expect("Master staged startup should reach readiness");
        assert_eq!(
            execution.scheduler.termination,
            SchedulerDrainTermination::HeadlessReady,
        );
        assert_eq!(execution.scheduler.final_tick, 2);
        assert_eq!(
            precompiled
                .persistent_state
                .as_ref()
                .and_then(|state| state.global(&FieldName::parse("trace").unwrap())),
            Some(&Value::text("NWCIS")),
        );
    }

    #[test]
    fn genesis_infers_all_typed_global_bare_new_destinations() {
        let source = concat!(
            "var/global/datum/controller/global_vars/GLOB = null\n",
            "var/global/datum/tracy/Tracy = null\n",
            "var/global/datum/debugger/Debugger = null\n",
            "var/global/datum/log_holder/logger = null\n",
            "var/global/datum/controller/master/Master = null\n",
            "/world/Genesis()\n\tGLOB.config_error_log = \"early.log\"\n\tTracy = new\n\tDebugger = new\n\tlogger = new\n\tMaster = new\n",
            "/datum/controller/global_vars\n\tvar/global/config_error_log\n",
            "/datum/tracy\n/datum/debugger\n/datum/log_holder\n/datum/controller/master\n\tvar/static/random_seed\n",
            "/datum/controller/master/New()\n\tif(!random_seed)\n\t\trandom_seed = 29051994\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("typed global bare new should execute");

        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/config_error_log"))
                .map(|variable| &variable.value),
            Some(&Value::text("early.log")),
            "a typed global receiver must bind owner-qualified static storage even while its datum value is null; vars={:?}",
            runtime.variables(),
        );

        for (name, expected) in [
            ("Tracy", "/datum/tracy"),
            ("Debugger", "/datum/debugger"),
            ("logger", "/datum/log_holder"),
            ("Master", "/datum/controller/master"),
        ] {
            let value = &runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with(&format!("/{name}")))
                .unwrap_or_else(|| panic!("missing global {name}"))
                .value;
            let Value::Datum(datum) = value else {
                panic!("{name} should contain a datum");
            };
            assert_eq!(
                runtime.heap().datum(*datum).unwrap().type_path().as_str(),
                expected
            );
        }
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/random_seed"))
                .map(|variable| &variable.value),
            Some(&Value::number(29_051_994.0)),
        );
    }

    #[test]
    fn glob_style_datum_vars_is_live_stable_and_copies_a_snapshot() {
        let source = concat!(
            "var/global/observed = 0\n",
            "/datum/globals\n\tvar/value = 1\n\tvar/global/shared = 2\n",
            "/datum/globals/proc/TestReflection()\n\tvalue = 3\n\tvar/list/reflection = vars\n\tvar/same_proxy = (reflection == vars)\n\treflection[\"value\"] += 2\n\treflection[\"shared\"] = 7\n\tvar/list/snapshot = reflection.Copy()\n\tvalue = 9\n\tglobal.observed = same_proxy + reflection[\"value\"] + reflection[\"shared\"] + snapshot[\"value\"] + snapshot[\"shared\"]\n",
            "/world/Genesis()\n\tvar/datum/globals/controller = new\n\tcontroller.TestReflection()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
            .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("datum vars reflection should execute during Genesis");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/observed"))
                .map(|variable| &variable.value),
            Some(&Value::number(29.0))
        );
    }

    #[test]
    fn sort_instance_list_defaults_exist_before_new_and_tim_sort() {
        let source = concat!(
            "var/global/datum/sort_instance/sorter = new /datum/sort_instance\n",
            "var/global/observed = 0\n",
            "/datum/sort_instance\n\tvar/list/runBases = list()\n\tvar/list/runLens = list()\n",
            "/datum/sort_instance/New()\n\trunBases.Add(1)\n",
            "/datum/sort_instance/proc/timSort()\n\trunBases.Cut()\n\trunLens.Cut()\n\treturn runBases.len + runLens.len\n",
            "/world/Genesis()\n\tglobal.observed = sorter.timSort()\n",
            "/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n")
            .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("world should allocate");
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .expect("sort instance defaults should precede New and timSort");
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|variable| variable.path.ends_with("/observed"))
                .map(|variable| &variable.value),
            Some(&Value::number(0.0))
        );
    }

    #[test]
    fn runtime_new_links_inherited_call_and_nested_new_defaults_before_constructor() {
        let source = concat!(
            "var/global/observed = 0\n/proc/make_base()\n\treturn 3\n/proc/make_child()\n\treturn 8\n",
            "/datum/token\n/datum/base\n\tvar/x = make_base()\n\tvar/datum/token/token = new /datum/token\n",
            "/datum/base/child\n\tx = make_child()\n/datum/base/child/New()\n\tglobal.observed = x + istype(token, /datum/token)\n",
            "/world/Genesis()\n\tnew /datum/base/child\n/area/test\n/turf/test\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse("\"a\" = (/turf/test, /area/test)\n(1,1,1) = {\"\na\n\"}\n").unwrap();
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");
        let allocation = allocate_world(&world, &mut runtime).unwrap();
        execute_initialization_plan(
            &compilation,
            &procedures,
            &index,
            &plan,
            &allocation,
            &mut runtime,
        )
        .unwrap();
        assert_eq!(
            runtime
                .variables()
                .iter()
                .find(|v| v.path.ends_with("/observed"))
                .map(|v| &v.value),
            Some(&Value::number(9.0))
        );
    }

    #[test]
    fn compatibility_sweep_collects_lifecycle_failures_without_hiding_good_targets() {
        let source = concat!(
            "/world/New()\n\tspawn(1) return 0\n",
            "/atom/proc/New()\n\treturn 0\n",
            "/area/test\n/turf/test\n/obj/test\n",
        );
        let (_fixture, compilation, runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/test, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\na\n\"}\n",
        ))
        .expect("map should parse");
        let world = build_plan(&map, &compilation);
        let plan = build_initialization_plan(&runtime, &index, &world, "test.dmm");

        let sweep = sweep_lifecycle_compatibility(&compilation, &procedures, &index, &plan);

        assert!(sweep.targets >= 2);
        assert!(sweep.compatible >= 1);
        assert_eq!(sweep.issues.len(), 1);
        assert!(!sweep.issues[0].message.is_empty());
        assert_eq!(
            sweep.issues[0].locations[0].procedure_path,
            "/world/proc/New"
        );
        assert_eq!(sweep.issues[0].locations[0].source.path, "types.dm");
    }

    #[test]
    fn runtime_audit_collects_independent_map_failures_in_one_execution() {
        let source = concat!(
            "/area/test\n/turf/test\n",
            "/obj/first\n/obj/first/Initialize()\n\tvar/list/missing\n\treturn missing[1]\n",
            "/obj/second\n/obj/second/Initialize()\n\tvar/list/missing\n\treturn missing[2]\n",
        );
        let (_fixture, compilation, mut runtime, index) = index(source);
        let procedures = ProcedureRegistry::build(&compilation);
        let map = parse(concat!(
            "\"a\" = (/obj/first, /turf/test, /area/test)\n",
            "\"b\" = (/obj/second, /turf/test, /area/test)\n",
            "(1,1,1) = {\"\nab\n\"}\n",
        ))
        .expect("two-cell audit map should parse");
        let world = build_plan(&map, &compilation);
        let compile_index = LifecycleIndex::build_compile_only(&compilation, &procedures);
        let mut precompiled =
            precompile_lifecycle_for_world(&compilation, &procedures, &compile_index, &world)
                .expect("both failing Initialize bodies should link");
        let plan = build_initialization_plan(&runtime, &index, &world, "audit.dmm");
        let allocation = allocate_world(&world, &mut runtime).expect("audit world should allocate");

        let error = audit_initialization_plan_with_precompiled(
            &index,
            &plan,
            &allocation,
            &mut runtime,
            SchedulerDrainLimits::default(),
            &mut precompiled,
        )
        .expect_err("audit should return its grouped failure count");

        assert!(matches!(
            error,
            InitializationExecutionError::AuditFailures { failures: 2 }
        ));
    }

    #[test]
    fn host_slice_budget_reacts_quickly_and_recovers_gradually() {
        let mut budget = HostSliceBudget::new(100_000, 1_000, 100_000, Duration::from_millis(10));

        budget.observe(Duration::from_millis(11));
        assert_eq!(budget.steps(), 50_000);
        budget.observe(Duration::from_millis(20));
        assert_eq!(budget.steps(), 25_000);

        budget.observe(Duration::from_millis(5));
        assert_eq!(budget.steps(), 31_250);
        budget.observe(Duration::from_millis(8));
        assert_eq!(budget.steps(), 31_250);
    }

    #[test]
    fn host_slice_budget_never_leaves_configured_bounds() {
        let mut budget = HostSliceBudget::new(999_999, 1_000, 100_000, Duration::from_millis(10));
        assert_eq!(budget.steps(), 100_000);
        for _ in 0..16 {
            budget.observe(Duration::from_secs(1));
        }
        assert_eq!(budget.steps(), 1_000);
        for _ in 0..32 {
            budget.observe(Duration::ZERO);
        }
        assert_eq!(budget.steps(), 100_000);
    }
}
