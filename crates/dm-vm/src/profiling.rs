//! Boot-phase observability: startup, atoms, and TGM instruction profiles,
//! the boot trace, and the shuttle-pipeline trace.
//!
//! Kept as a dedicated crate-root module so the interpreter run loop and
//! execution state can consult or emit diagnostics without whole-engine
//! knowledge of profile accounting.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::{CallFrame, ExecutionState, Module, ProcedureId};
use dm_value::{DatumId, FieldName, ListId, Value};

use crate::Instruction;

#[derive(Clone, Debug)]
pub(crate) enum ShuttleTracePostReturn {
    AtmosInit,
    NullifyNode { slot: Option<usize> },
}

#[derive(Debug)]
pub(crate) struct AtomsProfile {
    pub(crate) started: Instant,
    pub(crate) last_snapshot: Instant,
    pub(crate) startup_root: Option<String>,
    pub(crate) total_instructions: u64,
    pub(crate) instruction_categories: Option<[u64; STARTUP_INSTRUCTION_CATEGORY_COUNT]>,
    pub(crate) samples: HashMap<AtomsProfileProcedure, u64>,
    // Approximate wall time sampled at the existing 4,096-step checkpoints.
    // Native helpers retain logical instruction accounting, so this separates
    // expensive interpreter work from cheaply replayed reference budgets.
    pub(crate) wall_sample_nanos: HashMap<AtomsProfileProcedure, u128>,
    pub(crate) frame_entries: HashMap<AtomsProfileProcedure, u64>,
    pub(crate) paths: HashMap<AtomsProfileProcedure, String>,
    pub(crate) instruction_samples: HashMap<AtomsProfileInstruction, u64>,
    pub(crate) instruction_wall_nanos: HashMap<AtomsProfileInstruction, u128>,
    pub(crate) instruction_labels: HashMap<AtomsProfileInstruction, String>,
}

#[derive(Debug)]
pub(crate) struct TgmProfile {
    pub(crate) started: Instant,
    pub(crate) total_instructions: u64,
    pub(crate) procedure_samples: HashMap<AtomsProfileProcedure, u64>,
    pub(crate) instruction_samples: HashMap<AtomsProfileInstruction, u64>,
    pub(crate) paths: HashMap<AtomsProfileProcedure, String>,
    pub(crate) instruction_labels: HashMap<AtomsProfileInstruction, String>,
}

pub(crate) fn tgm_profiling_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_TGM").is_some())
}

pub(crate) fn emit_tgm_profile(profile: &TgmProfile) {
    let mut procedures = profile.procedure_samples.iter().collect::<Vec<_>>();
    procedures.sort_by_key(|(_, samples)| Reverse(**samples));
    eprintln!(
        "boot-vm: tgm-profile-summary elapsed_ms={} instructions={} procedures={}",
        profile.started.elapsed().as_millis(),
        profile.total_instructions,
        procedures.len()
    );
    for (key, samples) in procedures.into_iter().take(32) {
        eprintln!(
            "boot-vm: tgm-profile-procedure samples={} path={}",
            samples,
            profile.paths.get(key).map_or("<missing>", String::as_str)
        );
    }
    let mut instructions = profile.instruction_samples.iter().collect::<Vec<_>>();
    instructions.sort_by_key(|(_, samples)| Reverse(**samples));
    for (key, samples) in instructions.into_iter().take(64) {
        eprintln!(
            "boot-vm: tgm-profile-opcode samples={} path={} pc={} instruction={}",
            samples,
            profile
                .paths
                .get(&AtomsProfileProcedure {
                    module_identity: key.module_identity,
                    procedure: key.procedure,
                })
                .map_or("<missing>", String::as_str),
            key.instruction,
            profile
                .instruction_labels
                .get(key)
                .map_or("<missing>", String::as_str),
        );
    }
}

pub(crate) const STARTUP_INSTRUCTION_CATEGORY_COUNT: usize = 7;

pub(crate) fn startup_instruction_category(instruction: &Instruction) -> usize {
    match instruction {
        Instruction::IndexList
        | Instruction::IndexLocalList(_)
        | Instruction::NextLocalListIteration { .. }
        | Instruction::ListLength
        | Instruction::ListLengthLocal(_)
        | Instruction::Contains
        | Instruction::PrepareIteration => 0,
        Instruction::SetListIndex
        | Instruction::SetListIndexKeep
        | Instruction::CompoundListIndex(_)
        | Instruction::CompoundListIndexKeep(_)
        | Instruction::MutateListIndex { .. }
        | Instruction::LogicalOrEmptyListIndex => 1,
        Instruction::LoadField(_)
        | Instruction::LoadDeclaredField(_)
        | Instruction::LoadGlobal(_)
        | Instruction::InitialField(_)
        | Instruction::InitialDynamicField
        | Instruction::LoadDynamicField => 2,
        Instruction::StoreField(_)
        | Instruction::StoreFieldKeep(_)
        | Instruction::StoreGlobal(_)
        | Instruction::MutateField { .. }
        | Instruction::StoreDynamicField => 3,
        Instruction::Call { .. }
        | Instruction::CallCurrent { .. }
        | Instruction::CallParent { .. }
        | Instruction::CallDynamic { .. }
        | Instruction::StandardBuiltin { .. }
        | Instruction::NativeSrcMethod { .. }
        | Instruction::Return => 4,
        Instruction::JumpIfNull(_)
        | Instruction::JumpIfFalse(_)
        | Instruction::Jump(_)
        | Instruction::JumpIfArgumentSupplied { .. }
        | Instruction::IterationTypeFilter(_) => 5,
        _ => 6,
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AtomsProfileProcedure {
    pub(crate) module_identity: u64,
    pub(crate) procedure: ProcedureId,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct AtomsProfileInstruction {
    pub(crate) module_identity: u64,
    pub(crate) procedure: ProcedureId,
    pub(crate) instruction: usize,
}

pub(crate) fn shuttle_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        diagnostic_env_truthy(std::env::var("DREAM64_SHUTTLE_TRACE").ok().as_deref())
    })
}

pub(crate) fn diagnostic_env_truthy(value: Option<&str>) -> bool {
    value.is_some_and(|value| matches!(value, "1" | "true" | "yes" | "on"))
}

fn shuttle_trace_target_filter() -> &'static Option<String> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| {
        Some(
            std::env::var("DREAM64_SHUTTLE_TRACE_TARGET")
                .unwrap_or_else(|_| "/mixer/layer4".to_owned()),
        )
    })
}

pub(crate) fn shuttle_trace_is_late_shuttle_move(path: &str) -> bool {
    path.contains("lateShuttleMove")
}

pub(crate) fn shuttle_trace_is_nullify_node(path: &str) -> bool {
    path.contains("nullify_node@") || path.ends_with("/nullify_node")
}

pub(crate) fn shuttle_trace_is_atmos_init(path: &str) -> bool {
    path.contains("atmos_init@") || path.ends_with("/atmos_init")
}

fn shuttle_trace_matches_target(state: &ExecutionState, datum: DatumId) -> bool {
    shuttle_trace_target_filter()
        .as_ref()
        .is_some_and(|target| {
            state
                .heap
                .datum(datum)
                .is_ok_and(|datum| datum.type_path().as_str().contains(target))
        })
}

fn shuttle_trace_slot_from_value(value: &Value) -> Option<usize> {
    let value = value.as_number()?;
    if !value.is_finite() || value.fract() != 0.0 {
        return None;
    }
    let value = value.trunc() as i64;
    usize::try_from(value).ok().filter(|slot| *slot > 0)
}

pub(crate) fn shuttle_trace_slot_from_arguments(arguments: &[Value]) -> Option<usize> {
    arguments.first().and_then(shuttle_trace_slot_from_value)
}

fn shuttle_trace_turn_dir(direction: i32, angle: i32) -> i32 {
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let Some(index) = DIRECTIONS
        .iter()
        .position(|candidate| *candidate == direction)
    else {
        return direction;
    };
    let steps = angle / 45;
    DIRECTIONS[(index as i32 + steps).rem_euclid(8) as usize]
}

fn shuttle_trace_expected_slot_direction(
    dir_value: f32,
    flipped: bool,
    slot: usize,
) -> Option<i32> {
    if !dir_value.is_finite() {
        return None;
    }
    if dir_value.fract() != 0.0 || dir_value < i32::MIN as f32 || dir_value > i32::MAX as f32 {
        return None;
    }
    let direction = dir_value.trunc() as i32;
    let mut node1 = shuttle_trace_turn_dir(direction, -180);
    let node2 = shuttle_trace_turn_dir(direction, -90);
    let mut node3 = shuttle_trace_turn_dir(direction, 0);
    if flipped {
        node1 = shuttle_trace_turn_dir(node1, 180);
        node3 = shuttle_trace_turn_dir(node3, 180);
    }
    match slot {
        1 => Some(node1),
        2 => Some(node2),
        3 => Some(node3),
        _ => None,
    }
}

fn shuttle_trace_list_len(state: &ExecutionState, list: Option<ListId>) -> usize {
    list.and_then(|list| state.heap.list(list).ok())
        .map_or(0, dm_value::DmList::len)
}

fn shuttle_trace_list_slot(
    state: &ExecutionState,
    list: Option<ListId>,
    slot: usize,
) -> Option<Value> {
    let list = list?;
    state.heap.list(list).ok()?.get(slot).ok().cloned()
}

fn shuttle_trace_field_text(state: &ExecutionState, datum: DatumId, name: &str) -> String {
    let field = FieldName::parse(name).expect("built-in atom variable");
    match state.heap.datum_field(datum, &field) {
        Ok(value) => value.to_string(),
        Err(_) => "<missing>".to_owned(),
    }
}

fn shuttle_trace_field_u8(state: &ExecutionState, datum: DatumId, name: &str) -> Option<usize> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
        .filter(|value| value.is_finite() && value.fract() == 0.0)
        .and_then(|value| {
            let value = value as i64;
            usize::try_from(value).ok().filter(|slot| *slot > 0)
        })
}

fn shuttle_trace_field_bool(state: &ExecutionState, datum: DatumId, name: &str) -> bool {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
        .is_some_and(|value| value != 0.0)
}

fn shuttle_trace_field_number(state: &ExecutionState, datum: DatumId, name: &str) -> Option<f32> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    state
        .heap
        .datum_field(datum, &field)
        .ok()
        .and_then(Value::as_number)
}

fn shuttle_trace_list_field(state: &ExecutionState, datum: DatumId, name: &str) -> Option<ListId> {
    let field = FieldName::parse(name).expect("built-in atom variable");
    match state.heap.datum_field(datum, &field) {
        Ok(Value::List(list)) => Some(*list),
        _ => None,
    }
}

fn shuttle_trace_datum_type(state: &ExecutionState, target: DatumId) -> String {
    state.heap.datum(target).map_or_else(
        |_| "<missing>".to_owned(),
        |datum| datum.type_path().to_owned().to_string(),
    )
}

fn shuttle_trace_value_ref(value: Option<&Value>) -> String {
    match value {
        Some(Value::Datum(datum)) => format!("datum({datum:?})"),
        Some(Value::Null) => "null".to_owned(),
        Some(value) => format!("other({value})"),
        None => "none".to_owned(),
    }
}

pub(crate) fn shuttle_trace_prepare_call(
    module: &Module,
    state: &ExecutionState,
    caller: &CallFrame,
    procedure: ProcedureId,
    nullify_slot: Option<usize>,
    target_frame: &mut CallFrame,
) {
    target_frame.set_shuttle_trace_target(caller.shuttle_trace_target());
    target_frame.set_shuttle_trace_post_return(None);

    if !shuttle_trace_enabled() {
        return;
    }
    let Some(path) = module.procedure_path(procedure) else {
        return;
    };

    if shuttle_trace_is_late_shuttle_move(path)
        && let Value::Datum(target) = target_frame.src
        && shuttle_trace_matches_target(state, target)
    {
        target_frame.set_shuttle_trace_target(Some(target));
        shuttle_trace_emit_snapshot(state, target, "lateShuttleMove-entry", None);
        return;
    }

    let Some(shuttle_target) = target_frame.shuttle_trace_target() else {
        return;
    };
    if shuttle_trace_is_nullify_node(path) {
        let slot = nullify_slot;
        shuttle_trace_emit_snapshot(state, shuttle_target, "nullify-node-before", slot);
        target_frame
            .set_shuttle_trace_post_return(Some(ShuttleTracePostReturn::NullifyNode { slot }));
    }
    if shuttle_trace_is_atmos_init(path) {
        shuttle_trace_emit_snapshot(state, shuttle_target, "atmos-init-before", None);
        target_frame.set_shuttle_trace_post_return(Some(ShuttleTracePostReturn::AtmosInit));
    }
}

pub(crate) fn shuttle_trace_emit_snapshot(
    state: &ExecutionState,
    component: DatumId,
    event: &str,
    note: Option<usize>,
) {
    if !shuttle_trace_enabled() {
        return;
    }
    let component_type = shuttle_trace_datum_type(state, component);
    let target_dir = shuttle_trace_field_number(state, component, "dir").unwrap_or_default();
    let target_flipped = shuttle_trace_field_bool(state, component, "flipped");
    let device_type = shuttle_trace_field_u8(state, component, "device_type").unwrap_or(3);
    let nodes = shuttle_trace_list_field(state, component, "nodes");
    let parents = shuttle_trace_list_field(state, component, "parents");
    let node_len = shuttle_trace_list_len(state, nodes);
    let parent_len = shuttle_trace_list_len(state, parents);
    let slot_count = *[node_len, parent_len, device_type]
        .iter()
        .max()
        .unwrap_or(&3)
        .max(&1);
    let note = note.map_or_else(|| "n/a".to_owned(), |slot| slot.to_string());
    let target_location = shuttle_trace_field_text(state, component, "loc");
    let target_x = shuttle_trace_field_text(state, component, "x");
    let target_y = shuttle_trace_field_text(state, component, "y");
    let target_z = shuttle_trace_field_text(state, component, "z");
    eprintln!(
        "shuttle-trace event={event} component={component:?} type={component_type} note={note} \
device_type={device_type} target_dir={target_dir} target_flipped={target_flipped} \
target_loc={target_location} target_xyz={target_x},{target_y},{target_z}"
    );
    for slot in 1..=slot_count {
        let expected_direction =
            shuttle_trace_expected_slot_direction(target_dir, target_flipped, slot).unwrap_or(-1);
        let node_value = shuttle_trace_list_slot(state, nodes, slot);
        let parent_value = shuttle_trace_list_slot(state, parents, slot);
        let node = match node_value {
            Some(Value::Datum(node)) => Some(node),
            _ => None,
        };
        let parent = match parent_value {
            Some(Value::Datum(parent)) => Some(parent),
            _ => None,
        };
        let node_type = node
            .and_then(|datum| state.heap.datum(datum).ok())
            .map_or_else(
                || "<null>".to_owned(),
                |datum| datum.type_path().to_string(),
            );
        let parent_type = parent
            .and_then(|datum| state.heap.datum(datum).ok())
            .map_or_else(
                || "<null>".to_owned(),
                |datum| datum.type_path().to_string(),
            );
        let node_x = node.map_or_else(
            || "null".to_owned(),
            |datum| shuttle_trace_field_text(state, datum, "x"),
        );
        let node_y = node.map_or_else(
            || "null".to_owned(),
            |datum| shuttle_trace_field_text(state, datum, "y"),
        );
        let node_z = node.map_or_else(
            || "null".to_owned(),
            |datum| shuttle_trace_field_text(state, datum, "z"),
        );
        let node_dir = node.map_or_else(
            || "null".to_owned(),
            |datum| shuttle_trace_field_text(state, datum, "dir"),
        );
        let node_pipe = node.map_or_else(
            || "null".to_owned(),
            |datum| shuttle_trace_field_text(state, datum, "piping_layer"),
        );
        let node_loc = node
            .and_then(|datum| {
                let field = FieldName::parse("loc").expect("built-in atom variable");
                state.heap.datum_field(datum, &field).ok().cloned()
            })
            .map_or_else(
                || "null".to_owned(),
                |value| shuttle_trace_value_ref(Some(&value)),
            );
        eprintln!(
            "shuttle-trace event={event} component={component:?} slot={slot} \
expected_dir={expected_direction} node={node_ref} node_type={node_type} node_x={node_x} node_y={node_y} node_z={node_z} \
node_dir={node_dir} node_pipe={node_pipe} node_loc={node_loc} parent={parent_ref} parent_type={parent_type}",
            component = component,
            event = event,
            slot = slot,
            expected_direction = expected_direction,
            node_ref = shuttle_trace_value_ref(node_value.as_ref()),
            node_type = node_type,
            node_x = node_x,
            node_y = node_y,
            node_z = node_z,
            node_dir = node_dir,
            node_pipe = node_pipe,
            node_loc = node_loc,
            parent_ref = shuttle_trace_value_ref(parent_value.as_ref()),
            parent_type = parent_type
        );
    }
}

pub(crate) fn boot_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_BOOT_TRACE").is_some())
}

/// Wall-time threshold, if any, above which a single interpreter instruction is
/// reported to stderr. Gated by `DREAM64_TRACE_SLOW_INSTRUCTION`: unset disables
/// it, a bare/truthy value uses a 5 ms threshold, and a positive integer sets an
/// explicit millisecond threshold. Diagnostic only.
pub(crate) fn slow_instruction_trace_threshold() -> Option<Duration> {
    static THRESHOLD: OnceLock<Option<Duration>> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        let raw = std::env::var("DREAM64_TRACE_SLOW_INSTRUCTION").ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() || matches!(trimmed, "0" | "off" | "false" | "no") {
            return None;
        }
        let millis = trimmed
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .unwrap_or(5);
        Some(Duration::from_millis(millis))
    })
}

pub(crate) fn boot_dashboard_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_BOOT_DASHBOARD").is_some())
}

pub(crate) fn atoms_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_ATOMS").is_some())
}

pub(crate) fn startup_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_STARTUP").is_some())
}

pub(crate) fn startup_instruction_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("DREAM64_PROFILE_STARTUP_OPCODES").is_some())
}

pub(crate) const ATOMS_INITIALIZE_PATH: &str = "/datum/controller/subsystem/atoms/proc/Initialize";

pub(crate) fn is_atoms_initialize_path(path: &str) -> bool {
    path.split_once('@').map_or(path, |(base, _)| base) == ATOMS_INITIALIZE_PATH
}

pub(crate) fn is_subsystem_initialize_path(path: &str) -> bool {
    let path = path.split_once('@').map_or(path, |(base, _)| base);
    path.strip_prefix("/datum/controller/subsystem/")
        .and_then(|suffix| suffix.rsplit_once("/proc/"))
        .is_some_and(|(owner, selector)| !owner.is_empty() && selector == "Initialize")
}

fn atoms_profile_lines_with_event(profile: &AtomsProfile, event: &str) -> Vec<String> {
    let mut procedures = profile
        .samples
        .keys()
        .chain(profile.frame_entries.keys())
        .copied()
        .collect::<Vec<_>>();
    procedures.sort_unstable_by_key(|procedure| (procedure.module_identity, procedure.procedure));
    procedures.dedup();
    procedures.sort_by(|left, right| {
        let left_samples = profile.samples.get(left).copied().unwrap_or(0);
        let right_samples = profile.samples.get(right).copied().unwrap_or(0);
        let left_entries = profile.frame_entries.get(left).copied().unwrap_or(0);
        let right_entries = profile.frame_entries.get(right).copied().unwrap_or(0);
        right_samples
            .cmp(&left_samples)
            .then_with(|| right_entries.cmp(&left_entries))
            .then_with(|| {
                profile
                    .paths
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(profile.paths.get(right).map_or("<missing>", String::as_str))
            })
    });
    let sample_count = profile.samples.values().copied().sum::<u64>();
    let mut lines = vec![if let Some(root) = &profile.startup_root {
        format!(
            "boot-vm: startup-profile-{event} subsystem={root} elapsed_ms={} total_instructions={} samples={} procedures={}",
            profile.started.elapsed().as_millis(),
            profile.total_instructions,
            sample_count,
            procedures.len(),
        )
    } else {
        format!(
            "boot-vm: atoms-profile-{event} elapsed_ms={} total_instructions={} samples={} procedures={}",
            profile.started.elapsed().as_millis(),
            profile.total_instructions,
            sample_count,
            procedures.len(),
        )
    }];
    if let Some(counts) = profile.instruction_categories {
        let prefix = if let Some(root) = &profile.startup_root {
            format!("boot-vm: startup-profile-opcodes subsystem={root}")
        } else {
            "boot-vm: atoms-profile-opcodes".to_owned()
        };
        lines.push(format!(
            "{prefix} list_read={} list_write={} field_read={} field_write={} call={} branch={} other={}",
            counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6],
        ));
        let mut instructions = profile
            .instruction_samples
            .iter()
            .map(|(instruction, samples)| (*instruction, *samples))
            .collect::<Vec<_>>();
        instructions.sort_by(|(left, left_samples), (right, right_samples)| {
            right_samples.cmp(left_samples).then_with(|| {
                profile
                    .instruction_labels
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(
                        profile
                            .instruction_labels
                            .get(right)
                            .map_or("<missing>", String::as_str),
                    )
            })
        });
        for (rank, (instruction, samples)) in instructions.into_iter().take(30).enumerate() {
            let label = profile
                .instruction_labels
                .get(&instruction)
                .map_or("<missing>", String::as_str);
            lines.push(format!(
                "{prefix} hot_pc_rank={} samples={samples} {label}",
                rank + 1,
            ));
        }
        let mut wall_instructions = profile
            .instruction_wall_nanos
            .iter()
            .map(|(instruction, nanos)| (*instruction, *nanos))
            .collect::<Vec<_>>();
        wall_instructions.sort_by(|(left, left_nanos), (right, right_nanos)| {
            right_nanos.cmp(left_nanos).then_with(|| {
                profile
                    .instruction_labels
                    .get(left)
                    .map_or("<missing>", String::as_str)
                    .cmp(
                        profile
                            .instruction_labels
                            .get(right)
                            .map_or("<missing>", String::as_str),
                    )
            })
        });
        for (rank, (instruction, nanos)) in wall_instructions.into_iter().take(30).enumerate() {
            let label = profile
                .instruction_labels
                .get(&instruction)
                .map_or("<missing>", String::as_str);
            lines.push(format!(
                "{prefix} wall_pc_rank={} sampled_ms={} {label}",
                rank + 1,
                nanos / 1_000_000,
            ));
        }
    }
    let mut wall_samples = profile
        .wall_sample_nanos
        .iter()
        .map(|(procedure, nanos)| (*procedure, *nanos))
        .collect::<Vec<_>>();
    wall_samples.sort_by(|(left, left_nanos), (right, right_nanos)| {
        right_nanos.cmp(left_nanos).then_with(|| {
            profile
                .paths
                .get(left)
                .map_or("<missing>", String::as_str)
                .cmp(profile.paths.get(right).map_or("<missing>", String::as_str))
        })
    });
    for (rank, (procedure, nanos)) in wall_samples.into_iter().take(30).enumerate() {
        let path = profile
            .paths
            .get(&procedure)
            .map_or("<missing>", String::as_str);
        let prefix = if let Some(root) = &profile.startup_root {
            format!("boot-vm: startup-profile-wall subsystem={root}")
        } else {
            "boot-vm: atoms-profile-wall".to_owned()
        };
        lines.push(format!(
            "{prefix} rank={} sampled_ms={} procedure={path}",
            rank + 1,
            nanos / 1_000_000,
        ));
    }
    for (rank, procedure) in procedures.into_iter().take(30).enumerate() {
        let samples = profile.samples.get(&procedure).copied().unwrap_or(0);
        let entries = profile.frame_entries.get(&procedure).copied().unwrap_or(0);
        let path = profile
            .paths
            .get(&procedure)
            .map_or("<missing>", String::as_str);
        lines.push(if let Some(root) = &profile.startup_root {
            format!(
                "boot-vm: startup-profile-rank subsystem={root} rank={} samples={samples} entries={entries} procedure={path}",
                rank + 1,
            )
        } else {
            format!(
                "boot-vm: atoms-profile-rank rank={} samples={samples} entries={entries} procedure={path}",
                rank + 1,
            )
        });
    }
    lines
}

pub(crate) fn atoms_profile_lines(profile: &AtomsProfile) -> Vec<String> {
    atoms_profile_lines_with_event(profile, "summary")
}

pub(crate) fn atoms_profile_snapshot_lines_if_due(
    profile: &mut AtomsProfile,
    now: Instant,
    interval: Duration,
) -> Option<Vec<String>> {
    if now.duration_since(profile.last_snapshot) < interval {
        return None;
    }
    profile.last_snapshot = now;
    Some(atoms_profile_lines_with_event(profile, "snapshot"))
}

pub(crate) fn emit_atoms_profile(profile: &AtomsProfile) {
    for line in atoms_profile_lines(profile) {
        eprintln!("{line}");
    }
}

pub(crate) fn procedure_argument_trace_filter() -> &'static Option<String> {
    static FILTER: OnceLock<Option<String>> = OnceLock::new();
    FILTER.get_or_init(|| std::env::var("DREAM64_TRACE_PROC_ARGS").ok())
}

pub(crate) fn mark_boot_trace_frame(
    frame: &mut CallFrame,
    module: &Module,
    state: &ExecutionState,
    executed_steps: u64,
) {
    let trace_enabled = boot_trace_enabled();
    let argument_trace = procedure_argument_trace_filter();
    if !trace_enabled && !boot_dashboard_enabled() && argument_trace.is_none() {
        return;
    }
    let path = module
        .paths
        .get(frame.procedure.index())
        .map_or("<missing>", String::as_str);
    if argument_trace
        .as_deref()
        .is_some_and(|needle| path.contains(needle))
    {
        eprintln!(
            "boot-vm: proc-arguments path={path} src={} arguments=[{}]",
            boot_trace_describe_value(&frame.src, state),
            frame
                .arguments
                .iter()
                .map(|value| boot_trace_describe_value(value, state))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // Monkestation's title subsystem owns the authoritative startup display:
    // Master reports subsystem begin/end here, and Mapping/Assets use the same
    // API for their indented child rows. Mirror those semantic events into the
    // headless trace so the visible console can render the real checklist
    // instead of guessing progress from procedure materialization.
    if path.contains("/datum/controller/subsystem/title/proc/add_init_text@") {
        eprintln!(
            "boot-vm: init-display|event=add|category={}|name={}|stage={}|seconds={}",
            boot_trace_display_value(frame.arguments.first()),
            boot_trace_display_value(frame.arguments.get(1)),
            boot_trace_display_value(frame.arguments.get(2)),
            boot_trace_display_value(frame.arguments.get(3)),
        );
    } else if path.contains("/datum/controller/subsystem/title/proc/remove_init_text@") {
        eprintln!(
            "boot-vm: init-display|event=remove|category={}",
            boot_trace_display_value(frame.arguments.first()),
        );
    }
    // The polished boot dashboard needs only the authoritative title
    // subsystem events above. Keep the much heavier initializer/heartbeat
    // instrumentation behind DREAM64_BOOT_TRACE so visible production boots
    // can show real progress without paying full-trace overhead.
    if !trace_enabled {
        return;
    }
    let traced = path.contains("/datum/controller/global_vars/proc/InitGlobal")
        || [
            "/proc/make_datum_reference_lists",
            "/proc/init_sprite_accessories",
            "/proc/init_species_list",
            "/proc/init_hair_gradients",
            "/proc/init_keybindings",
            "/proc/init_emote_list",
            "/proc/init_crafting_recipes",
            "/proc/init_crafting_recipes_atoms",
            "/proc/init_religion_sects",
        ]
        .iter()
        .any(|suffix| path.ends_with(suffix));
    if !traced {
        return;
    }
    eprintln!("boot-vm: initializer-begin path={path}");
    let cold = frame.cold_mut();
    cold.boot_trace_started = Some(Instant::now());
    cold.boot_trace_heap = Some((
        state.heap.live_datum_count(),
        state.heap.live_list_count(),
        module.materialized_deferred_procedure_count(),
    ));
    cold.boot_trace_step = executed_steps;
}

fn boot_trace_describe_value(value: &Value, state: &ExecutionState) -> String {
    let Value::Datum(datum) = value else {
        return value.to_string();
    };
    let Ok(record) = state.heap.datum(*datum) else {
        return value.to_string();
    };
    let loc = record
        .field(&FieldName::parse("loc").expect("built-in loc field is valid"))
        .map_or_else(|_| "<unset>".to_owned(), ToString::to_string);
    let contents = record
        .field(&FieldName::parse("contents").expect("built-in contents field is valid"))
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok().map(dm_value::DmList::len),
            _ => None,
        })
        .map_or_else(|| "<unset>".to_owned(), |length| length.to_string());
    format!(
        "{}{{type={},loc={},contents_len={}}}",
        value,
        record.type_path(),
        loc,
        contents
    )
}

fn boot_trace_display_value(value: Option<&Value>) -> String {
    let raw = match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Text(text) | Value::File(text)) => text.to_string(),
        Some(Value::TypePath(path)) => path.to_string(),
        Some(Value::Number(number)) => number.to_f32().to_string(),
        Some(value) => value.to_string(),
    };
    raw.replace(['\r', '\n', '|'], " ")
}
