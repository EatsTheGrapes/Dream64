//! Per-procedure interpreter state: call frames, continuations, and frame construction.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

#[cfg(test)]
use crate::bytecode::Instruction;
use crate::bytecode::{ProcedureId, Program};
use crate::tgm_planner;
use crate::value_ops::{ExecutionContext, value_to_list_index};
use crate::{ExceptionHandler, PendingLocalPrompt, ShuttleTracePostReturn};
#[cfg(test)]
use dm_core::SourceSpan;
use dm_jit::NumericExecutionState;
use dm_value::{DatumId, ListId, PackedValue, TypePath, Value};
use smallvec::SmallVec;

use crate::execution::state::ExecutionState;

#[derive(Clone, Debug)]
pub(crate) struct CallFrame {
    pub(crate) procedure: ProcedureId,
    pub(crate) instruction: usize,
    // Most DM procedures use only a handful of locals and operand slots. Keep
    // those values inside the frame so millions of short startup calls do not
    // each pay for separate locals/stack heap allocations.
    pub(crate) locals: SmallVec<[Value; 8]>,
    pub(crate) stack: SmallVec<[Value; 8]>,
    pub(crate) result: Value,
    pub(crate) src: Value,
    pub(crate) usr: Value,
    // Retain all supplied values for the future DM `args` list, including
    // extras beyond the declared parameter slots.
    // The implicit DM `args` vector is needed for forwarding/default writes even
    // when its list identity is never materialized. Atom initialization calls
    // overwhelmingly supply only a few values, so keep that vector inline too.
    pub(crate) arguments: SmallVec<[Value; 8]>,
    // Materialized view of the special live `args` object. Parameter writes
    // keep its declared positions synchronized with `locals`.
    pub(crate) args_list: Option<ListId>,
    // Cache this per frame: StoreLocal is a dominant startup opcode and must
    // not rescan parameter_names for every ordinary local assignment.
    pub(crate) declared_argument_count: usize,
    pub(crate) supplied_parameters: SmallVec<[bool; 8]>,
    // Rare argument-name, exception, diagnostic, and shuttle continuation
    // state lives outside the hot frame header.
    pub(crate) cold: Option<Box<CallFrameCold>>,
    // A waitfor=FALSE boundary detaches from its caller only once. Later
    // sleeps in the already-detached continuation yield normally.
    pub(crate) detached_waitfor: bool,
    pub(crate) static_locals: SmallVec<[u16; 2]>,
    pub(crate) atoms_profile_entry_counted: bool,
    pub(crate) atoms_profile_root: bool,
    pub(crate) tgm_profile_root: bool,
}

/// Stable, native-width-independent identity for one suspended VM continuation.
///
/// The identity is serialized as an explicit `u64`; it never contains a host
/// pointer or depends on the 32-bit layout of BYOND's native runtime.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct VmContinuationId(pub(crate) u64);

impl VmContinuationId {
    /// Returns the portable integer representation used by snapshots and diagnostics.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OwnedContinuation {
    pub(crate) id: VmContinuationId,
    pub(crate) frames: Vec<CallFrame>,
}

impl OwnedContinuation {
    pub(crate) fn new(id: VmContinuationId, frames: Vec<CallFrame>) -> Self {
        debug_assert!(!frames.is_empty());
        Self { id, frames }
    }

    pub(crate) fn into_frames(self) -> Vec<CallFrame> {
        self.frames
    }
}

impl std::ops::Deref for OwnedContinuation {
    type Target = [CallFrame];

    fn deref(&self) -> &Self::Target {
        &self.frames
    }
}

impl<'a> IntoIterator for &'a OwnedContinuation {
    type Item = &'a CallFrame;
    type IntoIter = std::slice::Iter<'a, CallFrame>;

    fn into_iter(self) -> Self::IntoIter {
        self.frames.iter()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PackedNumericState {
    pub(crate) locals: SmallVec<[PackedValue; 8]>,
    pub(crate) stack: SmallVec<[PackedValue; 8]>,
    pub(crate) result: PackedValue,
}

impl PackedNumericState {
    pub(crate) fn from_rich(frame: &CallFrame) -> Option<Self> {
        fn numeric(value: &Value) -> Option<PackedValue> {
            let packed = PackedValue::try_from_value(value)?;
            packed.as_number_or_null()?;
            Some(packed)
        }
        Some(Self {
            locals: frame.locals.iter().map(numeric).collect::<Option<_>>()?,
            stack: frame.stack.iter().map(numeric).collect::<Option<_>>()?,
            result: numeric(&frame.result)?,
        })
    }

    pub(crate) fn materialize(self, frame: &mut CallFrame) {
        frame.locals.clear();
        frame
            .locals
            .extend(self.locals.into_iter().map(PackedValue::into_value));
        frame.stack.clear();
        frame
            .stack
            .extend(self.stack.into_iter().map(PackedValue::into_value));
        frame.result = self.result.into_value();
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CallFrameCold {
    pub(crate) pending_argument_names: Option<Vec<Option<String>>>,
    // Source lists consumed by arglist expansion. Re-entrant engine execution
    // cannot always see the outer caller's frames, so transfer these roots to
    // the callee until it returns.
    pub(crate) pending_argument_roots: SmallVec<[Value; 2]>,
    pub(crate) retained_call_roots: SmallVec<[Value; 2]>,
    pub(crate) exception_handlers: Vec<ExceptionHandler>,
    pub(crate) shuttle_trace_target: Option<DatumId>,
    pub(crate) shuttle_trace_post_return: Option<ShuttleTracePostReturn>,
    pub(crate) boot_trace_started: Option<Instant>,
    pub(crate) boot_trace_heap: Option<(usize, usize, usize)>,
    pub(crate) boot_trace_step: u64,
    // Engine constructors return the allocated value rather than New()'s
    // result. This is rare outside constructor continuations.
    pub(crate) caller_result_override: Option<Value>,
    // Connection construction may defer Login until the complete New chain.
    pub(crate) engine_post_return: Option<Box<CallFrame>>,
    // Native numeric state exists only while a guarded trace is suspended.
    pub(crate) numeric_jit_state: Option<NumericExecutionState>,
    // Null/number-only interpreter state stays packed across budget yields.
    // Pointer-bearing values side-exit before entering this domain, so the
    // ordinary rich frame remains the complete GC root set.
    pub(crate) packed_numeric_state: Option<PackedNumericState>,
    pub(crate) tgm_load: Option<TgmLoadContinuation>,
    pub(crate) ruin_scan: Option<RuinCandidateScan>,
    pub(crate) tgm_route_trace_mask: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct RuinCandidateScan {
    pub(crate) low: (i32, i32, i32),
    pub(crate) next: (i32, i32, i32),
    pub(crate) high: (i32, i32, i32),
    pub(crate) empty: bool,
    pub(crate) turfs: Vec<DatumId>,
    pub(crate) areas: Vec<Value>,
    pub(crate) validating: bool,
    pub(crate) validate_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TgmLoadPhase {
    Commit,
    AwaitCoordinate,
    Tick,
}

#[derive(Clone, Debug)]
pub(crate) struct TgmLoadContinuation {
    pub(crate) plan: Arc<tgm_planner::Plan>,
    pub(crate) cursor: tgm_planner::CommitCursor,
    pub(crate) phase: TgmLoadPhase,
    pub(crate) model_cache: Value,
    pub(crate) models: BTreeMap<Arc<str>, Value>,
    pub(crate) bounds: Value,
    pub(crate) coordinate_target: Option<(TypePath, ProcedureId)>,
}

impl TgmLoadContinuation {
    pub(crate) fn roots(&self) -> impl Iterator<Item = &Value> {
        std::iter::once(&self.model_cache)
            .chain(std::iter::once(&self.bounds))
            .chain(self.models.values())
    }
}

impl CallFrameCold {
    pub(crate) fn is_empty(&self) -> bool {
        self.pending_argument_names.is_none()
            && self.pending_argument_roots.is_empty()
            && self.retained_call_roots.is_empty()
            && self.exception_handlers.is_empty()
            && self.shuttle_trace_target.is_none()
            && self.shuttle_trace_post_return.is_none()
            && self.boot_trace_started.is_none()
            && self.boot_trace_heap.is_none()
            && self.boot_trace_step == 0
            && self.caller_result_override.is_none()
            && self.engine_post_return.is_none()
            && self.numeric_jit_state.is_none()
            && self.packed_numeric_state.is_none()
            && self.tgm_load.is_none()
            && self.ruin_scan.is_none()
            && self.tgm_route_trace_mask == 0
    }
}

impl CallFrame {
    pub(crate) fn prune_empty_cold(&mut self) {
        if self.cold.as_deref().is_some_and(CallFrameCold::is_empty) {
            self.cold = None;
        }
    }

    pub(crate) fn cold(&self) -> Option<&CallFrameCold> {
        self.cold.as_deref()
    }

    pub(crate) fn cold_mut(&mut self) -> &mut CallFrameCold {
        self.cold
            .get_or_insert_with(|| Box::new(CallFrameCold::default()))
    }

    pub(crate) fn pending_argument_names(&self) -> Option<&Vec<Option<String>>> {
        self.cold()
            .and_then(|cold| cold.pending_argument_names.as_ref())
    }

    pub(crate) fn set_pending_argument_names(&mut self, names: Vec<Option<String>>) {
        self.cold_mut().pending_argument_names = Some(names);
    }

    pub(crate) fn take_pending_argument_names(&mut self) -> Option<Vec<Option<String>>> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.pending_argument_names.take())
    }

    pub(crate) fn clear_pending_argument_names(&mut self) {
        if let Some(cold) = self.cold.as_deref_mut() {
            cold.pending_argument_names = None;
        }
    }

    pub(crate) fn pending_argument_roots(&self) -> &[Value] {
        self.cold()
            .map_or(&[], |cold| cold.pending_argument_roots.as_slice())
    }

    pub(crate) fn set_pending_argument_roots(&mut self, roots: SmallVec<[Value; 2]>) {
        if !roots.is_empty() || self.cold.is_some() {
            self.cold_mut().pending_argument_roots = roots;
        }
    }

    pub(crate) fn take_pending_argument_roots(&mut self) -> SmallVec<[Value; 2]> {
        self.cold.as_deref_mut().map_or_else(SmallVec::new, |cold| {
            std::mem::take(&mut cold.pending_argument_roots)
        })
    }

    pub(crate) fn clear_pending_argument_roots(&mut self) {
        if let Some(cold) = self.cold.as_deref_mut() {
            cold.pending_argument_roots.clear();
        }
    }

    pub(crate) fn retained_call_roots(&self) -> &[Value] {
        self.cold()
            .map_or(&[], |cold| cold.retained_call_roots.as_slice())
    }

    pub(crate) fn set_retained_call_roots(&mut self, roots: SmallVec<[Value; 2]>) {
        if !roots.is_empty() || self.cold.is_some() {
            self.cold_mut().retained_call_roots = roots;
        }
    }

    pub(crate) fn exception_handlers(&self) -> &[ExceptionHandler] {
        self.cold()
            .map_or(&[], |cold| cold.exception_handlers.as_slice())
    }

    pub(crate) fn exception_handlers_mut(&mut self) -> &mut Vec<ExceptionHandler> {
        &mut self.cold_mut().exception_handlers
    }

    pub(crate) fn shuttle_trace_target(&self) -> Option<DatumId> {
        self.cold().and_then(|cold| cold.shuttle_trace_target)
    }

    pub(crate) fn set_shuttle_trace_target(&mut self, target: Option<DatumId>) {
        if target.is_some() || self.cold.is_some() {
            self.cold_mut().shuttle_trace_target = target;
        }
    }

    pub(crate) fn take_shuttle_trace_post_return(&mut self) -> Option<ShuttleTracePostReturn> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.shuttle_trace_post_return.take())
    }

    pub(crate) fn set_shuttle_trace_post_return(&mut self, value: Option<ShuttleTracePostReturn>) {
        if value.is_some() || self.cold.is_some() {
            self.cold_mut().shuttle_trace_post_return = value;
        }
    }

    pub(crate) fn shuttle_trace_post_return(&self) -> Option<&ShuttleTracePostReturn> {
        self.cold()
            .and_then(|cold| cold.shuttle_trace_post_return.as_ref())
    }

    pub(crate) fn caller_result_override(&self) -> Option<&Value> {
        self.cold()
            .and_then(|cold| cold.caller_result_override.as_ref())
    }

    pub(crate) fn set_caller_result_override(&mut self, value: Option<Value>) {
        if value.is_some() || self.cold.is_some() {
            self.cold_mut().caller_result_override = value;
            self.prune_empty_cold();
        }
    }

    pub(crate) fn engine_post_return(&self) -> Option<&CallFrame> {
        self.cold()
            .and_then(|cold| cold.engine_post_return.as_deref())
    }

    pub(crate) fn set_engine_post_return(&mut self, frame: Option<Box<CallFrame>>) {
        if frame.is_some() || self.cold.is_some() {
            self.cold_mut().engine_post_return = frame;
        }
    }

    pub(crate) fn take_engine_post_return(&mut self) -> Option<Box<CallFrame>> {
        let frame = self
            .cold
            .as_deref_mut()
            .and_then(|cold| cold.engine_post_return.take());
        self.prune_empty_cold();
        frame
    }

    pub(crate) fn numeric_jit_state(&self) -> Option<&NumericExecutionState> {
        self.cold().and_then(|cold| cold.numeric_jit_state.as_ref())
    }

    pub(crate) fn numeric_jit_state_mut(&mut self) -> Option<&mut NumericExecutionState> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.numeric_jit_state.as_mut())
    }

    pub(crate) fn set_numeric_jit_state(&mut self, state: Option<NumericExecutionState>) {
        if state.is_some() || self.cold.is_some() {
            self.cold_mut().numeric_jit_state = state;
            self.prune_empty_cold();
        }
    }

    pub(crate) fn take_packed_numeric_state(&mut self) -> Option<PackedNumericState> {
        self.cold
            .as_deref_mut()
            .and_then(|cold| cold.packed_numeric_state.take())
    }

    pub(crate) fn set_packed_numeric_state(&mut self, state: Option<PackedNumericState>) {
        if state.is_some() || self.cold.is_some() {
            self.cold_mut().packed_numeric_state = state;
        }
        self.prune_empty_cold();
    }
}

pub(crate) enum FrameRunOutcome {
    Complete(Value),
    Yielded { frames: Vec<CallFrame>, delay: f32 },
    Prompted { id: u64, prompt: PendingLocalPrompt },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StepBudgetBehavior {
    Error,
    YieldScheduledContinuation,
}
pub(crate) fn make_frame(
    procedure: ProcedureId,
    program: &Program,
    arguments: &[Value],
    context: &ExecutionContext,
) -> CallFrame {
    make_frame_inline(
        procedure,
        program,
        arguments.iter().cloned().collect(),
        context,
    )
}

pub(crate) fn make_frame_owned(
    procedure: ProcedureId,
    program: &Program,
    arguments: impl Into<SmallVec<[Value; 8]>>,
    context: &ExecutionContext,
) -> CallFrame {
    make_frame_inline(procedure, program, arguments.into(), context)
}

fn make_frame_inline(
    procedure: ProcedureId,
    program: &Program,
    mut arguments: SmallVec<[Value; 8]>,
    context: &ExecutionContext,
) -> CallFrame {
    let declared_argument_count = declared_argument_count(program);
    let mut locals = SmallVec::<[Value; 8]>::from_elem(Value::Null, program.local_count);
    let bound_count = arguments
        .len()
        .min(program.parameter_count)
        .min(locals.len());
    locals[..bound_count].clone_from_slice(&arguments[..bound_count]);
    // BYOND's implicit `args` list has one entry for every declared
    // parameter even when the caller omitted it, padded with null, and then
    // retains any extra supplied arguments. This is observable in atom/New:
    // `new /obj` supplies no explicit location, but `/atom/New(loc, ...)`
    // can still assign `args[1]` before forwarding it to Initialize().
    let supplied_argument_count = arguments.len();
    arguments.resize(arguments.len().max(declared_argument_count), Value::Null);
    CallFrame {
        procedure,
        instruction: 0,
        locals,
        stack: SmallVec::new(),
        result: Value::Null,
        src: context.src.clone(),
        usr: context.usr.clone(),
        arguments,
        args_list: None,
        declared_argument_count,
        supplied_parameters: (0..program.parameter_count)
            .map(|index| index < supplied_argument_count)
            .collect(),
        cold: None,
        detached_waitfor: false,
        static_locals: SmallVec::new(),
        atoms_profile_entry_counted: false,
        atoms_profile_root: false,
        tgm_profile_root: false,
    }
}

pub(crate) fn make_frame_named(
    procedure: ProcedureId,
    program: &Program,
    arguments: &[Value],
    argument_names: &[Option<String>],
    context: &ExecutionContext,
) -> CallFrame {
    if argument_names.iter().all(Option::is_none) {
        return make_frame(procedure, program, arguments, context);
    }
    let mut positioned = vec![Value::Null; program.parameter_count];
    let mut extras = Vec::new();
    let mut supplied = vec![false; program.parameter_count];
    let mut next_positional = 0usize;
    for (index, value) in arguments.iter().enumerate() {
        let slot = argument_names
            .get(index)
            .and_then(Option::as_deref)
            .and_then(|name| {
                program
                    .parameter_names
                    .iter()
                    .position(|parameter| parameter == name)
            })
            .unwrap_or_else(|| {
                let slot = next_positional;
                next_positional += 1;
                slot
            });
        if slot < positioned.len() {
            positioned[slot] = value.clone();
            supplied[slot] = true;
        } else {
            extras.push(value.clone());
        }
    }
    let mut frame = make_frame(procedure, program, &positioned, context);
    frame.arguments = positioned[..declared_argument_count(program)]
        .iter()
        .cloned()
        .collect();
    frame.arguments.extend(extras);
    frame.supplied_parameters = supplied.into();
    frame
}

pub(crate) fn declared_argument_count(program: &Program) -> usize {
    // An unnamed trailing `...` reserves a compiler local slot but is not a
    // formal argument. BYOND's `args` pads omitted named parameters only; this
    // is why callback.New(thing, proc, ...) sees length(args) == 2 when no
    // captured callback arguments were supplied.
    program
        .parameter_names
        .iter()
        .rposition(|name| !name.is_empty())
        .map_or(0, |index| index + 1)
}

pub(crate) fn frame_context(frame: &CallFrame) -> ExecutionContext {
    ExecutionContext::new(frame.src.clone(), frame.usr.clone())
}

pub(crate) fn forwarded_frame_arguments(
    frame: &CallFrame,
    program: &Program,
) -> SmallVec<[Value; 8]> {
    // An argumentless self/parent call forwards the live parameter variables,
    // not the original pre-default call vector. Defaults and assignments have
    // already updated the parameter locals by this point. Preserve any extra
    // variadic values after the declared parameter slots.
    let mut arguments = frame.arguments.clone();
    for (index, value) in frame
        .locals
        .iter()
        .take(declared_argument_count(program))
        .enumerate()
    {
        arguments[index] = value.clone();
    }
    arguments
}

pub(crate) fn synchronize_frame_argument_write(
    frame: &mut CallFrame,
    program: &Program,
    key: &Value,
    value: Value,
) -> Result<(), String> {
    let index = value_to_list_index(key)?;
    if index == 0 || index > frame.arguments.len() {
        return Err(format!(
            "DM args position {index} exceeds length {}",
            frame.arguments.len(),
        ));
    }
    let slot = index - 1;
    frame.arguments[slot] = value.clone();
    if slot < declared_argument_count(program)
        && let Some(local) = frame.locals.get_mut(slot)
    {
        *local = value;
    }
    Ok(())
}
#[cfg(test)]
#[test]
#[ignore = "explicit call-frame layout measurement"]
fn call_frame_layout_measurement() {
    eprintln!(
        "call-frame-layout value_bytes={} frame_bytes={} rare_inline_bytes_avoided={}",
        std::mem::size_of::<Value>(),
        std::mem::size_of::<CallFrame>(),
        std::mem::size_of::<Option<Value>>()
            + std::mem::size_of::<Option<Box<CallFrame>>>()
            + std::mem::size_of::<Option<NumericExecutionState>>(),
    );
}

#[cfg(test)]
#[test]
fn call_frame_hot_header_stays_within_one_kibibyte() {
    assert!(
        std::mem::size_of::<CallFrame>() <= 1_024,
        "cold state leaked back into the hot call-frame header"
    );
}

#[cfg(all(test, target_pointer_width = "64"))]
#[test]
fn call_frame_cold_split_reduces_the_hot_header_to_768_bytes() {
    assert_eq!(std::mem::size_of::<CallFrame>(), 768);
    let avoided = std::mem::size_of::<Option<Value>>()
        + std::mem::size_of::<Option<Box<CallFrame>>>()
        + std::mem::size_of::<Option<NumericExecutionState>>();
    assert!(
        avoided >= 200,
        "rare state no longer provides a useful split"
    );
}

#[cfg(test)]
#[test]
fn rare_frame_state_allocates_cold_storage_only_while_live() {
    let program = Program {
        wait_for: true,
        parameter_count: 0,
        parameter_names: Vec::new(),
        verb_parameter_types: Vec::new(),
        verb_name: None,
        local_count: 0,
        source_spans: vec![SourceSpan::new(0, 1)],
        instructions: vec![Instruction::Return],
    };
    let mut frame = make_frame(ProcedureId(0), &program, &[], &ExecutionContext::default());
    assert!(frame.cold.is_none());
    frame.set_caller_result_override(Some(Value::number(7.0)));
    assert_eq!(frame.caller_result_override(), Some(&Value::number(7.0)));
    assert!(frame.cold.is_some());
    frame.set_caller_result_override(None);
    assert!(frame.cold.is_none());
}

pub(crate) trait InlineValueStackExt {
    fn split_off(&mut self, at: usize) -> Vec<Value>;
}

impl InlineValueStackExt for SmallVec<[Value; 8]> {
    fn split_off(&mut self, at: usize) -> Vec<Value> {
        self.drain(at..).collect()
    }
}
pub(crate) fn preserve_reentrant_frame_roots(
    state: &mut ExecutionState,
    frames: &[CallFrame],
) -> usize {
    fn preserve_frame(state: &mut ExecutionState, frame: &CallFrame) {
        state.host_value_roots.extend(frame.locals.iter().cloned());
        state.host_value_roots.extend(frame.stack.iter().cloned());
        state.host_value_roots.push(frame.result.clone());
        state.host_value_roots.push(frame.src.clone());
        state.host_value_roots.push(frame.usr.clone());
        state
            .host_value_roots
            .extend(frame.arguments.iter().cloned());
        state
            .host_value_roots
            .extend(frame.pending_argument_roots().iter().cloned());
        state
            .host_value_roots
            .extend(frame.retained_call_roots().iter().cloned());
        if let Some(tgm) = frame.cold().and_then(|cold| cold.tgm_load.as_ref()) {
            state.host_value_roots.extend(tgm.roots().cloned());
        }
        if let Some(scan) = frame.cold().and_then(|cold| cold.ruin_scan.as_ref()) {
            state
                .host_value_roots
                .extend(scan.turfs.iter().copied().map(Value::Datum));
            state.host_value_roots.extend(scan.areas.iter().cloned());
        }
        if let Some(list) = frame.args_list {
            state.host_value_roots.push(Value::List(list));
        }
        if let Some(value) = frame.caller_result_override() {
            state.host_value_roots.push(value.clone());
        }
        if let Some(frame) = frame.engine_post_return() {
            preserve_frame(state, frame);
        }
    }

    let original_len = state.host_value_roots.len();
    for frame in frames {
        preserve_frame(state, frame);
    }
    original_len
}
