//! Cranelift backend for statically safe Dream64 numeric traces.
//!
//! The first tier intentionally accepts a tiny, closed instruction language.
//! Selection from general DM bytecode belongs in `dm-vm`: if a procedure can
//! observe null, text, heap identity, suspension, dynamic dispatch, or runtime
//! errors, it must remain in the reference interpreter.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use smallvec::SmallVec;
use std::ffi::c_void;
/// One operation in a verified binary32 procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericInstruction {
    /// Push a binary32 constant.
    Constant(f32),
    /// Push a local passed to the trace.
    LoadLocal(u16),
    /// Pop a value into a procedure local.
    StoreLocal(u16),
    /// Push the region's implicit receiver marker. Runtime-inert (codegen
    /// pushes an arbitrary placeholder f32, identically to `Constant`) — the
    /// only thing that matters is that `validate` tracks this stack slot as
    /// `StackKind::Src`, so it type-checks as a receiver for
    /// `LoadFieldDynamic`/`StoreFieldDynamic` and nothing else.
    LoadSrc,
    /// Push a VM-guarded, materialized binary32 field.
    LoadField(u16),
    /// Pop into a materialized field and mark it dirty for VM writeback.
    StoreField(u16),
    /// Pop and discard a receiver placeholder, then push one named field of
    /// the region's implicit `src`, read live through the `load_field_dynamic`
    /// slow-path callback. Unlike `LoadField`, the field is not pre-fetched:
    /// the index addresses a per-region field-name table the VM resolves at
    /// call time, not the flat `fields` array. See the "Milestone 3" module
    /// doc below for why the receiver is always `src` and always implicit.
    LoadFieldDynamic(u16),
    /// Pop a value and a receiver placeholder, then write the value to one
    /// named field of the region's implicit `src` through the
    /// `store_field_dynamic` slow-path callback. The receiver placeholder is
    /// only ever `src` — proven at compile time by kind-tracking through
    /// `validate`, not by adjacency the way `LoadFieldDynamic`'s translator
    /// check is (a store's receiver sits under an arbitrary-length value
    /// expression, not immediately below the store).
    StoreFieldDynamic(u16),
    /// Set a VM-defined deferred action bit, committed after native exit.
    RaiseAction(u8),
    /// Duplicate the top operand.
    Duplicate,
    /// Discard the top operand.
    Pop,
    /// Add the top two stack operands.
    Add,
    /// Subtract the top operand from the preceding operand.
    Subtract,
    /// Multiply the top two stack operands.
    Multiply,
    /// Divide the preceding operand by the top operand.
    Divide,
    /// Negate the top operand.
    Negate,
    /// DM truth-value negation: `1.0` when the operand is `0.0`, else `0.0`.
    Not,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    /// Eager boolean conjunction: `1.0` when both operands are non-zero.
    And,
    /// Eager boolean disjunction: `1.0` when either operand is non-zero.
    Or,
    /// Continue execution at an absolute instruction index.
    Jump(u32),
    /// Pop a number and jump when it is zero (DM false for this numeric tier).
    JumpIfFalse(u32),
    /// Return the top stack value.
    Return,
}

/// Failure to validate or compile a numeric trace.
#[derive(Debug)]
pub enum CompileError {
    /// An instruction reads a local outside the declared input vector.
    InvalidLocal(u16),
    /// A field operation addresses outside the guarded field vector.
    InvalidField(u16),
    /// Dirty writeback currently uses one native mask.
    TooManyFields(usize),
    InvalidAction(u8),
    /// A branch points outside the procedure.
    InvalidTarget(u32),
    /// An instruction requires more operands than the trace has produced.
    StackUnderflow,
    /// A trace must finish with exactly one result.
    InvalidResultStack(usize),
    /// Two control-flow paths disagree about operand-stack shape.
    InconsistentStack {
        instruction: usize,
        first: usize,
        second: usize,
    },
    /// An instruction's operand is the wrong kind — for example, arithmetic
    /// on the region's implicit `src` marker (produced only by a translated
    /// `LoadSrc`, valid only as `LoadFieldDynamic`/`StoreFieldDynamic`'s
    /// receiver) rather than a number.
    InvalidOperandKind(usize),
    /// Two control-flow paths agree on operand-stack *depth* at a merge
    /// point but disagree about which slots hold `src` versus a number.
    InconsistentOperandKind(usize),
    /// Cranelift rejected the generated module.
    Backend(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLocal(local) => {
                write!(formatter, "numeric trace reads invalid local {local}")
            }
            Self::InvalidField(field) => {
                write!(formatter, "numeric trace reads invalid field {field}")
            }
            Self::TooManyFields(count) => {
                write!(formatter, "numeric trace has {count} fields, maximum is 64")
            }
            Self::InvalidAction(action) => {
                write!(formatter, "numeric trace uses invalid action bit {action}")
            }
            Self::InvalidTarget(target) => write!(
                formatter,
                "numeric trace jumps to invalid instruction {target}"
            ),
            Self::StackUnderflow => formatter.write_str("numeric trace stack underflow"),
            Self::InvalidResultStack(depth) => {
                write!(
                    formatter,
                    "numeric trace ends with stack depth {depth}, expected one"
                )
            }
            Self::InconsistentStack {
                instruction,
                first,
                second,
            } => write!(
                formatter,
                "numeric trace reaches instruction {instruction} with stack depths {first} and {second}",
            ),
            Self::InvalidOperandKind(instruction) => write!(
                formatter,
                "numeric trace instruction {instruction} has an operand of the wrong kind"
            ),
            Self::InconsistentOperandKind(instruction) => write!(
                formatter,
                "numeric trace reaches instruction {instruction} with disagreeing operand kinds"
            ),
            Self::Backend(message) => write!(formatter, "Cranelift backend failed: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

type NumericEntry = unsafe extern "C" fn(
    *mut f32,
    *mut f32,
    *mut f32,
    *mut u64,
    *mut u64,
    u32,
    u64,
    *mut c_void,
) -> u64;

// Native stack stores deliberately land in a heap allocation with a checked
// redzone.  Keeping this buffer inline would place an unchecked Cranelift store
// next to the owning VM CallFrame, so a backend/verifier defect could corrupt
// live DM Values before Rust regained control.
const STACK_REDZONE_WORDS: usize = 16;
const STACK_REDZONE_BITS: u32 = 0x7fc0_d64a;

fn numeric_stack_storage(depth: usize) -> SmallVec<[f32; 16]> {
    let logical_depth = depth.max(1);
    let mut stack = SmallVec::with_capacity(logical_depth + STACK_REDZONE_WORDS);
    stack.resize(logical_depth, 0.0);
    stack.extend(std::iter::repeat_n(
        f32::from_bits(STACK_REDZONE_BITS),
        STACK_REDZONE_WORDS,
    ));
    debug_assert!(
        stack.spilled(),
        "native operand stack must be heap isolated"
    );
    stack
}

#[derive(Clone, Debug, PartialEq)]
pub struct NumericExecutionState {
    pub locals: SmallVec<[f32; 8]>,
    pub stack: SmallVec<[f32; 16]>,
    /// Guarded numeric snapshots supplied by the VM. These are never heap pointers.
    pub fields: SmallVec<[f32; 8]>,
    /// Fields stored by native execution and requiring VM writeback at the exit.
    pub dirty_fields: u64,
    /// VM-defined deferred work requested by the trace (for example enqueueing an update).
    pub action_bits: u64,
    pub instruction: u32,
}

impl NumericExecutionState {
    /// Reports whether value snapshots remain inline. The native operand stack
    /// is intentionally excluded because it is heap-isolated behind a redzone.
    #[must_use]
    pub fn is_fully_inline(&self) -> bool {
        !self.locals.spilled() && !self.fields.spilled()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericRunOutcome {
    Returned {
        value: f32,
        steps: u32,
    },
    /// The trace ran out of its step budget mid-execution. Resuming with the
    /// same state (more budget) continues this same native run — the trace's
    /// own control-flow position is exactly what it was.
    BudgetExhausted {
        instruction: u32,
        steps: u32,
    },
    /// A `LoadFieldDynamic` callback declined (the field isn't a guarded
    /// number, or any other reason the VM needs the interpreter for). Unlike
    /// `BudgetExhausted`, retrying `run_budgeted` from this same `state` is
    /// not expected to make different progress — the caller should hand the
    /// rest of this call to the interpreter rather than resume native
    /// execution. `instruction` is exactly where the interpreter must
    /// continue.
    SideExit {
        instruction: u32,
        steps: u32,
    },
}

/// One VM-owned rooted-value block dispatcher. Native code never interprets
/// the values behind slot IDs and never retains a heap pointer across exit.
pub type RootedBlockDispatcher = unsafe extern "C" fn(
    context: *mut c_void,
    roots: *mut u32,
    root_count: u32,
    stack: *mut u32,
    stack_len: *mut u32,
    stack_capacity: u32,
    start_pc: u32,
    budget: u32,
) -> u64;

/// Exact result of a rooted-value block. The dispatcher materializes roots and
/// operand-stack slot IDs before returning every status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootedBlockOutcome {
    Completed { instruction: u32, steps: u16 },
    BudgetExhausted { instruction: u32, steps: u16 },
    SideExit { instruction: u32, steps: u16 },
    RuntimeError { instruction: u32, steps: u16 },
}

type RootedBlockEntry =
    unsafe extern "C" fn(*mut c_void, *mut u32, u32, *mut u32, *mut u32, u32, u32, u32) -> u64;

/// Cranelift entry stub for one VM-verified rooted-value basic block. The
/// dispatcher processes the whole block in one call; this is deliberately not
/// a per-op callback trampoline.
pub struct CompiledRootedBlock {
    module: JITModule,
    entry: RootedBlockEntry,
}

type SafeRootedCallback<'a> =
    dyn FnMut(&mut [u32], &mut [u32], &mut usize, u32, u32) -> RootedBlockOutcome + 'a;

struct SafeRootedDispatch<'a> {
    callback: &'a mut SafeRootedCallback<'a>,
}

impl CompiledRootedBlock {
    /// Runs one native batch entry while keeping all raw-pointer handling in
    /// this backend crate. The VM sees only rooted slot slices and a fixed
    /// stack buffer plus its initialized length, so dispatch cannot reallocate
    /// storage behind the native entry's pointer.
    pub fn run_with<'a>(
        &self,
        roots: &mut [u32],
        stack: &mut Vec<u32>,
        start_pc: u32,
        budget: u32,
        dispatch: &'a mut (
                    dyn FnMut(&mut [u32], &mut [u32], &mut usize, u32, u32) -> RootedBlockOutcome
                        + 'a
                ),
    ) -> RootedBlockOutcome {
        let mut context = SafeRootedDispatch { callback: dispatch };
        // SAFETY: `safe_rooted_dispatch` interprets this context as the exact
        // local `SafeDispatch` type and the call cannot outlive this frame.
        unsafe {
            self.run(
                (&mut context as *mut SafeRootedDispatch<'_>).cast(),
                roots,
                stack,
                start_pc,
                budget,
            )
        }
    }

    /// Executes with externally rooted slots and a materialized operand stack.
    ///
    /// # Safety
    /// `context` must satisfy the dispatcher contract supplied at compilation.
    pub unsafe fn run(
        &self,
        context: *mut c_void,
        roots: &mut [u32],
        stack: &mut Vec<u32>,
        start_pc: u32,
        budget: u32,
    ) -> RootedBlockOutcome {
        let _keep_code_alive = &self.module;
        let mut stack_len = u32::try_from(stack.len()).unwrap_or(u32::MAX);
        let packed = unsafe {
            (self.entry)(
                context,
                roots.as_mut_ptr(),
                u32::try_from(roots.len()).unwrap_or(u32::MAX),
                stack.as_mut_ptr(),
                &mut stack_len,
                u32::try_from(stack.capacity()).unwrap_or(u32::MAX),
                start_pc,
                budget,
            )
        };
        let new_len = usize::try_from(stack_len).unwrap_or(stack.capacity());
        assert!(
            new_len <= stack.capacity(),
            "rooted dispatcher exceeded stack capacity"
        );
        unsafe { stack.set_len(new_len) };
        let instruction = packed as u32;
        let steps = (packed >> 32) as u16;
        match (packed >> 56) as u8 {
            0 => RootedBlockOutcome::Completed { instruction, steps },
            1 => RootedBlockOutcome::BudgetExhausted { instruction, steps },
            2 => RootedBlockOutcome::SideExit { instruction, steps },
            _ => RootedBlockOutcome::RuntimeError { instruction, steps },
        }
    }
}

unsafe extern "C" fn safe_rooted_dispatch(
    context: *mut c_void,
    roots: *mut u32,
    root_count: u32,
    stack: *mut u32,
    stack_len: *mut u32,
    stack_capacity: u32,
    start_pc: u32,
    budget: u32,
) -> u64 {
    let context = unsafe { &mut *context.cast::<SafeRootedDispatch<'_>>() };
    let roots = unsafe { std::slice::from_raw_parts_mut(roots, root_count as usize) };
    let len = unsafe { *stack_len } as usize;
    let capacity = stack_capacity as usize;
    if len > capacity {
        return (3_u64 << 56) | start_pc as u64;
    }
    let materialized = unsafe { std::slice::from_raw_parts_mut(stack, capacity) };
    let mut new_len = len;
    let outcome = (context.callback)(roots, materialized, &mut new_len, start_pc, budget);
    if new_len > capacity {
        return (3_u64 << 56) | start_pc as u64;
    }
    unsafe {
        *stack_len = new_len as u32;
    }
    let (status, instruction, steps) = match outcome {
        RootedBlockOutcome::Completed { instruction, steps } => (0, instruction, steps),
        RootedBlockOutcome::BudgetExhausted { instruction, steps } => (1, instruction, steps),
        RootedBlockOutcome::SideExit { instruction, steps } => (2, instruction, steps),
        RootedBlockOutcome::RuntimeError { instruction, steps } => (3, instruction, steps),
    };
    (status << 56) | (u64::from(steps) << 32) | u64::from(instruction)
}

/// Compiles a rooted block whose dispatcher is supplied safely at execution.
pub fn compile_safe_rooted_block() -> Result<CompiledRootedBlock, CompileError> {
    compile_rooted_block(safe_rooted_dispatch)
}

/// Compiles a single-call native block entry around a VM-owned batch dispatcher.
/// The callback must materialize state on every return and charge before each
/// logical operation.
pub fn compile_rooted_block(
    dispatcher: RootedBlockDispatcher,
) -> Result<CompiledRootedBlock, CompileError> {
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    builder.symbol("dream64_rooted_block_dispatch", dispatcher as *const u8);
    let mut module = JITModule::new(builder);
    let mut dispatcher_signature = module.make_signature();
    for ty in [
        types::I64,
        types::I64,
        types::I32,
        types::I64,
        types::I64,
        types::I32,
        types::I32,
        types::I32,
    ] {
        dispatcher_signature.params.push(AbiParam::new(ty));
    }
    dispatcher_signature.returns.push(AbiParam::new(types::I64));
    let dispatcher_id = module
        .declare_function(
            "dream64_rooted_block_dispatch",
            Linkage::Import,
            &dispatcher_signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let mut context = module.make_context();
    context.func.signature = dispatcher_signature;
    let function_id = module
        .declare_function(
            "dream64_rooted_block",
            Linkage::Local,
            &context.func.signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let dispatcher_ref = module.declare_func_in_func(dispatcher_id, &mut context.func);
    let mut frontend_context = FunctionBuilderContext::new();
    {
        let mut function = FunctionBuilder::new(&mut context.func, &mut frontend_context);
        let block = function.create_block();
        function.append_block_params_for_function_params(block);
        function.switch_to_block(block);
        let args = function.block_params(block).to_vec();
        let call = function.ins().call(dispatcher_ref, &args);
        let result = function.inst_results(call)[0];
        function.ins().return_(&[result]);
        function.seal_all_blocks();
        function.finalize();
    }
    module
        .define_function(function_id, &mut context)
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    module.clear_context(&mut context);
    module
        .finalize_definitions()
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let entry = module.get_finalized_function(function_id);
    let entry = unsafe { std::mem::transmute::<*const u8, RootedBlockEntry>(entry) };
    Ok(CompiledRootedBlock { module, entry })
}

/// Executable native code for one verified numeric trace.
///
/// The owning module keeps the executable allocation alive. It is intentionally
/// not `Clone`, and invocation checks the local vector before entering native code.
pub struct CompiledNumericTrace {
    _module: JITModule,
    entry: NumericEntry,
    local_count: usize,
    instruction_count: usize,
    max_stack_depth: usize,
    reachable: Vec<bool>,
    field_count: usize,
}

impl CompiledNumericTrace {
    /// Executes the trace for exactly the local vector shape used at compilation.
    #[must_use]
    pub fn run(&self, locals: &[f32]) -> Option<f32> {
        if locals.len() != self.local_count {
            return None;
        }
        // SAFETY: `entry` comes from a finalized Cranelift function with the
        // exact `(pointer) -> f32` ABI. The module is retained by `self`, and
        // `locals` remains live and contains the validated number of elements.
        let mut state = self.initial_state(locals)?;
        match self.run_budgeted(&mut state, u32::MAX, &mut |_| None, &mut |_, _| false)? {
            NumericRunOutcome::Returned { value, .. } => Some(value),
            NumericRunOutcome::BudgetExhausted { .. } | NumericRunOutcome::SideExit { .. } => None,
        }
    }

    #[must_use]
    pub fn initial_state(&self, locals: &[f32]) -> Option<NumericExecutionState> {
        (locals.len() == self.local_count && self.field_count == 0).then(|| NumericExecutionState {
            locals: locals.iter().copied().collect(),
            stack: numeric_stack_storage(self.max_stack_depth),
            fields: SmallVec::new(),
            dirty_fields: 0,
            action_bits: 0,
            instruction: 0,
        })
    }

    /// Creates state after the VM has guarded every receiver, field binding,
    /// and initial value as binary32. Heap access cannot occur while native code runs.
    #[must_use]
    pub fn initial_state_with_fields(
        &self,
        locals: &[f32],
        fields: &[f32],
    ) -> Option<NumericExecutionState> {
        (locals.len() == self.local_count && fields.len() == self.field_count).then(|| {
            NumericExecutionState {
                locals: locals.iter().copied().collect(),
                stack: numeric_stack_storage(self.max_stack_depth),
                fields: fields.iter().copied().collect(),
                dirty_fields: 0,
                action_bits: 0,
                instruction: 0,
            }
        })
    }

    /// Runs at most `max_steps` bytecode instructions and leaves locals, operand
    /// stack, and the exact resume PC materialized in `state` on budget exit.
    ///
    /// `load_field` answers a `LoadFieldDynamic(field_index)` instruction with
    /// the current guarded numeric value of that field on the region's
    /// implicit `src`, or `None` to side-exit. `store_field` answers a
    /// `StoreFieldDynamic(field_index, value)` instruction with whether the
    /// guarded write succeeded; on `false` the trace also leaves the value
    /// that would have been stored in `state.stack[0]` (the same slot
    /// `Returned` uses), since the interpreter resuming this exact
    /// `StoreField` needs it rematerialized — see `try_run_region_numeric_jit`.
    /// Traces that never lower either instruction still take both
    /// parameters — they are simply never called — so callers with nothing
    /// to answer can pass `&mut |_| None` / `&mut |_, _| false`.
    pub fn run_budgeted(
        &self,
        state: &mut NumericExecutionState,
        max_steps: u32,
        load_field: &mut dyn FnMut(u32) -> Option<f32>,
        store_field: &mut dyn FnMut(u32, f32) -> bool,
    ) -> Option<NumericRunOutcome> {
        let redzone_start = self.max_stack_depth.max(1);
        if state.locals.len() != self.local_count
            || state.stack.len() != self.max_stack_depth.max(1) + STACK_REDZONE_WORDS
            || state.fields.len() != self.field_count
            || usize::try_from(state.instruction).ok()? >= self.instruction_count
            || !self.reachable[usize::try_from(state.instruction).ok()?]
            || state.stack[redzone_start..]
                .iter()
                .any(|word| word.to_bits() != STACK_REDZONE_BITS)
        {
            return None;
        }
        let mut dispatch = SafeFieldDispatch {
            load_field,
            store_field,
        };
        // SAFETY: `dispatch` outlives the call below (it is not returned or
        // stored), and `safe_load_field_dynamic`/`safe_store_field_dynamic`
        // — the only functions this module's compiled code can call through
        // `context_pointer` — both cast it back to this exact type.
        let context_pointer = (&mut dispatch as *mut SafeFieldDispatch<'_>).cast();
        let packed = unsafe {
            (self.entry)(
                state.locals.as_mut_ptr(),
                state.stack.as_mut_ptr(),
                state.fields.as_mut_ptr(),
                &mut state.dirty_fields,
                &mut state.action_bits,
                state.instruction,
                u64::from(max_steps),
                context_pointer,
            )
        };
        if state.stack[redzone_start..]
            .iter()
            .any(|word| word.to_bits() != STACK_REDZONE_BITS)
        {
            // Never re-enter code that wrote outside its verified operand
            // stack. The allocation boundary kept the VM frame untouched.
            return None;
        }
        let raw_instruction = packed as u32;
        let steps = (packed >> 32) as u32;
        if raw_instruction == u32::MAX {
            Some(NumericRunOutcome::Returned {
                value: state.stack[0],
                steps,
            })
        } else if raw_instruction & 0x8000_0000 != 0 {
            let instruction = raw_instruction & 0x7fff_ffff;
            state.instruction = instruction;
            Some(NumericRunOutcome::SideExit { instruction, steps })
        } else {
            state.instruction = raw_instruction;
            Some(NumericRunOutcome::BudgetExhausted {
                instruction: raw_instruction,
                steps,
            })
        }
    }
}

/// Compiles a verified binary32 stack procedure, including control flow, to native code.
///
/// Returning an error is the normal signal for the caller to retain interpreter
/// execution; compilation never weakens general DM semantics.
pub fn compile_numeric_trace(
    instructions: &[NumericInstruction],
    local_count: usize,
) -> Result<CompiledNumericTrace, CompileError> {
    compile_numeric_field_trace(instructions, local_count, 0, 0)
}

/// Context wrapper for `run_budgeted`'s `load_field`/`store_field` closures.
/// A thin, pointer-sized struct so the fat `&mut dyn FnMut` references can
/// cross the FFI boundary as a single `*mut c_void` — the same reason
/// `SafeRootedDispatch` exists for `CompiledRootedBlock`. One combined
/// struct, not two: both `dream64_load_field_dynamic` and
/// `dream64_store_field_dynamic` are registered against the *same*
/// `context_pointer` parameter of the compiled function (there is only one),
/// so they must agree on what type is behind it.
struct SafeFieldDispatch<'a> {
    load_field: &'a mut dyn FnMut(u32) -> Option<f32>,
    store_field: &'a mut dyn FnMut(u32, f32) -> bool,
}

/// Trampoline registered as `dream64_load_field_dynamic` in every compiled
/// numeric-field trace. Packs the answer as `(1 << 32) | value.to_bits()` on
/// success (so the packed value is always `>= 2^32`) or `0` on a declined
/// field (the trace side-exits and the interpreter resumes this instruction).
unsafe extern "C" fn safe_load_field_dynamic(context: *mut c_void, field_index: u32) -> u64 {
    let context = unsafe { &mut *context.cast::<SafeFieldDispatch<'_>>() };
    match (context.load_field)(field_index) {
        Some(value) => 0x1_0000_0000_u64 | u64::from(value.to_bits()),
        None => 0,
    }
}

/// Trampoline registered as `dream64_store_field_dynamic`. Returns `1` on a
/// guarded write, `0` on decline (the trace side-exits and the interpreter
/// resumes this instruction, receiver and value both still to be
/// materialized by the VM side — see `StoreFieldDynamic`'s codegen).
unsafe extern "C" fn safe_store_field_dynamic(
    context: *mut c_void,
    field_index: u32,
    value_bits: u32,
) -> u64 {
    let context = unsafe { &mut *context.cast::<SafeFieldDispatch<'_>>() };
    let value = f32::from_bits(value_bits);
    u64::from((context.store_field)(field_index, value))
}

/// Compiles a trace over guarded numeric field snapshots. The VM validates and
/// materializes fields before entry, then commits `dirty_fields` at every native exit.
pub fn compile_numeric_field_trace(
    instructions: &[NumericInstruction],
    local_count: usize,
    field_count: usize,
    dynamic_field_count: usize,
) -> Result<CompiledNumericTrace, CompileError> {
    if field_count > 64 {
        return Err(CompileError::TooManyFields(field_count));
    }
    let validation = validate(instructions, local_count, field_count, dynamic_field_count)?;

    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    builder.symbol(
        "dream64_load_field_dynamic",
        safe_load_field_dynamic as *const u8,
    );
    builder.symbol(
        "dream64_store_field_dynamic",
        safe_store_field_dynamic as *const u8,
    );
    let mut module = JITModule::new(builder);
    let mut load_field_dynamic_signature = module.make_signature();
    load_field_dynamic_signature
        .params
        .push(AbiParam::new(types::I64));
    load_field_dynamic_signature
        .params
        .push(AbiParam::new(types::I32));
    load_field_dynamic_signature
        .returns
        .push(AbiParam::new(types::I64));
    let load_field_dynamic_id = module
        .declare_function(
            "dream64_load_field_dynamic",
            Linkage::Import,
            &load_field_dynamic_signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let mut store_field_dynamic_signature = module.make_signature();
    for ty in [types::I64, types::I32, types::I32] {
        store_field_dynamic_signature.params.push(AbiParam::new(ty));
    }
    store_field_dynamic_signature
        .returns
        .push(AbiParam::new(types::I64));
    let store_field_dynamic_id = module
        .declare_function(
            "dream64_store_field_dynamic",
            Linkage::Import,
            &store_field_dynamic_signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let mut context = module.make_context();
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I32));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .params
        .push(AbiParam::new(types::I64));
    context
        .func
        .signature
        .returns
        .push(AbiParam::new(types::I64));
    let function = module
        .declare_function(
            "dream64_numeric_trace",
            Linkage::Local,
            &context.func.signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let load_field_dynamic_ref =
        module.declare_func_in_func(load_field_dynamic_id, &mut context.func);
    let store_field_dynamic_ref =
        module.declare_func_in_func(store_field_dynamic_id, &mut context.func);

    let mut frontend_context = FunctionBuilderContext::new();
    {
        let mut function_builder = FunctionBuilder::new(&mut context.func, &mut frontend_context);
        let entry_block = function_builder.create_block();
        function_builder.append_block_params_for_function_params(entry_block);
        let params = function_builder.block_params(entry_block).to_vec();
        let locals_pointer = params[0];
        let stack_pointer = params[1];
        let fields_pointer = params[2];
        let dirty_pointer = params[3];
        let action_pointer = params[4];
        let resume_pc = params[5];
        let budget = params[6];
        let context_pointer = params[7];
        let checks: Vec<_> = instructions
            .iter()
            .map(|_| function_builder.create_block())
            .collect();
        let bodies: Vec<_> = instructions
            .iter()
            .map(|_| function_builder.create_block())
            .collect();
        let exits: Vec<_> = instructions
            .iter()
            .map(|_| function_builder.create_block())
            .collect();
        for block in checks.iter().chain(bodies.iter()).chain(exits.iter()) {
            function_builder.append_block_param(*block, types::I64);
        }
        function_builder.switch_to_block(entry_block);
        let zero_steps = function_builder.ins().iconst(types::I64, 0);
        let reachable: Vec<_> = validation
            .depths
            .iter()
            .enumerate()
            .filter_map(|(pc, depth)| depth.map(|_| pc))
            .collect();
        let dispatches: Vec<_> = reachable
            .iter()
            .map(|_| function_builder.create_block())
            .collect();
        function_builder.ins().jump(dispatches[0], &[]);
        for (dispatch_index, dispatch) in dispatches.iter().copied().enumerate() {
            let pc = reachable[dispatch_index];
            function_builder.switch_to_block(dispatch);
            let expected = function_builder.ins().iconst(types::I32, pc as i64);
            let matches = function_builder
                .ins()
                .icmp(IntCC::Equal, resume_pc, expected);
            if dispatch_index + 1 < dispatches.len() {
                function_builder.ins().brif(
                    matches,
                    checks[pc],
                    &[cranelift_codegen::ir::BlockArg::Value(zero_steps)],
                    dispatches[dispatch_index + 1],
                    &[],
                );
            } else {
                function_builder.ins().jump(
                    checks[pc],
                    &[cranelift_codegen::ir::BlockArg::Value(zero_steps)],
                );
            }
        }
        function_builder.seal_block(entry_block);

        for (pc, instruction) in instructions.iter().enumerate() {
            if validation.depths[pc].is_none() {
                continue;
            }
            function_builder.switch_to_block(checks[pc]);
            let steps = function_builder.block_params(checks[pc])[0];
            let exhausted =
                function_builder
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, steps, budget);
            function_builder.ins().brif(
                exhausted,
                exits[pc],
                &[cranelift_codegen::ir::BlockArg::Value(steps)],
                bodies[pc],
                &[cranelift_codegen::ir::BlockArg::Value(steps)],
            );

            function_builder.switch_to_block(exits[pc]);
            let steps = function_builder.block_params(exits[pc])[0];
            let packed = pack_exit(&mut function_builder, pc as u32, steps);
            function_builder.ins().return_(&[packed]);

            function_builder.switch_to_block(bodies[pc]);
            let steps = function_builder.block_params(bodies[pc])[0];
            let next_steps = function_builder.ins().iadd_imm(steps, 1);
            let mut depth = validation.depths[pc].expect("reachable instruction");
            match *instruction {
                NumericInstruction::Constant(value) => {
                    let value = function_builder.ins().f32const(value);
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::LoadSrc => {
                    // Runtime-inert: only `validate`'s `StackKind::Src`
                    // tracking gives this meaning. The bit pattern is never
                    // read — `LoadFieldDynamic` discards it unconditionally
                    // and `StoreFieldDynamic` only checks (at compile time)
                    // that a slot popped here traces back to `LoadSrc`.
                    let value = function_builder.ins().f32const(0.0);
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::LoadLocal(local) => {
                    let value =
                        memory_load(&mut function_builder, locals_pointer, usize::from(local));
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::StoreLocal(local) => {
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    memory_store(
                        &mut function_builder,
                        locals_pointer,
                        usize::from(local),
                        value,
                    );
                }
                NumericInstruction::LoadField(field) => {
                    let value =
                        memory_load(&mut function_builder, fields_pointer, usize::from(field));
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::StoreField(field) => {
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    memory_store(
                        &mut function_builder,
                        fields_pointer,
                        usize::from(field),
                        value,
                    );
                    let dirty = function_builder.ins().load(
                        types::I64,
                        cranelift_codegen::ir::MemFlags::trusted(),
                        dirty_pointer,
                        0,
                    );
                    let mask = function_builder
                        .ins()
                        .iconst(types::I64, (1_u64 << field) as i64);
                    let dirty = function_builder.ins().bor(dirty, mask);
                    function_builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        dirty,
                        dirty_pointer,
                        0,
                    );
                }
                NumericInstruction::LoadFieldDynamic(field) => {
                    // The popped value is a placeholder the VM-side translator
                    // pushed in place of the bytecode's `LoadSrc` (see
                    // `numeric_trace_instructions`); the actual receiver is
                    // always this region's implicit `src`, threaded through
                    // `context_pointer` rather than the operand stack. See the
                    // "Milestone 3" module doc below for why this is sound.
                    let _ = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let field_index = function_builder.ins().iconst(types::I32, i64::from(field));
                    let call = function_builder
                        .ins()
                        .call(load_field_dynamic_ref, &[context_pointer, field_index]);
                    let packed = function_builder.inst_results(call)[0];
                    // Packing convention (`safe_load_field_dynamic`): success
                    // sets bit 32 and carries the f32 bits in the low 32 bits,
                    // so the packed value is >= 2^32 iff the callback found a
                    // guarded numeric field; failure is exactly 0.
                    let failed = function_builder.ins().icmp_imm(
                        IntCC::UnsignedLessThan,
                        packed,
                        0x1_0000_0000_i64,
                    );
                    let declined = function_builder.create_block();
                    let loaded = function_builder.create_block();
                    function_builder
                        .ins()
                        .brif(failed, declined, &[], loaded, &[]);

                    function_builder.switch_to_block(declined);
                    function_builder.seal_block(declined);
                    let side_exit = pack_side_exit(&mut function_builder, pc as u32, steps);
                    function_builder.ins().return_(&[side_exit]);

                    function_builder.switch_to_block(loaded);
                    function_builder.seal_block(loaded);
                    let value_bits = function_builder.ins().ireduce(types::I32, packed);
                    let value = function_builder.ins().bitcast(
                        types::F32,
                        cranelift_codegen::ir::MemFlags::new(),
                        value_bits,
                    );
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::StoreFieldDynamic(field) => {
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    // Discard the receiver placeholder for the same reason
                    // `LoadFieldDynamic` does: the real receiver is always
                    // this region's implicit `src`, proven by `validate`'s
                    // `StackKind` tracking, not carried through this slot.
                    let _ = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let field_index = function_builder.ins().iconst(types::I32, i64::from(field));
                    let value_bits = function_builder.ins().bitcast(
                        types::I32,
                        cranelift_codegen::ir::MemFlags::new(),
                        value,
                    );
                    let call = function_builder.ins().call(
                        store_field_dynamic_ref,
                        &[context_pointer, field_index, value_bits],
                    );
                    let result = function_builder.inst_results(call)[0];
                    let failed = function_builder.ins().icmp_imm(IntCC::Equal, result, 0);
                    let declined = function_builder.create_block();
                    let stored = function_builder.create_block();
                    function_builder
                        .ins()
                        .brif(failed, declined, &[], stored, &[]);

                    function_builder.switch_to_block(declined);
                    function_builder.seal_block(declined);
                    // Stash the value that would have been stored in stack
                    // slot 0 — the VM side re-materializes it from
                    // `state.stack[0]` onto `frame.stack` before letting the
                    // interpreter redo this exact `StoreField`
                    // (`try_run_region_numeric_jit`). Safe to clobber slot 0
                    // unconditionally: the whole native operand stack is
                    // abandoned the moment this trace side-exits, exactly
                    // like `Return`'s use of the same slot below.
                    memory_store(&mut function_builder, stack_pointer, 0, value);
                    let side_exit = pack_side_exit(&mut function_builder, pc as u32, steps);
                    function_builder.ins().return_(&[side_exit]);

                    function_builder.switch_to_block(stored);
                    function_builder.seal_block(stored);
                }
                NumericInstruction::RaiseAction(action) => {
                    let actions = function_builder.ins().load(
                        types::I64,
                        cranelift_codegen::ir::MemFlags::trusted(),
                        action_pointer,
                        0,
                    );
                    let mask = function_builder
                        .ins()
                        .iconst(types::I64, (1_u64 << action) as i64);
                    let actions = function_builder.ins().bor(actions, mask);
                    function_builder.ins().store(
                        cranelift_codegen::ir::MemFlags::trusted(),
                        actions,
                        action_pointer,
                        0,
                    );
                }
                NumericInstruction::Duplicate => {
                    let value = memory_load(&mut function_builder, stack_pointer, depth - 1);
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::Pop => {
                    let _ = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                }
                NumericInstruction::Negate => {
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let value = function_builder.ins().fneg(value);
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::Not => {
                    // DM truth-value negation: numeric_core.rs's reference
                    // formula is `f32::from(value == 0.0)`, mirrored bitwise
                    // here (NaN != 0.0, so `!NaN` is 0.0, same as that path).
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let zero = function_builder.ins().f32const(0.0);
                    let is_zero = function_builder.ins().fcmp(FloatCC::Equal, value, zero);
                    let one = function_builder.ins().f32const(1.0);
                    let value = function_builder.ins().select(is_zero, one, zero);
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
                NumericInstruction::Jump(target) => {
                    function_builder.ins().jump(
                        checks[target as usize],
                        &[cranelift_codegen::ir::BlockArg::Value(next_steps)],
                    );
                    continue;
                }
                NumericInstruction::JumpIfFalse(target) => {
                    let condition = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let zero = function_builder.ins().f32const(0.0);
                    let is_false = function_builder.ins().fcmp(FloatCC::Equal, condition, zero);
                    function_builder.ins().brif(
                        is_false,
                        checks[target as usize],
                        &[cranelift_codegen::ir::BlockArg::Value(next_steps)],
                        checks[pc + 1],
                        &[cranelift_codegen::ir::BlockArg::Value(next_steps)],
                    );
                    continue;
                }
                NumericInstruction::Return => {
                    let value = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    memory_store(&mut function_builder, stack_pointer, 0, value);
                    let packed = pack_exit(&mut function_builder, u32::MAX, next_steps);
                    function_builder.ins().return_(&[packed]);
                    continue;
                }
                operation => {
                    let right = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let left = memory_pop(&mut function_builder, stack_pointer, &mut depth);
                    let value = match operation {
                        NumericInstruction::Add => function_builder.ins().fadd(left, right),
                        NumericInstruction::Subtract => function_builder.ins().fsub(left, right),
                        NumericInstruction::Multiply => function_builder.ins().fmul(left, right),
                        NumericInstruction::Divide => function_builder.ins().fdiv(left, right),
                        NumericInstruction::Equal
                        | NumericInstruction::NotEqual
                        | NumericInstruction::LessThan
                        | NumericInstruction::LessThanOrEqual
                        | NumericInstruction::GreaterThan
                        | NumericInstruction::GreaterThanOrEqual => {
                            let cc = match operation {
                                NumericInstruction::Equal => FloatCC::Equal,
                                NumericInstruction::NotEqual => FloatCC::NotEqual,
                                NumericInstruction::LessThan => FloatCC::LessThan,
                                NumericInstruction::LessThanOrEqual => FloatCC::LessThanOrEqual,
                                NumericInstruction::GreaterThan => FloatCC::GreaterThan,
                                NumericInstruction::GreaterThanOrEqual => {
                                    FloatCC::GreaterThanOrEqual
                                }
                                _ => unreachable!(),
                            };
                            let predicate = function_builder.ins().fcmp(cc, left, right);
                            let one = function_builder.ins().f32const(1.0);
                            let zero = function_builder.ins().f32const(0.0);
                            function_builder.ins().select(predicate, one, zero)
                        }
                        NumericInstruction::And | NumericInstruction::Or => {
                            // Eager, non-short-circuiting: both operands are
                            // already on the stack. Matches numeric_core.rs's
                            // `left != 0.0 && right != 0.0` / `||` reference.
                            let zero = function_builder.ins().f32const(0.0);
                            let left_truthy =
                                function_builder.ins().fcmp(FloatCC::NotEqual, left, zero);
                            let right_truthy =
                                function_builder.ins().fcmp(FloatCC::NotEqual, right, zero);
                            let combined = if matches!(operation, NumericInstruction::And) {
                                function_builder.ins().band(left_truthy, right_truthy)
                            } else {
                                function_builder.ins().bor(left_truthy, right_truthy)
                            };
                            let one = function_builder.ins().f32const(1.0);
                            function_builder.ins().select(combined, one, zero)
                        }
                        _ => unreachable!("non-binary instructions handled above"),
                    };
                    memory_push(&mut function_builder, stack_pointer, &mut depth, value);
                }
            }
            if pc + 1 == instructions.len() {
                let packed = pack_exit(&mut function_builder, u32::MAX, next_steps);
                function_builder.ins().return_(&[packed]);
            } else {
                function_builder.ins().jump(
                    checks[pc + 1],
                    &[cranelift_codegen::ir::BlockArg::Value(next_steps)],
                );
            }
        }
        for (pc, ((check, body), exit)) in checks.into_iter().zip(bodies).zip(exits).enumerate() {
            if validation.depths[pc].is_some() {
                function_builder.seal_block(check);
                function_builder.seal_block(body);
                function_builder.seal_block(exit);
            }
        }
        for block in dispatches {
            function_builder.seal_block(block);
        }
        function_builder.finalize();
    }
    module
        .define_function(function, &mut context)
        .map_err(|error| CompileError::Backend(format!("{error:?}\n{}", context.func.display())))?;
    module.clear_context(&mut context);
    module
        .finalize_definitions()
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let pointer = module.get_finalized_function(function);
    // SAFETY: Cranelift finalized `function` with the signature constructed
    // above. This is the sole pointer-to-callable conversion in Dream64's JIT.
    let entry: NumericEntry = unsafe { std::mem::transmute(pointer) };
    Ok(CompiledNumericTrace {
        _module: module,
        entry,
        local_count,
        instruction_count: instructions.len(),
        max_stack_depth: validation.max_depth,
        reachable: validation.depths.iter().map(Option::is_some).collect(),
        field_count,
    })
}

fn memory_load(
    builder: &mut FunctionBuilder<'_>,
    pointer: cranelift_codegen::ir::Value,
    index: usize,
) -> cranelift_codegen::ir::Value {
    builder.ins().load(
        types::F32,
        cranelift_codegen::ir::MemFlags::trusted(),
        pointer,
        i32::try_from(index * 4).unwrap(),
    )
}
fn memory_store(
    builder: &mut FunctionBuilder<'_>,
    pointer: cranelift_codegen::ir::Value,
    index: usize,
    value: cranelift_codegen::ir::Value,
) {
    builder.ins().store(
        cranelift_codegen::ir::MemFlags::trusted(),
        value,
        pointer,
        i32::try_from(index * 4).unwrap(),
    );
}
fn memory_pop(
    builder: &mut FunctionBuilder<'_>,
    pointer: cranelift_codegen::ir::Value,
    depth: &mut usize,
) -> cranelift_codegen::ir::Value {
    *depth -= 1;
    memory_load(builder, pointer, *depth)
}

fn memory_push(
    builder: &mut FunctionBuilder<'_>,
    pointer: cranelift_codegen::ir::Value,
    depth: &mut usize,
    value: cranelift_codegen::ir::Value,
) {
    memory_store(builder, pointer, *depth, value);
    *depth += 1;
}

fn pack_exit(
    builder: &mut FunctionBuilder<'_>,
    instruction: u32,
    steps: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let shifted = builder.ins().ishl_imm(steps, 32);
    let pc = builder.ins().iconst(types::I64, i64::from(instruction));
    builder.ins().bor(shifted, pc)
}

/// Packs a `LoadFieldDynamic` decline exactly like `pack_exit`, but with the
/// low field's top bit set so `run_budgeted` reports `SideExit` rather than
/// `BudgetExhausted` — real instruction indices never set that bit (no
/// procedure has anywhere near `2^31` instructions), and `u32::MAX` (all
/// bits set, `Returned`'s sentinel) is unambiguous either way.
fn pack_side_exit(
    builder: &mut FunctionBuilder<'_>,
    instruction: u32,
    steps: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let shifted = builder.ins().ishl_imm(steps, 32);
    let pc = builder
        .ins()
        .iconst(types::I64, i64::from(instruction | 0x8000_0000));
    builder.ins().bor(shifted, pc)
}

struct Validation {
    depths: Vec<Option<usize>>,
    max_depth: usize,
}

/// Whether a tracked operand-stack slot holds a number or the region's
/// implicit `src` marker (produced only by a translated `LoadSrc`; valid only
/// as `LoadFieldDynamic`/`StoreFieldDynamic`'s receiver operand). This is
/// deliberately the smallest possible slice of the design doc's
/// `Unboxed | Rooted slot` operand model — the one non-numeric value
/// field-touching procedures reliably need — tracked precisely enough that a
/// non-`src` receiver is rejected by this type check, not silently treated
/// as `src`. See the "Milestone 3" module doc below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StackKind {
    Number,
    Src,
}

fn pop_number(pc: usize, stack: &mut SmallVec<[StackKind; 8]>) -> Result<(), CompileError> {
    match stack.pop() {
        Some(StackKind::Number) => Ok(()),
        Some(StackKind::Src) => Err(CompileError::InvalidOperandKind(pc)),
        None => Err(CompileError::StackUnderflow),
    }
}

fn pop_src(pc: usize, stack: &mut SmallVec<[StackKind; 8]>) -> Result<(), CompileError> {
    match stack.pop() {
        Some(StackKind::Src) => Ok(()),
        Some(StackKind::Number) => Err(CompileError::InvalidOperandKind(pc)),
        None => Err(CompileError::StackUnderflow),
    }
}

fn validate(
    instructions: &[NumericInstruction],
    local_count: usize,
    field_count: usize,
    dynamic_field_count: usize,
) -> Result<Validation, CompileError> {
    if instructions.is_empty() {
        return Err(CompileError::InvalidResultStack(0));
    }
    let mut depths = vec![None; instructions.len()];
    let mut kinds: Vec<Option<SmallVec<[StackKind; 8]>>> = vec![None; instructions.len()];
    depths[0] = Some(0);
    kinds[0] = Some(SmallVec::new());
    let mut work = vec![0usize];
    let mut max_depth = 0;
    while let Some(pc) = work.pop() {
        let mut stack = kinds[pc].clone().expect("queued reachable instruction");
        let instruction = instructions[pc];
        match instruction {
            NumericInstruction::Constant(_) => stack.push(StackKind::Number),
            NumericInstruction::LoadSrc => stack.push(StackKind::Src),
            NumericInstruction::LoadLocal(local) => {
                if usize::from(local) >= local_count {
                    return Err(CompileError::InvalidLocal(local));
                }
                stack.push(StackKind::Number);
            }
            NumericInstruction::StoreLocal(local) => {
                if usize::from(local) >= local_count {
                    return Err(CompileError::InvalidLocal(local));
                }
                pop_number(pc, &mut stack)?;
            }
            NumericInstruction::LoadField(field) => {
                if usize::from(field) >= field_count {
                    return Err(CompileError::InvalidField(field));
                }
                stack.push(StackKind::Number);
            }
            NumericInstruction::StoreField(field) => {
                if usize::from(field) >= field_count {
                    return Err(CompileError::InvalidField(field));
                }
                pop_number(pc, &mut stack)?;
            }
            NumericInstruction::RaiseAction(action) => {
                if action >= 64 {
                    return Err(CompileError::InvalidAction(action));
                }
            }
            NumericInstruction::Duplicate => {
                let top = *stack.last().ok_or(CompileError::StackUnderflow)?;
                stack.push(top);
            }
            NumericInstruction::Pop => {
                if stack.pop().is_none() {
                    return Err(CompileError::StackUnderflow);
                }
            }
            NumericInstruction::Negate | NumericInstruction::Not => match stack.last() {
                Some(StackKind::Number) => {}
                Some(StackKind::Src) => return Err(CompileError::InvalidOperandKind(pc)),
                None => return Err(CompileError::StackUnderflow),
            },
            NumericInstruction::LoadFieldDynamic(field) => {
                if usize::from(field) >= dynamic_field_count {
                    return Err(CompileError::InvalidField(field));
                }
                // The popped placeholder is discarded unconditionally by
                // codegen (the real receiver travels through the callback
                // context, not this stack), so unlike `StoreFieldDynamic`
                // below, its kind is never checked here — the VM-side
                // translator's adjacency rule is the whole soundness
                // argument for reads (see `numeric_trace_instructions`).
                if stack.pop().is_none() {
                    return Err(CompileError::StackUnderflow);
                }
                stack.push(StackKind::Number);
            }
            NumericInstruction::StoreFieldDynamic(field) => {
                if usize::from(field) >= dynamic_field_count {
                    return Err(CompileError::InvalidField(field));
                }
                pop_number(pc, &mut stack)?;
                pop_src(pc, &mut stack)?;
            }
            NumericInstruction::Add
            | NumericInstruction::Subtract
            | NumericInstruction::Multiply
            | NumericInstruction::Divide
            | NumericInstruction::Equal
            | NumericInstruction::NotEqual
            | NumericInstruction::LessThan
            | NumericInstruction::LessThanOrEqual
            | NumericInstruction::GreaterThan
            | NumericInstruction::GreaterThanOrEqual
            | NumericInstruction::And
            | NumericInstruction::Or => {
                pop_number(pc, &mut stack)?;
                pop_number(pc, &mut stack)?;
                stack.push(StackKind::Number);
            }
            NumericInstruction::Jump(target) => {
                add_edge(target, &stack, &mut depths, &mut kinds, &mut work)?;
                max_depth = max_depth.max(stack.len());
                continue;
            }
            NumericInstruction::JumpIfFalse(target) => {
                pop_number(pc, &mut stack)?;
                add_edge(target, &stack, &mut depths, &mut kinds, &mut work)?;
            }
            NumericInstruction::Return => {
                if stack.len() != 1 {
                    return Err(CompileError::InvalidResultStack(stack.len()));
                }
                pop_number(pc, &mut stack)?;
                max_depth = max_depth.max(1);
                continue;
            }
        }
        max_depth = max_depth.max(stack.len());
        if pc + 1 == instructions.len() {
            if stack.len() != 1 {
                return Err(CompileError::InvalidResultStack(stack.len()));
            }
            if stack[0] != StackKind::Number {
                return Err(CompileError::InvalidOperandKind(pc));
            }
        } else {
            add_edge((pc + 1) as u32, &stack, &mut depths, &mut kinds, &mut work)?;
        }
    }
    Ok(Validation { depths, max_depth })
}

fn add_edge(
    target: u32,
    stack: &SmallVec<[StackKind; 8]>,
    depths: &mut [Option<usize>],
    kinds: &mut [Option<SmallVec<[StackKind; 8]>>],
    work: &mut Vec<usize>,
) -> Result<(), CompileError> {
    let target_usize = usize::try_from(target).map_err(|_| CompileError::InvalidTarget(target))?;
    let Some(depth_slot) = depths.get_mut(target_usize) else {
        return Err(CompileError::InvalidTarget(target));
    };
    let depth = stack.len();
    match *depth_slot {
        None => {
            *depth_slot = Some(depth);
            kinds[target_usize] = Some(stack.clone());
            work.push(target_usize);
        }
        Some(first) if first != depth => {
            return Err(CompileError::InconsistentStack {
                instruction: target_usize,
                first,
                second: depth,
            });
        }
        Some(_) => {
            if kinds[target_usize]
                .as_ref()
                .is_some_and(|existing| existing != stack)
            {
                return Err(CompileError::InconsistentOperandKind(target_usize));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Baseline region JIT (docs/performance/baseline-region-jit.md)
// ---------------------------------------------------------------------------
//
// A "region" compiles a run of DM bytecode starting at a hot entry PC to
// native code, calling back into a Rust slow path for anything it cannot do
// inline. Milestone 1 proved the Cranelift compile/call/outcome-decode round
// trip with a stub that supported no bytecode and always deopted. Milestone 2
// (numeric core) is the first region that does real work: `dm-vm` installs a
// [`CompiledNumericTrace`] directly as the `PcCache::Region` payload — the
// existing binary32 trace compiler below *is* the region entry/exit model for
// an all-numeric procedure, so no separate region wrapper type is needed.
// `dm-vm`'s sidecar calls `initial_state`/`run_budgeted` exactly as the
// pre-region whole-procedure numeric JIT did; only the caching/warm-up model
// moved (from an unconditional first-call compile in a thread-local map, to a
// warm-up-counted compile installed at the procedure's PC-0 sidecar slot).
// Later milestones (fields, globals, calls) will need regions to do more than
// pure numerics, at which point this may grow into a dedicated wrapper type;
// until then `CompiledNumericTrace` fills the role directly.

#[cfg(test)]
mod tests {
    use super::{
        CompileError, NumericInstruction, NumericRunOutcome, compile_numeric_field_trace,
        compile_numeric_trace,
    };

    #[test]
    fn compiles_binary32_arithmetic() {
        let trace = compile_numeric_trace(
            &[
                NumericInstruction::LoadLocal(0),
                NumericInstruction::Constant(2.0),
                NumericInstruction::Multiply,
                NumericInstruction::LoadLocal(1),
                NumericInstruction::Add,
                NumericInstruction::Negate,
            ],
            2,
        )
        .expect("trace compiles");
        assert_eq!(trace.run(&[3.0, 4.0]), Some(-10.0));
        assert_eq!(trace.run(&[3.0]), None);
    }

    #[test]
    fn compiles_not_as_dm_truth_value_negation() {
        let trace = compile_numeric_trace(
            &[NumericInstruction::LoadLocal(0), NumericInstruction::Not],
            1,
        )
        .expect("trace compiles");
        assert_eq!(trace.run(&[0.0]), Some(1.0));
        assert_eq!(trace.run(&[1.0]), Some(0.0));
        assert_eq!(trace.run(&[-3.5]), Some(0.0));
        assert_eq!(trace.run(&[f32::NAN]), Some(0.0));
    }

    #[test]
    fn compiles_and_or_as_eager_canonicalized_booleans() {
        let and_trace = compile_numeric_trace(
            &[
                NumericInstruction::LoadLocal(0),
                NumericInstruction::LoadLocal(1),
                NumericInstruction::And,
            ],
            2,
        )
        .expect("and trace compiles");
        let or_trace = compile_numeric_trace(
            &[
                NumericInstruction::LoadLocal(0),
                NumericInstruction::LoadLocal(1),
                NumericInstruction::Or,
            ],
            2,
        )
        .expect("or trace compiles");
        for (left, right) in [(0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (3.0, -3.0)] {
            let expected_and = f32::from(left != 0.0 && right != 0.0);
            let expected_or = f32::from(left != 0.0 || right != 0.0);
            assert_eq!(
                and_trace.run(&[left, right]),
                Some(expected_and),
                "and({left}, {right})"
            );
            assert_eq!(
                or_trace.run(&[left, right]),
                Some(expected_or),
                "or({left}, {right})"
            );
        }
    }

    #[test]
    fn load_field_dynamic_reads_a_guarded_field_via_callback() {
        // The VM-side translator emits `Constant(0.0)` in place of the
        // bytecode's `LoadSrc` -- `LoadFieldDynamic` pops and discards that
        // placeholder; the real receiver is always this region's implicit
        // `src`, supplied to the callback out of band, never through the
        // operand stack (see the "Milestone 3" module doc below).
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::Constant(0.0),
                NumericInstruction::LoadFieldDynamic(0),
                NumericInstruction::Return,
            ],
            0,
            0,
            1,
        )
        .expect("dynamic field trace compiles");
        let mut state = trace.initial_state(&[]).unwrap();
        let outcome = trace
            .run_budgeted(
                &mut state,
                10,
                &mut |index| {
                    assert_eq!(index, 0, "only field-table index 0 was declared");
                    Some(42.0)
                },
                &mut |_, _| false,
            )
            .unwrap();
        assert_eq!(
            outcome,
            NumericRunOutcome::Returned {
                value: 42.0,
                steps: 3
            }
        );
    }

    #[test]
    fn load_field_dynamic_side_exits_when_the_callback_declines() {
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::Constant(0.0),
                NumericInstruction::LoadFieldDynamic(0),
                NumericInstruction::Return,
            ],
            0,
            0,
            1,
        )
        .expect("dynamic field trace compiles");
        let mut state = trace.initial_state(&[]).unwrap();
        let outcome = trace
            .run_budgeted(&mut state, 10, &mut |_| None, &mut |_, _| false)
            .unwrap();
        assert_eq!(
            outcome,
            NumericRunOutcome::SideExit {
                instruction: 1,
                steps: 1,
            },
            "declining must resume at the LoadFieldDynamic instruction itself, \
             having retired no steps for it"
        );
        // A SideExit is not "retry this same state and expect progress" the
        // way BudgetExhausted is, but `state.instruction` must still name the
        // exact bytecode a caller that does inspect it should continue from.
        assert_eq!(state.instruction, 1);
    }

    #[test]
    fn load_field_dynamic_rejects_an_out_of_range_field_index() {
        assert!(matches!(
            compile_numeric_field_trace(
                &[
                    NumericInstruction::Constant(0.0),
                    NumericInstruction::LoadFieldDynamic(1),
                    NumericInstruction::Return,
                ],
                0,
                0,
                1,
            ),
            Err(CompileError::InvalidField(1))
        ));
    }

    #[test]
    fn store_field_dynamic_writes_a_guarded_field_via_callback() {
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::LoadSrc,
                NumericInstruction::LoadLocal(0),
                NumericInstruction::StoreFieldDynamic(0),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Return,
            ],
            1,
            0,
            1,
        )
        .expect("dynamic field store trace compiles");
        let mut state = trace.initial_state(&[9.0]).unwrap();
        let mut received = None;
        let outcome = trace
            .run_budgeted(&mut state, 10, &mut |_| None, &mut |index, value| {
                received = Some((index, value));
                true
            })
            .unwrap();
        assert_eq!(received, Some((0, 9.0)));
        assert_eq!(
            outcome,
            NumericRunOutcome::Returned {
                value: 1.0,
                steps: 5
            }
        );
    }

    #[test]
    fn store_field_dynamic_side_exits_and_stashes_the_declined_value() {
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::LoadSrc,
                NumericInstruction::LoadLocal(0),
                NumericInstruction::StoreFieldDynamic(0),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Return,
            ],
            1,
            0,
            1,
        )
        .expect("dynamic field store trace compiles");
        let mut state = trace.initial_state(&[9.0]).unwrap();
        let outcome = trace
            .run_budgeted(&mut state, 10, &mut |_| None, &mut |_, _| false)
            .unwrap();
        assert_eq!(
            outcome,
            NumericRunOutcome::SideExit {
                instruction: 2,
                steps: 2,
            }
        );
        assert_eq!(
            state.stack[0], 9.0,
            "the value that would have been written must be recoverable from stack slot 0"
        );
    }

    #[test]
    fn validate_rejects_src_flowing_into_arithmetic() {
        assert!(matches!(
            compile_numeric_trace(
                &[
                    NumericInstruction::LoadSrc,
                    NumericInstruction::Constant(1.0),
                    NumericInstruction::Add,
                    NumericInstruction::Return,
                ],
                0,
            ),
            Err(CompileError::InvalidOperandKind(_))
        ));
    }

    #[test]
    fn validate_rejects_storing_src_into_a_local() {
        assert!(matches!(
            compile_numeric_trace(
                &[
                    NumericInstruction::LoadSrc,
                    NumericInstruction::StoreLocal(0),
                    NumericInstruction::Constant(0.0),
                    NumericInstruction::Return,
                ],
                1,
            ),
            Err(CompileError::InvalidOperandKind(_))
        ));
    }

    #[test]
    fn validate_rejects_store_field_dynamic_with_swapped_operand_kinds() {
        // Receiver and value in the wrong stack positions (a translator bug,
        // not something the real translator ever emits) must still be caught
        // by the type checker, not silently miscompiled.
        assert!(matches!(
            compile_numeric_field_trace(
                &[
                    NumericInstruction::Constant(5.0),
                    NumericInstruction::LoadSrc,
                    NumericInstruction::StoreFieldDynamic(0),
                    NumericInstruction::Constant(0.0),
                    NumericInstruction::Return,
                ],
                0,
                0,
                1,
            ),
            Err(CompileError::InvalidOperandKind(_))
        ));
    }

    #[test]
    fn rejects_unsafe_trace_shapes_for_interpreter_fallback() {
        assert!(matches!(
            compile_numeric_trace(&[NumericInstruction::Add], 0),
            Err(CompileError::StackUnderflow)
        ));
        assert!(matches!(
            compile_numeric_trace(&[NumericInstruction::LoadLocal(1)], 1),
            Err(CompileError::InvalidLocal(1))
        ));
    }

    #[test]
    fn native_operand_stack_is_heap_isolated_and_redzone_checked() {
        let trace = compile_numeric_trace(
            &[
                NumericInstruction::Constant(1.0),
                NumericInstruction::Return,
            ],
            0,
        )
        .expect("trace compiles");
        let mut state = trace.initial_state(&[]).expect("state shape matches");
        assert!(state.stack.spilled());
        let redzone = trace.max_stack_depth.max(1);
        state.stack[redzone] = 0.0;
        assert_eq!(
            trace.run_budgeted(&mut state, 2, &mut |_| None, &mut |_, _| false),
            None
        );
    }

    #[test]
    fn compiles_local_mutation_and_loop_backedge() {
        // sum = 0; while (n > 0) { sum += n; n -= 1 }; return sum
        let trace = compile_numeric_trace(
            &[
                NumericInstruction::Constant(0.0),
                NumericInstruction::StoreLocal(1),
                NumericInstruction::LoadLocal(0),
                NumericInstruction::Constant(0.0),
                NumericInstruction::GreaterThan,
                NumericInstruction::JumpIfFalse(17),
                NumericInstruction::LoadLocal(1),
                NumericInstruction::LoadLocal(0),
                NumericInstruction::Add,
                NumericInstruction::StoreLocal(1),
                NumericInstruction::LoadLocal(0),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Subtract,
                NumericInstruction::StoreLocal(0),
                NumericInstruction::Jump(2),
                NumericInstruction::Constant(999.0),
                NumericInstruction::Return,
                NumericInstruction::LoadLocal(1),
                NumericInstruction::Return,
            ],
            2,
        )
        .expect("loop compiles");
        assert_eq!(trace.run(&[5.0, 123.0]), Some(15.0));

        let mut state = trace.initial_state(&[5.0, 123.0]).unwrap();
        assert_eq!(
            trace.run_budgeted(&mut state, 0, &mut |_| None, &mut |_, _| false),
            Some(NumericRunOutcome::BudgetExhausted {
                instruction: 0,
                steps: 0
            })
        );
        let mut total_steps = 0;
        loop {
            match trace
                .run_budgeted(&mut state, 10, &mut |_| None, &mut |_, _| false)
                .unwrap()
            {
                NumericRunOutcome::BudgetExhausted { steps, .. } => {
                    assert_eq!(steps, 10);
                    total_steps += steps;
                }
                NumericRunOutcome::Returned { value, steps } => {
                    total_steps += steps;
                    assert_eq!(value, 15.0);
                    break;
                }
                NumericRunOutcome::SideExit { .. } => {
                    panic!("a trace with no LoadFieldDynamic cannot side-exit")
                }
            }
        }
        assert_eq!(total_steps, 73);
        assert_eq!(state.locals.as_slice(), &[0.0, 15.0]);
    }

    #[test]
    fn materialized_fields_write_back_and_raise_deferred_actions() {
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::Constant(0.0),
                NumericInstruction::StoreLocal(1),
                NumericInstruction::LoadLocal(1),
                NumericInstruction::LoadLocal(0),
                NumericInstruction::LessThan,
                NumericInstruction::JumpIfFalse(16),
                NumericInstruction::LoadField(0),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Add,
                NumericInstruction::StoreField(0),
                NumericInstruction::RaiseAction(2),
                NumericInstruction::LoadLocal(1),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Add,
                NumericInstruction::StoreLocal(1),
                NumericInstruction::Jump(2),
                NumericInstruction::LoadField(0),
                NumericInstruction::Return,
            ],
            2,
            1,
            0,
        )
        .expect("guarded field loop compiles");
        let mut state = trace
            .initial_state_with_fields(&[10.0, 0.0], &[7.0])
            .unwrap();
        assert!(
            state.is_fully_inline(),
            "ordinary field trace must not allocate"
        );
        loop {
            if matches!(
                trace
                    .run_budgeted(&mut state, 7, &mut |_| None, &mut |_, _| false)
                    .unwrap(),
                NumericRunOutcome::Returned { value: 17.0, .. }
            ) {
                break;
            }
        }
        assert_eq!(state.fields.as_slice(), &[17.0]);
        assert_eq!(state.dirty_fields, 1);
        assert_eq!(state.action_bits, 1 << 2);
    }

    #[test]
    fn duplicate_and_pop_preserve_stack_shape() {
        let trace = compile_numeric_trace(
            &[
                NumericInstruction::Constant(4.0),
                NumericInstruction::Duplicate,
                NumericInstruction::Add,
                NumericInstruction::Constant(99.0),
                NumericInstruction::Pop,
                NumericInstruction::Return,
            ],
            0,
        )
        .unwrap();
        assert_eq!(trace.run(&[]), Some(8.0));
    }

    #[test]
    #[ignore = "local release microbenchmark"]
    fn materialized_field_two_million_call_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        const CALLS: usize = 2_000_000;
        let trace = compile_numeric_field_trace(
            &[
                NumericInstruction::LoadField(0),
                NumericInstruction::Constant(1.0),
                NumericInstruction::Add,
                NumericInstruction::StoreField(0),
                NumericInstruction::LoadField(0),
                NumericInstruction::Return,
            ],
            0,
            1,
            0,
        )
        .unwrap();
        let mut native = trace.initial_state_with_fields(&[], &[0.0]).unwrap();
        let started = Instant::now();
        for _ in 0..CALLS {
            black_box(
                trace
                    .run_budgeted(&mut native, 6, &mut |_| None, &mut |_, _| false)
                    .unwrap(),
            );
        }
        let native_elapsed = started.elapsed();
        let mut rust_field = 0.0_f32;
        let started = Instant::now();
        for _ in 0..CALLS {
            rust_field = black_box(rust_field + 1.0);
        }
        let rust_elapsed = started.elapsed();
        eprintln!(
            "materialized-field calls={CALLS} native={native_elapsed:?} rust={rust_elapsed:?}"
        );
        assert_eq!(native.fields[0], rust_field);
    }

    #[repr(C)]
    struct RootedFixture {
        calls: u32,
    }

    unsafe extern "C" fn rooted_fixture_dispatch(
        context: *mut std::ffi::c_void,
        roots: *mut u32,
        root_count: u32,
        stack: *mut u32,
        stack_len: *mut u32,
        stack_capacity: u32,
        start_pc: u32,
        budget: u32,
    ) -> u64 {
        let fixture = unsafe { &mut *context.cast::<RootedFixture>() };
        fixture.calls += 1;
        if budget == 0 {
            return (1_u64 << 56) | u64::from(start_pc);
        }
        if root_count == 0 || unsafe { *stack_len } >= stack_capacity {
            return (2_u64 << 56) | (1_u64 << 32) | u64::from(start_pc + 1);
        }
        unsafe {
            let len = *stack_len as usize;
            *stack.add(len) = *roots;
            *stack_len += 1;
        }
        (1_u64 << 32) | u64::from(start_pc + 1)
    }

    #[test]
    fn rooted_block_materializes_stack_and_exact_budget_exit() {
        let trace = super::compile_rooted_block(rooted_fixture_dispatch).unwrap();
        let mut fixture = RootedFixture { calls: 0 };
        let mut roots = [73_u32];
        let mut stack = Vec::with_capacity(2);
        assert_eq!(
            unsafe {
                trace.run(
                    (&mut fixture as *mut RootedFixture).cast(),
                    &mut roots,
                    &mut stack,
                    9,
                    0,
                )
            },
            super::RootedBlockOutcome::BudgetExhausted {
                instruction: 9,
                steps: 0
            },
        );
        assert!(stack.is_empty());
        assert_eq!(
            unsafe {
                trace.run(
                    (&mut fixture as *mut RootedFixture).cast(),
                    &mut roots,
                    &mut stack,
                    9,
                    1,
                )
            },
            super::RootedBlockOutcome::Completed {
                instruction: 10,
                steps: 1
            },
        );
        assert_eq!(stack, [73]);
        assert_eq!(fixture.calls, 2);
    }

    #[test]
    #[ignore = "local release microbenchmark"]
    fn rooted_block_batch_entry_microbenchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        const CALLS: usize = 500_000;
        let trace = super::compile_rooted_block(rooted_fixture_dispatch).unwrap();
        let mut fixture = RootedFixture { calls: 0 };
        let mut roots = [1_u32];
        let started = Instant::now();
        for _ in 0..CALLS {
            let mut stack = Vec::with_capacity(1);
            black_box(unsafe {
                trace.run(
                    (&mut fixture as *mut RootedFixture).cast(),
                    &mut roots,
                    &mut stack,
                    0,
                    1,
                )
            });
        }
        eprintln!(
            "rooted-block batch calls={CALLS} elapsed={:?}",
            started.elapsed()
        );
    }
}
