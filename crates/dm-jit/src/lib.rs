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
/// One operation in a verified binary32 procedure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NumericInstruction {
    /// Push a binary32 constant.
    Constant(f32),
    /// Push a local passed to the trace.
    LoadLocal(u16),
    /// Pop a value into a procedure local.
    StoreLocal(u16),
    /// Push a VM-guarded, materialized binary32 field.
    LoadField(u16),
    /// Pop into a materialized field and mark it dirty for VM writeback.
    StoreField(u16),
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
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
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
            Self::Backend(message) => write!(formatter, "Cranelift backend failed: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

type NumericEntry =
    unsafe extern "C" fn(*mut f32, *mut f32, *mut f32, *mut u64, *mut u64, u32, u64) -> u64;

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
    Returned { value: f32, steps: u32 },
    BudgetExhausted { instruction: u32, steps: u32 },
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
        match self.run_budgeted(&mut state, u32::MAX)? {
            NumericRunOutcome::Returned { value, .. } => Some(value),
            NumericRunOutcome::BudgetExhausted { .. } => None,
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
    pub fn run_budgeted(
        &self,
        state: &mut NumericExecutionState,
        max_steps: u32,
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
        let packed = unsafe {
            (self.entry)(
                state.locals.as_mut_ptr(),
                state.stack.as_mut_ptr(),
                state.fields.as_mut_ptr(),
                &mut state.dirty_fields,
                &mut state.action_bits,
                state.instruction,
                u64::from(max_steps),
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
        let instruction = packed as u32;
        let steps = (packed >> 32) as u32;
        if instruction == u32::MAX {
            Some(NumericRunOutcome::Returned {
                value: state.stack[0],
                steps,
            })
        } else {
            state.instruction = instruction;
            Some(NumericRunOutcome::BudgetExhausted { instruction, steps })
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
    compile_numeric_field_trace(instructions, local_count, 0)
}

/// Compiles a trace over guarded numeric field snapshots. The VM validates and
/// materializes fields before entry, then commits `dirty_fields` at every native exit.
pub fn compile_numeric_field_trace(
    instructions: &[NumericInstruction],
    local_count: usize,
    field_count: usize,
) -> Result<CompiledNumericTrace, CompileError> {
    if field_count > 64 {
        return Err(CompileError::TooManyFields(field_count));
    }
    let validation = validate(instructions, local_count, field_count)?;

    let builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|error| CompileError::Backend(error.to_string()))?;
    let mut module = JITModule::new(builder);
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
        .returns
        .push(AbiParam::new(types::I64));
    let function = module
        .declare_function(
            "dream64_numeric_trace",
            Linkage::Local,
            &context.func.signature,
        )
        .map_err(|error| CompileError::Backend(error.to_string()))?;

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

struct Validation {
    depths: Vec<Option<usize>>,
    max_depth: usize,
}

fn validate(
    instructions: &[NumericInstruction],
    local_count: usize,
    field_count: usize,
) -> Result<Validation, CompileError> {
    if instructions.is_empty() {
        return Err(CompileError::InvalidResultStack(0));
    }
    let mut depths = vec![None; instructions.len()];
    depths[0] = Some(0);
    let mut work = vec![0usize];
    let mut max_depth = 0;
    while let Some(pc) = work.pop() {
        let mut depth = depths[pc].expect("queued reachable instruction");
        let instruction = instructions[pc];
        match instruction {
            NumericInstruction::Constant(_) => depth += 1,
            NumericInstruction::LoadLocal(local) => {
                if usize::from(local) >= local_count {
                    return Err(CompileError::InvalidLocal(local));
                }
                depth += 1;
            }
            NumericInstruction::StoreLocal(local) => {
                if usize::from(local) >= local_count {
                    return Err(CompileError::InvalidLocal(local));
                }
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
                depth -= 1;
            }
            NumericInstruction::LoadField(field) => {
                if usize::from(field) >= field_count {
                    return Err(CompileError::InvalidField(field));
                }
                depth += 1;
            }
            NumericInstruction::StoreField(field) => {
                if usize::from(field) >= field_count {
                    return Err(CompileError::InvalidField(field));
                }
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
                depth -= 1;
            }
            NumericInstruction::RaiseAction(action) => {
                if action >= 64 {
                    return Err(CompileError::InvalidAction(action));
                }
            }
            NumericInstruction::Duplicate => {
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
                depth += 1;
            }
            NumericInstruction::Pop => {
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
                depth -= 1;
            }
            NumericInstruction::Negate => {
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
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
            | NumericInstruction::GreaterThanOrEqual => {
                if depth < 2 {
                    return Err(CompileError::StackUnderflow);
                }
                depth -= 1;
            }
            NumericInstruction::Jump(target) => {
                add_edge(target, depth, &mut depths, &mut work)?;
                max_depth = max_depth.max(depth);
                continue;
            }
            NumericInstruction::JumpIfFalse(target) => {
                if depth < 1 {
                    return Err(CompileError::StackUnderflow);
                }
                depth -= 1;
                add_edge(target, depth, &mut depths, &mut work)?;
            }
            NumericInstruction::Return => {
                if depth != 1 {
                    return Err(CompileError::InvalidResultStack(depth));
                }
                max_depth = max_depth.max(depth);
                continue;
            }
        }
        max_depth = max_depth.max(depth);
        if pc + 1 == instructions.len() {
            if depth != 1 {
                return Err(CompileError::InvalidResultStack(depth));
            }
        } else {
            add_edge((pc + 1) as u32, depth, &mut depths, &mut work)?;
        }
    }
    Ok(Validation { depths, max_depth })
}

fn add_edge(
    target: u32,
    depth: usize,
    depths: &mut [Option<usize>],
    work: &mut Vec<usize>,
) -> Result<(), CompileError> {
    let target_usize = usize::try_from(target).map_err(|_| CompileError::InvalidTarget(target))?;
    let Some(slot) = depths.get_mut(target_usize) else {
        return Err(CompileError::InvalidTarget(target));
    };
    match *slot {
        None => {
            *slot = Some(depth);
            work.push(target_usize);
        }
        Some(first) if first != depth => {
            return Err(CompileError::InconsistentStack {
                instruction: target_usize,
                first,
                second: depth,
            });
        }
        Some(_) => {}
    }
    Ok(())
}

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
        assert_eq!(trace.run_budgeted(&mut state, 2), None);
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
            trace.run_budgeted(&mut state, 0),
            Some(NumericRunOutcome::BudgetExhausted {
                instruction: 0,
                steps: 0
            })
        );
        let mut total_steps = 0;
        loop {
            match trace.run_budgeted(&mut state, 10).unwrap() {
                NumericRunOutcome::BudgetExhausted { steps, .. } => {
                    assert_eq!(steps, 10);
                    total_steps += steps;
                }
                NumericRunOutcome::Returned { value, steps } => {
                    total_steps += steps;
                    assert_eq!(value, 15.0);
                    break;
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
                trace.run_budgeted(&mut state, 7).unwrap(),
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
        )
        .unwrap();
        let mut native = trace.initial_state_with_fields(&[], &[0.0]).unwrap();
        let started = Instant::now();
        for _ in 0..CALLS {
            black_box(trace.run_budgeted(&mut native, 6).unwrap());
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
}
