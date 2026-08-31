//! Interpreter support helpers: diagnostics, trace construction, and the
//! numeric/compound-assignment evaluations shared by the instruction dispatcher.
//!
//! The interpreter loop in `run` stays focused on frame advancement and
//! dispatch; the pure evaluations here have no access to `ExecutionState` and
//! can be reasoned about in isolation.

use crate::bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, Instruction, Module, ProcedureId,
};
use crate::execution::frame::CallFrame;
use crate::value_ops::{bitwise_binary, bitwise_shift, fractional_remainder, integer_remainder};
use crate::{CallTrace, RuntimeError};

pub(crate) fn execution_error(
    module: &Module,
    frames: &[CallFrame],
    message: impl Into<String>,
) -> RuntimeError {
    let instruction = frames.last().map_or(0, |frame| frame.instruction);
    let source_span = frames.last().and_then(|frame| {
        module
            .procedure(frame.procedure)
            .and_then(|program| program.source_spans.get(frame.instruction))
            .copied()
    });
    RuntimeError {
        message: message.into(),
        instruction,
        source_span,
        call_stack: frames
            .iter()
            .map(|frame| trace(module, frame.procedure, frame.instruction))
            .collect(),
    }
}

pub(crate) fn trace(module: &Module, procedure: ProcedureId, instruction: usize) -> CallTrace {
    CallTrace {
        procedure: module
            .procedure_path(procedure)
            .unwrap_or("<invalid procedure>")
            .to_owned(),
        instruction,
        source_span: module
            .procedure(procedure)
            .and_then(|program| program.source_spans.get(instruction))
            .copied(),
    }
}

pub(crate) fn execute_numeric_binary(instruction: &Instruction, left: f32, right: f32) -> f32 {
    match instruction {
        Instruction::Add => left + right,
        Instruction::Subtract => left - right,
        Instruction::Multiply => left * right,
        Instruction::Power => left.powf(right),
        Instruction::Divide => left / right,
        Instruction::Remainder => integer_remainder(left, right),
        Instruction::FractionalRemainder => fractional_remainder(left, right),
        Instruction::BitAnd => bitwise_binary(left, right, |left, right| left & right),
        Instruction::BitOr => bitwise_binary(left, right, |left, right| left | right),
        Instruction::BitXor => bitwise_binary(left, right, |left, right| left ^ right),
        Instruction::ShiftLeft => bitwise_shift(left, right, |left, right| left << right),
        Instruction::ShiftRight => bitwise_shift(left, right, |left, right| left >> right),
        Instruction::Less => f32::from(left < right),
        Instruction::LessEqual => f32::from(left <= right),
        Instruction::Greater => f32::from(left > right),
        Instruction::GreaterEqual => f32::from(left >= right),
        _ => unreachable!("instruction came from the numeric operation group"),
    }
}

pub(crate) fn execute_compound_list_index_operation(
    operator: CompoundListIndexOperator,
    left: f32,
    right: f32,
) -> f32 {
    match operator {
        CompoundListIndexOperator::Add => left + right,
        CompoundListIndexOperator::Subtract => left - right,
        CompoundListIndexOperator::Multiply => left * right,
        CompoundListIndexOperator::Divide => left / right,
        CompoundListIndexOperator::Remainder => integer_remainder(left, right),
        CompoundListIndexOperator::FractionalRemainder => fractional_remainder(left, right),
        CompoundListIndexOperator::BitAnd => {
            bitwise_binary(left, right, |left, right| left & right)
        }
        CompoundListIndexOperator::BitOr => bitwise_binary(left, right, |left, right| left | right),
        CompoundListIndexOperator::BitXor => {
            bitwise_binary(left, right, |left, right| left ^ right)
        }
        CompoundListIndexOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundListIndexOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
    }
}

pub(crate) fn compound_assignment_from_list_index(
    operator: CompoundListIndexOperator,
) -> CompoundAssignmentOperator {
    match operator {
        CompoundListIndexOperator::Add => CompoundAssignmentOperator::Add,
        CompoundListIndexOperator::Subtract => CompoundAssignmentOperator::Subtract,
        CompoundListIndexOperator::Multiply => CompoundAssignmentOperator::Multiply,
        CompoundListIndexOperator::Divide => CompoundAssignmentOperator::Divide,
        CompoundListIndexOperator::Remainder => CompoundAssignmentOperator::Remainder,
        CompoundListIndexOperator::FractionalRemainder => {
            CompoundAssignmentOperator::FractionalRemainder
        }
        CompoundListIndexOperator::BitAnd => CompoundAssignmentOperator::BitAnd,
        CompoundListIndexOperator::BitOr => CompoundAssignmentOperator::BitOr,
        CompoundListIndexOperator::BitXor => CompoundAssignmentOperator::BitXor,
        CompoundListIndexOperator::ShiftLeft => CompoundAssignmentOperator::ShiftLeft,
        CompoundListIndexOperator::ShiftRight => CompoundAssignmentOperator::ShiftRight,
    }
}
