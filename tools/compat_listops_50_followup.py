from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:1000]}")
    return text.replace(old, new, 1)

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    "use builtins::{execute_list_method, execute_standard_builtin, is_subtype, standard_builtin_arity};\n",
    "use builtins::{\n    execute_list_binary_operator, execute_list_compound_operator, execute_list_method,\n    execute_standard_builtin, is_subtype, standard_builtin_arity,\n};\n",
    "list operator VM imports",
)
text = replace_once(
    text,
    "    /// Numeric addition.\n    Add,\n",
    "    /// Executes a compound assignment while preserving type-specific mutation semantics.\n    CompoundAssignment(CompoundAssignmentOperator),\n    /// Numeric/list/text addition.\n    Add,\n",
    "compound assignment instruction",
)
text = replace_once(
    text,
    '''/// Numeric operation used by [`Instruction::CompoundListIndex`].
''',
    '''/// Calculate-and-assign operator used by [`Instruction::CompoundAssignment`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompoundAssignmentOperator {
    /// Addition assignment (`+=`).
    Add,
    /// Subtraction assignment (`-=`).
    Subtract,
    /// Multiplication assignment (`*=`).
    Multiply,
    /// Division assignment (`/=`).
    Divide,
    /// Remainder assignment (`%=`).
    Remainder,
    /// Bitwise/list-mask assignment (`&=`).
    BitAnd,
    /// Bitwise/list-union assignment (`|=`).
    BitOr,
    /// Bitwise/list-symmetric-difference assignment (`^=`).
    BitXor,
    /// Left-shift assignment (`<<=`).
    ShiftLeft,
    /// Right-shift assignment (`>>=`).
    ShiftRight,
}

/// Numeric operation used by [`Instruction::CompoundListIndex`].
''',
    "compound operator enum",
)
pattern = re.compile(r'''fn compound_instruction\(operator: &str\) -> Result<Instruction, CompileError> \{\n    Ok\(match operator \{.*?\n    \}\)\n\}\n''', re.S)
match = pattern.search(text)
if not match:
    raise SystemExit("compound_instruction function was not found")
new_fn = '''fn compound_instruction(operator: &str) -> Result<Instruction, CompileError> {
    let operator = match operator {
        "+=" => CompoundAssignmentOperator::Add,
        "-=" => CompoundAssignmentOperator::Subtract,
        "*=" => CompoundAssignmentOperator::Multiply,
        "/=" => CompoundAssignmentOperator::Divide,
        "%=" => CompoundAssignmentOperator::Remainder,
        "&=" => CompoundAssignmentOperator::BitAnd,
        "|=" => CompoundAssignmentOperator::BitOr,
        "^=" => CompoundAssignmentOperator::BitXor,
        "<<=" => CompoundAssignmentOperator::ShiftLeft,
        ">>=" => CompoundAssignmentOperator::ShiftRight,
        _ => return Err(compile_error(format!("unsupported compound operator {operator}"))),
    };
    Ok(Instruction::CompoundAssignment(operator))
}
'''
text = text[:match.start()] + new_fn + text[match.end():]

old_add = '''            Instruction::Add => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = match (left, right) {
                    (Value::Number(left), Value::Number(right)) => {
                        Value::number(left.to_f32() + right.to_f32())
                    }
                    (Value::Null, Value::Number(right)) => Value::number(right.to_f32()),
                    (Value::Number(left), Value::Null) => Value::number(left.to_f32()),
                    (Value::Null, Value::Null) => Value::number(0.0),
                    (Value::Text(left), Value::Text(right)) => {
                        Value::text(format!("{left}{right}"))
                    }
                    (left, right) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!(
                                "addition requires compatible DM values, received {left} and {right}"
                            ),
                        ));
                    }
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Subtract
            | Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::BitAnd
            | Instruction::BitOr
            | Instruction::BitXor
            | Instruction::ShiftLeft
            | Instruction::ShiftRight
'''
new_add = '''            Instruction::CompoundAssignment(operator) => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let Value::List(list) = left {
                    execute_list_compound_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_compound_assignment(operator, left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Add => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let value = if let Value::List(list) = left {
                    execute_list_binary_operator("+", list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    execute_scalar_add(left, right)
                        .map_err(|message| execution_error(module, &frames, message))?
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Subtract | Instruction::BitAnd | Instruction::BitOr | Instruction::BitXor => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let operator = match instruction {
                    Instruction::Subtract => "-",
                    Instruction::BitAnd => "&",
                    Instruction::BitOr => "|",
                    Instruction::BitXor => "^",
                    _ => unreachable!(),
                };
                let value = if let Value::List(list) = left {
                    execute_list_binary_operator(operator, list, &right, state)
                        .map_err(|message| execution_error(module, &frames, message))?
                } else {
                    let left = scalar_number_string(left)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    let right = scalar_number_string(right)
                        .map_err(|message| execution_error(module, &frames, message))?;
                    Value::number(execute_numeric_binary(&instruction, left, right))
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight
'''
text = replace_once(text, old_add, new_add, "binary and compound list dispatch")

helper_anchor = '''fn pop_number(stack: &mut Vec<Value>) -> Result<f32, String> {
'''
scalar_helpers = '''fn scalar_number_string(value: Value) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

fn execute_scalar_add(left: Value, right: Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(Value::number(left.to_f32() + right.to_f32()))
        }
        (Value::Null, Value::Number(right)) => Ok(Value::number(right.to_f32())),
        (Value::Number(left), Value::Null) => Ok(Value::number(left.to_f32())),
        (Value::Null, Value::Null) => Ok(Value::number(0.0)),
        (Value::Text(left), Value::Text(right)) => Ok(Value::text(format!("{left}{right}"))),
        (left, right) => Err(format!(
            "addition requires compatible DM values, received {left} and {right}"
        )),
    }
}

fn execute_scalar_compound_assignment(
    operator: CompoundAssignmentOperator,
    left: Value,
    right: Value,
) -> Result<Value, String> {
    if matches!(operator, CompoundAssignmentOperator::Add)
        && matches!((&left, &right), (Value::Text(_), Value::Text(_)))
    {
        return execute_scalar_add(left, right);
    }
    let left = scalar_number_string(left)?;
    let right = scalar_number_string(right)?;
    let value = match operator {
        CompoundAssignmentOperator::Add => left + right,
        CompoundAssignmentOperator::Subtract => left - right,
        CompoundAssignmentOperator::Multiply => left * right,
        CompoundAssignmentOperator::Divide => left / right,
        CompoundAssignmentOperator::Remainder => left % right,
        CompoundAssignmentOperator::BitAnd => bitwise_binary(left, right, |a, b| a & b),
        CompoundAssignmentOperator::BitOr => bitwise_binary(left, right, |a, b| a | b),
        CompoundAssignmentOperator::BitXor => bitwise_binary(left, right, |a, b| a ^ b),
        CompoundAssignmentOperator::ShiftLeft => shift_binary(left, right, true),
        CompoundAssignmentOperator::ShiftRight => shift_binary(left, right, false),
    };
    Ok(Value::number(value))
}

'''
text = replace_once(text, helper_anchor, scalar_helpers + helper_anchor, "scalar compound helpers")

test_anchor = '''    #[test]
    fn documented_list_methods_and_len_execute_natively() {
'''
tests = r'''    #[test]
    fn list_binary_operators_return_new_lists_without_mutating_the_left_operand() {
        let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2, 2, 3)\n\tvar/list/b = list(2, 4)\n\tvar/list/added = a + b\n\tvar/list/subtracted = a - b\n\tvar/list/unioned = a | b\n\tvar/list/masked = a & b\n\tvar/list/xored = a ^ b\n\treturn a.len + added.len + subtracted.len + unioned.len + masked.len + xored.len + (a[2] == 2) + (unioned[4] == 4)\n",
        )
        .expect("list operator source should parse");
        let module = compile_module(&source.definitions).expect("list operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(24.0))
        );
    }

    #[test]
    fn compound_list_operators_mutate_shared_alias_identity() {
        let source = parse(
            "/proc/run()\n\tvar/list/a = list(1, 2)\n\tvar/list/alias = a\n\ta += list(2, 3)\n\tvar/after_add = alias.len\n\ta -= 2\n\tvar/after_remove = alias.len\n\ta |= list(3, 4)\n\tvar/after_union = alias.len\n\ta &= list(1, 4)\n\tvar/after_mask = alias.len\n\ta ^= list(4, 5)\n\treturn after_add + after_remove + after_union + after_mask + alias.len + (alias[1] == 1) + (alias[2] == 5)\n",
        )
        .expect("compound list operator source should parse");
        let module = compile_module(&source.definitions).expect("compound list operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/run").unwrap(), &[]),
            Ok(Value::number(17.0))
        );
    }

'''
text = replace_once(text, test_anchor, tests + test_anchor, "list operator regressions")
p.write_text(text)
