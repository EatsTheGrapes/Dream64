from pathlib import Path
import re


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:1000]}")
    return text.replace(old, new, 1)


# Fix association assignment to attach a value to an existing list item rather
# than introducing a duplicate iteration entry, then expose an order-preserving
# containment helper for list operators.
p = Path("crates/dm-value/src/lib.rs")
text = p.read_text()
old = '''    pub fn set_key(&mut self, key: Value, value: Value) -> Option<Value> {
        if let Some((_, current)) = self
            .associative
            .iter_mut()
            .find(|(candidate, _)| candidate.semantic_eq(&key))
        {
            return Some(std::mem::replace(current, value));
        }
        self.order.push(ListOrder::Associative(key.clone()));
        self.associative.push((key, value));
        None
    }
'''
new = '''    pub fn set_key(&mut self, key: Value, value: Value) -> Option<Value> {
        if let Some((_, current)) = self
            .associative
            .iter_mut()
            .find(|(candidate, _)| candidate.semantic_eq(&key))
        {
            return Some(std::mem::replace(current, value));
        }
        if let Some((order_index, position)) = self.order.iter().enumerate().find_map(|(index, entry)| {
            let ListOrder::Positional(position) = entry else {
                return None;
            };
            self.positional[*position]
                .semantic_eq(&key)
                .then_some((index, *position))
        }) {
            let existing_key = self.positional.remove(position);
            for entry in &mut self.order {
                if let ListOrder::Positional(other) = entry
                    && *other > position
                {
                    *other -= 1;
                }
            }
            self.order[order_index] = ListOrder::Associative(existing_key.clone());
            self.associative.push((existing_key, value));
            return None;
        }
        self.order.push(ListOrder::Associative(key.clone()));
        self.associative.push((key, value));
        None
    }

    /// Returns whether an iteration entry is semantically equal to `value`.
    #[must_use]
    pub fn contains(&self, value: &Value) -> bool {
        self.positions()
            .any(|(_, candidate)| candidate.semantic_eq(value))
    }
'''
text = replace_once(text, old, new, "association conversion and contains")

test_anchor = '''    #[test]
    fn range_insert_swap_and_resize_preserve_order_and_associations() {
'''
test = '''    #[test]
    fn assigning_an_existing_item_as_a_key_preserves_length_and_order() {
        let mut list = DmList::default();
        list.add(text("key"));
        list.add(text("other"));
        assert_eq!(list.set_key(text("key"), Value::number(7.0)), None);
        assert_eq!(list.len(), 2);
        assert!(list.get(1).unwrap().semantic_eq(&text("key")));
        assert!(
            list.get_key(&text("key"))
                .unwrap()
                .semantic_eq(&Value::number(7.0))
        );
    }

'''
text = replace_once(text, test_anchor, test + test_anchor, "association conversion regression")
p.write_text(text)

# Add list operator helpers using the same ordered/associative storage used by
# native list procs.
p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
text = replace_once(
    text,
    "use super::ExecutionState;\n",
    "use super::{CompoundAssignmentOperator, ExecutionState};\n",
    "compound operator import",
)
anchor = '''pub(super) fn execute_list_method(
'''
helpers = r'''#[derive(Clone)]
struct ListOperatorEntry {
    key: Value,
    associated: Option<Value>,
}

fn list_operator_snapshot(list: ListId, state: &ExecutionState) -> Result<Vec<ListOperatorEntry>, String> {
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    list.positions()
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect::<Vec<_>>()
        .pipe(Ok)
}

trait Pipe: Sized {
    fn pipe<T>(self, function: impl FnOnce(Self) -> T) -> T {
        function(self)
    }
}
impl<T> Pipe for T {}

fn add_operator_entry(
    list: ListId,
    entry: ListOperatorEntry,
    state: &mut ExecutionState,
    only_if_absent: bool,
) -> Result<(), String> {
    let target = state.heap.list_mut(list).map_err(|error| error.to_string())?;
    if only_if_absent && target.contains(&entry.key) {
        return Ok(());
    }
    if let Some(associated) = entry.associated {
        target.set_key(entry.key, associated);
    } else {
        target.add(entry.key);
    }
    Ok(())
}

fn remove_all_operator_matches(
    list: ListId,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0;
    while state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .remove_last(value)
        .is_some()
    {
        removed += 1;
    }
    Ok(removed)
}

fn operator_rhs_entries(value: &Value, state: &ExecutionState) -> Result<Vec<ListOperatorEntry>, String> {
    if let Value::List(list) = value {
        list_operator_snapshot(*list, state)
    } else {
        Ok(vec![ListOperatorEntry {
            key: value.clone(),
            associated: None,
        }])
    }
}

pub(super) fn execute_list_binary_operator(
    operator: &str,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match operator {
        "+" => {
            let result = state.heap.copy_list(left).map_err(|error| error.to_string())?;
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, false)?;
            }
            Ok(Value::List(result))
        }
        "-" => {
            let result = state.heap.copy_list(left).map_err(|error| error.to_string())?;
            for entry in operator_rhs_entries(right, state)? {
                state
                    .heap
                    .list_mut(result)
                    .map_err(|error| error.to_string())?
                    .remove_last(&entry.key);
            }
            Ok(Value::List(result))
        }
        "|" => {
            let result = state.heap.allocate_list();
            for entry in list_operator_snapshot(left, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            Ok(Value::List(result))
        }
        "&" => {
            let result = state.heap.copy_list(left).map_err(|error| error.to_string())?;
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(result, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(result, &entry.key, state)?;
                }
            }
            Ok(Value::List(result))
        }
        "^" => {
            let result = state.heap.allocate_list();
            let left_entries = list_operator_snapshot(left, state)?;
            let right_entries = operator_rhs_entries(right, state)?;
            for entry in &left_entries {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry.clone(), state, true)?;
                }
            }
            for entry in right_entries {
                if !left_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry, state, true)?;
                }
            }
            Ok(Value::List(result))
        }
        _ => Err(format!("unsupported /list binary operator {operator:?}")),
    }
}

pub(super) fn execute_list_compound_operator(
    operator: CompoundAssignmentOperator,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match operator {
        CompoundAssignmentOperator::Add => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, false)?;
            }
        }
        CompoundAssignmentOperator::Subtract => {
            for entry in operator_rhs_entries(right, state)? {
                state
                    .heap
                    .list_mut(left)
                    .map_err(|error| error.to_string())?
                    .remove_last(&entry.key);
            }
        }
        CompoundAssignmentOperator::BitOr => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, true)?;
            }
        }
        CompoundAssignmentOperator::BitAnd => {
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(left, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                }
            }
        }
        CompoundAssignmentOperator::BitXor => {
            let right_entries = operator_rhs_entries(right, state)?;
            let original = list_operator_snapshot(left, state)?;
            for entry in right_entries {
                if original
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                } else {
                    add_operator_entry(left, entry, state, true)?;
                }
            }
        }
        CompoundAssignmentOperator::Multiply
        | CompoundAssignmentOperator::Divide
        | CompoundAssignmentOperator::Remainder
        | CompoundAssignmentOperator::ShiftLeft
        | CompoundAssignmentOperator::ShiftRight => {
            return Err(format!(
                "operator {operator:?} is not defined for a BYOND list"
            ));
        }
    }
    Ok(Value::List(left))
}

'''
text = replace_once(text, anchor, helpers + anchor, "list operator helper insertion")
p.write_text(text)

# Bytecode distinguishes compound assignment from ordinary binary operators so
# list += mutates the aliased list while list + allocates a shallow copy.
p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()
text = replace_once(
    text,
    "use builtins::{\n    execute_list_method, execute_standard_builtin, is_subtype, standard_builtin_arity,\n};\n",
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
                    let left = scalar_number(left)?;
                    let right = scalar_number(right)?;
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
scalar_helpers = '''fn scalar_number(value: Value) -> Result<f32, RuntimeError> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(RuntimeError {
            message: format!("numeric operation received {value}"),
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        }),
    }
}

fn scalar_number_string(value: Value) -> Result<f32, String> {
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

# scalar_number above should not manufacture RuntimeError; replace its use in the
# non-compound operator arm with the String-returning helper and remove it.
text = text.replace("                    let left = scalar_number(left)?;\n                    let right = scalar_number(right)?;\n", "                    let left = scalar_number_string(left)\n                        .map_err(|message| execution_error(module, &frames, message))?;\n                    let right = scalar_number_string(right)\n                        .map_err(|message| execution_error(module, &frames, message))?;\n", 1)
text = re.sub(r'''fn scalar_number\(value: Value\) -> Result<f32, RuntimeError> \{.*?\n\}\n\n''', '', text, count=1, flags=re.S)

# Regressions: non-mutating operators return distinct lists, while compound
# operators retain list identity so aliases observe the mutation.
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
            Ok(Value::number(23.0))
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
            Ok(Value::number(16.0))
        );
    }

'''
text = replace_once(text, test_anchor, tests + test_anchor, "list operator regressions")
p.write_text(text)
