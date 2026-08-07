from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:1500]}")
    return text.replace(old, new, 1)


# Lexer: add documented comparison and fractional-modulo spellings.  **= is
# intentionally left alone as a recognized/reserved token; BYOND's documented
# calculate-and-assign operator table does not define it.
p = Path("crates/dm-lexer/src/lib.rs")
text = p.read_text()
old = '''            "<<=", ">>=", "&&=", "||=", "**=", "...", "::", "?.", "?:", "?[", "==", "!=", "<=",
            ">=", "<<", ">>", "&&", "||", "++", "--", "+=", "-=", "*=", "/=", "%=", "&=", "|=",
            "^=", "~=", "**", "..", "/", ".", ":", "?", "=", "+", "-", "*", "%", "<", ">", "!",
            "~", "&", "|", "^", "#", "@",
'''
new = '''            "<=>", "<<=", ">>=", "&&=", "||=", "%%=", "**=", "...", "::", "?.", "?:", "?[",
            "==", "!=", "<>", "<=", ">=", "~!", "<<", ">>", "&&", "||", "++", "--", "+=", "-=",
            "*=", "/=", "%=", "&=", "|=", "^=", "~=", "%%", "**", "..", "/", ".", ":", "?",
            "=", "+", "-", "*", "%", "<", ">", "!", "~", "&", "|", "^", "#", "@",
'''
text = replace_once(text, old, new, "documented operator lexer")
p.write_text(text)


p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()

# Bytecode surface for fractional modulo, equivalence, and three-way compare.
text = replace_once(
    text,
    '''    /// Numeric remainder.
    Remainder,
    /// 32-bit integer bitwise conjunction.
''',
    '''    /// Legacy integer remainder (`%`), after truncating both operands.
    Remainder,
    /// Fractional remainder (`%%`) without integer truncation.
    FractionalRemainder,
    /// 24-bit integer bitwise conjunction.
''',
    "fractional remainder instruction",
)
text = text.replace("    /// 32-bit integer bitwise disjunction.\n", "    /// 24-bit integer bitwise disjunction.\n", 1)
text = text.replace("    /// 32-bit integer bitwise exclusive disjunction.\n", "    /// 24-bit integer bitwise exclusive disjunction.\n", 1)
text = text.replace("    /// 32-bit integer left shift.\n", "    /// 24-bit integer left shift.\n", 1)
text = text.replace("    /// 32-bit arithmetic right shift.\n", "    /// 24-bit logical right shift.\n", 1)
text = replace_once(
    text,
    '''    /// Inequality comparison.
    NotEqual,
    /// List membership comparison.
''',
    '''    /// Inequality comparison.
    NotEqual,
    /// Shallow BYOND equivalence comparison (`~=`).
    Equivalent,
    /// Negated shallow equivalence comparison (`~!`).
    NotEquivalent,
    /// Three-way comparison (`<=>`), yielding -1, 0, or 1.
    Compare,
    /// List membership comparison.
''',
    "comparison instructions",
)

# Compound fractional modulo for locals/fields/globals and indexed numeric values.
text = replace_once(
    text,
    '''    /// Remainder assignment (`%=`).
    Remainder,
    /// Bitwise/list-mask assignment (`&=`).
''',
    '''    /// Legacy integer remainder assignment (`%=`).
    Remainder,
    /// Fractional remainder assignment (`%%=`).
    FractionalRemainder,
    /// Bitwise/list-mask assignment (`&=`).
''',
    "compound fractional remainder enum",
)
text = replace_once(
    text,
    '''    /// Remainder assignment (`%=`).
    Remainder,
    /// Bitwise conjunction assignment (`&=`).
''',
    '''    /// Legacy integer remainder assignment (`%=`).
    Remainder,
    /// Fractional remainder assignment (`%%=`).
    FractionalRemainder,
    /// Bitwise conjunction assignment (`&=`).
''',
    "indexed compound fractional remainder enum",
)

# Assignment grammar.
text = text.replace('                | "%="\n                | "&="', '                | "%="\n                | "%%="\n                | "&="', 2)

# Binary precedence: aliases/equivalence compare like equality, spaceship like
# relational operators, and %% shares multiplicative precedence.
text = replace_once(
    text,
    '''        b"==" | b"!=" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" => Some(9),
''',
    '''        b"==" | b"!=" | b"<>" | b"~=" | b"~!" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"<=>" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" | b"%%" => Some(9),
''',
    "documented binary precedence",
)

# Lower && and || as true short-circuit control flow and preserve the actual
# operand value, as BYOND specifies.  Other operators retain ordinary eager
# binary lowering.
old = '''        Expression::Binary {
            operator,
            left,
            right,
        } => {
            emit_expression(left, locals, instructions, procedures)?;
            emit_expression(right, locals, instructions, procedures)?;
            instructions.push(match operator.as_str() {
                "+" => Instruction::Add,
                "-" => Instruction::Subtract,
                "*" => Instruction::Multiply,
                "**" => Instruction::Power,
                "/" => Instruction::Divide,
                "%" => Instruction::Remainder,
                "&" => Instruction::BitAnd,
                "|" => Instruction::BitOr,
                "^" => Instruction::BitXor,
                "<<" => Instruction::ShiftLeft,
                ">>" => Instruction::ShiftRight,
                "==" => Instruction::Equal,
                "!=" => Instruction::NotEqual,
                "in" => Instruction::Contains,
                "<" => Instruction::Less,
                "<=" => Instruction::LessEqual,
                ">" => Instruction::Greater,
                ">=" => Instruction::GreaterEqual,
                "&&" => Instruction::And,
                "||" => Instruction::Or,
                _ => {
                    return Err(compile_error(format!(
                        "unsupported binary operator {operator}"
                    )));
                }
            });
        }
'''
new = '''        Expression::Binary {
            operator,
            left,
            right,
        } => {
            if operator == "&&" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, false_jump, end)?;
            } else if operator == "||" {
                emit_expression(left, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                let end_jump = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let false_target = instructions.len();
                patch_jump(instructions, false_jump, false_target)?;
                instructions.push(Instruction::Pop);
                emit_expression(right, locals, instructions, procedures)?;
                let end = instructions.len();
                patch_jump(instructions, end_jump, end)?;
            } else {
                emit_expression(left, locals, instructions, procedures)?;
                emit_expression(right, locals, instructions, procedures)?;
                instructions.push(match operator.as_str() {
                    "+" => Instruction::Add,
                    "-" => Instruction::Subtract,
                    "*" => Instruction::Multiply,
                    "**" => Instruction::Power,
                    "/" => Instruction::Divide,
                    "%" => Instruction::Remainder,
                    "%%" => Instruction::FractionalRemainder,
                    "&" => Instruction::BitAnd,
                    "|" => Instruction::BitOr,
                    "^" => Instruction::BitXor,
                    "<<" => Instruction::ShiftLeft,
                    ">>" => Instruction::ShiftRight,
                    "==" => Instruction::Equal,
                    "!=" | "<>" => Instruction::NotEqual,
                    "~=" => Instruction::Equivalent,
                    "~!" => Instruction::NotEquivalent,
                    "<=>" => Instruction::Compare,
                    "in" => Instruction::Contains,
                    "<" => Instruction::Less,
                    "<=" => Instruction::LessEqual,
                    ">" => Instruction::Greater,
                    ">=" => Instruction::GreaterEqual,
                    _ => {
                        return Err(compile_error(format!(
                            "unsupported binary operator {operator}"
                        )));
                    }
                });
            }
        }
'''
text = replace_once(text, old, new, "short-circuit/value binary lowering")

# Compound operator mappings.
text = replace_once(
    text,
    '''        "%=" => CompoundAssignmentOperator::Remainder,
        "&=" => CompoundAssignmentOperator::BitAnd,
''',
    '''        "%=" => CompoundAssignmentOperator::Remainder,
        "%%=" => CompoundAssignmentOperator::FractionalRemainder,
        "&=" => CompoundAssignmentOperator::BitAnd,
''',
    "compound fractional mapping",
)
text = replace_once(
    text,
    '''        "%=" => CompoundListIndexOperator::Remainder,
        "&=" => CompoundListIndexOperator::BitAnd,
''',
    '''        "%=" => CompoundListIndexOperator::Remainder,
        "%%=" => CompoundListIndexOperator::FractionalRemainder,
        "&=" => CompoundListIndexOperator::BitAnd,
''',
    "indexed compound fractional mapping",
)

# Interpreter: fractional modulo joins numeric arithmetic, while relational
# comparison handles both number/null and text operands. Equivalence uses the
# heap because lists compare shallow contents and 516+ associated values.
old = '''            Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight
            | Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(execute_numeric_binary(
                        &instruction,
                        left,
                        right,
                    )));
            }
            Instruction::Equal | Instruction::NotEqual => {
'''
new = '''            Instruction::Multiply
            | Instruction::Power
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::FractionalRemainder
            | Instruction::ShiftLeft
            | Instruction::ShiftRight => {
                let right = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(execute_numeric_binary(
                        &instruction,
                        left,
                        right,
                    )));
            }
            Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual
            | Instruction::Compare => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let comparison = compare_values(&left, &right)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = match instruction {
                    Instruction::Less => comparison.is_some_and(|value| value.is_lt()),
                    Instruction::LessEqual => comparison.is_some_and(|value| value.is_le()),
                    Instruction::Greater => comparison.is_some_and(|value| value.is_gt()),
                    Instruction::GreaterEqual => comparison.is_some_and(|value| value.is_ge()),
                    Instruction::Compare => {
                        let value = comparison.map_or(0.0, |value| match value {
                            std::cmp::Ordering::Less => -1.0,
                            std::cmp::Ordering::Equal => 0.0,
                            std::cmp::Ordering::Greater => 1.0,
                        });
                        frames[frame_index].stack.push(Value::number(value));
                        continue;
                    }
                    _ => unreachable!(),
                };
                frames[frame_index].stack.push(Value::number(f32::from(result)));
            }
            Instruction::Equal | Instruction::NotEqual => {
'''
text = replace_once(text, old, new, "documented comparison interpreter")

# Equivalence and safe `in`: non-list RHS is simply false per BYOND.
anchor = '''            Instruction::Contains => {
'''
equiv = '''            Instruction::Equivalent | Instruction::NotEquivalent => {
                let right = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let left = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let equivalent = values_equivalent(&left, &right, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let result = if matches!(instruction, Instruction::NotEquivalent) {
                    !equivalent
                } else {
                    equivalent
                };
                frames[frame_index].stack.push(Value::number(f32::from(result)));
            }
'''
text = replace_once(text, anchor, equiv + anchor, "equivalence interpreter")
old = '''            Instruction::Contains => {
                let list = match pop(&mut frames[frame_index].stack) {
                    Ok(Value::List(list)) => list,
                    Ok(value) => {
                        return Err(execution_error(
                            module,
                            &frames,
                            format!("right operand of 'in' must be a list, received {value}"),
                        ));
                    }
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let needle = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let contains = state
                    .heap
                    .list(list)
                    .map_err(|error| execution_error(module, &frames, error.to_string()))?
                    .positions()
                    .any(|(_, value)| values_equal(&needle, value));
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(contains)));
            }
'''
new = '''            Instruction::Contains => {
                let container = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let needle = pop(&mut frames[frame_index].stack)
                    .map_err(|message| execution_error(module, &frames, message))?;
                let contains = if let Value::List(list) = container {
                    state
                        .heap
                        .list(list)
                        .map_err(|error| execution_error(module, &frames, error.to_string()))?
                        .positions()
                        .any(|(_, value)| values_equal(&needle, value))
                } else {
                    false
                };
                frames[frame_index].stack.push(Value::number(f32::from(contains)));
            }
'''
text = replace_once(text, old, new, "safe non-list membership")

# Numeric operator semantics: % truncates to integers; %% is fractional.  DM
# bitwise operations expose 24 effective bits rather than signed 32-bit math.
text = replace_once(
    text,
    '''        Instruction::Remainder => left % right,
        Instruction::BitAnd => bitwise_binary(left, right, |left, right| left & right),
''',
    '''        Instruction::Remainder => integer_remainder(left, right),
        Instruction::FractionalRemainder => fractional_remainder(left, right),
        Instruction::BitAnd => bitwise_binary(left, right, |left, right| left & right),
''',
    "numeric remainder semantics",
)
text = replace_once(
    text,
    '''        CompoundListIndexOperator::Remainder => left % right,
        CompoundListIndexOperator::BitAnd => {
''',
    '''        CompoundListIndexOperator::Remainder => integer_remainder(left, right),
        CompoundListIndexOperator::FractionalRemainder => fractional_remainder(left, right),
        CompoundListIndexOperator::BitAnd => {
''',
    "indexed remainder semantics",
)
text = replace_once(
    text,
    '''        CompoundAssignmentOperator::Remainder => left % right,
        CompoundAssignmentOperator::BitAnd => bitwise_binary(left, right, |a, b| a & b),
''',
    '''        CompoundAssignmentOperator::Remainder => integer_remainder(left, right),
        CompoundAssignmentOperator::FractionalRemainder => fractional_remainder(left, right),
        CompoundAssignmentOperator::BitAnd => bitwise_binary(left, right, |a, b| a & b),
''',
    "compound remainder semantics",
)

old = '''/// DM bitwise operations coerce their binary32 numeric operands to signed
/// 32-bit integers by truncation and return the resulting integer as a DM
/// number. Rust's float-to-int conversion also gives deterministic saturation
/// for values outside the integer range.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_binary(left: f32, right: f32, operation: impl FnOnce(i32, i32) -> i32) -> f32 {
    operation(left as i32, right as i32) as f32
}

/// Executes a DM bitwise complement after integer coercion.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_not(value: f32) -> f32 {
    (!(value as i32)) as f32
}

/// Executes a DM shift after integer coercion. Shift counts are masked to the
/// low five bits, matching the fixed-width 32-bit integer representation.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "DM bitwise operators intentionally bridge binary32 numbers and signed 32-bit integers"
)]
fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(i32, u32) -> i32) -> f32 {
    let count = u32::from_ne_bytes((right as i32).to_ne_bytes()) & 31;
    operation(left as i32, count) as f32
}
'''
new = '''const DM_BIT_MASK: u32 = (1 << 24) - 1;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dm_u24(value: f32) -> u32 {
    (value.trunc() as i64 as u32) & DM_BIT_MASK
}

#[allow(clippy::cast_precision_loss)]
fn bitwise_binary(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    (operation(dm_u24(left), dm_u24(right)) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_precision_loss)]
fn bitwise_not(value: f32) -> f32 {
    ((!dm_u24(value)) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    let count = right.trunc().max(0.0) as u32;
    if count >= 24 {
        return 0.0;
    }
    (operation(dm_u24(left), count) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn integer_remainder(left: f32, right: f32) -> f32 {
    let left = left.trunc() as i32;
    let right = right.trunc() as i32;
    if right == 0 {
        f32::NAN
    } else {
        (left % right) as f32
    }
}

fn fractional_remainder(left: f32, right: f32) -> f32 {
    if right == 0.0 {
        f32::NAN
    } else {
        right * (left / right).fract()
    }
}
'''
text = replace_once(text, old, new, "24-bit bitwise semantics")

# Existing shift callers use signed closures; make them u32 closures implicitly.
# The textual operators remain identical and infer the new integer type.

# Add comparison/equivalence helpers before jump validation.
anchor = '''fn validate_jump(target: usize, instruction_count: usize) -> Result<(), String> {
'''
helpers = '''fn compare_values(left: &Value, right: &Value) -> Result<Option<std::cmp::Ordering>, String> {
    match (left, right) {
        (Value::Text(left), Value::Text(right)) => Ok(Some(left.as_ref().cmp(right.as_ref()))),
        (Value::Null | Value::Number(_), Value::Null | Value::Number(_)) => {
            let left = match left {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            let right = match right {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            Ok(left.partial_cmp(&right))
        }
        _ => Err(format!("comparison requires two numbers or two text values, received {left} and {right}")),
    }
}

fn values_equivalent(left: &Value, right: &Value, heap: &ValueHeap) -> Result<bool, String> {
    let (Value::List(left_id), Value::List(right_id)) = (left, right) else {
        return Ok(left.semantic_eq(right));
    };
    let left = heap.list(*left_id).map_err(|error| error.to_string())?;
    let right = heap.list(*right_id).map_err(|error| error.to_string())?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for index in 1..=left.len() {
        let left_key = left.get(index).map_err(|error| error.to_string())?;
        let right_key = right.get(index).map_err(|error| error.to_string())?;
        if !left_key.semantic_eq(right_key) {
            return Ok(false);
        }
        let left_assoc = left.get_key(left_key).cloned().unwrap_or(Value::Null);
        let right_assoc = right.get_key(right_key).cloned().unwrap_or(Value::Null);
        if !left_assoc.semantic_eq(&right_assoc) {
            return Ok(false);
        }
    }
    Ok(true)
}

'''
text = replace_once(text, anchor, helpers + anchor, "comparison helper insertion")

# The legacy interpreter instructions remain for compatibility but newly
# compiled logical expressions no longer emit them. Their behavior is left
# unchanged because module bytecode is not a stable external ABI yet.

# Correct the old test that encoded signed-32-bit semantics and add focused
# regressions for the documented operators.
old_test_name = '    fn shift_operators_and_compound_assignments_use_signed_32_bit_semantics() {'
if old_test_name in text:
    text = text.replace(old_test_name, '    fn shift_operators_and_compound_assignments_use_byond_24_bit_semantics() {', 1)
# Update its expectations/comments for 24-bit shifts without depending on exact
# source formatting elsewhere.
text = text.replace('        // shifts preserve the sign bit, and a count of 33 masks to one.\n',
                    '        // BYOND shifts are limited to the low 24 bits; counts >=24 yield zero.\n', 1)
text = text.replace('            Ok(Value::number(64.0))\n        );\n        assert!(program.instructions.iter().any(|instruction| {',
                    '            Ok(Value::number(62.0))\n        );\n        assert!(program.instructions.iter().any(|instruction| {', 1)

test_anchor = '''    #[test]
    fn conditional_expressions_associate_right() {
'''
test = r'''    #[test]
    fn documented_operator_semantics_cover_short_circuit_modulo_compare_and_equivalence() {
        let source = parse(
            "/proc/probe()\n\tvar/list/a = list(\"key\" = 7, 2)\n\tvar/list/b = list(\"key\" = 7, 2)\n\tvar/list/c = list(\"key\" = 8, 2)\n\tvar/legacy = 5.9 % 2.1\n\tvar/fractional = 5.5 %% 2\n\tlegacy %= 2\n\tfractional %%= 1.25\n\tif((a ~= b) != 1 || (a ~! c) != 1)\n\t\treturn -100\n\tif((3 <=> 4) != -1 || (\"b\" <=> \"a\") != 1 || (1 <> 2) != 1)\n\t\treturn -101\n\tif((99 in null) != 0)\n\t\treturn -102\n\tvar/or_value = \"\" || \"fallback\"\n\tvar/and_value = \"left\" && \"right\"\n\tvar/skip_or = 1 || list()[99]\n\tvar/skip_and = 0 && list()[99]\n\tif(or_value != \"fallback\" || and_value != \"right\" || skip_or != 1 || skip_and != 0)\n\t\treturn -103\n\treturn legacy + fractional\n",
        )
        .expect("documented operator source should parse");
        let module = compile_module(&source.definitions).expect("documented operators should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(1.25))
        );
    }

    #[test]
    fn bitwise_operators_use_byonds_24_effective_bits() {
        let source = parse(
            "/proc/probe()\n\tvar/a = ~0\n\tvar/b = 1 << 24\n\tvar/c = 0xFFFFFF >> 23\n\treturn a + b + c\n",
        )
        .expect("bitwise source should parse");
        let module = compile_module(&source.definitions).expect("bitwise source should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/probe").unwrap(), &[]),
            Ok(Value::number(16_777_216.0))
        );
    }

'''
text = replace_once(text, test_anchor, test + test_anchor, "documented operator regressions")
p.write_text(text)


# Standard pure procedures: add inexpensive documented helpers that are useful
# throughout DM code without requiring renderer/network state.
p = Path("crates/dm-vm/src/builtins.rs")
text = p.read_text()
text = replace_once(
    text,
    '''        | "isinf" | "isnan" | "ckey" | "fexists" | "file2text" => (1, 1),
''',
    '''        | "isinf" | "isnan" | "ckey" | "fexists" | "file2text" | "lentext" | "list2params"
        | "params2list" => (1, 1),
''',
    "single-argument standard builtins",
)
text = replace_once(
    text,
    '''        "cmptext" | "cmptextEx" => (1, usize::MAX),
''',
    '''        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" => (0, usize::MAX),
        "num2text" => (1, 3),
''',
    "sort and num2text arity",
)
text = replace_once(
    text,
    '''        "file2text" => file2text(arguments, state),
        _ => Err(format!("unknown native DM builtin {name:?}")),
''',
    '''        "file2text" => file2text(arguments, state),
        "lentext" => lentext(arguments, state),
        "sorttext" => sorttext(arguments, state, false),
        "sorttextEx" | "sortText" => sorttext(arguments, state, true),
        "num2text" => num2text(arguments),
        "list2params" => list2params(arguments, state),
        "params2list" => params2list(arguments, state),
        _ => Err(format!("unknown native DM builtin {name:?}")),
''',
    "standard builtin dispatch",
)

anchor = '''fn unary_number(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
'''
helpers = r'''fn lentext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "lentext")?;
    Ok(Value::number(text.len() as f32))
}

fn sorttext(arguments: &[Value], state: &ExecutionState, exact: bool) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Ok(Value::number(0.0));
    }
    let values = arguments
        .iter()
        .map(|value| strict_text(value, state, "sorttext"))
        .collect::<Result<Vec<_>, _>>()?;
    let compare = |left: &str, right: &str| {
        if exact {
            left.cmp(right)
        } else {
            left.to_lowercase().cmp(&right.to_lowercase())
        }
    };
    let ascending = values.windows(2).all(|pair| compare(&pair[0], &pair[1]).is_lt());
    let descending = values.windows(2).all(|pair| compare(&pair[0], &pair[1]).is_gt());
    Ok(Value::number(if ascending { 1.0 } else if descending { -1.0 } else { 0.0 }))
}

fn num2text(arguments: &[Value]) -> Result<Value, String> {
    let value = number(&arguments[0], "num2text")?;
    if arguments.len() == 3 {
        let digits = number(&arguments[1], "num2text digits")?.trunc().max(0.0) as usize;
        let radix = number(&arguments[2], "num2text radix")?.trunc() as u32;
        if !(2..=36).contains(&radix) {
            return Err(format!("num2text radix {radix} is outside 2..=36"));
        }
        let negative = value.is_sign_negative();
        let mut integer = value.abs().trunc() as u32;
        let alphabet = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut encoded = Vec::new();
        loop {
            encoded.push(alphabet[(integer % radix) as usize] as char);
            integer /= radix;
            if integer == 0 {
                break;
            }
        }
        while encoded.len() < digits {
            encoded.push('0');
        }
        if negative {
            encoded.push('-');
        }
        encoded.reverse();
        return Ok(Value::text(encoded.into_iter().collect::<String>()));
    }
    let sigfig = arguments
        .get(1)
        .map_or(Ok(6_usize), |value| number(value, "num2text sigfig").map(|value| value.trunc().max(1.0) as usize))?;
    let plain = value.to_string();
    let significant_digits = plain.chars().filter(char::is_ascii_digit).count();
    if significant_digits <= sigfig || value == 0.0 {
        return Ok(Value::text(plain));
    }
    Ok(Value::text(format!("{:.*e}", sigfig.saturating_sub(1), value)))
}

fn form_encode(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => output.push(char::from(byte)),
            b' ' => output.push('+'),
            _ => output.push_str(&format!("%{byte:02X}")),
        }
    }
    output
}

fn form_decode(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).map_err(|error| error.to_string())?;
                let byte = u8::from_str_radix(hex, 16).map_err(|_| format!("invalid parameter escape %{hex}"))?;
                output.push(byte);
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output).map_err(|error| format!("parameter text is not UTF-8: {error}"))
}

fn list2params(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list_id) = arguments[0] else {
        return Err(format!("list2params requires a list, received {}", arguments[0]));
    };
    let list = state.heap.list(list_id).map_err(|error| error.to_string())?;
    let mut pairs = Vec::with_capacity(list.len());
    for (_, key) in list.positions() {
        let key_text = runtime_text(key, state, "list2params key")?;
        let associated = list.get_key(key).cloned().unwrap_or(Value::Null);
        let value_text = runtime_text(&associated, state, "list2params value")?;
        pairs.push(format!("{}={}", form_encode(&key_text), form_encode(&value_text)));
    }
    Ok(Value::text(pairs.join("&")))
}

fn params2list(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let params = strict_text(&arguments[0], state, "params2list")?;
    let result = state.heap.allocate_list();
    for part in params.split(['&', ';']) {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        let key = Value::text(form_decode(key)?);
        let value = Value::text(form_decode(value)?);
        state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?
            .set_key(key, value);
    }
    Ok(Value::List(result))
}

'''
text = replace_once(text, anchor, helpers + anchor, "pure documented builtin helpers")
p.write_text(text)
