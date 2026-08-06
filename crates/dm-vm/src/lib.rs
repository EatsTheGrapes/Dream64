//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::HashMap;
use std::fmt;

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind};

/// A value supported by the initial executable DM subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    /// DM `null`.
    Null,
    /// Compatibility-mode binary32 `num`.
    Number(DmNumberBits),
    /// Immutable text.
    Text(String),
}

impl Value {
    /// Creates a numeric value without widening it.
    #[must_use]
    pub const fn number(value: f32) -> Self {
        Self::Number(DmNumberBits::from_f32(value))
    }

    /// Returns the stored number when this value is numeric.
    #[must_use]
    pub const fn as_number(&self) -> Option<f32> {
        match self {
            Self::Number(number) => Some(number.to_f32()),
            Self::Null | Self::Text(_) => None,
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Number(number) => number.to_f32() != 0.0,
            Self::Text(text) => !text.is_empty(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => formatter.write_str("null"),
            Self::Number(number) => write!(formatter, "{}", number.to_f32()),
            Self::Text(text) => write!(formatter, "{text:?}"),
        }
    }
}

/// One instruction in the portable reference bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Instruction {
    /// Pushes `null`.
    PushNull,
    /// Pushes a numeric constant.
    PushNumber(DmNumberBits),
    /// Pushes a text constant.
    PushText(String),
    /// Pushes a local value.
    LoadLocal(u16),
    /// Pops into a local slot.
    StoreLocal(u16),
    /// Numeric addition.
    Add,
    /// Numeric subtraction.
    Subtract,
    /// Numeric multiplication.
    Multiply,
    /// Numeric division.
    Divide,
    /// Numeric remainder.
    Remainder,
    /// Numeric negation.
    Negate,
    /// DM truth-value negation.
    Not,
    /// Equality comparison.
    Equal,
    /// Inequality comparison.
    NotEqual,
    /// Less-than comparison.
    Less,
    /// Less-than-or-equal comparison.
    LessEqual,
    /// Greater-than comparison.
    Greater,
    /// Greater-than-or-equal comparison.
    GreaterEqual,
    /// Boolean conjunction.
    And,
    /// Boolean disjunction.
    Or,
    /// Returns the top stack value.
    Return,
}

/// A compiled procedure body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    /// Required positional argument count for this initial subset.
    pub parameter_count: usize,
    /// Number of local slots, including parameters.
    pub local_count: usize,
    /// Portable instructions in execution order.
    pub instructions: Vec<Instruction>,
    /// Source line associated with each instruction for diagnostics/debugging.
    pub source_spans: Vec<SourceSpan>,
}

/// Failure while compiling the initial executable subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompileError {
    /// Human-readable diagnostic.
    pub message: String,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CompileError {}

/// Failure while executing portable bytecode.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeError {
    /// Human-readable runtime diagnostic.
    pub message: String,
    /// Instruction index at which execution failed.
    pub instruction: usize,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at instruction {}",
            self.message, self.instruction
        )
    }
}

impl std::error::Error for RuntimeError {}

/// Compiles one procedure definition to portable stack bytecode.
///
/// The current vertical slice supports positional parameters, local `var`
/// declarations, `return`, numeric and text literals, local reads, unary
/// operators, and common binary operators.
///
/// # Errors
///
/// Returns [`CompileError`] for unsupported statements, malformed expressions,
/// unknown locals, or non-procedure definitions.
pub fn compile_procedure(definition: &Definition) -> Result<Program, CompileError> {
    if !matches!(
        definition.kind,
        DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
    ) {
        return Err(compile_error("definition is not executable"));
    }

    let mut locals = HashMap::new();
    for (index, parameter) in definition.parameters.iter().enumerate() {
        let name = parameter_name(&parameter.tokens)
            .ok_or_else(|| compile_error("procedure parameter has no name"))?;
        locals.insert(name.to_owned(), to_local_index(index)?);
    }

    let mut instructions = Vec::new();
    let mut source_spans = Vec::new();
    let mut has_return = false;
    for line in &definition.body {
        let Some(first) = line.tokens.first() else {
            continue;
        };
        let first_instruction = instructions.len();
        match &first.kind {
            TokenKind::Identifier(keyword) if keyword == "return" => {
                if line.tokens.len() == 1 {
                    instructions.push(Instruction::PushNull);
                } else {
                    compile_expression(&line.tokens[1..], &locals, &mut instructions)?;
                }
                instructions.push(Instruction::Return);
                has_return = true;
            }
            TokenKind::Identifier(keyword) if keyword == "var" => {
                compile_local(&line.tokens, &mut locals, &mut instructions)?;
            }
            _ => {
                return Err(compile_error(format!(
                    "unsupported statement beginning with {:?}",
                    first.kind
                )));
            }
        }
        source_spans.extend(std::iter::repeat_n(
            line.span,
            instructions.len().saturating_sub(first_instruction),
        ));
    }
    if !has_return {
        instructions.push(Instruction::PushNull);
        instructions.push(Instruction::Return);
        source_spans.extend([definition.span, definition.span]);
    }

    Ok(Program {
        parameter_count: definition.parameters.len(),
        local_count: locals.len(),
        instructions,
        source_spans,
    })
}

fn compile_local(
    tokens: &[SpannedToken],
    locals: &mut HashMap<String, u16>,
    instructions: &mut Vec<Instruction>,
) -> Result<(), CompileError> {
    let assignment = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
        .ok_or_else(|| compile_error("local declaration requires an initializer"))?;
    let name = tokens[1..assignment]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.clone()),
            _ => None,
        })
        .ok_or_else(|| compile_error("local declaration has no name"))?;
    if locals.contains_key(&name) {
        return Err(compile_error(format!("local {name:?} is already declared")));
    }
    compile_expression(&tokens[assignment + 1..], locals, instructions)?;
    let slot = to_local_index(locals.len())?;
    locals.insert(name, slot);
    instructions.push(Instruction::StoreLocal(slot));
    Ok(())
}

fn parameter_name(tokens: &[SpannedToken]) -> Option<&str> {
    let end = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
        .unwrap_or(tokens.len());
    tokens[..end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.as_str()),
            _ => None,
        })
}

fn to_local_index(index: usize) -> Result<u16, CompileError> {
    u16::try_from(index).map_err(|_| compile_error("procedure has more than 65536 locals"))
}

fn compile_expression(
    tokens: &[SpannedToken],
    locals: &HashMap<String, u16>,
    instructions: &mut Vec<Instruction>,
) -> Result<(), CompileError> {
    let expression = ExpressionParser::new(tokens).parse()?;
    emit_expression(&expression, locals, instructions)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Expression {
    Null,
    Number(DmNumberBits),
    Text(String),
    Local(String),
    Unary {
        operator: String,
        operand: Box<Self>,
    },
    Binary {
        operator: String,
        left: Box<Self>,
        right: Box<Self>,
    },
}

struct ExpressionParser<'a> {
    tokens: &'a [SpannedToken],
    index: usize,
}

impl<'a> ExpressionParser<'a> {
    const fn new(tokens: &'a [SpannedToken]) -> Self {
        Self { tokens, index: 0 }
    }

    fn parse(mut self) -> Result<Expression, CompileError> {
        let expression = self.parse_binary(1)?;
        if self.index != self.tokens.len() {
            return Err(compile_error(format!(
                "unexpected token {:?} in expression",
                self.tokens[self.index].kind
            )));
        }
        Ok(expression)
    }

    fn parse_binary(&mut self, minimum_precedence: u8) -> Result<Expression, CompileError> {
        let mut left = self.parse_unary()?;
        while let Some(operator) = self.current_operator() {
            let Some(precedence) = binary_precedence(operator) else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            let operator = operator.to_owned();
            self.index += 1;
            let right = self.parse_binary(precedence + 1)?;
            left = Expression::Binary {
                operator,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, CompileError> {
        if let Some(operator @ ("!" | "+" | "-")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            return Ok(Expression::Unary {
                operator,
                operand: Box::new(self.parse_unary()?),
            });
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| compile_error("expected an expression"))?;
        self.index += 1;
        match &token.kind {
            TokenKind::Number(spelling) => parse_number(spelling).map(Expression::Number),
            TokenKind::String(text) | TokenKind::RawString(text) | TokenKind::TextBlock(text) => {
                Ok(Expression::Text(text.clone()))
            }
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(Expression::Null),
            TokenKind::Identifier(identifier) => Ok(Expression::Local(identifier.clone())),
            TokenKind::Punctuation('(') => {
                let expression = self.parse_binary(1)?;
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    Some(TokenKind::Punctuation(')')) => {
                        self.index += 1;
                        Ok(expression)
                    }
                    _ => Err(compile_error("expected ')' after expression")),
                }
            }
            _ => Err(compile_error(format!(
                "unexpected token {:?} in expression",
                token.kind
            ))),
        }
    }

    fn current_operator(&self) -> Option<&str> {
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) => Some(operator),
            _ => None,
        }
    }
}

fn parse_number(spelling: &str) -> Result<DmNumberBits, CompileError> {
    let normalized = spelling.replace('_', "");
    let value = if let Some(hexadecimal) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        let integer = u32::from_str_radix(hexadecimal, 16)
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?;
        integer
            .to_string()
            .parse::<f32>()
            .expect("every u32 decimal spelling is a valid f32")
    } else {
        normalized
            .parse::<f32>()
            .map_err(|error| compile_error(format!("invalid number {spelling:?}: {error}")))?
    };
    Ok(DmNumberBits::from_f32(value))
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"==" | b"!=" => Some(3),
        b"<" | b"<=" | b">" | b">=" => Some(4),
        b"+" | b"-" => Some(5),
        b"*" | b"/" | b"%" => Some(6),
        _ => None,
    }
}

fn emit_expression(
    expression: &Expression,
    locals: &HashMap<String, u16>,
    instructions: &mut Vec<Instruction>,
) -> Result<(), CompileError> {
    match expression {
        Expression::Null => instructions.push(Instruction::PushNull),
        Expression::Number(number) => instructions.push(Instruction::PushNumber(*number)),
        Expression::Text(text) => instructions.push(Instruction::PushText(text.clone())),
        Expression::Local(name) => {
            let slot = locals
                .get(name)
                .copied()
                .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
            instructions.push(Instruction::LoadLocal(slot));
        }
        Expression::Unary { operator, operand } => {
            emit_expression(operand, locals, instructions)?;
            match operator.as_str() {
                "+" => {}
                "-" => instructions.push(Instruction::Negate),
                "!" => instructions.push(Instruction::Not),
                _ => {
                    return Err(compile_error(format!(
                        "unsupported unary operator {operator}"
                    )));
                }
            }
        }
        Expression::Binary {
            operator,
            left,
            right,
        } => {
            emit_expression(left, locals, instructions)?;
            emit_expression(right, locals, instructions)?;
            instructions.push(match operator.as_str() {
                "+" => Instruction::Add,
                "-" => Instruction::Subtract,
                "*" => Instruction::Multiply,
                "/" => Instruction::Divide,
                "%" => Instruction::Remainder,
                "==" => Instruction::Equal,
                "!=" => Instruction::NotEqual,
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
    }
    Ok(())
}

fn compile_error(message: impl Into<String>) -> CompileError {
    CompileError {
        message: message.into(),
    }
}

/// Executes one program to completion on the reference interpreter.
///
/// # Errors
///
/// Returns [`RuntimeError`] for arity errors, invalid bytecode stack/local
/// access, or operations on values of unsupported types.
pub fn execute(program: &Program, arguments: &[Value]) -> Result<Value, RuntimeError> {
    if arguments.len() != program.parameter_count {
        return Err(runtime_error(
            0,
            format!(
                "expected {} arguments, received {}",
                program.parameter_count,
                arguments.len()
            ),
        ));
    }
    let mut locals = vec![Value::Null; program.local_count];
    locals[..arguments.len()].clone_from_slice(arguments);
    let mut stack = Vec::new();
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
        match instruction {
            Instruction::PushNull => stack.push(Value::Null),
            Instruction::PushNumber(number) => stack.push(Value::Number(*number)),
            Instruction::PushText(text) => stack.push(Value::Text(text.clone())),
            Instruction::LoadLocal(slot) => {
                let value = locals.get(usize::from(*slot)).cloned().ok_or_else(|| {
                    runtime_error(instruction_index, format!("invalid local slot {slot}"))
                })?;
                stack.push(value);
            }
            Instruction::StoreLocal(slot) => {
                let value = pop(&mut stack, instruction_index)?;
                let local = locals.get_mut(usize::from(*slot)).ok_or_else(|| {
                    runtime_error(instruction_index, format!("invalid local slot {slot}"))
                })?;
                *local = value;
            }
            Instruction::Negate => {
                let value = pop_number(&mut stack, instruction_index)?;
                stack.push(Value::number(-value));
            }
            Instruction::Not => {
                let value = pop(&mut stack, instruction_index)?;
                stack.push(Value::number(f32::from(!value.truthy())));
            }
            Instruction::Add
            | Instruction::Subtract
            | Instruction::Multiply
            | Instruction::Divide
            | Instruction::Remainder
            | Instruction::Less
            | Instruction::LessEqual
            | Instruction::Greater
            | Instruction::GreaterEqual => {
                let right = pop_number(&mut stack, instruction_index)?;
                let left = pop_number(&mut stack, instruction_index)?;
                let result = match instruction {
                    Instruction::Add => left + right,
                    Instruction::Subtract => left - right,
                    Instruction::Multiply => left * right,
                    Instruction::Divide => left / right,
                    Instruction::Remainder => left % right,
                    Instruction::Less => f32::from(left < right),
                    Instruction::LessEqual => f32::from(left <= right),
                    Instruction::Greater => f32::from(left > right),
                    Instruction::GreaterEqual => f32::from(left >= right),
                    _ => unreachable!("instruction came from the numeric operation group"),
                };
                stack.push(Value::number(result));
            }
            Instruction::Equal | Instruction::NotEqual => {
                let right = pop(&mut stack, instruction_index)?;
                let left = pop(&mut stack, instruction_index)?;
                let equal = values_equal(&left, &right);
                let result = if matches!(instruction, Instruction::NotEqual) {
                    !equal
                } else {
                    equal
                };
                stack.push(Value::number(f32::from(result)));
            }
            Instruction::And | Instruction::Or => {
                let right = pop(&mut stack, instruction_index)?.truthy();
                let left = pop(&mut stack, instruction_index)?.truthy();
                let result = if matches!(instruction, Instruction::And) {
                    left && right
                } else {
                    left || right
                };
                stack.push(Value::number(f32::from(result)));
            }
            Instruction::Return => return pop(&mut stack, instruction_index),
        }
    }
    Err(runtime_error(
        program.instructions.len(),
        "program ended without Return",
    ))
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        (Value::Number(left), Value::Number(right)) => {
            left.to_f32().partial_cmp(&right.to_f32()) == Some(std::cmp::Ordering::Equal)
        }
        (Value::Text(left), Value::Text(right)) => left == right,
        _ => false,
    }
}

fn pop(stack: &mut Vec<Value>, instruction: usize) -> Result<Value, RuntimeError> {
    stack
        .pop()
        .ok_or_else(|| runtime_error(instruction, "bytecode stack underflow"))
}

fn pop_number(stack: &mut Vec<Value>, instruction: usize) -> Result<f32, RuntimeError> {
    let value = pop(stack, instruction)?;
    value
        .as_number()
        .ok_or_else(|| runtime_error(instruction, format!("numeric operation received {value}")))
}

fn runtime_error(instruction: usize, message: impl Into<String>) -> RuntimeError {
    RuntimeError {
        message: message.into(),
        instruction,
    }
}

#[cfg(test)]
mod tests {
    use dm_syntax::parse;

    use super::{Value, compile_procedure, execute};

    fn execute_source(source: &str, argument: f32) -> Value {
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        execute(&program, &[Value::number(argument)]).expect("procedure should execute")
    }

    #[test]
    fn compiles_locals_and_executes_binary32_arithmetic() {
        let source = "/proc/probe(input)\n\tvar/doubled = input * 2\n\treturn doubled + 3\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let result = execute(&program, &[Value::number(4.0)]).expect("procedure should execute");

        assert_eq!(result, Value::number(11.0));
        assert_eq!(program.instructions.len(), program.source_spans.len());
        assert_eq!(program.source_spans[0], syntax.definitions[0].body[0].span);
    }

    #[test]
    fn observes_operator_precedence_and_parentheses() {
        let result = execute_source("/proc/probe(input)\n\treturn (input + 3) * 2\n", 4.0);

        assert_eq!(result, Value::number(14.0));
    }

    #[test]
    fn rejects_unknown_locals_during_compilation() {
        let syntax =
            parse("/proc/probe(input)\n\treturn missing + input\n").expect("source should parse");
        let error = compile_procedure(&syntax.definitions[0])
            .expect_err("unknown local should fail compilation");

        assert!(error.message.contains("unknown local"));
    }
}
