//! Portable stack bytecode and the deterministic reference interpreter.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::HashMap;
use std::fmt;

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind, SourceLine};

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
    /// Discards the top stack value.
    Pop,
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
    /// Pops a condition and jumps to an absolute instruction when it is false.
    JumpIfFalse(usize),
    /// Jumps to an absolute instruction.
    Jump(usize),
    /// Skips a parameter default when that argument was explicitly supplied.
    JumpIfArgumentSupplied {
        /// Zero-based declared parameter index.
        parameter: u16,
        /// Absolute instruction after the parameter's default initializer.
        target: usize,
    },
    /// Calls a procedure with positional values popped from the stack.
    Call {
        /// Stable module-local procedure identity.
        procedure: ProcedureId,
        /// Number of positional values supplied by the caller.
        argument_count: u16,
    },
    /// Calls the currently executing procedure.
    CallCurrent {
        /// Explicit argument count, or `None` to reuse the frame's complete
        /// originally supplied argument vector.
        argument_count: Option<u16>,
    },
    /// Calls the semantically resolved parent implementation.
    CallParent {
        /// Resolved module-local target, or `None` when no parent exists.
        procedure: Option<ProcedureId>,
        /// Explicit argument count, or `None` to reuse the complete original
        /// argument vector of the current frame.
        argument_count: Option<u16>,
    },
    /// Returns the top stack value.
    Return,
}

/// Stable procedure identity within one compiled module.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProcedureId(u32);

impl ProcedureId {
    fn from_index(index: usize) -> Result<Self, CompileError> {
        u32::try_from(index)
            .map(Self)
            .map_err(|_| compile_error("module has more than u32::MAX procedures"))
    }

    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A compiled procedure body.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Program {
    /// Declared positional parameter count.
    pub parameter_count: usize,
    /// Number of local slots, including parameters.
    pub local_count: usize,
    /// Portable instructions in execution order.
    pub instructions: Vec<Instruction>,
    /// Source line associated with each instruction for diagnostics/debugging.
    pub source_spans: Vec<SourceSpan>,
}

/// A deterministic table of compiled procedures and their canonical paths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Module {
    procedures: Vec<Program>,
    paths: Vec<String>,
    names: HashMap<String, ProcedureId>,
}

impl Module {
    /// Looks up a procedure by canonical path, such as `/proc/main`.
    #[must_use]
    pub fn procedure_id(&self, path: &str) -> Option<ProcedureId> {
        self.names.get(path).copied()
    }

    /// Returns a compiled procedure by module-local identity.
    #[must_use]
    pub fn procedure(&self, procedure: ProcedureId) -> Option<&Program> {
        self.procedures.get(procedure.index())
    }

    /// Returns the canonical path associated with a procedure.
    #[must_use]
    pub fn procedure_path(&self, procedure: ProcedureId) -> Option<&str> {
        self.paths.get(procedure.index()).map(String::as_str)
    }

    /// Returns the stable identity at a procedure-spec index.
    #[must_use]
    pub fn procedure_id_at(&self, index: usize) -> Option<ProcedureId> {
        self.procedures.get(index)?;
        u32::try_from(index).ok().map(ProcedureId)
    }
}

/// One independently identified procedure body supplied by a semantic layer.
#[derive(Clone, Debug)]
pub struct ProcedureSpec<'definition> {
    /// Unique diagnostic path for stack traces and lookup.
    pub path: String,
    /// Parsed procedure definition to compile.
    pub definition: &'definition Definition,
    /// Index of the exact parent implementation in the same spec slice.
    pub parent: Option<usize>,
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
    /// Source span associated with the failing instruction, when available.
    pub source_span: Option<SourceSpan>,
    /// Active procedures from the entry point through the failing frame.
    pub call_stack: Vec<CallTrace>,
}

/// One source-mapped procedure in a runtime error's call stack.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CallTrace {
    /// Canonical procedure path.
    pub procedure: String,
    /// Instruction active in this frame.
    pub instruction: usize,
    /// Source span associated with the active instruction, when available.
    pub source_span: Option<SourceSpan>,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at instruction {}",
            self.message, self.instruction
        )?;
        if let Some(span) = self.source_span {
            write!(formatter, " (source {}..{})", span.start, span.end)?;
        }
        Ok(())
    }
}

impl std::error::Error for RuntimeError {}

/// Compiles one procedure definition to portable stack bytecode.
///
/// The current vertical slice supports positional parameters and safe default
/// expressions, local `var` declarations, assignment, structured control flow,
/// numeric and text literals, local reads, procedure calls, unary operators,
/// and common binary operators.
///
/// # Errors
///
/// Returns [`CompileError`] for unsupported statements, malformed expressions,
/// unknown locals, or non-procedure definitions.
pub fn compile_procedure(definition: &Definition) -> Result<Program, CompileError> {
    compile_procedure_with_resolver(definition, &HashMap::new())
}

/// Compiles a deterministic module from procedure definitions in source order.
///
/// This initial call-resolution slice exposes global `/proc/name` procedures to
/// unqualified `name(...)` expressions. Object dispatch and overloads belong to
/// the later object-tree semantic pass.
///
/// # Errors
///
/// Returns [`CompileError`] when a definition is not executable, a canonical
/// procedure path is duplicated, or any procedure body cannot be compiled.
pub fn compile_module(definitions: &[Definition]) -> Result<Module, CompileError> {
    let mut names = HashMap::new();
    let mut call_names = HashMap::new();
    let mut paths = Vec::with_capacity(definitions.len());
    for (index, definition) in definitions.iter().enumerate() {
        if !matches!(
            definition.kind,
            DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
        ) {
            return Err(compile_error(format!(
                "definition {} is not executable",
                definition.path
            )));
        }
        let procedure = ProcedureId::from_index(index)?;
        let path = definition.path.to_string();
        if names.insert(path.clone(), procedure).is_some() {
            return Err(compile_error(format!("duplicate procedure path {path:?}")));
        }
        let segments = definition.path.segments();
        if segments.len() == 2
            && matches!(segments[0].as_str(), "proc" | "verb")
            && call_names.insert(segments[1].clone(), procedure).is_some()
        {
            return Err(compile_error(format!(
                "ambiguous global procedure name {:?}",
                segments[1]
            )));
        }
        paths.push(path);
    }

    let procedures = definitions
        .iter()
        .map(|definition| compile_procedure_with_resolver(definition, &call_names))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
    })
}

/// Compiles procedure bodies whose exact parent implementations were resolved
/// by an independent semantic layer.
///
/// Spec order defines stable module-local identities. Parent indices may point
/// forward or backward, but must refer to this same slice. Diagnostic paths
/// must be unique. Unqualified global call resolution remains the concern of
/// [`compile_module`]; this API focuses on already-resolved implementation
/// chains.
///
/// # Errors
///
/// Returns [`CompileError`] for duplicate paths, invalid parent indices, or
/// procedure bodies outside the supported executable subset.
pub fn compile_module_specs(specs: &[ProcedureSpec<'_>]) -> Result<Module, CompileError> {
    let mut names = HashMap::new();
    let mut paths = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let procedure = ProcedureId::from_index(index)?;
        if names.insert(spec.path.clone(), procedure).is_some() {
            return Err(compile_error(format!(
                "duplicate procedure spec path {:?}",
                spec.path
            )));
        }
        if spec.parent.is_some_and(|parent| parent >= specs.len()) {
            return Err(compile_error(format!(
                "procedure spec {:?} has invalid parent index {:?}",
                spec.path, spec.parent
            )));
        }
        paths.push(spec.path.clone());
    }

    let procedures = specs
        .iter()
        .map(|spec| {
            let mut targets = HashMap::new();
            if let Some(parent) = spec.parent {
                targets.insert("..".to_owned(), ProcedureId::from_index(parent)?);
            }
            compile_procedure_with_resolver(spec.definition, &targets)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Module {
        procedures,
        paths,
        names,
    })
}

fn compile_procedure_with_resolver(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<Program, CompileError> {
    if !matches!(
        definition.kind,
        DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
    ) {
        return Err(compile_error("definition is not executable"));
    }

    let mut locals = LocalTable::default();
    for (index, parameter) in definition.parameters.iter().enumerate() {
        let name = parameter_name(&parameter.tokens)
            .ok_or_else(|| compile_error("procedure parameter has no name"))?;
        locals.insert_parameter(name.to_owned(), to_local_index(index)?);
    }

    let mut instructions = Vec::new();
    let mut source_spans = Vec::new();
    let mut loops = Vec::new();
    compile_parameter_defaults(
        definition,
        &locals,
        &mut instructions,
        &mut source_spans,
        procedures,
    )?;
    let falls_through = if let Some(first_line) = definition.body.first() {
        let block_indentation = indentation(first_line);
        let (next_line, falls_through) = compile_block(
            &definition.body,
            0,
            block_indentation,
            &mut locals,
            &mut instructions,
            &mut source_spans,
            procedures,
            &mut loops,
        )?;
        if next_line != definition.body.len() {
            return Err(compile_error("procedure body contains invalid indentation"));
        }
        falls_through
    } else {
        true
    };
    if falls_through {
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::PushNull,
            definition.span,
        );
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::Return,
            definition.span,
        );
    }

    Ok(Program {
        parameter_count: definition.parameters.len(),
        local_count: locals.slot_count,
        instructions,
        source_spans,
    })
}

#[derive(Default)]
struct LocalTable {
    names: HashMap<String, u16>,
    slot_count: usize,
}

impl LocalTable {
    fn insert_parameter(&mut self, name: String, slot: u16) {
        self.names.insert(name, slot);
        self.slot_count = self.slot_count.max(usize::from(slot) + 1);
    }

    fn declare(&mut self, name: String) -> Result<u16, CompileError> {
        if self.names.contains_key(&name) {
            return Err(compile_error(format!("local {name:?} is already declared")));
        }
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        self.names.insert(name, slot);
        Ok(slot)
    }

    fn get(&self, name: &str) -> Option<u16> {
        self.names.get(name).copied()
    }

    fn remove(&mut self, name: &str) {
        self.names.remove(name);
    }
}

fn compile_parameter_defaults(
    definition: &Definition,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    for (parameter_index, parameter) in definition.parameters.iter().enumerate() {
        let Some(assignment) = parameter.tokens.iter().position(
            |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        ) else {
            continue;
        };
        let default_tokens = &parameter.tokens[assignment + 1..];
        if default_tokens.is_empty() {
            return Err(compile_error("procedure parameter default is empty"));
        }
        let parameter_slot = to_local_index(parameter_index)?;
        let default_jump = instructions.len();
        push_instruction(
            instructions,
            source_spans,
            Instruction::JumpIfArgumentSupplied {
                parameter: parameter_slot,
                target: usize::MAX,
            },
            parameter.span,
        );
        let expression = ExpressionParser::new(default_tokens).parse()?;
        validate_parameter_default(&expression)?;
        let first_default_instruction = instructions.len();
        emit_expression(&expression, locals, instructions, procedures)?;
        instructions.push(Instruction::StoreLocal(parameter_slot));
        source_spans.extend(std::iter::repeat_n(
            parameter.span,
            instructions.len() - first_default_instruction,
        ));
        let end_target = instructions.len();
        patch_jump(instructions, default_jump, end_target)?;
    }
    Ok(())
}

fn validate_parameter_default(expression: &Expression) -> Result<(), CompileError> {
    match expression {
        Expression::Null | Expression::Number(_) | Expression::Text(_) => Ok(()),
        Expression::Local(name) => Err(compile_error(format!(
            "parameter default reference {name:?} requires BYOND conformance confirmation"
        ))),
        Expression::Call { .. }
        | Expression::CurrentCall { .. }
        | Expression::ParentCall { .. } => Err(compile_error(
            "procedure calls in parameter defaults are not supported",
        )),
        Expression::Unary { operand, .. } => validate_parameter_default(operand),
        Expression::Binary { left, right, .. } => {
            validate_parameter_default(left)?;
            validate_parameter_default(right)
        }
    }
}

struct LoopContext {
    continue_target: Option<usize>,
    continue_jumps: Vec<usize>,
    break_jumps: Vec<usize>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_block(
    lines: &[SourceLine],
    mut line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let mut falls_through = true;
    while let Some(line) = lines.get(line_index) {
        let line_indentation = indentation(line);
        if line_indentation < block_indentation {
            break;
        }
        if line_indentation > block_indentation {
            return Err(compile_error("unexpected indentation in procedure body"));
        }
        let first = line
            .tokens
            .first()
            .expect("syntax source lines always contain tokens");
        match &first.kind {
            TokenKind::Identifier(keyword) if keyword == "if" => {
                let (next_line, statement_falls_through) = compile_if(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                falls_through &= statement_falls_through;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "while" => {
                let next_line = compile_while(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "for" => {
                let next_line = compile_for(
                    lines,
                    line_index,
                    block_indentation,
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?;
                line_index = next_line;
                continue;
            }
            TokenKind::Identifier(keyword) if keyword == "else" => {
                return Err(compile_error("else without a matching if"));
            }
            TokenKind::Identifier(keyword) if keyword == "break" => {
                if line.tokens.len() != 1 {
                    return Err(compile_error("break does not accept an expression"));
                }
                let Some(loop_context) = loops.last_mut() else {
                    return Err(compile_error("break outside a loop"));
                };
                let jump = instructions.len();
                push_instruction(
                    instructions,
                    source_spans,
                    Instruction::Jump(usize::MAX),
                    line.span,
                );
                loop_context.break_jumps.push(jump);
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "continue" => {
                if line.tokens.len() != 1 {
                    return Err(compile_error("continue does not accept an expression"));
                }
                let Some(loop_context) = loops.last_mut() else {
                    return Err(compile_error("continue outside a loop"));
                };
                let target = loop_context.continue_target.unwrap_or(usize::MAX);
                let jump = instructions.len();
                push_instruction(
                    instructions,
                    source_spans,
                    Instruction::Jump(target),
                    line.span,
                );
                if loop_context.continue_target.is_none() {
                    loop_context.continue_jumps.push(jump);
                }
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "return" => {
                let first_instruction = instructions.len();
                if line.tokens.len() == 1 {
                    instructions.push(Instruction::PushNull);
                } else {
                    compile_expression(&line.tokens[1..], locals, instructions, procedures)?;
                }
                instructions.push(Instruction::Return);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "var" => {
                let first_instruction = instructions.len();
                let _ = compile_local(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(name)
                if matches!(
                    line.tokens.get(1).map(|token| &token.kind),
                    Some(TokenKind::Operator(operator)) if operator == "="
                ) =>
            {
                let slot = locals
                    .get(name)
                    .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
                let first_instruction = instructions.len();
                compile_expression(&line.tokens[2..], locals, instructions, procedures)?;
                instructions.push(Instruction::StoreLocal(slot));
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            _ => {
                return Err(compile_error(format!(
                    "unsupported statement beginning with {:?}",
                    first.kind
                )));
            }
        }
        line_index += 1;
    }
    Ok((line_index, falls_through))
}

#[allow(clippy::too_many_arguments)]
fn compile_while(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let condition_target = instructions.len();
    let condition = condition_tokens(&line.tokens, "while")?;
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - condition_target,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("while statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("while statement requires an indented body"));
    }

    loops.push(LoopContext {
        continue_target: Some(condition_target),
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active while context was pushed");
    let (after_body, _) = body?;
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(condition_target),
        line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    Ok(after_body)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let [initializer, condition, increment] = for_clauses(&line.tokens)?;
    let initializer_start = instructions.len();
    let scoped_local = compile_for_clause(initializer, true, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initializer_start,
    ));

    let condition_target = instructions.len();
    if condition.is_empty() {
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    } else {
        compile_expression(condition, locals, instructions, procedures)?;
    }
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - condition_target,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("for statement requires an indented body"));
    }
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    let loop_context = loops.pop().expect("the active for context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    let increment_start = instructions.len();
    compile_for_clause(increment, false, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - increment_start,
    ));
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(condition_target),
        line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    if let Some(scoped_local) = scoped_local {
        locals.remove(&scoped_local);
    }
    Ok(after_body)
}

fn for_clauses(tokens: &[SpannedToken]) -> Result<[&[SpannedToken]; 3], CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        header.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Err(compile_error("C-style for requires a parenthesized header"));
    }
    let clauses = &header[1..header.len() - 1];
    let mut separators = Vec::new();
    let mut depth = 0_usize;
    for (index, token) in clauses.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(';') if depth == 0 => separators.push(index),
            _ => {}
        }
    }
    if separators.len() != 2 {
        if clauses.iter().any(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "in"),
        ) {
            return Err(compile_error("for-in list iteration is not implemented"));
        }
        return Err(compile_error(
            "C-style for requires initializer, condition, and increment clauses separated by ';'",
        ));
    }
    Ok([
        &clauses[..separators[0]],
        &clauses[separators[0] + 1..separators[1]],
        &clauses[separators[1] + 1..],
    ])
}

fn compile_for_clause(
    tokens: &[SpannedToken],
    allow_declaration: bool,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<Option<String>, CompileError> {
    if tokens.is_empty() {
        return Ok(None);
    }
    if matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "var"
    ) {
        if !allow_declaration {
            return Err(compile_error(
                "for increment clause cannot declare a local variable",
            ));
        }
        return compile_local(tokens, locals, instructions, procedures).map(Some);
    }
    if let [first, operator, expression @ ..] = tokens
        && let (TokenKind::Identifier(name), TokenKind::Operator(operator)) =
            (&first.kind, &operator.kind)
        && operator == "="
    {
        let slot = locals
            .get(name)
            .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
        compile_expression(expression, locals, instructions, procedures)?;
        instructions.push(Instruction::StoreLocal(slot));
        return Ok(None);
    }
    if let Some((name, increment)) = local_increment(tokens) {
        let slot = locals
            .get(name)
            .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
        instructions.push(Instruction::LoadLocal(slot));
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
        instructions.push(if increment {
            Instruction::Add
        } else {
            Instruction::Subtract
        });
        instructions.push(Instruction::StoreLocal(slot));
        return Ok(None);
    }
    compile_expression(tokens, locals, instructions, procedures)?;
    instructions.push(Instruction::Pop);
    Ok(None)
}

fn local_increment(tokens: &[SpannedToken]) -> Option<(&str, bool)> {
    let [first, second] = tokens else {
        return None;
    };
    match (&first.kind, &second.kind) {
        (TokenKind::Identifier(name), TokenKind::Operator(operator))
        | (TokenKind::Operator(operator), TokenKind::Identifier(name))
            if matches!(operator.as_str(), "++" | "--") =>
        {
            Some((name, operator == "++"))
        }
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_if(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let line = &lines[line_index];
    let first_instruction = instructions.len();
    let condition = condition_tokens(&line.tokens, "if")?;
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - first_instruction,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("if statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("if statement requires an indented body"));
    }
    let (after_then, then_falls_through) = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    if !lines
        .get(after_then)
        .is_some_and(|candidate| is_else(candidate, block_indentation))
    {
        let end_target = instructions.len();
        patch_jump(instructions, false_jump, end_target)?;
        return Ok((after_then, true));
    }

    let else_line = &lines[after_then];
    let end_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(usize::MAX),
        else_line.span,
    );
    let else_target = instructions.len();
    patch_jump(instructions, false_jump, else_target)?;
    let else_child_index = after_then + 1;
    let else_child = lines
        .get(else_child_index)
        .ok_or_else(|| compile_error("else statement requires an indented body"))?;
    let else_indentation = indentation(else_child);
    if else_indentation <= block_indentation {
        return Err(compile_error("else statement requires an indented body"));
    }
    let (after_else, else_falls_through) = compile_block(
        lines,
        else_child_index,
        else_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    let end_target = instructions.len();
    patch_jump(instructions, end_jump, end_target)?;
    Ok((after_else, then_falls_through || else_falls_through))
}

fn condition_tokens<'a>(
    tokens: &'a [SpannedToken],
    keyword: &str,
) -> Result<&'a [SpannedToken], CompileError> {
    let expression = &tokens[1..];
    if matches!(
        expression.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        if !matches!(
            expression.last().map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            return Err(compile_error(format!("{keyword} condition is missing ')'")));
        }
        return Ok(&expression[1..expression.len() - 1]);
    }
    if expression.is_empty() {
        return Err(compile_error(format!("{keyword} requires a condition")));
    }
    Ok(expression)
}

fn indentation(line: &SourceLine) -> usize {
    line.indentation
        .tabs
        .saturating_mul(8)
        .saturating_add(line.indentation.spaces)
}

fn is_else(line: &SourceLine, expected_indentation: usize) -> bool {
    indentation(line) == expected_indentation
        && matches!(
            line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "else"
        )
}

fn push_instruction(
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    instruction: Instruction,
    span: SourceSpan,
) {
    instructions.push(instruction);
    source_spans.push(span);
}

fn patch_jump(
    instructions: &mut [Instruction],
    instruction_index: usize,
    target: usize,
) -> Result<(), CompileError> {
    match instructions.get_mut(instruction_index) {
        Some(
            Instruction::JumpIfFalse(destination)
            | Instruction::Jump(destination)
            | Instruction::JumpIfArgumentSupplied {
                target: destination,
                ..
            },
        ) => {
            *destination = target;
            Ok(())
        }
        _ => Err(compile_error("compiler attempted to patch a non-jump")),
    }
}

fn compile_local(
    tokens: &[SpannedToken],
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<String, CompileError> {
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
    compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
    let slot = locals.declare(name.clone())?;
    instructions.push(Instruction::StoreLocal(slot));
    Ok(name)
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
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let expression = ExpressionParser::new(tokens).parse()?;
    emit_expression(&expression, locals, instructions, procedures)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Expression {
    Null,
    Number(DmNumberBits),
    Text(String),
    Local(String),
    Call {
        procedure: String,
        arguments: Vec<Self>,
    },
    CurrentCall {
        arguments: Option<Vec<Self>>,
    },
    ParentCall {
        arguments: Option<Vec<Self>>,
    },
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
            TokenKind::Operator(operator)
                if operator == ".."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::ParentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Operator(operator)
                if operator == "."
                    && matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::CurrentCall {
                    arguments: if arguments.is_empty() {
                        None
                    } else {
                        Some(arguments)
                    },
                })
            }
            TokenKind::Number(spelling) => parse_number(spelling).map(Expression::Number),
            TokenKind::String(text) | TokenKind::RawString(text) | TokenKind::TextBlock(text) => {
                Ok(Expression::Text(text.clone()))
            }
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(Expression::Null),
            TokenKind::Identifier(identifier)
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) =>
            {
                let arguments = self.parse_call_arguments()?;
                Ok(Expression::Call {
                    procedure: identifier.clone(),
                    arguments,
                })
            }
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

    fn parse_call_arguments(&mut self) -> Result<Vec<Expression>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut arguments = Vec::new();
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            loop {
                arguments.push(self.parse_binary(1)?);
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    Some(TokenKind::Punctuation(',')) => self.index += 1,
                    Some(TokenKind::Punctuation(')')) => break,
                    _ => {
                        return Err(compile_error(
                            "expected ',' or ')' after procedure argument",
                        ));
                    }
                }
            }
        }
        self.index += 1;
        Ok(arguments)
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
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match expression {
        Expression::Null => instructions.push(Instruction::PushNull),
        Expression::Number(number) => instructions.push(Instruction::PushNumber(*number)),
        Expression::Text(text) => instructions.push(Instruction::PushText(text.clone())),
        Expression::Local(name) => {
            let slot = locals
                .get(name)
                .ok_or_else(|| compile_error(format!("unknown local {name:?}")))?;
            instructions.push(Instruction::LoadLocal(slot));
        }
        Expression::Call {
            procedure,
            arguments,
        } => {
            let target = procedures
                .get(procedure)
                .copied()
                .ok_or_else(|| compile_error(format!("unknown procedure {procedure:?}")))?;
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Call {
                procedure: target,
                argument_count,
            });
        }
        Expression::CurrentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                let count = u16::try_from(arguments.len())
                    .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                Some(count)
            } else {
                None
            };
            instructions.push(Instruction::CallCurrent { argument_count });
        }
        Expression::ParentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                let count = u16::try_from(arguments.len())
                    .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                Some(count)
            } else {
                None
            };
            instructions.push(Instruction::CallParent {
                procedure: procedures.get("..").copied(),
                argument_count,
            });
        }
        Expression::Unary { operator, operand } => {
            emit_expression(operand, locals, instructions, procedures)?;
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
            emit_expression(left, locals, instructions, procedures)?;
            emit_expression(right, locals, instructions, procedures)?;
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

/// Limits applied by the deterministic reference interpreter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionLimits {
    /// Maximum number of simultaneously active procedure frames.
    pub max_call_depth: usize,
    /// Maximum total bytecode instructions executed across all call frames.
    pub max_steps: u64,
}

impl Default for ExecutionLimits {
    fn default() -> Self {
        Self {
            max_call_depth: 1_024,
            max_steps: 10_000_000,
        }
    }
}

#[derive(Debug)]
struct CallFrame {
    procedure: ProcedureId,
    instruction: usize,
    locals: Vec<Value>,
    stack: Vec<Value>,
    // Retain all supplied values for the future DM `args` list, including
    // extras beyond the declared parameter slots.
    arguments: Vec<Value>,
}

/// Executes one standalone program to completion on the reference interpreter.
///
/// Calls cannot occur in a standalone program; use [`execute_module`] for
/// programs produced by [`compile_module`].
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid bytecode stack/local access or
/// operations on values of unsupported types.
pub fn execute(program: &Program, arguments: &[Value]) -> Result<Value, RuntimeError> {
    execute_with_limits(program, arguments, ExecutionLimits::default())
}

/// Executes one standalone program with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute`], including
/// call-depth or total-instruction budget exhaustion.
pub fn execute_with_limits(
    program: &Program,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let entry = ProcedureId(0);
    let module = Module {
        procedures: vec![program.clone()],
        paths: vec!["<standalone>".to_owned()],
        names: HashMap::new(),
    };
    execute_module_with_limits(&module, entry, arguments, limits)
}

/// Executes a procedure from a compiled module with default safety limits.
///
/// Declared parameters are bound positionally. Missing parameters are `null`,
/// and extra supplied values are retained in the frame for future `args`
/// support, matching DM's permissive call arity.
///
/// # Errors
///
/// Returns [`RuntimeError`] for invalid procedure identities or bytecode,
/// unsupported value operations, and call-depth exhaustion.
pub fn execute_module(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
) -> Result<Value, RuntimeError> {
    execute_module_with_limits(module, entry, arguments, ExecutionLimits::default())
}

/// Executes a module procedure with explicit deterministic safety limits.
///
/// # Errors
///
/// Returns [`RuntimeError`] for the same failures as [`execute_module`].
pub fn execute_module_with_limits(
    module: &Module,
    entry: ProcedureId,
    arguments: &[Value],
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let Some(program) = module.procedure(entry) else {
        return Err(RuntimeError {
            message: format!("invalid entry procedure {}", entry.index()),
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        });
    };
    if limits.max_call_depth == 0 {
        return Err(RuntimeError {
            message: "maximum call depth must be at least one".to_owned(),
            instruction: 0,
            source_span: program.source_spans.first().copied(),
            call_stack: vec![trace(module, entry, 0)],
        });
    }

    let frames = vec![make_frame(entry, program, arguments)];
    run_frames(module, frames, limits)
}

#[allow(clippy::too_many_lines)]
fn run_frames(
    module: &Module,
    mut frames: Vec<CallFrame>,
    limits: ExecutionLimits,
) -> Result<Value, RuntimeError> {
    let mut remaining_steps = limits.max_steps;
    loop {
        let frame_index = frames.len() - 1;
        let procedure = frames[frame_index].procedure;
        let instruction_index = frames[frame_index].instruction;
        let Some(program) = module.procedure(procedure) else {
            return Err(execution_error(
                module,
                &frames,
                format!("invalid procedure {}", procedure.index()),
            ));
        };
        let Some(instruction) = program.instructions.get(instruction_index).cloned() else {
            return Err(execution_error(
                module,
                &frames,
                "program ended without Return",
            ));
        };
        if remaining_steps == 0 {
            return Err(execution_error(
                module,
                &frames,
                format!("instruction budget of {} exhausted", limits.max_steps),
            ));
        }
        remaining_steps -= 1;

        match instruction {
            Instruction::PushNull => frames[frame_index].stack.push(Value::Null),
            Instruction::PushNumber(number) => {
                frames[frame_index].stack.push(Value::Number(number));
            }
            Instruction::PushText(text) => frames[frame_index].stack.push(Value::Text(text)),
            Instruction::LoadLocal(slot) => {
                let Some(value) = frames[frame_index].locals.get(usize::from(slot)).cloned() else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                frames[frame_index].stack.push(value);
            }
            Instruction::StoreLocal(slot) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let Some(local) = frames[frame_index].locals.get_mut(usize::from(slot)) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid local slot {slot}"),
                    ));
                };
                *local = value;
            }
            Instruction::Pop => {
                if let Err(message) = pop(&mut frames[frame_index].stack) {
                    return Err(execution_error(module, &frames, message));
                }
            }
            Instruction::Negate => {
                let value = match pop_number(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index].stack.push(Value::number(-value));
            }
            Instruction::Not => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(!value.truthy())));
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
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let equal = values_equal(&left, &right);
                let result = if matches!(instruction, Instruction::NotEqual) {
                    !equal
                } else {
                    equal
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::And | Instruction::Or => {
                let right = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value.truthy(),
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let left = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value.truthy(),
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let result = if matches!(instruction, Instruction::And) {
                    left && right
                } else {
                    left || right
                };
                frames[frame_index]
                    .stack
                    .push(Value::number(f32::from(result)));
            }
            Instruction::JumpIfFalse(target) => {
                let condition = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if !condition.truthy() {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Jump(target) => {
                if let Err(message) = validate_jump(target, program.instructions.len()) {
                    return Err(execution_error(module, &frames, message));
                }
                frames[frame_index].instruction = target;
                continue;
            }
            Instruction::JumpIfArgumentSupplied { parameter, target } => {
                if frames[frame_index].arguments.len() > usize::from(parameter) {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::Call {
                procedure: target,
                argument_count,
            } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let count = usize::from(argument_count);
                let stack_length = frames[frame_index].stack.len();
                if count > stack_length {
                    return Err(execution_error(module, &frames, "bytecode stack underflow"));
                }
                let arguments = frames[frame_index].stack.split_off(stack_length - count);
                let Some(target_program) = module.procedure(target) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid call target {}", target.index()),
                    ));
                };
                frames.push(make_frame(target, target_program, &arguments));
                continue;
            }
            Instruction::CallCurrent { argument_count } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let arguments = if let Some(argument_count) = argument_count {
                    let count = usize::from(argument_count);
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    frames[frame_index].stack.split_off(stack_length - count)
                } else {
                    frames[frame_index].arguments.clone()
                };
                frames.push(make_frame(procedure, program, &arguments));
                continue;
            }
            Instruction::CallParent {
                procedure: target,
                argument_count,
            } => {
                if frames.len() >= limits.max_call_depth {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("maximum call depth {} exceeded", limits.max_call_depth),
                    ));
                }
                let Some(target) = target else {
                    return Err(execution_error(
                        module,
                        &frames,
                        "parent procedure call has no resolved target",
                    ));
                };
                let arguments = if let Some(argument_count) = argument_count {
                    let count = usize::from(argument_count);
                    let stack_length = frames[frame_index].stack.len();
                    if count > stack_length {
                        return Err(execution_error(module, &frames, "bytecode stack underflow"));
                    }
                    frames[frame_index].stack.split_off(stack_length - count)
                } else {
                    frames[frame_index].arguments.clone()
                };
                let Some(target_program) = module.procedure(target) else {
                    return Err(execution_error(
                        module,
                        &frames,
                        format!("invalid parent call target {}", target.index()),
                    ));
                };
                frames.push(make_frame(target, target_program, &arguments));
                continue;
            }
            Instruction::Return => {
                let result = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                frames.pop();
                let Some(caller) = frames.last_mut() else {
                    return Ok(result);
                };
                caller.stack.push(result);
                caller.instruction += 1;
                continue;
            }
        }
        frames[frame_index].instruction += 1;
    }
}

fn make_frame(procedure: ProcedureId, program: &Program, arguments: &[Value]) -> CallFrame {
    let mut locals = vec![Value::Null; program.local_count];
    let bound_count = arguments
        .len()
        .min(program.parameter_count)
        .min(locals.len());
    locals[..bound_count].clone_from_slice(&arguments[..bound_count]);
    CallFrame {
        procedure,
        instruction: 0,
        locals,
        stack: Vec::new(),
        arguments: arguments.to_vec(),
    }
}

fn execution_error(
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

fn trace(module: &Module, procedure: ProcedureId, instruction: usize) -> CallTrace {
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

fn execute_numeric_binary(instruction: &Instruction, left: f32, right: f32) -> f32 {
    match instruction {
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
    }
}

fn validate_jump(target: usize, instruction_count: usize) -> Result<(), String> {
    if target > instruction_count {
        return Err(format!("invalid jump target {target}"));
    }
    Ok(())
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

fn pop(stack: &mut Vec<Value>) -> Result<Value, String> {
    stack
        .pop()
        .ok_or_else(|| "bytecode stack underflow".to_owned())
}

fn pop_number(stack: &mut Vec<Value>) -> Result<f32, String> {
    let value = pop(stack)?;
    value
        .as_number()
        .ok_or_else(|| format!("numeric operation received {value}"))
}

#[cfg(test)]
mod tests {
    use dm_syntax::parse;

    use super::{
        ExecutionLimits, Instruction, Value, compile_module, compile_procedure, execute,
        execute_module, execute_module_with_limits, execute_with_limits,
    };

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

    #[test]
    fn executes_assignment_and_nested_if_else_blocks() {
        let source = "/proc/clamp(input)\n\tvar/result = input\n\tif(result < 0)\n\t\tresult = 0\n\telse\n\t\tif(result > 10)\n\t\t\tresult = 10\n\treturn result\n";

        assert_eq!(execute_source(source, -2.0), Value::number(0.0));
        assert_eq!(execute_source(source, 7.0), Value::number(7.0));
        assert_eq!(execute_source(source, 18.0), Value::number(10.0));
    }

    #[test]
    fn recognizes_when_both_conditional_branches_return() {
        let source = "/proc/sign(input)\n\tif(input < 0)\n\t\treturn -1\n\telse\n\t\treturn 1\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(-2.0)]),
            Ok(Value::number(-1.0))
        );
        assert_eq!(
            execute(&program, &[Value::number(2.0)]),
            Ok(Value::number(1.0))
        );
        assert_eq!(program.instructions.len(), program.source_spans.len());
    }

    #[test]
    fn calls_forward_declared_procedures_with_positional_arguments() {
        let source = "/proc/main(input)\n\treturn add(input, 3)\n/proc/add(left, right)\n\treturn left + right\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(8.0)]),
            Ok(Value::number(11.0))
        );
    }

    #[test]
    fn executes_recursive_calls_on_explicit_frames() {
        let source = "/proc/factorial(input)\n\tif(input <= 1)\n\t\treturn 1\n\treturn input * factorial(input - 1)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/factorial")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(5.0)]),
            Ok(Value::number(120.0))
        );
    }

    #[test]
    fn binds_missing_arguments_to_null_and_retains_extra_arguments() {
        let source = "/proc/identity(input)\n\treturn input\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/identity")
            .expect("entry procedure should exist");

        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::Null));
        assert_eq!(
            execute_module(&module, entry, &[Value::number(7.0), Value::number(99.0)]),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn bounds_recursion_and_reports_the_source_mapped_call_stack() {
        let source = "/proc/recurse(input)\n\treturn recurse(input)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/recurse")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[Value::number(1.0)],
            ExecutionLimits {
                max_call_depth: 3,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("unbounded recursion should reach the explicit limit");

        assert!(error.message.contains("maximum call depth 3"));
        assert_eq!(error.call_stack.len(), 3);
        assert!(error.source_span.is_some());
        assert!(
            error
                .call_stack
                .iter()
                .all(|trace| trace.procedure == "/proc/recurse" && trace.source_span.is_some())
        );
    }

    #[test]
    fn maps_callee_runtime_errors_and_preserves_caller_context() {
        let source = "/proc/main()\n\treturn broken()\n/proc/broken()\n\treturn \"text\" + 1\n";
        let syntax = parse(source).expect("source should parse");
        let expected_span = syntax.definitions[1].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");
        let error =
            execute_module(&module, entry, &[]).expect_err("numeric operation on text should fail");

        assert!(error.message.contains("numeric operation received"));
        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 2);
        assert_eq!(error.call_stack[0].procedure, "/proc/main");
        assert_eq!(error.call_stack[1].procedure, "/proc/broken");
        assert_eq!(error.call_stack[1].source_span, Some(expected_span));
    }

    #[test]
    fn current_call_uses_explicit_positional_arguments() {
        let source =
            "/proc/countdown(value)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
        let syntax = parse(source).expect("source should parse");
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/countdown")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module(&module, entry, &[Value::number(4.0)]),
            Ok(Value::number(4.0))
        );
    }

    #[test]
    fn argumentless_current_call_reuses_original_frame_arguments() {
        let source = "/proc/recurse(value, stop)\n\tstop = 1\n\treturn .()\n";
        let syntax = parse(source).expect("source should parse");
        let call_span = syntax.definitions[0].body[1].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/recurse")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[Value::number(7.0), Value::Null, Value::number(99.0)],
            ExecutionLimits {
                max_call_depth: 4,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("reused original arguments should keep recursing");

        assert!(error.message.contains("maximum call depth 4"));
        assert_eq!(error.source_span, Some(call_span));
        assert_eq!(error.call_stack.len(), 4);
        assert!(error.call_stack.iter().all(|trace| {
            trace.procedure == "/proc/recurse" && trace.source_span == Some(call_span)
        }));
    }

    #[test]
    fn unresolved_parent_call_reports_source_mapped_runtime_error() {
        let syntax = parse("/proc/child()\n\treturn ..()\n").expect("source should parse");
        let span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute(&program, &[]).expect_err("unresolved parent should fail at runtime");

        assert_eq!(
            error.message,
            "parent procedure call has no resolved target"
        );
        assert_eq!(error.source_span, Some(span));
    }

    #[test]
    fn while_supports_zero_and_multiple_iterations() {
        let source = "/proc/count(limit)\n\tvar/result = 0\n\twhile(result < limit)\n\t\tresult = result + 1\n\treturn result\n";

        assert_eq!(execute_source(source, 0.0), Value::number(0.0));
        assert_eq!(execute_source(source, 5.0), Value::number(5.0));
    }

    #[test]
    fn break_and_continue_work_inside_nested_conditionals() {
        let source = "/proc/filter(limit)\n\tvar/index = 0\n\tvar/total = 0\n\twhile(index < limit)\n\t\tindex = index + 1\n\t\tif(index == 2)\n\t\t\tcontinue\n\t\tif(index > 4)\n\t\t\tbreak\n\t\ttotal = total + index\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let while_span = syntax.definitions[0].body[2].span;
        let continue_span = syntax.definitions[0].body[5].span;
        let break_span = syntax.definitions[0].body[7].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(8.0))
        );
        assert_eq!(program.instructions.len(), program.source_spans.len());
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::JumpIfFalse(_))
                && *span == while_span
        ));
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::Jump(_))
                && *span == continue_span
        ));
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::Jump(_))
                && *span == break_span
        ));
    }

    #[test]
    fn nested_loops_patch_break_and_continue_to_the_innermost_loop() {
        let source = "/proc/nested(limit)\n\tvar/outer = 0\n\tvar/total = 0\n\twhile(outer < limit)\n\t\touter = outer + 1\n\t\tvar/inner = 0\n\t\twhile(inner < 5)\n\t\t\tinner = inner + 1\n\t\t\tif(inner == 2)\n\t\t\t\tcontinue\n\t\t\tif(inner == 4)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

        assert_eq!(execute_source(source, 3.0), Value::number(6.0));
    }

    #[test]
    fn rejects_break_and_continue_outside_loops() {
        for (statement, expected) in [
            ("break", "break outside a loop"),
            ("continue", "continue outside a loop"),
        ] {
            let source = format!("/proc/invalid()\n\t{statement}\n");
            let syntax = parse(&source).expect("source should parse");
            let error = compile_procedure(&syntax.definitions[0])
                .expect_err("loop control outside a loop should fail");

            assert_eq!(error.message, expected);
        }
    }

    #[test]
    fn instruction_budget_terminates_an_infinite_while_with_source_context() {
        let source = "/proc/spin()\n\twhile(1)\n\t\tcontinue\n";
        let syntax = parse(source).expect("source should parse");
        let while_span = syntax.definitions[0].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/spin")
            .expect("entry procedure should exist");
        let error = execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_steps: 7,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("infinite loop should exhaust its instruction budget");

        assert_eq!(error.message, "instruction budget of 7 exhausted");
        assert_eq!(error.source_span, Some(while_span));
        assert_eq!(error.call_stack.len(), 1);
        assert_eq!(error.call_stack[0].procedure, "/proc/spin");
        assert_eq!(error.call_stack[0].source_span, Some(while_span));
    }

    #[test]
    fn exact_standalone_instruction_budget_completes_the_final_return() {
        let source = "/proc/increment(value)\n\treturn value + 1\n";
        let syntax = parse(source).expect("source should parse");
        let return_span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let exact_steps = u64::try_from(program.instructions.len())
            .expect("test program instruction count should fit u64");

        assert_eq!(
            execute_with_limits(
                &program,
                &[Value::number(4.0)],
                ExecutionLimits {
                    max_steps: exact_steps,
                    ..ExecutionLimits::default()
                },
            ),
            Ok(Value::number(5.0))
        );
        let error = execute_with_limits(
            &program,
            &[Value::number(4.0)],
            ExecutionLimits {
                max_steps: exact_steps - 1,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("one fewer step should stop before Return");
        assert_eq!(error.source_span, Some(return_span));
        assert_eq!(error.call_stack[0].procedure, "<standalone>");
    }

    #[test]
    fn instruction_budget_is_shared_across_procedure_calls() {
        let source = "/proc/main()\n\treturn helper()\n/proc/helper()\n\treturn 7\n";
        let syntax = parse(source).expect("source should parse");
        let helper_span = syntax.definitions[1].body[0].span;
        let module = compile_module(&syntax.definitions).expect("module should compile");
        let entry = module
            .procedure_id("/proc/main")
            .expect("entry procedure should exist");

        assert_eq!(
            execute_module_with_limits(
                &module,
                entry,
                &[],
                ExecutionLimits {
                    max_steps: 4,
                    ..ExecutionLimits::default()
                },
            ),
            Ok(Value::number(7.0))
        );
        let error = execute_module_with_limits(
            &module,
            entry,
            &[],
            ExecutionLimits {
                max_steps: 2,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("caller and callee should consume one shared budget");

        assert_eq!(error.source_span, Some(helper_span));
        assert_eq!(error.call_stack.len(), 2);
        assert_eq!(error.call_stack[0].procedure, "/proc/main");
        assert_eq!(error.call_stack[1].procedure, "/proc/helper");
        assert_eq!(error.call_stack[1].source_span, Some(helper_span));
    }

    #[test]
    fn c_style_for_supports_scoped_initializer_and_postfix_increment() {
        let source = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\ttotal = total + i\n\treturn total\n";

        assert_eq!(execute_source(source, 0.0), Value::number(0.0));
        assert_eq!(execute_source(source, 5.0), Value::number(10.0));

        let escaped =
            parse("/proc/invalid()\n\tfor(var/i = 0; i < 1; i++)\n\t\tcontinue\n\treturn i\n")
                .expect("source should parse");
        let error = compile_procedure(&escaped.definitions[0])
            .expect_err("for initializer should be scoped to its loop");
        assert_eq!(error.message, "unknown local \"i\"");
    }

    #[test]
    fn c_style_for_supports_prefix_decrement_and_optional_clauses() {
        let decrement = "/proc/sum(limit)\n\tvar/total = 0\n\tfor(var/i = limit; i > 0; --i)\n\t\ttotal = total + i\n\treturn total\n";
        assert_eq!(execute_source(decrement, 3.0), Value::number(6.0));

        let optional = "/proc/once()\n\tfor(;;)\n\t\tbreak\n\treturn 9\n";
        let syntax = parse(optional).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
    }

    #[test]
    fn for_continue_runs_increment_and_break_exits_the_loop() {
        let source = "/proc/filter(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tif(i == 1)\n\t\t\tcontinue\n\t\tif(i == 4)\n\t\t\tbreak\n\t\ttotal = total + i\n\treturn total\n";
        let syntax = parse(source).expect("source should parse");
        let for_span = syntax.definitions[0].body[1].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(5.0))
        );
        assert!(program.instructions.iter().zip(&program.source_spans).any(
            |(instruction, span)| matches!(instruction, Instruction::StoreLocal(_))
                && *span == for_span
        ));
    }

    #[test]
    fn nested_for_loops_patch_control_to_the_innermost_loop() {
        let source = "/proc/nested(limit)\n\tvar/total = 0\n\tfor(var/i = 0; i < limit; i++)\n\t\tfor(var/j = 0; j < 4; j++)\n\t\t\tif(j == 1)\n\t\t\t\tcontinue\n\t\t\tif(j == 3)\n\t\t\t\tbreak\n\t\t\ttotal = total + 1\n\treturn total\n";

        assert_eq!(execute_source(source, 3.0), Value::number(6.0));
    }

    #[test]
    fn infinite_for_obeys_step_budget_and_for_in_has_precise_diagnostic() {
        let source = "/proc/spin()\n\tfor(;;)\n\t\tcontinue\n";
        let syntax = parse(source).expect("source should parse");
        let for_span = syntax.definitions[0].body[0].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute_with_limits(
            &program,
            &[],
            ExecutionLimits {
                max_steps: 7,
                ..ExecutionLimits::default()
            },
        )
        .expect_err("infinite for should exhaust its step budget");
        assert_eq!(error.message, "instruction budget of 7 exhausted");
        assert_eq!(error.source_span, Some(for_span));

        let list_iteration =
            parse("/proc/list_loop(items)\n\tfor(var/item in items)\n\t\tcontinue\n")
                .expect("source should parse");
        let error = compile_procedure(&list_iteration.definitions[0])
            .expect_err("for-in should remain a distinct unsupported form");
        assert_eq!(error.message, "for-in list iteration is not implemented");
    }

    #[test]
    fn parameter_literal_default_applies_only_when_argument_is_omitted() {
        let source = "/proc/defaulted(value = 5)\n\treturn value\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(5.0)));
        assert_eq!(execute(&program, &[Value::Null]), Ok(Value::Null));
        assert_eq!(
            execute(&program, &[Value::number(9.0)]),
            Ok(Value::number(9.0))
        );

        let text = parse("/proc/text_default(value = \"fallback\")\n\treturn value\n")
            .expect("source should parse");
        let program = compile_procedure(&text.definitions[0]).expect("procedure should compile");
        assert_eq!(
            execute(&program, &[]),
            Ok(Value::Text("fallback".to_owned()))
        );
    }

    #[test]
    fn multiple_parameter_defaults_evaluate_in_declaration_order() {
        let source = "/proc/combine(first = 1 + 1, second = 3, third = 4)\n\treturn first + second + third\n";
        let syntax = parse(source).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");

        assert_eq!(execute(&program, &[]), Ok(Value::number(9.0)));
        assert_eq!(
            execute(&program, &[Value::number(10.0)]),
            Ok(Value::number(17.0))
        );
        let error = execute(
            &program,
            &[Value::number(10.0), Value::Null, Value::number(1.0)],
        )
        .expect_err("explicit null should suppress the second parameter default");
        assert_eq!(error.message, "numeric operation received null");
        assert_eq!(error.source_span, Some(syntax.definitions[0].body[0].span));
    }

    #[test]
    fn defaults_interact_with_explicit_and_argument_reusing_current_calls() {
        let countdown = "/proc/countdown(value = 3)\n\tif(value <= 0)\n\t\treturn 0\n\treturn 1 + .(value - 1)\n";
        let syntax = parse(countdown).expect("source should parse");
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        assert_eq!(execute(&program, &[]), Ok(Value::number(3.0)));

        let reapply = "/proc/reapply(value = 1)\n\tvalue = 0\n\treturn .()\n";
        let syntax = parse(reapply).expect("source should parse");
        let call_span = syntax.definitions[0].body[1].span;
        let program = compile_procedure(&syntax.definitions[0]).expect("procedure should compile");
        let error = execute_with_limits(
            &program,
            &[],
            ExecutionLimits {
                max_call_depth: 3,
                ..ExecutionLimits::default()
            },
        )
        .expect_err(".() should reuse omission and reapply the default in each frame");
        assert!(error.message.contains("maximum call depth 3"));
        assert_eq!(error.source_span, Some(call_span));
        assert_eq!(error.call_stack.len(), 3);
    }

    #[test]
    fn unsupported_parameter_default_forms_have_precise_diagnostics() {
        let reference = parse("/proc/reference(first = 2, second = first)\n\treturn second\n")
            .expect("source should parse");
        let error = compile_procedure(&reference.definitions[0])
            .expect_err("cross-parameter default needs conformance confirmation");
        assert_eq!(
            error.message,
            "parameter default reference \"first\" requires BYOND conformance confirmation"
        );

        let call = parse("/proc/call_default(value = helper())\n\treturn value\n")
            .expect("source should parse");
        let error = compile_procedure(&call.definitions[0])
            .expect_err("call defaults should remain outside the supported subset");
        assert_eq!(
            error.message,
            "procedure calls in parameter defaults are not supported"
        );
    }
}
