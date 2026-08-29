//! Statement lowering and the lexical scope table for the DM compiler.
//!
//! Owns `LocalTable` (procedure locals, parameters, and static
//! type/category annotations) plus the statement-level recursion
//! coordinator and lowering of each structured control-flow form (blocks,
//! assignment, `while`/`do`/`for` loops, `try`, `if`, `switch`) onto the
//! portable `Instruction` stream. Expressions are parsed and emitted by the
//! sibling `compile_expr` module.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind, SourceLine};
use dm_value::{FieldName, TypePath};

use crate::CompileError;
use crate::bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, Instruction, ProcedureId,
    VerbParameterType,
};

use crate::compile::compile_error;
use crate::compile_expr::{
    Expression, ExpressionParser, compile_expression, emit_expression, to_local_index,
};

pub(crate) struct LocalTable<'fields> {
    names: HashMap<String, u16>,
    types: HashMap<String, TypePath>,
    src_fields: &'fields BTreeMap<String, FieldName>,
    global_fields: &'fields BTreeMap<String, FieldName>,
    global_types: &'fields BTreeMap<String, TypePath>,
    pub(crate) slot_count: usize,
}

impl Default for LocalTable<'static> {
    fn default() -> Self {
        static EMPTY: std::sync::LazyLock<BTreeMap<String, FieldName>> =
            std::sync::LazyLock::new(BTreeMap::new);
        static EMPTY_TYPES: std::sync::LazyLock<BTreeMap<String, TypePath>> =
            std::sync::LazyLock::new(BTreeMap::new);
        Self::with_fields(&EMPTY, &EMPTY, &EMPTY_TYPES)
    }
}

impl<'fields> LocalTable<'fields> {
    pub(crate) fn with_fields(
        src_fields: &'fields BTreeMap<String, FieldName>,
        global_fields: &'fields BTreeMap<String, FieldName>,
        global_types: &'fields BTreeMap<String, TypePath>,
    ) -> Self {
        Self {
            names: HashMap::new(),
            types: HashMap::new(),
            src_fields,
            global_fields,
            global_types,
            slot_count: 0,
        }
    }
    pub(crate) fn insert_parameter(&mut self, name: String, slot: u16) {
        self.names.insert(name, slot);
        self.slot_count = self.slot_count.max(usize::from(slot) + 1);
    }

    pub(crate) fn reserve_parameter_slots(&mut self, count: usize) -> Result<(), CompileError> {
        // Keep unnamed varargs positions available to the frame binder and
        // ensure subsequent locals are allocated after every parameter.
        let count = usize::from(to_local_index(count)?);
        self.slot_count = self.slot_count.max(count);
        Ok(())
    }

    pub(crate) fn declare(&mut self, name: String) -> Result<u16, CompileError> {
        if self.names.contains_key(&name) {
            return Err(compile_error(format!("local {name:?} is already declared")));
        }
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        self.names.insert(name, slot);
        Ok(slot)
    }

    fn declare_hidden(&mut self) -> Result<u16, CompileError> {
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        Ok(slot)
    }

    pub(crate) fn get(&self, name: &str) -> Option<u16> {
        self.names.get(name).copied()
    }

    pub(crate) fn set_type(&mut self, name: String, type_path: TypePath) {
        self.types.insert(name, type_path);
    }

    fn local_type(&self, name: &str) -> Option<&TypePath> {
        self.types.get(name)
    }

    pub(crate) fn src_field(&self, name: &str) -> Option<&FieldName> {
        self.src_fields.get(name)
    }

    pub(crate) fn global_field(&self, name: &str) -> Option<&FieldName> {
        self.global_fields.get(name)
    }

    fn global_type(&self, name: &str) -> Option<&TypePath> {
        self.global_types.get(name)
    }

    pub(crate) fn receiver_static(
        &self,
        receiver: &Expression,
        name: &FieldName,
    ) -> Option<&FieldName> {
        let receiver = match receiver {
            Expression::Src => "src",
            Expression::Local(receiver) => receiver.as_str(),
            Expression::GlobalField(receiver) => receiver.as_str(),
            _ => return None,
        };
        self.global_fields
            .get(&format!("{receiver}.{}", name.as_str()))
    }

    fn remove(&mut self, name: &str) {
        self.names.remove(name);
        self.types.remove(name);
    }
}

pub(crate) fn compile_parameter_defaults(
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

pub(crate) struct LoopContext {
    continue_target: Option<usize>,
    continue_jumps: Vec<usize>,
    break_jumps: Vec<usize>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn compile_block(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    // DM locals are lexical to their block. Macro helpers routinely expand
    // repeated `do { var/_L = ... } while(0)` scopes; retaining those names
    // after the child block makes unrelated invocations collide.
    let saved_names = locals.names.clone();
    let result = compile_block_inner(
        lines,
        line_index,
        block_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    );
    locals.names = saved_names;
    result
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_block_inner(
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
            // `switch` is a statement in DM, not a procedure call.  Each
            // indented `if` arm is a case list (with comma-separated values
            // and `low to high` ranges), while `else` is the default arm.
            // Keep this distinct from ordinary `if`: a switch selector is
            // evaluated exactly once and every case compares against it.
            TokenKind::Identifier(keyword) if keyword == "switch" => {
                let (next_line, statement_falls_through) = compile_switch(
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
            TokenKind::Identifier(keyword) if keyword == "do" => {
                let next_line = compile_do_while(
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
            TokenKind::Identifier(keyword) if keyword == "try" => {
                let (next_line, statement_falls_through) = compile_try(
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
            TokenKind::Identifier(keyword) if keyword == "catch" => {
                return Err(compile_error("catch without a matching try"));
            }
            TokenKind::Identifier(keyword) if keyword == "else" => {
                return Err(compile_error("else without a matching if"));
            }
            TokenKind::Identifier(keyword) if keyword == "break" => {
                let depth = match line.tokens.as_slice() {
                    [_] => 1,
                    [_, SpannedToken { kind: TokenKind::Number(depth), .. }] => depth
                        .parse::<usize>()
                        .map_err(|_| compile_error("invalid labeled break depth"))?,
                    _ => {
                        return Err(compile_error("break does not accept an expression"));
                    }
                };
                if loops.is_empty() {
                    return Err(compile_error("break outside a loop"));
                }
                if depth == 0 || depth > loops.len() {
                    return Err(compile_error("break does not accept an expression"));
                }
                let target_loop = loops.len() - depth;
                let Some(loop_context) = loops.get_mut(target_loop) else {
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
                if loops.is_empty() {
                    return Err(compile_error("continue outside a loop"));
                }
                let depth = match line.tokens.as_slice() {
                    [_] => 1,
                    [_, SpannedToken { kind: TokenKind::Number(depth), .. }] => depth
                        .parse::<usize>()
                        .map_err(|_| compile_error("invalid labeled continue depth"))?,
                    _ => return Err(compile_error("continue does not accept an expression")),
                };
                if depth == 0 || depth > loops.len() {
                    return Err(compile_error("continue does not accept an expression"));
                }
                let target_loop = loops.len() - depth;
                let loop_context = &mut loops[target_loop];
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
                    // A bare DM `return` returns the procedure's current
                    // special result value (`.`), just like falling through.
                    // This is relied upon by cache-hit patterns that assign
                    // `.` and then use `if(.) return`.
                    instructions.push(Instruction::LoadResult);
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
            TokenKind::Identifier(keyword) if keyword == "CRASH" => {
                let first_instruction = instructions.len();
                compile_crash_statement(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            TokenKind::Identifier(keyword) if keyword == "throw" => {
                if line.tokens.len() == 1 {
                    return Err(compile_error("throw requires an expression"));
                }
                let first_instruction = instructions.len();
                compile_expression(&line.tokens[1..], locals, instructions, procedures)?;
                instructions.push(Instruction::Throw);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
                falls_through = false;
            }
            // `waitfor` is procedure metadata captured on `Program`; it has no
            // executable assignment at the declaration site.
            TokenKind::Identifier(keyword)
                if keyword == "set" && is_waitfor_directive(&line.tokens) => {}
            TokenKind::Identifier(keyword)
                if keyword == "set"
                    && matches!(line.tokens.get(1).map(|token| &token.kind), Some(TokenKind::Identifier(_)))
                    && matches!(line.tokens.get(2).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "=") =>
            {
                // Verb/procedure `set` directives (`name`, `category`, `desc`,
                // `hidden`, and friends) are declaration metadata and do not
                // execute when the procedure is called.
            }
            TokenKind::Identifier(keyword) if keyword == "var" => {
                let first_instruction = instructions.len();
                compile_local_declarations(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(keyword) | TokenKind::Operator(keyword)
                if keyword != "spawn" && top_level_assignment(&line.tokens).is_some() =>
            {
                let first_instruction = instructions.len();
                compile_assignment_statement(&line.tokens, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) if top_level_output(&line.tokens).is_some() => {
                let first_instruction = instructions.len();
                let output = top_level_output(&line.tokens).expect("output index was checked");
                compile_expression(&line.tokens[..output], locals, instructions, procedures)?;
                compile_expression(&line.tokens[output + 1..], locals, instructions, procedures)?;
                instructions.push(Instruction::Output);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_) if top_level_input(&line.tokens).is_some() => {
                let first_instruction = instructions.len();
                let input = top_level_input(&line.tokens).expect("input index was checked");
                let target = ExpressionParser::new(&line.tokens[input + 1..]).parse()?;
                match target {
                    Expression::Local(name) => {
                        if let Some(slot) = locals.get(&name) {
                            compile_expression(
                                &line.tokens[..input],
                                locals,
                                instructions,
                                procedures,
                            )?;
                            instructions.push(Instruction::Input);
                            instructions.push(Instruction::StoreLocal(slot));
                        } else if let Some(field) = locals.src_field(&name) {
                            instructions.push(Instruction::LoadSrc);
                            compile_expression(
                                &line.tokens[..input],
                                locals,
                                instructions,
                                procedures,
                            )?;
                            instructions.push(Instruction::Input);
                            instructions.push(Instruction::StoreField(field.clone()));
                        } else if let Some(global) = locals.global_field(&name) {
                            compile_expression(
                                &line.tokens[..input],
                                locals,
                                instructions,
                                procedures,
                            )?;
                            instructions.push(Instruction::Input);
                            instructions.push(Instruction::StoreGlobal(global.clone()));
                        } else {
                            return Err(compile_error(format!(
                                "savefile input target {name:?} is not writable"
                            )));
                        }
                    }
                    Expression::Field { receiver, name } => {
                        emit_expression(&receiver, locals, instructions, procedures)?;
                        compile_expression(
                            &line.tokens[..input],
                            locals,
                            instructions,
                            procedures,
                        )?;
                        instructions.push(Instruction::Input);
                        instructions.push(Instruction::StoreField(name));
                    }
                    Expression::Index { list, index } => {
                        emit_expression(&list, locals, instructions, procedures)?;
                        emit_expression(&index, locals, instructions, procedures)?;
                        compile_expression(
                            &line.tokens[..input],
                            locals,
                            instructions,
                            procedures,
                        )?;
                        instructions.push(Instruction::Input);
                        instructions.push(Instruction::SetListIndex);
                    }
                    _ => return Err(compile_error("savefile input target is not writable")),
                }
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Postfix/prefix increments are valid standalone statements as
            // well as for-loop clauses.  In particular, bare datum fields
            // such as `areasize++` resolve through `src` rather than a local
            // binding, so they must take the same lowering path as compound
            // assignments.
            TokenKind::Identifier(_) | TokenKind::Operator(_)
                if local_increment(&line.tokens).is_some() =>
            {
                let first_instruction = instructions.len();
                // `compile_for_clause` owns the shared prefix/postfix
                // increment lowering (including bare `src` fields).  The
                // standalone statement form has identical semantics.
                compile_for_clause(&line.tokens, false, locals, instructions, procedures)?;
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Operator(operator) if matches!(operator.as_str(), "++" | "--") => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Operator(operator) if operator == "." => {
                let first_instruction = instructions.len();
                if top_level_assignment(&line.tokens).is_some_and(|(index, _)| index == 1) {
                    compile_result_assignment(&line.tokens, locals, instructions, procedures)?;
                } else if top_level_assignment(&line.tokens).is_some() {
                    // The special result is also a regular expression value,
                    // so indexed writes such as `.[key] = value` use the same
                    // list-assignment lowering as any other expression.
                    compile_assignment_statement(
                        &line.tokens,
                        locals,
                        instructions,
                        procedures,
                    )?;
                } else {
                    compile_expression(&line.tokens, locals, instructions, procedures)?;
                    instructions.push(Instruction::Pop);
                }
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(keyword) if keyword == "call" => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(keyword) if keyword == "spawn" => {
                let first_instruction = instructions.len();
                let after_keyword = &line.tokens[1..];
                let rest = if matches!(
                    after_keyword.first().map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    let mut spawn = ExpressionParser::new(after_keyword);
                    let arguments = spawn.parse_call_arguments()?;
                    if arguments.len() > 1 {
                        return Err(compile_error(
                            "spawn accepts at most one delay argument before the spawned expression",
                        ));
                    }
                    if let Some(delay) = arguments.first() {
                        emit_expression(delay, locals, instructions, procedures)?;
                    } else {
                        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(0.0)));
                    }
                    &line.tokens[1 + spawn.index..]
                } else {
                    // BYOND's `spawn statement` and `spawn { ... }` forms are
                    // exactly `spawn(0)` with the parentheses omitted.
                    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(0.0)));
                    after_keyword
                };
                let spawn_instruction = instructions.len();
                instructions.push(Instruction::Spawn { entry: usize::MAX });
                let skip_spawned_body = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let spawned_entry = instructions.len();
                if rest.is_empty() {
                    source_spans.extend(std::iter::repeat_n(
                        line.span,
                        instructions.len() - first_instruction,
                    ));
                    let Some(first_body_line) = lines.get(line_index + 1) else {
                        return Err(compile_error("spawn requires a spawned statement"));
                    };
                    let body_indentation = indentation(first_body_line);
                    if body_indentation <= block_indentation {
                        return Err(compile_error(
                            "spawn requires an indented spawned statement",
                        ));
                    }
                    let (next_line, _) = compile_block(
                        lines,
                        line_index + 1,
                        body_indentation,
                        locals,
                        instructions,
                        source_spans,
                        procedures,
                        loops,
                    )?;
                    line_index = next_line;
                } else {
                    compile_expression(rest, locals, instructions, procedures)?;
                    instructions.push(Instruction::Pop);
                }
                instructions.push(Instruction::PushNull);
                instructions.push(Instruction::Return);
                let after_spawned_body = instructions.len();
                instructions[spawn_instruction] = Instruction::Spawn {
                    entry: spawned_entry,
                };
                instructions[skip_spawned_body] = Instruction::Jump(after_spawned_body);
                if rest.is_empty() {
                    source_spans.extend(std::iter::repeat_n(line.span, 2));
                } else {
                    source_spans.extend(std::iter::repeat_n(
                        line.span,
                        instructions.len() - first_instruction,
                    ));
                }
                if rest.is_empty() {
                    continue;
                }
            }
            // `new /type(...)` is also commonly written as a pure
            // side-effect statement, especially for controller singletons.
            TokenKind::Identifier(keyword) if keyword == "new" => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // A parent call is also a valid side-effect-only statement.  It
            // starts with the `..` operator rather than an identifier, so it
            // cannot share the ordinary static-call statement arm below.
            TokenKind::Operator(operator)
                if operator == ".."
                    && matches!(
                        line.tokens.get(1).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) =>
            {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Parenthesized expressions are valid as discarded-result
            // statements too.  Macro expansions commonly wrap an assignment
            // or a side-effecting call in parentheses, which means these
            // lines begin with punctuation rather than an identifier.
            TokenKind::Punctuation('(') => {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            // Calls may be used purely for their side effects.  `call(...)`
            // has its own syntax above, but ordinary static calls (including
            // datum helper calls such as `RegisterSignals(...)`) and dotted
            // datum calls such as `atom_storage.set_holdable(...)` both begin
            // with an identifier.  The latter have the opening parenthesis
            // after the receiver and selector rather than immediately after
            // the first identifier, so recognize any call-shaped expression
            // on the source line and lower its discarded result uniformly.
            TokenKind::Identifier(_)
                if line
                    .tokens
                    .iter()
                    .any(|token| token.kind == TokenKind::Punctuation('(')) =>
            {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
                source_spans.extend(std::iter::repeat_n(
                    line.span,
                    instructions.len() - first_instruction,
                ));
            }
            TokenKind::Identifier(_)
                if line.tokens.iter().any(|token| {
                    matches!(&token.kind, TokenKind::Operator(operator) if operator == "++" || operator == "--")
                }) =>
            {
                let first_instruction = instructions.len();
                compile_expression(&line.tokens, locals, instructions, procedures)?;
                instructions.push(Instruction::Pop);
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

fn is_waitfor_directive(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens,
        [
            SpannedToken {
                kind: TokenKind::Identifier(set),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(name),
                ..
            },
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(value),
                ..
            }
        ] if set == "set"
            && name == "waitfor"
            && operator == "="
            && matches!(value.as_str(), "TRUE" | "FALSE")
    ) || matches!(
        tokens,
        [
            SpannedToken {
                kind: TokenKind::Identifier(set),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(name),
                ..
            },
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Number(value),
                ..
            }
        ] if set == "set"
            && name == "waitfor"
            && operator == "="
            && matches!(value.as_str(), "0" | "1")
    )
}

pub(crate) fn procedure_wait_for(definition: &Definition) -> bool {
    !definition.body.iter().any(|line| {
        matches!(
            line.tokens.as_slice(),
            [
                SpannedToken { kind: TokenKind::Identifier(set), .. },
                SpannedToken { kind: TokenKind::Identifier(name), .. },
                SpannedToken { kind: TokenKind::Operator(operator), .. },
                SpannedToken { kind: TokenKind::Identifier(value), .. }
            ] if set == "set" && name == "waitfor" && operator == "=" && value == "FALSE"
        ) || matches!(
            line.tokens.as_slice(),
            [
                SpannedToken { kind: TokenKind::Identifier(set), .. },
                SpannedToken { kind: TokenKind::Identifier(name), .. },
                SpannedToken { kind: TokenKind::Operator(operator), .. },
                SpannedToken { kind: TokenKind::Number(value), .. }
            ] if set == "set" && name == "waitfor" && operator == "=" && value == "0"
        )
    })
}

pub(crate) fn procedure_verb_name(definition: &Definition) -> Option<String> {
    (definition.kind == DefinitionKind::Verb)
        .then(|| {
            definition
                .body
                .iter()
                .find_map(|line| match line.tokens.as_slice() {
                    [
                        SpannedToken {
                            kind: TokenKind::Identifier(set),
                            ..
                        },
                        SpannedToken {
                            kind: TokenKind::Identifier(name),
                            ..
                        },
                        SpannedToken {
                            kind: TokenKind::Operator(operator),
                            ..
                        },
                        SpannedToken {
                            kind: TokenKind::String(value),
                            ..
                        },
                    ] if set == "set" && name == "name" && operator == "=" => Some(value.clone()),
                    _ => None,
                })
        })
        .flatten()
}

fn compile_crash_statement(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let Some((first, rest)) = tokens.split_first() else {
        return Err(compile_error("CRASH requires a message expression"));
    };
    if !matches!(&first.kind, TokenKind::Identifier(keyword) if keyword == "CRASH")
        || rest.len() < 2
        || !matches!(
            rest.first().map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        )
        || !matches!(
            rest.last().map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        )
    {
        return Err(compile_error(
            "CRASH requires one parenthesized message expression",
        ));
    }
    let expression = &rest[1..rest.len() - 1];
    if expression.is_empty() {
        instructions.push(Instruction::PushText(Arc::from("CRASH")));
    } else {
        compile_expression(expression, locals, instructions, procedures)?;
    }
    instructions.push(Instruction::Crash);
    Ok(())
}

fn top_level_assignment(tokens: &[SpannedToken]) -> Option<(usize, &str)> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator)
                if matches!(
                    operator.as_str(),
                    "=" | ":="
                        | "+="
                        | "-="
                        | "*="
                        | "/="
                        | "%="
                        | "%%="
                        | "&="
                        | "|="
                        | "^="
                        | "<<="
                        | ">>="
                        | "&&="
                        | "||="
                ) && depth == 0 =>
            {
                return Some((index, operator));
            }
            _ => {}
        }
    }
    None
}

fn top_level_output(tokens: &[SpannedToken]) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator) if operator == "<<" && depth == 0 => {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn top_level_input(tokens: &[SpannedToken]) -> Option<usize> {
    let mut depth = 0_usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator) if operator == ">>" && depth == 0 => {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::too_many_lines)]
fn compile_assignment_statement(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let (assignment, operator) = top_level_assignment(tokens)
        .ok_or_else(|| compile_error("assignment statement requires '='"))?;
    let operator = if operator == ":=" { "=" } else { operator };
    if matches!(operator, "||=" | "&&=") {
        compile_expression(tokens, locals, instructions, procedures)?;
        instructions.push(Instruction::Pop);
        return Ok(());
    }
    if assignment == 0 || assignment + 1 == tokens.len() {
        return Err(compile_error("assignment requires a target and value"));
    }
    let target = ExpressionParser::new(&tokens[..assignment]).parse()?;
    match target {
        Expression::Local(name) => {
            let local = locals.get(&name);
            let field = locals.src_field(&name).cloned();
            let global = locals.global_field(&name).cloned();
            let Some(slot) = local else {
                if field.is_none() && global.is_none() {
                    return Err(compile_error(format!("unknown local {name:?}")));
                }
                if let Some(global) = global {
                    if operator != "=" {
                        instructions.push(Instruction::LoadGlobal(global.clone()));
                    }
                    let mut value = ExpressionParser::new(&tokens[assignment + 1..]).parse()?;
                    if let Expression::New { type_path, .. } = &mut value
                        && type_path.is_none()
                        && let Some(inferred) = locals.global_type(&name)
                    {
                        *type_path = Some(Box::new(Expression::TypePath(inferred.clone())));
                    }
                    infer_contextual_locate(&mut value, locals.global_type(&name));
                    emit_expression(&value, locals, instructions, procedures)?;
                    if operator != "=" {
                        instructions.push(compound_instruction(operator)?);
                    }
                    instructions.push(Instruction::StoreGlobal(global));
                    return Ok(());
                }
                if operator == "=" {
                    instructions.push(Instruction::LoadSrc);
                } else {
                    instructions.push(Instruction::LoadSrc);
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(
                        field.clone().expect("field was checked"),
                    ));
                }
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreField(field.expect("field was checked")));
                return Ok(());
            };
            if operator != "=" {
                instructions.push(Instruction::LoadLocal(slot));
            }
            let mut value = ExpressionParser::new(&tokens[assignment + 1..]).parse()?;
            if let Expression::New { type_path, .. } = &mut value
                && type_path.is_none()
                && let Some(inferred) = locals.local_type(&name)
            {
                *type_path = Some(Box::new(Expression::TypePath(inferred.clone())));
            }
            infer_contextual_locate(&mut value, locals.local_type(&name));
            emit_expression(&value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreLocal(slot));
        }
        Expression::Index { list, index } => {
            if operator == "=" {
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                if let Expression::Field { receiver, name } = list.as_ref()
                    && name.as_str() == "vars"
                {
                    emit_expression(receiver, locals, instructions, procedures)?;
                    emit_expression(&index, locals, instructions, procedures)?;
                    instructions.push(Instruction::PrepareRhsFirstIndexAssignment);
                    instructions.push(Instruction::StoreDynamicField);
                    return Ok(());
                }
                emit_expression(&list, locals, instructions, procedures)?;
                emit_expression(&index, locals, instructions, procedures)?;
                instructions.push(Instruction::PrepareRhsFirstIndexAssignment);
                instructions.push(Instruction::SetListIndex);
            } else {
                emit_expression(&list, locals, instructions, procedures)?;
                emit_expression(&index, locals, instructions, procedures)?;
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(&list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(&index, locals, instructions, procedures)?;
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[end_jump] = Instruction::Jump(end);
        }
        Expression::Field { receiver, name } => {
            if let Some(storage) = locals
                .receiver_static(receiver.as_ref(), &name)
                .or_else(|| {
                    matches!(receiver.as_ref(), Expression::Src)
                        .then(|| locals.global_field(name.as_str()))
                        .flatten()
                })
            {
                if operator != "=" {
                    instructions.push(Instruction::LoadGlobal(storage.clone()));
                }
                compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreGlobal(storage.clone()));
                return Ok(());
            }
            emit_expression(&receiver, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreField(name));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(&receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreField(name));
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[end_jump] = Instruction::Jump(end);
        }
        Expression::GlobalField(name) => {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreGlobal(name));
        }
        Expression::Src => {
            if operator != "=" {
                return Err(compile_error("src only supports direct assignment"));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            instructions.push(Instruction::StoreSrc);
        }
        Expression::Usr => {
            if operator != "=" {
                return Err(compile_error("usr only supports direct assignment"));
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            instructions.push(Instruction::StoreUsr);
        }
        Expression::Result => {
            if operator != "=" {
                instructions.push(Instruction::LoadResult);
            }
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreResult);
        }
        Expression::Unary {
            operator: unary_operator,
            operand,
        } if unary_operator == "*" => {
            if let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::LoadLocalRaw(slot));
            } else {
                emit_expression(&operand, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
        }
        _ => return Err(compile_error("assignment target is not writable")),
    }
    Ok(())
}

pub(crate) fn compound_instruction(operator: &str) -> Result<Instruction, CompileError> {
    let operator = match operator {
        "+=" => CompoundAssignmentOperator::Add,
        "-=" => CompoundAssignmentOperator::Subtract,
        "*=" => CompoundAssignmentOperator::Multiply,
        "/=" => CompoundAssignmentOperator::Divide,
        "%=" => CompoundAssignmentOperator::Remainder,
        "%%=" => CompoundAssignmentOperator::FractionalRemainder,
        "&=" => CompoundAssignmentOperator::BitAnd,
        "|=" => CompoundAssignmentOperator::BitOr,
        "^=" => CompoundAssignmentOperator::BitXor,
        "<<=" => CompoundAssignmentOperator::ShiftLeft,
        ">>=" => CompoundAssignmentOperator::ShiftRight,
        _ => {
            return Err(compile_error(format!(
                "unsupported compound operator {operator}"
            )));
        }
    };
    Ok(Instruction::CompoundAssignment(operator))
}

pub(crate) fn compound_list_index_operator(
    operator: &str,
) -> Result<CompoundListIndexOperator, CompileError> {
    Ok(match operator {
        "+=" => CompoundListIndexOperator::Add,
        "-=" => CompoundListIndexOperator::Subtract,
        "*=" => CompoundListIndexOperator::Multiply,
        "/=" => CompoundListIndexOperator::Divide,
        "%=" => CompoundListIndexOperator::Remainder,
        "%%=" => CompoundListIndexOperator::FractionalRemainder,
        "&=" => CompoundListIndexOperator::BitAnd,
        "|=" => CompoundListIndexOperator::BitOr,
        "^=" => CompoundListIndexOperator::BitXor,
        "<<=" => CompoundListIndexOperator::ShiftLeft,
        ">>=" => CompoundListIndexOperator::ShiftRight,
        _ => {
            return Err(compile_error(format!(
                "unsupported compound operator {operator:?}"
            )));
        }
    })
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

    loops.push(LoopContext {
        continue_target: Some(condition_target),
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
        .map(|(_, falls_through)| (line_index + 1, falls_through))
    } else {
        let child_index = line_index + 1;
        let Some(child) = lines.get(child_index) else {
            // BYOND permits an empty while whose condition performs all
            // useful work, including postfix/prefix mutation idioms.
            return finish_while_body(
                line_index + 1,
                condition_target,
                false_jump,
                line,
                loops,
                instructions,
                source_spans,
            );
        };
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return finish_while_body(
                line_index + 1,
                condition_target,
                false_jump,
                line,
                loops,
                instructions,
                source_spans,
            );
        }
        compile_block(
            lines,
            child_index,
            child_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
    };
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

#[allow(clippy::too_many_arguments)]
fn finish_while_body(
    after_body: usize,
    condition_target: usize,
    false_jump: usize,
    line: &SourceLine,
    loops: &mut Vec<LoopContext>,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
) -> Result<usize, CompileError> {
    let loop_context = loops.pop().expect("the active while context was pushed");
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, condition_target)?;
    }
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

/// Compiles BYOND's post-test `do`/`while` loop form.  The trailing `while`
/// belongs to the `do` statement, at its original indentation, rather than
/// beginning a second statement after the body.
#[allow(clippy::too_many_arguments)]
fn compile_do_while(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<usize, CompileError> {
    let do_line = &lines[line_index];
    if do_line.tokens.len() != 1 {
        return Err(compile_error("do statement does not accept a condition"));
    }
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("do statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("do statement requires an indented body"));
    }

    let body_target = instructions.len();
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
    let loop_context = loops.pop().expect("the active do context was pushed");
    let (while_index, _) = body?;
    let while_line = lines
        .get(while_index)
        .ok_or_else(|| compile_error("do statement requires a trailing while condition"))?;
    if indentation(while_line) != block_indentation
        || !matches!(
            while_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "while"
        )
    {
        return Err(compile_error(
            "do statement requires a trailing while condition",
        ));
    }

    let condition_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, condition_target)?;
    }
    let condition = condition_tokens(&while_line.tokens, "while")?;
    let condition_start = instructions.len();
    compile_expression(condition, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        while_line.span,
        instructions.len() - condition_start,
    ));
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        while_line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::Jump(body_target),
        while_line.span,
    );
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    Ok(while_index + 1)
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
    if let Some((local_name, type_path)) = for_type_parts(&line.tokens)? {
        return compile_for_in(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            &local_name,
            true,
            &[],
            Some(&type_path),
            Some(&type_path),
            Some(&type_path),
        );
    }
    if let Some((first, second, iterable, declared)) = for_assoc_parts(&line.tokens)? {
        return compile_for_assoc(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            first,
            second,
            iterable,
            declared,
        );
    }
    if !for_header_uses_c_style(&line.tokens)
        && let Some((local_name, declared, start, end, step)) = for_to_parts(&line.tokens)?
    {
        return compile_for_to(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            &local_name,
            declared,
            start,
            end,
            step,
        );
    }
    if let Some((local_name, declared, iterable, declared_type, filter_type)) =
        for_in_parts(&line.tokens)?
    {
        return compile_for_in(
            lines,
            line_index,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
            &local_name,
            declared,
            iterable,
            None,
            declared_type.as_ref(),
            filter_type.as_ref(),
        );
    }
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
    let child_indentation = lines.get(child_index).map(indentation);
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if child_indentation.is_some_and(|indent| indent > block_indentation) {
        compile_block(
            lines,
            child_index,
            child_indentation.expect("checked"),
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
    } else {
        Ok((child_index, true))
    };
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

fn for_header_uses_c_style(tokens: &[SpannedToken]) -> bool {
    let mut depth = 0usize;
    let separators = tokens
        .iter()
        .enumerate()
        .filter_map(|(index, token)| match token.kind {
            TokenKind::Punctuation('(' | '[') => {
                depth += 1;
                None
            }
            TokenKind::Punctuation(')' | ']') => {
                depth = depth.saturating_sub(1);
                None
            }
            TokenKind::Punctuation(';' | ',') if depth == 1 => Some(index),
            _ => None,
        })
        .collect::<Vec<_>>();
    separators.len() >= 2
        || separators.first().is_some_and(|separator| {
            tokens[*separator + 1..tokens.len().saturating_sub(1)]
                .iter()
                .any(|_| true)
        })
}

/// Compiles DM's inclusive numeric range loop, `for(var/i in first to last)`.
/// The end expression is evaluated once, matching the normal DM range-loop
/// header semantics and avoiding re-evaluating a mutable field on each turn.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_to(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    local_name: &str,
    declared: bool,
    start: &[SpannedToken],
    end: &[SpannedToken],
    step: Option<&[SpannedToken]>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let field_target = (!declared)
        .then(|| locals.src_field(local_name).cloned())
        .flatten();
    let item_slot = if declared {
        locals.declare(local_name.to_owned())?
    } else if let Some(slot) = locals.get(local_name) {
        slot
    } else if field_target.is_some() {
        locals.declare_hidden()?
    } else {
        return Err(compile_error(format!("unknown local {local_name:?}")));
    };
    let current_slot = locals.declare_hidden()?;
    let end_slot = locals.declare_hidden()?;
    let step_slot = step.map(|_| locals.declare_hidden()).transpose()?;

    let initialization_start = instructions.len();
    compile_expression(start, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(current_slot));
    compile_expression(end, locals, instructions, procedures)?;
    instructions.push(Instruction::StoreLocal(end_slot));
    if let Some(step) = step {
        compile_expression(step, locals, instructions, procedures)?;
        instructions.push(Instruction::StoreLocal(
            step_slot.expect("an explicit range step has a hidden slot"),
        ));
    }
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initialization_start,
    ));

    let condition_target = instructions.len();
    // `step` controls both the increment and direction.  Keep the bounds
    // inclusive, just like BYOND: positive steps run while `i <= end` and
    // negative steps run while `i >= end`.  The step expression is evaluated
    // once at loop entry, rather than once per iteration.
    if let Some(step_slot) = step_slot {
        for instruction in [
            Instruction::LoadLocal(step_slot),
            Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
            Instruction::GreaterEqual,
            Instruction::LoadLocal(current_slot),
            Instruction::LoadLocal(end_slot),
            Instruction::LessEqual,
            Instruction::And,
            Instruction::LoadLocal(step_slot),
            Instruction::PushNumber(DmNumberBits::from_f32(0.0)),
            Instruction::Less,
            Instruction::LoadLocal(current_slot),
            Instruction::LoadLocal(end_slot),
            Instruction::GreaterEqual,
            Instruction::And,
            Instruction::Or,
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
    } else {
        for instruction in [
            Instruction::LoadLocal(current_slot),
            Instruction::LoadLocal(end_slot),
            Instruction::LessEqual,
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
    }
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );
    // BYOND does not assign an existing iterator when the range is empty.
    // Keep the candidate in a hidden slot until the entry condition succeeds.
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(current_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(item_slot),
        line.span,
    );
    if let Some(field) = &field_target {
        for instruction in [
            Instruction::LoadSrc,
            Instruction::LoadLocal(item_slot),
            Instruction::StoreField(field.clone()),
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
    }
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("for-to statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("for-to statement requires an indented body"));
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
    let loop_context = loops.pop().expect("the active for-to context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    if let Some(field) = &field_target {
        push_instruction(instructions, source_spans, Instruction::LoadSrc, line.span);
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadField(field.clone()),
            line.span,
        );
    } else {
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadLocal(item_slot),
            line.span,
        );
    }
    let increment = step_slot.map_or(
        Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
        Instruction::LoadLocal,
    );
    for instruction in [
        increment,
        Instruction::Add,
        Instruction::StoreLocal(current_slot),
        Instruction::Jump(condition_target),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    if declared {
        locals.remove(local_name);
    }
    Ok(after_body)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_in(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    local_name: &str,
    declared: bool,
    iterable: &[SpannedToken],
    type_instances: Option<&TypePath>,
    declared_type: Option<&TypePath>,
    filter_type: Option<&TypePath>,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let result_target = !declared && local_name == ".";
    let field_target = (!declared && !result_target)
        .then(|| locals.src_field(local_name).cloned())
        .flatten();
    let global_target = (!declared && !result_target && field_target.is_none())
        .then(|| locals.global_field(local_name).cloned())
        .flatten();
    let item_slot = if result_target {
        locals.declare_hidden()?
    } else if declared {
        let slot = locals.declare(local_name.to_owned())?;
        if let Some(type_path) = declared_type {
            locals.set_type(local_name.to_owned(), type_path.clone());
        }
        slot
    } else if let Some(slot) = locals.get(local_name) {
        slot
    } else if field_target.is_some() || global_target.is_some() {
        locals.declare_hidden()?
    } else {
        return Err(compile_error(format!("unknown local {local_name:?}")));
    };
    let list_slot = locals.declare_hidden()?;
    let index_slot = locals.declare_hidden()?;

    let initialization_start = instructions.len();
    if let Some(type_path) = type_instances {
        instructions.push(Instruction::TypeInstances(type_path.clone()));
    } else {
        compile_expression(iterable, locals, instructions, procedures)?;
    }
    instructions.push(Instruction::PrepareIteration);
    instructions.push(Instruction::StoreLocal(list_slot));
    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    instructions.push(Instruction::StoreLocal(index_slot));
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - initialization_start,
    ));

    let condition_target = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(index_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::ListLengthLocal(list_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::LessEqual,
        line.span,
    );
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );

    push_instruction(
        instructions,
        source_spans,
        Instruction::LoadLocal(index_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::IndexLocalList(list_slot),
        line.span,
    );
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(item_slot),
        line.span,
    );
    let filter_jump = filter_type.map(|type_path| {
        for instruction in [
            Instruction::LoadLocal(item_slot),
            Instruction::IterationTypeFilter(type_path.clone()),
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
        let jump = instructions.len();
        push_instruction(
            instructions,
            source_spans,
            Instruction::JumpIfFalse(usize::MAX),
            line.span,
        );
        jump
    });
    if let Some(field) = &field_target {
        for instruction in [
            Instruction::LoadSrc,
            Instruction::LoadLocal(item_slot),
            Instruction::StoreField(field.clone()),
        ] {
            push_instruction(instructions, source_spans, instruction, line.span);
        }
    } else if let Some(global) = &global_target {
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadLocal(item_slot),
            line.span,
        );
        push_instruction(
            instructions,
            source_spans,
            Instruction::StoreGlobal(global.clone()),
            line.span,
        );
    }
    if result_target {
        push_instruction(
            instructions,
            source_spans,
            Instruction::LoadLocal(item_slot),
            line.span,
        );
        push_instruction(
            instructions,
            source_spans,
            Instruction::StoreResult,
            line.span,
        );
    }

    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
        .map(|(_, falls_through)| (line_index + 1, falls_through))
    } else {
        let child_index = line_index + 1;
        let child = lines
            .get(child_index)
            .ok_or_else(|| compile_error("for-in statement requires an indented body"))?;
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return Err(compile_error("for-in statement requires an indented body"));
        }
        compile_block(
            lines,
            child_index,
            child_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
    };
    let loop_context = loops.pop().expect("the active for-in context was pushed");
    let (after_body, _) = body?;

    let increment_target = instructions.len();
    if let Some(filter_jump) = filter_jump {
        patch_jump(instructions, filter_jump, increment_target)?;
    }
    for continue_jump in loop_context.continue_jumps {
        patch_jump(instructions, continue_jump, increment_target)?;
    }
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
        Instruction::Add,
        Instruction::StoreLocal(index_slot),
        Instruction::Jump(condition_target),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end_target = instructions.len();
    patch_jump(instructions, false_jump, end_target)?;
    for break_jump in loop_context.break_jumps {
        patch_jump(instructions, break_jump, end_target)?;
    }
    if declared {
        locals.remove(local_name);
    }
    Ok(after_body)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_for_assoc(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
    first: &[SpannedToken],
    second: &[SpannedToken],
    iterable: &[SpannedToken],
    declared: bool,
) -> Result<usize, CompileError> {
    let line = &lines[line_index];
    let (first_target, first_name) = parse_for_target(first, declared, locals)?;
    let (second_target, second_name) = parse_for_target(second, declared, locals)?;
    let list_slot = locals.declare_hidden()?;
    let index_slot = locals.declare_hidden()?;
    let key_slot = locals.declare_hidden()?;
    let value_slot = locals.declare_hidden()?;
    let start = instructions.len();
    compile_expression(iterable, locals, instructions, procedures)?;
    instructions.push(Instruction::PrepareIteration);
    instructions.push(Instruction::StoreLocal(list_slot));
    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
    instructions.push(Instruction::StoreLocal(index_slot));
    source_spans.extend(std::iter::repeat_n(line.span, instructions.len() - start));
    let condition = instructions.len();
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::ListLengthLocal(list_slot),
        Instruction::LessEqual,
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let false_jump = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::JumpIfFalse(usize::MAX),
        line.span,
    );
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::IndexLocalList(list_slot),
        Instruction::StoreLocal(key_slot),
        Instruction::LoadLocal(key_slot),
        Instruction::IndexLocalList(list_slot),
        Instruction::StoreLocal(value_slot),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    emit_for_target_store(&first_target, key_slot, locals, instructions, procedures)?;
    emit_for_target_store(&second_target, value_slot, locals, instructions, procedures)?;
    source_spans.extend(std::iter::repeat_n(
        line.span,
        instructions.len() - source_spans.len(),
    ));
    loops.push(LoopContext {
        continue_target: None,
        continue_jumps: Vec::new(),
        break_jumps: Vec::new(),
    });
    let body = if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
        .map(|(_, falls_through)| (line_index + 1, falls_through))
    } else {
        let child_index = line_index + 1;
        let child = lines
            .get(child_index)
            .ok_or_else(|| compile_error("for-in statement requires an indented body"))?;
        let child_indent = indentation(child);
        if child_indent <= block_indentation {
            return Err(compile_error("for-in statement requires an indented body"));
        }
        compile_block(
            lines,
            child_index,
            child_indent,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )
    };
    let context = loops.pop().expect("assoc loop context pushed");
    let (after_body, _) = body?;
    let increment = instructions.len();
    for jump in context.continue_jumps {
        patch_jump(instructions, jump, increment)?;
    }
    for instruction in [
        Instruction::LoadLocal(index_slot),
        Instruction::PushNumber(DmNumberBits::from_f32(1.0)),
        Instruction::Add,
        Instruction::StoreLocal(index_slot),
        Instruction::Jump(condition),
    ] {
        push_instruction(instructions, source_spans, instruction, line.span);
    }
    let end = instructions.len();
    patch_jump(instructions, false_jump, end)?;
    for jump in context.break_jumps {
        patch_jump(instructions, jump, end)?;
    }
    if let Some(name) = first_name {
        locals.remove(&name);
    }
    if let Some(name) = second_name {
        locals.remove(&name);
    }
    Ok(after_body)
}

fn parse_for_target(
    tokens: &[SpannedToken],
    declared: bool,
    locals: &mut LocalTable,
) -> Result<(Expression, Option<String>), CompileError> {
    if declared {
        let name = tokens
            .iter()
            .rev()
            .find_map(|token| match &token.kind {
                TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
                _ => None,
            })
            .ok_or_else(|| compile_error("associative loop declaration has no name"))?;
        locals.declare(name.clone())?;
        return Ok((Expression::Local(name.clone()), Some(name)));
    }
    Ok((ExpressionParser::new(tokens).parse()?, None))
}

fn emit_for_target_store(
    target: &Expression,
    slot: u16,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(target) = locals.get(name) {
                instructions.push(Instruction::LoadLocal(slot));
                instructions.push(Instruction::StoreLocal(target));
            } else if let Some(field) = locals.src_field(name) {
                // An undeclared associative-loop target follows normal DM
                // assignment lookup. It may therefore name an existing src
                // field (`for(cointype in typesof(...))`) rather than a local.
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LoadLocal(slot));
                instructions.push(Instruction::StoreField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LoadLocal(slot));
                instructions.push(Instruction::StoreGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadLocal(slot));
            instructions.push(Instruction::SetListIndex);
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadLocal(slot));
            instructions.push(Instruction::SetListIndex);
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let null_target = instructions.len();
            instructions.push(Instruction::Pop);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(null_target);
            instructions[end_jump] = Instruction::Jump(end);
        }
        _ => return Err(compile_error("associative loop target is not writable")),
    }
    Ok(())
}

fn for_type_parts(tokens: &[SpannedToken]) -> Result<Option<(String, TypePath)>, CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        header.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Ok(None);
    }
    let inner = &header[1..header.len() - 1];
    if !matches!(inner.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
        || inner.iter().any(|token| {
            matches!(&token.kind,
            TokenKind::Identifier(name) if matches!(name.as_str(), "in" | "to"))
                || matches!(token.kind, TokenKind::Punctuation(',' | ';'))
        })
    {
        return Ok(None);
    }
    let names = inner
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if names.len() < 2 {
        return Ok(None);
    }
    let local = names.last().expect("length checked").clone();
    let path = format!("/{}", names[..names.len() - 1].join("/"));
    let path = TypePath::parse(&path).map_err(|error| compile_error(error.to_string()))?;
    Ok(Some((local, path)))
}

#[allow(clippy::type_complexity)]
fn for_assoc_parts(
    tokens: &[SpannedToken],
) -> Result<Option<(&[SpannedToken], &[SpannedToken], &[SpannedToken], bool)>, CompileError> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|t| &t.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return Ok(None);
    }
    let mut depth = 0usize;
    let mut closing = None;
    for (index, token) in header.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(') => depth += 1,
            TokenKind::Punctuation(')') => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closing = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(closing) = closing else {
        return Ok(None);
    };
    let inner = &header[1..closing];
    let Some(in_pos) = inner
        .iter()
        .position(|t| matches!(&t.kind, TokenKind::Identifier(n) if n == "in"))
    else {
        return Ok(None);
    };
    let targets = &inner[..in_pos];
    let iterable = &inner[in_pos + 1..];
    let mut depth = 0usize;
    let mut comma = None;
    for (index, token) in targets.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[') => depth += 1,
            TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(',') if depth == 0 => comma = Some(index),
            _ => {}
        }
    }
    let Some(comma) = comma else {
        return Ok(None);
    };
    if iterable.is_empty() || targets[..comma].is_empty() || targets[comma + 1..].is_empty() {
        return Err(compile_error(
            "associative for-in requires two targets and an iterable",
        ));
    }
    let declared =
        matches!(targets.first().map(|t| &t.kind), Some(TokenKind::Identifier(n)) if n == "var");
    Ok(Some((
        &targets[..comma],
        &targets[comma + 1..],
        iterable,
        declared,
    )))
}

fn for_in_parts(
    tokens: &[SpannedToken],
) -> Result<
    Option<(
        String,
        bool,
        &[SpannedToken],
        Option<TypePath>,
        Option<TypePath>,
    )>,
    CompileError,
> {
    let header = &tokens[1..];
    if !matches!(
        header.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return Ok(None);
    }
    let mut depth = 0usize;
    let mut closing = None;
    for (index, token) in header.iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(') => depth += 1,
            TokenKind::Punctuation(')') => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    closing = Some(index);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(closing) = closing else {
        return Ok(None);
    };
    let clauses = &header[1..closing];
    let clauses = if matches!(
        clauses.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(';'))
    ) && clauses[..clauses.len().saturating_sub(1)]
        .iter()
        .all(|token| token.kind != TokenKind::Punctuation(';'))
    {
        &clauses[..clauses.len() - 1]
    } else if clauses
        .iter()
        .any(|token| token.kind == TokenKind::Punctuation(';'))
    {
        return Ok(None);
    } else {
        clauses
    };
    let separators = top_level_keyword_positions(clauses, "in");
    if separators.len() > 1 {
        return Err(compile_error(
            "for-in header contains multiple 'in' keywords",
        ));
    }
    let Some(separator) = separators.first().copied() else {
        return Ok(None);
    };
    let declaration = &clauses[..separator];
    let iterable = &clauses[separator + 1..];
    if iterable.is_empty() {
        return Err(compile_error("for-in requires an iterable expression"));
    }
    let declared = matches!(
        declaration.first().map(|token| &token.kind),
        Some(TokenKind::Identifier(identifier)) if identifier == "var"
    );
    // A typed loop declaration may carry a cast qualifier after the local,
    // e.g. `var/turf/area_turf as anything`.  The qualifier describes the
    // iteration mode, not a second local name.  Restrict the name search to
    // the declaration portion before `as`, otherwise the old reverse scan
    // incorrectly registered `anything` and left `area_turf` unresolved in
    // the loop body.
    let declaration_end = declaration
        .iter()
        .position(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "as"),
        )
        .unwrap_or(declaration.len());
    let local_name = if matches!(declaration, [SpannedToken { kind: TokenKind::Operator(operator), .. }] if operator == ".") {
        Some(".".to_owned())
    } else { declaration[..declaration_end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) if identifier != "var" => Some(identifier.clone()),
            _ => None,
        }) } .ok_or_else(|| compile_error("for-in variable declaration has no name"))?;
    let iterates_as_anything = declaration.windows(2).any(|tokens| {
        matches!(&tokens[0].kind, TokenKind::Identifier(identifier) if identifier == "as")
            && matches!(&tokens[1].kind, TokenKind::Identifier(identifier) if identifier == "anything")
    });
    // `as anything` explicitly disables the declaration's runtime type
    // filter. This is commonly used for typed loop variables that iterate
    // type paths (for example `var/datum/language/T as anything in
    // typesof(...)`). The declared type remains useful to the semantic pass,
    // but the VM must not discard those non-datum values here.
    let declared_type = declared
        .then(|| declared_local_type(declaration, &local_name))
        .flatten();
    let filter_type = (!iterates_as_anything)
        .then(|| declared_type.clone())
        .flatten();
    Ok(Some((
        local_name,
        declared,
        iterable,
        declared_type,
        filter_type,
    )))
}

/// Recognizes `for(var/name in first to last [step increment])`, rather than treating the
/// range's `to` keyword as the beginning of a normal iterable expression.
#[allow(clippy::type_complexity)]
fn for_to_parts(
    tokens: &[SpannedToken],
) -> Result<
    Option<(
        String,
        bool,
        &[SpannedToken],
        &[SpannedToken],
        Option<&[SpannedToken]>,
    )>,
    CompileError,
> {
    let (local_name, declared, iterable) =
        if let Some((name, declared, iterable, _, _)) = for_in_parts(tokens)? {
            (name, declared, iterable)
        } else {
            let header = &tokens[1..];
            if !matches!(
                header.first().map(|token| &token.kind),
                Some(TokenKind::Punctuation('('))
            ) || !matches!(
                header.last().map(|token| &token.kind),
                Some(TokenKind::Punctuation(')'))
            ) {
                return Ok(None);
            }
            let clauses = &header[1..header.len() - 1];
            let separators = top_level_keyword_positions(clauses, "to");
            let [to_separator] = separators.as_slice() else {
                return Ok(None);
            };
            let before_to = &clauses[..*to_separator];
            let Some(assignment) = before_to.iter().rposition(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
            ) else {
                return Ok(None);
            };
            let declaration = &before_to[..assignment];
            let declared = matches!(
                declaration.first().map(|token| &token.kind),
                Some(TokenKind::Identifier(identifier)) if identifier == "var"
            );
            let local_name = declaration
                .iter()
                .rev()
                .find_map(|token| match &token.kind {
                    TokenKind::Identifier(identifier) if identifier != "var" => {
                        Some(identifier.clone())
                    }
                    _ => None,
                })
                .ok_or_else(|| compile_error("for-to variable declaration has no name"))?;
            let start = &before_to[assignment + 1..];
            let iterable = &clauses[assignment + 1..];
            debug_assert!(iterable.starts_with(start));
            (local_name, declared, iterable)
        };
    let separators = top_level_keyword_positions(iterable, "to");
    let [separator] = separators.as_slice() else {
        return Ok(None);
    };
    let start = &iterable[..*separator];
    let after_to = &iterable[*separator + 1..];
    let after_to = after_to
        .iter()
        .position(|token| token.kind == TokenKind::Punctuation(';'))
        .map_or(after_to, |end| &after_to[..end]);
    // The first top-level `step` begins the increment expression. Subsequent
    // occurrences are ordinary identifiers inside that expression (for
    // example, `step step` when the increment is held in a local named
    // `step`).
    let step_separator = top_level_keyword_positions(after_to, "step")
        .into_iter()
        .next();
    let (end, step) = match step_separator {
        None => (after_to, None),
        Some(separator) => (&after_to[..separator], Some(&after_to[separator + 1..])),
    };
    if start.is_empty() || end.is_empty() {
        return Err(compile_error("for-to range requires both bounds"));
    }
    if step.is_some_and(<[SpannedToken]>::is_empty) {
        return Err(compile_error("for-to range step requires an increment"));
    }
    Ok(Some((local_name, declared, start, end, step)))
}

/// Finds DM header keywords outside nested calls, indexes, and list literals.
/// Range bounds may legally refer to locals named `to` or `step` inside a
/// nested expression, so only a top-level occurrence can delimit a `for`
/// range clause.
fn top_level_keyword_positions(tokens: &[SpannedToken], keyword: &str) -> Vec<usize> {
    let mut depth = 0usize;
    let mut positions = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Identifier(identifier) if depth == 0 && identifier == keyword => {
                positions.push(index);
            }
            _ => {}
        }
    }
    positions
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
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(';' | ',') if depth == 0 => separators.push(index),
            _ => {}
        }
    }
    if separators.is_empty() && clauses.is_empty() {
        return Ok([clauses, clauses, clauses]);
    }
    if separators.len() == 1 {
        let separator = separators[0];
        return Ok([
            &clauses[..separator],
            &clauses[separator + 1..],
            &clauses[0..0],
        ]);
    }
    if separators.len() != 2 {
        if clauses.iter().any(
            |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "in"),
        ) {
            return Err(compile_error("for-in list iteration is not implemented"));
        }
        return Err(compile_error(
            "C-style for requires initializer, condition, and increment clauses separated by ';' or ','",
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
        // In C-style headers BYOND accepts a declaration followed by an
        // `in range` type-filter-looking suffix. It does not iterate that
        // range; the suffix qualifies the initializer and the declared value
        // remains the ordinary left-hand initializer.
        let tokens = top_level_keyword_positions(tokens, "in")
            .first()
            .map_or(tokens, |separator| &tokens[..*separator]);
        let separators = tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| {
                matches!(&token.kind, TokenKind::Operator(operator) if operator == "&&")
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        if !separators.is_empty() {
            let mut start = 0usize;
            let mut last = None;
            for end in separators.into_iter().chain(std::iter::once(tokens.len())) {
                let declaration = &tokens[start..end];
                if !matches!(declaration.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
                {
                    return Err(compile_error(
                        "combined for initializer must contain variable declarations",
                    ));
                }
                last = Some(compile_local(
                    declaration,
                    locals,
                    instructions,
                    procedures,
                )?);
                start = end + 1;
            }
            return Ok(last);
        }
        return compile_local(tokens, locals, instructions, procedures).map(Some);
    }
    if let [first, operator, expression @ ..] = tokens
        && let (TokenKind::Identifier(name), TokenKind::Operator(operator)) =
            (&first.kind, &operator.kind)
        && operator == "="
    {
        if let Some(slot) = locals.get(name) {
            compile_expression(expression, locals, instructions, procedures)?;
            instructions.push(Instruction::StoreLocal(slot));
        } else if let Some(field) = locals.src_field(name) {
            instructions.push(Instruction::LoadSrc);
            compile_expression(expression, locals, instructions, procedures)?;
            instructions.push(Instruction::StoreField(field.clone()));
        } else if let Some(global) = locals.global_field(name) {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(global.clone()));
            }
            compile_expression(expression, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreGlobal(global.clone()));
        } else {
            return Err(compile_error(format!("unknown local {name:?}")));
        }
        return Ok(None);
    }
    if let Some((name, increment)) = local_increment(tokens) {
        let local = locals.get(name);
        let field = locals.src_field(name).cloned();
        let global = locals.global_field(name).cloned();
        if let Some(slot) = local {
            instructions.push(Instruction::LoadLocal(slot));
        } else if let Some(field) = &field {
            instructions.push(Instruction::LoadSrc);
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::LoadField(field.clone()));
        } else if let Some(global) = &global {
            instructions.push(Instruction::LoadGlobal(global.clone()));
        } else {
            return Err(compile_error(format!("unknown local {name:?}")));
        }
        instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
        instructions.push(if increment {
            Instruction::Add
        } else {
            Instruction::Subtract
        });
        if let Some(slot) = local {
            instructions.push(Instruction::StoreLocal(slot));
        } else if field.is_some() {
            instructions.push(Instruction::StoreField(field.expect("field was checked")));
        } else {
            instructions.push(Instruction::StoreGlobal(
                global.expect("global was checked"),
            ));
        }
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
fn compile_try(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let try_line = &lines[line_index];
    if try_line.tokens.len() != 1 {
        return Err(compile_error("try does not accept an expression"));
    }
    let child_index = line_index + 1;
    let child = lines
        .get(child_index)
        .ok_or_else(|| compile_error("try statement requires an indented body"))?;
    let child_indentation = indentation(child);
    if child_indentation <= block_indentation {
        return Err(compile_error("try statement requires an indented body"));
    }

    let handler_instruction = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::BeginTry {
            catch: usize::MAX,
            end: usize::MAX,
            local: None,
        },
        try_line.span,
    );
    let (catch_index, try_falls_through) = compile_block(
        lines,
        child_index,
        child_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    let catch_line = lines
        .get(catch_index)
        .filter(|line| {
            indentation(line) == block_indentation
                && matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "catch")
        })
        .ok_or_else(|| compile_error("try requires a matching catch"))?;
    let catch_local_name = parse_catch_local(&catch_line.tokens)?;
    let catch_local = catch_local_name
        .as_ref()
        .map(|_| locals.declare_hidden())
        .transpose()?;

    let protected_end = instructions.len();
    push_instruction(
        instructions,
        source_spans,
        Instruction::EndTry,
        catch_line.span,
    );
    // A terminating protected body cannot reach the catch-skipping branch.
    // Omitting that dead instruction also keeps a try/catch whose two arms
    // terminate from pointing one past the complete procedure.
    let end_jump = try_falls_through.then(|| {
        let jump = instructions.len();
        push_instruction(
            instructions,
            source_spans,
            Instruction::Jump(usize::MAX),
            catch_line.span,
        );
        jump
    });
    let catch_target = instructions.len();
    instructions[handler_instruction] = Instruction::BeginTry {
        catch: catch_target,
        end: protected_end,
        local: catch_local,
    };

    let catch_child_index = catch_index + 1;
    let catch_indentation = lines.get(catch_child_index).map(indentation);
    // An empty catch is legal (`catch` followed by the next sibling
    // statement) and simply consumes the thrown value. A try itself may not
    // be empty, which also preserves BYOND's OD0015 diagnostic for an empty
    // try/catch pair.
    if catch_indentation.is_none_or(|indentation| indentation <= block_indentation) {
        let end_target = instructions.len();
        if let Some(end_jump) = end_jump {
            patch_jump(instructions, end_jump, end_target)?;
        }
        return Ok((catch_child_index, true));
    }
    let catch_indentation = catch_indentation.expect("indentation was checked");
    let saved_names = locals.names.clone();
    if let (Some(name), Some(slot)) = (catch_local_name, catch_local) {
        locals.names.insert(name, slot);
    }
    let (next_line, catch_falls_through) = compile_block(
        lines,
        catch_child_index,
        catch_indentation,
        locals,
        instructions,
        source_spans,
        procedures,
        loops,
    )?;
    locals.names = saved_names;
    let end_target = instructions.len();
    if let Some(end_jump) = end_jump {
        patch_jump(instructions, end_jump, end_target)?;
    }
    Ok((next_line, try_falls_through || catch_falls_through))
}

fn parse_catch_local(tokens: &[SpannedToken]) -> Result<Option<String>, CompileError> {
    if tokens.len() == 1 {
        return Ok(None);
    }
    if !matches!(
        tokens.get(1).map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        tokens.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return Err(compile_error("catch variable requires parentheses"));
    }
    let inner = &tokens[2..tokens.len() - 1];
    if inner.is_empty() {
        return Ok(None);
    }
    if !matches!(inner.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "var")
    {
        return Err(compile_error(
            "catch binding must be a variable declaration",
        ));
    }
    let name = inner.iter().rev().find_map(|token| match &token.kind {
        TokenKind::Identifier(name) if name != "var" => Some(name.clone()),
        _ => None,
    });
    name.map(Some)
        .ok_or_else(|| compile_error("catch variable declaration requires a name"))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
    // DM permits a single statement after the closing condition delimiter,
    // e.g. `if (ready) continue` and `if (missing) return`.  SourceLine
    // keeps that statement on the same physical line, so compile it through
    // the ordinary block machinery using a synthetic one-line block.  This
    // deliberately also preserves `break`/`continue` loop context and all
    // ordinary statement lowering instead of special-casing return here.
    let (after_then, then_falls_through) = if let Some(body) = inline_conditional_body(&line.tokens)
        && matches!(body.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "do")
    {
        // Macro expansions frequently produce `if(condition) do { ... }
        // while(0)`. The brace normalizer has already placed the compact do
        // body on subsequent logical lines, so retain that tail while
        // replacing only the leading conditional with its inline statement.
        let mut inline_lines = lines[line_index..].to_vec();
        inline_lines[0].tokens = body.to_vec();
        let consumed = compile_do_while(
            &inline_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (line_index + consumed, true)
    } else if let Some(body) = inline_conditional_body(&line.tokens) {
        let mut inline_line = line.clone();
        inline_line.tokens = body.to_vec();
        let (_, falls_through) = compile_block(
            std::slice::from_ref(&inline_line),
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (line_index + 1, falls_through)
    } else {
        let child_index = line_index + 1;
        let child = lines
            .get(child_index)
            .ok_or_else(|| compile_error("if statement requires an indented body"))?;
        let child_indentation = indentation(child);
        if child_indentation <= block_indentation {
            return Err(compile_error("if statement requires an indented body"));
        }
        compile_block(
            lines,
            child_index,
            child_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?
    };
    if !lines
        .get(after_then)
        .is_some_and(|candidate| is_else(candidate, block_indentation))
    {
        let end_target = instructions.len();
        patch_jump(instructions, false_jump, end_target)?;
        return Ok((after_then, true));
    }

    let else_line = &lines[after_then];
    // Only a live then arm needs to skip over the else body. Emitting this
    // branch after a terminating `return`, `throw`, or loop-control statement
    // leaves unreachable bytecode whose target can be the program boundary
    // when the else arm terminates too.
    let end_jump = then_falls_through.then(|| {
        let jump = instructions.len();
        push_instruction(
            instructions,
            source_spans,
            Instruction::Jump(usize::MAX),
            else_line.span,
        );
        jump
    });
    let else_target = instructions.len();
    patch_jump(instructions, false_jump, else_target)?;
    let (after_else, else_falls_through) = if is_else_if(else_line) {
        // `else if` is a nested conditional in DM.  Re-present the tail of
        // the source as an `if` block so its condition and any inline body
        // take the same lowering path as a top-level conditional.
        let mut nested_lines = lines[after_then..].to_vec();
        nested_lines[0].tokens = nested_lines[0].tokens[1..].to_vec();
        let (after_nested, falls_through) = compile_if(
            &nested_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (after_then + after_nested, falls_through)
    } else if let Some(body) = inline_else_body(&else_line.tokens) {
        // `else for(...)` and `else while(...)` keep their controlled body on
        // the following indented lines. Preserve the remaining source rather
        // than compiling only a synthetic header line.
        let mut inline_lines = lines[after_then..].to_vec();
        inline_lines[0].tokens = body.to_vec();
        let (consumed, falls_through) = compile_block(
            &inline_lines,
            0,
            block_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?;
        (after_then + consumed, falls_through)
    } else {
        let else_child_index = after_then + 1;
        let else_child = lines
            .get(else_child_index)
            .ok_or_else(|| compile_error("else statement requires an indented body"))?;
        let else_indentation = indentation(else_child);
        if else_indentation <= block_indentation {
            return Err(compile_error("else statement requires an indented body"));
        }
        compile_block(
            lines,
            else_child_index,
            else_indentation,
            locals,
            instructions,
            source_spans,
            procedures,
            loops,
        )?
    };
    let end_target = instructions.len();
    if let Some(end_jump) = end_jump {
        patch_jump(instructions, end_jump, end_target)?;
    }
    Ok((after_else, then_falls_through || else_falls_through))
}

/// Compiles DM's selector-based `switch` statement.
///
/// Unlike C, DM switch arms do not fall through.  Case arms are written as
/// `if(value)` (or `if(first to last)`) below the selector and are therefore
/// not ordinary conditional statements despite sharing their spelling.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_switch(
    lines: &[SourceLine],
    line_index: usize,
    block_indentation: usize,
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    procedures: &HashMap<String, ProcedureId>,
    loops: &mut Vec<LoopContext>,
) -> Result<(usize, bool), CompileError> {
    let switch_line = &lines[line_index];
    let selector = condition_tokens(&switch_line.tokens, "switch")?;
    let selector_start = instructions.len();
    compile_expression(selector, locals, instructions, procedures)?;
    let selector_slot = locals.declare_hidden()?;
    push_instruction(
        instructions,
        source_spans,
        Instruction::StoreLocal(selector_slot),
        switch_line.span,
    );
    source_spans.extend(std::iter::repeat_n(
        switch_line.span,
        instructions.len() - selector_start - 1,
    ));

    let first_case_index = line_index + 1;
    let first_case = lines
        .get(first_case_index)
        .ok_or_else(|| compile_error("switch statement requires an indented case body"))?;
    let case_indentation = indentation(first_case);
    if case_indentation <= block_indentation {
        return Err(compile_error(
            "switch statement requires an indented case body",
        ));
    }

    let mut next_case_index = first_case_index;
    let mut end_jumps = Vec::new();
    let mut saw_default = false;
    while let Some(case_line) = lines.get(next_case_index) {
        let current_indentation = indentation(case_line);
        if current_indentation < case_indentation {
            break;
        }
        if current_indentation > case_indentation {
            return Err(compile_error("unexpected indentation in switch statement"));
        }
        let is_case = matches!(
            case_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "if"
        );
        let is_default = matches!(
            case_line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "else"
        );
        if !is_case && !is_default {
            return Err(compile_error(
                "switch statement requires if cases or an else default",
            ));
        }
        if saw_default {
            return Err(compile_error("switch case cannot follow an else default"));
        }
        if is_default {
            saw_default = true;
        } else {
            let condition_start = instructions.len();
            emit_switch_case_condition(
                condition_tokens(&case_line.tokens, "switch case")?,
                selector_slot,
                locals,
                instructions,
                procedures,
            )?;
            source_spans.extend(std::iter::repeat_n(
                case_line.span,
                instructions.len() - condition_start,
            ));
        }
        let false_jump = if is_case {
            let jump = instructions.len();
            push_instruction(
                instructions,
                source_spans,
                Instruction::JumpIfFalse(usize::MAX),
                case_line.span,
            );
            Some(jump)
        } else {
            None
        };
        let inline_case_body = if is_default && case_line.tokens.len() > 1 {
            Some(&case_line.tokens[1..])
        } else {
            inline_conditional_body(&case_line.tokens)
        };
        let after_body = if let Some(body) = inline_case_body {
            let mut inline_line = case_line.clone();
            inline_line.tokens = body.to_vec();
            compile_block(
                std::slice::from_ref(&inline_line),
                0,
                case_indentation,
                locals,
                instructions,
                source_spans,
                procedures,
                loops,
            )?;
            next_case_index + 1
        } else {
            let body_index = next_case_index + 1;
            let body_indentation = lines.get(body_index).map(indentation);
            if body_indentation.is_some_and(|indent| indent > case_indentation) {
                compile_block(
                    lines,
                    body_index,
                    body_indentation.expect("checked"),
                    locals,
                    instructions,
                    source_spans,
                    procedures,
                    loops,
                )?
                .0
            } else {
                // A macro may deliberately expand a case body to a lone
                // semicolon (`EMPTY_BLOCK_GUARD`). The syntax normalizer
                // removes that empty statement; the case remains a valid
                // no-op and falls through to the end of the switch.
                body_index
            }
        };
        if !saw_default {
            let end_jump = instructions.len();
            push_instruction(
                instructions,
                source_spans,
                Instruction::Jump(usize::MAX),
                case_line.span,
            );
            end_jumps.push(end_jump);
        }
        if let Some(jump) = false_jump {
            let next_case_target = instructions.len();
            patch_jump(instructions, jump, next_case_target)?;
        }
        next_case_index = after_body;
        if saw_default {
            if lines
                .get(next_case_index)
                .is_some_and(|next| indentation(next) == case_indentation)
            {
                return Err(compile_error("switch case cannot follow an else default"));
            }
            break;
        }
    }
    let end_target = instructions.len();
    for jump in end_jumps {
        patch_jump(instructions, jump, end_target)?;
    }
    Ok((next_case_index, true))
}

fn emit_switch_case_condition(
    tokens: &[SpannedToken],
    selector_slot: u16,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let alternatives = split_switch_tokens(tokens, ',')?;
    if alternatives.is_empty() {
        return Err(compile_error("switch case requires at least one value"));
    }
    let alternative_count = alternatives
        .last()
        .is_some_and(|alternative| alternative.is_empty())
        .then(|| alternatives.len().saturating_sub(1))
        .unwrap_or(alternatives.len());
    if alternative_count == 0 {
        return Err(compile_error("switch case requires at least one value"));
    }
    for (alternative_index, alternative) in alternatives[..alternative_count].iter().enumerate() {
        if alternative.is_empty() {
            return Err(compile_error("switch case contains an empty value"));
        }
        let range = split_switch_keyword(alternative, "to")?;
        if let Some((lower, upper)) = range {
            if lower.is_empty() || upper.is_empty() {
                return Err(compile_error("switch range requires both bounds"));
            }
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(lower, locals, instructions, procedures)?;
            instructions.push(Instruction::GreaterEqual);
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(upper, locals, instructions, procedures)?;
            instructions.push(Instruction::LessEqual);
            instructions.push(Instruction::And);
        } else {
            instructions.push(Instruction::LoadLocal(selector_slot));
            compile_expression(alternative, locals, instructions, procedures)?;
            instructions.push(Instruction::Equal);
        }
        if alternative_index > 0 {
            instructions.push(Instruction::Or);
        }
    }
    Ok(())
}

fn split_switch_tokens(
    tokens: &[SpannedToken],
    separator: char,
) -> Result<Vec<&[SpannedToken]>, CompileError> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    compile_error("switch case contains unmatched closing punctuation")
                })?;
            }
            TokenKind::Punctuation(punctuation) if *punctuation == separator && depth == 0 => {
                groups.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(compile_error(
            "switch case contains unmatched opening punctuation",
        ));
    }
    groups.push(&tokens[start..]);
    Ok(groups)
}

#[allow(clippy::type_complexity)]
fn split_switch_keyword<'a>(
    tokens: &'a [SpannedToken],
    keyword: &str,
) -> Result<Option<(&'a [SpannedToken], &'a [SpannedToken])>, CompileError> {
    let mut depth = 0usize;
    let mut found = None;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1).ok_or_else(|| {
                    compile_error("switch range contains unmatched closing punctuation")
                })?;
            }
            TokenKind::Identifier(name)
                if name == keyword && depth == 0 && found.replace(index).is_some() =>
            {
                return Err(compile_error(
                    "switch range contains multiple 'to' keywords",
                ));
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(compile_error(
            "switch range contains unmatched opening punctuation",
        ));
    }
    Ok(found.map(|index| (&tokens[..index], &tokens[index + 1..])))
}

pub(crate) fn condition_tokens<'a>(
    tokens: &'a [SpannedToken],
    keyword: &str,
) -> Result<&'a [SpannedToken], CompileError> {
    let mut expression = &tokens[1..];
    // The preprocessor can retain the opening brace from a compact C-style
    // conditional such as `if (condition) {`.  Block structure remains
    // indentation-based in the lowered syntax, so it is not expression input.
    if matches!(
        expression.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation('{'))
    ) {
        expression = &expression[..expression.len() - 1];
    }
    if matches!(
        expression.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        let mut depth = 0usize;
        for (index, token) in expression.iter().enumerate() {
            match &token.kind {
                TokenKind::Punctuation('(') => depth += 1,
                TokenKind::Punctuation(')') => {
                    depth = depth.checked_sub(1).ok_or_else(|| {
                        compile_error(format!("{keyword} condition is missing '('"))
                    })?;
                    if depth == 0 {
                        return Ok(&expression[1..index]);
                    }
                }
                _ => {}
            }
        }
        return Err(compile_error(format!("{keyword} condition is missing ')'")));
    }
    if expression.is_empty() {
        return Err(compile_error(format!("{keyword} requires a condition")));
    }
    Ok(expression)
}

/// Returns the statement written after a parenthesized conditional on the
/// same physical source line.  A trailing `{` belongs to the preprocessor's
/// compact brace form and is not an inline DM statement.
fn inline_conditional_body(tokens: &[SpannedToken]) -> Option<&[SpannedToken]> {
    let expression = tokens.get(1..)?;
    if !matches!(
        expression.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return None;
    }
    let mut depth = 0usize;
    for (index, token) in expression.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(') => depth += 1,
            TokenKind::Punctuation(')') => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    let body = &expression[index + 1..];
                    return (!body.is_empty()
                        && !matches!(
                            body.first().map(|token| &token.kind),
                            Some(TokenKind::Punctuation('{'))
                        ))
                    .then_some(body);
                }
            }
            _ => {}
        }
    }
    None
}

/// Returns a body written directly after `else`, such as `else return`.
/// `else if` deliberately remains a nested conditional form and is handled
/// by the regular indented parser path.
fn inline_else_body(tokens: &[SpannedToken]) -> Option<&[SpannedToken]> {
    let body = tokens.get(1..)?;
    (!body.is_empty()
        && !matches!(
            body.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "if"
        )
        && !matches!(
            body.first().map(|token| &token.kind),
            Some(TokenKind::Punctuation('{'))
        ))
    .then_some(body)
}

fn is_else_if(line: &SourceLine) -> bool {
    matches!(
        line.tokens.as_slice(),
        [
            SpannedToken {
                kind: TokenKind::Identifier(else_keyword),
                ..
            },
            SpannedToken {
                kind: TokenKind::Identifier(if_keyword),
                ..
            },
            ..
        ] if else_keyword == "else" && if_keyword == "if"
    )
}

pub(crate) fn indentation(line: &SourceLine) -> usize {
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

pub(crate) fn push_instruction(
    instructions: &mut Vec<Instruction>,
    source_spans: &mut Vec<SourceSpan>,
    instruction: Instruction,
    span: SourceSpan,
) {
    instructions.push(instruction);
    source_spans.push(span);
}

pub(crate) fn patch_jump(
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

fn compile_result_assignment(
    tokens: &[SpannedToken],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let Some(TokenKind::Operator(assignment)) = tokens.get(1).map(|token| &token.kind) else {
        return Err(compile_error(
            "special return value '.' requires an assignment",
        ));
    };
    if tokens.len() < 3 {
        return Err(compile_error(
            "special return value assignment requires an expression",
        ));
    }
    if assignment != "=" {
        instructions.push(Instruction::LoadResult);
    }
    compile_expression(&tokens[2..], locals, instructions, procedures)?;
    if assignment != "=" {
        instructions.push(match assignment.as_str() {
            "+=" => Instruction::Add,
            "-=" => Instruction::Subtract,
            "*=" => Instruction::Multiply,
            "/=" => Instruction::Divide,
            "%=" => Instruction::Remainder,
            "&=" => Instruction::BitAnd,
            "|=" => Instruction::BitOr,
            "^=" => Instruction::BitXor,
            "<<=" => Instruction::ShiftLeft,
            ">>=" => Instruction::ShiftRight,
            _ => {
                return Err(compile_error(format!(
                    "unsupported special return value assignment operator {assignment:?}"
                )));
            }
        });
    }
    instructions.push(Instruction::StoreResult);
    Ok(())
}

fn compile_local(
    tokens: &[SpannedToken],
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<String, CompileError> {
    let assignment = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="));
    let declaration_end = assignment.unwrap_or(tokens.len());
    let suffix = tokens[1..declaration_end]
        .iter()
        .position(|token| matches!(token.kind, TokenKind::Punctuation('[')))
        .map_or(declaration_end, |offset| 1 + offset);
    let name = tokens[1..suffix]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.clone()),
            _ => None,
        })
        .ok_or_else(|| compile_error("local declaration has no name"))?;
    let declared_type = declared_local_type(tokens, &name);
    let slot = locals.declare(name.clone())?;
    if let Some(type_path) = declared_type.clone() {
        locals.set_type(name.clone(), type_path);
    }
    let is_static = tokens[1..declaration_end].iter().any(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "static"),
    );
    let static_jump = is_static.then(|| {
        let index = instructions.len();
        instructions.push(Instruction::LoadStaticLocalOrJump {
            slot,
            target: usize::MAX,
        });
        index
    });
    if let Some(assignment) = assignment {
        let mut value = ExpressionParser::new(&tokens[assignment + 1..]).parse()?;
        if let Expression::New { type_path, .. } = &mut value
            && type_path.is_none()
            && let Some(inferred) = declared_type.as_ref()
        {
            *type_path = Some(Box::new(Expression::TypePath(inferred.clone())));
        }
        infer_contextual_locate(&mut value, declared_type.as_ref());
        emit_expression(&value, locals, instructions, procedures)?;
    } else if suffix < declaration_end {
        let mut dimensions = 0u8;
        let mut cursor = suffix;
        while cursor < declaration_end {
            if !matches!(tokens[cursor].kind, TokenKind::Punctuation('[')) {
                cursor += 1;
                continue;
            }
            let mut bracket_depth = 1usize;
            let close = (cursor + 1..declaration_end)
                .find(|&index| {
                    match tokens[index].kind {
                        TokenKind::Punctuation('[') => bracket_depth += 1,
                        TokenKind::Punctuation(']') => {
                            bracket_depth = bracket_depth.saturating_sub(1)
                        }
                        _ => {}
                    }
                    bracket_depth == 0
                })
                .ok_or_else(|| compile_error("array declaration has an unclosed dimension"))?;
            compile_expression(&tokens[cursor + 1..close], locals, instructions, procedures)?;
            dimensions = dimensions
                .checked_add(1)
                .ok_or_else(|| compile_error("too many array dimensions"))?;
            cursor = close + 1;
        }
        instructions.push(Instruction::MakeArray(dimensions));
    } else {
        // Typed and untyped local declarations without an initializer begin
        // as null in DM.
        instructions.push(Instruction::PushNull);
    }
    if is_static {
        instructions.push(Instruction::InitializeStaticLocal(slot));
    }
    instructions.push(Instruction::StoreLocal(slot));
    if let Some(jump) = static_jump {
        let target = instructions.len();
        instructions[jump] = Instruction::LoadStaticLocalOrJump { slot, target };
    }
    Ok(name)
}

fn infer_contextual_locate(expression: &mut Expression, declared_type: Option<&TypePath>) {
    let Some(declared_type) = declared_type else {
        return;
    };
    let arguments = match expression {
        Expression::Locate { arguments } | Expression::LocateIn { arguments, .. } => arguments,
        _ => return,
    };
    if arguments.is_empty() {
        arguments.push(Expression::TypePath(declared_type.clone()));
    }
}

fn declared_local_type(tokens: &[SpannedToken], name: &str) -> Option<TypePath> {
    let declaration_end = tokens
        .iter()
        .position(|token| {
            matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                || matches!(&token.kind, TokenKind::Identifier(identifier) if matches!(identifier.as_str(), "as" | "in"))
        })
        .unwrap_or(tokens.len());
    let name_index = tokens[..declaration_end].iter().rposition(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == name),
    )?;
    let var_index = tokens[..name_index].iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "var"),
    )?;
    let segments = tokens[var_index + 1..name_index]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier)
                if !matches!(identifier.as_str(), "static" | "global" | "tmp" | "final") =>
            {
                Some(identifier.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    declared_type_path(&segments)
}

fn declared_type_path(segments: &[String]) -> Option<TypePath> {
    if segments.is_empty() {
        return None;
    }
    // In `list/datum/member/items`, only `list` is the variable's runtime
    // type. The remaining path is BYOND's optional element-type annotation;
    // `/list/datum/member` is not a list subtype.
    let segments = if segments.first().is_some_and(|segment| segment == "list") {
        &segments[..1]
    } else {
        segments
    };
    TypePath::parse(&format!("/{}", segments.join("/"))).ok()
}

fn compile_local_declarations(
    tokens: &[SpannedToken],
    locals: &mut LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    let mut depth = 0_usize;
    let mut start = 1_usize;
    let mut parts = Vec::new();
    for (index, token) in tokens.iter().enumerate().skip(1) {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Operator(operator) if operator == "?[" => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(',') if depth == 0 => {
                parts.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    parts.push(&tokens[start..]);
    for part in parts {
        if part.is_empty() {
            return Err(compile_error("local declaration after ',' is empty"));
        }
        let mut declaration = Vec::with_capacity(part.len() + 1);
        declaration.push(tokens[0].clone());
        declaration.extend_from_slice(part);
        compile_local(&declaration, locals, instructions, procedures)?;
    }
    Ok(())
}

pub(crate) fn parameter_name(tokens: &[SpannedToken]) -> Option<&str> {
    let end = tokens
        .iter()
        .position(|token| {
            matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                || matches!(&token.kind, TokenKind::Identifier(identifier) if matches!(identifier.as_str(), "as" | "in"))
        })
        .unwrap_or(tokens.len());
    tokens[..end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.as_str()),
            _ => None,
        })
}

pub(crate) fn verb_parameter_type(tokens: &[SpannedToken]) -> VerbParameterType {
    let Some(as_index) = tokens.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == "as"),
    ) else {
        return VerbParameterType::Anything;
    };
    let types = tokens[as_index + 1..]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let non_null = types
        .iter()
        .copied()
        .filter(|value| *value != "null")
        .collect::<Vec<_>>();
    match non_null.as_slice() {
        ["text" | "command_text"] => VerbParameterType::Text,
        ["message"] => VerbParameterType::Message,
        ["num"] => VerbParameterType::Number,
        ["color"] => VerbParameterType::Color,
        ["file" | "icon" | "sound"] => VerbParameterType::File,
        ["anything"] | [] => VerbParameterType::Anything,
        values
            if values
                .iter()
                .all(|value| ["obj", "mob", "turf", "area"].contains(value)) =>
        {
            let mask = values.iter().fold(0, |mask, value| {
                mask | match *value {
                    "obj" => 1,
                    "mob" => 2,
                    "turf" => 4,
                    "area" => 8,
                    _ => 0,
                }
            });
            VerbParameterType::Atom(mask)
        }
        _ => VerbParameterType::Anything,
    }
}

pub(crate) fn declared_parameter_type(tokens: &[SpannedToken], name: &str) -> Option<TypePath> {
    let declaration_end = tokens
        .iter()
        .position(|token| {
            matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                || matches!(&token.kind, TokenKind::Identifier(identifier) if matches!(identifier.as_str(), "as" | "in"))
        })
        .unwrap_or(tokens.len());
    let name_index = tokens[..declaration_end].iter().rposition(
        |token| matches!(&token.kind, TokenKind::Identifier(identifier) if identifier == name),
    )?;
    let segments = tokens[..name_index]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier)
                if !matches!(
                    identifier.as_str(),
                    "var" | "static" | "global" | "tmp" | "final"
                ) =>
            {
                Some(identifier.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    declared_type_path(&segments)
}

pub(crate) fn expression_static_type(
    expression: &Expression,
    locals: &LocalTable<'_>,
) -> Option<TypePath> {
    match expression {
        Expression::Local(name) => locals
            .local_type(name)
            .or_else(|| locals.global_type(name))
            .cloned(),
        Expression::GlobalField(name) => locals.global_type(name.as_str()).cloned(),
        Expression::New {
            type_path: Some(type_path),
            ..
        } => match type_path.as_ref() {
            Expression::TypePath(path) => Some(path.clone()),
            _ => None,
        },
        _ => None,
    }
}
