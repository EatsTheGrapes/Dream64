from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}\n--- needle ---\n{old}")
    p.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# dm-project: BYOND predefined __LINE__ compiler macro, including nested macros.
# ---------------------------------------------------------------------------
project = "crates/dm-project/src/lib.rs"
replace_once(
    project,
    '''                if name == "__FILE__" {
                    self.append_original(
                        source,
                        SourceSpan::new(span.start + literal_start, span.start + offset),
                    );
                    let invocation =
                        SourceSpan::new(span.start + offset, span.start + identifier_end);
                    self.append_replacement(&format!("{file_macro:?}"), invocation);
                    offset = identifier_end;
                    literal_start = offset;
                    continue;
                }
''',
    '''                if matches!(name, "__FILE__" | "__LINE__") {
                    self.append_original(
                        source,
                        SourceSpan::new(span.start + literal_start, span.start + offset),
                    );
                    let invocation =
                        SourceSpan::new(span.start + offset, span.start + identifier_end);
                    let replacement = if name == "__FILE__" {
                        format!("{file_macro:?}")
                    } else {
                        source.as_bytes()[..invocation.start]
                            .iter()
                            .filter(|byte| **byte == b'\\n')
                            .count()
                            .saturating_add(1)
                            .to_string()
                    };
                    self.append_replacement(&replacement, invocation);
                    offset = identifier_end;
                    literal_start = offset;
                    continue;
                }
''',
)
replace_once(
    project,
    '''                    let replacement = expand_macro(
                        name,
                        arguments.as_deref(),
                        macros,
                        &mut Vec::new(),
                        &file_macro,
                    )
''',
    '''                    let line_macro = source.as_bytes()[..invocation.start]
                        .iter()
                        .filter(|byte| **byte == b'\\n')
                        .count()
                        .saturating_add(1);
                    let replacement = expand_macro(
                        name,
                        arguments.as_deref(),
                        macros,
                        &mut Vec::new(),
                        &file_macro,
                        line_macro,
                    )
''',
)
replace_once(
    project,
    '''fn expand_macro(
    name: &str,
    arguments: Option<&[String]>,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
) -> Result<String, String> {
    if name == "__FILE__" {
        return Ok(format!("{file_macro:?}"));
    }
''',
    '''fn expand_macro(
    name: &str,
    arguments: Option<&[String]>,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
    if name == "__FILE__" {
        return Ok(format!("{file_macro:?}"));
    }
    if name == "__LINE__" {
        return Ok(line_macro.to_string());
    }
''',
)
replace_once(
    project,
    '''                substitute_function_macro(
                    name, definition, parameters, arguments, macros, stack, file_macro,
                )
''',
    '''                substitute_function_macro(
                    name,
                    definition,
                    parameters,
                    arguments,
                    macros,
                    stack,
                    file_macro,
                    line_macro,
                )
''',
)
replace_once(
    project,
    '''        expand_replacement(&definition.replacement, macros, stack, file_macro)
''',
    '''        expand_replacement(
            &definition.replacement,
            macros,
            stack,
            file_macro,
            line_macro,
        )
''',
)
replace_once(
    project,
    '''fn expand_replacement(
    replacement: &str,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
) -> Result<String, String> {
''',
    '''fn expand_replacement(
    replacement: &str,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
''',
)
replace_once(
    project,
    '''            if name == "__FILE__" {
                let file_literal = format!("{file_macro:?}");
                output.push_str(&file_literal);
                offset = end;
                continue;
            }
''',
    '''            if name == "__FILE__" {
                let file_literal = format!("{file_macro:?}");
                output.push_str(&file_literal);
                offset = end;
                continue;
            }
            if name == "__LINE__" {
                output.push_str(&line_macro.to_string());
                offset = end;
                continue;
            }
''',
)
replace_once(
    project,
    '''                            output.push_str(&expand_macro(
                                name,
                                Some(&arguments),
                                macros,
                                stack,
                                file_macro,
                            )?);
''',
    '''                            output.push_str(&expand_macro(
                                name,
                                Some(&arguments),
                                macros,
                                stack,
                                file_macro,
                                line_macro,
                            )?);
''',
)
replace_once(
    project,
    '''                    output.push_str(&expand_macro(name, None, macros, stack, file_macro)?);
''',
    '''                    output.push_str(&expand_macro(
                        name,
                        None,
                        macros,
                        stack,
                        file_macro,
                        line_macro,
                    )?);
''',
)
replace_once(
    project,
    '''fn substitute_function_macro(
    name: &str,
    definition: &MacroDefinition,
    parameters: &MacroParameters,
    arguments: &[String],
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
) -> Result<String, String> {
''',
    '''fn substitute_function_macro(
    name: &str,
    definition: &MacroDefinition,
    parameters: &MacroParameters,
    arguments: &[String],
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
''',
)
replace_once(
    project,
    '''    expand_replacement(&substituted, macros, stack, file_macro)
''',
    '''    expand_replacement(&substituted, macros, stack, file_macro, line_macro)
''',
)
replace_once(
    project,
    '''    #[test]
    fn shares_defines_across_recursive_includes() {
''',
    '''    #[test]
    fn expands_predefined_line_macro_at_the_invocation_line() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#define SOURCE_LINE __LINE__\\n",
            "/proc/source_line()\\n",
            "\\treturn SOURCE_LINE\\n",
        );
        fs::write(scratch.path().join("world.dme"), source)
            .expect("line macro fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("predefined line macro should expand");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should remain UTF-8");

        assert!(!expanded.contains("__LINE__"));
        assert!(!expanded.contains("SOURCE_LINE"));
        assert!(expanded.contains("\\treturn 3\\n"), "expanded source was {expanded:?}");
    }

    #[test]
    fn shares_defines_across_recursive_includes() {
''',
)

# ---------------------------------------------------------------------------
# dm-vm: block(), copytext/copytext_char(), and lexical block-local scopes.
# ---------------------------------------------------------------------------
vm = "crates/dm-vm/src/lib.rs"
replace_once(
    vm,
    '''    ReplaceText {
        /// Number of supplied arguments (three through five).
        argument_count: u8,
        /// Whether matches are case-sensitive.
        exact: bool,
        /// Whether optional bounds count Unicode scalar values.
        character_indices: bool,
    },
    /// Produces a deterministic pseudo-random integer in an inclusive range.
''',
    '''    ReplaceText {
        /// Number of supplied arguments (three through five).
        argument_count: u8,
        /// Whether matches are case-sensitive.
        exact: bool,
        /// Whether optional bounds count Unicode scalar values.
        character_indices: bool,
    },
    /// Copies a bounded section of text using BYOND's 1-based positions.
    CopyText {
        /// Number of supplied arguments (one through three).
        argument_count: u8,
        /// Whether positions count Unicode scalar values rather than bytes.
        character_indices: bool,
    },
    /// Enumerates every materialized turf in an inclusive 3D rectangular block.
    Block {
        /// Number of supplied arguments: two turfs, or three through six coordinates.
        argument_count: u8,
    },
    /// Produces a deterministic pseudo-random integer in an inclusive range.
''',
)
replace_once(
    vm,
    '''#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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
''',
    '''#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_block(
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
''',
)
replace_once(
    vm,
    '''    ReplaceText {
        arguments: Vec<Self>,
        exact: bool,
        character_indices: bool,
    },
    Rand {
''',
    '''    ReplaceText {
        arguments: Vec<Self>,
        exact: bool,
        character_indices: bool,
    },
    CopyText {
        arguments: Vec<Self>,
        character_indices: bool,
    },
    Block {
        arguments: Vec<Self>,
    },
    Rand {
''',
)
replace_once(
    vm,
    '''                } else if identifier == "length" {
''',
    '''                } else if matches!(identifier.as_str(), "copytext" | "copytext_char") {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=3).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::CopyText {
                        arguments,
                        character_indices: identifier == "copytext_char",
                    })
                } else if identifier == "length" {
''',
)
replace_once(
    vm,
    '''                } else if identifier == "typesof" {
''',
    '''                } else if identifier == "block" {
                    let arguments = self.parse_call_arguments()?;
                    if !(arguments.len() == 2 || (3..=6).contains(&arguments.len())) {
                        return Err(compile_error(format!(
                            "block requires two turfs or three through six coordinates, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Block { arguments })
                } else if identifier == "typesof" {
''',
)
replace_once(
    vm,
    '''        Expression::Length { value } => {
''',
    '''        Expression::CopyText {
            arguments,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::CopyText {
                argument_count: u8::try_from(arguments.len())
                    .expect("copytext argument count was validated by the parser"),
                character_indices: *character_indices,
            });
        }
        Expression::Block { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Block {
                argument_count: u8::try_from(arguments.len())
                    .expect("block argument count was validated by the parser"),
            });
        }
        Expression::Length { value } => {
''',
)
replace_once(
    vm,
    '''        | Expression::ReplaceText { arguments, .. }
        | Expression::Rand { arguments }
''',
    '''        | Expression::ReplaceText { arguments, .. }
        | Expression::CopyText { arguments, .. }
        | Expression::Block { arguments }
        | Expression::Rand { arguments }
''',
)
replace_once(
    vm,
    '''            Instruction::Length => {
''',
    '''            Instruction::CopyText {
                argument_count,
                character_indices,
            } => {
                let count = usize::from(argument_count);
                if !(1..=3).contains(&count) || frames[frame_index].stack.len() < count {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid copytext builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = copy_text_builtin(&arguments, character_indices, &state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(Value::text(value));
            }
            Instruction::Length => {
''',
)
replace_once(
    vm,
    '''            Instruction::TypesOf => {
''',
    '''            Instruction::Block { argument_count } => {
                let count = usize::from(argument_count);
                if !(count == 2 || (3..=6).contains(&count))
                    || frames[frame_index].stack.len() < count
                {
                    return Err(execution_error(
                        module,
                        &frames,
                        "invalid block builtin stack",
                    ));
                }
                let arguments = {
                    let stack = &mut frames[frame_index].stack;
                    stack.split_off(stack.len() - count)
                };
                let value = block_builtin(&arguments, &mut state.heap)
                    .map_err(|message| execution_error(module, &frames, message))?;
                frames[frame_index].stack.push(value);
            }
            Instruction::TypesOf => {
''',
)
replace_once(
    vm,
    '''fn builtin_text(value: &Value, heap: &ValueHeap, context: &str) -> Result<String, String> {
''',
    '''fn copy_text_builtin(
    arguments: &[Value],
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<String, String> {
    let source = builtin_text(&arguments[0], heap, "copytext text")?;
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = signed_text_index(arguments.get(1), 1)?;
    let end = signed_text_index(arguments.get(2), 0)?;
    let start = resolve_text_position(start, logical_length);
    let end = if end == 0 {
        logical_length.saturating_add(1)
    } else {
        resolve_text_position(end, logical_length)
    };
    if end <= start {
        return Ok(String::new());
    }
    let start = start.saturating_sub(1);
    let end = end.saturating_sub(1);
    let (start, end) = if character_indices {
        (character_offset(&source, start), character_offset(&source, end))
    } else {
        (
            previous_char_boundary(&source, start),
            previous_char_boundary(&source, end),
        )
    };
    Ok(source[start..end].to_owned())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "DM text positions are integralized from binary32 at the language boundary"
)]
fn signed_text_index(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => {
            let number = number.to_f32();
            if !number.is_finite() {
                return Ok(default);
            }
            Ok(number.trunc() as i64)
        }
        Some(value) => Err(format!("copytext bounds require a number, received {value}")),
    }
}

fn resolve_text_position(position: i64, logical_length: usize) -> usize {
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn builtin_text(value: &Value, heap: &ValueHeap, context: &str) -> Result<String, String> {
''',
)
replace_once(
    vm,
    '''/// Resolves BYOND's `range()` over the materialized headless world.
''',
    '''/// Resolves BYOND's `block()` over materialized headless turfs.
fn block_builtin(arguments: &[Value], heap: &mut ValueHeap) -> Result<Value, String> {
    let list = heap.allocate_list();
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");

    let datum_coordinates = |value: &Value, heap: &ValueHeap| -> Option<(f32, f32, f32)> {
        let Value::Datum(datum) = value else {
            return None;
        };
        let datum = heap.datum(*datum).ok()?;
        let path = datum.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            return None;
        }
        Some((
            datum.field(&x).ok()?.as_number()?,
            datum.field(&y).ok()?.as_number()?,
            datum.field(&z).ok()?.as_number()?,
        ))
    };
    let numeric = |value: &Value| value.as_number().filter(|number| number.is_finite());

    let (start, end) = match arguments {
        [start, end] => {
            let Some(start) = datum_coordinates(start, heap) else {
                return Ok(Value::List(list));
            };
            let Some(end) = datum_coordinates(end, heap) else {
                return Ok(Value::List(list));
            };
            (start, end)
        }
        [start_x, start_y, start_z, rest @ ..] if rest.len() <= 3 => {
            let (Some(start_x), Some(start_y), Some(start_z)) =
                (numeric(start_x), numeric(start_y), numeric(start_z))
            else {
                return Ok(Value::List(list));
            };
            let end_x = rest
                .first()
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_x);
            let end_y = rest
                .get(1)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_y);
            let end_z = rest
                .get(2)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_z);
            ((start_x, start_y, start_z), (end_x, end_y, end_z))
        }
        _ => return Err("block requires two turfs or three through six coordinates".to_owned()),
    };

    // Accept either corner ordering while preserving the inclusive rectangular
    // volume described by the two endpoints. This is important to movement
    // code whose source/destination order naturally changes with direction.
    let low = (start.0.min(end.0), start.1.min(end.1), start.2.min(end.2));
    let high = (start.0.max(end.0), start.1.max(end.1), start.2.max(end.2));
    let matching = heap
        .datums()
        .filter_map(|(datum, candidate)| {
            let path = candidate.type_path().as_str();
            if path != "/turf" && !path.starts_with("/turf/") {
                return None;
            }
            let candidate_x = candidate.field(&x).ok()?.as_number()?;
            let candidate_y = candidate.field(&y).ok()?.as_number()?;
            let candidate_z = candidate.field(&z).ok()?.as_number()?;
            (candidate_x >= low.0
                && candidate_x <= high.0
                && candidate_y >= low.1
                && candidate_y <= high.1
                && candidate_z >= low.2
                && candidate_z <= high.2)
                .then_some(datum)
        })
        .collect::<Vec<_>>();
    let result = heap
        .list_mut(list)
        .expect("a newly allocated list handle must be live");
    for datum in matching {
        result.add(Value::Datum(datum));
    }
    Ok(Value::List(list))
}

/// Resolves BYOND's `range()` over the materialized headless world.
''',
)
replace_once(
    vm,
    '''    #[test]
    fn random_builtins_are_deterministic_and_respect_their_bounds() {
''',
    '''    #[test]
    fn repeated_nested_blocks_may_redeclare_macro_locals() {
        let source = parse(
            "/proc/repeated_scopes()\\n\\tvar/total = 0\\n\\tdo { var/_L = 1; total += _L; } while(0)\\n\\tdo { var/_L = 2; total += _L; } while(0)\\n\\treturn total\\n",
        )
        .expect("repeated scoped locals should parse");
        let module = compile_module(&source.definitions)
            .expect("nested blocks should permit repeated local names");
        let entry = module.procedure_id("/proc/repeated_scopes").expect("entry");
        assert_eq!(execute_module(&module, entry, &[]), Ok(Value::number(3.0)));
    }

    #[test]
    fn copytext_char_uses_character_positions_and_negative_offsets() {
        let source = parse(
            "/proc/middle()\\n\\treturn copytext_char(\"AéB\", 2, 3)\\n/proc/tail()\\n\\treturn copytext_char(\"Hi there\", -5)\\n",
        )
        .expect("copytext_char source should parse");
        let module = compile_module(&source.definitions).expect("copytext_char should compile");
        let middle = module.procedure_id("/proc/middle").expect("middle");
        let tail = module.procedure_id("/proc/tail").expect("tail");
        assert_eq!(execute_module(&module, middle, &[]), Ok(Value::text("é")));
        assert_eq!(execute_module(&module, tail, &[]), Ok(Value::text("there")));
    }

    #[test]
    fn block_enumerates_inclusive_turf_rectangles() {
        let source = parse("/proc/box(start, finish)\\n\\treturn block(start, finish)\\n")
            .expect("block source should parse");
        let module = compile_module(&source.definitions).expect("block should compile");
        let entry = module.procedure_id("/proc/box").expect("box");
        let mut state = ExecutionState::new();
        let turf_path = TypePath::parse("/turf/test").expect("turf path");
        let mut turfs = Vec::new();
        for (x_value, y_value) in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)] {
            let turf = state.heap_mut().allocate_datum(turf_path.clone());
            state.heap_mut().set_datum_field(turf, field("x"), Value::number(x_value)).unwrap();
            state.heap_mut().set_datum_field(turf, field("y"), Value::number(y_value)).unwrap();
            state.heap_mut().set_datum_field(turf, field("z"), Value::number(1.0)).unwrap();
            turfs.push(turf);
        }
        let result = execute_module_in_state(
            &module,
            entry,
            &[Value::Datum(turfs[3]), Value::Datum(turfs[0])],
            &mut state,
        )
        .expect("block should execute");
        let Value::List(list) = result else {
            panic!("block should return a list");
        };
        assert_eq!(state.heap().list(list).expect("block list").len(), 4);
    }

    #[test]
    fn random_builtins_are_deterministic_and_respect_their_bounds() {
''',
)
