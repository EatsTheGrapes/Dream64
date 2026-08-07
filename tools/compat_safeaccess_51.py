from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}\n---OLD---\n{old[:1200]}")
    return text.replace(old, new, 1)


# Lexer: BYOND 512 added both ?. and ?: null-conditional member operators.
p = Path("crates/dm-lexer/src/lib.rs")
text = p.read_text()
old = '''            "<<=", ">>=", "&&=", "||=", "**=", "...", "::", "?.", "?[", "==", "!=", "<=", ">=",
'''
new = '''            "<<=", ">>=", "&&=", "||=", "**=", "...", "::", "?.", "?:", "?[", "==", "!=", "<=", ">=",
'''
text = replace_once(text, old, new, "lexer null-conditional colon")

# Add a small lexical regression before the existing operator-focused test tail.
anchor = '''    #[test]
    fn separates_arithmetic_operators_from_numbers() {
'''
test = '''    #[test]
    fn retains_null_conditional_member_and_index_operators() {
        let tokens = lex("value?.field value?:dynamic values?[key]\\n")
            .expect("null-conditional operators should lex");
        let operators: Vec<_> = tokens
            .into_iter()
            .filter_map(|token| match token.kind {
                TokenKind::Operator(operator) => Some(operator),
                _ => None,
            })
            .collect();
        assert_eq!(operators, ["?.", "?:", "?["]);
    }

'''
text = replace_once(text, anchor, test + anchor, "lexer safe access regression")
p.write_text(text)

p = Path("crates/dm-vm/src/lib.rs")
text = p.read_text()

# Bytecode branch that tests specifically for null, not generic DM falsiness.
text = replace_once(
    text,
    '''    /// Pops a condition and jumps to an absolute instruction when it is false.
    JumpIfFalse(usize),
''',
    '''    /// Pops a value and jumps when it is exactly DM `null`.
    ///
    /// Null-conditional member/index/call lowering duplicates the receiver
    /// before this instruction, leaving the original receiver as the result
    /// on the skipped path while evaluating it only once.
    JumpIfNull(usize),
    /// Pops a condition and jumps to an absolute instruction when it is false.
    JumpIfFalse(usize),
''',
    "JumpIfNull instruction",
)

# Preserve safe-vs-ordinary access in the expression tree.
text = replace_once(
    text,
    '''    Field {
        receiver: Box<Self>,
        name: FieldName,
    },
    GlobalField(FieldName),
''',
    '''    Field {
        receiver: Box<Self>,
        name: FieldName,
    },
    SafeField {
        receiver: Box<Self>,
        name: FieldName,
    },
    GlobalField(FieldName),
''',
    "SafeField expression",
)
text = replace_once(
    text,
    '''    DynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    Index {
        list: Box<Self>,
        index: Box<Self>,
    },
''',
    '''    DynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    SafeDynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    Index {
        list: Box<Self>,
        index: Box<Self>,
    },
    SafeIndex {
        list: Box<Self>,
        index: Box<Self>,
    },
''',
    "safe call/index expressions",
)

# Parser: keep ordinary colon chaining equivalent to dot in this dynamic VM,
# but preserve the ? prefix for null-conditional lowering.
old = '''            let starts_list_index = matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('['))
            ) || matches!(self.current_operator(), Some("?["));
            if starts_list_index {
                self.index += 1;
                let index = self.parse_binary(1)?;
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(']'))
                ) {
                    return Err(compile_error("expected ']' after list index"));
                }
                self.index += 1;
                expression = Expression::Index {
                    list: Box::new(expression),
                    index: Box::new(index),
                };
                continue;
            }
            if matches!(self.current_operator(), Some("." | "?.")) {
                self.index += 1;
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected a field name after '.'"));
                };
                let name =
                    FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
                self.index += 1;
                expression = if matches!(expression, Expression::GlobalNamespace) {
                    Expression::GlobalField(name)
                } else {
                    Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }
                };
                continue;
            }
'''
new = '''            let safe_list_index = matches!(self.current_operator(), Some("?["));
            let starts_list_index = matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('['))
            ) || safe_list_index;
            if starts_list_index {
                self.index += 1;
                let index = self.parse_binary(1)?;
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(']'))
                ) {
                    return Err(compile_error("expected ']' after list index"));
                }
                self.index += 1;
                expression = if safe_list_index {
                    Expression::SafeIndex {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                } else {
                    Expression::Index {
                        list: Box::new(expression),
                        index: Box::new(index),
                    }
                };
                continue;
            }
            if matches!(self.current_operator(), Some("." | ":" | "?." | "?:")) {
                let safe_member = matches!(self.current_operator(), Some("?." | "?:"));
                self.index += 1;
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected a field name after member access"));
                };
                let name =
                    FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
                self.index += 1;
                expression = if matches!(expression, Expression::GlobalNamespace) {
                    Expression::GlobalField(name)
                } else if safe_member {
                    Expression::SafeField {
                        receiver: Box::new(expression),
                        name,
                    }
                } else {
                    Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }
                };
                continue;
            }
'''
text = replace_once(text, old, new, "safe access parser")

old = '''                let Expression::Field { receiver, name } = expression else {
                    break;
                };
                let arguments = self.parse_call_arguments()?;
                expression = Expression::DynamicCall {
                    target: receiver,
                    procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                    arguments,
                };
                continue;
'''
new = '''                expression = match expression {
                    Expression::Field { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::DynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                        }
                    }
                    Expression::SafeField { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::SafeDynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                        }
                    }
                    other => {
                        expression = other;
                        break;
                    }
                };
                continue;
'''
text = replace_once(text, old, new, "safe dynamic call parser")

# Expression lowering helpers use a direct patch rather than generic truthy
# branches so receiver false values such as 0 do not masquerade as null.
text = replace_once(
    text,
    '''        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadField(name.clone()));
        }
        Expression::GlobalField(name) => {
''',
    '''        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::LoadField(name.clone()));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            instructions.push(Instruction::LoadField(name.clone()));
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::GlobalField(name) => {
''',
    "emit SafeField",
)
text = replace_once(
    text,
    '''        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic { argument_count });
        }
        Expression::List(entries) => {
''',
    '''        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic { argument_count });
        }
        Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(procedure, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic { argument_count });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::List(entries) => {
''',
    "emit SafeDynamicCall",
)
text = replace_once(
    text,
    '''        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
        }
        Expression::Unary { operator, operand } => {
''',
    '''        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
        }
        Expression::SafeIndex { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::IndexList);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Unary { operator, operand } => {
''',
    "emit SafeIndex",
)

# Safe assignment expressions: skip both the read and RHS when null, leaving
# null as the expression result on that path.
text = replace_once(
    text,
    '''        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
        }
        Expression::Index { list, index } => {
''',
    '''        Expression::Field { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
        }
        Expression::SafeField { receiver, name } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            if operator != "=" {
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::LoadField(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::StoreFieldKeep(name.clone()));
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::Index { list, index } => {
''',
    "safe field assignment expression",
)
text = replace_once(
    text,
    '''        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            if operator == "=" {
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::SetListIndexKeep);
            } else {
                // CompoundListIndex consumes the list, key, and right operand
                // and leaves no value, so retain an independent copy of the
                // computed result is not possible without a temporary. Keep
                // compound assignment expressions explicit until the VM has a
                // value-preserving variant.
                return Err(compile_error(
                    "compound list assignment is not supported as an expression",
                ));
            }
        }
        _ => return Err(compile_error("assignment target is not writable")),
''',
    '''        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            if operator == "=" {
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::SetListIndexKeep);
            } else {
                // CompoundListIndex consumes the list, key, and right operand
                // and leaves no value, so retain an independent copy of the
                // computed result is not possible without a temporary. Keep
                // compound assignment expressions explicit until the VM has a
                // value-preserving variant.
                return Err(compile_error(
                    "compound list assignment is not supported as an expression",
                ));
            }
        }
        Expression::SafeIndex { list, index } => {
            if operator != "=" {
                return Err(compile_error(
                    "compound null-conditional list assignment is not supported as an expression",
                ));
            }
            emit_expression(list, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            emit_expression(index, locals, instructions, procedures)?;
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::SetListIndexKeep);
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        _ => return Err(compile_error("assignment target is not writable")),
''',
    "safe index assignment expression",
)

# Statement assignment requires an explicit cleanup branch because normal
# assignments leave no value on the stack, while the skipped safe path still
# holds the original null receiver.
text = replace_once(
    text,
    '''        Expression::Index { list, index } => {
            emit_expression(&list, locals, instructions, procedures)?;
            emit_expression(&index, locals, instructions, procedures)?;
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
                instructions.push(Instruction::CompoundListIndex(
                    compound_list_index_operator(operator)?,
                ));
            }
        }
        Expression::Field { receiver, name } => {
''',
    '''        Expression::Index { list, index } => {
            emit_expression(&list, locals, instructions, procedures)?;
            emit_expression(&index, locals, instructions, procedures)?;
            compile_expression(&tokens[assignment + 1..], locals, instructions, procedures)?;
            if operator == "=" {
                instructions.push(Instruction::SetListIndex);
            } else {
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
''',
    "safe index assignment statement",
)
text = replace_once(
    text,
    '''        Expression::Field { receiver, name } => {
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
        Expression::GlobalField(name) => {
''',
    '''        Expression::Field { receiver, name } => {
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
''',
    "safe field assignment statement",
)

# Initializer binding recursively traverses safe access just like ordinary
# member/index/call nodes.
text = replace_once(
    text,
    '''        Expression::Field { receiver, .. } => {
            bind_initializer_expression(receiver, bindings)?;
        }
''',
    '''        Expression::Field { receiver, .. } | Expression::SafeField { receiver, .. } => {
            bind_initializer_expression(receiver, bindings)?;
        }
''',
    "initializer safe field traversal",
)
text = replace_once(
    text,
    '''        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(procedure, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
''',
    '''        Expression::DynamicCall {
            target,
            procedure,
            arguments,
        }
        | Expression::SafeDynamicCall {
            target,
            procedure,
            arguments,
        } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(procedure, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
''',
    "initializer safe call traversal",
)
text = replace_once(
    text,
    '''        Expression::Index { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
''',
    '''        Expression::Index { list, index } | Expression::SafeIndex { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
''',
    "initializer safe index traversal",
)

# Runtime null-only branch.
text = replace_once(
    text,
    '''            Instruction::JumpIfFalse(target) => {
''',
    '''            Instruction::JumpIfNull(target) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if matches!(value, Value::Null) {
                    if let Err(message) = validate_jump(target, program.instructions.len()) {
                        return Err(execution_error(module, &frames, message));
                    }
                    frames[frame_index].instruction = target;
                    continue;
                }
            }
            Instruction::JumpIfFalse(target) => {
''',
    "execute JumpIfNull",
)

# Regression tests: reads, index reads, calls, chaining colon, and writes all
# short-circuit. The side-effect counter proves RHS/argument expressions are
# not evaluated when the receiver is null.
test_anchor = '''    #[test]
    fn dotted_datum_calls_lower_to_dynamic_dispatch() {
'''
tests = r'''    #[test]
    fn null_conditional_field_index_and_call_short_circuit_without_rhs_evaluation() {
        let source = parse(
            "/datum/example/proc/read(value, list/values)\n\tvar/a = value?.field\n\tvar/b = values?[bump()]\n\tvar/c = value?:take(bump())\n\tvalue?.field = bump()\n\tvalues?[bump()] = bump()\n\treturn isnull(a) + isnull(b) + isnull(c) + GLOB.calls\n/datum/example/proc/take(value)\n\treturn value\n/proc/bump()\n\tGLOB.calls += 1\n\treturn 1\n",
        )
        .expect("null-conditional source should parse");
        let mut specs = Vec::new();
        specs.push(ProcedureSpec {
            path: "/datum/example/proc/read@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::from([("bump".to_owned(), ProcedureId(2))]),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        specs.push(ProcedureSpec {
            path: "/datum/example/proc/take@0".to_owned(),
            definition: &source.definitions[1],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        specs.push(ProcedureSpec {
            path: "/proc/bump@0".to_owned(),
            definition: &source.definitions[2],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::from([("calls".to_owned(), field("calls"))]),
        });
        let module = compile_module_specs(&specs).expect("null-conditional source should compile");
        let mut state = ExecutionState::new();
        state.set_global(field("calls"), Value::number(0.0));
        assert_eq!(
            execute_module_in_state(
                &module,
                module.procedure_id_at(0).expect("read entry"),
                &[Value::Null, Value::Null],
                &mut state,
            ),
            Ok(Value::number(3.0))
        );
        assert_eq!(state.global(&field("calls")), Some(&Value::number(0.0)));
    }

    #[test]
    fn null_conditional_access_executes_normally_for_live_receivers() {
        let source = parse(
            "/datum/example/proc/read(list/values)\n\tvar/a = src?.field\n\tvar/b = values?[1]\n\treturn a + b\n",
        )
        .expect("live null-conditional source should parse");
        let module = compile_module_specs(&[ProcedureSpec {
            path: "/datum/example/proc/read@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("field".to_owned(), field("field"))]),
            global_fields: BTreeMap::new(),
        }])
        .expect("live null-conditional source should compile");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(datum, field("field"), Value::number(4.0))
            .unwrap();
        let list = state.heap_mut().allocate_list();
        state.heap_mut().list_mut(list).unwrap().add(Value::number(5.0));
        assert_eq!(
            execute_module_in_context(
                &module,
                module.procedure_id_at(0).expect("read entry"),
                &[Value::List(list)],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

'''
text = replace_once(text, test_anchor, tests + test_anchor, "safe access regressions")
p.write_text(text)
