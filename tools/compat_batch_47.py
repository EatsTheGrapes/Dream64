from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}\n---OLD---\n{old}")
    p.write_text(text.replace(old, new, 1))

vm = "crates/dm-vm/src/lib.rs"
sem = "crates/dm-semantics/src/lib.rs"

# Preserve assignment-expression values when writing datum fields.
replace_once(
    vm,
    "    /// Pops a value and datum receiver, then writes one named field.\n    StoreField(FieldName),\n",
    "    /// Pops a value and datum receiver, then writes one named field.\n    StoreField(FieldName),\n    /// Stores one datum field while preserving the assigned value on the stack.\n    StoreFieldKeep(FieldName),\n",
)
replace_once(
    vm,
    '''            Instruction::StoreField(name) => {
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let datum = match datum_receiver(&receiver, "field write") {
                    Ok(datum) => datum,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Err(error) = state.heap.set_datum_field(datum, name, value) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
            }
''',
    '''            Instruction::StoreField(name) | Instruction::StoreFieldKeep(name) => {
                let keep = matches!(instruction, Instruction::StoreFieldKeep(_));
                let value = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let receiver = match pop(&mut frames[frame_index].stack) {
                    Ok(value) => value,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                let datum = match datum_receiver(&receiver, "field write") {
                    Ok(datum) => datum,
                    Err(message) => return Err(execution_error(module, &frames, message)),
                };
                if let Err(error) = state.heap.set_datum_field(datum, name, value.clone()) {
                    return Err(execution_error(module, &frames, error.to_string()));
                }
                if keep {
                    frames[frame_index].stack.push(value);
                }
            }
''',
)
replace_once(
    vm,
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
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreField(name.clone()));
        }
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
''',
)

# BYOND direction constants used directly by atmos/movement code.
replace_once(
    vm,
    '''        "FALSE" | "BLEND_DEFAULT" => Some(0.0),
        "TRUE" | "BLEND_OVERLAY" | "KEEP_TOGETHER" => Some(1.0),
        "BLEND_ADD" | "KEEP_APART" => Some(2.0),
        "BLEND_SUBTRACT" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" => Some(4.0),
        "BLEND_INSET_OVERLAY" => Some(5.0),
''',
    '''        "FALSE" | "BLEND_DEFAULT" => Some(0.0),
        "TRUE" | "BLEND_OVERLAY" | "KEEP_TOGETHER" | "NORTH" => Some(1.0),
        "BLEND_ADD" | "KEEP_APART" | "SOUTH" => Some(2.0),
        "BLEND_SUBTRACT" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" | "EAST" => Some(4.0),
        "BLEND_INSET_OVERLAY" | "NORTHEAST" => Some(5.0),
        "SOUTHEAST" => Some(6.0),
        "WEST" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" => Some(16.0),
        "DOWN" => Some(32.0),
''',
)

# Lazy logical assignment operators. Lower through the existing conditional and
# plain-assignment machinery; this preserves short-circuiting for the ordinary
# local/field/list targets used by tgstation/Monkestation helpers.
old_ops = '"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>="'
new_ops = '"=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" | "&&=" | "||="'
p = Path(vm)
text = p.read_text()
if text.count(old_ops) < 2:
    raise SystemExit(f"expected assignment operator catalog at least twice, found {text.count(old_ops)}")
text = text.replace(old_ops, new_ops)
p.write_text(text)

replace_once(
    vm,
    '''        let operator = operator.clone();
        self.index += 1;
        let value = self.parse_assignment()?;
        Ok(Expression::Assignment {
            target: Box::new(target),
            operator,
            value: Box::new(value),
        })
''',
    '''        let operator = operator.clone();
        self.index += 1;
        let value = self.parse_assignment()?;
        if operator == "||=" {
            let assignment = Expression::Assignment {
                target: Box::new(target.clone()),
                operator: "=".to_owned(),
                value: Box::new(value),
            };
            return Ok(Expression::Conditional {
                condition: Box::new(target.clone()),
                when_true: Box::new(target),
                when_false: Box::new(assignment),
            });
        }
        if operator == "&&=" {
            let assignment = Expression::Assignment {
                target: Box::new(target.clone()),
                operator: "=".to_owned(),
                value: Box::new(value),
            };
            return Ok(Expression::Conditional {
                condition: Box::new(target.clone()),
                when_true: Box::new(assignment),
                when_false: Box::new(target),
            });
        }
        Ok(Expression::Assignment {
            target: Box::new(target),
            operator,
            value: Box::new(value),
        })
''',
)

# Standalone ||= / &&= statements go through expression lowering so they share
# the short-circuit behavior above instead of numeric compound assignment.
replace_once(
    vm,
    '''    let (assignment, operator) = top_level_assignment(tokens)
        .ok_or_else(|| compile_error("assignment statement requires '='"))?;
    if assignment == 0 || assignment + 1 == tokens.len() {
''',
    '''    let (assignment, operator) = top_level_assignment(tokens)
        .ok_or_else(|| compile_error("assignment statement requires '='"))?;
    if matches!(operator, "||=" | "&&=") {
        compile_expression(tokens, locals, instructions, procedures)?;
        instructions.push(Instruction::Pop);
        return Ok(());
    }
    if assignment == 0 || assignment + 1 == tokens.len() {
''',
)

# isicon() classifier. Dream64 currently preserves resource literal paths as
# text in headless mode, so .dmi resources participate alongside /icon datums.
replace_once(
    vm,
    '''    /// Whether every value is a valid DM location (an atom).
    IsLoc,
    /// Whether a datum or type path belongs to an optional type hierarchy.
    IsType,
''',
    '''    /// Whether every value is a valid DM location (an atom).
    IsLoc,
    /// Whether the value is an icon datum or a headless icon resource.
    IsIcon,
    /// Whether a datum or type path belongs to an optional type hierarchy.
    IsType,
''',
)
replace_once(
    vm,
    '''        "isturf" => Some(TypePredicateKind::IsTurf),
        "isloc" => Some(TypePredicateKind::IsLoc),
        "istype" => Some(TypePredicateKind::IsType),
''',
    '''        "isturf" => Some(TypePredicateKind::IsTurf),
        "isloc" => Some(TypePredicateKind::IsLoc),
        "isicon" => Some(TypePredicateKind::IsIcon),
        "istype" => Some(TypePredicateKind::IsType),
''',
)
replace_once(
    vm,
    '''        TypePredicateKind::IsLoc => Ok(arguments.iter().all(|value| {
''',
    '''        TypePredicateKind::IsIcon => match value {
            Value::Text(text) => Ok(text.to_ascii_lowercase().ends_with(".dmi")),
            Value::Datum(datum) => {
                let path = heap
                    .datum(*datum)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                Ok(path == "/icon" || path.starts_with("/icon/"))
            }
            _ => Ok(false),
        },
        TypePredicateKind::IsLoc => Ok(arguments.iter().all(|value| {
''',
)

# Numeric min/max builtins. This covers the scalar/list numeric forms exercised
# by movement and lifecycle code and preserves BYOND's single-list argument form.
replace_once(
    sem,
    '''    "\treturn output\\n",
);
const STANDARD_BUILTIN_NAMES: [&str; 6] =
    ["isarea", "ismob", "isobj", "get_dir", "istext", "orange"];
''',
    '''    "\treturn output\\n",
    "/proc/min(...)\\n",
    "\tvar/list/values = args\\n",
    "\tif(length(args) == 1 && islist(args[1]))\\n",
    "\t\tvalues = args[1]\\n",
    "\tif(!length(values))\\n",
    "\t\treturn null\\n",
    "\tvar/result = values[1]\\n",
    "\tfor(var/value in values)\\n",
    "\t\tif(value < result)\\n",
    "\t\t\tresult = value\\n",
    "\treturn result\\n",
    "/proc/max(...)\\n",
    "\tvar/list/values = args\\n",
    "\tif(length(args) == 1 && islist(args[1]))\\n",
    "\t\tvalues = args[1]\\n",
    "\tif(!length(values))\\n",
    "\t\treturn null\\n",
    "\tvar/result = values[1]\\n",
    "\tfor(var/value in values)\\n",
    "\t\tif(value > result)\\n",
    "\t\t\tresult = value\\n",
    "\treturn result\\n",
);
const STANDARD_BUILTIN_NAMES: [&str; 8] = [
    "isarea", "ismob", "isobj", "get_dir", "istext", "orange", "min", "max",
];
''',
)

# Regression tests for the newly exposed Monkestation shapes.
p = Path(vm)
text = p.read_text()
anchor = '''    #[test]\n    fn shared_value_migration_preserves_scalar_execution() {\n'''
tests = r'''    #[test]
    fn logical_assignment_short_circuits_locals_fields_and_list_entries() {
        let source = parse(
            "/datum/example/proc/run()\n\tvar/local\n\tlocal ||= 3\n\tvar/list/values = list()\n\tvalues[1] ||= 4\n\tsrc.flag ||= 5\n\treturn local + values[1] + src.flag\n",
        )
        .expect("logical assignment source should parse");
        let module = compile_module_specs(&[ProcedureSpec {
            path: "/datum/example/proc/run@0".to_owned(),
            definition: &source.definitions[0],
            parent: None,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::from([("flag".to_owned(), field("flag"))]),
            global_fields: BTreeMap::new(),
        }])
        .expect("logical assignments should compile");
        let entry = module.procedure_id_at(0).expect("entry");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(datum, field("flag"), Value::Null)
            .unwrap();
        assert_eq!(
            execute_module_in_context(
                &module,
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(12.0))
        );
    }

    #[test]
    fn plane_macro_nested_scope_keeps_cached_locals_visible() {
        let source = parse(
            "/proc/plane_macro(flag, other)\n\tvar/output = 0\n\tdo { if(flag) { var/_cached_plane = 7; var/_our_turf = other; if(_our_turf) { output = _cached_plane; } else if(other) { output = _cached_plane; } else { output = _cached_plane; } } else { output = 2; } } while(0)\n\treturn output\n",
        )
        .expect("plane macro source should parse");
        let module = compile_module(&source.definitions).expect("plane macro scope should compile");
        let entry = module.procedure_id("/proc/plane_macro").expect("entry");
        assert_eq!(
            execute_module(&module, entry, &[Value::number(1.0), Value::number(1.0)]),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn direction_and_icon_builtins_cover_lifecycle_shapes() {
        let source = parse(
            "/proc/directions()\n\treturn NORTH + SOUTH + EAST + WEST + NORTHEAST + NORTHWEST + SOUTHEAST + SOUTHWEST\n/proc/icon_resource()\n\treturn isicon('icons/test.dmi')\n",
        )
        .expect("builtin source should parse");
        let module = compile_module(&source.definitions).expect("builtins should compile");
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/directions").unwrap(), &[]),
            Ok(Value::number(45.0))
        );
        assert_eq!(
            execute_module(&module, module.procedure_id("/proc/icon_resource").unwrap(), &[]),
            Ok(Value::number(1.0))
        );
    }

'''
if text.count(anchor) != 1:
    raise SystemExit(f"test anchor expected once, found {text.count(anchor)}")
text = text.replace(anchor, tests + anchor, 1)
p.write_text(text)
