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

# Fix the combined StoreField arm without moving the instruction payload before
# checking whether this is the keep-value variant.
replace_once(
    vm,
    '''            Instruction::StoreField(name) | Instruction::StoreFieldKeep(name) => {
                let keep = matches!(instruction, Instruction::StoreFieldKeep(_));
''',
    '''            Instruction::StoreField(ref name) | Instruction::StoreFieldKeep(ref name) => {
                let keep = matches!(instruction, Instruction::StoreFieldKeep(_));
''',
)
replace_once(
    vm,
    '''                if let Err(error) = state.heap.set_datum_field(datum, name, value.clone()) {
''',
    '''                if let Err(error) = state
                    .heap
                    .set_datum_field(datum, name.clone(), value.clone())
                {
''',
)

# Append numeric min/max to the actual current standard builtin table.
p = Path(sem)
text = p.read_text()
anchor = r'''    "\treturn output\n",
);
const STANDARD_BUILTIN_NAMES: [&str; 6] =
    ["isarea", "ismob", "isobj", "get_dir", "istext", "orange"];
'''
replacement = r'''    "\treturn output\n",
    "/proc/min(...)\n",
    "\tvar/list/values = args\n",
    "\tif(length(args) == 1 && islist(args[1]))\n",
    "\t\tvalues = args[1]\n",
    "\tif(!length(values))\n",
    "\t\treturn null\n",
    "\tvar/result = values[1]\n",
    "\tfor(var/value in values)\n",
    "\t\tif(value < result)\n",
    "\t\t\tresult = value\n",
    "\treturn result\n",
    "/proc/max(...)\n",
    "\tvar/list/values = args\n",
    "\tif(length(args) == 1 && islist(args[1]))\n",
    "\t\tvalues = args[1]\n",
    "\tif(!length(values))\n",
    "\t\treturn null\n",
    "\tvar/result = values[1]\n",
    "\tfor(var/value in values)\n",
    "\t\tif(value > result)\n",
    "\t\t\tresult = value\n",
    "\treturn result\n",
);
const STANDARD_BUILTIN_NAMES: [&str; 8] = [
    "isarea", "ismob", "isobj", "get_dir", "istext", "orange", "min", "max",
];
'''
if text.count(anchor) != 1:
    raise SystemExit(f"standard builtin anchor expected once, found {text.count(anchor)}")
p.write_text(text.replace(anchor, replacement, 1))

# Add regressions that were after the failed standard-builtin patch in the
# original script and therefore were not reached.
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
if tests not in text:
    if text.count(anchor) != 1:
        raise SystemExit(f"test anchor expected once, found {text.count(anchor)}")
    text = text.replace(anchor, tests + anchor, 1)
p.write_text(text)
