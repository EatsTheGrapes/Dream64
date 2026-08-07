from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}")
    file.write_text(text.replace(old, new, 1))


# __FILE__ is a predefined compiler macro, including when referenced from
# inside another macro replacement.
project = "crates/dm-project/src/lib.rs"
replace_once(
    project,
    '''    fn append_expanded_source(\n        &mut self,\n        source: &str,\n        span: SourceSpan,\n        macros: &HashMap<String, MacroDefinition>,\n        path: &Path,\n    ) -> Result<(), ProjectError> {\n        let text = &source[span.start..span.end];\n''',
    '''    fn append_expanded_source(\n        &mut self,\n        source: &str,\n        span: SourceSpan,\n        macros: &HashMap<String, MacroDefinition>,\n        path: &Path,\n    ) -> Result<(), ProjectError> {\n        let file_macro = path.to_string_lossy().replace('\\\\', "/");\n        let text = &source[span.start..span.end];\n''',
)
replace_once(
    project,
    '''                let name = &text[offset..identifier_end];\n                if let Some(definition) = macros.get(name) {\n''',
    '''                let name = &text[offset..identifier_end];\n                if name == "__FILE__" {\n                    self.append_original(\n                        source,\n                        SourceSpan::new(span.start + literal_start, span.start + offset),\n                    );\n                    let invocation =\n                        SourceSpan::new(span.start + offset, span.start + identifier_end);\n                    self.append_replacement(&format!("{file_macro:?}"), invocation);\n                    offset = identifier_end;\n                    literal_start = offset;\n                    continue;\n                }\n                if let Some(definition) = macros.get(name) {\n''',
)
replace_once(
    project,
    '''                        expand_macro(name, arguments.as_deref(), macros, &mut Vec::new()).map_err(\n                            |message| ProjectError::MacroExpansion {\n''',
    '''                        expand_macro(\n                            name,\n                            arguments.as_deref(),\n                            macros,\n                            &mut Vec::new(),\n                            &file_macro,\n                        )\n                        .map_err(|message| ProjectError::MacroExpansion {\n''',
)
replace_once(
    project,
    '''                                message,\n                            },\n                        )?;\n''',
    '''                                message,\n                            })?;\n''',
)
replace_once(
    project,
    '''fn expand_macro(\n    name: &str,\n    arguments: Option<&[String]>,\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n) -> Result<String, String> {\n''',
    '''fn expand_macro(\n    name: &str,\n    arguments: Option<&[String]>,\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n    file_macro: &str,\n) -> Result<String, String> {\n    if name == "__FILE__" {\n        return Ok(format!("{file_macro:?}"));\n    }\n''',
)
replace_once(
    project,
    '''                substitute_function_macro(name, definition, parameters, arguments, macros, stack)\n''',
    '''                substitute_function_macro(\n                    name,\n                    definition,\n                    parameters,\n                    arguments,\n                    macros,\n                    stack,\n                    file_macro,\n                )\n''',
)
replace_once(
    project,
    '''        expand_replacement(&definition.replacement, macros, stack)\n''',
    '''        expand_replacement(&definition.replacement, macros, stack, file_macro)\n''',
)
replace_once(
    project,
    '''fn expand_replacement(\n    replacement: &str,\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n) -> Result<String, String> {\n''',
    '''fn expand_replacement(\n    replacement: &str,\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n    file_macro: &str,\n) -> Result<String, String> {\n''',
)
replace_once(
    project,
    '''            let name = &replacement[offset..end];\n            if let Some(definition) = macros.get(name) {\n''',
    '''            let name = &replacement[offset..end];\n            if name == "__FILE__" {\n                output.push_str(&format!("{file_macro:?}"));\n                offset = end;\n                continue;\n            }\n            if let Some(definition) = macros.get(name) {\n''',
)
text = Path(project).read_text()
text = text.replace(
    "expand_macro(name, Some(&arguments), macros, stack)?",
    "expand_macro(name, Some(&arguments), macros, stack, file_macro)?",
)
text = text.replace(
    "expand_macro(name, None, macros, stack)?",
    "expand_macro(name, None, macros, stack, file_macro)?",
)
Path(project).write_text(text)
replace_once(
    project,
    '''fn substitute_function_macro(\n    name: &str,\n    definition: &MacroDefinition,\n    parameters: &MacroParameters,\n    arguments: &[String],\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n) -> Result<String, String> {\n''',
    '''fn substitute_function_macro(\n    name: &str,\n    definition: &MacroDefinition,\n    parameters: &MacroParameters,\n    arguments: &[String],\n    macros: &HashMap<String, MacroDefinition>,\n    stack: &mut Vec<String>,\n    file_macro: &str,\n) -> Result<String, String> {\n''',
)
replace_once(
    project,
    '''    expand_replacement(&substituted, macros, stack)\n}\n''',
    '''    expand_replacement(&substituted, macros, stack, file_macro)\n}\n''',
)
marker = '''    #[test]\n    fn shares_defines_across_recursive_includes() {\n'''
test = '''    #[test]\n    fn expands_predefined_file_macro_inside_user_macros() {\n        let scratch = ScratchDirectory::new();\n        let source = concat!(\n            "#define SOURCE_FILE __FILE__\\n",\n            "/proc/source_file()\\n",\n            "\\treturn SOURCE_FILE\\n",\n        );\n        fs::write(scratch.path().join("world.dme"), source)\n            .expect("file macro fixture should be written");\n\n        let project = Project::load(scratch.path().join("world.dme"))\n            .expect("predefined file macro should expand");\n        let expanded = project.files[0]\n            .compiler_text()\n            .expect("expanded source should remain UTF-8");\n\n        assert!(!expanded.contains("__FILE__"));\n        assert!(!expanded.contains("SOURCE_FILE"));\n        assert!(expanded.contains("world.dme"));\n    }\n\n'''
replace_once(project, marker, test + marker)

semantics = "crates/dm-semantics/src/lib.rs"
old_builtin = '''const STANDARD_LOCATION_BUILTINS: &str = concat!(\n    "/proc/isarea(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /area))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n    "/proc/ismob(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /mob))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n    "/proc/isobj(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /obj))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n);\nconst STANDARD_LOCATION_BUILTIN_NAMES: [&str; 3] = ["isarea", "ismob", "isobj"];\n'''
new_builtin = '''const STANDARD_BUILTINS: &str = concat!(\n    "/proc/isarea(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /area))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n    "/proc/ismob(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /mob))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n    "/proc/isobj(...)\\n",\n    "\\tfor(var/location in args)\\n",\n    "\\t\\tif(!istype(location, /obj))\\n",\n    "\\t\\t\\treturn 0\\n",\n    "\\treturn 1\\n",\n    "/proc/get_dir(reference, target)\\n",\n    "\\tif(!istype(reference, /atom) || !istype(target, /atom))\\n",\n    "\\t\\treturn 0\\n",\n    "\\tvar/direction = 0\\n",\n    "\\tif(target.y > reference.y)\\n",\n    "\\t\\tdirection |= 1\\n",\n    "\\telse if(target.y < reference.y)\\n",\n    "\\t\\tdirection |= 2\\n",\n    "\\tif(target.x > reference.x)\\n",\n    "\\t\\tdirection |= 4\\n",\n    "\\telse if(target.x < reference.x)\\n",\n    "\\t\\tdirection |= 8\\n",\n    "\\treturn direction\\n",\n    "/proc/istext(value)\\n",\n    "\\treturn !isnull(value) && !isnum(value) && !ispath(value) && !islist(value) && !istype(value)\\n",\n    "/proc/orange(first, second = usr)\\n",\n    "\\tvar/distance\\n",\n    "\\tvar/center\\n",\n    "\\tif(isnum(first))\\n",\n    "\\t\\tdistance = first\\n",\n    "\\t\\tcenter = second\\n",\n    "\\telse\\n",\n    "\\t\\tcenter = first\\n",\n    "\\t\\tdistance = second\\n",\n    "\\tvar/output = list()\\n",\n    "\\tfor(var/atom/candidate in range(distance, center))\\n",\n    "\\t\\tif(candidate == center || candidate.loc == center)\\n",\n    "\\t\\t\\tcontinue\\n",\n    "\\t\\toutput[length(output) + 1] = candidate\\n",\n    "\\treturn output\\n",\n);\nconst STANDARD_BUILTIN_NAMES: [&str; 6] =\n    ["isarea", "ismob", "isobj", "get_dir", "istext", "orange"];\n'''
replace_once(semantics, old_builtin, new_builtin)
text = Path(semantics).read_text()
text = text.replace("STANDARD_LOCATION_BUILTINS", "STANDARD_BUILTINS")
text = text.replace("STANDARD_LOCATION_BUILTIN_NAMES", "STANDARD_BUILTIN_NAMES")
Path(semantics).write_text(text)
replace_once(semantics, '''        "/datum" => &["type"],\n''', '''        "/datum" => &["tag", "type"],\n''')
replace_once(
    semantics,
    '''    fn lowers_standard_datum_type_field_for_all_datums() {\n        let compilation = TestProject::compile("/datum/example\\n\\tproc/read()\\n\\t\\treturn type\\n");\n''',
    '''    fn lowers_standard_datum_type_field_for_all_datums() {\n        let compilation = TestProject::compile(\n            "/datum/example\\n\\tproc/read()\\n\\t\\treturn list(type, tag)\\n",\n        );\n''',
)
marker = '''    #[test]\n    fn selected_dynamic_literal_call_includes_matching_method_implementation() {\n'''
test = '''    #[test]\n    fn links_direction_text_and_orange_standard_builtins() {\n        let compilation = TestProject::compile(concat!(\n            "/proc/classify()\\n",\n            "\\treturn istext(\\\"hello\\\") + istext(3)\\n",\n            "/atom/example/proc/neighbors(other)\\n",\n            "\\treturn get_dir(src, other) + length(orange(1, src))\\n",\n        ));\n        let registry = ProcedureRegistry::build(&compilation);\n        registry\n            .compile_vm(&compilation)\n            .expect("standard direction/text/orange builtins should link");\n        assert_eq!(\n            execute_effective(&compilation, "/proc/classify", &[]),\n            Ok(Value::number(1.0))\n        );\n    }\n\n'''
replace_once(semantics, marker, test + marker)

runtime = "crates/dm-runtime/src/lib.rs"
replace_once(
    runtime,
    '''fn materialize_builtin_atom_defaults(\n    heap: &mut ValueHeap,\n    datum: DatumId,\n    is_atom: bool,\n    is_movable: bool,\n) -> Result<(), ValueError> {\n    if !is_atom {\n        return Ok(());\n    }\n''',
    '''fn materialize_builtin_atom_defaults(\n    heap: &mut ValueHeap,\n    datum: DatumId,\n    is_atom: bool,\n    is_movable: bool,\n) -> Result<(), ValueError> {\n    // Every /datum has BYOND's built-in tag field, even though it has no\n    // source declaration in user projects.\n    let tag = FieldName::parse("tag").expect("built-in datum field name is valid");\n    if heap.datum_field(datum, &tag).is_err() {\n        heap.set_datum_field(datum, tag, Value::Null)?;\n    }\n    if !is_atom {\n        return Ok(());\n    }\n''',
)
