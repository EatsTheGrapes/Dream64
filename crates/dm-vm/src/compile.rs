//! DM source -> portable bytecode compiler.
//!
//! This module owns lowering parsed procedure/initializer definitions and the
//! local statement/expression grammar into the portable `Instruction` stream
//! that [`crate::bytecode`] defines and the interpreter consumes.
//!
//! It intentionally depends on [`crate::bytecode`] for the IR data shapes and
//! reaches a couple of shared types ([`crate::CompileError`],
//! [`crate::ProcedureSpec`]) and the spec catalog helper at the crate root.
//! The compiler has no execution logic itself.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, OnceLock};

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind, lex};
use dm_syntax::{Definition, DefinitionKind, SourceLine};
use dm_value::{FieldName, TypePath};

use crate::builtins::standard_builtin_arity;
use crate::bytecode::{
    CompoundAssignmentOperator, CompoundListIndexOperator, DeferredProcedure, InitializerBinding,
    InitializerCallNameIndex, InitializerCompileContext, InitializerProgram, Instruction,
    ListEntryKind, Module, ProcedureId, Program, TypePredicateKind, VerbParameterType,
    next_module_identity,
};
use crate::{
    CompileError, ProcedureSpec, TEXT_MACRO_A, TEXT_MACRO_A_UPPER, TEXT_MACRO_IMPROPER,
    TEXT_MACRO_OBJECT, TEXT_MACRO_ORDINAL, TEXT_MACRO_PLURAL, TEXT_MACRO_POSSESSIVE,
    TEXT_MACRO_POSSESSIVE_ADJECTIVE, TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER,
    TEXT_MACRO_POSSESSIVE_UPPER, TEXT_MACRO_PROPER, TEXT_MACRO_REFLEXIVE, TEXT_MACRO_ROMAN,
    TEXT_MACRO_ROMAN_UPPER, TEXT_MACRO_SUBJECT, TEXT_MACRO_SUBJECT_UPPER, TEXT_MACRO_THE,
    TEXT_MACRO_THE_UPPER, procedure_type_catalog_from_specs,
};

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

/// Lowers one variable initializer expression to existing VM bytecode.
///
/// Bare names are resolved only through `bindings`; there are no implicit or
/// fabricated built-ins. When `procedures` is supplied, unqualified calls may
/// resolve to global `/proc/name` entries already present in that module.
/// Initializer tokens retain their expanded-source span on every instruction.
///
/// # Errors
///
/// Returns [`CompileError`] for malformed syntax, unresolved identifiers or
/// calls, and expression forms that have no initializer execution context.
pub fn compile_initializer(
    tokens: &[SpannedToken],
    bindings: &BTreeMap<String, InitializerBinding>,
    procedures: Option<&Module>,
) -> Result<InitializerProgram, CompileError> {
    let mut module = procedures.cloned().unwrap_or_else(|| Module {
        identity: next_module_identity(),
        procedures: Vec::new(),
        paths: Vec::new(),
        names: HashMap::new(),
        dynamic_names: HashMap::new(),
        deferred: Arc::new(HashMap::new()),
        procedure_types: Vec::new(),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    });
    let entry = compile_initializer_into_module(tokens, bindings, &mut module)?;
    Ok(InitializerProgram { module, entry })
}

/// Appends one initializer entry point to an existing linked module without
/// cloning or recompiling its project procedures.
pub fn compile_initializer_into_module(
    tokens: &[SpannedToken],
    bindings: &BTreeMap<String, InitializerBinding>,
    module: &mut Module,
) -> Result<ProcedureId, CompileError> {
    // Appending changes procedure IDs and instruction ranges, so a previously
    // validated sidecar can no longer be trusted.
    module.clear_compact_wordcode();
    let context = initializer_compile_context(module);
    let program = compile_initializer_program(tokens, bindings, &context)?;
    append_initializer_program(module, program)
}

/// Snapshots global initializer call names once for deterministic parallel lowering.
pub fn initializer_compile_context(module: &mut Module) -> InitializerCompileContext {
    let names = Arc::clone(
        &module
            .initializer_call_names
            .get_or_insert_with(|| {
                let mut names = HashMap::new();
                for (path, procedure) in &module.dynamic_names {
                    if let Some(name) = path.strip_prefix("/proc/")
                        && !name.contains('/')
                    {
                        names.insert(
                            name.split('@').next().unwrap_or(name).to_owned(),
                            *procedure,
                        );
                    }
                }
                InitializerCallNameIndex {
                    names: Arc::new(names),
                    module_names_scanned: module.dynamic_names.len(),
                }
            })
            .names,
    );
    InitializerCompileContext { names }
}

/// Lowers one initializer without mutating its destination module.
pub fn compile_initializer_program(
    tokens: &[SpannedToken],
    bindings: &BTreeMap<String, InitializerBinding>,
    context: &InitializerCompileContext,
) -> Result<Program, CompileError> {
    let mut expression = ExpressionParser::new(tokens).parse()?;
    bind_initializer_expression(&mut expression, bindings)?;
    let mut instructions = Vec::new();
    emit_expression(
        &expression,
        &LocalTable::default(),
        &mut instructions,
        &context.names,
    )?;
    instructions.push(Instruction::Return);
    let source_span = match (tokens.first(), tokens.last()) {
        (Some(first), Some(last)) => SourceSpan::new(first.span.start, last.span.end),
        _ => return Err(compile_error("expected an initializer expression")),
    };
    specialize_local_list_iteration_headers(&mut instructions);
    Ok(Program {
        wait_for: true,
        parameter_count: 0,
        parameter_names: Vec::new(),
        verb_parameter_types: Vec::new(),
        verb_name: None,
        local_count: 0,
        source_spans: vec![source_span; instructions.len()],
        instructions,
    })
}

/// Publishes one prepared initializer program in owner-thread source order.
pub fn append_initializer_program(
    module: &mut Module,
    program: Program,
) -> Result<ProcedureId, CompileError> {
    let entry = ProcedureId::from_index(module.procedures.len())?;
    module.procedures.push(Arc::new(program));
    module.paths.push("<initializer>".to_owned());
    module
        .dynamic_names
        .insert("<initializer>".to_owned(), entry);
    Ok(entry)
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
    compile_module_with_global_fields(definitions, &BTreeMap::new())
}

/// Compiles a module with an explicit registry of bare global variable names.
/// This is useful for syntax-only consumers that retain procedure definitions
/// separately from the declaration tree while preserving strict name checks.
pub fn compile_module_with_global_fields(
    definitions: &[Definition],
    global_fields: &BTreeMap<String, FieldName>,
) -> Result<Module, CompileError> {
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

    let procedures = parallel_collect_ordered(definitions.len(), |index| {
        let definition = &definitions[index];
        compile_procedure_with_resolver_and_fields(
            definition,
            &call_names,
            &BTreeMap::new(),
            global_fields,
            &BTreeMap::new(),
        )
        .map(Arc::new)
    })?;
    Ok(Module {
        identity: next_module_identity(),
        procedures,
        dynamic_names: dynamic_name_index(&paths)?,
        paths,
        names,
        deferred: Arc::new(HashMap::new()),
        procedure_types: definitions
            .iter()
            .filter(|definition| definition.kind == DefinitionKind::Procedure)
            .filter_map(|definition| TypePath::parse(&definition.path.to_string()).ok())
            .collect(),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
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
    let global_types = vec![BTreeMap::new(); specs.len()];
    compile_module_specs_with_global_types(specs, &global_types)
}

/// Compiles procedure specs with declared global types used to infer bare
/// `new` expressions from their assignment destinations.
pub fn compile_module_specs_with_global_types(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
) -> Result<Module, CompileError> {
    if specs.len() != global_types.len() {
        return Err(compile_error(
            "procedure spec/global type table length mismatch",
        ));
    }
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
    let call_index = SpecCallIndex::build(&paths)?;

    let targets = specs
        .iter()
        .map(|spec| resolved_spec_targets(spec, &call_index))
        .collect::<Result<Vec<_>, _>>()?;
    let procedures = parallel_collect_ordered(specs.len(), |index| {
        let spec = &specs[index];
        compile_procedure_with_resolver_and_fields(
            spec.definition,
            &targets[index],
            &spec.src_fields,
            &spec.global_fields,
            &global_types[index],
        )
        .map(Arc::new)
        .map_err(|error| compile_error(format!("{}: {}", spec.path, error.message)))
    })?;
    Ok(Module {
        identity: next_module_identity(),
        procedures,
        dynamic_names: dynamic_name_index(&paths)?,
        paths,
        names,
        deferred: Arc::new(HashMap::new()),
        procedure_types: procedure_type_catalog_from_specs(specs),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    })
}

/// Symbolically links every procedure spec while compiling only the requested
/// eager indices. Deferred bodies retain stable module-local identities and
/// are lowered exactly once when execution first dispatches to them.
///
/// This is intended for genuinely dynamic DM calls whose runtime receiver
/// cannot be proven statically. Linking all candidate symbols preserves
/// virtual dispatch without making cold boot compile every same-name body.
///
/// # Errors
///
/// Returns [`CompileError`] for an invalid spec table or an eager body that
/// cannot be lowered. A deferred-body lowering failure is reported when that
/// body is first selected.
pub fn compile_module_specs_selective(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
    eager_indices: &BTreeSet<usize>,
) -> Result<Module, CompileError> {
    compile_module_specs_selective_with_errors(specs, global_types, eager_indices, &BTreeMap::new())
}

/// Selective symbolic linking with source-aware semantic failures retained on
/// deferred symbols and raised only if runtime dispatch materializes them.
pub fn compile_module_specs_selective_with_errors(
    specs: &[ProcedureSpec<'_>],
    global_types: &[BTreeMap<String, TypePath>],
    eager_indices: &BTreeSet<usize>,
    deferred_errors: &BTreeMap<usize, CompileError>,
) -> Result<Module, CompileError> {
    if specs.len() != global_types.len() {
        return Err(compile_error(
            "procedure spec/global type table length mismatch",
        ));
    }
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
    let call_index = SpecCallIndex::build(&paths)?;
    let targets = specs
        .iter()
        .map(|spec| resolved_spec_targets(spec, &call_index))
        .collect::<Result<Vec<_>, _>>()?;
    let eager_programs = parallel_collect_ordered(specs.len(), |index| {
        if !eager_indices.contains(&index) {
            return Ok(None);
        }
        let spec = &specs[index];
        compile_procedure_with_resolver_and_fields(
            spec.definition,
            &targets[index],
            &spec.src_fields,
            &spec.global_fields,
            &global_types[index],
        )
        .map(|program| Some(Arc::new(program)))
        .map_err(|error| compile_error(format!("{}: {}", spec.path, error.message)))
    })?;
    let mut procedures = Vec::with_capacity(specs.len());
    let mut deferred = HashMap::new();
    for (index, (spec, global_types)) in specs.iter().zip(global_types).enumerate() {
        if let Some(program) = eager_programs[index].clone() {
            procedures.push(program);
        } else {
            let procedure = ProcedureId::from_index(index)?;
            procedures.push(Arc::new(Program {
                wait_for: true,
                parameter_count: 0,
                parameter_names: Vec::new(),
                verb_parameter_types: Vec::new(),
                verb_name: None,
                local_count: 0,
                instructions: Vec::new(),
                source_spans: Vec::new(),
            }));
            deferred.insert(
                procedure,
                DeferredProcedure {
                    definition: Arc::new(spec.definition.clone()),
                    targets: Arc::new(targets[index].clone()),
                    src_fields: Arc::new(spec.src_fields.clone()),
                    global_fields: Arc::new(spec.global_fields.clone()),
                    global_types: Arc::new(global_types.clone()),
                    preflight_error: deferred_errors.get(&index).cloned(),
                    compiled: Arc::new(OnceLock::new()),
                },
            );
        }
    }
    Ok(Module {
        identity: next_module_identity(),
        procedures,
        dynamic_names: dynamic_name_index(&paths)?,
        paths,
        names,
        deferred: Arc::new(deferred),
        procedure_types: procedure_type_catalog_from_specs(specs),
        initializer_call_names: None,
        compact_wordcode: Default::default(),
        semantic_digests: Default::default(),
    })
}

/// Runs independent lowering jobs across the host's available CPUs while
/// publishing their results strictly in source order. Stable reassembly keeps
/// the first diagnostic and serialized module bytes independent of scheduling.
pub(crate) fn parallel_collect_ordered<T, E, F>(len: usize, compile: F) -> Result<Vec<T>, E>
where
    T: Send,
    E: Send,
    F: Fn(usize) -> Result<T, E> + Sync,
{
    const PARALLEL_THRESHOLD: usize = 64;
    let workers = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(len.max(1));
    if workers == 1 || len < PARALLEL_THRESHOLD {
        return (0..len).map(compile).collect();
    }

    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let compile = &compile;
        for worker in 0..workers {
            let sender = sender.clone();
            scope.spawn(move || {
                for index in (worker..len).step_by(workers) {
                    if sender.send((index, compile(index))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
    });
    let mut results = (0..len).map(|_| None).collect::<Vec<_>>();
    for (index, result) in receiver {
        results[index] = Some(result);
    }
    results
        .into_iter()
        .map(|result| result.expect("every lowering worker reports exactly one result"))
        .collect()
}

fn resolved_spec_targets(
    spec: &ProcedureSpec<'_>,
    call_index: &SpecCallIndex,
) -> Result<HashMap<String, ProcedureId>, CompileError> {
    let mut targets = HashMap::new();
    if let Some(parent) = spec.parent {
        targets.insert("..".to_owned(), ProcedureId::from_index(parent)?);
    }
    targets.extend(static_call_targets(
        &spec.path,
        call_index,
        referenced_static_call_names(spec.definition),
    ));
    for (selector, target) in &spec.static_calls {
        targets.insert(selector.clone(), ProcedureId::from_index(*target)?);
    }
    Ok(targets)
}

struct SpecCallIndex {
    latest_by_base_path: HashMap<String, ProcedureId>,
}

pub(crate) fn dynamic_name_index(
    paths: &[String],
) -> Result<HashMap<String, ProcedureId>, CompileError> {
    let mut index = HashMap::new();
    for (position, path) in paths.iter().enumerate() {
        let (base_path, suffix) = path
            .split_once('@')
            .map_or((path.as_str(), None), |(base, suffix)| (base, Some(suffix)));
        let procedure = ProcedureId::from_index(position)?;
        if matches!(suffix, Some("dream64_builtin" | "dream64_native")) {
            // Engine declarations are fallback surfaces. A real project body
            // at the same canonical path retains BYOND override precedence
            // for dynamic calls, even though generated specs are appended.
            index.entry(base_path.to_owned()).or_insert(procedure);
        } else {
            index.insert(base_path.to_owned(), procedure);
        }
    }
    Ok(index)
}

impl SpecCallIndex {
    fn build(paths: &[String]) -> Result<Self, CompileError> {
        let mut latest_by_base_path = HashMap::new();
        for (position, path) in paths.iter().enumerate() {
            let Some((_, _)) = path.rsplit_once("/proc/") else {
                continue;
            };
            let (base_path, suffix) = path
                .split_once('@')
                .map_or((path.as_str(), None), |(base, suffix)| (base, Some(suffix)));
            // Match the old reverse scan: the last spec for a base path wins.
            let procedure = ProcedureId::from_index(position)?;
            if matches!(suffix, Some("dream64_builtin" | "dream64_native")) {
                latest_by_base_path
                    .entry(base_path.to_owned())
                    .or_insert(procedure);
            } else {
                latest_by_base_path.insert(base_path.to_owned(), procedure);
            }
        }
        Ok(Self {
            latest_by_base_path,
        })
    }
}

fn static_call_targets(
    path: &str,
    index: &SpecCallIndex,
    selectors: impl IntoIterator<Item = String>,
) -> HashMap<String, ProcedureId> {
    let Some((owner, _)) = path.rsplit_once("/proc/") else {
        return HashMap::new();
    };
    let mut targets = HashMap::new();
    for name in selectors {
        let mut current_owner = owner;
        loop {
            let expected = if current_owner.is_empty() {
                format!("/proc/{name}")
            } else {
                format!("{current_owner}/proc/{name}")
            };
            if let Some(procedure) = index.latest_by_base_path.get(&expected) {
                targets.insert(name.clone(), *procedure);
                break;
            }
            let Some((parent, _)) = current_owner.rsplit_once('/') else {
                break;
            };
            current_owner = parent;
        }
    }
    targets
}

fn referenced_static_call_names(definition: &Definition) -> std::collections::BTreeSet<String> {
    let mut names = std::collections::BTreeSet::new();
    let token_groups = std::iter::once(&definition.header)
        .chain(
            definition
                .parameters
                .iter()
                .map(|parameter| &parameter.tokens),
        )
        .chain(definition.body.iter().map(|line| &line.tokens));
    for tokens in token_groups {
        for pair in tokens.windows(2) {
            if let [
                SpannedToken {
                    kind: TokenKind::Identifier(name),
                    ..
                },
                SpannedToken {
                    kind: TokenKind::Punctuation('('),
                    ..
                },
            ] = pair
                && !matches!(
                    name.as_str(),
                    "if" | "for" | "while" | "switch" | "catch" | "spawn" | "new"
                )
            {
                names.insert(name.clone());
            }
        }
    }
    names
}

fn compile_procedure_with_resolver(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<Program, CompileError> {
    compile_procedure_with_resolver_and_fields(
        definition,
        procedures,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &BTreeMap::new(),
    )
}

pub(crate) fn compile_procedure_with_resolver_and_fields(
    definition: &Definition,
    procedures: &HashMap<String, ProcedureId>,
    src_fields: &BTreeMap<String, FieldName>,
    global_fields: &BTreeMap<String, FieldName>,
    global_types: &BTreeMap<String, TypePath>,
) -> Result<Program, CompileError> {
    if !matches!(
        definition.kind,
        DefinitionKind::Procedure | DefinitionKind::ProcedureOverride | DefinitionKind::Verb
    ) {
        return Err(compile_error("definition is not executable"));
    }

    let mut locals = LocalTable::with_fields(src_fields, global_fields, global_types);
    for (index, parameter) in definition.parameters.iter().enumerate() {
        // BYOND permits an unnamed trailing `...` parameter.  It still
        // occupies an argument position, but cannot be referenced by name.
        if let Some(name) = parameter_name(&parameter.tokens) {
            locals.insert_parameter(name.to_owned(), to_local_index(index)?);
            if let Some(type_path) = declared_parameter_type(&parameter.tokens, name) {
                locals.set_type(name.to_owned(), type_path);
            }
        }
    }
    locals.reserve_parameter_slots(definition.parameters.len())?;
    // `args` is an implicit, per-call list in every DM procedure.  It must
    // be a local (rather than a fabricated global) so recursive and nested
    // calls retain their own complete supplied argument vectors.
    let args_slot = locals.declare("args".to_owned())?;

    let mut instructions = Vec::new();
    let mut source_spans = Vec::new();
    let mut loops = Vec::new();
    // BYOND exposes `args` as a per-call list, but materializing it eagerly in
    // every procedure turns short hot-path calls (notably map loading and
    // InitAtom) into millions of otherwise unreachable list identities. The
    // value is unobservable when the special local is never referenced, so
    // retain exact behavior while allocating it only for procedures whose
    // expanded syntax can read or write `args`.
    let uses_args = definition
        .parameters
        .iter()
        .flat_map(|parameter| parameter.tokens.iter())
        .chain(definition.body.iter().flat_map(|line| line.tokens.iter()))
        .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "args"));
    if uses_args {
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::MakeArgs,
            definition.span,
        );
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::StoreLocal(args_slot),
            definition.span,
        );
    }
    compile_parameter_defaults(
        definition,
        &locals,
        &mut instructions,
        &mut source_spans,
        procedures,
    )?;
    // The DM preprocessor is allowed to expand a macro into several
    // statements on one logical source line.  The common `QDEL_NULL(x)`
    // helper, for example, becomes `qdel(x); x = null`.  Keep statement
    // separators out of the expression parser by turning only top-level
    // semicolons into ordinary logical lines before lowering the body.
    let joined_body = join_delimited_physical_lines(&definition.body);
    let mut body = normalize_labeled_loops(split_top_level_semicolon_statements(&joined_body));
    // A backslash joins physical DM lines before parsing.  Syntax retains the
    // marker for source mapping, but it has no expression-level meaning.
    for line in &mut body {
        line.tokens
            .retain(|token| !matches!(token.kind, TokenKind::LineContinuation));
    }
    if definition.kind == DefinitionKind::Verb {
        // `set hidden/category/name/...` lines on verbs are declaration
        // metadata, not executable assignments.
        body.retain(|line| {
            !matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(keyword)) if keyword == "set")
        });
    }
    let falls_through = if let Some(first_line) = body.first() {
        let block_indentation = indentation(first_line);
        let (next_line, falls_through) = compile_block(
            &body,
            0,
            block_indentation,
            &mut locals,
            &mut instructions,
            &mut source_spans,
            procedures,
            &mut loops,
        )?;
        if next_line != body.len() {
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
            Instruction::LoadResult,
            definition.span,
        );
        push_instruction(
            &mut instructions,
            &mut source_spans,
            Instruction::Return,
            definition.span,
        );
    }

    specialize_local_list_iteration_headers(&mut instructions);
    if locals.slot_count > u16::MAX as usize {
        return Err(compile_error(
            "procedure exceeds the VM ABI limit of 65535 local slots",
        ));
    }
    if instructions.len() > u16::MAX as usize {
        return Err(compile_error(
            "procedure exceeds the VM ABI limit of 65535 instructions",
        ));
    }
    Ok(Program {
        wait_for: procedure_wait_for(definition),
        parameter_count: definition.parameters.len(),
        parameter_names: definition
            .parameters
            .iter()
            .map(|parameter| {
                parameter_name(&parameter.tokens)
                    .unwrap_or_default()
                    .to_owned()
            })
            .collect(),
        verb_parameter_types: definition
            .parameters
            .iter()
            .map(|parameter| verb_parameter_type(&parameter.tokens))
            .collect(),
        verb_name: procedure_verb_name(definition),
        local_count: locals.slot_count,
        instructions,
        source_spans,
    })
}

fn specialize_local_list_iteration_headers(instructions: &mut [Instruction]) {
    for index in 0..instructions.len().saturating_sub(6) {
        let Some(
            [
                Instruction::LoadLocal(index_slot),
                Instruction::ListLengthLocal(list_slot),
                Instruction::LessEqual,
                Instruction::JumpIfFalse(exit),
                Instruction::LoadLocal(fetch_index),
                Instruction::IndexLocalList(fetch_list),
                Instruction::StoreLocal(item_slot),
            ],
        ) = instructions.get(index..index + 7)
        else {
            continue;
        };
        if index_slot != fetch_index || list_slot != fetch_list {
            continue;
        }
        let specialized = Instruction::NextLocalListIteration {
            list_slot: *list_slot,
            index_slot: *index_slot,
            item_slot: *item_slot,
            exit: *exit,
        };
        instructions[index] = specialized;
    }
}

/// Joins physical source lines while a parenthesized/bracketed expression is
/// still open. Dream Maker treats their indentation and leading commas as
/// whitespace inside the expression rather than as new statements.
fn join_delimited_physical_lines(lines: &[SourceLine]) -> Vec<SourceLine> {
    let mut result = Vec::new();
    let mut index = 0usize;
    while index < lines.len() {
        let mut line = lines[index].clone();
        let mut depth = delimiter_delta(&line.tokens);
        while depth > 0 && index + 1 < lines.len() {
            index += 1;
            let continuation = &lines[index];
            depth += delimiter_delta(&continuation.tokens);
            line.tokens.extend(continuation.tokens.iter().cloned());
            line.span = SourceSpan::new(line.span.start, continuation.span.end);
        }
        result.push(line);
        index += 1;
    }
    result
}

fn delimiter_delta(tokens: &[SpannedToken]) -> isize {
    tokens
        .iter()
        .fold(0isize, |depth, token| match &token.kind {
            TokenKind::Punctuation('(' | '[') => depth + 1,
            TokenKind::Operator(operator) if operator == "?[" => depth + 1,
            TokenKind::Punctuation(')' | ']') => depth - 1,
            _ => depth,
        })
}

/// Expands DM's macro-style statement separators and compact brace bodies.
///
/// BYOND macros commonly use C-style compact bodies even though ordinary DM
/// source is indentation based: `if (!value) { value = list(); } value += x;`.
/// The preprocessor leaves that expansion on the invocation's logical line.
/// Re-present its braces and semicolons as indentation-based logical lines
/// before statement lowering.  Parenthesized and indexed expressions retain
/// their punctuation unchanged. Empty statements (including a physical line
/// containing only `}`) are legal and discarded.
fn split_top_level_semicolon_statements(lines: &[SourceLine]) -> Vec<SourceLine> {
    fn ends_with_type_path(tokens: &[SpannedToken]) -> bool {
        let mut index = tokens.len();
        let mut segments = 0usize;
        while index >= 2 {
            if !matches!(tokens[index - 1].kind, TokenKind::Identifier(_))
                || !matches!(&tokens[index - 2].kind, TokenKind::Operator(operator) if operator == "/")
            {
                break;
            }
            segments += 1;
            index -= 2;
        }
        segments > 0
    }

    let mut result = Vec::with_capacity(lines.len());
    // A preprocessor expansion may contain a backslash-continued compact
    // brace body. The syntax layer preserves those continuations as separate
    // physical SourceLines, so brace nesting must survive the line boundary.
    // Resetting it per line incorrectly made declarations from the opening
    // line unavailable to an `else` branch emitted on the following line.
    let mut brace_depth = 0usize;
    for line in lines {
        let mut statement = Vec::new();
        let mut grouping_depth = 0usize;
        let base_indentation = indentation(line);
        let mut emit = |tokens: &mut Vec<SpannedToken>, brace_depth: usize| {
            if tokens.is_empty() {
                return;
            }
            let mut logical_line = line.clone();
            logical_line.indentation.tabs = 0;
            logical_line.indentation.spaces = base_indentation.saturating_add(brace_depth);
            logical_line.tokens = std::mem::take(tokens);
            result.push(logical_line);
        };
        for token in &line.tokens {
            match &token.kind {
                TokenKind::Punctuation('(' | '[') => {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Operator(operator) if operator == "?[" => {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation(')' | ']') => {
                    grouping_depth = grouping_depth.saturating_sub(1);
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('{') if grouping_depth > 0 => {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('{')
                    if grouping_depth == 0
                        && statement.iter().any(
                            |token| matches!(&token.kind, TokenKind::Identifier(name) if name == "new"),
                        )
                        || grouping_depth == 0 && ends_with_type_path(&statement) =>
                {
                    grouping_depth += 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('}') if grouping_depth > 0 => {
                    grouping_depth -= 1;
                    statement.push(token.clone());
                }
                TokenKind::Punctuation('{') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                    brace_depth += 1;
                }
                TokenKind::Punctuation('}') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                    brace_depth = brace_depth.saturating_sub(1);
                }
                TokenKind::Punctuation(';') if grouping_depth == 0 => {
                    emit(&mut statement, brace_depth);
                }
                _ => statement.push(token.clone()),
            }
        }
        emit(&mut statement, brace_depth);
    }
    result
}

fn normalize_labeled_loops(mut lines: Vec<SourceLine>) -> Vec<SourceLine> {
    let mut index = 0;
    while index + 1 < lines.len() {
        let label = match lines[index].tokens.as_slice() {
            [
                SpannedToken {
                    kind: TokenKind::Identifier(label),
                    ..
                },
                SpannedToken {
                    kind: TokenKind::Operator(colon),
                    ..
                },
            ] if colon == ":" => label.clone(),
            _ => {
                index += 1;
                continue;
            }
        };
        let base = indentation(&lines[index]);
        let is_loop = matches!(
            lines[index + 1].tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if matches!(keyword.as_str(), "for" | "while" | "do")
        );
        if !is_loop || indentation(&lines[index + 1]) < base {
            index += 1;
            continue;
        }
        let indentation_delta = indentation(&lines[index + 1]) - base;
        lines.remove(index);
        lines[index].indentation.tabs = 0;
        lines[index].indentation.spaces = base;
        let mut active_loop_indents = vec![base];
        let mut cursor = index + 1;
        while cursor < lines.len() && indentation(&lines[cursor]) > base {
            lines[cursor].indentation.tabs = 0;
            lines[cursor].indentation.spaces =
                indentation(&lines[cursor]).saturating_sub(indentation_delta);
            let current_indent = indentation(&lines[cursor]);
            while active_loop_indents
                .last()
                .is_some_and(|indent| *indent >= current_indent)
            {
                active_loop_indents.pop();
            }
            if matches!(
                lines[cursor].tokens.first().map(|token| &token.kind),
                Some(TokenKind::Identifier(keyword)) if matches!(keyword.as_str(), "for" | "while" | "do")
            ) {
                active_loop_indents.push(current_indent);
            }
            if matches!(
                lines[cursor].tokens.as_slice(),
                [SpannedToken { kind: TokenKind::Identifier(keyword), .. }, SpannedToken { kind: TokenKind::Identifier(target), .. }]
                    if matches!(keyword.as_str(), "break" | "continue") && target == &label
            ) {
                lines[cursor].tokens[1].kind =
                    TokenKind::Number(active_loop_indents.len().to_string());
            }
            cursor += 1;
        }
        index = cursor;
    }
    lines
}

struct LocalTable<'fields> {
    names: HashMap<String, u16>,
    types: HashMap<String, TypePath>,
    src_fields: &'fields BTreeMap<String, FieldName>,
    global_fields: &'fields BTreeMap<String, FieldName>,
    global_types: &'fields BTreeMap<String, TypePath>,
    slot_count: usize,
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
    fn with_fields(
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
    fn insert_parameter(&mut self, name: String, slot: u16) {
        self.names.insert(name, slot);
        self.slot_count = self.slot_count.max(usize::from(slot) + 1);
    }

    fn reserve_parameter_slots(&mut self, count: usize) -> Result<(), CompileError> {
        // Keep unnamed varargs positions available to the frame binder and
        // ensure subsequent locals are allocated after every parameter.
        let count = usize::from(to_local_index(count)?);
        self.slot_count = self.slot_count.max(count);
        Ok(())
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

    fn declare_hidden(&mut self) -> Result<u16, CompileError> {
        let slot = to_local_index(self.slot_count)?;
        self.slot_count += 1;
        Ok(slot)
    }

    fn get(&self, name: &str) -> Option<u16> {
        self.names.get(name).copied()
    }

    fn set_type(&mut self, name: String, type_path: TypePath) {
        self.types.insert(name, type_path);
    }

    fn local_type(&self, name: &str) -> Option<&TypePath> {
        self.types.get(name)
    }

    fn src_field(&self, name: &str) -> Option<&FieldName> {
        self.src_fields.get(name)
    }

    fn global_field(&self, name: &str) -> Option<&FieldName> {
        self.global_fields.get(name)
    }

    fn global_type(&self, name: &str) -> Option<&TypePath> {
        self.global_types.get(name)
    }

    fn receiver_static(&self, receiver: &Expression, name: &FieldName) -> Option<&FieldName> {
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

struct LoopContext {
    continue_target: Option<usize>,
    continue_jumps: Vec<usize>,
    break_jumps: Vec<usize>,
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
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

fn procedure_wait_for(definition: &Definition) -> bool {
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

fn procedure_verb_name(definition: &Definition) -> Option<String> {
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

fn compound_instruction(operator: &str) -> Result<Instruction, CompileError> {
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

fn compound_list_index_operator(operator: &str) -> Result<CompoundListIndexOperator, CompileError> {
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

fn parameter_name(tokens: &[SpannedToken]) -> Option<&str> {
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

fn verb_parameter_type(tokens: &[SpannedToken]) -> VerbParameterType {
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

fn declared_parameter_type(tokens: &[SpannedToken], name: &str) -> Option<TypePath> {
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

fn expression_static_type(expression: &Expression, locals: &LocalTable<'_>) -> Option<TypePath> {
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

pub(crate) fn to_local_index(index: usize) -> Result<u16, CompileError> {
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
pub(crate) enum Expression {
    Null,
    Number(DmNumberBits),
    Text(String),
    File(String),
    TypePath(TypePath),
    ModifiedTypePath {
        base: TypePath,
        overrides: Vec<(FieldName, Self)>,
    },
    New {
        type_path: Option<Box<Self>>,
        arguments: Vec<Self>,
        overrides: Vec<(FieldName, Self)>,
    },
    Regex {
        arguments: Vec<Self>,
    },
    MutableAppearance {
        arguments: Vec<Self>,
    },
    Matrix {
        arguments: Vec<Self>,
    },
    Vector {
        arguments: Vec<Self>,
    },
    ReplaceText {
        arguments: Vec<Self>,
        exact: bool,
        character_indices: bool,
    },
    CopyText {
        arguments: Vec<Self>,
        character_indices: bool,
    },
    StandardBuiltin {
        name: String,
        arguments: Vec<Self>,
    },
    NativeSrcMethod {
        name: String,
        arguments: Vec<Self>,
    },
    ExternalCall {
        library: Box<Self>,
        function: Box<Self>,
        arguments: Vec<Self>,
    },
    Animate {
        arguments: Vec<(Option<String>, Self)>,
    },
    Filter {
        arguments: Vec<(Option<String>, Self)>,
    },
    Crash(Box<Self>),
    Sleep(Box<Self>),
    Initial(Box<Self>),
    Block {
        arguments: Vec<Self>,
    },
    Rand {
        arguments: Vec<Self>,
    },
    Roll {
        arguments: Vec<Self>,
    },
    Pick {
        entries: Vec<(Option<Self>, Self)>,
    },
    Prob(Box<Self>),
    Round {
        arguments: Vec<Self>,
    },
    Length {
        value: Box<Self>,
    },
    Ref {
        value: Box<Self>,
    },
    GetStep {
        source: Box<Self>,
        direction: Box<Self>,
    },
    GetStepTowards {
        source: Box<Self>,
        target: Box<Self>,
    },
    Range {
        arguments: Vec<Self>,
    },
    TypesOf {
        arguments: Vec<Self>,
    },
    HasCall {
        receiver: Box<Self>,
        selector: Box<Self>,
    },
    TypePredicate {
        kind: TypePredicateKind,
        arguments: Vec<Self>,
    },
    Local(String),
    Src,
    Usr,
    Caller,
    World,
    GlobalNamespace,
    Field {
        receiver: Box<Self>,
        name: FieldName,
    },
    SafeField {
        receiver: Box<Self>,
        name: FieldName,
    },
    GlobalField(FieldName),
    Result,
    Call {
        procedure: String,
        arguments: Vec<Self>,
    },
    NamedArgument {
        name: String,
        value: Box<Self>,
    },
    /// A list expansion used only in an enclosing call or constructor
    /// argument list (`target(arglist(values))`).
    ArgList(Box<Self>),
    Locate {
        arguments: Vec<Self>,
    },
    LocateIn {
        arguments: Vec<Self>,
        container: Box<Self>,
    },
    CurrentCall {
        arguments: Option<Vec<Self>>,
    },
    ParentCall {
        arguments: Option<Vec<Self>>,
    },
    DynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
        null_receiver_is_global: bool,
    },
    SafeDynamicCall {
        target: Box<Self>,
        procedure: Box<Self>,
        arguments: Vec<Self>,
    },
    List(Vec<ListExpressionEntry>),
    AssociativeList(Vec<ListExpressionEntry>),
    Index {
        list: Box<Self>,
        index: Box<Self>,
    },
    SafeIndex {
        list: Box<Self>,
        index: Box<Self>,
    },
    Unary {
        operator: String,
        operand: Box<Self>,
    },
    Mutation {
        target: Box<Self>,
        delta: i8,
        prefix: bool,
    },
    Binary {
        operator: String,
        left: Box<Self>,
        right: Box<Self>,
    },
    Conditional {
        condition: Box<Self>,
        when_true: Box<Self>,
        when_false: Box<Self>,
    },
    LogicalOrAssignment {
        target: Box<Self>,
        value: Box<Self>,
    },
    Assignment {
        target: Box<Self>,
        operator: String,
        value: Box<Self>,
    },
}

fn expression_null_propagates(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::SafeField { .. }
            | Expression::SafeIndex { .. }
            | Expression::SafeDynamicCall { .. }
    )
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ListExpressionEntry {
    Positional(Expression),
    Associative { key: Expression, value: Expression },
}

/// Numeric constants supplied by the BYOND language rather than by project
/// source. Keep this deliberately finite: an unrecognised identifier must
/// continue through ordinary local/field resolution and retain its useful
/// diagnostic instead of silently becoming a number.
fn dm_builtin_text_constant(identifier: &str) -> Option<&'static str> {
    match identifier {
        "UNIX" => Some("UNIX"),
        "MS_WINDOWS" => Some("MS Windows"),
        "MALE" => Some("male"),
        "FEMALE" => Some("female"),
        "NEUTER" => Some("neuter"),
        "PLURAL" => Some("plural"),
        _ => None,
    }
}

pub(crate) fn dm_builtin_numeric_constant(identifier: &str) -> Option<f32> {
    match identifier {
        "FALSE"
        | "BLEND_DEFAULT"
        | "MATRIX_COPY"
        | "MOB_PERSPECTIVE"
        | "TOPDOWN_MAP"
        | "LINEAR_EASING"
        | "COLORSPACE_RGB"
        | "MOUSE_INACTIVE_POINTER"
        | "NO_STEPS"
        | "PROFILE_START"
        | "PROFILE_REFRESH"
        | "FILTER_COLOR_RGB"
        | "UNIFORM_RAND"
        | "ICON_ADD" => Some(0.0),
        "FLOAT_LAYER" => Some(-1.0),
        "TRUE"
        | "SOUND_STREAM"
        | "NORMAL_RAND"
        | "MASK_INVERSE"
        | "FILTER_OVERLAY"
        | "FILTER_COLOR_HSV"
        | "OUTLINE_SHARP"
        | "WAVE_SIDEWAYS"
        | "ICON_SUBTRACT"
        | "BLEND_OVERLAY"
        | "KEEP_TOGETHER"
        | "NORTH"
        | "EYE_PERSPECTIVE"
        | "AREA_LAYER"
        | "SINE_EASING"
        | "ANIMATION_END_NOW"
        | "COLORSPACE_HSV"
        | "VIS_INHERIT_ICON"
        | "MOUSE_ACTIVE_POINTER"
        | "FORWARD_STEPS"
        | "BLIND"
        | "PROFILE_STOP" => Some(1.0),
        "CONTROL_FREAK_SKIN" => Some(1.0),
        "CONTROL_FREAK_MACROS" => Some(2.0),
        "JSON_PRETTY_PRINT" => Some(1.0),
        "BLEND_ADD"
        | "LINEAR_RAND"
        | "MASK_SWAP"
        | "FILTER_UNDERLAY"
        | "FILTER_COLOR_HSL"
        | "OUTLINE_SQUARE"
        | "WAVE_BOUNDED"
        | "KEEP_APART"
        | "SOUTH"
        | "EDGE_PERSPECTIVE"
        | "TURF_LAYER"
        | "CIRCULAR_EASING"
        | "ANIMATION_LINEAR_TRANSFORM"
        | "COLORSPACE_HSL"
        | "VIS_INHERIT_ICON_STATE"
        | "SLIDE_STEPS"
        | "PROFILE_CLEAR"
        | "PROFILE_RESTART"
        | "ICON_MULTIPLY" => Some(2.0),
        "BLEND_SUBTRACT" | "SQUARE_RAND" | "FILTER_COLOR_HCY" | "OBJ_LAYER" | "CUBIC_EASING"
        | "COLORSPACE_HCY" | "MOUSE_DRAG_POINTER" | "SYNC_STEPS" | "ICON_OVERLAY" => Some(3.0),
        "BLEND_MULTIPLY" | "LONG_GLIDE" | "EAST" | "MATRIX_INVERT" | "MOB_LAYER"
        | "BOUNCE_EASING" | "ANIMATION_PARALLEL" | "VIS_INHERIT_DIR" | "MOUSE_DROP_POINTER"
        | "SEE_MOBS" | "SEEMOBS" | "PROFILE_AVERAGE" => Some(4.0),
        "SOUND_UPDATE" => Some(16.0),
        "BLEND_INSET_OVERLAY"
        | "NORTHEAST"
        | "MATRIX_ROTATE"
        | "FLY_LAYER"
        | "ELASTIC_EASING"
        | "MOUSE_ARROW_POINTER"
        | "ICON_OR" => Some(5.0),
        "SOUTHEAST"
        | "MATRIX_SCALE"
        | "BACK_EASING"
        | "MOUSE_CROSSHAIRS_POINTER"
        | "ICON_UNDERLAY" => Some(6.0),
        "MATRIX_TRANSLATE" | "QUAD_EASING" | "MOUSE_HAND_POINTER" => Some(7.0),
        "WEST" | "RESET_TRANSFORM" | "JUMP_EASING" | "ANIMATION_SLICE" | "VIS_INHERIT_LAYER"
        | "SEE_OBJS" | "SEEOBJS" => Some(8.0),
        "NORTHWEST" => Some(9.0),
        "SOUTHWEST" => Some(10.0),
        "UP" | "RESET_COLOR" | "ANIMATION_END_LOOP" | "VIS_INHERIT_PLANE" | "SEE_TURFS"
        | "SEETURFS" => Some(16.0),
        "DOWN" | "RESET_ALPHA" | "VIS_INHERIT_ID" | "SEE_SELF" => Some(32.0),
        // Appearance flags are BYOND bitflags. Keep the complete contiguous
        // built-in flag family here rather than teaching project code about
        // individual flags as each one is encountered.
        // These make an overlay/image ignore the corresponding value
        // inherited from its parent.
        "PIXEL_SCALE" | "EASE_IN" | "VIS_UNDERLAY" | "SEE_INFRA" => Some(64.0),
        "TILE_BOUND" | "MATRIX_MODIFY" | "EASE_OUT" | "VIS_HIDE" => Some(128.0),
        "INHERIT_ID" | "ANIMATION_RELATIVE" | "SEE_PIXELS" => Some(256.0),
        "NO_CLIENT_COLOR" | "ANIMATION_CONTINUE" | "SEE_THRU" => Some(512.0),
        "RESET_CONTENTS" | "SEE_BLACKNESS" => Some(1024.0),
        "PLANE_MASTER" => Some(2048.0),
        "PASS_MOUSE" => Some(4096.0),
        "TILE_MOVER" => Some(8192.0),
        "EFFECTS_LAYER" => Some(5000.0),
        "TOPDOWN_LAYER" => Some(10000.0),
        "BACKGROUND_LAYER" => Some(20000.0),
        "FLOAT_PLANE" => Some(-32767.0),
        "TILED_ICON_MAP" => Some(32768.0),
        _ => None,
    }
}

pub(crate) struct ExpressionParser<'a> {
    tokens: &'a [SpannedToken],
    index: usize,
    /// While parsing the true arm of `?:`, a bare colon terminates that arm
    /// instead of selecting a dynamic field.  Outside that one context DM's
    /// `datum:field` syntax remains a normal postfix operation, including in
    /// the false arm (`condition ? datum : datum:type`).
    conditional_true_arm: bool,
}

impl<'a> ExpressionParser<'a> {
    pub(crate) const fn new(tokens: &'a [SpannedToken]) -> Self {
        Self {
            tokens,
            index: 0,
            conditional_true_arm: false,
        }
    }

    pub(crate) fn parse(mut self) -> Result<Expression, CompileError> {
        let expression = self.parse_assignment()?;
        if self.index != self.tokens.len() {
            return Err(compile_error(format!(
                "unexpected token {:?} in expression",
                self.tokens[self.index].kind
            )));
        }
        Ok(expression)
    }

    /// Parses right-associative assignment expressions. DM permits an
    /// assignment anywhere an expression is accepted, for example
    /// `(GLOB.initialized = TRUE)` in a macro expansion.
    fn parse_assignment(&mut self) -> Result<Expression, CompileError> {
        let target = self.parse_conditional()?;
        let Some(TokenKind::Operator(operator)) =
            self.tokens.get(self.index).map(|token| &token.kind)
        else {
            return Ok(target);
        };
        if !matches!(
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
        ) {
            return Ok(target);
        }
        let operator = if operator == ":=" {
            "=".to_owned()
        } else {
            operator.clone()
        };
        self.index += 1;
        let value = self.parse_assignment()?;
        if operator == "||=" {
            return Ok(Expression::LogicalOrAssignment {
                target: Box::new(target),
                value: Box::new(value),
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
    }

    /// Parses DM's right-associative `condition ? when_true : when_false`
    /// expression.  It deliberately sits below every binary operator, so a
    /// condition such as `a || b ? c : d` is parsed as expected.
    fn parse_conditional(&mut self) -> Result<Expression, CompileError> {
        let condition = self.parse_binary(1)?;
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Operator(operator)) if operator == "?"
        ) {
            return Ok(condition);
        }
        self.index += 1;
        let enclosing_true_arm = self.conditional_true_arm;
        self.conditional_true_arm = true;
        let when_true = self.parse_assignment()?;
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) if operator == ":" => self.index += 1,
            _ => return Err(compile_error("expected ':' in conditional expression")),
        }
        // The false arm is still inside an enclosing true arm, if there is
        // one.  In `a ? b ? c : d : e`, that outer colon must terminate the
        // nested expression rather than becoming dynamic access `d:e`.
        self.conditional_true_arm = enclosing_true_arm;
        let when_false = self.parse_assignment()?;
        Ok(Expression::Conditional {
            condition: Box::new(condition),
            when_true: Box::new(when_true),
            when_false: Box::new(when_false),
        })
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
            let right_precedence = if operator == "**" {
                precedence
            } else {
                precedence + 1
            };
            let right = self.parse_binary(right_precedence)?;
            left = if operator == "in" {
                // `value in lower to upper` is BYOND's inclusive range
                // predicate. `to` is a keyword delimiter rather than a
                // general arithmetic operator, so lower it directly to the
                // two comparisons while the left operand is still available.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Identifier(keyword)) if keyword == "to"
                ) {
                    self.index += 1;
                    let upper = self.parse_binary(right_precedence)?;
                    Expression::Binary {
                        operator: "&&".to_owned(),
                        left: Box::new(Expression::Binary {
                            operator: ">=".to_owned(),
                            left: Box::new(left.clone()),
                            right: Box::new(right),
                        }),
                        right: Box::new(Expression::Binary {
                            operator: "<=".to_owned(),
                            left: Box::new(left),
                            right: Box::new(upper),
                        }),
                    }
                } else {
                    match left {
                        Expression::Locate { arguments } => Expression::LocateIn {
                            arguments,
                            container: Box::new(right),
                        },
                        left => Expression::Binary {
                            operator,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                    }
                }
            } else {
                Expression::Binary {
                    operator,
                    left: Box::new(left),
                    right: Box::new(right),
                }
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expression, CompileError> {
        // Prefix mutation is an expression in DM, not merely a statement:
        // `values[++i]` first updates i, then uses the new value. Reuse the
        // assignment lowering so every assignable target retains its normal
        // single-evaluation behavior.
        if let Some(operator @ ("++" | "--")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            let target = self.parse_unary()?;
            return Ok(Expression::Mutation {
                target: Box::new(target),
                delta: if operator == "++" { 1 } else { -1 },
                prefix: true,
            });
        }
        if let Some(operator @ ("!" | "+" | "-" | "~" | "&" | "*")) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            return Ok(Expression::Unary {
                operator,
                operand: Box::new(self.parse_unary()?),
            });
        }
        let mut expression = self.parse_primary()?;
        loop {
            let safe_list_index = matches!(self.current_operator(), Some("?["));
            let starts_list_index = matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('['))
            ) || safe_list_index;
            if starts_list_index {
                self.index += 1;
                // An index is a full DM expression. In particular, ternaries
                // and assignments are legal here (`values[flag ? a : b]`).
                // Parsing only the binary-precedence layer left the `?` in
                // front of the closing bracket and produced a misleading
                // "expected ']'" diagnostic.
                let index = self.parse_assignment()?;
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(']'))
                ) {
                    return Err(compile_error("expected ']' after list index"));
                }
                self.index += 1;
                expression = if safe_list_index || expression_null_propagates(&expression) {
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
            if matches!(self.current_operator(), Some("::")) {
                self.index += 1;
                let Some(TokenKind::Identifier(qualified)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected identifier after '::'"));
                };
                let qualified = qualified.clone();
                self.index += 1;
                if qualified == "name"
                    && let Expression::TypePath(path) = &expression
                    && let Some((_, procedure_name)) = path.as_str().rsplit_once("/proc/")
                {
                    expression = Expression::Text(procedure_name.to_owned());
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    expression = Expression::Call {
                        procedure: qualified,
                        arguments: self.parse_call_arguments()?,
                    };
                } else {
                    let name = FieldName::parse(&qualified)
                        .map_err(|error| compile_error(error.to_string()))?;
                    expression = Expression::Initial(Box::new(Expression::Field {
                        receiver: Box::new(expression),
                        name,
                    }));
                }
                continue;
            }
            if matches!(self.current_operator(), Some("." | "?." | "?:"))
                || (matches!(self.current_operator(), Some(":"))
                    && (!self.conditional_true_arm
                        || (self.colon_member_is_lexically_attached()
                            && self.conditional_true_arm_has_later_colon()))
                    && matches!(
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                        Some(TokenKind::Identifier(_))
                    ))
            {
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
                let propagate_null = safe_member || expression_null_propagates(&expression);
                expression = if matches!(expression, Expression::GlobalNamespace) {
                    Expression::GlobalField(name)
                } else if propagate_null {
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
            // `input(...) as null|anything in choices` is prompt metadata, not
            // a cast. Retain it in the internal builtin selector and append
            // the evaluated choice list so a connected client can display the
            // correct modal while headless execution keeps its default path.
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Identifier(keyword)) if keyword == "as"
            ) {
                self.index += 1;
                let mut prompt_types = Vec::new();
                loop {
                    match self.tokens.get(self.index).map(|token| &token.kind) {
                        Some(TokenKind::Identifier(keyword)) if keyword == "in" => break,
                        Some(TokenKind::Punctuation(',' | ')' | ']' | '}')) | None => {
                            if let Expression::StandardBuiltin { name, .. } = &mut expression
                                && name == "input"
                            {
                                *name = format!("input@{}", prompt_types.join("+"));
                            }
                            return Ok(expression);
                        }
                        Some(TokenKind::Identifier(prompt_type)) => {
                            prompt_types.push(prompt_type.to_ascii_lowercase());
                            self.index += 1;
                        }
                        _ => self.index += 1,
                    }
                }
                self.index += 1;
                let choices = self.parse_assignment()?;
                if let Expression::StandardBuiltin { name, arguments } = &mut expression
                    && name == "input"
                {
                    prompt_types.push("list".to_owned());
                    *name = format!("input@{}", prompt_types.join("+"));
                    arguments.push(choices);
                }
                continue;
            }
            // A datum procedure call is a postfix operation in DM.  The
            // regular `name(...)` arm in `parse_primary` handles static
            // calls, while `receiver.name(...)` must retain both the datum
            // receiver and its dynamically-selected procedure name.  This
            // occurs extensively in lifecycle code after macro expansion
            // (for example signal dispatch helpers).
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('('))
            ) {
                expression = match expression {
                    Expression::Field { receiver, name } => {
                        let arguments = self.parse_call_arguments()?;
                        Expression::DynamicCall {
                            target: receiver,
                            procedure: Box::new(Expression::Text(name.as_str().to_owned())),
                            arguments,
                            null_receiver_is_global: false,
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
                    // A second argument list invokes the procedure selector
                    // produced by the preceding expression.  DreamMaker uses
                    // this for `call_ext(library, function)(arguments)` as
                    // well as ordinary `call(...)(...)` selectors.
                    other => Expression::DynamicCall {
                        target: Box::new(Expression::Null),
                        procedure: Box::new(other),
                        arguments: self.parse_call_arguments()?,
                        null_receiver_is_global: true,
                    },
                };
                continue;
            }
            if let Some(operator @ ("++" | "--")) = self.current_operator() {
                let delta = if operator == "++" { 1 } else { -1 };
                self.index += 1;
                expression = Expression::Mutation {
                    target: Box::new(expression),
                    delta,
                    prefix: false,
                };
                continue;
            }
            break;
        }
        Ok(expression)
    }

    fn colon_member_is_lexically_attached(&self) -> bool {
        let Some(colon) = self.tokens.get(self.index) else {
            return false;
        };
        let Some(name) = self.tokens.get(self.index + 1) else {
            return false;
        };
        colon.span.end == name.span.start
    }

    /// Inside the true arm of `?:`, an attached `:name` is ambiguous with the
    /// ternary separator (`condition ? value:null`). It can only be dynamic
    /// member access when another colon remains to terminate the conditional
    /// (`condition ? datum:field : fallback`). That delimiter can be outside
    /// grouping which began before the member, as in a macro-expanded
    /// `condition ? list[(inner ? value : value:type)] : fallback`.
    fn conditional_true_arm_has_later_colon(&self) -> bool {
        for token in self.tokens.iter().skip(self.index + 2) {
            if matches!(&token.kind, TokenKind::Operator(operator) if operator == ":") {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primary(&mut self) -> Result<Expression, CompileError> {
        let token = self
            .tokens
            .get(self.index)
            .ok_or_else(|| compile_error("expected an expression"))?;
        self.index += 1;
        match &token.kind {
            // Type paths are expression values in DM: `/obj/item/tool` is
            // distinct from text and is accepted by builtins such as
            // `istype`, `ispath`, and `new`. The lexer exposes every slash as
            // an operator, so consume the complete slash-delimited sequence
            // here before ordinary binary division is considered.
            TokenKind::Operator(operator) if operator == "/" => {
                let mut path = String::new();
                loop {
                    let Some(TokenKind::Identifier(segment)) =
                        self.tokens.get(self.index).map(|token| &token.kind)
                    else {
                        // BYOND accepts a canonical type path with a trailing
                        // slash (commonly used as an associative-list key).
                        // The slash has already been consumed; canonicalize it
                        // away once at least one real segment was collected.
                        if !path.is_empty() {
                            break;
                        }
                        return Err(compile_error("expected a type path segment after '/'"));
                    };
                    path.push('/');
                    path.push_str(segment);
                    self.index += 1;
                    if !matches!(self.current_operator(), Some("/")) {
                        break;
                    }
                    self.index += 1;
                }
                let base =
                    TypePath::parse(&path).map_err(|error| compile_error(error.to_string()))?;
                let overrides = self.parse_modified_type_overrides()?;
                if overrides.is_empty() {
                    Ok(Expression::TypePath(base))
                } else {
                    Ok(Expression::ModifiedTypePath { base, overrides })
                }
            }
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
            TokenKind::Operator(operator) if operator == "." => Ok(Expression::Result),
            TokenKind::Number(spelling) => parse_number(spelling).map(Expression::Number),
            // Resource literals are first-class file values. Keep them
            // distinct from ordinary text so BYOND's `isfile()` contract is
            // observable by project code such as the runtime DMM reader.
            TokenKind::String(text) => parse_interpolated_string(text),
            TokenKind::RawString(text) | TokenKind::TextBlock(text) => {
                Ok(Expression::Text(text.clone()))
            }
            TokenKind::Resource(text) => {
                let normalized = text.replace('\\', "/");
                Ok(Expression::File(
                    normalized
                        .strip_prefix("./")
                        .unwrap_or(&normalized)
                        .to_owned(),
                ))
            }
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(Expression::Null),
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_numeric_constant(identifier) =>
            {
                Ok(Expression::Number(DmNumberBits::from_f32(value)))
            }
            TokenKind::Identifier(identifier)
                if let Some(value) = dm_builtin_text_constant(identifier) =>
            {
                Ok(Expression::Text(value.to_owned()))
            }
            TokenKind::Operator(operator) if operator == "::" => {
                let Some(TokenKind::Identifier(name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                else {
                    return Err(compile_error("expected global identifier after '::'"));
                };
                let name = name.clone();
                self.index += 1;
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::Call {
                        procedure: name,
                        arguments: self.parse_call_arguments()?,
                    })
                } else {
                    FieldName::parse(&name)
                        .map(Expression::GlobalField)
                        .map_err(|error| compile_error(error.to_string()))
                }
            }
            TokenKind::Identifier(identifier) if identifier == "src" => Ok(Expression::Src),
            TokenKind::Identifier(identifier) if identifier == "usr" => Ok(Expression::Usr),
            TokenKind::Identifier(identifier) if identifier == "caller" => Ok(Expression::Caller),
            TokenKind::Identifier(identifier) if identifier == "world" => Ok(Expression::World),
            TokenKind::Identifier(identifier) if identifier == "locs" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("locs").expect("built-in locs field name is valid"),
            }),
            TokenKind::Identifier(identifier) if identifier == "vars" => Ok(Expression::Field {
                receiver: Box::new(Expression::Src),
                name: FieldName::parse("vars").expect("built-in vars field name is valid"),
            }),
            // Only lowercase `global` is BYOND's built-in namespace. `GLOB`
            // in SS13 codebases is an ordinary declared global datum.
            TokenKind::Identifier(identifier) if identifier == "global" => {
                Ok(Expression::GlobalNamespace)
            }
            TokenKind::Identifier(identifier) if matches!(self.tokens.get(self.index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "::") =>
            {
                let mut qualifiers = Vec::new();
                let mut next_token = self.tokens.get(self.index).map(|token| &token.kind);
                while let Some(TokenKind::Operator(operator)) = next_token {
                    if operator != "::" {
                        break;
                    }
                    self.index += 1;
                    let token = self
                        .tokens
                        .get(self.index)
                        .ok_or_else(|| compile_error("expected namespace qualifier after '::'"))?;
                    let TokenKind::Identifier(qualified) = &token.kind else {
                        return Err(compile_error("expected identifier after '::'"));
                    };
                    qualifiers.push(qualified.clone());
                    self.index += 1;
                    next_token = self.tokens.get(self.index).map(|token| &token.kind);
                }

                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: qualifiers
                            .last()
                            .expect("namespace chain has a qualifier")
                            .clone(),
                        arguments,
                    })
                } else {
                    let mut receiver = Expression::Local(identifier.clone());
                    for qualifier in qualifiers {
                        let name = FieldName::parse(&qualifier)
                            .map_err(|error| compile_error(error.to_string()))?;
                        receiver = Expression::Initial(Box::new(Expression::Field {
                            receiver: Box::new(receiver),
                            name,
                        }));
                    }
                    Ok(receiver)
                }
            }
            TokenKind::Identifier(identifier) if identifier == "new" => {
                // `new /path(args)` is the common explicit form.  An
                // unqualified `new(args)` constructs the current datum type.
                // Keep the constructor arguments in the AST even though the
                // headless VM currently only establishes object identity.
                if matches!(self.current_operator(), Some("/")) {
                    let type_path = self.parse_primary()?;
                    let overrides = self.parse_modified_type_overrides()?;
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                        overrides,
                    })
                } else if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: self.parse_call_arguments()?,
                        overrides: Vec::new(),
                    })
                } else if let Some(TokenKind::Identifier(type_name)) =
                    self.tokens.get(self.index).map(|token| &token.kind)
                {
                    // DM also permits a runtime type expression, for example
                    // `new starting_organ(src)`.  This is distinct from
                    // unqualified `new(...)`: the identifier is the type to
                    // instantiate, not a constructor argument.
                    // Do not delegate this to `parse_unary`: its ordinary
                    // identifier rule interprets the following `(` as a
                    // static procedure call. Here it belongs to `new`.
                    let mut type_path = if type_name == "src" {
                        Expression::Src
                    } else {
                        Expression::Local(type_name.clone())
                    };
                    self.index += 1;
                    while matches!(self.current_operator(), Some(".")) {
                        self.index += 1;
                        let Some(TokenKind::Identifier(field)) =
                            self.tokens.get(self.index).map(|token| &token.kind)
                        else {
                            return Err(compile_error(
                                "runtime new type field access requires an identifier",
                            ));
                        };
                        type_path = Expression::Field {
                            receiver: Box::new(type_path),
                            name: FieldName::parse(field)
                                .map_err(|error| compile_error(error.to_string()))?,
                        };
                        self.index += 1;
                    }
                    let arguments = if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('('))
                    ) {
                        self.parse_call_arguments()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expression::New {
                        type_path: Some(Box::new(type_path)),
                        arguments,
                        overrides: Vec::new(),
                    })
                } else {
                    Ok(Expression::New {
                        type_path: None,
                        arguments: Vec::new(),
                        overrides: Vec::new(),
                    })
                }
            }
            TokenKind::Identifier(identifier) if identifier == "call_ext" => {
                let selectors = self.parse_call_arguments()?;
                let [library, function] = selectors.as_slice() else {
                    return Err(compile_error(
                        "call_ext requires a library and exported function selector",
                    ));
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call_ext selector requires an argument list"));
                }
                Ok(Expression::ExternalCall {
                    library: Box::new(library.clone()),
                    function: Box::new(function.clone()),
                    arguments: self.parse_call_arguments()?,
                })
            }
            TokenKind::Identifier(identifier) if identifier == "call" => {
                let selectors = self.parse_call_arguments()?;
                let (target, procedure, null_receiver_is_global) = match selectors.as_slice() {
                    [procedure] => (Expression::Null, procedure.clone(), true),
                    [target, procedure] => (target.clone(), procedure.clone(), false),
                    _ => {
                        return Err(compile_error(
                            "call requires a procedure or a receiver and procedure",
                        ));
                    }
                };
                if !matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    return Err(compile_error("call selector requires an argument list"));
                }
                Ok(Expression::DynamicCall {
                    target: Box::new(target),
                    procedure: Box::new(procedure),
                    arguments: self.parse_call_arguments()?,
                    null_receiver_is_global,
                })
            }
            TokenKind::Identifier(identifier)
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) =>
            {
                if identifier == "CRASH" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "CRASH requires exactly one argument, received {}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Crash(Box::new(
                        arguments.pop().expect("CRASH argument count was validated"),
                    )))
                } else if identifier == "list" {
                    Ok(Expression::List(self.parse_list_arguments()?))
                } else if identifier == "alist" {
                    Ok(Expression::AssociativeList(self.parse_list_arguments()?))
                } else if identifier == "arglist" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "arglist requires exactly one list, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ArgList(Box::new(
                        arguments.pop().expect("argument count was validated"),
                    )))
                } else if let Some(kind) = type_predicate_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    let valid_count = match kind {
                        TypePredicateKind::IsType | TypePredicateKind::IsPath => {
                            (1..=2).contains(&arguments.len())
                        }
                        // BYOND's location classifiers accept multiple values
                        // and succeed only when every supplied value matches.
                        TypePredicateKind::IsLoc
                        | TypePredicateKind::IsMovable
                        | TypePredicateKind::IsTurf => !arguments.is_empty(),
                        _ => arguments.len() == 1,
                    };
                    if !valid_count {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypePredicate { kind, arguments })
                } else if identifier == "initial" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "initial requires exactly one variable reference, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Initial(Box::new(
                        arguments.pop().expect("validated initial argument"),
                    )))
                } else if identifier == "regex" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "regex requires a pattern and optional flags, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Regex { arguments })
                } else if identifier == "mutable_appearance" {
                    Ok(Expression::MutableAppearance {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "matrix" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 6 {
                        return Err(compile_error("matrix accepts at most six arguments"));
                    }
                    Ok(Expression::Matrix { arguments })
                } else if identifier == "vector" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 3 {
                        return Err(compile_error("vector accepts at most three arguments"));
                    }
                    Ok(Expression::Vector { arguments })
                } else if let Some((exact, character_indices)) = replacetext_kind(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if !(3..=5).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "{identifier} requires text, needle, replacement, and optional start/end; received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::ReplaceText {
                        arguments,
                        exact,
                        character_indices,
                    })
                } else if matches!(identifier.as_str(), "copytext" | "copytext_char") {
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
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "length requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Length {
                        value: Box::new(
                            arguments
                                .pop()
                                .expect("length argument count was validated"),
                        ),
                    })
                } else if identifier == "ref" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "ref requires exactly one argument, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Ref {
                        value: Box::new(arguments.pop().expect("ref argument count was validated")),
                    })
                } else if identifier == "get_step" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step requires exactly an atom/turf and direction, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let direction = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    let source = arguments
                        .pop()
                        .expect("get_step argument count was validated");
                    Ok(Expression::GetStep {
                        source: Box::new(source),
                        direction: Box::new(direction),
                    })
                } else if identifier == "get_step_towards" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "get_step_towards requires exactly a source and target, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let target = arguments.pop().expect("argument count validated");
                    let source = arguments.pop().expect("argument count validated");
                    Ok(Expression::GetStepTowards {
                        source: Box::new(source),
                        target: Box::new(target),
                    })
                } else if identifier == "range" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "range requires a distance and optional center, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Range { arguments })
                } else if identifier == "block" {
                    let arguments = self.parse_call_arguments()?;
                    if !(arguments.len() == 2 || (3..=6).contains(&arguments.len())) {
                        return Err(compile_error(format!(
                            "block requires two turfs or three through six coordinates, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Block { arguments })
                } else if identifier == "typesof" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.is_empty() || arguments.len() > usize::from(u8::MAX) {
                        return Err(compile_error(format!(
                            "typesof requires between one and 255 type arguments, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::TypesOf { arguments })
                } else if identifier == "hascall" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 2 {
                        return Err(compile_error(format!(
                            "hascall requires a receiver and procedure selector, received {} arguments",
                            arguments.len()
                        )));
                    }
                    let selector = arguments.pop().expect("hascall arity was validated");
                    let receiver = arguments.pop().expect("hascall arity was validated");
                    Ok(Expression::HasCall {
                        receiver: Box::new(receiver),
                        selector: Box::new(selector),
                    })
                } else if identifier == "rand" {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() > 2 {
                        return Err(compile_error(format!(
                            "rand accepts zero, one, or two numeric bounds, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Rand { arguments })
                } else if identifier == "roll" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "roll requires dice or a dice count and side count, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Roll { arguments })
                } else if identifier == "pick" {
                    Ok(Expression::Pick {
                        entries: self.parse_pick_arguments()?,
                    })
                } else if identifier == "prob" {
                    let arguments = self.parse_call_arguments()?;
                    let [chance] = arguments.as_slice() else {
                        return Err(compile_error(format!(
                            "prob requires exactly one percentage, received {} arguments",
                            arguments.len()
                        )));
                    };
                    Ok(Expression::Prob(Box::new(chance.clone())))
                } else if identifier == "round" {
                    let arguments = self.parse_call_arguments()?;
                    if !(1..=2).contains(&arguments.len()) {
                        return Err(compile_error(format!(
                            "round requires a number and optional multiple, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Round { arguments })
                } else if identifier == "sleep" {
                    let mut arguments = self.parse_call_arguments()?;
                    if arguments.len() != 1 {
                        return Err(compile_error(format!(
                            "sleep requires exactly one delay, received {} arguments",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::Sleep(Box::new(
                        arguments.pop().expect("sleep argument count was validated"),
                    )))
                } else if identifier == "locate" {
                    Ok(Expression::Locate {
                        arguments: self.parse_call_arguments()?,
                    })
                } else if identifier == "animate" {
                    Ok(Expression::Animate {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "filter" {
                    Ok(Expression::Filter {
                        arguments: self.parse_named_call_arguments()?,
                    })
                } else if identifier == "nameof" {
                    self.parse_nameof_expression()
                } else if matches!(
                    identifier.as_str(),
                    "MapColors"
                        | "Blend"
                        | "SetIntensity"
                        | "Scale"
                        | "Crop"
                        | "Shift"
                        | "Width"
                        | "Height"
                        | "DrawBox"
                        | "Insert"
                        | "GetPixel"
                        | "Add"
                        | "Subtract"
                        | "Multiply"
                        | "Translate"
                        | "Invert"
                        | "Turn"
                ) {
                    Ok(Expression::NativeSrcMethod {
                        name: identifier.clone(),
                        arguments: self.parse_call_arguments()?,
                    })
                } else if let Some((minimum, maximum)) = standard_builtin_arity(identifier) {
                    let arguments = self.parse_call_arguments()?;
                    if arguments.len() < minimum || arguments.len() > maximum {
                        return Err(compile_error(format!(
                            "{identifier} received {} arguments; expected {minimum} through {maximum}",
                            arguments.len()
                        )));
                    }
                    Ok(Expression::StandardBuiltin {
                        name: identifier.clone(),
                        arguments,
                    })
                } else {
                    let arguments = self.parse_call_arguments()?;
                    Ok(Expression::Call {
                        procedure: identifier.clone(),
                        arguments,
                    })
                }
            }
            TokenKind::Identifier(identifier) => Ok(Expression::Local(identifier.clone())),
            TokenKind::Punctuation('(') => {
                let expression = self.parse_assignment()?;
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    Some(TokenKind::Punctuation(')')) => {
                        self.index += 1;
                        Ok(expression)
                    }
                    found => Err(compile_error(format!(
                        "expected ')' after expression; found {found:?}; next {:?}",
                        self.tokens.get(self.index + 1).map(|token| &token.kind),
                    ))),
                }
            }
            _ => Err(compile_error(format!(
                "unexpected token {:?} in expression",
                token.kind
            ))),
        }
    }

    /// Parses BYOND's compile-time `nameof(reference)` form.
    ///
    /// The argument is a reference grammar rather than an ordinary runtime
    /// expression.  In particular, tgstation uses all of these shapes:
    /// `nameof(.proc/name)`, `nameof(/datum/example.proc/name)`, and
    /// `nameof(type::field)`.  Each evaluates to the referenced member's
    /// final textual component.  Retaining that component is sufficient for
    /// headless callback and signal registration and also supports
    /// `NAMEOF_STATIC` without pretending its compile-time reference is a
    /// datum field read.
    fn parse_nameof_expression(&mut self) -> Result<Expression, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut nesting = 0_usize;
        let mut final_name = None;
        loop {
            let token = self
                .tokens
                .get(self.index)
                .ok_or_else(|| compile_error("expected ')' after nameof reference"))?;
            match &token.kind {
                TokenKind::Punctuation('(') => nesting += 1,
                TokenKind::Punctuation(')') if nesting == 0 => {
                    self.index += 1;
                    break;
                }
                TokenKind::Punctuation(')') => nesting -= 1,
                TokenKind::Identifier(name) => final_name = Some(name.clone()),
                _ => {}
            }
            self.index += 1;
        }
        final_name
            .map(Expression::Text)
            .ok_or_else(|| compile_error("nameof requires a named reference"))
    }

    fn parse_call_arguments(&mut self) -> Result<Vec<Expression>, CompileError> {
        Ok(self
            .parse_named_call_arguments()?
            .into_iter()
            .map(|(name, expression)| {
                name.map_or(expression.clone(), |name| Expression::NamedArgument {
                    name,
                    value: Box::new(expression),
                })
            })
            .collect())
    }

    fn parse_named_call_arguments(
        &mut self,
    ) -> Result<Vec<(Option<String>, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ) {
            return Err(compile_error("expected '(' before call arguments"));
        }
        self.index += 1;
        let mut arguments = Vec::new();
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            loop {
                // Empty positional slots are legal DM and mean that the
                // callee's default value should be used.  The VM represents
                // an omitted interior slot as null until its call ABI carries
                // a distinct omission marker.
                if matches!(
                    self.tokens.get(self.index).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(','))
                ) {
                    arguments.push((None, Expression::Null));
                    self.index += 1;
                    if matches!(
                        self.tokens.get(self.index).map(|token| &token.kind),
                        Some(TokenKind::Punctuation(')'))
                    ) {
                        break;
                    }
                    continue;
                }
                // BYOND permits keyword-style call arguments, e.g.
                // `do_after(user, 4 SECONDS, target = src)`.  The current
                // execution ABI is positional, but retaining the source
                // order here is still the correct lowering for its existing
                // subset and, importantly, lets the compiler continue on to
                // report the next unsupported construct instead of rejecting
                // the call syntax itself.
                let name = match (
                    self.tokens.get(self.index).map(|token| &token.kind),
                    self.tokens.get(self.index + 1).map(|token| &token.kind),
                ) {
                    (Some(TokenKind::Identifier(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    (Some(TokenKind::String(name)), Some(TokenKind::Operator(operator)))
                        if operator == "=" =>
                    {
                        Some(name.clone())
                    }
                    _ => None,
                };
                if name.is_some() {
                    self.index += 2;
                }
                arguments.push((name, self.parse_assignment()?));
                match self.tokens.get(self.index).map(|token| &token.kind) {
                    // DM's weighted `pick()` syntax separates a weight from
                    // its candidate with `;`, e.g. `pick(10; red, 1; blue)`.
                    // The headless call ABI is positional, so retaining both
                    // expressions is the most faithful representation it can
                    // currently carry.
                    Some(TokenKind::Punctuation(',' | ';')) => {
                        self.index += 1;
                        // DM accepts a trailing separator in a parenthesized
                        // argument list, including multiline calls.  Do not
                        // attempt to parse the closing parenthesis as the
                        // next argument expression.
                        if matches!(
                            self.tokens.get(self.index).map(|token| &token.kind),
                            Some(TokenKind::Punctuation(')'))
                        ) {
                            break;
                        }
                    }
                    Some(TokenKind::Punctuation(')')) => break,
                    _ => {
                        return Err(compile_error(format!(
                            "expected ',' or ')' after procedure argument, received {:?}",
                            self.tokens.get(self.index).map(|token| &token.kind)
                        )));
                    }
                }
            }
        }
        self.index += 1;
        Ok(arguments)
    }

    /// Parses `pick()` entries while retaining its `weight; candidate` form.
    fn parse_pick_arguments(
        &mut self,
    ) -> Result<Vec<(Option<Expression>, Expression)>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            let first = self.parse_assignment()?;
            let entry = if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation(';'))
            ) {
                self.index += 1;
                (Some(first), self.parse_assignment()?)
            } else {
                (None, first)
            };
            entries.push(entry);
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after pick entry")),
            }
        }
        if entries.is_empty() {
            return Err(compile_error("pick requires at least one candidate"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn parse_modified_type_overrides(
        &mut self,
    ) -> Result<Vec<(FieldName, Expression)>, CompileError> {
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('{'))
        ) {
            return Ok(Vec::new());
        }
        self.index += 1;
        let mut overrides = Vec::new();
        loop {
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation('}'))
            ) {
                self.index += 1;
                return Ok(overrides);
            }
            let Some(TokenKind::Identifier(name)) =
                self.tokens.get(self.index).map(|token| &token.kind)
            else {
                return Err(compile_error("modified type requires a field name"));
            };
            let name = FieldName::parse(name).map_err(|error| compile_error(error.to_string()))?;
            self.index += 1;
            if !matches!(self.current_operator(), Some("=")) {
                return Err(compile_error("modified type field requires '='"));
            }
            self.index += 1;
            let start = self.index;
            let mut depth = 0_usize;
            while let Some(token) = self.tokens.get(self.index) {
                match token.kind {
                    TokenKind::Punctuation('(' | '[') => depth += 1,
                    TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
                    TokenKind::Punctuation('}' | ';') if depth == 0 => break,
                    _ => {}
                }
                self.index += 1;
            }
            if start == self.index {
                return Err(compile_error("modified type field value is empty"));
            }
            let value = ExpressionParser::new(&self.tokens[start..self.index]).parse()?;
            overrides.push((name, value));
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(';')) => self.index += 1,
                Some(TokenKind::Punctuation('}')) => {}
                _ => return Err(compile_error("modified type requires ';' or '}'")),
            }
        }
    }

    fn parse_list_arguments(&mut self) -> Result<Vec<ListExpressionEntry>, CompileError> {
        debug_assert!(matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        ));
        self.index += 1;
        let mut entries = Vec::new();
        while !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            // BYOND treats an omitted interior list/call argument as null,
            // while a single trailing comma contributes no extra entry.
            // Monk's species perk lists intentionally rely on this before
            // filtering nulls with list_clear_nulls().
            if matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Punctuation(','))
            ) {
                entries.push(ListExpressionEntry::Positional(Expression::Null));
                self.index += 1;
                continue;
            }
            // The unparenthesized `=` in a list literal introduces an
            // associative entry rather than an assignment expression. A
            // parenthesized assignment still reaches `parse_assignment` via
            // primary-expression parsing.
            let key_or_value = self.parse_conditional()?;
            if matches!(self.current_operator(), Some("=")) {
                self.index += 1;
                let value = self.parse_conditional()?;
                entries.push(ListExpressionEntry::Associative {
                    key: key_or_value,
                    value,
                });
            } else {
                entries.push(ListExpressionEntry::Positional(key_or_value));
            }
            match self.tokens.get(self.index).map(|token| &token.kind) {
                Some(TokenKind::Punctuation(',')) => self.index += 1,
                Some(TokenKind::Punctuation(')')) => break,
                _ => return Err(compile_error("expected ',' or ')' after list entry")),
            }
        }
        if !matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        ) {
            return Err(compile_error("expected ')' after list entries"));
        }
        self.index += 1;
        Ok(entries)
    }

    fn current_operator(&self) -> Option<&str> {
        match self.tokens.get(self.index).map(|token| &token.kind) {
            Some(TokenKind::Operator(operator)) => Some(operator),
            Some(TokenKind::Identifier(identifier)) if identifier == "in" => Some(identifier),
            _ => None,
        }
    }
}

/// Classifies BYOND's four `replacetext` builtin spellings without treating
/// them as project procedures.  The `_char` variants use character positions,
/// while `Ex` means exact (case-sensitive) matching.
fn replacetext_kind(identifier: &str) -> Option<(bool, bool)> {
    match identifier {
        "replacetext" => Some((false, false)),
        "replacetextEx" => Some((true, false)),
        "replacetext_char" => Some((false, true)),
        "replacetextEx_char" => Some((true, true)),
        _ => None,
    }
}

/// Identifies the compiler-handled BYOND value predicates.
fn type_predicate_kind(identifier: &str) -> Option<TypePredicateKind> {
    match identifier {
        "isnull" => Some(TypePredicateKind::IsNull),
        "isnum" => Some(TypePredicateKind::IsNum),
        "ispath" => Some(TypePredicateKind::IsPath),
        "islist" => Some(TypePredicateKind::IsList),
        "ismovable" => Some(TypePredicateKind::IsMovable),
        "isturf" => Some(TypePredicateKind::IsTurf),
        "isloc" => Some(TypePredicateKind::IsLoc),
        "isicon" => Some(TypePredicateKind::IsIcon),
        "istype" => Some(TypePredicateKind::IsType),
        _ => None,
    }
}

fn parse_number(spelling: &str) -> Result<DmNumberBits, CompileError> {
    let normalized = spelling.replace('_', "");
    let value = if matches!(normalized.as_str(), "1#INF" | "1.#INF") {
        f32::INFINITY
    } else if matches!(normalized.as_str(), "1#IND" | "1.#IND") {
        f32::NAN
    } else if let Some(hexadecimal) = normalized
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

fn parse_interpolated_string(text: &str) -> Result<Expression, CompileError> {
    const ESCAPED_OPEN: char = '\u{e000}';
    const ESCAPED_CLOSE: char = '\u{e001}';
    // Protect escaped brackets before looking for interpolation holes. This
    // must consume escape pairs rather than using `str::replace`: in
    // `\\\\[value]` the first pair denotes a literal backslash and the
    // bracket still begins interpolation.
    let mut protected = String::with_capacity(text.len());
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character != '\\' {
            protected.push(character);
            continue;
        }
        let Some(escaped) = characters.next() else {
            protected.push('\\');
            break;
        };
        match escaped {
            '[' => protected.push(ESCAPED_OPEN),
            ']' => protected.push(ESCAPED_CLOSE),
            _ => {
                protected.push('\\');
                protected.push(escaped);
            }
        }
    }
    let text = protected;
    let literal = |text: &str| {
        decode_quoted_text_fragment(&text.replace(ESCAPED_OPEN, "[").replace(ESCAPED_CLOSE, "]"))
    };
    let mut template = String::with_capacity(text.len());
    let mut interpolations = Vec::new();
    let mut cursor = 0_usize;
    while let Some(relative_open) = text[cursor..].find('[') {
        let open = cursor + relative_open;
        let Some(close) = interpolated_expression_close(&text, open + 1) else {
            break;
        };
        if text[open + 1..close].trim().is_empty() {
            cursor = close + 1;
            continue;
        }
        if open > cursor {
            template.push_str(&literal(&text[cursor..open]));
        }
        let tokens = lex(&text[open + 1..close])
            .map_err(|error| {
                compile_error(format!("invalid embedded expression: {}", error.message))
            })?
            .into_iter()
            .filter(|token| {
                !matches!(
                    token.kind,
                    TokenKind::LineStart { .. } | TokenKind::Newline | TokenKind::LineContinuation
                )
            })
            .collect::<Vec<_>>();
        template.push_str("[]");
        interpolations.push(ExpressionParser::new(&tokens).parse()?);
        cursor = close + 1;
    }
    if interpolations.is_empty() {
        return Ok(Expression::Text(literal(&text)));
    }
    if cursor < text.len() {
        template.push_str(&literal(&text[cursor..]));
    }
    let mut arguments = Vec::with_capacity(interpolations.len() + 1);
    arguments.push(Expression::Text(template));
    arguments.extend(interpolations);
    Ok(Expression::StandardBuiltin {
        name: "text".to_owned(),
        arguments,
    })
}

/// Decode escapes in an ordinary double-quoted DM string fragment. Raw
/// strings are represented by a different token kind and intentionally never
/// pass through this function.
fn decode_quoted_text_fragment(text: &str) -> String {
    let mut decoded = String::with_capacity(text.len());
    let mut cursor = 0_usize;
    while cursor < text.len() {
        let character = text[cursor..]
            .chars()
            .next()
            .expect("cursor is inside quoted text");
        cursor += character.len_utf8();
        if character != '\\' {
            decoded.push(character);
            continue;
        }
        if cursor == text.len() {
            decoded.push('\\');
            break;
        }

        // Text macros win over their single-character escape prefixes:
        // `\\the` and `\\th` are format operations, while `\\t` is a tab.
        let remaining = &text[cursor..];
        let macro_match = [
            ("improper", TEXT_MACRO_IMPROPER, true),
            ("himself", TEXT_MACRO_REFLEXIVE, false),
            ("Himself", TEXT_MACRO_REFLEXIVE, false),
            ("herself", TEXT_MACRO_REFLEXIVE, false),
            ("Herself", TEXT_MACRO_REFLEXIVE, false),
            ("proper", TEXT_MACRO_PROPER, true),
            ("Roman", TEXT_MACRO_ROMAN_UPPER, true),
            ("roman", TEXT_MACRO_ROMAN, true),
            ("Hers", TEXT_MACRO_POSSESSIVE_UPPER, false),
            ("hers", TEXT_MACRO_POSSESSIVE, false),
            ("The", TEXT_MACRO_THE_UPPER, true),
            ("the", TEXT_MACRO_THE, true),
            ("She", TEXT_MACRO_SUBJECT_UPPER, false),
            ("she", TEXT_MACRO_SUBJECT, false),
            ("His", TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER, false),
            ("his", TEXT_MACRO_POSSESSIVE_ADJECTIVE, false),
            ("him", TEXT_MACRO_OBJECT, false),
            ("An", TEXT_MACRO_A_UPPER, true),
            ("an", TEXT_MACRO_A, true),
            ("He", TEXT_MACRO_SUBJECT_UPPER, false),
            ("he", TEXT_MACRO_SUBJECT, false),
            ("th", TEXT_MACRO_ORDINAL, false),
            ("A", TEXT_MACRO_A_UPPER, true),
            ("a", TEXT_MACRO_A, true),
            ("s", TEXT_MACRO_PLURAL, false),
        ]
        .into_iter()
        .find(|(spelling, _, _)| remaining.starts_with(spelling));
        if let Some((spelling, marker, prefix)) = macro_match {
            decoded.push(marker);
            cursor += spelling.len();
            if prefix && text[cursor..].starts_with(' ') {
                cursor += 1;
            }
            continue;
        }

        let escaped = text[cursor..]
            .chars()
            .next()
            .expect("checked escaped text exists");
        cursor += escaped.len_utf8();
        match escaped {
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            '\\' => decoded.push('\\'),
            '"' => decoded.push('"'),
            '[' => decoded.push('['),
            ']' => decoded.push(']'),
            // BYOND has additional display-format escapes (for example
            // `\\the` and `\\proper`). Keep those intact until the text
            // formatting layer interprets them instead of silently deleting
            // their escape marker.
            other => {
                decoded.push('\\');
                decoded.push(other);
            }
        }
    }
    decoded
}

pub(crate) fn interpolated_expression_close(text: &str, start: usize) -> Option<usize> {
    let mut depth = 1usize;
    let mut cursor = start;
    let mut quote = None;
    let mut escaped = false;
    while cursor < text.len() {
        let character = text[cursor..].chars().next()?;
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
        } else {
            match character {
                '"' | '\'' => quote = Some(character),
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(cursor);
                    }
                }
                _ => {}
            }
        }
        cursor += character.len_utf8();
    }
    None
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"|" => Some(3),
        b"^" => Some(4),
        b"&" => Some(5),
        b"==" | b"!=" | b"<>" | b"~=" | b"~!" => Some(6),
        b"<<" | b">>" | b"<" | b"<=" | b">" | b">=" | b"<=>" | b"in" => Some(7),
        b"+" | b"-" => Some(8),
        b"*" | b"/" | b"%" | b"%%" => Some(9),
        b"**" => Some(10),
        _ => None,
    }
}

#[allow(clippy::too_many_lines)]
/// Emits an associative-list key, preserving macro-expanded named arguments.
///
/// Macro wrappers such as `AddComponent(...)` expand named arguments into
/// `list(name = value)`. The original call grammar is no longer visible, so
/// an unbound bare name here is a textual associative key, not an assignment
/// target. Bound locals and fields retain their ordinary expression meaning.
fn emit_associative_list_key(
    key: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    // In DM's list-constructor grammar, a bare identifier to the left of `=`
    // is a named/text key even when a local, field, or global with the same
    // spelling exists. Dynamic keys use an explicit expression instead.
    if let Expression::Local(name) = key {
        instructions.push(Instruction::PushText(Arc::from(name.as_str())));
        return Ok(());
    }
    emit_expression(key, locals, instructions, procedures)
}

/// Marker used by call-like instructions to consume the count produced by
/// [`Instruction::ExpandArgumentLists`].  A source procedure cannot have
/// this many arguments, so it is unambiguous in the compact bytecode ABI.
pub(crate) const EXPANDED_ARGUMENT_COUNT: u16 = u16::MAX;

/// Emits a call argument vector, retaining BYOND's runtime `arglist()`
/// expansion semantics.  Ordinary expressions preserve the compact static
/// count; an expansion emits a small preparation instruction and returns the
/// sentinel consumed by the following call-like instruction.
fn emit_call_arguments(
    arguments: &[Expression],
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<u16, CompileError> {
    let argument_count = u16::try_from(arguments.len())
        .map_err(|_| compile_error("call has more than 65535 positional arguments"))?;
    let mut expanded_indices = Vec::new();
    for (index, argument) in arguments.iter().enumerate() {
        if let Expression::ArgList(value) = argument {
            expanded_indices.push(to_local_index(index)?);
            emit_expression(value, locals, instructions, procedures)?;
        } else if let Expression::NamedArgument { value, .. } = argument {
            emit_expression(value, locals, instructions, procedures)?;
        } else {
            emit_expression(argument, locals, instructions, procedures)?;
        }
    }
    if expanded_indices.is_empty() {
        Ok(argument_count)
    } else {
        instructions.push(Instruction::ExpandArgumentLists {
            argument_count,
            argument_names: arguments
                .iter()
                .map(|argument| match argument {
                    Expression::NamedArgument { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect(),
            expanded_indices,
        });
        Ok(EXPANDED_ARGUMENT_COUNT)
    }
}

#[allow(clippy::too_many_lines)]
fn emit_expression(
    expression: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match expression {
        Expression::NamedArgument { value, .. } => {
            emit_expression(value, locals, instructions, procedures)?;
        }
        Expression::Null => instructions.push(Instruction::PushNull),
        Expression::Number(number) => instructions.push(Instruction::PushNumber(*number)),
        Expression::Text(text) => {
            instructions.push(Instruction::PushText(Arc::from(text.as_str())))
        }
        Expression::File(path) => instructions.push(Instruction::PushFile(path.clone())),
        Expression::TypePath(path) => instructions.push(Instruction::PushTypePath(path.clone())),
        Expression::ModifiedTypePath { base, overrides } => {
            instructions.push(Instruction::PushTypePath(base.clone()));
            for (_, value) in overrides {
                emit_expression(value, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeModifiedTypePath {
                fields: overrides
                    .iter()
                    .map(|(field, _)| field.clone())
                    .collect::<Vec<_>>()
                    .into(),
            });
        }
        Expression::New {
            type_path,
            arguments,
            overrides,
        } => {
            let Some(type_path) = type_path else {
                return Err(compile_error(
                    "inferred new has no statically resolved destination type",
                ));
            };
            emit_expression(type_path, locals, instructions, procedures)?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::AllocateDatum {
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
            });
            for (name, value) in overrides {
                instructions.push(Instruction::Duplicate);
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::StoreField(name.clone()));
            }
        }
        Expression::Regex { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeRegex {
                argument_count: u8::try_from(arguments.len())
                    .expect("regex argument count was validated by the parser"),
            });
        }
        Expression::MutableAppearance { arguments } => {
            // SS13 projects commonly provide `/proc/mutable_appearance` as a
            // behavior-rich helper around the engine datum. Just like qdel
            // and the other engine fallbacks, that project procedure wins.
            // This is especially important for named arguments: Monkestation's
            // human overlay path supplies `layer` and `appearance_flags`, and
            // the helper also applies its omitted defaults before returning.
            if let Some(procedure) = procedures.get("mutable_appearance").copied() {
                let argument_count =
                    emit_call_arguments(arguments, locals, instructions, procedures)?;
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            } else {
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                instructions.push(Instruction::MakeMutableAppearance {
                    argument_count: u16::try_from(arguments.len()).map_err(|_| {
                        compile_error(
                            "mutable_appearance has more than 65535 constructor arguments",
                        )
                    })?,
                });
            }
        }
        Expression::Matrix { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeMatrix {
                argument_count: u8::try_from(arguments.len())
                    .expect("matrix argument count was validated"),
            });
        }
        Expression::Vector { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::MakeVector {
                argument_count: u8::try_from(arguments.len())
                    .expect("vector argument count was validated"),
            });
        }
        Expression::ReplaceText {
            arguments,
            exact,
            character_indices,
        } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ReplaceText {
                argument_count: u8::try_from(arguments.len())
                    .expect("replacetext argument count was validated by the parser"),
                exact: *exact,
                character_indices: *character_indices,
            });
        }
        Expression::CopyText {
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
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Length);
        }
        Expression::Ref { value } => {
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Ref);
        }
        Expression::GetStep { source, direction } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(direction, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStep);
        }
        Expression::GetStepTowards { source, target } => {
            emit_expression(source, locals, instructions, procedures)?;
            emit_expression(target, locals, instructions, procedures)?;
            instructions.push(Instruction::GetStepTowards);
        }
        Expression::Range { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Range {
                argument_count: u8::try_from(arguments.len())
                    .expect("range argument count was validated by the parser"),
            });
        }
        Expression::TypesOf { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::TypesOf {
                argument_count: u8::try_from(arguments.len())
                    .expect("typesof argument count was validated by the parser"),
            });
        }
        Expression::HasCall { receiver, selector } => {
            emit_expression(receiver, locals, instructions, procedures)?;
            emit_expression(selector, locals, instructions, procedures)?;
            instructions.push(Instruction::HasCall);
        }
        Expression::Rand { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Rand {
                argument_count: u8::try_from(arguments.len())
                    .expect("rand argument count was validated by the parser"),
            });
        }
        Expression::Roll { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Roll {
                argument_count: u8::try_from(arguments.len())
                    .expect("roll argument count was validated by the parser"),
            });
        }
        Expression::Pick { entries } => {
            if let [(None, Expression::ArgList(value))] = entries.as_slice() {
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::ExpandArgumentLists {
                    argument_count: 1,
                    argument_names: vec![None],
                    expanded_indices: vec![0],
                });
                instructions.push(Instruction::PickExpandedArguments);
                return Ok(());
            }
            let mut weighted = Vec::with_capacity(entries.len());
            for (weight, candidate) in entries {
                weighted.push(weight.is_some());
                if let Some(weight) = weight {
                    emit_expression(weight, locals, instructions, procedures)?;
                }
                emit_expression(candidate, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Pick { weighted });
        }
        Expression::Prob(chance) => {
            emit_expression(chance, locals, instructions, procedures)?;
            instructions.push(Instruction::Prob);
        }
        Expression::Round { arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Round {
                argument_count: u8::try_from(arguments.len())
                    .expect("round argument count was validated by the parser"),
            });
        }
        Expression::TypePredicate { kind, arguments } => {
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            let inferred_type = (*kind == TypePredicateKind::IsType && arguments.len() == 1)
                .then(|| expression_static_type(&arguments[0], locals))
                .flatten();
            let argument_count = arguments.len() + usize::from(inferred_type.is_some());
            if let Some(type_path) = inferred_type {
                instructions.push(Instruction::PushTypePath(type_path));
            }
            instructions.push(Instruction::TypePredicate {
                kind: *kind,
                argument_count: u8::try_from(argument_count)
                    .expect("predicate argument count was validated by the parser"),
            });
        }
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::LoadLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LoadField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LoadGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::Src => instructions.push(Instruction::LoadSrc),
        Expression::Usr => instructions.push(Instruction::LoadUsr),
        Expression::Caller => instructions.push(Instruction::LoadCaller),
        Expression::World => instructions.push(Instruction::LoadGlobal(
            FieldName::parse("world").expect("built-in world global name is valid"),
        )),
        Expression::GlobalNamespace => {
            return Err(compile_error("global namespace requires a field name"));
        }
        Expression::Field { receiver, name } => {
            if name.as_str() == "vars" {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadDatumVars);
            } else if let Some(storage) =
                locals.receiver_static(receiver.as_ref(), name).or_else(|| {
                    matches!(receiver.as_ref(), Expression::Src)
                        .then(|| locals.global_field(name.as_str()))
                        .flatten()
                })
            {
                instructions.push(Instruction::LoadGlobal(storage.clone()));
            } else {
                let declared = expression_static_type(receiver, locals).is_some();
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(if declared {
                    Instruction::LoadDeclaredField(name.clone())
                } else {
                    Instruction::LoadField(name.clone())
                });
            }
        }
        Expression::SafeField { receiver, name } => {
            let declared = expression_static_type(receiver, locals).is_some();
            emit_expression(receiver, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            let null_jump = instructions.len();
            instructions.push(Instruction::JumpIfNull(usize::MAX));
            instructions.push(if declared {
                Instruction::LoadDeclaredField(name.clone())
            } else {
                Instruction::LoadField(name.clone())
            });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::GlobalField(name) => {
            if name.as_str() == "vars" {
                instructions.push(Instruction::LoadGlobalVars);
            } else {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
        }
        Expression::Result => instructions.push(Instruction::LoadResult),
        Expression::ArgList(_) => {
            return Err(compile_error(
                "arglist may only appear in a call or constructor argument list",
            ));
        }
        Expression::StandardBuiltin { name, arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("native builtin has more than 65535 arguments"))?;
            // `newlist(/type, ...)` is syntax sugar for constructing each
            // argument with an ordinary zero-argument `new`, then collecting
            // the resulting objects into a list.  Lower it to AllocateDatum
            // so inherited defaults, instance initializers, New(), scheduler
            // suspension, and atom registration are identical to explicit
            // construction.  A project-defined /proc/newlist still wins.
            if name == "newlist" && !procedures.contains_key(name) {
                for argument in arguments {
                    if matches!(argument, Expression::NamedArgument { .. }) {
                        return Err(compile_error("newlist does not take named arguments"));
                    }
                    emit_expression(argument, locals, instructions, procedures)?;
                    instructions.push(Instruction::AllocateDatum {
                        argument_count: 0,
                        argument_names: Vec::new(),
                    });
                }
                instructions.push(Instruction::MakeList(argument_count));
                return Ok(());
            }
            for argument in arguments {
                if let Expression::ArgList(value) = argument {
                    // A single expanded list is already the native ABI used
                    // by list-aware builtins such as min/max.
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            // DM source may deliberately replace a global procedure whose name
            // also has an engine fallback (tgstation's /proc/qdel is the
            // important case). A real project procedure wins over the native
            // fallback exactly like any other global proc declaration.
            if let Some(procedure) = procedures.get(name).copied() {
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            } else {
                instructions.push(Instruction::StandardBuiltin {
                    name: name.clone(),
                    argument_count,
                    argument_names: arguments
                        .iter()
                        .map(|argument| match argument {
                            Expression::NamedArgument { name, .. } => Some(name.clone()),
                            _ => None,
                        })
                        .collect(),
                });
            }
        }
        Expression::NativeSrcMethod { name, arguments } => {
            // Several BYOND engine method names are also valid project proc
            // names. Monkestation's legacy spritesheet datum declares
            // `/datum/asset/spritesheet/proc/Insert`; that project method must
            // win over `/icon.Insert`, just like project global builtins do.
            if let Some(procedure) = procedures.get(name).copied() {
                let argument_count =
                    emit_call_arguments(arguments, locals, instructions, procedures)?;
                instructions.push(Instruction::Call {
                    procedure,
                    argument_count,
                    argument_names: vec![None; arguments.len()],
                });
            } else {
                let argument_count = u16::try_from(arguments.len())
                    .map_err(|_| compile_error("native method has more than 65535 arguments"))?;
                for argument in arguments {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
                instructions.push(Instruction::NativeSrcMethod {
                    name: name.clone(),
                    argument_count,
                });
            }
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            emit_expression(library, locals, instructions, procedures)?;
            emit_expression(function, locals, instructions, procedures)?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::ExternalCall {
                argument_count: u16::try_from(arguments.len())
                    .map_err(|_| compile_error("external call has more than 65535 arguments"))?,
            });
        }
        Expression::Animate { arguments } => {
            let mut expanded_indices = Vec::new();
            for (index, (_, argument)) in arguments.iter().enumerate() {
                if let Expression::ArgList(value) = argument {
                    expanded_indices.push(to_local_index(index)?);
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            instructions.push(Instruction::Animate {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
                expanded_indices,
            });
        }
        Expression::Filter { arguments } => {
            let mut expanded_indices = Vec::new();
            for (index, (_, argument)) in arguments.iter().enumerate() {
                if let Expression::ArgList(value) = argument {
                    expanded_indices.push(to_local_index(index)?);
                    emit_expression(value, locals, instructions, procedures)?;
                } else {
                    emit_expression(argument, locals, instructions, procedures)?;
                }
            }
            instructions.push(Instruction::MakeFilter {
                argument_names: arguments.iter().map(|(name, _)| name.clone()).collect(),
                expanded_indices,
            });
        }
        Expression::Crash(message) => {
            emit_expression(message, locals, instructions, procedures)?;
            instructions.push(Instruction::Crash);
            // Keep expression stack shape valid for unreachable continuation.
            instructions.push(Instruction::PushNull);
        }
        Expression::Sleep(delay) => {
            emit_expression(delay, locals, instructions, procedures)?;
            instructions.push(Instruction::Sleep);
        }
        Expression::Initial(reference) => match reference.as_ref() {
            Expression::Field { receiver, name } => {
                if let Some(storage) = locals.receiver_static(receiver, name) {
                    // Static initialization is materialized before procedures
                    // run and occupies its qualified persistent slot.
                    instructions.push(Instruction::LoadInitialGlobal(storage.clone()));
                } else {
                    emit_expression(receiver, locals, instructions, procedures)?;
                    instructions.push(Instruction::InitialField(name.clone()));
                }
            }
            Expression::Local(name) => {
                let field = locals.src_field(name).ok_or_else(|| {
                    compile_error(format!("initial target {name:?} is not an instance field"))
                })?;
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::InitialField(field.clone()));
            }
            Expression::SafeField { receiver, name } => {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let null_jump = instructions.len();
                instructions.push(Instruction::JumpIfNull(usize::MAX));
                instructions.push(Instruction::InitialField(name.clone()));
                let end = instructions.len();
                instructions[null_jump] = Instruction::JumpIfNull(end);
            }
            Expression::Index { list, index } if matches!(list.as_ref(), Expression::Field { name, .. } if name.as_str() == "vars") =>
            {
                let Expression::Field { receiver, .. } = list.as_ref() else {
                    unreachable!("vars index guard established a field receiver")
                };
                emit_expression(receiver, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::InitialDynamicField);
            }
            _ => return Err(compile_error("initial requires a field reference")),
        },
        Expression::Call {
            procedure,
            arguments,
        } => {
            let target = procedures
                .get(procedure)
                .copied()
                .ok_or_else(|| compile_error(format!("unknown procedure {procedure:?}")))?;
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::Call {
                procedure: target,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
            });
        }
        Expression::Locate { arguments } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            instructions.push(Instruction::Locate { argument_count });
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            let argument_count = u16::try_from(arguments.len())
                .map_err(|_| compile_error("locate has more than 65535 positional arguments"))?;
            for argument in arguments {
                emit_expression(argument, locals, instructions, procedures)?;
            }
            emit_expression(container, locals, instructions, procedures)?;
            instructions.push(Instruction::LocateIn { argument_count });
        }
        Expression::CurrentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallCurrent { argument_count });
        }
        Expression::ParentCall { arguments } => {
            let argument_count = if let Some(arguments) = arguments {
                Some(emit_call_arguments(
                    arguments,
                    locals,
                    instructions,
                    procedures,
                )?)
            } else {
                None
            };
            instructions.push(Instruction::CallParent {
                procedure: procedures.get("..").copied(),
                argument_count,
            });
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
            null_receiver_is_global,
        } => {
            emit_expression(target, locals, instructions, procedures)?;
            let static_selector = if let Expression::Text(selector) = procedure.as_ref() {
                Some(selector.clone())
            } else {
                emit_expression(procedure, locals, instructions, procedures)?;
                None
            };
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic {
                static_selector,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
                null_receiver_is_global: *null_receiver_is_global,
            });
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
            let static_selector = if let Expression::Text(selector) = procedure.as_ref() {
                Some(selector.clone())
            } else {
                emit_expression(procedure, locals, instructions, procedures)?;
                None
            };
            let argument_count = emit_call_arguments(arguments, locals, instructions, procedures)?;
            instructions.push(Instruction::CallDynamic {
                static_selector,
                argument_count,
                argument_names: arguments
                    .iter()
                    .map(|argument| match argument {
                        Expression::NamedArgument { name, .. } => Some(name.clone()),
                        _ => None,
                    })
                    .collect(),
                null_receiver_is_global: false,
            });
            let end = instructions.len();
            instructions[null_jump] = Instruction::JumpIfNull(end);
        }
        Expression::List(entries) => {
            let mut kinds = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Positional);
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        emit_associative_list_key(key, locals, instructions, procedures)?;
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Associative);
                    }
                }
            }
            instructions.push(Instruction::MakeListEntries(kinds));
        }
        Expression::AssociativeList(entries) => {
            let mut kinds = Vec::with_capacity(entries.len());
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Positional);
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        emit_associative_list_key(key, locals, instructions, procedures)?;
                        emit_expression(value, locals, instructions, procedures)?;
                        kinds.push(ListEntryKind::Associative);
                    }
                }
            }
            instructions.push(Instruction::MakeAssociativeListEntries(kinds));
        }
        Expression::Index { list, index } => {
            if let Expression::Field { receiver, name } = list.as_ref()
                && name.as_str() == "vars"
            {
                emit_expression(receiver, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::LoadDynamicField);
            } else {
                if let Expression::Local(name) = list.as_ref()
                    && let Some(slot) = locals.get(name)
                {
                    emit_expression(index, locals, instructions, procedures)?;
                    instructions.push(Instruction::IndexLocalList(slot));
                } else {
                    emit_expression(list, locals, instructions, procedures)?;
                    emit_expression(index, locals, instructions, procedures)?;
                    instructions.push(Instruction::IndexList);
                }
            }
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
            if operator == "&"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::AddressLocal(slot));
                return Ok(());
            }
            if operator == "*"
                && let Expression::Local(name) = operand.as_ref()
                && let Some(slot) = locals.get(name)
            {
                instructions.push(Instruction::LoadLocalRaw(slot));
                instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                instructions.push(Instruction::IndexList);
                return Ok(());
            }
            emit_expression(operand, locals, instructions, procedures)?;
            match operator.as_str() {
                "+" => {}
                "-" => instructions.push(Instruction::Negate),
                "!" => instructions.push(Instruction::Not),
                "~" => instructions.push(Instruction::BitNot),
                "&" => instructions.push(Instruction::MakeList(1)),
                "*" => {
                    instructions.push(Instruction::PushNumber(DmNumberBits::from_f32(1.0)));
                    instructions.push(Instruction::IndexList);
                }
                _ => {
                    return Err(compile_error(format!(
                        "unsupported unary operator {operator}"
                    )));
                }
            }
        }
        Expression::Mutation {
            target,
            delta,
            prefix,
        } => emit_mutation_expression(target, *delta, *prefix, locals, instructions, procedures)?,
        Expression::Binary {
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
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            emit_expression(condition, locals, instructions, procedures)?;
            let false_jump = instructions.len();
            instructions.push(Instruction::JumpIfFalse(usize::MAX));
            emit_expression(when_true, locals, instructions, procedures)?;
            let end_jump = instructions.len();
            instructions.push(Instruction::Jump(usize::MAX));
            let false_target = instructions.len();
            patch_jump(instructions, false_jump, false_target)?;
            emit_expression(when_false, locals, instructions, procedures)?;
            let end_target = instructions.len();
            patch_jump(instructions, end_jump, end_target)?;
        }
        Expression::LogicalOrAssignment { target, value } => {
            if !matches!(value.as_ref(), Expression::List(entries) if entries.is_empty())
                || !emit_logical_or_empty_list_assignment(target, locals, instructions, procedures)?
            {
                // Keep every non-empty RHS on the general logical-assignment
                // lowering. The superinstructions below are deliberately
                // exact to the overwhelmingly common empty-list constructor.
                emit_expression(target, locals, instructions, procedures)?;
                let false_jump = instructions.len();
                instructions.push(Instruction::JumpIfFalse(usize::MAX));
                emit_expression(target, locals, instructions, procedures)?;
                let end_jump = instructions.len();
                instructions.push(Instruction::Jump(usize::MAX));
                let false_target = instructions.len();
                patch_jump(instructions, false_jump, false_target)?;
                emit_assignment_expression(target, "=", value, locals, instructions, procedures)?;
                let end_target = instructions.len();
                patch_jump(instructions, end_jump, end_target)?;
            }
        }
        Expression::Assignment {
            target,
            operator,
            value,
        } => emit_assignment_expression(target, operator, value, locals, instructions, procedures)?,
    }
    Ok(())
}

fn emit_logical_or_empty_list_assignment(
    target: &Expression,
    locals: &LocalTable<'_>,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<bool, CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::LogicalOrEmptyListLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::LogicalOrEmptyListField(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::LogicalOrEmptyListGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) if name.as_str() != "vars" => {
            instructions.push(Instruction::LogicalOrEmptyListGlobal(name.clone()));
        }
        Expression::Field { receiver, name } if name.as_str() != "vars" => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name).or_else(|| {
                matches!(receiver.as_ref(), Expression::Src)
                    .then(|| locals.global_field(name.as_str()))
                    .flatten()
            }) {
                instructions.push(Instruction::LogicalOrEmptyListGlobal(storage.clone()));
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::LogicalOrEmptyListField(name.clone()));
            }
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::LogicalOrEmptyListIndex);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn emit_mutation_expression(
    target: &Expression,
    delta: i8,
    prefix: bool,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                instructions.push(Instruction::MutateLocal {
                    slot,
                    delta,
                    prefix,
                });
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                instructions.push(Instruction::MutateField {
                    name: field.clone(),
                    delta,
                    prefix,
                });
            } else if let Some(global) = locals.global_field(name) {
                instructions.push(Instruction::MutateGlobal {
                    name: global.clone(),
                    delta,
                    prefix,
                });
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => instructions.push(Instruction::MutateGlobal {
            name: name.clone(),
            delta,
            prefix,
        }),
        Expression::Field { receiver, name } => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name) {
                instructions.push(Instruction::MutateGlobal {
                    name: storage.clone(),
                    delta,
                    prefix,
                });
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::MutateField {
                    name: name.clone(),
                    delta,
                    prefix,
                });
            }
        }
        Expression::SafeField { receiver, name } => {
            if let Some(storage) = locals.receiver_static(receiver.as_ref(), name) {
                instructions.push(Instruction::MutateGlobal {
                    name: storage.clone(),
                    delta,
                    prefix,
                });
            } else {
                emit_expression(receiver, locals, instructions, procedures)?;
                instructions.push(Instruction::Duplicate);
                let null_jump = instructions.len();
                instructions.push(Instruction::JumpIfNull(usize::MAX));
                instructions.push(Instruction::MutateField {
                    name: name.clone(),
                    delta,
                    prefix,
                });
                let end = instructions.len();
                instructions[null_jump] = Instruction::JumpIfNull(end);
            }
        }
        Expression::Index { list, index } => {
            emit_expression(list, locals, instructions, procedures)?;
            emit_expression(index, locals, instructions, procedures)?;
            instructions.push(Instruction::MutateListIndex { delta, prefix });
        }
        Expression::Result => instructions.push(Instruction::MutateResult { delta, prefix }),
        _ => return Err(compile_error("increment/decrement target is not writable")),
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn emit_assignment_expression(
    target: &Expression,
    operator: &str,
    value: &Expression,
    locals: &LocalTable,
    instructions: &mut Vec<Instruction>,
    procedures: &HashMap<String, ProcedureId>,
) -> Result<(), CompileError> {
    match target {
        Expression::Result => {
            if operator != "=" {
                instructions.push(Instruction::LoadResult);
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreResult);
        }
        Expression::Usr => {
            if operator != "=" {
                return Err(compile_error("usr only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreUsr);
        }
        Expression::Local(name) => {
            if let Some(slot) = locals.get(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadLocal(slot));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreLocal(slot));
            } else if let Some(field) = locals.src_field(name) {
                instructions.push(Instruction::LoadSrc);
                if operator != "=" {
                    instructions.push(Instruction::Duplicate);
                    instructions.push(Instruction::LoadField(field.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::StoreFieldKeep(field.clone()));
            } else if let Some(global) = locals.global_field(name) {
                if operator != "=" {
                    instructions.push(Instruction::LoadGlobal(global.clone()));
                }
                emit_expression(value, locals, instructions, procedures)?;
                if operator != "=" {
                    instructions.push(compound_instruction(operator)?);
                }
                instructions.push(Instruction::Duplicate);
                instructions.push(Instruction::StoreGlobal(global.clone()));
            } else {
                return Err(compile_error(format!("unknown local {name:?}")));
            }
        }
        Expression::GlobalField(name) => {
            if operator != "=" {
                instructions.push(Instruction::LoadGlobal(name.clone()));
            }
            emit_expression(value, locals, instructions, procedures)?;
            if operator != "=" {
                instructions.push(compound_instruction(operator)?);
            }
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreGlobal(name.clone()));
        }
        Expression::Src => {
            if operator != "=" {
                return Err(compile_error("src only supports direct assignment"));
            }
            emit_expression(value, locals, instructions, procedures)?;
            instructions.push(Instruction::Duplicate);
            instructions.push(Instruction::StoreSrc);
        }
        Expression::Field { receiver, name } => {
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
            if operator == "=" {
                emit_expression(value, locals, instructions, procedures)?;
                emit_expression(list, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                instructions.push(Instruction::PrepareRhsFirstIndexAssignment);
                instructions.push(Instruction::SetListIndexKeep);
            } else {
                emit_expression(list, locals, instructions, procedures)?;
                emit_expression(index, locals, instructions, procedures)?;
                emit_expression(value, locals, instructions, procedures)?;
                instructions.push(Instruction::CompoundListIndexKeep(
                    compound_list_index_operator(operator)?,
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
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn bind_initializer_expression(
    expression: &mut Expression,
    bindings: &BTreeMap<String, InitializerBinding>,
) -> Result<(), CompileError> {
    match expression {
        Expression::World => {}
        Expression::Local(name) => {
            let binding = bindings
                .get(name)
                .ok_or_else(|| compile_error(format!("unresolved initializer name {name:?}")))?;
            *expression = match binding {
                InitializerBinding::Global(field) => Expression::GlobalField(field.clone()),
                InitializerBinding::SrcField(field) => Expression::Field {
                    receiver: Box::new(Expression::Src),
                    name: field.clone(),
                },
            };
        }
        Expression::Field { receiver, name } | Expression::SafeField { receiver, name } => {
            if let Expression::Local(receiver_name) = receiver.as_ref()
                && let Some(InitializerBinding::Global(storage)) =
                    bindings.get(&format!("{receiver_name}.{}", name.as_str()))
            {
                *expression = Expression::GlobalField(storage.clone());
            } else {
                bind_initializer_expression(receiver, bindings)?;
            }
        }
        Expression::NamedArgument { value, .. } => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::Call { arguments, .. }
        | Expression::StandardBuiltin { arguments, .. }
        | Expression::NativeSrcMethod { arguments, .. }
        | Expression::Regex { arguments }
        | Expression::MutableAppearance { arguments }
        | Expression::Matrix { arguments }
        | Expression::Vector { arguments }
        | Expression::ReplaceText { arguments, .. }
        | Expression::CopyText { arguments, .. }
        | Expression::Block { arguments }
        | Expression::Rand { arguments }
        | Expression::Roll { arguments }
        | Expression::Round { arguments }
        | Expression::Range { arguments }
        | Expression::TypePredicate { arguments, .. }
        | Expression::Locate { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::ExternalCall {
            library,
            function,
            arguments,
        } => {
            bind_initializer_expression(library, bindings)?;
            bind_initializer_expression(function, bindings)?;
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Animate { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::Filter { arguments } => {
            for (_, argument) in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::TypesOf { arguments } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
        }
        Expression::HasCall { receiver, selector } => {
            bind_initializer_expression(receiver, bindings)?;
            bind_initializer_expression(selector, bindings)?;
        }
        Expression::Length { value }
        | Expression::Ref { value }
        | Expression::Initial(value)
        | Expression::Sleep(value)
        | Expression::Crash(value) => {
            bind_initializer_expression(value, bindings)?;
        }
        Expression::ArgList(value) => bind_initializer_expression(value, bindings)?,
        Expression::GetStep { source, direction } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(direction, bindings)?;
        }
        Expression::GetStepTowards { source, target } => {
            bind_initializer_expression(source, bindings)?;
            bind_initializer_expression(target, bindings)?;
        }
        Expression::Prob(chance) => bind_initializer_expression(chance, bindings)?,
        Expression::Pick { entries } => {
            for (weight, candidate) in entries {
                if let Some(weight) = weight {
                    bind_initializer_expression(weight, bindings)?;
                }
                bind_initializer_expression(candidate, bindings)?;
            }
        }
        Expression::New {
            type_path,
            arguments,
            overrides,
        } => {
            if let Some(type_path) = type_path {
                bind_initializer_expression(type_path, bindings)?;
            }
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
            }
        }
        Expression::ModifiedTypePath { overrides, .. } => {
            for (_, value) in overrides {
                bind_initializer_expression(value, bindings)?;
            }
        }
        Expression::LocateIn {
            arguments,
            container,
        } => {
            for argument in arguments {
                bind_initializer_expression(argument, bindings)?;
            }
            bind_initializer_expression(container, bindings)?;
        }
        Expression::DynamicCall {
            target,
            procedure,
            arguments,
            ..
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
        Expression::List(entries) | Expression::AssociativeList(entries) => {
            for entry in entries {
                match entry {
                    ListExpressionEntry::Positional(value) => {
                        bind_initializer_expression(value, bindings)?;
                    }
                    ListExpressionEntry::Associative { key, value } => {
                        // A bare key in `list(name = value)` is named-argument
                        // syntax and therefore the text "name", even if an
                        // initializer binding with that spelling exists.
                        let bare_text_key = matches!(key, Expression::Local(_));
                        if !bare_text_key {
                            bind_initializer_expression(key, bindings)?;
                        }
                        bind_initializer_expression(value, bindings)?;
                    }
                }
            }
        }
        Expression::Index { list, index } | Expression::SafeIndex { list, index } => {
            bind_initializer_expression(list, bindings)?;
            bind_initializer_expression(index, bindings)?;
        }
        Expression::Unary { operand, .. }
        | Expression::Mutation {
            target: operand, ..
        } => {
            bind_initializer_expression(operand, bindings)?;
        }
        Expression::Binary { left, right, .. } => {
            bind_initializer_expression(left, bindings)?;
            bind_initializer_expression(right, bindings)?;
        }
        Expression::Conditional {
            condition,
            when_true,
            when_false,
        } => {
            bind_initializer_expression(condition, bindings)?;
            bind_initializer_expression(when_true, bindings)?;
            bind_initializer_expression(when_false, bindings)?;
        }
        Expression::LogicalOrAssignment { target, value }
        | Expression::Assignment { target, value, .. } => {
            bind_initializer_expression(target, bindings)?;
            bind_initializer_expression(value, bindings)?;
        }
        Expression::CurrentCall { .. }
        | Expression::ParentCall { .. }
        | Expression::Result
        | Expression::Caller => {
            return Err(compile_error(
                "current-procedure state is unavailable in a variable initializer",
            ));
        }
        Expression::Null
        | Expression::Number(_)
        | Expression::Text(_)
        | Expression::File(_)
        | Expression::TypePath(_)
        | Expression::Src
        | Expression::Usr
        | Expression::GlobalNamespace
        | Expression::GlobalField(_) => {}
    }
    Ok(())
}

pub(crate) fn compile_error(message: impl Into<String>) -> CompileError {
    CompileError {
        message: message.into(),
    }
}
