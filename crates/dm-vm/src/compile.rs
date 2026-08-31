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

use dm_core::SourceSpan;
use dm_lexer::{SpannedToken, TokenKind};
use dm_syntax::{Definition, DefinitionKind, SourceLine};
use dm_value::{FieldName, TypePath};

use crate::bytecode::{
    DeferredProcedure, InitializerBinding, InitializerCallNameIndex, InitializerCompileContext,
    InitializerProgram, Instruction, Module, ProcedureId, Program, next_module_identity,
};
use crate::{CompileError, ProcedureSpec, procedure_type_catalog_from_specs};

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

// Re-export the compiler submodules under the stable `crate::compile::*`
// paths so the bytecode IR, interpreter, native acceleration, and tests keep
// their existing references regardless of the expression/statement split.
#[allow(unused_imports)]
pub(crate) use crate::compile_expr::{
    EXPANDED_ARGUMENT_COUNT, Expression, ExpressionParser, ListExpressionEntry,
    bind_initializer_expression, dm_builtin_numeric_constant, emit_expression,
    interpolated_expression_close, to_local_index,
};
#[allow(unused_imports)]
pub(crate) use crate::compile_stmt::{
    LocalTable, compile_block, compile_parameter_defaults, condition_tokens,
    declared_parameter_type, indentation, parameter_name, procedure_verb_name, procedure_wait_for,
    push_instruction, verb_parameter_type,
};
pub(crate) fn compile_error(message: impl Into<String>) -> CompileError {
    CompileError {
        message: message.into(),
    }
}
