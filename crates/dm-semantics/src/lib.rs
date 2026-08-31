//! Procedure-level semantic indexing for Dream Maker projects.
//!
//! The registry converts canonical procedure nodes into source-ordered
//! implementation chains. Each implementation records the exact predecessor a
//! future `..()` lowering should invoke: the previous implementation on the
//! same type, or the inherited procedure's effective implementation.
//!
//! Inheritance follows the object tree's resolved hierarchy, including the final
//! source-ordered constant `parent_type = /some/type` assignment on each type.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_globals::{StorageClass, VariableRegistry};
use dm_lexer::TokenKind;
use dm_object_tree::{CodePath, NodeId, NodeKind};
use dm_value::{FieldName, TypePath};

mod builtins;
mod executable;
mod ids;
mod ir;
mod registry;

use builtins::{
    NATIVE_PARENT_BUILTINS, STANDARD_BUILTINS, compiler_type_predicate, native_member_index,
    native_parent_index,
};
use ids::{implementation_id, procedure_id};
use ir::effective_target;

pub use executable::{ExecutableProcedureStats, ExecutableProcedures};
pub use ids::{ProcedureId, ProcedureImplementationId};
pub use ir::{
    Procedure, ProcedureClosureStats, ProcedureImplementation, ProcedureImplementationKind,
    ProcedureRegistryBuildStats,
};
pub use registry::ProcedureRegistry;

fn normalize_upward_paths(
    compilation: &Compilation,
    owner: Option<NodeId>,
    definition: &dm_syntax::Definition,
    global_types: &BTreeMap<String, TypePath>,
) -> dm_syntax::Definition {
    let mut normalized = definition.clone();
    let mut local_types = BTreeMap::<String, dm_syntax::DefinitionPath>::new();
    for parameter in &normalized.parameters {
        if let Some(name) = parameter_declaration_name(&parameter.tokens)
            && let Some(path) = declared_type_path(&parameter.tokens, name)
        {
            local_types.insert(name.to_owned(), path);
        }
    }
    let contextual_anchor = owner
        .and_then(|node| compilation.code_tree().node(node))
        .map(|node| dm_syntax::DefinitionPath::new(node.path.segments().to_vec()));
    for line in &mut normalized.body {
        line.tokens = compilation
            .code_tree()
            .normalize_upward_paths(contextual_anchor.as_ref(), &line.tokens);
        if matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
        {
            let identifiers = line
                .tokens
                .iter()
                .take_while(|token| {
                    !matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                        && !matches!(token.kind, TokenKind::Punctuation('['))
                })
                .filter_map(|token| match &token.kind {
                    TokenKind::Identifier(name) => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let end = identifiers
                .iter()
                .position(|name| *name == "as")
                .unwrap_or(identifiers.len());
            let declared_name = identifiers[..end].last().map(|name| (*name).to_owned());
            if let Some(name) = declared_name
                && let Some(path) = declared_type_path(&line.tokens, &name)
            {
                qualify_contextual_new(&mut line.tokens, &path);
                local_types.insert(name, path);
            }
        }
        // Function-like macros commonly retain compact `; if(...) { ... }`
        // bodies on one source line. Discover every typed local declaration,
        // not only a declaration occupying the first token.
        for var_index in line
            .tokens
            .iter()
            .enumerate()
            .filter_map(|(index, token)| {
                matches!(&token.kind, TokenKind::Identifier(name) if name == "var").then_some(index)
            })
            .collect::<Vec<_>>()
        {
            let end = line.tokens[var_index..]
                .iter()
                .position(|token| {
                    matches!(
                        token.kind,
                        TokenKind::Punctuation(';') | TokenKind::Punctuation('}')
                    ) || matches!(&token.kind, TokenKind::Identifier(name) if name == "in")
                        || matches!(token.kind, TokenKind::Punctuation('['))
                })
                .map_or(line.tokens.len(), |offset| var_index + offset);
            let declaration = &line.tokens[var_index..end];
            let assignment = declaration
                .iter()
                .position(|token| matches!(&token.kind, TokenKind::Operator(op) if op == "="))
                .unwrap_or(declaration.len());
            if let Some(name) =
                declaration[..assignment]
                    .iter()
                    .rev()
                    .find_map(|token| match &token.kind {
                        TokenKind::Identifier(name) if name != "var" => Some(name.as_str()),
                        _ => None,
                    })
                && let Some(path) = declared_type_path(declaration, name)
            {
                local_types.insert(name.to_owned(), path);
            }
        }
        qualify_compact_local_assignments(&mut line.tokens, &local_types);
        // BYOND's inferred `new` is destination typed.  Preserve that context
        // through the complete RHS (including parentheses, simple assignment
        // wrappers and ternary arms), but never invent the executing datum's
        // type for a contextless expression.
        if let Some(assignment) = top_level_simple_assignment(&line.tokens) {
            let target = &line.tokens[..assignment];
            let expected =
                inferred_assignment_type(compilation, owner, target, &local_types, global_types);
            if let Some(expected) = expected {
                let start = if line.tokens[..assignment]
                    .iter()
                    .any(|token| matches!(token.kind, TokenKind::Punctuation('[')))
                {
                    0
                } else {
                    assignment + 1
                };
                qualify_contextual_new_from(&mut line.tokens, start, &expected);
            }
        }
        if matches!(line.tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "var")
        {
            let assignment = line.tokens.iter().position(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
            );
            // A declaration annotation belongs to the declaration side of
            // `=`.  The same `as` token on the RHS is expression syntax,
            // notably `input(...) as num|null`, and must survive lowering.
            let declaration_end = assignment.unwrap_or(line.tokens.len());
            if let Some(annotation) = line.tokens[..declaration_end].iter().position(
                |token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"),
            ) {
                line.tokens.drain(annotation..declaration_end);
            }
        }
    }
    for parameter in &mut normalized.parameters {
        let declared_name = parameter_declaration_name(&parameter.tokens).map(str::to_owned);
        if let Some(name) = declared_name
            && let Some(path) = declared_type_path(&parameter.tokens, &name)
        {
            qualify_contextual_new(&mut parameter.tokens, &path);
        }
        let assignment = parameter.tokens.iter().position(
            |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
        );
        let annotation = parameter
            .tokens
            .iter()
            .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"));
        if let Some(annotation) = annotation
            && assignment.is_none_or(|assignment| annotation < assignment)
        {
            let end = assignment.unwrap_or(parameter.tokens.len());
            parameter.tokens.drain(annotation..end);
        } else if let (Some(annotation), Some(assignment)) = (annotation, assignment)
            && annotation > assignment
        {
            parameter.tokens.drain(annotation..);
        }
    }
    normalized
}

/// Expands BYOND's compiler-owned `__PROC__` pseudo-macro to the canonical
/// current procedure reference. Unlike an ordinary preprocessor macro this
/// requires the resolved object-tree identity, so expansion belongs here.
fn expand_proc_pseudo_macro(definition: &mut dm_syntax::Definition, path: &CodePath) {
    fn expand(tokens: &mut Vec<dm_lexer::SpannedToken>, segments: &[String]) {
        let mut index = 0;
        while index < tokens.len() {
            if !matches!(&tokens[index].kind, TokenKind::Identifier(name) if name == "__PROC__") {
                index += 1;
                continue;
            }
            let span = tokens[index].span;
            let replacement = segments
                .iter()
                .flat_map(|segment| {
                    [
                        dm_lexer::SpannedToken {
                            kind: TokenKind::Operator("/".to_owned()),
                            span,
                        },
                        dm_lexer::SpannedToken {
                            kind: TokenKind::Identifier(segment.clone()),
                            span,
                        },
                    ]
                })
                .collect::<Vec<_>>();
            let replacement_len = replacement.len();
            tokens.splice(index..=index, replacement);
            index += replacement_len;
        }
    }

    let segments = path.segments();
    expand(&mut definition.header, segments);
    for parameter in &mut definition.parameters {
        expand(&mut parameter.tokens, segments);
    }
    for line in &mut definition.body {
        expand(&mut line.tokens, segments);
    }
}

fn qualify_contextual_new(
    tokens: &mut Vec<dm_lexer::SpannedToken>,
    expected_type: &dm_syntax::DefinitionPath,
) {
    qualify_contextual_new_from(tokens, 0, expected_type);
}

fn qualify_contextual_new_from(
    tokens: &mut Vec<dm_lexer::SpannedToken>,
    start: usize,
    expected_type: &dm_syntax::DefinitionPath,
) {
    let mut index = start;
    while index < tokens.len() {
        if matches!(&tokens[index].kind, TokenKind::Identifier(name) if name == "new")
            && !matches!(tokens.get(index + 1).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
            && !matches!(
                tokens.get(index + 1).map(|token| &token.kind),
                Some(TokenKind::Identifier(_))
            )
        {
            let span = tokens[index].span;
            let mut inserted = Vec::new();
            for segment in expected_type.segments() {
                inserted.push(dm_lexer::SpannedToken {
                    kind: TokenKind::Operator("/".to_owned()),
                    span,
                });
                inserted.push(dm_lexer::SpannedToken {
                    kind: TokenKind::Identifier(segment.clone()),
                    span,
                });
            }
            let inserted_len = inserted.len();
            tokens.splice(index + 1..index + 1, inserted);
            index += inserted_len;
        }
        index += 1;
    }
}

fn top_level_simple_assignment(tokens: &[dm_lexer::SpannedToken]) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator)
                if matches!(operator.as_str(), "=" | "||=" | "&&=") && depth == 0 =>
            {
                return Some(index);
            }
            _ => {}
        }
    }
    None
}

fn qualify_compact_local_assignments(
    tokens: &mut Vec<dm_lexer::SpannedToken>,
    locals: &BTreeMap<String, dm_syntax::DefinitionPath>,
) {
    let mut index = 0;
    while index + 2 < tokens.len() {
        let receiver = match &tokens[index].kind {
            TokenKind::Identifier(name) => name.clone(),
            _ => {
                index += 1;
                continue;
            }
        };
        if matches!(&tokens[index + 1].kind, TokenKind::Operator(op) if matches!(op.as_str(), "=" | "||=" | "&&="))
            && matches!(&tokens[index + 2].kind, TokenKind::Identifier(name) if name == "new")
            && let Some(path) = locals.get(&receiver)
        {
            qualify_contextual_new_from(tokens, index + 2, path);
        }
        index += 1;
    }
}

fn inferred_assignment_type(
    compilation: &Compilation,
    owner: Option<NodeId>,
    target: &[dm_lexer::SpannedToken],
    locals: &BTreeMap<String, dm_syntax::DefinitionPath>,
    globals: &BTreeMap<String, TypePath>,
) -> Option<dm_syntax::DefinitionPath> {
    if target
        .iter()
        .any(|token| matches!(token.kind, TokenKind::Punctuation('[')))
        && let Some(receiver) = target.iter().find_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        && let Some(path) = locals.get(receiver)
    {
        return Some(path.clone());
    }
    let identifiers = target
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) if name != "var" => Some(name.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let name = *identifiers.last()?;
    // A plain destination obeys normal local -> src -> global resolution.
    if identifiers.len() == 1 {
        if let Some(path) = locals.get(name) {
            return Some(path.clone());
        }
        if let Some(path) = declared_member_type(compilation, owner, name) {
            return Some(path);
        }
        return globals.get(name).map(|path| {
            dm_syntax::DefinitionPath::new(
                path.as_str()[1..].split('/').map(str::to_owned).collect(),
            )
        });
    }
    // `src.field` is statically typed even though its receiver is implicit at
    // runtime. More general chains are handled when every member declaration
    // supplies a concrete type.
    let mut current = if identifiers[0] == "src" {
        owner
    } else {
        locals
            .get(identifiers[0])
            .and_then(|path| compilation.code_tree().find(path))
            .or_else(|| {
                globals.get(identifiers[0]).and_then(|path| {
                    let path = dm_syntax::DefinitionPath::new(
                        path.as_str()[1..].split('/').map(str::to_owned).collect(),
                    );
                    compilation.code_tree().find(&path)
                })
            })
            .or_else(|| {
                declared_member_type(compilation, owner, identifiers[0])
                    .and_then(|path| compilation.code_tree().find(&path))
            })
    };
    let start = 1;
    let mut result = None;
    for member in &identifiers[start..] {
        result = declared_member_type(compilation, current, member);
        current = result
            .as_ref()
            .and_then(|path| compilation.code_tree().find(path));
    }
    result
}

fn declared_member_type(
    compilation: &Compilation,
    mut owner: Option<NodeId>,
    name: &str,
) -> Option<dm_syntax::DefinitionPath> {
    while let Some(type_id) = owner {
        let type_node = compilation.code_tree().node(type_id)?;
        if let Some(path) = engine_builtin_member_type(&type_node.path, name) {
            return Some(path);
        }
        let mut segments = type_node.path.segments().to_vec();
        segments.extend(["var".to_owned(), name.to_owned()]);
        if let Some(field) = compilation
            .code_tree()
            .find(&dm_syntax::DefinitionPath::new(segments))
            .and_then(|id| compilation.code_tree().node(id))
            && let Some(declaration) = field
                .declarations
                .iter()
                .rev()
                .find_map(|id| compilation.code_tree().declaration(*id))
            && let Some(definition) = compilation
                .syntax(declaration.file_id)
                .and_then(|syntax| syntax.definitions.get(declaration.definition_index))
            && let Some(path) = declared_type_path(&definition.header, name)
        {
            return Some(path);
        }
        owner = type_node.parent_type;
    }
    None
}

/// Statically typed reference fields supplied by the BYOND object model
/// rather than a project `/var` declaration. OpenDream exposes the same
/// reciprocal `Mob.Client` and `Client.Mob` types in its runtime metadata.
fn engine_builtin_member_type(owner: &CodePath, name: &str) -> Option<dm_syntax::DefinitionPath> {
    let owner = owner.to_string();
    let path = if (owner == "/mob" || owner.starts_with("/mob/")) && name == "client" {
        "/client"
    } else if (owner == "/client" || owner.starts_with("/client/")) && name == "mob" {
        "/mob"
    } else {
        return None;
    };
    Some(dm_syntax::DefinitionPath::new(
        path.trim_start_matches('/')
            .split('/')
            .map(str::to_owned)
            .collect(),
    ))
}

#[derive(Default)]
struct ConstBindings {
    globals: BTreeSet<String>,
    fields: BTreeMap<NodeId, BTreeSet<String>>,
    field_types: BTreeMap<NodeId, BTreeMap<String, NodeId>>,
    scalar_field_types: BTreeMap<NodeId, ScalarConstraint>,
    scalar_field_conflicts: Vec<String>,
    unresolved_type_conflicts: Vec<String>,
}

impl ConstBindings {
    fn build(compilation: &Compilation) -> Self {
        let mut bindings = Self::default();
        let registry = VariableRegistry::build(compilation);
        for entry in registry.entries() {
            let Some(name) = entry.path.rsplit('/').next() else {
                continue;
            };
            if entry.modifiers.constant {
                if let Some(owner) = &entry.owner {
                    bindings
                        .fields
                        .entry(owner.node)
                        .or_default()
                        .insert(name.to_owned());
                } else {
                    bindings.globals.insert(name.to_owned());
                }
            }
            let Some(owner) = &entry.owner else {
                continue;
            };
            let Some(definition) = compilation
                .syntax(entry.file_id)
                .and_then(|syntax| syntax.definitions.get(entry.definition_index))
            else {
                continue;
            };
            if entry.assignment == dm_globals::AssignmentKind::Declaration
                && let Some(constraint) = scalar_constraint(&definition.header)
            {
                bindings.scalar_field_types.insert(entry.node, constraint);
            }
            if definition.header.iter().any(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
            ) && let Some(path) = declared_type_path(&definition.header, name)
                && !is_known_declared_type(compilation, &path)
            {
                bindings.unresolved_type_conflicts.push(format!(
                    "unknown declared type `{path}` for variable {}",
                    entry.path
                ));
            }
            if let Some(type_node) = declared_type_node(compilation, &definition.header, name) {
                bindings
                    .field_types
                    .entry(owner.node)
                    .or_default()
                    .entry(name.to_owned())
                    .or_insert(type_node);
            }
        }
        for entry in registry
            .entries()
            .iter()
            .filter(|entry| entry.assignment == dm_globals::AssignmentKind::Override)
        {
            let Some(initializer) = &entry.initializer else {
                continue;
            };
            let Some(actual) = proven_literal_scalar_type(&initializer.tokens) else {
                continue;
            };
            let Some(expected) = bindings.effective_scalar_field(compilation, entry.node) else {
                continue;
            };
            if actual != expected.kind && !(actual == ScalarType::Null && expected.allows_null) {
                bindings.scalar_field_conflicts.push(format!(
                    "cannot assign {actual:?} to field override {} declared as {:?}",
                    entry.path, expected.kind
                ));
            }
        }
        bindings
    }

    fn field_is_const(
        &self,
        compilation: &Compilation,
        mut owner: Option<NodeId>,
        name: &str,
    ) -> bool {
        while let Some(node) = owner {
            if self
                .fields
                .get(&node)
                .is_some_and(|fields| fields.contains(name))
            {
                return true;
            }
            owner = compilation
                .code_tree()
                .node(node)
                .and_then(|type_node| type_node.parent_type);
        }
        false
    }

    fn field_type(
        &self,
        compilation: &Compilation,
        mut owner: Option<NodeId>,
        name: &str,
    ) -> Option<NodeId> {
        while let Some(node) = owner {
            if let Some(field_type) = self
                .field_types
                .get(&node)
                .and_then(|fields| fields.get(name))
            {
                return Some(*field_type);
            }
            owner = compilation
                .code_tree()
                .node(node)
                .and_then(|type_node| type_node.parent_type);
        }
        None
    }

    fn effective_scalar_field(
        &self,
        compilation: &Compilation,
        mut node: NodeId,
    ) -> Option<ScalarConstraint> {
        loop {
            if let Some(constraint) = self.scalar_field_types.get(&node) {
                return Some(*constraint);
            }
            node = compilation.code_tree().node(node)?.inherited_member?;
        }
    }
}

fn validate_const_assignments(
    definition: &dm_syntax::Definition,
    owner: Option<NodeId>,
    compilation: &Compilation,
    bindings: &ConstBindings,
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
) -> Result<(), dm_vm::CompileError> {
    if let Some(message) = bindings.scalar_field_conflicts.first() {
        return Err(dm_vm::CompileError {
            message: message.clone(),
        });
    }
    if let Some(message) = bindings.unresolved_type_conflicts.first() {
        return Err(dm_vm::CompileError {
            message: message.clone(),
        });
    }
    let mut locals = BTreeSet::new();
    let mut const_locals = BTreeSet::new();
    let mut local_types = BTreeMap::new();
    let mut scalar_types = BTreeMap::new();
    let mut known_truthy = BTreeSet::new();
    let procedure_node = registry
        .procedure(implementation.procedure())
        .map(|procedure| procedure.node);
    if let Some(path) = procedure_return_type_path(&definition.header)
        && !is_known_declared_type(compilation, &path)
    {
        return Err(dm_vm::CompileError {
            message: format!("unknown declared procedure return type `{path}`"),
        });
    }
    if let Some(node) = procedure_node {
        validate_override_return_signature(compilation, node)?;
    }
    for parameter in &definition.parameters {
        let Some(name) = parameter_declaration_name(&parameter.tokens) else {
            continue;
        };
        locals.insert(name.to_owned());
        validate_declared_type_exists(compilation, &parameter.tokens, name)?;
        if let Some(type_node) = declared_type_node(compilation, &parameter.tokens, name) {
            local_types.insert(name.to_owned(), type_node);
        }
        if let Some(constraint) = scalar_constraint(&parameter.tokens) {
            scalar_types.insert(name.to_owned(), constraint);
        }
    }

    for line in &definition.body {
        let tokens = &line.tokens;
        let assignment = tokens.iter().position(|token| {
            matches!(&token.kind, TokenKind::Operator(operator) if is_assignment_operator(operator))
        });
        let declaration_end = tokens
            .iter()
            .position(|token| matches!(token.kind, TokenKind::Punctuation('[')))
            .unwrap_or_else(|| assignment.unwrap_or(tokens.len()));
        let identifiers: Vec<_> = tokens
            .iter()
            .take(declaration_end)
            .filter_map(|token| match &token.kind {
                TokenKind::Identifier(name) => Some(name.as_str()),
                _ => None,
            })
            .collect();

        if identifiers.first() == Some(&"var") {
            // A comma-separated declaration introduces every declarator. The
            // final declarator may own the initializer (`var/i, ch, len = 3`),
            // but preceding bare names must still be visible to flow checks.
            for grouped_name in grouped_local_declaration_names(tokens) {
                locals.insert(grouped_name.clone());
                scalar_types
                    .entry(grouped_name)
                    .or_insert(ScalarConstraint::exact(ScalarType::Dynamic));
            }
            let name_index = identifiers
                .iter()
                .position(|name| *name == "as")
                .unwrap_or(identifiers.len());
            if let Some(name) = identifiers[..name_index].last().copied() {
                locals.insert(name.to_owned());
                validate_declared_type_exists(compilation, tokens, name)?;
                if let Some(type_node) = declared_type_node(compilation, tokens, name) {
                    local_types.insert(name.to_owned(), type_node);
                }
                if let Some(constraint) = scalar_constraint(tokens) {
                    if let Some(assignment) = assignment
                        && let Some(actual) = proven_scalar_type(
                            compilation,
                            registry,
                            implementation,
                            &tokens[assignment + 1..],
                            &scalar_types,
                            &local_types,
                            owner,
                            bindings,
                            &known_truthy,
                        )
                    {
                        validate_scalar_assignment(name, constraint, actual)?;
                    }
                    scalar_types.insert(name.to_owned(), constraint);
                } else if identifiers.contains(&"const") {
                    if let Some(assignment) = assignment
                        && let Some(actual) = proven_scalar_type(
                            compilation,
                            registry,
                            implementation,
                            &tokens[assignment + 1..],
                            &scalar_types,
                            &local_types,
                            owner,
                            bindings,
                            &known_truthy,
                        )
                    {
                        scalar_types.insert(name.to_owned(), ScalarConstraint::exact(actual));
                    }
                } else {
                    scalar_types.insert(
                        name.to_owned(),
                        ScalarConstraint::exact(ScalarType::Dynamic),
                    );
                }
                if assignment.is_some_and(|assignment| {
                    expression_is_proven_truthy(&tokens[assignment + 1..])
                }) {
                    known_truthy.insert(name.to_owned());
                }
                if identifiers.contains(&"const") {
                    const_locals.insert(name.to_owned());
                }
                if let (Some(expected), Some(actual)) = (
                    local_types.get(name).copied(),
                    assignment.and_then(|assignment| {
                        proven_datum_expression_type(
                            compilation,
                            &tokens[assignment + 1..],
                            &local_types,
                            owner,
                            bindings,
                            &known_truthy,
                            registry,
                            implementation,
                        )
                    }),
                ) {
                    validate_type_assignment(compilation, name, expected, actual)?;
                }
            }
            continue;
        }

        let Some(assignment_index) = assignment else {
            // BYOND return annotations constrain the declared proc signature
            // (and therefore callers), but do not reject body values. BYOND
            // 516 accepts path-, scalar-, local-, and call-result mismatches
            // here; runtime filtering/coercion remains the proc contract.
            continue;
        };
        let assigned = &tokens[..assignment_index];
        let mut bare_name = match assigned {
            [token] => match &token.kind {
                TokenKind::Identifier(name) => Some(name.as_str()),
                _ => None,
            },
            _ => None,
        };
        if bare_name.is_none()
            && assignment_index == 0
            && matches!(
                tokens.first().map(|token| &token.kind),
                Some(TokenKind::Operator(operator)) if operator == "++" || operator == "--"
            )
        {
            bare_name = match tokens.get(1).map(|token| &token.kind) {
                Some(TokenKind::Identifier(name)) => Some(name.as_str()),
                _ => None,
            };
        }
        if let Some((receiver, name)) = assigned_receiver_field(assigned) {
            let receiver_type = if receiver == "src" {
                owner
            } else {
                local_types
                    .get(receiver)
                    .copied()
                    .or_else(|| bindings.field_type(compilation, owner, receiver))
            };
            if receiver_type.is_some() && bindings.field_is_const(compilation, receiver_type, name)
            {
                return Err(dm_vm::CompileError {
                    message: format!("cannot assign to const variable `{name}`"),
                });
            }
            continue;
        }
        let Some(name) = bare_name else {
            continue;
        };

        let forbidden = if locals.contains(name) {
            const_locals.contains(name)
        } else {
            bindings.field_is_const(compilation, owner, name) || bindings.globals.contains(name)
        };
        if forbidden {
            return Err(dm_vm::CompileError {
                message: format!("cannot assign to const variable `{name}`"),
            });
        }
        if locals.contains(name) {
            known_truthy.remove(name);
        }
        if matches!(
            tokens.get(assignment_index).map(|token| &token.kind),
            Some(TokenKind::Operator(operator)) if operator == "="
        ) && matches!(
            tokens.get(assignment_index + 1).map(|token| &token.kind),
            Some(TokenKind::Identifier(keyword)) if keyword == "new"
        ) && let (Some(expected), Some(actual)) = (
            local_types.get(name).copied(),
            proven_datum_expression_type(
                compilation,
                &tokens[assignment_index + 1..],
                &local_types,
                owner,
                bindings,
                &known_truthy,
                registry,
                implementation,
            ),
        ) {
            validate_type_assignment(compilation, name, expected, actual)?;
        }
        if let (Some(expected), Some(actual)) = (
            scalar_types.get(name).copied(),
            proven_scalar_type(
                compilation,
                registry,
                implementation,
                &tokens[assignment_index + 1..],
                &scalar_types,
                &local_types,
                owner,
                bindings,
                &known_truthy,
            ),
        ) {
            validate_scalar_assignment(name, expected, actual)?;
        }
        if locals.contains(name) && expression_is_proven_truthy(&tokens[assignment_index + 1..]) {
            known_truthy.insert(name.to_owned());
        }
    }
    Ok(())
}

fn validate_override_return_signature(
    compilation: &Compilation,
    node: NodeId,
) -> Result<(), dm_vm::CompileError> {
    let tree_node = compilation
        .code_tree()
        .node(node)
        .expect("procedure node should exist");
    let Some(parent) = tree_node.inherited_member else {
        return Ok(());
    };
    let direct_scalar = tree_node.declarations.iter().rev().find_map(|declaration| {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)?
            .definitions
            .get(declaration.definition_index)?;
        procedure_scalar_return(&definition.header)
    });
    if let (Some(child), Some(parent)) =
        (direct_scalar, effective_scalar_return(compilation, parent))
        && (child.kind != parent.kind || child.allows_null != parent.allows_null)
    {
        return Err(dm_vm::CompileError {
            message: "procedure override changes its inherited scalar return type".to_owned(),
        });
    }
    let direct_datum = tree_node.declarations.iter().rev().find_map(|declaration| {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)?
            .definitions
            .get(declaration.definition_index)?;
        procedure_return_type_node(compilation, &definition.header)
    });
    if let (Some(child), Some(parent)) = (direct_datum, effective_datum_return(compilation, parent))
    {
        validate_type_assignment(compilation, "return", parent, child)?;
    }
    Ok(())
}

fn validate_type_assignment(
    compilation: &Compilation,
    name: &str,
    expected: NodeId,
    actual: NodeId,
) -> Result<(), dm_vm::CompileError> {
    let tree = compilation.code_tree();
    let mut current = Some(actual);
    while let Some(node) = current {
        if node == expected {
            return Ok(());
        }
        current = tree.node(node).and_then(|node| node.parent_type);
    }
    // DreamMaker's path annotations permit a value declared as a base path to
    // flow into a variable declared as one of its subpaths. The runtime value
    // can still be an instance of that subtype. Unrelated type branches remain
    // a compile-time mismatch.
    let mut current = Some(expected);
    while let Some(node) = current {
        if node == actual {
            return Ok(());
        }
        current = tree.node(node).and_then(|node| node.parent_type);
    }
    // A datum path on a DM local/parameter is a static hint used for member
    // lookup and inferred `new`; it is not a Rust-style assignment barrier.
    // BYOND accepts values from unrelated branches here (including values
    // supplied through `as mob|obj|turf` call sites) and leaves runtime
    // predicates/casts to the program. Keep the stricter check for declared
    // procedure return contracts, which Dream64 uses to narrow call results.
    if name != "return" {
        return Ok(());
    }
    let expected = tree
        .node(expected)
        .map(|node| node.path.to_string())
        .unwrap_or_else(|| "<unknown>".to_owned());
    let actual = tree
        .node(actual)
        .map(|node| node.path.to_string())
        .unwrap_or_else(|| "<unknown>".to_owned());
    Err(dm_vm::CompileError {
        message: format!(
            "cannot assign {actual} to typed variable `{name}` declared as {expected}"
        ),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScalarType {
    Number,
    Text,
    Null,
    Dynamic,
}

fn expression_is_proven_truthy(tokens: &[dm_lexer::SpannedToken]) -> bool {
    matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "new")
        || matches!(tokens, [token] if matches!(&token.kind, TokenKind::Number(value) if value != "0" && value != "0.0"))
        || matches!(tokens, [token] if matches!(&token.kind, TokenKind::String(value) | TokenKind::RawString(value) if !value.is_empty()))
}

#[derive(Clone, Copy, Debug)]
struct ScalarConstraint {
    kind: ScalarType,
    allows_null: bool,
}

impl ScalarConstraint {
    const fn exact(kind: ScalarType) -> Self {
        Self {
            kind,
            allows_null: matches!(kind, ScalarType::Null),
        }
    }
}

fn scalar_constraint(tokens: &[dm_lexer::SpannedToken]) -> Option<ScalarConstraint> {
    let annotation = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"))?;
    let identifiers: BTreeSet<_> = tokens[annotation + 1..]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    let kind = if identifiers.contains("num") {
        ScalarType::Number
    } else if identifiers.contains("text") {
        ScalarType::Text
    } else {
        return None;
    };
    Some(ScalarConstraint {
        kind,
        allows_null: identifiers.contains("null"),
    })
}

fn procedure_scalar_return(tokens: &[dm_lexer::SpannedToken]) -> Option<ScalarConstraint> {
    let closing = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Punctuation(')'))?;
    scalar_constraint(&tokens[closing + 1..])
}

fn effective_scalar_return(compilation: &Compilation, node: NodeId) -> Option<ScalarConstraint> {
    let tree_node = compilation.code_tree().node(node)?;
    for declaration in tree_node.declarations.iter().rev() {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)
            .and_then(|syntax| syntax.definitions.get(declaration.definition_index))?;
        if let Some(constraint) = procedure_scalar_return(&definition.header) {
            return Some(constraint);
        }
    }
    tree_node
        .inherited_member
        .and_then(|parent| effective_scalar_return(compilation, parent))
}

fn effective_datum_return(compilation: &Compilation, node: NodeId) -> Option<NodeId> {
    let tree_node = compilation.code_tree().node(node)?;
    for declaration in tree_node.declarations.iter().rev() {
        let declaration = compilation.code_tree().declaration(*declaration)?;
        let definition = compilation
            .syntax(declaration.file_id)
            .and_then(|syntax| syntax.definitions.get(declaration.definition_index))?;
        if let Some(return_type) = procedure_return_type_node(compilation, &definition.header) {
            return Some(return_type);
        }
    }
    tree_node
        .inherited_member
        .and_then(|parent| effective_datum_return(compilation, parent))
}

fn statically_called_procedure(
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
) -> Option<NodeId> {
    let TokenKind::Identifier(selector) = &tokens.first()?.kind else {
        return None;
    };
    if !matches!(
        tokens.get(1).map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) || !matches!(
        tokens.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) {
        return None;
    }
    let target = registry.static_call_target(implementation, selector, compilation)?;
    registry
        .procedure(target.procedure())
        .map(|procedure| procedure.node)
}

fn proven_scalar_type(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
    tokens: &[dm_lexer::SpannedToken],
    locals: &BTreeMap<String, ScalarConstraint>,
    datum_locals: &BTreeMap<String, NodeId>,
    owner: Option<NodeId>,
    bindings: &ConstBindings,
    known_truthy: &BTreeSet<String>,
) -> Option<ScalarType> {
    if let Some(inferred) = infer_scalar_composite(
        compilation,
        registry,
        implementation,
        tokens,
        locals,
        datum_locals,
        owner,
        bindings,
        known_truthy,
    ) {
        return Some(inferred);
    }
    let direct = match tokens {
        [token] => match &token.kind {
            TokenKind::Identifier(name) => locals.get(name).map(|constraint| constraint.kind),
            _ => proven_literal_scalar_type(tokens),
        },
        _ => proven_literal_scalar_type(tokens),
    };
    direct
        .or_else(|| {
            statically_called_procedure(registry, implementation, compilation, tokens)
                .and_then(|node| effective_scalar_return(compilation, node))
                .map(|constraint| constraint.kind)
        })
        .or_else(|| {
            let (receiver, member, call) = receiver_member_expression(tokens)?;
            let receiver_type =
                proven_receiver_type(compilation, receiver, datum_locals, owner, bindings)?;
            if call {
                find_member_node(compilation, receiver_type, "proc", member)
                    .and_then(|node| effective_scalar_return(compilation, node))
                    .map(|constraint| constraint.kind)
            } else {
                find_member_node(compilation, receiver_type, "var", member)
                    .and_then(|node| bindings.effective_scalar_field(compilation, node))
                    .map(|constraint| constraint.kind)
            }
        })
        .or_else(|| {
            let (receiver, member) = parenthesized_receiver_method(tokens)?;
            let receiver_type = proven_datum_expression_type(
                compilation,
                receiver,
                datum_locals,
                owner,
                bindings,
                known_truthy,
                registry,
                implementation,
            )?;
            find_member_node(compilation, receiver_type, "proc", member)
                .and_then(|node| effective_scalar_return(compilation, node))
                .map(|constraint| constraint.kind)
        })
}

#[allow(clippy::too_many_arguments)]
fn infer_scalar_composite(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
    tokens: &[dm_lexer::SpannedToken],
    locals: &BTreeMap<String, ScalarConstraint>,
    datum_locals: &BTreeMap<String, NodeId>,
    owner: Option<NodeId>,
    bindings: &ConstBindings,
    known_truthy: &BTreeSet<String>,
) -> Option<ScalarType> {
    if matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) && matches!(
        tokens.last().map(|token| &token.kind),
        Some(TokenKind::Punctuation(')'))
    ) && matching_closing(tokens, 0, '(', ')') == Some(tokens.len() - 1)
    {
        return proven_scalar_type(
            compilation,
            registry,
            implementation,
            &tokens[1..tokens.len() - 1],
            locals,
            datum_locals,
            owner,
            bindings,
            known_truthy,
        );
    }
    if matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "..")
        && tokens.len() == 3
        && matches!(
            tokens.get(1).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        )
        && matches!(
            tokens.get(2).map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        )
    {
        let parent = registry.implementation(implementation)?.parent_target?;
        let node = registry.procedure(parent.procedure())?.node;
        return effective_scalar_return(compilation, node).map(|constraint| constraint.kind);
    }
    if let Some((question, colon)) = top_level_ternary(tokens) {
        if condition_is_known_truthy(&tokens[..question], known_truthy) {
            return proven_scalar_type(
                compilation,
                registry,
                implementation,
                &tokens[question + 1..colon],
                locals,
                datum_locals,
                owner,
                bindings,
                known_truthy,
            );
        }
        let left = proven_scalar_type(
            compilation,
            registry,
            implementation,
            &tokens[question + 1..colon],
            locals,
            datum_locals,
            owner,
            bindings,
            known_truthy,
        )?;
        let right = proven_scalar_type(
            compilation,
            registry,
            implementation,
            &tokens[colon + 1..],
            locals,
            datum_locals,
            owner,
            bindings,
            known_truthy,
        )?;
        return (left == right).then_some(left);
    }
    if let Some((index, operator)) = top_level_binary(tokens) {
        let left = proven_scalar_type(
            compilation,
            registry,
            implementation,
            &tokens[..index],
            locals,
            datum_locals,
            owner,
            bindings,
            known_truthy,
        )?;
        let right = proven_scalar_type(
            compilation,
            registry,
            implementation,
            &tokens[index + 1..],
            locals,
            datum_locals,
            owner,
            bindings,
            known_truthy,
        )?;
        return match operator {
            "+" if left == right && matches!(left, ScalarType::Number | ScalarType::Text) => {
                Some(left)
            }
            "-" | "*" | "/" | "%" if left == ScalarType::Number && right == ScalarType::Number => {
                Some(ScalarType::Number)
            }
            _ => None,
        };
    }
    inline_list_index_scalar(
        compilation,
        registry,
        implementation,
        tokens,
        locals,
        datum_locals,
        owner,
        bindings,
        known_truthy,
    )
}

fn receiver_member_expression(tokens: &[dm_lexer::SpannedToken]) -> Option<(&str, &str, bool)> {
    let [receiver, dot, member, rest @ ..] = tokens else {
        return None;
    };
    let TokenKind::Identifier(receiver) = &receiver.kind else {
        return None;
    };
    if !matches!(&dot.kind, TokenKind::Operator(operator) if operator == ".") {
        return None;
    }
    let TokenKind::Identifier(member) = &member.kind else {
        return None;
    };
    let call = !rest.is_empty()
        && matches!(
            rest.first().map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        )
        && matches!(
            rest.last().map(|token| &token.kind),
            Some(TokenKind::Punctuation(')'))
        );
    if !rest.is_empty() && !call {
        return None;
    }
    Some((receiver, member, call))
}

fn parenthesized_receiver_method(
    tokens: &[dm_lexer::SpannedToken],
) -> Option<(&[dm_lexer::SpannedToken], &str)> {
    if !matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) {
        return None;
    }
    let close = matching_closing(tokens, 0, '(', ')')?;
    if close + 5 != tokens.len()
        || !matches!(&tokens[close + 1].kind, TokenKind::Operator(operator) if operator == ".")
        || !matches!(&tokens[close + 3].kind, TokenKind::Punctuation('('))
        || !matches!(&tokens[close + 4].kind, TokenKind::Punctuation(')'))
    {
        return None;
    }
    let TokenKind::Identifier(member) = &tokens[close + 2].kind else {
        return None;
    };
    Some((&tokens[1..close], member))
}

fn condition_is_known_truthy(
    tokens: &[dm_lexer::SpannedToken],
    known_truthy: &BTreeSet<String>,
) -> bool {
    matches!(tokens, [token] if matches!(&token.kind, TokenKind::Identifier(name) if known_truthy.contains(name)))
}

fn proven_receiver_type(
    compilation: &Compilation,
    receiver: &str,
    datum_locals: &BTreeMap<String, NodeId>,
    owner: Option<NodeId>,
    bindings: &ConstBindings,
) -> Option<NodeId> {
    if receiver == "src" {
        owner
    } else {
        datum_locals
            .get(receiver)
            .copied()
            .or_else(|| bindings.field_type(compilation, owner, receiver))
    }
}

fn find_member_node(
    compilation: &Compilation,
    mut owner: NodeId,
    namespace: &str,
    member: &str,
) -> Option<NodeId> {
    loop {
        let owner_node = compilation.code_tree().node(owner)?;
        let mut segments = owner_node.path.segments().to_vec();
        segments.push(namespace.to_owned());
        segments.push(member.to_owned());
        let path = dm_syntax::DefinitionPath::new(segments);
        if let Some(node) = compilation.code_tree().find(&path) {
            return Some(node);
        }
        owner = owner_node.parent_type?;
    }
}

fn proven_literal_scalar_type(tokens: &[dm_lexer::SpannedToken]) -> Option<ScalarType> {
    match tokens {
        [token] => match &token.kind {
            TokenKind::Number(_) => Some(ScalarType::Number),
            TokenKind::String(_) | TokenKind::RawString(_) => Some(ScalarType::Text),
            TokenKind::Identifier(name) if name == "null" => Some(ScalarType::Null),
            _ => None,
        },
        _ => None,
    }
}

fn matching_closing(
    tokens: &[dm_lexer::SpannedToken],
    opening: usize,
    open: char,
    close: char,
) -> Option<usize> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().skip(opening) {
        match token.kind {
            TokenKind::Punctuation(value) if value == open => depth += 1,
            TokenKind::Punctuation(value) if value == close => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn top_level_ternary(tokens: &[dm_lexer::SpannedToken]) -> Option<(usize, usize)> {
    let mut depth = 0usize;
    let mut question = None;
    for (index, token) in tokens.iter().enumerate() {
        match &token.kind {
            TokenKind::Punctuation('(' | '[') => depth += 1,
            TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator) if depth == 0 && operator == "?" => {
                question = Some(index)
            }
            TokenKind::Operator(operator) if depth == 0 && operator == ":" => {
                if let Some(question) = question {
                    return Some((question, index));
                }
            }
            _ => {}
        }
    }
    None
}

fn top_level_binary(tokens: &[dm_lexer::SpannedToken]) -> Option<(usize, &str)> {
    let mut depth = 0usize;
    for (index, token) in tokens.iter().enumerate().rev() {
        match &token.kind {
            TokenKind::Punctuation(')' | ']') => depth += 1,
            TokenKind::Punctuation('(' | '[') => depth = depth.saturating_sub(1),
            TokenKind::Operator(operator)
                if depth == 0 && matches!(operator.as_str(), "+" | "-" | "*" | "/" | "%") =>
            {
                return Some((index, operator));
            }
            _ => {}
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn inline_list_index_scalar(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
    tokens: &[dm_lexer::SpannedToken],
    locals: &BTreeMap<String, ScalarConstraint>,
    datum_locals: &BTreeMap<String, NodeId>,
    owner: Option<NodeId>,
    bindings: &ConstBindings,
    known_truthy: &BTreeSet<String>,
) -> Option<ScalarType> {
    if !matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "list")
        || !matches!(
            tokens.get(1).map(|token| &token.kind),
            Some(TokenKind::Punctuation('('))
        )
    {
        return None;
    }
    let close = matching_closing(tokens, 1, '(', ')')?;
    if !matches!(
        tokens.get(close + 1).map(|token| &token.kind),
        Some(TokenKind::Punctuation('['))
    ) || matching_closing(tokens, close + 1, '[', ']') != Some(tokens.len() - 1)
    {
        return None;
    }
    let mut entry_start = 2usize;
    let mut depth = 0usize;
    let mut inferred = None;
    for index in 2..=close {
        let separator = index == close
            || (depth == 0 && matches!(tokens[index].kind, TokenKind::Punctuation(',')));
        if separator {
            let entry = proven_scalar_type(
                compilation,
                registry,
                implementation,
                &tokens[entry_start..index],
                locals,
                datum_locals,
                owner,
                bindings,
                known_truthy,
            )?;
            if inferred.is_some_and(|previous| previous != entry) {
                return None;
            }
            inferred = Some(entry);
            entry_start = index + 1;
            continue;
        }
        match tokens[index].kind {
            TokenKind::Punctuation('(' | '[') => depth += 1,
            TokenKind::Punctuation(')' | ']') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    inferred
}

fn validate_scalar_assignment(
    name: &str,
    expected: ScalarConstraint,
    actual: ScalarType,
) -> Result<(), dm_vm::CompileError> {
    if expected.kind == ScalarType::Dynamic
        || actual == expected.kind
        || (actual == ScalarType::Null && expected.allows_null)
    {
        return Ok(());
    }
    Err(dm_vm::CompileError {
        message: format!(
            "cannot assign {actual:?} to typed variable `{name}` declared as {:?}",
            expected.kind
        ),
    })
}

fn proven_datum_expression_type(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
    local_types: &BTreeMap<String, NodeId>,
    owner: Option<NodeId>,
    bindings: &ConstBindings,
    known_truthy: &BTreeSet<String>,
    registry: &ProcedureRegistry,
    implementation: ProcedureImplementationId,
) -> Option<NodeId> {
    if let Some((question, colon)) = top_level_ternary(tokens) {
        if condition_is_known_truthy(&tokens[..question], known_truthy) {
            return proven_datum_expression_type(
                compilation,
                &tokens[question + 1..colon],
                local_types,
                owner,
                bindings,
                known_truthy,
                registry,
                implementation,
            );
        }
        let left = proven_datum_expression_type(
            compilation,
            &tokens[question + 1..colon],
            local_types,
            owner,
            bindings,
            known_truthy,
            registry,
            implementation,
        )?;
        let right = proven_datum_expression_type(
            compilation,
            &tokens[colon + 1..],
            local_types,
            owner,
            bindings,
            known_truthy,
            registry,
            implementation,
        )?;
        return (left == right).then_some(left);
    }
    if let [token] = tokens
        && let TokenKind::Identifier(name) = &token.kind
    {
        return local_types
            .get(name)
            .copied()
            .or_else(|| bindings.field_type(compilation, owner, name));
    }
    if let Some(node) = statically_called_procedure(registry, implementation, compilation, tokens) {
        return effective_datum_return(compilation, node);
    }
    if let Some((receiver, member, call)) = receiver_member_expression(tokens) {
        let receiver_type =
            proven_receiver_type(compilation, receiver, local_types, owner, bindings)?;
        if call {
            return find_member_node(compilation, receiver_type, "proc", member)
                .and_then(|node| effective_datum_return(compilation, node));
        }
        return bindings.field_type(compilation, Some(receiver_type), member);
    }
    if let Some((receiver, member)) = parenthesized_receiver_method(tokens) {
        let receiver_type = proven_datum_expression_type(
            compilation,
            receiver,
            local_types,
            owner,
            bindings,
            known_truthy,
            registry,
            implementation,
        )?;
        return find_member_node(compilation, receiver_type, "proc", member)
            .and_then(|node| effective_datum_return(compilation, node));
    }
    let mut index = usize::from(
        matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "new"),
    );
    if !matches!(tokens.get(index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
    {
        return None;
    }
    let mut segments = Vec::new();
    while matches!(tokens.get(index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
    {
        let Some(TokenKind::Identifier(segment)) = tokens.get(index + 1).map(|token| &token.kind)
        else {
            return None;
        };
        segments.push(segment.clone());
        index += 2;
    }
    compilation
        .code_tree()
        .find(&dm_syntax::DefinitionPath::new(segments))
}

fn assigned_receiver_field(tokens: &[dm_lexer::SpannedToken]) -> Option<(&str, &str)> {
    match tokens {
        [receiver, dot, field] if matches!(&dot.kind, TokenKind::Operator(operator) if operator == ".") =>
        {
            let TokenKind::Identifier(receiver) = &receiver.kind else {
                return None;
            };
            let TokenKind::Identifier(field) = &field.kind else {
                return None;
            };
            Some((receiver, field))
        }
        _ => None,
    }
}

fn declared_type_node(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
    variable_name: &str,
) -> Option<NodeId> {
    compilation
        .code_tree()
        .find(&declared_type_path(tokens, variable_name)?)
}

fn parameter_declaration_name(tokens: &[dm_lexer::SpannedToken]) -> Option<&str> {
    let end = tokens
        .iter()
        .position(|token| {
            matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                || matches!(&token.kind, TokenKind::Identifier(name) if matches!(name.as_str(), "as" | "in"))
        })
        .unwrap_or(tokens.len());
    tokens[..end]
        .iter()
        .rev()
        .find_map(|token| match &token.kind {
            TokenKind::Identifier(name) if name != "var" => Some(name.as_str()),
            _ => None,
        })
}

fn validate_declared_type_exists(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
    variable_name: &str,
) -> Result<(), dm_vm::CompileError> {
    let Some(path) = declared_type_path(tokens, variable_name) else {
        return Ok(());
    };
    if is_known_declared_type(compilation, &path) {
        return Ok(());
    }
    Err(dm_vm::CompileError {
        message: format!("unknown declared type `{path}` for variable `{variable_name}`"),
    })
}

fn is_known_declared_type(compilation: &Compilation, path: &dm_syntax::DefinitionPath) -> bool {
    if compilation.code_tree().find(path).is_some() {
        return true;
    }
    let segments = path.segments();
    for length in (1..segments.len()).rev() {
        let ancestor = dm_syntax::DefinitionPath::new(segments[..length].to_vec());
        if let Some(node) = compilation
            .code_tree()
            .find(&ancestor)
            .and_then(|id| compilation.code_tree().node(id))
        {
            return !node.is_standard();
        }
    }
    false
}

fn declared_type_path(
    tokens: &[dm_lexer::SpannedToken],
    variable_name: &str,
) -> Option<dm_syntax::DefinitionPath> {
    let assignment = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
        .unwrap_or(tokens.len());
    let header = &tokens[..assignment];
    // In `var/i, ch, len = ...`, slash segments belonging to earlier
    // declarators are not a type prefix for `len`. Restrict type inference to
    // the declarator containing the requested name while retaining `var` as
    // the declaration marker.
    let name_position = header.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(name) if name == variable_name),
    )?;
    let mut depth = 0usize;
    let mut declarator_start = 0usize;
    for (index, token) in header[..name_position].iter().enumerate() {
        match token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            TokenKind::Punctuation(',') if depth == 0 => declarator_start = index + 1,
            _ => {}
        }
    }
    let owned_header;
    let header = if declarator_start == 0 {
        header
    } else {
        owned_header = std::iter::once(tokens[0].clone())
            .chain(header[declarator_start..].iter().cloned())
            .collect::<Vec<_>>();
        &owned_header
    };
    if let Some(name_index) = header.iter().position(
        |token| matches!(&token.kind, TokenKind::Identifier(name) if name == variable_name),
    ) && matches!(
        header.get(name_index + 1).map(|token| &token.kind),
        Some(TokenKind::Punctuation('['))
    ) {
        return Some(dm_syntax::DefinitionPath::new(vec!["list".to_owned()]));
    }
    if let Some(as_index) = header
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"))
        && header[as_index + 1..]
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "|"))
    {
        // `as mob|obj|turf` is a union constraint, not the path
        // `/mob/obj/turf`. No single exact receiver type is proven.
        return None;
    }
    let identifiers: Vec<_> = header
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // `as ...` and a following `in ...` are input/verb constraints, never a
    // declared datum path.  Preserve an explicit type before the variable
    // (`mob/M as mob in players`) and otherwise leave the parameter untyped
    // (`message as message`, `target as turf in world`).
    let constraint = identifiers
        .iter()
        .position(|name| *name == "as")
        .unwrap_or(identifiers.len());
    let declaration = &identifiers[..constraint];
    let name_index = declaration
        .iter()
        .rposition(|name| *name == variable_name)?;
    let type_start = declaration
        .iter()
        .position(|name| *name == "var")
        .map_or(0, |index| index + 1);
    let segments: Vec<String> = declaration[type_start..name_index]
        .iter()
        .filter(|name| !matches!(**name, "global" | "static" | "tmp" | "const" | "final"))
        .map(|name| (**name).to_owned())
        .collect();
    if segments.is_empty() {
        return None;
    }
    // DM permits a list declaration to carry an element type after `list`, as
    // in `var/list/datum/item/items`.  The variable itself is still a /list;
    // the following path is not a subtype rooted beneath /list.
    if matches!(segments.first().map(String::as_str), Some("list" | "alist")) {
        return Some(dm_syntax::DefinitionPath::new(vec![segments[0].clone()]));
    }
    Some(dm_syntax::DefinitionPath::new(segments))
}

fn grouped_local_declaration_names(tokens: &[dm_lexer::SpannedToken]) -> Vec<String> {
    let assignment = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
        .unwrap_or(tokens.len());
    let mut start = 1usize;
    let mut depth = 0usize;
    let mut names = Vec::new();
    for end in 1..=assignment {
        let separator = end == assignment
            || (depth == 0
                && matches!(
                    tokens.get(end).map(|token| &token.kind),
                    Some(TokenKind::Punctuation(','))
                ));
        if separator {
            if let Some(name) =
                tokens[start..end]
                    .iter()
                    .rev()
                    .find_map(|token| match &token.kind {
                        TokenKind::Identifier(name) if name != "as" => Some(name.clone()),
                        _ => None,
                    })
            {
                names.push(name);
            }
            start = end + 1;
            continue;
        }
        match tokens[end].kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    names
}

fn procedure_return_type_node(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
) -> Option<NodeId> {
    compilation
        .code_tree()
        .find(&procedure_return_type_path(tokens)?)
}

fn procedure_return_type_path(
    tokens: &[dm_lexer::SpannedToken],
) -> Option<dm_syntax::DefinitionPath> {
    let closing = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Punctuation(')'))?;
    let annotation = tokens[closing + 1..]
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"))?
        + closing
        + 1;
    let segments = tokens[annotation + 1..]
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(segment)
                if !matches!(segment.as_str(), "null" | "num" | "text") =>
            {
                Some(segment.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return None;
    }
    Some(dm_syntax::DefinitionPath::new(segments))
}

fn is_assignment_operator(operator: &str) -> bool {
    matches!(
        operator,
        "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" | "++" | "--"
    )
}

/// Returns bare names for project-wide variables, including globals introduced
/// by macros such as `SUBSYSTEM_DEF(mapping)`. A global has no owning type;
/// static and instance variables remain deliberately excluded. BYOND's built-in
/// `world` singleton is also available as a bare global even though it has no
/// source declaration.
fn declared_global_fields(compilation: &Compilation) -> BTreeMap<String, FieldName> {
    let mut fields: BTreeMap<String, FieldName> = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::Variable && node.owner_type.is_none())
        .filter_map(|node| {
            let name = node.path.segments().last()?;
            FieldName::parse(name)
                .ok()
                .map(|field| (name.clone(), field))
        })
        .collect();
    fields.insert(
        "world".to_owned(),
        FieldName::parse("world").expect("built-in world global name is valid"),
    );
    fields
}

fn direct_static_fields(
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, FieldName>> {
    let mut fields = BTreeMap::<NodeId, BTreeMap<String, FieldName>>::new();
    for entry in registry
        .entries()
        .iter()
        // DM spells type-owned shared storage both `var/static` and
        // `var/global`.  The registry preserves that spelling as distinct
        // storage classes, but both lower to the same owner-qualified VM slot.
        .filter(|entry| matches!(entry.storage, StorageClass::Static | StorageClass::Global))
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        fields
            .entry(owner)
            .or_default()
            .insert(name.to_owned(), FieldName::static_storage(&entry.path));
    }
    fields
}

#[cfg(test)]
fn inherited_static_fields(
    compilation: &Compilation,
    owner: Option<NodeId>,
    direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    cache: &mut BTreeMap<NodeId, BTreeMap<String, FieldName>>,
) -> BTreeMap<String, FieldName> {
    let Some(owner) = owner else {
        return BTreeMap::new();
    };
    if let Some(fields) = cache.get(&owner) {
        return fields.clone();
    }
    let tree = compilation.code_tree();
    let mut hierarchy = Vec::new();
    let mut current = Some(owner);
    while let Some(node) = current {
        hierarchy.push(node);
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    hierarchy.reverse();
    let mut fields = BTreeMap::new();
    for node in hierarchy {
        if let Some(direct) = direct_fields.get(&node) {
            fields.extend(direct.clone());
        }
    }
    cache.insert(owner, fields.clone());
    fields
}

fn referenced_inherited_fields(
    compilation: &Compilation,
    owner: Option<NodeId>,
    direct_fields: &BTreeMap<NodeId, BTreeMap<String, FieldName>>,
    referenced: &BTreeSet<String>,
    include_standard: bool,
) -> BTreeMap<String, FieldName> {
    let Some(mut current) = owner else {
        return BTreeMap::new();
    };
    let tree = compilation.code_tree();
    let mut unresolved = referenced.clone();
    let mut fields = BTreeMap::new();
    while !unresolved.is_empty() {
        let mut available = direct_fields.get(&current).cloned().unwrap_or_default();
        if include_standard {
            standard_instance_fields(tree.node(current).map(|node| &node.path), &mut available);
        }
        if !available.is_empty() {
            let resolved = unresolved
                .iter()
                .filter_map(|name| {
                    available
                        .get(name)
                        .map(|field| (name.clone(), field.clone()))
                })
                .collect::<Vec<_>>();
            for (name, field) in resolved {
                unresolved.remove(&name);
                fields.insert(name, field);
            }
        }
        let Some(parent) = tree.node(current).and_then(|node| node.parent_type) else {
            break;
        };
        current = parent;
    }
    fields
}

fn declared_receiver_types(
    definition: &dm_syntax::Definition,
) -> BTreeMap<String, dm_syntax::DefinitionPath> {
    let mut types = BTreeMap::new();
    for parameter in &definition.parameters {
        let Some(name) = parameter_declaration_name(&parameter.tokens) else {
            continue;
        };
        if let Some(path) = declared_type_path(&parameter.tokens, name) {
            types.insert(name.to_owned(), path);
        }
    }
    for line in &definition.body {
        let is_local_declaration = matches!(
            line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(name)) if name == "var"
        );
        let is_for_declaration = matches!(
            line.tokens.first().map(|token| &token.kind),
            Some(TokenKind::Identifier(name)) if name == "for"
        ) && line
            .tokens
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "var"));
        if !is_local_declaration && !is_for_declaration {
            continue;
        }
        let Some(name) = parameter_declaration_name(&line.tokens) else {
            continue;
        };
        if let Some(path) = declared_type_path(&line.tokens, name) {
            types.insert(name.to_owned(), path);
        }
    }
    types
}

fn declared_global_types(compilation: &Compilation) -> BTreeMap<String, TypePath> {
    let mut types = compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::Variable && node.owner_type.is_none())
        .filter_map(|node| {
            let name = node.path.segments().last()?;
            let declaration = node
                .declarations
                .iter()
                .rev()
                .find_map(|id| compilation.code_tree().declaration(*id))?;
            let definition = compilation
                .syntax(declaration.file_id)
                .and_then(|syntax| syntax.definitions.get(declaration.definition_index))?;
            let path = declared_type_path(&definition.header, name)?;
            TypePath::parse(&path.to_string())
                .ok()
                .map(|path| (name.clone(), path))
        })
        .collect::<BTreeMap<_, _>>();
    if compilation
        .code_tree()
        .nodes()
        .iter()
        .any(|node| node.kind == NodeKind::Type && node.path.to_string() == "/world")
    {
        types.insert(
            "world".to_owned(),
            TypePath::parse("/world").expect("built-in world type path is valid"),
        );
    }
    types
}

fn declared_field_types(compilation: &Compilation) -> BTreeMap<NodeId, BTreeMap<String, NodeId>> {
    let mut types = BTreeMap::<NodeId, BTreeMap<String, NodeId>>::new();
    for node in compilation
        .code_tree()
        .nodes()
        .iter()
        .filter(|node| node.kind == NodeKind::Variable)
    {
        let Some(owner) = node.owner_type else {
            continue;
        };
        let Some(name) = node.path.segments().last() else {
            continue;
        };
        let Some(definition) = node
            .declarations
            .iter()
            .filter_map(|id| compilation.code_tree().declaration(*id))
            .filter_map(|declaration| {
                compilation
                    .syntax(declaration.file_id)
                    .and_then(|syntax| syntax.definitions.get(declaration.definition_index))
            })
            .find(|definition| declared_type_path(&definition.header, name).is_some())
        else {
            continue;
        };
        let Some(path) = declared_type_path(&definition.header, name) else {
            continue;
        };
        let Some(field_type) = compilation.code_tree().find(&path) else {
            continue;
        };
        types
            .entry(owner)
            .or_default()
            .insert(name.clone(), field_type);
    }
    types
}

fn inherited_declared_field_type(
    compilation: &Compilation,
    field_types: &BTreeMap<NodeId, BTreeMap<String, NodeId>>,
    mut owner: NodeId,
    name: &str,
) -> Option<NodeId> {
    loop {
        if let Some(field_type) = field_types.get(&owner).and_then(|fields| fields.get(name)) {
            return Some(*field_type);
        }
        owner = compilation.code_tree().node(owner)?.parent_type?;
    }
}

fn dynamic_call_literal_selectors(definition: &dm_syntax::Definition) -> BTreeSet<String> {
    let mut selectors = BTreeSet::new();
    for line in &definition.body {
        let tokens = &line.tokens;
        for call_index in 0..tokens.len().saturating_sub(1) {
            if !matches!(&tokens[call_index].kind, TokenKind::Identifier(name) if name == "call")
                || !matches!(tokens[call_index + 1].kind, TokenKind::Punctuation('('))
            {
                continue;
            }
            let mut depth = 1usize;
            let mut separator = None;
            for (offset, token) in tokens[call_index + 2..].iter().enumerate() {
                match &token.kind {
                    TokenKind::Punctuation('(') => depth += 1,
                    TokenKind::Punctuation(')') => {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                    TokenKind::Punctuation(',') if depth == 1 => {
                        separator = Some(call_index + 2 + offset);
                        break;
                    }
                    _ => {}
                }
            }
            let selector_index = separator.map_or(call_index + 2, |index| index + 1);
            if let Some(TokenKind::String(selector) | TokenKind::RawString(selector)) =
                tokens.get(selector_index).map(|token| &token.kind)
            {
                selectors.insert(selector.clone());
            }
        }
    }
    selectors
}

#[derive(Default)]
struct ConstructionDependencies {
    targets: BTreeSet<ProcedureImplementationId>,
    unbounded: bool,
}

fn constructor_targets_by_ancestor(
    compilation: &Compilation,
    procedures: &[Procedure],
    by_owner_name: &BTreeMap<(Option<NodeId>, String), ProcedureId>,
) -> BTreeMap<NodeId, BTreeSet<ProcedureImplementationId>> {
    let mut result = BTreeMap::<NodeId, BTreeSet<ProcedureImplementationId>>::new();
    for ((owner, name), procedure) in by_owner_name {
        if name != "New" {
            continue;
        }
        let Some(mut ancestor) = *owner else {
            continue;
        };
        let Some(target) = procedures[procedure.index()].effective_target else {
            continue;
        };
        loop {
            result.entry(ancestor).or_default().insert(target);
            let Some(parent) = compilation
                .code_tree()
                .node(ancestor)
                .and_then(|node| node.parent_type)
            else {
                break;
            };
            ancestor = parent;
        }
    }
    result
}

fn effective_constructor_target(
    compilation: &Compilation,
    procedures: &[Procedure],
    by_owner_name: &BTreeMap<(Option<NodeId>, String), ProcedureId>,
    mut owner: NodeId,
) -> Option<ProcedureImplementationId> {
    loop {
        if let Some(target) = by_owner_name
            .get(&(Some(owner), "New".to_owned()))
            .and_then(|procedure| procedures[procedure.index()].effective_target)
        {
            return Some(target);
        }
        owner = compilation.code_tree().node(owner)?.parent_type?;
    }
}

fn construction_dependencies(
    definition: &dm_syntax::Definition,
    compilation: &Compilation,
    procedures: &[Procedure],
    by_owner_name: &BTreeMap<(Option<NodeId>, String), ProcedureId>,
    targets_by_ancestor: &BTreeMap<NodeId, BTreeSet<ProcedureImplementationId>>,
) -> ConstructionDependencies {
    let mut result = ConstructionDependencies::default();
    let declared = declared_receiver_types(definition)
        .into_iter()
        .filter_map(|(name, path)| compilation.code_tree().find(&path).map(|node| (name, node)))
        .collect::<BTreeMap<_, _>>();
    let mut families = declared.clone();

    // A loop variable drawn from typesof/subtypesof(/base) is a type path
    // bounded to that hierarchy even when the loop declaration itself omits a
    // datum annotation (a common tgstation bootstrap idiom).
    for line in &definition.body {
        let tokens = &line.tokens;
        let Some(in_index) = tokens
            .iter()
            .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "in"))
        else {
            continue;
        };
        let Some(variable) = tokens[..in_index]
            .iter()
            .rev()
            .find_map(|token| match &token.kind {
                TokenKind::Identifier(name)
                    if !matches!(name.as_str(), "for" | "var" | "as" | "anything") =>
                {
                    Some(name.clone())
                }
                _ => None,
            })
        else {
            continue;
        };
        let source = &tokens[in_index + 1..];
        let Some(function_index) = source.iter().position(|token| {
            matches!(&token.kind, TokenKind::Identifier(name) if matches!(name.as_str(), "typesof" | "subtypesof" | "typecacheof"))
        }) else {
            continue;
        };
        let path_start = source[function_index + 1..]
            .iter()
            .position(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "/"),
            )
            .map(|offset| function_index + 1 + offset);
        let Some(path_start) = path_start else {
            continue;
        };
        if let Some(owner) = type_node_from_tokens(compilation, source, path_start) {
            families.insert(variable, owner);
        }
    }

    let project_newlist = by_owner_name
        .keys()
        .any(|(owner, name)| owner.is_none() && name == "newlist");
    for line in &definition.body {
        let tokens = &line.tokens;
        if !project_newlist {
            for newlist_index in tokens.iter().enumerate().filter_map(|(index, token)| {
                matches!(&token.kind, TokenKind::Identifier(name) if name == "newlist")
                    .then_some(index)
            }) {
                if !matches!(
                    tokens.get(newlist_index + 1).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                ) {
                    continue;
                }
                let mut cursor = newlist_index + 2;
                let mut argument_start = cursor;
                let mut depth = 0usize;
                let mut closed = false;
                while cursor < tokens.len() {
                    let boundary = match tokens[cursor].kind {
                        TokenKind::Punctuation('(' | '[' | '{') => {
                            depth += 1;
                            false
                        }
                        TokenKind::Punctuation(')') if depth == 0 => {
                            closed = true;
                            true
                        }
                        TokenKind::Punctuation(')' | ']' | '}') => {
                            depth = depth.saturating_sub(1);
                            false
                        }
                        TokenKind::Punctuation(',') if depth == 0 => true,
                        _ => false,
                    };
                    if boundary {
                        let argument = &tokens[argument_start..cursor];
                        if !argument.is_empty() {
                            if matches!(argument.first().map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
                            {
                                if let Some(owner) = type_node_from_tokens(compilation, argument, 0)
                                    && let Some(target) = effective_constructor_target(
                                        compilation,
                                        procedures,
                                        by_owner_name,
                                        owner,
                                    )
                                {
                                    result.targets.insert(target);
                                }
                            } else {
                                result.unbounded = true;
                            }
                        }
                        argument_start = cursor + 1;
                        if closed {
                            break;
                        }
                    }
                    cursor += 1;
                }
                if !closed {
                    result.unbounded = true;
                }
            }
        }
        for new_index in tokens.iter().enumerate().filter_map(|(index, token)| {
            matches!(&token.kind, TokenKind::Identifier(name) if name == "new").then_some(index)
        }) {
            let next = new_index + 1;
            if matches!(tokens.get(next).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
            {
                if let Some(owner) = type_node_from_tokens(compilation, tokens, next)
                    && let Some(target) =
                        effective_constructor_target(compilation, procedures, by_owner_name, owner)
                {
                    result.targets.insert(target);
                }
                continue;
            }
            if let Some(TokenKind::Identifier(name)) = tokens.get(next).map(|token| &token.kind) {
                if let Some(owner) = families.get(name)
                    && let Some(targets) = targets_by_ancestor.get(owner)
                {
                    result.targets.extend(targets.iter().copied());
                } else {
                    result.unbounded = true;
                }
                continue;
            }
            // Bare `new(...)` is destination-typed. Recover a declared local
            // type from the assignment target; otherwise remain conservative.
            let assignment = tokens[..new_index].iter().rposition(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
            );
            let destination = assignment.and_then(|assignment| {
                tokens[..assignment]
                    .iter()
                    .rev()
                    .find_map(|token| match &token.kind {
                        TokenKind::Identifier(name) if name != "var" => Some(name),
                        _ => None,
                    })
            });
            if let Some(owner) = destination.and_then(|name| declared.get(name)) {
                if let Some(target) =
                    effective_constructor_target(compilation, procedures, by_owner_name, *owner)
                {
                    result.targets.insert(target);
                }
            } else {
                result.unbounded = true;
            }
        }
    }
    result
}

fn type_node_from_tokens(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
    mut index: usize,
) -> Option<NodeId> {
    let mut segments = Vec::new();
    while matches!(tokens.get(index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
    {
        let TokenKind::Identifier(segment) = tokens.get(index + 1).map(|token| &token.kind)? else {
            break;
        };
        segments.push(segment.clone());
        index += 2;
    }
    (!segments.is_empty())
        .then(|| {
            compilation
                .code_tree()
                .find(&dm_syntax::DefinitionPath::new(segments))
        })
        .flatten()
}

fn static_proc_reference_paths(
    definition: &dm_syntax::Definition,
    procedure_path: &CodePath,
) -> BTreeSet<String> {
    let procedure_path = procedure_path.to_string();
    let owner_path = procedure_path
        .split_once("/proc/")
        .map(|(owner, _)| owner.to_owned());
    fn collect(
        tokens: &[dm_lexer::SpannedToken],
        owner_path: Option<&str>,
        paths: &mut BTreeSet<String>,
    ) {
        let mut index = 0usize;
        while index < tokens.len() {
            if let TokenKind::String(value) | TokenKind::RawString(value) = &tokens[index].kind {
                let segments = value
                    .strip_prefix('/')
                    .map(|path| path.split('/').collect::<Vec<_>>())
                    .unwrap_or_default();
                if let Some(proc_index) = segments.iter().position(|segment| *segment == "proc")
                    && proc_index + 1 < segments.len()
                    && segments
                        .iter()
                        .all(|segment| !segment.is_empty() && is_identifier_text(segment))
                {
                    paths.insert(value.clone());
                }
            }
            // tgstation's PROC_REF(name) expands to nameof(.proc/name).
            // Although the resulting selector is text at runtime, the
            // receiver is the current datum and the target family is exactly
            // bounded by the owning type at compile time.
            if matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == ".")
                && matches!(tokens.get(index + 1).map(|token| &token.kind), Some(TokenKind::Identifier(segment)) if segment == "proc")
                && matches!(tokens.get(index + 2).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
                && let Some(TokenKind::Identifier(name)) =
                    tokens.get(index + 3).map(|token| &token.kind)
                && let Some(owner_path) = owner_path
            {
                paths.insert(format!("{owner_path}/proc/{name}"));
                index += 4;
                continue;
            }
            if !matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == "/") {
                index += 1;
                continue;
            }
            let start = index;
            let mut segments = Vec::new();
            while index + 1 < tokens.len()
                && matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == "/")
            {
                let TokenKind::Identifier(segment) = &tokens[index + 1].kind else {
                    break;
                };
                segments.push(segment.clone());
                index += 2;
            }
            // tgstation's TYPE_PROC_REF(/owner/type, name) expands to
            // nameof(/owner/type.proc/name). Retain that exact callback even
            // though the parser represents the owner path and `.proc/name`
            // as separate token runs.
            if !segments.is_empty()
                && matches!(tokens.get(index).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == ".")
                && matches!(tokens.get(index + 1).map(|token| &token.kind), Some(TokenKind::Identifier(segment)) if segment == "proc")
                && matches!(tokens.get(index + 2).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
                && let Some(TokenKind::Identifier(name)) =
                    tokens.get(index + 3).map(|token| &token.kind)
            {
                paths.insert(format!("/{}/proc/{name}", segments.join("/")));
                index += 4;
                continue;
            }
            if let Some(proc_index) = segments.iter().position(|segment| segment == "proc")
                && proc_index + 1 < segments.len()
            {
                paths.insert(format!("/{}", segments.join("/")));
            }
            if index == start {
                index += 1;
            }
        }
    }

    let mut paths = BTreeSet::new();
    // The full header begins with this procedure's own canonical `/proc/...`
    // declaration path. Treating that syntax as a first-class reference made
    // every body retain itself and, before path indexing, triggered one full
    // registry scan per procedure. Parameter token lists retain the only
    // header expressions that can genuinely reference another procedure.
    for parameter in &definition.parameters {
        collect(&parameter.tokens, owner_path.as_deref(), &mut paths);
    }
    for line in &definition.body {
        collect(&line.tokens, owner_path.as_deref(), &mut paths);
    }
    paths
}

fn static_procedure_type_families(definition: &dm_syntax::Definition) -> BTreeSet<Vec<String>> {
    fn collect(tokens: &[dm_lexer::SpannedToken], families: &mut BTreeSet<Vec<String>>) {
        for (index, token) in tokens.iter().enumerate() {
            if !matches!(&token.kind, TokenKind::Identifier(name) if name == "typesof")
                || !matches!(
                    tokens.get(index + 1).map(|token| &token.kind),
                    Some(TokenKind::Punctuation('('))
                )
            {
                continue;
            }
            let mut cursor = index + 2;
            let mut segments = Vec::new();
            while matches!(tokens.get(cursor).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
            {
                let Some(TokenKind::Identifier(segment)) =
                    tokens.get(cursor + 1).map(|token| &token.kind)
                else {
                    break;
                };
                segments.push(segment.clone());
                cursor += 2;
            }
            if segments.last().is_some_and(|segment| segment == "proc") {
                families.insert(segments);
            }
        }
    }

    let mut families = BTreeSet::new();
    for parameter in &definition.parameters {
        collect(&parameter.tokens, &mut families);
    }
    for line in &definition.body {
        collect(&line.tokens, &mut families);
    }
    families
}

fn is_identifier_text(value: &str) -> bool {
    let mut characters = value.chars();
    characters
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn member_call_dependencies(
    definition: &dm_syntax::Definition,
    owner: Option<NodeId>,
    compilation: &Compilation,
    global_types: &BTreeMap<String, TypePath>,
    field_types: &BTreeMap<NodeId, BTreeMap<String, NodeId>>,
    by_owner_name: &BTreeMap<(Option<NodeId>, String), ProcedureId>,
    dynamic_targets: &BTreeMap<String, Vec<ProcedureImplementationId>>,
    procedures: &[Procedure],
) -> (
    BTreeSet<ProcedureImplementationId>,
    BTreeSet<ProcedureImplementationId>,
    BTreeSet<String>,
) {
    let receiver_types = declared_receiver_types(definition);
    let mut flowing_types = receiver_types
        .iter()
        .filter_map(|(name, path)| {
            compilation
                .code_tree()
                .find(path)
                .map(|node| (name.clone(), node))
        })
        .collect::<BTreeMap<_, _>>();
    let declared_names = flowing_types.keys().cloned().collect::<BTreeSet<_>>();
    let mut exact = BTreeSet::new();
    let mut typed_virtual = BTreeSet::new();
    let mut unresolved = BTreeSet::new();
    for line in &definition.body {
        // Interpolated DM text stores its embedded expressions inside a single
        // String token. The VM reparses those expressions when lowering the
        // body, so their member calls must participate in symbolic linking as
        // well. Receiver typing cannot be recovered reliably from decoded
        // string text (nested quoted arguments are legal), therefore retain
        // matching member symbols conservatively and leave exact virtual
        // selection to runtime dispatch.
        for token in &line.tokens {
            if let TokenKind::String(text) | TokenKind::RawString(text) = &token.kind {
                collect_text_member_call_selectors(text, &mut unresolved);
            }
        }
        for selector_index in 2..line.tokens.len().saturating_sub(1) {
            let dot = &line.tokens[selector_index - 1];
            let selector = &line.tokens[selector_index];
            let open = &line.tokens[selector_index + 1];
            if !matches!(&dot.kind, TokenKind::Operator(operator) if operator == "." || operator == "?.")
                || !matches!(open.kind, TokenKind::Punctuation('('))
            {
                continue;
            }
            let TokenKind::Identifier(selector) = &selector.kind else {
                continue;
            };
            let receiver_end = selector_index - 1;
            // `world` is an engine-owned, typed global receiver. Prefer that
            // exact binding before examining a larger enclosing expression:
            // macro-expanded arguments such as
            // `list.Add("key", world.some_proc())` otherwise let the broad
            // prefix scan incorrectly prove `list` as the receiver and retain
            // the wrong same-named member family.
            let receiver_node = matches!(
                line.tokens
                    .get(receiver_end.saturating_sub(1))
                    .map(|token| &token.kind),
                Some(TokenKind::Identifier(identifier)) if identifier == "world"
            )
            .then(|| {
                compilation
                    .code_tree()
                    .find(&dm_syntax::DefinitionPath::new(vec!["world".to_owned()]))
            })
            .flatten()
            .or_else(|| {
                (0..receiver_end).find_map(|start| {
                    proven_receiver_expression_type(
                        compilation,
                        owner,
                        &line.tokens[start..receiver_end],
                        &flowing_types,
                        global_types,
                        field_types,
                        by_owner_name,
                        procedures,
                    )
                })
            });
            let Some(mut receiver_node) = receiver_node else {
                unresolved.insert(selector.clone());
                continue;
            };
            let declared_receiver = receiver_node;
            let mut target = None;
            loop {
                if let Some(procedure) = by_owner_name
                    .get(&(Some(receiver_node), selector.clone()))
                    .and_then(|id| procedures.get(id.index()))
                    && let Some(effective) = effective_target(procedures, procedure.id)
                {
                    target = Some(effective);
                    break;
                }
                let Some(parent) = compilation
                    .code_tree()
                    .node(receiver_node)
                    .and_then(|node| node.parent_type)
                else {
                    break;
                };
                receiver_node = parent;
            }
            if let Some(target) = target {
                exact.insert(target);
                // A typed DM variable may hold any compatible subtype. Keep
                // runtime virtual dispatch while narrowing the linked module
                // to overrides beneath the declared receiver type.
                for &candidate in dynamic_targets.get(selector).into_iter().flatten() {
                    let Some(candidate_owner) = procedures
                        .get(candidate.procedure().index())
                        .and_then(|procedure| procedure.owner_type)
                    else {
                        continue;
                    };
                    if type_is_descendant_or_same(compilation, candidate_owner, declared_receiver) {
                        if candidate != target {
                            typed_virtual.insert(candidate);
                        }
                    }
                }
            } else {
                unresolved.insert(selector.clone());
            }
        }
        if let Some(assignment) = top_level_simple_assignment(&line.tokens) {
            let lhs = &line.tokens[..assignment];
            let rhs = &line.tokens[assignment + 1..];
            let local = lhs.iter().rev().find_map(|token| match &token.kind {
                TokenKind::Identifier(name) if !matches!(name.as_str(), "var" | "as") => {
                    Some(name.clone())
                }
                _ => None,
            });
            if let Some(local) = local
                && !lhs.iter().any(|token| {
                    matches!(&token.kind, TokenKind::Operator(operator) if operator == "." || operator == "?.")
                        || matches!(token.kind, TokenKind::Punctuation('['))
                })
                && !declared_names.contains(&local)
            {
                if let Some(alias_type) = proven_field_chain_type(
                    compilation,
                    owner,
                    rhs,
                    &flowing_types,
                    global_types,
                    field_types,
                ) {
                    flowing_types.insert(local, alias_type);
                } else {
                    flowing_types.remove(&local);
                }
            }
        }
    }
    (exact, typed_virtual, unresolved)
}

fn receiver_root_type(
    compilation: &Compilation,
    owner: Option<NodeId>,
    name: &str,
    local_types: &BTreeMap<String, NodeId>,
    global_types: &BTreeMap<String, TypePath>,
    field_types: &BTreeMap<NodeId, BTreeMap<String, NodeId>>,
) -> Option<NodeId> {
    if name == "src" {
        return owner;
    }
    local_types
        .get(name)
        .copied()
        .or_else(|| {
            global_types.get(name).and_then(|path| {
                let segments = path
                    .as_str()
                    .trim_matches('/')
                    .split('/')
                    .map(str::to_owned)
                    .collect::<Vec<_>>();
                compilation
                    .code_tree()
                    .find(&dm_syntax::DefinitionPath::new(segments))
            })
        })
        .or_else(|| {
            owner.and_then(|owner| {
                inherited_declared_field_type(compilation, field_types, owner, name)
            })
        })
}

fn proven_field_chain_type(
    compilation: &Compilation,
    owner: Option<NodeId>,
    tokens: &[dm_lexer::SpannedToken],
    local_types: &BTreeMap<String, NodeId>,
    global_types: &BTreeMap<String, TypePath>,
    field_types: &BTreeMap<NodeId, BTreeMap<String, NodeId>>,
) -> Option<NodeId> {
    let TokenKind::Identifier(first) = &tokens.first()?.kind else {
        return None;
    };
    let mut current = receiver_root_type(
        compilation,
        owner,
        first,
        local_types,
        global_types,
        field_types,
    )?;
    let mut index = 1;
    while index < tokens.len() {
        if !matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == "." || operator == "?.")
        {
            return None;
        }
        let TokenKind::Identifier(member) = &tokens.get(index + 1)?.kind else {
            return None;
        };
        current = inherited_declared_field_type(compilation, field_types, current, member)?;
        index += 2;
    }
    Some(current)
}

#[allow(clippy::too_many_arguments)]
fn proven_receiver_expression_type(
    compilation: &Compilation,
    owner: Option<NodeId>,
    tokens: &[dm_lexer::SpannedToken],
    local_types: &BTreeMap<String, NodeId>,
    global_types: &BTreeMap<String, TypePath>,
    field_types: &BTreeMap<NodeId, BTreeMap<String, NodeId>>,
    by_owner_name: &BTreeMap<(Option<NodeId>, String), ProcedureId>,
    procedures: &[Procedure],
) -> Option<NodeId> {
    if let Some((question, colon)) = top_level_ternary(tokens) {
        let truthy = proven_receiver_expression_type(
            compilation,
            owner,
            &tokens[question + 1..colon],
            local_types,
            global_types,
            field_types,
            by_owner_name,
            procedures,
        )?;
        let falsey = proven_receiver_expression_type(
            compilation,
            owner,
            &tokens[colon + 1..],
            local_types,
            global_types,
            field_types,
            by_owner_name,
            procedures,
        )?;
        return (truthy == falsey).then_some(truthy);
    }
    if let Some(field) = proven_field_chain_type(
        compilation,
        owner,
        tokens,
        local_types,
        global_types,
        field_types,
    ) {
        return Some(field);
    }
    if matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Punctuation('('))
    ) && matching_closing(tokens, 0, '(', ')') == Some(tokens.len() - 1)
    {
        return proven_receiver_expression_type(
            compilation,
            owner,
            &tokens[1..tokens.len() - 1],
            local_types,
            global_types,
            field_types,
            by_owner_name,
            procedures,
        );
    }
    let close = tokens.len().checked_sub(1)?;
    if !matches!(tokens.get(close)?.kind, TokenKind::Punctuation(')')) {
        return None;
    }
    let open = (0..close).rev().find(|&index| {
        matches!(tokens[index].kind, TokenKind::Punctuation('('))
            && matching_closing(tokens, index, '(', ')') == Some(close)
    })?;
    let TokenKind::Identifier(selector) = &tokens.get(open.checked_sub(1)?)?.kind else {
        return None;
    };
    let procedure_node = if open >= 3
        && matches!(&tokens[open - 2].kind, TokenKind::Operator(operator) if operator == "." || operator == "?.")
    {
        let receiver_type = proven_receiver_expression_type(
            compilation,
            owner,
            &tokens[..open - 2],
            local_types,
            global_types,
            field_types,
            by_owner_name,
            procedures,
        )?;
        find_member_node(compilation, receiver_type, "proc", selector)
    } else if open == 1 {
        let mut current = owner;
        let mut found = None;
        while let Some(node) = current {
            if let Some(procedure) = by_owner_name
                .get(&(Some(node), selector.clone()))
                .and_then(|id| procedures.get(id.index()))
            {
                found = Some(procedure.node);
                break;
            }
            current = compilation
                .code_tree()
                .node(node)
                .and_then(|node| node.parent_type);
        }
        found.or_else(|| {
            by_owner_name
                .get(&(None, selector.clone()))
                .and_then(|id| procedures.get(id.index()))
                .map(|procedure| procedure.node)
        })
    } else {
        None
    }?;
    effective_datum_return(compilation, procedure_node)
}

fn type_is_descendant_or_same(
    compilation: &Compilation,
    mut candidate: NodeId,
    expected: NodeId,
) -> bool {
    loop {
        if candidate == expected {
            return true;
        }
        let Some(parent) = compilation
            .code_tree()
            .node(candidate)
            .and_then(|node| node.parent_type)
        else {
            return false;
        };
        candidate = parent;
    }
}

fn static_call_selectors(definition: &dm_syntax::Definition) -> BTreeSet<String> {
    let mut selectors = BTreeSet::new();
    // Default argument expressions execute as part of the procedure call and
    // therefore participate in the same static call graph as its body.
    collect_call_selectors(&definition.header, &mut selectors);
    for line in &definition.body {
        collect_call_selectors(&line.tokens, &mut selectors);
        for token in &line.tokens {
            if let TokenKind::String(text) | TokenKind::RawString(text) = &token.kind {
                collect_text_call_selectors(text, &mut selectors);
            }
        }
    }
    selectors
}

fn referenced_identifiers(definition: &dm_syntax::Definition) -> BTreeSet<String> {
    fn collect(tokens: &[dm_lexer::SpannedToken], names: &mut BTreeSet<String>) {
        for token in tokens {
            match &token.kind {
                TokenKind::Identifier(name) => {
                    names.insert(name.clone());
                }
                TokenKind::String(text) | TokenKind::RawString(text) => {
                    // The token stores decoded DM text without its outer
                    // quotes. Re-lexing the whole value is unreliable when
                    // interpolation itself contains quoted arguments (for
                    // example `[join(values, ", ")]`). Conservatively retain
                    // every identifier-shaped word; later binding lookup
                    // filters this set to real src/global fields.
                    let bytes = text.as_bytes();
                    let mut index = 0;
                    while index < bytes.len() {
                        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
                            let start = index;
                            index += 1;
                            while index < bytes.len()
                                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
                            {
                                index += 1;
                            }
                            names.insert(text[start..index].to_owned());
                        } else {
                            index += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }

    let mut names = BTreeSet::new();
    collect(&definition.header, &mut names);
    for line in &definition.body {
        collect(&line.tokens, &mut names);
    }
    names
}

fn collect_text_call_selectors(text: &str, selectors: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index].is_ascii_alphabetic() || bytes[index] == b'_' {
            let start = index;
            index += 1;
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
            {
                index += 1;
            }
            let mut next = index;
            while next < bytes.len() && bytes[next].is_ascii_whitespace() {
                next += 1;
            }
            if bytes.get(next) == Some(&b'(') {
                selectors.insert(text[start..index].to_owned());
            }
        } else {
            index += 1;
        }
    }
}

fn collect_text_member_call_selectors(text: &str, selectors: &mut BTreeSet<String>) {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'.' {
            index += 1;
            continue;
        }
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_whitespace() {
            index += 1;
        }
        let start = index;
        if bytes
            .get(index)
            .is_none_or(|byte| !byte.is_ascii_alphabetic() && *byte != b'_')
        {
            continue;
        }
        index += 1;
        while index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_')
        {
            index += 1;
        }
        let mut next = index;
        while next < bytes.len() && bytes[next].is_ascii_whitespace() {
            next += 1;
        }
        if bytes.get(next) == Some(&b'(') {
            selectors.insert(text[start..index].to_owned());
        }
    }
}

fn collect_call_selectors(tokens: &[dm_lexer::SpannedToken], selectors: &mut BTreeSet<String>) {
    for index in 0..tokens.len().saturating_sub(1) {
        if index > 0 {
            match &tokens[index - 1].kind {
                TokenKind::Operator(operator) if matches!(operator.as_str(), "." | "?.") => {
                    continue;
                }
                TokenKind::Operator(operator)
                    if operator == ":" && !ternary_colon(tokens, index - 1) =>
                {
                    continue;
                }
                _ => {}
            }
        }
        if let (TokenKind::Identifier(name), TokenKind::Punctuation('(')) =
            (&tokens[index].kind, &tokens[index + 1].kind)
        {
            selectors.insert(name.clone());
        }
    }
}

fn ternary_colon(tokens: &[dm_lexer::SpannedToken], colon: usize) -> bool {
    // A ternary can appear inside any delimited expression, notably
    // `new(user ? user : drop_location())`. Keep an independent pending count
    // at each delimiter depth so a colon in that expression is not mistaken
    // for DM's dynamic member-access syntax.
    let mut pending_by_depth = vec![0usize];
    for token in &tokens[..colon] {
        match &token.kind {
            TokenKind::Punctuation('(' | '[' | '{') => pending_by_depth.push(0),
            TokenKind::Operator(operator) if operator == "?[" => pending_by_depth.push(0),
            TokenKind::Punctuation(')' | ']' | '}') if pending_by_depth.len() > 1 => {
                pending_by_depth.pop();
            }
            TokenKind::Operator(operator) if operator == "?" => {
                *pending_by_depth
                    .last_mut()
                    .expect("the root delimiter depth always exists") += 1;
            }
            TokenKind::Operator(operator)
                if operator == ":"
                    && pending_by_depth.last().is_some_and(|pending| *pending > 0) =>
            {
                *pending_by_depth
                    .last_mut()
                    .expect("the root delimiter depth always exists") -= 1;
            }
            _ => {}
        }
    }
    pending_by_depth.last().is_some_and(|pending| *pending > 0)
}

fn direct_instance_fields(
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, FieldName>> {
    let mut fields = BTreeMap::<NodeId, BTreeMap<String, FieldName>>::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage == StorageClass::Instance)
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        if let Ok(field) = FieldName::parse(name) {
            fields
                .entry(owner)
                .or_default()
                .insert(name.to_owned(), field);
        }
    }
    fields
}

fn direct_instance_field_types(
    compilation: &Compilation,
    registry: &VariableRegistry,
) -> BTreeMap<NodeId, BTreeMap<String, TypePath>> {
    let mut types = BTreeMap::<NodeId, BTreeMap<String, TypePath>>::new();
    for entry in registry
        .entries()
        .iter()
        .filter(|entry| entry.storage == StorageClass::Instance)
    {
        let Some(owner) = entry.owner.as_ref().map(|owner| owner.node) else {
            continue;
        };
        let Some(name) = entry.path.rsplit('/').next() else {
            continue;
        };
        let Some(definition) = compilation
            .syntax(entry.file_id)
            .and_then(|syntax| syntax.definitions.get(entry.definition_index))
        else {
            continue;
        };
        let Some(path) = declared_type_path(&definition.header, name) else {
            continue;
        };
        let Ok(path) = TypePath::parse(&path.to_string()) else {
            continue;
        };
        types
            .entry(owner)
            .or_default()
            .insert(name.to_owned(), path);
    }
    types
}

fn referenced_inherited_field_types(
    compilation: &Compilation,
    owner: NodeId,
    direct_types: &BTreeMap<NodeId, BTreeMap<String, TypePath>>,
    referenced: &BTreeSet<String>,
) -> BTreeMap<String, TypePath> {
    let tree = compilation.code_tree();
    let mut current = Some(owner);
    let mut unresolved = referenced.clone();
    let mut types = BTreeMap::new();
    while let Some(node) = current {
        if let Some(available) = direct_types.get(&node) {
            let resolved = unresolved
                .iter()
                .filter_map(|name| available.get(name).map(|path| (name.clone(), path.clone())))
                .collect::<Vec<_>>();
            for (name, path) in resolved {
                unresolved.remove(&name);
                types.insert(name, path);
            }
        }
        if unresolved.is_empty() {
            break;
        }
        current = tree.node(node).and_then(|type_node| type_node.parent_type);
    }
    types
}

/// Adds the fields supplied by BYOND's built-in datum and atom hierarchies.
///
/// The object tree deliberately seeds only standard *types*, since their
/// members have no user source declaration. VM lowering still needs the
/// corresponding names, however, so bare reads such as `type`, `loc`, and
/// `dir` lower exactly like declared `src` fields. Keep this catalog at the
/// semantic boundary: atom-only names must not become visible on arbitrary
/// `/datum`s.
fn standard_instance_fields(path: Option<&CodePath>, fields: &mut BTreeMap<String, FieldName>) {
    let Some(path) = path else {
        return;
    };
    let path = path.to_string();
    let names = standard_instance_field_names(&path);
    if names.is_empty() {
        return;
    }
    for name in names {
        // All catalog entries are fixed, valid DM identifiers.
        fields.insert(
            (*name).to_owned(),
            FieldName::parse(name).expect("standard field name is valid"),
        );
    }
}

/// Returns the engine-owned fields declared directly by a built-in DM type.
///
/// This is public so runtime materialization tests can enforce that the
/// semantic catalog and concrete datum defaults never drift apart.
#[doc(hidden)]
#[must_use]
pub fn standard_instance_field_names(path: &str) -> &'static [&'static str] {
    match path {
        // Every datum exposes its canonical runtime type through this
        // read-only built-in field. The VM materializes its value from the
        // datum record rather than from a user-declared default.
        "/datum" => &["datum_flags", "tag", "type", "parent_type", "vars"],
        "/world" => &[
            "system_type",
            "icon_size",
            "tick_lag",
            "fps",
            "timezone",
            "cpu",
            "time",
            "timeofday",
            "realtime",
            "maxx",
            "maxy",
            "maxz",
            "params",
            "log",
            "name",
            "hub",
            "hub_password",
            "internet_address",
            "address",
            "status",
            "port",
            "area",
            "mob",
            "turf",
            "byond_version",
            "byond_build",
            "cache_lifespan",
            "executor",
            "game_state",
            "host",
            "loop_checks",
            "map_format",
            "map_cpu",
            "movement_mode",
            "process",
            "reachable",
            "sleep_offline",
            "tick_usage",
            "url",
            "version",
            "view",
            "visibility",
        ],
        "/atom" => &[
            "alpha",
            "appearance",
            "appearance_flags",
            "blend_mode",
            "color",
            "contents",
            "density",
            "desc",
            "dir",
            "gender",
            "filters",
            "icon",
            "icon_state",
            "invisibility",
            "layer",
            "loc",
            "luminosity",
            "maptext",
            "maptext_height",
            "maptext_width",
            "maptext_x",
            "maptext_y",
            "mouse_opacity",
            "mouse_over_pointer",
            "name",
            "opacity",
            "overlays",
            "particles",
            "plane",
            "pixel_x",
            "pixel_y",
            "pixel_w",
            "pixel_z",
            "render_source",
            "render_target",
            "suffix",
            "text",
            "transform",
            "underlays",
            "vis_contents",
            "vis_locs",
            "vis_flags",
            "verbs",
            "x",
            "y",
            "z",
        ],
        "/atom/movable" => &[
            "animate_movement",
            "bound_height",
            "bound_width",
            "bound_x",
            "bound_y",
            "glide_size",
            "locs",
            "screen_loc",
            "step_x",
            "step_y",
            "step_size",
        ],
        "/mob" => &[
            "ckey",
            "client",
            "eye",
            "key",
            "perspective",
            "see_in_dark",
            "see_infrared",
            "see_invisible",
            "sight",
        ],
        "/client" => &[
            "address",
            "ckey",
            "computer_id",
            "connection",
            "control_freak",
            "dir",
            "gender",
            "byond_build",
            "byond_version",
            "key",
            "eye",
            "fps",
            "images",
            "inactivity",
            "mob",
            "mouse_pointer_icon",
            "perspective",
            "pixel_w",
            "pixel_x",
            "pixel_y",
            "pixel_z",
            "screen",
            "statobj",
            "verbs",
            "view",
        ],
        "/matrix" => &["a", "b", "c", "d", "e", "f"],
        // BYOND exposes the state of the most recent regex operation as
        // ordinary fields. Map readers in tg-derived projects use `next`
        // directly to advance a global regex sweep.
        "/regex" => &["text", "flags", "match", "index", "group", "next"],
        // `/sound` is an engine value with fields supplied by BYOND even when
        // no project declaration exists. OpenDream exposes the core fields
        // through DreamObjectSound; BYOND also exposes constructor controls.
        "/sound" => &[
            "file",
            "repeat",
            "wait",
            "channel",
            "volume",
            "frequency",
            "pan",
            "offset",
        ],
        "/particles" => &[
            "color",
            "width",
            "height",
            "count",
            "spawning",
            "bound1",
            "bound2",
            "gravity",
            "gradient",
            "color_change",
            "transform",
            "icon",
            "icon_state",
            "lifespan",
            "fadein",
            "fade",
            "position",
            "velocity",
            "scale",
            "grow",
            "rotation",
            "spin",
            "friction",
            "drift",
        ],
        // `/image` is an engine-owned appearance datum rather than an atom,
        // but BYOND exposes the same mutable appearance surface used by
        // overlays. These fields exist without user declarations.
        "/image" => &[
            "alpha",
            "appearance",
            "appearance_flags",
            "blend_mode",
            "color",
            "dir",
            "icon",
            "icon_state",
            "layer",
            "loc",
            "name",
            "overlays",
            "plane",
            "pixel_x",
            "pixel_y",
            "pixel_w",
            "pixel_z",
            "transform",
            "underlays",
            "vis_contents",
        ],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};
    use dm_globals::VariableRegistry;
    use dm_value::{FieldName, TypePath};
    use dm_vm::{
        ExecutionContext, ExecutionState, Instruction, RuntimeError, Value, execute_module,
        execute_module_in_context, execute_module_in_state,
    };

    use super::{
        ExecutableProcedures, Procedure, ProcedureImplementationKind, ProcedureRegistry,
        direct_static_fields, inherited_static_fields,
    };

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn compile(source: &str) -> Compilation {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-semantics-{}-{}",
                ordinal,
                std::process::id()
            ));
            fs::create_dir(&root).expect("test project directory should be created");
            let project = Self { root };
            fs::write(project.root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(project.root.join("types.dm"), source).expect("source should be written");
            CompilerDatabase::new()
                .compile(project.root.join("world.dme"))
                .expect("test project should compile")
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                eprintln!(
                    "failed to clean test project {}: {error}",
                    self.root.display()
                );
            }
        }
    }

    #[test]
    fn compiled_executable_artifact_round_trips_eager_module_and_semantic_mapping() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/value()\n\t\treturn 1\n/datum/child\n\tparent_type = /datum/base\n\tvalue()\n\t\treturn ..() + 1\n/proc/read(datum/child/source)\n\treturn source.value()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry
            .compile_vm_all_symbolic_deferred(&compilation)
            .expect("symbolic module should link")
            .into_fully_eager()
            .expect("fixture procedures should lower eagerly");

        let encoded = executable
            .encode_compiled_artifact()
            .expect("eager executable should encode");
        let segments = executable
            .encode_compiled_artifact_segments()
            .expect("segmented executable should encode");
        assert_eq!(segments.len(), 3);
        assert_eq!(segments.concat(), encoded);
        assert_eq!(
            encoded,
            executable
                .encode_compiled_artifact()
                .expect("encoding should be deterministic")
        );
        let decoded = ExecutableProcedures::decode_compiled_artifact(&encoded)
            .expect("executable should decode");
        assert_eq!(decoded.module(), executable.module());
        assert_eq!(decoded.stats(), executable.stats());
        for procedure in registry.procedures() {
            for implementation in &procedure.implementations {
                let before = executable
                    .implementation(implementation.id)
                    .expect("linked implementation should exist");
                let after = decoded
                    .implementation(implementation.id)
                    .expect("decoded implementation should exist");
                assert_eq!(
                    executable.module().procedure_path(before),
                    decoded.module().procedure_path(after)
                );
            }
        }
        assert_eq!(decoded.module().deferred_procedure_count(), 0);

        let mut bad_header = encoded.clone();
        bad_header[0] ^= 0xff;
        assert!(ExecutableProcedures::decode_compiled_artifact(&bad_header).is_err());
        assert!(
            ExecutableProcedures::decode_compiled_artifact(&encoded[..encoded.len() - 1]).is_err()
        );
        let mut trailing = encoded;
        trailing.push(0);
        assert!(ExecutableProcedures::decode_compiled_artifact(&trailing).is_err());
    }

    #[test]
    fn avd_empty_variadic_signature_preserves_rhs_input_constraints() {
        let compilation = TestProject::compile(concat!(
            "/datum/admin_verb/set_server_fps/__avd_do_verb(client/user,)\n",
            "\tvar/cfg_fps = 20\n",
            "\tvar/new_fps = round(input(user, \"FPS\", \"FPS\", 20) as num | null)\n",
            "\tif(new_fps <= 0)\n",
            "\t\treturn cfg_fps\n",
            "\treturn new_fps\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry
            .compile_vm_all_symbolic_deferred(&compilation)
            .expect("AVD-shaped symbolic module should link")
            .into_fully_eager()
            .expect("RHS input constraints must survive semantic normalization");

        assert_eq!(executable.module().deferred_procedure_count(), 0);
        assert!(executable.module().procedure_paths().any(|path| {
            path.starts_with("/datum/admin_verb/set_server_fps/proc/__avd_do_verb@")
        }));
    }

    fn procedure_by_path<'registry>(
        registry: &'registry ProcedureRegistry,
        path: &str,
    ) -> &'registry Procedure {
        registry
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == path)
            .expect("procedure path should exist")
    }

    fn execute_effective(
        compilation: &Compilation,
        path: &str,
        arguments: &[Value],
    ) -> Result<Value, RuntimeError> {
        let registry = ProcedureRegistry::build(compilation);
        let procedure = procedure_by_path(&registry, path);
        let target = procedure
            .effective_target
            .expect("procedure should have an effective implementation");
        let executable = registry
            .compile_vm(compilation)
            .expect("procedure registry should compile to VM bytecode");
        let entry = executable
            .implementation(target)
            .expect("effective implementation should have a VM identity");
        execute_module(executable.module(), entry, arguments)
    }

    #[test]
    fn independent_body_compilation_links_known_external_calls_to_stubs() {
        let compilation = TestProject::compile(
            "/proc/helper()\n\treturn 1\n/proc/caller()\n\treturn \"[helper()]\"\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let caller = procedure_by_path(&registry, "/proc/caller")
            .effective_target
            .expect("caller implementation should exist");
        let results = registry.compile_vm_bodies_independently(&compilation, [caller]);
        assert_eq!(results.len(), 1);
        results
            .into_iter()
            .next()
            .expect("caller result should exist")
            .1
            .expect("known external call should lower through an inert inventory stub");
    }

    #[test]
    fn indexes_a_base_procedure_with_source_identity() {
        let compilation = TestProject::compile("/datum/base\n\tproc/run()\n\t\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert!(procedure.owner_type.is_some());
        assert_eq!(procedure.implementations.len(), 1);
        assert_eq!(
            procedure.implementations[0].kind,
            ProcedureImplementationKind::Declaration
        );
        assert_eq!(procedure.implementations[0].definition_index, 1);
        assert!(!procedure.implementations[0].span.is_empty());
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[0].id)
        );
    }

    #[test]
    fn rejects_writes_to_global_local_and_inherited_const_variables() {
        let cases = [
            (
                "global",
                "var/const/answer = 42\n/proc/RunTest()\n\tanswer = 7\n",
                "answer",
            ),
            (
                "local",
                "/proc/RunTest()\n\tvar/const/answer = 42\n\tanswer += 1\n",
                "answer",
            ),
            (
                "inherited",
                "/datum/base\n\tvar/const/answer = 42\n/datum/base/child\n\tproc/change()\n\t\tanswer = 7\n",
                "answer",
            ),
            (
                "prefix mutation",
                "/proc/RunTest()\n\tvar/const/answer = 42\n\t++answer\n",
                "answer",
            ),
            (
                "typed local receiver",
                "/obj\n\tvar/const/answer = 42\n/proc/RunTest()\n\tvar/obj/o = new\n\to.answer = 7\n",
                "answer",
            ),
            (
                "typed field receiver",
                "/obj\n\tvar/const/answer = 42\n/datum/holder\n\tvar/obj/item\n\tproc/change()\n\t\titem.answer = 7\n",
                "answer",
            ),
        ];

        for (label, source, name) in cases {
            let compilation = TestProject::compile(source);
            let error = ProcedureRegistry::build(&compilation)
                .compile_vm(&compilation)
                .expect_err("const assignment should fail compilation");
            assert!(
                error.message.contains(&format!("const variable `{name}`")),
                "{label}: {}",
                error.message
            );
        }
    }

    #[test]
    fn mutable_local_shadowing_a_const_field_remains_assignable() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/const/answer = 42\n\tproc/read()\n\t\tvar/answer = 1\n\t\tanswer = 2\n\t\treturn answer\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("mutable local should shadow the const field");
    }

    #[test]
    fn typed_receiver_const_check_does_not_guess_from_the_field_name() {
        let compilation = TestProject::compile(
            "/obj\n\tvar/const/answer = 42\n/datum/other\n\tvar/answer = 1\n/proc/RunTest()\n\tvar/datum/other/o = new\n\to.answer = 7\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("the proven receiver field is mutable despite a same-name const elsewhere");
    }

    #[test]
    fn typed_parameters_and_locals_allow_runtime_cross_branch_values() {
        let incompatible = TestProject::compile(
            "/proc/replace(turf/bar as turf)\n\tbar = new /obj(null)\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&incompatible)
            .compile_vm(&incompatible)
            .expect("BYOND path annotations do not reject unrelated runtime assignments");

        let compatible = TestProject::compile(
            "/obj/item\n/proc/replace(obj/bar as obj)\n\tbar = new /obj/item(null)\n/proc/local()\n\tvar/obj/bar = new /obj/item\n\treturn bar\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("subtype construction should satisfy the declared type");
    }

    #[test]
    fn validates_typed_sources_and_proven_datum_return_paths() {
        let dynamic_assignment = TestProject::compile(
            "/datum/base\n/obj/item\n/proc/copy()\n\tvar/datum/base/target\n\tvar/obj/item/source\n\ttarget = source\n",
        );
        ProcedureRegistry::build(&dynamic_assignment)
            .compile_vm(&dynamic_assignment)
            .expect("DreamMaker permits runtime values to flow through path annotations");

        let incompatible_return = TestProject::compile(
            "/datum/base\n/obj/item\n/proc/build() as /datum/base\n\treturn /obj/item\n",
        );
        ProcedureRegistry::build(&incompatible_return)
            .compile_vm(&incompatible_return)
            .expect("BYOND return annotations do not reject body values");

        let compatible = TestProject::compile(
            "/datum/base\n/datum/base/child\n/proc/copy() as /datum/base\n\tvar/datum/base/target\n\tvar/datum/base/child/source\n\ttarget = source\n\treturn /datum/base/child\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("subtype sources and return paths should satisfy base constraints");
    }

    #[test]
    fn validates_proven_scalar_annotations_without_rejecting_null_unions() {
        let bad_local = TestProject::compile(
            "/proc/value() as num\n\tvar/const/result = \"wrong\"\n\treturn result\n",
        );
        ProcedureRegistry::build(&bad_local)
            .compile_vm(&bad_local)
            .expect("BYOND scalar return annotations do not reject body values");

        let bad_parameter = TestProject::compile(
            "/proc/value(var/input = \"text\" as text) as num\n\treturn input\n",
        );
        ProcedureRegistry::build(&bad_parameter)
            .compile_vm(&bad_parameter)
            .expect("BYOND permits dynamic scalar return values");

        let compatible = TestProject::compile(
            "/proc/number() as num\n\tvar/value = 5 as num|null\n\tvalue = null\n\tvalue = 7\n\treturn value\n/proc/text_value(var/input = \"ok\" as text) as text\n\treturn input\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("matching scalar annotations and nullable assignments should compile");
    }

    #[test]
    fn inherits_override_return_constraints_and_propagates_static_call_types() {
        let inherited_mismatch = TestProject::compile(
            "/datum/proc/value() as num\n\treturn 5\n/datum/child/value()\n\treturn \"wrong\"\n",
        );
        ProcedureRegistry::build(&inherited_mismatch)
            .compile_vm(&inherited_mismatch)
            .expect("an inherited annotation constrains signatures, not body values");

        let changed_signature = TestProject::compile(
            "/datum/proc/value() as num\n\treturn 5\n/datum/child/value() as text\n\treturn \"wrong\"\n",
        );
        assert!(
            ProcedureRegistry::build(&changed_signature)
                .compile_vm(&changed_signature)
                .expect_err("override cannot replace numeric return with text")
                .message
                .contains("changes its inherited scalar return type")
        );

        let call_mismatch = TestProject::compile(
            "/proc/text_value() as text\n\treturn \"text\"\n/proc/number_value() as num\n\treturn text_value()\n",
        );
        ProcedureRegistry::build(&call_mismatch)
            .compile_vm(&call_mismatch)
            .expect("BYOND permits a differently annotated call result to be returned");

        let compatible = TestProject::compile(
            "/datum/base\n/datum/base/child\n/proc/build_child() as /datum/base/child\n\treturn /datum/base/child\n/proc/build() as /datum/base\n\treturn build_child()\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("statically called subtype return should satisfy base return");
    }

    #[test]
    fn validates_scalar_field_overrides_against_late_inherited_declarations() {
        let incompatible = TestProject::compile(
            "/datum/base/child\n\tvalue = \"wrong\"\n/datum/base\n\tvar/value = 5 as num\n/proc/RunTest()\n\treturn\n",
        );
        assert!(
            ProcedureRegistry::build(&incompatible)
                .compile_vm(&incompatible)
                .expect_err("text override cannot satisfy inherited numeric field")
                .message
                .contains("field override /datum/base/child/var/value")
        );

        let compatible = TestProject::compile(
            "/datum/base/child\n\tvalue = null\n/datum/base\n\tvar/value = 5 as num|null\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("nullable inherited field should accept null subtype default");
    }

    #[test]
    fn infers_returns_from_proven_receiver_fields_and_methods() {
        let field_mismatch = TestProject::compile(
            "/datum/value_holder\n\tvar/bar = 5 as num\n\tproc/read() as text\n\t\tvar/datum/value_holder/D = new\n\t\treturn D.bar\n",
        );
        ProcedureRegistry::build(&field_mismatch)
            .compile_vm(&field_mismatch)
            .expect("typed fields remain runtime values at a return site");

        let method_mismatch = TestProject::compile(
            "/datum/producer/proc/value() as text\n\treturn \"text\"\n/proc/read() as num\n\tvar/datum/producer/P = new\n\treturn P.value()\n",
        );
        ProcedureRegistry::build(&method_mismatch)
            .compile_vm(&method_mismatch)
            .expect("typed method results remain runtime values at a return site");

        let compatible = TestProject::compile(
            "/datum/base\n/datum/base/child\n/datum/producer/proc/value() as /datum/base/child\n\treturn /datum/base/child\n/proc/read() as /datum/base\n\tvar/datum/producer/P = new\n\treturn P.value()\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("typed receiver method subtype should satisfy base return");
    }

    #[test]
    fn late_base_signature_constrains_early_override_chain() {
        let compilation = TestProject::compile(
            "/datum/do/re/mi/fa/so/f()\n\treturn 5\n/datum/do/re/f()\n\treturn ..() + \" re\"\n/datum/do/re/mi/fa/f()\n\treturn ..() + \" fa\"\n/datum/do/re/mi/f()\n\treturn ..() + \" mi\"\n/datum/do/proc/f() as text\n\treturn \"do\"\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("late inherited signatures do not constrain override body values");
    }

    #[test]
    fn infers_only_proven_scalar_composite_results() {
        let incompatible =
            TestProject::compile("/proc/ternary_value() as text\n\treturn 1 ? 2 : 3\n");
        ProcedureRegistry::build(&incompatible)
            .compile_vm(&incompatible)
            .expect("return annotations do not reject numeric ternaries");

        let list_mismatch =
            TestProject::compile("/proc/list_value() as text\n\treturn list(1, 2, 3)[1]\n");
        ProcedureRegistry::build(&list_mismatch)
            .compile_vm(&list_mismatch)
            .expect("return annotations do not reject list-index results");

        let compatible = TestProject::compile(
            "/datum/proc/value() as text\n\treturn \"base\"\n/datum/child/value()\n\treturn ..() + \" child\"\n/proc/number() as num\n\treturn (1 ? 2 : 3) + list(4, 5)[1]\n",
        );
        ProcedureRegistry::build(&compatible)
            .compile_vm(&compatible)
            .expect("matching proven composites should compile");
    }

    #[test]
    fn mutable_unannotated_locals_do_not_acquire_static_scalar_types() {
        let compilation =
            TestProject::compile("/datum/proc/foo() as num\n\tvar/meep = 5\n\treturn meep\n");
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("dynamic locals may be returned from annotated procedures");
    }

    #[test]
    fn comma_grouped_bare_locals_do_not_form_a_fake_type_path() {
        let compilation = TestProject::compile(
            "/proc/is_guest_key(key)\n\tvar/i, ch, len = 3\n\ti = 1\n\tch = 2\n\treturn i + ch + len\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("grouped bare locals must remain independent untyped declarations");
    }

    #[test]
    fn narrows_truthy_ternaries_and_invalidates_facts_on_dynamic_writes() {
        let narrowed = TestProject::compile(
            "/datum/test1\n/datum/test2/proc/meep() as num\n\treturn 5\n/datum/test3/proc/meep() as text\n\treturn \"bad\"\n/proc/read() as num\n\tvar/datum/test1/T1 = new\n\tvar/datum/test2/T2 = new\n\tvar/datum/test3/T3 = new\n\treturn (T1 ? T2 : T3).meep()\n",
        );
        ProcedureRegistry::build(&narrowed)
            .compile_vm(&narrowed)
            .expect("a local initialized with new is proven truthy");

        let invalidated = TestProject::compile(
            "/datum/test1\n/datum/test2/proc/meep() as num\n\treturn 5\n/datum/test3/proc/meep() as text\n\treturn \"bad\"\n/proc/read(value) as num\n\tvar/datum/test1/T1 = new\n\tvar/datum/test2/T2 = new\n\tvar/datum/test3/T3 = new\n\tT1 = value\n\treturn (T1 ? T2 : T3).meep()\n",
        );
        ProcedureRegistry::build(&invalidated)
            .compile_vm(&invalidated)
            .expect("an unknown write invalidates the truth fact and stays unchecked");
    }

    #[test]
    fn rejects_unknown_declared_types_without_confusing_type_named_fields() {
        let unknown = TestProject::compile(
            "/datum/later\n\tvar/datum/laterrr/aa = new(0)\n/proc/RunTest()\n\treturn\n",
        );
        assert!(
            ProcedureRegistry::build(&unknown)
                .compile_vm(&unknown)
                .expect_err("unknown field declaration type must be rejected")
                .message
                .contains("unknown declared type `/datum/laterrr`")
        );

        let name_clash = TestProject::compile(
            "var/datum/later/later\n/datum/later\n\tvar/datum/later/later = new(0)\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&name_clash)
            .compile_vm(&name_clash)
            .expect("a field may have the same name as its declared type");

        let typed_list = TestProject::compile(
            "/datum/item\n/datum/holder\n\tvar/final/list/datum/item/items = list()\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&typed_list)
            .compile_vm(&typed_list)
            .expect("a typed-list element path must not be treated as a /list subtype");

        let project_descendant = TestProject::compile(
            "/obj/item/weapon\n/datum/holder\n\tvar/obj/item/weapon/gun/ballistic/owner_gun\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&project_descendant)
            .compile_vm(&project_descendant)
            .expect("BYOND accepts an annotated descendant beneath a project-defined type");

        let unresolved_annotation = TestProject::compile(
            "/datum/holder\n\tvar/datum/forward_declared_later/value\n/proc/RunTest()\n\treturn\n",
        );
        ProcedureRegistry::build(&unresolved_annotation)
            .compile_vm(&unresolved_annotation)
            .expect("BYOND accepts unresolved field annotations without an initializer");

        let unknown_local =
            TestProject::compile("/proc/read()\n\tvar/datum/missing/value\n\treturn\n");
        assert!(
            ProcedureRegistry::build(&unknown_local)
                .compile_vm(&unknown_local)
                .expect_err("unknown local declaration type must be rejected")
                .message
                .contains("unknown declared type `/datum/missing`")
        );

        let unknown_parameter =
            TestProject::compile("/proc/read(var/datum/missing/value)\n\treturn\n");
        assert!(
            ProcedureRegistry::build(&unknown_parameter)
                .compile_vm(&unknown_parameter)
                .expect_err("unknown parameter declaration type must be rejected")
                .message
                .contains("unknown declared type `/datum/missing`")
        );

        let unknown_return =
            TestProject::compile("/proc/read() as /datum/missing\n\treturn null\n");
        assert!(
            ProcedureRegistry::build(&unknown_return)
                .compile_vm(&unknown_return)
                .expect_err("unknown procedure return type must be rejected")
                .message
                .contains("unknown declared procedure return type `/datum/missing`")
        );
    }

    #[test]
    fn accepts_remaining_declared_type_inference_shapes() {
        let cases = [
            (
                "inherited typed field override",
                "/datum/test/thing\n\tvar/list/foo = list()\n/datum/test/thing/stuff\n\tfoo = new()\n/proc/RunTest()\n\treturn\n",
            ),
            (
                "nested list assignment",
                "/proc/RunTest()\n\tvar/list/L = list()\n\tL[new()] = new()\n\treturn\n",
            ),
            (
                "late derived field",
                "/datum/later\n\tvar/datum/pointless_base/a\n/datum/pointless_base/derived/var/x = 7\n/proc/RunTest()\n\tvar/datum/later/L = new\n\tL.a = new /datum/pointless_base/derived()\n\treturn\n",
            ),
            (
                "BYOND input-qualified parameters",
                "/area/target\n/datum/thing\n/mob/player\n/client/proc/jump_to_area(area/target in world)\n\treturn target\n/client/proc/debug_variables(datum/thing in world)\n\treturn thing\n/proc/togglebuildmode(mob/M in global.player_list)\n\treturn M\n",
            ),
        ];
        for (label, source) in cases {
            let compilation = TestProject::compile(source);
            ProcedureRegistry::build(&compilation)
                .compile_vm(&compilation)
                .unwrap_or_else(|error| panic!("{label} should compile: {error:?}"));
        }
    }

    #[test]
    fn contextual_new_in_typed_list_assignment_allocates_the_list_type() {
        let compilation = TestProject::compile(
            "/proc/build()\n\tvar/list/L = list()\n\tL[new()] = new()\n\treturn L[1]\n",
        );
        let result = execute_effective(&compilation, "/proc/build", &[]);
        assert!(matches!(result, Ok(Value::List(_))), "result: {result:?}");
    }

    #[test]
    fn inferred_new_uses_every_statically_proven_destination_family() {
        let cases = [
            (
                "typed local wrapper and ternary",
                "/datum/item\n/proc/build(var/flag)\n\tvar/datum/item/value = (flag ? new() : new())\n\treturn value.type\n",
            ),
            (
                "implicit src field",
                "/datum/holder\n\tvar/datum/item/value\n\tproc/build()\n\t\tvalue = new()\n\t\treturn value.type\n/datum/item\n",
            ),
            (
                "explicit and chained member fields",
                "/datum/outer\n\tvar/datum/inner/child\n\tproc/build()\n\t\tchild.value = new()\n\t\treturn child.value.type\n/datum/inner\n\tvar/datum/item/value\n/datum/item\n",
            ),
            (
                "typed global",
                "/var/datum/item/shared\n/proc/build()\n\tshared = new()\n\treturn shared.type\n/datum/item\n",
            ),
        ];
        for (label, source) in cases {
            let compilation = TestProject::compile(source);
            ProcedureRegistry::build(&compilation)
                .compile_vm(&compilation)
                .unwrap_or_else(|error| panic!("{label} should compile: {error:?}"));
        }
    }

    #[test]
    fn title_icon_field_assignment_infers_icon_and_preserves_the_resource() {
        let compilation = TestProject::compile(concat!(
            "/datum/controller/subsystem/title\n",
            "\tvar/icon/icon\n",
            "\tproc/Initialize()\n",
            "\t\ticon = new(fcopy_rsc(\"icons/runtime/default_title.dmi\"))\n",
            "\t\treturn icon.icon\n",
            "/proc/run_title_initialize()\n",
            "\tvar/datum/controller/subsystem/title/title = new\n",
            "\treturn title.Initialize()\n",
        ));

        assert_eq!(
            execute_effective(&compilation, "/proc/run_title_initialize", &[]),
            Ok(Value::file("icons/runtime/default_title.dmi")),
        );
    }

    #[test]
    fn inferred_new_follows_typed_global_controller_member_destination() {
        let compilation = TestProject::compile(
            "/datum/ghost_arena\n\tNew(var/source, var/marker)\n/datum/controller/global_vars\n\tvar/datum/ghost_arena/ghost_arena\n\tvar/first_arena_marker\n/var/global/datum/controller/global_vars/GLOB = new /datum/controller/global_vars\n/obj/effect/ghost_arena_corner/Initialize()\n\tGLOB.ghost_arena = new(src, GLOB.first_arena_marker)\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("typed GLOB member must qualify inferred new");
    }

    #[test]
    fn inferred_new_uses_logical_assignment_and_compact_macro_local_context() {
        for source in [
            "/datum/cassette\n\tvar/datum/cassette_data/cassette_data\n\tproc/LateInitialize()\n\t\tcassette_data ||= new\n/datum/cassette_data\n",
            "/datum/sort_instance\n/var/global/datum/sort_instance/shared_sorter = new /datum/sort_instance\n/proc/sortTim()\n\tvar/datum/sort_instance/sorter = shared_sorter; if(isnull(sorter)){ sorter = new; }\n",
        ] {
            let compilation = TestProject::compile(source);
            ProcedureRegistry::build(&compilation)
                .compile_vm(&compilation)
                .expect("destination context must survive logical/compact assignment");
        }
    }

    #[test]
    fn inferred_new_uses_slash_typed_parameter_without_var_keyword() {
        let compilation = TestProject::compile(
            "/datum/tgui\n/proc/ui_interact(mob/user, datum/tgui/ui)\n\tif(!ui)\n\t\tui = new(user)\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("typed parameter destination must qualify new");
    }

    #[test]
    fn inferred_new_follows_builtin_mob_client_into_a_typed_safe_field() {
        let compilation = TestProject::compile(concat!(
            "/datum/meta_token_holder\n",
            "\tvar/client/owner\n",
            "/client\n",
            "\tvar/datum/meta_token_holder/client_token_holder\n",
            "/mob/Login()\n",
            "\tclient?.client_token_holder = new(client)\n",
        ));
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("the built-in typed mob.client edge must qualify safe-field new");
    }

    #[test]
    fn inferred_new_uses_typed_parameter_default_and_for_receiver_field() {
        let compilation = TestProject::compile(
            "/datum/point\n/proc/copy_to(datum/point/p = new)\n\treturn p\n/datum/gas\n/obj/pipe\n\tvar/datum/gas/air_temporary\n/proc/store(var/list/members)\n\tfor(var/obj/pipe/member in members)\n\t\tmember.air_temporary = new\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("parameter defaults and typed loop receivers qualify new");
    }

    #[test]
    fn inferred_new_rejects_contextless_and_unresolved_destinations() {
        for source in [
            "/proc/build()\n\treturn new()\n",
            "/proc/build()\n\tvar/value\n\tvalue = new()\n",
        ] {
            let compilation = TestProject::compile(source);
            let error = ProcedureRegistry::build(&compilation)
                .compile_vm(&compilation)
                .expect_err("unproven inferred new must be rejected");
            assert!(
                error
                    .message
                    .contains("no statically resolved destination type")
            );
        }
    }

    #[test]
    fn typed_receiver_static_access_uses_one_inherited_qualified_slot() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/static/shared = 3\n/datum/child\n\tparent_type = /datum/base\n/proc/read(var/datum/child/other)\n\tother.shared = 9\n\treturn initial(other.shared)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry
            .compile_vm(&compilation)
            .expect("typed static member access");
        let procedure = procedure_by_path(&registry, "/proc/read")
            .effective_target
            .and_then(|id| executable.implementation(id))
            .expect("read implementation");
        let instructions = &executable
            .module()
            .procedure(procedure)
            .expect("program")
            .instructions;
        let stores = instructions
            .iter()
            .filter_map(|instruction| match instruction {
                dm_vm::Instruction::StoreGlobal(field) => Some(field.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let initials = instructions
            .iter()
            .filter_map(|instruction| match instruction {
                dm_vm::Instruction::LoadInitialGlobal(field) => Some(field.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(stores.len(), 1);
        assert_eq!(initials, stores);
    }

    #[test]
    fn typed_for_in_receiver_reads_inherited_static_list_by_shared_identity() {
        let compilation = TestProject::compile(
            "/datum/bodypart_overlay\n\tvar/static/list/all_layers = list(1, 2, 4)\n/datum/bodypart_overlay/mutant\n/proc/read_layers(list/bodypart_overlays)\n\tvar/list/first\n\tfor(var/datum/bodypart_overlay/overlay as anything in bodypart_overlays)\n\t\tfirst = overlay.all_layers\n\treturn first\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/read_layers")
            .effective_target
            .expect("read_layers implementation");
        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("typed loop receiver static access should lower");
        let program = executable
            .module()
            .procedure(executable.implementation(entry).unwrap())
            .expect("read_layers program");
        let storage = FieldName::static_storage("/datum/bodypart_overlay/var/all_layers");
        assert!(program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::LoadGlobal(field) if field == &storage)
        ));
        assert!(!program.instructions.iter().any(
            |instruction| matches!(instruction, Instruction::LoadField(field) if field.as_str() == "all_layers")
        ));

        let mut state = ExecutionState::new();
        let shared = state.heap_mut().allocate_list();
        for layer in [1.0, 2.0, 4.0] {
            state
                .heap_mut()
                .list_mut(shared)
                .unwrap()
                .add(Value::number(layer));
        }
        let overlay = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/bodypart_overlay/mutant").unwrap());
        let overlays = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .list_mut(overlays)
            .unwrap()
            .add(Value::Datum(overlay));
        state.set_global(storage, Value::List(shared));

        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[Value::List(overlays)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::List(shared)),
            "every instance receiver must observe the one inherited static list",
        );
    }

    #[test]
    fn typed_global_receiver_static_access_does_not_dereference_null() {
        let compilation = TestProject::compile(
            "var/global/datum/globals/GLOB\n/datum/globals\n\tvar/global/config_error_log\n/world/Genesis()\n\tGLOB.config_error_log = \"early.log\"\n\treturn GLOB.config_error_log\n",
        );
        assert_eq!(
            super::declared_global_types(&compilation)
                .get("GLOB")
                .map(dm_value::TypePath::as_str),
            Some("/datum/globals")
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry
            .compile_vm(&compilation)
            .expect("typed global static");
        let procedure = procedure_by_path(&registry, "/world/proc/Genesis")
            .effective_target
            .and_then(|id| executable.implementation(id))
            .expect("Genesis implementation");
        let instructions = &executable
            .module()
            .procedure(procedure)
            .unwrap()
            .instructions;
        let qualified = dm_value::FieldName::static_storage("/datum/globals/var/config_error_log");
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::StoreGlobal(field) if field == &qualified)
        ), "{instructions:?}");
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::LoadGlobal(field) if field == &qualified)
        ));
        assert!(!instructions.iter().any(|instruction| matches!(
            instruction,
            dm_vm::Instruction::LoadField(_) | dm_vm::Instruction::StoreField(_)
        )));
    }

    #[test]
    fn typed_global_receiver_static_increment_uses_qualified_shared_slot() {
        let compilation = TestProject::compile(
            "var/global/datum/controller/master/Master\n/datum/controller/master\n\tvar/static/restart_count = 0\n/proc/Recreate_MC()\n\treturn ++Master.restart_count\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry
            .compile_vm(&compilation)
            .expect("master static mutation");
        let procedure = procedure_by_path(&registry, "/proc/Recreate_MC")
            .effective_target
            .and_then(|id| executable.implementation(id))
            .expect("Recreate_MC implementation");
        let instructions = &executable
            .module()
            .procedure(procedure)
            .unwrap()
            .instructions;
        let qualified =
            dm_value::FieldName::static_storage("/datum/controller/master/var/restart_count");
        assert!(instructions.iter().any(|instruction| matches!(
            instruction,
            dm_vm::Instruction::MutateGlobal { name, delta: 1, prefix: true } if name == &qualified
        )), "{instructions:?}");
        assert!(!instructions.iter().any(|instruction| matches!(
            instruction,
            dm_vm::Instruction::MutateField { name, .. } if name.as_str() == "restart_count"
        )));
    }

    #[test]
    fn bare_type_static_in_owner_method_uses_qualified_shared_slot() {
        let compilation = TestProject::compile(
            "/datum/controller/master\n\tvar/static/random_seed\n/datum/controller/master/New()\n\tif(!random_seed)\n\t\trandom_seed = 7\n\treturn random_seed\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry.compile_vm(&compilation).expect("master static");
        let procedure = procedure_by_path(&registry, "/datum/controller/master/proc/New")
            .effective_target
            .and_then(|id| executable.implementation(id))
            .expect("New implementation");
        let instructions = &executable
            .module()
            .procedure(procedure)
            .unwrap()
            .instructions;
        let qualified =
            dm_value::FieldName::static_storage("/datum/controller/master/var/random_seed");
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::LoadGlobal(field) if field == &qualified)
        ), "{instructions:?}");
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::StoreGlobal(field) if field == &qualified)
        ), "{instructions:?}");
    }

    #[test]
    fn true_instance_field_wins_over_inherited_same_name_static() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/static/value\n/datum/base/child\n\tvar/value\n/datum/base/child/proc/Run()\n\tvalue = 4\n\treturn value\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let executable = registry.compile_vm(&compilation).expect("field collision");
        let procedure = procedure_by_path(&registry, "/datum/base/child/proc/Run")
            .effective_target
            .and_then(|id| executable.implementation(id))
            .expect("Run implementation");
        let instructions = &executable
            .module()
            .procedure(procedure)
            .unwrap()
            .instructions;
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::StoreField(field) if field.as_str() == "value")
        ));
        assert!(instructions.iter().any(
            |instruction| matches!(instruction, dm_vm::Instruction::LoadField(field) if field.as_str() == "value")
        ));
    }

    #[test]
    fn resolves_upward_search_path_expressions_before_vm_lowering() {
        let cases = [
            (
                "deep search",
                "/datum/a/b/c\n/datum/d\n/proc/RunTest()\n\treturn /datum/a/b/c.d\n",
                "/proc/RunTest",
                "/datum/d",
            ),
            (
                "procedure namespace",
                "/atom/proc/fn()\n\treturn\n/proc/RunTest()\n\treturn /atom./proc/fn\n",
                "/proc/RunTest",
                "/atom/proc/fn",
            ),
            (
                "contextual search",
                "/datum/foo\n/datum/bar/proc/find()\n\treturn .foo\n",
                "/datum/bar/proc/find",
                "/datum/foo",
            ),
            (
                "empty suffix",
                "/datum/foo\n/proc/RunTest()\n\treturn /datum/foo.\n",
                "/proc/RunTest",
                "/datum/foo",
            ),
        ];

        for (label, source, procedure, expected) in cases {
            let compilation = TestProject::compile(source);
            assert_eq!(
                execute_effective(&compilation, procedure, &[]),
                Ok(Value::TypePath(TypePath::parse(expected).unwrap())),
                "{label}"
            );
        }
    }

    #[test]
    fn function_macro_brace_blocks_keep_locals_visible_to_nested_children() {
        let compilation = TestProject::compile(
            "#define WRAP(value) \\\n\tdo {\\
\t\tif(value) {\\
\t\t\tvar/_cached_plane = value;\\
\t\t\tif(_cached_plane) {\\
\t\t\t\tvalue = _cached_plane;\\
\t\t\t}\\
\t\t}\\
\t} while(FALSE)\n\n/proc/run(value)\n\tWRAP(value)\n\treturn value\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        registry
            .compile_vm(&compilation)
            .expect("macro-expanded brace locals should compile");
    }

    #[test]
    fn multiline_macro_brace_blocks_keep_locals_visible_to_nested_children() {
        let compilation = TestProject::compile(
            r#"#define GET_NEW_PLANE(new_value, multiplier) (blacklist?["[new_value]"] ? new_value : (new_value) - multiplier)
#define WRAP(value) \
	do {\
		if(value) {\
			var/_cached_plane = value;\
			var/turf/_our_turf = value;\
			if(_our_turf) {\
				value = GET_NEW_PLANE(_cached_plane, 1);\
			} else if(value) {\
				value = _cached_plane;\
			}\
		}\
	} while(FALSE)

/proc/run(value, blacklist)
	WRAP(value)
	return value
"#,
        );
        let registry = ProcedureRegistry::build(&compilation);
        registry
            .compile_vm(&compilation)
            .expect("macro-expanded brace locals should compile");
    }

    #[test]
    fn typed_global_macro_declaration_is_visible_as_a_bare_global() {
        let compilation = TestProject::compile(
            r#"#define GLOBAL_REAL(X, Typepath) var/global##Typepath/##X;

GLOBAL_REAL(Master, /datum/controller/master)

/proc/run()
	return Master
"#,
        );
        let registry = ProcedureRegistry::build(&compilation);
        registry
            .compile_vm(&compilation)
            .expect("typed global declarations should resolve by bare name");
    }

    #[test]
    fn typed_global_proc_parameters_keep_if_lines_in_the_procedure_body() {
        let compilation = TestProject::compile(
            "/proc/overwrite_field_if_available(datum/record/base, datum/record/other, field_name)\n\tif(!istype(base) || !istype(other))\n\t\treturn\n\tif(other.vars[field_name])\n\t\tbase.vars[field_name] = other.vars[field_name]\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        assert!(
            registry
                .procedures()
                .iter()
                .all(|procedure| procedure.path.to_string() != "/proc/if"),
            "if statements must not become phantom global procedures"
        );
        assert!(
            registry.procedures().iter().any(|procedure| {
                procedure.path.to_string() == "/proc/overwrite_field_if_available"
            }),
            "the typed global procedure should be indexed"
        );
    }

    #[test]
    fn links_a_child_override_to_the_inherited_effective_body() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/run");
        let child = procedure_by_path(&registry, "/datum/base/child/proc/run");

        assert_eq!(child.inherited_procedure, Some(base.id));
        assert_eq!(child.implementations.len(), 1);
        assert_eq!(
            child.implementations[0].parent_target,
            base.effective_target
        );
    }

    #[test]
    fn chains_multiple_reopenings_in_expanded_source_order() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn 2\n/datum/base\n\trun()\n\t\treturn 3\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert_eq!(procedure.implementations.len(), 3);
        assert!(
            procedure
                .implementations
                .windows(2)
                .all(|pair| pair[0].ordinal < pair[1].ordinal)
        );
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.implementations[1].parent_target,
            Some(procedure.implementations[0].id)
        );
        assert_eq!(
            procedure.implementations[2].parent_target,
            Some(procedure.implementations[1].id)
        );
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[2].id)
        );
    }

    #[test]
    fn indexes_a_global_procedure_without_an_owner_or_parent() {
        let compilation = TestProject::compile("/proc/global_run()\n\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/proc/global_run");

        assert_eq!(procedure.owner_type, None);
        assert_eq!(procedure.inherited_procedure, None);
        assert_eq!(procedure.implementations[0].parent_target, None);
    }

    #[test]
    fn follows_a_resolved_explicit_parent_type() {
        let compilation = TestProject::compile(
            "/datum/original\n\tproc/run()\n\t\treturn 1\n/datum/alternate\n\tproc/run()\n\t\treturn 2\n/custom\n\tparent_type = /datum/alternate\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let original = procedure_by_path(&registry, "/datum/original/proc/run");
        let alternate = procedure_by_path(&registry, "/datum/alternate/proc/run");
        let custom = procedure_by_path(&registry, "/custom/proc/run");

        assert_ne!(custom.inherited_procedure, Some(original.id));
        assert_eq!(custom.inherited_procedure, Some(alternate.id));
        assert_eq!(
            custom.implementations[0].parent_target,
            alternate.effective_target
        );
    }

    #[test]
    fn lowers_bare_inherited_fields_as_src_fields_after_local_resolution() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/loading_id = 1\n/datum/base/child\n\tproc/run(loading_id)\n\t\tvar/local = loading_id\n\t\tloading_id = local\n\t\treturn src.loading_id\n/datum/base/child\n\tproc/use_field()\n\t\tloading_id += 1\n\t\treturn loading_id\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/child/proc/use_field");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                procedure.implementations.iter().map(|body| body.id),
            )
            .expect("bare inherited field should compile");
        let entry = executable
            .implementation(procedure.effective_target.expect("procedure has a body"))
            .expect("implementation should be present");
        let program = executable
            .module()
            .procedure(entry)
            .expect("program should exist");

        assert!(program.instructions.windows(3).any(|instructions| matches!(
            instructions,
            [Instruction::LoadSrc, Instruction::Duplicate, Instruction::LoadField(field)]
                if field.as_str() == "loading_id"
        )));
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::StoreField(field) if field.as_str() == "loading_id"
        )));
    }

    #[test]
    fn lowers_existing_instance_field_as_undeclared_for_loop_target() {
        let compilation = TestProject::compile(
            "/obj/machine\n\tvar/cointype = /obj/coin\n\tInitialize()\n\t\tfor(cointype in typesof(/obj/coin))\n\t\t\tvar/obj/coin/value = new cointype\n/obj/coin\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/obj/machine/proc/Initialize");
        registry
            .compile_vm_implementations(
                &compilation,
                procedure.implementations.iter().map(|body| body.id),
            )
            .expect("an undeclared for target should bind an existing src field");
    }

    #[test]
    fn lowers_standard_atom_fields_only_for_their_builtin_hierarchy() {
        let compilation = TestProject::compile(
            "/atom/proc/offsets()\n\tif(pixel_x == 0 && pixel_y == 0)\n\t\treturn list(pixel_w, pixel_z)\n/obj/example\n\tproc/read()\n\t\tloc = src\n\t\tpixel_x += 1\n\t\talpha -= 1\n\t\treturn list(dir, color, desc, blend_mode, alpha, appearance_flags, layer, plane, transform, overlays, underlays, vis_contents, vis_locs, x, y, z)\n\tDestroy()\n\t\tvis_locs = null\n\t\tif(length(vis_contents))\n\t\t\tvis_contents.Cut()\n/datum/example\n\tproc/read()\n\t\treturn alpha\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let object = procedure_by_path(&registry, "/obj/example/proc/read");
        registry
            .compile_vm_implementations(
                &compilation,
                object.implementations.iter().map(|body| body.id),
            )
            .expect("standard atom fields should compile as src fields");
        let destroy = procedure_by_path(&registry, "/obj/example/proc/Destroy");
        registry
            .compile_vm_implementations(
                &compilation,
                destroy.implementations.iter().map(|body| body.id),
            )
            .expect("vis_locs and vis_contents should bind in Destroy");
        let atom = procedure_by_path(&registry, "/atom/proc/offsets");
        registry
            .compile_vm_implementations(
                &compilation,
                atom.implementations.iter().map(|body| body.id),
            )
            .expect("pixel offsets are engine fields on /atom itself");

        let datum = procedure_by_path(&registry, "/datum/example/proc/read");
        let error = registry
            .compile_vm_implementations(
                &compilation,
                datum.implementations.iter().map(|body| body.id),
            )
            .expect_err("atom fields must not become datum locals");
        assert!(
            error.message.contains("unknown local \"alpha\""),
            "unexpected diagnostic: {}",
            error.message
        );
    }

    #[test]
    fn switch_arm_local_does_not_hide_documented_atom_and_particle_fields() {
        let compilation = TestProject::compile(concat!(
            "/obj/machinery/chem_recipe_debug/proc/ui_act(action)\n",
            "\tswitch(action)\n",
            "\t\tif(\"setTargetList\")\n",
            "\t\t\tvar/text = \"local\"\n",
            "\t\t\tif(!text)\n",
            "\t\t\t\treturn 1\n",
            "\t\tif(\"setEdit\")\n",
            "\t\t\tif(!text)\n",
            "\t\t\t\treturn 2\n",
            "/particles/proc/return_ui_representation()\n",
            "\treturn color_change\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let ui_act = procedure_by_path(&registry, "/obj/machinery/chem_recipe_debug/proc/ui_act");
        let particles = procedure_by_path(&registry, "/particles/proc/return_ui_representation");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                [
                    ui_act.effective_target.expect("ui_act body"),
                    particles
                        .effective_target
                        .expect("particle representation body"),
                ],
            )
            .expect("BYOND engine fields should bind outside the local's lexical switch arm");

        for (target, expected_field) in [
            (ui_act.effective_target.unwrap(), "text"),
            (particles.effective_target.unwrap(), "color_change"),
        ] {
            let entry = executable
                .implementation(target)
                .expect("selected implementation should be linked");
            let program = executable
                .module()
                .procedure(entry)
                .expect("selected implementation should have bytecode");
            assert!(program.instructions.iter().any(|instruction| matches!(
                instruction,
                Instruction::LoadField(field) if field.as_str() == expected_field
            )));
        }
    }

    #[test]
    fn mulebot_initialize_binds_deprecated_atom_suffix_field() {
        let compilation = TestProject::compile(
            "/atom\n/mob\n/mob/living\n/mob/living/simple_animal\n/mob/living/simple_animal/bot\n/mob/living/simple_animal/bot/mulebot\n\tvar/id\n\tproc/set_id(value)\n\t\tid = value\n\tInitialize(mapload)\n\t\tset_id(suffix || id || \"fallback\")\n\t\tsuffix = null\n\t\treturn suffix\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let initialize = procedure_by_path(
            &registry,
            "/mob/living/simple_animal/bot/mulebot/proc/Initialize",
        );
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                initialize.implementations.iter().map(|body| body.id),
            )
            .expect("BYOND's deprecated /atom/suffix field must bind through mob inheritance");
        let entry = executable
            .implementation(initialize.effective_target.expect("Initialize has a body"))
            .expect("Initialize implementation should be compiled");
        let program = executable
            .module()
            .procedure(entry)
            .expect("Initialize program should exist");

        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::LoadField(field) if field.as_str() == "suffix"
        )));
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            Instruction::StoreField(field) if field.as_str() == "suffix"
        )));
    }

    #[test]
    fn lowers_documented_image_and_mob_engine_fields() {
        let compilation = TestProject::compile(
            "/image/proc/update()\n\toverlays += src\n\tappearance_flags |= 1\n\tdir = 4\n\treturn overlays\n/mob/proc/update_vision()\n\tsight |= 1\n\tsee_invisible = 2\n\treturn sight + see_invisible + initial(sight) + initial(see_invisible)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        for path in ["/image/proc/update", "/mob/proc/update_vision"] {
            let procedure = procedure_by_path(&registry, path);
            registry
                .compile_vm_implementations(
                    &compilation,
                    [procedure.effective_target.expect("procedure body")],
                )
                .unwrap_or_else(|error| panic!("{path} should bind engine fields: {error:?}"));
        }
    }

    #[test]
    fn lowers_client_matrix_and_atom_appearance_engine_fields() {
        let compilation = TestProject::compile(
            "/client/proc/read_engine_state()\n\treturn list(connection, address, computer_id, view, screen, verbs)\n/matrix/proc/read_components()\n\treturn a + b + c + d + e + f\n/atom/proc/read_appearance()\n\treturn list(appearance, filters)\n/atom/movable/proc/read_step_offsets()\n\treturn step_x + step_y\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        for path in [
            "/client/proc/read_engine_state",
            "/matrix/proc/read_components",
            "/atom/proc/read_appearance",
            "/atom/movable/proc/read_step_offsets",
        ] {
            let procedure = procedure_by_path(&registry, path);
            registry
                .compile_vm_implementations(
                    &compilation,
                    [procedure.effective_target.expect("procedure body")],
                )
                .unwrap_or_else(|error| panic!("{path} should bind engine fields: {error:?}"));
        }
    }

    #[test]
    fn client_mouse_pointer_icon_is_a_null_initialized_engine_field() {
        let compilation = TestProject::compile(
            "/client/MouseDown(value)\n\tif(initial(mouse_pointer_icon))\n\t\treturn \"unexpected initial pointer\"\n\tmouse_pointer_icon = value\n\treturn mouse_pointer_icon\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/client/proc/MouseDown");
        let target = procedure
            .effective_target
            .expect("MouseDown should have a body");
        let executable = registry
            .compile_vm_implementations(&compilation, [target])
            .expect("the documented client field should bind during lowering");
        let entry = executable
            .implementation(target)
            .expect("MouseDown should be linked");
        let mut state = ExecutionState::new();
        let client = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/client").unwrap());
        let pointer = Value::file("cursor.dmi");
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                std::slice::from_ref(&pointer),
                &mut state,
                &ExecutionContext::new(Value::Datum(client), Value::Null),
            ),
            Ok(pointer.clone())
        );
        assert_eq!(
            state
                .heap()
                .datum_field(client, &FieldName::parse("mouse_pointer_icon").unwrap()),
            Ok(&pointer)
        );
    }

    #[test]
    fn compiler_predicates_bypass_synthetic_static_call_targets() {
        let compilation = TestProject::compile(
            "/proc/check(atom/value)\n\treturn isturf(value) + isnull(value) + istype(value, /atom)\n",
        );
        let executable = ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("compiler predicates should link");
        let entry = executable
            .module()
            .effective_procedure_id("/proc/check")
            .expect("check procedure");
        let program = executable.module().procedure(entry).expect("check body");
        assert_eq!(
            program
                .instructions
                .iter()
                .filter(|instruction| matches!(instruction, Instruction::TypePredicate { .. }))
                .count(),
            3
        );
        assert!(
            !program
                .instructions
                .iter()
                .any(|instruction| matches!(instruction, Instruction::Call { .. })),
            "language predicates must not pay a synthetic procedure call: {:?}",
            program.instructions
        );
    }

    #[test]
    fn lowers_standard_datum_type_field_for_all_datums() {
        let compilation =
            TestProject::compile("/datum/example\n\tproc/read()\n\t\treturn list(type, tag)\n");
        let registry = ProcedureRegistry::build(&compilation);
        let datum = procedure_by_path(&registry, "/datum/example/proc/read");
        registry
            .compile_vm_implementations(
                &compilation,
                datum.implementations.iter().map(|body| body.id),
            )
            .expect("datum type should compile as its built-in src field");
    }

    #[test]
    fn lowers_documented_world_host_fields_as_builtin_src_fields() {
        let compilation = TestProject::compile(
            "/world/proc/read_host()\n\tworld.log = file(\"data/dd.log\")\n\treturn list(name, hub, hub_password, internet_address, address, status, port, params, log, area, mob, turf, byond_version, byond_build, cache_lifespan, executor, game_state, host, loop_checks, map_format, map_cpu, movement_mode, process, reachable, sleep_offline, tick_usage, url, version, view, visibility)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/world/proc/read_host");
        registry
            .compile_vm_implementations(
                &compilation,
                [procedure.effective_target.expect("procedure body")],
            )
            .expect("documented world host fields should lower as src fields");
    }

    #[test]
    fn links_standard_location_predicates_as_variadic_builtins() {
        let compilation = TestProject::compile(concat!(
            "/proc/valid_locations()\n",
            "\treturn isarea(new /area, new /area/station) + isobj(new /obj, new /obj/item) + ismob(new /mob, new /mob/living)\n",
            "/proc/invalid_locations()\n",
            "\treturn isarea(new /turf) + isobj(new /mob) + ismob(new /obj)\n",
        ));

        assert_eq!(
            execute_effective(&compilation, "/proc/valid_locations", &[]),
            Ok(Value::number(3.0))
        );
        assert_eq!(
            execute_effective(&compilation, "/proc/invalid_locations", &[]),
            Ok(Value::number(0.0))
        );
    }

    #[test]
    fn links_direction_text_and_orange_standard_builtins() {
        let compilation = TestProject::compile(concat!(
            "/proc/classify()\n",
            "\treturn istext(\"hello\") + istext(3)\n",
            "/atom/example/proc/neighbors(other)\n",
            "\treturn get_dir(src, other) + length(orange(1, src))\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        registry
            .compile_vm(&compilation)
            .expect("standard direction/text/orange builtins should link");
        assert_eq!(
            execute_effective(&compilation, "/proc/classify", &[]),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn selected_dynamic_literal_call_includes_matching_method_implementation() {
        let compilation = TestProject::compile(
            "/datum/receiver\n\tproc/entry()\n\t\treturn call(src, \"register\")()\n\tproc/register()\n\t\treturn 9\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/receiver/proc/entry");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("literal dynamic method should be included");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn typesof_procedure_family_links_generated_managed_global_initializers() {
        let compilation = TestProject::compile(concat!(
            "/datum/controller/global_vars/var/global/list/species_list\n",
            "/datum/controller/global_vars/var/global/list/crafting_recipes\n",
            "/datum/controller/global_vars/proc/InitGlobalspecies_list()\n",
            "\tspecies_list = list()\n",
            "/datum/controller/global_vars/proc/InitGlobalcrafting_recipes()\n",
            "\tcrafting_recipes = list()\n",
            "/datum/controller/global_vars/Initialize()\n",
            "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
            "\t\tcall(src, glob_proc)()\n",
            "/datum/unrelated/proc/not_an_initializer()\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let initialize =
            procedure_by_path(&registry, "/datum/controller/global_vars/proc/Initialize")
                .effective_target
                .expect("Initialize implementation");
        let closure = registry.implementation_closure(&compilation, [initialize]);

        for path in [
            "/datum/controller/global_vars/proc/InitGlobalspecies_list",
            "/datum/controller/global_vars/proc/InitGlobalcrafting_recipes",
        ] {
            let target = procedure_by_path(&registry, path)
                .effective_target
                .expect("generated managed-global initializer");
            assert!(closure.contains(&target), "closure omitted {path}");
        }
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/not_an_initializer")
            .effective_target
            .expect("unrelated implementation");
        assert!(!closure.contains(&unrelated));

        registry
            .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
            .expect("bounded procedure family must be linked into the symbolic module");
    }

    #[test]
    fn typesof_procedure_family_index_ignores_large_unrelated_registry() {
        let mut source = String::from(concat!(
            "/datum/target/proc/first()\n",
            "/datum/target/proc/second()\n",
            "/datum/target/proc/Initialize()\n",
            "\tfor(var/path in typesof(/datum/target/proc))\n",
            "\t\tcall(src, path)()\n",
        ));
        for index in 0..2_048 {
            source.push_str(&format!(
                "/datum/unrelated_{index}/proc/run()\n\treturn {index}\n"
            ));
        }
        let compilation = TestProject::compile(&source);
        let registry = ProcedureRegistry::build(&compilation);
        let initialize = procedure_by_path(&registry, "/datum/target/proc/Initialize")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [initialize]);
        for path in ["/datum/target/proc/first", "/datum/target/proc/second"] {
            assert!(
                closure.contains(&procedure_by_path(&registry, path).effective_target.unwrap()),
                "indexed family omitted {path}",
            );
        }
        assert!(
            !closure.contains(
                &procedure_by_path(&registry, "/datum/unrelated_2047/proc/run")
                    .effective_target
                    .unwrap()
            )
        );
    }

    #[test]
    fn lazy_registry_matches_eager_dependencies_and_defers_body_analysis() {
        let compilation = TestProject::compile(concat!(
            "/datum/base/proc/New()\n",
            "/datum/base/proc/ping()\n\treturn 1\n",
            "/datum/base/child/New()\n\t..()\n",
            "/datum/base/child/ping()\n\treturn 2\n",
            "/datum/runner/proc/run(datum/base/value)\n",
            "\tfor(var/path in typesof(/datum/base/proc))\n",
            "\t\tcall(value, path)()\n",
            "\tvar/datum/base/child/item = new\n",
            "\treturn item.ping()\n",
        ));
        let eager = ProcedureRegistry::build(&compilation);
        let lazy = ProcedureRegistry::build_lazy(&compilation);
        assert!(!lazy.dependencies_initialized());
        assert_eq!(lazy.procedures(), eager.procedures());
        let root = procedure_by_path(&eager, "/datum/runner/proc/run")
            .effective_target
            .unwrap();
        assert_eq!(
            lazy.implementation_closure_with_stats(&compilation, [root]),
            eager.implementation_closure_with_stats(&compilation, [root]),
        );
        assert!(lazy.dependencies_initialized());
        assert_eq!(lazy.build_stats(), eager.build_stats());
        assert_eq!(
            lazy.compile_vm_implementations_symbolic_dynamic(&compilation, [root])
                .map(|executable| executable.stats().clone()),
            eager
                .compile_vm_implementations_symbolic_dynamic(&compilation, [root])
                .map(|executable| executable.stats().clone()),
        );
    }

    #[test]
    fn managed_global_macro_retains_underscore_named_initializer() {
        let compilation = TestProject::compile(concat!(
            "#define GLOBAL_MANAGED(X, InitValue) /datum/controller/global_vars/proc/InitGlobal##X(){ X = InitValue; }\n",
            "#define GLOBAL_RAW(X) /datum/controller/global_vars/var/global##X\n",
            "#define GLOBAL_LIST_INIT(X, InitValue) GLOBAL_RAW(/list/##X); GLOBAL_MANAGED(X, InitValue)\n",
            "#define GLOBAL_LIST_EMPTY(X) GLOBAL_LIST_INIT(X, list())\n",
            "GLOBAL_LIST_EMPTY(all_huds)\n",
            "GLOBAL_LIST_EMPTY(huds_by_category)\n",
            "GLOBAL_LIST_INIT(huds, list(1))\n",
            "/datum/controller/global_vars/Initialize()\n",
            "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
            "\t\tcall(src, glob_proc)()\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        for path in [
            "/datum/controller/global_vars/proc/InitGlobalall_huds",
            "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
            "/datum/controller/global_vars/proc/InitGlobalhuds",
        ] {
            assert!(
                registry
                    .procedures()
                    .iter()
                    .any(|procedure| procedure.path.to_string() == path),
                "managed global macro omitted {path}"
            );
        }
    }

    #[test]
    fn reopened_initglobal_is_enumerated_and_invoked_by_procedure_typesof() {
        let compilation = TestProject::compile(concat!(
            "/datum/controller/global_vars\n\tvar/trace = 0\n",
            "/datum/controller/global_vars/proc/InitGlobalhuds_by_category()\n\ttrace += 1\n",
            "/datum/controller/global_vars/InitGlobalhuds_by_category()\n\t..()\n\ttrace += 10\n",
            "/datum/controller/global_vars/proc/InitGlobalhuds()\n\ttrace *= 2\n",
            "/datum/controller/global_vars/Initialize()\n",
            "\tfor(var/glob_proc in typesof(/datum/controller/global_vars/proc))\n",
            "\t\tcall(src, glob_proc)()\n",
            "\treturn trace\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let initialize =
            procedure_by_path(&registry, "/datum/controller/global_vars/proc/Initialize")
                .effective_target
                .unwrap();
        let closure = registry.implementation_closure(&compilation, [initialize]);
        for path in [
            "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
            "/datum/controller/global_vars/proc/InitGlobalhuds",
        ] {
            let target = procedure_by_path(&registry, path).effective_target.unwrap();
            assert!(closure.contains(&target), "closure omitted {path}");
        }
        let executable = registry
            .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
            .expect("reopened managed global family should compile");
        let catalog = executable
            .module()
            .procedure_type_paths()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        for path in [
            "/datum/controller/global_vars/proc/InitGlobalhuds_by_category",
            "/datum/controller/global_vars/proc/InitGlobalhuds",
        ] {
            assert!(
                catalog.iter().any(|entry| entry == path),
                "catalog omitted {path}: {catalog:?}"
            );
        }
        let entry = executable.implementation(initialize).unwrap();
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/controller/global_vars").unwrap());
        state
            .heap_mut()
            .set_datum_field(
                receiver,
                FieldName::parse("trace").unwrap(),
                Value::number(0.0),
            )
            .unwrap();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(22.0))
        );
    }

    #[test]
    fn typed_global_member_call_links_exact_runtime_receiver_target() {
        let compilation = TestProject::compile(
            "var/global/datum/log_holder/logger\n/proc/entry()\n\treturn logger.Log(4)\n/datum/log_holder/proc/Log(value)\n\treturn value + 3\n/datum/unrelated/proc/Log(value)\n\treturn value + 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry");
        let entry_target = entry.effective_target.expect("entry implementation");
        let (closure, _) = registry.implementation_closure_with_stats(&compilation, [entry_target]);
        let log_target = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
            .effective_target
            .expect("logger implementation");
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
            .effective_target
            .expect("unrelated implementation");
        assert!(closure.contains(&log_target));
        assert!(!closure.contains(&unrelated));

        let executable = registry
            .compile_vm_implementations(&compilation, [entry_target])
            .expect("typed member target should link");
        let mut state = ExecutionState::new();
        let logger = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/log_holder").unwrap());
        state.set_global(
            dm_value::FieldName::parse("logger").unwrap(),
            Value::Datum(logger),
        );
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry_target).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn interpolated_world_member_call_links_runtime_candidates() {
        let compilation = TestProject::compile(
            "/world/proc/get_world_state_for_logging()\n\treturn 7\n/datum/log_entry/proc/render()\n\tvar/list/entries = list()\n\tentries.Add(\"[world.get_world_state_for_logging()]\")\n\treturn entries[1]\n/datum/unrelated/proc/get_world_state_for_logging()\n\treturn 99\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/log_entry/proc/render")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        let world = procedure_by_path(&registry, "/world/proc/get_world_state_for_logging")
            .effective_target
            .unwrap();
        let unrelated = procedure_by_path(
            &registry,
            "/datum/unrelated/proc/get_world_state_for_logging",
        )
        .effective_target
        .unwrap();
        assert!(closure.contains(&world));
        assert!(closure.contains(&unrelated));

        let executable = registry
            .compile_vm_implementations_symbolic_dynamic(&compilation, [entry])
            .expect("macro-shaped nested world member target should link");
        assert!(executable.implementation(world).is_some());
        let mut state = ExecutionState::new();
        let world_datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        state.set_global(
            dm_value::FieldName::parse("world").unwrap(),
            Value::Datum(world_datum),
        );
        let log_entry = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/log_entry").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(log_entry), Value::Null),
            ),
            Ok(Value::text("7"))
        );
    }

    #[test]
    fn typed_global_field_chain_links_exact_runtime_receiver_target() {
        let compilation = TestProject::compile(
            "var/global/datum/globals/GLOB\n/datum/globals/var/datum/log_holder/logger\n/proc/entry()\n\treturn GLOB.logger.Log(4)\n/datum/log_holder/proc/Log(value)\n\treturn value + 3\n/datum/unrelated/proc/Log(value)\n\treturn value + 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .expect("entry implementation");
        let closure = registry.implementation_closure(&compilation, [entry]);
        let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
            .effective_target
            .expect("logger implementation");
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
            .effective_target
            .expect("unrelated implementation");
        assert!(closure.contains(&log));
        assert!(!closure.contains(&unrelated));
    }

    #[test]
    fn implicit_inherited_typed_field_member_call_links_exact_target() {
        let compilation = TestProject::compile(
            "/datum/base/var/datum/log_holder/logger\n/datum/child\n\tparent_type = /datum/base\n\tproc/entry()\n\t\treturn logger.Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/child/proc/entry")
            .effective_target
            .expect("entry implementation");
        let closure = registry.implementation_closure(&compilation, [entry]);
        let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
            .effective_target
            .expect("logger implementation");
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
            .effective_target
            .expect("unrelated implementation");
        assert!(closure.contains(&log));
        assert!(!closure.contains(&unrelated));
    }

    #[test]
    fn proven_untyped_local_alias_narrows_until_dynamic_reassignment() {
        let compilation = TestProject::compile(
            "var/global/datum/globals/GLOB\n/datum/globals/var/datum/log_holder/logger\n/proc/narrowed()\n\tvar/alias = GLOB.logger\n\treturn alias.Log()\n/proc/invalidated(value)\n\tvar/alias = GLOB.logger\n\talias = value\n\treturn alias.Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let log = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
            .effective_target
            .unwrap();
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
            .effective_target
            .unwrap();
        let narrowed = procedure_by_path(&registry, "/proc/narrowed")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [narrowed]);
        assert!(closure.contains(&log));
        assert!(!closure.contains(&unrelated));

        let invalidated = procedure_by_path(&registry, "/proc/invalidated")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [invalidated]);
        assert!(closure.contains(&log));
        assert!(closure.contains(&unrelated));
    }

    #[test]
    fn typed_procedure_return_chain_narrows_member_dispatch() {
        let compilation = TestProject::compile(
            "/proc/get_logger() as /datum/log_holder\n\treturn null\n/datum/provider/proc/get_logger() as /datum/log_holder\n\treturn null\n/proc/from_global()\n\treturn get_logger()?.Log()\n/datum/provider/proc/from_member()\n\treturn (src.get_logger()).Log()\n/datum/log_holder/proc/Log()\n\treturn 7\n/datum/unrelated/proc/Log()\n\treturn 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/Log")
            .effective_target
            .unwrap();
        for path in ["/proc/from_global", "/datum/provider/proc/from_member"] {
            let entry = procedure_by_path(&registry, path).effective_target.unwrap();
            let closure = registry.implementation_closure(&compilation, [entry]);
            assert!(
                !closure.contains(&unrelated),
                "typed return receiver in {path} must not link unrelated Log"
            );
        }
    }

    #[test]
    fn untyped_member_call_links_broad_candidates_and_dispatches_runtime_override() {
        let compilation = TestProject::compile(
            "/proc/entry(receiver)\n\treturn receiver.Log()\n/datum/base/proc/Log()\n\treturn 1\n/datum/child\n\tparent_type = /datum/base\n/datum/child/Log()\n\treturn 2\n/datum/other/proc/Log()\n\treturn 3\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry");
        let entry_target = entry.effective_target.expect("entry implementation");
        let (closure, _) = registry.implementation_closure_with_stats(&compilation, [entry_target]);
        for path in [
            "/datum/base/proc/Log",
            "/datum/child/proc/Log",
            "/datum/other/proc/Log",
        ] {
            assert!(
                closure.contains(
                    &procedure_by_path(&registry, path)
                        .effective_target
                        .expect("member implementation")
                ),
                "untyped receiver must retain candidate {path}"
            );
        }
        let executable = registry
            .compile_vm_implementations(&compilation, [entry_target])
            .expect("broad dynamic member closure should link");
        let mut state = ExecutionState::new();
        let child = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/child").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry_target).unwrap(),
                &[Value::Datum(child)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(2.0))
        );
    }

    #[test]
    fn untyped_member_candidates_are_symbolic_until_runtime_dispatch() {
        let compilation = TestProject::compile(
            "/proc/entry(receiver)\n\treturn receiver.Log()\n/datum/child/proc/Log()\n\treturn 2\n/datum/unrelated/proc/Log()\n\treturn 100\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .expect("entry implementation");
        assert_eq!(
            registry.eager_implementation_closure(&compilation, [entry]),
            BTreeSet::from([entry]),
            "genuinely untyped candidates must remain symbolic at the boot gate"
        );
        let executable = registry
            .compile_vm_implementations_symbolic_dynamic(&compilation, [entry])
            .expect("symbolic dynamic module should link");
        assert_eq!(executable.module().deferred_procedure_count(), 2);
        assert_eq!(
            executable.module().materialized_deferred_procedure_count(),
            0
        );

        let mut state = ExecutionState::new();
        let child = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/child").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[Value::Datum(child)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(2.0))
        );
        assert_eq!(
            executable.module().materialized_deferred_procedure_count(),
            1,
            "only the runtime-selected override should compile"
        );
    }

    #[test]
    fn all_symbolic_bootstrap_module_defers_unreachable_invalid_body() {
        let compilation = TestProject::compile(
            "/proc/reached()\n\treturn 7\n/proc/unreachable_invalid()\n\tvar/const/answer = 42\n\tanswer = 9\n\treturn answer\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let reached = procedure_by_path(&registry, "/proc/reached")
            .effective_target
            .expect("reached implementation");
        let invalid = procedure_by_path(&registry, "/proc/unreachable_invalid")
            .effective_target
            .expect("invalid implementation");
        let executable = registry
            .compile_vm_all_symbolic_deferred(&compilation)
            .expect("unreached body errors must remain deferred");
        assert_eq!(executable.module().deferred_procedure_count(), 2);
        assert_eq!(
            executable.module().materialized_deferred_procedure_count(),
            0
        );

        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(reached).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            executable.module().materialized_deferred_procedure_count(),
            1,
            "only the reached initializer callee should lower"
        );
        assert!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(invalid).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            )
            .is_err(),
            "the deferred validation error must surface if the bad body is reached"
        );
    }

    #[test]
    fn initializer_frontier_omits_unrelated_procedure_specs() {
        let compilation = TestProject::compile(
            "/proc/reached()\n\treturn helper()\n/proc/helper()\n\treturn 7\n/proc/unrelated_invalid()\n\tvar/const/answer = 42\n\tanswer = 9\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let reached = procedure_by_path(&registry, "/proc/reached")
            .effective_target
            .expect("reached implementation");
        let invalid = procedure_by_path(&registry, "/proc/unrelated_invalid")
            .effective_target
            .expect("invalid implementation");
        let executable = registry
            .compile_vm_initializer_frontier_symbolic_deferred(&compilation, ["reached"])
            .expect("frontier should link");
        assert_eq!(
            executable.module().deferred_procedure_count(),
            2,
            "the named root and its static callee should be retained"
        );
        assert!(
            executable.implementation(invalid).is_none(),
            "an unrelated body must not even receive a bootstrap module spec"
        );
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(reached).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(7.0))
        );
        assert_eq!(
            executable.module().materialized_deferred_procedure_count(),
            2
        );
    }

    #[test]
    fn construction_closure_narrows_typed_and_subtypesof_families() {
        let compilation = TestProject::compile(
            "/proc/from_loop()\n\tfor(var/path in subtypesof(/datum/base))\n\t\tvar/datum/value = new path\n/proc/from_typesof()\n\tfor(var/path in typesof(/datum/base))\n\t\tvar/datum/value = new path\n/proc/from_typecache()\n\tfor(var/datum/base/path as anything in typecacheof(path = /datum/base, ignore_root_path = TRUE))\n\t\tvar/datum/value = new path\n/proc/from_typed(datum/base/path)\n\tvar/datum/value = new path\n/proc/from_unknown(path)\n\tvar/datum/value = new path\n/proc/from_newlist()\n\treturn newlist(/datum/base/child)\n/proc/from_dynamic_newlist(path)\n\treturn newlist(path)\n/datum/base/New()\n/datum/base/child/New()\n/datum/unrelated/New()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let target = |path| {
            procedure_by_path(&registry, path)
                .effective_target
                .expect("implementation")
        };
        let base_new = target("/datum/base/proc/New");
        let child_new = target("/datum/base/child/proc/New");
        let unrelated_new = target("/datum/unrelated/proc/New");

        for entry in [
            "/proc/from_loop",
            "/proc/from_typesof",
            "/proc/from_typecache",
            "/proc/from_typed",
        ] {
            let closure = registry.implementation_closure(&compilation, [target(entry)]);
            assert!(
                closure.contains(&base_new),
                "{entry} should retain base New"
            );
            assert!(
                closure.contains(&child_new),
                "{entry} should retain descendant New"
            );
            assert!(
                !closure.contains(&unrelated_new),
                "{entry} must omit unrelated New"
            );
        }

        let unknown = registry.implementation_closure(&compilation, [target("/proc/from_unknown")]);
        assert!(unknown.contains(&base_new));
        assert!(unknown.contains(&child_new));
        assert!(
            unknown.contains(&unrelated_new),
            "a genuinely untyped construction must retain all New candidates"
        );

        let newlist = registry.implementation_closure(&compilation, [target("/proc/from_newlist")]);
        assert!(newlist.contains(&child_new));
        assert!(!newlist.contains(&unrelated_new));
        let dynamic_newlist =
            registry.implementation_closure(&compilation, [target("/proc/from_dynamic_newlist")]);
        assert!(dynamic_newlist.contains(&base_new));
        assert!(dynamic_newlist.contains(&child_new));
        assert!(dynamic_newlist.contains(&unrelated_new));
    }

    #[test]
    fn typed_virtual_overrides_are_symbolic_but_declared_base_is_eager() {
        let compilation = TestProject::compile(
            "/proc/entry(datum/base/receiver)\n\treturn receiver.Log()\n/datum/base/proc/Log()\n\treturn 1\n/datum/base/child/Log()\n\treturn expensive_helper()\n/datum/base/child/proc/expensive_helper()\n\treturn 2\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let base = procedure_by_path(&registry, "/datum/base/proc/Log")
            .effective_target
            .unwrap();
        let child = procedure_by_path(&registry, "/datum/base/child/proc/Log")
            .effective_target
            .unwrap();
        let helper = procedure_by_path(&registry, "/datum/base/child/proc/expensive_helper")
            .effective_target
            .unwrap();
        let eager = registry.eager_implementation_closure(&compilation, [entry]);
        assert!(eager.contains(&base));
        assert!(!eager.contains(&child));
        assert!(!eager.contains(&helper));
        let full = registry.implementation_closure(&compilation, [entry]);
        assert!(full.contains(&child));
        assert!(full.contains(&helper));
    }

    #[test]
    fn typed_local_member_call_after_dynamic_new_retains_admin_verb_method_family() {
        let compilation = TestProject::compile(concat!(
            "/datum/admin_verb/proc/__avd_check_should_exist()\n\treturn 1\n",
            "/datum/admin_verb/AdminVOX/__avd_check_should_exist()\n\treturn 0\n",
            "/datum/controller/subsystem/admin_verbs/proc/setup_verb_list()\n",
            "\tvar/datum/admin_verb/verb_type = /datum/admin_verb/AdminVOX\n",
            "\tvar/datum/admin_verb/verb_singleton = new verb_type\n",
            "\treturn verb_singleton.__avd_check_should_exist()\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let setup = procedure_by_path(
            &registry,
            "/datum/controller/subsystem/admin_verbs/proc/setup_verb_list",
        )
        .effective_target
        .unwrap();
        let base = procedure_by_path(&registry, "/datum/admin_verb/proc/__avd_check_should_exist")
            .effective_target
            .unwrap();
        let override_target = procedure_by_path(
            &registry,
            "/datum/admin_verb/AdminVOX/proc/__avd_check_should_exist",
        )
        .effective_target
        .unwrap();
        let closure = registry.implementation_closure(&compilation, [setup]);
        assert!(closure.contains(&base), "base method must be retained");
        assert!(
            closure.contains(&override_target),
            "compatible generated overrides must be retained"
        );
    }

    #[test]
    fn complete_symbolic_lifecycle_table_keeps_unproven_runtime_methods_deferred() {
        let compilation = TestProject::compile(concat!(
            "/proc/startup()\n\treturn 1\n",
            "/datum/runtime_only/proc/invoke_from_data()\n\treturn 9\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let startup = procedure_by_path(&registry, "/proc/startup")
            .effective_target
            .unwrap();
        let runtime_only =
            procedure_by_path(&registry, "/datum/runtime_only/proc/invoke_from_data")
                .effective_target
                .unwrap();
        assert!(
            !registry
                .implementation_closure(&compilation, [startup])
                .contains(&runtime_only),
            "fixture must be outside the statically proven closure"
        );
        let executable = registry
            .compile_vm_all_symbolic_with_eager_roots(&compilation, [startup])
            .expect("complete deferred table should link");
        assert!(executable.implementation(startup).is_some());
        assert!(executable.implementation(runtime_only).is_some());
        assert!(executable.module().deferred_procedure_count() >= 1);
    }

    #[test]
    fn proc_pseudo_macro_is_the_current_canonical_procedure_reference() {
        let compilation = TestProject::compile(
            "/datum/example/proc/reenter(again)\n\tif(again)\n\t\treturn call(src, __PROC__)(0)\n\treturn 7\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/example/proc/reenter");
        let target = procedure.effective_target.unwrap();
        let executable = registry
            .compile_vm_implementations(&compilation, [target])
            .expect("__PROC__ should lower as a procedure reference");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(target).unwrap(),
                &[Value::number(1.0)],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn caller_exposes_the_actual_calling_frame_as_a_callee_datum() {
        let compilation = TestProject::compile(
            "/datum/example/proc/outer()\n\treturn inner()\n/datum/example/proc/inner()\n\treturn caller.src == src && isnull(caller.caller)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let outer = procedure_by_path(&registry, "/datum/example/proc/outer");
        let target = outer.effective_target.unwrap();
        let executable = registry
            .compile_vm_implementations(&compilation, [target])
            .expect("caller should lower as an implicit proc variable");
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(target).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn dependency_closure_uses_preindexed_dynamic_selector_candidates() {
        let compilation = TestProject::compile(
            "/proc/entry()\n\treturn call(src, \"register\")()\n/datum/one/proc/register()\n\treturn 1\n/datum/two/proc/register()\n\treturn 2\n/datum/irrelevant/proc/alpha()\n\treturn 3\n/datum/irrelevant/proc/beta()\n\treturn 4\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .expect("entry implementation");
        let (closure, stats) = registry.implementation_closure_with_stats(&compilation, [entry]);

        assert_eq!(
            closure.len(),
            3,
            "entry and both matching methods are reachable"
        );
        assert_eq!(stats.dynamic_selectors_resolved, 1);
        assert_eq!(
            stats.dynamic_candidates_considered, 2,
            "unrelated procedures must not be scanned as dynamic candidates"
        );
        assert_eq!(stats.bodies_visited, 3);
    }

    #[test]
    fn first_class_proc_path_links_through_local_argument_and_field_call() {
        let compilation = TestProject::compile(
            "/datum/sorter\n\tvar/cmp\n/datum/sorter/proc/run(comparator)\n\tvar/local_cmp = comparator\n\tsrc.cmp = local_cmp\n\treturn call(src.cmp)(2, 7)\n/proc/cmp_subsystem_init(a, b)\n\treturn b - a\n/proc/unrelated()\n\treturn 99\n/proc/entry(comparator = /proc/cmp_subsystem_init)\n\tvar/list/refs = list(/proc/cmp_subsystem_init, /datum/sorter)\n\tvar/datum/sorter/sorter = new\n\treturn sorter.run(refs[1] || comparator)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let comparator = procedure_by_path(&registry, "/proc/cmp_subsystem_init")
            .effective_target
            .unwrap();
        let unrelated = procedure_by_path(&registry, "/proc/unrelated")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(closure.contains(&comparator));
        assert!(!closure.contains(&unrelated));

        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("first-class comparator reference should link");
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(5.0))
        );
    }

    #[test]
    fn relative_proc_ref_retains_signal_callback() {
        let compilation = TestProject::compile(
            "/datum/handler/proc/register()\n\tvar/callback = nameof(.proc/new_item_created)\n\treturn call(src, callback)()\n/datum/handler/proc/new_item_created()\n\treturn 42\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let register = procedure_by_path(&registry, "/datum/handler/proc/register")
            .effective_target
            .unwrap();
        let callback = procedure_by_path(&registry, "/datum/handler/proc/new_item_created")
            .effective_target
            .unwrap();

        assert!(
            registry
                .implementation_closure(&compilation, [register])
                .contains(&callback),
            "PROC_REF-style nameof(.proc/name) callbacks must remain linked",
        );
    }

    #[test]
    fn typed_proc_ref_retains_signal_callback_for_subtype_receiver() {
        let compilation = TestProject::compile(
            "/datum/module/proc/register(datum/module/syndicate/receiver)\n\tvar/callback = nameof(/datum/module.proc/add_overlay)\n\treturn call(receiver, callback)()\n/datum/module/proc/add_overlay()\n\treturn 42\n/datum/module/syndicate\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let register = procedure_by_path(&registry, "/datum/module/proc/register")
            .effective_target
            .unwrap();
        let callback = procedure_by_path(&registry, "/datum/module/proc/add_overlay")
            .effective_target
            .unwrap();

        assert!(
            registry
                .implementation_closure(&compilation, [register])
                .contains(&callback),
            "TYPE_PROC_REF-style nameof(/owner.proc/name) must retain the callback",
        );
        let executable = registry
            .compile_vm_implementations(&compilation, [register])
            .expect("typed callback should link");
        let mut state = ExecutionState::new();
        state.set_type_parents(
            [
                (TypePath::parse("/datum").unwrap(), None),
                (
                    TypePath::parse("/datum/module").unwrap(),
                    Some(TypePath::parse("/datum").unwrap()),
                ),
                (
                    TypePath::parse("/datum/module/syndicate").unwrap(),
                    Some(TypePath::parse("/datum/module").unwrap()),
                ),
            ]
            .into(),
        );
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/module/syndicate").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(register).unwrap(),
                &[Value::Datum(receiver)],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(42.0)),
        );
    }

    #[test]
    fn literal_text2path_proc_reference_retains_inferable_symbol() {
        let compilation = TestProject::compile(
            "/proc/cmp_value(a, b)\n\treturn a - b\n/proc/entry()\n\tvar/cmp = text2path(\"/proc/cmp_value\")\n\treturn call(cmp)(9, 4)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        assert_eq!(
            registry.build_stats().static_proc_reference_index_lookups,
            1,
            "one literal reference should require one indexed lookup regardless of registry size"
        );
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let comparator = procedure_by_path(&registry, "/proc/cmp_value")
            .effective_target
            .unwrap();
        assert!(
            registry
                .implementation_closure(&compilation, [entry])
                .contains(&comparator)
        );
        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .unwrap();
        let mut state = ExecutionState::new();
        state.set_type_paths([TypePath::parse("/proc/cmp_value").unwrap()]);
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(5.0))
        );
    }

    #[test]
    fn project_sort_wrapper_and_comparator_reference_are_retained_transitively() {
        let compilation = TestProject::compile(concat!(
            "/proc/cmp_desc(left, right)\n\treturn right - left\n",
            "/proc/sort_list(values, comparator)\n",
            "\treturn call(comparator)(values[1], values[2])\n",
            "/proc/entry()\n\treturn sort_list(list(1, 3), /proc/cmp_desc)\n",
            "/proc/unrelated()\n\treturn 99\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let wrapper = procedure_by_path(&registry, "/proc/sort_list")
            .effective_target
            .unwrap();
        let comparator = procedure_by_path(&registry, "/proc/cmp_desc")
            .effective_target
            .unwrap();
        let unrelated = procedure_by_path(&registry, "/proc/unrelated")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(closure.contains(&wrapper));
        assert!(closure.contains(&comparator));
        assert!(!closure.contains(&unrelated));

        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("project sort wrapper and comparator should link transitively");
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(2.0)),
        );
    }

    #[test]
    fn inherited_bare_call_retains_and_dispatches_runtime_subtype_override() {
        let compilation = TestProject::compile(concat!(
            "/atom/proc/add_debris_element()\n\treturn 1\n",
            "/obj/Initialize()\n\treturn add_debris_element()\n",
            "/obj/effect/statclick/ticket_list\n",
            "/obj/structure/barricade/wooden/add_debris_element()\n\treturn 9\n",
            "/datum/unrelated/add_debris_element()\n\treturn 100\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let initialize = procedure_by_path(&registry, "/obj/proc/Initialize")
            .effective_target
            .unwrap();
        let base = procedure_by_path(&registry, "/atom/proc/add_debris_element")
            .effective_target
            .unwrap();
        let override_target = procedure_by_path(
            &registry,
            "/obj/structure/barricade/wooden/proc/add_debris_element",
        )
        .effective_target
        .unwrap();
        let unrelated = procedure_by_path(&registry, "/datum/unrelated/proc/add_debris_element")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [initialize]);
        assert!(closure.contains(&base));
        assert!(closure.contains(&override_target));
        assert!(!closure.contains(&unrelated));
        assert!(
            !registry
                .eager_implementation_closure(&compilation, [initialize])
                .contains(&override_target),
            "compatible virtual overrides should stay deferred until dispatched",
        );

        let executable = registry
            .compile_vm_implementations_symbolic_dynamic(&compilation, [initialize])
            .unwrap();
        let mut state = ExecutionState::new();
        let wooden = TypePath::parse("/obj/structure/barricade/wooden").unwrap();
        state.set_type_parents(BTreeMap::from([
            (
                wooden.clone(),
                Some(TypePath::parse("/obj/structure/barricade").unwrap()),
            ),
            (
                TypePath::parse("/obj/structure/barricade").unwrap(),
                Some(TypePath::parse("/obj/structure").unwrap()),
            ),
            (
                TypePath::parse("/obj/structure").unwrap(),
                Some(TypePath::parse("/obj").unwrap()),
            ),
            (
                TypePath::parse("/obj").unwrap(),
                Some(TypePath::parse("/atom").unwrap()),
            ),
        ]));
        let receiver = state.heap_mut().allocate_datum(wooden);
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(initialize).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(9.0)),
        );
    }

    #[test]
    fn inherited_bare_call_after_nested_ternary_colon_is_retained() {
        let compilation = TestProject::compile(concat!(
            "/atom/proc/drop_location()\n\treturn 7\n",
            "/obj/proc/forward(value)\n\treturn value\n",
            "/obj/proc/click_alt(user)\n",
            "\treturn forward(user ? user : drop_location())\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let click_alt = procedure_by_path(&registry, "/obj/proc/click_alt")
            .effective_target
            .expect("click_alt body");
        let drop_location = procedure_by_path(&registry, "/atom/proc/drop_location")
            .effective_target
            .expect("drop_location body");

        assert!(
            registry
                .implementation_closure(&compilation, [click_alt])
                .contains(&drop_location),
            "a ternary nested in a call must retain its bare inherited false-arm call",
        );
        registry
            .compile_vm_implementations(&compilation, [click_alt])
            .expect("the retained nested-ternary call should resolve during lowering");
    }

    #[test]
    fn explicit_construction_links_and_runs_glob_new_before_returning() {
        let compilation = TestProject::compile(
            "/var/global/datum/controller/global_vars/GLOB\n/datum/controller/global_vars/New(marker)\n\tGLOB = src\n\tsrc.marker = marker\n/proc/entry()\n\tvar/datum/controller/global_vars/created = new /datum/controller/global_vars(17)\n\treturn GLOB == created && created.marker == 17\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let constructor = procedure_by_path(&registry, "/datum/controller/global_vars/proc/New")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(closure.contains(&constructor));

        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("constructor dependency should link");
        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn subtype_construction_uses_inherited_new_once_with_arguments() {
        let compilation = TestProject::compile(
            "/var/global/constructor_calls = 0\n/datum/base\n\tvar/marker\n/datum/base/New(marker)\n\tconstructor_calls += 1\n\tsrc.marker = marker\n/datum/base/child\n/proc/entry()\n\tvar/datum/base/child/created = new /datum/base/child(23)\n\treturn constructor_calls * 100 + created.marker\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let inherited = procedure_by_path(&registry, "/datum/base/proc/New")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(closure.contains(&inherited));

        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("inherited constructor dependency should link");
        let mut state = ExecutionState::new();
        state.set_global(
            FieldName::parse("constructor_calls").unwrap(),
            Value::number(0.0),
        );
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(123.0))
        );
    }

    #[test]
    fn dynamic_subsystem_catalog_construction_runs_each_generated_new_and_sets_global() {
        let compilation = TestProject::compile(
            "/var/global/datum/controller/subsystem/processing/dcs/SSdcs\n/datum/controller/subsystem\n/datum/controller/subsystem/processing\n/datum/controller/subsystem/processing/dcs/New()\n\tSSdcs = src\n/proc/entry()\n\tvar/list/subsystem_types = typesof(/datum/controller/subsystem) - /datum/controller/subsystem\n\tfor(var/I in subsystem_types)\n\t\tnew I\n\treturn istype(SSdcs, /datum/controller/subsystem/processing/dcs)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let constructor = procedure_by_path(
            &registry,
            "/datum/controller/subsystem/processing/dcs/proc/New",
        )
        .effective_target
        .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(closure.contains(&constructor));
        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("dynamic subsystem constructor family should link");
        let mut state = ExecutionState::new();
        let subsystem = TypePath::parse("/datum/controller/subsystem").unwrap();
        let processing = TypePath::parse("/datum/controller/subsystem/processing").unwrap();
        let dcs = TypePath::parse("/datum/controller/subsystem/processing/dcs").unwrap();
        state.set_type_paths([subsystem.clone(), processing.clone(), dcs.clone()]);
        state.set_type_parents(BTreeMap::from([
            (
                subsystem.clone(),
                Some(TypePath::parse("/datum/controller").unwrap()),
            ),
            (processing.clone(), Some(subsystem)),
            (dcs, Some(processing)),
        ]));
        state.set_global(FieldName::parse("SSdcs").unwrap(), Value::Null);
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn nested_list_assignment_preserves_get_element_result_value() {
        let compilation = TestProject::compile(
            "/datum/element\n/datum/element/child\n/datum/manager\n\tvar/list/elements_by_type\n/datum/manager/New()\n\telements_by_type = list()\n/datum/manager/proc/GetElement(list/arguments)\n\tvar/datum/element/eletype = arguments[1]\n\tvar/element_id = eletype\n\t. = elements_by_type[element_id]\n\tif(.)\n\t\treturn\n\t. = elements_by_type[element_id] = new eletype\n/proc/entry()\n\tvar/datum/manager/manager = new\n\tmanager.GetElement(list(/datum/element/child))\n\treturn istype(manager.GetElement(list(/datum/element/child)), /datum/element/child)\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/entry")
            .effective_target
            .unwrap();
        let executable = registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("GetElement-shaped procedure should compile");
        let mut state = ExecutionState::new();
        state.set_type_parents(BTreeMap::from([
            (
                TypePath::parse("/datum/element/child").unwrap(),
                Some(TypePath::parse("/datum/element").unwrap()),
            ),
            (
                TypePath::parse("/datum/manager").unwrap(),
                Some(TypePath::parse("/datum").unwrap()),
            ),
        ]));
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                executable.implementation(entry).unwrap(),
                &[],
                &mut state,
                &ExecutionContext::default(),
            ),
            Ok(Value::number(1.0))
        );
    }

    #[test]
    fn typed_global_member_call_links_exact_method_not_bare_global_proc() {
        let compilation = TestProject::compile(
            "/datum/log_holder/proc/Log()\n\treturn 1\n/proc/Log()\n\treturn 2\n/var/global/datum/log_holder/logger = new /datum/log_holder\n/proc/log_world()\n\treturn logger.Log()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/log_world")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        let member = procedure_by_path(&registry, "/datum/log_holder/proc/Log")
            .effective_target
            .unwrap();
        let bare = procedure_by_path(&registry, "/proc/Log")
            .effective_target
            .unwrap();
        assert!(closure.contains(&member));
        assert!(
            !closure.contains(&bare),
            "member syntax must not become a bare static call"
        );
        registry
            .compile_vm_implementations(&compilation, [entry])
            .expect("logger.Log linked");
    }

    #[test]
    fn ternary_false_arm_global_call_is_not_confused_with_colon_member_call() {
        let compilation = TestProject::compile(
            "/proc/format_text(var/x)\n\treturn x\n/proc/get_area_name(var/x, var/format_text)\n\treturn x\n/datum/holder/proc/get_area_name()\n\treturn 0\n/proc/run(var/datum/holder/A)\n\tvar/location = A ? format_text(A.name) : get_area_name(src, format_text=TRUE)\n\tA:get_area_name()\n\treturn location\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/proc/run")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [entry]);
        assert!(
            closure.contains(
                &procedure_by_path(&registry, "/proc/get_area_name")
                    .effective_target
                    .unwrap()
            )
        );
    }

    #[test]
    fn union_as_annotation_does_not_form_a_fake_type_path() {
        let compilation = TestProject::compile(
            "/proc/run(var/atom/location as mob|obj|turf)\n\treturn location\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("DM union annotation must remain valid");
    }

    #[test]
    fn input_constraints_do_not_form_fake_declared_types() {
        let compilation = TestProject::compile(
            "/proc/plain(message as message)\n\treturn message\n/proc/typed(mob/M as mob in world)\n\treturn M\n/proc/untyped(target as turf in world)\n\treturn target\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("input constraints must not be interpreted as datum paths");
    }

    #[test]
    fn suffix_array_local_is_a_list_not_a_fake_declared_type() {
        let compilation = TestProject::compile(
            "/area/misc/hilbertshotel/proc/storeRoom(roomSize)\n\tvar/storage[roomSize]\n\treturn storage\n",
        );
        ProcedureRegistry::build(&compilation)
            .compile_vm(&compilation)
            .expect("suffix array local must compile as a list");
    }

    #[test]
    fn suffix_array_instance_fields_are_registered_and_inherited() {
        let compilation = TestProject::compile(
            "/datum/dna\n\tvar/mutation_index[4]\n\tproc/set_entry(index, value)\n\t\tmutation_index[index] = value\n\t\treturn mutation_index[index]\n/mob/living/carbon\n\tvar/list/overlays_standing[8]\n/mob/living/carbon/human/proc/read_overlay(index)\n\treturn overlays_standing[index]\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        for path in [
            "/datum/dna/proc/set_entry",
            "/mob/living/carbon/human/proc/read_overlay",
        ] {
            let procedure = procedure_by_path(&registry, path);
            registry
                .compile_vm_implementations(
                    &compilation,
                    [procedure.effective_target.expect("procedure body")],
                )
                .unwrap_or_else(|error| {
                    panic!("{path} should inherit suffix-array field: {error:?}")
                });
        }
    }

    #[test]
    fn module_specs_copy_only_bindings_referenced_by_each_body() {
        let compilation = TestProject::compile(
            "var/global/used = 4\nvar/global/unused_one = 8\nvar/global/unused_two = 16\n/datum/example\n\tvar/field_used = 3\n\tvar/field_unused = 9\n\tproc/run()\n\t\treturn used + field_used\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let run = procedure_by_path(&registry, "/datum/example/proc/run");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                [run.effective_target.expect("run implementation")],
            )
            .expect("referenced bindings compile");

        assert_eq!(executable.stats().global_field_bindings, 1);
        assert_eq!(executable.stats().src_field_bindings, 1);
        assert_eq!(executable.stats().static_registry_builds, 1);
        let entry = executable
            .implementation(run.effective_target.expect("run implementation"))
            .expect("run is linked");
        let mut state = ExecutionState::new();
        state.set_global(
            dm_value::FieldName::parse("used").unwrap(),
            Value::number(4.0),
        );
        let datum = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/example").unwrap());
        state
            .heap_mut()
            .set_datum_field(
                datum,
                dm_value::FieldName::parse("field_used").unwrap(),
                Value::number(3.0),
            )
            .unwrap();
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(datum), Value::Null),
            ),
            Ok(Value::number(7.0))
        );
    }

    #[test]
    fn module_binding_lookup_work_depends_on_references_not_global_inventory() {
        let compile = |unused_globals: usize| {
            let mut source = "var/global/used = 4\n/proc/run()\n\treturn used\n".to_owned();
            for index in 0..unused_globals {
                source.push_str(&format!("var/global/unused_{index} = {index}\n"));
            }
            let compilation = TestProject::compile(&source);
            let registry = ProcedureRegistry::build(&compilation);
            let run = procedure_by_path(&registry, "/proc/run")
                .effective_target
                .expect("run implementation");
            let executable = registry
                .compile_vm_implementations(&compilation, [run])
                .expect("run should link");
            (
                executable.stats().global_binding_index_lookups,
                executable.stats().typed_global_index_lookups,
                executable.stats().global_field_bindings,
            )
        };

        assert_eq!(
            compile(2),
            compile(200),
            "unreferenced project globals must not increase per-body binding work"
        );
    }

    #[test]
    fn inherited_field_binding_work_depends_on_references_not_owner_inventory() {
        let compile = |unused_fields: usize| {
            let mut source = "/datum/base\n\tvar/used = 4\n".to_owned();
            for index in 0..unused_fields {
                source.push_str(&format!("\tvar/unused_{index} = {index}\n"));
            }
            source.push_str("/datum/base/child/proc/read()\n\treturn used\n");
            let compilation = TestProject::compile(&source);
            let registry = ProcedureRegistry::build(&compilation);
            let read = procedure_by_path(&registry, "/datum/base/child/proc/read")
                .effective_target
                .expect("read implementation");
            let executable = registry
                .compile_vm_implementations(&compilation, [read])
                .expect("inherited field should link");
            (
                executable.stats().inherited_field_name_lookups,
                executable.stats().src_field_bindings,
            )
        };

        assert_eq!(compile(2), compile(200));
    }

    #[test]
    fn indexed_static_fields_preserve_inheritance_shadowing_and_build_once() {
        let compilation = TestProject::compile(
            "/datum/base\n\tvar/static/shared = 1\n\tproc/read_base()\n\t\treturn shared\n/datum/child\n\tparent_type = /datum/base\n\tvar/static/shared = 2\n\tproc/read_child()\n\t\treturn shared\n/datum/reader/proc/read_receiver(datum/base/value)\n\treturn value.shared\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/read_base");
        let child = procedure_by_path(&registry, "/datum/child/proc/read_child");
        let receiver = procedure_by_path(&registry, "/datum/reader/proc/read_receiver");
        let targets = [base, child, receiver]
            .into_iter()
            .map(|procedure| procedure.effective_target.expect("procedure body"));
        let executable = registry
            .compile_vm_implementations(&compilation, targets)
            .expect("inherited and receiver statics should compile");

        assert_eq!(executable.stats().static_registry_builds, 1);
        assert_eq!(
            executable.stats().global_field_bindings,
            5,
            "typed slash-parameter receiver contributes its qualified static binding",
        );

        let variables = VariableRegistry::build(&compilation);
        let direct = direct_static_fields(&variables);
        let mut cache = BTreeMap::new();
        let child_node = compilation
            .code_tree()
            .find(&dm_syntax::DefinitionPath::new(vec![
                "datum".to_owned(),
                "child".to_owned(),
            ]))
            .expect("child type");
        let inherited =
            inherited_static_fields(&compilation, Some(child_node), &direct, &mut cache);
        assert_eq!(
            inherited.get("shared"),
            Some(&dm_value::FieldName::static_storage(
                "/datum/child/var/shared"
            )),
            "child static must shadow its inherited static"
        );
    }

    #[test]
    fn owner_static_binds_in_new_assignment_and_increment() {
        let compilation = TestProject::compile(
            "/datum/conversation\n\tvar/static/uid = 0\n\tvar/id\n/datum/conversation/New()\n\tid = uid\n\tuid++\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let variables = VariableRegistry::build(&compilation);
        let direct = direct_static_fields(&variables);
        assert!(
            direct.values().any(|fields| fields.contains_key("uid")),
            "registry entries: {:?}",
            variables.entries()
        );
        let new = procedure_by_path(&registry, "/datum/conversation/proc/New");
        let mut cache = BTreeMap::new();
        let inherited = inherited_static_fields(&compilation, new.owner_type, &direct, &mut cache);
        assert!(
            inherited.contains_key("uid"),
            "owner={:?} inherited={inherited:?}",
            new.owner_type
        );
        registry
            .compile_vm_implementations(&compilation, [new.effective_target.expect("New body")])
            .expect("owner static should bind as a qualified global slot");
    }

    #[test]
    fn world_profile_override_parent_call_reaches_engine_native() {
        let compilation =
            TestProject::compile("/world/Profile(command, type, format)\n\treturn ..()\n");
        let registry = ProcedureRegistry::build(&compilation);
        let profile = procedure_by_path(&registry, "/world/proc/Profile");
        let target = profile.effective_target.expect("profile override");
        let executable = registry
            .compile_vm_implementations(&compilation, [target])
            .expect("native parent should link");
        let entry = executable
            .implementation(target)
            .expect("override is linked");
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());

        let result = execute_module_in_context(
            executable.module(),
            entry,
            &[Value::number(2.0)],
            &mut state,
            &ExecutionContext::new(Value::Datum(world), Value::Null),
        )
        .expect("native profile call should execute");
        let Value::List(profile) = result else {
            panic!("non-JSON profile data should be a list");
        };
        assert_eq!(state.heap().list(profile).unwrap().len(), 6);
    }

    #[test]
    fn headless_byond_membership_query_is_callable_and_false() {
        let compilation = TestProject::compile(
            "/client/proc/check_member()\n\treturn IsByondMember() || src.IsByondMember()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/client/proc/check_member");
        let target = procedure.effective_target.expect("procedure body");
        let executable = registry
            .compile_vm_implementations(&compilation, [target])
            .expect("the engine membership query should link in headless mode");
        let entry = executable
            .implementation(target)
            .expect("check_member linked");
        let mut state = ExecutionState::new();
        state.set_type_parents(
            [
                (TypePath::parse("/datum").unwrap(), None),
                (
                    TypePath::parse("/client").unwrap(),
                    Some(TypePath::parse("/datum").unwrap()),
                ),
            ]
            .into(),
        );
        let client = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/client").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(client), Value::Null),
            )
            .expect("membership query should execute"),
            Value::number(0.0),
        );
    }

    #[test]
    fn world_config_and_open_port_overrides_reach_engine_natives() {
        let compilation = TestProject::compile(concat!(
            "/world/SetConfig(config_set, param, value)\n\treturn ..()\n",
            "/world/GetConfig(config_set, param)\n\treturn ..()\n",
            "/world/OpenPort(port)\n\treturn ..()\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let targets = [
            "/world/proc/SetConfig",
            "/world/proc/GetConfig",
            "/world/proc/OpenPort",
        ]
        .map(|path| procedure_by_path(&registry, path).effective_target.unwrap());
        let executable = registry
            .compile_vm_implementations(&compilation, targets)
            .expect("world native parents should link");
        let mut state = ExecutionState::new();
        let world = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/world").unwrap());
        let context = ExecutionContext::new(Value::Datum(world), Value::Null);
        let execute_target = |target, arguments: &[Value], state: &mut ExecutionState| {
            execute_module_in_context(
                executable.module(),
                executable.implementation(target).unwrap(),
                arguments,
                state,
                &context,
            )
            .unwrap()
        };
        assert_eq!(
            execute_target(
                targets[0],
                &[
                    Value::text("env"),
                    Value::text("DREAM64_TEST"),
                    Value::text("set")
                ],
                &mut state,
            ),
            Value::Null
        );
        assert_eq!(
            execute_target(
                targets[1],
                &[Value::text("env"), Value::text("DREAM64_TEST")],
                &mut state,
            ),
            Value::text("set")
        );
        assert_eq!(
            execute_target(targets[2], &[Value::number(4321.0)], &mut state),
            Value::number(1.0)
        );
        assert_eq!(
            state
                .heap()
                .datum_field(world, &dm_value::FieldName::parse("port").unwrap())
                .unwrap(),
            &Value::number(4321.0)
        );
    }

    #[test]
    fn lifecycle_bodies_bind_implicit_type_fields_on_every_datum_receiver() {
        let compilation = TestProject::compile(
            "/obj/example/proc/read_type()\n\treturn type\n/obj/example/proc/read_parent()\n\treturn parent_type\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let read_type = procedure_by_path(&registry, "/obj/example/proc/read_type");
        let read_parent = procedure_by_path(&registry, "/obj/example/proc/read_parent");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                [
                    read_type.effective_target.unwrap(),
                    read_parent.effective_target.unwrap(),
                ],
            )
            .expect("implicit datum type fields should compile");
        assert_eq!(executable.stats().src_field_bindings, 2);
    }

    #[test]
    fn typed_atom_fields_are_inherited_by_obj_lifecycle_bodies() {
        let compilation = TestProject::compile(
            "/datum/reagents\n/atom\n\tvar/datum/reagents/reagents = null\n/obj/item/example/proc/Initialize()\n\treturn reagents\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let initialize = procedure_by_path(&registry, "/obj/item/example/proc/Initialize");
        let executable = registry
            .compile_vm_implementations(&compilation, [initialize.effective_target.unwrap()])
            .expect("typed /atom fields should be inherited by /obj procedures");
        assert_eq!(executable.stats().src_field_bindings, 1);
    }

    #[test]
    fn interpolated_text_retains_inherited_fields_with_nested_quoted_arguments() {
        let compilation = TestProject::compile(
            r#"/datum/reagents
	var/reagent_list
/atom
	var/datum/reagents/reagents = null
/proc/pretty(value, join_text)
	return value
/obj/item/example/proc/Initialize()
	return "contents: [pretty(reagents.reagent_list, join_text = ", ")]"
"#,
        );
        let registry = ProcedureRegistry::build(&compilation);
        let initialize = procedure_by_path(&registry, "/obj/item/example/proc/Initialize");
        registry
            .compile_vm_implementations(
                &compilation,
                registry
                    .implementation_closure(&compilation, [initialize.effective_target.unwrap()]),
            )
            .expect("interpolated inherited receiver fields should compile");
    }

    #[test]
    fn selected_static_call_uses_object_tree_ancestor_not_lexical_path_ancestor() {
        let compilation = TestProject::compile(
            "/datum/proc/RegisterSignals()\n\treturn 42\n/area/centcom/proc/Initialize()\n\treturn RegisterSignals()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/area/centcom/proc/Initialize");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("a call inherited from /datum should be linked into the selected module");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        assert_eq!(
            execute_module(executable.module(), entry, &[]),
            Ok(Value::number(42.0))
        );
    }

    #[test]
    fn selected_method_includes_and_resolves_direct_helper_calls() {
        let compilation = TestProject::compile(
            "/datum/receiver\n\tproc/entry()\n\t\treturn helper()\n\tproc/helper()\n\t\treturn 9\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let entry = procedure_by_path(&registry, "/datum/receiver/proc/entry");
        let executable = registry
            .compile_vm_implementations(
                &compilation,
                entry
                    .implementations
                    .iter()
                    .map(|implementation| implementation.id),
            )
            .expect("direct helper method should be included");
        let entry = executable
            .implementation(entry.effective_target.expect("entry has a body"))
            .expect("entry should be linked");
        let mut state = ExecutionState::new();
        let receiver = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/datum/receiver").unwrap());
        assert_eq!(
            execute_module_in_context(
                executable.module(),
                entry,
                &[],
                &mut state,
                &ExecutionContext::new(Value::Datum(receiver), Value::Null),
            ),
            Ok(Value::number(9.0))
        );
    }

    #[test]
    fn owner_bare_call_resolves_callable_verb_like_byond_proc_dispatch() {
        let compilation = TestProject::compile(
            "/mob/living/proc/say()\n\treturn succumb()\n/mob/living/verb/succumb()\n\treturn 5\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let say = procedure_by_path(&registry, "/mob/living/proc/say")
            .effective_target
            .unwrap();
        let succumb = procedure_by_path(&registry, "/mob/living/verb/succumb")
            .effective_target
            .unwrap();
        let closure = registry.implementation_closure(&compilation, [say]);
        assert!(closure.contains(&succumb));
        registry
            .compile_vm_implementations(&compilation, [say])
            .expect("verbs are callable by bare owner method name");
    }

    #[test]
    fn executes_an_inherited_override_and_reuses_omitted_arguments() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run(value = 2)\n\t\treturn value + 1\n/datum/base/child\n\trun(value = 2)\n\t\treturn ..() + 10\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/datum/base/child/proc/run", &[]),
            Ok(Value::number(13.0))
        );
        assert_eq!(
            execute_effective(&compilation, "/datum/base/child/proc/run", &[Value::Null]),
            Ok(Value::number(13.0)),
            "BYOND applies the declared default to an explicitly null parameter before forwarding it to the parent",
        );
    }

    #[test]
    fn executes_multiple_reopenings_through_the_exact_parent_chain() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n/datum/base\n\trun()\n\t\treturn ..() + 1\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/datum/base/proc/run", &[]),
            Ok(Value::number(3.0))
        );
    }

    #[test]
    fn executes_parent_target_from_explicit_parent_type_and_explicit_arguments() {
        let compilation = TestProject::compile(
            "/datum/alternate\n\tproc/run(value = 1)\n\t\treturn value * 2\n/custom\n\tparent_type = /datum/alternate\n\trun(value = 3)\n\t\treturn ..(value + 1)\n",
        );

        assert_eq!(
            execute_effective(&compilation, "/custom/proc/run", &[]),
            Ok(Value::number(8.0))
        );
    }

    #[test]
    fn terminal_new_and_del_parent_calls_resolve_to_engine_hooks() {
        let compilation = TestProject::compile(
            "/datum/species\n\tNew()\n\t\treturn ..()\n\tDel()\n\t\treturn ..()\n",
        );
        assert_eq!(
            execute_effective(&compilation, "/datum/species/proc/New", &[]),
            Ok(Value::Null),
            "a subtype constructor may terminate at BYOND's engine /datum/New",
        );
        assert_eq!(
            execute_effective(&compilation, "/datum/species/proc/Del", &[]),
            Ok(Value::Null),
            "a subtype destructor may terminate at BYOND's engine /datum/Del",
        );
    }

    #[test]
    fn movable_bump_parent_chain_terminates_at_the_engine_native() {
        let compilation = TestProject::compile(concat!(
            "/atom/movable/Bump(atom/obstacle)\n",
            "\t. = ..()\n",
            "\treturn isnull(.) * 7\n",
            "/obj/crate/Bump(atom/obstacle)\n",
            "\treturn ..() + 1\n",
        ));

        assert_eq!(
            execute_effective(&compilation, "/atom/movable/proc/Bump", &[Value::Null]),
            Ok(Value::number(7.0)),
            "the project base override must observe BYOND's null terminal result",
        );
        assert_eq!(
            execute_effective(&compilation, "/obj/crate/proc/Bump", &[Value::Null]),
            Ok(Value::number(8.0)),
            "a descendant override must traverse the project base before the native terminal",
        );
    }

    #[test]
    fn descendant_movable_bump_can_reach_the_native_without_a_source_base() {
        let compilation =
            TestProject::compile("/obj/crate/Bump(atom/obstacle)\n\treturn isnull(..())\n");

        assert_eq!(
            execute_effective(&compilation, "/obj/crate/proc/Bump", &[Value::Null]),
            Ok(Value::number(1.0)),
        );
    }

    #[test]
    fn unrelated_bump_name_does_not_bind_to_the_movable_native() {
        let compilation = TestProject::compile("/datum/example/Bump()\n\treturn ..()\n");
        let error = execute_effective(&compilation, "/datum/example/proc/Bump", &[])
            .expect_err("a datum procedure named Bump has no engine movable parent");

        assert_eq!(
            error.message,
            "parent procedure call has no resolved target"
        );
    }

    #[test]
    fn engine_generator_icon_and_walk_surfaces_lower_eagerly_and_execute() {
        let compilation = TestProject::compile(concat!(
            "/proc/Rand()\n\treturn 99\n",
            "/generator/proc/RandList()\n\treturn Rand()\n",
            "/icon/proc/Opaque(background = \"#000000\")\n",
            "\tSwapColor(null, background)\n",
            "\treturn src\n",
            "/proc/generator_result()\n",
            "\tvar/generator/value = generator(\"num\", 4, 4)\n",
            "\treturn value.RandList()\n",
            "/proc/icon_result()\n",
            "\tvar/icon/value = icon()\n",
            "\treturn value.Opaque()\n",
            "/proc/_walk(ref, dir, lag)\n\twalk(ref, dir, lag)\n",
            "/proc/_walk_towards(ref, target, lag)\n\twalk_towards(ref, target, lag)\n",
            "/proc/_walk_to(ref, target, minimum, lag)\n",
            "\twalk_to(ref, target, minimum, lag)\n",
            "/proc/_walk_away(ref, target, maximum, lag)\n",
            "\twalk_away(ref, target, maximum, lag)\n",
            "/proc/_walk_rand(ref, lag)\n\twalk_rand(ref, lag)\n",
        ));
        let registry = ProcedureRegistry::build(&compilation);
        let generator_target = procedure_by_path(&registry, "/proc/generator_result")
            .effective_target
            .unwrap();
        let icon_target = procedure_by_path(&registry, "/proc/icon_result")
            .effective_target
            .unwrap();
        let executable = registry
            .compile_vm_all_symbolic_deferred(&compilation)
            .expect("engine surface fixture should link")
            .into_fully_eager()
            .expect("all Monk-shaped engine calls should lower eagerly");
        assert_eq!(executable.module().deferred_procedure_count(), 0);

        let mut state = ExecutionState::new();
        assert_eq!(
            execute_module_in_state(
                executable.module(),
                executable.implementation(generator_target).unwrap(),
                &[],
                &mut state,
            ),
            Ok(Value::number(4.0)),
            "the engine-owned generator member must win over a same-name global proc",
        );
        let Value::Datum(icon) = execute_module_in_state(
            executable.module(),
            executable.implementation(icon_target).unwrap(),
            &[],
            &mut state,
        )
        .expect("the native icon member should execute") else {
            panic!("Opaque should return its icon receiver")
        };
        let operations_field = FieldName::parse("_dream64_icon_operations").unwrap();
        let Value::List(operations) = state
            .heap()
            .datum_field(icon, &operations_field)
            .expect("SwapColor should record an icon operation")
        else {
            panic!("icon operations should be stored in a list")
        };
        let [(_, Value::List(operation))] = state
            .heap()
            .list(*operations)
            .unwrap()
            .positions()
            .collect::<Vec<_>>()
            .as_slice()
        else {
            panic!("Opaque should perform exactly one icon operation")
        };
        assert_eq!(
            state.heap().list(*operation).unwrap().get(1),
            Ok(&Value::text("SwapColor")),
        );
    }

    #[test]
    fn project_generator_member_overrides_engine_rand_in_full_and_independent_modules() {
        let compilation = TestProject::compile(concat!(
            "/generator/Rand()\n\treturn 99\n",
            "/generator/proc/RandList()\n\treturn Rand()\n",
            "/proc/run()\n",
            "\tvar/generator/value = generator(\"num\", 4, 4)\n",
            "\treturn value.RandList()\n",
        ));
        assert_eq!(
            execute_effective(&compilation, "/proc/run", &[]),
            Ok(Value::number(99.0)),
        );

        let registry = ProcedureRegistry::build(&compilation);
        let rand_list = procedure_by_path(&registry, "/generator/proc/RandList")
            .effective_target
            .unwrap();
        let mut independently = registry.compile_vm_bodies_independently(&compilation, [rand_list]);
        let (compiled_id, result) = independently
            .pop()
            .expect("independent RandList body should be present");
        assert_eq!(compiled_id, rand_list);
        result.expect("a project member override should remain a valid independent call target");
    }

    #[test]
    fn missing_parent_target_is_a_source_mapped_runtime_error() {
        let compilation = TestProject::compile("/proc/orphan()\n\treturn ..()\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/proc/orphan");
        let implementation = procedure.implementations[0];
        assert_eq!(implementation.parent_target, None);
        let expected_span = compilation
            .syntax(implementation.file_id)
            .expect("source syntax should exist")
            .definitions[implementation.definition_index]
            .body[0]
            .span;
        let error = execute_effective(&compilation, "/proc/orphan", &[])
            .expect_err("orphan parent call should fail at runtime");

        assert_eq!(
            error.message,
            "parent procedure call has no resolved target"
        );
        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 1);
    }

    #[test]
    fn engine_owned_topic_and_click_methods_supply_terminal_parent_targets() {
        let compilation = TestProject::compile(
            "/datum/Topic(href, list/href_list)\n\treturn ..()\n/client/Click(object, location, control, params)\n\treturn ..()\n/proc/run()\n\tvar/datum/target = new\n\tvar/client/user = new\n\treturn isnull(target.Topic(\"x\", list())) + isnull(user.Click(null, null, null, null))\n",
        );
        assert_eq!(
            execute_effective(&compilation, "/proc/run", &[]),
            Ok(Value::number(2.0)),
        );
    }

    #[test]
    fn engine_owned_client_click_dispatches_the_addressed_atom() {
        let compilation = TestProject::compile(
            "var/global/clicked = 0\n/atom/Click(location, control, params)\n\tclicked = (control == \"map\" && params == \"left=1\")\n\treturn 7\n/client/Click(object, location, control, params)\n\treturn ..()\n/proc/run()\n\tvar/atom/target = new\n\tvar/client/user = new\n\treturn user.Click(target, null, \"map\", \"left=1\") + clicked * 10\n",
        );
        assert_eq!(
            execute_effective(&compilation, "/proc/run", &[]),
            Ok(Value::number(17.0)),
        );
    }

    #[test]
    fn unqualified_istype_uses_typed_src_field_declarations() {
        let compilation = TestProject::compile(
            "/obj/item\n/obj/item/space\n/obj/item/explorer\n/datum/holder\n\tvar/obj/item/space/suit\n\tproc/check()\n\t\treturn istype(suit)\n/proc/run()\n\tvar/datum/holder/holder = new\n\tholder.suit = new /obj/item/explorer\n\tvar/incompatible = holder.check()\n\tholder.suit = new /obj/item/space\n\treturn incompatible * 10 + holder.check()\n",
        );
        assert_eq!(
            execute_effective(&compilation, "/proc/run", &[]),
            Ok(Value::number(1.0)),
        );
    }

    #[test]
    fn parent_failure_preserves_both_source_mapped_frames() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn \"text\" + 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/run");
        let base_implementation = base.implementations[0];
        let expected_span = compilation
            .syntax(base_implementation.file_id)
            .expect("source syntax should exist")
            .definitions[base_implementation.definition_index]
            .body[0]
            .span;
        let error = execute_effective(&compilation, "/datum/base/child/proc/run", &[])
            .expect_err("parent numeric failure should propagate");

        assert_eq!(error.source_span, Some(expected_span));
        assert_eq!(error.call_stack.len(), 2);
        assert!(
            error.call_stack[0]
                .procedure
                .contains("/datum/base/child/proc/run")
        );
        assert!(
            error.call_stack[1]
                .procedure
                .contains("/datum/base/proc/run")
        );
        assert_eq!(error.call_stack[1].source_span, Some(expected_span));
    }
}
