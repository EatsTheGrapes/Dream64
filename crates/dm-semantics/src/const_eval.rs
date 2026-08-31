//! Compile-time constant evaluation: the `ConstBindings` index of `const`
//! globals/fields and their declared/scalar types, `validate_const_assignments`
//! (the per-body const-write and declared-type checker), and the
//! destination-type inference that drives contextual `new` qualification.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_globals::VariableRegistry;
use dm_lexer::TokenKind;
use dm_object_tree::{CodePath, NodeId};
use dm_value::TypePath;

use super::{
    ProcedureImplementationId, ProcedureRegistry, ScalarConstraint, ScalarType,
    assigned_receiver_field, declared_type_node, declared_type_path, expression_is_proven_truthy,
    grouped_local_declaration_names, is_assignment_operator, is_known_declared_type,
    parameter_declaration_name, procedure_return_type_path, proven_datum_expression_type,
    proven_literal_scalar_type, proven_scalar_type, scalar_constraint,
    validate_declared_type_exists, validate_override_return_signature, validate_scalar_assignment,
    validate_type_assignment,
};

pub(crate) fn inferred_assignment_type(
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
pub(crate) struct ConstBindings {
    globals: BTreeSet<String>,
    fields: BTreeMap<NodeId, BTreeSet<String>>,
    field_types: BTreeMap<NodeId, BTreeMap<String, NodeId>>,
    scalar_field_types: BTreeMap<NodeId, ScalarConstraint>,
    scalar_field_conflicts: Vec<String>,
    unresolved_type_conflicts: Vec<String>,
}

impl ConstBindings {
    pub(crate) fn build(compilation: &Compilation) -> Self {
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

    pub(crate) fn field_is_const(
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

    pub(crate) fn field_type(
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

    pub(crate) fn effective_scalar_field(
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

pub(crate) fn validate_const_assignments(
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
