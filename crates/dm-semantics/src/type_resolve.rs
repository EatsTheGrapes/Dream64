//! Declared-type resolution: turning DM `var/type/name` and `as`-annotation
//! syntax into object-tree nodes, resolving procedure return-type annotations,
//! proving the datum type of an expression, and building the project-wide
//! global-field / global-type / instance-field-type maps the registry and
//! dependency passes consume.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_lexer::TokenKind;
use dm_object_tree::{NodeId, NodeKind};
use dm_value::{FieldName, TypePath};

use super::{
    ConstBindings, ProcedureImplementationId, ProcedureRegistry, condition_is_known_truthy,
    effective_datum_return, find_member_node, parenthesized_receiver_method, proven_receiver_type,
    receiver_member_expression, statically_called_procedure, top_level_ternary,
};

pub(crate) fn proven_datum_expression_type(
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

pub(crate) fn assigned_receiver_field(tokens: &[dm_lexer::SpannedToken]) -> Option<(&str, &str)> {
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

pub(crate) fn declared_type_node(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
    variable_name: &str,
) -> Option<NodeId> {
    compilation
        .code_tree()
        .find(&declared_type_path(tokens, variable_name)?)
}

pub(crate) fn parameter_declaration_name(tokens: &[dm_lexer::SpannedToken]) -> Option<&str> {
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

pub(crate) fn validate_declared_type_exists(
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

pub(crate) fn is_known_declared_type(
    compilation: &Compilation,
    path: &dm_syntax::DefinitionPath,
) -> bool {
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

pub(crate) fn declared_type_path(
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

pub(crate) fn grouped_local_declaration_names(tokens: &[dm_lexer::SpannedToken]) -> Vec<String> {
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

pub(crate) fn procedure_return_type_node(
    compilation: &Compilation,
    tokens: &[dm_lexer::SpannedToken],
) -> Option<NodeId> {
    compilation
        .code_tree()
        .find(&procedure_return_type_path(tokens)?)
}

pub(crate) fn procedure_return_type_path(
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

pub(crate) fn is_assignment_operator(operator: &str) -> bool {
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
pub(crate) fn declared_global_fields(compilation: &Compilation) -> BTreeMap<String, FieldName> {
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

pub(crate) fn declared_receiver_types(
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

pub(crate) fn declared_global_types(compilation: &Compilation) -> BTreeMap<String, TypePath> {
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

pub(crate) fn declared_field_types(
    compilation: &Compilation,
) -> BTreeMap<NodeId, BTreeMap<String, NodeId>> {
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

pub(crate) fn inherited_declared_field_type(
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
