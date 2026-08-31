//! Non-call semantic dependency analysis for a procedure body: the datum
//! types constructed via `new` / `newlist`, procedure paths referenced through
//! `PROC_REF` / `typesof(.../proc)`, and the exact/virtual member-call targets
//! reachable through statically proven receiver expressions.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_lexer::TokenKind;
use dm_object_tree::{CodePath, NodeId};
use dm_value::TypePath;

use super::{
    Procedure, ProcedureId, ProcedureImplementationId, collect_text_member_call_selectors,
    declared_receiver_types, effective_datum_return, effective_target, find_member_node,
    inherited_declared_field_type, matching_closing, top_level_simple_assignment,
    top_level_ternary, type_node_from_tokens,
};

pub(crate) fn dynamic_call_literal_selectors(
    definition: &dm_syntax::Definition,
) -> BTreeSet<String> {
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
pub(crate) struct ConstructionDependencies {
    pub(crate) targets: BTreeSet<ProcedureImplementationId>,
    pub(crate) unbounded: bool,
}

pub(crate) fn constructor_targets_by_ancestor(
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

pub(crate) fn construction_dependencies(
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

pub(crate) fn static_proc_reference_paths(
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

pub(crate) fn static_procedure_type_families(
    definition: &dm_syntax::Definition,
) -> BTreeSet<Vec<String>> {
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

pub(crate) fn member_call_dependencies(
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

pub(crate) fn type_is_descendant_or_same(
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
