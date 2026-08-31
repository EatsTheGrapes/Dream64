//! Scalar (num/text/null) type inference over DM expression token streams:
//! ScalarType/ScalarConstraint, the `as num|text|null` annotation reader,
//! effective scalar/datum return resolution across override chains, and the
//! recursive proven-scalar-type / composite-inference walk plus its small
//! expression-shape token helpers.

use std::collections::{BTreeMap, BTreeSet};

use dm_compiler::Compilation;
use dm_lexer::TokenKind;
use dm_object_tree::NodeId;

use super::{
    ConstBindings, ProcedureImplementationId, ProcedureRegistry, procedure_return_type_node,
    proven_datum_expression_type,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ScalarType {
    Number,
    Text,
    Null,
    Dynamic,
}

pub(crate) fn expression_is_proven_truthy(tokens: &[dm_lexer::SpannedToken]) -> bool {
    matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Identifier(name)) if name == "new")
        || matches!(tokens, [token] if matches!(&token.kind, TokenKind::Number(value) if value != "0" && value != "0.0"))
        || matches!(tokens, [token] if matches!(&token.kind, TokenKind::String(value) | TokenKind::RawString(value) if !value.is_empty()))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScalarConstraint {
    pub(crate) kind: ScalarType,
    pub(crate) allows_null: bool,
}

impl ScalarConstraint {
    pub(crate) const fn exact(kind: ScalarType) -> Self {
        Self {
            kind,
            allows_null: matches!(kind, ScalarType::Null),
        }
    }
}

pub(crate) fn scalar_constraint(tokens: &[dm_lexer::SpannedToken]) -> Option<ScalarConstraint> {
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

pub(crate) fn procedure_scalar_return(
    tokens: &[dm_lexer::SpannedToken],
) -> Option<ScalarConstraint> {
    let closing = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Punctuation(')'))?;
    scalar_constraint(&tokens[closing + 1..])
}

pub(crate) fn effective_scalar_return(
    compilation: &Compilation,
    node: NodeId,
) -> Option<ScalarConstraint> {
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

pub(crate) fn effective_datum_return(compilation: &Compilation, node: NodeId) -> Option<NodeId> {
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

pub(crate) fn statically_called_procedure(
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

pub(crate) fn proven_scalar_type(
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
pub(crate) fn infer_scalar_composite(
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

pub(crate) fn receiver_member_expression(
    tokens: &[dm_lexer::SpannedToken],
) -> Option<(&str, &str, bool)> {
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

pub(crate) fn parenthesized_receiver_method(
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

pub(crate) fn condition_is_known_truthy(
    tokens: &[dm_lexer::SpannedToken],
    known_truthy: &BTreeSet<String>,
) -> bool {
    matches!(tokens, [token] if matches!(&token.kind, TokenKind::Identifier(name) if known_truthy.contains(name)))
}

pub(crate) fn proven_receiver_type(
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

pub(crate) fn find_member_node(
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

pub(crate) fn proven_literal_scalar_type(tokens: &[dm_lexer::SpannedToken]) -> Option<ScalarType> {
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

pub(crate) fn matching_closing(
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

pub(crate) fn top_level_ternary(tokens: &[dm_lexer::SpannedToken]) -> Option<(usize, usize)> {
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

pub(crate) fn top_level_binary(tokens: &[dm_lexer::SpannedToken]) -> Option<(usize, &str)> {
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
pub(crate) fn inline_list_index_scalar(
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
