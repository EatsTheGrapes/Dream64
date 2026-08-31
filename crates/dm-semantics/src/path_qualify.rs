//! Header/body token rewrites applied before VM lowering: resolving
//! upward-search path expressions against the object tree, expanding the
//! `__PROC__` pseudo-macro, and qualifying BYOND's destination-typed
//! contextual `new` so the bytecode compiler sees an explicit type path.

use std::collections::BTreeMap;

use dm_compiler::Compilation;
use dm_lexer::TokenKind;
use dm_object_tree::{CodePath, NodeId};
use dm_value::TypePath;

use super::{declared_type_path, inferred_assignment_type, parameter_declaration_name};

pub(crate) fn normalize_upward_paths(
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
pub(crate) fn expand_proc_pseudo_macro(definition: &mut dm_syntax::Definition, path: &CodePath) {
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

pub(crate) fn top_level_simple_assignment(tokens: &[dm_lexer::SpannedToken]) -> Option<usize> {
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

pub(crate) fn type_node_from_tokens(
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
