//! Syntactic call-graph collection: the set of bare and member call selectors
//! a procedure body invokes (including selectors embedded in interpolated
//! strings) and the identifier set used to resolve src/global field bindings.

use std::collections::BTreeSet;

use dm_lexer::TokenKind;

pub(crate) fn static_call_selectors(definition: &dm_syntax::Definition) -> BTreeSet<String> {
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

pub(crate) fn referenced_identifiers(definition: &dm_syntax::Definition) -> BTreeSet<String> {
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

pub(crate) fn collect_text_member_call_selectors(text: &str, selectors: &mut BTreeSet<String>) {
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
