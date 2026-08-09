//! A loss-aware first-stage lexer for Dream Maker source text.

#![cfg_attr(not(test), deny(missing_docs))]

use dm_core::SourceSpan;

/// A token and its byte range in the original source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpannedToken {
    /// The recognized token.
    pub kind: TokenKind,
    /// The token's byte range.
    pub span: SourceSpan,
}

/// Lexical units needed by the preprocessor and parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TokenKind {
    /// Leading indentation retained without assigning block semantics yet.
    LineStart {
        /// Number of leading tab bytes.
        tabs: usize,
        /// Number of leading space bytes.
        spaces: usize,
    },
    /// A line break. CRLF is represented as one token.
    Newline,
    /// A backslash followed by a line break, joining physical source lines.
    LineContinuation,
    /// A DM identifier or keyword.
    Identifier(String),
    /// A numeric literal retained as source text until semantic conversion.
    Number(String),
    /// A double-quoted string with escapes still present.
    String(String),
    /// A raw `@{"..."}` string, which may span physical lines.
    RawString(String),
    /// A `{"..."}` text block, which may span physical lines.
    TextBlock(String),
    /// A single-quoted resource literal with escapes still present.
    Resource(String),
    /// Punctuation with a fixed textual spelling.
    Punctuation(char),
    /// An operator, preferring the longest recognized spelling.
    Operator(String),
}

/// A lexical error that can be reported without discarding prior tokens.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LexError {
    /// Human-readable description of the problem.
    pub message: String,
    /// Source range that caused the problem.
    pub span: SourceSpan,
}

/// Tokenizes source while preserving indentation and literal spelling.
///
/// This stage intentionally does not decide whether indentation is valid or
/// whether an identifier is a keyword. Those rules belong to later phases and
/// will be validated against reference-compiler fixtures.
///
/// # Errors
///
/// Returns a [`LexError`] for an unterminated comment or quoted literal, or for
/// a source character that has no recognized DM token spelling.
pub fn lex(source: &str) -> Result<Vec<SpannedToken>, LexError> {
    Lexer::new(source).run()
}

struct Lexer<'source> {
    source: &'source str,
    offset: usize,
    at_line_start: bool,
    tokens: Vec<SpannedToken>,
}

impl<'source> Lexer<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            offset: 0,
            at_line_start: true,
            tokens: Vec::new(),
        }
    }

    fn run(mut self) -> Result<Vec<SpannedToken>, LexError> {
        while self.offset < self.source.len() {
            if self.at_line_start {
                self.lex_indentation();
            }
            let Some(current) = self.current_char() else {
                break;
            };
            match current {
                '\r' | '\n' => self.lex_newline(),
                '\\' if self.has_line_continuation() => self.lex_line_continuation(),
                ' ' | '\t' => self.advance_char(),
                '/' if self.remaining().starts_with("//") => self.skip_line_comment(),
                '/' if self.remaining().starts_with("/*") => self.skip_block_comment()?,
                'a'..='z' | 'A'..='Z' | '_' => self.lex_identifier(),
                '\\' => self.lex_escaped_identifier()?,
                '0'..='9' => self.lex_number(),
                '@' if self.remaining().starts_with("@@") => self.lex_at_raw_string()?,
                '@' if self.remaining().starts_with("@(") => self.lex_complex_raw_string()?,
                '@' if self.remaining().starts_with("@{") => self.lex_braced_string(true)?,
                // DM also accepts the compact C#-style spelling used heavily
                // for regular expressions: `@"\\d+"`.  Unlike an ordinary
                // quoted string its backslashes are literal, so it must not
                // go through `scan_quoted_body` (which consumes escapes).
                '@' if self.remaining().starts_with("@\"") => self.lex_at_quoted_raw_string()?,
                // BYOND accepts @'...' as a raw text literal as well. Unlike a
                // plain single-quoted literal this is text, not a resource.
                '@' if self.remaining().starts_with("@'") => {
                    self.lex_at_single_quoted_raw_string()?
                }
                '"' => self.lex_quoted('"', false)?,
                '\'' => self.lex_quoted('\'', true)?,
                '{' if self.remaining().starts_with("{\"") => self.lex_braced_string(false)?,
                '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' => {
                    self.push_char(TokenKind::Punctuation(current));
                }
                _ => self.lex_operator()?,
            }
        }
        Ok(self.tokens)
    }

    fn lex_indentation(&mut self) {
        let start = self.offset;
        let mut tabs = 0;
        let mut spaces = 0;
        while let Some(current) = self.current_char() {
            match current {
                '\t' => tabs += 1,
                ' ' => spaces += 1,
                _ => break,
            }
            self.advance_char();
        }
        self.tokens.push(SpannedToken {
            kind: TokenKind::LineStart { tabs, spaces },
            span: SourceSpan::new(start, self.offset),
        });
        self.at_line_start = false;
    }

    fn lex_newline(&mut self) {
        let start = self.offset;
        if self.remaining().starts_with("\r\n") {
            self.offset += 2;
        } else {
            self.advance_char();
        }
        self.tokens.push(SpannedToken {
            kind: TokenKind::Newline,
            span: SourceSpan::new(start, self.offset),
        });
        self.at_line_start = true;
    }

    fn has_line_continuation(&self) -> bool {
        self.remaining().starts_with("\\\n") || self.remaining().starts_with("\\\r\n")
    }

    fn lex_line_continuation(&mut self) {
        let start = self.offset;
        self.offset += if self.remaining().starts_with("\\\r\n") {
            3
        } else {
            2
        };
        self.tokens.push(SpannedToken {
            kind: TokenKind::LineContinuation,
            span: SourceSpan::new(start, self.offset),
        });
        self.at_line_start = true;
    }

    fn skip_line_comment(&mut self) {
        while !matches!(self.current_char(), None | Some('\r' | '\n')) {
            self.advance_char();
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        let start = self.offset;
        self.offset += 2;
        let mut depth = 1_usize;
        while self.offset < self.source.len() {
            if self.remaining().starts_with("//") {
                self.offset += 2;
                self.skip_line_comment();
                continue;
            }
            if self.remaining().starts_with("/*") {
                self.offset += 2;
                depth += 1;
                continue;
            }
            if self.remaining().starts_with("*/") {
                self.offset += 2;
                depth -= 1;
                if depth != 0 {
                    continue;
                }
                if self.at_line_start {
                    let line_start = self.source[..self.offset]
                        .rfind(['\r', '\n'])
                        .map_or(0, |newline| newline + 1);
                    let mut tabs = 0;
                    let mut spaces = 0;
                    for character in self.source[line_start..].chars() {
                        match character {
                            '\t' => tabs += 1,
                            ' ' => spaces += 1,
                            _ => break,
                        }
                    }
                    self.tokens.push(SpannedToken {
                        kind: TokenKind::LineStart { tabs, spaces },
                        span: SourceSpan::new(line_start, self.offset),
                    });
                    self.at_line_start = false;
                }
                return Ok(());
            }
            if matches!(self.current_char(), Some('\r' | '\n')) {
                self.lex_newline();
            } else {
                self.advance_char();
            }
        }
        Err(LexError {
            message: "unterminated block comment".to_owned(),
            span: SourceSpan::new(start, self.offset),
        })
    }

    fn lex_identifier(&mut self) {
        let start = self.offset;
        let mut spelling = String::new();
        while let Some(character) = self.current_char() {
            if character.is_ascii_alphanumeric() || character == '_' {
                spelling.push(character);
                self.advance_char();
            } else if character == '\\' && !self.has_line_continuation() {
                self.advance_char();
                let Some(escaped) = self.current_char() else {
                    break;
                };
                spelling.push(escaped);
                self.advance_char();
            } else {
                break;
            }
        }
        self.push_range(start, TokenKind::Identifier(spelling));
    }

    fn lex_escaped_identifier(&mut self) -> Result<(), LexError> {
        let start = self.offset;
        self.advance_char();
        let Some(escaped) = self.current_char() else {
            return Err(LexError {
                message: "escaped identifier is missing a character".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        self.advance_char();
        let mut spelling = escaped.to_string();
        while let Some(character) = self.current_char() {
            if character.is_ascii_alphanumeric() || character == '_' {
                spelling.push(character);
                self.advance_char();
            } else if character == '\\' && !self.has_line_continuation() {
                self.advance_char();
                let Some(escaped) = self.current_char() else {
                    break;
                };
                spelling.push(escaped);
                self.advance_char();
            } else {
                break;
            }
        }
        self.push_range(start, TokenKind::Identifier(spelling));
        Ok(())
    }

    fn lex_number(&mut self) {
        let start = self.offset;
        if self.remaining().starts_with("0x") || self.remaining().starts_with("0X") {
            self.offset += 2;
            self.advance_while(|character| character.is_ascii_hexdigit() || character == '_');
            self.push_range(
                start,
                TokenKind::Number(self.source[start..self.offset].to_owned()),
            );
            return;
        }

        self.advance_while(|character| character.is_ascii_digit() || character == '_');
        if self.remaining().starts_with("#INF") || self.remaining().starts_with("#IND") {
            self.offset += 4;
            self.push_range(
                start,
                TokenKind::Number(self.source[start..self.offset].to_owned()),
            );
            return;
        }
        if self.current_char() == Some('.') {
            self.advance_char();
            if self.remaining().starts_with("#INF") || self.remaining().starts_with("#IND") {
                self.offset += 4;
                self.push_range(
                    start,
                    TokenKind::Number(self.source[start..self.offset].to_owned()),
                );
                return;
            }
            self.advance_while(|character| character.is_ascii_digit() || character == '_');
        }
        if matches!(self.current_char(), Some('e' | 'E')) {
            self.advance_char();
            if matches!(self.current_char(), Some('+' | '-')) {
                self.advance_char();
            }
            self.advance_while(|character| character.is_ascii_digit() || character == '_');
        }
        self.push_range(
            start,
            TokenKind::Number(self.source[start..self.offset].to_owned()),
        );
    }

    fn lex_complex_raw_string(&mut self) -> Result<(), LexError> {
        let start = self.offset;
        self.offset += 2;
        let delimiter_start = self.offset;
        let Some(relative_close) = self.remaining().find(')') else {
            self.offset = self.source.len();
            return Err(LexError {
                message: "unterminated raw string delimiter".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let delimiter_end = self.offset + relative_close;
        let delimiter = &self.source[delimiter_start..delimiter_end];
        if delimiter.is_empty() {
            return Err(LexError {
                message: "raw string delimiter cannot be empty".to_owned(),
                span: SourceSpan::new(start, delimiter_end + 1),
            });
        }
        self.offset = delimiter_end + 1;
        if self.remaining().starts_with("\r\n") {
            self.offset += 2;
        } else if matches!(self.current_char(), Some('\r' | '\n')) {
            self.advance_char();
        }
        let content_start = self.offset;
        let Some(relative_end) = self.remaining().find(delimiter) else {
            self.offset = self.source.len();
            return Err(LexError {
                message: "unterminated complex raw string".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let mut content_end = self.offset + relative_end;
        if self.source[..content_end].ends_with("\r\n") {
            content_end -= 2;
        } else if self.source[..content_end].ends_with(['\r', '\n']) {
            content_end -= 1;
        }
        let content = self.source[content_start..content_end].to_owned();
        self.offset += relative_end + delimiter.len();
        self.push_range(start, TokenKind::RawString(content));
        Ok(())
    }

    fn lex_quoted(&mut self, quote: char, resource: bool) -> Result<(), LexError> {
        let start = self.offset;
        self.advance_char();
        let content_start = self.offset;
        self.scan_quoted_body(quote, quote == '"', start)?;
        let content_end = self.offset - quote.len_utf8();
        let content = self.source[content_start..content_end].to_owned();
        let kind = if resource {
            TokenKind::Resource(content)
        } else {
            TokenKind::String(content)
        };
        self.push_range(start, kind);
        Ok(())
    }

    fn lex_braced_string(&mut self, raw: bool) -> Result<(), LexError> {
        let start = self.offset;
        self.offset += usize::from(raw) + 1;
        if self.current_char() != Some('"') {
            return Err(LexError {
                message: "braced string must begin with a quote".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        }
        self.advance_char();
        let content_start = self.offset;
        let Some(relative_end) = self.remaining().find("\"}") else {
            self.offset = self.source.len();
            return Err(LexError {
                message: "unterminated braced string".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let content_end = self.offset + relative_end;
        let content = self.source[content_start..content_end].to_owned();
        self.offset = content_end + 2;
        let kind = if raw {
            TokenKind::RawString(content)
        } else {
            TokenKind::TextBlock(content)
        };
        self.push_range(start, kind);
        Ok(())
    }

    fn lex_at_raw_string(&mut self) -> Result<(), LexError> {
        let start = self.offset;
        self.offset += 2;
        let content_start = self.offset;
        let Some(relative_end) = self.remaining().find('@') else {
            self.offset = self.source.len();
            return Err(LexError {
                message: "unterminated at-delimited raw string".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let content_end = self.offset + relative_end;
        let content = self.source[content_start..content_end].to_owned();
        self.offset = content_end + 1;
        self.push_range(start, TokenKind::RawString(content));
        Ok(())
    }

    fn lex_at_quoted_raw_string(&mut self) -> Result<(), LexError> {
        self.lex_at_quote_raw_string('"')
    }

    fn lex_at_single_quoted_raw_string(&mut self) -> Result<(), LexError> {
        self.lex_at_quote_raw_string('\'')
    }

    fn lex_at_quote_raw_string(&mut self, quote: char) -> Result<(), LexError> {
        let start = self.offset;
        // Skip the at-sign and quote delimiter. Raw strings do not treat a
        // backslash as an escape; the next quote ends the literal.
        self.offset += 2;
        let content_start = self.offset;
        let Some(relative_end) = self.remaining().find(quote) else {
            self.offset = self.source.len();
            return Err(LexError {
                message: "unterminated at-quoted raw string".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let content_end = self.offset + relative_end;
        let content = self.source[content_start..content_end].to_owned();
        self.offset = content_end + 1;
        self.push_range(start, TokenKind::RawString(content));
        Ok(())
    }

    fn scan_quoted_body(
        &mut self,
        quote: char,
        allows_interpolation: bool,
        opening_offset: usize,
    ) -> Result<(), LexError> {
        let mut interpolation_depth = 0_usize;
        while let Some(current) = self.current_char() {
            if current == '\\' {
                self.advance_char();
                self.advance_char();
                continue;
            }
            if current == quote && interpolation_depth == 0 {
                self.advance_char();
                return Ok(());
            }
            if allows_interpolation && current == '[' {
                interpolation_depth += 1;
                self.advance_char();
                continue;
            }
            if interpolation_depth > 0 {
                if current == ']' {
                    interpolation_depth -= 1;
                    self.advance_char();
                    continue;
                }
                if current == '"' {
                    let nested_start = self.offset;
                    self.advance_char();
                    self.scan_quoted_body('"', true, nested_start)?;
                    continue;
                }
            }
            self.advance_char();
        }
        Err(LexError {
            message: "unterminated quoted literal".to_owned(),
            span: SourceSpan::new(opening_offset, self.offset),
        })
    }

    fn lex_operator(&mut self) -> Result<(), LexError> {
        const OPERATORS: &[&str] = &[
            "<=>", "<<=", ">>=", "&&=", "||=", "%%=", "**=", "...", "::", "?.", "?:", "?[", "==",
            "!=", "<>", "<=", ">=", "~!", "<<", ">>", "&&", "||", "++", "--", ":=", "+=", "-=",
            "*=", "/=", "%=", "&=", "|=", "^=", "~=", "%%", "**", "..", "/", ".", ":", "?", "=",
            "+", "-", "*", "%", "<", ">", "!", "~", "&", "|", "^", "#", "@",
        ];
        let Some(operator) = OPERATORS
            .iter()
            .find(|operator| self.remaining().starts_with(**operator))
        else {
            let start = self.offset;
            self.advance_char();
            return Err(LexError {
                message: "unrecognized source character".to_owned(),
                span: SourceSpan::new(start, self.offset),
            });
        };
        let start = self.offset;
        self.offset += operator.len();
        self.push_range(start, TokenKind::Operator((*operator).to_owned()));
        Ok(())
    }

    fn current_char(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn remaining(&self) -> &str {
        &self.source[self.offset..]
    }

    fn advance_char(&mut self) {
        if let Some(current) = self.current_char() {
            self.offset += current.len_utf8();
        }
    }

    fn advance_while(&mut self, predicate: impl Fn(char) -> bool) {
        while self.current_char().is_some_and(&predicate) {
            self.advance_char();
        }
    }

    fn push_char(&mut self, kind: TokenKind) {
        let start = self.offset;
        self.advance_char();
        self.push_range(start, kind);
    }

    fn push_range(&mut self, start: usize, kind: TokenKind) {
        self.tokens.push(SpannedToken {
            kind,
            span: SourceSpan::new(start, self.offset),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{TokenKind, lex};

    #[test]
    fn retains_dm_paths_and_indentation() {
        let tokens = lex("/datum/example\n\tvar/value = 1\n").expect("source should lex");
        let kinds: Vec<_> = tokens.into_iter().map(|token| token.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::LineStart { tabs: 0, spaces: 0 },
                TokenKind::Operator("/".to_owned()),
                TokenKind::Identifier("datum".to_owned()),
                TokenKind::Operator("/".to_owned()),
                TokenKind::Identifier("example".to_owned()),
                TokenKind::Newline,
                TokenKind::LineStart { tabs: 1, spaces: 0 },
                TokenKind::Identifier("var".to_owned()),
                TokenKind::Operator("/".to_owned()),
                TokenKind::Identifier("value".to_owned()),
                TokenKind::Operator("=".to_owned()),
                TokenKind::Number("1".to_owned()),
                TokenKind::Newline,
            ]
        );
    }

    #[test]
    fn rejects_unterminated_literals() {
        let error = lex("value = \"missing").expect_err("literal should fail");

        assert_eq!(error.message, "unterminated quoted literal");
    }

    #[test]
    fn retains_null_conditional_member_and_index_operators() {
        let tokens = lex("value?.field value?:dynamic values?[key]\n")
            .expect("null-conditional operators should lex");
        let operators: Vec<_> = tokens
            .into_iter()
            .filter_map(|token| match token.kind {
                TokenKind::Operator(operator) => Some(operator),
                _ => None,
            })
            .collect();
        assert_eq!(operators, ["?.", "?:", "?["]);
    }

    #[test]
    fn separates_arithmetic_operators_from_numbers() {
        let tokens = lex("result = 1+2e-3\n").expect("source should lex");
        let kinds: Vec<_> = tokens.into_iter().map(|token| token.kind).collect();

        assert!(kinds.windows(3).any(|window| {
            window
                == [
                    TokenKind::Number("1".to_owned()),
                    TokenKind::Operator("+".to_owned()),
                    TokenKind::Number("2e-3".to_owned()),
                ]
        }));
    }

    #[test]
    fn comments_do_not_hide_line_structure() {
        let tokens = lex("// first\n/* second\nthird */\n").expect("comments should lex");
        let newlines = tokens
            .iter()
            .filter(|token| token.kind == TokenKind::Newline)
            .count();

        assert_eq!(newlines, 3);
    }

    #[test]
    fn block_comments_nest_and_line_comments_hide_closers() {
        lex("/* outer /* inner */ outer */\nvalue\n").expect("nested comments should close");

        let error = lex("/* outer\n// */\nvalue\n").expect_err("line comment hides closer");
        assert_eq!(error.message, "unterminated block comment");
        assert_eq!(error.span.start, 0);
    }

    #[test]
    fn retains_preprocessor_line_continuations() {
        let tokens = lex("#define FLAG value \\\r\n\t| other\n").expect("source should lex");

        assert!(
            tokens
                .iter()
                .any(|token| token.kind == TokenKind::LineContinuation)
        );
        assert_eq!(
            tokens
                .iter()
                .filter(|token| token.kind == TokenKind::Newline)
                .count(),
            1
        );
    }

    #[test]
    fn merges_escaped_identifier_components_without_changing_line_splices() {
        let tokens =
            lex("ES\\KE \\+suffix\\-part\nnext\\\nline\n").expect("escaped identifiers should lex");
        let identifiers: Vec<_> = tokens
            .iter()
            .filter_map(|token| match &token.kind {
                TokenKind::Identifier(identifier) => Some(identifier.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(identifiers, ["ESKE", "+suffix-part", "next", "line"]);
        assert!(
            tokens
                .iter()
                .any(|token| token.kind == TokenKind::LineContinuation)
        );
    }

    #[test]
    fn retains_nested_strings_inside_text_expressions() {
        let source = "value = \"outer [format(\"nested [other(\"deep\")]\")] end\"\n";
        let tokens = lex(source).expect("nested text expression should lex");
        let strings: Vec<_> = tokens
            .iter()
            .filter_map(|token| match &token.kind {
                TokenKind::String(content) => Some(content.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(
            strings,
            vec!["outer [format(\"nested [other(\"deep\")]\")] end"]
        );
    }

    #[test]
    fn retains_multiline_raw_strings() {
        let source = "value = @{\"first \\\\n+second \\\"quoted\\\"\"}\n";
        let tokens = lex(source).expect("raw string should lex");

        assert!(tokens.iter().any(|token| {
            matches!(
                &token.kind,
                TokenKind::RawString(content) if content == "first \\\\n+second \\\"quoted\\\""
            )
        }));
    }

    #[test]
    fn retains_interpolated_text_blocks() {
        let source = "value = {\"first\n[second]\n\"}\n";
        let tokens = lex(source).expect("text block should lex");

        assert!(tokens.iter().any(|token| {
            matches!(
                &token.kind,
                TokenKind::TextBlock(content) if content == "first\n[second]\n"
            )
        }));
    }

    #[test]
    fn retains_at_delimited_raw_strings() {
        let source = "value = @@[a-z \\\\n+\"quoted\"]@\n";
        let tokens = lex(source).expect("at-delimited raw string should lex");

        assert!(tokens.iter().any(|token| {
            matches!(
                &token.kind,
                TokenKind::RawString(content) if content == "[a-z \\\\n+\"quoted\"]"
            )
        }));
    }

    #[test]
    fn retains_at_quoted_raw_strings_without_unescaping_backslashes() {
        let source = "value = @\"[\\n\\t]\\\\d+\"\n";
        let tokens = lex(source).expect("at-quoted raw string should lex");

        assert!(tokens.iter().any(|token| {
            matches!(
                &token.kind,
                TokenKind::RawString(content) if content == "[\\n\\t]\\\\d+"
            )
        }));
    }

    #[test]
    fn retains_single_quoted_raw_text_as_text_not_resource() {
        let source = "value = @'^(\\d+)/(.*)$'\n";
        let tokens = lex(source).expect("single-quoted raw text should lex");

        assert!(tokens.iter().any(|token| {
            matches!(
                &token.kind,
                TokenKind::RawString(content) if content == "^(\\d+)/(.*)$"
            )
        }));
        assert!(
            !tokens
                .iter()
                .any(|token| matches!(token.kind, TokenKind::Resource(_)))
        );
    }
}
