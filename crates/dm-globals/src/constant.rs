//! Conservative evaluation of initializer expressions that require no runtime reads.

use std::cmp::Ordering;

use dm_core::{DmNumberBits, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};

/// A value proven constructible without reading runtime state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstantValue {
    /// DM `null`.
    Null,
    /// A compatibility-mode binary32 number.
    Number(DmNumberBits),
    /// Immutable decoded text.
    Text(String),
    /// A canonical absolute type path.
    TypePath(String),
    /// An ordered list construction whose entries are all constant.
    List(Vec<ConstantListEntry>),
}

/// One ordered entry in a constant list construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstantListEntry {
    /// A value appended at the next one-based numeric index.
    Positional(ConstantValue),
    /// A key and value inserted as an association.
    Associative {
        /// Constant association key.
        key: ConstantValue,
        /// Constant association value.
        value: ConstantValue,
    },
}

/// Top-level shape of a successfully evaluated initializer.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ConstantValueShape {
    /// Null value.
    Null,
    /// Binary32 number.
    Number,
    /// Immutable text.
    Text,
    /// Absolute type path.
    TypePath,
    /// Ordered list construction.
    List,
}

impl ConstantValue {
    /// Returns the top-level value shape for deterministic inventory counts.
    #[must_use]
    pub const fn shape(&self) -> ConstantValueShape {
        match self {
            Self::Null => ConstantValueShape::Null,
            Self::Number(_) => ConstantValueShape::Number,
            Self::Text(_) => ConstantValueShape::Text,
            Self::TypePath(_) => ConstantValueShape::TypePath,
            Self::List(_) => ConstantValueShape::List,
        }
    }

    fn truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Number(number) => number.to_f32() != 0.0,
            Self::Text(text) => !text.is_empty(),
            Self::TypePath(_) | Self::List(_) => true,
        }
    }
}

/// Result of attempting conservative constant evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConstantEvaluation {
    /// Every operation and operand was proven constant.
    Value(ConstantValue),
    /// Runtime behavior is required or semantics are not yet proven.
    Unsupported(UnsupportedConstant),
}

/// Precise reason constant evaluation stopped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsupportedConstant {
    /// Stable unsupported syntax or semantic category.
    pub category: UnsupportedCategory,
    /// Expanded-source token range that established the category.
    pub span: SourceSpan,
}

/// Categories deliberately rejected by the conservative evaluator.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum UnsupportedCategory {
    /// No expression followed the assignment operator.
    EmptyExpression,
    /// An identifier would read a constant, variable, or runtime name.
    Identifier,
    /// A procedure or built-in call would execute runtime behavior.
    Call,
    /// A `new` expression would allocate a runtime datum.
    NewExpression,
    /// Text contains interpolation or an unimplemented escape form.
    DynamicText,
    /// A resource literal requires resource-table semantics.
    ResourceLiteral,
    /// An operator does not yet have confirmed constant semantics.
    UnsupportedOperator,
    /// Proven operand types do not support the requested operation.
    TypeMismatch,
    /// Indexing, member access, or another dynamic expression is present.
    DynamicExpression,
    /// Tokens do not form a complete supported expression.
    InvalidSyntax,
    /// A numeric spelling cannot be represented as binary32.
    InvalidNumber,
}

/// Evaluates one complete initializer token sequence conservatively.
#[must_use]
pub fn evaluate_constant(tokens: &[SpannedToken]) -> ConstantEvaluation {
    let Some(first) = tokens.first() else {
        return ConstantEvaluation::Unsupported(UnsupportedConstant {
            category: UnsupportedCategory::EmptyExpression,
            span: SourceSpan::new(0, 0),
        });
    };
    let mut parser = ConstantParser { tokens, index: 0 };
    match parser.parse_binary(1) {
        Ok(value) if parser.index == tokens.len() => ConstantEvaluation::Value(value),
        Ok(_) => {
            let trailing = tokens.get(parser.index).unwrap_or(first);
            ConstantEvaluation::Unsupported(UnsupportedConstant {
                category: match trailing.kind {
                    TokenKind::Operator(_) => UnsupportedCategory::UnsupportedOperator,
                    TokenKind::Punctuation('[') => UnsupportedCategory::DynamicExpression,
                    _ => UnsupportedCategory::InvalidSyntax,
                },
                span: trailing.span,
            })
        }
        Err(unsupported) => ConstantEvaluation::Unsupported(unsupported),
    }
}

struct ConstantParser<'tokens> {
    tokens: &'tokens [SpannedToken],
    index: usize,
}

impl ConstantParser<'_> {
    fn parse_binary(
        &mut self,
        minimum_precedence: u8,
    ) -> Result<ConstantValue, UnsupportedConstant> {
        let mut left = self.parse_unary()?;
        while let Some((operator, span)) = self.current_operator() {
            let Some(precedence) = binary_precedence(operator) else {
                break;
            };
            if precedence < minimum_precedence {
                break;
            }
            let operator = operator.to_owned();
            self.index += 1;
            let right = self.parse_binary(precedence + 1)?;
            left = evaluate_binary(&operator, &left, &right, span)?;
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<ConstantValue, UnsupportedConstant> {
        if let Some((operator @ ("!" | "+" | "-"), span)) = self.current_operator() {
            let operator = operator.to_owned();
            self.index += 1;
            let operand = self.parse_unary()?;
            return evaluate_unary(&operator, &operand, span);
        }
        self.parse_primary()
    }

    #[allow(clippy::too_many_lines)]
    fn parse_primary(&mut self) -> Result<ConstantValue, UnsupportedConstant> {
        let Some(token) = self.tokens.get(self.index) else {
            return Err(UnsupportedConstant {
                category: UnsupportedCategory::InvalidSyntax,
                span: self
                    .tokens
                    .last()
                    .map_or(SourceSpan::new(0, 0), |last| last.span),
            });
        };
        self.index += 1;
        match &token.kind {
            TokenKind::Number(spelling) => parse_number(spelling)
                .map(ConstantValue::Number)
                .map_err(|()| UnsupportedConstant {
                    category: UnsupportedCategory::InvalidNumber,
                    span: token.span,
                }),
            TokenKind::String(text) | TokenKind::TextBlock(text) => decode_text(text)
                .map(ConstantValue::Text)
                .map_err(|()| UnsupportedConstant {
                    category: UnsupportedCategory::DynamicText,
                    span: token.span,
                }),
            TokenKind::RawString(text) => Ok(ConstantValue::Text(text.clone())),
            TokenKind::Resource(_) => Err(UnsupportedConstant {
                category: UnsupportedCategory::ResourceLiteral,
                span: token.span,
            }),
            TokenKind::Identifier(identifier) if identifier == "null" => Ok(ConstantValue::Null),
            TokenKind::Identifier(identifier) if identifier == "TRUE" => Ok(number_value(1.0)),
            TokenKind::Identifier(identifier) if identifier == "FALSE" => Ok(number_value(0.0)),
            TokenKind::Identifier(identifier) if identifier == "new" => Err(UnsupportedConstant {
                category: UnsupportedCategory::NewExpression,
                span: token.span,
            }),
            TokenKind::Identifier(identifier) if self.current_punctuation('(') => {
                if identifier == "list" {
                    self.parse_list()
                } else {
                    Err(UnsupportedConstant {
                        category: UnsupportedCategory::Call,
                        span: token.span,
                    })
                }
            }
            TokenKind::Identifier(_) => Err(UnsupportedConstant {
                category: UnsupportedCategory::Identifier,
                span: token.span,
            }),
            TokenKind::Operator(operator) if operator == "/" => self.parse_type_path(token.span),
            TokenKind::Punctuation('(') => {
                let value = self.parse_binary(1)?;
                if !self.current_punctuation(')') {
                    return Err(UnsupportedConstant {
                        category: UnsupportedCategory::InvalidSyntax,
                        span: self.current_span(token.span),
                    });
                }
                self.index += 1;
                Ok(value)
            }
            TokenKind::Operator(_) => Err(UnsupportedConstant {
                category: UnsupportedCategory::UnsupportedOperator,
                span: token.span,
            }),
            _ => Err(UnsupportedConstant {
                category: UnsupportedCategory::InvalidSyntax,
                span: token.span,
            }),
        }
    }

    fn parse_type_path(
        &mut self,
        opening_span: SourceSpan,
    ) -> Result<ConstantValue, UnsupportedConstant> {
        let mut segments = Vec::new();
        while let Some(SpannedToken {
            kind: TokenKind::Identifier(segment),
            ..
        }) = self.tokens.get(self.index)
        {
            segments.push(segment.clone());
            self.index += 1;
            if !matches!(
                self.tokens.get(self.index).map(|token| &token.kind),
                Some(TokenKind::Operator(operator)) if operator == "/"
            ) || !matches!(
                self.tokens.get(self.index + 1).map(|token| &token.kind),
                Some(TokenKind::Identifier(_))
            ) {
                break;
            }
            self.index += 1;
        }
        if segments.is_empty() {
            return Err(UnsupportedConstant {
                category: UnsupportedCategory::InvalidSyntax,
                span: opening_span,
            });
        }
        Ok(ConstantValue::TypePath(format!("/{}", segments.join("/"))))
    }

    fn parse_list(&mut self) -> Result<ConstantValue, UnsupportedConstant> {
        debug_assert!(self.current_punctuation('('));
        self.index += 1;
        let mut entries = Vec::new();
        while !self.current_punctuation(')') {
            let key_or_value = self.parse_binary(1)?;
            if matches!(self.current_operator(), Some(("=", _))) {
                self.index += 1;
                let value = self.parse_binary(1)?;
                entries.push(ConstantListEntry::Associative {
                    key: key_or_value,
                    value,
                });
            } else {
                entries.push(ConstantListEntry::Positional(key_or_value));
            }
            if self.current_punctuation(',') {
                self.index += 1;
            } else if !self.current_punctuation(')') {
                return Err(UnsupportedConstant {
                    category: UnsupportedCategory::InvalidSyntax,
                    span: self.current_span(self.tokens[self.index - 1].span),
                });
            }
        }
        self.index += 1;
        Ok(ConstantValue::List(entries))
    }

    fn current_operator(&self) -> Option<(&str, SourceSpan)> {
        self.tokens
            .get(self.index)
            .and_then(|token| match &token.kind {
                TokenKind::Operator(operator) => Some((operator.as_str(), token.span)),
                _ => None,
            })
    }

    fn current_punctuation(&self, punctuation: char) -> bool {
        matches!(
            self.tokens.get(self.index).map(|token| &token.kind),
            Some(TokenKind::Punctuation(current)) if *current == punctuation
        )
    }

    fn current_span(&self, fallback: SourceSpan) -> SourceSpan {
        self.tokens
            .get(self.index)
            .map_or(fallback, |token| token.span)
    }
}

fn parse_number(spelling: &str) -> Result<DmNumberBits, ()> {
    let normalized = spelling.replace('_', "");
    let value = if let Some(hexadecimal) = normalized
        .strip_prefix("0x")
        .or_else(|| normalized.strip_prefix("0X"))
    {
        u32::from_str_radix(hexadecimal, 16)
            .map_err(|_| ())?
            .to_string()
            .parse::<f32>()
            .map_err(|_| ())?
    } else {
        normalized.parse::<f32>().map_err(|_| ())?
    };
    Ok(DmNumberBits::from_f32(value))
}

fn decode_text(text: &str) -> Result<String, ()> {
    let mut decoded = String::with_capacity(text.len());
    let mut characters = text.chars();
    while let Some(character) = characters.next() {
        if character == '[' {
            return Err(());
        }
        if character != '\\' {
            decoded.push(character);
            continue;
        }
        let escaped = characters.next().ok_or(())?;
        decoded.push(match escaped {
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            '\\' => '\\',
            '"' => '"',
            '[' => '[',
            ']' => ']',
            _ => return Err(()),
        });
    }
    Ok(decoded)
}

fn evaluate_unary(
    operator: &str,
    operand: &ConstantValue,
    span: SourceSpan,
) -> Result<ConstantValue, UnsupportedConstant> {
    match operator {
        "!" => Ok(number_value(f32::from(!operand.truthy()))),
        "+" => number_operand(operand, span).map(number_value),
        "-" => number_operand(operand, span).map(|number| number_value(-number)),
        _ => Err(UnsupportedConstant {
            category: UnsupportedCategory::UnsupportedOperator,
            span,
        }),
    }
}

fn evaluate_binary(
    operator: &str,
    left: &ConstantValue,
    right: &ConstantValue,
    span: SourceSpan,
) -> Result<ConstantValue, UnsupportedConstant> {
    match operator {
        "&&" => Ok(number_value(f32::from(left.truthy() && right.truthy()))),
        "||" => Ok(number_value(f32::from(left.truthy() || right.truthy()))),
        "==" | "!=" => {
            if matches!(left, ConstantValue::List(_)) || matches!(right, ConstantValue::List(_)) {
                return type_mismatch(span);
            }
            let equal = semantic_equal(left, right);
            Ok(number_value(f32::from(if operator == "==" {
                equal
            } else {
                !equal
            })))
        }
        "+" | "-" | "*" | "/" | "%" | "<" | "<=" | ">" | ">=" => {
            let left = number_operand(left, span)?;
            let right = number_operand(right, span)?;
            let result = match operator {
                "+" => left + right,
                "-" => left - right,
                "*" => left * right,
                "/" => left / right,
                "%" => left % right,
                "<" => f32::from(left < right),
                "<=" => f32::from(left <= right),
                ">" => f32::from(left > right),
                ">=" => f32::from(left >= right),
                _ => unreachable!("operator came from the numeric group"),
            };
            Ok(number_value(result))
        }
        _ => Err(UnsupportedConstant {
            category: UnsupportedCategory::UnsupportedOperator,
            span,
        }),
    }
}

fn number_operand(value: &ConstantValue, span: SourceSpan) -> Result<f32, UnsupportedConstant> {
    if let ConstantValue::Number(number) = value {
        Ok(number.to_f32())
    } else {
        type_mismatch(span)
    }
}

fn type_mismatch<T>(span: SourceSpan) -> Result<T, UnsupportedConstant> {
    Err(UnsupportedConstant {
        category: UnsupportedCategory::TypeMismatch,
        span,
    })
}

fn semantic_equal(left: &ConstantValue, right: &ConstantValue) -> bool {
    match (left, right) {
        (ConstantValue::Null, ConstantValue::Null) => true,
        (ConstantValue::Number(left), ConstantValue::Number(right)) => {
            left.to_f32().partial_cmp(&right.to_f32()) == Some(Ordering::Equal)
        }
        (ConstantValue::Text(left), ConstantValue::Text(right))
        | (ConstantValue::TypePath(left), ConstantValue::TypePath(right)) => left == right,
        _ => false,
    }
}

const fn binary_precedence(operator: &str) -> Option<u8> {
    match operator.as_bytes() {
        b"||" => Some(1),
        b"&&" => Some(2),
        b"==" | b"!=" => Some(3),
        b"<" | b"<=" | b">" | b">=" => Some(4),
        b"+" | b"-" => Some(5),
        b"*" | b"/" | b"%" => Some(6),
        _ => None,
    }
}

fn number_value(value: f32) -> ConstantValue {
    ConstantValue::Number(DmNumberBits::from_f32(value))
}
