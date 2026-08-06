//! Loss-aware parsing for Dream Maker text maps (`.dmm`).

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::BTreeMap;
use std::fmt;

use dm_core::SourceSpan;

/// A parsed DMM map with key definitions and coordinate blocks in source order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Map {
    /// Uniform byte width of map keys.
    pub key_width: usize,
    /// Canonical key table.
    pub keys: BTreeMap<String, KeyDefinition>,
    /// Coordinate blocks in source order.
    pub blocks: Vec<MapBlock>,
}

/// One quoted map key and the atom initializers it expands to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyDefinition {
    /// Quoted key text.
    pub key: String,
    /// Atom initializers in creation order.
    pub atoms: Vec<AtomInitializer>,
    /// Complete source span of the definition.
    pub span: SourceSpan,
}

/// One type path plus an optional raw map-variable initializer block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtomInitializer {
    /// Absolute DM type path.
    pub path: String,
    /// Raw text between `{` and `}`, excluding the braces.
    pub variables: Option<String>,
    /// Source-ordered assignments parsed from the variable block.
    pub variable_assignments: Vec<MapVariableAssignment>,
    /// Complete source span of this initializer.
    pub span: SourceSpan,
}

/// One lossless, source-mapped map variable assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapVariableAssignment {
    /// Assignment target exactly as written, excluding surrounding whitespace.
    pub name: String,
    /// Source span of [`Self::name`].
    pub name_span: SourceSpan,
    /// Unevaluated value syntax.
    pub value: MapValue,
    /// Complete assignment text, excluding surrounding whitespace and `;`.
    pub raw: String,
    /// Complete source span corresponding to [`Self::raw`].
    pub span: SourceSpan,
}

/// One losslessly retained map value expression.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapValue {
    /// Conservative syntactic shape of the value.
    pub kind: MapValueKind,
    /// Value text exactly as written, excluding surrounding whitespace.
    pub raw: String,
    /// Source span corresponding to [`Self::raw`].
    pub span: SourceSpan,
}

/// Conservative syntax categories for map values.
///
/// These categories do not resolve paths, unescape strings, expand lists, or
/// otherwise evaluate DM expressions.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MapValueKind {
    /// A complete double-quoted text literal.
    Text,
    /// A complete single-quoted resource literal.
    Resource,
    /// A `list(...)` or `newlist(...)` expression.
    List,
    /// An absolute DM path expression.
    Path,
    /// A numeric literal spelling.
    Number,
    /// The literal `null`.
    Null,
    /// One bare identifier, such as `NORTH` or `TRUE`.
    Identifier,
    /// Any value requiring later DM expression parsing or evaluation.
    Expression,
}

impl MapValueKind {
    /// Returns a stable spelling for diagnostics and corpus reports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Resource => "resource",
            Self::List => "list",
            Self::Path => "path",
            Self::Number => "number",
            Self::Null => "null",
            Self::Identifier => "identifier",
            Self::Expression => "expression",
        }
    }
}

/// One coordinate payload beginning at an explicit world coordinate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MapBlock {
    /// One-based X coordinate.
    pub x: i32,
    /// One-based Y coordinate.
    pub y: i32,
    /// One-based Z coordinate.
    pub z: i32,
    /// Rows split into map keys.
    pub rows: Vec<Vec<String>>,
    /// Complete source span of the block.
    pub span: SourceSpan,
}

/// A source-mapped DMM syntax or structural error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    /// Human-readable explanation.
    pub message: String,
    /// Relevant byte range.
    pub span: SourceSpan,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} at bytes {}..{}",
            self.message, self.span.start, self.span.end
        )
    }
}

impl std::error::Error for ParseError {}

/// Parses a Dream Maker text map.
///
/// The parser accepts canonical DMM and TGM-style multiline coordinate blocks,
/// retains atom variable blocks losslessly, and validates that every grid key
/// has a definition.
///
/// # Errors
///
/// Returns [`ParseError`] for malformed delimiters, inconsistent key widths,
/// duplicate keys, invalid coordinates, or references to unknown keys.
pub fn parse(source: &str) -> Result<Map, ParseError> {
    Parser::new(source).parse()
}

struct Parser<'source> {
    source: &'source str,
    offset: usize,
}

impl<'source> Parser<'source> {
    const fn new(source: &'source str) -> Self {
        Self { source, offset: 0 }
    }

    fn parse(mut self) -> Result<Map, ParseError> {
        let mut keys = BTreeMap::new();
        let mut blocks = Vec::new();
        let mut key_width = None;
        loop {
            self.skip_space_and_comments();
            if self.offset == self.source.len() {
                break;
            }
            match self.peek() {
                Some(b'"') => {
                    let definition = self.parse_key()?;
                    if definition.key.is_empty() {
                        return Err(Self::error_at(definition.span, "map key cannot be empty"));
                    }
                    if let Some(width) = key_width {
                        if definition.key.len() != width {
                            return Err(Self::error_at(
                                definition.span,
                                "map keys do not have a uniform byte width",
                            ));
                        }
                    } else {
                        key_width = Some(definition.key.len());
                    }
                    if keys.insert(definition.key.clone(), definition).is_some() {
                        return Err(self.error_here("duplicate map key"));
                    }
                }
                Some(b'(') => blocks.push(self.parse_block()?),
                _ => return Err(self.error_here("expected a key definition or coordinate block")),
            }
        }
        let key_width =
            key_width.ok_or_else(|| self.error_here("map contains no key definitions"))?;
        for block in &mut blocks {
            for row in &mut block.rows {
                let raw = row.pop().expect("unvalidated rows contain raw payload");
                if raw.len() % key_width != 0 {
                    return Err(Self::error_at(
                        block.span,
                        "map row length is not divisible by the key width",
                    ));
                }
                for start in (0..raw.len()).step_by(key_width) {
                    if !raw.is_char_boundary(start) || !raw.is_char_boundary(start + key_width) {
                        return Err(Self::error_at(
                            block.span,
                            "map key splits a UTF-8 character",
                        ));
                    }
                    let key = raw[start..start + key_width].to_owned();
                    if !keys.contains_key(&key) {
                        return Err(Self::error_at(
                            block.span,
                            format!("coordinate block references unknown key {key:?}"),
                        ));
                    }
                    row.push(key);
                }
            }
        }
        Ok(Map {
            key_width,
            keys,
            blocks,
        })
    }

    fn parse_key(&mut self) -> Result<KeyDefinition, ParseError> {
        let start = self.offset;
        let key = self.quoted_string()?;
        self.skip_space_and_comments();
        self.expect(b'=')?;
        self.skip_space_and_comments();
        self.expect(b'(')?;
        let mut atoms = Vec::new();
        loop {
            self.skip_space_and_comments();
            if self.consume(b')') {
                break;
            }
            atoms.push(self.parse_atom()?);
            self.skip_space_and_comments();
            if self.consume(b',') {
                continue;
            }
            self.expect(b')')?;
            break;
        }
        Ok(KeyDefinition {
            key,
            atoms,
            span: SourceSpan::new(start, self.offset),
        })
    }

    fn parse_atom(&mut self) -> Result<AtomInitializer, ParseError> {
        let start = self.offset;
        if self.peek() != Some(b'/') {
            return Err(self.error_here("atom initializer must begin with an absolute type path"));
        }
        while matches!(self.peek(), Some(byte) if !byte.is_ascii_whitespace() && !matches!(byte, b'{' | b',' | b')'))
        {
            self.offset += 1;
        }
        let path = self.source[start..self.offset].to_owned();
        self.skip_space_and_comments();
        let (variables, variable_assignments) = if self.consume(b'{') {
            let body_start = self.offset;
            let end = self.scan_balanced_brace()?;
            (
                Some(self.source[body_start..end].to_owned()),
                parse_variable_assignments(self.source, body_start, end)?,
            )
        } else {
            (None, Vec::new())
        };
        Ok(AtomInitializer {
            path,
            variables,
            variable_assignments,
            span: SourceSpan::new(start, self.offset),
        })
    }

    fn scan_balanced_brace(&mut self) -> Result<usize, ParseError> {
        let mut depth = 1usize;
        let mut quote = None;
        let mut escaped = false;
        let mut line_comment = false;
        let mut block_comment = false;
        while let Some(byte) = self.peek() {
            self.offset += 1;
            if line_comment {
                if byte == b'\n' {
                    line_comment = false;
                }
                continue;
            }
            if block_comment {
                if byte == b'*' && self.peek() == Some(b'/') {
                    self.offset += 1;
                    block_comment = false;
                }
                continue;
            }
            if let Some(delimiter) = quote {
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == delimiter {
                    quote = None;
                }
                continue;
            }
            if byte == b'/' && self.peek() == Some(b'/') {
                self.offset += 1;
                line_comment = true;
                continue;
            }
            if byte == b'/' && self.peek() == Some(b'*') {
                self.offset += 1;
                block_comment = true;
                continue;
            }
            match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(self.offset - 1);
                    }
                }
                _ => {}
            }
        }
        Err(self.error_here("unterminated atom variable block"))
    }

    fn parse_block(&mut self) -> Result<MapBlock, ParseError> {
        let start = self.offset;
        self.expect(b'(')?;
        let x = self.integer()?;
        self.expect(b',')?;
        let y = self.integer()?;
        self.expect(b',')?;
        let z = self.integer()?;
        self.expect(b')')?;
        self.skip_space_and_comments();
        self.expect(b'=')?;
        self.skip_horizontal_space();
        self.expect(b'{')?;
        let payload = self.quoted_string()?;
        self.expect(b'}')?;
        let rows = payload
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| vec![line.trim_end_matches('\r').to_owned()])
            .collect();
        Ok(MapBlock {
            x,
            y,
            z,
            rows,
            span: SourceSpan::new(start, self.offset),
        })
    }

    fn integer(&mut self) -> Result<i32, ParseError> {
        self.skip_horizontal_space();
        let start = self.offset;
        if self.peek() == Some(b'-') {
            self.offset += 1;
        }
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.offset += 1;
        }
        self.source[start..self.offset]
            .parse()
            .map_err(|_| Self::error_at(SourceSpan::new(start, self.offset), "invalid coordinate"))
    }

    fn quoted_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut value = String::new();
        let mut segment_start = self.offset;
        while let Some(byte) = self.peek() {
            if byte == b'"' {
                value.push_str(&self.source[segment_start..self.offset]);
                self.offset += 1;
                return Ok(value);
            }
            if byte != b'\\' {
                let character = self.remaining().chars().next().expect("source remains");
                self.offset += character.len_utf8();
                continue;
            }
            value.push_str(&self.source[segment_start..self.offset]);
            self.offset += 1;
            let escaped = self
                .remaining()
                .chars()
                .next()
                .ok_or_else(|| self.error_here("unterminated quoted-string escape"))?;
            self.offset += escaped.len_utf8();
            value.push(match escaped {
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
            segment_start = self.offset;
        }
        Err(self.error_here("unterminated quoted string"))
    }

    fn skip_space_and_comments(&mut self) {
        loop {
            while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
                self.offset += 1;
            }
            if self.remaining().starts_with("//") {
                while self.peek().is_some_and(|byte| byte != b'\n') {
                    self.offset += 1;
                }
            } else {
                break;
            }
        }
    }

    fn skip_horizontal_space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.offset += 1;
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), ParseError> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(self.error_here(format!("expected {:?}", char::from(expected))))
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.source.as_bytes().get(self.offset).copied()
    }

    fn remaining(&self) -> &str {
        &self.source[self.offset..]
    }

    fn error_here(&self, message: impl Into<String>) -> ParseError {
        Self::error_at(SourceSpan::new(self.offset, self.offset), message)
    }

    fn error_at(span: SourceSpan, message: impl Into<String>) -> ParseError {
        ParseError {
            message: message.into(),
            span,
        }
    }
}

fn parse_variable_assignments(
    source: &str,
    body_start: usize,
    body_end: usize,
) -> Result<Vec<MapVariableAssignment>, ParseError> {
    let mut assignments = Vec::new();
    let mut segment_start = body_start;
    let mut offset = body_start;
    let mut delimiters = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_comment = false;

    while offset < body_end {
        let byte = source.as_bytes()[offset];
        if line_comment {
            offset += 1;
            if byte == b'\n' {
                line_comment = false;
            }
            continue;
        }
        if block_comment {
            if byte == b'*' && source.as_bytes().get(offset + 1) == Some(&b'/') {
                offset += 2;
                block_comment = false;
            } else {
                offset += 1;
            }
            continue;
        }
        if let Some(delimiter) = quote {
            offset += 1;
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                quote = None;
            }
            continue;
        }
        if byte == b'/' && source.as_bytes().get(offset + 1) == Some(&b'/') {
            line_comment = true;
            offset += 2;
            continue;
        }
        if byte == b'/' && source.as_bytes().get(offset + 1) == Some(&b'*') {
            block_comment = true;
            offset += 2;
            continue;
        }
        match byte {
            b'"' | b'\'' => {
                quote = Some(byte);
                offset += 1;
            }
            b'(' | b'[' | b'{' => {
                delimiters.push((byte, offset));
                offset += 1;
            }
            b')' | b']' | b'}' => {
                let Some((opening, _)) = delimiters.pop() else {
                    return Err(ParseError {
                        message: format!("unexpected {:?} in map variable value", char::from(byte)),
                        span: SourceSpan::new(offset, offset + 1),
                    });
                };
                if !delimiters_match(opening, byte) {
                    return Err(ParseError {
                        message: format!("mismatched {:?} in map variable value", char::from(byte)),
                        span: SourceSpan::new(offset, offset + 1),
                    });
                }
                offset += 1;
            }
            b';' if delimiters.is_empty() => {
                push_variable_assignment(source, segment_start, offset, &mut assignments)?;
                offset += 1;
                segment_start = offset;
            }
            _ => offset += 1,
        }
    }

    finish_variable_assignments(
        source,
        segment_start,
        body_end,
        &delimiters,
        quote,
        block_comment,
        &mut assignments,
    )?;
    Ok(assignments)
}

fn finish_variable_assignments(
    source: &str,
    segment_start: usize,
    body_end: usize,
    delimiters: &[(u8, usize)],
    quote: Option<u8>,
    block_comment: bool,
    assignments: &mut Vec<MapVariableAssignment>,
) -> Result<(), ParseError> {
    if let Some((opening, opening_offset)) = delimiters.last().copied() {
        return Err(ParseError {
            message: format!(
                "unterminated {:?} in map variable value",
                char::from(opening)
            ),
            span: SourceSpan::new(opening_offset, opening_offset + 1),
        });
    }
    if quote.is_some() {
        return Err(ParseError {
            message: "unterminated quoted map variable value".to_owned(),
            span: SourceSpan::new(body_end, body_end),
        });
    }
    if block_comment {
        return Err(ParseError {
            message: "unterminated block comment in map variable value".to_owned(),
            span: SourceSpan::new(body_end, body_end),
        });
    }
    push_variable_assignment(source, segment_start, body_end, assignments)?;
    Ok(())
}

fn push_variable_assignment(
    source: &str,
    start: usize,
    end: usize,
    assignments: &mut Vec<MapVariableAssignment>,
) -> Result<(), ParseError> {
    let Some((start, end)) = trimmed_range(source, start, end) else {
        return Ok(());
    };
    let equals = find_assignment_equals(source, start, end).ok_or_else(|| ParseError {
        message: "map variable assignment is missing '='".to_owned(),
        span: SourceSpan::new(start, end),
    })?;
    let Some((name_start, name_end)) = trimmed_range(source, start, equals) else {
        return Err(ParseError {
            message: "map variable assignment has no target".to_owned(),
            span: SourceSpan::new(start, equals),
        });
    };
    let Some((value_start, value_end)) = trimmed_range(source, equals + 1, end) else {
        return Err(ParseError {
            message: "map variable assignment has no value".to_owned(),
            span: SourceSpan::new(equals + 1, end),
        });
    };
    let raw_value = &source[value_start..value_end];
    assignments.push(MapVariableAssignment {
        name: source[name_start..name_end].to_owned(),
        name_span: SourceSpan::new(name_start, name_end),
        value: MapValue {
            kind: classify_value(raw_value),
            raw: raw_value.to_owned(),
            span: SourceSpan::new(value_start, value_end),
        },
        raw: source[start..end].to_owned(),
        span: SourceSpan::new(start, end),
    });
    Ok(())
}

fn find_assignment_equals(source: &str, start: usize, end: usize) -> Option<usize> {
    let mut offset = start;
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    while offset < end {
        let byte = source.as_bytes()[offset];
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == delimiter {
                quote = None;
            }
        } else {
            match byte {
                b'"' | b'\'' => quote = Some(byte),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = depth.saturating_sub(1),
                b'=' if depth == 0 => return Some(offset),
                _ => {}
            }
        }
        offset += 1;
    }
    None
}

fn trimmed_range(source: &str, mut start: usize, mut end: usize) -> Option<(usize, usize)> {
    while start < end && source.as_bytes()[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && source.as_bytes()[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    (start < end).then_some((start, end))
}

const fn delimiters_match(opening: u8, closing: u8) -> bool {
    matches!(
        (opening, closing),
        (b'(', b')') | (b'[', b']') | (b'{', b'}')
    )
}

fn classify_value(raw: &str) -> MapValueKind {
    if complete_quoted_literal(raw, b'"') {
        MapValueKind::Text
    } else if complete_quoted_literal(raw, b'\'') {
        MapValueKind::Resource
    } else if call_spelling(raw, "list") || call_spelling(raw, "newlist") {
        MapValueKind::List
    } else if raw.starts_with('/') {
        MapValueKind::Path
    } else if raw == "null" {
        MapValueKind::Null
    } else if raw.parse::<f64>().is_ok() {
        MapValueKind::Number
    } else if raw.bytes().enumerate().all(|(index, byte)| {
        byte == b'_' || byte.is_ascii_alphanumeric() && (index != 0 || !byte.is_ascii_digit())
    }) {
        MapValueKind::Identifier
    } else {
        MapValueKind::Expression
    }
}

fn call_spelling(raw: &str, name: &str) -> bool {
    raw.strip_prefix(name)
        .is_some_and(|remainder| remainder.trim_start().starts_with('('))
}

fn complete_quoted_literal(raw: &str, delimiter: u8) -> bool {
    if raw.as_bytes().first() != Some(&delimiter) {
        return false;
    }
    let mut escaped = false;
    for (index, byte) in raw.bytes().enumerate().skip(1) {
        if escaped {
            escaped = false;
        } else if byte == b'\\' {
            escaped = true;
        } else if byte == delimiter {
            return raw[index + 1..].trim().is_empty();
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parses_keys_atoms_variables_and_grid_blocks() {
        let source = "// generated\n\"aa\" = (/obj/item{name = \"tool\"}, /turf/open, /area/test)\n\"bb\" = (/turf/closed, /area/test)\n(1,2,3) = {\"\naabb\nbbaa\n\"}\n";
        let map = parse(source).expect("fixture should parse");
        assert_eq!(map.key_width, 2);
        assert_eq!(map.keys.len(), 2);
        assert_eq!(map.keys["aa"].atoms[0].path, "/obj/item");
        assert_eq!(
            map.keys["aa"].atoms[0].variables.as_deref(),
            Some("name = \"tool\"")
        );
        assert_eq!(
            (map.blocks[0].x, map.blocks[0].y, map.blocks[0].z),
            (1, 2, 3)
        );
        assert_eq!(map.blocks[0].rows[0], ["aa", "bb"]);
        assert_eq!(map.blocks[0].rows[1], ["bb", "aa"]);
    }

    #[test]
    fn rejects_unknown_grid_keys() {
        let error =
            parse("\"a\" = (/turf)\n(1,1,1) = {\"\nb\n\"}\n").expect_err("unknown key should fail");
        assert!(error.message.contains("unknown key"));
    }

    #[test]
    fn preserves_utf8_quoted_keys() {
        let map =
            parse("\"é\" = (/turf)\n(1,1,1) = {\"\né\n\"}\n").expect("Unicode key should parse");
        assert_eq!(map.key_width, 2);
        assert_eq!(map.blocks[0].rows[0], ["é"]);
    }
}
