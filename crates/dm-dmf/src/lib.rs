//! Loss-aware parsing for BYOND-compatible Dream Maker interface (`.dmf`) files.
//!
//! The initial parser recognizes `window`, `menu`, and `macro` sections, their
//! `elem` children, and ordered key/value properties. It preserves raw values,
//! decoded quoted values, comments, and byte-accurate source spans while
//! recovering from malformed lines to produce actionable diagnostics.
//!
//! This stage does not yet interpret control-specific property schemas, resolve
//! references between controls/windows/menus/macros, expand preprocessing
//! directives, or join multiline/continued property values. Those concerns are
//! deliberately left for validation and lowering passes once reference fixtures
//! establish their exact compatibility behavior.

#![cfg_attr(not(test), deny(missing_docs))]

use dm_core::SourceSpan;

/// A parsed DMF document in deterministic source order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Document {
    /// Top-level windows, menus, and macro sets in encounter order.
    pub sections: Vec<Section>,
    /// Standalone comments in encounter order.
    pub comments: Vec<Comment>,
    /// Recoverable parser diagnostics in encounter order.
    pub diagnostics: Vec<Diagnostic>,
    /// Original source length in bytes.
    pub source_len: usize,
}

/// A typed top-level DMF definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Section {
    /// A client window containing controls.
    Window(Window),
    /// A menu containing menu entries.
    Menu(Menu),
    /// A macro set containing key bindings.
    MacroSet(MacroSet),
}

impl Section {
    /// Returns the section's identifier.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Window(section) => &section.name,
            Self::Menu(section) => &section.name,
            Self::MacroSet(section) => &section.name,
        }
    }

    /// Returns the complete byte range occupied by the section.
    #[must_use]
    pub const fn span(&self) -> SourceSpan {
        match self {
            Self::Window(section) => section.span,
            Self::Menu(section) => section.span,
            Self::MacroSet(section) => section.span,
        }
    }
}

/// A DMF window and its controls.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Window {
    /// Window identifier from the quoted header.
    pub name: String,
    /// Complete source range of the window block.
    pub span: SourceSpan,
    /// Controls in source order.
    pub controls: Vec<Control>,
}

/// A control belonging to a [`Window`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Control {
    /// Optional control identifier; anonymous `elem` entries are valid DMF.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// A DMF menu and its entries.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Menu {
    /// Menu identifier from the quoted header.
    pub name: String,
    /// Complete source range of the menu block.
    pub span: SourceSpan,
    /// Entries in source order.
    pub entries: Vec<MenuEntry>,
}

/// One element in a [`Menu`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MenuEntry {
    /// Optional entry identifier; menu separators and actions may be anonymous.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// A named group of client keyboard macros.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MacroSet {
    /// Macro-set identifier from the quoted header.
    pub name: String,
    /// Complete source range of the macro block.
    pub span: SourceSpan,
    /// Macro bindings in source order.
    pub macros: Vec<Macro>,
}

/// One keyboard binding in a [`MacroSet`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Macro {
    /// Optional binding identifier.
    pub id: Option<String>,
    /// Complete source range of the element block.
    pub span: SourceSpan,
    /// Properties in source order.
    pub properties: Vec<Property>,
}

/// One ordered `key = value` assignment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Property {
    /// Property name exactly as spelled, excluding surrounding whitespace.
    pub key: String,
    /// Complete source range of the property line.
    pub span: SourceSpan,
    /// Byte range containing the property name.
    pub key_span: SourceSpan,
    /// Loss-aware property value.
    pub value: PropertyValue,
}

/// A property value retaining both source spelling and decoded content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PropertyValue {
    /// Exact trimmed source spelling, including quotes when present.
    pub raw: String,
    /// Content with surrounding quotes removed and basic escapes decoded.
    pub decoded: String,
    /// Lexical representation used by the value.
    pub kind: ValueKind,
    /// Byte range containing the trimmed raw value.
    pub span: SourceSpan,
}

/// Lexical form of a DMF property value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueKind {
    /// An unquoted token or compound value such as `10,20` or `640x480`.
    Bare,
    /// A double-quoted text value.
    Quoted,
    /// A single-quoted resource path.
    Resource,
}

/// A retained full-line DMF comment.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Comment {
    /// Exact comment spelling excluding indentation.
    pub raw: String,
    /// Byte range of the comment text.
    pub span: SourceSpan,
}

/// Impact of a parser diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticSeverity {
    /// Parsing recovered, but the source is ambiguous or suspicious.
    Warning,
    /// The line could not be represented according to the supported grammar.
    Error,
}

/// Machine-readable parser diagnostic category.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticKind {
    /// A section or element header was malformed.
    MalformedHeader,
    /// An element appeared before any window, menu, or macro section.
    ElementOutsideSection,
    /// A property appeared before any element in its section.
    PropertyOutsideElement,
    /// A property name or assignment was malformed.
    MalformedProperty,
    /// A quoted name or value did not terminate on its physical line.
    UnterminatedQuote,
    /// Non-comment source followed an otherwise complete quoted value.
    TrailingCharacters,
    /// The line is not part of the supported DMF grammar.
    UnknownStatement,
    /// A property name repeats within one element.
    DuplicateProperty,
    /// A section or element identifier repeats in the same scope.
    DuplicateIdentifier,
}

/// An actionable, byte-located parser problem.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Machine-readable category.
    pub kind: DiagnosticKind,
    /// Whether the problem invalidates the line.
    pub severity: DiagnosticSeverity,
    /// Human-readable explanation.
    pub message: String,
    /// Relevant source byte range.
    pub span: SourceSpan,
}

/// Parses a DMF source document while retaining recoverable source information.
#[must_use]
pub fn parse(source: &str) -> Document {
    Parser::new(source).run()
}

struct Parser<'source> {
    source: &'source str,
    document: Document,
    current_section: Option<usize>,
}

impl<'source> Parser<'source> {
    fn new(source: &'source str) -> Self {
        Self {
            source,
            document: Document {
                source_len: source.len(),
                ..Document::default()
            },
            current_section: None,
        }
    }

    fn run(mut self) -> Document {
        let mut start = 0usize;
        while start < self.source.len() {
            let newline = self.source[start..]
                .find('\n')
                .map_or(self.source.len(), |offset| start + offset);
            let content_end = newline
                .checked_sub(1)
                .filter(|index| self.source.as_bytes()[*index] == b'\r')
                .unwrap_or(newline);
            self.parse_line(start, content_end);
            start = if newline < self.source.len() {
                newline + 1
            } else {
                self.source.len()
            };
        }
        self.document
    }

    fn parse_line(&mut self, start: usize, end: usize) {
        let line = &self.source[start..end];
        let indentation = line.len() - line.trim_start_matches([' ', '\t']).len();
        let trimmed = &line[indentation..];
        if trimmed.is_empty() {
            return;
        }
        if is_full_line_comment(trimmed) {
            self.document.comments.push(Comment {
                raw: trimmed.to_owned(),
                span: SourceSpan::new(start + indentation, end),
            });
            return;
        }
        let inline_comment = inline_comment_start(trimmed);
        if let Some(comment_start) = inline_comment {
            self.document.comments.push(Comment {
                raw: trimmed[comment_start..].to_owned(),
                span: SourceSpan::new(absolute_offset(start, indentation, comment_start), end),
            });
        }
        let code_end = inline_comment.unwrap_or(trimmed.len());
        let code = trimmed[..code_end].trim_end();
        if code.is_empty() {
            return;
        }
        let absolute = start + indentation;

        let (keyword, tail) = leading_keyword(code);
        match keyword {
            "window" | "menu" | "macro" if !tail.trim_start().starts_with('=') => {
                self.parse_section(keyword, tail, absolute, end);
                return;
            }
            "elem" => {
                self.parse_element(tail, absolute, end);
                return;
            }
            _ => {}
        }
        if code.contains('=') {
            self.parse_property(code, absolute, end);
            return;
        }
        self.error(
            DiagnosticKind::UnknownStatement,
            "expected window, menu, macro, elem, or key = value",
            SourceSpan::new(absolute, end),
        );
    }

    fn parse_section(&mut self, keyword: &str, tail: &str, start: usize, end: usize) {
        let Some(name) = self.parse_quoted_name(tail, start + keyword.len()) else {
            return;
        };
        if self
            .document
            .sections
            .iter()
            .any(|section| section.name() == name)
        {
            self.warning(
                DiagnosticKind::DuplicateIdentifier,
                format!("duplicate top-level DMF identifier {name:?}"),
                SourceSpan::new(start, end),
            );
        }
        let span = SourceSpan::new(start, end);
        let section = match keyword {
            "window" => Section::Window(Window {
                name,
                span,
                controls: Vec::new(),
            }),
            "menu" => Section::Menu(Menu {
                name,
                span,
                entries: Vec::new(),
            }),
            "macro" => Section::MacroSet(MacroSet {
                name,
                span,
                macros: Vec::new(),
            }),
            _ => unreachable!("caller recognizes section keywords"),
        };
        self.document.sections.push(section);
        self.current_section = Some(self.document.sections.len() - 1);
    }

    fn parse_element(&mut self, tail: &str, start: usize, end: usize) {
        let Some(section_index) = self.current_section else {
            self.error(
                DiagnosticKind::ElementOutsideSection,
                "elem must follow a window, menu, or macro header",
                SourceSpan::new(start, end),
            );
            return;
        };
        let trimmed_tail = tail.trim();
        let id = if trimmed_tail.is_empty() {
            None
        } else {
            let Some(id) = self.parse_quoted_name(tail, start + "elem".len()) else {
                return;
            };
            Some(id)
        };
        if id
            .as_deref()
            .is_some_and(|id| section_has_element(&self.document.sections[section_index], id))
        {
            self.warning(
                DiagnosticKind::DuplicateIdentifier,
                format!(
                    "duplicate elem identifier {:?} in this section",
                    id.as_deref().expect("duplicate lookup requires an id")
                ),
                SourceSpan::new(start, end),
            );
        }
        let span = SourceSpan::new(start, end);
        match &mut self.document.sections[section_index] {
            Section::Window(section) => section.controls.push(Control {
                id,
                span,
                properties: Vec::new(),
            }),
            Section::Menu(section) => section.entries.push(MenuEntry {
                id,
                span,
                properties: Vec::new(),
            }),
            Section::MacroSet(section) => section.macros.push(Macro {
                id,
                span,
                properties: Vec::new(),
            }),
        }
        extend_section_span(&mut self.document.sections[section_index], end);
    }

    fn parse_property(&mut self, code: &str, start: usize, end: usize) {
        let Some(section_index) = self.current_section else {
            self.error(
                DiagnosticKind::PropertyOutsideElement,
                "property must belong to an elem inside a section",
                SourceSpan::new(start, end),
            );
            return;
        };
        if !section_has_elements(&self.document.sections[section_index]) {
            self.error(
                DiagnosticKind::PropertyOutsideElement,
                "property must follow an elem header",
                SourceSpan::new(start, end),
            );
            return;
        }
        let assignment = code.find('=').expect("caller found an assignment");
        let key_source = &code[..assignment];
        let key = key_source.trim();
        let key_offset = key_source.find(key).unwrap_or(0);
        if key.is_empty()
            || !key
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            self.error(
                DiagnosticKind::MalformedProperty,
                "property name may contain only letters, digits, '_', '-', and '.'",
                SourceSpan::new(start, start + assignment),
            );
            return;
        }
        let value_source = &code[assignment + 1..];
        let value_trimmed = value_source.trim();
        let value_offset = assignment + 1 + value_source.find(value_trimmed).unwrap_or(0);
        let value_span = SourceSpan::new(
            start + value_offset,
            start + value_offset + value_trimmed.len(),
        );
        let value = self.parse_value(value_trimmed, value_span);
        let duplicate = last_properties(&self.document.sections[section_index])
            .is_some_and(|properties| properties.iter().any(|property| property.key == key));
        if duplicate {
            self.warning(
                DiagnosticKind::DuplicateProperty,
                format!("property {key:?} repeats within this elem; source order is preserved"),
                SourceSpan::new(start + key_offset, start + key_offset + key.len()),
            );
        }
        let property = Property {
            key: key.to_owned(),
            span: SourceSpan::new(start, end),
            key_span: SourceSpan::new(start + key_offset, start + key_offset + key.len()),
            value,
        };
        last_properties_mut(&mut self.document.sections[section_index])
            .expect("element existence was checked")
            .push(property);
        extend_last_element_span(&mut self.document.sections[section_index], end);
        extend_section_span(&mut self.document.sections[section_index], end);
    }

    fn parse_quoted_name(&mut self, tail: &str, base: usize) -> Option<String> {
        let leading = tail.len() - tail.trim_start_matches([' ', '\t']).len();
        let value = tail[leading..].trim_end();
        let span = SourceSpan::new(base + leading, base + leading + value.len());
        let parsed = quoted_content(value, '"');
        match parsed {
            QuotedResult::Complete { decoded, consumed } if value[consumed..].trim().is_empty() => {
                Some(decoded)
            }
            QuotedResult::Complete { consumed, .. } => {
                self.error(
                    DiagnosticKind::TrailingCharacters,
                    "unexpected source after quoted identifier",
                    SourceSpan::new(span.start + consumed, span.end),
                );
                None
            }
            QuotedResult::Unterminated => {
                self.error(
                    DiagnosticKind::UnterminatedQuote,
                    "unterminated quoted identifier",
                    span,
                );
                None
            }
            QuotedResult::NotQuoted => {
                self.error(
                    DiagnosticKind::MalformedHeader,
                    "DMF section and elem identifiers must be double quoted",
                    span,
                );
                None
            }
        }
    }

    fn parse_value(&mut self, raw: &str, span: SourceSpan) -> PropertyValue {
        let quote = raw
            .chars()
            .next()
            .filter(|character| matches!(character, '"' | '\''));
        let Some(quote) = quote else {
            return PropertyValue {
                raw: raw.to_owned(),
                decoded: raw.to_owned(),
                kind: ValueKind::Bare,
                span,
            };
        };
        let kind = if quote == '"' {
            ValueKind::Quoted
        } else {
            ValueKind::Resource
        };
        match quoted_content(raw, quote) {
            QuotedResult::Complete { decoded, consumed } => {
                if !raw[consumed..].trim().is_empty() {
                    self.error(
                        DiagnosticKind::TrailingCharacters,
                        "unexpected source after quoted property value",
                        SourceSpan::new(span.start + consumed, span.end),
                    );
                }
                PropertyValue {
                    raw: raw.to_owned(),
                    decoded,
                    kind,
                    span,
                }
            }
            QuotedResult::Unterminated => {
                self.error(
                    DiagnosticKind::UnterminatedQuote,
                    "unterminated quoted property value",
                    span,
                );
                PropertyValue {
                    raw: raw.to_owned(),
                    decoded: raw[quote.len_utf8()..].to_owned(),
                    kind,
                    span,
                }
            }
            QuotedResult::NotQuoted => unreachable!("the first character was checked"),
        }
    }

    fn error(&mut self, kind: DiagnosticKind, message: impl Into<String>, span: SourceSpan) {
        self.document.diagnostics.push(Diagnostic {
            kind,
            severity: DiagnosticSeverity::Error,
            message: message.into(),
            span,
        });
    }

    fn warning(&mut self, kind: DiagnosticKind, message: impl Into<String>, span: SourceSpan) {
        self.document.diagnostics.push(Diagnostic {
            kind,
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            span,
        });
    }
}

enum QuotedResult {
    Complete { decoded: String, consumed: usize },
    Unterminated,
    NotQuoted,
}

fn quoted_content(source: &str, quote: char) -> QuotedResult {
    if !source.starts_with(quote) {
        return QuotedResult::NotQuoted;
    }
    let mut decoded = String::new();
    let mut escaped = false;
    for (offset, character) in source[quote.len_utf8()..].char_indices() {
        if escaped {
            match character {
                'n' => decoded.push('\n'),
                'r' => decoded.push('\r'),
                't' => decoded.push('\t'),
                '\\' => decoded.push('\\'),
                other if other == quote => decoded.push(other),
                other => {
                    decoded.push('\\');
                    decoded.push(other);
                }
            }
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        if character == quote {
            return QuotedResult::Complete {
                decoded,
                consumed: quote.len_utf8() + offset + character.len_utf8(),
            };
        }
        decoded.push(character);
    }
    QuotedResult::Unterminated
}

fn leading_keyword(source: &str) -> (&str, &str) {
    let end = source.find(char::is_whitespace).unwrap_or(source.len());
    (&source[..end], &source[end..])
}

fn is_full_line_comment(source: &str) -> bool {
    source.starts_with("//") || source.starts_with('#') || source.starts_with(';')
}

fn inline_comment_start(source: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut quote = None;
    let mut escaped = false;
    let mut index = 0usize;
    while index + 1 < bytes.len() {
        let byte = bytes[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if quote.is_some() && byte == b'\\' {
            escaped = true;
            index += 1;
            continue;
        }
        if quote == Some(byte) {
            quote = None;
            index += 1;
            continue;
        }
        if quote.is_none() && matches!(byte, b'"' | b'\'') {
            quote = Some(byte);
            index += 1;
            continue;
        }
        if quote.is_none()
            && byte == b'/'
            && bytes[index + 1] == b'/'
            && index
                .checked_sub(1)
                .is_none_or(|previous| bytes[previous].is_ascii_whitespace())
        {
            return Some(index);
        }
        index += 1;
    }
    None
}

const fn absolute_offset(line_start: usize, indentation: usize, relative: usize) -> usize {
    line_start + indentation + relative
}

fn section_has_element(section: &Section, id: &str) -> bool {
    match section {
        Section::Window(section) => section
            .controls
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
        Section::Menu(section) => section
            .entries
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
        Section::MacroSet(section) => section
            .macros
            .iter()
            .any(|element| element.id.as_deref() == Some(id)),
    }
}

fn section_has_elements(section: &Section) -> bool {
    match section {
        Section::Window(section) => !section.controls.is_empty(),
        Section::Menu(section) => !section.entries.is_empty(),
        Section::MacroSet(section) => !section.macros.is_empty(),
    }
}

fn last_properties(section: &Section) -> Option<&[Property]> {
    match section {
        Section::Window(section) => section
            .controls
            .last()
            .map(|element| element.properties.as_slice()),
        Section::Menu(section) => section
            .entries
            .last()
            .map(|element| element.properties.as_slice()),
        Section::MacroSet(section) => section
            .macros
            .last()
            .map(|element| element.properties.as_slice()),
    }
}

fn last_properties_mut(section: &mut Section) -> Option<&mut Vec<Property>> {
    match section {
        Section::Window(section) => section
            .controls
            .last_mut()
            .map(|element| &mut element.properties),
        Section::Menu(section) => section
            .entries
            .last_mut()
            .map(|element| &mut element.properties),
        Section::MacroSet(section) => section
            .macros
            .last_mut()
            .map(|element| &mut element.properties),
    }
}

fn extend_section_span(section: &mut Section, end: usize) {
    match section {
        Section::Window(section) => section.span.end = end,
        Section::Menu(section) => section.span.end = end,
        Section::MacroSet(section) => section.span.end = end,
    }
}

fn extend_last_element_span(section: &mut Section, end: usize) {
    match section {
        Section::Window(section) => {
            section
                .controls
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
        Section::Menu(section) => {
            section
                .entries
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
        Section::MacroSet(section) => {
            section
                .macros
                .last_mut()
                .expect("caller checked element existence")
                .span
                .end = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DiagnosticKind, DiagnosticSeverity, Section, ValueKind, parse};

    #[test]
    fn parses_synthetic_monk_compatibility_shapes() {
        let source = include_str!("../fixtures/compatibility.dmf");
        let document = parse(source);

        assert!(document.diagnostics.is_empty());
        assert_eq!(document.sections.len(), 3);
        assert_eq!(document.comments.len(), 1);
        let Section::MacroSet(macros) = &document.sections[0] else {
            panic!("first section should be a macro set");
        };
        let Section::Menu(menu) = &document.sections[1] else {
            panic!("second section should be a menu");
        };
        let Section::Window(window) = &document.sections[2] else {
            panic!("third section should be a window");
        };
        assert_eq!(macros.macros.len(), 1);
        assert_eq!(menu.entries.len(), 2);
        assert_eq!(menu.entries[0].id, None);
        assert_eq!(window.controls.len(), 2);
        let icon = window.controls[1]
            .properties
            .iter()
            .find(|property| property.key == "icon")
            .expect("resource property should exist");
        assert_eq!(icon.value.kind, ValueKind::Resource);
        assert_eq!(icon.value.decoded, "icons\\ui\\app.png");
        let status = window.controls[1]
            .properties
            .iter()
            .find(|property| property.key == "on-status")
            .expect("quoted command should exist");
        assert_eq!(status.value.kind, ValueKind::Quoted);
        assert!(status.value.decoded.contains("\"status.text="));
        assert_eq!(document.source_len, source.len());
    }

    #[test]
    fn retains_property_order_raw_values_and_byte_spans() {
        let source = "window \"w\"\r\n\telem \"map\"\r\n\t\ttype = MAP\r\n\t\tsize = 640x480\r\n";
        let document = parse(source);
        let Section::Window(window) = &document.sections[0] else {
            panic!("section should be a window");
        };
        let properties = &window.controls[0].properties;

        assert_eq!(properties[0].key, "type");
        assert_eq!(properties[0].value.raw, "MAP");
        assert_eq!(properties[1].key, "size");
        assert_eq!(properties[1].value.raw, "640x480");
        for property in properties {
            assert_eq!(
                &source[property.key_span.start..property.key_span.end],
                property.key
            );
            assert_eq!(
                &source[property.value.span.start..property.value.span.end],
                property.value.raw
            );
        }
    }

    #[test]
    fn retains_comments_without_treating_colors_as_comments() {
        let source = "# full-line comment\nwindow \"w\"\n\telem \"label\"\n\t\ttext-color = #ffffff // trailing comment\n";
        let document = parse(source);
        let Section::Window(window) = &document.sections[0] else {
            panic!("section should be a window");
        };

        assert_eq!(document.comments[0].raw, "# full-line comment");
        assert_eq!(document.comments[1].raw, "// trailing comment");
        assert_eq!(window.controls[0].properties[0].value.raw, "#ffffff");
        assert!(document.diagnostics.is_empty());
    }

    #[test]
    fn reports_and_recovers_from_actionable_errors() {
        let source = "elem \"orphan\"\nwindow missing-quotes\nwindow \"w\"\n\ttype = MAIN\n\telem \"c\"\n\t\tname = \"unterminated\n\t\ttype = MAP\n\t\ttype = BROWSER\nunknown syntax\n";
        let document = parse(source);
        let kinds: Vec<_> = document
            .diagnostics
            .iter()
            .map(|diagnostic| diagnostic.kind)
            .collect();

        assert!(kinds.contains(&DiagnosticKind::ElementOutsideSection));
        assert!(kinds.contains(&DiagnosticKind::MalformedHeader));
        assert!(kinds.contains(&DiagnosticKind::PropertyOutsideElement));
        assert!(kinds.contains(&DiagnosticKind::UnterminatedQuote));
        assert!(kinds.contains(&DiagnosticKind::DuplicateProperty));
        assert!(kinds.contains(&DiagnosticKind::UnknownStatement));
        assert!(document.diagnostics.iter().all(|diagnostic| {
            diagnostic.span.end <= source.len() && !diagnostic.message.is_empty()
        }));
        assert!(document.diagnostics.iter().any(|diagnostic| {
            diagnostic.kind == DiagnosticKind::DuplicateProperty
                && diagnostic.severity == DiagnosticSeverity::Warning
        }));
    }
}
