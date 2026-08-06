//! Declaration-oriented syntax structures for Dream Maker source.
//!
//! This stage indexes definitions and retains procedure bodies as source lines.
//! Expression and statement parsing will consume those bodies in a later stage.

#![cfg_attr(not(test), deny(missing_docs))]

use std::fmt;

use dm_core::SourceSpan;
use dm_lexer::{LexError, SpannedToken, TokenKind, lex};

/// A parsed DM source file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SyntaxFile {
    /// Definitions in source order. Parent indices always point backward.
    pub definitions: Vec<Definition>,
}

/// One indexed definition in the DM code tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Definition {
    /// Canonical absolute path, including `proc`, `verb`, or `var` nodes.
    pub path: DefinitionPath,
    /// The semantic category inferred from the declaration header.
    pub kind: DefinitionKind,
    /// Nearest enclosing definition when indentation supplied a relative path.
    pub parent: Option<usize>,
    /// Indentation on the first physical line of the declaration.
    pub indentation: Indentation,
    /// Byte range occupied by the complete logical header.
    pub span: SourceSpan,
    /// Tokens from the declaration header, excluding indentation and newlines.
    pub header: Vec<SpannedToken>,
    /// Procedure parameters retained as token groups for the semantic parser.
    pub parameters: Vec<ParameterSyntax>,
    /// Opaque logical lines belonging to a procedure or verb body.
    pub body: Vec<SourceLine>,
}

/// A canonical path in the DM definition tree.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DefinitionPath {
    segments: Vec<String>,
}

impl DefinitionPath {
    /// Creates a canonical absolute path from non-empty segments.
    ///
    /// # Panics
    ///
    /// Panics when `segments` is empty or contains an empty segment.
    #[must_use]
    pub fn new(segments: Vec<String>) -> Self {
        assert!(!segments.is_empty(), "a definition path cannot be empty");
        assert!(
            segments.iter().all(|segment| !segment.is_empty()),
            "a definition path cannot contain an empty segment"
        );
        Self { segments }
    }

    /// Returns the path segments without the leading slash.
    #[must_use]
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Returns whether this path ends with the supplied segments.
    #[must_use]
    pub fn ends_with(&self, suffix: &[&str]) -> bool {
        self.segments.len() >= suffix.len()
            && self.segments[self.segments.len() - suffix.len()..]
                .iter()
                .map(String::as_str)
                .eq(suffix.iter().copied())
    }
}

impl fmt::Display for DefinitionPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "/{}", self.segments.join("/"))
    }
}

/// The kind of node introduced or changed by a declaration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DefinitionKind {
    /// An object/type node.
    Type,
    /// A procedure declared through an explicit `proc` path node.
    Procedure,
    /// A procedure override written directly below a type path.
    ProcedureOverride,
    /// A verb declared through a `verb` path node.
    Verb,
    /// A variable introduced through an explicit `var` path node.
    Variable,
    /// An inherited variable value overridden below a type path.
    VariableOverride,
}

impl DefinitionKind {
    const fn owns_procedure_body(self) -> bool {
        matches!(self, Self::Procedure | Self::ProcedureOverride | Self::Verb)
    }
}

/// Leading whitespace retained without prematurely enforcing a tab width.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Indentation {
    /// Leading tab bytes.
    pub tabs: usize,
    /// Leading space bytes.
    pub spaces: usize,
}

impl Indentation {
    fn column(self) -> usize {
        self.tabs.saturating_mul(8).saturating_add(self.spaces)
    }

    fn is_deeper_than(self, other: Self) -> bool {
        self.column() > other.column()
    }
}

/// An unparsed procedure parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterSyntax {
    /// Complete token range of the parameter.
    pub span: SourceSpan,
    /// Parameter tokens excluding the separating comma.
    pub tokens: Vec<SpannedToken>,
}

/// A logical source line retained for later statement parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLine {
    /// Indentation from the first physical line.
    pub indentation: Indentation,
    /// Byte range covering the logical line.
    pub span: SourceSpan,
    /// Tokens with indentation and physical newlines removed.
    pub tokens: Vec<SpannedToken>,
}

/// A syntax-layer error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SyntaxError {
    /// The source could not be tokenized.
    Lex(LexError),
    /// A logical line ended with unmatched opening punctuation.
    UnclosedDelimiter(SourceSpan),
}

impl fmt::Display for SyntaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lex(error) => write!(
                formatter,
                "{} at {}..{}",
                error.message, error.span.start, error.span.end
            ),
            Self::UnclosedDelimiter(span) => {
                write!(
                    formatter,
                    "unclosed delimiter at {}..{}",
                    span.start, span.end
                )
            }
        }
    }
}

impl std::error::Error for SyntaxError {}

/// Parses definition headers and builds their indentation hierarchy.
///
/// Procedure statements are deliberately retained as opaque [`SourceLine`]
/// values. This keeps code-tree construction independent from the expression
/// parser while preserving all tokens and source ranges needed by that parser.
///
/// # Errors
///
/// Returns [`SyntaxError::Lex`] when tokenization fails, or
/// [`SyntaxError::UnclosedDelimiter`] when the file ends in a continued logical
/// line.
pub fn parse(source: &str) -> Result<SyntaxFile, SyntaxError> {
    let tokens = lex(source).map_err(SyntaxError::Lex)?;
    let lines = build_logical_lines(tokens)?;
    Ok(Parser::new(lines).run())
}

struct Parser {
    lines: Vec<SourceLine>,
    definitions: Vec<Definition>,
    stack: Vec<Context>,
}

enum Context {
    Definition(usize),
    Namespace {
        indentation: Indentation,
        segments: Vec<String>,
        owner: Option<usize>,
        kind: NamespaceKind,
    },
}

#[derive(Clone, Copy)]
enum NamespaceKind {
    Variable,
    Procedure,
    Verb,
}

impl Parser {
    fn new(lines: Vec<SourceLine>) -> Self {
        Self {
            lines,
            definitions: Vec::new(),
            stack: Vec::new(),
        }
    }

    fn run(mut self) -> SyntaxFile {
        let mut lines = std::mem::take(&mut self.lines).into_iter().peekable();
        while let Some(line) = lines.next() {
            self.pop_finished_definitions(line.indentation);
            if let Some(parent) = self.procedure_body_owner()
                && line
                    .indentation
                    .is_deeper_than(self.definitions[parent].indentation)
            {
                self.definitions[parent].body.push(line);
                continue;
            }

            if let Some((segments, kind)) = self.classify_namespace(&line, lines.peek()) {
                let (base_path, owner) = self.indentation_base();
                let was_absolute = starts_with_slash(&line.tokens);
                let mut path = if was_absolute {
                    Vec::new()
                } else {
                    base_path.unwrap_or_default().to_vec()
                };
                path.extend(segments);
                self.stack.push(Context::Namespace {
                    indentation: line.indentation,
                    segments: path,
                    owner: if was_absolute { None } else { owner },
                    kind,
                });
                continue;
            }

            let Some(mut candidate) = classify_line(&line) else {
                continue;
            };
            let namespace_kind = self.namespace_kind();
            if let Some(kind) = namespace_kind {
                candidate.kind = kind.definition_kind();
            }
            let (base_path, owner) = self.indentation_base();
            let parent = if candidate.was_absolute { None } else { owner };
            let path = canonicalize_path(
                if candidate.was_absolute {
                    None
                } else {
                    base_path
                },
                &candidate,
            );
            let parameters = if candidate.kind.owns_procedure_body() {
                parse_parameters(&line.tokens)
            } else {
                Vec::new()
            };
            let index = self.definitions.len();
            self.definitions.push(Definition {
                path,
                kind: candidate.kind,
                parent,
                indentation: line.indentation,
                span: line.span,
                parameters,
                header: line.tokens,
                body: Vec::new(),
            });
            self.stack.push(Context::Definition(index));
        }

        SyntaxFile {
            definitions: self.definitions,
        }
    }

    fn pop_finished_definitions(&mut self, indentation: Indentation) {
        while let Some(context) = self.stack.last() {
            let context_indentation = match context {
                Context::Definition(index) => self.definitions[*index].indentation,
                Context::Namespace { indentation, .. } => *indentation,
            };
            if indentation.is_deeper_than(context_indentation) {
                break;
            }
            self.stack.pop();
        }
    }

    fn procedure_body_owner(&self) -> Option<usize> {
        match self.stack.last() {
            Some(Context::Definition(index))
                if self.definitions[*index].kind.owns_procedure_body() =>
            {
                Some(*index)
            }
            _ => None,
        }
    }

    fn indentation_base(&self) -> (Option<&[String]>, Option<usize>) {
        match self.stack.last() {
            Some(Context::Definition(index))
                if self.definitions[*index].kind == DefinitionKind::Type =>
            {
                (Some(self.definitions[*index].path.segments()), Some(*index))
            }
            Some(Context::Namespace {
                segments, owner, ..
            }) => (Some(segments), *owner),
            _ => (None, None),
        }
    }

    fn namespace_kind(&self) -> Option<NamespaceKind> {
        match self.stack.last() {
            Some(Context::Namespace { kind, .. }) => Some(*kind),
            _ => None,
        }
    }

    fn classify_namespace(
        &self,
        line: &SourceLine,
        next_line: Option<&SourceLine>,
    ) -> Option<(Vec<String>, NamespaceKind)> {
        if !next_line.is_some_and(|next| next.indentation.is_deeper_than(line.indentation))
            || line.tokens.is_empty()
            || !is_path_spelling(&line.tokens)
        {
            return None;
        }
        let segments = path_segments(&line.tokens);
        let kind = namespace_kind_from_segments(&segments).or_else(|| self.namespace_kind())?;
        Some((segments, kind))
    }
}

impl NamespaceKind {
    const fn definition_kind(self) -> DefinitionKind {
        match self {
            Self::Variable => DefinitionKind::Variable,
            Self::Procedure => DefinitionKind::Procedure,
            Self::Verb => DefinitionKind::Verb,
        }
    }
}

struct Candidate {
    segments: Vec<String>,
    kind: DefinitionKind,
    was_absolute: bool,
}

fn classify_line(line: &SourceLine) -> Option<Candidate> {
    if line.tokens.is_empty() || is_preprocessor_line(&line.tokens) {
        return None;
    }

    let header_end = line
        .tokens
        .iter()
        .position(|token| match &token.kind {
            TokenKind::Punctuation('(') => true,
            TokenKind::Operator(operator) => operator == "=",
            _ => false,
        })
        .unwrap_or(line.tokens.len());
    let path_tokens = &line.tokens[..header_end];
    let was_absolute = starts_with_slash(path_tokens);
    let segments = path_segments(path_tokens);
    if segments.is_empty() || !is_path_spelling(path_tokens) {
        return None;
    }

    let parameter_open = line
        .tokens
        .iter()
        .position(|token| token.kind == TokenKind::Punctuation('('));
    let assignment = line
        .tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="));
    let has_parameters = parameter_open.is_some_and(|open_index| {
        assignment.is_none_or(|assignment_index| open_index < assignment_index)
    });
    let has_assignment = assignment.is_some();
    let kind = if segments.iter().any(|segment| segment == "verb") {
        DefinitionKind::Verb
    } else if segments.iter().any(|segment| segment == "proc") {
        DefinitionKind::Procedure
    } else if segments.iter().any(|segment| segment == "var") {
        DefinitionKind::Variable
    } else if has_parameters {
        DefinitionKind::ProcedureOverride
    } else if has_assignment {
        DefinitionKind::VariableOverride
    } else {
        DefinitionKind::Type
    };

    Some(Candidate {
        segments,
        kind,
        was_absolute,
    })
}

fn starts_with_slash(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Operator(operator)) if operator == "/"
    )
}

fn path_segments(tokens: &[SpannedToken]) -> Vec<String> {
    tokens
        .iter()
        .filter_map(|token| match &token.kind {
            TokenKind::Identifier(identifier) => Some(identifier.clone()),
            _ => None,
        })
        .collect()
}

fn namespace_kind_from_segments(segments: &[String]) -> Option<NamespaceKind> {
    if segments.iter().any(|segment| segment == "verb") {
        Some(NamespaceKind::Verb)
    } else if segments.iter().any(|segment| segment == "proc") {
        Some(NamespaceKind::Procedure)
    } else if segments.iter().any(|segment| segment == "var") {
        Some(NamespaceKind::Variable)
    } else {
        None
    }
}

fn is_preprocessor_line(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens.first().map(|token| &token.kind),
        Some(TokenKind::Operator(operator)) if operator == "#"
    )
}

fn is_path_spelling(tokens: &[SpannedToken]) -> bool {
    tokens.iter().all(|token| {
        matches!(token.kind, TokenKind::Identifier(_))
            || matches!(&token.kind, TokenKind::Operator(operator) if operator == "/")
    })
}

fn canonicalize_path(base: Option<&[String]>, candidate: &Candidate) -> DefinitionPath {
    let mut segments = if candidate.was_absolute {
        Vec::new()
    } else {
        base.unwrap_or_default().to_vec()
    };
    segments.extend(candidate.segments.iter().cloned());

    match candidate.kind {
        DefinitionKind::ProcedureOverride => {
            let name = segments
                .pop()
                .expect("a classified procedure always has a name");
            segments.push("proc".to_owned());
            segments.push(name);
        }
        DefinitionKind::VariableOverride => {
            let name = segments
                .pop()
                .expect("a classified variable always has a name");
            segments.push("var".to_owned());
            segments.push(name);
        }
        DefinitionKind::Variable => {
            let variable_node = segments
                .iter()
                .rposition(|segment| segment == "var")
                .expect("a classified variable has a var path node");
            if variable_node + 1 < segments.len() {
                let name = segments
                    .last()
                    .expect("a variable path has a final segment")
                    .clone();
                segments.truncate(variable_node + 1);
                segments.push(name);
            }
        }
        _ => {}
    }

    DefinitionPath::new(segments)
}

fn parse_parameters(tokens: &[SpannedToken]) -> Vec<ParameterSyntax> {
    let Some(open_index) = tokens
        .iter()
        .position(|token| token.kind == TokenKind::Punctuation('('))
    else {
        return Vec::new();
    };
    let Some(close_index) = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Punctuation(')'))
    else {
        return Vec::new();
    };
    if close_index <= open_index + 1 {
        return Vec::new();
    }

    let mut parameters = Vec::new();
    let mut start = open_index + 1;
    let mut delimiter_depth = 0usize;
    for index in open_index + 1..close_index {
        match tokens[index].kind {
            TokenKind::Punctuation('(' | '[' | '{') => delimiter_depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                delimiter_depth = delimiter_depth.saturating_sub(1);
            }
            TokenKind::Punctuation(',') if delimiter_depth == 0 => {
                push_parameter(&mut parameters, &tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    push_parameter(&mut parameters, &tokens[start..close_index]);
    parameters
}

fn push_parameter(parameters: &mut Vec<ParameterSyntax>, tokens: &[SpannedToken]) {
    let Some(first) = tokens.first() else {
        return;
    };
    let last = tokens.last().expect("a non-empty slice has a last token");
    parameters.push(ParameterSyntax {
        span: SourceSpan::new(first.span.start, last.span.end),
        tokens: tokens.to_vec(),
    });
}

fn build_logical_lines(tokens: Vec<SpannedToken>) -> Result<Vec<SourceLine>, SyntaxError> {
    let mut lines = Vec::new();
    let mut indentation = Indentation::default();
    let mut line_start = None;
    let mut line_tokens: Vec<SpannedToken> = Vec::new();
    let mut delimiter_stack: Vec<SourceSpan> = Vec::new();

    for token in tokens {
        match token.kind {
            TokenKind::LineStart { tabs, spaces } if line_tokens.is_empty() => {
                if line_start.is_none() {
                    indentation = Indentation { tabs, spaces };
                    line_start = Some(token.span.start);
                }
            }
            TokenKind::Newline if delimiter_stack.is_empty() => {
                if !line_tokens.is_empty() {
                    let start = line_start.unwrap_or(line_tokens[0].span.start);
                    lines.push(SourceLine {
                        indentation,
                        span: SourceSpan::new(start, token.span.end),
                        tokens: std::mem::take(&mut line_tokens),
                    });
                }
                indentation = Indentation::default();
                line_start = None;
            }
            TokenKind::Newline | TokenKind::LineContinuation | TokenKind::LineStart { .. } => {}
            TokenKind::Punctuation('(' | '[' | '{') => {
                delimiter_stack.push(token.span);
                line_tokens.push(token);
            }
            TokenKind::Punctuation(')' | ']' | '}') => {
                delimiter_stack.pop();
                line_tokens.push(token);
            }
            _ => line_tokens.push(token),
        }
    }

    if let Some(opening_span) = delimiter_stack.first().copied() {
        return Err(SyntaxError::UnclosedDelimiter(opening_span));
    }
    if !line_tokens.is_empty() {
        let start = line_start.unwrap_or(line_tokens[0].span.start);
        let end = line_tokens
            .last()
            .expect("a non-empty token list has a last token")
            .span
            .end;
        lines.push(SourceLine {
            indentation,
            span: SourceSpan::new(start, end),
            tokens: line_tokens,
        });
    }

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::{DefinitionKind, parse};

    fn paths(source: &str) -> Vec<(String, DefinitionKind, Option<usize>)> {
        parse(source)
            .expect("source should parse")
            .definitions
            .into_iter()
            .map(|definition| {
                (
                    definition.path.to_string(),
                    definition.kind,
                    definition.parent,
                )
            })
            .collect()
    }

    #[test]
    fn indexes_the_seed_fixture() {
        let source = include_str!("../../../fixtures/compiler/basic/basic.dm");
        let syntax = parse(source).expect("fixture should parse");
        let indexed: Vec<_> = syntax
            .definitions
            .iter()
            .map(|definition| (definition.path.to_string(), definition.kind))
            .collect();

        assert_eq!(
            indexed,
            vec![
                ("/world".to_owned(), DefinitionKind::Type),
                (
                    "/world/var/name".to_owned(),
                    DefinitionKind::VariableOverride,
                ),
                ("/datum/probe_base".to_owned(), DefinitionKind::Type),
                (
                    "/datum/probe_base/var/value".to_owned(),
                    DefinitionKind::Variable,
                ),
                (
                    "/datum/probe_base/proc/compute".to_owned(),
                    DefinitionKind::Procedure,
                ),
                ("/datum/probe_base/child".to_owned(), DefinitionKind::Type,),
                (
                    "/datum/probe_base/child/var/value".to_owned(),
                    DefinitionKind::VariableOverride,
                ),
                (
                    "/datum/probe_base/child/proc/compute".to_owned(),
                    DefinitionKind::ProcedureOverride,
                ),
                ("/proc/run_probe".to_owned(), DefinitionKind::Procedure),
            ]
        );
        assert_eq!(syntax.definitions[4].parameters.len(), 1);
        assert_eq!(syntax.definitions[4].body.len(), 1);
        assert_eq!(syntax.definitions[8].body.len(), 3);
    }

    #[test]
    fn equates_relative_paths_with_indentation() {
        let indexed = paths("datum\n\tchild\n\t\tproc/run()\n\t\t\treturn 1\n");

        assert_eq!(
            indexed,
            vec![
                ("/datum".to_owned(), DefinitionKind::Type, None),
                ("/datum/child".to_owned(), DefinitionKind::Type, Some(0)),
                (
                    "/datum/child/proc/run".to_owned(),
                    DefinitionKind::Procedure,
                    Some(1),
                ),
            ]
        );
    }

    #[test]
    fn absolute_path_ignores_indentation_context() {
        let indexed = paths("/datum/outer\n\t/datum/independent\n");

        assert_eq!(
            indexed,
            vec![
                ("/datum/outer".to_owned(), DefinitionKind::Type, None),
                ("/datum/independent".to_owned(), DefinitionKind::Type, None,),
            ]
        );
    }

    #[test]
    fn supports_multiline_parameter_lists() {
        let source = "/datum/example/proc/run(\n\tfirst,\n\tlist/second = list(1, 2),\n\t...\n+)\n\treturn first\n";
        let syntax = parse(source).expect("multiline declaration should parse");

        assert_eq!(syntax.definitions.len(), 1);
        assert_eq!(syntax.definitions[0].parameters.len(), 3);
        assert_eq!(syntax.definitions[0].body.len(), 1);
    }

    #[test]
    fn keeps_a_typed_variable_type_out_of_its_code_tree_path() {
        let indexed = paths("/datum/example\n\tvar/datum/species/current_species\n");

        assert_eq!(
            indexed,
            vec![
                ("/datum/example".to_owned(), DefinitionKind::Type, None),
                (
                    "/datum/example/var/current_species".to_owned(),
                    DefinitionKind::Variable,
                    Some(0),
                ),
            ]
        );
    }

    #[test]
    fn does_not_index_local_variables_as_code_tree_definitions() {
        let source = "/proc/run()\n\tvar/local = 1\n\tif(local)\n\t\tvar/nested = 2\n";
        let syntax = parse(source).expect("procedure should parse");

        assert_eq!(syntax.definitions.len(), 1);
        assert_eq!(syntax.definitions[0].body.len(), 3);
    }

    #[test]
    fn treats_grouped_variables_as_declarations_owned_by_the_type() {
        let indexed =
            paths("/datum/example\n\tvar\n\t\tfirst\n\t\tsecond = 2\n\t\tlist\n\t\t\tnames\n");

        assert_eq!(
            indexed,
            vec![
                ("/datum/example".to_owned(), DefinitionKind::Type, None),
                (
                    "/datum/example/var/first".to_owned(),
                    DefinitionKind::Variable,
                    Some(0),
                ),
                (
                    "/datum/example/var/second".to_owned(),
                    DefinitionKind::Variable,
                    Some(0),
                ),
                (
                    "/datum/example/var/names".to_owned(),
                    DefinitionKind::Variable,
                    Some(0),
                ),
            ]
        );
    }

    #[test]
    fn supports_typed_grouped_variable_paths() {
        let indexed = paths("/datum/example\n\tvar/list\n\t\tentries\n");

        assert_eq!(
            indexed,
            vec![
                ("/datum/example".to_owned(), DefinitionKind::Type, None),
                (
                    "/datum/example/var/entries".to_owned(),
                    DefinitionKind::Variable,
                    Some(0),
                ),
            ]
        );
    }

    #[test]
    fn treats_grouped_procedures_as_executable_declarations() {
        let source = "/datum/example\n\tproc\n\t\tcalculate(value)\n\t\t\treturn value\n";
        let syntax = parse(source).expect("grouped procedure should parse");

        assert_eq!(
            paths(source),
            vec![
                ("/datum/example".to_owned(), DefinitionKind::Type, None),
                (
                    "/datum/example/proc/calculate".to_owned(),
                    DefinitionKind::Procedure,
                    Some(0),
                ),
            ]
        );
        assert_eq!(syntax.definitions[1].parameters.len(), 1);
        assert_eq!(syntax.definitions[1].body.len(), 1);
    }

    #[test]
    fn treats_grouped_verbs_as_executable_declarations() {
        let source = "/mob/example\n\tverb\n\t\tinspect_target(target)\n\t\t\treturn target\n";
        let syntax = parse(source).expect("grouped verb should parse");

        assert_eq!(
            paths(source),
            vec![
                ("/mob/example".to_owned(), DefinitionKind::Type, None),
                (
                    "/mob/example/verb/inspect_target".to_owned(),
                    DefinitionKind::Verb,
                    Some(0),
                ),
            ]
        );
        assert_eq!(syntax.definitions[1].body.len(), 1);
    }

    #[test]
    fn distinguishes_initializer_calls_from_procedure_parameters() {
        let source = r#"
var/global/list/global_list = list("entry")
var/global/global_new = new /datum/example()
var/global/global_call = build_global()

/datum/example
	var/list/instance_list = list()
	var/instance_new = new()
	var/instance_call = build_instance()
	var/static/list/static_list = list()
	var/static/static_new = new /datum()
	var/static/static_call = build_static()
	override_list = list()
	override_new = new()
	override_call = build_override()
	proc/declared(argument = list())
		return argument
	overridden(argument = build_default())
		return argument
"#;
        let syntax = parse(source).expect("initializer expressions should parse");
        let indexed: Vec<_> = syntax
            .definitions
            .iter()
            .map(|definition| (definition.path.to_string(), definition.kind))
            .collect();

        assert_eq!(
            indexed,
            vec![
                ("/var/global_list".to_owned(), DefinitionKind::Variable),
                ("/var/global_new".to_owned(), DefinitionKind::Variable),
                ("/var/global_call".to_owned(), DefinitionKind::Variable),
                ("/datum/example".to_owned(), DefinitionKind::Type),
                (
                    "/datum/example/var/instance_list".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/instance_new".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/instance_call".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/static_list".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/static_new".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/static_call".to_owned(),
                    DefinitionKind::Variable
                ),
                (
                    "/datum/example/var/override_list".to_owned(),
                    DefinitionKind::VariableOverride
                ),
                (
                    "/datum/example/var/override_new".to_owned(),
                    DefinitionKind::VariableOverride
                ),
                (
                    "/datum/example/var/override_call".to_owned(),
                    DefinitionKind::VariableOverride
                ),
                (
                    "/datum/example/proc/declared".to_owned(),
                    DefinitionKind::Procedure
                ),
                (
                    "/datum/example/proc/overridden".to_owned(),
                    DefinitionKind::ProcedureOverride
                ),
            ]
        );
        assert!(
            syntax.definitions[..13]
                .iter()
                .all(|definition| definition.parameters.is_empty() && definition.body.is_empty())
        );
        assert!(
            syntax.definitions[13..]
                .iter()
                .all(|definition| definition.parameters.len() == 1 && definition.body.len() == 1)
        );
    }
}
