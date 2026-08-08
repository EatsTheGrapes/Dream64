//! Project-wide orchestration for the Dream64 compiler frontend.
//!
//! This crate is the stable boundary between project discovery, per-file
//! syntax parsing, and global object-tree construction. Later semantic and
//! bytecode stages can consume one [`Compilation`] without rebuilding or
//! reordering source files.

#![cfg_attr(not(test), deny(missing_docs))]

use std::fmt;
use std::path::{Path, PathBuf};

use dm_core::{FileId, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_object_tree::{
    BuildOutput, CodeTree, DefinitionUnit, DiagnosticKind as TreeDiagnosticKind,
    DiagnosticSeverity as TreeDiagnosticSeverity, NodeId, TreeDiagnostic,
};
use dm_project::{FileKind, PragmaSeverity, Project, ProjectError};
use dm_syntax::{SyntaxError, SyntaxFile};

/// Reusable entry point for deterministic project compilations.
///
/// The database is currently stateless. Giving callers a stable orchestration
/// object now leaves room for caches and dependency invalidation without
/// changing the compilation call site later.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompilerDatabase;

impl CompilerDatabase {
    /// Creates an empty compiler database.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Loads and compiles the frontend representation of a `.dme` project.
    ///
    /// Files are parsed and supplied to the object-tree builder in the
    /// project's deterministic first-include order. A syntax error is retained
    /// as a diagnostic and prevents only that file from contributing
    /// declarations. Project discovery errors remain fatal because they make
    /// the include order incomplete.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when project discovery or source loading
    /// fails.
    pub fn compile(&self, root_file: impl AsRef<Path>) -> Result<Compilation, CompilerError> {
        let project = Project::load(root_file).map_err(CompilerError::Project)?;
        Ok(compile_project(project))
    }
}

/// A complete frontend snapshot for one project.
#[derive(Debug)]
pub struct Compilation {
    project: Project,
    syntax_files: Vec<Option<SyntaxFile>>,
    code_tree: CodeTree,
    declarations: Vec<CompilationDeclaration>,
    diagnostics: Vec<Diagnostic>,
    stats: CompilationStats,
}

impl Compilation {
    /// Returns the discovered project and its source/include tables.
    #[must_use]
    pub const fn project(&self) -> &Project {
        &self.project
    }

    /// Returns parsed syntax by stable project-local file identity.
    ///
    /// Non-source resources and source files with syntax errors return `None`.
    #[must_use]
    pub fn syntax(&self, file_id: FileId) -> Option<&SyntaxFile> {
        self.syntax_files.get(file_id.index())?.as_ref()
    }

    /// Maps a syntax-layer compiler-view range back to original file bytes.
    #[must_use]
    pub fn original_span(&self, file_id: FileId, compiler_span: SourceSpan) -> Option<SourceSpan> {
        self.project
            .file(file_id)
            .map(|file| file.original_span(compiler_span))
    }

    /// Returns the global canonical object tree.
    #[must_use]
    pub const fn code_tree(&self) -> &CodeTree {
        &self.code_tree
    }

    /// Returns declarations in true preprocessor expansion order.
    ///
    /// This is the authoritative global declaration sequence for semantic
    /// compilation. Unlike physical file order, it interleaves declarations
    /// before and after an `#include` with declarations spliced from the
    /// included file.
    #[must_use]
    pub fn declarations(&self) -> &[CompilationDeclaration] {
        &self.declarations
    }

    /// Returns all recoverable frontend diagnostics in deterministic order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Returns deterministic counters for tooling and regression checks.
    #[must_use]
    pub const fn stats(&self) -> &CompilationStats {
        &self.stats
    }
}

/// One parsed declaration placed in the expanded project source stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompilationDeclaration {
    /// Stable position in global preprocessor expansion order.
    pub ordinal: usize,
    /// Physical source file containing the declaration.
    pub file_id: FileId,
    /// Index into that file's [`SyntaxFile::definitions`] table.
    pub definition_index: usize,
    /// Canonical object-tree node receiving the declaration.
    pub node: NodeId,
    /// Declaration header range in the physical source file.
    pub span: SourceSpan,
}

/// Deterministic summary of one frontend compilation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompilationStats {
    /// Every file discovered from the root project.
    pub project_files: usize,
    /// Environment and DM source files successfully parsed.
    pub parsed_files: usize,
    /// Total bytes across all discovered project files.
    pub project_bytes: u64,
    /// Parsed declaration headers before global tree construction.
    pub definitions: usize,
    /// Canonical nodes, including implicit parent types.
    pub code_nodes: usize,
    /// Source declarations in preprocessor expansion order.
    pub code_declarations: usize,
    /// Informational diagnostics.
    pub notes: usize,
    /// Warning diagnostics.
    pub warnings: usize,
    /// Error diagnostics.
    pub errors: usize,
}

/// Severity shared by syntax and object-tree diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticSeverity {
    /// Informational evidence that does not invalidate compilation.
    Note,
    /// A recoverable issue that should be shown to the developer.
    Warning,
    /// A source or structural error.
    Error,
}

/// Compiler stage and category that produced a diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiagnosticKind {
    /// Tokenization or declaration syntax failed in one source file.
    Syntax,
    /// The same file identity reached the object tree twice.
    DuplicateFileUnit,
    /// One explicit declaration repeated another declaration.
    DuplicateDeclaration,
    /// One canonical path occupied incompatible namespaces.
    ConflictingNodeKind,
    /// A procedure, verb, or variable path had no valid owning type.
    MalformedMemberPath,
    /// A native `matrix()` constructor has arguments that imply an invalid signature.
    SuspiciousMatrixCall,
    /// A procedure parameter incorrectly uses a global `/var/...` path.
    ProcArgumentGlobal,
    /// A numeric builtin will substitute zero for a constant non-numeric argument.
    FallbackBuiltinArgument,
    /// A constant builtin argument is outside the builtin's valid numeric domain.
    BadArgument,
    /// A builtin call supplies too few or too many arguments.
    InvalidArgumentCount,
    /// A quoted string contains a malformed embedded expression or text macro.
    InvalidStringInterpolation,
    /// An assignment attempts to modify a compile-time readonly member such as `type`.
    ReadOnlyAssignment,
}

/// A source location with both stable identity and display path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticLocation {
    /// Stable project-local file identity.
    pub file_id: FileId,
    /// Canonical absolute file path.
    pub path: PathBuf,
    /// Relevant byte range in the file, when known.
    pub span: Option<SourceSpan>,
}

/// A recoverable compiler diagnostic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Diagnostic {
    /// Compiler stage and machine-readable category.
    pub kind: DiagnosticKind,
    /// Impact of the issue.
    pub severity: DiagnosticSeverity,
    /// Human-readable explanation.
    pub message: String,
    /// Primary source location, when available.
    pub location: Option<DiagnosticLocation>,
    /// Earlier or otherwise related source location, when available.
    pub related: Option<DiagnosticLocation>,
}

/// Fatal failure that prevented a complete project snapshot.
#[derive(Debug)]
pub enum CompilerError {
    /// Project discovery, preprocessing, or source loading failed.
    Project(ProjectError),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project(error) => write!(formatter, "project loading failed: {error}"),
        }
    }
}

impl std::error::Error for CompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Project(error) => Some(error),
        }
    }
}

fn compile_project(project: Project) -> Compilation {
    let mut syntax_files = Vec::with_capacity(project.files.len());
    let mut diagnostics = Vec::new();
    let mut parsed_files = 0usize;
    let mut definitions = 0usize;

    for file in &project.files {
        if !matches!(file.kind, FileKind::Environment | FileKind::Source) {
            syntax_files.push(None);
            continue;
        }

        // Source-bearing project files have already been UTF-8 validated by
        // the loader while it scanned preprocessing directives.
        let source = file
            .compiler_text()
            .expect("project loader validates source-bearing files as UTF-8");
        match dm_syntax::parse(source) {
            Ok(syntax) => {
                parsed_files += 1;
                definitions += syntax.definitions.len();
                syntax_files.push(Some(syntax));
            }
            Err(error) => {
                let compiler_span = match &error {
                    SyntaxError::Lex(error) => error.span,
                    SyntaxError::UnclosedDelimiter(span) => *span,
                    SyntaxError::InfixNewline { span, .. } => *span,
                };
                diagnostics.push(syntax_diagnostic(
                    file.id,
                    file.path.clone(),
                    file.original_span(compiler_span),
                    &error,
                ));
                syntax_files.push(None);
            }
        }
    }

    let units = expanded_definition_units(&project, &syntax_files);
    let BuildOutput {
        tree,
        diagnostics: tree_diagnostics,
    } = dm_object_tree::build_definitions(&units);
    let declarations = tree
        .declarations()
        .iter()
        .map(|declaration| CompilationDeclaration {
            ordinal: declaration.ordinal,
            file_id: declaration.file_id,
            definition_index: declaration.definition_index,
            node: declaration.node,
            span: project
                .file(declaration.file_id)
                .expect("tree declarations refer to supplied project files")
                .original_span(declaration.span),
        })
        .collect::<Vec<_>>();
    diagnostics.extend(
        tree_diagnostics
            .iter()
            .map(|diagnostic| map_tree_diagnostic(&project, &tree, diagnostic)),
    );
    for (index, syntax) in syntax_files.iter().enumerate() {
        let Some(syntax) = syntax else { continue };
        let file = &project.files[index];
        diagnostics.extend(matrix_lint_diagnostics(file, syntax));
        diagnostics.extend(proc_argument_diagnostics(file, syntax));
        diagnostics.extend(numeric_builtin_diagnostics(file, syntax));
        diagnostics.extend(builtin_arity_diagnostics(file, syntax));
        diagnostics.extend(string_interpolation_diagnostics(file, syntax));
        diagnostics.extend(readonly_member_diagnostics(file, syntax));
    }
    diagnostics.retain_mut(|diagnostic| {
        let Some(severity) = project.diagnostic_severity(diagnostic_pragma_name(diagnostic)) else {
            return true;
        };
        match severity {
            PragmaSeverity::Disabled => false,
            PragmaSeverity::Notice => {
                diagnostic.severity = DiagnosticSeverity::Note;
                true
            }
            PragmaSeverity::Warning => {
                diagnostic.severity = DiagnosticSeverity::Warning;
                true
            }
            PragmaSeverity::Error => {
                diagnostic.severity = DiagnosticSeverity::Error;
                true
            }
        }
    });

    let mut stats = CompilationStats {
        project_files: project.files.len(),
        parsed_files,
        project_bytes: project
            .files
            .iter()
            .map(|file| file.contents.len() as u64)
            .sum(),
        definitions,
        code_nodes: tree.nodes().len(),
        code_declarations: declarations.len(),
        ..CompilationStats::default()
    };
    for diagnostic in &diagnostics {
        match diagnostic.severity {
            DiagnosticSeverity::Note => stats.notes += 1,
            DiagnosticSeverity::Warning => stats.warnings += 1,
            DiagnosticSeverity::Error => stats.errors += 1,
        }
    }

    Compilation {
        project,
        syntax_files,
        code_tree: tree,
        declarations,
        diagnostics,
        stats,
    }
}

fn diagnostic_pragma_name(diagnostic: &Diagnostic) -> &'static str {
    match diagnostic.kind {
        DiagnosticKind::Syntax => "BadToken",
        DiagnosticKind::DuplicateFileUnit => "FileAlreadyIncluded",
        DiagnosticKind::DuplicateDeclaration if diagnostic.message.contains("/var/") => {
            "DuplicateVariable"
        }
        DiagnosticKind::DuplicateDeclaration => "DuplicateProcDefinition",
        DiagnosticKind::ConflictingNodeKind => "InvalidOverride",
        DiagnosticKind::MalformedMemberPath => "DanglingVarType",
        DiagnosticKind::SuspiciousMatrixCall => "SuspiciousMatrixCall",
        DiagnosticKind::ProcArgumentGlobal => "ProcArgumentGlobal",
        DiagnosticKind::FallbackBuiltinArgument => "FallbackBuiltinArgument",
        DiagnosticKind::BadArgument => "BadArgument",
        DiagnosticKind::InvalidArgumentCount => "InvalidArgumentCount",
        DiagnosticKind::InvalidStringInterpolation => "BadExpression",
        DiagnosticKind::ReadOnlyAssignment => "InvalidReference",
    }
}

fn readonly_member_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    const ASSIGNMENTS: &[&str] = &[
        "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&&=",
        "||=", "%%=", "**=",
    ];
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        for line in &definition.body {
            for window in line.tokens.windows(3) {
                let member_access = matches!(&window[0].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | "?." | ":" | "?:"));
                let readonly = matches!(&window[1].kind, TokenKind::Identifier(name) if name == "type");
                let assignment = matches!(&window[2].kind, TokenKind::Operator(operator) if ASSIGNMENTS.contains(&operator.as_str()));
                if !(member_access && readonly && assignment) {
                    continue;
                }
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::ReadOnlyAssignment,
                    severity: DiagnosticSeverity::Error,
                    message: "the type member is compile-time readonly".to_owned(),
                    location: Some(DiagnosticLocation {
                        file_id: file.id,
                        path: file.path.clone(),
                        span: Some(file.original_span(SourceSpan::new(
                            window[0].span.start,
                            window[2].span.end,
                        ))),
                    }),
                    related: None,
                });
            }
        }
    }
    diagnostics
}

fn string_interpolation_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        lint_string_tokens(file, &definition.header, &mut diagnostics);
        for line in &definition.body {
            lint_string_tokens(file, &line.tokens, &mut diagnostics);
        }
    }
    diagnostics
}

fn lint_string_tokens(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for (index, token) in tokens.iter().enumerate() {
        let TokenKind::String(content) = &token.kind else {
            continue;
        };
        let is_text_template = index >= 2
            && tokens[index - 1].kind == TokenKind::Punctuation('(')
            && matches!(&tokens[index - 2].kind, TokenKind::Identifier(name) if name == "text")
            && (index < 3
                || !matches!(&tokens[index - 3].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | ":" | "::" | "?." | "?:")));
        if is_text_template {
            continue;
        }
        let trimmed = content.trim_end();
        let empty_expression = interpolation_contents(content)
            .any(|expression| expression.trim().is_empty() || expression.trim() == ";");
        let adjacent_expressions =
            interpolation_contents(content).any(|expression| expression.contains("\"\""));
        let dangling_text_macro = trimmed.ends_with("\\proper") || trimmed.ends_with("\\improper");
        let message = if empty_expression {
            Some("expected an expression inside string interpolation")
        } else if adjacent_expressions {
            Some("expected the end of the embedded expression")
        } else if dangling_text_macro {
            Some("text macro requires a following interpolated expression or text")
        } else {
            None
        };
        let Some(message) = message else { continue };
        diagnostics.push(Diagnostic {
            kind: DiagnosticKind::InvalidStringInterpolation,
            severity: DiagnosticSeverity::Error,
            message: message.to_owned(),
            location: Some(DiagnosticLocation {
                file_id: file.id,
                path: file.path.clone(),
                span: Some(file.original_span(token.span)),
            }),
            related: None,
        });
    }
}

fn interpolation_contents(content: &str) -> impl Iterator<Item = &str> {
    let mut expressions = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut escaped = false;
    for (offset, character) in content.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if character == '\\' {
            escaped = true;
            continue;
        }
        match character {
            '[' => {
                if depth == 0 {
                    start = offset + 1;
                }
                depth += 1;
            }
            ']' if depth != 0 => {
                depth -= 1;
                if depth == 0 {
                    expressions.push(&content[start..offset]);
                }
            }
            _ => {}
        }
    }
    expressions.into_iter()
}

fn numeric_builtin_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        lint_numeric_builtin_calls(file, &definition.header, &mut diagnostics);
        for line in &definition.body {
            lint_numeric_builtin_calls(file, &line.tokens, &mut diagnostics);
        }
    }
    diagnostics
}

fn lint_numeric_builtin_calls(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    diagnostics: &mut Vec<Diagnostic>,
) {
    const NUMERIC_BUILTINS: &[&str] = &[
        "abs", "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sqrt", "log", "rgb",
    ];
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let TokenKind::Identifier(name) = &tokens[index].kind else {
            index += 1;
            continue;
        };
        let is_member = index != 0
            && matches!(&tokens[index - 1].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | ":" | "::" | "?." | "?:"));
        if is_member
            || !NUMERIC_BUILTINS.contains(&name.as_str())
            || tokens[index + 1].kind != TokenKind::Punctuation('(')
        {
            index += 1;
            continue;
        }
        let Some((close, arguments)) = call_arguments(tokens, index + 1) else {
            index += 1;
            continue;
        };
        if (name == "log" || name == "arctan") && !matches!(arguments.len(), 1 | 2) {
            index = close + 1;
            continue;
        }
        if name == "rgb" && !(3..=5).contains(&arguments.len()) {
            index = close + 1;
            continue;
        }
        if name != "log" && name != "arctan" && name != "rgb" && arguments.len() != 1 {
            index = close + 1;
            continue;
        }
        for argument in &arguments {
            if constant_non_number(argument) {
                push_builtin_diagnostic(
                    file,
                    argument,
                    DiagnosticKind::FallbackBuiltinArgument,
                    DiagnosticSeverity::Warning,
                    format!("constant non-numeric argument to {name}() is treated as 0"),
                    diagnostics,
                );
            }
        }
        for (argument_index, argument) in arguments.iter().enumerate() {
            let Some(number) = constant_number(argument) else {
                continue;
            };
            let invalid = match name.as_str() {
                "arcsin" | "arccos" => !(-1.0..=1.0).contains(&number),
                "sqrt" => number < 0.0,
                "log" => number < 0.0 && argument_index < 2,
                _ => false,
            };
            if invalid {
                push_builtin_diagnostic(
                    file,
                    argument,
                    DiagnosticKind::BadArgument,
                    DiagnosticSeverity::Error,
                    format!("constant argument {number} is outside the domain of {name}()"),
                    diagnostics,
                );
            }
        }
        index = close + 1;
    }
}

fn builtin_arity_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        lint_builtin_arities(file, &definition.header, &mut diagnostics);
        for line in &definition.body {
            lint_builtin_arities(file, &line.tokens, &mut diagnostics);
        }
    }
    diagnostics
}

fn lint_builtin_arities(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let TokenKind::Identifier(name) = &tokens[index].kind else {
            index += 1;
            continue;
        };
        let is_member = index != 0
            && matches!(&tokens[index - 1].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | ":" | "::" | "?." | "?:"));
        if is_member || tokens[index + 1].kind != TokenKind::Punctuation('(') {
            index += 1;
            continue;
        }
        let Some((close, arguments)) = call_arguments(tokens, index + 1) else {
            index += 1;
            continue;
        };
        let valid = match name.as_str() {
            "image" => !arguments.is_empty(),
            "addtext" => arguments.len() >= 2,
            "rgb" => (3..=5).contains(&arguments.len()),
            _ => {
                index += 1;
                continue;
            }
        };
        if !valid {
            let compiler_span = SourceSpan::new(tokens[index].span.start, tokens[close].span.end);
            diagnostics.push(Diagnostic {
                kind: DiagnosticKind::InvalidArgumentCount,
                severity: DiagnosticSeverity::Error,
                message: format!("invalid argument count {} for {name}()", arguments.len()),
                location: Some(DiagnosticLocation {
                    file_id: file.id,
                    path: file.path.clone(),
                    span: Some(file.original_span(compiler_span)),
                }),
                related: None,
            });
        }
        index = close + 1;
    }
}

fn push_builtin_diagnostic(
    file: &dm_project::ProjectFile,
    argument: &[SpannedToken],
    kind: DiagnosticKind,
    severity: DiagnosticSeverity,
    message: String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some((first, last)) = argument.first().zip(argument.last()) else {
        return;
    };
    diagnostics.push(Diagnostic {
        kind,
        severity,
        message,
        location: Some(DiagnosticLocation {
            file_id: file.id,
            path: file.path.clone(),
            span: Some(file.original_span(SourceSpan::new(first.span.start, last.span.end))),
        }),
        related: None,
    });
}

fn constant_non_number(tokens: &[SpannedToken]) -> bool {
    matches!(
        tokens,
        [SpannedToken {
            kind: TokenKind::String(_)
                | TokenKind::RawString(_)
                | TokenKind::TextBlock(_)
                | TokenKind::Resource(_),
            ..
        }]
    ) || matches!(tokens, [SpannedToken { kind: TokenKind::Identifier(name), .. }] if name == "null")
}

fn constant_number(tokens: &[SpannedToken]) -> Option<f32> {
    match tokens {
        [
            SpannedToken {
                kind: TokenKind::Number(number),
                ..
            },
        ] => number.parse().ok(),
        [
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Number(number),
                ..
            },
        ] if operator == "-" => number.parse::<f32>().ok().map(|number| -number),
        _ => None,
    }
}

fn proc_argument_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        for parameter in &definition.parameters {
            if !matches!(parameter.tokens.as_slice(), [
                SpannedToken { kind: TokenKind::Operator(slash), .. },
                SpannedToken { kind: TokenKind::Identifier(var), .. },
                ..
            ] if slash == "/" && var == "var")
            {
                continue;
            }
            diagnostics.push(Diagnostic {
                kind: DiagnosticKind::ProcArgumentGlobal,
                severity: DiagnosticSeverity::Warning,
                message: "procedure arguments cannot use the global /var path".to_owned(),
                location: Some(DiagnosticLocation {
                    file_id: file.id,
                    path: file.path.clone(),
                    span: Some(file.original_span(parameter.span)),
                }),
                related: None,
            });
        }
    }
    diagnostics
}

fn matrix_lint_diagnostics(file: &dm_project::ProjectFile, syntax: &SyntaxFile) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        for line in &definition.body {
            lint_matrix_calls(file, &line.tokens, &mut diagnostics);
        }
    }
    diagnostics
}

fn lint_matrix_calls(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut index = 0usize;
    while index + 1 < tokens.len() {
        let is_matrix =
            matches!(&tokens[index].kind, TokenKind::Identifier(name) if name == "matrix");
        let is_member = index != 0
            && matches!(&tokens[index - 1].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | ":" | "::" | "?." | "?:"));
        if !is_matrix || is_member || tokens[index + 1].kind != TokenKind::Punctuation('(') {
            index += 1;
            continue;
        }
        let Some((close, arguments)) = call_arguments(tokens, index + 1) else {
            index += 1;
            continue;
        };
        let suspicious = match arguments.len() {
            2..=4 => arguments
                .last()
                .and_then(|argument| constant_matrix_opcode(argument))
                .is_some_and(|valid| !valid),
            5 => true,
            _ => false,
        };
        if suspicious {
            let compiler_span =
                SourceSpan::new(tokens[index + 1].span.start, tokens[close].span.end);
            diagnostics.push(Diagnostic {
                kind: DiagnosticKind::SuspiciousMatrixCall,
                severity: DiagnosticSeverity::Warning,
                message: if arguments.len() == 5 {
                    "calling matrix() with 5 arguments will always error at runtime".to_owned()
                } else {
                    "matrix() arguments have an invalid opcode or insufficient arguments".to_owned()
                },
                location: Some(DiagnosticLocation {
                    file_id: file.id,
                    path: file.path.clone(),
                    span: Some(file.original_span(compiler_span)),
                }),
                related: None,
            });
        }
        index = close + 1;
    }
}

fn call_arguments(tokens: &[SpannedToken], open: usize) -> Option<(usize, Vec<&[SpannedToken]>)> {
    let mut depth = 1usize;
    let mut start = open + 1;
    let mut arguments = Vec::new();
    for index in open + 1..tokens.len() {
        match tokens[index].kind {
            TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
            TokenKind::Punctuation(')' | ']' | '}') => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    if index != start || !arguments.is_empty() {
                        arguments.push(&tokens[start..index]);
                    }
                    return Some((index, arguments));
                }
            }
            TokenKind::Punctuation(',') if depth == 1 => {
                arguments.push(&tokens[start..index]);
                start = index + 1;
            }
            _ => {}
        }
    }
    None
}

fn constant_matrix_opcode(tokens: &[SpannedToken]) -> Option<bool> {
    match tokens {
        [
            SpannedToken {
                kind: TokenKind::Number(number),
                ..
            },
        ] => {
            let opcode = number.parse::<i32>().ok()?;
            Some((0..=8).contains(&(opcode & !128)))
        }
        [
            SpannedToken {
                kind: TokenKind::Operator(operator),
                ..
            },
            SpannedToken {
                kind: TokenKind::Number(number),
                ..
            },
        ] if operator == "-" => {
            let opcode = -number.parse::<i32>().ok()?;
            Some((0..=8).contains(&(opcode & !128)))
        }
        [
            SpannedToken {
                kind:
                    TokenKind::String(_)
                    | TokenKind::RawString(_)
                    | TokenKind::TextBlock(_)
                    | TokenKind::Resource(_),
                ..
            },
        ] => Some(false),
        [
            SpannedToken {
                kind: TokenKind::Identifier(name),
                ..
            },
        ] if matches!(name.as_str(), "null" | "TRUE" | "FALSE") => Some(false),
        _ => None,
    }
}

fn expanded_definition_units<'syntax>(
    project: &Project,
    syntax_files: &'syntax [Option<SyntaxFile>],
) -> Vec<DefinitionUnit<'syntax>> {
    let mut units = Vec::new();
    let mut next_definition = vec![0usize; project.files.len()];
    for segment in project.expansion_segments() {
        let Some(syntax) = syntax_files[segment.file_id.index()].as_ref() else {
            continue;
        };
        let definition_index = &mut next_definition[segment.file_id.index()];
        let file = project
            .file(segment.file_id)
            .expect("expansion segments refer to project files");
        while *definition_index < syntax.definitions.len()
            && file
                .original_span(syntax.definitions[*definition_index].span)
                .start
                < segment.span.start
        {
            *definition_index += 1;
        }
        while *definition_index < syntax.definitions.len() {
            let definition = &syntax.definitions[*definition_index];
            if file.original_span(definition.span).start >= segment.span.end {
                break;
            }
            units.push(DefinitionUnit {
                file_id: segment.file_id,
                definition_index: *definition_index,
                definition,
            });
            *definition_index += 1;
        }
    }
    units
}

fn syntax_diagnostic(
    file_id: FileId,
    path: PathBuf,
    span: SourceSpan,
    error: &SyntaxError,
) -> Diagnostic {
    Diagnostic {
        kind: DiagnosticKind::Syntax,
        severity: DiagnosticSeverity::Error,
        message: error.to_string(),
        location: Some(DiagnosticLocation {
            file_id,
            path,
            span: Some(span),
        }),
        related: None,
    }
}

fn map_tree_diagnostic(
    project: &Project,
    tree: &CodeTree,
    diagnostic: &TreeDiagnostic,
) -> Diagnostic {
    let location = diagnostic
        .current
        .and_then(|declaration| tree.declaration(declaration))
        .map(|declaration| declaration_location(project, declaration.file_id, declaration.span))
        .or_else(|| {
            diagnostic.file_id.map(|file_id| DiagnosticLocation {
                file_id,
                path: project
                    .file(file_id)
                    .expect("tree diagnostics refer to supplied project files")
                    .path
                    .clone(),
                span: None,
            })
        });
    let related = diagnostic
        .previous
        .and_then(|declaration| tree.declaration(declaration))
        .map(|declaration| declaration_location(project, declaration.file_id, declaration.span));
    Diagnostic {
        kind: match diagnostic.kind {
            TreeDiagnosticKind::DuplicateFileUnit => DiagnosticKind::DuplicateFileUnit,
            TreeDiagnosticKind::DuplicateDeclaration => DiagnosticKind::DuplicateDeclaration,
            TreeDiagnosticKind::ConflictingNodeKind => DiagnosticKind::ConflictingNodeKind,
            TreeDiagnosticKind::MalformedMemberPath => DiagnosticKind::MalformedMemberPath,
        },
        severity: match diagnostic.severity {
            TreeDiagnosticSeverity::Note => DiagnosticSeverity::Note,
            TreeDiagnosticSeverity::Warning => DiagnosticSeverity::Warning,
            TreeDiagnosticSeverity::Error => DiagnosticSeverity::Error,
        },
        message: tree_diagnostic_message(diagnostic),
        location,
        related,
    }
}

fn declaration_location(
    project: &Project,
    file_id: FileId,
    span: SourceSpan,
) -> DiagnosticLocation {
    DiagnosticLocation {
        file_id,
        path: project
            .file(file_id)
            .expect("tree declarations refer to supplied project files")
            .path
            .clone(),
        span: Some(
            project
                .file(file_id)
                .expect("tree declarations refer to supplied project files")
                .original_span(span),
        ),
    }
}

fn tree_diagnostic_message(diagnostic: &TreeDiagnostic) -> String {
    let path = diagnostic
        .path
        .as_ref()
        .map_or_else(String::new, |path| format!(" for {path}"));
    match diagnostic.kind {
        TreeDiagnosticKind::DuplicateFileUnit => {
            "source file was supplied to the object tree more than once".to_owned()
        }
        TreeDiagnosticKind::DuplicateDeclaration => {
            format!("duplicate explicit declaration{path}")
        }
        TreeDiagnosticKind::ConflictingNodeKind => {
            format!("conflicting declaration namespace{path}")
        }
        TreeDiagnosticKind::MalformedMemberPath => {
            format!("member declaration has no valid owning type{path}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_core::SourceSpan;
    use dm_object_tree::NodeKind;

    use super::{CompilerDatabase, DiagnosticKind};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn new() -> Self {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-compiler-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("test project directory should be created");
            Self { root }
        }

        fn write(&self, relative_path: &str, contents: &str) {
            let path = self.root.join(relative_path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("test source directory should be created");
            }
            fs::write(path, contents).expect("test source should be written");
        }

        fn path(&self, relative_path: &str) -> PathBuf {
            self.root.join(relative_path)
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.root) {
                eprintln!(
                    "failed to clean test project {}: {error}",
                    self.root.display()
                );
            }
        }
    }

    fn relative(path: &Path) -> &str {
        path.file_name()
            .and_then(|name| name.to_str())
            .expect("test paths should have Unicode file names")
    }

    #[test]
    fn retains_syntax_by_file_id_and_builds_in_include_order() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "#include \"first.dm\"\n#include \"second.dm\"\n#include \"map.dmm\"\n",
        );
        fixture.write("first.dm", "/datum/sample/proc/run()\n\treturn 1\n");
        fixture.write("second.dm", "/datum/sample\n\trun()\n\t\treturn 2\n");
        fixture.write("map.dmm", "\"a\" = (/turf)\n");

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("mini project should compile");
        let files = &compilation.project().files;
        assert_eq!(relative(&files[0].path), "world.dme");
        assert_eq!(relative(&files[1].path), "first.dm");
        assert_eq!(relative(&files[2].path), "second.dm");
        assert_eq!(relative(&files[3].path), "map.dmm");
        assert!(compilation.syntax(files[1].id).is_some());
        assert!(compilation.syntax(files[3].id).is_none());

        let procedure = compilation
            .code_tree()
            .nodes()
            .iter()
            .find(|node| node.path.to_string() == "/datum/sample/proc/run")
            .expect("procedure should be globally indexed");
        let declarations: Vec<_> = procedure
            .declarations
            .iter()
            .map(|id| {
                compilation
                    .code_tree()
                    .declaration(*id)
                    .expect("declaration id should be valid")
            })
            .collect();
        assert_eq!(declarations.len(), 2);
        assert_eq!(declarations[0].file_id, files[1].id);
        assert_eq!(declarations[1].file_id, files[2].id);
        assert_eq!(procedure.kind, NodeKind::Procedure);
    }

    #[test]
    fn splices_included_declarations_between_surrounding_declarations() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"outer.dm\"\n");
        fixture.write(
            "outer.dm",
            "/datum/order/var/first\n#include \"middle.dm\"\n/datum/order/var/third\n#include \"middle.dm\"\n",
        );
        fixture.write("middle.dm", "/datum/order/var/second\n");

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("splice-order project should compile");
        let ordered_paths: Vec<_> = compilation
            .declarations()
            .iter()
            .map(|declaration| {
                compilation
                    .syntax(declaration.file_id)
                    .expect("ordered declarations come from parsed source")
                    .definitions[declaration.definition_index]
                    .path
                    .to_string()
            })
            .collect();

        assert_eq!(
            ordered_paths,
            [
                "/datum/order/var/first",
                "/datum/order/var/second",
                "/datum/order/var/third",
            ]
        );
        assert_eq!(
            compilation
                .declarations()
                .iter()
                .map(|declaration| declaration.ordinal)
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
    }

    #[test]
    fn parses_only_selected_conditional_branches_at_original_spans() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"conditions.dm\"\n");
        let source = "/datum/before\n#if 0\n/datum/hidden\n#if 1\n/datum/nested_hidden\n#endif\n#else\n/* span-preserving comment */\n/datum/selected\n#if 1\n/datum/nested_selected\n#else\n/datum/other_hidden\n#endif\n#endif\n/datum/after\n";
        fixture.write("conditions.dm", source);

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("conditional project should compile");
        let source_file = compilation
            .project()
            .files
            .iter()
            .find(|file| relative(&file.path) == "conditions.dm")
            .expect("conditional source should be discovered");
        let syntax = compilation
            .syntax(source_file.id)
            .expect("conditional source should parse");
        let paths: Vec<_> = syntax
            .definitions
            .iter()
            .map(|definition| definition.path.to_string())
            .collect();

        assert_eq!(
            paths,
            [
                "/datum/before",
                "/datum/selected",
                "/datum/nested_selected",
                "/datum/after",
            ]
        );
        let selected = syntax
            .definitions
            .iter()
            .find(|definition| definition.path.to_string() == "/datum/selected")
            .expect("selected else declaration should exist");
        assert_eq!(
            selected.span.start,
            source
                .find("/datum/selected")
                .expect("selected declaration should exist in original source")
        );
        assert_eq!(
            &source[selected.span.start..selected.span.end],
            "/datum/selected\n"
        );
        assert_eq!(compilation.stats().definitions, 4);
        assert_eq!(compilation.stats().code_declarations, 4);
    }

    #[test]
    fn maps_expanded_declarations_and_diagnostics_to_macro_invocations() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "#include \"defines.dm\"\n#include \"mapped.dm\"\n#include \"broken.dm\"\n",
        );
        fixture.write(
            "defines.dm",
            "#define ROOT /datum\n#define MAPPED ROOT/mapped\n#define BROKEN ☃\n",
        );
        fixture.write("mapped.dm", "MAPPED\n");
        fixture.write("broken.dm", "BROKEN\n");

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("macro diagnostics should be recoverable");
        let mapped_file = compilation
            .project()
            .files
            .iter()
            .find(|file| relative(&file.path) == "mapped.dm")
            .expect("mapped source should be discovered");
        let syntax = compilation
            .syntax(mapped_file.id)
            .expect("mapped declaration should parse");
        assert_eq!(syntax.definitions[0].path.to_string(), "/datum/mapped");
        assert_eq!(
            compilation.original_span(mapped_file.id, syntax.definitions[0].span),
            Some(SourceSpan::new(0, "MAPPED\n".len()))
        );
        let declaration = compilation
            .declarations()
            .iter()
            .find(|declaration| declaration.file_id == mapped_file.id)
            .expect("mapped declaration should enter the tree");
        assert_eq!(declaration.span, SourceSpan::new(0, "MAPPED\n".len()));

        let diagnostic = compilation
            .diagnostics()
            .iter()
            .find(|diagnostic| diagnostic.kind == DiagnosticKind::Syntax)
            .expect("invalid replacement should produce a syntax diagnostic");
        let location = diagnostic
            .location
            .as_ref()
            .expect("macro diagnostic should retain its invocation");
        assert_eq!(relative(&location.path), "broken.dm");
        assert_eq!(location.span, Some(SourceSpan::new(0, "BROKEN".len())));
    }

    #[test]
    fn aggregates_syntax_and_tree_diagnostics_with_paths_and_spans() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "#include \"good.dm\"\n#include \"broken.dm\"\n#include \"duplicate.dm\"\n",
        );
        fixture.write("good.dm", "/datum/example/var/value\n");
        fixture.write("broken.dm", "/datum/broken/proc/run(\n");
        fixture.write("duplicate.dm", "/datum/example/var/value\n");

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("recoverable diagnostics should not abort compilation");
        let syntax = compilation
            .diagnostics()
            .iter()
            .find(|diagnostic| diagnostic.kind == DiagnosticKind::Syntax)
            .expect("syntax diagnostic should be retained");
        assert_eq!(
            relative(
                &syntax
                    .location
                    .as_ref()
                    .expect("syntax diagnostic should have a location")
                    .path
            ),
            "broken.dm"
        );
        assert!(
            syntax
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        );

        let duplicate = compilation
            .diagnostics()
            .iter()
            .find(|diagnostic| diagnostic.kind == DiagnosticKind::DuplicateDeclaration)
            .expect("duplicate declaration should be retained");
        assert_eq!(
            relative(
                &duplicate
                    .location
                    .as_ref()
                    .expect("duplicate should have a primary location")
                    .path
            ),
            "duplicate.dm"
        );
        assert_eq!(
            relative(
                &duplicate
                    .related
                    .as_ref()
                    .expect("duplicate should link the earlier declaration")
                    .path
            ),
            "good.dm"
        );
        assert!(
            duplicate
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        );
        assert!(
            duplicate
                .related
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        );
        assert_eq!(compilation.stats().errors, 1);
        assert_eq!(compilation.stats().notes, 1);
    }

    #[test]
    fn applies_project_pragma_severity_to_frontend_diagnostics() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma DuplicateProcDefinition error\n",
                "#include \"duplicate.dm\"\n",
            ),
        );
        fixture.write(
            "duplicate.dm",
            "/datum/example/proc/run()\n\treturn 1\n/datum/example/proc/run()\n\treturn 2\n",
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("pragma-controlled diagnostics should compile recoverably");
        let duplicate = compilation
            .diagnostics()
            .iter()
            .find(|diagnostic| diagnostic.kind == DiagnosticKind::DuplicateDeclaration)
            .expect("duplicate procedure should emit a diagnostic");

        assert_eq!(duplicate.severity, super::DiagnosticSeverity::Error);
        assert_eq!(compilation.stats().errors, 1);
    }

    #[test]
    fn emits_and_controls_suspicious_matrix_constructor_diagnostics() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma SuspiciousMatrixCall error\n",
                "/proc/run()\n",
                "\tvar/bad = matrix(\"x\", \"y\", \"not an opcode\")\n",
                "\tvar/five = matrix(1, 2, 3, 4, 5)\n",
                "\tvar/good_copy = matrix(1, 0)\n",
                "\tvar/good_modify = matrix(1, 129)\n",
                "\tvar/dynamic = matrix(1, opcode)\n",
                "\tvar/member = holder.matrix(1, \"bad\")\n",
                "\tvar/normal = matrix(1, 0, 0, 1, 0, 0)\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("matrix linting should be recoverable");
        let matrix_diagnostics: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::SuspiciousMatrixCall)
            .collect();

        assert_eq!(matrix_diagnostics.len(), 2);
        assert!(
            matrix_diagnostics
                .iter()
                .all(|diagnostic| diagnostic.severity == super::DiagnosticSeverity::Error)
        );
        assert!(matrix_diagnostics.iter().all(|diagnostic| {
            diagnostic
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        }));
    }

    #[test]
    fn disables_suspicious_matrix_diagnostics_via_pragma() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "#pragma SuspiciousMatrixCall disabled\n/proc/run()\n\treturn matrix(1, 2, 3, 4, 5)\n",
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("disabled matrix lint should compile");

        assert!(
            compilation
                .diagnostics()
                .iter()
                .all(|diagnostic| diagnostic.kind != DiagnosticKind::SuspiciousMatrixCall)
        );
    }

    #[test]
    fn diagnoses_global_var_paths_only_in_procedure_parameters() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma ProcArgumentGlobal error\n",
                "/var/global_value\n",
                "/proc/bad(/var/value)\n",
                "\treturn value\n",
                "/proc/good(var/value, datum/typed)\n",
                "\treturn value\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("argument lint should be recoverable");
        let diagnostics: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::ProcArgumentGlobal)
            .collect();

        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, super::DiagnosticSeverity::Error);
        assert!(
            diagnostics[0]
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        );
    }

    #[test]
    fn distinguishes_duplicate_variable_and_procedure_pragma_names() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma DuplicateVariable disabled\n",
                "#pragma DuplicateProcDefinition error\n",
                "/datum/example/var/value\n",
                "/datum/example/var/value\n",
                "/datum/example/proc/run()\n",
                "\treturn 1\n",
                "/datum/example/proc/run()\n",
                "\treturn 2\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("duplicate declarations should be recoverable");
        let duplicates: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::DuplicateDeclaration)
            .collect();

        assert_eq!(duplicates.len(), 1);
        assert_eq!(duplicates[0].severity, super::DiagnosticSeverity::Error);
        assert!(duplicates[0].message.contains("/proc/"));
    }

    #[test]
    fn validates_constant_numeric_builtin_arguments_without_flagging_dynamic_values() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma FallbackBuiltinArgument error\n",
                "/proc/run(dynamic)\n",
                "\tvar/fallback = sin(\"bad\")\n",
                "\tvar/domain = sqrt(-1)\n",
                "\tvar/good = arcsin(1) + log(2, 4)\n",
                "\tvar/unknown = cos(dynamic)\n",
                "\tvar/member = holder.sin(\"bad\")\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("builtin diagnostics should be recoverable");
        let fallback: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::FallbackBuiltinArgument)
            .collect();
        let bad_argument: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::BadArgument)
            .collect();

        assert_eq!(fallback.len(), 1);
        assert_eq!(fallback[0].severity, super::DiagnosticSeverity::Error);
        assert_eq!(bad_argument.len(), 1);
        assert_eq!(bad_argument[0].severity, super::DiagnosticSeverity::Error);
    }

    #[test]
    fn validates_builtin_arities_and_global_rgb_constant_arguments() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma FallbackBuiltinArgument error\n",
                "/datum/example/var/color = rgb(1, 2, null)\n",
                "/proc/run(dynamic)\n",
                "\timage()\n",
                "\taddtext(\"only one\")\n",
                "\trgb(1, 2)\n",
                "\timage(dynamic)\n",
                "\taddtext(\"a\", dynamic)\n",
                "\trgb(1, 2, 3)\n",
                "\tholder.image()\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("arity diagnostics should be recoverable");

        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::InvalidArgumentCount)
                .count(),
            3
        );
        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::FallbackBuiltinArgument)
                .count(),
            1
        );
    }

    #[test]
    fn rejects_type_member_writes_without_rejecting_reads_or_mutable_members() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/proc/run(list/value)\n",
                "\tvalue.type = /list\n",
                "\tvalue.type += 1\n",
                "\tvar/read_type = value.type\n",
                "\tvalue.len = 2\n",
                "\tvalue.dynamic = 3\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("readonly diagnostics should be recoverable");
        let diagnostics: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::ReadOnlyAssignment)
            .collect();

        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.location.as_ref().and_then(|item| item.span).is_some())
        );
    }

    #[test]
    fn rejects_malformed_string_interpolations_and_accepts_valid_nested_text() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/proc/run(value)\n",
                "\tvar/empty = \"[;]\"\n",
                "\tvar/adjacent = \"[\"a\"\"b\"]\"\n",
                "\tvar/dangling = \"Example \\proper\"\n",
                "\tvar/good = \"prefix [value] suffix\"\n",
                "\tvar/nested = \"[format(\"nested [value]\")]\"\n",
                "\tvar/proper_name = \"\\proper thing\"\n",
                "\tvar/template = text(\"[] [ ]\", value, value)\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("string diagnostics should be recoverable");
        let diagnostics: Vec<_> = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::InvalidStringInterpolation)
            .collect();

        assert_eq!(diagnostics.len(), 3);
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        }));
    }

    #[test]
    fn reports_stable_stats_for_the_same_project() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/example\n\tvar/value = 3\n\tproc/read()\n\t\treturn value\n",
        );
        let database = CompilerDatabase::new();

        let first = database
            .compile(fixture.path("world.dme"))
            .expect("first compilation should succeed");
        let second = database
            .compile(fixture.path("world.dme"))
            .expect("second compilation should succeed");

        assert_eq!(first.stats(), second.stats());
        assert_eq!(first.stats().project_files, 2);
        assert_eq!(first.stats().parsed_files, 2);
        assert_eq!(first.stats().definitions, 3);
        assert_eq!(first.stats().code_declarations, 3);
        assert_eq!(first.stats().errors, 0);
    }
}
