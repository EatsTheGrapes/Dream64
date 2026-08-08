//! Deterministic loading and include discovery for DM environment projects.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str;

use dm_core::{FileId, SourceSpan};

const TARGET_DM_VERSION: i64 = 516;
const TARGET_DM_BUILD: i64 = 1663;

/// A fully discovered project in first-include order.
#[derive(Debug)]
pub struct Project {
    /// Canonical project directory containing the root environment file.
    pub root_directory: PathBuf,
    /// Files in deterministic first-discovery order.
    pub files: Vec<ProjectFile>,
    /// Include operations in source encounter order.
    pub includes: Vec<IncludeEdge>,
    /// Active compiler diagnostic policies in source encounter order.
    pub diagnostic_pragmas: Vec<DiagnosticPragma>,
}

impl Project {
    /// Loads a `.dme` and recursively discovers quoted source includes.
    ///
    /// Quoted includes are searched relative to the including file first and
    /// then relative to the root project directory. A canonical path is loaded
    /// only once, matching DM's include behavior. Angle-bracket system includes
    /// are recorded but resolved later by the standard-library layer.
    ///
    /// # Errors
    ///
    /// Returns [`ProjectError`] when the root or an include cannot be read,
    /// when a source file is not UTF-8, or when an include escapes the project
    /// directory.
    pub fn load(root_file: impl AsRef<Path>) -> Result<Self, ProjectError> {
        Loader::new(root_file.as_ref())?.load()
    }

    /// Retrieves a file by its stable project-local identity.
    #[must_use]
    pub fn file(&self, id: FileId) -> Option<&ProjectFile> {
        self.files.get(id.index())
    }

    /// Returns the last configured severity for a named compiler diagnostic.
    #[must_use]
    pub fn diagnostic_severity(&self, name: &str) -> Option<PragmaSeverity> {
        self.diagnostic_pragmas
            .iter()
            .rev()
            .find(|pragma| pragma.name == name)
            .map(|pragma| pragma.severity)
    }

    /// Returns the project source stream after quoted includes are spliced in.
    ///
    /// Segments appear in the same deterministic depth-first order as textual
    /// `#include` expansion. Include directive lines are excluded. A physical
    /// file contributes segments only at its first include, matching the
    /// loader's include-once behavior; later includes of the same file expand
    /// to no source.
    #[must_use]
    pub fn expansion_segments(&self) -> Vec<ExpansionSegment> {
        if self.files.is_empty() {
            return Vec::new();
        }

        let mut includes_by_file = vec![Vec::new(); self.files.len()];
        for include in &self.includes {
            includes_by_file[include.source.index()].push(include);
        }
        for includes in &mut includes_by_file {
            includes.sort_by_key(|include| (include.span.start, include.span.end));
        }

        let mut segments = Vec::new();
        let mut expanded = vec![false; self.files.len()];
        expanded[0] = true;
        let mut stack = vec![ExpansionFrame::new(self.files[0].id)];
        while !stack.is_empty() {
            let frame_index = stack.len() - 1;
            let file_id = stack[frame_index].file_id;
            let includes = &includes_by_file[file_id.index()];
            if stack[frame_index].next_include < includes.len() {
                let include = includes[stack[frame_index].next_include];
                stack[frame_index].next_include += 1;
                let cursor = stack[frame_index].cursor;
                if cursor < include.span.start {
                    segments.push(ExpansionSegment {
                        file_id,
                        span: SourceSpan::new(cursor, include.span.start),
                    });
                }
                stack[frame_index].cursor = cursor.max(include.span.end);

                if let IncludeTarget::File(target) = include.target
                    && !expanded[target.index()]
                {
                    expanded[target.index()] = true;
                    stack.push(ExpansionFrame::new(target));
                }
                continue;
            }

            let cursor = stack[frame_index].cursor;
            let file_end = self.files[file_id.index()].contents.len();
            if cursor < file_end {
                segments.push(ExpansionSegment {
                    file_id,
                    span: SourceSpan::new(cursor, file_end),
                });
            }
            stack.pop();
        }
        segments
    }
}

/// Severity selected by a Dream Maker `#pragma` diagnostic policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PragmaSeverity {
    /// Do not emit the diagnostic.
    Disabled,
    /// Emit an informational diagnostic.
    Notice,
    /// Emit a warning.
    Warning,
    /// Emit a compilation error.
    Error,
}

/// One source-ordered `#pragma DiagnosticName severity` policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticPragma {
    /// OpenDream/Dream Maker diagnostic name.
    pub name: String,
    /// Configured severity.
    pub severity: PragmaSeverity,
    /// File containing the directive.
    pub source: FileId,
    /// Complete directive range.
    pub span: SourceSpan,
}

/// A contiguous part of one file in the fully expanded project source stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionSegment {
    /// Physical project file supplying the bytes.
    pub file_id: FileId,
    /// Byte range retained from that physical file.
    pub span: SourceSpan,
}

#[derive(Clone, Copy, Debug)]
struct ExpansionFrame {
    file_id: FileId,
    next_include: usize,
    cursor: usize,
}

impl ExpansionFrame {
    const fn new(file_id: FileId) -> Self {
        Self {
            file_id,
            next_include: 0,
            cursor: 0,
        }
    }
}

/// A file known to the project loader.
#[derive(Debug)]
pub struct ProjectFile {
    /// Stable identity assigned in first-discovery order.
    pub id: FileId,
    /// Canonical absolute path.
    pub path: PathBuf,
    /// Path relative to the project directory.
    pub relative_path: PathBuf,
    /// File role inferred from its extension.
    pub kind: FileKind,
    /// Original bytes. Source consumers may call [`Self::text`].
    pub contents: Vec<u8>,
    compiler_contents: Option<Vec<u8>>,
    compiler_source_map: Vec<SourceMapping>,
}

impl ProjectFile {
    /// Returns UTF-8 source text.
    ///
    /// # Errors
    ///
    /// Returns [`str::Utf8Error`] if this file is not UTF-8 text.
    pub fn text(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(&self.contents)
    }

    /// Returns source suitable for syntax parsing after conditional masking.
    ///
    /// Preprocessor directive lines and source in inactive conditional branches
    /// are replaced with same-length ASCII whitespace. Line endings and all
    /// active ordinary source bytes are preserved, so offsets in the returned
    /// text refer directly to [`Self::contents`]. Files without directives
    /// borrow the original byte buffer without making a copy.
    ///
    /// This stage intentionally does not perform general macro replacement.
    ///
    /// # Errors
    ///
    /// Returns [`str::Utf8Error`] if this file is not UTF-8 text.
    pub fn compiler_text(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(self.compiler_contents.as_deref().unwrap_or(&self.contents))
    }

    /// Maps a compiler-view byte range back to this file's original bytes.
    ///
    /// Bytes copied from ordinary source map one-to-one. Text produced by a
    /// macro replacement maps to the complete identifier that invoked the
    /// macro. A range spanning several mapping entries covers their combined
    /// original range.
    #[must_use]
    pub fn original_span(&self, compiler_span: SourceSpan) -> SourceSpan {
        if self.compiler_source_map.is_empty() {
            return compiler_span;
        }
        let start = self.map_compiler_offset(compiler_span.start, false);
        let end = self.map_compiler_offset(compiler_span.end, true);
        SourceSpan::new(start.min(end), end.max(start))
    }

    fn map_compiler_offset(&self, offset: usize, end: bool) -> usize {
        let probe = if end && offset != 0 {
            offset - 1
        } else {
            offset
        };
        let mapping = self
            .compiler_source_map
            .iter()
            .find(|mapping| {
                mapping.expanded.start <= probe
                    && (probe < mapping.expanded.end
                        || (mapping.expanded.is_empty() && probe == mapping.expanded.start))
            })
            .or_else(|| self.compiler_source_map.last());
        let Some(mapping) = mapping else {
            return offset.min(self.contents.len());
        };
        if mapping.expanded.len() == mapping.original.len() {
            let relative = offset
                .saturating_sub(mapping.expanded.start)
                .min(mapping.expanded.len());
            mapping.original.start + relative
        } else if end {
            mapping.original.end
        } else {
            mapping.original.start
        }
        .min(self.contents.len())
    }
}

/// One contiguous relationship between compiler-view and original bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceMapping {
    expanded: SourceSpan,
    original: SourceSpan,
}

/// Role of a file in a DM project.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FileKind {
    /// Root or nested Dream Maker environment source.
    Environment,
    /// Dream Maker language source.
    Source,
    /// Text map source.
    Map,
    /// Interface/skin definition.
    Interface,
    /// Any other resource file.
    Resource,
}

impl FileKind {
    fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("dme") => Self::Environment,
            Some("dm") => Self::Source,
            Some("dmm") => Self::Map,
            Some("dmf") => Self::Interface,
            _ => Self::Resource,
        }
    }

    fn can_contain_includes(self) -> bool {
        matches!(self, Self::Environment | Self::Source)
    }
}

/// One `#include` operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncludeEdge {
    /// File containing the directive.
    pub source: FileId,
    /// Resolved project file or deferred system include.
    pub target: IncludeTarget,
    /// Include spelling between quotes or angle brackets.
    pub spelling: String,
    /// Byte range of the complete directive line.
    pub span: SourceSpan,
}

/// Destination of an include operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IncludeTarget {
    /// A quoted include resolved to a project file.
    File(FileId),
    /// An angle-bracket include supplied by the engine standard library.
    System(String),
}

/// Failure while loading a project or include graph.
#[derive(Debug)]
pub enum ProjectError {
    /// Filesystem operation failed.
    Io {
        /// Path involved in the operation.
        path: PathBuf,
        /// Underlying platform error.
        source: io::Error,
    },
    /// A source-bearing file was not UTF-8.
    InvalidUtf8 {
        /// Invalid source file.
        path: PathBuf,
        /// Decoder error.
        source: str::Utf8Error,
    },
    /// A quoted include did not resolve to a file.
    MissingInclude {
        /// File containing the directive.
        source_file: PathBuf,
        /// Include spelling from the directive.
        spelling: String,
    },
    /// A resolved include escaped the canonical project directory.
    OutsideProject {
        /// File containing the directive.
        source_file: PathBuf,
        /// Canonical resolved target.
        target: PathBuf,
    },
    /// A conditional-compilation directive was malformed.
    Preprocessor {
        /// File containing the directive.
        path: PathBuf,
        /// Byte offset of the directive.
        offset: usize,
        /// Explanation of the malformed directive.
        message: String,
    },
    /// An object-like macro recursively expanded itself or exceeded limits.
    MacroExpansion {
        /// File containing the macro invocation.
        path: PathBuf,
        /// Original byte offset of the invocation.
        offset: usize,
        /// Deterministic expansion failure explanation.
        message: String,
    },
}

impl fmt::Display for ProjectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(formatter, "{}: {source}", path.display()),
            Self::InvalidUtf8 { path, source } => {
                write!(formatter, "{} is not UTF-8: {source}", path.display())
            }
            Self::MissingInclude {
                source_file,
                spelling,
            } => write!(
                formatter,
                "{} includes missing file {spelling:?}",
                source_file.display()
            ),
            Self::OutsideProject {
                source_file,
                target,
            } => write!(
                formatter,
                "{} includes path outside project: {}",
                source_file.display(),
                target.display()
            ),
            Self::Preprocessor {
                path,
                offset,
                message,
            } => write!(
                formatter,
                "{}:{offset}: preprocessor error: {message}",
                path.display()
            ),
            Self::MacroExpansion {
                path,
                offset,
                message,
            } => write!(
                formatter,
                "{}:{offset}: macro expansion error: {message}",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ProjectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::InvalidUtf8 { source, .. } => Some(source),
            Self::MissingInclude { .. }
            | Self::OutsideProject { .. }
            | Self::Preprocessor { .. }
            | Self::MacroExpansion { .. } => None,
        }
    }
}

struct Loader {
    root_file: PathBuf,
    root_directory: PathBuf,
    files: Vec<ProjectFile>,
    includes: Vec<(usize, IncludeEdge)>,
    identities: HashMap<PathBuf, FileId>,
    macros: HashMap<String, MacroDefinition>,
    next_include_ordinal: usize,
    warning_directive_is_error: bool,
    duplicate_include_is_error: bool,
    diagnostic_pragmas: Vec<DiagnosticPragma>,
}

impl Loader {
    fn new(root_file: &Path) -> Result<Self, ProjectError> {
        let root_file = canonicalize(root_file)?;
        let root_directory = root_file
            .parent()
            .expect("a canonical file path has a parent")
            .to_path_buf();
        let macros = HashMap::from([
            (
                "DM_VERSION".to_owned(),
                MacroDefinition::object(TARGET_DM_VERSION.to_string()),
            ),
            (
                "DM_BUILD".to_owned(),
                MacroDefinition::object(TARGET_DM_BUILD.to_string()),
            ),
        ]);
        Ok(Self {
            root_file,
            root_directory,
            files: Vec::new(),
            includes: Vec::new(),
            identities: HashMap::new(),
            macros,
            next_include_ordinal: 0,
            warning_directive_is_error: false,
            duplicate_include_is_error: false,
            diagnostic_pragmas: Vec::new(),
        })
    }

    fn load(mut self) -> Result<Project, ProjectError> {
        let root_file = self.root_file.clone();
        self.load_file(&root_file)?;
        self.includes.sort_by_key(|(ordinal, _)| *ordinal);
        Ok(Project {
            root_directory: self.root_directory,
            files: self.files,
            includes: self.includes.into_iter().map(|(_, edge)| edge).collect(),
            diagnostic_pragmas: self.diagnostic_pragmas,
        })
    }

    fn load_file(&mut self, path: &Path) -> Result<FileId, ProjectError> {
        let path = canonicalize(path)?;
        if let Some(id) = self.identities.get(&path).copied() {
            return Ok(id);
        }
        if !path.starts_with(&self.root_directory) {
            return Err(ProjectError::OutsideProject {
                source_file: self.root_file.clone(),
                target: path,
            });
        }

        let contents = fs::read(&path).map_err(|source| ProjectError::Io {
            path: path.clone(),
            source,
        })?;
        let kind = FileKind::from_path(&path);
        let id = FileId::from_index(self.files.len());
        let relative_path = path
            .strip_prefix(&self.root_directory)
            .expect("project path was checked against its root")
            .to_path_buf();
        self.identities.insert(path.clone(), id);
        self.files.push(ProjectFile {
            id,
            path: path.clone(),
            relative_path,
            kind,
            contents,
            compiler_contents: None,
            compiler_source_map: Vec::new(),
        });

        if kind.can_contain_includes() {
            let text =
                self.files[id.index()]
                    .text()
                    .map_err(|source| ProjectError::InvalidUtf8 {
                        path: path.clone(),
                        source,
                    })?;
            let directives = scan_directives(text);
            self.process_directives(id, &path, directives)?;
        }

        Ok(id)
    }

    #[allow(clippy::too_many_lines)]
    fn process_directives(
        &mut self,
        source: FileId,
        path: &Path,
        directives: Vec<Directive>,
    ) -> Result<(), ProjectError> {
        let original = self.files[source.index()].contents.clone();
        let original_text = str::from_utf8(&original).expect("source UTF-8 was validated");
        let mut compiler_source = CompilerSourceBuilder::new();
        let mut conditionals = Vec::new();
        let mut cursor = 0usize;
        let mut deferred_expansion: Option<(usize, String)> = None;
        for directive in directives {
            let span = directive.span;
            let ordinary_span = SourceSpan::new(cursor, span.start);
            if conditional_active(&conditionals) {
                if let Some((start, mut deferred)) = deferred_expansion.take() {
                    deferred.push_str(&original_text[ordinary_span.start..ordinary_span.end]);
                    match CompilerSourceBuilder::expand_deferred_source(
                        &deferred,
                        &mut self.macros,
                        path,
                    )? {
                        Some(contents) => compiler_source.append_replacement(
                            &contents,
                            SourceSpan::new(start, ordinary_span.end),
                        ),
                        None => deferred_expansion = Some((start, deferred)),
                    }
                } else {
                    let checkpoint = compiler_source.checkpoint();
                    match compiler_source.append_expanded_source(
                        original_text,
                        ordinary_span,
                        &mut self.macros,
                        path,
                    ) {
                        Ok(()) => {}
                        Err(ProjectError::MacroExpansion { ref message, .. })
                            if message == "unterminated function macro invocation" =>
                        {
                            compiler_source.restore(checkpoint);
                            deferred_expansion = Some((
                                ordinary_span.start,
                                original_text[ordinary_span.start..ordinary_span.end].to_owned(),
                            ));
                        }
                        Err(error) => return Err(error),
                    }
                }
            } else {
                if let Some((_, deferred)) = &mut deferred_expansion {
                    deferred.push_str(&masked_text(original_text, ordinary_span));
                } else {
                    compiler_source.append_masked(original_text, ordinary_span);
                }
            }
            if let Some((_, deferred)) = &mut deferred_expansion {
                deferred.push_str(&masked_text(original_text, span));
            } else {
                compiler_source.append_masked(original_text, span);
            }
            cursor = span.end;
            self.process_directive(source, path, directive, &mut conditionals)?;
        }
        if let Some(frame) = conditionals.last() {
            return Err(preprocessor_error(
                path,
                self.files[source.index()].contents.len(),
                format!(
                    "unterminated conditional opened at byte {} ({} conditional directive{} still open)",
                    frame.opening_offset,
                    conditionals.len(),
                    if conditionals.len() == 1 { "" } else { "s" },
                ),
            ));
        }
        let tail = SourceSpan::new(cursor, original.len());
        if conditional_active(&conditionals) {
            if let Some((start, mut deferred)) = deferred_expansion.take() {
                deferred.push_str(&original_text[tail.start..tail.end]);
                let contents = CompilerSourceBuilder::expand_deferred_source(
                    &deferred,
                    &mut self.macros,
                    path,
                )?
                .ok_or_else(|| ProjectError::MacroExpansion {
                    path: path.to_path_buf(),
                    offset: start,
                    message: "unterminated function macro invocation".to_owned(),
                })?;
                compiler_source.append_replacement(&contents, SourceSpan::new(start, tail.end));
            } else {
                compiler_source.append_expanded_source(
                    original_text,
                    tail,
                    &mut self.macros,
                    path,
                )?;
            }
        } else {
            compiler_source.append_masked(original_text, tail);
        }
        if compiler_source.contents != original {
            self.files[source.index()].compiler_contents = Some(compiler_source.contents);
            self.files[source.index()].compiler_source_map = compiler_source.mappings;
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn process_directive(
        &mut self,
        source: FileId,
        path: &Path,
        directive: Directive,
        conditionals: &mut Vec<ConditionalFrame>,
    ) -> Result<(), ProjectError> {
        let active = conditional_active(conditionals);
        let offset = directive.span.start;
        match directive.kind {
            DirectiveKind::If(expression) => {
                let condition = self.evaluate(path, offset, &expression)?;
                conditionals.push(ConditionalFrame::new(active, condition, offset));
            }
            DirectiveKind::Ifdef(name) => conditionals.push(ConditionalFrame::new(
                active,
                self.macros.contains_key(&name),
                offset,
            )),
            DirectiveKind::Ifndef(name) => conditionals.push(ConditionalFrame::new(
                active,
                !self.macros.contains_key(&name),
                offset,
            )),
            DirectiveKind::Elif(expression) => {
                let frame = conditionals.last_mut().ok_or_else(|| {
                    preprocessor_error(path, offset, "#elif without matching #if")
                })?;
                if frame.branch_state == ConditionalBranchState::Else {
                    return Err(preprocessor_error(
                        path,
                        offset,
                        format!(
                            "#elif after #else for conditional opened at byte {}",
                            frame.opening_offset
                        ),
                    ));
                }
                let condition = self.evaluate(path, offset, &expression)?;
                frame.activate_alternative(condition);
            }
            DirectiveKind::Else => {
                let frame = conditionals.last_mut().ok_or_else(|| {
                    preprocessor_error(path, offset, "#else without matching #if")
                })?;
                if frame.branch_state == ConditionalBranchState::Else {
                    return Err(preprocessor_error(
                        path,
                        offset,
                        format!(
                            "duplicate #else for conditional opened at byte {}",
                            frame.opening_offset
                        ),
                    ));
                }
                frame.activate_else();
            }
            DirectiveKind::Endif => {
                conditionals.pop().ok_or_else(|| {
                    preprocessor_error(path, offset, "#endif without matching #if")
                })?;
            }
            DirectiveKind::Define {
                name,
                value,
                parameters,
            } if active => {
                if parameters
                    .as_ref()
                    .is_some_and(|parameters| !parameters.valid)
                {
                    return Err(preprocessor_error(
                        path,
                        offset,
                        "variadic macro parameter must be last",
                    ));
                }
                self.macros.insert(
                    name,
                    MacroDefinition {
                        replacement: value,
                        parameters,
                    },
                );
            }
            DirectiveKind::Undef(name) if active => {
                self.macros.remove(&name);
            }
            DirectiveKind::Include {
                spelling,
                delimiter,
            } if active => {
                self.process_include(source, path, spelling, delimiter, directive.span)?;
            }
            DirectiveKind::Error(message) if active => {
                return Err(preprocessor_error(
                    path,
                    offset,
                    format!("#error{}", directive_message_suffix(&message)),
                ));
            }
            DirectiveKind::Warning(message) if active && self.warning_directive_is_error => {
                return Err(preprocessor_error(
                    path,
                    offset,
                    format!("#warn{}", directive_message_suffix(&message)),
                ));
            }
            DirectiveKind::Pragma(value) if active => {
                self.apply_pragma(source, directive.span, &value);
            }
            DirectiveKind::Warning(_)
            | DirectiveKind::Pragma(_)
            | DirectiveKind::Define { .. }
            | DirectiveKind::Undef(_)
            | DirectiveKind::Include { .. }
            | DirectiveKind::Error(_) => {}
            DirectiveKind::Malformed(message) => {
                return Err(preprocessor_error(path, offset, message));
            }
        }
        Ok(())
    }

    fn apply_pragma(&mut self, source: FileId, span: SourceSpan, value: &str) {
        let mut words = value.split_whitespace();
        let name = words.next();
        let severity = words.next();
        match (name, severity) {
            (Some("WarningDirective"), Some("error")) => {
                self.warning_directive_is_error = true;
            }
            (Some("WarningDirective"), Some("warning" | "disabled")) => {
                self.warning_directive_is_error = false;
            }
            (Some("FileAlreadyIncluded"), Some("error")) => {
                self.duplicate_include_is_error = true;
            }
            (Some("FileAlreadyIncluded"), Some("warning" | "disabled")) => {
                self.duplicate_include_is_error = false;
            }
            _ => {}
        }
        let severity = match severity {
            Some("disabled") => Some(PragmaSeverity::Disabled),
            Some("notice" | "info") => Some(PragmaSeverity::Notice),
            Some("warning" | "warn") => Some(PragmaSeverity::Warning),
            Some("error") => Some(PragmaSeverity::Error),
            _ => None,
        };
        if let (Some(name), Some(severity)) = (name, severity) {
            self.diagnostic_pragmas.push(DiagnosticPragma {
                name: name.to_owned(),
                severity,
                source,
                span,
            });
        }
    }

    fn process_include(
        &mut self,
        source: FileId,
        path: &Path,
        spelling: String,
        delimiter: IncludeDelimiter,
        span: SourceSpan,
    ) -> Result<(), ProjectError> {
        let ordinal = self.next_include_ordinal;
        self.next_include_ordinal += 1;
        let target = match delimiter {
            IncludeDelimiter::System => IncludeTarget::System(spelling.clone()),
            IncludeDelimiter::Quoted => {
                let target_path = self.resolve_quoted(path, &spelling)?;
                if self.duplicate_include_is_error && self.identities.contains_key(&target_path) {
                    return Err(preprocessor_error(
                        path,
                        span.start,
                        format!("file already included: {spelling:?}"),
                    ));
                }
                IncludeTarget::File(self.load_file(&target_path)?)
            }
        };
        self.includes.push((
            ordinal,
            IncludeEdge {
                source,
                target,
                spelling,
                span,
            },
        ));
        Ok(())
    }

    fn evaluate(&self, path: &Path, offset: usize, expression: &str) -> Result<bool, ProjectError> {
        ConditionParser::new(expression, &self.macros, path, &self.root_directory)
            .parse()
            .map(|value| value != 0)
            .map_err(|message| preprocessor_error(path, offset, message))
    }

    fn resolve_quoted(&self, source_file: &Path, spelling: &str) -> Result<PathBuf, ProjectError> {
        let portable = spelling.replace(['\\', '/'], std::path::MAIN_SEPARATOR_STR);
        let include_path = Path::new(&portable);
        let source_candidate = source_file
            .parent()
            .expect("a loaded source file has a parent")
            .join(include_path);
        let root_candidate = self.root_directory.join(include_path);
        let candidate = if source_candidate.is_file() {
            source_candidate
        } else if root_candidate.is_file() {
            root_candidate
        } else {
            return Err(ProjectError::MissingInclude {
                source_file: source_file.to_path_buf(),
                spelling: spelling.to_owned(),
            });
        };
        let canonical = canonicalize(&candidate)?;
        if !canonical.starts_with(&self.root_directory) {
            return Err(ProjectError::OutsideProject {
                source_file: source_file.to_path_buf(),
                target: canonical,
            });
        }
        Ok(canonical)
    }
}

fn canonicalize(path: &Path) -> Result<PathBuf, ProjectError> {
    path.canonicalize().map_err(|source| ProjectError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[derive(Clone, Copy)]
enum IncludeDelimiter {
    Quoted,
    System,
}

struct Directive {
    kind: DirectiveKind,
    span: SourceSpan,
}

enum DirectiveKind {
    Include {
        spelling: String,
        delimiter: IncludeDelimiter,
    },
    Define {
        name: String,
        value: String,
        parameters: Option<MacroParameters>,
    },
    Undef(String),
    If(String),
    Ifdef(String),
    Ifndef(String),
    Elif(String),
    Else,
    Endif,
    Error(String),
    Warning(String),
    Pragma(String),
    Malformed(String),
}

fn directive_message_suffix(message: &str) -> String {
    let message = message.trim();
    if message.is_empty() {
        String::new()
    } else {
        format!(" {message}")
    }
}

#[derive(Clone, Debug)]
struct MacroDefinition {
    replacement: String,
    parameters: Option<MacroParameters>,
}

#[derive(Clone, Debug)]
struct MacroParameters {
    fixed: Vec<String>,
    variadic: Option<String>,
    valid: bool,
}

impl MacroDefinition {
    fn object(replacement: String) -> Self {
        Self {
            replacement,
            parameters: None,
        }
    }
}

struct ConditionalFrame {
    parent_active: bool,
    active: bool,
    branch_state: ConditionalBranchState,
    opening_offset: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ConditionalBranchState {
    NoneTaken,
    Taken,
    Else,
}

impl ConditionalFrame {
    fn new(parent_active: bool, condition: bool, opening_offset: usize) -> Self {
        Self {
            parent_active,
            active: parent_active && condition,
            branch_state: if condition {
                ConditionalBranchState::Taken
            } else {
                ConditionalBranchState::NoneTaken
            },
            opening_offset,
        }
    }

    fn activate_alternative(&mut self, condition: bool) {
        let selected = self.branch_state == ConditionalBranchState::NoneTaken && condition;
        self.active = self.parent_active && selected;
        if selected {
            self.branch_state = ConditionalBranchState::Taken;
        }
    }

    fn activate_else(&mut self) {
        self.active = self.parent_active && self.branch_state == ConditionalBranchState::NoneTaken;
        self.branch_state = ConditionalBranchState::Else;
    }
}

fn conditional_active(conditionals: &[ConditionalFrame]) -> bool {
    conditionals
        .last()
        .is_none_or(|frame: &ConditionalFrame| frame.active)
}

struct CompilerSourceBuilder {
    contents: Vec<u8>,
    mappings: Vec<SourceMapping>,
}

impl CompilerSourceBuilder {
    const fn new() -> Self {
        Self {
            contents: Vec::new(),
            mappings: Vec::new(),
        }
    }

    fn checkpoint(&self) -> (usize, usize) {
        (self.contents.len(), self.mappings.len())
    }

    fn restore(&mut self, checkpoint: (usize, usize)) {
        self.contents.truncate(checkpoint.0);
        self.mappings.truncate(checkpoint.1);
    }

    fn expand_deferred_source(
        source: &str,
        macros: &mut HashMap<String, MacroDefinition>,
        path: &Path,
    ) -> Result<Option<String>, ProjectError> {
        let mut expanded = Self::new();
        match expanded.append_expanded_source(
            source,
            SourceSpan::new(0, source.len()),
            macros,
            path,
        ) {
            Ok(()) => Ok(Some(
                String::from_utf8(expanded.contents)
                    .expect("macro expansion preserves UTF-8 source"),
            )),
            Err(ProjectError::MacroExpansion { ref message, .. })
                if message == "unterminated function macro invocation" =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn append_expanded_source(
        &mut self,
        source: &str,
        span: SourceSpan,
        macros: &mut HashMap<String, MacroDefinition>,
        path: &Path,
    ) -> Result<(), ProjectError> {
        let file_macro = path.to_string_lossy().replace('\\', "/");
        let text = &source[span.start..span.end];
        let mut offset = 0usize;
        let mut literal_start = 0usize;
        while offset < text.len() {
            if let Some(end) = protected_text_end(text, offset) {
                offset = end;
                continue;
            }
            let byte = text.as_bytes()[offset];
            if is_identifier_start(byte) {
                let identifier_end = identifier_end(text, offset);
                let name = &text[offset..identifier_end];
                if matches!(name, "__FILE__" | "__LINE__") {
                    self.append_original(
                        source,
                        SourceSpan::new(span.start + literal_start, span.start + offset),
                    );
                    let invocation =
                        SourceSpan::new(span.start + offset, span.start + identifier_end);
                    let replacement = if name == "__FILE__" {
                        format!("{file_macro:?}")
                    } else {
                        source_line_number(source, invocation.start).to_string()
                    };
                    self.append_replacement(&replacement, invocation);
                    offset = identifier_end;
                    literal_start = offset;
                    continue;
                }
                if let Some(definition) = macros.get(name).cloned() {
                    let invocation_end = if definition.parameters.is_some() {
                        let open = skip_horizontal_whitespace(text, identifier_end);
                        if text.as_bytes().get(open) != Some(&b'(') {
                            offset = identifier_end;
                            continue;
                        }
                        parse_macro_arguments(text, open)
                            .map_err(|message| ProjectError::MacroExpansion {
                                path: path.to_path_buf(),
                                offset: span.start + offset,
                                message,
                            })?
                            .1
                    } else {
                        identifier_end
                    };
                    self.append_original(
                        source,
                        SourceSpan::new(span.start + literal_start, span.start + offset),
                    );
                    let invocation =
                        SourceSpan::new(span.start + offset, span.start + invocation_end);
                    let arguments = definition.parameters.as_ref().map(|_| {
                        let open = skip_horizontal_whitespace(text, identifier_end);
                        parse_macro_arguments(text, open)
                            .expect("arguments were validated above")
                            .0
                    });
                    let line_macro = source_line_number(source, invocation.start);
                    let replacement = expand_macro(
                        name,
                        arguments.as_deref(),
                        macros,
                        &mut Vec::new(),
                        &file_macro,
                        line_macro,
                    )
                    .map_err(|message| ProjectError::MacroExpansion {
                        path: path.to_path_buf(),
                        offset: invocation.start,
                        message,
                    })?;
                    let line_end = text[invocation_end..]
                        .find(['\r', '\n'])
                        .map_or(text.len(), |relative| invocation_end + relative);
                    if replacement.trim() == "#define" {
                        let generated = format!("#define{}", &text[invocation_end..line_end]);
                        apply_generated_define(&generated, macros).map_err(|message| {
                            ProjectError::MacroExpansion {
                                path: path.to_path_buf(),
                                offset: invocation.start,
                                message,
                            }
                        })?;
                        offset = line_end;
                    } else if let Some((visible, definition)) = split_generated_define(&replacement)
                    {
                        self.append_replacement(visible, invocation);
                        apply_generated_define(definition, macros).map_err(|message| {
                            ProjectError::MacroExpansion {
                                path: path.to_path_buf(),
                                offset: invocation.start,
                                message,
                            }
                        })?;
                        offset = invocation_end;
                    } else {
                        self.append_replacement(&replacement, invocation);
                        offset = invocation_end;
                    }
                    literal_start = offset;
                    continue;
                }
                offset = identifier_end;
                continue;
            }
            offset += text[offset..]
                .chars()
                .next()
                .expect("offset is inside source text")
                .len_utf8();
        }
        self.append_original(
            source,
            SourceSpan::new(span.start + literal_start, span.end),
        );
        Ok(())
    }

    fn append_original(&mut self, source: &str, span: SourceSpan) {
        if span.is_empty() {
            return;
        }
        let expanded_start = self.contents.len();
        self.contents
            .extend_from_slice(&source.as_bytes()[span.start..span.end]);
        self.mappings.push(SourceMapping {
            expanded: SourceSpan::new(expanded_start, self.contents.len()),
            original: span,
        });
    }

    fn append_masked(&mut self, source: &str, span: SourceSpan) {
        if span.is_empty() {
            return;
        }
        let expanded_start = self.contents.len();
        self.contents
            .extend(source.as_bytes()[span.start..span.end].iter().map(|byte| {
                if matches!(*byte, b'\r' | b'\n') {
                    *byte
                } else {
                    b' '
                }
            }));
        self.mappings.push(SourceMapping {
            expanded: SourceSpan::new(expanded_start, self.contents.len()),
            original: span,
        });
    }

    fn append_replacement(&mut self, replacement: &str, invocation: SourceSpan) {
        if replacement.is_empty() {
            return;
        }
        let expanded_start = self.contents.len();
        self.contents.extend_from_slice(replacement.as_bytes());
        self.mappings.push(SourceMapping {
            expanded: SourceSpan::new(expanded_start, self.contents.len()),
            original: invocation,
        });
    }
}

fn masked_text(source: &str, span: SourceSpan) -> String {
    source.as_bytes()[span.start..span.end]
        .iter()
        .map(|byte| {
            if matches!(*byte, b'\r' | b'\n') {
                char::from(*byte)
            } else {
                ' '
            }
        })
        .collect()
}

fn split_generated_define(expansion: &str) -> Option<(&str, &str)> {
    let mut offset = 0usize;
    while offset < expansion.len() {
        if let Some(end) = protected_text_end(expansion, offset) {
            offset = end;
            continue;
        }
        if expansion.as_bytes()[offset] == b'#' {
            let after_hash = skip_horizontal_whitespace(expansion, offset + 1);
            if expansion[after_hash..].starts_with("define")
                && expansion[after_hash + "define".len()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            {
                return Some((&expansion[..offset], &expansion[offset..]));
            }
        }
        offset += expansion[offset..]
            .chars()
            .next()
            .expect("offset is inside macro expansion")
            .len_utf8();
    }
    None
}

fn apply_generated_define(
    directive: &str,
    macros: &mut HashMap<String, MacroDefinition>,
) -> Result<(), String> {
    let Some(kind) = parse_directive_line(directive) else {
        return Err("macro generated a malformed #define directive".to_owned());
    };
    let DirectiveKind::Define {
        name,
        value,
        parameters,
    } = kind
    else {
        return Err("macro generated a non-define directive".to_owned());
    };
    macros.insert(
        name,
        MacroDefinition {
            replacement: value,
            parameters,
        },
    );
    Ok(())
}

fn source_line_number(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())]
        .matches('\n')
        .count()
        .saturating_add(1)
}

const MAX_MACRO_EXPANSION_DEPTH: usize = 64;

fn expand_macro(
    name: &str,
    arguments: Option<&[String]>,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
    if name == "__FILE__" {
        return Ok(format!("{file_macro:?}"));
    }
    if name == "__LINE__" {
        return Ok(line_macro.to_string());
    }
    if let Some(cycle_start) = stack.iter().position(|entry| entry == name) {
        let mut cycle = stack[cycle_start..].to_vec();
        cycle.push(name.to_owned());
        return Err(format!("recursive macro expansion: {}", cycle.join(" -> ")));
    }
    if stack.len() >= MAX_MACRO_EXPANSION_DEPTH {
        return Err(format!(
            "macro expansion exceeded {MAX_MACRO_EXPANSION_DEPTH} levels while expanding {name}"
        ));
    }
    let Some(definition) = macros.get(name) else {
        return Ok(name.to_owned());
    };
    stack.push(name.to_owned());
    let result = if let Some(parameters) = &definition.parameters {
        arguments.map_or_else(
            || Err(format!("function macro {name} requires a call")),
            |arguments| {
                substitute_function_macro(
                    name, definition, parameters, arguments, macros, stack, file_macro, line_macro,
                )
            },
        )
    } else {
        expand_replacement(
            &definition.replacement,
            macros,
            stack,
            file_macro,
            line_macro,
        )
    };
    stack.pop();
    result
}

fn expand_replacement(
    replacement: &str,
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
    let mut output = String::with_capacity(replacement.len());
    let mut offset = 0usize;
    while offset < replacement.len() {
        if let Some(end) = protected_text_end(replacement, offset) {
            output.push_str(&replacement[offset..end]);
            offset = end;
            continue;
        }
        let byte = replacement.as_bytes()[offset];
        if is_identifier_start(byte) {
            let end = identifier_end(replacement, offset);
            let name = &replacement[offset..end];
            if name == "__FILE__" {
                let file_literal = format!("{file_macro:?}");
                output.push_str(&file_literal);
                offset = end;
                continue;
            }
            if name == "__LINE__" {
                output.push_str(&line_macro.to_string());
                offset = end;
                continue;
            }
            if let Some(definition) = macros.get(name) {
                if definition.parameters.is_some() {
                    let open = skip_horizontal_whitespace(replacement, end);
                    if replacement.as_bytes().get(open) == Some(&b'(') {
                        let (arguments, invocation_end) = parse_macro_arguments(replacement, open)?;
                        if stack.iter().any(|active| active == name) {
                            output.push_str(&replacement[offset..invocation_end]);
                        } else {
                            output.push_str(&expand_macro(
                                name,
                                Some(&arguments),
                                macros,
                                stack,
                                file_macro,
                                line_macro,
                            )?);
                        }
                        offset = invocation_end;
                        continue;
                    }
                    output.push_str(name);
                } else {
                    output.push_str(&expand_macro(
                        name, None, macros, stack, file_macro, line_macro,
                    )?);
                }
            } else {
                output.push_str(name);
            }
            offset = end;
            continue;
        }
        let character = replacement[offset..]
            .chars()
            .next()
            .expect("offset is inside macro replacement text");
        output.push(character);
        offset += character.len_utf8();
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn substitute_function_macro(
    name: &str,
    definition: &MacroDefinition,
    parameters: &MacroParameters,
    arguments: &[String],
    macros: &HashMap<String, MacroDefinition>,
    stack: &mut Vec<String>,
    file_macro: &str,
    line_macro: usize,
) -> Result<String, String> {
    if arguments.len() < parameters.fixed.len()
        || (parameters.variadic.is_none() && arguments.len() > parameters.fixed.len())
    {
        return Err(format!(
            "function macro {name} expects {} argument{} but received {}",
            parameters.fixed.len(),
            if parameters.fixed.len() == 1 { "" } else { "s" },
            arguments.len()
        ));
    }
    let mut substitutions = HashMap::new();
    for (parameter, argument) in parameters.fixed.iter().zip(arguments) {
        substitutions.insert(parameter.as_str(), splice_continuations(argument.trim()));
    }
    if let Some(variadic) = &parameters.variadic {
        substitutions.insert(
            variadic.as_str(),
            arguments[parameters.fixed.len()..]
                .iter()
                .map(|argument| splice_continuations(argument.trim()))
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    let replacement = &definition.replacement;
    let mut substituted = String::with_capacity(replacement.len());
    let mut offset = 0usize;
    while offset < replacement.len() {
        if let Some(end) = protected_text_end(replacement, offset) {
            substituted.push_str(&replacement[offset..end]);
            offset = end;
            continue;
        }
        if replacement[offset..].starts_with("##") {
            offset += 2;
            continue;
        }
        if replacement.as_bytes()[offset] == b'#' {
            let parameter_start = skip_horizontal_whitespace(replacement, offset + 1);
            if replacement
                .as_bytes()
                .get(parameter_start)
                .is_some_and(|byte| is_identifier_start(*byte))
            {
                let parameter_end = identifier_end(replacement, parameter_start);
                let parameter = &replacement[parameter_start..parameter_end];
                if let Some(argument) = substitutions.get(parameter) {
                    substituted.push('"');
                    for character in argument.chars() {
                        if matches!(character, '\\' | '"') {
                            substituted.push('\\');
                        }
                        substituted.push(character);
                    }
                    substituted.push('"');
                    offset = parameter_end;
                    continue;
                }
            }
        }
        if is_identifier_start(replacement.as_bytes()[offset]) {
            let end = identifier_end(replacement, offset);
            let parameter = &replacement[offset..end];
            if let Some(argument) = substitutions.get(parameter) {
                substituted.push_str(argument);
            } else {
                substituted.push_str(parameter);
            }
            offset = end;
            continue;
        }
        let character = replacement[offset..]
            .chars()
            .next()
            .expect("offset is inside function replacement text");
        substituted.push(character);
        offset += character.len_utf8();
    }
    expand_replacement(&substituted, macros, stack, file_macro, line_macro)
}

fn parse_macro_arguments(source: &str, open: usize) -> Result<(Vec<String>, usize), String> {
    let mut arguments = Vec::new();
    let mut stack = vec![b')'];
    let mut offset = open + 1;
    let mut argument_start = offset;
    while offset < source.len() {
        if let Some(end) = protected_text_end(source, offset) {
            offset = end;
            continue;
        }
        let byte = source.as_bytes()[offset];
        match byte {
            b'(' => stack.push(b')'),
            b'[' => stack.push(b']'),
            b'{' => stack.push(b'}'),
            b')' | b']' | b'}' if stack.last() == Some(&byte) => {
                stack.pop();
                if stack.is_empty() {
                    let final_argument = source[argument_start..offset].trim();
                    if !final_argument.is_empty() || !arguments.is_empty() {
                        arguments.push(final_argument.to_owned());
                    }
                    return Ok((arguments, offset + 1));
                }
            }
            b',' if stack.len() == 1 => {
                arguments.push(source[argument_start..offset].trim().to_owned());
                argument_start = offset + 1;
            }
            _ => {}
        }
        offset += source[offset..]
            .chars()
            .next()
            .expect("offset is inside macro invocation")
            .len_utf8();
    }
    Err("unterminated function macro invocation".to_owned())
}

fn skip_horizontal_whitespace(source: &str, start: usize) -> usize {
    start
        + source[start..]
            .chars()
            .take_while(|character| matches!(*character, ' ' | '\t' | '\r' | '\n'))
            .map(char::len_utf8)
            .sum::<usize>()
}

fn protected_text_end(source: &str, offset: usize) -> Option<usize> {
    let remaining = &source[offset..];
    if remaining.starts_with("//") {
        return Some(
            remaining
                .find('\n')
                .map_or(source.len(), |end| offset + end),
        );
    }
    if let Some(comment) = remaining.strip_prefix("/*") {
        return Some(
            comment
                .find("*/")
                .map_or(source.len(), |end| offset + 2 + end + 2),
        );
    }
    if let Some(raw_string) = remaining.strip_prefix("{\"") {
        return Some(
            raw_string
                .find("\"}")
                .map_or(source.len(), |end| offset + 2 + end + 2),
        );
    }
    if matches!(source.as_bytes()[offset], b'\"' | b'\'') {
        return Some(quoted_text_end(source, offset, source.as_bytes()[offset]));
    }
    None
}

fn quoted_text_end(source: &str, start: usize, delimiter: u8) -> usize {
    let mut cursor = start + 1;
    let mut escaped = false;
    while cursor < source.len() {
        let byte = source.as_bytes()[cursor];
        if escaped {
            escaped = false;
            cursor += 1;
            continue;
        }
        if byte == b'\\' {
            escaped = true;
            cursor += 1;
            continue;
        }
        if delimiter == b'"' && byte == b'[' {
            cursor = interpolation_end(source, cursor + 1);
            continue;
        }
        cursor += 1;
        if byte == delimiter {
            break;
        }
    }
    cursor
}

fn interpolation_end(source: &str, start: usize) -> usize {
    let mut cursor = start;
    let mut depth = 1usize;
    while cursor < source.len() {
        let byte = source.as_bytes()[cursor];
        if matches!(byte, b'"' | b'\'') {
            cursor = quoted_text_end(source, cursor, byte);
            continue;
        }
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                cursor += 1;
                if depth == 0 {
                    return cursor;
                }
                continue;
            }
            _ => {}
        }
        cursor += source[cursor..]
            .chars()
            .next()
            .expect("offset is inside string interpolation")
            .len_utf8();
    }
    source.len()
}

const fn is_identifier_start(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn identifier_end(source: &str, start: usize) -> usize {
    start
        + source.as_bytes()[start..]
            .iter()
            .take_while(|byte| byte.is_ascii_alphanumeric() || **byte == b'_')
            .count()
}

fn scan_directives(source: &str) -> Vec<Directive> {
    let mut directives = Vec::new();
    let mut offset = 0;
    let mut in_block_comment = false;
    while offset < source.len() {
        let physical_end = physical_line_end(source, offset);
        let physical_line = &source[offset..physical_end];
        let line = physical_line.strip_suffix('\n').unwrap_or(physical_line);
        let uncommented = remove_comments(line, &mut in_block_comment);
        if let Some(mut kind) = parse_directive_line(&uncommented) {
            let directive_end = if matches!(&kind, DirectiveKind::Define { .. }) {
                extended_define_end(source, offset, physical_end, &uncommented)
            } else {
                physical_end
            };
            if let DirectiveKind::Define {
                name,
                value,
                parameters,
            } = &mut kind
            {
                let (complete_value, complete_parameters) =
                    complete_define(source, offset, directive_end, name);
                *value = complete_value;
                *parameters = complete_parameters;
            }
            directives.push(Directive {
                kind,
                span: SourceSpan::new(offset, directive_end),
            });
            offset = directive_end;
        } else {
            offset = physical_end;
        }
    }
    directives
}

fn complete_define(
    source: &str,
    start: usize,
    end: usize,
    name: &str,
) -> (String, Option<MacroParameters>) {
    let directive = source[start..end]
        .trim_end_matches(['\r', '\n'])
        .trim_start();
    let hash = directive.find('#').expect("a parsed directive contains #");
    let after_keyword = directive[hash + 1..]
        .trim_start()
        .strip_prefix("define")
        .expect("a define directive has the define keyword")
        .trim_start();
    let after_name = after_keyword
        .strip_prefix(name)
        .expect("parsed macro name is present in its directive");
    let (replacement, parameters) = if after_name.starts_with('(') {
        let parameters_end = after_name
            .find(')')
            .map_or(after_name.len(), |index| index + 1);
        (
            &after_name[parameters_end..],
            Some(parse_macro_parameters(
                &after_name[1..parameters_end.saturating_sub(1)],
            )),
        )
    } else {
        (after_name, None)
    };
    let replacement = replacement.trim_start();
    if replacement.starts_with("{\"") {
        (replacement.to_owned(), parameters)
    } else {
        (
            strip_macro_comments(&splice_continuations(replacement)),
            parameters,
        )
    }
}

fn parse_macro_parameters(source: &str) -> MacroParameters {
    let mut fixed = Vec::new();
    let mut variadic = None;
    let mut valid = true;
    for parameter in source
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        if variadic.is_some() {
            valid = false;
        }
        if parameter == "..." {
            variadic = Some("__VA_ARGS__".to_owned());
        } else if let Some(name) = parameter.strip_suffix("...") {
            variadic = Some(name.trim().to_owned());
        } else {
            fixed.push(parameter.to_owned());
        }
    }
    MacroParameters {
        fixed,
        variadic,
        valid,
    }
}

fn splice_continuations(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut offset = 0usize;
    while offset < bytes.len() {
        if bytes[offset] == b'\\' {
            if offset + 1 == bytes.len() {
                offset += 1;
                continue;
            }
            if bytes.get(offset + 1) == Some(&b'\n') {
                offset += 2;
                while matches!(bytes.get(offset), Some(b' ' | b'\t')) {
                    offset += 1;
                }
                continue;
            }
            if bytes.get(offset + 1) == Some(&b'\r') && bytes.get(offset + 2) == Some(&b'\n') {
                offset += 3;
                while matches!(bytes.get(offset), Some(b' ' | b'\t')) {
                    offset += 1;
                }
                continue;
            }
        }
        output.push(bytes[offset]);
        offset += 1;
    }
    String::from_utf8(output).expect("removing ASCII continuations preserves UTF-8")
}

fn strip_macro_comments(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut in_block_comment = false;
    for physical_line in source.split_inclusive('\n') {
        let line = physical_line.strip_suffix('\n').unwrap_or(physical_line);
        output.push_str(&remove_comments(line, &mut in_block_comment));
        if physical_line.ends_with('\n') {
            output.push('\n');
        }
    }
    output.trim_end().to_owned()
}

fn extended_define_end(
    source: &str,
    line_start: usize,
    initial_end: usize,
    uncommented: &str,
) -> usize {
    if let Some(raw_start) = uncommented.find("{\"") {
        let original_raw_start = line_start + raw_start;
        if let Some(relative_end) = source[original_raw_start + 2..].find("\"}") {
            let raw_end = original_raw_start + 2 + relative_end + 2;
            return physical_line_end(source, raw_end);
        }
        return source.len();
    }

    let mut end = initial_end;
    let mut current_start = line_start;
    while source[current_start..end]
        .trim_end_matches(['\r', '\n'])
        .trim_end()
        .ends_with('\\')
        && end < source.len()
    {
        current_start = end;
        end = physical_line_end(source, current_start);
    }
    end
}

fn physical_line_end(source: &str, start: usize) -> usize {
    source[start..]
        .find('\n')
        .map_or(source.len(), |relative| start + relative + 1)
}

fn remove_comments(line: &str, in_block_comment: &mut bool) -> String {
    let bytes = line.as_bytes();
    let mut uncommented = Vec::with_capacity(bytes.len());
    let mut offset = 0;
    let mut quote = None;
    let mut escaped = false;

    while offset < bytes.len() {
        if *in_block_comment {
            if bytes[offset..].starts_with(b"*/") {
                *in_block_comment = false;
                offset += 2;
                uncommented.push(b' ');
            } else {
                offset += 1;
            }
            continue;
        }

        if let Some(delimiter) = quote {
            let byte = bytes[offset];
            uncommented.push(byte);
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

        if bytes[offset..].starts_with(b"//") {
            break;
        }
        if bytes[offset..].starts_with(b"/*") {
            *in_block_comment = true;
            offset += 2;
            uncommented.push(b' ');
            continue;
        }

        let byte = bytes[offset];
        if matches!(byte, b'\"' | b'\'') {
            quote = Some(byte);
        }
        uncommented.push(byte);
        offset += 1;
    }

    String::from_utf8(uncommented).expect("removing ASCII comment bytes preserves UTF-8")
}

fn parse_directive_line(line: &str) -> Option<DirectiveKind> {
    let directive = line.trim_start().strip_prefix('#')?.trim_start();
    let keyword_end = directive
        .find(char::is_whitespace)
        .unwrap_or(directive.len());
    let keyword = &directive[..keyword_end];
    let value = directive[keyword_end..].trim_start();
    match keyword {
        "include" => parse_include(value),
        "define" => {
            let name_end = value
                .find(|character: char| character.is_whitespace() || character == '(')
                .unwrap_or(value.len());
            let name = value[..name_end].to_owned();
            if name.is_empty() {
                return None;
            }
            let macro_value = if value[name_end..].starts_with('(') {
                "1"
            } else {
                value[name_end..].trim_start()
            };
            Some(DirectiveKind::Define {
                name,
                value: macro_value.to_owned(),
                parameters: value[name_end..].starts_with('(').then(|| MacroParameters {
                    fixed: Vec::new(),
                    variadic: None,
                    valid: true,
                }),
            })
        }
        "undef" => identifier(value).map(DirectiveKind::Undef),
        "if" => Some(if value.is_empty() {
            DirectiveKind::Malformed("#if requires a conditional expression".to_owned())
        } else {
            DirectiveKind::If(value.to_owned())
        }),
        "ifdef" => Some(parse_conditional_identifier(
            "#ifdef",
            value,
            DirectiveKind::Ifdef,
        )),
        "ifndef" => Some(parse_conditional_identifier(
            "#ifndef",
            value,
            DirectiveKind::Ifndef,
        )),
        "elif" => Some(if value.is_empty() {
            DirectiveKind::Malformed("#elif requires a conditional expression".to_owned())
        } else {
            DirectiveKind::Elif(value.to_owned())
        }),
        "else" => Some(if value.is_empty() {
            DirectiveKind::Else
        } else {
            DirectiveKind::Malformed("#else does not accept arguments".to_owned())
        }),
        "endif" => Some(if value.is_empty() {
            DirectiveKind::Endif
        } else {
            DirectiveKind::Malformed("#endif does not accept arguments".to_owned())
        }),
        "error" => Some(DirectiveKind::Error(value.to_owned())),
        "warn" => Some(DirectiveKind::Warning(value.to_owned())),
        "pragma" => Some(DirectiveKind::Pragma(value.to_owned())),
        _ => None,
    }
}

fn parse_conditional_identifier(
    directive: &str,
    value: &str,
    constructor: impl FnOnce(String) -> DirectiveKind,
) -> DirectiveKind {
    let Some(name) = identifier(value) else {
        return DirectiveKind::Malformed(format!("{directive} requires a macro name"));
    };
    if !value[name.len()..].trim().is_empty() {
        return DirectiveKind::Malformed(format!("{directive} accepts exactly one macro name"));
    }
    constructor(name)
}

fn parse_include(value: &str) -> Option<DirectiveKind> {
    let (spelling, delimiter) = if let Some(quoted) = value.strip_prefix('"') {
        let end = quoted.find('"')?;
        (&quoted[..end], IncludeDelimiter::Quoted)
    } else {
        let system = value.strip_prefix('<')?;
        let end = system.find('>')?;
        (&system[..end], IncludeDelimiter::System)
    };
    Some(DirectiveKind::Include {
        spelling: spelling.to_owned(),
        delimiter,
    })
}

fn identifier(value: &str) -> Option<String> {
    let end = value
        .find(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .unwrap_or(value.len());
    (end > 0).then(|| value[..end].to_owned())
}

fn preprocessor_error(path: &Path, offset: usize, message: impl Into<String>) -> ProjectError {
    ProjectError::Preprocessor {
        path: path.to_path_buf(),
        offset,
        message: message.into(),
    }
}

struct ConditionParser<'a> {
    source: &'a str,
    offset: usize,
    macros: &'a HashMap<String, MacroDefinition>,
    source_file: &'a Path,
    root_directory: &'a Path,
}

impl<'a> ConditionParser<'a> {
    fn new(
        source: &'a str,
        macros: &'a HashMap<String, MacroDefinition>,
        source_file: &'a Path,
        root_directory: &'a Path,
    ) -> Self {
        Self {
            source: source.split("//").next().unwrap_or(source),
            offset: 0,
            macros,
            source_file,
            root_directory,
        }
    }

    fn parse(mut self) -> Result<i64, String> {
        let value = self.parse_or()?;
        self.skip_whitespace();
        if self.offset != self.source.len() {
            return Err(format!(
                "unsupported conditional expression near {:?}",
                &self.source[self.offset..]
            ));
        }
        Ok(value)
    }

    fn parse_or(&mut self) -> Result<i64, String> {
        let mut value = self.parse_and()?;
        while self.consume("||") {
            let right = self.parse_and()?;
            value = i64::from(value != 0 || right != 0);
        }
        Ok(value)
    }

    fn parse_and(&mut self) -> Result<i64, String> {
        let mut value = self.parse_comparison()?;
        while self.consume("&&") {
            let right = self.parse_comparison()?;
            value = i64::from(value != 0 && right != 0);
        }
        Ok(value)
    }

    fn parse_comparison(&mut self) -> Result<i64, String> {
        let left = self.parse_unary()?;
        let operation = ["==", "!=", "<=", ">=", "<", ">"]
            .into_iter()
            .find(|operation| self.consume(operation));
        let Some(operation) = operation else {
            return Ok(left);
        };
        let right = self.parse_unary()?;
        Ok(i64::from(match operation {
            "==" => left == right,
            "!=" => left != right,
            "<=" => left <= right,
            ">=" => left >= right,
            "<" => left < right,
            ">" => left > right,
            _ => unreachable!("comparison operator came from a fixed list"),
        }))
    }

    fn parse_unary(&mut self) -> Result<i64, String> {
        if self.consume("!") {
            return Ok(i64::from(self.parse_unary()? == 0));
        }
        if self.consume("-") {
            return Ok(-self.parse_unary()?);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<i64, String> {
        if self.consume("(") {
            let value = self.parse_or()?;
            if !self.consume(")") {
                return Err("expected ')' in conditional expression".to_owned());
            }
            return Ok(value);
        }
        if self.consume_word("defined") {
            let parenthesized = self.consume("(");
            let name = self
                .parse_identifier()
                .ok_or_else(|| "expected a macro name after defined".to_owned())?;
            if parenthesized && !self.consume(")") {
                return Err("expected ')' after defined macro name".to_owned());
            }
            return Ok(i64::from(self.macros.contains_key(name)));
        }
        if self.consume_word("fexists") {
            if !self.consume("(") {
                return Err("expected '(' after fexists".to_owned());
            }
            let spelling = self.parse_quoted_path()?;
            if !self.consume(")") {
                return Err("expected ')' after fexists path".to_owned());
            }
            let portable = spelling.replace(['\\', '/'], std::path::MAIN_SEPARATOR_STR);
            let relative = Path::new(&portable);
            let beside_source = self
                .source_file
                .parent()
                .expect("a loaded source file has a parent")
                .join(relative);
            let from_root = self.root_directory.join(relative);
            return Ok(i64::from(beside_source.exists() || from_root.exists()));
        }
        if let Some(number) = self.parse_number()? {
            return Ok(number);
        }
        if let Some(name) = self.parse_identifier() {
            return Ok(resolve_macro_number(name, self.macros, 0));
        }
        Err("expected a value in conditional expression".to_owned())
    }

    fn parse_quoted_path(&mut self) -> Result<String, String> {
        self.skip_whitespace();
        if self.source.as_bytes().get(self.offset) != Some(&b'"') {
            return Err("fexists requires a quoted path".to_owned());
        }
        self.offset += 1;
        let mut path = String::new();
        while self.offset < self.source.len() {
            let character = self.source[self.offset..]
                .chars()
                .next()
                .expect("offset is inside conditional expression");
            self.offset += character.len_utf8();
            match character {
                '"' => return Ok(path),
                '\\' => {
                    let escaped = self.source[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| "unterminated fexists path".to_owned())?;
                    self.offset += escaped.len_utf8();
                    path.push(escaped);
                }
                _ => path.push(character),
            }
        }
        Err("unterminated fexists path".to_owned())
    }

    fn parse_number(&mut self) -> Result<Option<i64>, String> {
        self.skip_whitespace();
        let remaining = &self.source[self.offset..];
        let (radix, prefix) = if remaining.starts_with("0x") || remaining.starts_with("0X") {
            (16, 2)
        } else {
            (10, 0)
        };
        let digit_count = remaining[prefix..]
            .chars()
            .take_while(|character| character.is_digit(radix))
            .map(char::len_utf8)
            .sum::<usize>();
        if digit_count == 0 {
            return Ok(None);
        }
        let end = prefix + digit_count;
        let spelling = &remaining[prefix..end];
        let number = i64::from_str_radix(spelling, radix)
            .map_err(|error| format!("invalid integer {spelling:?}: {error}"))?;
        self.offset += end;
        Ok(Some(number))
    }

    fn parse_identifier(&mut self) -> Option<&'a str> {
        self.skip_whitespace();
        let remaining = &self.source[self.offset..];
        let length = remaining
            .chars()
            .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
            .map(char::len_utf8)
            .sum::<usize>();
        if length == 0 {
            return None;
        }
        self.offset += length;
        Some(&remaining[..length])
    }

    fn consume_word(&mut self, word: &str) -> bool {
        self.skip_whitespace();
        let remaining = &self.source[self.offset..];
        if !remaining.starts_with(word) {
            return false;
        }
        let boundary = remaining[word.len()..].chars().next();
        if boundary.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_') {
            return false;
        }
        self.offset += word.len();
        true
    }

    fn consume(&mut self, spelling: &str) -> bool {
        self.skip_whitespace();
        if self.source[self.offset..].starts_with(spelling) {
            self.offset += spelling.len();
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        self.offset += self.source[self.offset..]
            .chars()
            .take_while(|character| character.is_whitespace())
            .map(char::len_utf8)
            .sum::<usize>();
    }
}

fn resolve_macro_number(
    name: &str,
    macros: &HashMap<String, MacroDefinition>,
    depth: usize,
) -> i64 {
    if depth >= 32 {
        return 0;
    }
    let Some(definition) = macros.get(name) else {
        return match name {
            "TRUE" => 1,
            _ => 0,
        };
    };
    let value = definition.replacement.trim();
    if value.is_empty() {
        return 1;
    }
    if let Some(hexadecimal) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        return i64::from_str_radix(hexadecimal, 16).unwrap_or(0);
    }
    if let Ok(number) = value.parse() {
        return number;
    }
    if value
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return resolve_macro_number(value, macros, depth + 1);
    }
    1
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    use dm_core::SourceSpan;

    use super::{IncludeTarget, PragmaSeverity, Project, ProjectError};

    struct ScratchDirectory(PathBuf);

    impl ScratchDirectory {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock should follow the Unix epoch")
                .as_nanos();
            let path =
                std::env::temp_dir().join(format!("dm64-project-test-{}-{unique}", process::id()));
            fs::create_dir(&path).expect("scratch directory should be created");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for ScratchDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("scratch directory should be removed");
        }
    }

    fn preprocessor_error_for(source: &str) -> (usize, String) {
        let scratch = ScratchDirectory::new();
        let environment = scratch.path().join("world.dme");
        fs::write(&environment, source).expect("environment should be written");
        match Project::load(environment).expect_err("project should reject malformed directives") {
            ProjectError::Preprocessor {
                offset, message, ..
            } => (offset, message),
            error => panic!("expected a preprocessor error, got {error}"),
        }
    }

    #[test]
    fn loads_quoted_includes_once_in_first_discovery_order() {
        let scratch = ScratchDirectory::new();
        let nested = scratch.path().join("nested");
        fs::create_dir(&nested).expect("nested directory should be created");
        fs::write(
            scratch.path().join("world.dme"),
            "#include \"nested\\first.dm\"\n#include \"shared.dm\"\n",
        )
        .expect("environment should be written");
        fs::write(
            nested.join("first.dm"),
            "#include \"../shared.dm\"\n/datum/first\n",
        )
        .expect("first source should be written");
        fs::write(scratch.path().join("shared.dm"), "/datum/shared\n")
            .expect("shared source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("project should load successfully");

        assert_eq!(project.files.len(), 3);
        assert_eq!(project.includes.len(), 3);
        assert_eq!(project.files[0].relative_path, Path::new("world.dme"));
        assert_eq!(
            project.files[1].relative_path,
            Path::new("nested").join("first.dm")
        );
        assert_eq!(project.files[2].relative_path, Path::new("shared.dm"));
        assert!(matches!(
            project.includes[1].target,
            IncludeTarget::File(id) if id.index() == 2
        ));
        assert!(matches!(
            project.includes[2].target,
            IncludeTarget::File(id) if id.index() == 2
        ));
    }

    #[test]
    fn records_system_includes_without_resolving_them() {
        let scratch = ScratchDirectory::new();
        fs::write(scratch.path().join("world.dme"), "#include <stddef.dm>\n")
            .expect("environment should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("project should load successfully");

        assert_eq!(project.files.len(), 1);
        assert!(matches!(
            &project.includes[0].target,
            IncludeTarget::System(spelling) if spelling == "stddef.dm"
        ));
    }

    #[test]
    fn skips_includes_in_inactive_conditional_branches() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#ifdef NOT_DEFINED\n#include \"missing.dm\"\n#else\n#include \"active.dm\"\n#endif\n",
        )
        .expect("environment should be written");
        fs::write(scratch.path().join("active.dm"), "/datum/active\n")
            .expect("active source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("inactive missing include should be ignored");

        assert_eq!(project.files.len(), 2);
        assert_eq!(project.includes.len(), 1);
        assert_eq!(project.includes[0].spelling, "active.dm");
    }

    #[test]
    fn masks_directives_and_nested_inactive_branches_without_shifting_bytes() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#include \"conditions.dm\"\n",
        )
        .expect("environment should be written");
        let source = "/datum/before\n#if 0\n/datum/hidden\n#if 1\n/datum/nested_hidden\n#endif\n#else\n/* active comment */\n/datum/selected\n#if 1\n/datum/nested_selected\n#else\n/datum/other_hidden\n#endif\n#endif\n/datum/after\n";
        fs::write(scratch.path().join("conditions.dm"), source)
            .expect("conditional source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("conditional source should load");
        let file = &project.files[1];
        let compiler_source = file
            .compiler_text()
            .expect("compiler source should remain UTF-8");

        assert_eq!(compiler_source.len(), source.len());
        for (offset, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                assert_eq!(compiler_source.as_bytes()[offset], b'\n');
            }
        }
        for hidden in [
            "/datum/hidden",
            "/datum/nested_hidden",
            "/datum/other_hidden",
        ] {
            let start = source.find(hidden).expect("hidden source should exist");
            assert!(
                compiler_source.as_bytes()[start..start + hidden.len()]
                    .iter()
                    .all(|byte| *byte == b' ')
            );
        }
        for active in [
            "/datum/before",
            "/* active comment */",
            "/datum/selected",
            "/datum/nested_selected",
            "/datum/after",
        ] {
            let start = source.find(active).expect("active source should exist");
            assert_eq!(&compiler_source[start..start + active.len()], active);
        }
        assert!(!compiler_source.contains("#if"));
        assert!(!compiler_source.contains("#else"));
        assert!(!compiler_source.contains("#endif"));
    }

    #[test]
    fn borrows_byte_identical_source_when_no_directives_need_masking() {
        let scratch = ScratchDirectory::new();
        fs::write(scratch.path().join("world.dme"), "#include \"plain.dm\"\n")
            .expect("environment should be written");
        fs::write(
            scratch.path().join("plain.dm"),
            "/datum/plain\n\tvar/value = 1\n",
        )
        .expect("plain source should be written");

        let project =
            Project::load(scratch.path().join("world.dme")).expect("plain source should load");
        let file = &project.files[1];
        let original = file.text().expect("plain source should be UTF-8");
        let compiler_source = file
            .compiler_text()
            .expect("compiler source should be UTF-8");

        assert_eq!(compiler_source, original);
        assert_eq!(compiler_source.as_ptr(), original.as_ptr());
    }

    #[test]
    fn masks_complete_multiline_define_values() {
        let scratch = ScratchDirectory::new();
        let source = "#define DOCUMENT {\"\n#include \"not_a_real_include.dm\"\n#if 0\nraw text\n#endif\n\"}\n#define CONTINUED \"first\\\nsecond\\\nthird\"\n/datum/after_macros\n";
        fs::write(scratch.path().join("world.dme"), source).expect("environment should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("directives inside macro values should not be evaluated");
        let compiler_source = project.files[0]
            .compiler_text()
            .expect("masked macro source should be UTF-8");
        let declaration_start = source
            .find("/datum/after_macros")
            .expect("ordinary declaration should exist");

        assert_eq!(compiler_source.len(), source.len());
        assert!(
            compiler_source.as_bytes()[..declaration_start]
                .iter()
                .all(|byte| matches!(*byte, b' ' | b'\r' | b'\n'))
        );
        assert_eq!(
            &compiler_source[declaration_start..],
            "/datum/after_macros\n"
        );
        assert!(project.includes.is_empty());
    }

    #[test]
    fn multiline_macro_continuations_do_not_add_definition_indentation() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            r"#define WRAP(value) \
	do { \
		if (value) { \
			value = 1; \
		} \
	} while (0)
/proc/example(value)
	WRAP(value)
",
        )
        .expect("environment should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("continued macro source should load");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should be UTF-8");

        // The invocation contributes its one tab of procedure indentation.
        // Formatting tabs from the macro definition must not survive ahead
        // of the first replacement token and turn this into a nested block.
        assert!(
            expanded.contains("\tdo { if (value) { value = 1; } } while (0)"),
            "expanded source was {expanded:?}"
        );
        assert!(!expanded.contains("\t\tdo {"));
    }

    #[test]
    fn expands_nested_object_macros_across_includes_with_source_mapping() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#include \"defines.dm\"\n#include \"uses.dm\"\n",
        )
        .expect("environment should be written");
        fs::write(
            scratch.path().join("defines.dm"),
            "#define ROOT /datum\n#define NESTED ROOT/example // replacement comment\n#define FUNCTION(value) ROOT/value\n",
        )
        .expect("macro definitions should be written");
        let uses = "NESTED\n\"NESTED\"\n// NESTED\n/* NESTED */\n{\"NESTED\"}\nFUNCTION(test)\n#undef NESTED\nNESTED\n";
        fs::write(scratch.path().join("uses.dm"), uses).expect("macro uses should be written");

        let project =
            Project::load(scratch.path().join("world.dme")).expect("object macros should expand");
        let file = &project.files[2];
        let expanded = file
            .compiler_text()
            .expect("expanded source should be UTF-8");

        assert!(expanded.starts_with("/datum/example\n"));
        assert!(expanded.contains("\"NESTED\"\n// NESTED\n/* NESTED */\n{\"NESTED\"}"));
        assert!(expanded.contains("/datum/test"));
        assert!(expanded.ends_with("      \nNESTED\n"));
        let expanded_start = expanded
            .find("/datum/example")
            .expect("nested replacement should exist");
        let original_start = uses.find("NESTED").expect("invocation should exist");
        assert_eq!(
            file.original_span(SourceSpan::new(
                expanded_start,
                expanded_start + "/datum/example".len(),
            )),
            SourceSpan::new(original_start, original_start + "NESTED".len())
        );
    }

    #[test]
    fn reports_deterministic_recursive_and_depth_macro_errors() {
        let scratch = ScratchDirectory::new();
        let recursive_source = "#define FIRST SECOND\n#define SECOND FIRST\nFIRST\n";
        fs::write(scratch.path().join("world.dme"), recursive_source)
            .expect("recursive macros should be written");
        let recursion = Project::load(scratch.path().join("world.dme"))
            .expect_err("recursive macros should fail");
        assert!(matches!(
            recursion,
            ProjectError::MacroExpansion { offset, ref message, .. }
                if offset == recursive_source.rfind("FIRST").expect("invocation should exist")
                    && message == "recursive macro expansion: FIRST -> SECOND -> FIRST"
        ));

        let depth_scratch = ScratchDirectory::new();
        let mut depth_source = String::new();
        for index in 0..=64 {
            writeln!(depth_source, "#define LEVEL_{index} LEVEL_{}", index + 1)
                .expect("writing to a string should succeed");
        }
        depth_source.push_str("LEVEL_0\n");
        fs::write(depth_scratch.path().join("world.dme"), depth_source)
            .expect("deep macros should be written");
        let depth = Project::load(depth_scratch.path().join("world.dme"))
            .expect_err("over-deep macros should fail");
        assert!(matches!(
            depth,
            ProjectError::MacroExpansion { ref message, .. }
                if message == "macro expansion exceeded 64 levels while expanding LEVEL_64"
        ));
    }

    #[test]
    fn expands_function_macros_with_nested_and_variadic_arguments() {
        let scratch = ScratchDirectory::new();
        let source = "#define ROOT /datum\n#define WRAP(first, second) list(first, second)\n#define FORWARD(arguments...) WRAP(arguments)\n#define STRINGIFY(value) #value\n#define TYPE(value) ROOT/##value\nWRAP(call(1, 2), \"comma, text\")\nFORWARD(alpha, list(beta, gamma))\nSTRINGIFY(alpha + beta)\nTYPE(example)\n";
        fs::write(scratch.path().join("world.dme"), source)
            .expect("function macros should be written");

        let project =
            Project::load(scratch.path().join("world.dme")).expect("function macros should expand");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded functions should be UTF-8");

        assert!(expanded.contains("list(call(1, 2), \"comma, text\")"));
        assert!(expanded.contains("list(alpha, list(beta, gamma))"));
        assert!(expanded.contains("\"alpha + beta\""));
        assert!(expanded.contains("/datum/example"));
        assert!(!expanded.contains("WRAP("));
        assert!(!expanded.contains("FORWARD("));
        let invocation_start = source
            .find("TYPE(example)")
            .expect("type invocation should exist");
        let expanded_start = expanded
            .find("/datum/example")
            .expect("expanded type should exist");
        assert_eq!(
            project.files[0].original_span(SourceSpan::new(
                expanded_start,
                expanded_start + "/datum/example".len(),
            )),
            SourceSpan::new(invocation_start, invocation_start + "TYPE(example)".len())
        );
    }

    #[test]
    fn applies_define_directives_generated_by_object_and_function_macros() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#define EMPTY\n",
            "#define DEFINE #define\n",
            "#define DEFINE_WITH_EMPTY EMPTY #define\n",
            "#define DEFINE_NESTED(name, value) DEFINE name value\n",
            "#define DEFINE_AFTER_STATEMENT(name, value) var/kept = 4; DEFINE name value\n",
            "DEFINE A 1\n",
            "DEFINE_WITH_EMPTY B 2\n",
            "DEFINE_NESTED(C, 3)\n",
            "EMPTY DEFINE D 4\n",
            " DEFINE E 5\n",
            "DEFINE_AFTER_STATEMENT(F, 6)\n",
            "/proc/check()\n",
            "\treturn A + B + C + D + E + F\n",
        );
        fs::write(scratch.path().join("world.dme"), source)
            .expect("generated-directive fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("generated define directives should preprocess");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should remain UTF-8");

        assert!(expanded.contains("var/kept = 4;"));
        assert!(expanded.contains("return 1 + 2 + 3 + 4 + 5 + 6"));
        assert!(!expanded.contains("#define"));
    }

    #[test]
    fn preserves_multiline_macro_invocations_across_conditional_directives() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#define MAKE_LIST(values...) list(values)\n",
            "/proc/check()\n",
            "\tvar/first = MAKE_LIST(\n",
            "\t\t1,\n",
            "\t\t#ifdef OMIT_MIDDLE\n",
            "\t\t2,\n",
            "\t\t#endif\n",
            "\t\t3\n",
            "\t)\n",
        );
        fs::write(scratch.path().join("world.dme"), source)
            .expect("embedded-directive fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("directives inside macro arguments should preprocess");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should remain UTF-8");

        assert!(expanded.contains("list(1,"));
        assert!(expanded.contains('3'));
        assert!(!expanded.contains("OMIT_MIDDLE"));
        assert!(!expanded.contains("\n\t\t2,"));
    }

    #[test]
    fn expands_predefined_file_macro_inside_user_macros() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#define SOURCE_FILE __FILE__\n",
            "/proc/source_file()\n",
            "\treturn SOURCE_FILE\n",
        );
        fs::write(scratch.path().join("world.dme"), source)
            .expect("file macro fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("predefined file macro should expand");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should remain UTF-8");

        assert!(!expanded.contains("__FILE__"));
        assert!(!expanded.contains("SOURCE_FILE"));
        assert!(expanded.contains("world.dme"));
    }

    #[test]
    fn expands_predefined_line_macro_at_the_invocation_line() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#define SOURCE_LINE __LINE__\n",
            "/proc/source_line()\n",
            "\treturn SOURCE_LINE\n",
        );
        fs::write(scratch.path().join("world.dme"), source)
            .expect("line macro fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("predefined line macro should expand");
        let expanded = project.files[0]
            .compiler_text()
            .expect("expanded source should remain UTF-8");

        assert!(!expanded.contains("__LINE__"));
        assert!(!expanded.contains("SOURCE_LINE"));
        assert!(
            expanded.contains("\treturn 3\n"),
            "expanded source was {expanded:?}"
        );
    }

    #[test]
    fn shares_defines_across_recursive_includes() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#include \"defines.dm\"\n#include \"conditional.dm\"\n",
        )
        .expect("environment should be written");
        fs::write(
            scratch.path().join("defines.dm"),
            "#define FEATURE_LEVEL 3\n",
        )
        .expect("defines source should be written");
        fs::write(
            scratch.path().join("conditional.dm"),
            "#if defined(FEATURE_LEVEL) && FEATURE_LEVEL >= 3\n#include \"feature.dm\"\n#endif\n",
        )
        .expect("conditional source should be written");
        fs::write(scratch.path().join("feature.dm"), "/datum/feature\n")
            .expect("feature source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("cross-file macro should activate the include");

        assert_eq!(project.files.len(), 4);
        assert_eq!(project.includes.len(), 3);
        assert_eq!(project.includes[2].spelling, "feature.dm");
    }

    #[test]
    fn exposes_target_compiler_version_to_conditions() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#if (DM_VERSION == 516 && DM_BUILD >= 1663)\n#include \"version.dm\"\n#endif\n",
        )
        .expect("environment should be written");
        fs::write(scratch.path().join("version.dm"), "/datum/version\n")
            .expect("version source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("target version condition should parse");

        assert_eq!(project.files.len(), 2);
        assert_eq!(project.includes.len(), 1);
    }

    #[test]
    fn evaluates_fexists_relative_to_source_and_project_root() {
        let scratch = ScratchDirectory::new();
        fs::create_dir(scratch.path().join("nested"))
            .expect("nested source directory should be created");
        fs::write(
            scratch.path().join("world.dme"),
            "#include \"nested/check.dm\"\n",
        )
        .expect("environment should be written");
        fs::write(scratch.path().join("root.resource"), b"root")
            .expect("root resource should be written");
        fs::write(
            scratch.path().join("nested").join("local.resource"),
            b"local",
        )
        .expect("local resource should be written");
        fs::write(
            scratch.path().join("nested").join("check.dm"),
            concat!(
                "#if fexists(\"local.resource\")\n",
                "#include \"local.dm\"\n",
                "#endif\n",
                "#if fexists(\"root.resource\")\n",
                "#include \"root.dm\"\n",
                "#endif\n",
                "#if fexists(\"missing.resource\")\n",
                "#include \"missing.dm\"\n",
                "#endif\n",
            ),
        )
        .expect("conditional source should be written");
        fs::write(
            scratch.path().join("nested").join("local.dm"),
            "/datum/local\n",
        )
        .expect("local include should be written");
        fs::write(scratch.path().join("root.dm"), "/datum/root\n")
            .expect("root include should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("fexists conditions should load");

        assert_eq!(project.includes.len(), 3);
        assert!(
            project
                .includes
                .iter()
                .any(|edge| edge.spelling == "local.dm")
        );
        assert!(
            project
                .includes
                .iter()
                .any(|edge| edge.spelling == "root.dm")
        );
        assert!(
            !project
                .includes
                .iter()
                .any(|edge| edge.spelling == "missing.dm")
        );
    }

    #[test]
    fn rejects_malformed_fexists_conditions() {
        let (_, unquoted) = preprocessor_error_for("#if fexists(file.dm)\n#endif\n");
        assert_eq!(unquoted, "fexists requires a quoted path");

        let (_, unclosed) = preprocessor_error_for("#if fexists(\"file.dm\"\n#endif\n");
        assert_eq!(unclosed, "expected ')' after fexists path");
    }

    #[test]
    fn masks_warning_and_pragma_directives_without_rejecting_source() {
        let scratch = ScratchDirectory::new();
        let source = concat!(
            "#pragma WarningDirective warning\n",
            "#warn this is a warning, not DM source\n",
            "/datum/after_warning\n",
        );
        fs::write(scratch.path().join("world.dme"), source).expect("environment should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("warnings and pragmas should not reject the project");
        let compiler_source = project.files[0]
            .compiler_text()
            .expect("compiler source should remain UTF-8");

        assert!(!compiler_source.contains("#warn"));
        assert!(!compiler_source.contains("#pragma"));
        assert!(compiler_source.contains("/datum/after_warning"));
    }

    #[test]
    fn reports_error_directives_only_in_active_branches() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "#if 0\n#error inactive failure\n#endif\n/datum/valid\n",
        )
        .expect("environment should be written");
        Project::load(scratch.path().join("world.dme"))
            .expect("an inactive error directive should be ignored");

        fs::write(
            scratch.path().join("world.dme"),
            "#error \"active failure\"\n",
        )
        .expect("environment should be replaced");
        let error = Project::load(scratch.path().join("world.dme"))
            .expect_err("an active error directive should reject the project");
        assert!(matches!(
            error,
            ProjectError::Preprocessor { ref message, .. }
                if message == "#error \"active failure\""
        ));
    }

    #[test]
    fn applies_warning_and_duplicate_include_pragma_severity() {
        let warning_scratch = ScratchDirectory::new();
        fs::write(
            warning_scratch.path().join("world.dme"),
            "#pragma WarningDirective error\n#warn promoted warning\n",
        )
        .expect("warning fixture should be written");
        let warning = Project::load(warning_scratch.path().join("world.dme"))
            .expect_err("the warning pragma should promote warnings to errors");
        assert!(matches!(
            warning,
            ProjectError::Preprocessor { ref message, .. }
                if message == "#warn promoted warning"
        ));

        let include_scratch = ScratchDirectory::new();
        fs::write(
            include_scratch.path().join("world.dme"),
            concat!(
                "#pragma FileAlreadyIncluded error\n",
                "#include \"shared.dm\"\n",
                "#include \"./shared.dm\"\n",
            ),
        )
        .expect("include fixture should be written");
        fs::write(include_scratch.path().join("shared.dm"), "/datum/shared\n")
            .expect("included file should be written");
        let duplicate = Project::load(include_scratch.path().join("world.dme"))
            .expect_err("the include pragma should reject duplicate canonical files");
        assert!(matches!(
            duplicate,
            ProjectError::Preprocessor { ref message, .. }
                if message == "file already included: \"./shared.dm\""
        ));
    }

    #[test]
    fn preserves_generic_diagnostic_pragma_metadata_in_source_order() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            concat!(
                "#pragma SuspiciousMatrixCall warning\n",
                "#pragma SuspiciousMatrixCall error\n",
                "#pragma EmptyProc disabled\n",
            ),
        )
        .expect("pragma fixture should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("diagnostic pragmas should load");

        assert_eq!(project.diagnostic_pragmas.len(), 3);
        assert_eq!(
            project.diagnostic_severity("SuspiciousMatrixCall"),
            Some(PragmaSeverity::Error)
        );
        assert_eq!(
            project.diagnostic_severity("EmptyProc"),
            Some(PragmaSeverity::Disabled)
        );
    }

    #[test]
    fn ignores_directives_in_block_comments_and_accepts_comments_as_whitespace() {
        let scratch = ScratchDirectory::new();
        fs::write(
            scratch.path().join("world.dme"),
            "/*\n#include \"missing.dm\"\n#if 0\n*/\n/* leading */ #define FEATURE 1 /* trailing\ncontinued */\n#if FEATURE /* condition */\n#include /* separator */ \"active.dm\"\n#endif\n",
        )
        .expect("environment should be written");
        fs::write(scratch.path().join("active.dm"), "/datum/active\n")
            .expect("active source should be written");

        let project = Project::load(scratch.path().join("world.dme"))
            .expect("commented directives should not affect preprocessing");

        assert_eq!(project.files.len(), 2);
        assert_eq!(project.includes.len(), 1);
        assert_eq!(project.includes[0].spelling, "active.dm");
    }

    #[test]
    fn rejects_duplicate_else_and_elif_after_else() {
        let (_, duplicate_message) = preprocessor_error_for("#if 0\n#else\n#else\n#endif\n");
        assert_eq!(
            duplicate_message,
            "duplicate #else for conditional opened at byte 0"
        );

        let (_, elif_message) = preprocessor_error_for("#if 0\n#else\n#elif 1\n#endif\n");
        assert_eq!(
            elif_message,
            "#elif after #else for conditional opened at byte 0"
        );
    }

    #[test]
    fn validates_conditional_directive_arguments() {
        let (_, empty_elif) = preprocessor_error_for("#if 0\n#elif\n#endif\n");
        assert_eq!(empty_elif, "#elif requires a conditional expression");

        let (_, else_arguments) = preprocessor_error_for("#if 0\n#else extra\n#endif\n");
        assert_eq!(else_arguments, "#else does not accept arguments");

        let (_, multiple_names) = preprocessor_error_for("#ifdef FIRST SECOND\n#endif\n");
        assert_eq!(multiple_names, "#ifdef accepts exactly one macro name");
    }

    #[test]
    fn rejects_variadic_macro_parameters_before_the_final_position() {
        let (_, message) =
            preprocessor_error_for("#define INVALID(first..., second..., third...) list(first)\n");

        assert_eq!(message, "variadic macro parameter must be last");
    }

    #[test]
    fn reports_unmatched_elif_before_parsing_its_expression() {
        let (offset, message) = preprocessor_error_for("#elif )\n");

        assert_eq!(offset, 0);
        assert_eq!(message, "#elif without matching #if");
    }

    #[test]
    fn reports_the_innermost_unterminated_conditional_and_stack_depth() {
        let source = "#if 1\n#if 1\n";
        let (offset, message) = preprocessor_error_for(source);

        assert_eq!(offset, source.len());
        assert_eq!(
            message,
            "unterminated conditional opened at byte 6 (2 conditional directives still open)"
        );
    }

    #[test]
    fn rejects_includes_outside_the_project_directory() {
        let scratch = ScratchDirectory::new();
        let project_directory = scratch.path().join("project");
        fs::create_dir(&project_directory).expect("project directory should be created");
        fs::write(scratch.path().join("outside.dm"), "/datum/outside\n")
            .expect("outside source should be written");
        fs::write(
            project_directory.join("world.dme"),
            "#include \"../outside.dm\"\n",
        )
        .expect("environment should be written");

        let error = Project::load(project_directory.join("world.dme"))
            .expect_err("outside include should be rejected");

        assert!(matches!(error, ProjectError::OutsideProject { .. }));
    }
}
