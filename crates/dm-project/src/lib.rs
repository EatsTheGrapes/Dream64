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
            | Self::Preprocessor { .. } => None,
        }
    }
}

struct Loader {
    root_file: PathBuf,
    root_directory: PathBuf,
    files: Vec<ProjectFile>,
    includes: Vec<(usize, IncludeEdge)>,
    identities: HashMap<PathBuf, FileId>,
    macros: HashMap<String, String>,
    next_include_ordinal: usize,
}

impl Loader {
    fn new(root_file: &Path) -> Result<Self, ProjectError> {
        let root_file = canonicalize(root_file)?;
        let root_directory = root_file
            .parent()
            .expect("a canonical file path has a parent")
            .to_path_buf();
        let macros = HashMap::from([
            ("DM_VERSION".to_owned(), TARGET_DM_VERSION.to_string()),
            ("DM_BUILD".to_owned(), TARGET_DM_BUILD.to_string()),
        ]);
        Ok(Self {
            root_file,
            root_directory,
            files: Vec::new(),
            includes: Vec::new(),
            identities: HashMap::new(),
            macros,
            next_include_ordinal: 0,
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

    fn process_directives(
        &mut self,
        source: FileId,
        path: &Path,
        directives: Vec<Directive>,
    ) -> Result<(), ProjectError> {
        if directives.is_empty() {
            return Ok(());
        }

        let mut compiler_contents = self.files[source.index()].contents.clone();
        let mut conditionals = Vec::new();
        let mut cursor = 0usize;
        for directive in directives {
            let span = directive.span;
            if !conditional_active(&conditionals) {
                mask_source(&mut compiler_contents, SourceSpan::new(cursor, span.start));
            }
            mask_source(&mut compiler_contents, span);
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
        if !conditional_active(&conditionals) {
            mask_source(
                &mut compiler_contents,
                SourceSpan::new(cursor, self.files[source.index()].contents.len()),
            );
        }
        self.files[source.index()].compiler_contents = Some(compiler_contents);
        Ok(())
    }

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
            DirectiveKind::Define { name, value } if active => {
                self.macros.insert(name, value);
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
            DirectiveKind::Define { .. }
            | DirectiveKind::Undef(_)
            | DirectiveKind::Include { .. } => {}
            DirectiveKind::Malformed(message) => {
                return Err(preprocessor_error(path, offset, message));
            }
        }
        Ok(())
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
        ConditionParser::new(expression, &self.macros)
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
    },
    Undef(String),
    If(String),
    Ifdef(String),
    Ifndef(String),
    Elif(String),
    Else,
    Endif,
    Malformed(String),
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

fn mask_source(contents: &mut [u8], span: SourceSpan) {
    for byte in &mut contents[span.start..span.end] {
        if !matches!(*byte, b'\r' | b'\n') {
            *byte = b' ';
        }
    }
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
        if let Some(kind) = parse_directive_line(&uncommented) {
            let directive_end = if matches!(&kind, DirectiveKind::Define { .. }) {
                extended_define_end(source, offset, physical_end, &uncommented)
            } else {
                physical_end
            };
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
    macros: &'a HashMap<String, String>,
}

impl<'a> ConditionParser<'a> {
    fn new(source: &'a str, macros: &'a HashMap<String, String>) -> Self {
        Self {
            source: source.split("//").next().unwrap_or(source),
            offset: 0,
            macros,
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
        if let Some(number) = self.parse_number()? {
            return Ok(number);
        }
        if let Some(name) = self.parse_identifier() {
            return Ok(resolve_macro_number(name, self.macros, 0));
        }
        Err("expected a value in conditional expression".to_owned())
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

fn resolve_macro_number(name: &str, macros: &HashMap<String, String>, depth: usize) -> i64 {
    if depth >= 32 {
        return 0;
    }
    let Some(value) = macros.get(name) else {
        return match name {
            "TRUE" => 1,
            _ => 0,
        };
    };
    let value = value.trim();
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
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{IncludeTarget, Project, ProjectError};

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
