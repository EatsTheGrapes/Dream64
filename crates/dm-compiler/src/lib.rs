//! Project-wide orchestration for the Dream64 compiler frontend.
//!
//! This crate is the stable boundary between project discovery, per-file
//! syntax parsing, and global object-tree construction. Later semantic and
//! bytecode stages can consume one [`Compilation`] without rebuilding or
//! reordering source files.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use dm_core::{FileId, SourceSpan};
use dm_lexer::{SpannedToken, TokenKind};
use dm_object_tree::{
    BuildOutput, CodeTree, DefinitionUnit, DiagnosticKind as TreeDiagnosticKind,
    DiagnosticSeverity as TreeDiagnosticSeverity, NodeId, TreeDiagnostic,
};
use dm_project::{FileKind, PragmaSeverity, Project, ProjectDefines, ProjectError};
use dm_syntax::{
    Definition, DefinitionKind, DefinitionPath, Indentation, ParameterSyntax, SourceLine,
    SyntaxError, SyntaxFile,
};
use persistent_database::{
    Digest, InputDependency, MAX_SECTION_PAYLOAD_BYTES, PersistentCompilerDatabase,
    SectionDependency, StableIdEntry,
};

const LINKED_FRONTEND_SECTION: u64 = 1;
/// Persistent section containing the fully linked executable module.
pub const PERSISTENT_EXECUTABLE_SECTION: u64 = 2;
/// Reserved page-ID range for the persistent executable section.
pub const PERSISTENT_EXECUTABLE_PAGE_BASE: u64 = 20_000_000;
const INPUT_SECTION_BASE: u64 = 1_000;
const LINKED_PAYLOAD_PAGE_BASE: u64 = 10_000_000;

pub mod persistent_database;

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
        self.compile_with_defines(root_file, &ProjectDefines::new())
    }

    /// Loads and compiles a `.dme` frontend, seeding the preprocessor with
    /// caller-supplied `-D` defines.
    ///
    /// # Errors
    ///
    /// Returns [`CompilerError`] when project discovery or source loading
    /// fails.
    pub fn compile_with_defines(
        &self,
        root_file: impl AsRef<Path>,
        defines: &ProjectDefines,
    ) -> Result<Compilation, CompilerError> {
        let project =
            Project::load_with_defines(root_file, defines).map_err(CompilerError::Project)?;
        Ok(compile_project(project))
    }

    /// Compiles a project through an exact persistent preprocessing cache.
    ///
    /// The returned boolean is `true` only when the cached project snapshot
    /// was accepted after byte-for-byte validation of every discovered file.
    /// Syntax and semantic products are still rebuilt from that immutable
    /// snapshot, so compiler changes cannot reuse stale AST or object-tree
    /// data.
    ///
    /// # Errors
    ///
    /// Returns the same project discovery and source errors as [`Self::compile`]
    /// when the cache is absent or stale.
    pub fn compile_cached(
        &self,
        root_file: impl AsRef<Path>,
        cache_file: impl AsRef<Path>,
    ) -> Result<(Compilation, bool), CompilerError> {
        let (compilation, stats) = self.compile_cached_with_stats(root_file, cache_file)?;
        Ok((compilation, stats.project_snapshot_hit))
    }

    /// Compiles a project through the preprocessing and parsed-syntax caches.
    ///
    /// The syntax cache is considered only after the project snapshot has
    /// validated every discovered source/resource fingerprint. Its format also
    /// embeds a fingerprint of the lexer and declaration parser sources, so a
    /// frontend grammar change cannot silently reuse stale tokens or syntax.
    /// Object-tree and semantic products are deliberately rebuilt on every
    /// call and therefore never cross an engine-version boundary.
    ///
    /// # Errors
    ///
    /// Returns the same project discovery and source errors as [`Self::compile`]
    /// when the cache is absent or stale.
    pub fn compile_cached_with_stats(
        &self,
        root_file: impl AsRef<Path>,
        cache_file: impl AsRef<Path>,
    ) -> Result<(Compilation, CompilationCacheStats), CompilerError> {
        self.compile_with_mode(root_file, cache_file, BuildMode::Incremental)
    }

    /// Compiles using an explicit persistent-build policy.
    ///
    /// Incremental builds may reuse both cache tiers. Clean builds discard the
    /// project and parsed-syntax payloads at the caller-provided cache path
    /// before rebuilding and repopulating them. Fresh builds neither read nor
    /// write those persistent caches.
    ///
    /// # Errors
    ///
    /// Returns project discovery or source-loading errors. Cache removal and
    /// cache writes remain best-effort and cannot make valid source fail.
    pub fn compile_with_mode(
        &self,
        root_file: impl AsRef<Path>,
        cache_file: impl AsRef<Path>,
        mode: BuildMode,
    ) -> Result<(Compilation, CompilationCacheStats), CompilerError> {
        let cache_file = cache_file.as_ref();
        if mode == BuildMode::Fresh {
            let compilation = self.compile(root_file)?;
            let source_file_count = compilation
                .project
                .files
                .iter()
                .filter(|file| matches!(file.kind, FileKind::Environment | FileKind::Source))
                .count();
            return Ok((
                compilation,
                CompilationCacheStats {
                    build_mode: mode,
                    project_snapshot_hit: false,
                    parsed_syntax_hit: false,
                    syntax_files_reused: 0,
                    syntax_files_reparsed: source_file_count,
                },
            ));
        }
        if mode == BuildMode::Clean {
            let _ = fs::remove_file(cache_file);
            let _ = fs::remove_file(parsed_syntax_cache_path(cache_file));
        }
        let (project, project_snapshot_hit) =
            Project::load_cached(root_file, cache_file).map_err(CompilerError::Project)?;
        let syntax_cache_file = parsed_syntax_cache_path(cache_file);
        let cached_syntax = fs::read(&syntax_cache_file)
            .ok()
            .and_then(|bytes| decode_parsed_syntax_cache(&bytes, &project));
        let source_file_count = project
            .files
            .iter()
            .filter(|file| matches!(file.kind, FileKind::Environment | FileKind::Source))
            .count();
        let (syntax_files, syntax_diagnostics, parsed_syntax_hit, syntax_files_reused) =
            cached_syntax
                .map(|(syntax, diagnostics, exact, reused)| (syntax, diagnostics, exact, reused))
                .unwrap_or_else(|| {
                    let (syntax, diagnostics) = parse_project_syntax(&project);
                    (syntax, diagnostics, false, 0)
                });
        if !parsed_syntax_hit {
            if let Some(parent) = syntax_cache_file.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let encoded = encode_parsed_syntax_cache(&project, &syntax_files);
            let _ = fs::write(&syntax_cache_file, encoded);
        }
        Ok((
            compile_project_from_syntax(project, syntax_files, syntax_diagnostics),
            CompilationCacheStats {
                build_mode: mode,
                project_snapshot_hit,
                parsed_syntax_hit,
                syntax_files_reused,
                syntax_files_reparsed: source_file_count.saturating_sub(syntax_files_reused),
            },
        ))
    }

    /// Compiles through the versioned persistent compiler database.
    ///
    /// An exact build/input match restores the linked frontend section without
    /// parsing or rebuilding the object tree. A miss reuses unchanged syntax
    /// files, rebuilds the linked section, and atomically replaces the vNext
    /// database with updated dependency and stable-ID metadata.
    pub fn compile_persistent(
        &self,
        root_file: impl AsRef<Path>,
        database_file: impl AsRef<Path>,
        mode: BuildMode,
        semantic_digest: Digest,
        build_configuration_digest: Digest,
    ) -> Result<(Compilation, PersistentCompilationStats), CompilerError> {
        let database_file = database_file.as_ref();
        let frontend_cache = persistent_frontend_cache_path(database_file);
        if mode == BuildMode::Clean {
            let _ = fs::remove_file(database_file);
            let _ = fs::remove_file(&frontend_cache);
            let _ = fs::remove_file(parsed_syntax_cache_path(&frontend_cache));
        }
        let (project, project_snapshot_hit) = if mode == BuildMode::Fresh {
            (
                Project::load(root_file).map_err(CompilerError::Project)?,
                false,
            )
        } else {
            Project::load_cached_exact(root_file, &frontend_cache)
                .map_err(CompilerError::Project)?
        };
        self.compile_persistent_prevalidated(
            project,
            project_snapshot_hit,
            database_file,
            mode,
            semantic_digest,
            build_configuration_digest,
        )
    }

    /// Compiles a project snapshot already validated by the caller.
    ///
    /// This avoids a duplicate filesystem byte scan when a lifecycle owner has
    /// already selected metadata validation or strict exact hashing.
    pub fn compile_persistent_prevalidated(
        &self,
        project: Project,
        project_snapshot_hit: bool,
        database_file: impl AsRef<Path>,
        mode: BuildMode,
        semantic_digest: Digest,
        build_configuration_digest: Digest,
    ) -> Result<(Compilation, PersistentCompilationStats), CompilerError> {
        let database_file = database_file.as_ref();
        let frontend_cache = persistent_frontend_cache_path(database_file);
        let inputs = persistent_inputs(&project);
        let prior = (mode == BuildMode::Incremental)
            .then(|| PersistentCompilerDatabase::read(database_file).ok())
            .flatten();
        let changed_inputs = prior.as_ref().map_or_else(
            || (0..inputs.len() as u64).collect(),
            |database| {
                if database.matches_build(&semantic_digest, &build_configuration_digest) {
                    database.changed_inputs(&inputs)
                } else {
                    (0..inputs.len() as u64).collect()
                }
            },
        );
        let invalidated_sections = prior.as_ref().map_or_else(Vec::new, |database| {
            database.invalidated_sections(&changed_inputs)
        });
        if let Some(database) = &prior
            && database.matches_build(&semantic_digest, &build_configuration_digest)
            && changed_inputs.is_empty()
            && let Some(payload) = linked_frontend_payload(database)
            && let Ok(compilation) = Compilation::decode_compiled_artifact(&payload)
        {
            return Ok((
                compilation,
                PersistentCompilationStats {
                    build_mode: mode,
                    project_snapshot_hit,
                    parsed_syntax_hit: true,
                    syntax_files_reused: inputs.len(),
                    syntax_files_reparsed: 0,
                    linked_sections_reused: 1,
                    linked_sections_rebuilt: 0,
                    changed_inputs: 0,
                    invalidated_sections: 0,
                },
            ));
        }

        let syntax_cache_file = parsed_syntax_cache_path(&frontend_cache);
        let cached_syntax = (mode != BuildMode::Fresh)
            .then(|| fs::read(&syntax_cache_file).ok())
            .flatten()
            .and_then(|bytes| decode_parsed_syntax_cache(&bytes, &project));
        let source_file_count = project
            .files
            .iter()
            .filter(|file| matches!(file.kind, FileKind::Environment | FileKind::Source))
            .count();
        let (syntax_files, syntax_diagnostics, parsed_syntax_hit, syntax_files_reused) =
            cached_syntax.unwrap_or_else(|| {
                let (syntax, diagnostics) = parse_project_syntax(&project);
                (syntax, diagnostics, false, 0)
            });
        if mode != BuildMode::Fresh && !parsed_syntax_hit {
            if let Some(parent) = syntax_cache_file.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let _ = fs::write(
                &syntax_cache_file,
                encode_parsed_syntax_cache(&project, &syntax_files),
            );
        }
        let compilation = compile_project_from_syntax(project, syntax_files, syntax_diagnostics);
        if mode != BuildMode::Fresh {
            let database = persistent_database_for_compilation(
                &compilation,
                inputs,
                semantic_digest,
                build_configuration_digest,
                prior.as_ref(),
            );
            database
                .write_atomic(database_file)
                .map_err(|error| CompilerError::Persistent(error.to_string()))?;
        }
        Ok((
            compilation,
            PersistentCompilationStats {
                build_mode: mode,
                project_snapshot_hit,
                parsed_syntax_hit,
                syntax_files_reused,
                syntax_files_reparsed: source_file_count.saturating_sub(syntax_files_reused),
                linked_sections_reused: 0,
                linked_sections_rebuilt: 1,
                changed_inputs: changed_inputs.len(),
                invalidated_sections: invalidated_sections.len(),
            },
        ))
    }
}

/// Persistent-cache policy for one compiler invocation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BuildMode {
    /// Compile without reading or updating persistent caches.
    Fresh,
    /// Reuse valid persistent products and replace stale products.
    #[default]
    Incremental,
    /// Rebuild and repopulate persistent products at the selected cache path.
    Clean,
}

/// Cache hits observed while compiling one project.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompilationCacheStats {
    /// Persistent-cache policy selected for this compilation.
    pub build_mode: BuildMode,
    /// The immutable preprocessed project snapshot passed filesystem validation.
    pub project_snapshot_hit: bool,
    /// Parsed declaration syntax was restored without lexing or parsing sources.
    pub parsed_syntax_hit: bool,
    /// Source files restored without lexing or parsing.
    pub syntax_files_reused: usize,
    /// Source files lexed and parsed for this compilation.
    pub syntax_files_reparsed: usize,
}

/// Stage-level reuse and invalidation observed by the vNext compiler database.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PersistentCompilationStats {
    /// Requested persistent build policy.
    pub build_mode: BuildMode,
    /// Preprocessed project snapshot reused.
    pub project_snapshot_hit: bool,
    /// Every parsed syntax file reused.
    pub parsed_syntax_hit: bool,
    /// Syntax files restored from cache.
    pub syntax_files_reused: usize,
    /// Syntax files reparsed.
    pub syntax_files_reparsed: usize,
    /// Linked frontend sections restored directly.
    pub linked_sections_reused: usize,
    /// Linked frontend sections rebuilt.
    pub linked_sections_rebuilt: usize,
    /// Project inputs whose identity or content changed.
    pub changed_inputs: usize,
    /// Existing dependency-graph sections invalidated by those changes.
    pub invalidated_sections: usize,
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
    /// Encodes the complete immutable frontend result for inclusion in a
    /// Dream64 compiled executable artifact.
    ///
    /// Loading this payload restores the preprocessed project, parsed syntax,
    /// canonical object tree, declaration stream, diagnostics, and counters
    /// without rerunning lexing, parsing, or object-tree construction.
    #[doc(hidden)]
    #[must_use]
    pub fn encode_compiled_artifact(&self) -> Vec<u8> {
        let segments = self.encode_compiled_artifact_segments();
        let total_length = segments.iter().map(Vec::len).sum();
        let mut output = Vec::with_capacity(total_length);
        for segment in segments {
            output.extend_from_slice(&segment);
        }
        debug_assert_eq!(output.len(), total_length);
        output
    }

    /// Encodes the frontend artifact as ordered byte segments without copying
    /// its large project, syntax, and object-tree components into a second
    /// contiguous allocation.
    #[doc(hidden)]
    #[must_use]
    pub fn encode_compiled_artifact_segments(&self) -> Vec<Vec<u8>> {
        let project = self.project.encode_compiled_artifact();
        let syntax = encode_parsed_syntax_cache(&self.project, &self.syntax_files);
        let code_tree = self.code_tree.encode_compiled_artifact();
        let mut header = COMPILATION_ARTIFACT_MAGIC.to_vec();
        syntax_cache_write_len(&mut header, project.len());
        let mut syntax_header = Vec::with_capacity(std::mem::size_of::<u64>());
        syntax_cache_write_len(&mut syntax_header, syntax.len());
        let mut code_tree_header = Vec::with_capacity(std::mem::size_of::<u64>());
        syntax_cache_write_len(&mut code_tree_header, code_tree.len());
        let mut tail = Vec::new();

        syntax_cache_write_len(&mut tail, self.declarations.len());
        for declaration in &self.declarations {
            syntax_cache_write_len(&mut tail, declaration.ordinal);
            syntax_cache_write_len(&mut tail, declaration.file_id.index());
            syntax_cache_write_len(&mut tail, declaration.definition_index);
            syntax_cache_write_len(&mut tail, declaration.node.index());
            syntax_cache_write_span(&mut tail, declaration.span);
        }

        syntax_cache_write_len(&mut tail, self.diagnostics.len());
        for diagnostic in &self.diagnostics {
            tail.push(compilation_artifact_diagnostic_kind(diagnostic.kind));
            tail.push(match diagnostic.severity {
                DiagnosticSeverity::Note => 0,
                DiagnosticSeverity::Warning => 1,
                DiagnosticSeverity::Error => 2,
            });
            syntax_cache_write_string(&mut tail, &diagnostic.message);
            compilation_artifact_write_location(&mut tail, diagnostic.location.as_ref());
            compilation_artifact_write_location(&mut tail, diagnostic.related.as_ref());
        }
        compilation_artifact_write_stats(&mut tail, self.stats);
        vec![
            header,
            project,
            syntax_header,
            syntax,
            code_tree_header,
            code_tree,
            tail,
        ]
    }

    /// Decodes and validates one immutable frontend artifact payload.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error for a corrupt, truncated, incompatible, or
    /// internally inconsistent payload.
    #[doc(hidden)]
    pub fn decode_compiled_artifact(bytes: &[u8]) -> Result<Self, String> {
        let mut input = Cursor::new(bytes);
        let mut magic = vec![0; COMPILATION_ARTIFACT_MAGIC.len()];
        input
            .read_exact(&mut magic)
            .map_err(|_| "compiled frontend is truncated before its header".to_owned())?;
        if magic != COMPILATION_ARTIFACT_MAGIC {
            return Err("compiled frontend has an unsupported header".to_owned());
        }
        let project_bytes = compilation_artifact_read_bytes(&mut input, "project")?;
        let project = Project::decode_compiled_artifact(project_bytes)?;
        let syntax_bytes = compilation_artifact_read_bytes(&mut input, "syntax")?;
        let (syntax_files, _, _, _) = decode_parsed_syntax_cache(syntax_bytes, &project)
            .ok_or_else(|| "compiled frontend syntax payload is invalid".to_owned())?;
        let code_tree_bytes = compilation_artifact_read_bytes(&mut input, "code tree")?;
        let code_tree = CodeTree::decode_compiled_artifact(code_tree_bytes)?;

        let declaration_count = compilation_artifact_read_count(&mut input, "declaration count")?;
        let mut declarations = Vec::with_capacity(declaration_count);
        for _ in 0..declaration_count {
            let ordinal = compilation_artifact_read_len(&mut input, "declaration ordinal")?;
            let file_id = FileId::from_index(compilation_artifact_read_len(
                &mut input,
                "declaration file identity",
            )?);
            let definition_index = compilation_artifact_read_len(&mut input, "definition index")?;
            let node_index = compilation_artifact_read_len(&mut input, "node identity")?;
            let node = code_tree
                .nodes()
                .get(node_index)
                .ok_or_else(|| format!("compiled frontend references missing node {node_index}"))?
                .id;
            let span = compilation_artifact_read_span(&mut input)?;
            declarations.push(CompilationDeclaration {
                ordinal,
                file_id,
                definition_index,
                node,
                span,
            });
        }

        let diagnostic_count = compilation_artifact_read_count(&mut input, "diagnostic count")?;
        let mut diagnostics = Vec::with_capacity(diagnostic_count);
        for _ in 0..diagnostic_count {
            let kind = compilation_artifact_read_diagnostic_kind(&mut input)?;
            let severity = match syntax_cache_read_byte(&mut input)
                .ok_or_else(|| "compiled frontend is truncated at diagnostic severity".to_owned())?
            {
                0 => DiagnosticSeverity::Note,
                1 => DiagnosticSeverity::Warning,
                2 => DiagnosticSeverity::Error,
                tag => {
                    return Err(format!(
                        "compiled frontend has unknown diagnostic severity {tag}"
                    ));
                }
            };
            let message = syntax_cache_read_string(&mut input)
                .ok_or_else(|| "compiled frontend has invalid diagnostic text".to_owned())?;
            let location = compilation_artifact_read_location(&mut input, &project)?;
            let related = compilation_artifact_read_location(&mut input, &project)?;
            diagnostics.push(Diagnostic {
                kind,
                severity,
                message,
                location,
                related,
            });
        }
        let stats = compilation_artifact_read_stats(&mut input)?;
        if input.position() != bytes.len() as u64 {
            return Err("compiled frontend contains trailing bytes".to_owned());
        }
        if syntax_files.len() != project.files.len() {
            return Err("compiled frontend syntax/project table lengths disagree".to_owned());
        }
        for declaration in &declarations {
            let Some(syntax) = syntax_files
                .get(declaration.file_id.index())
                .and_then(Option::as_ref)
            else {
                return Err(format!(
                    "compiled frontend declaration references missing syntax file {}",
                    declaration.file_id.index()
                ));
            };
            if declaration.definition_index >= syntax.definitions.len() {
                return Err(format!(
                    "compiled frontend declaration references missing definition {}:{}",
                    declaration.file_id.index(),
                    declaration.definition_index
                ));
            }
        }
        Ok(Self {
            project,
            syntax_files,
            code_tree,
            declarations,
            diagnostics,
            stats,
        })
    }

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
    /// An assignment or loop iterator attempts to write a declared constant.
    WriteToConstant,
    /// A statically untyped value is dereferenced with the typed member operator.
    UntypedDereference,
    /// A plain datum is indexed despite having no index semantics.
    InvalidIndexOperation,
    /// A runtime-search `:` operator use controlled by pragma.
    RuntimeSearchOperator,
    /// A constant initializer or assignment conflicts with a declared variable type.
    InvalidVarType,
    /// A variable declaration names a type path that does not exist.
    UndefinedType,
    /// An inherited final variable is overridden.
    FinalVariableOverride,
    /// Redeclarations disagree about whether a variable is constant.
    ConflictingVariableModifier,
    /// An inherited global/static variable is initialized again.
    GlobalVariableReinitialization,
    /// `nameof()` received an expression rather than a nameable reference.
    InvalidNameofTarget,
    /// A declaration requiring a compile-time value contains a runtime expression.
    InvalidConstantInitializer,
    /// A single-quoted resource literal does not resolve to a file.
    MissingResource,
    /// A weighted `pick()` entry uses a procedure call as its weight.
    InvalidWeightedPick,
    /// Macro expansion produced a numeric declaration path component.
    InvalidExpandedDeclarationPath,
    /// A procedure override attempts to redefine its inherited return type.
    ReturnTypeRedefinition,
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
    /// Persistent compiler database serialization or installation failed.
    Persistent(String),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Project(error) => write!(formatter, "project loading failed: {error}"),
            Self::Persistent(error) => write!(formatter, "persistent compilation failed: {error}"),
        }
    }
}

impl std::error::Error for CompilerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Project(error) => Some(error),
            Self::Persistent(_) => None,
        }
    }
}

const PARSED_SYNTAX_CACHE_MAGIC: &[u8] = b"DREAM64-PARSED-SYNTAX\0\x02";
const PARSED_SYNTAX_FRONTEND_FINGERPRINT: u64 = syntax_frontend_fingerprint();

const fn hash_source(mut hash: u64, source: &[u8]) -> u64 {
    let mut index = 0;
    while index < source.len() {
        hash ^= source[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

const fn syntax_frontend_fingerprint() -> u64 {
    let hash = hash_source(
        0xcbf2_9ce4_8422_2325,
        include_bytes!("../../dm-lexer/src/lib.rs"),
    );
    hash_source(hash, include_bytes!("../../dm-syntax/src/lib.rs"))
}

fn parsed_syntax_cache_path(project_cache: &Path) -> PathBuf {
    let mut name = project_cache.file_name().unwrap_or_default().to_os_string();
    name.push(".syntax");
    project_cache.with_file_name(name)
}

fn persistent_frontend_cache_path(database: &Path) -> PathBuf {
    let mut name = database.file_name().unwrap_or_default().to_os_string();
    name.push(".frontend");
    database.with_file_name(name)
}

fn digest32(bytes: &[u8]) -> Digest {
    let first = md5::compute(bytes).0;
    let mut salted = Vec::with_capacity(bytes.len() + 1);
    salted.push(1);
    salted.extend_from_slice(bytes);
    let second = md5::compute(salted).0;
    let mut digest = [0; 32];
    digest[..16].copy_from_slice(&first);
    digest[16..].copy_from_slice(&second);
    digest
}

fn persistent_inputs(project: &Project) -> Vec<InputDependency> {
    project
        .files
        .iter()
        .map(|file| {
            InputDependency::new(
                &file.relative_path,
                digest32(&file.contents),
                file.contents.len() as u64,
            )
        })
        .collect()
}

fn persistent_database_for_compilation(
    compilation: &Compilation,
    inputs: Vec<InputDependency>,
    semantic_digest: Digest,
    build_configuration_digest: Digest,
    prior: Option<&PersistentCompilerDatabase>,
) -> PersistentCompilerDatabase {
    let prior_ids = prior
        .into_iter()
        .flat_map(|database| &database.stable_ids)
        .map(|entry| ((entry.namespace.clone(), entry.name.clone()), entry.id))
        .collect::<HashMap<_, _>>();
    let mut next_ids = HashMap::<String, u64>::new();
    for entry in prior.into_iter().flat_map(|database| &database.stable_ids) {
        next_ids
            .entry(entry.namespace.clone())
            .and_modify(|next| *next = (*next).max(entry.id.saturating_add(1)))
            .or_insert(entry.id.saturating_add(1));
    }
    let stable_ids = compilation
        .code_tree
        .nodes()
        .iter()
        .map(|node| {
            let namespace = match node.kind {
                dm_object_tree::NodeKind::Type => "type",
                dm_object_tree::NodeKind::Procedure | dm_object_tree::NodeKind::Verb => "procedure",
                dm_object_tree::NodeKind::Variable => "field",
            };
            let name = node.path.to_string();
            let id = prior_ids
                .get(&(namespace.to_owned(), name.clone()))
                .copied()
                .unwrap_or_else(|| {
                    let next = next_ids.entry(namespace.to_owned()).or_default();
                    let id = *next;
                    *next = next.saturating_add(1);
                    id
                });
            StableIdEntry {
                namespace: namespace.to_owned(),
                name,
                id,
            }
        })
        .collect();
    let linked_segments = compilation.encode_compiled_artifact_segments();
    let input_sections = inputs
        .iter()
        .enumerate()
        .map(|(index, input)| SectionDependency {
            section_id: INPUT_SECTION_BASE + index as u64,
            section_dependencies: vec![],
            input_dependencies: vec![index as u64],
            content_digest: input.content_digest,
            payload: vec![],
        });
    let input_section_ids = (0..inputs.len())
        .map(|index| INPUT_SECTION_BASE + index as u64)
        .collect::<Vec<_>>();
    let mut sections = input_sections.collect::<Vec<_>>();
    let mut linked_hasher = md5::Context::new();
    let mut page_ids = Vec::new();
    let mut next_page_id = LINKED_PAYLOAD_PAGE_BASE;
    for segment in linked_segments {
        for page in segment.chunks(MAX_SECTION_PAYLOAD_BYTES) {
            linked_hasher.consume(page);
            page_ids.push(next_page_id);
            sections.push(SectionDependency {
                section_id: next_page_id,
                section_dependencies: input_section_ids.clone(),
                input_dependencies: vec![],
                content_digest: digest32(page),
                payload: page.to_vec(),
            });
            next_page_id += 1;
        }
    }
    let linked_digest = digest32_from_md5(linked_hasher.compute().0);
    sections.push(SectionDependency {
        section_id: LINKED_FRONTEND_SECTION,
        section_dependencies: page_ids,
        input_dependencies: vec![],
        content_digest: linked_digest,
        payload: vec![],
    });
    let preserved_runtime = prior.and_then(|database| {
        let manifest = database
            .sections
            .iter()
            .find(|section| section.section_id == PERSISTENT_EXECUTABLE_SECTION)?
            .clone();
        if manifest.content_digest == [0; 32]
            || manifest.section_dependencies.iter().any(|id| {
                !(PERSISTENT_EXECUTABLE_PAGE_BASE..PERSISTENT_EXECUTABLE_PAGE_BASE + 1_000_000)
                    .contains(id)
            })
        {
            return None;
        }
        let page_ids = manifest.section_dependencies.clone();
        let pages = database
            .sections
            .iter()
            .filter(|section| page_ids.contains(&section.section_id))
            .cloned()
            .collect::<Vec<_>>();
        (pages.len() == page_ids.len()).then_some((manifest, pages))
    });
    if let Some((manifest, mut pages)) = preserved_runtime {
        for page in &mut pages {
            page.section_dependencies = vec![LINKED_FRONTEND_SECTION];
        }
        sections.extend(pages);
        sections.push(manifest);
    } else {
        sections.push(SectionDependency {
            section_id: PERSISTENT_EXECUTABLE_SECTION,
            section_dependencies: vec![LINKED_FRONTEND_SECTION],
            input_dependencies: vec![],
            content_digest: [0; 32],
            payload: vec![],
        });
    }
    PersistentCompilerDatabase {
        semantic_digest,
        build_configuration_digest,
        inputs,
        stable_ids,
        sections,
    }
}

fn linked_frontend_payload(database: &PersistentCompilerDatabase) -> Option<Vec<u8>> {
    let manifest = database
        .sections
        .iter()
        .find(|section| section.section_id == LINKED_FRONTEND_SECTION)?;
    if !manifest.payload.is_empty() {
        return Some(manifest.payload.clone());
    }
    let by_id = database
        .sections
        .iter()
        .map(|section| (section.section_id, section))
        .collect::<HashMap<_, _>>();
    let total = manifest
        .section_dependencies
        .iter()
        .try_fold(0_usize, |total, id| {
            total.checked_add(by_id.get(id)?.payload.len())
        })?;
    let mut payload = Vec::with_capacity(total);
    for id in &manifest.section_dependencies {
        payload.extend_from_slice(&by_id.get(id)?.payload);
    }
    Some(payload)
}

fn digest32_from_md5(first: [u8; 16]) -> Digest {
    let second = md5::compute(first).0;
    let mut digest = [0; 32];
    digest[..16].copy_from_slice(&first);
    digest[16..].copy_from_slice(&second);
    digest
}

fn syntax_source_fingerprint(file: &dm_project::ProjectFile) -> u64 {
    file.compiler_text().map_or(0, |source| {
        hash_source(
            hash_source(
                0xcbf2_9ce4_8422_2325,
                file.relative_path.to_string_lossy().as_bytes(),
            ),
            source.as_bytes(),
        )
    })
}

fn encode_parsed_syntax_cache(project: &Project, syntax_files: &[Option<SyntaxFile>]) -> Vec<u8> {
    let mut output = Vec::new();
    output.extend_from_slice(PARSED_SYNTAX_CACHE_MAGIC);
    syntax_cache_write_u64(&mut output, PARSED_SYNTAX_FRONTEND_FINGERPRINT);
    syntax_cache_write_len(&mut output, syntax_files.len());
    for (file, syntax) in project.files.iter().zip(syntax_files) {
        syntax_cache_write_u64(&mut output, syntax_source_fingerprint(file));
        let Some(syntax) = syntax else {
            output.push(0);
            continue;
        };
        output.push(1);
        syntax_cache_write_len(&mut output, syntax.definitions.len());
        for definition in &syntax.definitions {
            syntax_cache_write_strings(&mut output, definition.path.segments());
            output.push(match definition.kind {
                DefinitionKind::Type => 0,
                DefinitionKind::Procedure => 1,
                DefinitionKind::ProcedureOverride => 2,
                DefinitionKind::Verb => 3,
                DefinitionKind::Variable => 4,
                DefinitionKind::VariableOverride => 5,
            });
            match definition.parent {
                Some(parent) => {
                    output.push(1);
                    syntax_cache_write_len(&mut output, parent);
                }
                None => output.push(0),
            }
            syntax_cache_write_indentation(&mut output, definition.indentation);
            syntax_cache_write_span(&mut output, definition.span);
            syntax_cache_write_tokens(&mut output, &definition.header);
            syntax_cache_write_len(&mut output, definition.parameters.len());
            for parameter in &definition.parameters {
                syntax_cache_write_span(&mut output, parameter.span);
                syntax_cache_write_tokens(&mut output, &parameter.tokens);
            }
            syntax_cache_write_len(&mut output, definition.body.len());
            for line in &definition.body {
                syntax_cache_write_indentation(&mut output, line.indentation);
                syntax_cache_write_span(&mut output, line.span);
                syntax_cache_write_tokens(&mut output, &line.tokens);
            }
        }
    }
    output
}

fn decode_parsed_syntax_cache(
    bytes: &[u8],
    project: &Project,
) -> Option<(Vec<Option<SyntaxFile>>, Vec<Diagnostic>, bool, usize)> {
    let mut input = Cursor::new(bytes);
    let mut magic = vec![0; PARSED_SYNTAX_CACHE_MAGIC.len()];
    input.read_exact(&mut magic).ok()?;
    if magic != PARSED_SYNTAX_CACHE_MAGIC
        || syntax_cache_read_u64(&mut input)? != PARSED_SYNTAX_FRONTEND_FINGERPRINT
        || syntax_cache_read_len(&mut input)? != project.files.len()
    {
        return None;
    }
    let mut syntax_files = Vec::with_capacity(project.files.len());
    let mut diagnostics = Vec::new();
    let mut exact = true;
    let mut reused = 0;
    for file in &project.files {
        let cached_fingerprint = syntax_cache_read_u64(&mut input)?;
        let source_matches = cached_fingerprint == syntax_source_fingerprint(file);
        exact &= source_matches;
        let present = syntax_cache_read_byte(&mut input)?;
        let source_bearing = matches!(file.kind, FileKind::Environment | FileKind::Source);
        if source_bearing && source_matches && present == 1 {
            reused += 1;
        }
        if present == 0 {
            if !source_bearing {
                syntax_files.push(None);
                continue;
            }
            let parsed = parse_one_syntax_file(file, &mut diagnostics);
            if source_matches && parsed.is_some() {
                return None;
            }
            syntax_files.push(parsed);
            continue;
        }
        if present != 1 || !source_bearing {
            return None;
        }
        let definition_count = syntax_cache_read_len(&mut input)?;
        let mut definitions = Vec::with_capacity(definition_count.min(1_000_000));
        for definition_index in 0..definition_count {
            let path = DefinitionPath::new(syntax_cache_read_strings(&mut input)?);
            let kind = match syntax_cache_read_byte(&mut input)? {
                0 => DefinitionKind::Type,
                1 => DefinitionKind::Procedure,
                2 => DefinitionKind::ProcedureOverride,
                3 => DefinitionKind::Verb,
                4 => DefinitionKind::Variable,
                5 => DefinitionKind::VariableOverride,
                _ => return None,
            };
            let parent = match syntax_cache_read_byte(&mut input)? {
                0 => None,
                1 => Some(syntax_cache_read_len(&mut input)?),
                _ => return None,
            };
            if parent.is_some_and(|parent| parent >= definition_index) {
                return None;
            }
            let indentation = syntax_cache_read_indentation(&mut input)?;
            let span = syntax_cache_read_span(&mut input)?;
            let header = syntax_cache_read_tokens(&mut input)?;
            let parameter_count = syntax_cache_read_len(&mut input)?;
            let mut parameters = Vec::with_capacity(parameter_count.min(1_000_000));
            for _ in 0..parameter_count {
                parameters.push(ParameterSyntax {
                    span: syntax_cache_read_span(&mut input)?,
                    tokens: syntax_cache_read_tokens(&mut input)?,
                });
            }
            let line_count = syntax_cache_read_len(&mut input)?;
            let mut body = Vec::with_capacity(line_count.min(1_000_000));
            for _ in 0..line_count {
                body.push(SourceLine {
                    indentation: syntax_cache_read_indentation(&mut input)?,
                    span: syntax_cache_read_span(&mut input)?,
                    tokens: syntax_cache_read_tokens(&mut input)?,
                });
            }
            definitions.push(Definition {
                path,
                kind,
                parent,
                indentation,
                span,
                header,
                parameters,
                body,
            });
        }
        let cached = Some(SyntaxFile { definitions });
        syntax_files.push(if source_matches {
            cached
        } else {
            parse_one_syntax_file(file, &mut diagnostics)
        });
    }
    (input.position() == bytes.len() as u64).then_some((syntax_files, diagnostics, exact, reused))
}

fn syntax_cache_write_tokens(output: &mut Vec<u8>, tokens: &[SpannedToken]) {
    syntax_cache_write_len(output, tokens.len());
    for token in tokens {
        match &token.kind {
            TokenKind::LineStart { tabs, spaces } => {
                output.push(0);
                syntax_cache_write_len(output, *tabs);
                syntax_cache_write_len(output, *spaces);
            }
            TokenKind::Newline => output.push(1),
            TokenKind::LineContinuation => output.push(2),
            TokenKind::Identifier(value) => syntax_cache_write_tagged_string(output, 3, value),
            TokenKind::Number(value) => syntax_cache_write_tagged_string(output, 4, value),
            TokenKind::String(value) => syntax_cache_write_tagged_string(output, 5, value),
            TokenKind::RawString(value) => syntax_cache_write_tagged_string(output, 6, value),
            TokenKind::TextBlock(value) => syntax_cache_write_tagged_string(output, 7, value),
            TokenKind::Resource(value) => syntax_cache_write_tagged_string(output, 8, value),
            TokenKind::Punctuation(value) => {
                output.push(9);
                output.extend_from_slice(&u32::from(*value).to_le_bytes());
            }
            TokenKind::Operator(value) => syntax_cache_write_tagged_string(output, 10, value),
        }
        syntax_cache_write_span(output, token.span);
    }
}

fn syntax_cache_read_tokens(input: &mut Cursor<&[u8]>) -> Option<Vec<SpannedToken>> {
    let count = syntax_cache_read_len(input)?;
    let mut tokens = Vec::with_capacity(count.min(1_000_000));
    for _ in 0..count {
        let kind = match syntax_cache_read_byte(input)? {
            0 => TokenKind::LineStart {
                tabs: syntax_cache_read_len(input)?,
                spaces: syntax_cache_read_len(input)?,
            },
            1 => TokenKind::Newline,
            2 => TokenKind::LineContinuation,
            3 => TokenKind::Identifier(syntax_cache_read_string(input)?),
            4 => TokenKind::Number(syntax_cache_read_string(input)?),
            5 => TokenKind::String(syntax_cache_read_string(input)?),
            6 => TokenKind::RawString(syntax_cache_read_string(input)?),
            7 => TokenKind::TextBlock(syntax_cache_read_string(input)?),
            8 => TokenKind::Resource(syntax_cache_read_string(input)?),
            9 => {
                let mut bytes = [0; 4];
                input.read_exact(&mut bytes).ok()?;
                TokenKind::Punctuation(char::from_u32(u32::from_le_bytes(bytes))?)
            }
            10 => TokenKind::Operator(syntax_cache_read_string(input)?),
            _ => return None,
        };
        tokens.push(SpannedToken {
            kind,
            span: syntax_cache_read_span(input)?,
        });
    }
    Some(tokens)
}

fn syntax_cache_write_tagged_string(output: &mut Vec<u8>, tag: u8, value: &str) {
    output.push(tag);
    syntax_cache_write_string(output, value);
}

fn syntax_cache_write_strings(output: &mut Vec<u8>, values: &[String]) {
    syntax_cache_write_len(output, values.len());
    for value in values {
        syntax_cache_write_string(output, value);
    }
}

fn syntax_cache_read_strings(input: &mut Cursor<&[u8]>) -> Option<Vec<String>> {
    let count = syntax_cache_read_len(input)?;
    if count == 0 {
        return None;
    }
    let mut values = Vec::with_capacity(count.min(1_000_000));
    for _ in 0..count {
        values.push(syntax_cache_read_string(input)?);
    }
    Some(values)
}

fn syntax_cache_write_indentation(output: &mut Vec<u8>, indentation: Indentation) {
    syntax_cache_write_len(output, indentation.tabs);
    syntax_cache_write_len(output, indentation.spaces);
}

fn syntax_cache_read_indentation(input: &mut Cursor<&[u8]>) -> Option<Indentation> {
    Some(Indentation {
        tabs: syntax_cache_read_len(input)?,
        spaces: syntax_cache_read_len(input)?,
    })
}

fn syntax_cache_write_span(output: &mut Vec<u8>, span: SourceSpan) {
    syntax_cache_write_len(output, span.start);
    syntax_cache_write_len(output, span.end);
}

fn syntax_cache_read_span(input: &mut Cursor<&[u8]>) -> Option<SourceSpan> {
    let start = syntax_cache_read_len(input)?;
    let end = syntax_cache_read_len(input)?;
    (start <= end).then(|| SourceSpan::new(start, end))
}

fn syntax_cache_write_string(output: &mut Vec<u8>, value: &str) {
    syntax_cache_write_len(output, value.len());
    output.extend_from_slice(value.as_bytes());
}

fn syntax_cache_read_string(input: &mut Cursor<&[u8]>) -> Option<String> {
    let length = syntax_cache_read_len(input)?;
    let remaining = input
        .get_ref()
        .len()
        .checked_sub(usize::try_from(input.position()).ok()?)?;
    if length > remaining {
        return None;
    }
    let mut bytes = vec![0; length];
    input.read_exact(&mut bytes).ok()?;
    String::from_utf8(bytes).ok()
}

fn syntax_cache_write_len(output: &mut Vec<u8>, value: usize) {
    syntax_cache_write_u64(output, value as u64);
}

fn syntax_cache_write_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn syntax_cache_read_len(input: &mut Cursor<&[u8]>) -> Option<usize> {
    usize::try_from(syntax_cache_read_u64(input)?).ok()
}

fn syntax_cache_read_u64(input: &mut Cursor<&[u8]>) -> Option<u64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).ok()?;
    Some(u64::from_le_bytes(bytes))
}

fn syntax_cache_read_byte(input: &mut Cursor<&[u8]>) -> Option<u8> {
    let mut byte = [0];
    input.read_exact(&mut byte).ok()?;
    Some(byte[0])
}

const COMPILATION_ARTIFACT_MAGIC: &[u8] = b"DREAM64-COMPILATION\0\x01";
const MAX_COMPILATION_ARTIFACT_ITEMS: usize = 16_777_216;

fn compilation_artifact_read_bytes<'artifact>(
    input: &mut Cursor<&'artifact [u8]>,
    what: &str,
) -> Result<&'artifact [u8], String> {
    let length = syntax_cache_read_len(input)
        .ok_or_else(|| format!("compiled frontend is truncated at {what} length"))?;
    let start = usize::try_from(input.position())
        .map_err(|_| format!("compiled frontend {what} offset exceeds this platform"))?;
    let end = start
        .checked_add(length)
        .filter(|end| *end <= input.get_ref().len())
        .ok_or_else(|| format!("compiled frontend {what} section is truncated"))?;
    input.set_position(end as u64);
    Ok(&input.get_ref()[start..end])
}

fn compilation_artifact_read_len(input: &mut Cursor<&[u8]>, what: &str) -> Result<usize, String> {
    syntax_cache_read_len(input)
        .ok_or_else(|| format!("compiled frontend is truncated while reading {what}"))
}

fn compilation_artifact_read_count(input: &mut Cursor<&[u8]>, what: &str) -> Result<usize, String> {
    let count = compilation_artifact_read_len(input, what)?;
    if count > MAX_COMPILATION_ARTIFACT_ITEMS {
        return Err(format!(
            "compiled frontend {what} exceeds the limit of {MAX_COMPILATION_ARTIFACT_ITEMS}"
        ));
    }
    Ok(count)
}

fn compilation_artifact_read_span(input: &mut Cursor<&[u8]>) -> Result<SourceSpan, String> {
    let start = compilation_artifact_read_len(input, "source span start")?;
    let end = compilation_artifact_read_len(input, "source span end")?;
    if start > end {
        return Err("compiled frontend contains an inverted source span".to_owned());
    }
    Ok(SourceSpan::new(start, end))
}

fn compilation_artifact_write_location(
    output: &mut Vec<u8>,
    location: Option<&DiagnosticLocation>,
) {
    let Some(location) = location else {
        output.push(0);
        return;
    };
    output.push(1);
    syntax_cache_write_len(output, location.file_id.index());
    syntax_cache_write_string(output, &location.path.to_string_lossy());
    match location.span {
        Some(span) => {
            output.push(1);
            syntax_cache_write_span(output, span);
        }
        None => output.push(0),
    }
}

fn compilation_artifact_read_location(
    input: &mut Cursor<&[u8]>,
    project: &Project,
) -> Result<Option<DiagnosticLocation>, String> {
    match syntax_cache_read_byte(input)
        .ok_or_else(|| "compiled frontend is truncated at diagnostic location".to_owned())?
    {
        0 => Ok(None),
        1 => {
            let file_id = FileId::from_index(compilation_artifact_read_len(
                input,
                "diagnostic file identity",
            )?);
            if file_id.index() >= project.files.len() {
                return Err(format!(
                    "compiled frontend diagnostic references missing file {}",
                    file_id.index()
                ));
            }
            let path = PathBuf::from(
                syntax_cache_read_string(input)
                    .ok_or_else(|| "compiled frontend has invalid diagnostic path".to_owned())?,
            );
            let span = match syntax_cache_read_byte(input)
                .ok_or_else(|| "compiled frontend is truncated at diagnostic span tag".to_owned())?
            {
                0 => None,
                1 => Some(compilation_artifact_read_span(input)?),
                tag => {
                    return Err(format!(
                        "compiled frontend has invalid diagnostic span tag {tag}"
                    ));
                }
            };
            Ok(Some(DiagnosticLocation {
                file_id,
                path,
                span,
            }))
        }
        tag => Err(format!(
            "compiled frontend has invalid diagnostic location tag {tag}"
        )),
    }
}

const fn compilation_artifact_diagnostic_kind(kind: DiagnosticKind) -> u8 {
    match kind {
        DiagnosticKind::Syntax => 0,
        DiagnosticKind::DuplicateFileUnit => 1,
        DiagnosticKind::DuplicateDeclaration => 2,
        DiagnosticKind::ConflictingNodeKind => 3,
        DiagnosticKind::MalformedMemberPath => 4,
        DiagnosticKind::SuspiciousMatrixCall => 5,
        DiagnosticKind::ProcArgumentGlobal => 6,
        DiagnosticKind::FallbackBuiltinArgument => 7,
        DiagnosticKind::BadArgument => 8,
        DiagnosticKind::InvalidArgumentCount => 9,
        DiagnosticKind::InvalidStringInterpolation => 10,
        DiagnosticKind::ReadOnlyAssignment => 11,
        DiagnosticKind::WriteToConstant => 12,
        DiagnosticKind::UntypedDereference => 13,
        DiagnosticKind::InvalidIndexOperation => 14,
        DiagnosticKind::RuntimeSearchOperator => 15,
        DiagnosticKind::InvalidVarType => 16,
        DiagnosticKind::UndefinedType => 17,
        DiagnosticKind::FinalVariableOverride => 18,
        DiagnosticKind::ConflictingVariableModifier => 19,
        DiagnosticKind::GlobalVariableReinitialization => 20,
        DiagnosticKind::InvalidNameofTarget => 21,
        DiagnosticKind::InvalidConstantInitializer => 22,
        DiagnosticKind::MissingResource => 23,
        DiagnosticKind::InvalidWeightedPick => 24,
        DiagnosticKind::InvalidExpandedDeclarationPath => 25,
        DiagnosticKind::ReturnTypeRedefinition => 26,
    }
}

fn compilation_artifact_read_diagnostic_kind(
    input: &mut Cursor<&[u8]>,
) -> Result<DiagnosticKind, String> {
    match syntax_cache_read_byte(input)
        .ok_or_else(|| "compiled frontend is truncated at diagnostic kind".to_owned())?
    {
        0 => Ok(DiagnosticKind::Syntax),
        1 => Ok(DiagnosticKind::DuplicateFileUnit),
        2 => Ok(DiagnosticKind::DuplicateDeclaration),
        3 => Ok(DiagnosticKind::ConflictingNodeKind),
        4 => Ok(DiagnosticKind::MalformedMemberPath),
        5 => Ok(DiagnosticKind::SuspiciousMatrixCall),
        6 => Ok(DiagnosticKind::ProcArgumentGlobal),
        7 => Ok(DiagnosticKind::FallbackBuiltinArgument),
        8 => Ok(DiagnosticKind::BadArgument),
        9 => Ok(DiagnosticKind::InvalidArgumentCount),
        10 => Ok(DiagnosticKind::InvalidStringInterpolation),
        11 => Ok(DiagnosticKind::ReadOnlyAssignment),
        12 => Ok(DiagnosticKind::WriteToConstant),
        13 => Ok(DiagnosticKind::UntypedDereference),
        14 => Ok(DiagnosticKind::InvalidIndexOperation),
        15 => Ok(DiagnosticKind::RuntimeSearchOperator),
        16 => Ok(DiagnosticKind::InvalidVarType),
        17 => Ok(DiagnosticKind::UndefinedType),
        18 => Ok(DiagnosticKind::FinalVariableOverride),
        19 => Ok(DiagnosticKind::ConflictingVariableModifier),
        20 => Ok(DiagnosticKind::GlobalVariableReinitialization),
        21 => Ok(DiagnosticKind::InvalidNameofTarget),
        22 => Ok(DiagnosticKind::InvalidConstantInitializer),
        23 => Ok(DiagnosticKind::MissingResource),
        24 => Ok(DiagnosticKind::InvalidWeightedPick),
        25 => Ok(DiagnosticKind::InvalidExpandedDeclarationPath),
        26 => Ok(DiagnosticKind::ReturnTypeRedefinition),
        tag => Err(format!(
            "compiled frontend has unknown diagnostic kind {tag}"
        )),
    }
}

fn compilation_artifact_write_stats(output: &mut Vec<u8>, stats: CompilationStats) {
    for value in [
        stats.project_files as u64,
        stats.parsed_files as u64,
        stats.project_bytes,
        stats.definitions as u64,
        stats.code_nodes as u64,
        stats.code_declarations as u64,
        stats.notes as u64,
        stats.warnings as u64,
        stats.errors as u64,
    ] {
        syntax_cache_write_u64(output, value);
    }
}

fn compilation_artifact_read_stats(input: &mut Cursor<&[u8]>) -> Result<CompilationStats, String> {
    let mut next = || {
        syntax_cache_read_u64(input)
            .ok_or_else(|| "compiled frontend is truncated in its statistics".to_owned())
    };
    let project_files = usize::try_from(next()?)
        .map_err(|_| "compiled frontend project file count is too large".to_owned())?;
    let parsed_files = usize::try_from(next()?)
        .map_err(|_| "compiled frontend parsed file count is too large".to_owned())?;
    let project_bytes = next()?;
    let definitions = usize::try_from(next()?)
        .map_err(|_| "compiled frontend definition count is too large".to_owned())?;
    let code_nodes = usize::try_from(next()?)
        .map_err(|_| "compiled frontend node count is too large".to_owned())?;
    let code_declarations = usize::try_from(next()?)
        .map_err(|_| "compiled frontend tree declaration count is too large".to_owned())?;
    let notes = usize::try_from(next()?)
        .map_err(|_| "compiled frontend note count is too large".to_owned())?;
    let warnings = usize::try_from(next()?)
        .map_err(|_| "compiled frontend warning count is too large".to_owned())?;
    let errors = usize::try_from(next()?)
        .map_err(|_| "compiled frontend error count is too large".to_owned())?;
    Ok(CompilationStats {
        project_files,
        parsed_files,
        project_bytes,
        definitions,
        code_nodes,
        code_declarations,
        notes,
        warnings,
        errors,
    })
}

fn compile_project(project: Project) -> Compilation {
    let (syntax_files, syntax_diagnostics) = parse_project_syntax(&project);
    compile_project_from_syntax(project, syntax_files, syntax_diagnostics)
}

fn parse_project_syntax(project: &Project) -> (Vec<Option<SyntaxFile>>, Vec<Diagnostic>) {
    let mut syntax_files = Vec::with_capacity(project.files.len());
    let mut diagnostics = Vec::new();

    for file in &project.files {
        if !matches!(file.kind, FileKind::Environment | FileKind::Source) {
            syntax_files.push(None);
            continue;
        }

        syntax_files.push(parse_one_syntax_file(file, &mut diagnostics));
    }
    (syntax_files, diagnostics)
}

fn parse_one_syntax_file(
    file: &dm_project::ProjectFile,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<SyntaxFile> {
    // Source-bearing project files have already been UTF-8 validated by the
    // loader while it scanned preprocessing directives.
    let source = file
        .compiler_text()
        .expect("project loader validates source-bearing files as UTF-8");
    match dm_syntax::parse(source) {
        Ok(syntax) => Some(syntax),
        Err(error) => {
            let compiler_span = syntax_error_span(&error);
            diagnostics.push(syntax_diagnostic(
                file.id,
                file.path.clone(),
                file.original_span(compiler_span),
                &error,
            ));
            None
        }
    }
}

fn syntax_error_span(error: &SyntaxError) -> SourceSpan {
    match error {
        SyntaxError::Lex(error) => error.span,
        SyntaxError::UnclosedDelimiter(span) | SyntaxError::InfixNewline { span, .. } => *span,
    }
}

fn compile_project_from_syntax(
    project: Project,
    syntax_files: Vec<Option<SyntaxFile>>,
    mut diagnostics: Vec<Diagnostic>,
) -> Compilation {
    let parsed_files = syntax_files
        .iter()
        .filter(|syntax| syntax.is_some())
        .count();
    let definitions = syntax_files
        .iter()
        .flatten()
        .map(|syntax| syntax.definitions.len())
        .sum();

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
    diagnostics.extend(variable_modifier_diagnostics(
        &project,
        &syntax_files,
        &tree,
    ));
    diagnostics.extend(constant_cycle_diagnostics(&project, &syntax_files));
    diagnostics.extend(procedure_return_override_diagnostics(
        &project,
        &syntax_files,
        &tree,
    ));
    for (index, syntax) in syntax_files.iter().enumerate() {
        let Some(syntax) = syntax else { continue };
        let file = &project.files[index];
        diagnostics.extend(preprocessed_source_diagnostics(file));
        diagnostics.extend(matrix_lint_diagnostics(file, syntax));
        diagnostics.extend(proc_argument_diagnostics(file, syntax));
        diagnostics.extend(numeric_builtin_diagnostics(file, syntax));
        diagnostics.extend(builtin_arity_diagnostics(file, syntax));
        diagnostics.extend(string_interpolation_diagnostics(file, syntax));
        diagnostics.extend(readonly_member_diagnostics(file, syntax));
        diagnostics.extend(const_write_diagnostics(file, syntax));
        diagnostics.extend(reference_operator_diagnostics(file, syntax));
        diagnostics.extend(variable_type_diagnostics(file, syntax));
        diagnostics.extend(undefined_local_type_diagnostics(file, syntax, &tree));
        diagnostics.extend(nameof_diagnostics(file, syntax));
        diagnostics.extend(constant_initializer_diagnostics(file, syntax));
        diagnostics.extend(resource_and_weighted_pick_diagnostics(file, syntax));
    }
    diagnostics.retain_mut(|diagnostic| {
        let Some(severity) = project.diagnostic_severity(diagnostic_pragma_name(diagnostic)) else {
            return diagnostic.kind != DiagnosticKind::RuntimeSearchOperator;
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
        DiagnosticKind::WriteToConstant => "WriteToConstant",
        DiagnosticKind::UntypedDereference => "InvalidReference",
        DiagnosticKind::InvalidIndexOperation => "InvalidIndexOperation",
        DiagnosticKind::RuntimeSearchOperator => "RuntimeSearchOperator",
        DiagnosticKind::InvalidVarType => "InvalidVarType",
        DiagnosticKind::UndefinedType => "UnknownType",
        DiagnosticKind::FinalVariableOverride => "InvalidOverride",
        DiagnosticKind::ConflictingVariableModifier => "InvalidOverride",
        DiagnosticKind::GlobalVariableReinitialization => "InvalidOverride",
        DiagnosticKind::InvalidNameofTarget => "BadArgument",
        DiagnosticKind::InvalidConstantInitializer => "InvalidInitialValue",
        DiagnosticKind::MissingResource => "MissingResource",
        DiagnosticKind::InvalidWeightedPick => "BadArgument",
        DiagnosticKind::InvalidExpandedDeclarationPath => "BadToken",
        DiagnosticKind::ReturnTypeRedefinition => "InvalidReturnType",
    }
}

fn variable_type_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut declared_types: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for definition in &syntax.definitions {
        if definition.kind != dm_syntax::DefinitionKind::Variable {
            continue;
        }
        let Some(expected) = type_annotation(&definition.header) else {
            continue;
        };
        let segments = definition.path.segments();
        let Some(var_index) = segments.iter().position(|segment| segment == "var") else {
            continue;
        };
        let owner = format!("/{}", segments[..var_index].join("/"));
        let name = segments.last().expect("variable path has a name").clone();
        declared_types
            .entry(name)
            .or_default()
            .push((owner, expected));
    }

    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Variable | dm_syntax::DefinitionKind::VariableOverride
        ) {
            let segments = definition.path.segments();
            let Some(name) = segments.last() else {
                continue;
            };
            let explicit = type_annotation(&definition.header);
            if let Some(expected) = explicit.as_deref() {
                if initializer_conflicts(&definition.header, expected) {
                    push_variable_type_diagnostic(
                        file,
                        definition.span,
                        DiagnosticKind::InvalidVarType,
                        "variable initializer conflicts with its declared type",
                        &mut diagnostics,
                    );
                }
            } else if definition.kind == dm_syntax::DefinitionKind::VariableOverride {
                let owner_end = segments
                    .iter()
                    .position(|segment| segment == "var")
                    .unwrap_or(segments.len() - 1);
                let owner = format!("/{}", segments[..owner_end].join("/"));
                if let Some(expected) = nearest_declared_type(name, &owner, &declared_types)
                    && initializer_conflicts(&definition.header, expected)
                {
                    push_variable_type_diagnostic(
                        file,
                        definition.span,
                        DiagnosticKind::InvalidVarType,
                        "variable override conflicts with inherited type",
                        &mut diagnostics,
                    );
                }
            }
        }
        if matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            lint_typed_local_new_assignments(file, definition, &mut diagnostics);
        }
    }
    diagnostics
}

fn undefined_local_type_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
    tree: &CodeTree,
) -> Vec<Diagnostic> {
    const BUILTIN_TYPES: &[&str] = &[
        "area",
        "atom",
        "client",
        "database",
        "datum",
        "icon",
        "image",
        "list",
        "matrix",
        "mob",
        "mutable_appearance",
        "obj",
        "regex",
        "savefile",
        "sound",
        "turf",
        "world",
    ];
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Variable | dm_syntax::DefinitionKind::VariableOverride
        ) {
            continue;
        }
        let Some((_, _, Some(type_path))) = local_declaration(&definition.header) else {
            continue;
        };
        let type_segments = type_path
            .trim_start_matches('/')
            .split('/')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if type_segments.is_empty()
            || (type_segments.len() == 1 && BUILTIN_TYPES.contains(&type_segments[0].as_str()))
            || tree
                .find(&dm_syntax::DefinitionPath::new(type_segments.clone()))
                .is_some()
        {
            continue;
        }
        push_reference_diagnostic(
            file,
            definition.span,
            DiagnosticKind::UndefinedType,
            DiagnosticSeverity::Error,
            &format!("undefined variable type: /{}", type_segments.join("/")),
            &mut diagnostics,
        );
    }
    for definition in &syntax.definitions {
        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            continue;
        }
        for line in &definition.body {
            let Some((_, _, Some(type_path))) = local_declaration(&line.tokens) else {
                continue;
            };
            let segments = type_path
                .trim_start_matches('/')
                .split('/')
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if segments.is_empty()
                || (segments.len() == 1 && BUILTIN_TYPES.contains(&segments[0].as_str()))
                || tree
                    .find(&dm_syntax::DefinitionPath::new(segments))
                    .is_some()
            {
                continue;
            }
            let span = line.tokens.first().map_or(line.span, |token| token.span);
            push_reference_diagnostic(
                file,
                span,
                DiagnosticKind::UndefinedType,
                DiagnosticSeverity::Error,
                &format!("undefined variable type: {type_path}"),
                &mut diagnostics,
            );
        }
    }
    diagnostics
}

fn constant_cycle_diagnostics(
    project: &Project,
    syntax_files: &[Option<SyntaxFile>],
) -> Vec<Diagnostic> {
    let mut declarations = HashMap::<String, (FileId, SourceSpan, Vec<String>)>::new();
    for (file_index, syntax) in syntax_files.iter().enumerate() {
        let Some(syntax) = syntax else { continue };
        for definition in &syntax.definitions {
            let segments = definition.path.segments();
            if definition.kind != dm_syntax::DefinitionKind::Variable
                || segments.first().is_none_or(|segment| segment != "var")
                || !has_identifier(&definition.header, "const")
            {
                continue;
            }
            let Some(name) = segments.last().cloned() else {
                continue;
            };
            let dependencies = definition
                .header
                .iter()
                .skip_while(|token| {
                    !matches!(&token.kind, TokenKind::Operator(operator) if operator == "=")
                })
                .skip(1)
                .filter_map(|token| match &token.kind {
                    TokenKind::Identifier(identifier) => Some(identifier.clone()),
                    _ => None,
                })
                .collect();
            declarations.insert(
                name,
                (
                    FileId::from_index(file_index),
                    definition.span,
                    dependencies,
                ),
            );
        }
    }
    let graph = declarations
        .iter()
        .map(|(name, (_, _, dependencies))| (name.clone(), dependencies.clone()))
        .collect::<HashMap<_, _>>();
    let mut diagnostics = Vec::new();
    for (name, (file_id, span, _)) in declarations {
        let Some(dependencies) = graph.get(&name) else {
            continue;
        };
        let cyclic = dependencies.iter().any(|dependency| {
            dependency == &name
                || constant_dependency_reaches(dependency, &name, &graph, &mut HashSet::new())
        });
        if !cyclic {
            continue;
        }
        let Some(file) = project.file(file_id) else {
            continue;
        };
        push_reference_diagnostic(
            file,
            span,
            DiagnosticKind::InvalidConstantInitializer,
            DiagnosticSeverity::Error,
            "constant initializer contains a cyclic reference",
            &mut diagnostics,
        );
    }
    diagnostics
}

fn constant_dependency_reaches(
    current: &str,
    target: &str,
    graph: &HashMap<String, Vec<String>>,
    visited: &mut HashSet<String>,
) -> bool {
    if !visited.insert(current.to_owned()) {
        return false;
    }
    graph.get(current).is_some_and(|dependencies| {
        dependencies.iter().any(|dependency| {
            dependency == target || constant_dependency_reaches(dependency, target, graph, visited)
        })
    })
}

fn constant_initializer_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    const CONSTANT_CALLS: &[&str] = &[
        "abs", "arccos", "arcsin", "arctan", "ckey", "ckeyEx", "cos", "rgb", "sin", "sqrt", "tan",
    ];
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Variable | dm_syntax::DefinitionKind::VariableOverride
        ) && has_identifier(&definition.header, "const")
        {
            let invalid_call = definition.header.windows(2).any(|window| {
                matches!(&window[0].kind, TokenKind::Identifier(name) if !CONSTANT_CALLS.contains(&name.as_str()))
                    && window[1].kind == TokenKind::Punctuation('(')
            });
            if invalid_call {
                push_reference_diagnostic(
                    file,
                    definition.span,
                    DiagnosticKind::InvalidConstantInitializer,
                    DiagnosticSeverity::Error,
                    "constant initializer calls a runtime procedure",
                    &mut diagnostics,
                );
            }
        }

        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            continue;
        }
        let parameter_names = definition
            .parameters
            .iter()
            .filter_map(|parameter| {
                parameter
                    .tokens
                    .iter()
                    .rev()
                    .find_map(|token| match &token.kind {
                        TokenKind::Identifier(name) => Some(name.clone()),
                        _ => None,
                    })
            })
            .collect::<HashSet<_>>();
        let mut local_names = HashSet::new();
        for line in &definition.body {
            let Some((name, _, _)) = local_declaration(&line.tokens) else {
                continue;
            };
            let is_static =
                has_identifier(&line.tokens, "static") || has_identifier(&line.tokens, "global");
            let equals = line.tokens.iter().position(
                |token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="),
            );
            if is_static
                && equals.is_some_and(|equals| {
                    line.tokens[equals + 1..].iter().any(|token| {
                        matches!(&token.kind, TokenKind::Identifier(value) if parameter_names.contains(value) || local_names.contains(value))
                    })
                })
            {
                push_reference_diagnostic(
                    file,
                    line.span,
                    DiagnosticKind::InvalidConstantInitializer,
                    DiagnosticSeverity::Error,
                    "static initializer references a runtime local value",
                    &mut diagnostics,
                );
            }
            local_names.insert(name);
        }
    }
    diagnostics
}

fn has_identifier(tokens: &[SpannedToken], expected: &str) -> bool {
    tokens
        .iter()
        .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == expected))
}

fn resource_and_weighted_pick_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        lint_resources_and_weighted_pick(file, &definition.header, &mut diagnostics);
        for line in &definition.body {
            lint_resources_and_weighted_pick(file, &line.tokens, &mut diagnostics);
        }
    }
    diagnostics
}

fn lint_resources_and_weighted_pick(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    diagnostics: &mut Vec<Diagnostic>,
) {
    for token in tokens {
        let TokenKind::Resource(resource) = &token.kind else {
            continue;
        };
        let relative = resource.replace('\\', "/");
        let exists = file
            .path
            .parent()
            .is_some_and(|parent| parent.join(relative).is_file());
        if !exists {
            push_reference_diagnostic(
                file,
                token.span,
                DiagnosticKind::MissingResource,
                DiagnosticSeverity::Error,
                &format!("resource file does not exist: '{resource}'"),
                diagnostics,
            );
        }
    }

    let mut index = 0usize;
    while index + 1 < tokens.len() {
        if !matches!(&tokens[index].kind, TokenKind::Identifier(name) if name == "pick")
            || tokens[index + 1].kind != TokenKind::Punctuation('(')
        {
            index += 1;
            continue;
        }
        let Some((close, _)) = call_arguments(tokens, index + 1) else {
            index += 1;
            continue;
        };
        let mut depth = 0usize;
        let mut entry_start = index + 2;
        for cursor in index + 2..close {
            match tokens[cursor].kind {
                TokenKind::Punctuation('(' | '[' | '{') => depth += 1,
                TokenKind::Punctuation(')' | ']' | '}') => depth = depth.saturating_sub(1),
                TokenKind::Punctuation(',') if depth == 0 => entry_start = cursor + 1,
                TokenKind::Punctuation(';') if depth == 0 => {
                    let weight = &tokens[entry_start..cursor];
                    let calls_procedure = weight.windows(2).any(|window| {
                        matches!(&window[0].kind, TokenKind::Identifier(_))
                            && window[1].kind == TokenKind::Punctuation('(')
                    });
                    if calls_procedure && !weight.is_empty() {
                        let span = SourceSpan::new(
                            weight[0].span.start,
                            weight.last().expect("non-empty weight").span.end,
                        );
                        push_reference_diagnostic(
                            file,
                            span,
                            DiagnosticKind::InvalidWeightedPick,
                            DiagnosticSeverity::Error,
                            "weighted pick weights cannot call procedures",
                            diagnostics,
                        );
                    }
                }
                _ => {}
            }
        }
        index = close + 1;
    }
}

fn preprocessed_source_diagnostics(file: &dm_project::ProjectFile) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    if let Ok(source) = file.text()
        && let Err(error) = dm_lexer::lex(source)
        && error.message == "unterminated block comment"
    {
        diagnostics.push(Diagnostic {
            kind: DiagnosticKind::Syntax,
            severity: DiagnosticSeverity::Error,
            message: error.message,
            location: Some(DiagnosticLocation {
                file_id: file.id,
                path: file.path.clone(),
                span: Some(error.span),
            }),
            related: None,
        });
    }

    let Ok(source) = file.compiler_text() else {
        return diagnostics;
    };
    if let Ok(tokens) = dm_lexer::lex(source) {
        let mut saw_var = false;
        let mut saw_initializer = false;
        for (index, token) in tokens.iter().enumerate() {
            if matches!(token.kind, TokenKind::Newline) {
                saw_var = false;
                saw_initializer = false;
                continue;
            }
            if matches!(&token.kind, TokenKind::Identifier(name) if name == "var") {
                saw_var = true;
            }
            if matches!(&token.kind, TokenKind::Operator(operator) if operator == "=") {
                saw_initializer = true;
            }
            if saw_var
                && !saw_initializer
                && matches!(&token.kind, TokenKind::Operator(operator) if operator == "/")
                && matches!(
                    tokens.get(index + 1).map(|token| &token.kind),
                    Some(TokenKind::Number(_))
                )
            {
                let number = &tokens[index + 1];
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::InvalidExpandedDeclarationPath,
                    severity: DiagnosticSeverity::Error,
                    message: "numeric value cannot be used as a declaration path component"
                        .to_owned(),
                    location: Some(DiagnosticLocation {
                        file_id: file.id,
                        path: file.path.clone(),
                        span: Some(file.original_span(number.span)),
                    }),
                    related: None,
                });
            }
        }
    }
    diagnostics
}

fn procedure_return_override_diagnostics(
    project: &Project,
    syntax_files: &[Option<SyntaxFile>],
    tree: &CodeTree,
) -> Vec<Diagnostic> {
    let definition_for = |declaration: &dm_object_tree::TreeDeclaration| {
        syntax_files
            .get(declaration.file_id.index())?
            .as_ref()?
            .definitions
            .get(declaration.definition_index)
    };
    let mut diagnostics = Vec::new();
    for node in tree
        .nodes()
        .iter()
        .filter(|node| node.kind == dm_object_tree::NodeKind::Procedure)
    {
        let Some(inherited) = node
            .inherited_member
            .and_then(|inherited| tree.node(inherited))
        else {
            continue;
        };
        let inherited_return = inherited
            .declarations
            .iter()
            .filter_map(|id| tree.declaration(*id))
            .filter_map(&definition_for)
            .find_map(|definition| type_annotation(&definition.header));
        let Some(inherited_return) = inherited_return else {
            continue;
        };
        for declaration in node
            .declarations
            .iter()
            .filter_map(|id| tree.declaration(*id))
        {
            let Some(definition) = definition_for(declaration) else {
                continue;
            };
            let Some(current_return) = type_annotation(&definition.header) else {
                continue;
            };
            if current_return != inherited_return {
                push_tree_semantic_diagnostic(
                    project,
                    declaration,
                    DiagnosticKind::ReturnTypeRedefinition,
                    "procedure return type cannot be redefined from parent",
                    &mut diagnostics,
                );
            }
        }
    }
    diagnostics
}

fn variable_modifier_diagnostics(
    project: &Project,
    syntax_files: &[Option<SyntaxFile>],
    tree: &CodeTree,
) -> Vec<Diagnostic> {
    let definition_for = |declaration: &dm_object_tree::TreeDeclaration| {
        syntax_files
            .get(declaration.file_id.index())?
            .as_ref()?
            .definitions
            .get(declaration.definition_index)
    };
    let has_modifier = |definition: &dm_syntax::Definition, modifier: &str| {
        definition
            .header
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == modifier))
    };
    let initialized = |definition: &dm_syntax::Definition| {
        definition
            .header
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
    };
    let mut diagnostics = Vec::new();

    let mut const_state = HashMap::<Vec<String>, bool>::new();
    for (file_index, syntax) in syntax_files.iter().enumerate() {
        let Some(syntax) = syntax else { continue };
        let Some(file) = project.files.get(file_index) else {
            continue;
        };
        for definition in &syntax.definitions {
            if !matches!(
                definition.kind,
                dm_syntax::DefinitionKind::Variable | dm_syntax::DefinitionKind::VariableOverride
            ) {
                continue;
            }
            let segments = definition.path.segments();
            let Some(name) = segments.last() else {
                continue;
            };
            let mut key = segments[..segments.len() - 1]
                .iter()
                .filter(|segment| {
                    !matches!(
                        segment.as_str(),
                        "var" | "const" | "static" | "global" | "tmp" | "final"
                    )
                })
                .cloned()
                .collect::<Vec<_>>();
            key.push(name.clone());
            let is_const = has_modifier(definition, "const")
                || segments.iter().any(|segment| segment == "const");
            if let Some(previous) = const_state.insert(key, is_const)
                && previous != is_const
            {
                diagnostics.push(Diagnostic {
                    kind: DiagnosticKind::ConflictingVariableModifier,
                    severity: DiagnosticSeverity::Error,
                    message: "variable redeclaration changes its const modifier".to_owned(),
                    location: Some(DiagnosticLocation {
                        file_id: file.id,
                        path: file.path.clone(),
                        span: Some(file.original_span(definition.span)),
                    }),
                    related: None,
                });
            }
        }
    }

    for node in tree
        .nodes()
        .iter()
        .filter(|node| node.kind == dm_object_tree::NodeKind::Variable)
    {
        let declarations = node
            .declarations
            .iter()
            .filter_map(|id| tree.declaration(*id))
            .collect::<Vec<_>>();

        let Some(inherited_id) = node.inherited_member else {
            continue;
        };
        let Some(inherited) = tree.node(inherited_id) else {
            continue;
        };
        let inherited_definitions = inherited
            .declarations
            .iter()
            .filter_map(|id| tree.declaration(*id))
            .filter_map(&definition_for)
            .collect::<Vec<_>>();
        let inherited_final = inherited_definitions
            .iter()
            .any(|definition| has_modifier(definition, "final"));
        let inherited_global = inherited_definitions.iter().any(|definition| {
            ["const", "global", "static"]
                .iter()
                .any(|modifier| has_modifier(definition, modifier))
        });
        for declaration in declarations {
            let Some(definition) = definition_for(declaration) else {
                continue;
            };
            if inherited_final {
                push_tree_semantic_diagnostic(
                    project,
                    declaration,
                    DiagnosticKind::FinalVariableOverride,
                    "final variable cannot be overridden",
                    &mut diagnostics,
                );
            } else if inherited_global && initialized(definition) {
                push_tree_semantic_diagnostic(
                    project,
                    declaration,
                    DiagnosticKind::GlobalVariableReinitialization,
                    "inherited global variable cannot be reinitialized",
                    &mut diagnostics,
                );
            }
        }
    }
    diagnostics
}

fn push_tree_semantic_diagnostic(
    project: &Project,
    declaration: &dm_object_tree::TreeDeclaration,
    kind: DiagnosticKind,
    message: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(file) = project.file(declaration.file_id) else {
        return;
    };
    diagnostics.push(Diagnostic {
        kind,
        severity: DiagnosticSeverity::Error,
        message: message.to_owned(),
        location: Some(DiagnosticLocation {
            file_id: file.id,
            path: file.path.clone(),
            span: Some(file.original_span(declaration.span)),
        }),
        related: None,
    });
}

fn type_annotation(tokens: &[SpannedToken]) -> Option<String> {
    let index = tokens
        .iter()
        .rposition(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "as"))?;
    match tokens.get(index + 1).map(|token| &token.kind) {
        Some(TokenKind::Identifier(name)) => Some(name.clone()),
        _ => None,
    }
}

fn initializer_conflicts(tokens: &[SpannedToken], expected: &str) -> bool {
    let Some(equal) = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Operator(operator) if operator == "="))
    else {
        return false;
    };
    match tokens.get(equal + 1).map(|token| &token.kind) {
        Some(TokenKind::String(_) | TokenKind::RawString(_) | TokenKind::TextBlock(_)) => {
            expected == "num"
        }
        Some(TokenKind::Number(_)) => expected == "text",
        _ => false,
    }
}

fn nearest_declared_type<'a>(
    name: &str,
    owner: &str,
    declarations: &'a HashMap<String, Vec<(String, String)>>,
) -> Option<&'a str> {
    declarations
        .get(name)?
        .iter()
        .filter(|(candidate, _)| owner == candidate || owner.starts_with(&format!("{candidate}/")))
        .max_by_key(|(candidate, _)| candidate.len())
        .map(|(_, expected)| expected.as_str())
}

fn lint_typed_local_new_assignments(
    file: &dm_project::ProjectFile,
    definition: &dm_syntax::Definition,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut local_types = HashMap::new();
    for line in &definition.body {
        if let Some((name, _, Some(type_path))) = local_declaration(&line.tokens) {
            local_types.insert(name, type_path);
        }
        for (index, token) in line.tokens.iter().enumerate() {
            let TokenKind::Identifier(name) = &token.kind else {
                continue;
            };
            let Some(expected) = local_types.get(name) else {
                continue;
            };
            if !matches!(line.tokens.get(index + 1).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "=")
                || !matches!(line.tokens.get(index + 2).map(|token| &token.kind), Some(TokenKind::Identifier(word)) if word == "new")
                || !matches!(line.tokens.get(index + 3).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/")
            {
                continue;
            }
            let Some(TokenKind::Identifier(actual)) =
                line.tokens.get(index + 4).map(|token| &token.kind)
            else {
                continue;
            };
            if expected != &format!("/{actual}")
                && !format!("/{actual}").starts_with(&format!("{expected}/"))
            {
                push_variable_type_diagnostic(
                    file,
                    token.span,
                    DiagnosticKind::InvalidVarType,
                    "new path conflicts with local variable type",
                    diagnostics,
                );
            }
        }
    }
}

fn push_variable_type_diagnostic(
    file: &dm_project::ProjectFile,
    span: SourceSpan,
    kind: DiagnosticKind,
    message: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    diagnostics.push(Diagnostic {
        kind,
        severity: DiagnosticSeverity::Warning,
        message: message.to_owned(),
        location: Some(DiagnosticLocation {
            file_id: file.id,
            path: file.path.clone(),
            span: Some(file.original_span(span)),
        }),
        related: None,
    });
}

fn reference_operator_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            continue;
        }
        let mut untyped = HashSet::new();
        let mut local_types = HashMap::new();
        for line in &definition.body {
            if let Some((name, _, type_path)) = local_declaration(&line.tokens) {
                if let Some(type_path) = type_path {
                    local_types.insert(name, type_path);
                } else {
                    untyped.insert(name);
                }
            }
            for (index, token) in line.tokens.iter().enumerate() {
                let TokenKind::Identifier(name) = &token.kind else {
                    continue;
                };
                if untyped.contains(name)
                    && matches!(line.tokens.get(index + 1).map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == ".")
                {
                    push_reference_diagnostic(
                        file,
                        token.span,
                        DiagnosticKind::UntypedDereference,
                        DiagnosticSeverity::Error,
                        "typed member access requires a statically typed value",
                        &mut diagnostics,
                    );
                }
                if local_types.get(name).is_some_and(|path| path == "/datum")
                    && matches!(
                        line.tokens.get(index + 1).map(|token| &token.kind),
                        Some(TokenKind::Punctuation('['))
                    )
                {
                    push_reference_diagnostic(
                        file,
                        token.span,
                        DiagnosticKind::InvalidIndexOperation,
                        DiagnosticSeverity::Warning,
                        "plain datum values cannot be indexed",
                        &mut diagnostics,
                    );
                }
            }
            for token in &line.tokens {
                if matches!(&token.kind, TokenKind::Operator(operator) if operator == ":") {
                    push_reference_diagnostic(
                        file,
                        token.span,
                        DiagnosticKind::RuntimeSearchOperator,
                        DiagnosticSeverity::Warning,
                        "runtime-search operator defers member validation until runtime",
                        &mut diagnostics,
                    );
                }
            }
        }
    }
    diagnostics
}

fn push_reference_diagnostic(
    file: &dm_project::ProjectFile,
    span: SourceSpan,
    kind: DiagnosticKind,
    severity: DiagnosticSeverity,
    message: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    diagnostics.push(Diagnostic {
        kind,
        severity,
        message: message.to_owned(),
        location: Some(DiagnosticLocation {
            file_id: file.id,
            path: file.path.clone(),
            span: Some(file.original_span(span)),
        }),
        related: None,
    });
}

fn const_write_diagnostics(file: &dm_project::ProjectFile, syntax: &SyntaxFile) -> Vec<Diagnostic> {
    let mut global_consts = HashSet::new();
    let mut field_consts: HashMap<String, HashSet<String>> = HashMap::new();
    for definition in &syntax.definitions {
        if !definition
            .header
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "const"))
        {
            continue;
        }
        let segments = definition.path.segments();
        let Some(name) = segments.last().cloned() else {
            continue;
        };
        if segments.first().is_some_and(|segment| segment == "var") {
            global_consts.insert(name);
        } else if let Some(var_index) = segments.iter().position(|segment| segment == "var") {
            let owner = format!("/{}", segments[..var_index].join("/"));
            field_consts.entry(owner).or_default().insert(name);
        }
    }

    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            continue;
        }
        let mut locals = HashSet::new();
        let mut local_types = HashMap::new();
        for line in &definition.body {
            if let Some((name, is_const, type_path)) = local_declaration(&line.tokens) {
                if is_const {
                    locals.insert(name.clone());
                }
                if let Some(type_path) = type_path {
                    local_types.insert(name, type_path);
                }
            }
            lint_const_writes(
                file,
                &line.tokens,
                &locals,
                &local_types,
                &global_consts,
                &field_consts,
                &mut diagnostics,
            );
        }
    }
    diagnostics
}

fn local_declaration(tokens: &[SpannedToken]) -> Option<(String, bool, Option<String>)> {
    let var_index = tokens
        .iter()
        .position(|token| matches!(&token.kind, TokenKind::Identifier(name) if name == "var"))?;
    let mut segments = Vec::new();
    let mut index = var_index + 1;
    while index + 1 < tokens.len()
        && matches!(&tokens[index].kind, TokenKind::Operator(operator) if operator == "/")
    {
        let TokenKind::Identifier(segment) = &tokens[index + 1].kind else {
            break;
        };
        segments.push(segment.clone());
        index += 2;
    }
    let name = segments.last()?.clone();
    let is_const = segments.iter().any(|segment| segment == "const");
    let type_segments: Vec<_> = segments[..segments.len() - 1]
        .iter()
        .filter(|segment| {
            !matches!(
                segment.as_str(),
                "const" | "static" | "global" | "tmp" | "final"
            )
        })
        .cloned()
        .collect();
    // BYOND's suffix array declaration syntax (`var/name[]`, `var/name[5]`,
    // and typed variants) declares a list value even though `list` is not a
    // path segment before the variable name.
    let is_array = matches!(
        tokens.get(index).map(|token| &token.kind),
        Some(TokenKind::Punctuation('['))
    );
    let type_path = if is_array {
        Some("/list".to_owned())
    } else {
        (!type_segments.is_empty()).then(|| format!("/{}", type_segments.join("/")))
    };
    Some((name, is_const, type_path))
}

#[allow(clippy::too_many_arguments)]
fn lint_const_writes(
    file: &dm_project::ProjectFile,
    tokens: &[SpannedToken],
    locals: &HashSet<String>,
    local_types: &HashMap<String, String>,
    globals: &HashSet<String>,
    fields: &HashMap<String, HashSet<String>>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    const ASSIGNMENTS: &[&str] = &[
        "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&&=", "||=", "%%=",
        "**=",
    ];
    for (index, token) in tokens.iter().enumerate() {
        let TokenKind::Identifier(name) = &token.kind else {
            continue;
        };
        let declaration = tokens[..index]
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(word) if word == "var"));
        let assigned = tokens.get(index + 1).is_some_and(|token| matches!(&token.kind, TokenKind::Operator(operator) if ASSIGNMENTS.contains(&operator.as_str())));
        let loop_write = tokens.get(index + 1).is_some_and(
            |token| matches!(&token.kind, TokenKind::Identifier(word) if word == "in"),
        ) && tokens[..index]
            .iter()
            .any(|token| matches!(&token.kind, TokenKind::Identifier(word) if word == "for"));
        let bare_const = !declaration
            && (assigned || loop_write)
            && (locals.contains(name) || globals.contains(name));
        let field_const = assigned
            && index >= 2
            && matches!(&tokens[index - 1].kind, TokenKind::Operator(operator) if operator == ".")
            && matches!(&tokens[index - 2].kind, TokenKind::Identifier(receiver) if local_types.get(receiver).is_some_and(|owner| fields.get(owner).is_some_and(|names| names.contains(name))));
        if bare_const || field_const {
            diagnostics.push(Diagnostic {
                kind: DiagnosticKind::WriteToConstant,
                severity: DiagnosticSeverity::Error,
                message: format!("cannot write to constant {name}"),
                location: Some(DiagnosticLocation {
                    file_id: file.id,
                    path: file.path.clone(),
                    span: Some(file.original_span(token.span)),
                }),
                related: None,
            });
        }
    }
}

fn readonly_member_diagnostics(
    file: &dm_project::ProjectFile,
    syntax: &SyntaxFile,
) -> Vec<Diagnostic> {
    const ASSIGNMENTS: &[&str] = &[
        "=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&&=", "||=", "%%=",
        "**=",
    ];
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        for line in &definition.body {
            for window in line.tokens.windows(3) {
                let member_access = matches!(&window[0].kind, TokenKind::Operator(operator) if matches!(operator.as_str(), "." | "?." | ":" | "?:"));
                let readonly =
                    matches!(&window[1].kind, TokenKind::Identifier(name) if name == "type");
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

fn nameof_diagnostics(file: &dm_project::ProjectFile, syntax: &SyntaxFile) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for definition in &syntax.definitions {
        if !matches!(
            definition.kind,
            dm_syntax::DefinitionKind::Procedure
                | dm_syntax::DefinitionKind::ProcedureOverride
                | dm_syntax::DefinitionKind::Verb
        ) {
            continue;
        }
        let segments = definition.path.segments();
        let global_context = segments
            .iter()
            .position(|segment| matches!(segment.as_str(), "proc" | "verb"))
            .is_some_and(|index| index == 0);
        for line in &definition.body {
            let mut index = 0usize;
            while index + 1 < line.tokens.len() {
                if !matches!(&line.tokens[index].kind, TokenKind::Identifier(name) if name == "nameof")
                    || line.tokens[index + 1].kind != TokenKind::Punctuation('(')
                {
                    index += 1;
                    continue;
                }
                let Some((close, arguments)) = call_arguments(&line.tokens, index + 1) else {
                    index += 1;
                    continue;
                };
                let valid =
                    arguments.len() == 1 && valid_nameof_target(arguments[0], global_context);
                if !valid {
                    let span =
                        SourceSpan::new(line.tokens[index].span.start, line.tokens[close].span.end);
                    push_reference_diagnostic(
                        file,
                        span,
                        DiagnosticKind::InvalidNameofTarget,
                        DiagnosticSeverity::Error,
                        "nameof() requires a variable, member, procedure reference, or type path",
                        &mut diagnostics,
                    );
                }
                index = close + 1;
            }
        }
    }
    diagnostics
}

fn valid_nameof_target(tokens: &[SpannedToken], global_context: bool) -> bool {
    if tokens.len() == 1 {
        return matches!(&tokens[0].kind, TokenKind::Identifier(name) if name != "__TYPE__" || !global_context);
    }
    let path = matches!(tokens.first().map(|token| &token.kind), Some(TokenKind::Operator(operator)) if operator == "/");
    if path {
        return tokens.iter().enumerate().all(|(index, token)| {
            if index % 2 == 0 {
                matches!(&token.kind, TokenKind::Operator(operator) if operator == "/")
            } else {
                matches!(&token.kind, TokenKind::Identifier(_))
            }
        });
    }
    tokens.iter().enumerate().all(|(index, token)| {
        if index % 2 == 0 {
            matches!(&token.kind, TokenKind::Identifier(_))
        } else {
            matches!(&token.kind, TokenKind::Operator(operator) if operator == ".")
        }
    })
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

    use super::{
        BuildMode, Compilation, CompilerDatabase, DiagnosticKind, parsed_syntax_cache_path,
    };

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

    #[test]
    fn compiled_frontend_artifact_round_trips_without_recompiling_the_tree() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write(
            "types.dm",
            "/datum/base\n\tvar/value = 7\n\tproc/read()\n\t\treturn value\n/datum/child\n\tparent_type = /datum/base\n\tread()\n\t\treturn ..()\n",
        );
        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("fixture should compile");

        let encoded = compilation.encode_compiled_artifact();
        assert_eq!(encoded, compilation.encode_compiled_artifact());
        let segments = compilation.encode_compiled_artifact_segments();
        assert_eq!(segments.len(), 7);
        assert_eq!(segments.concat(), encoded);
        let decoded =
            Compilation::decode_compiled_artifact(&encoded).expect("artifact should decode");
        assert_eq!(decoded.stats(), compilation.stats());
        assert_eq!(decoded.declarations(), compilation.declarations());
        assert_eq!(decoded.diagnostics(), compilation.diagnostics());
        assert_eq!(decoded.code_tree(), compilation.code_tree());
        assert_eq!(
            decoded.project().content_fingerprint(),
            compilation.project().content_fingerprint()
        );
        assert!(
            decoded
                .project()
                .files
                .iter()
                .map(|file| decoded.syntax(file.id))
                .eq(compilation
                    .project()
                    .files
                    .iter()
                    .map(|file| compilation.syntax(file.id)))
        );

        let mut bad_header = encoded.clone();
        bad_header[0] ^= 0xff;
        assert!(Compilation::decode_compiled_artifact(&bad_header).is_err());
        assert!(Compilation::decode_compiled_artifact(&encoded[..encoded.len() - 1]).is_err());
        let mut trailing = encoded;
        trailing.push(0);
        assert!(Compilation::decode_compiled_artifact(&trailing).is_err());
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
    fn validates_typed_variable_initialization_overrides_and_new_paths() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma InvalidVarType error\n",
                "/datum/base\n",
                "\tvar/value = 5 as num\n",
                "\tvar/optional = null as num\n",
                "\tvar/text_value = \"ok\" as text\n",
                "/datum/base/child\n",
                "\tvalue = \"bad\"\n",
                "/proc/run()\n",
                "\tvar/turf/location\n",
                "\tlocation = new /obj\n",
                "\tvar/obj/item\n",
                "\titem = new /obj\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("typed variable diagnostics should be recoverable");

        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::InvalidVarType)
                .count(),
            2
        );
    }

    #[test]
    fn validates_typed_untyped_index_and_runtime_search_reference_operators() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "#pragma InvalidIndexOperation error\n",
                "#pragma RuntimeSearchOperator error\n",
                "/proc/run()\n",
                "\tvar/untyped = new /obj\n",
                "\tvar/datum/plain = new\n",
                "\tvar/list/items = list()\n",
                "\tvar/obj/typed = new\n",
                "\tuntyped.value\n",
                "\tplain[\"key\"]\n",
                "\ttyped:dynamic_proc()\n",
                "\ttyped.value\n",
                "\titems[1]\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("reference diagnostics should be recoverable");

        for kind in [
            DiagnosticKind::UntypedDereference,
            DiagnosticKind::InvalidIndexOperation,
            DiagnosticKind::RuntimeSearchOperator,
        ] {
            assert_eq!(
                compilation
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| diagnostic.kind == kind)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn suffix_array_declarations_are_statically_lists() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/proc/run()\n",
                "\tvar/untyped = new /obj\n",
                "\tvar/array[5]\n",
                "\tvar/datum/typed_array[]\n",
                "\tuntyped.value\n",
                "\tarray.len\n",
                "\ttyped_array.len\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("array declarations should compile");

        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::UntypedDereference)
                .count(),
            1,
            "only the genuinely untyped value should be rejected"
        );
    }

    #[test]
    fn rejects_undefined_local_type_paths_without_flagging_real_types() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/datum/known\n",
                "/datum/holder\n",
                "\tvar/datum/missing/field\n",
                "/proc/run()\n",
                "\tvar/datum/absent/bad\n",
                "\tvar/datum/known/good\n",
                "\tvar/obj/builtin\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("undefined type diagnostics should be recoverable");

        let diagnostics = compilation
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.kind == DiagnosticKind::UndefinedType)
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("/datum/missing"))
        );
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("/datum/absent"))
        );
    }

    #[test]
    fn enforces_inherited_variable_modifiers() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/datum\n",
                "\tvar/final/final_value = 1\n",
                "\tvar/ordinary = 1\n",
                "/datum/child\n",
                "\tfinal_value = 2\n",
                "\tordinary = 2\n",
                "/atom/var/a = 5\n",
                "/atom/const/a = 4\n",
                "/datum/var/const/global_value = 5\n",
                "/turf/global_value = 4\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("modifier diagnostics should be recoverable");

        for kind in [
            DiagnosticKind::FinalVariableOverride,
            DiagnosticKind::ConflictingVariableModifier,
            DiagnosticKind::GlobalVariableReinitialization,
        ] {
            assert_eq!(
                compilation
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| diagnostic.kind == kind)
                    .count(),
                1,
                "expected exactly one {kind:?} diagnostic"
            );
        }
    }

    #[test]
    fn validates_nameof_targets_by_syntax_and_scope() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "/datum/proc/scoped()\n",
                "\tnameof(__TYPE__)\n",
                "\tnameof(src.name)\n",
                "/proc/run()\n",
                "\tvar/list/items = list()\n",
                "\tnameof(__TYPE__)\n",
                "\tnameof(items[1])\n",
                "\tnameof(items)\n",
                "\tnameof(/datum)\n",
                "\tnameof(/proc/run)\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("nameof diagnostics should be recoverable");

        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::InvalidNameofTarget)
                .count(),
            2
        );
    }

    #[test]
    fn rejects_cyclic_and_runtime_constant_initializers() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "var/const/A = B\n",
                "var/const/B = A\n",
                "/proc/runtime_value()\n",
                "\treturn 1\n",
                "var/const/bad_call = rgb(runtime_value(), 0, 0)\n",
                "var/const/good_call = rgb(1, 2, 3)\n",
                "/proc/run(var/datum/item)\n",
                "\tvar/static/bad_static = item.type\n",
                "\tvar/static/good_static = /datum\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("constant diagnostics should be recoverable");

        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| {
                    diagnostic.kind == DiagnosticKind::InvalidConstantInitializer
                })
                .count(),
            4,
            "two cycle members, one runtime call, and one local static initializer"
        );
    }

    #[test]
    fn validates_resource_literals_and_weighted_pick_weights() {
        let fixture = TestProject::new();
        fixture.write("asset.txt", "available");
        fixture.write(
            "world.dme",
            concat!(
                "/proc/run()\n",
                "\tvar/weight = 50\n",
                "\tvar/good_resource = 'asset.txt'\n",
                "\tvar/bad_resource = 'missing.txt'\n",
                "\tvar/good_pick = pick(weight; 1, 20; 2)\n",
                "\tvar/bad_pick = pick(prob(50); 1)\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("resource and pick diagnostics should be recoverable");

        for kind in [
            DiagnosticKind::MissingResource,
            DiagnosticKind::InvalidWeightedPick,
        ] {
            assert_eq!(
                compilation
                    .diagnostics()
                    .iter()
                    .filter(|diagnostic| diagnostic.kind == kind)
                    .count(),
                1
            );
        }
    }

    #[test]
    fn rejects_preprocessed_paths_comments_and_return_type_redefinitions() {
        let macro_fixture = TestProject::new();
        macro_fixture.write(
            "world.dme",
            "#define NAME 1\nvar/const/NAME = 5\n/proc/run()\n\tvar/const/NAME = 6\n",
        );
        let macro_compilation = CompilerDatabase::new()
            .compile(macro_fixture.path("world.dme"))
            .expect("expanded path diagnostics should be recoverable");
        assert_eq!(
            macro_compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| {
                    diagnostic.kind == DiagnosticKind::InvalidExpandedDeclarationPath
                })
                .count(),
            2
        );

        let comment_fixture = TestProject::new();
        comment_fixture.write(
            "world.dme",
            "/*\n/*\n*/\n*/\n/*\n// */\n/proc/run()\n\treturn\n",
        );
        let comment_compilation = CompilerDatabase::new()
            .compile(comment_fixture.path("world.dme"))
            .expect("comment diagnostics should be recoverable");
        assert!(
            comment_compilation.diagnostics().iter().any(|diagnostic| {
                diagnostic.kind == DiagnosticKind::Syntax
                    && diagnostic.message.starts_with("unterminated block comment")
            }),
            "diagnostics: {:?}",
            comment_compilation.diagnostics()
        );

        let return_fixture = TestProject::new();
        return_fixture.write(
            "world.dme",
            concat!(
                "/datum/proc/value() as num\n",
                "\treturn 1\n",
                "/datum/child/value() as text\n",
                "\treturn \"bad\"\n",
                "/datum/valid/value()\n",
                "\treturn \"inherited annotation\"\n",
            ),
        );
        let return_compilation = CompilerDatabase::new()
            .compile(return_fixture.path("world.dme"))
            .expect("return override diagnostics should be recoverable");
        assert_eq!(
            return_compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::ReturnTypeRedefinition)
                .count(),
            1
        );
    }

    #[test]
    fn runtime_search_lint_is_disabled_without_pragma() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "/proc/run(obj/value)\n\tvalue:dynamic_proc()\n",
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("disabled lint should compile");

        assert!(
            compilation
                .diagnostics()
                .iter()
                .all(|diagnostic| diagnostic.kind != DiagnosticKind::RuntimeSearchOperator)
        );
    }

    #[test]
    fn rejects_const_writes_and_loop_reuse_without_flagging_initializers() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            concat!(
                "var/const/global_value = 1\n",
                "/obj/var/const/fixed = 2\n",
                "/obj/var/mutable = 3\n",
                "/datum/var/fixed = 4\n",
                "/proc/run()\n",
                "\tvar/const/local_value = 5\n",
                "\tglobal_value = 2\n",
                "\tlocal_value = 6\n",
                "\tfor(local_value in 1 to 3)\n",
                "\t\treturn\n",
                "\tvar/obj/object = new\n",
                "\tobject.fixed = 7\n",
                "\tobject.mutable = 8\n",
                "\tvar/datum/other = new\n",
                "\tother.fixed = 9\n",
            ),
        );

        let compilation = CompilerDatabase::new()
            .compile(fixture.path("world.dme"))
            .expect("constant diagnostics should be recoverable");
        assert_eq!(
            compilation
                .diagnostics()
                .iter()
                .filter(|diagnostic| diagnostic.kind == DiagnosticKind::WriteToConstant)
                .count(),
            4
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
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic
                .location
                .as_ref()
                .and_then(|item| item.span)
                .is_some()
        }));
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

    #[test]
    fn parsed_syntax_cache_hits_and_invalidates_with_project_and_format() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/example\n\tvar/value = 3\n");
        let cache = fixture.path("cache/project.bin");
        let database = CompilerDatabase::new();

        let (cold, cold_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("cold cached compilation should succeed");
        assert!(!cold_cache.project_snapshot_hit);
        assert!(!cold_cache.parsed_syntax_hit);
        assert_eq!(cold_cache.syntax_files_reused, 0);
        assert_eq!(cold_cache.syntax_files_reparsed, 2);

        let (warm, warm_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("warm cached compilation should succeed");
        assert!(warm_cache.project_snapshot_hit);
        assert!(warm_cache.parsed_syntax_hit);
        assert_eq!(warm_cache.syntax_files_reused, 2);
        assert_eq!(warm_cache.syntax_files_reparsed, 0);
        assert_eq!(cold.stats(), warm.stats());

        let syntax_cache = parsed_syntax_cache_path(&cache);
        fs::write(&syntax_cache, b"obsolete-format")
            .expect("syntax cache corruption should be writable");
        let (_, corrupt_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("a corrupt syntax cache should rebuild safely");
        assert!(corrupt_cache.project_snapshot_hit);
        assert!(!corrupt_cache.parsed_syntax_hit);
        let (_, repaired_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("the rebuilt syntax cache should be reusable");
        assert!(repaired_cache.parsed_syntax_hit);

        fixture.write(
            "types.dm",
            "/datum/example\n\tvar/value = 300\n\tvar/changed = TRUE\n",
        );
        let (changed, changed_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("a changed project should invalidate both cache tiers");
        assert!(!changed_cache.project_snapshot_hit);
        assert!(!changed_cache.parsed_syntax_hit);
        assert_eq!(changed_cache.syntax_files_reused, 1);
        assert_eq!(changed_cache.syntax_files_reparsed, 1);
        assert_eq!(changed.stats().definitions, 3);
    }

    #[test]
    fn explicit_build_modes_control_persistent_cache_reuse() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/example\n\tvar/value = 3\n");
        let cache = fixture.path("cache/project.bin");
        let database = CompilerDatabase::new();

        let (_, cold) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Incremental)
            .expect("cold incremental compilation should populate caches");
        assert_eq!(cold.build_mode, BuildMode::Incremental);
        assert!(!cold.project_snapshot_hit);

        let (_, warm) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Incremental)
            .expect("warm incremental compilation should reuse caches");
        assert!(warm.project_snapshot_hit);
        assert!(warm.parsed_syntax_hit);

        let (_, fresh) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Fresh)
            .expect("fresh compilation should bypass caches");
        assert_eq!(fresh.build_mode, BuildMode::Fresh);
        assert!(!fresh.project_snapshot_hit);
        assert!(!fresh.parsed_syntax_hit);

        let (_, still_warm) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Incremental)
            .expect("fresh compilation must not disturb persistent caches");
        assert!(still_warm.project_snapshot_hit);
        assert!(still_warm.parsed_syntax_hit);

        let (_, clean) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Clean)
            .expect("clean compilation should rebuild and repopulate caches");
        assert_eq!(clean.build_mode, BuildMode::Clean);
        assert!(!clean.project_snapshot_hit);
        assert!(!clean.parsed_syntax_hit);

        let (_, after_clean) = database
            .compile_with_mode(fixture.path("world.dme"), &cache, BuildMode::Incremental)
            .expect("clean compilation should leave reusable caches");
        assert!(after_clean.project_snapshot_hit);
        assert!(after_clean.parsed_syntax_hit);
    }

    #[test]
    fn persistent_database_reuses_linked_stage_and_invalidates_dependents() {
        let fixture = TestProject::new();
        fixture.write("world.dme", "#include \"types.dm\"\n");
        fixture.write("types.dm", "/datum/example\n\tvar/value = 3\n");
        let database_file = fixture.path("cache/project.d64cdb");
        let database = CompilerDatabase::new();

        let (cold, cold_stats) = database
            .compile_persistent(
                fixture.path("world.dme"),
                &database_file,
                BuildMode::Incremental,
                [1; 32],
                [2; 32],
            )
            .expect("cold persistent compilation should succeed");
        assert_eq!(cold_stats.linked_sections_rebuilt, 1);
        assert_eq!(cold_stats.linked_sections_reused, 0);
        let first_database =
            crate::persistent_database::PersistentCompilerDatabase::read(&database_file).unwrap();
        let original_type_id = first_database
            .stable_ids
            .iter()
            .find(|entry| entry.namespace == "type" && entry.name == "/datum/example")
            .unwrap()
            .id;

        let (warm, warm_stats) = database
            .compile_persistent(
                fixture.path("world.dme"),
                &database_file,
                BuildMode::Incremental,
                [1; 32],
                [2; 32],
            )
            .expect("warm persistent compilation should restore linked output");
        assert_eq!(warm_stats.linked_sections_reused, 1);
        assert_eq!(warm_stats.linked_sections_rebuilt, 0);
        assert_eq!(warm.stats(), cold.stats());

        fixture.write("types.dm", "/aaa\n/datum/example\n\tvar/value = 4\n");
        let (changed, changed_stats) = database
            .compile_persistent(
                fixture.path("world.dme"),
                &database_file,
                BuildMode::Incremental,
                [1; 32],
                [2; 32],
            )
            .expect("changed persistent compilation should rebuild dependents");
        assert_eq!(changed_stats.changed_inputs, 1);
        assert_eq!(changed_stats.syntax_files_reused, 1);
        assert_eq!(changed_stats.syntax_files_reparsed, 1);
        assert_eq!(changed_stats.linked_sections_rebuilt, 1);
        assert!(changed_stats.invalidated_sections >= 3);
        assert_eq!(changed.stats().definitions, cold.stats().definitions + 1);
        let changed_database =
            crate::persistent_database::PersistentCompilerDatabase::read(&database_file).unwrap();
        assert_eq!(
            changed_database
                .stable_ids
                .iter()
                .find(|entry| entry.namespace == "type" && entry.name == "/datum/example")
                .unwrap()
                .id,
            original_type_id,
        );
    }

    #[test]
    fn parsed_syntax_cache_preserves_errors_without_discarding_valid_files() {
        let fixture = TestProject::new();
        fixture.write(
            "world.dme",
            "#include \"valid.dm\"\n#include \"invalid.dm\"\n",
        );
        fixture.write("valid.dm", "/datum/example\n\tvar/value = 3\n");
        fixture.write(
            "invalid.dm",
            "/datum/broken\n\tvar/value = \"unterminated\n",
        );
        let cache = fixture.path("cache/project.bin");
        let database = CompilerDatabase::new();

        let (cold, cold_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("cold compilation with a syntax diagnostic should complete");
        assert!(!cold_cache.parsed_syntax_hit);
        let (warm, warm_cache) = database
            .compile_cached_with_stats(fixture.path("world.dme"), &cache)
            .expect("warm compilation should restore valid files and reproduce the error");
        assert!(warm_cache.parsed_syntax_hit);
        assert_eq!(cold.stats(), warm.stats());
        assert_eq!(cold.diagnostics().len(), warm.diagnostics().len());
        assert_eq!(cold.diagnostics()[0].message, warm.diagnostics()[0].message);
        assert!(warm.syntax(warm.project().files[1].id).is_some());
        assert!(warm.syntax(warm.project().files[2].id).is_none());
    }
}
