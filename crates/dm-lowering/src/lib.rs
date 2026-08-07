//! Recoverable, project-wide lowering from DM semantics to portable VM programs.
//!
//! Every semantic procedure implementation is attempted independently. A body
//! that cannot be lowered therefore becomes an inventory diagnostic instead of
//! preventing unrelated, supported bodies from becoming executable.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use dm_compiler::{Compilation, CompilerDatabase, CompilerError};
use dm_core::{FileId, SourceSpan};
use dm_semantics::{ProcedureImplementation, ProcedureImplementationId, ProcedureRegistry};
use dm_vm::{Instruction, Module, ProcedureSpec, Program};

/// Broad, stable reason that an implementation could not be lowered.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum UnsupportedCategory {
    /// The semantic implementation no longer has a corresponding syntax body.
    MissingDefinition,
    /// The definition or body is not executable procedure syntax.
    Definition,
    /// Indentation made the procedure body structurally invalid.
    Indentation,
    /// A procedure parameter is unsupported or malformed.
    Parameter,
    /// Local-variable declaration or lookup failed.
    Local,
    /// A named procedure call could not be resolved.
    CallResolution,
    /// A parent call had no target or its dependency chain could not lower.
    ParentCall,
    /// A statement form is outside the current VM subset.
    Statement,
    /// An expression or operator is outside the current VM subset.
    Expression,
    /// The failure does not yet have a more specific stable category.
    Other,
}

impl UnsupportedCategory {
    /// Returns a stable CLI/report spelling.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MissingDefinition => "missing-definition",
            Self::Definition => "definition",
            Self::Indentation => "indentation",
            Self::Parameter => "parameter",
            Self::Local => "local",
            Self::CallResolution => "call-resolution",
            Self::ParentCall => "parent-call",
            Self::Statement => "statement",
            Self::Expression => "expression",
            Self::Other => "other",
        }
    }
}

/// One implementation that can execute in the current VM subset.
#[derive(Clone, Debug)]
pub struct LoweredImplementation {
    /// Semantic identity of the source body.
    pub implementation: ProcedureImplementationId,
    /// Canonical procedure path shared by its override chain.
    pub procedure_path: String,
    /// Physical source file containing the implementation.
    pub file_id: FileId,
    /// Project-relative physical source path.
    pub source_path: String,
    /// Header span mapped back to the original, unmasked source bytes.
    pub original_span: SourceSpan,
    /// Executable program for this exact implementation.
    pub program: Program,
    /// Parent-linked module when this implementation requires an override chain.
    pub module: Option<Module>,
    /// Module-local index of this implementation when [`Self::module`] exists.
    pub module_entry_index: Option<usize>,
}

/// Source-mapped record for one implementation outside the current VM subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoweringDiagnostic {
    /// Stable unsupported-feature category.
    pub category: UnsupportedCategory,
    /// Detailed diagnostic produced by the lowering attempt.
    pub message: String,
    /// Semantic identity of the source body.
    pub implementation: ProcedureImplementationId,
    /// Canonical procedure path shared by its override chain.
    pub procedure_path: String,
    /// Physical source file containing the implementation.
    pub file_id: FileId,
    /// Project-relative physical source path.
    pub source_path: String,
    /// Header span in compiler-view bytes.
    pub compiler_span: SourceSpan,
    /// Header span mapped back to original, unmasked source bytes.
    pub original_span: SourceSpan,
}

/// Deterministic aggregate for one unsupported-feature category.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockerSummary {
    /// Stable unsupported-feature category.
    pub category: UnsupportedCategory,
    /// Number of implementations blocked by this category.
    pub count: usize,
    /// First detailed diagnostic in semantic registry order.
    pub example_message: String,
}

/// Deterministic project-wide lowering counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LoweringStats {
    /// Canonical procedures present in the semantic registry.
    pub procedures: usize,
    /// Source implementations attempted.
    pub implementations: usize,
    /// Implementations successfully lowered.
    pub lowered: usize,
    /// Lowered implementations retaining an exact parent-linked module.
    pub parent_linked: usize,
    /// Implementations outside the current executable subset.
    pub unsupported: usize,
}

/// Recoverable executable inventory for one frontend compilation.
#[derive(Debug)]
pub struct LoweringInventory {
    lowered: Vec<LoweredImplementation>,
    diagnostics: Vec<LoweringDiagnostic>,
    blockers: Vec<BlockerSummary>,
    stats: LoweringStats,
}

impl LoweringInventory {
    /// Returns successful implementations in deterministic semantic order.
    #[must_use]
    pub fn lowered(&self) -> &[LoweredImplementation] {
        &self.lowered
    }

    /// Returns source-mapped failures in deterministic semantic order.
    #[must_use]
    pub fn diagnostics(&self) -> &[LoweringDiagnostic] {
        &self.diagnostics
    }

    /// Returns blocker categories by descending frequency, then stable label.
    #[must_use]
    pub fn blockers(&self) -> &[BlockerSummary] {
        &self.blockers
    }

    /// Returns deterministic project counters.
    #[must_use]
    pub const fn stats(&self) -> &LoweringStats {
        &self.stats
    }

    /// Finds one successfully lowered semantic implementation.
    #[must_use]
    pub fn implementation(
        &self,
        implementation: ProcedureImplementationId,
    ) -> Option<&LoweredImplementation> {
        self.lowered
            .iter()
            .find(|lowered| lowered.implementation == implementation)
    }
}

/// Loads a `.dme` project and inventories every semantic implementation.
///
/// Unsupported procedure syntax is retained in the returned inventory. Only
/// project discovery or source-loading failures abort this operation.
///
/// # Errors
///
/// Returns [`CompilerError`] when the frontend cannot discover or load the
/// project.
pub fn lower_project(root_file: impl AsRef<Path>) -> Result<LoweringInventory, CompilerError> {
    let compilation = CompilerDatabase::new().compile(root_file)?;
    Ok(lower_compilation(&compilation))
}

/// Attempts every implementation in an existing frontend compilation.
#[must_use]
pub fn lower_compilation(compilation: &Compilation) -> LoweringInventory {
    let registry = ProcedureRegistry::build(compilation);
    let mut lowered = Vec::new();
    let mut diagnostics = Vec::new();

    for procedure in registry.procedures() {
        for implementation in &procedure.implementations {
            let context =
                SourceContext::new(compilation, implementation, procedure.path.to_string());
            let Some(definition) = compilation
                .syntax(implementation.file_id)
                .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
            else {
                diagnostics.push(context.diagnostic(
                    UnsupportedCategory::MissingDefinition,
                    "semantic implementation has no retained syntax definition".to_owned(),
                ));
                continue;
            };

            match dm_vm::compile_procedure(definition) {
                Ok(program) if !has_unresolved_parent_call(&program) => {
                    lowered.push(context.lowered(program, None, None));
                }
                Ok(_) if implementation.parent_target.is_none() => {
                    diagnostics.push(context.diagnostic(
                        UnsupportedCategory::ParentCall,
                        "parent procedure call has no resolved semantic target".to_owned(),
                    ));
                }
                Ok(_) => match compile_parent_chain(compilation, &registry, implementation.id) {
                    Ok((module, entry_index, program)) => {
                        lowered.push(context.lowered(program, Some(module), Some(entry_index)));
                    }
                    Err(message) => diagnostics.push(context.diagnostic(
                        UnsupportedCategory::ParentCall,
                        format!("parent implementation chain could not lower: {message}"),
                    )),
                },
                Err(error) => diagnostics
                    .push(context.diagnostic(classify_message(&error.message), error.message)),
            }
        }
    }

    let blockers = summarize_blockers(&diagnostics);
    let stats = LoweringStats {
        procedures: registry.procedures().len(),
        implementations: lowered.len() + diagnostics.len(),
        lowered: lowered.len(),
        parent_linked: lowered.iter().filter(|body| body.module.is_some()).count(),
        unsupported: diagnostics.len(),
    };
    LoweringInventory {
        lowered,
        diagnostics,
        blockers,
        stats,
    }
}

struct SourceContext {
    implementation: ProcedureImplementationId,
    procedure_path: String,
    file_id: FileId,
    source_path: String,
    compiler_span: SourceSpan,
    original_span: SourceSpan,
}

impl SourceContext {
    fn new(
        compilation: &Compilation,
        implementation: &ProcedureImplementation,
        procedure_path: String,
    ) -> Self {
        let source_path = compilation
            .project()
            .file(implementation.file_id)
            .map_or_else(
                || format!("<file:{}>", implementation.file_id.index()),
                |file| file.relative_path.to_string_lossy().into_owned(),
            );
        Self {
            implementation: implementation.id,
            procedure_path,
            file_id: implementation.file_id,
            source_path,
            compiler_span: implementation.span,
            original_span: compilation
                .original_span(implementation.file_id, implementation.span)
                .unwrap_or(implementation.span),
        }
    }

    fn diagnostic(&self, category: UnsupportedCategory, message: String) -> LoweringDiagnostic {
        LoweringDiagnostic {
            category,
            message,
            implementation: self.implementation,
            procedure_path: self.procedure_path.clone(),
            file_id: self.file_id,
            source_path: self.source_path.clone(),
            compiler_span: self.compiler_span,
            original_span: self.original_span,
        }
    }

    fn lowered(
        &self,
        program: Program,
        module: Option<Module>,
        module_entry_index: Option<usize>,
    ) -> LoweredImplementation {
        LoweredImplementation {
            implementation: self.implementation,
            procedure_path: self.procedure_path.clone(),
            file_id: self.file_id,
            source_path: self.source_path.clone(),
            original_span: self.original_span,
            program,
            module,
            module_entry_index,
        }
    }
}

fn compile_parent_chain(
    compilation: &Compilation,
    registry: &ProcedureRegistry,
    target: ProcedureImplementationId,
) -> Result<(Module, usize, Program), String> {
    let mut reverse_chain = Vec::new();
    let mut seen = BTreeSet::new();
    let mut cursor = Some(target);
    while let Some(implementation_id) = cursor {
        if !seen.insert(implementation_id) {
            return Err("semantic parent implementation cycle".to_owned());
        }
        let implementation = registry
            .implementation(implementation_id)
            .ok_or_else(|| "semantic parent implementation is missing".to_owned())?;
        reverse_chain.push(implementation_id);
        cursor = implementation.parent_target;
    }
    reverse_chain.reverse();

    let indices: BTreeMap<_, _> = reverse_chain
        .iter()
        .copied()
        .enumerate()
        .map(|(index, implementation)| (implementation, index))
        .collect();
    let mut specs = Vec::with_capacity(reverse_chain.len());
    for implementation_id in &reverse_chain {
        let implementation = registry
            .implementation(*implementation_id)
            .ok_or_else(|| "semantic implementation disappeared".to_owned())?;
        let procedure = registry
            .procedure(implementation_id.procedure())
            .ok_or_else(|| "semantic procedure disappeared".to_owned())?;
        let definition = compilation
            .syntax(implementation.file_id)
            .and_then(|syntax| syntax.definitions.get(implementation.definition_index))
            .ok_or_else(|| format!("missing syntax definition for {}", procedure.path))?;
        let parent = implementation
            .parent_target
            .map(|parent| {
                indices.get(&parent).copied().ok_or_else(|| {
                    format!("parent implementation for {} is missing", procedure.path)
                })
            })
            .transpose()?;
        specs.push(ProcedureSpec {
            path: format!("{}@{}", procedure.path, implementation.ordinal),
            definition,
            parent,
            static_calls: BTreeMap::new(),
            src_fields: BTreeMap::new(),
            global_fields: BTreeMap::new(),
        });
    }
    let entry_index = reverse_chain
        .iter()
        .position(|implementation| *implementation == target)
        .ok_or_else(|| "target is missing from its semantic parent chain".to_owned())?;
    let module = dm_vm::compile_module_specs(&specs).map_err(|error| error.message)?;
    let program = module
        .procedure_id_at(entry_index)
        .and_then(|entry| module.procedure(entry))
        .cloned()
        .ok_or_else(|| "compiled parent chain has no target entry".to_owned())?;
    Ok((module, entry_index, program))
}

fn has_unresolved_parent_call(program: &Program) -> bool {
    program.instructions.iter().any(|instruction| {
        matches!(
            instruction,
            Instruction::CallParent {
                procedure: None,
                ..
            }
        )
    })
}

fn classify_message(message: &str) -> UnsupportedCategory {
    let lower = message.to_ascii_lowercase();
    if lower.contains("indent") {
        UnsupportedCategory::Indentation
    } else if lower.contains("parameter") || lower.contains("argument") {
        UnsupportedCategory::Parameter
    } else if lower.contains("local") {
        UnsupportedCategory::Local
    } else if lower.contains("unknown procedure") || lower.contains("call") {
        UnsupportedCategory::CallResolution
    } else if lower.contains("statement") || lower.contains("for-in") {
        UnsupportedCategory::Statement
    } else if lower.contains("expression") || lower.contains("operator") || lower.contains("token")
    {
        UnsupportedCategory::Expression
    } else if lower.contains("not executable") || lower.contains("definition") {
        UnsupportedCategory::Definition
    } else {
        UnsupportedCategory::Other
    }
}

fn summarize_blockers(diagnostics: &[LoweringDiagnostic]) -> Vec<BlockerSummary> {
    let mut grouped: BTreeMap<UnsupportedCategory, (usize, String)> = BTreeMap::new();
    for diagnostic in diagnostics {
        let entry = grouped
            .entry(diagnostic.category)
            .or_insert_with(|| (0, diagnostic.message.clone()));
        entry.0 += 1;
    }
    let mut blockers: Vec<_> = grouped
        .into_iter()
        .map(|(category, (count, example_message))| BlockerSummary {
            category,
            count,
            example_message,
        })
        .collect();
    blockers.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.category.cmp(&right.category))
    });
    blockers
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::CompilerDatabase;
    use dm_vm::{Value, execute_module};

    use super::{UnsupportedCategory, lower_compilation};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn new(environment: &str, files: &[(&str, &str)]) -> Self {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-lowering-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("test project directory should be created");
            fs::write(root.join("world.dme"), environment).expect("environment should be written");
            for (path, source) in files {
                fs::write(root.join(path), source).expect("source should be written");
            }
            Self { root }
        }

        fn environment(&self) -> PathBuf {
            self.root.join("world.dme")
        }
    }

    impl Drop for TestProject {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn inventories_supported_and_unsupported_bodies_without_aborting() {
        let project = TestProject::new(
            "#include \"supported.dm\"\n#include \"unsupported.dm\"\n",
            &[
                ("supported.dm", "/proc/ready()\n\treturn 42\n"),
                ("unsupported.dm", "/proc/later()\n\tspawn()\n"),
            ],
        );
        let compilation = CompilerDatabase::new()
            .compile(project.environment())
            .expect("mini-project should compile");
        let inventory = lower_compilation(&compilation);

        assert_eq!(inventory.stats().implementations, 2);
        assert_eq!(inventory.stats().lowered, 1);
        assert_eq!(inventory.stats().unsupported, 1);
        assert_eq!(
            inventory.stats().implementations,
            inventory.stats().lowered + inventory.stats().unsupported
        );
        let diagnostic = &inventory.diagnostics()[0];
        assert_eq!(diagnostic.procedure_path, "/proc/later");
        assert_eq!(diagnostic.source_path, "unsupported.dm");
        assert_eq!(diagnostic.category, UnsupportedCategory::CallResolution);
        assert!(!diagnostic.original_span.is_empty());
    }

    #[test]
    fn retains_an_executable_module_for_an_inherited_parent_call() {
        let project = TestProject::new(
            "#include \"types.dm\"\n",
            &[(
                "types.dm",
                "/datum/base\n\tproc/run()\n\t\treturn 2\n/datum/base/child\n\trun()\n\t\treturn ..() + 5\n",
            )],
        );
        let compilation = CompilerDatabase::new()
            .compile(project.environment())
            .expect("mini-project should compile");
        let inventory = lower_compilation(&compilation);
        let child = inventory
            .lowered()
            .iter()
            .find(|body| body.procedure_path == "/datum/base/child/proc/run")
            .expect("child override should lower");
        let module = child
            .module
            .as_ref()
            .expect("parent chain should be retained");
        let entry = module
            .procedure_id_at(
                child
                    .module_entry_index
                    .expect("entry index should be retained"),
            )
            .expect("entry should exist");

        assert_eq!(inventory.stats().parent_linked, 1);
        assert_eq!(execute_module(module, entry, &[]), Ok(Value::number(7.0)));
    }

    #[test]
    fn one_unsupported_parent_chain_does_not_poison_an_unrelated_body() {
        let project = TestProject::new(
            "#include \"types.dm\"\n",
            &[(
                "types.dm",
                "/datum/base\n\tproc/run()\n\t\treturn missing_call()\n/datum/base/child\n\trun()\n\t\treturn ..()\n/proc/healthy()\n\treturn 9\n",
            )],
        );
        let compilation = CompilerDatabase::new()
            .compile(project.environment())
            .expect("mini-project should compile");
        let inventory = lower_compilation(&compilation);

        assert!(
            inventory
                .lowered()
                .iter()
                .any(|body| body.procedure_path == "/proc/healthy")
        );
        assert_eq!(inventory.stats().lowered, 1);
        assert_eq!(inventory.stats().unsupported, 2);
        assert_eq!(
            inventory.blockers()[0].category,
            UnsupportedCategory::CallResolution
        );
    }
}
