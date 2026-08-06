//! Procedure-level semantic indexing for Dream Maker projects.
//!
//! The registry converts canonical procedure nodes into source-ordered
//! implementation chains. Each implementation records the exact predecessor a
//! future `..()` lowering should invoke: the previous implementation on the
//! same type, or the inherited procedure's effective implementation.
//!
//! Inheritance follows the object tree's resolved hierarchy, including the final
//! source-ordered constant `parent_type = /some/type` assignment on each type.

#![cfg_attr(not(test), deny(missing_docs))]

use std::collections::BTreeMap;

use dm_compiler::Compilation;
use dm_core::{FileId, SourceSpan};
use dm_object_tree::{CodePath, NodeId, NodeKind};
use dm_syntax::DefinitionKind;

/// Tree-local identity of a canonical procedure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureId(u32);

impl ProcedureId {
    /// Returns this identity's index in [`ProcedureRegistry::procedures`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tree-local identity of one body in a procedure's override chain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureImplementationId {
    procedure: ProcedureId,
    index: u32,
}

impl ProcedureImplementationId {
    /// Returns the canonical procedure containing this implementation.
    #[must_use]
    pub const fn procedure(self) -> ProcedureId {
        self.procedure
    }

    /// Returns this implementation's source-order index within its procedure.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index as usize
    }
}

/// Syntactic role used to introduce a procedure implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcedureImplementationKind {
    /// A body declared through an explicit `proc` path node.
    Declaration,
    /// A body written directly beneath an owning type.
    Override,
}

/// One source body in a canonical procedure's override chain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcedureImplementation {
    /// Tree-local identity of this implementation.
    pub id: ProcedureImplementationId,
    /// Physical source file containing the declaration.
    pub file_id: FileId,
    /// Index into the corresponding syntax file's definition table.
    pub definition_index: usize,
    /// Position in the preprocessor-expanded declaration stream.
    pub ordinal: usize,
    /// Complete source span of the procedure header.
    pub span: SourceSpan,
    /// Whether source used a declaration or override spelling.
    pub kind: ProcedureImplementationKind,
    /// Exact implementation a future `..()` expression should invoke.
    pub parent_target: Option<ProcedureImplementationId>,
}

/// One canonical procedure and all bodies assigned to it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Procedure {
    /// Tree-local canonical procedure identity.
    pub id: ProcedureId,
    /// Corresponding canonical object-tree node.
    pub node: NodeId,
    /// Canonical absolute procedure path.
    pub path: CodePath,
    /// Owning type, or `None` for a global procedure under `/proc`.
    pub owner_type: Option<NodeId>,
    /// Nearest same-name procedure in the default parent hierarchy.
    pub inherited_procedure: Option<ProcedureId>,
    /// Implementations in preprocessor-expanded source order.
    pub implementations: Vec<ProcedureImplementation>,
    /// Body selected when this canonical procedure is dispatched.
    pub effective_target: Option<ProcedureImplementationId>,
}

/// Project-wide registry of canonical procedures and override chains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcedureRegistry {
    procedures: Vec<Procedure>,
    by_node: BTreeMap<NodeId, ProcedureId>,
    by_path: BTreeMap<CodePath, ProcedureId>,
}

impl ProcedureRegistry {
    /// Builds a registry from the compiler's accepted canonical declarations.
    #[must_use]
    pub fn build(compilation: &Compilation) -> Self {
        let tree = compilation.code_tree();
        let procedure_nodes: Vec<_> = tree
            .nodes()
            .iter()
            .filter(|node| node.kind == NodeKind::Procedure)
            .collect();
        let by_node: BTreeMap<_, _> = procedure_nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, procedure_id(index)))
            .collect();
        let by_path = procedure_nodes
            .iter()
            .map(|node| (node.path.clone(), by_node[&node.id]))
            .collect();

        let mut procedures: Vec<_> = procedure_nodes
            .into_iter()
            .map(|node| {
                let id = by_node[&node.id];
                let implementations = node
                    .declarations
                    .iter()
                    .filter_map(|declaration_id| tree.declaration(*declaration_id))
                    .filter_map(|declaration| {
                        let kind = match declaration.kind {
                            DefinitionKind::Procedure => ProcedureImplementationKind::Declaration,
                            DefinitionKind::ProcedureOverride => {
                                ProcedureImplementationKind::Override
                            }
                            DefinitionKind::Type
                            | DefinitionKind::Verb
                            | DefinitionKind::Variable
                            | DefinitionKind::VariableOverride => return None,
                        };
                        Some((declaration, kind))
                    })
                    .enumerate()
                    .map(|(index, (declaration, kind))| ProcedureImplementation {
                        id: implementation_id(id, index),
                        file_id: declaration.file_id,
                        definition_index: declaration.definition_index,
                        ordinal: declaration.ordinal,
                        span: declaration.span,
                        kind,
                        parent_target: None,
                    })
                    .collect::<Vec<_>>();
                Procedure {
                    id,
                    node: node.id,
                    path: node.path.clone(),
                    owner_type: node.owner_type,
                    inherited_procedure: node
                        .inherited_member
                        .and_then(|parent| by_node.get(&parent).copied()),
                    effective_target: implementations.last().map(|body| body.id),
                    implementations,
                }
            })
            .collect();

        for procedure_index in 0..procedures.len() {
            let inherited_target = procedures[procedure_index]
                .inherited_procedure
                .and_then(|parent| effective_target(&procedures, parent));
            for implementation_index in 0..procedures[procedure_index].implementations.len() {
                let parent_target = if implementation_index == 0 {
                    inherited_target
                } else {
                    Some(implementation_id(
                        procedures[procedure_index].id,
                        implementation_index - 1,
                    ))
                };
                procedures[procedure_index].implementations[implementation_index].parent_target =
                    parent_target;
            }
        }

        Self {
            procedures,
            by_node,
            by_path,
        }
    }

    /// Returns canonical procedures in object-tree node order.
    #[must_use]
    pub fn procedures(&self) -> &[Procedure] {
        &self.procedures
    }

    /// Looks up a canonical procedure by registry identity.
    #[must_use]
    pub fn procedure(&self, id: ProcedureId) -> Option<&Procedure> {
        self.procedures.get(id.index())
    }

    /// Looks up a canonical procedure by object-tree node identity.
    #[must_use]
    pub fn find_node(&self, node: NodeId) -> Option<ProcedureId> {
        self.by_node.get(&node).copied()
    }

    /// Looks up a canonical procedure by absolute code path.
    #[must_use]
    pub fn find_path(&self, path: &CodePath) -> Option<ProcedureId> {
        self.by_path.get(path).copied()
    }

    /// Looks up one procedure implementation by its composite identity.
    #[must_use]
    pub fn implementation(
        &self,
        id: ProcedureImplementationId,
    ) -> Option<&ProcedureImplementation> {
        self.procedure(id.procedure())?
            .implementations
            .get(id.index())
    }
}

fn effective_target(
    procedures: &[Procedure],
    procedure: ProcedureId,
) -> Option<ProcedureImplementationId> {
    let procedure = procedures.get(procedure.index())?;
    procedure.effective_target.or_else(|| {
        procedure
            .inherited_procedure
            .and_then(|parent| effective_target(procedures, parent))
    })
}

fn procedure_id(index: usize) -> ProcedureId {
    ProcedureId(u32::try_from(index).expect("a registry cannot contain more than u32::MAX procs"))
}

fn implementation_id(
    procedure: ProcedureId,
    implementation_index: usize,
) -> ProcedureImplementationId {
    ProcedureImplementationId {
        procedure,
        index: u32::try_from(implementation_index)
            .expect("a procedure cannot contain more than u32::MAX implementations"),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use dm_compiler::{Compilation, CompilerDatabase};

    use super::{Procedure, ProcedureImplementationKind, ProcedureRegistry};

    static NEXT_PROJECT: AtomicU64 = AtomicU64::new(0);

    struct TestProject {
        root: PathBuf,
    }

    impl TestProject {
        fn compile(source: &str) -> Compilation {
            let ordinal = NEXT_PROJECT.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "dream64-dm-semantics-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&root).expect("test project directory should be created");
            let project = Self { root };
            fs::write(project.root.join("world.dme"), "#include \"types.dm\"\n")
                .expect("environment should be written");
            fs::write(project.root.join("types.dm"), source).expect("source should be written");
            CompilerDatabase::new()
                .compile(project.root.join("world.dme"))
                .expect("test project should compile")
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

    fn procedure_by_path<'registry>(
        registry: &'registry ProcedureRegistry,
        path: &str,
    ) -> &'registry Procedure {
        registry
            .procedures()
            .iter()
            .find(|procedure| procedure.path.to_string() == path)
            .expect("procedure path should exist")
    }

    #[test]
    fn indexes_a_base_procedure_with_source_identity() {
        let compilation = TestProject::compile("/datum/base\n\tproc/run()\n\t\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert!(procedure.owner_type.is_some());
        assert_eq!(procedure.implementations.len(), 1);
        assert_eq!(
            procedure.implementations[0].kind,
            ProcedureImplementationKind::Declaration
        );
        assert_eq!(procedure.implementations[0].definition_index, 1);
        assert!(!procedure.implementations[0].span.is_empty());
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[0].id)
        );
    }

    #[test]
    fn links_a_child_override_to_the_inherited_effective_body() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base/child\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let base = procedure_by_path(&registry, "/datum/base/proc/run");
        let child = procedure_by_path(&registry, "/datum/base/child/proc/run");

        assert_eq!(child.inherited_procedure, Some(base.id));
        assert_eq!(child.implementations.len(), 1);
        assert_eq!(
            child.implementations[0].parent_target,
            base.effective_target
        );
    }

    #[test]
    fn chains_multiple_reopenings_in_expanded_source_order() {
        let compilation = TestProject::compile(
            "/datum/base\n\tproc/run()\n\t\treturn 1\n/datum/base\n\trun()\n\t\treturn 2\n/datum/base\n\trun()\n\t\treturn 3\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/datum/base/proc/run");

        assert_eq!(procedure.implementations.len(), 3);
        assert!(
            procedure
                .implementations
                .windows(2)
                .all(|pair| pair[0].ordinal < pair[1].ordinal)
        );
        assert_eq!(procedure.implementations[0].parent_target, None);
        assert_eq!(
            procedure.implementations[1].parent_target,
            Some(procedure.implementations[0].id)
        );
        assert_eq!(
            procedure.implementations[2].parent_target,
            Some(procedure.implementations[1].id)
        );
        assert_eq!(
            procedure.effective_target,
            Some(procedure.implementations[2].id)
        );
    }

    #[test]
    fn indexes_a_global_procedure_without_an_owner_or_parent() {
        let compilation = TestProject::compile("/proc/global_run()\n\treturn 1\n");
        let registry = ProcedureRegistry::build(&compilation);
        let procedure = procedure_by_path(&registry, "/proc/global_run");

        assert_eq!(procedure.owner_type, None);
        assert_eq!(procedure.inherited_procedure, None);
        assert_eq!(procedure.implementations[0].parent_target, None);
    }

    #[test]
    fn follows_a_resolved_explicit_parent_type() {
        let compilation = TestProject::compile(
            "/datum/original\n\tproc/run()\n\t\treturn 1\n/datum/alternate\n\tproc/run()\n\t\treturn 2\n/custom\n\tparent_type = /datum/alternate\n\trun()\n\t\treturn ..()\n",
        );
        let registry = ProcedureRegistry::build(&compilation);
        let original = procedure_by_path(&registry, "/datum/original/proc/run");
        let alternate = procedure_by_path(&registry, "/datum/alternate/proc/run");
        let custom = procedure_by_path(&registry, "/custom/proc/run");

        assert_ne!(custom.inherited_procedure, Some(original.id));
        assert_eq!(custom.inherited_procedure, Some(alternate.id));
        assert_eq!(
            custom.implementations[0].parent_target,
            alternate.effective_target
        );
    }
}
