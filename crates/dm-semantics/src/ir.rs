//! The typed semantic IR: canonical [`Procedure`] records, their source-ordered
//! [`ProcedureImplementation`] chains, the deterministic build/closure counters,
//! and `effective_target`, the inheritance-aware body selector shared by the
//! registry and every dependency walk.

use dm_core::{FileId, SourceSpan};
use dm_object_tree::{CodePath, NodeId};

use super::{ProcedureId, ProcedureImplementationId};

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

/// Deterministic counters for project procedure indexing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcedureRegistryBuildStats {
    /// Static procedure references resolved through the canonical path index.
    pub static_proc_reference_index_lookups: usize,
}

/// Deterministic counters for one dependency-closure resolution.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcedureClosureStats {
    /// Unique procedure bodies visited.
    pub bodies_visited: usize,
    /// Static call selectors resolved through the owner/name index.
    pub static_selectors_resolved: usize,
    /// Dynamic literal selectors resolved through the name index.
    pub dynamic_selectors_resolved: usize,
    /// Exact implementations considered for dynamic literal selectors.
    pub dynamic_candidates_considered: usize,
}

pub(crate) fn effective_target(
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
