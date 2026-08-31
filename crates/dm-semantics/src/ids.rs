//! Dense tree-local identities for canonical procedures and their override
//! bodies, plus the private constructors the registry builder uses to mint them
//! from bounded `usize` indices.

/// Tree-local identity of a canonical procedure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureId(pub(crate) u32);

impl ProcedureId {
    /// Reconstructs an identity from a validated persistent procedure index.
    #[doc(hidden)]
    #[must_use]
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("procedure index exceeds u32"))
    }
    /// Returns this identity's index in [`ProcedureRegistry::procedures`].
    ///
    /// [`ProcedureRegistry::procedures`]: crate::ProcedureRegistry::procedures
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Tree-local identity of one body in a procedure's override chain.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProcedureImplementationId {
    pub(crate) procedure: ProcedureId,
    pub(crate) index: u32,
}

impl ProcedureImplementationId {
    /// Reconstructs an implementation identity from validated persistent indices.
    #[doc(hidden)]
    #[must_use]
    pub fn from_indices(procedure: usize, implementation: usize) -> Self {
        Self {
            procedure: ProcedureId::from_index(procedure),
            index: u32::try_from(implementation).expect("implementation index exceeds u32"),
        }
    }
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

pub(crate) fn procedure_id(index: usize) -> ProcedureId {
    ProcedureId(u32::try_from(index).expect("a registry cannot contain more than u32::MAX procs"))
}

pub(crate) fn implementation_id(
    procedure: ProcedureId,
    implementation_index: usize,
) -> ProcedureImplementationId {
    ProcedureImplementationId {
        procedure,
        index: u32::try_from(implementation_index)
            .expect("a procedure cannot contain more than u32::MAX implementations"),
    }
}
