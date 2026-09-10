//! Per-`(procedure, PC)` execution sidecar: the mutable inline-cache storage
//! the interpreter keeps *beside* the immutable `Arc<Program>` bytecode.
//!
//! `docs/performance/boot-architecture-research.md` ranks a real adaptive
//! Tier-1 baseline as the only lever with the reach to close the five-minute
//! cold-boot gap, and its first requirement is "give each program a compact
//! mutable execution form and a PC-local cache sidecar — do not add a
//! process-global hash lookup to every opcode".
//!
//! One [`ProcedureSidecar`] holds a dense array with one [`PcCache`] per
//! bytecode instruction. A specialization family (field read, and later global
//! access, resolved calls, list ops) installs a guarded form into the slot for
//! its PC; every hit still revalidates against live heap/catalog state, so the
//! cache only ever affects hit rate, never semantics. Cold slots are one
//! pointer-sized niche and never allocate.

use std::collections::HashMap;

use crate::bytecode::ProcedureId;
use crate::execution::state::FieldSlotCache;

/// The specialization state for one bytecode instruction. Every non-`Cold`
/// variant boxes its payload so an all-`Cold` procedure array stays at one
/// machine word per instruction.
#[derive(Default)]
pub(crate) enum PcCache {
    /// No guarded form installed yet — the ordinary interpreter path runs and
    /// may install one after resolving.
    #[default]
    Cold,
    /// Polymorphic `(receiver type → resolution)` inline cache for a
    /// `LoadField` / `LoadDeclaredField` site. See [`FieldSlotCache`].
    FieldRead(Box<FieldSlotCache>),
}

/// A procedure's PC-indexed sidecar array. Allocated once, on the procedure's
/// first execution, and retained for the process lifetime; entries are never
/// removed, so the array's heap address is stable.
pub(crate) struct ProcedureSidecar {
    pcs: Box<[PcCache]>,
}

impl ProcedureSidecar {
    /// A fresh all-`Cold` sidecar for a procedure with `instruction_count`
    /// bytecode instructions.
    pub(crate) fn new(instruction_count: usize) -> Self {
        Self {
            pcs: (0..instruction_count).map(|_| PcCache::Cold).collect(),
        }
    }

    /// The cache slot for one instruction index, or `None` if the index is out
    /// of range (a fused/synthetic instruction with no source PC).
    pub(crate) fn slot_mut(&mut self, instruction_index: usize) -> Option<&mut PcCache> {
        self.pcs.get_mut(instruction_index)
    }

    /// Every installed field-read inline cache in this procedure.
    #[cfg(test)]
    pub(crate) fn field_read_caches(&self) -> impl Iterator<Item = &FieldSlotCache> {
        self.pcs.iter().filter_map(|slot| match slot {
            PcCache::FieldRead(cache) => Some(cache.as_ref()),
            PcCache::Cold => None,
        })
    }
}

/// All procedure sidecars for one runtime world, keyed by
/// `(module identity, procedure)`. A newtype so its accessors are a disjoint
/// field borrow of `ExecutionState`, leaving the heap borrowable alongside.
#[derive(Default)]
pub(crate) struct ProgramSidecars {
    by_procedure: HashMap<(u64, ProcedureId), ProcedureSidecar>,
}

impl ProgramSidecars {
    /// Drops every cache. Called when the type / shared-var / initial-value
    /// catalogs are replaced, or on ready-world snapshot restore.
    pub(crate) fn clear(&mut self) {
        self.by_procedure.clear();
    }

    /// The field-read inline cache installed at one `(module, procedure, PC)`
    /// call site, or `None` if the site has never resolved a datum field read.
    /// Non-creating — the hit and invalidation paths use this.
    pub(crate) fn field_read_cache(
        &mut self,
        module_identity: u64,
        procedure: ProcedureId,
        instruction_index: usize,
    ) -> Option<&mut FieldSlotCache> {
        match self
            .by_procedure
            .get_mut(&(module_identity, procedure))?
            .slot_mut(instruction_index)?
        {
            PcCache::FieldRead(cache) => Some(cache),
            PcCache::Cold => None,
        }
    }

    /// The field-read inline cache at a call site, allocating the procedure's
    /// sidecar array and promoting the PC's slot on first use. The miss path
    /// installs a resolution through this.
    pub(crate) fn field_read_cache_or_install(
        &mut self,
        module_identity: u64,
        procedure: ProcedureId,
        instruction_index: usize,
        instruction_count: usize,
    ) -> Option<&mut FieldSlotCache> {
        let slot = self
            .by_procedure
            .entry((module_identity, procedure))
            .or_insert_with(|| ProcedureSidecar::new(instruction_count))
            .slot_mut(instruction_index)?;
        if matches!(slot, PcCache::Cold) {
            *slot = PcCache::FieldRead(Box::default());
        }
        match slot {
            PcCache::FieldRead(cache) => Some(cache),
            PcCache::Cold => unreachable!("slot was just promoted to FieldRead"),
        }
    }

    /// The most receiver types any single field-read call site is tracking.
    #[cfg(test)]
    pub(crate) fn widest_field_read_site(&self) -> usize {
        self.by_procedure
            .values()
            .flat_map(ProcedureSidecar::field_read_caches)
            .map(FieldSlotCache::tracked_type_count)
            .max()
            .unwrap_or(0)
    }
}
