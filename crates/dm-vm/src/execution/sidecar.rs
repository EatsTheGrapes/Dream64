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
//!
//! The run loop resolves a procedure's sidecar **once per procedure switch**
//! (alongside the immutable [`Program`](crate::bytecode::Program)) and threads
//! `&mut ProcedureSidecar` through `dispatch_instruction`, so a per-PC cache
//! access is a plain array index — never a map lookup on the hot path.

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
    /// A procedure entry (PC 0) has been observed this many times, warming up
    /// toward a region-compile attempt. See
    /// `docs/performance/baseline-region-jit.md`.
    RegionCounting(u16),
    /// A compiled region installed at this call site's entry PC.
    Region(Box<dm_jit::CompiledRegion>),
    /// A region-compile attempt at this site failed or the shape was
    /// unsupported; never retried.
    RegionRejected,
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

    /// The field-read inline cache at one PC, or `None` if nothing has been
    /// installed there yet (or the index is out of range). The hit and
    /// invalidation paths use this.
    pub(crate) fn field_read_cache(
        &mut self,
        instruction_index: usize,
    ) -> Option<&mut FieldSlotCache> {
        match self.pcs.get_mut(instruction_index)? {
            PcCache::FieldRead(cache) => Some(cache),
            // `Cold`: nothing installed yet. The `Region*` variants only ever
            // occupy PC 0 (see `poll_region_at_entry`) and claim that slot for
            // region-compile bookkeeping instead — a procedure whose very
            // first instruction is a `LoadField` simply never gets a field
            // cache at PC 0, same as any other declined quickening attempt.
            PcCache::Cold
            | PcCache::RegionCounting(_)
            | PcCache::Region(_)
            | PcCache::RegionRejected => None,
        }
    }

    /// The field-read inline cache at one PC, promoting the slot on first use.
    /// The miss path installs a resolution through this.
    pub(crate) fn field_read_cache_or_install(
        &mut self,
        instruction_index: usize,
    ) -> Option<&mut FieldSlotCache> {
        let slot = self.pcs.get_mut(instruction_index)?;
        if matches!(slot, PcCache::Cold) {
            *slot = PcCache::FieldRead(Box::default());
        }
        match slot {
            PcCache::FieldRead(cache) => Some(cache),
            PcCache::Cold => unreachable!("slot was just promoted to FieldRead"),
            // See `field_read_cache` above: PC 0 already claimed for a region
            // attempt, decline rather than clobber it.
            PcCache::RegionCounting(_) | PcCache::Region(_) | PcCache::RegionRejected => None,
        }
    }

    /// Every installed field-read inline cache in this procedure.
    #[cfg(test)]
    pub(crate) fn field_read_caches(&self) -> impl Iterator<Item = &FieldSlotCache> {
        self.pcs.iter().filter_map(|slot| match slot {
            PcCache::FieldRead(cache) => Some(cache.as_ref()),
            PcCache::Cold
            | PcCache::RegionCounting(_)
            | PcCache::Region(_)
            | PcCache::RegionRejected => None,
        })
    }

    /// Procedure entries observed at PC 0 before Milestone 1 attempts to
    /// compile a region there. Small and arbitrary: the compiled body does
    /// nothing yet (see `docs/performance/baseline-region-jit.md`), so this
    /// threshold only proves the warm-up/compile/install machinery behaves —
    /// it is not evidence for the right threshold once regions do real work.
    const REGION_ENTRY_THRESHOLD: u16 = 16;

    /// Called once per procedure entry (`instruction_index == 0`). Advances
    /// the PC-0 slot toward a region-compile attempt; a no-op once the slot
    /// holds anything else (a region, a rejection, or — rare, but possible for
    /// a one-instruction procedure — an installed field-read cache).
    pub(crate) fn poll_region_at_entry(&mut self) {
        let Some(slot) = self.pcs.first_mut() else {
            return;
        };
        match slot {
            PcCache::Cold => *slot = PcCache::RegionCounting(1),
            PcCache::RegionCounting(count) => {
                if *count + 1 >= Self::REGION_ENTRY_THRESHOLD {
                    *slot = match dm_jit::compile_trivial_region() {
                        Ok(region) => PcCache::Region(Box::new(region)),
                        Err(_) => PcCache::RegionRejected,
                    };
                } else {
                    *count += 1;
                }
            }
            PcCache::FieldRead(_) | PcCache::Region(_) | PcCache::RegionRejected => {}
        }
    }

    /// Runs the region installed at PC 0, if any. `entry_pc` is always `0` in
    /// Milestone 1 (a region only ever installs at a procedure's own entry);
    /// later milestones that compile from loop headers pass the header PC.
    pub(crate) fn run_region_at_entry(
        &self,
        entry_pc: u32,
        budget: u32,
    ) -> Option<dm_jit::RegionOutcome> {
        match self.pcs.first()? {
            PcCache::Region(region) => Some(region.run(entry_pc, budget)),
            _ => None,
        }
    }

    /// Whether a region has been compiled and installed at PC 0.
    #[cfg(test)]
    pub(crate) fn has_region_at_entry(&self) -> bool {
        matches!(self.pcs.first(), Some(PcCache::Region(_)))
    }
}

/// All procedure sidecars for one runtime world, keyed by
/// `(module identity, procedure)`. `run_frames` lends the whole map out as a
/// value disjoint from `ExecutionState` for one run, then holds `&mut` to the
/// active procedure's entry across that procedure's instructions, re-resolving
/// only on a procedure switch.
#[derive(Default)]
pub(crate) struct ProgramSidecars {
    by_procedure: HashMap<(u64, ProcedureId), ProcedureSidecar>,
}

impl ProgramSidecars {
    /// Drops every cache. Called *between* `run_frames` invocations only —
    /// when the type / shared-var / initial-value catalogs are replaced, or on
    /// ready-world snapshot restore. Never reachable from `dispatch_instruction`.
    pub(crate) fn clear(&mut self) {
        self.by_procedure.clear();
    }

    /// The sidecar for one `(module, procedure)`, allocating an all-`Cold` array
    /// of `instruction_count` slots on first use. The run loop calls this only
    /// when the executing procedure changes and holds the borrow across that
    /// procedure's instructions.
    pub(crate) fn resolve(
        &mut self,
        module_identity: u64,
        procedure: ProcedureId,
        instruction_count: usize,
    ) -> &mut ProcedureSidecar {
        self.by_procedure
            .entry((module_identity, procedure))
            .or_insert_with(|| ProcedureSidecar::new(instruction_count))
    }

    /// The most receiver types any single field-read call site is tracking.
    #[cfg(test)]
    pub(crate) fn widest_field_read_site(&self) -> usize {
        self.by_procedure
            .values()
            .flat_map(|sidecar| sidecar.field_read_caches())
            .map(FieldSlotCache::tracked_type_count)
            .max()
            .unwrap_or(0)
    }

    /// Whether a region has been compiled and installed at `procedure`'s entry.
    #[cfg(test)]
    pub(crate) fn region_installed(&self, module_identity: u64, procedure: ProcedureId) -> bool {
        self.by_procedure
            .get(&(module_identity, procedure))
            .is_some_and(ProcedureSidecar::has_region_at_entry)
    }
}
