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

use crate::CompiledRegion;
use crate::bytecode::{Module, ProcedureId, Program};
use crate::execution::region_compile_worker::AsyncRegionCompile;
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
    /// A compile request for this site has been submitted to the background
    /// worker (`region_compile_worker`) and is awaiting a result — never
    /// entered when background compilation isn't enabled for this world
    /// (`ExecutionState::enable_async_region_compile`), in which case a
    /// warm site compiles straight to `Region`/`RegionRejected` instead, as
    /// every milestone before this one did.
    RegionCompiling,
    /// A compiled region installed at this PC — a procedure's own entry
    /// (PC 0), or, since Milestone 7, a call-resume candidate elsewhere.
    /// Since Milestone 3, `CompiledRegion` bundles the whole-procedure
    /// binary32 trace with the dense `FieldName` table its
    /// `LoadFieldDynamic` instructions index into (`dm-jit` never sees a
    /// `FieldName`, only indices) — see `compile_region_trace_at`.
    Region(Box<CompiledRegion>),
    /// A region-compile attempt at this site failed or the shape was
    /// unsupported; never retried.
    RegionRejected,
}

/// The memoized packed-dispatch profitability verdict for one PC.
///
/// The prediction scan walks up to 24 instructions forward from a PC, reading
/// only the immutable bytecode and following static `Jump` targets. It consults
/// no frame, heap, or scheduler state, so its answer for a given PC is fixed for
/// the life of the program and one evaluation is enough.
///
/// Boot telemetry in `docs/performance/boot-architecture-research.md` measured
/// ~207M block entries re-running that scan to decline 220M times against 26k
/// uses, and calls memoizing it "a cache-the-verdict cleanup worth ~10 s on its
/// own".
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum PackedRunVerdict {
    /// Not scanned at this PC yet.
    #[default]
    Unknown,
    /// The scan reached its horizon; packed dispatch is worth entering.
    Profitable,
    /// The scan bailed out; the rich path handles this PC.
    Unprofitable,
}

/// A procedure's PC-indexed sidecar array. Allocated once, on the procedure's
/// first execution, and retained for the process lifetime; entries are never
/// removed, so the array's heap address is stable.
pub(crate) struct ProcedureSidecar {
    pcs: Box<[PcCache]>,
    /// Packed-dispatch verdicts, parallel to `pcs`.
    ///
    /// Kept out of [`PcCache`] deliberately: that array holds one variant per
    /// PC, and field-read and region bookkeeping already contend for the slot.
    /// A verdict must not be able to evict a field cache, or vice versa.
    ///
    /// Allocated on the first packed-dispatch question for this procedure.
    /// Only PCs that pass `numeric_dispatch_candidate` ever ask, so procedures
    /// that never enter a numeric block never pay the byte-per-instruction.
    packed_run: Option<Box<[PackedRunVerdict]>>,
}

impl ProcedureSidecar {
    /// A fresh all-`Cold` sidecar for a procedure with `instruction_count`
    /// bytecode instructions.
    pub(crate) fn new(instruction_count: usize) -> Self {
        Self {
            pcs: (0..instruction_count).map(|_| PcCache::Cold).collect(),
            packed_run: None,
        }
    }

    /// The packed-dispatch profitability verdict for one PC, running `compute`
    /// and memoizing it the first time a PC is asked about.
    ///
    /// `compute` must be the pure bytecode scan described on
    /// [`PackedRunVerdict`]; anything consulting runtime state would be wrong to
    /// cache here. A PC outside this procedure's range falls through to
    /// `compute` uncached rather than growing the array.
    pub(crate) fn packed_run_profitable(
        &mut self,
        instruction_index: usize,
        compute: impl FnOnce() -> bool,
    ) -> bool {
        let instruction_count = self.pcs.len();
        if instruction_index >= instruction_count {
            return compute();
        }
        let verdicts = self.packed_run.get_or_insert_with(|| {
            vec![PackedRunVerdict::Unknown; instruction_count].into_boxed_slice()
        });
        match verdicts[instruction_index] {
            PackedRunVerdict::Profitable => true,
            PackedRunVerdict::Unprofitable => false,
            PackedRunVerdict::Unknown => {
                let profitable = compute();
                verdicts[instruction_index] = if profitable {
                    PackedRunVerdict::Profitable
                } else {
                    PackedRunVerdict::Unprofitable
                };
                profitable
            }
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
            // `Cold`: nothing installed yet. The `Region*` variants occupy
            // PC 0 and — since milestone 7 — any call-resume candidate PC
            // (see `poll_region_at`) and claim that slot for region-compile
            // bookkeeping instead — a procedure whose very first instruction
            // is a `LoadField` simply never gets a field cache at PC 0, same
            // as any other declined quickening attempt.
            PcCache::Cold
            | PcCache::RegionCounting(_)
            | PcCache::RegionCompiling
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
            PcCache::RegionCounting(_)
            | PcCache::RegionCompiling
            | PcCache::Region(_)
            | PcCache::RegionRejected => None,
        }
    }

    /// Every installed field-read inline cache in this procedure.
    #[cfg(test)]
    pub(crate) fn field_read_caches(&self) -> impl Iterator<Item = &FieldSlotCache> {
        self.pcs.iter().filter_map(|slot| match slot {
            PcCache::FieldRead(cache) => Some(cache.as_ref()),
            PcCache::Cold
            | PcCache::RegionCounting(_)
            | PcCache::RegionCompiling
            | PcCache::Region(_)
            | PcCache::RegionRejected => None,
        })
    }

    /// Procedure entries observed at PC 0 before a region-compile attempt.
    /// Small and arbitrary — chosen in Milestone 1 to prove the warm-up/
    /// compile/install machinery behaves, not tuned as the right threshold;
    /// still unrevisited in Milestone 2.
    const REGION_ENTRY_THRESHOLD: u16 = 16;

    /// Called once per procedure entry (`instruction_index == 0`), or once
    /// per milestone-7 call-resume candidate PC that a side-exit's own
    /// analysis has proven safe and registered via
    /// `register_region_candidate` — `run.rs`'s `region_site` check decides
    /// which. Advances that slot toward a region-compile attempt; a no-op
    /// once the slot holds anything else (a region, a rejection, a pending
    /// background compile, or — rare, but possible for a one-instruction
    /// procedure — an installed field-read cache). Each candidate PC's slot
    /// is an independent cell in the same dense array PC 0 already used, so
    /// this needs no separate bookkeeping structure for the extra
    /// candidates.
    ///
    /// `async_compile: None` compiles synchronously and installs the result
    /// immediately, exactly as every milestone before background compilation
    /// existed did (and as every existing region-JIT test still assumes —
    /// see `region_compile_worker`'s module doc). `Some` submits the compile
    /// to the background worker and marks the slot `RegionCompiling`
    /// instead; the result arrives later via `install_region_result`, driven
    /// by `ProgramSidecars::drain_async_region_results`.
    pub(crate) fn poll_region_at(
        &mut self,
        module: &Module,
        program: &Program,
        procedure: ProcedureId,
        pc: usize,
        async_compile: Option<&AsyncRegionCompile>,
    ) {
        let Some(slot) = self.pcs.get_mut(pc) else {
            return;
        };
        match slot {
            PcCache::Cold => *slot = PcCache::RegionCounting(1),
            PcCache::RegionCounting(count) => {
                if *count + 1 >= Self::REGION_ENTRY_THRESHOLD {
                    match async_compile {
                        Some(async_compile) => {
                            async_compile.enqueue(procedure, pc);
                            *slot = PcCache::RegionCompiling;
                        }
                        None => {
                            *slot = match crate::compile_region_trace_at(module, program, pc) {
                                Some(region) => PcCache::Region(Box::new(region)),
                                None => PcCache::RegionRejected,
                            };
                        }
                    }
                } else {
                    *count += 1;
                }
            }
            PcCache::FieldRead(_)
            | PcCache::RegionCompiling
            | PcCache::Region(_)
            | PcCache::RegionRejected => {}
        }
    }

    /// Installs a background compile's result (`region_compile_worker`) once
    /// it arrives — a no-op if the slot isn't still `RegionCompiling` (it
    /// always will be in practice: exactly one request is ever sent per PC,
    /// the moment it first transitions out of `RegionCounting`), so a
    /// duplicate or late-for-a-cleared-world result can never clobber
    /// something else.
    pub(crate) fn install_region_result(&mut self, pc: usize, region: Option<CompiledRegion>) {
        let Some(slot @ PcCache::RegionCompiling) = self.pcs.get_mut(pc) else {
            return;
        };
        *slot = match region {
            Some(region) => PcCache::Region(Box::new(region)),
            None => PcCache::RegionRejected,
        };
    }

    /// The region compiled at `pc` (0 for a procedure's own entry, or a
    /// milestone-7 call-resume candidate elsewhere), if any. Callers drive
    /// it exactly as the pre-region whole-procedure numeric JIT drove its
    /// own cached trace: build/resume a `NumericExecutionState` from the
    /// caller's frame and call `run_budgeted`.
    pub(crate) fn region_at(&self, pc: usize) -> Option<&CompiledRegion> {
        match self.pcs.get(pc)? {
            PcCache::Region(region) => Some(region),
            _ => None,
        }
    }

    /// Milestone 7: whether `pc` has ever been registered as a call-resume
    /// candidate (or is itself already a `Region`/`RegionRejected`) — i.e.,
    /// whether the run loop should even bother polling/attempting a region
    /// here at all. `Cold` (the overwhelming majority of ordinary
    /// instructions, which are never candidates) short-circuits this to a
    /// single array read, so per-instruction dispatch pays no cost for
    /// procedures where no side-exit has ever registered anything.
    pub(crate) fn is_region_site(&self, pc: usize) -> bool {
        !matches!(
            self.pcs.get(pc),
            None | Some(PcCache::Cold | PcCache::FieldRead(_))
        )
    }

    /// Milestone 7: registers `pc` as a call-resume candidate the first time
    /// a side-exit's own analysis proves it safe (`safe_call_resume_pc`) —
    /// a no-op if the slot already holds anything else (already registered,
    /// already compiled, already rejected, or claimed by a field-read cache
    /// first). Does not itself attempt a region-compile; `poll_region_at`
    /// does that once this slot has been observed enough times.
    pub(crate) fn register_region_candidate(&mut self, pc: usize) {
        if let Some(slot @ PcCache::Cold) = self.pcs.get_mut(pc) {
            *slot = PcCache::RegionCounting(0);
        }
    }

    /// Whether a region has been compiled and installed at PC 0.
    #[cfg(test)]
    pub(crate) fn has_region_at_entry(&self) -> bool {
        matches!(self.pcs.first(), Some(PcCache::Region(_)))
    }

    /// Whether a region has been compiled and installed at an arbitrary
    /// `pc` — milestone 7's call-resume regions, unlike the entry region
    /// `has_region_at_entry` checks, can live anywhere in the array.
    #[cfg(test)]
    pub(crate) fn has_region_at(&self, pc: usize) -> bool {
        matches!(self.pcs.get(pc), Some(PcCache::Region(_)))
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

    /// Installs every background region compile that has finished since the
    /// last drain. Called at a procedure switch (`run.rs`) rather than every
    /// instruction — dispatch never blocks on this, and a compile that
    /// finishes between drains just sits in the channel a little longer, at
    /// no cost to anything (it isn't consulted until `region_at`/`is_region_site`
    /// next look at that PC, which only happens once this world reaches it
    /// again). A result for a procedure this world no longer tracks (its
    /// sidecar was cleared, e.g. by a ready-world snapshot restore, while the
    /// compile was in flight) is simply dropped — the slot it would have
    /// installed into no longer exists.
    pub(crate) fn drain_async_region_results(&mut self, async_compile: &AsyncRegionCompile) {
        for result in async_compile.drain_ready() {
            if result.module_identity != async_compile.module_identity() {
                continue;
            }
            if let Some(sidecar) = self
                .by_procedure
                .get_mut(&(result.module_identity, result.procedure))
            {
                sidecar.install_region_result(result.entry_pc, result.region);
            }
        }
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

    /// Whether a region has been compiled and installed at an arbitrary `pc`
    /// within `procedure` — milestone 7's own call-resume candidates, which
    /// unlike the entry region `region_installed` checks are not necessarily
    /// at PC 0.
    #[cfg(test)]
    pub(crate) fn region_installed_at(
        &self,
        module_identity: u64,
        procedure: ProcedureId,
        pc: usize,
    ) -> bool {
        self.by_procedure
            .get(&(module_identity, procedure))
            .is_some_and(|sidecar| sidecar.has_region_at(pc))
    }
}
