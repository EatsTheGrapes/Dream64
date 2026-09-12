//! Background Cranelift compilation for the region tier
//! (`docs/performance/baseline-region-jit.md`).
//!
//! Every region compile (`compile_region_trace_at`) was, until this module
//! existed, run synchronously inline on whatever thread called
//! `ProcedureSidecar::poll_region_at` — the boot's own interpreter thread.
//! Milestone 7's own boot-parity measurement found this is a real cost: many
//! more independent call-resume candidates than PC-0 entries alone means many
//! more synchronous compile attempts (successful and rejected) on the boot
//! critical path, and Cranelift codegen is not free. `compile_region_trace_at`
//! only ever reads `Module`/`Program` — immutable, `Send + Sync` data with no
//! dependency on the executing frame or `ExecutionState` — so it is exactly
//! the kind of "immutable-input work" the design doc's own "Risks" section
//! already flagged as workable on a worker thread.
//!
//! **Opt-in, not a default.** Every existing region-JIT test asserts that a
//! region installs *synchronously*, the moment a warm-up counter crosses
//! threshold (e.g. "20 complete calls must be enough to cross the warm-up
//! threshold" — asserted immediately, no polling). Enabling this
//! unconditionally would make that timing asynchronous and flake those tests
//! against real wall-clock/thread-scheduling variance. Instead,
//! `ExecutionState::enable_async_region_compile` is called explicitly, only
//! by the real boot entry point (`dm_runtime::RuntimeImage::decode_linked_artifact`)
//! — every existing test's `ExecutionState::new()` leaves `async_region_compile`
//! `None` and keeps today's exact synchronous behavior, unchanged.
//!
//! One background thread, spawned lazily and shared process-wide (this is a
//! long-running server process; there is no shutdown story to build, the same
//! way `worker_lane`'s `std::thread::scope` batches need none). Each
//! `ExecutionState` that opts in gets its own reply channel, so results are
//! never misdelivered across independently-booted worlds (relevant for tests
//! that opt in, run in parallel with other tests).

use std::sync::mpsc;
use std::sync::{Arc, OnceLock};
use std::thread;

use crate::CompiledRegion;
use crate::bytecode::{Module, ProcedureId};

struct RegionCompileRequest {
    module: Arc<Module>,
    procedure: ProcedureId,
    module_identity: u64,
    entry_pc: usize,
    reply: mpsc::Sender<RegionCompileResult>,
}

pub(crate) struct RegionCompileResult {
    pub(crate) procedure: ProcedureId,
    pub(crate) module_identity: u64,
    pub(crate) entry_pc: usize,
    pub(crate) region: Option<CompiledRegion>,
}

/// The lazily-spawned, process-wide compile thread's request queue. A
/// `Sender` is cheap to clone (an `Arc`-backed handle) and safe to share
/// across every `AsyncRegionCompile` that opts in.
fn region_compile_worker_sender() -> mpsc::Sender<RegionCompileRequest> {
    static SENDER: OnceLock<mpsc::Sender<RegionCompileRequest>> = OnceLock::new();
    SENDER.get_or_init(spawn_region_compile_worker).clone()
}

fn spawn_region_compile_worker() -> mpsc::Sender<RegionCompileRequest> {
    let (tx, rx) = mpsc::channel::<RegionCompileRequest>();
    thread::Builder::new()
        .name("region-jit-compile".to_owned())
        .spawn(move || {
            while let Ok(request) = rx.recv() {
                // Mirrors `compile_region_trace_at`'s own leaf-call-inlining
                // path: an arbitrary `Call` target, resolved the identical
                // way the interpreter itself resolves it (including
                // deferred-procedure compilation, safe under concurrent
                // callers via `Arc<OnceLock<_>>` — see `resolve_procedure`).
                let region = request
                    .module
                    .resolve_procedure(request.procedure)
                    .ok()
                    .and_then(|program| {
                        crate::compile_region_trace_at(&request.module, program, request.entry_pc)
                    });
                let _ = request.reply.send(RegionCompileResult {
                    procedure: request.procedure,
                    module_identity: request.module_identity,
                    entry_pc: request.entry_pc,
                    region,
                });
            }
        })
        // Thread creation failing means the OS is out of resources — an
        // unrecoverable condition this boot is not going to survive anyway;
        // no graceful degradation path is worth building for it.
        .expect("spawning the region-JIT background compile thread");
    tx
}

/// One `ExecutionState`'s opt-in to background region compilation: an
/// `Arc<Module>` cloned once (a real but one-time cost — cheap `Arc` bumps
/// for `procedures`/`deferred`, a genuine deep clone for `paths`/`names`,
/// still negligible next to a multi-hundred-second boot) so every compile
/// request after this can hand the worker thread an owned reference without
/// re-cloning, plus this world's own private reply channel.
pub(crate) struct AsyncRegionCompile {
    module: Arc<Module>,
    module_identity: u64,
    reply_tx: mpsc::Sender<RegionCompileResult>,
    reply_rx: mpsc::Receiver<RegionCompileResult>,
}

impl AsyncRegionCompile {
    pub(crate) fn new(module: &Module) -> Self {
        let (reply_tx, reply_rx) = mpsc::channel();
        Self {
            module: Arc::new(module.clone()),
            module_identity: module.identity.0,
            reply_tx,
            reply_rx,
        }
    }

    pub(crate) fn module_identity(&self) -> u64 {
        self.module_identity
    }

    /// Submits `procedure`'s region-at-`entry_pc` compile to the background
    /// worker. Fire-and-forget: if the worker thread's own channel is
    /// somehow closed (never happens outside a prior `expect` panic), the
    /// request is silently dropped and that one site simply never gets a
    /// region — always safe, since a region is an optimization, never a
    /// correctness requirement.
    pub(crate) fn enqueue(&self, procedure: ProcedureId, entry_pc: usize) {
        let request = RegionCompileRequest {
            module: Arc::clone(&self.module),
            procedure,
            module_identity: self.module_identity,
            entry_pc,
            reply: self.reply_tx.clone(),
        };
        let _ = region_compile_worker_sender().send(request);
    }

    /// Every finished compile result available right now, without blocking
    /// — the caller drains this at a natural, already-paid-for point (a
    /// procedure switch) rather than paying a channel check on every
    /// instruction.
    pub(crate) fn drain_ready(&self) -> impl Iterator<Item = RegionCompileResult> + '_ {
        self.reply_rx.try_iter()
    }
}
