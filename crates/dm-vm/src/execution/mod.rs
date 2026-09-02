//! Execution-state and call-frame machinery shared by the deterministic run
//! front doors.
//!
//! `ExecutionState` owns every mutable runtime value: the heap, globals, the
//! scheduler, local-client sessions, and type metadata. `CallFrame` is the
//! hot per-procedure state the interpreter advances, with rare argument,
//! exception, and diagnostic state pushed behind `CallFrameCold` so ordinary
//! frames stay small.

mod frame;
mod heap_gc;
mod host_services;
mod interpreter;
mod run;
mod run_support;
mod scheduler;
mod state;
mod support;
mod type_metadata;
mod world_geometry;

pub use frame::VmContinuationId;
pub use scheduler::advance_scheduler;
pub use state::ExecutionState;
pub use support::{ContinuationMetrics, DeclaredFieldQuickeningMetrics};

#[cfg(test)]
pub(crate) use frame::make_frame_owned;
pub(crate) use frame::{
    CallFrame, CallFrameCold, FrameRunOutcome, OwnedContinuation, PackedNumericState,
    RuinCandidateScan, StepBudgetBehavior, TgmLoadContinuation, TgmLoadPhase,
    declared_argument_count, frame_context, make_frame,
};
#[cfg(test)]
pub(crate) use heap_gc::adaptive_heap_collection_growth;
#[cfg(test)]
pub(crate) use heap_gc::{
    BULK_INIT_LOW_YIELD_STREAK, DEFAULT_HEAP_IDENTITY_CEILING, LARGE_HEAP_STREAK_IDENTITIES,
    MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH, MAXIMUM_LOW_YIELD_COLLECTION_GROWTH,
    MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH, MINIMUM_HEAP_COLLECTION_GROWTH,
    bulk_init_aware_collection_growth, collection_counts_toward_bulk_init_streak,
};
pub(crate) use run::run_frames;
pub(crate) use run_support::trace;
pub(crate) use scheduler::schedule_frames;
