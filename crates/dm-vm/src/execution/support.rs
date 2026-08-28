//! Observability and accounting metrics for the execution engine.
//!
//! Module layout: this crate splits `execution` into `state`, `frame`, `scheduler`, `run`,
//! and `support` so the deterministic engine stays navigable by concern.

/// Allocation-neutral scheduler metrics for continuation-layout benchmarks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ContinuationMetrics {
    /// Suspended continuations in scheduled and currently due queues.
    pub continuations: usize,
    /// Total call frames retained by those continuations.
    pub frames: usize,
    /// Frames carrying allocated cold exception/debug/argument state.
    pub cold_frames: usize,
    /// Rich values retained in locals, operand stacks, and argument vectors.
    pub retained_values: usize,
    /// Current native hot-frame header size, useful as a migration baseline.
    pub frame_header_bytes: usize,
    /// Header bytes avoided by moving rare state behind the cold allocation.
    pub rare_inline_bytes_avoided: usize,
}

/// Runtime counters for invalidation-safe static-field slot quickening.
///
/// The historical type name is retained for API compatibility. Counters now
/// include ordinary `datum.field` bytecode as well as statically declared
/// reads; dynamic `datum.vars[name]` accesses are not included.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeclaredFieldQuickeningMetrics {
    /// Reads served by a validated dense slot.
    pub hits: u64,
    /// Reads that resolved and installed or refreshed a slot.
    pub misses: u64,
    /// Cached slots rejected after a datum layout mutation.
    pub invalidations: u64,
}
