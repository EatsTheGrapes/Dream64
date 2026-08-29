//! Embedded native accelerations for the reference interpreter.
//!
//! The interpreter is contract-bound to a deterministic bytecode algorithm,
//! but for a narrow set of *canonical* procedure shapes it shortcuts straight
//! into native Rust when the compiled program provably matches the shape.
//! World-clock numerics, `type2parent`/`istext` compaction, the camera chunk,
//! `RegisterSignal`, rooted-list, and lumcount loops, and — for Monke's `.dmm`
//! pipeline — the TGM/ruin/dmm-load drives all live here so that trust
//! criteria, instrumentation counters, and canonical matchers stay co-located
//! with the code that uses them.
//!
//! The cluster is organized into three sibling modules: `numeric_core` (world
//! datum, clock, and numeric dispatch), `tgm_ruin` (Monke TGM/ruin loading and
//! the `type2parent`/`istext` canonicalizers with their counters), and
//! `fastpath_jit` (trace-compiled fast paths). The hub re-exports the public
//! surface the crate root and the sibling modules consume.

mod fastpath_jit;
mod numeric_core;
mod tgm_ruin;

// Numeric-core dispatch counters are public crate APIs for the profiler.
pub use self::numeric_core::packed_dispatch_counters;

// TGM/ruin instrumentation counters are public crate APIs for the profiler.
pub use self::tgm_ruin::{
    native_build_coordinate_prefix_metrics, native_discover_offset_activations,
    native_ruin_area_rejection_samples, native_ruin_batch_metrics,
    native_ruin_rejection_cache_hits, native_ruin_rejection_causes, native_ruin_scan_metrics,
    native_tgm_build_cache_metrics, native_tgm_commit_samples, native_tgm_continuation_rejections,
    native_tgm_load_activations, native_tgm_load_metrics, native_tgm_route_samples,
    native_tgm_target_cache_metrics,
};

pub(crate) use self::fastpath_jit::{
    execute_compact_fast_instruction, try_run_camera_chunk_fast_path,
    try_run_discover_offset_fast_path, try_run_dmm_preload_measurement_fast_path, try_run_guarded_jit,
    try_run_parsed_dmm_new_fast_path, try_run_register_signal_fast_path, try_run_rooted_list_jit,
};

pub(crate) use self::numeric_core::{
    advance_headless_world_clock, false_tick_check_target, numeric_dispatch_candidate,
    set_world_numeric_field, try_run_numeric_dispatch_block, try_run_numeric_local_update,
    try_run_numeric_loop_branch, world_numeric_field,
};

pub(crate) use self::tgm_ruin::{
    TgmDrive, canonical_istext, canonical_static_native_builtin, canonical_tgm_load_path,
    canonical_type2parent, canonical_type2parent_target, drive_ruin_candidate_scan, drive_tgm_load,
    trace_tgm_route, try_run_build_coordinate_prefix, try_run_ruin_affected_turfs_batch,
    try_run_tgm_build_cache_simple_member,
};

// Items reached only from the integration tests keep their exports test-gated
// so non-test builds never observe an unused re-export.
#[cfg(test)]
pub(crate) use self::{
    fastpath_jit::{
        REGISTER_SIGNAL_FAST_CACHE, compile_lumcount_trace, compile_register_signal_trace,
        compile_rooted_list_trace, discover_offset_native, jit_disabled, numeric_jit_prefix_candidate,
        numeric_trace_instructions,
    },
    numeric_core::{try_run_packed_numeric_dispatch_block, try_run_rich_numeric_dispatch_block},
    tgm_ruin::{
        CANONICAL_MONKE_BUILD_COORDINATE_DIGEST, CANONICAL_TYPE2PARENT_SOURCE,
        build_tgm_load_continuation, canonical_type2parent_program, revalidated_ruin_rejection,
        ruin_scan_attach_at_call, run_ruin_affected_turfs_batch, tgm_attach_location,
    },
};