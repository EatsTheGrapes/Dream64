//! Heap collection, GC roots, and scheduler-progress metrics.
//!
//! Split out of `state.rs`: the adaptive list/datum reachability collector
//! (`maybe_collect_unreachable_lists`) and its growth policy, the host-
//! boundary heap compaction entry points, and the allocation-neutral
//! continuation / bounded-scheduler telemetry used by runtime benchmarks.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use crate::bytecode::Module;
use crate::{
    NativeWalk, PendingPromptContinuation, QuiescentHeapCompaction, boot_dashboard_enabled,
    boot_trace_enabled,
};
use dm_jit::NumericExecutionState;
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueHeap};

use crate::execution::frame::CallFrame;
use crate::execution::state::ExecutionState;
use crate::execution::support::ContinuationMetrics;

impl ExecutionState {
    pub(crate) fn rebuild_initial_value_roots(&mut self) {
        let mut datum_roots = Vec::new();
        let mut list_roots = Vec::new();
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.initial_values.values().flat_map(BTreeMap::values),
        );
        datum_roots.sort_unstable();
        datum_roots.dedup();
        list_roots.sort_unstable();
        list_roots.dedup();
        self.initial_value_datum_roots = Arc::from(datum_roots);
        self.initial_value_list_roots = Arc::from(list_roots);
    }

    /// Returns allocation-neutral suspended-frame metrics for runtime benchmarks.
    #[must_use]
    pub fn continuation_metrics(&self) -> ContinuationMetrics {
        let mut metrics = ContinuationMetrics {
            frame_header_bytes: std::mem::size_of::<CallFrame>(),
            rare_inline_bytes_avoided: std::mem::size_of::<Option<Value>>()
                + std::mem::size_of::<Option<Box<CallFrame>>>()
                + std::mem::size_of::<Option<NumericExecutionState>>(),
            ..ContinuationMetrics::default()
        };
        for continuation in self
            .scheduled_spawns
            .iter()
            .chain(self.scheduler_inflight.iter())
            .map(|spawn| &spawn.frames)
        {
            metrics.continuations += 1;
            metrics.frames += continuation.len();
            for frame in continuation {
                metrics.cold_frames += usize::from(frame.cold.is_some());
                metrics.retained_values += frame.locals.len()
                    + frame.stack.len()
                    + frame.arguments.len()
                    + frame.pending_argument_roots().len()
                    + frame.retained_call_roots().len()
                    + 3
                    + usize::from(frame.args_list.is_some())
                    + usize::from(frame.caller_result_override().is_some());
            }
        }
        metrics
    }

    /// Formats a bounded snapshot of suspended DM continuations for shutdown diagnostics.
    ///
    /// This performs no sampling or bookkeeping while bytecode executes. Hosts call it only
    /// after a scheduler bound terminates, so startup hot loops pay no telemetry overhead.
    #[must_use]
    pub fn bounded_scheduler_progress(&self, module: &Module) -> Vec<String> {
        let mut tasks = self.scheduled_spawns.iter().collect::<Vec<_>>();
        tasks.sort_unstable_by_key(|task| (task.due_tick, task.sequence));
        let mut lines = Vec::new();
        for (task_index, task) in tasks.into_iter().take(4).enumerate() {
            for (frame_index, frame) in task.frames.iter().rev().take(8).enumerate() {
                let path = module
                    .procedure_path(frame.procedure)
                    .unwrap_or("<missing>");
                let program = module.procedure(frame.procedure);
                let source = program
                    .and_then(|program| program.source_spans.get(frame.instruction))
                    .map_or_else(|| "<missing>".to_owned(), |span| format!("{span:?}"));
                let parameters = program.map_or_else(String::new, |program| {
                    program
                        .parameter_names
                        .iter()
                        .zip(frame.locals.iter())
                        .filter(|(name, _)| !name.is_empty())
                        .take(4)
                        .map(|(name, value)| {
                            format!("{name}={}", bounded_progress_value(value, &self.heap))
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                });
                let map_progress = map_loader_progress(frame, path, &self.heap);
                lines.push(format!(
                    "task={task_index} continuation={} due_tick={} sequence={} depth={} frame={} procedure={} instruction={} source={} stack={} locals={} parameters=[{}]{}",
                    task.frames.id.get(),
                    task.due_tick,
                    task.sequence,
                    task.frames.len(),
                    frame_index,
                    path,
                    frame.instruction,
                    source,
                    frame.stack.len(),
                    frame.locals.len(),
                    parameters,
                    map_progress,
                ));
            }
        }
        lines
    }

    /// Releases values returned across completed host execution calls.
    ///
    /// Callers that have consumed those results can use this before the next
    /// collection cycle so temporary datum/list identities are not retained
    /// for the lifetime of the server.
    pub fn release_host_value_roots(&mut self) -> usize {
        let released = self.host_value_roots.len();
        self.host_value_roots.clear();
        released
    }

    /// Forces one full heap collection at a host-owned quiescent boundary.
    ///
    /// Startup calls this only after VM execution has returned all active
    /// frames to the scheduler-owned queues, so the normal global, scheduler,
    /// client, world, and host roots are complete without an extra frame set.
    pub fn compact_quiescent_heap(&mut self) -> QuiescentHeapCompaction {
        let before_datums = self.heap.live_datum_count();
        let before_lists = self.heap.live_list_count();
        let started = Instant::now();
        self.next_list_collection = 1;
        self.maybe_collect_unreachable_lists(&[]);
        QuiescentHeapCompaction {
            reclaimed_datums: before_datums.saturating_sub(self.heap.live_datum_count()),
            reclaimed_lists: before_lists.saturating_sub(self.heap.live_list_count()),
            elapsed: started.elapsed(),
        }
    }

    pub(crate) fn maybe_collect_unreachable_lists(&mut self, active_frames: &[CallFrame]) {
        if self.next_list_collection == 0 {
            self.next_list_collection = MINIMUM_HEAP_COLLECTION_GROWTH;
        }
        let before_lists = self.heap.live_list_count();
        let before_datums = self.heap.live_datum_count();
        let before = before_lists.saturating_add(before_datums);
        if before < self.next_list_collection {
            return;
        }
        let collection_started = Instant::now();

        let mut datum_roots = Vec::new();
        let mut list_roots = Vec::new();
        extend_heap_root_ids(&mut datum_roots, &mut list_roots, self.globals.values());
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.initial_globals.values(),
        );
        datum_roots.extend_from_slice(&self.initial_value_datum_roots);
        list_roots.extend_from_slice(&self.initial_value_list_roots);
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.procedure_static_locals
                .values()
                .flat_map(|slots| slots.values()),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.savefiles
                .values()
                .flat_map(|savefile| savefile.entries.values()),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.environment_overrides.values().flatten(),
        );
        extend_heap_root_ids(
            &mut datum_roots,
            &mut list_roots,
            self.last_animation_target.iter(),
        );
        extend_heap_root_ids(&mut datum_roots, &mut list_roots, &self.host_value_roots);
        datum_roots.extend(self.initial_prototypes.values().copied());
        datum_roots.extend(self.savefiles.keys().copied());
        datum_roots.extend(
            self.savefile_entries
                .iter()
                .flat_map(|(entry, (savefile, _))| [*entry, *savefile]),
        );
        datum_roots.extend(self.deleting_datums.iter().copied());
        datum_roots.extend(self.client.session_datums());
        datum_roots.extend(self.client.mob_datums());
        datum_roots.extend(self.compact_default_datums.iter().copied());
        datum_roots.extend(self.world_turfs.values().copied());
        datum_roots.extend(self.world_areas.values().copied());
        datum_roots.extend(self.default_world_area);
        datum_roots.extend(self.native_walks.keys().copied());
        datum_roots.extend(self.native_walks.values().filter_map(NativeWalk::target));
        list_roots.extend(self.global_vars_proxy);
        for (datum, list) in self
            .datum_vars_by_datum
            .iter()
            .filter(|(datum, _)| self.heap.datum(**datum).is_ok())
        {
            datum_roots.push(*datum);
            list_roots.push(*list);
        }
        for prompt in self.client.pending_prompts() {
            datum_roots.push(prompt.1.client);
            extend_heap_root_ids(&mut datum_roots, &mut list_roots, &prompt.1.choices);
        }

        let mut add_frame_roots = |frame: &CallFrame| {
            // Engine-owned post-return continuations are suspended frames, not
            // disposable metadata. A collection may run while their child is
            // active, so root the complete boxed continuation chain exactly as
            // we do for ordinary scheduler frames.
            let mut frame = Some(frame);
            while let Some(current) = frame {
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.locals);
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.stack);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.result),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.src),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    std::iter::once(&current.usr),
                );
                extend_heap_root_ids(&mut datum_roots, &mut list_roots, &current.arguments);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.pending_argument_roots(),
                );
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.retained_call_roots(),
                );
                if let Some(tgm) = current.cold().and_then(|cold| cold.tgm_load.as_ref()) {
                    extend_heap_root_ids(&mut datum_roots, &mut list_roots, tgm.roots());
                }
                if let Some(scan) = current.cold().and_then(|cold| cold.ruin_scan.as_ref()) {
                    datum_roots.extend(scan.turfs.iter().copied());
                    extend_heap_root_ids(&mut datum_roots, &mut list_roots, &scan.areas);
                }
                list_roots.extend(current.args_list);
                extend_heap_root_ids(
                    &mut datum_roots,
                    &mut list_roots,
                    current.caller_result_override(),
                );
                frame = current.engine_post_return();
            }
        };
        for frame in active_frames {
            add_frame_roots(frame);
        }
        for spawn in &self.scheduled_spawns {
            for frame in &spawn.frames {
                add_frame_roots(frame);
            }
        }
        for spawn in &self.scheduler_inflight {
            for frame in &spawn.frames {
                add_frame_roots(frame);
            }
        }
        for (_, prompt) in self.client.pending_prompts() {
            match &prompt.continuation {
                PendingPromptContinuation::Frames(frames) => {
                    for frame in frames {
                        add_frame_roots(frame);
                    }
                }
                PendingPromptContinuation::Verb(invocation) => {
                    add_frame_roots(&invocation.frame);
                }
            }
        }

        let collection = self
            .heap
            .collect_unreachable_values_from_ids_with_stats(&datum_roots, &list_roots);
        let reclaimed_datums = collection.reclaimed_datums;
        let reclaimed_lists = collection.reclaimed_lists;
        let heap = &self.heap;
        self.associative_lists
            .retain(|list| heap.list(*list).is_ok());
        self.reference_lists.retain(|list| heap.list(*list).is_ok());
        self.datum_vars_proxies
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.datum_vars_by_datum
            .retain(|datum, list| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.contents_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.vis_contents_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.vis_locs_owners
            .retain(|list, datum| heap.list(*list).is_ok() && heap.datum(*datum).is_ok());
        self.compact_default_datums
            .retain(|datum| heap.datum(*datum).is_ok());
        self.native_walks.retain(|movable, walk| {
            heap.datum(*movable).is_ok()
                && walk
                    .target()
                    .is_none_or(|target| heap.datum(target).is_ok())
        });
        if self
            .global_vars_proxy
            .is_some_and(|list| heap.list(list).is_err())
        {
            self.global_vars_proxy = None;
        }

        let after_lists = self.heap.live_list_count();
        let after_datums = self.heap.live_datum_count();
        let after = after_lists.saturating_add(after_datums);
        let reclaimed = reclaimed_datums.saturating_add(reclaimed_lists);
        // A run of collections that each free almost nothing means the heap is
        // in a monotonic bulk-allocation phase (map load, `SSatoms` init).
        // Track the streak so the growth window can widen past its low-yield
        // cap and stop punctuating that phase with dozens of full-heap walks.
        let visited = after.saturating_add(reclaimed);
        if reclaimed.saturating_mul(NEAR_ZERO_YIELD_RECIPROCAL) <= visited {
            self.low_yield_collection_streak = self.low_yield_collection_streak.saturating_add(1);
        } else {
            self.low_yield_collection_streak = 0;
        }
        let growth = bulk_init_aware_collection_growth(
            after,
            reclaimed,
            self.low_yield_collection_streak,
            self.heap_identity_ceiling,
        );
        self.next_list_collection = after.saturating_add(growth);
        if boot_trace_enabled() || boot_dashboard_enabled() {
            let elapsed_ms = collection_started.elapsed().as_millis();
            let lists = collection.list_storage;
            let datum_storage = collection.datum_storage;
            let datums = collection.datum_arena;
            let list_arena = collection.list_arena;
            eprintln!(
                "boot-vm: heap-gc datums_before={before_datums} datums_after={after_datums} datums_reclaimed={reclaimed_datums} lists_before={before_lists} lists_after={after_lists} lists_reclaimed={reclaimed_lists} growth={growth} next={} elapsed_ms={elapsed_ms} list_storage_allocated={} list_payload_len={} list_payload_cap={} list_order_len={} list_order_cap={} list_prefix_retained={} list_prefix_compacted={} list_prefix_entries_compacted={} list_vectors_shrunk={} list_capacity_bytes_reclaimed={} list_shared_shrink_candidates={} list_assoc_indexes={} list_assoc_index_len={} list_assoc_index_cap={} list_assoc_index_ratio_bins={:?} list_assoc_indexes_shrunk={} list_assoc_index_cap_reclaimed={} list_assoc_index_bytes_reclaimed={} list_remove_indexes={} list_remove_key_len={} list_remove_key_cap={} list_remove_position_len={} list_remove_position_cap={} list_remove_removed_len={} list_remove_removed_cap={} list_remove_indexes_dropped={} list_shared_derived_candidates={} datum_field_len={} datum_field_cap={} datum_shared_name_datums={} datum_shared_name_logical_slots={} datum_shared_name_layouts={} datum_shared_name_physical_slots={} datum_shared_name_bytes_saved={} datum_field_vectors_shrunk={} datum_capacity_bytes_reclaimed={} datum_field_indexes={} datum_field_index_len={} datum_field_index_cap={} datum_field_index_ratio_bins={:?} datum_field_indexes_shrunk={} datum_field_index_cap_reclaimed={} datum_field_index_bytes_reclaimed={} datum_field_indexes_deduplicated={} datum_field_index_dedupe_bytes_reclaimed={} datum_field_physical_indexes={} datum_field_physical_index_len={} datum_field_physical_index_cap={} datum_field_index_fingerprint_collisions={} datum_field_index_fingerprints_computed={} datum_field_index_pointer_cache_hits={} datum_field_index_exact_layout_comparisons={} datum_arena_live={} datum_arena_slots={} datum_arena_free={} datum_arena_chunks={} datum_arena_reserved={} list_arena_live={} list_arena_slots={} list_arena_free={} list_arena_chunks={} list_arena_reserved={}",
                self.next_list_collection,
                lists.allocated_lists,
                lists.payload_len,
                lists.payload_capacity,
                lists.order_len,
                lists.order_capacity,
                lists.prefix_retained,
                lists.compacted_lists,
                lists.compacted_prefix_entries,
                lists.shrunk_vectors,
                lists.reclaimed_capacity_bytes,
                lists.shared_shrink_candidates,
                lists.associative_indexes,
                lists.associative_index_len,
                lists.associative_index_capacity,
                lists.associative_index_ratio_bins,
                lists.shrunk_associative_indexes,
                lists.reclaimed_associative_index_capacity,
                lists.reclaimed_associative_index_bytes,
                lists.positional_remove_indexes,
                lists.positional_remove_key_len,
                lists.positional_remove_key_capacity,
                lists.positional_remove_position_len,
                lists.positional_remove_position_capacity,
                lists.positional_remove_removed_len,
                lists.positional_remove_removed_capacity,
                lists.dropped_positional_remove_indexes,
                lists.shared_derived_index_candidates,
                datum_storage.field_len,
                datum_storage.field_capacity,
                datum_storage.shared_field_name_datums,
                datum_storage.shared_field_name_logical_slots,
                datum_storage.shared_field_name_layouts,
                datum_storage.shared_field_name_physical_slots,
                datum_storage.shared_field_name_bytes_saved,
                datum_storage.shrunk_field_vectors,
                datum_storage.reclaimed_capacity_bytes,
                datum_storage.field_indexes,
                datum_storage.field_index_len,
                datum_storage.field_index_capacity,
                datum_storage.field_index_ratio_bins,
                datum_storage.shrunk_field_indexes,
                datum_storage.reclaimed_field_index_capacity,
                datum_storage.reclaimed_field_index_bytes,
                datum_storage.deduplicated_field_indexes,
                datum_storage.deduplicated_field_index_bytes,
                datum_storage.physical_field_indexes,
                datum_storage.physical_field_index_len,
                datum_storage.physical_field_index_capacity,
                datum_storage.field_index_fingerprint_collisions,
                datum_storage.field_index_fingerprints_computed,
                datum_storage.field_index_pointer_cache_hits,
                datum_storage.field_index_exact_layout_comparisons,
                datums.live,
                datums.slots,
                datums.free,
                datums.chunks,
                datums.reserved,
                list_arena.live,
                list_arena.slots,
                list_arena.free,
                list_arena.chunks,
                list_arena.reserved,
            );
            if std::env::var_os("DREAM64_PROFILE_MEMORY").is_some() {
                let mut counts = HashMap::<TypePath, usize>::new();
                for (_, datum) in self.heap.datums() {
                    *counts.entry(datum.type_path().clone()).or_default() += 1;
                }
                let mut counts = counts.into_iter().collect::<Vec<_>>();
                counts.sort_unstable_by(|(left_path, left_count), (right_path, right_count)| {
                    right_count
                        .cmp(left_count)
                        .then_with(|| left_path.as_str().cmp(right_path.as_str()))
                });
                let summary = counts
                    .into_iter()
                    .take(20)
                    .map(|(path, count)| format!("{}={count}", path.as_str()))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("boot-vm: heap-type-census {summary}");
            }
        }
    }
}

fn bounded_progress_value(value: &Value, heap: &ValueHeap) -> String {
    match value {
        Value::Text(text) => {
            let truncated = text.chars().count() > 48;
            let mut text = text.chars().take(48).collect::<String>();
            if truncated {
                text.push('…');
            }
            format!("{text:?}")
        }
        Value::List(list) => heap.list(*list).map_or_else(
            |_| "list(stale)".to_owned(),
            |values| format!("list(len={})", values.len()),
        ),
        Value::Datum(datum) => heap.datum(*datum).map_or_else(
            |_| "datum(stale)".to_owned(),
            |value| format!("datum({})", value.type_path()),
        ),
        _ => value.to_string(),
    }
}

fn map_loader_progress(frame: &CallFrame, path: &str, heap: &ValueHeap) -> String {
    if !path.contains("/datum/parsed_map/proc/") {
        return String::new();
    }
    let Value::Datum(src) = frame.src else {
        return String::new();
    };
    let fields = ["map_format", "grid_models", "modelCache", "bounds"]
        .into_iter()
        .filter_map(|name| {
            let field = FieldName::parse(name).ok()?;
            let value = heap.datum_field(src, &field).ok()?;
            Some(format!("{name}={}", bounded_progress_value(value, heap)))
        })
        .collect::<Vec<_>>();
    if !fields.is_empty() {
        format!(" map=[{}]", fields.join(","))
    } else {
        Default::default()
    }
}

pub(crate) const MINIMUM_HEAP_COLLECTION_GROWTH: usize = 65_536;
// Keep the low-yield window bounded tightly enough for production hosts with
// small Windows commit limits. Boot194 exhausted commit after this window
// expanded to 836,706 live identities, before the next reachability pass.
pub(crate) const MAXIMUM_LOW_YIELD_COLLECTION_GROWTH: usize = 262_144;
pub(crate) const MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH: usize = 262_144;
pub(crate) const MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH: usize = 262_144;

/// A collection whose reclaim count is at most `1 / NEAR_ZERO_YIELD_RECIPROCAL`
/// of the identities it visited counts as near-zero-yield for bulk-init phase
/// detection. Deliberately tighter than the 5% "low yield" growth bucket below:
/// this is meant to fire only when a pass frees essentially nothing.
pub(crate) const NEAR_ZERO_YIELD_RECIPROCAL: usize = 100;

/// Consecutive near-zero-yield collections that identify a monotonic
/// bulk-allocation phase (map load, `SSatoms.InitializeAtoms`). Until the streak
/// reaches this length the base growth policy is unchanged.
pub(crate) const BULK_INIT_LOW_YIELD_STREAK: u32 = 3;

/// Default ceiling on live heap identities that forces a collection regardless
/// of recent reclaim yield. Bounds committed memory during a long bulk-init
/// phase where the yield heuristic would otherwise keep widening the window.
/// Overridable with `DREAM64_HEAP_IDENTITY_CEILING`.
pub(crate) const DEFAULT_HEAP_IDENTITY_CEILING: usize = 16_777_216;

/// Resolves the live-identity ceiling from the environment once per state.
pub(crate) fn resolve_heap_identity_ceiling() -> usize {
    std::env::var("DREAM64_HEAP_IDENTITY_CEILING")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value >= MINIMUM_HEAP_COLLECTION_GROWTH)
        .unwrap_or(DEFAULT_HEAP_IDENTITY_CEILING)
}

/// Chooses the next collection window, widening it past the low-yield cap once a
/// run of near-zero-yield passes shows the heap is in a monotonic
/// bulk-allocation phase.
///
/// During `SSatoms.InitializeAtoms` almost every identity stays reachable (each
/// of ~1M space turfs lazily materialises structurally-identical overlay
/// lists), so [`adaptive_heap_collection_growth`] pins the window to
/// [`MAXIMUM_LOW_YIELD_COLLECTION_GROWTH`] and every ~260k allocations pays
/// another multi-second full-heap walk. Once `low_yield_streak` reaches
/// [`BULK_INIT_LOW_YIELD_STREAK`] the window instead runs toward `ceiling` (by
/// at most the current live size, so growth stays a bounded per-pass doubling),
/// collapsing dozens of passes into a handful. `ceiling` still forces a pass, so
/// committed memory stays bounded; above it the base policy resumes.
///
/// This changes only *when* collections run, never what they observe: a
/// collection is already required to be observationally transparent, and the
/// window is a deterministic function of heap counts plus collection history,
/// so identical executions still collect at identical points.
pub(crate) fn bulk_init_aware_collection_growth(
    live: usize,
    reclaimed: usize,
    low_yield_streak: u32,
    ceiling: usize,
) -> usize {
    let base = adaptive_heap_collection_growth(live, reclaimed);
    if low_yield_streak < BULK_INIT_LOW_YIELD_STREAK || live >= ceiling {
        return base;
    }
    let headroom = ceiling - live;
    base.max(headroom.min(live))
}

/// Chooses how many additional heap identities may be allocated before the
/// next full reachability pass.
///
/// A large, mostly-live production heap is expensive to rescan and receives a
/// 25% growth window after a pass that reclaimed at most 5% of the identities
/// it visited. More productive passes tighten the window because they indicate
/// that transient allocation pressure is building. Absolute caps bound both
/// transient garbage and the live-object growth between collections.
pub(crate) fn adaptive_heap_collection_growth(live: usize, reclaimed: usize) -> usize {
    let visited = live.saturating_add(reclaimed);
    let (growth_divisor, maximum_growth) = if reclaimed <= visited.saturating_div(20) {
        (4, MAXIMUM_LOW_YIELD_COLLECTION_GROWTH)
    } else if reclaimed <= visited.saturating_div(5) {
        (10, MAXIMUM_MODERATE_YIELD_COLLECTION_GROWTH)
    } else {
        (40, MAXIMUM_HIGH_YIELD_COLLECTION_GROWTH)
    };
    live.saturating_div(growth_divisor)
        .max(MINIMUM_HEAP_COLLECTION_GROWTH)
        .min(maximum_growth)
}

fn extend_heap_root_ids<'a>(
    datum_roots: &mut Vec<DatumId>,
    list_roots: &mut Vec<ListId>,
    values: impl IntoIterator<Item = &'a Value>,
) {
    for value in values {
        extend_heap_root_value(datum_roots, list_roots, value);
    }
}

fn extend_heap_root_value(
    datum_roots: &mut Vec<DatumId>,
    list_roots: &mut Vec<ListId>,
    value: &Value,
) {
    match value {
        Value::Datum(datum) => datum_roots.push(*datum),
        Value::List(list) => list_roots.push(*list),
        Value::ModifiedTypePath(path) => {
            for (_, value) in path.overrides() {
                extend_heap_root_value(datum_roots, list_roots, value);
            }
        }
        Value::Null | Value::Number(_) | Value::Text(_) | Value::File(_) | Value::TypePath(_) => {}
    }
}
