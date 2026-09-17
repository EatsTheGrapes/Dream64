//! The persistent headless scheduler loop and its entry paths: public-endpoint
//! discovery, per-launch random seeding, lobby-generation activation, and the
//! startup scheduler drain limits.

use std::collections::hash_map::RandomState;
use std::env;
use std::hash::{BuildHasher, Hasher};
use std::net::IpAddr;
use std::process::{Command as ProcessCommand, ExitCode};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dm_lifecycle::ipc::{BootPhase, LoopbackIpc};
use dm_lifecycle::{
    HeadlessReadinessProbe, HostSliceBudget, SchedulerDrainLimits,
    advance_persistent_scheduler_responsive, readiness_probe_matches,
};
use dm_runtime::RuntimeImage;

pub(crate) fn report_public_endpoint(port: u16) {
    let _ = std::thread::Builder::new()
        .name("dream64-public-ip".to_owned())
        .spawn(move || {
            let output = ProcessCommand::new("curl")
                .args([
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    "5",
                    "https://api.ipify.org",
                ])
                .output();
            let Ok(output) = output else {
                eprintln!("server-network: public IP discovery unavailable");
                return;
            };
            let address = String::from_utf8_lossy(&output.stdout);
            match address.trim().parse::<IpAddr>() {
                Ok(address) => eprintln!(
                    "server-network: public-endpoint={address}:{port} tcp-port-forward-required=true"
                ),
                Err(_) => eprintln!("server-network: public IP discovery unavailable"),
            }
        });
}

pub(crate) fn launch_random_seed() -> (u64, &'static str) {
    launch_random_seed_from(env::var("DREAM64_RANDOM_SEED").ok().as_deref())
}

pub(crate) fn launch_random_seed_from(value: Option<&str>) -> (u64, &'static str) {
    if let Some(value) = value
        && let Ok(seed) = value.parse::<u64>()
        && seed != 0
    {
        return (seed, "environment");
    }
    (fresh_launch_random_seed(), "host-entropy")
}

pub(crate) fn fresh_launch_random_seed() -> u64 {
    // `RandomState::new()` obtains independently keyed entropy from the host.
    // Mix in launch-local values as domain separation and avoid the all-zero
    // state used by deterministic unit tests.
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    hasher.write_u32(std::process::id());
    let seed = hasher.finish();
    if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    }
}

pub(crate) fn run_persistent_server_loop(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    startup_ipc: Option<LoopbackIpc>,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> ExitCode {
    let mut ipc_address = startup_ipc.expect("production boot bound loopback IPC");
    if let Err(error) =
        activate_lobby_generation(runtime, precompiled, &mut ipc_address, lobby_readiness)
    {
        eprintln!("generation activation: {error}");
        return ExitCode::FAILURE;
    }
    ipc_address
        .commit_boot_phase(
            ipc_address.startup_generation(),
            BootPhase::RuntimeStartedReadOnly,
            "Lobby ready — round countdown active",
        )
        .expect("activated lobby belongs to current boot generation");
    let startup_generation = ipc_address.startup_generation();
    if let Err(error) = ipc_address.accept_startup_clients(startup_generation) {
        eprintln!("generation activation commit: {error}");
        return ExitCode::FAILURE;
    }
    if let Some(state) = precompiled.persistent_state_mut() {
        ipc_address.enable_session_interaction(state);
    }
    precompiled.mark_profile_steady_state();
    eprintln!(
        "server-progress: loopback-ipc={} startup=accepting lobby=pregame",
        ipc_address.local_addr()
    );
    let max_slices = env::var_os("DREAM64_BOOT_MAX_SLICES")
        .and_then(|limit| {
            limit
                .to_str()
                .and_then(|value| value.parse::<u64>().ok())
                .map(Some)
                .or_else(|| {
                    eprintln!("DREAM64_BOOT_MAX_SLICES ignored: not a valid u64: {limit:?}");
                    None
                })
        })
        .flatten();
    let mut slices = 0u64;
    let mut host_budget = HostSliceBudget::new(100_000, 1_000, 100_000, Duration::from_millis(10));
    let mut max_vm_slice = Duration::ZERO;
    let mut over_target_slices = 0u64;
    loop {
        let slice_started = Instant::now();
        let tick_duration = precompiled.persistent_tick_duration();
        ipc_address.apply_lifecycle_tick_boundary(precompiled);
        let vm_started = Instant::now();
        let scheduled_steps = host_budget.steps();
        let scheduler = match advance_persistent_scheduler_responsive(
            precompiled,
            runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 1,
            },
            scheduled_steps,
            None,
        ) {
            Ok(scheduler) => scheduler,
            Err(error) => {
                eprintln!("persistent scheduler: {error}");
                return ExitCode::FAILURE;
            }
        };
        let vm_elapsed = vm_started.elapsed();
        max_vm_slice = max_vm_slice.max(vm_elapsed);
        over_target_slices =
            over_target_slices.saturating_add(u64::from(vm_elapsed > Duration::from_millis(10)));
        host_budget.observe(vm_elapsed);
        ipc_address.apply_lifecycle_tick_boundary(precompiled);
        slices = slices.saturating_add(1);
        if slices == 1 || slices % 100 == 0 || vm_elapsed >= Duration::from_millis(50) {
            eprintln!(
                "server-progress: scheduler slice={} tick={} rounds={} completed={} failed={} pending={} termination={:?} vm_us={} vm_max_us={} next_step_budget={} over_10ms={} host_loop_us={}",
                slices,
                scheduler.final_tick,
                scheduler.rounds,
                scheduler.completed_tasks,
                scheduler.failed_tasks,
                scheduler.pending_tasks,
                scheduler.termination,
                vm_elapsed.as_micros(),
                max_vm_slice.as_micros(),
                host_budget.steps(),
                over_target_slices,
                slice_started.elapsed().as_micros(),
            );
        }
        if let Some(limit) = max_slices
            && slices >= limit
        {
            eprintln!("boot-progress: reached DREAM64_BOOT_MAX_SLICES={limit}; stopping");
            for line in precompiled.bounded_scheduler_progress() {
                eprintln!("boot-progress: shutdown-dm-frame {line}");
            }
            report_boot_profiles(precompiled, "boot-max-slices");
            return ExitCode::SUCCESS;
        }
        if let Some(remaining) = tick_duration.checked_sub(slice_started.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
}

/// Diagnostic: emit the always-on field-slot quickening ratio, plus — when
/// enabled — the whole-boot instruction-category histogram
/// (`DREAM64_PROFILE_INSTRUCTIONS`) and the per-procedure self-time top
/// (`DREAM64_PROFILE_PROC_STEPS`).
fn report_boot_profiles(precompiled: &dm_lifecycle::PrecompiledLifecycle, at: &str) {
    let fq = precompiled.field_quickening_totals();
    let fq_total = fq.hits + fq.misses;
    #[allow(clippy::cast_precision_loss)]
    let fq_hit_pct = if fq_total == 0 {
        0.0
    } else {
        (fq.hits as f64 / fq_total as f64) * 100.0
    };
    let (eiv_hits, eiv_cold) = precompiled.effective_initial_value_totals();
    eprintln!(
        "boot-profile at={at} field_quickening hits={} misses={} invalidations={} hit_pct={fq_hit_pct:.1} present_hits={} absent_hits={} eiv_hits={eiv_hits} eiv_cold={eiv_cold}",
        fq.hits, fq.misses, fq.invalidations, fq.present_hits, fq.absent_hits,
    );
    // JIT / numeric fast-path telemetry — gated by the same profiling flags,
    // so this is free on a production boot. Answers whether the Cranelift
    // whole-procedure JIT contributes at all and whether the numeric
    // basic-block interpreter work is concentrated enough to be worth native
    // compilation.
    let (
        compiled_numeric,
        rejected_numeric,
        compiled_lumcount,
        rejected_lumcount,
        jit_runs,
        jit_steps,
    ) = dm_vm::guarded_jit_telemetry();
    let (packed_entries, packed_declines) = dm_vm::packed_dispatch_counters();
    let (block_entries, block_steps, block_len) = dm_vm::numeric_block_telemetry();
    eprintln!(
        "boot-profile at={at} jit_guarded numeric_compiled={compiled_numeric} numeric_rejected={rejected_numeric} lumcount_compiled={compiled_lumcount} lumcount_rejected={rejected_lumcount} runs={jit_runs} steps={jit_steps}"
    );
    let (alloc_count, alloc_total_ns, alloc_roots_ns, alloc_init_ns, alloc_init_count) =
        dm_vm::datum_alloc_telemetry();
    if alloc_count > 0 {
        eprintln!(
            "boot-profile at={at} datum_alloc allocations={alloc_count} total_ms={} per_alloc_ns={} roots_ms={} roots_pct={} initializer_ms={} initializer_pct={} initializer_programs={alloc_init_count} programs_per_alloc={}",
            alloc_total_ns / 1_000_000,
            alloc_total_ns.checked_div(alloc_count).unwrap_or(0),
            alloc_roots_ns / 1_000_000,
            alloc_roots_ns
                .saturating_mul(100)
                .checked_div(alloc_total_ns)
                .unwrap_or(0),
            alloc_init_ns / 1_000_000,
            alloc_init_ns
                .saturating_mul(100)
                .checked_div(alloc_total_ns)
                .unwrap_or(0),
            alloc_init_count
                .saturating_mul(100)
                .checked_div(alloc_count)
                .unwrap_or(0),
        );
    }
    eprintln!(
        "boot-profile at={at} numeric_blocks entries={block_entries} steps={block_steps} avg={} len_1_4={} len_5_16={} len_17_64={} len_65_256={} len_257up={} packed_entries={packed_entries} packed_declines={packed_declines}",
        block_steps.checked_div(block_entries).unwrap_or(0),
        block_len[0],
        block_len[1],
        block_len[2],
        block_len[3],
        block_len[4],
    );
    for line in dm_vm::numeric_block_site_report(30) {
        eprintln!("boot-profile at={at} {line}");
    }
    // Whole-heap scans by unqualified `locate()` / `TypeInstances` — reveals an
    // O(n) `locate(/type)` pattern that grows with the heap during atom init
    // (the move_turf_to_area class of bug).
    let (loc_ty, loc_ty_datums, loc_tag, loc_tag_datums, ti_scans, ti_datums) =
        dm_vm::locate_scan_telemetry();
    eprintln!(
        "boot-profile at={at} locate_scans type_scans={loc_ty} type_scan_datums={loc_ty_datums} tag_scans={loc_tag} tag_scan_datums={loc_tag_datums} type_instances_scans={ti_scans} type_instances_scan_datums={ti_datums}"
    );
    for line in precompiled.instruction_profile_lines(false) {
        eprintln!("boot-profile at={at} {line}");
    }
    for line in precompiled.instruction_profile_lines(true) {
        eprintln!("boot-profile at={at} {line}");
    }
    for (rank, line) in precompiled
        .proc_step_profile_lines(30)
        .into_iter()
        .enumerate()
    {
        eprintln!("boot-profile at={at} proc_rank={} {line}", rank + 1);
    }
    for (rank, line) in precompiled.pc_profile_lines(40).into_iter().enumerate() {
        eprintln!("boot-profile at={at} procedure_pc_rank={} {line}", rank + 1);
    }
}

/// Diagnostic: dump the Master Controller / `SSticker` state that governs the
/// activation-phase pregame gate. Gated by `DREAM64_TRACE_TICKER`.
///
/// Fields are read raw from datum storage with no initial-value fallback, so a
/// field that Monke only ever leaves at its compile-time initial value reads as
/// absent. That is itself the signal for `SSticker.current_state`: absent means
/// still `GAME_STATE_STARTUP`; `fire()` writing `GAME_STATE_PREGAME` is what the
/// readiness probe waits for and what shows up here as `current_state=1`.
fn trace_ticker_state(precompiled: &mut dm_lifecycle::PrecompiledLifecycle, slice: u64) {
    if env::var_os("DREAM64_TRACE_TICKER").is_none() {
        return;
    }
    let Some(state) = precompiled.persistent_state_mut() else {
        return;
    };
    let dump = |state: &dm_vm::ExecutionState, label: &str, datum, names: &[&str]| {
        let dm_value::Value::Datum(datum) = datum else {
            eprintln!("ticker-trace slice={slice} {label}=<not a datum: {datum:?}>");
            return;
        };
        let mut parts = Vec::new();
        for name in names {
            let Ok(field) = dm_value::FieldName::parse(name) else {
                continue;
            };
            match state.heap().datum_field(datum, &field) {
                Ok(dm_value::Value::Datum(d)) => parts.push(format!("{name}=datum({d:?})")),
                Ok(dm_value::Value::List(l)) => {
                    let len = state.heap().list(*l).map_or(0, dm_value::DmList::len);
                    parts.push(format!("{name}=list(len={len})"));
                }
                Ok(value) => parts.push(format!("{name}={value:?}")),
                Err(_) => parts.push(format!("{name}=<unwritten>")),
            }
        }
        eprintln!("ticker-trace slice={slice} {label}: {}", parts.join(" "));
    };
    let global = |state: &dm_vm::ExecutionState, name: &str| {
        dm_value::FieldName::parse(name)
            .ok()
            .and_then(|field| state.global(&field).cloned())
            .unwrap_or(dm_value::Value::Null)
    };

    let ssticker = global(state, "SSticker");
    dump(
        state,
        "SSticker",
        ssticker,
        &["current_state", "times_fired", "initialized"],
    );
    let master = global(state, "Master");
    dump(
        state,
        "Master",
        master,
        &[
            "current_runlevel",
            "processing",
            "last_type_processed",
            "iteration",
        ],
    );
}

/// Slice ceiling for the lobby-generation activation loop.
///
/// The loop advances Monke's post-`Initialize` Master Controller tail one
/// scheduler tick per slice until `SSticker` enters pregame. A stock
/// `tgstation.d64` reaches pregame in ~3.7k ticks, so the default keeps a >2x
/// margin: a healthy boot always exits early on the readiness check, while a
/// genuinely stuck controller tail is still bounded rather than spinning
/// forever. Override with `DREAM64_ACTIVATION_MAX_SLICES`.
const DEFAULT_ACTIVATION_MAX_SLICES: u64 = 8_000;
const _: () = assert!(
    DEFAULT_ACTIVATION_MAX_SLICES >= 7_000,
    "activation slice ceiling must stay well above the ~3.7k ticks a stock boot needs to reach pregame",
);

/// Per-slice VM wall-time target for the activation loop.
///
/// The steady-state persistent loop stays latency-sensitive for connected
/// sessions, but the activation phase serves only a read-only "finishing
/// initialization" lobby, so it runs for throughput. One Master Controller tick
/// in this phase costs ~200-250ms of VM time on the reference machine; a target
/// below that makes every slice overshoot, so the adaptive step controller
/// collapses to its floor, no tick ever completes, and the boot livelocks short
/// of pregame. Override with `DREAM64_SCHEDULER_WALL_BUDGET_MS`.
const DEFAULT_ACTIVATION_WALL_BUDGET_MS: u64 = 400;
const _: () = assert!(
    DEFAULT_ACTIVATION_WALL_BUDGET_MS >= 300,
    "activation wall budget must stay above the ~200-250ms cost of one activation-phase MC tick",
);

/// Step floor for the activation loop's adaptive budget. Large enough that even
/// a slice which overshoots the wall target still clears one tick's fixed
/// per-round setup cost and makes forward progress.
const ACTIVATION_MIN_SLICE_STEPS: u64 = 250_000;

fn activation_max_slices_from(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|slices| *slices > 0)
        .unwrap_or(DEFAULT_ACTIVATION_MAX_SLICES)
}

fn activation_wall_budget_ms_from(raw: Option<&str>) -> u64 {
    raw.and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|millis| *millis > 0)
        .unwrap_or(DEFAULT_ACTIVATION_WALL_BUDGET_MS)
}

fn activate_lobby_generation(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    ipc: &mut LoopbackIpc,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> Result<(), String> {
    let Some(lobby_readiness) = lobby_readiness else {
        return Ok(());
    };
    let is_ready = |precompiled: &mut dm_lifecycle::PrecompiledLifecycle| {
        precompiled
            .persistent_state_mut()
            .is_some_and(|state| readiness_probe_matches(state, lobby_readiness))
    };
    if is_ready(precompiled) {
        return Ok(());
    }

    ipc.commit_boot_phase(
        ipc.startup_generation(),
        BootPhase::Lifecycle,
        "Finishing subsystem initialization",
    )?;
    ipc.show_startup_lobby(ipc.startup_generation())?;
    eprintln!(
        "boot-progress: generation activation gate opened read_only=true target=SSticker.current_state"
    );

    // A reconnecting native client retries once per second. Give it a short,
    // bounded window to enter the restored lobby before Monk's final
    // Master.Initialize tail performs population-sensitive storyteller setup.
    let attach_grace = env::var("DREAM64_ACTIVATION_ATTACH_GRACE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(3_000);
    let grace_deadline = Instant::now() + Duration::from_millis(attach_grace);
    while Instant::now() < grace_deadline {
        ipc.apply_lifecycle_tick_boundary(precompiled);
        std::thread::sleep(Duration::from_millis(10));
    }

    let max_slices =
        activation_max_slices_from(env::var("DREAM64_ACTIVATION_MAX_SLICES").ok().as_deref());
    let target_millis = activation_wall_budget_ms_from(
        env::var("DREAM64_SCHEDULER_WALL_BUDGET_MS").ok().as_deref(),
    );
    // The activation phase is not latency-sensitive — no session interacts with
    // the read-only lobby yet — so it runs large slices for throughput. Native
    // quickening then translates into more startup progress per slice rather
    // than merely finishing a fixed instruction count sooner. The step floor is
    // high enough that even a slice which overshoots the wall target still
    // pushes one Master Controller tick past its fixed setup cost; below that a
    // stock boot livelocks here, well short of `SSticker` pregame.
    let mut activation_budget = HostSliceBudget::new(
        ACTIVATION_MIN_SLICE_STEPS,
        ACTIVATION_MIN_SLICE_STEPS,
        10_000_000,
        Duration::from_millis(target_millis),
    );
    for slice in 1..=max_slices {
        ipc.apply_lifecycle_tick_boundary(precompiled);
        if is_ready(precompiled) {
            eprintln!(
                "boot-progress: generation activation complete slices={} lobby=pregame",
                slice - 1,
            );
            return Ok(());
        }
        let vm_started = Instant::now();
        let scheduled_steps = activation_budget.steps();
        let scheduler = advance_persistent_scheduler_responsive(
            precompiled,
            runtime,
            SchedulerDrainLimits {
                max_ticks: 1,
                max_rounds: 1,
            },
            scheduled_steps,
            Some(Duration::from_millis(target_millis)),
        )
        .map_err(|error| error.to_string())?;
        let vm_elapsed = vm_started.elapsed();
        activation_budget.observe(vm_elapsed);
        if slice == 1 || slice % 500 == 0 {
            trace_ticker_state(precompiled, slice);
        }
        if slice == 1 || slice % 100 == 0 {
            let (gc_count, gc_ms, exec_steps) = precompiled.boot_execution_totals();
            eprintln!(
                "boot-progress: generation activation slice={slice} tick={} rounds={} completed={} failed={} pending={} termination={:?} vm_us={} step_budget={} next_step_budget={} gc_count={gc_count} gc_ms={gc_ms} exec_steps={exec_steps}",
                scheduler.final_tick,
                scheduler.rounds,
                scheduler.completed_tasks,
                scheduler.failed_tasks,
                scheduler.pending_tasks,
                scheduler.termination,
                vm_elapsed.as_micros(),
                scheduled_steps,
                activation_budget.steps(),
            );
        }
        ipc.apply_lifecycle_tick_boundary(precompiled);
        if is_ready(precompiled) {
            let (gc_count, gc_ms, exec_steps) = precompiled.boot_execution_totals();
            eprintln!(
                "boot-progress: generation activation complete slices={slice} tick={} completed={} pending={} lobby=pregame gc_count={gc_count} gc_ms={gc_ms} exec_steps={exec_steps} uncaught_runtimes={}",
                scheduler.final_tick,
                scheduler.completed_tasks,
                scheduler.pending_tasks,
                // Per-site reporting goes quiet after a few examples, so this is
                // the only place the real total for the activation phase shows up.
                dm_vm::uncaught_runtime_count(),
            );
            report_boot_profiles(precompiled, "activation-complete");
            return Ok(());
        }
    }
    report_activation_timeout_diagnostics(precompiled);
    Err(format!(
        "Monke lobby did not enter pregame within {max_slices} activation slices"
    ))
}

/// Dumps the native-quickening counters and the parked DM frames after the
/// activation loop gives up, so a stuck controller tail is diagnosable from the
/// boot log alone.
fn report_activation_timeout_diagnostics(precompiled: &mut dm_lifecycle::PrecompiledLifecycle) {
    for line in precompiled.bounded_scheduler_progress() {
        eprintln!("boot-progress: activation-timeout-dm-frame {line}");
    }
    let (ruin_batches, ruin_steps) = dm_vm::native_ruin_batch_metrics();
    let (ruin_scan_activations, ruin_scan_cells, ruin_scan_rejections, ruin_scan_successes) =
        dm_vm::native_ruin_scan_metrics();
    let (ruin_flag_rejections, ruin_area_rejections) = dm_vm::native_ruin_rejection_causes();
    let (tgm_cells, tgm_safepoints, tgm_commits) = dm_vm::native_tgm_load_metrics();
    let (tgm_target_resolutions, tgm_target_cache_hits) = dm_vm::native_tgm_target_cache_metrics();
    let (tgm_build_cache_members, tgm_build_cache_logical_steps) =
        dm_vm::native_tgm_build_cache_metrics();
    let (build_coordinate_prefixes, build_coordinate_fallbacks) =
        dm_vm::native_build_coordinate_prefix_metrics();
    eprintln!(
        "boot-progress: native-quickening tgm_load_activations={} tgm_cells={} tgm_safepoints={} tgm_commits={} tgm_target_resolutions={} tgm_target_cache_hits={} tgm_build_cache_members={} tgm_build_cache_logical_steps={} build_coordinate_prefixes={} build_coordinate_fallbacks={} discover_offset_activations={} ruin_batches={} ruin_logical_steps={} ruin_scan_activations={} ruin_scan_cells={} ruin_scan_rejections={} ruin_scan_successes={} ruin_flag_rejections={} ruin_area_rejections={} ruin_rejection_cache_hits={}",
        dm_vm::native_tgm_load_activations(),
        tgm_cells,
        tgm_safepoints,
        tgm_commits,
        tgm_target_resolutions,
        tgm_target_cache_hits,
        tgm_build_cache_members,
        tgm_build_cache_logical_steps,
        build_coordinate_prefixes,
        build_coordinate_fallbacks,
        dm_vm::native_discover_offset_activations(),
        ruin_batches,
        ruin_steps,
        ruin_scan_activations,
        ruin_scan_cells,
        ruin_scan_rejections,
        ruin_scan_successes,
        ruin_flag_rejections,
        ruin_area_rejections,
        dm_vm::native_ruin_rejection_cache_hits(),
    );
    for sample in dm_vm::native_tgm_commit_samples() {
        eprintln!("boot-progress: tgm-commit {sample}");
    }
    for sample in dm_vm::native_tgm_continuation_rejections() {
        eprintln!("boot-progress: tgm-continuation-rejection {sample}");
    }
    for sample in dm_vm::native_tgm_route_samples() {
        eprintln!("boot-progress: tgm-route {sample}");
    }
    for sample in dm_vm::native_ruin_area_rejection_samples() {
        eprintln!("boot-progress: ruin-area-rejection {sample}");
    }
}

pub(crate) fn startup_scheduler_limits() -> SchedulerDrainLimits {
    let defaults = SchedulerDrainLimits::default();
    let max_rounds = env::var_os("DREAM64_STARTUP_MAX_ROUNDS")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<usize>().ok()))
        .unwrap_or(defaults.max_rounds);
    let max_ticks = env::var_os("DREAM64_STARTUP_MAX_TICKS")
        .and_then(|value| value.to_str().and_then(|value| value.parse::<u64>().ok()))
        .unwrap_or(defaults.max_ticks);
    SchedulerDrainLimits {
        max_ticks,
        max_rounds,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_ACTIVATION_MAX_SLICES, DEFAULT_ACTIVATION_WALL_BUDGET_MS,
        activation_max_slices_from, activation_wall_budget_ms_from, fresh_launch_random_seed,
        launch_random_seed_from,
    };

    #[test]
    fn activation_slice_ceiling_falls_back_on_absent_or_invalid_overrides() {
        for raw in [None, Some("  "), Some("0"), Some("not-a-number")] {
            assert_eq!(
                activation_max_slices_from(raw),
                DEFAULT_ACTIVATION_MAX_SLICES
            );
        }
        assert_eq!(activation_max_slices_from(Some(" 60000 ")), 60_000);
    }

    #[test]
    fn activation_wall_budget_falls_back_on_absent_or_invalid_overrides() {
        for raw in [None, Some("0"), Some("garbage")] {
            assert_eq!(
                activation_wall_budget_ms_from(raw),
                DEFAULT_ACTIVATION_WALL_BUDGET_MS
            );
        }
        assert_eq!(activation_wall_budget_ms_from(Some("1000")), 1_000);
    }

    #[test]
    fn launch_entropy_never_uses_the_deterministic_test_seed() {
        let first = fresh_launch_random_seed();
        let second = fresh_launch_random_seed();
        assert_ne!(first, 0);
        assert_ne!(second, 0);
        assert_ne!(first, second);
    }

    #[test]
    fn launch_seed_can_be_replayed_from_the_environment() {
        assert_eq!(
            launch_random_seed_from(Some("8675309")),
            (8_675_309, "environment")
        );
    }
}
