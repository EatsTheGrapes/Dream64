//! The persistent headless scheduler loop and its entry paths: public-endpoint
//! discovery, per-launch random seeding, prewarmed-standby activation handoff,
//! lobby-generation activation, and the startup scheduler drain limits.

use std::collections::hash_map::RandomState;
use std::env;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead as _, BufReader, Read as _};
use std::net::{IpAddr, TcpListener};
use std::process::{Command as ProcessCommand, ExitCode};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dm_lifecycle::ipc::{BootPhase, LoopbackIpc, parse_loopback_address};
use dm_lifecycle::{
    HeadlessReadinessProbe, HostSliceBudget, SchedulerDrainLimits,
    advance_persistent_scheduler_responsive, readiness_probe_matches,
};
use dm_runtime::RuntimeImage;

use super::cli::ProductionReadyWorldIdentity;

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

pub(crate) fn run_prewarmed_standby(
    runtime: &mut RuntimeImage,
    precompiled: &mut dm_lifecycle::PrecompiledLifecycle,
    identity: &ProductionReadyWorldIdentity,
    control_address: &str,
    lobby_readiness: Option<&HeadlessReadinessProbe>,
) -> ExitCode {
    let control_address = match parse_loopback_address(control_address) {
        Ok(address) => address,
        Err(error) => {
            eprintln!("prewarm standby address: {error}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(control_address) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("prewarm standby bind {control_address}: {error}");
            return ExitCode::FAILURE;
        }
    };
    eprintln!(
        "boot-progress: prewarmed standby ready control={} deployment={:?} seed={}",
        control_address, identity.deployment_id, identity.random_seed,
    );
    let expected = format!("ACTIVATE {}", identity.deployment_id);
    loop {
        let (stream, peer) = match listener.accept() {
            Ok(connection) => connection,
            Err(error) => {
                eprintln!("prewarm standby accept: {error}");
                return ExitCode::FAILURE;
            }
        };
        let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
        let mut command = String::new();
        match BufReader::new(stream).take(4_096).read_line(&mut command) {
            Ok(_) if command.trim() == expected => {
                eprintln!(
                    "boot-progress: prewarmed standby activation accepted peer={peer} deployment={:?}",
                    identity.deployment_id,
                );
                break;
            }
            Ok(_) if command.trim() == format!("CANCEL {}", identity.deployment_id) => {
                eprintln!(
                    "boot-progress: prewarmed standby cancelled deployment={:?}",
                    identity.deployment_id,
                );
                return ExitCode::SUCCESS;
            }
            Ok(_) => eprintln!("prewarm standby rejected command from {peer}"),
            Err(error) => eprintln!("prewarm standby read from {peer}: {error}"),
        }
    }
    drop(listener);

    let ipc_address = env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "0.0.0.0:51664".to_owned());
    let ipc_address = match parse_loopback_address(&ipc_address) {
        Ok(address) => address,
        Err(error) => {
            eprintln!("loopback IPC: {error}");
            return ExitCode::FAILURE;
        }
    };
    let timeout = env::var("DREAM64_HANDOFF_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(Duration::from_secs(30), Duration::from_millis);
    let deadline = Instant::now() + timeout;
    let ipc = loop {
        match LoopbackIpc::bind_starting(ipc_address, "Activating prepared world") {
            Ok(ipc) => break ipc,
            Err(error) if Instant::now() < deadline => {
                eprintln!(
                    "boot-progress: handoff waiting for ipc={} reason={error}",
                    ipc_address
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                eprintln!(
                    "boot-progress: handoff failed ipc={} reason={error}",
                    ipc_address
                );
                return ExitCode::FAILURE;
            }
        }
    };
    report_public_endpoint(ipc.local_addr().port());
    eprintln!(
        "boot-progress: prewarmed handoff complete ipc={} deployment={:?}",
        ipc.local_addr(),
        identity.deployment_id,
    );
    run_persistent_server_loop(runtime, precompiled, Some(ipc), lobby_readiness)
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
            return ExitCode::SUCCESS;
        }
        if let Some(remaining) = tick_duration.checked_sub(slice_started.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
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

    let max_slices = env::var("DREAM64_ACTIVATION_MAX_SLICES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000);
    let target_millis = env::var("DREAM64_SCHEDULER_WALL_BUDGET_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(50)
        .max(1);
    // Native quickening must translate into more startup progress, not merely
    // finish the same fixed 100k legacy instructions earlier. The VM's wall
    // deadline remains authoritative for responsiveness; this controller
    // raises the logical ceiling while slices are comfortably under it.
    let mut activation_budget = HostSliceBudget::new(
        100_000,
        10_000,
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
        )
        .map_err(|error| error.to_string())?;
        let vm_elapsed = vm_started.elapsed();
        activation_budget.observe(vm_elapsed);
        if slice == 1 || slice % 100 == 0 {
            eprintln!(
                "boot-progress: generation activation slice={slice} tick={} rounds={} completed={} failed={} pending={} termination={:?} vm_us={} step_budget={} next_step_budget={}",
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
            eprintln!(
                "boot-progress: generation activation complete slices={slice} tick={} completed={} pending={} lobby=pregame",
                scheduler.final_tick, scheduler.completed_tasks, scheduler.pending_tasks,
            );
            return Ok(());
        }
    }
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
    Err(format!(
        "Monke lobby did not enter pregame within {max_slices} activation slices"
    ))
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
    use super::{fresh_launch_random_seed, launch_random_seed_from};

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
