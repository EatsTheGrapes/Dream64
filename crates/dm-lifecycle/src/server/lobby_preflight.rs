//! `lobby-preflight` / `lobby-preview`: run `/world` Genesis and `/world/New`
//! against a minimal one-tile world, attach a local guest, and drain the
//! scheduler far enough to confirm the project's lobby UI comes up (optionally
//! staying resident to serve it).

use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use dm_lifecycle::ipc::{LoopbackIpc, parse_loopback_address};
use dm_lifecycle::{LifecycleIndex, LifecycleKind, LifecycleResolution};
use dm_runtime::RuntimeImage;
use dm_semantics::ExecutableProcedures;
use dm_value::{FieldName, TypePath, Value};
use dm_vm::{ExecutionLimits, advance_scheduler};

use super::server_loop::report_public_endpoint;

pub(crate) fn run_lobby_preflight(
    environment: &Path,
    world_params: &str,
    index: &LifecycleIndex,
    runtime: &mut RuntimeImage,
    executable: &ExecutableProcedures,
    serve: bool,
) -> ExitCode {
    const MAX_TICKS: u64 = 1_000;
    let started = Instant::now();
    let world_path = TypePath::parse("/world").expect("built-in world path is valid");
    let world = match runtime
        .canonical_world()
        .map(Ok)
        .unwrap_or_else(|| runtime.allocate_datum(&world_path))
    {
        Ok(world) => world,
        Err(error) => {
            eprintln!("lobby-preflight: world allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let genesis = match index
        .find_path("/world")
        .map(|lifecycle| lifecycle.targets.get(LifecycleKind::Genesis))
    {
        Some(LifecycleResolution::Resolved(target)) => Some(target),
        Some(LifecycleResolution::Absent) | None => None,
        Some(LifecycleResolution::Unsupported(issue)) => {
            eprintln!(
                "lobby-preflight: world Genesis is unsupported: {}",
                issue.message
            );
            return ExitCode::FAILURE;
        }
    };
    let mut state = runtime.take_execution_state();
    let world_params = match state.decode_params_list(world_params) {
        Ok(params) => params,
        Err(error) => {
            eprintln!("lobby-preflight: world params decoding failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = state.heap_mut().set_datum_field(
        world,
        FieldName::parse("params").expect("built-in world params field is valid"),
        world_params,
    ) {
        eprintln!("lobby-preflight: world params assignment failed: {error}");
        return ExitCode::FAILURE;
    }
    state.set_global(
        FieldName::parse("world").expect("built-in world global is valid"),
        Value::Datum(world),
    );
    if let Some(target) = genesis {
        let Some(entry) = executable.implementation(target.implementation) else {
            eprintln!("lobby-preflight: world Genesis VM target is missing");
            return ExitCode::FAILURE;
        };
        if let Err(error) = dm_vm::execute_module_in_context(
            executable.module(),
            entry,
            &[],
            &mut state,
            &dm_vm::ExecutionContext::new(Value::Datum(world), Value::Null),
        ) {
            eprintln!("lobby-preflight: world Genesis failed: {error}");
            return ExitCode::FAILURE;
        }
        for tick in 0..=MAX_TICKS {
            if state.scheduled_task_count() == 0 {
                break;
            }
            if let Err(error) = advance_scheduler(
                executable.module(),
                u64::from(tick != 0),
                ExecutionLimits::default(),
                &mut state,
            ) {
                eprintln!("lobby-preflight: world Genesis continuation failed: {error}");
                return ExitCode::FAILURE;
            }
            if tick == MAX_TICKS && state.scheduled_task_count() != 0 {
                eprintln!("lobby-preflight: world Genesis exceeded {MAX_TICKS} ticks");
                return ExitCode::FAILURE;
            }
        }
    }
    let world_new = match index
        .find_path("/world")
        .map(|lifecycle| lifecycle.targets.get(LifecycleKind::New))
    {
        Some(LifecycleResolution::Resolved(target)) => target,
        Some(LifecycleResolution::Absent) | None => {
            eprintln!("lobby-preflight: world/New is absent");
            return ExitCode::FAILURE;
        }
        Some(LifecycleResolution::Unsupported(issue)) => {
            eprintln!(
                "lobby-preflight: world/New is unsupported: {}",
                issue.message
            );
            return ExitCode::FAILURE;
        }
    };
    let Some(entry) = executable.implementation(world_new.implementation) else {
        eprintln!("lobby-preflight: world/New VM target is missing");
        return ExitCode::FAILURE;
    };
    if let Err(error) = dm_vm::execute_module_in_context(
        executable.module(),
        entry,
        &[],
        &mut state,
        &dm_vm::ExecutionContext::new(Value::Datum(world), Value::Null),
    ) {
        eprintln!("lobby-preflight: world/New failed: {error}");
        return ExitCode::FAILURE;
    }
    for tick in 0..=MAX_TICKS {
        if state.scheduled_task_count() == 0 {
            break;
        }
        if let Err(error) = advance_scheduler(
            executable.module(),
            u64::from(tick != 0),
            ExecutionLimits::default(),
            &mut state,
        ) {
            eprintln!("lobby-preflight: world/New continuation failed: {error}");
            return ExitCode::FAILURE;
        }
        if tick == MAX_TICKS && state.scheduled_task_count() != 0 {
            eprintln!("lobby-preflight: world/New exceeded {MAX_TICKS} ticks");
            return ExitCode::FAILURE;
        }
    }
    let area = match state
        .allocate_compact_map_datum(TypePath::parse("/area").expect("built-in area path is valid"))
    {
        Ok(area) => area,
        Err(error) => {
            eprintln!("lobby-preflight: minimal area allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let turf = match state
        .allocate_compact_map_datum(TypePath::parse("/turf").expect("built-in turf path is valid"))
    {
        Ok(turf) => turf,
        Err(error) => {
            eprintln!("lobby-preflight: minimal turf allocation failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    for (name, value) in [("x", 1), ("y", 1), ("z", 1)] {
        if let Err(error) = state.heap_mut().set_datum_field(
            turf,
            FieldName::parse(name).expect("built-in coordinate field is valid"),
            Value::number(value as f32),
        ) {
            eprintln!("lobby-preflight: minimal turf coordinate failed: {error}");
            return ExitCode::FAILURE;
        }
    }
    if let Err(error) = state.heap_mut().set_datum_field(
        turf,
        FieldName::parse("loc").expect("built-in loc field is valid"),
        Value::Datum(area),
    ) {
        eprintln!("lobby-preflight: minimal turf area assignment failed: {error}");
        return ExitCode::FAILURE;
    }
    state.rebuild_world_geometry();
    if serve {
        let ipc_address =
            env::var("DREAM64_IPC_ADDR").unwrap_or_else(|_| "0.0.0.0:51664".to_owned());
        let mut ipc = match parse_loopback_address(&ipc_address).and_then(LoopbackIpc::bind) {
            Ok(ipc) => ipc,
            Err(error) => {
                eprintln!("lobby-preview: loopback IPC: {error}");
                return ExitCode::FAILURE;
            }
        };
        report_public_endpoint(ipc.local_addr().port());
        eprintln!(
            "lobby-preview: ready project={} loopback-ipc={} hidden_guests=0 elapsed_ms={}",
            environment.display(),
            ipc.local_addr(),
            started.elapsed().as_millis()
        );
        let mut slices = 0u64;
        loop {
            let slice_started = Instant::now();
            ipc.apply_executable_tick_boundary(executable, &mut state);
            if let Err(error) = advance_scheduler(
                executable.module(),
                1,
                ExecutionLimits::default(),
                &mut state,
            ) {
                eprintln!(
                    "lobby-preview: scheduler failed: {}\ncall stack: {:#?}",
                    error.message, error.call_stack
                );
                return ExitCode::FAILURE;
            }
            slices = slices.saturating_add(1);
            if slices == 1 || slices % 100 == 0 {
                eprintln!(
                    "lobby-preview: scheduler slice={} tick={} pending={}",
                    slices,
                    state.scheduler_tick(),
                    state.scheduled_task_count()
                );
            }
            let tick = std::time::Duration::from_millis(100);
            if let Some(remaining) = tick.checked_sub(slice_started.elapsed()) {
                std::thread::sleep(remaining);
            }
        }
    }
    let attached = match state.connect_local_guest(executable.module()) {
        Ok(attached) => attached,
        Err(error) => {
            eprintln!("lobby-preflight: guest attach failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let attached_mob_type = state
        .heap()
        .datum(attached.mob)
        .map(|datum| datum.type_path().to_string())
        .unwrap_or_else(|_| "<stale>".to_owned());
    eprintln!("lobby-preflight: attached mob_type={attached_mob_type}");
    let skin_controls = state.client_session(attached.client).map_or(0, |session| {
        session
            .ui()
            .tree()
            .windows
            .iter()
            .map(|window| window.controls.len())
            .sum()
    });
    let mut events = Vec::new();
    for tick in 0..=MAX_TICKS {
        if let Err(error) = advance_scheduler(
            executable.module(),
            u64::from(tick != 0),
            ExecutionLimits::default(),
            &mut state,
        ) {
            let bound_mob = state
                .heap()
                .datum_field(
                    attached.client,
                    &FieldName::parse("mob").expect("client mob field"),
                )
                .cloned()
                .unwrap_or(Value::Null);
            let hud = state
                .heap()
                .datum_field(
                    attached.mob,
                    &FieldName::parse("hud_used").expect("mob hud field"),
                )
                .cloned()
                .unwrap_or(Value::Null);
            eprintln!(
                "lobby-preflight: /client/New failed tick={tick} skin_controls={skin_controls} ui_events={} elapsed_ms={} bound_mob={} hud_used={} error={}\ncall stack: {:#?}",
                events.len(),
                started.elapsed().as_millis(),
                bound_mob,
                hud,
                error.message,
                error.call_stack
            );
            return ExitCode::FAILURE;
        }
        events.extend(state.take_local_client_outbound_events(attached.client));
        if state.scheduled_task_count() == 0 {
            if events.is_empty() {
                eprintln!(
                    "lobby-preflight: /client/New completed without UI events ticks={tick} skin_controls={skin_controls} elapsed_ms={}",
                    started.elapsed().as_millis()
                );
                return ExitCode::FAILURE;
            }
            eprintln!(
                "lobby-preflight: ready project={} ticks={} skin_controls={} ui_events={} elapsed_ms={}",
                environment.display(),
                tick,
                skin_controls,
                events.len(),
                started.elapsed().as_millis()
            );
            return ExitCode::SUCCESS;
        }
    }
    eprintln!(
        "lobby-preflight: /client/New did not complete within {MAX_TICKS} ticks pending={} ui_events={} elapsed_ms={}",
        state.scheduled_task_count(),
        events.len(),
        started.elapsed().as_millis()
    );
    ExitCode::FAILURE
}
