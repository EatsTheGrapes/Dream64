//! Deterministic scheduler draining for post-initialization and persistent
//! server host slices.
//!
//! This module owns [`SchedulerDrainLimits`], [`HostSliceBudget`],
//! [`SchedulerDrainTermination`], [`SchedulerDrain`], and the `drain_*`/`advance_*`
//! pipeline.
//!
//! # Hidden coupling
//!
//! [`drain_startup_scheduler`] returns `Result<SchedulerDrain, InitializationExecutionError>`.
//! [`advance_persistent_scheduler`] and [`advance_persistent_scheduler_responsive`]
//! access [`PrecompiledLifecycle`] fields directly. These couplings are preserved
//! from the original monolith. They will be decoupled in the future `execute.rs`
//! extraction.

use std::env;
use std::time::{Duration, Instant};

use dm_runtime::RuntimeImage;
use dm_semantics::ExecutableProcedures;
use dm_vm::{ExecutionLimits, Module, RuntimeError, advance_scheduler};

use crate::execute::InitializationExecutionError;
use crate::precompile::PrecompiledLifecycle;
use crate::readiness::{HeadlessReadinessProbe, readiness_probe_matches};

/// Safety bounds for the post-initialization deterministic scheduler drain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SchedulerDrainLimits {
    /// Maximum scheduler ticks advanced from the lifecycle completion tick.
    pub max_ticks: u64,
    /// Maximum dispatch rounds, including zero-delay rescheduling rounds.
    pub max_rounds: usize,
}

impl Default for SchedulerDrainLimits {
    fn default() -> Self {
        Self {
            max_ticks: 10_000,
            max_rounds: 10_000,
        }
    }
}

/// Adaptive instruction budget for latency-sensitive persistent host slices.
///
/// The VM remains single-owner. This controller only changes how frequently
/// execution returns to the host so sockets, timers, and completed immutable
/// worker jobs can be serviced.
#[derive(Clone, Debug)]
pub struct HostSliceBudget {
    current_steps: u64,
    minimum_steps: u64,
    maximum_steps: u64,
    target: Duration,
}

impl HostSliceBudget {
    /// Creates a bounded controller, clamping the initial budget into range.
    #[must_use]
    pub fn new(
        initial_steps: u64,
        minimum_steps: u64,
        maximum_steps: u64,
        target: Duration,
    ) -> Self {
        let minimum_steps = minimum_steps.max(1);
        let maximum_steps = maximum_steps.max(minimum_steps);
        Self {
            current_steps: initial_steps.clamp(minimum_steps, maximum_steps),
            minimum_steps,
            maximum_steps,
            target: target.max(Duration::from_micros(1)),
        }
    }

    /// Instruction ceiling for the next persistent scheduler round.
    #[must_use]
    pub const fn steps(&self) -> u64 {
        self.current_steps
    }

    /// Records VM wall time and adjusts the following instruction ceiling.
    ///
    /// An over-target slice halves immediately. Sustained slices below half
    /// the target recover gradually, avoiding oscillation around the target.
    pub fn observe(&mut self, elapsed: Duration) {
        if elapsed > self.target {
            self.current_steps = (self.current_steps / 2).max(self.minimum_steps);
        } else if elapsed <= self.target / 2 {
            let growth = (self.current_steps / 4).max(1);
            self.current_steps = self
                .current_steps
                .saturating_add(growth)
                .min(self.maximum_steps);
        }
    }
}

/// Honest reason the bounded post-initialization scheduler drain stopped.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SchedulerDrainTermination {
    /// No scheduled work remains.
    #[default]
    StableIdle,
    /// The configured codebase-owned readiness marker was observed while
    /// persistent scheduled work remained.
    HeadlessReady,
    /// Work remains beyond the configured tick budget.
    TickLimit,
    /// Work remains after the configured dispatch-round budget.
    RoundLimit,
}

/// Deterministic post-initialization scheduler summary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerDrain {
    /// Scheduler tick at which draining stopped.
    pub final_tick: u64,
    /// Number of dispatch rounds performed.
    pub rounds: usize,
    /// Number of tasks which ran to completion.
    pub completed_tasks: usize,
    /// Number of persistent scheduled threads terminated by an isolated
    /// runtime error during this drain.
    ///
    /// Startup drains continue by default and report any startup-thread
    /// failure in this field.
    pub failed_tasks: usize,
    /// Tasks still pending when draining stopped.
    pub pending_tasks: usize,
    /// Why draining stopped.
    pub termination: SchedulerDrainTermination,
}

pub(crate) fn drain_startup_scheduler(
    executable: &ExecutableProcedures,
    state: &mut dm_vm::ExecutionState,
    limits: SchedulerDrainLimits,
    readiness: Option<&HeadlessReadinessProbe>,
    mut startup_service: Option<&mut dyn FnMut(&ExecutableProcedures, &mut dm_vm::ExecutionState)>,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let start_tick = state.scheduler_tick();
    let tick_limit = start_tick.saturating_add(limits.max_ticks);
    let wall_clock_budget = scheduler_wall_clock_budget();
    let drain_started = Instant::now();
    let mut drain = SchedulerDrain {
        final_tick: start_tick,
        ..SchedulerDrain::default()
    };
    loop {
        if let Some(service) = startup_service.as_deref_mut() {
            service(executable, state);
        }
        if state.scheduled_task_count() == 0 {
            break;
        }
        if readiness.is_some_and(|probe| readiness_probe_matches(state, probe)) {
            drain.termination = SchedulerDrainTermination::HeadlessReady;
            break;
        }
        if drain.rounds >= limits.max_rounds {
            drain.termination = SchedulerDrainTermination::RoundLimit;
            break;
        }
        let remaining_wall_budget = match wall_clock_budget {
            Some(budget) => match budget.checked_sub(drain_started.elapsed()) {
                Some(remaining) => Some(remaining),
                None => {
                    drain.termination = SchedulerDrainTermination::RoundLimit;
                    break;
                }
            },
            None => None,
        };
        let next_tick = state
            .next_scheduled_tick()
            .expect("a non-empty scheduler has an earliest task");
        if next_tick > tick_limit {
            drain.termination = SchedulerDrainTermination::TickLimit;
            break;
        }
        let advance = next_tick.saturating_sub(state.scheduler_tick());
        match advance_scheduler(
            executable.module(),
            advance,
            ExecutionLimits {
                max_steps: startup_scheduler_max_steps(),
                wall_clock_budget: remaining_wall_budget,
                ..ExecutionLimits::default()
            },
            state,
        ) {
            Ok(completed) => {
                drain.rounds += 1;
                drain.completed_tasks += completed.len();
                drop(completed);
            }
            Err(error) => {
                if !startup_fail_fast_on_error() && scheduler_budget_exhausted(&error) {
                    drain.rounds += 1;
                    state.release_host_value_roots();
                    drain.final_tick = state.scheduler_tick();
                    continue;
                }
                if startup_fail_fast_on_error() {
                    return Err(InitializationExecutionError::Scheduler(error));
                }
                drain.rounds += 1;
                drain.failed_tasks = drain.failed_tasks.saturating_add(1);
                eprintln!(
                    "startup-runtime: isolated scheduled thread failure (continuing): {error}"
                );
            }
        }
        state.release_host_value_roots();
        drain.final_tick = state.scheduler_tick();
        if drain.rounds == 1 || drain.rounds % 1000 == 0 {
            eprintln!(
                "boot-progress: startup-scheduler slice={} tick={} completed={} failed={} pending={}",
                drain.rounds,
                drain.final_tick,
                drain.completed_tasks,
                drain.failed_tasks,
                state.scheduled_task_count()
            );
        }
    }
    drain.pending_tasks = state.scheduled_task_count();
    if readiness.is_some_and(|probe| readiness_probe_matches(state, probe)) {
        drain.termination = SchedulerDrainTermination::HeadlessReady;
    } else if drain.pending_tasks == 0 {
        drain.termination = SchedulerDrainTermination::StableIdle;
    }
    eprintln!(
        "boot-progress: scheduler termination={:?} tick={} rounds={} completed={} pending={}",
        drain.termination,
        drain.final_tick,
        drain.rounds,
        drain.completed_tasks,
        drain.pending_tasks
    );
    if !matches!(
        drain.termination,
        SchedulerDrainTermination::HeadlessReady | SchedulerDrainTermination::StableIdle
    ) {
        for line in state.bounded_scheduler_progress(executable.module()) {
            eprintln!("boot-progress: bounded-dm-frame {line}");
        }
    }
    Ok(drain)
}

fn startup_fail_fast_on_error() -> bool {
    static STARTUP_CONTINUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STARTUP_CONTINUE.get_or_init(|| {
        env::var_os("DREAM64_STRICT_STARTUP_ERRORS").is_some()
            || env::var_os("DREAM64_FAIL_FAST_STARTUP_ERRORS").is_some()
            || env::var_os("DREAM64_STARTUP_FATAL").is_some()
    })
}

fn scheduler_budget_exhausted(error: &RuntimeError) -> bool {
    error
        .message
        .strip_prefix("instruction budget of ")
        .and_then(|rest| rest.strip_suffix(" exhausted"))
        .is_some()
}

/// Advances persistent scheduled server work in a bounded host-loop slice.
/// Pending continuations remain in the runtime image for the next slice.
pub fn advance_persistent_scheduler(
    precompiled: &mut PrecompiledLifecycle,
    _runtime: &mut RuntimeImage,
    limits: SchedulerDrainLimits,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let mut state = precompiled
        .persistent_state
        .take()
        .expect("persistent scheduler requires completed precompiled lifecycle execution");
    let result = drain_persistent_scheduler(
        precompiled.executable.module(),
        &mut state,
        limits,
        ExecutionLimits::default(),
    );
    precompiled.persistent_state = Some(state);
    result.map_err(InitializationExecutionError::Scheduler)
}

/// Advances persistent work with an instruction-bounded cooperative dispatch.
/// Budget exhaustion retains the scheduled continuation at the same tick, so
/// the host can service transport queues before resuming exact VM state.
pub fn advance_persistent_scheduler_responsive(
    precompiled: &mut PrecompiledLifecycle,
    _runtime: &mut RuntimeImage,
    limits: SchedulerDrainLimits,
    max_steps_per_round: u64,
) -> Result<SchedulerDrain, InitializationExecutionError> {
    let mut state = precompiled
        .persistent_state
        .take()
        .expect("persistent scheduler requires completed precompiled lifecycle execution");
    let result = drain_persistent_scheduler(
        precompiled.executable.module(),
        &mut state,
        limits,
        ExecutionLimits {
            max_steps: max_steps_per_round.max(1),
            wall_clock_budget: scheduler_wall_clock_budget(),
            ..ExecutionLimits::default()
        },
    );
    precompiled.persistent_state = Some(state);
    result.map_err(InitializationExecutionError::Scheduler)
}

fn scheduler_wall_clock_budget() -> Option<Duration> {
    const DEFAULT_MILLIS: u64 = 50;
    let millis = std::env::var("DREAM64_SCHEDULER_WALL_BUDGET_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_MILLIS);
    (millis > 0).then(|| Duration::from_millis(millis))
}

fn startup_scheduler_max_steps() -> u64 {
    const DEFAULT_STEPS: u64 = 100_000;
    std::env::var("DREAM64_STARTUP_SCHEDULER_MAX_STEPS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_STEPS)
        .max(1)
}

/// Canonical path of the Master Controller initialization thread. If an
/// uncaught runtime error tears this `set waitfor = 0` continuation down, the
/// codebase never advances `init_stage_completed`, `Master.Loop` spins on the
/// unchanged stage forever, and the world never becomes ready. Isolating that
/// failure the way ordinary background threads are isolated would turn a boot
/// abort into a silent multi-minute hang, so it is surfaced as fatal instead.
const MASTER_INITIALIZE_PROC: &str = "/datum/controller/master/proc/Initialize";

fn runtime_error_unwound_master_initialize(error: &RuntimeError) -> bool {
    error
        .call_stack
        .iter()
        .any(|frame| frame.procedure.split('@').next() == Some(MASTER_INITIALIZE_PROC))
}

fn drain_persistent_scheduler(
    module: &Module,
    state: &mut dm_vm::ExecutionState,
    limits: SchedulerDrainLimits,
    execution_limits: ExecutionLimits,
) -> Result<SchedulerDrain, RuntimeError> {
    let start_tick = state.scheduler_tick();
    let tick_limit = start_tick.saturating_add(limits.max_ticks);
    let mut drain = SchedulerDrain {
        final_tick: start_tick,
        ..SchedulerDrain::default()
    };

    while state.scheduled_task_count() != 0 {
        if drain.rounds >= limits.max_rounds {
            drain.termination = SchedulerDrainTermination::RoundLimit;
            break;
        }
        let next_tick = state
            .next_scheduled_tick()
            .expect("a non-empty scheduler has an earliest task");
        if next_tick > tick_limit {
            drain.termination = SchedulerDrainTermination::TickLimit;
            break;
        }
        let advance = next_tick.saturating_sub(state.scheduler_tick());
        drain.rounds = drain.rounds.saturating_add(1);
        match advance_scheduler(module, advance, execution_limits, state) {
            Ok(completed) => {
                drain.completed_tasks = drain.completed_tasks.saturating_add(completed.len());
                drop(completed);
                state.release_host_value_roots();
            }
            Err(error) => {
                if scheduler_budget_exhausted(&error) {
                    state.release_host_value_roots();
                    drain.final_tick = state.scheduler_tick();
                    continue;
                }
                state.release_host_value_roots();
                if runtime_error_unwound_master_initialize(&error) {
                    // Losing the MC initialization thread is unrecoverable: no
                    // other continuation advances `init_stage_completed`, so the
                    // world never becomes ready. Surface it instead of spinning.
                    eprintln!(
                        "server-runtime: fatal scheduled thread failure — the Master Controller initialization thread was terminated by an uncaught runtime error: {error}"
                    );
                    return Err(error);
                }
                // `advance_scheduler` drops only the failing continuation and
                // restores every later due task to scheduler state. Match the
                // server scheduler's thread isolation here: report the full
                // source-mapped failure, then keep draining the other work.
                drain.failed_tasks = drain.failed_tasks.saturating_add(1);
                eprintln!(
                    "server-runtime: isolated scheduled thread failure (continuing): {error}"
                );
            }
        }
        drain.final_tick = state.scheduler_tick();
    }

    // A persistent server owns a clock even when no DM continuation is
    // pending. Advance an otherwise idle/between-task slice to its bounded
    // tick boundary. RoundLimit is the exception: same-tick work must retain
    // its tick and source order for the next host slice.
    if drain.termination != SchedulerDrainTermination::RoundLimit
        && state.scheduler_tick() < tick_limit
    {
        drain.rounds = drain.rounds.saturating_add(1);
        let completed = advance_scheduler(
            module,
            tick_limit.saturating_sub(state.scheduler_tick()),
            execution_limits,
            state,
        )
        .expect("no scheduled task is due before the validated persistent tick boundary");
        drain.completed_tasks = drain.completed_tasks.saturating_add(completed.len());
        drop(completed);
        state.release_host_value_roots();
        drain.final_tick = state.scheduler_tick();
    }

    drain.pending_tasks = state.scheduled_task_count();
    if drain.pending_tasks == 0 {
        drain.termination = SchedulerDrainTermination::StableIdle;
    } else if drain.termination != SchedulerDrainTermination::RoundLimit {
        drain.termination = SchedulerDrainTermination::TickLimit;
    }
    Ok(drain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dm_syntax::parse;
    use dm_vm::{CallTrace, ExecutionState, compile_module, execute_module_in_state};

    fn runtime_error_with_stack(procedures: &[&str]) -> RuntimeError {
        RuntimeError {
            message: "CRASH: boom".to_owned(),
            instruction: 0,
            source_span: None,
            call_stack: procedures
                .iter()
                .map(|procedure| CallTrace {
                    procedure: (*procedure).to_owned(),
                    instruction: 0,
                    source_span: None,
                })
                .collect(),
        }
    }

    #[test]
    fn master_initialize_unwind_detection_ignores_the_procedure_id_suffix() {
        assert!(runtime_error_unwound_master_initialize(
            &runtime_error_with_stack(&[
                "/datum/controller/master/proc/Initialize@5486",
                "/datum/controller/master/proc/init_subsystem@5487",
            ])
        ));
        assert!(runtime_error_unwound_master_initialize(
            &runtime_error_with_stack(&["/datum/controller/master/proc/Initialize",])
        ));
        assert!(!runtime_error_unwound_master_initialize(
            &runtime_error_with_stack(&[
                "/datum/controller/master/proc/InitializeSomethingElse@1",
                "/datum/controller/master/proc/Loop@5490",
            ])
        ));
        assert!(!runtime_error_unwound_master_initialize(
            &runtime_error_with_stack(&[])
        ));
    }

    /// A crafted `set waitfor = 0` `Master.Initialize` continuation that fails
    /// after it has already been detached must abort the persistent drain, and
    /// an identically shaped background thread must still be isolated.
    #[test]
    fn persistent_drain_is_fatal_only_when_master_initialize_unwinds() {
        const SOURCE: &str = concat!(
            "/datum/controller/master/proc/Initialize()\n",
            "\tset waitfor = 0\n",
            "\tsleep(1)\n",
            "\tboom_helper()\n",
            "/proc/boom_helper()\n",
            "\tCRASH(\"mc boom\")\n",
            "/proc/background_thread()\n",
            "\tset waitfor = 0\n",
            "\tsleep(1)\n",
            "\tCRASH(\"background boom\")\n",
            "/proc/boot_master()\n",
            "\tvar/datum/controller/master/controller = new /datum/controller/master\n",
            "\tcontroller.Initialize()\n",
            "/proc/boot_background()\n",
            "\tbackground_thread()\n",
        );
        let syntax = parse(SOURCE).expect("scheduler fixture should parse");
        let module = compile_module(&syntax.definitions).expect("scheduler fixture should compile");

        let limits = SchedulerDrainLimits {
            max_ticks: 10,
            max_rounds: 10,
        };

        let mut mc_state = ExecutionState::new();
        let boot_master = module
            .procedure_id("/proc/boot_master")
            .expect("boot_master entry");
        execute_module_in_state(&module, boot_master, &[], &mut mc_state)
            .expect("boot detaches the waitfor=0 Master.Initialize continuation");
        assert_eq!(mc_state.scheduled_task_count(), 1);
        let error =
            drain_persistent_scheduler(&module, &mut mc_state, limits, ExecutionLimits::default())
                .expect_err("losing Master.Initialize must abort the drain");
        assert!(runtime_error_unwound_master_initialize(&error));

        let mut bg_state = ExecutionState::new();
        let boot_background = module
            .procedure_id("/proc/boot_background")
            .expect("boot_background entry");
        execute_module_in_state(&module, boot_background, &[], &mut bg_state)
            .expect("boot detaches the waitfor=0 background continuation");
        assert_eq!(bg_state.scheduled_task_count(), 1);
        let bg_drain =
            drain_persistent_scheduler(&module, &mut bg_state, limits, ExecutionLimits::default())
                .expect("an ordinary background thread failure stays isolated");
        assert_eq!(bg_drain.failed_tasks, 1);
    }
}
