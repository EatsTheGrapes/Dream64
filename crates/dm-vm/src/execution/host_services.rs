//! Host-service job queues: environment overrides, external timers, and the
//! iconforge / SQL deterministic worker handoffs.
//!
//! Split out of `state.rs`: small `ExecutionState` helpers that mediate
//! between DM builtins and the host's out-of-band services.

use std::path::PathBuf;
use std::time::Instant;

use dm_value::Value;

use crate::execution::state::ExecutionState;

impl ExecutionState {
    pub(crate) fn environment_override(&self, name: &str) -> Option<&Option<Value>> {
        self.environment_overrides.get(name)
    }

    pub(crate) fn set_environment_override(&mut self, name: String, value: Option<Value>) {
        self.environment_overrides.insert(name, value);
    }

    pub(crate) fn reset_external_timer(&mut self, name: String) {
        self.external_timers.insert(name, Instant::now());
    }

    pub(crate) fn external_timer_milliseconds(&self, name: &str) -> f64 {
        self.external_timers
            .get(name)
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0)
    }

    pub(crate) fn external_timer_microseconds(&self, name: &str) -> f64 {
        self.external_timers
            .get(name)
            .map_or(0.0, |started| started.elapsed().as_secs_f64() * 1_000_000.0)
    }

    pub(crate) fn enqueue_iconforge_job(&mut self, result: String) -> String {
        self.iconforge_next_job = self.iconforge_next_job.saturating_add(1);
        let id = format!("dream64-iconforge-{}", self.iconforge_next_job);
        self.iconforge_jobs.insert(id.clone(), (false, result));
        id
    }

    pub(crate) fn poll_iconforge_job(&mut self, id: &str) -> Option<String> {
        let (polled, _) = self.iconforge_jobs.get(id)?;
        if !*polled {
            // Jobs are launched concurrently. One pending observation advances
            // the deterministic headless worker so the caller's next sweep can
            // collect every completed result without one sleep per job.
            for (polled, _) in self.iconforge_jobs.values_mut() {
                *polled = true;
            }
            return Some("NO RESULTS YET".to_owned());
        }
        self.iconforge_jobs.remove(id).map(|(_, result)| result)
    }

    pub(crate) fn load_iconforge_gags_config(&mut self, path: String, source: PathBuf) {
        self.iconforge_gags_configs.insert(path, source);
    }

    pub(crate) fn has_iconforge_gags_config(&self, path: &str) -> bool {
        self.iconforge_gags_configs.contains_key(path)
    }

    pub(crate) fn iconforge_gags_source(&self, path: &str) -> Option<&std::path::Path> {
        self.iconforge_gags_configs.get(path).map(PathBuf::as_path)
    }

    pub(crate) fn enqueue_sql_job(&mut self, result: String) -> String {
        self.sql_next_job = self.sql_next_job.saturating_add(1);
        let id = format!("dream64-sql-{}", self.sql_next_job);
        self.sql_jobs.insert(id.clone(), (false, result));
        id
    }

    pub(crate) fn poll_sql_job(&mut self, id: &str) -> Option<String> {
        let (polled, _) = self.sql_jobs.get(id)?;
        if !*polled {
            for (polled, _) in self.sql_jobs.values_mut() {
                *polled = true;
            }
            return Some("NO RESULTS YET".to_owned());
        }
        self.sql_jobs.remove(id).map(|(_, result)| result)
    }
}
