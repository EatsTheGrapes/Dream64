//! Binary-only support modules for the `dream64-server` orchestrator.
//!
//! `main.rs` owns the top-level boot control flow; these submodules hold the
//! coherent pieces it drives (command-line parsing, compiled-artifact
//! preparation, the ready-world snapshot cache, the persistent scheduler loop,
//! lobby preflighting, and the human-readable reporting helpers).

pub(crate) mod artifact_pipeline;
pub(crate) mod cli;
pub(crate) mod lobby_preflight;
pub(crate) mod ready_world;
pub(crate) mod reporting;
pub(crate) mod server_loop;
