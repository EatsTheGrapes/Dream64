//! Loopback-only framed IPC for local headless clients.
use crate::PrecompiledLifecycle;
use dm_value::DatumId;
use dm_vm::{ExecutionState, LocalClientUiEvent};

use std::{
    collections::BTreeMap,
    net::{SocketAddr, TcpListener},
    sync::{
        Arc, RwLock,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

#[path = "ipc/wire.rs"]
mod wire;

#[path = "ipc/text_encoding.rs"]
mod text_encoding;

use text_encoding::{
    encode_retained_ui_events, encode_snapshot, format_state, hex, read_project_resource,
    read_project_resource_chunk,
};
use wire::{Command, Request, parse_command, read_frame, write_frame};
// Resource bytes are hex encoded on this text protocol. Keep a chunk well
// below half the frame ceiling so headers and future metadata remain bounded.
const MAX_RESOURCE_CHUNK_BYTES: u32 = 256 * 1024;
static NEXT_STARTUP_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Scheduler-owned endpoint for a loopback IPC listener.
pub struct LoopbackIpc {
    address: SocketAddr,
    requests: Receiver<Request>,
    sessions: BTreeMap<String, DatumId>,
    next_session: u64,
    ui_sequences: BTreeMap<String, u64>,
    retained_ui: BTreeMap<String, Vec<(u64, LocalClientUiEvent)>>,
    readiness: BTreeMap<String, SessionReadiness>,
    startup_gate: Option<Arc<StartupGate>>,
    startup_generation: u64,
}

struct StartupGate {
    state: AtomicU64,
    phase: RwLock<String>,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
/// Owner-thread boot milestones committed under one startup-generation token.
pub enum BootPhase {
    /// Inputs and compiled artifact identity are being validated.
    ArtifactValidation = 0,
    /// Immutable program, resource, and structural tables are being linked.
    StructuralLoad = 1,
    /// Map prototypes and initializer work are being planned.
    WorldPlan = 2,
    /// The live world and map heap are being allocated on the owner thread.
    WorldAllocation = 3,
    /// Ordered DM lifecycle and subsystem startup code is executing.
    Lifecycle = 4,
    /// Runtime startup finished and clients may observe a read-only lobby.
    RuntimeStartedReadOnly = 5,
    /// The current generation may accept interactive client work.
    Interactive = 6,
}

impl StartupGate {
    const PHASE_BITS: u32 = 3;

    const fn state(generation: u64, readiness: BootPhase) -> u64 {
        (generation << Self::PHASE_BITS) | readiness as u64
    }

    const fn generation(state: u64) -> u64 {
        state >> Self::PHASE_BITS
    }

    const fn readiness(state: u64) -> u8 {
        (state & ((1 << Self::PHASE_BITS) - 1)) as u8
    }

    fn advance(&self, generation: u64, target: BootPhase) -> Result<(), &'static str> {
        let target = target as u8;
        loop {
            let state = self.state.load(Ordering::Acquire);
            if generation != Self::generation(state) {
                return Err("stale-startup-generation");
            }
            let current = Self::readiness(state);
            if target < current {
                return Err("startup-readiness-regression");
            }
            if target == current {
                return Ok(());
            }
            if self
                .state
                .compare_exchange(
                    state,
                    Self::state(generation, target.try_into().expect("valid readiness")),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    fn at_least(&self, readiness: BootPhase) -> bool {
        Self::readiness(self.state.load(Ordering::Acquire)) >= readiness as u8
    }

    fn rearm(&self, generation: u64, next_generation: u64) -> Result<(), &'static str> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            if generation != Self::generation(state) {
                return Err("stale-startup-generation");
            }
            if self
                .state
                .compare_exchange(
                    state,
                    Self::state(next_generation, BootPhase::ArtifactValidation),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }
}

impl TryFrom<u8> for BootPhase {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ArtifactValidation),
            1 => Ok(Self::StructuralLoad),
            2 => Ok(Self::WorldPlan),
            3 => Ok(Self::WorldAllocation),
            4 => Ok(Self::Lifecycle),
            5 => Ok(Self::RuntimeStartedReadOnly),
            6 => Ok(Self::Interactive),
            _ => Err(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct SessionReadiness {
    skin: bool,
    resources: bool,
    input: bool,
}

#[derive(Clone, Copy)]
enum ReadinessPhase {
    Skin,
    Resources,
    Input,
}

impl SessionReadiness {
    const fn interactive(self) -> bool {
        self.skin && self.resources && self.input
    }

    fn advance(&mut self, phase: ReadinessPhase) -> Result<(), &'static str> {
        match phase {
            ReadinessPhase::Skin => self.skin = true,
            ReadinessPhase::Resources if self.skin => self.resources = true,
            ReadinessPhase::Input if self.skin && self.resources => self.input = true,
            ReadinessPhase::Resources => return Err("skin-not-ready"),
            ReadinessPhase::Input => return Err("resources-not-ready"),
        }
        Ok(())
    }
}

impl LoopbackIpc {
    fn advance_readiness(
        &mut self,
        session: &str,
        phase: ReadinessPhase,
        state: &mut ExecutionState,
    ) -> String {
        let Some(client) = self.sessions.get(session).copied() else {
            return "error unknown-session".into();
        };
        let readiness = self.readiness.entry(session.to_owned()).or_default();
        if let Err(error) = readiness.advance(phase) {
            return format!("error {error}");
        }
        if readiness.interactive() && self.startup_interactive() {
            if let Err(error) = state.set_local_client_interactive(client, true) {
                return format!("error {error}");
            }
        }
        let phase = match phase {
            ReadinessPhase::Skin => "skin_ready",
            ReadinessPhase::Resources => "resources_ready",
            ReadinessPhase::Input => "input_ready",
        };
        format!("ok {phase} protocol=7 session={session}")
    }

    /// Binds the configured listener and starts its framing thread.
    pub fn bind(address: SocketAddr) -> Result<Self, String> {
        Self::bind_with_startup_gate(address, None)
    }

    /// Binds during host preflight, before the VM is ready to create clients.
    /// Early attach attempts receive the current phase instead of blocking the
    /// native client event loop on a scheduler boundary that does not yet exist.
    pub fn bind_starting(address: SocketAddr, phase: &str) -> Result<Self, String> {
        let generation = NEXT_STARTUP_GENERATION.fetch_add(1, Ordering::Relaxed);
        Self::bind_with_startup_gate(
            address,
            Some(Arc::new(StartupGate {
                state: AtomicU64::new(StartupGate::state(
                    generation,
                    BootPhase::ArtifactValidation,
                )),
                phase: RwLock::new(phase.to_owned()),
            })),
        )
    }

    fn bind_with_startup_gate(
        address: SocketAddr,
        startup_gate: Option<Arc<StartupGate>>,
    ) -> Result<Self, String> {
        let listener = TcpListener::bind(address).map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let (sender, requests) = mpsc::channel();
        let serve_gate = startup_gate.clone();
        thread::Builder::new()
            .name("dream64-loopback-ipc".into())
            .spawn(move || serve(listener, &sender, serve_gate.as_deref()))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            address,
            requests,
            sessions: BTreeMap::new(),
            next_session: 1,
            ui_sequences: BTreeMap::new(),
            retained_ui: BTreeMap::new(),
            readiness: BTreeMap::new(),
            startup_generation: startup_gate.as_ref().map_or(0, |gate| {
                StartupGate::generation(gate.state.load(Ordering::Relaxed))
            }),
            startup_gate,
        })
    }

    /// Updates the human-readable phase returned to clients during preflight.
    fn set_startup_description(&self, phase: &str) {
        let Some(gate) = &self.startup_gate else {
            return;
        };
        if let Ok(mut current) = gate.phase.write() {
            *current = phase.to_owned();
        }
    }

    /// Allows subsequent attach requests to enter the scheduler-owned VM.
    pub fn accept_startup_clients(&self, generation: u64) -> Result<(), &'static str> {
        if let Some(gate) = &self.startup_gate {
            gate.advance(generation, BootPhase::Interactive)?;
        }
        Ok(())
    }

    /// Exposes a read-only live lobby once Master begins subsystem work.
    pub fn show_startup_lobby(&self, generation: u64) -> Result<(), &'static str> {
        if let Some(gate) = &self.startup_gate {
            gate.advance(generation, BootPhase::RuntimeStartedReadOnly)?;
        }
        Ok(())
    }

    /// Token required to commit readiness for this listener's boot generation.
    #[must_use]
    pub const fn startup_generation(&self) -> u64 {
        self.startup_generation
    }

    /// Commits a typed boot phase and publishes its client-facing description.
    pub fn commit_boot_phase(
        &self,
        generation: u64,
        phase: BootPhase,
        description: &str,
    ) -> Result<(), &'static str> {
        if let Some(gate) = &self.startup_gate {
            gate.advance(generation, phase)?;
            if let Ok(mut current) = gate.phase.write() {
                *current = description.to_owned();
            }
        }
        Ok(())
    }

    /// Invalidates every outstanding readiness token and returns the token for
    /// a fresh startup cycle on the already-bound transport.
    pub fn rearm_startup(&mut self, phase: &str) -> Result<u64, &'static str> {
        let Some(gate) = &self.startup_gate else {
            return Ok(0);
        };
        let next = NEXT_STARTUP_GENERATION.fetch_add(1, Ordering::Relaxed);
        gate.rearm(self.startup_generation, next)?;
        self.startup_generation = next;
        self.set_startup_description(phase);
        Ok(next)
    }

    /// Enables input for clients that attached to the read-only startup lobby.
    pub fn enable_session_interaction(&self, state: &mut ExecutionState) {
        for (session, client) in &self.sessions {
            if self
                .readiness
                .get(session)
                .is_some_and(|ready| ready.interactive())
            {
                let _ = state.set_local_client_interactive(*client, true);
            }
        }
    }

    fn startup_interactive(&self) -> bool {
        self.startup_gate
            .as_ref()
            .is_none_or(|gate| gate.at_least(BootPhase::Interactive))
    }
    /// Returns the actual bound loopback address.
    #[must_use]
    pub const fn local_addr(&self) -> SocketAddr {
        self.address
    }
    /// Applies queued requests exclusively on the scheduler thread.
    pub fn apply_tick_boundary(&mut self, state: &mut ExecutionState) -> usize {
        let mut count = 0;
        while let Ok(request) = self.requests.try_recv() {
            let response = self.apply(request.command, state);
            let _ = request.response.send(response);
            count += 1;
        }
        count
    }

    /// Applies queued requests to an explicitly linked executable/state pair.
    /// This is used by lightweight hosts which have run world initialization
    /// without constructing the full map lifecycle wrapper.
    pub fn apply_executable_tick_boundary(
        &mut self,
        executable: &dm_semantics::ExecutableProcedures,
        state: &mut ExecutionState,
    ) -> usize {
        let mut count = 0;
        while let Ok(request) = self.requests.try_recv() {
            let response = match request.command {
                Command::Attach => match state.connect_local_guest(executable.module()) {
                    Ok(attached) => {
                        if let Err(error) =
                            state.set_local_client_interactive(attached.client, false)
                        {
                            let _ = request.response.send(format!("error {error}"));
                            count += 1;
                            continue;
                        }
                        let session = format!("s{}", self.next_session);
                        self.next_session += 1;
                        self.sessions.insert(session.clone(), attached.client);
                        self.ui_sequences.insert(session.clone(), 1);
                        self.retained_ui.insert(session.clone(), Vec::new());
                        self.readiness
                            .insert(session.clone(), SessionReadiness::default());
                        eprintln!(
                            "server-progress: client-attached session={} client={:?} pending={}",
                            session,
                            attached.client,
                            state.scheduled_task_count()
                        );
                        format_state("attach", &session, &attached, state.scheduler_tick())
                    }
                    Err(error) => format!("error {error}"),
                },
                Command::ScreenPointer {
                    session,
                    index,
                    generation,
                    event,
                    location,
                    params,
                } => {
                    if !self.startup_interactive() {
                        let _ = request
                            .response
                            .send("error server-starting-read-only".into());
                        count += 1;
                        continue;
                    }
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match state.queue_local_screen_pointer(
                        executable.module(),
                        client,
                        index,
                        generation,
                        event,
                        &location,
                        &params,
                    ) {
                        Ok(()) => format!("ok screen_pointer protocol=3 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::MapPointer {
                    session,
                    index,
                    generation,
                    x,
                    y,
                    z,
                    control,
                    params,
                } => {
                    if !self.startup_interactive() {
                        let _ = request
                            .response
                            .send("error server-starting-read-only".into());
                        count += 1;
                        continue;
                    }
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match state.queue_local_map_pointer(
                        executable.module(),
                        client,
                        index,
                        generation,
                        x,
                        y,
                        z,
                        &control,
                        &params,
                    ) {
                        Ok(()) => format!("ok map_pointer protocol=6 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::BrowserTopic { session, topic } => {
                    // Embedded browsers must complete their BYOND Topic
                    // handshakes while the startup lobby is read-only. Monk's
                    // media player queues every play call until its `ready`
                    // topic is accepted, and the stat/output panels use the
                    // same initialization channel. Pointer and verb input stay
                    // gated until HeadlessReady.
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match state.queue_local_browser_topic(executable.module(), client, &topic) {
                        Ok(()) => {
                            eprintln!(
                                "server-progress: browser-topic session={} bytes={}",
                                session,
                                topic.len()
                            );
                            format!("ok browser_topic protocol=4 client={session}")
                        }
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::ClientCommand { session, command } => {
                    if !self.startup_interactive() {
                        let _ = request
                            .response
                            .send("error server-starting-read-only".into());
                        count += 1;
                        continue;
                    }
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match state.queue_local_client_command(executable.module(), client, &command) {
                        Ok(()) => {
                            eprintln!(
                                "server-progress: client-command session={} command={:?}",
                                session, command
                            );
                            format!("ok client_command protocol=5 client={session}")
                        }
                        Err(error) => format!("error {error}"),
                    }
                }
                command => self.apply(command, state),
            };
            let _ = request.response.send(response);
            count += 1;
        }
        count
    }

    /// Applies queued requests to a booted lifecycle, allowing Attach to queue
    /// the project's `/client/New()` frame against its linked module.
    pub fn apply_lifecycle_tick_boundary(&mut self, lifecycle: &mut PrecompiledLifecycle) -> usize {
        let mut count = 0;
        while let Ok(request) = self.requests.try_recv() {
            let response = match request.command {
                Command::Attach => match lifecycle.connect_local_guest() {
                    Ok(attached) => {
                        let session = format!("s{}", self.next_session);
                        self.next_session += 1;
                        self.sessions.insert(session.clone(), attached.client);
                        self.ui_sequences.insert(session.clone(), 1);
                        self.retained_ui.insert(session.clone(), Vec::new());
                        self.readiness
                            .insert(session.clone(), SessionReadiness::default());
                        let tick = match lifecycle.persistent_state_mut() {
                            Some(state) => {
                                if let Err(error) =
                                    state.set_local_client_interactive(attached.client, false)
                                {
                                    let _ = request.response.send(format!("error {error}"));
                                    count += 1;
                                    continue;
                                }
                                state.scheduler_tick()
                            }
                            None => 0,
                        };
                        format_state("attach", &session, &attached, tick)
                    }
                    Err(error) => format!("error {error}"),
                },
                Command::ScreenPointer {
                    session,
                    index,
                    generation,
                    event,
                    location,
                    params,
                } => {
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match lifecycle.queue_local_screen_pointer(
                        client, index, generation, event, &location, &params,
                    ) {
                        Ok(()) => format!("ok screen_pointer protocol=3 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::MapPointer {
                    session,
                    index,
                    generation,
                    x,
                    y,
                    z,
                    control,
                    params,
                } => {
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match lifecycle.queue_local_map_pointer(
                        client, index, generation, x, y, z, &control, &params,
                    ) {
                        Ok(()) => format!("ok map_pointer protocol=6 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::BrowserTopic { session, topic } => {
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match lifecycle.queue_local_browser_topic(client, &topic) {
                        Ok(()) => format!("ok browser_topic protocol=4 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                Command::ClientCommand { session, command } => {
                    let Some(client) = self.sessions.get(&session).copied() else {
                        let _ = request.response.send("error unknown-session".into());
                        count += 1;
                        continue;
                    };
                    match lifecycle.queue_local_client_command(client, &command) {
                        Ok(()) => format!("ok client_command protocol=5 client={session}"),
                        Err(error) => format!("error {error}"),
                    }
                }
                command => match lifecycle.persistent_state_mut() {
                    Some(state) => self.apply(command, state),
                    None => "error persistent world is not ready for clients".to_owned(),
                },
            };
            let _ = request.response.send(response);
            count += 1;
        }
        count
    }
    fn apply(&mut self, command: Command, state: &mut ExecutionState) -> String {
        match command {
            Command::Ping => "ok ping protocol=1".to_owned(),
            Command::Attach => match state.create_attached_local_client() {
                Ok(attached) => {
                    if let Err(error) = state.set_local_client_interactive(attached.client, false) {
                        return format!("error {error}");
                    }
                    let session = format!("s{}", self.next_session);
                    self.next_session += 1;
                    self.sessions.insert(session.clone(), attached.client);
                    self.ui_sequences.insert(session.clone(), 1);
                    self.retained_ui.insert(session.clone(), Vec::new());
                    self.readiness
                        .insert(session.clone(), SessionReadiness::default());
                    format_state("attach", &session, &attached, state.scheduler_tick())
                }
                Err(error) => format!("error {error}"),
            },
            Command::MapSnapshot { session } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                let Ok(center) = state.local_client_view_coordinates(client) else {
                    return "error stale-session".into();
                };
                let snapshot = state.local_client_map_snapshot_for(Some(client), center.2);
                eprintln!(
                    "server-progress: map-snapshot session={} tiles={} screen={}",
                    session,
                    snapshot.tiles.len(),
                    snapshot.screen.len()
                );
                encode_snapshot(&session, state.scheduler_tick(), center, snapshot)
            }
            Command::ScreenSnapshot { session } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                // An impossible Z level skips expensive turf/occupant
                // appearance expansion while retaining the attached client's
                // authoritative screen list.
                let center = state
                    .local_client_view_coordinates(client)
                    .unwrap_or((1, 1, 1));
                let snapshot = state.local_client_map_snapshot_for(Some(client), i32::MIN);
                encode_snapshot(&session, state.scheduler_tick(), center, snapshot)
            }
            Command::Move { session, direction } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                match state
                    .queue_local_movement(client, direction)
                    .and_then(|()| state.apply_local_client_commands())
                    .and_then(|v| {
                        v.into_iter()
                            .last()
                            .ok_or_else(|| "movement was not committed".into())
                    }) {
                    Ok(attached) => {
                        format_state("move", &session, &attached, state.scheduler_tick())
                    }
                    Err(error) => format!("error {error}"),
                }
            }
            Command::Resource { session, path } => {
                if !self.sessions.contains_key(&session) {
                    return "error unknown-session".into();
                }
                match read_project_resource(state, &path) {
                    Ok(bytes) => format!(
                        "ok resource protocol=2 pathhex={} datahex={}",
                        hex(path.as_bytes()),
                        hex(&bytes)
                    ),
                    Err(error) => format!("error {error}"),
                }
            }
            Command::ResourceChunk {
                session,
                path,
                offset,
                length,
            } => {
                if !self.sessions.contains_key(&session) {
                    return "error unknown-session".into();
                }
                match read_project_resource_chunk(state, &path, offset, length) {
                    Ok(chunk) => format!(
                        "ok resource_chunk protocol=3 pathhex={} offset={} total={} eof={} datahex={}",
                        hex(path.as_bytes()),
                        offset,
                        chunk.total,
                        u8::from(chunk.eof),
                        hex(&chunk.bytes)
                    ),
                    Err(error) => format!("error {error}"),
                }
            }
            Command::UiEvents { session } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                // Stop-and-wait bounds the transport-owned backlog and leaves
                // newer VM events in the authoritative client queue until the
                // current batch is acknowledged. Re-polls therefore replay an
                // identical batch instead of growing an unbounded duplicate.
                let retained = self.retained_ui.entry(session.clone()).or_default();
                let events = if retained.is_empty() {
                    state.take_local_client_outbound_events(client)
                } else {
                    Vec::new()
                };
                if !events.is_empty() {
                    eprintln!(
                        "server-progress: ui-events session={} count={}",
                        session,
                        events.len()
                    );
                }
                let sequence = self.ui_sequences.entry(session.clone()).or_insert(1);
                retained.extend(events.into_iter().map(|event| {
                    let assigned = *sequence;
                    *sequence = sequence.saturating_add(1);
                    (assigned, event)
                }));
                encode_retained_ui_events(&session, retained)
            }
            Command::UiAck { session, sequence } => {
                if !self.sessions.contains_key(&session) {
                    return "error unknown-session".into();
                }
                let retained = self.retained_ui.entry(session.clone()).or_default();
                retained.retain(|(event_sequence, _)| *event_sequence > sequence);
                format!("ok ui_ack protocol=6 session={session} sequence={sequence}")
            }
            Command::SkinReady { session } => {
                self.advance_readiness(&session, ReadinessPhase::Skin, state)
            }
            Command::ResourcesReady { session } => {
                self.advance_readiness(&session, ReadinessPhase::Resources, state)
            }
            Command::InputReady { session } => {
                self.advance_readiness(&session, ReadinessPhase::Input, state)
            }
            Command::ScreenPointer { .. } => {
                "error screen pointer requires linked lifecycle".into()
            }
            Command::MapPointer { .. } => "error map pointer requires linked lifecycle".into(),
            Command::BrowserTopic { .. } => "error browser topic requires linked lifecycle".into(),
            Command::ClientCommand { .. } => {
                "error client command requires linked lifecycle".into()
            }
            Command::PromptResponse {
                session,
                id,
                response,
            } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                match state.submit_local_prompt_response(client, id, response) {
                    Ok(()) => format!("ok prompt_response protocol=7 client={session} id={id}"),
                    Err(error) => format!("error {error}"),
                }
            }
        }
    }
}

fn serve(listener: TcpListener, sender: &Sender<Request>, startup_gate: Option<&StartupGate>) {
    for connection in listener.incoming() {
        let Ok(mut stream) = connection else { continue };
        while let Ok(frame) = read_frame(&mut stream) {
            let response = match parse_command(&frame) {
                Ok(Command::Ping) => "ok ping protocol=1".to_owned(),
                Ok(Command::Attach)
                    if startup_gate
                        .is_some_and(|gate| !gate.at_least(BootPhase::RuntimeStartedReadOnly)) =>
                {
                    let phase = startup_gate
                        .and_then(|gate| gate.phase.read().ok().map(|phase| phase.clone()))
                        .unwrap_or_else(|| "Starting server".to_owned());
                    format!("error server-starting phasehex={}", hex(phase.as_bytes()))
                }
                Ok(command) => {
                    let (response, receive) = mpsc::sync_channel(1);
                    if sender.send(Request { command, response }).is_err() {
                        break;
                    }
                    receive
                        .recv()
                        .unwrap_or_else(|_| "error server-stopped".into())
                }
                Err(error) => format!("error {error}"),
            };
            if write_frame(&mut stream, response.as_bytes()).is_err() {
                break;
            }
        }
    }
}
/// Parses a configured TCP listener address.
pub fn parse_loopback_address(value: &str) -> Result<SocketAddr, String> {
    value.parse::<SocketAddr>().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::text_encoding::{ResourceChunk, unhex};
    use super::*;
    use dm_vm::LocalClientMapSnapshot;
    use std::net::TcpStream;
    use std::time::{Duration, Instant};
    #[test]
    fn accepts_local_and_public_listener_addresses() {
        assert!(parse_loopback_address("127.0.0.1:0").is_ok());
        assert!(parse_loopback_address("0.0.0.0:1").is_ok());
        assert!(parse_loopback_address("not-an-address").is_err());
    }

    #[test]
    fn socket_thread_waits_for_scheduler_boundary() {
        let mut ipc = LoopbackIpc::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let address = ipc.local_addr();
        let client = std::thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            write_frame(&mut stream, b"attach").unwrap();
            String::from_utf8(read_frame(&mut stream).unwrap()).unwrap()
        });
        let mut state = ExecutionState::new();
        let deadline = Instant::now() + Duration::from_secs(2);
        while ipc.apply_tick_boundary(&mut state) == 0 {
            assert!(Instant::now() < deadline, "IPC request was not queued");
            std::thread::yield_now();
        }
        assert!(client.join().unwrap().starts_with("error cannot attach"));
    }

    #[test]
    fn ping_bypasses_scheduler_boundary_during_long_vm_slice() {
        let _ipc = LoopbackIpc::bind("127.0.0.1:0".parse().unwrap()).unwrap();
        let mut stream = TcpStream::connect(_ipc.local_addr()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_millis(250)))
            .unwrap();
        let started = Instant::now();
        for _ in 0..100 {
            write_frame(&mut stream, b"ping").unwrap();
            assert_eq!(read_frame(&mut stream).unwrap(), b"ok ping protocol=1");
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(250),
            "transport-only ping waited behind an absent scheduler boundary: {elapsed:?}"
        );
        eprintln!(
            "loopback-ping count=100 elapsed={elapsed:?} average_us={}",
            elapsed.as_micros() / 100
        );
    }

    #[test]
    fn starting_listener_reports_phase_without_waiting_for_scheduler() {
        let ipc = LoopbackIpc::bind_starting(
            "127.0.0.1:0".parse().unwrap(),
            "Preflighting subsystem plans",
        )
        .unwrap();
        let mut stream = TcpStream::connect(ipc.local_addr()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        write_frame(&mut stream, b"attach").unwrap();
        assert_eq!(
            String::from_utf8(read_frame(&mut stream).unwrap()).unwrap(),
            format!(
                "error server-starting phasehex={}",
                hex(b"Preflighting subsystem plans")
            )
        );
        ipc.commit_boot_phase(
            ipc.startup_generation(),
            BootPhase::WorldAllocation,
            "Allocating map world",
        )
        .unwrap();
        write_frame(&mut stream, b"attach").unwrap();
        assert_eq!(
            String::from_utf8(read_frame(&mut stream).unwrap()).unwrap(),
            format!(
                "error server-starting phasehex={}",
                hex(b"Allocating map world")
            )
        );
    }

    #[test]
    fn startup_gate_rejects_stale_generation_commits() {
        let gate = StartupGate {
            state: AtomicU64::new(StartupGate::state(41, BootPhase::ArtifactValidation)),
            phase: RwLock::new("Loading world".into()),
        };

        assert_eq!(
            gate.advance(40, BootPhase::RuntimeStartedReadOnly),
            Err("stale-startup-generation")
        );
        assert!(!gate.at_least(BootPhase::RuntimeStartedReadOnly));
        assert_eq!(gate.advance(41, BootPhase::RuntimeStartedReadOnly), Ok(()));
        assert!(gate.at_least(BootPhase::RuntimeStartedReadOnly));
        assert!(!gate.at_least(BootPhase::Interactive));
    }

    #[test]
    fn startup_gate_is_monotonic_and_idempotent() {
        let gate = StartupGate {
            state: AtomicU64::new(StartupGate::state(7, BootPhase::ArtifactValidation)),
            phase: RwLock::new("Starting".into()),
        };

        assert_eq!(gate.advance(7, BootPhase::Interactive), Ok(()));
        assert_eq!(gate.advance(7, BootPhase::Interactive), Ok(()));
        assert_eq!(
            gate.advance(7, BootPhase::RuntimeStartedReadOnly),
            Err("startup-readiness-regression")
        );
        assert!(gate.at_least(BootPhase::Interactive));
    }

    #[test]
    fn boot_coordinator_accepts_the_complete_ordered_phase_sequence() {
        let gate = StartupGate {
            state: AtomicU64::new(StartupGate::state(19, BootPhase::ArtifactValidation)),
            phase: RwLock::new("Validating artifact".into()),
        };

        for phase in [
            BootPhase::StructuralLoad,
            BootPhase::WorldPlan,
            BootPhase::WorldAllocation,
            BootPhase::Lifecycle,
            BootPhase::RuntimeStartedReadOnly,
            BootPhase::Interactive,
        ] {
            assert_eq!(gate.advance(19, phase), Ok(()));
            assert!(gate.at_least(phase));
        }
        assert_eq!(
            gate.advance(19, BootPhase::Lifecycle),
            Err("startup-readiness-regression")
        );
    }

    #[test]
    fn rearming_startup_atomically_invalidates_the_previous_token() {
        let gate = StartupGate {
            state: AtomicU64::new(StartupGate::state(11, BootPhase::Interactive)),
            phase: RwLock::new("Ready".into()),
        };

        assert_eq!(gate.rearm(11, 12), Ok(()));
        assert!(!gate.at_least(BootPhase::RuntimeStartedReadOnly));
        assert_eq!(
            gate.advance(11, BootPhase::Interactive),
            Err("stale-startup-generation")
        );
        assert_eq!(gate.advance(12, BootPhase::RuntimeStartedReadOnly), Ok(()));
    }

    #[test]
    fn snapshot_v4_encodes_map_screen_trees_and_mouse_policy() {
        let mut state = ExecutionState::new();
        let datum = state
            .heap_mut()
            .allocate_datum(dm_value::TypePath::parse("/obj/test").unwrap());
        let child = dm_vm::LocalClientAppearance {
            datum,
            type_path: "/mutable_appearance".into(),
            icon: Some("icons/a b\t雪.dmi".into()),
            icon_state: Some("state\nunsafe".into()),
            dir: 4,
            layer: -1.5,
            plane: 7.0,
            appearance_flags: 4096,
            mouse_opacity: 2,
            pixel_x: 1.0,
            pixel_y: 2.0,
            pixel_w: 3.0,
            pixel_z: 4.0,
            color: Some("#ff00ff".into()),
            alpha: 127.5,
            maptext: None,
            maptext_width: 0.0,
            maptext_height: 0.0,
            maptext_x: 0.0,
            maptext_y: 0.0,
            underlays: vec![],
            overlays: vec![],
        };
        let parent = dm_vm::LocalClientAppearance {
            overlays: vec![child],
            ..dm_vm::LocalClientAppearance {
                datum,
                type_path: "/turf/open".into(),
                icon: None,
                icon_state: None,
                dir: 2,
                layer: 1.0,
                plane: 0.0,
                appearance_flags: 0,
                mouse_opacity: 1,
                pixel_x: 0.0,
                pixel_y: 0.0,
                pixel_w: 0.0,
                pixel_z: 0.0,
                color: None,
                alpha: 255.0,
                maptext: None,
                maptext_width: 0.0,
                maptext_height: 0.0,
                maptext_x: 0.0,
                maptext_y: 0.0,
                underlays: vec![],
                overlays: vec![],
            }
        };
        let snapshot = LocalClientMapSnapshot {
            width: 1,
            height: 1,
            z: 1,
            tiles: vec![dm_vm::LocalClientMapTile {
                x: 1,
                y: 1,
                type_path: "/turf/open\nunsafe".into(),
                color: None,
                occupants: vec![datum],
                appearances: vec![parent],
            }],
            screen: vec![dm_vm::LocalClientScreenAppearance {
                map_control: Some("map".into()),
                screen_loc: "CENTER,CENTER".into(),
                insertion: 0,
                appearance: dm_vm::LocalClientAppearance {
                    datum,
                    type_path: "/obj/screen".into(),
                    icon: None,
                    icon_state: Some("lobby".into()),
                    dir: 2,
                    layer: 20.0,
                    plane: 20.0,
                    appearance_flags: 64,
                    mouse_opacity: 0,
                    pixel_x: 0.0,
                    pixel_y: 0.0,
                    pixel_w: 0.0,
                    pixel_z: 0.0,
                    color: None,
                    alpha: 255.0,
                    maptext: None,
                    maptext_width: 0.0,
                    maptext_height: 0.0,
                    maptext_x: 0.0,
                    maptext_y: 0.0,
                    underlays: vec![],
                    overlays: vec![],
                },
            }],
        };
        let encoded = encode_snapshot("s1", 9, (4, 5, 1), snapshot);
        assert!(encoded.starts_with("ok map_snapshot protocol=4"));
        assert!(encoded.lines().any(|line| line.starts_with("S ")));
        assert_eq!(
            encoded
                .lines()
                .filter(|line| line.starts_with("A "))
                .count(),
            3
        );
        assert!(!encoded.contains("unsafe"));
        assert!(encoded.lines().any(|line| line.ends_with("4096 2")));
        assert!(encoded.lines().any(|line| line.ends_with("64 0")));
        assert_eq!(String::from_utf8(unhex("e99baa").unwrap()).unwrap(), "雪");
    }

    #[test]
    fn resource_reader_roundtrips_binary_and_rejects_traversal() {
        let root = std::env::temp_dir().join(format!(
            "dream64-ipc-resource-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("icons")).unwrap();
        let bytes = [0_u8, 255, b'\n', b'\t', 17];
        std::fs::write(root.join("icons/test.dmi"), bytes).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        assert_eq!(
            read_project_resource(&state, "icons/test.dmi").unwrap(),
            bytes
        );
        assert!(read_project_resource(&state, "../secret").is_err());
        assert_eq!(
            read_project_resource_chunk(&state, "icons/test.dmi", 2, 2).unwrap(),
            ResourceChunk {
                bytes: vec![b'\n', b'\t'],
                total: 5,
                eof: false,
            }
        );
        assert_eq!(
            read_project_resource_chunk(&state, "icons/test.dmi", 4, 2).unwrap(),
            ResourceChunk {
                bytes: vec![17],
                total: 5,
                eof: true,
            }
        );
        assert!(read_project_resource_chunk(&state, "icons/test.dmi", 6, 1).is_err());
        assert!(read_project_resource_chunk(&state, "../secret", 0, 1).is_err());
        assert_eq!(unhex(&hex(&bytes)).unwrap(), bytes);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ui_event_batch_preserves_type_order_payloads_and_sequence() {
        let events = vec![
            LocalClientUiEvent::Winset {
                control: "main.output".into(),
                parameters: "text=hello\nworld".into(),
            },
            LocalClientUiEvent::Output {
                control: "output".into(),
                message: "雪".into(),
            },
            LocalClientUiEvent::BrowseResource {
                name: "empty.bin".into(),
                bytes: vec![],
            },
            LocalClientUiEvent::Browse {
                window: "status".into(),
                html: "<b>ready</b>".into(),
            },
            LocalClientUiEvent::Prompt {
                id: 7,
                kind: dm_vm::LocalClientPromptKind::List,
                title: "Choose".into(),
                message: "Role".into(),
                default: "Engineer".into(),
                choices: vec!["Engineer".into(), "Doctor".into()],
                can_cancel: true,
            },
            LocalClientUiEvent::Sound {
                file: Some("sound/lobby.ogg".into()),
                channel: 7,
                repeat: true,
                volume: 80.0,
                frequency: 22050.0,
                pan: -25.0,
            },
        ]
        .into_iter()
        .enumerate()
        .map(|(index, event)| (41 + index as u64, event))
        .collect::<Vec<_>>();
        let encoded = encode_retained_ui_events("s7", &events);
        let lines = encoded.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], "ok ui_events protocol=6 client=s7 count=6");
        assert!(lines[1].starts_with("U 41 winset "));
        assert!(lines[2].starts_with("U 42 output "));
        assert!(lines[3].ends_with(" browse_resource 656d7074792e62696e -"));
        assert!(lines[4].starts_with("U 44 browse "));
        assert_eq!(
            lines[5],
            "U 45 prompt 7 list 1 43686f6f7365 526f6c65 456e67696e656572 456e67696e656572,446f63746f72"
        );
        assert_eq!(
            lines[6],
            "U 46 sound 7 1 80 22050 -25 736f756e642f6c6f6262792e6f6767"
        );
        assert!(!encoded.contains("hello\nworld"));
        assert!(!encoded.contains('雪'));
    }

    #[test]
    fn retained_ui_replays_identically_until_acknowledged() {
        let mut retained = vec![
            (
                11,
                LocalClientUiEvent::Output {
                    control: "output".into(),
                    message: "first".into(),
                },
            ),
            (
                12,
                LocalClientUiEvent::Browse {
                    window: "status".into(),
                    html: "<b>second</b>".into(),
                },
            ),
        ];
        let initial = encode_retained_ui_events("s1", &retained);
        assert_eq!(encode_retained_ui_events("s1", &retained), initial);

        retained.retain(|(sequence, _)| *sequence > 11);
        let after_partial_ack = encode_retained_ui_events("s1", &retained);
        assert!(!after_partial_ack.contains("U 11 "));
        assert!(after_partial_ack.contains("U 12 browse "));

        retained.retain(|(sequence, _)| *sequence > 12);
        assert_eq!(
            encode_retained_ui_events("s1", &retained),
            "ok ui_events protocol=6 client=s1 count=0\n"
        );
    }

    #[test]
    fn readiness_requires_skin_then_resources_then_input() {
        let mut readiness = SessionReadiness::default();
        assert_eq!(
            readiness.advance(ReadinessPhase::Resources),
            Err("skin-not-ready")
        );
        assert_eq!(
            readiness.advance(ReadinessPhase::Input),
            Err("resources-not-ready")
        );
        assert!(!readiness.interactive());

        readiness.advance(ReadinessPhase::Skin).unwrap();
        assert!(!readiness.interactive());
        readiness.advance(ReadinessPhase::Resources).unwrap();
        assert!(!readiness.interactive());
        readiness.advance(ReadinessPhase::Input).unwrap();
        assert!(readiness.interactive());

        // Readiness notifications are idempotent across harmless retries.
        readiness.advance(ReadinessPhase::Skin).unwrap();
        readiness.advance(ReadinessPhase::Resources).unwrap();
        readiness.advance(ReadinessPhase::Input).unwrap();
        assert!(readiness.interactive());
    }
}
