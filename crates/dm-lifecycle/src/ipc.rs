//! Loopback-only framed IPC for local headless clients.
use crate::PrecompiledLifecycle;
use dm_value::DatumId;
use dm_vm::{
    ExecutionState, LocalClientMapSnapshot, LocalClientPromptResponse, LocalClientState,
    LocalClientUiEvent, LocalMovementDirection,
};
use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender},
    },
    thread,
};
const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
enum Command {
    Ping,
    Attach,
    MapSnapshot {
        session: String,
    },
    ScreenSnapshot {
        session: String,
    },
    Move {
        session: String,
        direction: LocalMovementDirection,
    },
    Resource {
        session: String,
        path: String,
    },
    UiEvents {
        session: String,
    },
    UiAck {
        session: String,
        sequence: u64,
    },
    SkinReady {
        session: String,
    },
    ResourcesReady {
        session: String,
    },
    InputReady {
        session: String,
    },
    ScreenPointer {
        session: String,
        index: u32,
        generation: u32,
        event: dm_vm::LocalScreenPointerEvent,
        location: String,
        params: String,
    },
    MapPointer {
        session: String,
        index: u32,
        generation: u32,
        x: i32,
        y: i32,
        z: i32,
        control: String,
        params: String,
    },
    BrowserTopic {
        session: String,
        topic: String,
    },
    ClientCommand {
        session: String,
        command: String,
    },
    PromptResponse {
        session: String,
        id: u64,
        response: LocalClientPromptResponse,
    },
}
struct Request {
    command: Command,
    response: SyncSender<String>,
}

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
}

struct StartupGate {
    accepting: AtomicBool,
    interactive: AtomicBool,
    phase: RwLock<String>,
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
        Self::bind_with_startup_gate(
            address,
            Some(Arc::new(StartupGate {
                accepting: AtomicBool::new(false),
                interactive: AtomicBool::new(false),
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
            startup_gate,
        })
    }

    /// Updates the human-readable phase returned to clients during preflight.
    pub fn set_startup_phase(&self, phase: &str) {
        let Some(gate) = &self.startup_gate else {
            return;
        };
        if let Ok(mut current) = gate.phase.write() {
            *current = phase.to_owned();
        }
    }

    /// Allows subsequent attach requests to enter the scheduler-owned VM.
    pub fn accept_startup_clients(&self) {
        if let Some(gate) = &self.startup_gate {
            gate.interactive.store(true, Ordering::Release);
            gate.accepting.store(true, Ordering::Release);
        }
    }

    /// Exposes a read-only live lobby once Master begins subsystem work.
    pub fn show_startup_lobby(&self) {
        if let Some(gate) = &self.startup_gate {
            gate.accepting.store(true, Ordering::Release);
        }
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
            .is_none_or(|gate| gate.interactive.load(Ordering::Acquire))
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
                    if startup_gate.is_some_and(|gate| !gate.accepting.load(Ordering::Acquire)) =>
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
fn parse_command(frame: &[u8]) -> Result<Command, String> {
    let mut p = std::str::from_utf8(frame)
        .map_err(|_| "frame is not UTF-8".to_owned())?
        .split_ascii_whitespace();
    match p.next() {
        Some("ping") if p.next().is_none() => Ok(Command::Ping),
        Some("attach") if p.next().is_none() => Ok(Command::Attach),
        Some("map_snapshot") => {
            let session = p
                .next()
                .ok_or("map_snapshot session is missing")?
                .to_owned();
            if p.next().is_some() {
                Err("map_snapshot has trailing arguments".into())
            } else {
                Ok(Command::MapSnapshot { session })
            }
        }
        Some("screen_snapshot") => {
            let session = p
                .next()
                .ok_or("screen_snapshot session is missing")?
                .to_owned();
            if p.next().is_some() {
                Err("screen_snapshot has trailing arguments".into())
            } else {
                Ok(Command::ScreenSnapshot { session })
            }
        }
        Some("move") => {
            let session = p.next().ok_or("move session is missing")?.to_owned();
            let direction = match p.next() {
                Some("north") => LocalMovementDirection::North,
                Some("south") => LocalMovementDirection::South,
                Some("east") => LocalMovementDirection::East,
                Some("west") => LocalMovementDirection::West,
                _ => return Err("move direction is invalid".into()),
            };
            if p.next().is_some() {
                Err("move has trailing arguments".into())
            } else {
                Ok(Command::Move { session, direction })
            }
        }
        Some("resource") => {
            let session = p.next().ok_or("resource session is missing")?.to_owned();
            let path = p.next().ok_or("resource path is missing")?;
            if p.next().is_some() {
                return Err("resource has trailing arguments".into());
            }
            let path = String::from_utf8(unhex(path)?)
                .map_err(|_| "resource path is not UTF-8".to_owned())?;
            Ok(Command::Resource { session, path })
        }
        Some("ui_events") => {
            let session = p.next().ok_or("ui_events session is missing")?.to_owned();
            if p.next().is_some() {
                Err("ui_events has trailing arguments".into())
            } else {
                Ok(Command::UiEvents { session })
            }
        }
        Some("ui_ack") => {
            let session = p.next().ok_or("ui_ack session is missing")?.to_owned();
            let sequence = p
                .next()
                .ok_or("ui_ack sequence is missing")?
                .parse()
                .map_err(|_| "ui_ack sequence is invalid")?;
            if p.next().is_some() {
                Err("ui_ack has trailing arguments".into())
            } else {
                Ok(Command::UiAck { session, sequence })
            }
        }
        Some(command @ ("skin_ready" | "resources_ready" | "input_ready")) => {
            let session = p.next().ok_or("readiness session is missing")?.to_owned();
            if p.next().is_some() {
                return Err("readiness command has trailing arguments".into());
            }
            Ok(match command {
                "skin_ready" => Command::SkinReady { session },
                "resources_ready" => Command::ResourcesReady { session },
                "input_ready" => Command::InputReady { session },
                _ => unreachable!(),
            })
        }
        Some("screen_pointer") => {
            let session = p
                .next()
                .ok_or("screen_pointer session is missing")?
                .to_owned();
            let target = p.next().ok_or("screen_pointer target is missing")?;
            let (index, generation) = target
                .split_once(':')
                .ok_or("screen_pointer target is invalid")?;
            let index =
                u32::from_str_radix(index, 16).map_err(|_| "screen_pointer index is invalid")?;
            let generation = u32::from_str_radix(generation, 16)
                .map_err(|_| "screen_pointer generation is invalid")?;
            let event = match p.next() {
                Some("entered") => dm_vm::LocalScreenPointerEvent::Entered,
                Some("exited") => dm_vm::LocalScreenPointerEvent::Exited,
                Some("click") => dm_vm::LocalScreenPointerEvent::Click,
                _ => return Err("screen_pointer event is invalid".into()),
            };
            let decode_text = |value: &str| -> Result<String, String> {
                let bytes = if value == "-" {
                    Vec::new()
                } else {
                    unhex(value)?
                };
                String::from_utf8(bytes).map_err(|_| "screen_pointer field is not UTF-8".into())
            };
            let location = decode_text(p.next().ok_or("screen_pointer location is missing")?)?;
            let params = decode_text(p.next().ok_or("screen_pointer params are missing")?)?;
            if p.next().is_some() {
                return Err("screen_pointer has trailing arguments".into());
            }
            Ok(Command::ScreenPointer {
                session,
                index,
                generation,
                event,
                location,
                params,
            })
        }
        Some("map_pointer") => {
            let session = p.next().ok_or("map_pointer session is missing")?.to_owned();
            let target = p.next().ok_or("map_pointer target is missing")?;
            let (index, generation) = target
                .split_once(':')
                .ok_or("map_pointer target is invalid")?;
            let index =
                u32::from_str_radix(index, 16).map_err(|_| "map_pointer index is invalid")?;
            let generation = u32::from_str_radix(generation, 16)
                .map_err(|_| "map_pointer generation is invalid")?;
            let x = p
                .next()
                .ok_or("map_pointer x is missing")?
                .parse()
                .map_err(|_| "map_pointer x is invalid")?;
            let y = p
                .next()
                .ok_or("map_pointer y is missing")?
                .parse()
                .map_err(|_| "map_pointer y is invalid")?;
            let z = p
                .next()
                .ok_or("map_pointer z is missing")?
                .parse()
                .map_err(|_| "map_pointer z is invalid")?;
            let decode_text = |value: &str| -> Result<String, String> {
                let bytes = if value == "-" {
                    Vec::new()
                } else {
                    unhex(value)?
                };
                String::from_utf8(bytes).map_err(|_| "map_pointer field is not UTF-8".into())
            };
            let control = decode_text(p.next().ok_or("map_pointer control is missing")?)?;
            let params = decode_text(p.next().ok_or("map_pointer params are missing")?)?;
            if p.next().is_some() {
                return Err("map_pointer has trailing arguments".into());
            }
            Ok(Command::MapPointer {
                session,
                index,
                generation,
                x,
                y,
                z,
                control,
                params,
            })
        }
        Some("browser_topic") => {
            let session = p
                .next()
                .ok_or("browser_topic session is missing")?
                .to_owned();
            let topic = p.next().ok_or("browser_topic payload is missing")?;
            if p.next().is_some() {
                return Err("browser_topic has trailing arguments".into());
            }
            let topic = String::from_utf8(unhex(topic)?)
                .map_err(|_| "browser_topic payload is not UTF-8".to_owned())?;
            Ok(Command::BrowserTopic { session, topic })
        }
        Some("client_command") => {
            let session = p
                .next()
                .ok_or("client_command session is missing")?
                .to_owned();
            let command = p.next().ok_or("client_command payload is missing")?;
            if p.next().is_some() {
                return Err("client_command has trailing arguments".into());
            }
            let command = String::from_utf8(unhex(command)?)
                .map_err(|_| "client_command payload is not UTF-8".to_owned())?;
            Ok(Command::ClientCommand { session, command })
        }
        Some("prompt_response") => {
            let session = p
                .next()
                .ok_or("prompt_response session is missing")?
                .to_owned();
            let id = p
                .next()
                .ok_or("prompt_response id is missing")?
                .parse()
                .map_err(|_| "prompt_response id is invalid")?;
            let kind = p.next().ok_or("prompt_response kind is missing")?;
            let payload = p.next().ok_or("prompt_response payload is missing")?;
            if p.next().is_some() {
                return Err("prompt_response has trailing arguments".into());
            }
            let response = match kind {
                "null" if payload == "-" => LocalClientPromptResponse::Null,
                "text" => LocalClientPromptResponse::Text(
                    String::from_utf8(unhex(payload)?)
                        .map_err(|_| "prompt_response text is not UTF-8")?,
                ),
                "number" => LocalClientPromptResponse::Number(
                    payload
                        .parse()
                        .map_err(|_| "prompt_response number is invalid")?,
                ),
                "choice" => LocalClientPromptResponse::Choice(
                    payload
                        .parse()
                        .map_err(|_| "prompt_response choice is invalid")?,
                ),
                _ => return Err("prompt_response kind is invalid".into()),
            };
            Ok(Command::PromptResponse {
                session,
                id,
                response,
            })
        }
        Some(_) => Err("unknown command".into()),
        None => Err("empty command".into()),
    }
}
fn format_state(kind: &str, session: &str, state: &LocalClientState, tick: u64) -> String {
    format!(
        "ok {kind} protocol=1 client={session} mob=[0xd{:06x}] tick={tick} x={} y={} z={}",
        state.mob.index() + 1,
        state.x,
        state.y,
        state.z
    )
}
fn encode_snapshot(
    session: &str,
    tick: u64,
    center: (i32, i32, i32),
    snapshot: LocalClientMapSnapshot,
) -> String {
    let mut out = format!(
        "ok map_snapshot protocol=3 session={session} tick={tick} width={} height={} x={} y={} z={} tiles={} screen={}\n",
        snapshot.width,
        snapshot.height,
        center.0,
        center.1,
        center.2,
        snapshot.tiles.len(),
        snapshot.screen.len()
    );
    for tile in snapshot.tiles {
        use std::fmt::Write as _;
        let color = optional_hex(tile.color.as_deref());
        let occupants = tile
            .occupants
            .iter()
            .map(|id| format!("{:x}:{:x}", id.index(), id.generation()))
            .collect::<Vec<_>>()
            .join(",");
        let _ = writeln!(
            out,
            "T {} {} {} {} {} {}",
            tile.x,
            tile.y,
            hex(tile.type_path.as_bytes()),
            color,
            occupants,
            tile.appearances.len(),
        );
        for appearance in &tile.appearances {
            encode_appearance(&mut out, appearance);
        }
    }
    for screen in snapshot.screen {
        use std::fmt::Write as _;
        let _ = writeln!(
            out,
            "S {:x}:{:x} {} {} {}",
            screen.appearance.datum.index(),
            screen.appearance.datum.generation(),
            screen.insertion,
            optional_hex(screen.map_control.as_deref()),
            optional_hex(Some(screen.screen_loc.as_str()))
        );
        encode_appearance(&mut out, &screen.appearance);
    }
    out
}

fn encode_appearance(out: &mut String, appearance: &dm_vm::LocalClientAppearance) {
    use std::fmt::Write as _;
    let _ = writeln!(
        out,
        "A {:x}:{:x} {} {} {} {} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {} {:08x} {} {} {} {:08x} {:08x} {:08x} {:08x}",
        appearance.datum.index(),
        appearance.datum.generation(),
        hex(appearance.type_path.as_bytes()),
        optional_hex(appearance.icon.as_deref()),
        optional_hex(appearance.icon_state.as_deref()),
        appearance.dir,
        appearance.layer.to_bits(),
        appearance.plane.to_bits(),
        appearance.pixel_x.to_bits(),
        appearance.pixel_y.to_bits(),
        appearance.pixel_w.to_bits(),
        appearance.pixel_z.to_bits(),
        optional_hex(appearance.color.as_deref()),
        appearance.alpha.to_bits(),
        appearance.underlays.len(),
        appearance.overlays.len(),
        optional_hex(appearance.maptext.as_deref()),
        appearance.maptext_width.to_bits(),
        appearance.maptext_height.to_bits(),
        appearance.maptext_x.to_bits(),
        appearance.maptext_y.to_bits(),
    );
    for child in &appearance.underlays {
        encode_appearance(out, child);
    }
    for child in &appearance.overlays {
        encode_appearance(out, child);
    }
}

fn optional_hex(value: Option<&str>) -> String {
    value.map_or_else(
        || "-".to_owned(),
        |value| {
            if value.is_empty() {
                "~".to_owned()
            } else {
                hex(value.as_bytes())
            }
        },
    )
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn unhex(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("hex field has odd length".to_owned());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16)
                .map_err(|_| "hex field is invalid".to_owned())
        })
        .collect()
}

fn read_project_resource(state: &ExecutionState, path: &str) -> Result<Vec<u8>, String> {
    use std::path::{Component, Path};
    let root = state
        .project_root()
        .ok_or_else(|| "project root is unavailable".to_owned())?;
    let relative = Path::new(path);
    if relative.is_absolute()
        || relative.components().any(|part| {
            matches!(
                part,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("resource path escapes project root".to_owned());
    }
    let root = root.canonicalize().map_err(|error| error.to_string())?;
    let target = root
        .join(relative)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    if !target.starts_with(&root) {
        return Err("resource path escapes project root".to_owned());
    }
    std::fs::read(target).map_err(|error| error.to_string())
}

fn encode_retained_ui_events(session: &str, events: &[(u64, LocalClientUiEvent)]) -> String {
    use std::fmt::Write as _;
    let mut output = format!(
        "ok ui_events protocol=6 client={session} count={}\n",
        events.len()
    );
    for (sequence, event) in events {
        match event.clone() {
            LocalClientUiEvent::Winset {
                control,
                parameters,
            } => {
                let _ = writeln!(
                    output,
                    "U {sequence} winset {} {}",
                    required_hex(control.as_bytes()),
                    required_hex(parameters.as_bytes())
                );
            }
            LocalClientUiEvent::Output { control, message } => {
                let _ = writeln!(
                    output,
                    "U {sequence} output {} {}",
                    required_hex(control.as_bytes()),
                    required_hex(message.as_bytes())
                );
            }
            LocalClientUiEvent::BrowseResource { name, bytes } => {
                let _ = writeln!(
                    output,
                    "U {sequence} browse_resource {} {}",
                    required_hex(name.as_bytes()),
                    required_hex(&bytes)
                );
            }
            LocalClientUiEvent::Browse { window, html } => {
                let _ = writeln!(
                    output,
                    "U {sequence} browse {} {}",
                    required_hex(window.as_bytes()),
                    required_hex(html.as_bytes())
                );
            }
            LocalClientUiEvent::Prompt {
                id,
                kind,
                title,
                message,
                default,
                choices,
                can_cancel,
            } => {
                let kind = match kind {
                    dm_vm::LocalClientPromptKind::Text => "text",
                    dm_vm::LocalClientPromptKind::Message => "message",
                    dm_vm::LocalClientPromptKind::Number => "number",
                    dm_vm::LocalClientPromptKind::Color => "color",
                    dm_vm::LocalClientPromptKind::File => "file",
                    dm_vm::LocalClientPromptKind::List => "list",
                    dm_vm::LocalClientPromptKind::Alert => "alert",
                };
                let choices = if choices.is_empty() {
                    "-".to_owned()
                } else {
                    choices
                        .iter()
                        .map(|choice| required_hex(choice.as_bytes()))
                        .collect::<Vec<_>>()
                        .join(",")
                };
                let _ = writeln!(
                    output,
                    "U {sequence} prompt {id} {kind} {} {} {} {} {}",
                    u8::from(can_cancel),
                    required_hex(title.as_bytes()),
                    required_hex(message.as_bytes()),
                    required_hex(default.as_bytes()),
                    choices,
                );
            }
            LocalClientUiEvent::Sound {
                file,
                channel,
                repeat,
                volume,
                frequency,
                pan,
            } => {
                let path = file
                    .as_deref()
                    .map_or_else(|| "-".to_owned(), |path| required_hex(path.as_bytes()));
                let _ = writeln!(
                    output,
                    "U {sequence} sound {channel} {} {volume} {frequency} {pan} {path}",
                    u8::from(repeat),
                );
            }
        }
    }
    output
}

fn required_hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".to_owned()
    } else {
        hex(bytes)
    }
}
fn read_frame(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut h = [0; 4];
    stream.read_exact(&mut h)?;
    let len = u32::from_be_bytes(h) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IPC frame is too large",
        ));
    }
    let mut p = vec![0; len];
    stream.read_exact(&mut p)?;
    Ok(p)
}
fn write_frame(stream: &mut TcpStream, payload: &[u8]) -> io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "IPC frame is too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(payload)
}
/// Parses a configured TCP listener address.
pub fn parse_loopback_address(value: &str) -> Result<SocketAddr, String> {
    value.parse::<SocketAddr>().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    #[test]
    fn parses_session_commands() {
        assert_eq!(parse_command(b"attach"), Ok(Command::Attach));
        assert_eq!(
            parse_command(b"map_snapshot s1"),
            Ok(Command::MapSnapshot {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"resource s1 69636f6e732f746573742e646d69"),
            Ok(Command::Resource {
                session: "s1".into(),
                path: "icons/test.dmi".into(),
            })
        );
        assert_eq!(
            parse_command(b"ui_events s1"),
            Ok(Command::UiEvents {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"ui_ack s1 47"),
            Ok(Command::UiAck {
                session: "s1".into(),
                sequence: 47,
            })
        );
        assert_eq!(
            parse_command(b"skin_ready s1"),
            Ok(Command::SkinReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"resources_ready s1"),
            Ok(Command::ResourcesReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"input_ready s1"),
            Ok(Command::InputReady {
                session: "s1".into()
            })
        );
        assert_eq!(
            parse_command(b"move s1 east"),
            Ok(Command::Move {
                session: "s1".into(),
                direction: LocalMovementDirection::East
            })
        );
        assert_eq!(
            parse_command(b"screen_pointer s1 a:2 click - 6d6f7573652d783d31"),
            Ok(Command::ScreenPointer {
                session: "s1".into(),
                index: 10,
                generation: 2,
                event: dm_vm::LocalScreenPointerEvent::Click,
                location: String::new(),
                params: "mouse-x=1".into(),
            })
        );
        assert_eq!(
            parse_command(b"map_pointer s1 a:2 5 7 1 - 6c6566743d31"),
            Ok(Command::MapPointer {
                session: "s1".into(),
                index: 10,
                generation: 2,
                x: 5,
                y: 7,
                z: 1,
                control: String::new(),
                params: "left=1".into(),
            })
        );
        assert_eq!(
            parse_command(b"browser_topic s1 62796f6e643a2f2f3f616374696f6e3d7265616479"),
            Ok(Command::BrowserTopic {
                session: "s1".into(),
                topic: "byond://?action=ready".into(),
            })
        );
        assert_eq!(
            parse_command(b"client_command s1 726566726573682d7467756920226c6f626279206e6f7722"),
            Ok(Command::ClientCommand {
                session: "s1".into(),
                command: "refresh-tgui \"lobby now\"".into(),
            })
        );
        assert_eq!(
            parse_command(b"prompt_response s1 9 text 68656c6c6f"),
            Ok(Command::PromptResponse {
                session: "s1".into(),
                id: 9,
                response: LocalClientPromptResponse::Text("hello".into()),
            })
        );
        assert_eq!(
            parse_command(b"prompt_response s1 10 choice 2"),
            Ok(Command::PromptResponse {
                session: "s1".into(),
                id: 10,
                response: LocalClientPromptResponse::Choice(2),
            })
        );
    }
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
        ipc.set_startup_phase("Allocating map world");
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
    fn snapshot_v3_hex_encodes_map_and_screen_appearance_trees() {
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
        assert!(encoded.starts_with("ok map_snapshot protocol=3"));
        assert!(encoded.lines().any(|line| line.starts_with("S ")));
        assert_eq!(
            encoded
                .lines()
                .filter(|line| line.starts_with("A "))
                .count(),
            3
        );
        assert!(!encoded.contains("unsafe"));
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
