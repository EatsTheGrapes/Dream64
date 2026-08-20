//! Loopback-only framed IPC for local headless clients.
use crate::PrecompiledLifecycle;
use dm_value::DatumId;
use dm_vm::{
    ExecutionState, LocalClientMapSnapshot, LocalClientState, LocalClientUiEvent,
    LocalMovementDirection,
};
use std::{
    collections::BTreeMap,
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::mpsc::{self, Receiver, Sender, SyncSender},
    thread,
};
const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Command {
    Attach,
    MapSnapshot {
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
}
impl LoopbackIpc {
    /// Binds a localhost listener and starts its framing thread.
    pub fn bind(address: SocketAddr) -> Result<Self, String> {
        if !address.ip().is_loopback() {
            return Err("IPC listener must bind to a loopback address".into());
        }
        let listener = TcpListener::bind(address).map_err(|e| e.to_string())?;
        let address = listener.local_addr().map_err(|e| e.to_string())?;
        let (sender, requests) = mpsc::channel();
        thread::Builder::new()
            .name("dream64-loopback-ipc".into())
            .spawn(move || serve(listener, &sender))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            address,
            requests,
            sessions: BTreeMap::new(),
            next_session: 1,
            ui_sequences: BTreeMap::new(),
        })
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
                        let tick = lifecycle
                            .persistent_state_mut()
                            .map_or(0, |state| state.scheduler_tick());
                        format_state("attach", &session, &attached, tick)
                    }
                    Err(error) => format!("error {error}"),
                },
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
            Command::Attach => match state.create_attached_local_client() {
                Ok(attached) => {
                    let session = format!("s{}", self.next_session);
                    self.next_session += 1;
                    self.sessions.insert(session.clone(), attached.client);
                    self.ui_sequences.insert(session.clone(), 1);
                    format_state("attach", &session, &attached, state.scheduler_tick())
                }
                Err(error) => format!("error {error}"),
            },
            Command::MapSnapshot { session } => {
                let Some(client) = self.sessions.get(&session).copied() else {
                    return "error unknown-session".into();
                };
                let Ok(attached) = state.local_client_state(client) else {
                    return "error stale-session".into();
                };
                encode_snapshot(
                    &session,
                    state.scheduler_tick(),
                    state.local_client_map_snapshot(attached.z),
                )
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
                let events = state.take_local_client_outbound_events(client);
                let sequence = self.ui_sequences.entry(session.clone()).or_insert(1);
                encode_ui_events(&session, sequence, events)
            }
        }
    }
}

fn serve(listener: TcpListener, sender: &Sender<Request>) {
    for connection in listener.incoming() {
        let Ok(mut stream) = connection else { continue };
        while let Ok(frame) = read_frame(&mut stream) {
            let response = match parse_command(&frame) {
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
fn encode_snapshot(session: &str, tick: u64, snapshot: LocalClientMapSnapshot) -> String {
    let mut out = format!(
        "ok map_snapshot protocol=2 session={session} tick={tick} width={} height={} z={} tiles={}\n",
        snapshot.width,
        snapshot.height,
        snapshot.z,
        snapshot.tiles.len()
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
    out
}

fn encode_appearance(out: &mut String, appearance: &dm_vm::LocalClientAppearance) {
    use std::fmt::Write as _;
    let _ = writeln!(
        out,
        "A {:x}:{:x} {} {} {} {} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {} {:08x} {} {}",
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

fn encode_ui_events(
    session: &str,
    next_sequence: &mut u64,
    events: Vec<LocalClientUiEvent>,
) -> String {
    use std::fmt::Write as _;
    let mut output = format!(
        "ok ui_events protocol=2 client={session} count={}\n",
        events.len()
    );
    for event in events {
        let sequence = *next_sequence;
        *next_sequence = next_sequence.saturating_add(1);
        match event {
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
/// Parses and validates a configured loopback address.
pub fn parse_loopback_address(value: &str) -> Result<SocketAddr, String> {
    let a = value.parse::<SocketAddr>().map_err(|e| e.to_string())?;
    if a.ip().is_loopback() {
        Ok(a)
    } else {
        Err("IPC address must be loopback".into())
    }
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
            parse_command(b"move s1 east"),
            Ok(Command::Move {
                session: "s1".into(),
                direction: LocalMovementDirection::East
            })
        );
    }
    #[test]
    fn loopback_only() {
        assert!(parse_loopback_address("127.0.0.1:0").is_ok());
        assert!(parse_loopback_address("0.0.0.0:1").is_err());
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
    fn snapshot_v2_hex_encodes_nested_production_fields() {
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
        };
        let encoded = encode_snapshot("s1", 9, snapshot);
        assert!(encoded.starts_with("ok map_snapshot protocol=2"));
        assert_eq!(
            encoded
                .lines()
                .filter(|line| line.starts_with("A "))
                .count(),
            2
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
        let mut sequence = 41;
        let encoded = encode_ui_events(
            "s7",
            &mut sequence,
            vec![
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
            ],
        );
        let lines = encoded.lines().collect::<Vec<_>>();
        assert_eq!(lines[0], "ok ui_events protocol=2 client=s7 count=4");
        assert!(lines[1].starts_with("U 41 winset "));
        assert!(lines[2].starts_with("U 42 output "));
        assert!(lines[3].ends_with(" browse_resource 656d7074792e62696e -"));
        assert!(lines[4].starts_with("U 44 browse "));
        assert_eq!(sequence, 45);
        assert!(!encoded.contains("hello\nworld"));
        assert!(!encoded.contains('雪'));
    }
}
