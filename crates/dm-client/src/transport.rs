use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};

use dm_compiler::CompilerDatabase;
use dm_lifecycle::artifact::CompiledArtifact;
use dm_project::Project;
use dm_runtime::RuntimeImage;
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::ExecutionState;
use dm_world::{AtomCategory, WorldCoordinate};

use super::{Appearance, ClientPromptKind, ClientPrompt, InboundUiCommand, SoundUpdate};

pub(crate) const MAX_SNAPSHOT_RESOURCES: usize = 4_096;
pub(crate) const MAX_RESOURCE_BYTES: usize = 16 * 1024 * 1024;
pub(crate) const RESOURCE_CHUNK_BYTES: u32 = 256 * 1024;
pub(crate) const MAX_SNAPSHOT_RESOURCE_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SnapshotResourceBudget {
    pub(crate) count: usize,
    pub(crate) bytes: usize,
}

impl SnapshotResourceBudget {
    pub(crate) fn reserve(&mut self, bytes: usize) -> Result<(), &'static str> {
        self.count = self.count.checked_add(1).ok_or("resource count overflow")?;
        if self.count > MAX_SNAPSHOT_RESOURCES {
            return Err("snapshot resource count exceeds limit");
        }
        self.bytes = self
            .bytes
            .checked_add(bytes)
            .ok_or("snapshot resource byte accounting overflow")?;
        if self.bytes > MAX_SNAPSHOT_RESOURCE_BYTES {
            return Err("snapshot resource bytes exceed limit");
        }
        Ok(())
    }
}

pub(crate) struct WorldScene {
    pub(crate) cells: BTreeMap<(i32, i32, i32), u32>,
    pub(crate) turfs: BTreeMap<(i32, i32, i32), DatumId>,
    pub(crate) player: WorldCoordinate,
    pub(crate) player_mob: DatumId,
    pub(crate) label: String,
}

/// A transport-neutral map image returned to a connected client.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MapSnapshot {
    pub(crate) center: WorldCoordinate,
    pub(crate) cells: BTreeMap<(i32, i32, i32), u32>,
    pub(crate) turf_targets: BTreeMap<(i32, i32, i32), (u32, u32)>,
    pub(crate) appearances: BTreeMap<(i32, i32, i32), Vec<Appearance>>,
    pub(crate) screen: Vec<ScreenAppearance>,
    pub(crate) resources: BTreeMap<PathBuf, Vec<u8>>,
}

pub(crate) fn snapshot_has_lobby_screen(snapshot: &MapSnapshot) -> bool {
    snapshot.screen.iter().any(|screen| {
        screen.type_path.to_ascii_lowercase().contains("splash")
            || screen.appearances.iter().any(|appearance| {
                appearance
                    .resource
                    .to_string_lossy()
                    .to_ascii_lowercase()
                    .contains("background_monke.dmi")
            })
    })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScreenAppearance {
    pub(crate) datum_index: u32,
    pub(crate) datum_generation: u32,
    pub(crate) map_control: Option<String>,
    pub(crate) screen_loc: String,
    pub(crate) type_path: String,
    pub(crate) insertion: usize,
    pub(crate) appearances: Vec<Appearance>,
}

/// In-process implementation of the client/server exchange. Keeping the
/// window behind this boundary makes a future TCP transport a codec swap
/// rather than another renderer implementation.
pub(crate) struct LoopbackTransport {
    pub(crate) scene: WorldScene,
    attached: Option<DatumId>,
}

pub(crate) enum ClientTransport {
    Pending(PendingRemoteTransport),
    Remote(RemoteTransport),
    Offline(LoopbackTransport),
}

pub(crate) struct PendingRemoteTransport {
    pub(crate) address: SocketAddr,
    pub(crate) record: Option<PathBuf>,
    pub(crate) next_attempt: std::time::Instant,
    pub(crate) last_error: Option<String>,
}

pub(crate) enum ClientPromptResponse {
    Null,
    Text(String),
    Number(f32),
    Choice(usize),
}

impl ClientTransport {
    pub(crate) fn try_connect(&mut self) -> bool {
        let Self::Pending(pending) = self else {
            return false;
        };
        if std::time::Instant::now() < pending.next_attempt {
            return false;
        }
        pending.next_attempt = std::time::Instant::now() + std::time::Duration::from_secs(1);
        match RemoteTransport::connect(pending.address, pending.record.as_deref()).and_then(
            |mut transport| {
                transport.attach()?;
                Ok(transport)
            },
        ) {
            Ok(transport) => {
                eprintln!("client-server-ready: {}", transport.label);
                *self = Self::Remote(transport);
                true
            }
            Err(error) => {
                let status = startup_status_from_error(&error).unwrap_or(error);
                if pending.last_error.as_deref() != Some(&status) {
                    eprintln!("client-server-waiting: {status}");
                    pending.last_error = Some(status);
                }
                false
            }
        }
    }

    pub(crate) fn request_snapshot(&mut self) -> Result<MapSnapshot, String> {
        match self {
            Self::Pending(_) => Err("server is still starting".to_owned()),
            Self::Remote(transport) => transport.request_snapshot(),
            Self::Offline(transport) => Ok(transport.request_snapshot(10, 7)),
        }
    }

    pub(crate) fn request_screen_snapshot(&mut self) -> Result<MapSnapshot, String> {
        match self {
            Self::Pending(_) => Err("server is still starting".to_owned()),
            Self::Remote(transport) => transport.request_screen_snapshot(),
            Self::Offline(transport) => Ok(transport.request_snapshot(10, 7)),
        }
    }

    pub(crate) fn send_movement(
        &mut self,
        runtime: &mut ExecutionState,
        dx: i32,
        dy: i32,
    ) -> Result<Option<MapSnapshot>, String> {
        match self {
            Self::Pending(_) => Ok(None),
            Self::Remote(transport) => transport.send_movement(dx, dy),
            Self::Offline(transport) => Ok(transport.send_movement(runtime, dx, dy)),
        }
    }

    pub(crate) fn label(&self) -> &str {
        match self {
            Self::Pending(pending) => pending.last_error.as_deref().unwrap_or("Starting server"),
            Self::Remote(transport) => &transport.label,
            Self::Offline(transport) => transport.label(),
        }
    }

    pub(crate) fn poll_ui_events(&mut self) -> Result<Vec<(u64, InboundUiCommand)>, String> {
        match self {
            Self::Pending(_) => Ok(Vec::new()),
            Self::Remote(transport) => transport.poll_ui_events(),
            Self::Offline(_) => Ok(Vec::new()),
        }
    }

    pub(crate) fn acknowledge_ui(&mut self, sequence: u64) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.acknowledge_ui(sequence),
        }
    }

    pub(crate) fn mark_resources_ready(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.send_readiness("resources_ready"),
        }
    }

    pub(crate) fn mark_skin_ready(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.send_readiness("skin_ready"),
        }
    }

    pub(crate) fn mark_input_ready(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.send_readiness("input_ready"),
        }
    }

    pub(crate) fn send_screen_pointer(
        &mut self,
        target: (u32, u32),
        event: &str,
        location: &str,
        params: &str,
    ) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => {
                transport.send_screen_pointer(target, event, location, params)
            }
            Self::Offline(_) => Ok(()),
        }
    }

    pub(crate) fn send_map_pointer(
        &mut self,
        target: (u32, u32),
        coordinate: WorldCoordinate,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => {
                transport.send_map_pointer(target, coordinate, control, params)
            }
            Self::Offline(_) => Ok(()),
        }
    }

    pub(crate) fn send_browser_topic(&mut self, topic: &str) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => transport.send_browser_topic(topic),
            Self::Offline(_) => Ok(()),
        }
    }

    pub(crate) fn send_command(&mut self, command: &str) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => transport.send_command(command),
            // Replays contain server-to-client state, not a live VM to mutate.
            Self::Offline(_) => Ok(()),
        }
    }

    pub(crate) fn reconnect(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(pending) => {
                pending.next_attempt = std::time::Instant::now();
                pending.last_error = None;
                Ok(())
            }
            Self::Remote(transport) => {
                let address = transport
                    .address
                    .ok_or("a replay transport cannot reconnect")?;
                *self = Self::Pending(PendingRemoteTransport {
                    address,
                    record: None,
                    next_attempt: std::time::Instant::now(),
                    last_error: None,
                });
                Ok(())
            }
            Self::Offline(_) => Err("an offline client cannot reconnect".to_owned()),
        }
    }

    pub(crate) fn request_resource(&mut self, path: &str) -> Result<Vec<u8>, String> {
        match self {
            Self::Pending(_) => Err(format!(
                "server is still starting; resource {path:?} unavailable"
            )),
            Self::Remote(transport) => transport.request_resource(path),
            Self::Offline(_) => Err(format!("offline world has no resource {path:?}")),
        }
    }

    pub(crate) fn send_prompt_response(
        &mut self,
        id: u64,
        response: ClientPromptResponse,
    ) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => transport.send_prompt_response(id, response),
            Self::Offline(_) => Ok(()),
        }
    }

    pub(crate) fn is_live(&self) -> bool {
        matches!(
            self,
            Self::Remote(RemoteTransport {
                stream: ProtocolStream::Live(_),
                ..
            })
        )
    }
}

pub(crate) struct RemoteTransport {
    stream: ProtocolStream,
    address: Option<SocketAddr>,
    label: String,
    client_token: Option<String>,
    center: Option<WorldCoordinate>,
    recorder: Option<ReplayRecorder>,
    resource_cache: BTreeMap<String, Vec<u8>>,
}

const REPLAY_MAGIC: &[u8] = b"D64REPLAY\0\x01";

pub(crate) enum ProtocolStream {
    Live(TcpStream),
    Replay(ReplayReader),
}

pub(crate) struct ReplayRecorder {
    file: std::fs::File,
    recorded_resources: BTreeSet<String>,
}

impl ReplayRecorder {
    pub(crate) fn create(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::create(path)
            .map_err(|error| format!("create replay {}: {error}", path.display()))?;
        file.write_all(REPLAY_MAGIC)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            file,
            recorded_resources: BTreeSet::new(),
        })
    }

    pub(crate) fn record(&mut self, command: &str, response: &str) -> Result<(), String> {
        if command.starts_with("resource ") && !self.recorded_resources.insert(command.to_owned()) {
            return Ok(());
        }
        write_replay_blob(&mut self.file, command.as_bytes())?;
        write_replay_blob(&mut self.file, response.as_bytes())?;
        self.file.flush().map_err(|error| error.to_string())
    }
}

pub(crate) struct ReplayReader {
    responses: BTreeMap<String, VecDeque<String>>,
    repeatable_resources: BTreeMap<String, String>,
}

impl ReplayReader {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::open(path)
            .map_err(|error| format!("open replay {}: {error}", path.display()))?;
        let mut magic = vec![0; REPLAY_MAGIC.len()];
        file.read_exact(&mut magic)
            .map_err(|error| error.to_string())?;
        if magic != REPLAY_MAGIC {
            return Err(format!("{} is not a Dream64 replay", path.display()));
        }
        let mut responses = BTreeMap::<String, VecDeque<String>>::new();
        let mut repeatable_resources = BTreeMap::new();
        loop {
            let Some(command) = read_replay_blob(&mut file)? else {
                break;
            };
            let response = read_replay_blob(&mut file)?
                .ok_or_else(|| "truncated replay response".to_owned())?;
            let command =
                String::from_utf8(command).map_err(|_| "replay command is not UTF-8".to_owned())?;
            let response = String::from_utf8(response)
                .map_err(|_| "replay response is not UTF-8".to_owned())?;
            if command.starts_with("resource ") {
                repeatable_resources.insert(command.clone(), response.clone());
            }
            responses.entry(command).or_default().push_back(response);
        }
        Ok(Self {
            responses,
            repeatable_resources,
        })
    }

    pub(crate) fn exchange(&mut self, command: &str) -> Result<String, String> {
        if let Some(response) = self
            .responses
            .get_mut(command)
            .and_then(VecDeque::pop_front)
        {
            return Ok(response);
        }
        if command.starts_with("ui_events ") {
            return Ok("ok ui_events count=0\n".to_owned());
        }
        if let Some(kind) = [
            "browser_topic ",
            "client_command ",
            "screen_pointer ",
            "map_pointer ",
            "prompt_response ",
        ]
        .into_iter()
        .find(|prefix| command.starts_with(prefix))
        {
            let kind = kind.trim_end();
            return Ok(format!("ok {kind} replay=ignored\n"));
        }
        if let Some(response) = self.repeatable_resources.get(command) {
            return Ok(response.clone());
        }
        // A UI event can request a late browser asset that was unavailable
        // during capture (for example Monk's hidden command-bar spy page).
        // Replays are immutable snapshots, so serve an empty deterministic
        // resource instead of repeatedly surfacing a non-fatal transport error.
        if command.starts_with("resource ") {
            return Ok("ok resource datahex=".to_owned());
        }
        Err(format!("replay has no remaining response for {command:?}"))
    }
}

fn write_replay_blob(file: &mut std::fs::File, bytes: &[u8]) -> Result<(), String> {
    let length = u32::try_from(bytes.len()).map_err(|_| "replay entry exceeds 4 GiB".to_owned())?;
    file.write_all(&length.to_le_bytes())
        .map_err(|error| error.to_string())?;
    file.write_all(bytes).map_err(|error| error.to_string())
}

fn read_replay_blob(file: &mut std::fs::File) -> Result<Option<Vec<u8>>, String> {
    let mut length = [0; 4];
    match file.read_exact(&mut length) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error.to_string()),
    }
    let mut bytes = vec![0; u32::from_le_bytes(length) as usize];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(Some(bytes))
}

impl RemoteTransport {
    pub(crate) fn is_replay(&self) -> bool {
        matches!(self.stream, ProtocolStream::Replay(_))
    }

    pub(crate) fn request_resource(&mut self, path: &str) -> Result<Vec<u8>, String> {
        if let Some(bytes) = self.resource_cache.get(path) {
            return Ok(bytes.clone());
        }
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?
            .to_owned();
        // Protocol-2 recordings contain the historical whole-resource command.
        // Live transports use bounded protocol-3 chunks so hex expansion can
        // never exceed the server's 1 MiB frame limit.
        if !self.is_replay() {
            let mut bytes = Vec::new();
            let mut expected_total = None;
            loop {
                let offset = u64::try_from(bytes.len())
                    .map_err(|_| "resource offset exceeds 64-bit range".to_owned())?;
                let response = self.exchange(&format!(
                    "resource_chunk {session} {} {offset} {RESOURCE_CHUNK_BYTES}",
                    encode_hex(path.as_bytes())
                ))?;
                require_ok(&response, "resource_chunk")?;
                let fields = response_fields(&response);
                let response_offset = fields
                    .get("offset")
                    .ok_or("resource chunk omitted offset")?
                    .parse::<u64>()
                    .map_err(|_| "resource chunk offset is invalid")?;
                if response_offset != offset {
                    return Err("resource chunk offset does not match request".to_owned());
                }
                let total = fields
                    .get("total")
                    .ok_or("resource chunk omitted total length")?
                    .parse::<u64>()
                    .map_err(|_| "resource chunk total length is invalid")?;
                if total > MAX_RESOURCE_BYTES as u64 {
                    return Err(format!(
                        "invalid resource {path}: payload exceeds resource byte limit"
                    ));
                }
                if expected_total
                    .replace(total)
                    .is_some_and(|old| old != total)
                {
                    return Err("resource length changed during transfer".to_owned());
                }
                let encoded = fields
                    .get("datahex")
                    .ok_or_else(|| format!("resource chunk omitted data for {path}"))?;
                let chunk = decode_hex_bounded(encoded, RESOURCE_CHUNK_BYTES as usize)
                    .map_err(|error| format!("invalid resource {path}: {error}"))?;
                if chunk.is_empty() && offset != total {
                    return Err("resource chunk made no progress".to_owned());
                }
                bytes.extend_from_slice(&chunk);
                if bytes.len() as u64 > total {
                    return Err("resource chunks exceed advertised length".to_owned());
                }
                let eof = match fields.get("eof").map(String::as_str) {
                    Some("0") => false,
                    Some("1") => true,
                    _ => return Err("resource chunk EOF flag is invalid".to_owned()),
                };
                if eof {
                    if bytes.len() as u64 != total {
                        return Err("resource ended before advertised length".to_owned());
                    }
                    self.resource_cache.insert(path.to_owned(), bytes.clone());
                    return Ok(bytes);
                }
            }
        }
        let response = self.exchange(&format!(
            "resource {session} {}",
            encode_hex(path.as_bytes())
        ))?;
        require_ok(&response, "resource")?;
        let fields = response_fields(&response);
        let encoded = fields
            .get("datahex")
            .ok_or_else(|| format!("resource response omitted data for {path}"))?;
        let bytes = decode_hex_bounded(encoded, MAX_RESOURCE_BYTES)
            .map_err(|error| format!("invalid resource {path}: {error}"))?;
        self.resource_cache.insert(path.to_owned(), bytes.clone());
        Ok(bytes)
    }

    pub(crate) fn send_browser_topic(&mut self, topic: &str) -> Result<(), String> {
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let response = self.exchange(&format!(
            "browser_topic {session} {}",
            encode_hex(topic.as_bytes())
        ))?;
        require_ok(&response, "browser_topic")
    }

    pub(crate) fn send_command(&mut self, command: &str) -> Result<(), String> {
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let response = self.exchange(&format!(
            "client_command {session} {}",
            encode_hex(command.as_bytes())
        ))?;
        require_ok(&response, "client_command")
    }

    pub(crate) fn send_prompt_response(
        &mut self,
        id: u64,
        response: ClientPromptResponse,
    ) -> Result<(), String> {
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let (kind, payload) = match response {
            ClientPromptResponse::Null => ("null", "-".to_owned()),
            ClientPromptResponse::Text(value) => ("text", encode_hex(value.as_bytes())),
            ClientPromptResponse::Number(value) => ("number", value.to_string()),
            ClientPromptResponse::Choice(index) => ("choice", index.to_string()),
        };
        let response =
            self.exchange(&format!("prompt_response {session} {id} {kind} {payload}"))?;
        require_ok(&response, "prompt_response")
    }

    pub(crate) fn send_screen_pointer(
        &mut self,
        target: (u32, u32),
        event: &str,
        location: &str,
        params: &str,
    ) -> Result<(), String> {
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let command = format!(
            "screen_pointer {session} {:x}:{:x} {event} {} {}",
            target.0,
            target.1,
            if location.is_empty() {
                "-".into()
            } else {
                encode_hex(location.as_bytes())
            },
            if params.is_empty() {
                "-".into()
            } else {
                encode_hex(params.as_bytes())
            }
        );
        let response = self.exchange(&command)?;
        require_ok(&response, "screen_pointer")
    }

    pub(crate) fn send_map_pointer(
        &mut self,
        target: (u32, u32),
        coordinate: WorldCoordinate,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let text_field = |value: &str| {
            if value.is_empty() {
                "-".to_owned()
            } else {
                encode_hex(value.as_bytes())
            }
        };
        let response = self.exchange(&format!(
            "map_pointer {session} {:x}:{:x} {} {} {} {} {}",
            target.0,
            target.1,
            coordinate.x,
            coordinate.y,
            coordinate.z,
            text_field(control),
            text_field(params),
        ))?;
        require_ok(&response, "map_pointer")
    }

    pub(crate) fn connect(address: SocketAddr, record: Option<&Path>) -> Result<Self, String> {
        let stream =
            TcpStream::connect(address).map_err(|error| format!("connect {address}: {error}"))?;
        stream
            .set_nodelay(true)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            stream: ProtocolStream::Live(stream),
            address: Some(address),
            label: format!("server {address}"),
            client_token: None,
            center: None,
            recorder: record.map(ReplayRecorder::create).transpose()?,
            resource_cache: BTreeMap::new(),
        })
    }

    pub(crate) fn replay(path: &Path) -> Result<Self, String> {
        Ok(Self {
            stream: ProtocolStream::Replay(ReplayReader::open(path)?),
            address: None,
            label: format!("replay {}", path.display()),
            client_token: None,
            center: None,
            recorder: None,
            resource_cache: BTreeMap::new(),
        })
    }

    pub(crate) fn attach(&mut self) -> Result<(), String> {
        let response = self.exchange("attach")?;
        require_ok(&response, "attach")?;
        let fields = response_fields(&response);
        self.client_token = fields.get("client").cloned();
        self.center = coordinate_fields(&fields);
        Ok(())
    }

    pub(crate) fn request_snapshot(&mut self) -> Result<MapSnapshot, String> {
        self.request_snapshot_command("map_snapshot")
    }

    pub(crate) fn request_screen_snapshot(&mut self) -> Result<MapSnapshot, String> {
        self.request_snapshot_command("screen_snapshot")
    }

    pub(crate) fn request_snapshot_command(&mut self, command: &str) -> Result<MapSnapshot, String> {
        let token = self
            .client_token
            .as_deref()
            .ok_or_else(|| "server attach response did not identify the client".to_owned())?
            .to_owned();
        let response = self.exchange(&format!("{command} {token}"))?;
        require_ok(&response, command)?;
        let header = response_fields(&response);
        let mut cells = BTreeMap::new();
        let mut turf_targets = BTreeMap::new();
        let mut appearances = BTreeMap::<(i32, i32, i32), Vec<Appearance>>::new();
        let mut screen = Vec::new();
        let inferred_z = header
            .get("z")
            .and_then(|value| value.parse().ok())
            .or_else(|| self.center.map(|center| center.z))
            .unwrap_or(1);
        let lines = response.lines().skip(1).collect::<Vec<_>>();
        let mut cursor = 0;
        while cursor < lines.len() {
            let fields = lines[cursor].split_ascii_whitespace().collect::<Vec<_>>();
            cursor += 1;
            if fields.first() == Some(&"S") && fields.len() == 5 {
                let (datum_index, datum_generation) = fields[1]
                    .split_once(':')
                    .and_then(|(index, generation)| {
                        Some((
                            u32::from_str_radix(index, 16).ok()?,
                            u32::from_str_radix(generation, 16).ok()?,
                        ))
                    })
                    .unwrap_or((u32::MAX, u32::MAX));
                let insertion = fields[2].parse().unwrap_or(0);
                let map_control = decode_optional_hex_text(fields[3]);
                let screen_loc = decode_optional_hex_text(fields[4]).unwrap_or_default();
                // Protocol-3 recordings made before the screen-loc repair
                // mistook named TOP:/BOTTOM: axes for map-control prefixes.
                // Reconstitute those rows so existing offline captures render
                // with the same coordinates as a newly connected client.
                let (map_control, screen_loc) = normalize_screen_selector(map_control, screen_loc);
                let type_path = lines
                    .get(cursor)
                    .and_then(|line| line.split_ascii_whitespace().nth(2))
                    .and_then(decode_hex_text)
                    .unwrap_or_default();
                let mut flattened = Vec::new();
                parse_appearance_tree(&lines, &mut cursor, &mut flattened)?;
                screen.push(ScreenAppearance {
                    datum_index,
                    datum_generation,
                    map_control,
                    screen_loc,
                    type_path,
                    insertion,
                    appearances: flattened,
                });
                continue;
            }
            if fields.first() != Some(&"T") || !matches!(fields.len(), 6 | 7) {
                continue;
            }
            let appearance_field = fields.len() - 1;
            let (Ok(x), Ok(y), Ok(appearance_count)) = (
                fields[1].parse::<i32>(),
                fields[2].parse::<i32>(),
                fields[appearance_field].parse::<usize>(),
            ) else {
                continue;
            };
            let path = decode_hex_text(fields[3]).unwrap_or_else(|| "/turf".to_owned());
            let color = decode_optional_hex_text(fields[4])
                .as_deref()
                .and_then(parse_snapshot_color)
                .unwrap_or_else(|| path_color(&path));
            cells.insert((x, y, inferred_z), color);
            let mut flattened = Vec::new();
            for appearance_index in 0..appearance_count {
                if appearance_index == 0
                    && let Some(identity) =
                        lines.get(cursor).and_then(|line| appearance_identity(line))
                {
                    turf_targets.insert((x, y, inferred_z), identity);
                }
                parse_appearance_tree(&lines, &mut cursor, &mut flattened)?;
            }
            appearances.insert((x, y, inferred_z), flattened);
        }
        let center = match (
            header.get("x").and_then(|value| value.parse().ok()),
            header.get("y").and_then(|value| value.parse().ok()),
        ) {
            (Some(x), Some(y)) => WorldCoordinate {
                x,
                y,
                z: inferred_z,
            },
            _ => self.center.unwrap_or_else(|| {
                let max_x = cells.keys().map(|(x, _, _)| *x).max().unwrap_or(1);
                let max_y = cells.keys().map(|(_, y, _)| *y).max().unwrap_or(1);
                WorldCoordinate {
                    x: (max_x + 1) / 2,
                    y: (max_y + 1) / 2,
                    z: inferred_z,
                }
            }),
        };
        self.center = Some(center);
        let mut resources = BTreeMap::new();
        let paths = appearances
            .iter()
            .filter(|((x, y, z), _)| {
                // A maximized half-window viewport is roughly 30x32 tiles at
                // the Monk skin's native 32px scale. Cache the complete visible
                // neighborhood, not the old 21x15 development fixture radius.
                *z == center.z && (*x - center.x).abs() <= 32 && (*y - center.y).abs() <= 32
            })
            .flat_map(|(_, appearances)| appearances)
            .filter(|appearance| !appearance.resource.as_os_str().is_empty())
            .map(|appearance| appearance.resource.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let paths = paths
            .into_iter()
            .chain(
                screen
                    .iter()
                    .flat_map(|screen| &screen.appearances)
                    .filter(|appearance| !appearance.resource.as_os_str().is_empty())
                    .map(|appearance| appearance.resource.clone()),
            )
            .collect::<std::collections::BTreeSet<_>>();
        if paths.len() > MAX_SNAPSHOT_RESOURCES {
            return Err(format!(
                "snapshot advertises {} resources; limit is {MAX_SNAPSHOT_RESOURCES}",
                paths.len()
            ));
        }
        let mut resource_budget = SnapshotResourceBudget::default();
        for path in paths {
            let path_text = path.to_string_lossy();
            let data = self.request_resource(&path_text)?;
            resource_budget.reserve(data.len()).map_err(str::to_owned)?;
            resources.insert(path, data);
        }
        Ok(MapSnapshot {
            center,
            cells,
            turf_targets,
            appearances,
            screen,
            resources,
        })
    }

    pub(crate) fn send_movement(&mut self, dx: i32, dy: i32) -> Result<Option<MapSnapshot>, String> {
        if self.is_replay() {
            return Ok(None);
        }
        let token = self
            .client_token
            .as_deref()
            .ok_or_else(|| "server attach response did not identify the client".to_owned())?;
        let center = self.center.ok_or_else(|| {
            "server attach response did not include client coordinates".to_owned()
        })?;
        let direction = match (dx, dy) {
            (0, 1) => "north",
            (0, -1) => "south",
            (1, 0) => "east",
            (-1, 0) => "west",
            _ => return Err("client movement must be cardinal".to_owned()),
        };
        let command = format!("move {token} {direction}");
        let response = self.exchange(&command)?;
        require_ok(&response, "move")?;
        self.center = coordinate_fields(&response_fields(&response)).or(Some(WorldCoordinate {
            x: center.x + dx,
            y: center.y + dy,
            z: center.z,
        }));
        self.request_snapshot().map(Some)
    }

    pub(crate) fn poll_ui_events(&mut self) -> Result<Vec<(u64, InboundUiCommand)>, String> {
        let token = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?
            .to_owned();
        let response = self.exchange(&format!("ui_events {token}"))?;
        require_ok(&response, "ui_events")?;
        response.lines().skip(1).map(parse_ui_event).collect()
    }

    pub(crate) fn acknowledge_ui(&mut self, sequence: u64) -> Result<(), String> {
        let token = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?
            .to_owned();
        let ProtocolStream::Live(stream) = &mut self.stream else {
            // ACK is transport reliability metadata, not a user-visible replay
            // operation. Keeping it out of recordings preserves compatibility
            // with protocol-2 captures and prevents replay cursors from being
            // coupled to presentation timing.
            return Ok(());
        };
        write_frame(stream, format!("ui_ack {token} {sequence}").as_bytes())
            .map_err(|error| error.to_string())?;
        let response = String::from_utf8(read_frame(stream).map_err(|error| error.to_string())?)
            .map_err(|_| "server response is not UTF-8".to_owned())?;
        require_ok(&response, "ui_ack")
    }

    pub(crate) fn send_readiness(&mut self, command: &str) -> Result<(), String> {
        let token = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?
            .to_owned();
        let ProtocolStream::Live(stream) = &mut self.stream else {
            return Ok(());
        };
        write_frame(stream, format!("{command} {token}").as_bytes())
            .map_err(|error| error.to_string())?;
        let response = String::from_utf8(read_frame(stream).map_err(|error| error.to_string())?)
            .map_err(|_| "server response is not UTF-8".to_owned())?;
        require_ok(&response, command)
    }

    pub(crate) fn exchange(&mut self, command: &str) -> Result<String, String> {
        let response = match &mut self.stream {
            ProtocolStream::Live(stream) => {
                write_frame(stream, command.as_bytes()).map_err(|error| error.to_string())?;
                let payload = read_frame(stream).map_err(|error| error.to_string())?;
                String::from_utf8(payload).map_err(|_| "server response is not UTF-8".to_owned())?
            }
            ProtocolStream::Replay(reader) => reader.exchange(command)?,
        };
        if let Some(recorder) = &mut self.recorder {
            recorder.record(command, &response)?;
        }
        Ok(response)
    }
}

pub(crate) fn parse_ui_event(line: &str) -> Result<(u64, InboundUiCommand), String> {
    let fields = line.split_ascii_whitespace().collect::<Vec<_>>();
    if fields.first() != Some(&"U") || fields.len() < 4 {
        return Err(format!("invalid UI event row: {line}"));
    }
    let sequence = fields[1].parse().map_err(|_| "invalid UI event sequence")?;
    let text = |index| {
        (fields[index] == "-")
            .then(String::new)
            .or_else(|| decode_hex_text(fields[index]))
            .ok_or_else(|| "invalid UI event text".to_owned())
    };
    let command = match (fields[2], fields.len()) {
        ("link", 4) => InboundUiCommand::Link { url: text(3)? },
        ("winset", 5) => InboundUiCommand::WinSet {
            control: text(3)?,
            parameters: text(4)?,
        },
        ("output", 5) => InboundUiCommand::Output {
            control: (fields[3] != "-").then(|| text(3)).transpose()?,
            message: text(4)?,
        },
        ("browse_resource", 5) => InboundUiCommand::BrowseResource {
            name: text(3)?,
            data: if fields[4] == "-" {
                Vec::new()
            } else {
                decode_hex(fields[4]).ok_or("invalid browse resource bytes")?
            },
        },
        ("browse", 5) => InboundUiCommand::Browse {
            control: text(3)?,
            html: text(4)?,
        },
        ("prompt", 10) => {
            let kind = match fields[4] {
                "text" => ClientPromptKind::Text,
                "message" => ClientPromptKind::Message,
                "number" => ClientPromptKind::Number,
                "color" => ClientPromptKind::Color,
                "file" => ClientPromptKind::File,
                "list" => ClientPromptKind::List,
                "alert" => ClientPromptKind::Alert,
                _ => return Err("invalid prompt kind".to_owned()),
            };
            let choices = if fields[9] == "-" {
                Vec::new()
            } else {
                fields[9]
                    .split(',')
                    .map(|choice| decode_hex_text(choice).ok_or("invalid prompt choice".to_owned()))
                    .collect::<Result<Vec<_>, _>>()?
            };
            InboundUiCommand::Prompt(ClientPrompt {
                id: fields[3].parse().map_err(|_| "invalid prompt id")?,
                kind,
                can_cancel: match fields[5] {
                    "0" => false,
                    "1" => true,
                    _ => return Err("invalid prompt cancellation flag".to_owned()),
                },
                title: text(6)?,
                message: text(7)?,
                default: text(8)?,
                choices,
                edit: String::new(),
                selected: 0,
            })
        }
        ("sound", 9) => InboundUiCommand::Sound(SoundUpdate {
            channel: fields[3].parse().map_err(|_| "invalid sound channel")?,
            repeat: match fields[4] {
                "0" => false,
                "1" => true,
                _ => return Err("invalid sound repeat flag".to_owned()),
            },
            volume: fields[5].parse().map_err(|_| "invalid sound volume")?,
            frequency: fields[6].parse().map_err(|_| "invalid sound frequency")?,
            pan: fields[7].parse().map_err(|_| "invalid sound pan")?,
            file: (fields[8] != "-").then(|| text(8)).transpose()?,
        }),
        _ => return Err(format!("unsupported UI event row: {line}")),
    };
    Ok((sequence, command))
}

pub(crate) fn parse_byond_url(url: &str) -> Result<(String, BTreeMap<String, String>), String> {
    let url = url
        .strip_prefix("byond://")
        .ok_or("browser call is not a byond:// URL")?;
    let (path, query) = url.split_once('?').unwrap_or((url, ""));
    let mut parameters = BTreeMap::new();
    for field in query.split('&').filter(|field| !field.is_empty()) {
        let (name, value) = field.split_once('=').unwrap_or((field, ""));
        parameters.insert(percent_decode_form(name)?, percent_decode_form(value)?);
    }
    Ok((path.trim_matches('/').to_ascii_lowercase(), parameters))
}

pub(crate) fn percent_decode_form(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut cursor = 0;
    while cursor < bytes.len() {
        match bytes[cursor] {
            b'+' => {
                decoded.push(b' ');
                cursor += 1;
            }
            b'%' if cursor + 2 < bytes.len() => {
                let high = hex_nibble(bytes[cursor + 1]).ok_or("invalid URL escape")?;
                let low = hex_nibble(bytes[cursor + 2]).ok_or("invalid URL escape")?;
                decoded.push((high << 4) | low);
                cursor += 3;
            }
            b'%' => return Err("truncated URL escape".to_owned()),
            byte => {
                decoded.push(byte);
                cursor += 1;
            }
        }
    }
    String::from_utf8(decoded).map_err(|_| "browser URL is not UTF-8".to_owned())
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

pub(crate) fn quote_dmf_assignment(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

pub(crate) fn valid_byond_callback(callback: &str) -> bool {
    let indexed_callback = callback
        .strip_prefix("Byond.__callbacks__[")
        .and_then(|value| value.strip_suffix(']'))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()));
    indexed_callback
        || callback.split('.').all(|segment| {
            let mut bytes = segment.bytes();
            bytes
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || matches!(byte, b'_' | b'$'))
                && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$'))
        })
}

pub(crate) fn write_frame(stream: &mut impl Write, payload: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(payload)
}

pub(crate) fn read_frame(stream: &mut impl Read) -> std::io::Result<Vec<u8>> {
    // A dense 255x255 Z-level is several MiB even in the compact tabular
    // encoding. Requests remain tiny, but snapshot replies need a bounded
    // ceiling comfortably above a production map plane.
    const MAX_FRAME: usize = 64 * 1024 * 1024;
    let mut header = [0; 4];
    stream.read_exact(&mut header)?;
    let len = u32::from_be_bytes(header) as usize;
    if len > MAX_FRAME {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "server frame exceeds 64 MiB",
        ));
    }
    let mut payload = vec![0; len];
    stream.read_exact(&mut payload)?;
    Ok(payload)
}

fn require_ok(response: &str, operation: &str) -> Result<(), String> {
    response
        .starts_with("ok ")
        .then_some(())
        .ok_or_else(|| format!("{operation} rejected: {response}"))
}

fn response_fields(response: &str) -> BTreeMap<String, String> {
    response
        .lines()
        .next()
        .into_iter()
        .flat_map(str::split_ascii_whitespace)
        .filter_map(|part| part.split_once('='))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}

fn startup_status_from_error(error: &str) -> Option<String> {
    let phase = error
        .split_ascii_whitespace()
        .find_map(|field| field.strip_prefix("phasehex="))?;
    let phase = String::from_utf8(decode_hex(phase)?).ok()?;
    Some(format!("Server boot: {phase}"))
}

fn coordinate_fields(fields: &BTreeMap<String, String>) -> Option<WorldCoordinate> {
    Some(WorldCoordinate {
        x: fields.get("x")?.parse().ok()?,
        y: fields.get("y")?.parse().ok()?,
        z: fields.get("z")?.parse().ok()?,
    })
}

fn path_color(path: &str) -> u32 {
    let hash = path.bytes().fold(0_u32, |hash, byte| {
        hash.wrapping_mul(33).wrapping_add(u32::from(byte))
    });
    0xff00_0000 | (0x38 + (hash & 0x3f)) << 16 | (0x38 + ((hash >> 6) & 0x3f)) << 8 | 0x55
}

fn parse_snapshot_color(value: &str) -> Option<u32> {
    let value = value.trim();
    if value.is_empty() || value == "-" || value.eq_ignore_ascii_case("null") {
        return None;
    }
    let hex = value.strip_prefix('#')?;
    match hex.len() {
        6 => u32::from_str_radix(hex, 16)
            .ok()
            .map(|rgb| 0xff00_0000 | rgb),
        8 => u32::from_str_radix(hex, 16).ok(),
        _ => None,
    }
}

pub(crate) fn parse_appearance_tree(
    lines: &[&str],
    cursor: &mut usize,
    output: &mut Vec<Appearance>,
) -> Result<(), String> {
    parse_appearance_tree_for(lines, cursor, output, None)
}

fn appearance_identity(line: &str) -> Option<(u32, u32)> {
    let mut fields = line.split_ascii_whitespace();
    (fields.next()? == "A").then_some(())?;
    let (index, generation) = fields.next()?.split_once(':')?;
    Some((
        u32::from_str_radix(index, 16).ok()?,
        u32::from_str_radix(generation, 16).ok()?,
    ))
}

fn parse_appearance_tree_for(
    lines: &[&str],
    cursor: &mut usize,
    output: &mut Vec<Appearance>,
    root: Option<(u32, u32)>,
) -> Result<(), String> {
    let line = lines.get(*cursor).ok_or("truncated appearance tree")?;
    *cursor += 1;
    let mut fields = line.split_ascii_whitespace().collect::<Vec<_>>();
    // Protocol 2 originally rendered Some("") as an empty whitespace token.
    // Live servers using that first codec therefore omit the icon-state column.
    // Normalize those rows while the explicit `~` representation rolls out.
    if fields.first() == Some(&"A") && fields.len() == 15 {
        fields.insert(4, "~");
    }
    if fields.first() != Some(&"A") || !matches!(fields.len(), 16 | 21 | 23) {
        return Err("invalid appearance row".to_owned());
    }
    let identity = fields[1]
        .split_once(':')
        .and_then(|(index, generation)| {
            Some((
                u32::from_str_radix(index, 16).ok()?,
                u32::from_str_radix(generation, 16).ok()?,
            ))
        })
        .ok_or("invalid appearance identity")?;
    let root = root.unwrap_or(identity);
    let underlays = fields[14]
        .parse::<usize>()
        .map_err(|_| "invalid underlay count")?;
    let overlays = fields[15]
        .parse::<usize>()
        .map_err(|_| "invalid overlay count")?;
    for _ in 0..underlays {
        parse_appearance_tree_for(lines, cursor, output, Some(root))?;
    }
    let icon = decode_optional_hex_text(fields[3]).filter(|icon| !icon.is_empty());
    let maptext = fields
        .get(16)
        .and_then(|value| decode_optional_hex_text(value))
        .filter(|value| !value.is_empty());
    if icon.is_some() || maptext.is_some() {
        let numeric = |index| u32::from_str_radix(fields[index], 16).map(f32::from_bits);
        let color = decode_optional_hex_text(fields[12])
            .as_deref()
            .and_then(parse_snapshot_color)
            .map(|argb| [(argb >> 16) as u8, (argb >> 8) as u8, argb as u8])
            .unwrap_or([255; 3]);
        output.push(Appearance {
            datum_index: root.0,
            datum_generation: root.1,
            resource: icon.map_or_else(PathBuf::new, PathBuf::from),
            state: decode_optional_hex_text(fields[4]).unwrap_or_default(),
            direction: fields[5].parse().unwrap_or(2),
            frame: 1,
            layer: numeric(6).unwrap_or(0.0),
            plane: numeric(7).unwrap_or(0.0),
            appearance_flags: fields
                .get(21)
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
            mouse_opacity: fields
                .get(22)
                .and_then(|value| value.parse().ok())
                .unwrap_or(1),
            pixel_x: numeric(8).unwrap_or(0.0).round() as i32,
            pixel_y: numeric(9).unwrap_or(0.0).round() as i32,
            color,
            alpha: numeric(13).unwrap_or(255.0).clamp(0.0, 255.0).round() as u8,
            maptext,
            maptext_width: fields
                .get(17)
                .and_then(|value| u32::from_str_radix(value, 16).ok())
                .map(f32::from_bits)
                .unwrap_or(0.0)
                .round() as i32,
            maptext_height: fields
                .get(18)
                .and_then(|value| u32::from_str_radix(value, 16).ok())
                .map(f32::from_bits)
                .unwrap_or(0.0)
                .round() as i32,
            maptext_x: fields
                .get(19)
                .and_then(|value| u32::from_str_radix(value, 16).ok())
                .map(f32::from_bits)
                .unwrap_or(0.0)
                .round() as i32,
            maptext_y: fields
                .get(20)
                .and_then(|value| u32::from_str_radix(value, 16).ok())
                .map(f32::from_bits)
                .unwrap_or(0.0)
                .round() as i32,
        });
    }
    for _ in 0..overlays {
        parse_appearance_tree_for(lines, cursor, output, Some(root))?;
    }
    Ok(())
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    value
        .len()
        .is_multiple_of(2)
        .then(|| {
            (0..value.len())
                .step_by(2)
                .map(|index| u8::from_str_radix(&value[index..index + 2], 16).ok())
                .collect::<Option<Vec<_>>>()
        })
        .flatten()
}

pub(crate) fn decode_hex_bounded(value: &str, max_bytes: usize) -> Result<Vec<u8>, &'static str> {
    if !value.len().is_multiple_of(2) {
        return Err("hex payload has odd length");
    }
    let decoded_len = value.len() / 2;
    if decoded_len > max_bytes {
        return Err("payload exceeds resource byte limit");
    }
    decode_hex(value).ok_or("hex payload contains an invalid digit")
}

fn decode_hex_text(value: &str) -> Option<String> {
    String::from_utf8(decode_hex(value)?).ok()
}

fn decode_optional_hex_text(value: &str) -> Option<String> {
    match value {
        "-" => None,
        "~" => Some(String::new()),
        _ => decode_hex_text(value),
    }
}

pub(crate) fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

impl LoopbackTransport {
    pub(crate) fn new(scene: WorldScene) -> Self {
        Self {
            scene,
            attached: None,
        }
    }

    pub(crate) fn attach(&mut self, runtime: &mut ExecutionState, client: DatumId) -> Result<(), String> {
        self.scene.attach_local_player(runtime, client)?;
        self.attached = Some(client);
        Ok(())
    }

    pub(crate) fn request_snapshot(&self, radius_x: i32, radius_y: i32) -> MapSnapshot {
        let center = self.scene.player;
        let cells = self
            .scene
            .cells
            .iter()
            .filter(|((x, y, z), _)| {
                *z == center.z
                    && (*x - center.x).abs() <= radius_x
                    && (*y - center.y).abs() <= radius_y
            })
            .map(|(coordinate, color)| (*coordinate, *color))
            .collect();
        MapSnapshot {
            center,
            cells,
            turf_targets: self
                .scene
                .turfs
                .iter()
                .map(|(coordinate, datum)| (*coordinate, (datum.index(), datum.generation())))
                .collect(),
            appearances: BTreeMap::new(),
            screen: Vec::new(),
            resources: BTreeMap::new(),
        }
    }

    pub(crate) fn send_movement(
        &mut self,
        runtime: &mut ExecutionState,
        dx: i32,
        dy: i32,
    ) -> Option<MapSnapshot> {
        self.attached?;
        self.scene
            .move_player(runtime, dx, dy)
            .then(|| self.request_snapshot(10, 7))
    }

    pub(crate) fn label(&self) -> &str {
        &self.scene.label
    }
}

impl WorldScene {
    pub(crate) fn boot(world: &Path, requested_map: Option<&Path>) -> Result<(ExecutionState, Self), String> {
        eprintln!("client-world: compiling {}", world.display());
        let compilation = load_compilation(world)?;
        let map_path = requested_map
            .ok_or_else(|| "a real-world client launch requires --map <.dmm>".to_owned())?;
        let source = std::fs::read_to_string(map_path)
            .map_err(|error| format!("{}: {error}", map_path.display()))?;
        eprintln!("client-world: parsing {}", map_path.display());
        let map =
            dm_map::parse(&source).map_err(|error| format!("{}: {error}", map_path.display()))?;
        let plan = dm_world::build_plan(&map, &compilation);
        let mut cells = BTreeMap::new();
        for cell in plan.cells() {
            let color = plan.template(&cell.key).map_or(0xff253444, tile_color);
            cells.insert(
                (cell.coordinate.x, cell.coordinate.y, cell.coordinate.z),
                color,
            );
        }
        let mut image = RuntimeImage::from_compilation(&compilation)
            .map_err(|error| format!("runtime image: {error}"))?;
        let allocation = dm_world::allocate_world(&plan, &mut image)
            .map_err(|error| format!("world allocation: {error}"))?;
        let mut turfs = BTreeMap::new();
        for snapshot in allocation.snapshots() {
            if let Some(turf) = snapshot.turf {
                turfs.insert(
                    (
                        snapshot.coordinate.x,
                        snapshot.coordinate.y,
                        snapshot.coordinate.z,
                    ),
                    turf,
                );
            }
        }
        let player = turfs
            .keys()
            .next()
            .map(|&(x, y, z)| WorldCoordinate { x, y, z })
            .ok_or_else(|| format!("{} contains no allocated turfs", map_path.display()))?;
        let mut runtime = image.take_execution_state();
        let player_mob = image
            .allocate_datum_in_state(
                &TypePath::parse("/mob").expect("engine /mob path is valid"),
                &mut runtime,
            )
            .map_err(|error| format!("local player allocation: {error}"))?;
        Ok((
            runtime,
            Self {
                cells,
                turfs,
                player,
                player_mob,
                label: map_path.display().to_string(),
            },
        ))
    }

    pub(crate) fn attach_local_player(
        &mut self,
        runtime: &mut ExecutionState,
        client: DatumId,
    ) -> Result<(), String> {
        runtime
            .heap_mut()
            .set_datum_field(
                client,
                FieldName::parse("mob").expect("engine mob field is valid"),
                Value::Datum(self.player_mob),
            )
            .map_err(|error| error.to_string())?;
        self.sync_player_fields(runtime)
    }

    pub(crate) fn move_player(&mut self, runtime: &mut ExecutionState, dx: i32, dy: i32) -> bool {
        let next = WorldCoordinate {
            x: self.player.x + dx,
            y: self.player.y + dy,
            z: self.player.z,
        };
        if self.cells.contains_key(&(next.x, next.y, next.z)) {
            self.player = next;
            self.sync_player_fields(runtime).is_ok()
        } else {
            false
        }
    }

    pub(crate) fn sync_player_fields(&mut self, runtime: &mut ExecutionState) -> Result<(), String> {
        let turf = self.turf_for_player();
        let heap = runtime.heap_mut();
        for (field, coordinate) in [
            ("x", self.player.x),
            ("y", self.player.y),
            ("z", self.player.z),
        ] {
            heap.set_datum_field(
                self.player_mob,
                FieldName::parse(field).expect("engine coordinate field is valid"),
                Value::number(coordinate as f32),
            )
            .map_err(|error| error.to_string())?;
        }
        heap.set_datum_field(
            self.player_mob,
            FieldName::parse("loc").expect("engine loc field is valid"),
            Value::Datum(turf),
        )
        .map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(crate) fn turf_for_player(&self) -> DatumId {
        let coordinate = (self.player.x, self.player.y, self.player.z);
        *self
            .turfs
            .get(&coordinate)
            .expect("local player only moves into allocated map turfs")
    }
}

fn load_compilation(world: &Path) -> Result<dm_compiler::Compilation, String> {
    let project = Project::load(world).map_err(|error| format!("project: {error}"))?;
    let artifact_path = world.with_extension("d64");
    if artifact_path.is_file() {
        let fingerprint = *project.content_fingerprint().as_bytes();
        if let Ok(artifact) = CompiledArtifact::read_from(&artifact_path, fingerprint) {
            if let Some(section) = artifact.section(1) {
                eprintln!("client-world: decoding {}", artifact_path.display());
                return dm_compiler::Compilation::decode_compiled_artifact(section.payload())
                    .map_err(|error| format!("compiled artifact: {error}"));
            }
        }
    }
    eprintln!("client-world: compiling {}", world.display());
    CompilerDatabase::new()
        .compile(world)
        .map_err(|error| format!("{error:?}"))
}

fn tile_color(template: &dm_world::CellTemplate) -> u32 {
    let mut hash = 0_u32;
    for initializer in &template.initializers {
        for byte in initializer.path.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(u32::from(byte));
        }
    }
    let movable = template.initializers.iter().any(|initializer| {
        matches!(
            initializer.resolution,
            dm_world::InitializerResolution::Resolved {
                category: AtomCategory::Movable,
                ..
            }
        )
    });
    let base = if movable { 0x70 } else { 0x38 };
    0xff000000
        | ((base + (hash & 0x1f)) << 16)
        | ((base + ((hash >> 5) & 0x1f)) << 8)
        | (base + ((hash >> 10) & 0x1f))
}

fn named_screen_prefix(prefix: &str) -> bool {
    matches!(
        prefix.trim().to_ascii_uppercase().as_str(),
        "TOP" | "BOTTOM" | "NORTH" | "SOUTH" | "LEFT" | "RIGHT" | "EAST" | "WEST" | "CENTER"
    )
}

pub(crate) fn normalize_screen_selector(
    map_control: Option<String>,
    screen_loc: String,
) -> (Option<String>, String) {
    if map_control.as_deref().is_some_and(named_screen_prefix) {
        let prefix = map_control.expect("checked map control exists");
        (None, format!("{prefix}:{screen_loc}"))
    } else {
        (map_control, screen_loc)
    }
}
