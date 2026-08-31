//! The in-process BYOND client model: connected sessions, map/appearance
//! transport snapshots, and the modal prompt/verb continuation machinery.
//!
//! The engine executes DM with no external display. A *local client* is the
//! authoritative in-VM session a host (`dm-lifecycle`) can attach, move,
//! receive map snapshots from, and answer prompts for. Prompt continuations
//! suspend the calling DM frame and re-schedule it when the host answers.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use crate::builtins;
use crate::bytecode::{Module, VerbParameterType};
use crate::value_ops::ExecutionContext;
use crate::{
    CallFrame, ExecutionState, HeapReference, OwnedContinuation, allocate_initialized_datum,
    assign_datum_field, datum_field_or_initial, dynamic_call_target_named, is_atom_type_path,
    make_frame, parse_heap_reference, schedule_frames,
};
use dm_dmf::{
    ClientSession, ControlTree, Diagnostic, DiagnosticSeverity, UiEvent, parse as parse_dmf,
};
use dm_value::{DatumId, FieldName, TypePath, Value};

/// A local client could not be created from the supplied DMF skin.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalClientError {
    /// DMF diagnostics that prevented session creation.
    pub diagnostics: Vec<Diagnostic>,
}

/// One cardinal movement requested by a locally attached client.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalMovementDirection {
    /// Increase Y by one.
    North,
    /// Decrease Y by one.
    South,
    /// Increase X by one.
    East,
    /// Decrease X by one.
    West,
}

/// Authoritative location of one locally attached client and mob.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalClientState {
    /// Client datum identity.
    pub client: DatumId,
    /// Controlled mob identity.
    pub mob: DatumId,
    /// Current turf X coordinate.
    pub x: i32,
    /// Current turf Y coordinate.
    pub y: i32,
    /// Current turf Z coordinate.
    pub z: i32,
}

/// One authoritative UI operation emitted by DM for a connected local client.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalClientUiEvent {
    /// Open a URL requested by BYOND's `link()` builtin.
    Link {
        /// Absolute URL supplied by game code.
        url: String,
    },
    /// Mutate named DMF control properties.
    Winset {
        /// Target DMF control address.
        control: String,
        /// BYOND winset parameter string.
        parameters: String,
    },
    /// Append a message to an output control.
    Output {
        /// Target output control address.
        control: String,
        /// Text appended to the control.
        message: String,
    },
    /// Register a browser-visible resource under a logical name.
    BrowseResource {
        /// Browser-visible logical resource name.
        name: String,
        /// Complete resource payload.
        bytes: Vec<u8>,
    },
    /// Display HTML in a browser control.
    Browse {
        /// BYOND browser window/control selector.
        window: String,
        /// HTML document body.
        html: String,
    },
    /// Display a modal prompt and suspend the calling DM continuation.
    Prompt {
        /// Stable response token scoped to the connected client.
        id: u64,
        /// Native prompt presentation and response conversion.
        kind: LocalClientPromptKind,
        /// Window caption.
        title: String,
        /// Prompt body.
        message: String,
        /// Initial editable value or selected button.
        default: String,
        /// Alert buttons or list-picker display values.
        choices: Vec<String>,
        /// Whether closing the prompt may yield null.
        can_cancel: bool,
    },
    /// Play, replace, or stop one BYOND sound channel.
    Sound {
        /// Project-relative audio resource; `None` stops the channel.
        file: Option<String>,
        /// BYOND channel number, where zero is fire-and-forget.
        channel: i32,
        /// Whether playback loops.
        repeat: bool,
        /// Volume percentage.
        volume: f32,
        /// Requested playback frequency.
        frequency: f32,
        /// Stereo pan from -100 through 100.
        pan: f32,
    },
}

/// Native presentation class for one local-client prompt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalClientPromptKind {
    /// Single-line text or command text.
    Text,
    /// Multi-line message text.
    Message,
    /// Floating-point number.
    Number,
    /// Color text such as `#rrggbb`.
    Color,
    /// Project-relative file/icon/sound path.
    File,
    /// One of a fixed set of values.
    List,
    /// One of one through three alert buttons.
    Alert,
}

/// Typed answer supplied by the native client for a pending prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum LocalClientPromptResponse {
    /// User cancelled a nullable prompt.
    Null,
    /// Text, message, or color response.
    Text(String),
    /// Numeric response.
    Number(f32),
    /// Zero-based alert/list choice index.
    Choice(usize),
}

/// One stable map cell copied out of the runtime heap.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientMapTile {
    /// Turf X coordinate.
    pub x: i32,
    /// Turf Y coordinate.
    pub y: i32,
    /// Canonical turf type path.
    pub type_path: String,
    /// DM-visible color converted to stable text when present.
    pub color: Option<String>,
    /// Materialized atom identities currently contained by this turf.
    pub occupants: Vec<DatumId>,
    /// Turf and contained atoms in stable plane/layer/insertion draw order.
    pub appearances: Vec<LocalClientAppearance>,
}

/// Owned DM appearance data required by the local renderer.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientAppearance {
    /// Source atom or appearance datum when one exists.
    pub datum: DatumId,
    /// Canonical runtime type path.
    pub type_path: String,
    /// Backing DMI/resource path after unwrapping `/icon` objects.
    pub icon: Option<String>,
    /// Selected icon state.
    pub icon_state: Option<String>,
    /// BYOND direction bitfield.
    pub dir: i32,
    /// Draw layer.
    pub layer: f32,
    /// Draw plane.
    pub plane: f32,
    /// BYOND appearance behavior bitfield, kept at the VM's narrow integer width.
    pub appearance_flags: i32,
    /// BYOND mouse hit-test policy value.
    pub mouse_opacity: i32,
    /// X pixel offset.
    pub pixel_x: f32,
    /// Y pixel offset.
    pub pixel_y: f32,
    /// W pixel offset.
    pub pixel_w: f32,
    /// Z pixel offset.
    pub pixel_z: f32,
    /// DM color represented as stable transport text.
    pub color: Option<String>,
    /// Alpha in BYOND's 0 through 255 range.
    pub alpha: f32,
    /// HTML-like BYOND maptext attached to this appearance.
    pub maptext: Option<String>,
    /// Maptext box width in pixels.
    pub maptext_width: f32,
    /// Maptext box height in pixels.
    pub maptext_height: f32,
    /// Maptext X offset in pixels.
    pub maptext_x: f32,
    /// Maptext Y offset in pixels.
    pub maptext_y: f32,
    /// Nested underlays in stable plane/layer/insertion order.
    pub underlays: Vec<LocalClientAppearance>,
    /// Nested overlays in stable plane/layer/insertion order.
    pub overlays: Vec<LocalClientAppearance>,
}

/// Owned map snapshot suitable for transport to a local client.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientMapSnapshot {
    /// Maximum X coordinate represented by the world at this Z level.
    pub width: i32,
    /// Maximum Y coordinate represented by the world at this Z level.
    pub height: i32,
    /// Selected Z level.
    pub z: i32,
    /// Tiles in deterministic Y-then-X world index order.
    pub tiles: Vec<LocalClientMapTile>,
    /// HUD/screen atoms from the attached client's `screen` list.
    pub screen: Vec<LocalClientScreenAppearance>,
}

/// One client-screen appearance and its BYOND screen-space selector.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalClientScreenAppearance {
    /// Optional DMF map-control selector prefix.
    pub map_control: Option<String>,
    /// BYOND `screen_loc` expression used for viewport placement.
    pub screen_loc: String,
    /// Stable insertion position in `client.screen`.
    pub insertion: usize,
    /// Fully expanded appearance tree.
    pub appearance: LocalClientAppearance,
}

/// Mouse transition delivered to a client-owned screen atom.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalScreenPointerEvent {
    /// Pointer entered the atom's visible pixels.
    Entered,
    /// Pointer left the atom's visible pixels.
    Exited,
    /// Primary pointer button activated the atom.
    Click,
}

/// Mutable per-world client session state.
///
/// Extracted from `ExecutionState` to establish a clear subsystem boundary
/// for the in-process BYOND client model.
#[derive(Debug, Default)]
pub struct ClientState {
    /// Connected client sessions keyed by client datum.
    pub client_sessions: BTreeMap<DatumId, ClientSession>,
    /// Clients with interactive input focus.
    pub interactive_local_clients: HashSet<DatumId>,
    /// Active DMF skin for the local client.
    pub local_client_skin: Option<ControlTree>,
    /// Outbound UI events queued for each client.
    pub local_client_outbound_events: BTreeMap<DatumId, Vec<LocalClientUiEvent>>,
    /// Client -> controlled mob mapping.
    pub local_client_mobs: BTreeMap<DatumId, DatumId>,
    /// Queued movement commands awaiting scheduler application.
    pub local_client_commands: Vec<(u64, DatumId, LocalMovementDirection)>,
    /// Monotonic sequence for client commands.
    pub local_client_command_sequence: u64,
    /// Monotonic sequence for guest clients.
    pub local_guest_sequence: u64,
    /// Monotonic sequence for modal prompts.
    pub local_prompt_sequence: u64,
    /// Suspended prompt continuations awaiting host response.
    pub(crate) pending_local_prompts: BTreeMap<u64, PendingLocalPrompt>,
}

impl ClientState {
    /// Creates a new empty client state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether a client has an active session.
    pub fn has_session(&self, client: DatumId) -> bool {
        self.client_sessions.contains_key(&client)
    }

    /// Installs a client session with its control tree.
    pub fn install_session(&mut self, client: DatumId, tree: ControlTree) {
        self.client_sessions
            .entry(client)
            .and_modify(|session| *session = ClientSession::new(tree.clone()))
            .or_insert_with(|| ClientSession::new(tree));
    }

    /// Sets or replaces the local client skin.
    pub fn set_skin(&mut self, tree: ControlTree) {
        self.local_client_skin = Some(tree);
    }

    /// Takes all outbound UI events, leaving the queue empty.
    pub fn take_outbound_events(&mut self) -> BTreeMap<DatumId, Vec<LocalClientUiEvent>> {
        std::mem::take(&mut self.local_client_outbound_events)
    }

    /// Emits a UI event to a connected client.
    pub fn emit_ui_event(&mut self, client: DatumId, event: LocalClientUiEvent) {
        if self.client_sessions.contains_key(&client) {
            self.local_client_outbound_events
                .entry(client)
                .or_default()
                .push(event);
        }
    }

    /// Returns the number of pending prompts.
    pub fn pending_prompt_count(&self) -> usize {
        self.pending_local_prompts.len()
    }

    /// Registers a pending prompt continuation.
    pub(crate) fn register_prompt(&mut self, id: u64, prompt: PendingLocalPrompt) {
        self.pending_local_prompts.insert(id, prompt);
    }

    /// Removes and returns a pending prompt by ID.
    pub(crate) fn take_prompt(&mut self, id: u64) -> Option<PendingLocalPrompt> {
        self.pending_local_prompts.remove(&id)
    }

    /// Returns an iterator over pending prompts.
    pub(crate) fn pending_prompts(&self) -> impl Iterator<Item = (&u64, &PendingLocalPrompt)> {
        self.pending_local_prompts.iter()
    }

    /// Sets interactive status for a client.
    pub fn set_interactive(&mut self, client: DatumId, interactive: bool) {
        if interactive {
            self.interactive_local_clients.insert(client);
        } else {
            self.interactive_local_clients.remove(&client);
        }
    }

    /// Returns whether a client is interactive.
    pub fn is_interactive(&self, client: DatumId) -> bool {
        self.interactive_local_clients.contains(&client)
    }

    /// Attaches a client to a mob.
    pub fn attach_mob(&mut self, client: DatumId, mob: DatumId) {
        self.local_client_mobs.insert(client, mob);
    }

    /// Returns the mob attached to a client.
    pub fn attached_mob(&self, client: DatumId) -> Option<DatumId> {
        self.local_client_mobs.get(&client).copied()
    }

    /// Removes a client's attached mob.
    pub fn detach_mob(&mut self, client: DatumId) -> Option<DatumId> {
        self.local_client_mobs.remove(&client)
    }

    /// Queues a movement command for a client.
    pub fn queue_command(&mut self, client: DatumId, direction: LocalMovementDirection) -> u64 {
        let sequence = self.local_client_command_sequence;
        self.local_client_command_sequence = sequence.saturating_add(1);
        self.local_client_commands
            .push((sequence, client, direction));
        sequence
    }

    /// Takes all queued commands, leaving the queue empty.
    pub fn take_commands(&mut self) -> Vec<(u64, DatumId, LocalMovementDirection)> {
        std::mem::take(&mut self.local_client_commands)
    }

    /// Generates the next prompt sequence ID.
    pub fn next_prompt_id(&mut self) -> u64 {
        self.local_prompt_sequence = self.local_prompt_sequence.saturating_add(1);
        self.local_prompt_sequence
    }

    /// Generates the next guest sequence ID.
    pub fn next_guest_id(&mut self) -> u64 {
        self.local_guest_sequence = self.local_guest_sequence.saturating_add(1);
        self.local_guest_sequence
    }

    /// Returns all client session datums for GC root tracking.
    pub fn session_datums(&self) -> impl Iterator<Item = DatumId> + '_ {
        self.client_sessions.keys().copied()
    }

    /// Returns all attached mob datums for GC root tracking.
    pub fn mob_datums(&self) -> impl Iterator<Item = DatumId> + '_ {
        self.local_client_mobs
            .keys()
            .copied()
            .chain(self.local_client_mobs.values().copied())
    }

    /// Returns all prompt client datums for GC root tracking.
    pub fn prompt_client_datums(&self) -> impl Iterator<Item = DatumId> + '_ {
        self.pending_local_prompts.values().map(|p| p.client)
    }
}

impl fmt::Display for LocalClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "DMF contains {} error diagnostic(s)",
            self.diagnostics.len()
        )
    }
}

impl std::error::Error for LocalClientError {}

#[derive(Clone, Debug, Default)]
pub(crate) struct SavefileState {
    pub(crate) entries: HashMap<String, Value>,
    pub(crate) cd: String,
}

#[derive(Clone, Debug)]
pub(crate) struct ExceptionHandler {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) catch: usize,
    pub(crate) local: Option<u16>,
    pub(crate) stack_depth: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct ScheduledSpawn {
    pub(crate) due_tick: u64,
    pub(crate) sequence: u64,
    pub(crate) frames: OwnedContinuation,
}

#[derive(Clone, Debug)]
pub(crate) struct PendingLocalPrompt {
    pub(crate) client: DatumId,
    pub(crate) kind: LocalClientPromptKind,
    pub(crate) choices: Vec<Value>,
    pub(crate) can_cancel: bool,
    pub(crate) continuation: PendingPromptContinuation,
}

#[derive(Clone, Debug)]
pub(crate) enum PendingPromptContinuation {
    Frames(Vec<CallFrame>),
    Verb(PendingVerbInvocation),
}

#[derive(Clone, Debug)]
pub(crate) struct PendingVerbInvocation {
    pub(crate) frame: CallFrame,
    pub(crate) parameter_types: Vec<VerbParameterType>,
    pub(crate) parameter_names: Vec<String>,
    pub(crate) verb_name: String,
    pub(crate) parameter: usize,
}

pub(crate) struct LocalPromptSpec {
    pub(crate) id: u64,
    pub(crate) client: DatumId,
    pub(crate) kind: LocalClientPromptKind,
    pub(crate) choices: Vec<Value>,
    pub(crate) can_cancel: bool,
    pub(crate) event: LocalClientUiEvent,
}

fn local_prompt_client(
    state: &ExecutionState,
    arguments: &[Value],
    usr: &Value,
) -> Option<DatumId> {
    arguments
        .first()
        .into_iter()
        .chain(std::iter::once(usr))
        .filter_map(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        })
        .find_map(|datum| {
            if state.client.is_interactive(datum) {
                Some(datum)
            } else {
                state
                    .client
                    .local_client_mobs
                    .iter()
                    .find_map(|(client, mob)| {
                        (*mob == datum && state.client.is_interactive(*client)).then_some(*client)
                    })
            }
        })
}

fn prompt_value_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Text(value)) => value.to_string(),
        Some(Value::Number(value)) => value.to_f32().to_string(),
        Some(value) => value.to_string(),
    }
}

pub(crate) fn local_prompt_spec(
    name: &str,
    arguments: &[Value],
    usr: &Value,
    state: &mut ExecutionState,
) -> Result<Option<LocalPromptSpec>, String> {
    let base_name = name.split_once('@').map_or(name, |(name, _)| name);
    if !matches!(base_name, "input" | "alert") {
        return Ok(None);
    }
    let Some(client) = local_prompt_client(state, arguments, usr) else {
        return Ok(None);
    };
    let explicit_usr = arguments
        .first()
        .is_some_and(|value| matches!(value, Value::Datum(_) | Value::Null));
    let base = usize::from(explicit_usr);
    let (kind, title, message, default, choices, can_cancel) = if base_name == "alert" {
        let choices = arguments
            .iter()
            .skip(base + 2)
            .filter(|value| !matches!(value, Value::Null))
            .cloned()
            .collect::<Vec<_>>();
        let choices = if choices.is_empty() {
            vec![Value::text("Ok")]
        } else {
            choices
        };
        (
            LocalClientPromptKind::Alert,
            prompt_value_text(arguments.get(base + 1)),
            prompt_value_text(arguments.get(base)),
            prompt_value_text(choices.first()),
            choices,
            false,
        )
    } else {
        let type_marker = name.split_once('@').map_or("", |(_, marker)| marker);
        let list = type_marker.split('+').any(|part| part == "list");
        let choices = if list {
            arguments
                .last()
                .and_then(|value| match value {
                    Value::List(list) => state.heap.list(*list).ok(),
                    _ => None,
                })
                .map(|values| values.positions().map(|(_, value)| value.clone()).collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let default = arguments.get(base + 2);
        let kind = if list {
            LocalClientPromptKind::List
        } else if type_marker.split('+').any(|part| part == "num")
            || (type_marker.is_empty() && matches!(default, Some(Value::Number(_))))
        {
            LocalClientPromptKind::Number
        } else if type_marker.split('+').any(|part| part == "message") {
            LocalClientPromptKind::Message
        } else if type_marker.split('+').any(|part| part == "color") {
            LocalClientPromptKind::Color
        } else if type_marker
            .split('+')
            .any(|part| matches!(part, "file" | "icon" | "sound"))
        {
            LocalClientPromptKind::File
        } else {
            LocalClientPromptKind::Text
        };
        (
            kind,
            prompt_value_text(arguments.get(base + 1)),
            prompt_value_text(arguments.get(base)),
            prompt_value_text(default),
            choices,
            type_marker.split('+').any(|part| part == "null"),
        )
    };
    let id = state.client.next_prompt_id();
    let display_choices = choices
        .iter()
        .map(|value| prompt_value_text(Some(value)))
        .collect();
    Ok(Some(LocalPromptSpec {
        id,
        client,
        kind,
        choices,
        can_cancel,
        event: LocalClientUiEvent::Prompt {
            id,
            kind,
            title,
            message,
            default,
            choices: display_choices,
            can_cancel,
        },
    }))
}

pub(crate) fn register_prompt(state: &mut ExecutionState, id: u64, prompt: PendingLocalPrompt) {
    state.client.register_prompt(id, prompt);
}

fn collect_prompt_appearance_datums(
    appearance: &LocalClientAppearance,
    seen: &mut HashSet<DatumId>,
    values: &mut Vec<Value>,
) {
    if seen.insert(appearance.datum) {
        values.push(Value::Datum(appearance.datum));
    }
    for child in appearance
        .underlays
        .iter()
        .chain(appearance.overlays.iter())
    {
        collect_prompt_appearance_datums(child, seen, values);
    }
}

fn local_verb_prompt_candidates(state: &ExecutionState, client: DatumId) -> Vec<Value> {
    let mut seen = HashSet::new();
    let mut values = Vec::new();
    if let Ok(attached) = state.local_client_state(client) {
        let snapshot = state.local_client_map_snapshot_for(Some(client), attached.z);
        for tile in snapshot.tiles {
            for appearance in &tile.appearances {
                collect_prompt_appearance_datums(appearance, &mut seen, &mut values);
            }
            for occupant in tile.occupants {
                if seen.insert(occupant) {
                    values.push(Value::Datum(occupant));
                }
            }
        }
        for screen in snapshot.screen {
            collect_prompt_appearance_datums(&screen.appearance, &mut seen, &mut values);
        }
    }
    for datum in [Some(client), state.client.attached_mob(client)]
        .into_iter()
        .flatten()
    {
        if seen.insert(datum) {
            values.push(Value::Datum(datum));
        }
    }
    values
}

fn verb_atom_type_allows(state: &ExecutionState, value: &Value, mask: u8) -> bool {
    let Value::Datum(datum) = value else {
        return false;
    };
    let Ok(datum) = state.heap.datum(*datum) else {
        return false;
    };
    let path = datum.type_path().as_str();
    (mask & 1 != 0 && (path == "/obj" || path.starts_with("/obj/")))
        || (mask & 2 != 0 && (path == "/mob" || path.starts_with("/mob/")))
        || (mask & 4 != 0 && (path == "/turf" || path.starts_with("/turf/")))
        || (mask & 8 != 0 && (path == "/area" || path.starts_with("/area/")))
}

fn local_verb_choice_label(state: &ExecutionState, value: &Value) -> String {
    let Value::Datum(datum) = value else {
        return prompt_value_text(Some(value));
    };
    state.heap.datum(*datum).map_or_else(
        |_| value.to_string(),
        |value| format!("{} [0x{:x}]", value.type_path().as_str(), datum.index() + 1),
    )
}

pub(crate) fn queue_next_verb_prompt(
    state: &mut ExecutionState,
    client: DatumId,
    mut invocation: PendingVerbInvocation,
) -> Result<(), String> {
    let Some(parameter) = invocation
        .frame
        .supplied_parameters
        .iter()
        .position(|supplied| !supplied)
    else {
        schedule_frames(state, vec![invocation.frame], 0.0);
        return Ok(());
    };
    invocation.parameter = parameter;
    let kind = match invocation.parameter_types[parameter] {
        VerbParameterType::Text => LocalClientPromptKind::Text,
        VerbParameterType::Message => LocalClientPromptKind::Message,
        VerbParameterType::Number => LocalClientPromptKind::Number,
        VerbParameterType::Color => LocalClientPromptKind::Color,
        VerbParameterType::File => LocalClientPromptKind::File,
        VerbParameterType::Atom(_)
        | VerbParameterType::Anything
        | VerbParameterType::Unsupported => LocalClientPromptKind::List,
    };
    let choices = if kind == LocalClientPromptKind::List {
        let candidates = local_verb_prompt_candidates(state, client);
        match invocation.parameter_types[parameter] {
            VerbParameterType::Atom(mask) => {
                let mut filtered = candidates
                    .into_iter()
                    .filter(|value| verb_atom_type_allows(state, value, mask))
                    .collect::<Vec<_>>();
                if filtered.is_empty() {
                    filtered.extend(
                        state
                            .heap
                            .datums()
                            .map(|(datum, _)| Value::Datum(datum))
                            .filter(|value| verb_atom_type_allows(state, value, mask))
                            .take(256),
                    );
                }
                filtered
            }
            _ => candidates,
        }
    } else {
        Vec::new()
    };
    let display_choices = choices
        .iter()
        .map(|value| local_verb_choice_label(state, value))
        .collect::<Vec<_>>();
    let id = state.client.next_prompt_id();
    let parameter_name = invocation
        .parameter_names
        .get(parameter)
        .filter(|name| !name.is_empty())
        .cloned()
        .unwrap_or_else(|| format!("Argument {}", parameter + 1));
    state.emit_local_client_ui_event(
        client,
        LocalClientUiEvent::Prompt {
            id,
            kind,
            title: invocation.verb_name.clone(),
            message: parameter_name,
            default: String::new(),
            choices: display_choices,
            can_cancel: true,
        },
    );
    state.client.register_prompt(
        id,
        PendingLocalPrompt {
            client,
            kind,
            choices,
            can_cancel: true,
            continuation: PendingPromptContinuation::Verb(invocation),
        },
    );
    Ok(())
}

/// Host-facing orchestration for the in-process BYOND client model.
///
/// These methods coordinate the runtime heap, the scheduler, and the
/// [`ClientState`] store to open sessions, attach mobs, deliver map and
/// appearance snapshots, route pointer input, and drive modal prompt and
/// verb continuations.
impl ExecutionState {
    /// Allocates a local `/client` and atomically installs its parsed DMF skin.
    ///
    /// # Errors
    ///
    /// Returns the parser diagnostics without allocating a client when the
    /// supplied DMF contains any errors.
    ///
    /// # Panics
    ///
    /// Panics only if the engine's built-in `/client` type path becomes invalid.
    pub fn open_local_client(&mut self, dmf_source: &str) -> Result<DatumId, LocalClientError> {
        let document = parse_dmf(dmf_source);
        let diagnostics: Vec<_> = document
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
            .cloned()
            .collect();
        if !diagnostics.is_empty() {
            return Err(LocalClientError { diagnostics });
        }
        let client = self
            .heap
            .allocate_datum(TypePath::parse("/client").expect("the engine client path is valid"));
        self.install_client_session(client, ControlTree::from_document(&document));
        Ok(client)
    }

    /// Installs a skin-backed UI session for a connected client datum.
    pub fn install_client_session(&mut self, client: DatumId, tree: ControlTree) {
        self.client.install_session(client, tree);
    }

    /// Sets the parsed skin cloned into subsequently connected local clients.
    pub fn set_local_client_skin(&mut self, tree: ControlTree) {
        self.client.set_skin(tree);
    }

    /// Drains authoritative UI operations in exact DM execution order.
    #[must_use]
    pub fn take_local_client_outbound_events(
        &mut self,
        client: DatumId,
    ) -> Vec<LocalClientUiEvent> {
        self.client
            .take_outbound_events()
            .remove(&client)
            .unwrap_or_default()
    }

    pub(crate) fn emit_local_client_ui_event(
        &mut self,
        client: DatumId,
        event: LocalClientUiEvent,
    ) {
        self.client.emit_ui_event(client, event);
    }

    /// Returns the number of DM continuations waiting for native prompt input.
    #[must_use]
    pub fn pending_local_prompt_count(&self) -> usize {
        self.client.pending_prompt_count()
    }

    /// Supplies one typed native prompt answer and schedules its suspended DM
    /// continuation at the current scheduler boundary.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown prompt, the wrong client, cancellation
    /// of a required prompt, an invalid number, or an out-of-range choice.
    pub fn submit_local_prompt_response(
        &mut self,
        client: DatumId,
        id: u64,
        response: LocalClientPromptResponse,
    ) -> Result<(), String> {
        let prompt = self
            .client
            .take_prompt(id)
            .ok_or_else(|| format!("unknown local prompt {id}"))?;
        if prompt.client != client {
            self.client.register_prompt(id, prompt);
            return Err(format!("local prompt {id} belongs to another client"));
        }
        if matches!(response, LocalClientPromptResponse::Null)
            && matches!(&prompt.continuation, PendingPromptContinuation::Verb(_))
            && prompt.can_cancel
        {
            return Ok(());
        }
        let value = match response {
            LocalClientPromptResponse::Null if prompt.can_cancel => Value::Null,
            LocalClientPromptResponse::Null => {
                self.client.register_prompt(id, prompt);
                return Err(format!("local prompt {id} cannot be cancelled"));
            }
            LocalClientPromptResponse::Text(value)
                if matches!(
                    prompt.kind,
                    LocalClientPromptKind::Text
                        | LocalClientPromptKind::Message
                        | LocalClientPromptKind::Color
                        | LocalClientPromptKind::File
                ) =>
            {
                if prompt.kind == LocalClientPromptKind::File {
                    Value::File(value.into())
                } else {
                    Value::text(value)
                }
            }
            LocalClientPromptResponse::Text(value)
                if prompt.kind == LocalClientPromptKind::Number =>
            {
                Value::number(value.parse::<f32>().map_err(|_| {
                    self.client.register_prompt(id, prompt.clone());
                    format!("local prompt {id} requires a number")
                })?)
            }
            LocalClientPromptResponse::Number(value)
                if prompt.kind == LocalClientPromptKind::Number && value.is_finite() =>
            {
                Value::number(value)
            }
            LocalClientPromptResponse::Choice(index)
                if matches!(
                    prompt.kind,
                    LocalClientPromptKind::List | LocalClientPromptKind::Alert
                ) =>
            {
                prompt.choices.get(index).cloned().ok_or_else(|| {
                    self.client.register_prompt(id, prompt.clone());
                    format!("local prompt {id} choice {index} is out of range")
                })?
            }
            _ => {
                self.client.register_prompt(id, prompt);
                return Err(format!(
                    "local prompt {id} received an incompatible response"
                ));
            }
        };
        match prompt.continuation {
            PendingPromptContinuation::Frames(mut frames) => {
                let frame = frames
                    .last_mut()
                    .ok_or_else(|| format!("local prompt {id} has no continuation"))?;
                frame.stack.push(value);
                schedule_frames(self, frames, 0.0);
            }
            PendingPromptContinuation::Verb(mut invocation) => {
                let parameter = invocation.parameter;
                let value = if invocation.parameter_types[parameter] == VerbParameterType::File {
                    match value {
                        Value::Text(path) => Value::File(path),
                        value => value,
                    }
                } else {
                    value
                };
                let Some(local) = invocation.frame.locals.get_mut(parameter) else {
                    return Err(format!("local prompt {id} verb parameter is invalid"));
                };
                *local = value.clone();
                if parameter >= invocation.frame.arguments.len() {
                    invocation
                        .frame
                        .arguments
                        .resize(parameter + 1, Value::Null);
                }
                invocation.frame.arguments[parameter] = value;
                invocation.frame.supplied_parameters[parameter] = true;
                queue_next_verb_prompt(self, client, invocation)?;
            }
        }
        Ok(())
    }

    /// Returns the UI session associated with a connected client datum.
    #[must_use]
    pub fn client_session(&self, client: DatumId) -> Option<&ClientSession> {
        self.client.client_sessions.get(&client)
    }

    /// Returns the mutable UI session associated with a connected client datum.
    pub fn client_session_mut(&mut self, client: DatumId) -> Option<&mut ClientSession> {
        self.client.client_sessions.get_mut(&client)
    }

    /// Enables or disables modal prompt suspension for a window-attached
    /// client. Skin-only preflight clients remain non-interactive so startup
    /// probes cannot deadlock waiting for a UI response that has no consumer.
    pub fn set_local_client_interactive(
        &mut self,
        client: DatumId,
        interactive: bool,
    ) -> Result<(), String> {
        if !self.client.has_session(client) {
            return Err("local client has no installed UI session".to_owned());
        }
        self.client.set_interactive(client, interactive);
        Ok(())
    }

    /// Drains local UI events emitted by one connected client.
    #[must_use]
    pub fn take_client_events(&mut self, client: DatumId) -> Vec<UiEvent> {
        self.client
            .client_sessions
            .get_mut(&client)
            .map_or_else(Vec::new, ClientSession::take_events)
    }

    /// Binds an existing local client to an existing mob datum.
    ///
    /// # Errors
    ///
    /// Returns an error for stale identities or non-client/non-mob runtime types.
    pub fn attach_local_client(
        &mut self,
        client: DatumId,
        mob: DatumId,
    ) -> Result<LocalClientState, String> {
        let client_path = self
            .heap
            .datum(client)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str();
        if client_path != "/client" && !client_path.starts_with("/client/") {
            return Err("local controller identity is not a /client".to_owned());
        }
        let mob_path = self
            .heap
            .datum(mob)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str();
        if mob_path != "/mob" && !mob_path.starts_with("/mob/") {
            return Err("local controlled identity is not a /mob".to_owned());
        }
        assign_datum_field(
            self,
            client,
            FieldName::parse("mob").unwrap(),
            Value::Datum(mob),
        )?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("client").unwrap(),
            Value::Datum(client),
        )?;
        if self.local_client_coordinates(mob).is_err() {
            let turf = self
                .world_turfs
                .first_key_value()
                .map(|(_, turf)| *turf)
                .ok_or_else(|| "local client cannot attach to an empty world".to_owned())?;
            assign_datum_field(
                self,
                mob,
                FieldName::parse("loc").unwrap(),
                Value::Datum(turf),
            )?;
        }
        self.client.attach_mob(client, mob);
        self.local_client_state(client)
    }

    /// Allocates a local client and mob and places the mob on the first indexed turf.
    ///
    /// # Errors
    ///
    /// Returns an error when the authoritative world has no materialized turf.
    pub fn create_attached_local_client(&mut self) -> Result<LocalClientState, String> {
        let turf = self.world_turfs.values().next().copied().ok_or_else(|| {
            "cannot attach a local client before the map is materialized".to_owned()
        })?;
        let client = allocate_initialized_datum(
            self,
            TypePath::parse("/client").expect("engine client path is valid"),
        )?;
        let mob_type = self.connection_mob_type();
        let mob = allocate_initialized_datum(self, mob_type)?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("loc").unwrap(),
            Value::Datum(turf),
        )?;
        self.attach_local_client(client, mob)
    }

    pub(crate) fn create_pending_local_client(&mut self) -> Result<LocalClientState, String> {
        let turf = self.world_turfs.values().next().copied().ok_or_else(|| {
            "cannot attach a local client before the map is materialized".to_owned()
        })?;
        let client = allocate_initialized_datum(
            self,
            TypePath::parse("/client").expect("engine client path is valid"),
        )?;
        let mob = allocate_initialized_datum(self, self.connection_mob_type())?;
        assign_datum_field(
            self,
            mob,
            FieldName::parse("loc").unwrap(),
            Value::Datum(turf),
        )?;
        self.client.attach_mob(client, mob);
        self.local_client_state(client)
    }

    pub(crate) fn connection_mob_type(&self) -> TypePath {
        let fallback = TypePath::parse("/mob").expect("engine mob path is valid");
        let mob_field = FieldName::parse("mob").expect("engine world mob field is valid");
        self.heap
            .datums()
            .find(|(_, datum)| {
                let path = datum.type_path().as_str();
                path == "/world" || path.starts_with("/world/")
            })
            .and_then(|(world, _)| datum_field_or_initial(self, world, &mob_field).ok())
            .and_then(|value| match value {
                Value::TypePath(path) if builtins::is_subtype(self, &path, &fallback) => Some(path),
                Value::ModifiedTypePath(path)
                    if builtins::is_subtype(self, path.base(), &fallback) =>
                {
                    Some(path.base().clone())
                }
                _ => None,
            })
            .unwrap_or(fallback)
    }

    /// Creates a deterministic loopback guest and queues its project-defined
    /// `/client/New()` hook at the current scheduler boundary.
    ///
    /// The client session and client/mob relationship are installed before the
    /// frame becomes runnable, so `New()` observes the same fully connected
    /// identity that later UI builtins use. A sleeping hook remains an ordinary
    /// scheduled continuation; runtime failures are returned by
    /// [`advance_scheduler`].
    ///
    /// # Errors
    ///
    /// Returns an error when the world has no turf, the client cannot be bound,
    /// or the runtime client type has no effective `New` implementation.
    pub fn connect_local_guest(&mut self, module: &Module) -> Result<LocalClientState, String> {
        if self.client.local_client_skin.is_none()
            && let Some(root) = self.project_root.as_deref()
        {
            let skin_path = root.join("interface").join("skin.dmf");
            if skin_path.is_file() {
                let source = std::fs::read_to_string(&skin_path).map_err(|error| {
                    format!(
                        "failed to read local client skin {}: {error}",
                        skin_path.display()
                    )
                })?;
                let document = parse_dmf(&source);
                if let Some(diagnostic) = document
                    .diagnostics
                    .iter()
                    .find(|diagnostic| diagnostic.severity == DiagnosticSeverity::Error)
                {
                    return Err(format!(
                        "local client skin {} is invalid: {}",
                        skin_path.display(),
                        diagnostic.message
                    ));
                }
                self.client.set_skin(ControlTree::from_document(&document));
            }
        }
        let attached = self.create_pending_local_client()?;
        self.populate_local_verb_inventory(module, attached.client)?;
        self.populate_local_verb_inventory(module, attached.mob)?;
        let sequence = self.client.next_guest_id();
        let key = format!("Guest-{sequence}");
        for (name, value) in [
            ("key", Value::text(key.as_str())),
            ("ckey", Value::text(key.to_ascii_lowercase())),
            ("address", Value::text("127.0.0.1")),
            (
                "computer_id",
                Value::text(format!("dream64-local-{sequence}")),
            ),
            ("connection", Value::text("seeker")),
            ("byond_version", Value::number(516.0)),
            ("byond_build", Value::number(1680.0)),
        ] {
            let field = FieldName::parse(name).expect("guest identity field is valid");
            if datum_field_or_initial(self, attached.client, &field).is_ok() {
                assign_datum_field(self, attached.client, field, value)?;
            }
        }
        self.install_client_session(
            attached.client,
            self.client.local_client_skin.clone().unwrap_or_default(),
        );

        let receiver = Value::Datum(attached.client);
        let (procedure, context) = dynamic_call_target_named(
            module,
            self,
            &receiver,
            "New",
            &ExecutionContext::new(receiver.clone(), receiver.clone()),
            false,
        )?;
        let program = module.resolve_procedure(procedure)?;
        let frame = make_frame(procedure, program, &[], &context);
        schedule_frames(self, vec![frame], 0.0);
        Ok(attached)
    }

    /// Queues cardinal movement for deterministic application at a scheduler boundary.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is not attached or the controlled mob is stale.
    pub fn queue_local_movement(
        &mut self,
        client: DatumId,
        direction: LocalMovementDirection,
    ) -> Result<(), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;
        self.client.queue_command(client, direction);
        Ok(())
    }

    /// Queues a browser `byond://` request through the attached client's
    /// effective `Topic()` implementation.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is detached, the parameter string is
    /// invalid, or no effective client `Topic` procedure can be resolved.
    pub fn queue_local_browser_topic(
        &mut self,
        module: &Module,
        client: DatumId,
        topic: &str,
    ) -> Result<(), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached".to_owned())?;
        self.heap.datum(client).map_err(|error| error.to_string())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;
        let query = topic
            .strip_prefix("byond://")
            .unwrap_or(topic)
            .strip_prefix('?')
            .unwrap_or_else(|| topic.strip_prefix("byond://").unwrap_or(topic));
        let href_list = self.decode_params_list(query)?;
        let hsrc = match &href_list {
            Value::List(list) => self
                .heap
                .list(*list)
                .ok()
                .and_then(|values| values.get_key(&Value::text("src")).ok())
                .and_then(|value| match value {
                    Value::Text(reference) => match parse_heap_reference(reference) {
                        Some(HeapReference::Datum(index)) => {
                            self.heap.datum_id_at_index(index).map(Value::Datum)
                        }
                        _ => None,
                    },
                    _ => None,
                })
                .unwrap_or(Value::Null),
            _ => Value::Null,
        };
        let receiver = Value::Datum(client);
        let usr = Value::Datum(mob);
        let (procedure, context) = dynamic_call_target_named(
            module,
            self,
            &receiver,
            "Topic",
            // BYOND dispatches /client/Topic() with src set to the client and
            // usr set to that client's mob. SS13 security middleware rejects
            // browser messages when this relationship is not preserved.
            &ExecutionContext::new(receiver.clone(), usr),
            false,
        )?;
        let program = module.resolve_procedure(procedure)?;
        let arguments = [Value::text(topic), href_list, hsrc, Value::number(0.0)];
        schedule_frames(
            self,
            vec![make_frame(procedure, program, &arguments, &context)],
            0.0,
        );
        Ok(())
    }

    /// Resolves and queues one BYOND command against the attached client's
    /// verb inventory. Client verbs take precedence over mob verbs, matching
    /// the command surface exposed by a connected BYOND client.
    ///
    /// # Errors
    ///
    /// Returns an error when the client is detached, the command is malformed,
    /// no matching verb exists, or its supplied argument count is invalid.
    pub fn queue_local_client_command(
        &mut self,
        module: &Module,
        client: DatumId,
        command: &str,
    ) -> Result<(), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        self.heap.datum(client).map_err(|error| error.to_string())?;
        self.heap.datum(mob).map_err(|error| error.to_string())?;

        let (command_name, raw_arguments) = split_client_command(command)?;
        let normalized_command = normalize_client_command_name(command_name);
        let client_receiver = Value::Datum(client);
        let mob_receiver = Value::Datum(mob);
        let caller = ExecutionContext::new(client_receiver.clone(), mob_receiver.clone());
        let mut resolved = None;
        for receiver in [&client_receiver, &mob_receiver] {
            let Value::Datum(datum) = receiver else {
                unreachable!("local command receivers are datums")
            };
            for verb_path in self.local_verb_inventory(*datum)? {
                let path = verb_path.as_str();
                let Some((_, selector)) = path.rsplit_once("/verb/") else {
                    continue;
                };
                let selector = selector.split('@').next().unwrap_or(selector);
                let Some(procedure) = module
                    .effective_procedure_id(path)
                    .or_else(|| module.procedure_id(path))
                else {
                    continue;
                };
                let Some(program) = module.procedure(procedure) else {
                    continue;
                };
                let verb_command_name = program.verb_name.as_deref().unwrap_or(selector);
                if normalize_client_command_name(verb_command_name) != normalized_command {
                    continue;
                }
                let explicit_selector = format!("verb/{selector}");
                if let Ok(target) = dynamic_call_target_named(
                    module,
                    self,
                    receiver,
                    &explicit_selector,
                    &caller,
                    false,
                ) {
                    resolved = Some(target);
                    break;
                }
            }
            if resolved.is_some() {
                break;
            }
        }
        let (procedure, context) =
            resolved.ok_or_else(|| format!("unknown client command {command_name:?}"))?;
        let program = module.resolve_procedure(procedure)?;
        let arguments = parse_client_command_arguments(raw_arguments)?;
        if arguments.len() > program.parameter_count {
            return Err(format!(
                "client command {command_name:?} accepts at most {} argument(s), received {}",
                program.parameter_count,
                arguments.len()
            ));
        }
        let mut values = vec![Value::Null; program.parameter_count];
        let mut supplied = vec![false; program.parameter_count];
        for (index, argument) in arguments.into_iter().enumerate() {
            match program.verb_parameter_types[index] {
                VerbParameterType::Text | VerbParameterType::Message | VerbParameterType::Color => {
                    values[index] = Value::text(argument);
                    supplied[index] = true;
                }
                VerbParameterType::Number => {
                    values[index] = Value::number(argument.parse::<f32>().map_err(|_| {
                        format!(
                            "invalid number argument {argument:?} for client command {command_name:?}",
                        )
                    })?);
                    supplied[index] = true;
                }
                VerbParameterType::File => {
                    values[index] = Value::File(argument.into());
                    supplied[index] = true;
                }
                VerbParameterType::Atom(_)
                | VerbParameterType::Anything
                | VerbParameterType::Unsupported => {}
            }
        }
        let mut frame = make_frame(procedure, program, &values, &context);
        frame.supplied_parameters = supplied.into();
        if frame.supplied_parameters.iter().all(|supplied| *supplied) {
            schedule_frames(self, vec![frame], 0.0);
        } else {
            queue_next_verb_prompt(
                self,
                client,
                PendingVerbInvocation {
                    frame,
                    parameter_types: program.verb_parameter_types.clone(),
                    parameter_names: program.parameter_names.clone(),
                    verb_name: program
                        .verb_name
                        .clone()
                        .unwrap_or_else(|| command_name.to_owned()),
                    parameter: 0,
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn populate_local_verb_inventory(
        &mut self,
        module: &Module,
        datum: DatumId,
    ) -> Result<(), String> {
        let runtime_type = self
            .heap
            .datum(datum)
            .map_err(|error| error.to_string())?
            .type_path()
            .clone();
        let mut defaults = Vec::new();
        for path in module.procedure_paths() {
            let canonical = path.split_once('@').map_or(path, |(path, _)| path);
            let Some((owner, _)) = canonical.rsplit_once("/verb/") else {
                continue;
            };
            let Ok(owner) = TypePath::parse(owner) else {
                continue;
            };
            if builtins::is_subtype(self, &runtime_type, &owner)
                && let Ok(verb) = TypePath::parse(canonical)
                && !defaults.contains(&verb)
            {
                defaults.push(verb);
            }
        }
        let verbs_field = FieldName::parse("verbs").expect("engine verbs field is valid");
        let list = if let Ok(Value::List(list)) = self.heap.datum_field(datum, &verbs_field) {
            *list
        } else {
            let list = self.heap.allocate_list();
            self.heap
                .set_datum_field(datum, verbs_field, Value::List(list))
                .map_err(|error| error.to_string())?;
            list
        };
        let existing = self
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        let list = self
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        for verb in defaults {
            let value = Value::TypePath(verb);
            if !existing.contains(&value) {
                list.add(value);
            }
        }
        Ok(())
    }

    pub(crate) fn local_verb_inventory(&self, datum: DatumId) -> Result<Vec<TypePath>, String> {
        let verbs = datum_field_or_initial(
            self,
            datum,
            &FieldName::parse("verbs").expect("engine verbs field is valid"),
        )
        .map_err(|error| error.to_string())?;
        let Value::List(verbs) = verbs else {
            return Ok(Vec::new());
        };
        Ok(self
            .heap
            .list(verbs)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::TypePath(path) => Some(path.clone()),
                Value::ModifiedTypePath(path) => Some(path.base().clone()),
                _ => None,
            })
            .collect())
    }

    /// Applies every queued local command in stable enqueue order.
    ///
    /// This is the host's scheduler-boundary commit point. No movement mutates
    /// the live world before this method is called.
    ///
    /// # Errors
    ///
    /// Returns an error if an attached datum or its authoritative turf is stale.
    pub fn apply_local_client_commands(&mut self) -> Result<Vec<LocalClientState>, String> {
        let mut commands = self.client.take_commands();
        commands.sort_by_key(|(sequence, _, _)| *sequence);
        let mut committed = Vec::with_capacity(commands.len());
        for (_, client, direction) in commands {
            let mob = self
                .client
                .attached_mob(client)
                .ok_or_else(|| "local client detached before movement commit".to_owned())?;
            let current = self.local_client_coordinates(mob)?;
            let (dx, dy) = match direction {
                LocalMovementDirection::North => (0, 1),
                LocalMovementDirection::South => (0, -1),
                LocalMovementDirection::East => (1, 0),
                LocalMovementDirection::West => (-1, 0),
            };
            if let Some(destination) = self.turf_at(current.0 + dx, current.1 + dy, current.2) {
                assign_datum_field(
                    self,
                    mob,
                    FieldName::parse("loc").unwrap(),
                    Value::Datum(destination),
                )?;
            }
            committed.push(self.local_client_state(client)?);
        }
        Ok(committed)
    }

    /// Returns the authoritative location for an attached local client.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown client or a mob outside a turf.
    pub fn local_client_state(&self, client: DatumId) -> Result<LocalClientState, String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let (x, y, z) = self.local_client_coordinates(mob)?;
        Ok(LocalClientState {
            client,
            mob,
            x,
            y,
            z,
        })
    }

    /// Returns the turf coordinates observed by a client camera. BYOND uses
    /// `client.eye` for map projection and falls back to the controlled mob
    /// when no explicit eye is installed.
    pub fn local_client_view_coordinates(
        &self,
        client: DatumId,
    ) -> Result<(i32, i32, i32), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let eye = datum_field_or_initial(
            self,
            client,
            &FieldName::parse("eye").expect("client eye field"),
        )
        .map_err(|error| error.to_string())?;
        if let Value::Datum(eye) = eye {
            if let Some(coordinate) = self
                .world_turfs
                .iter()
                .find_map(|(coordinate, turf)| (*turf == eye).then_some(*coordinate))
            {
                return Ok(coordinate);
            }
            if let Ok(coordinate) = self.local_client_coordinates(eye) {
                return Ok(coordinate);
            }
        }
        self.local_client_coordinates(mob)
    }

    /// Copies one Z level into a stable transport-owned map snapshot.
    #[must_use]
    pub fn local_client_map_snapshot(&self, z: i32) -> LocalClientMapSnapshot {
        self.local_client_map_snapshot_for(None, z)
    }

    /// Copies a Z level plus the selected client's screen HUD appearances.
    #[must_use]
    pub fn local_client_map_snapshot_for(
        &self,
        client: Option<DatumId>,
        z: i32,
    ) -> LocalClientMapSnapshot {
        let color_field = FieldName::parse("color").unwrap();
        let mut tiles = self
            .world_turfs
            .iter()
            .filter_map(|(&(x, y, cell_z), &turf)| {
                (cell_z == z)
                    .then(|| {
                        let datum = self.heap.datum(turf).ok()?;
                        let color = datum_field_or_initial(self, turf, &color_field)
                            .ok()
                            .and_then(|value| {
                                (!matches!(value, Value::Null)).then(|| value.to_string())
                            });
                        let occupants: Vec<DatumId> = datum_field_or_initial(
                            self,
                            turf,
                            &FieldName::parse("contents").unwrap(),
                        )
                        .ok()
                        .and_then(|value| match value {
                            Value::List(list) => self.heap.list(list).ok(),
                            _ => None,
                        })
                        .map(|contents| {
                            contents
                                .positions()
                                .filter_map(|(_, value)| match value {
                                    Value::Datum(id) => Some(*id),
                                    _ => None,
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                        let mut draw_datums = Vec::with_capacity(occupants.len() + 1);
                        draw_datums.push(turf);
                        draw_datums.extend(occupants.iter().copied());
                        let mut appearances = draw_datums
                            .into_iter()
                            .enumerate()
                            .filter_map(|(order, datum)| {
                                self.local_client_appearance(datum, 0, &mut HashSet::new())
                                    .map(|appearance| (order, appearance))
                            })
                            .collect::<Vec<_>>();
                        appearances.sort_by(|(left_order, left), (right_order, right)| {
                            left.plane
                                .total_cmp(&right.plane)
                                .then_with(|| left.layer.total_cmp(&right.layer))
                                .then_with(|| left_order.cmp(right_order))
                        });
                        let appearances = appearances
                            .into_iter()
                            .map(|(_, appearance)| appearance)
                            .collect();
                        Some(LocalClientMapTile {
                            x,
                            y,
                            type_path: datum.type_path().as_str().to_owned(),
                            color,
                            occupants,
                            appearances,
                        })
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        tiles.sort_by_key(|tile| (tile.y, tile.x));
        let width = tiles.iter().map(|tile| tile.x).max().unwrap_or(0);
        let height = tiles.iter().map(|tile| tile.y).max().unwrap_or(0);
        let mut screen = client
            .and_then(|client| {
                datum_field_or_initial(
                    self,
                    client,
                    &FieldName::parse("screen").expect("client screen field"),
                )
                .ok()
            })
            .and_then(|value| match value {
                Value::List(list) => self.heap.list(list).ok(),
                _ => None,
            })
            .map(|list| {
                list.positions()
                    .filter_map(|(_, value)| match value {
                        Value::Datum(datum) => Some(*datum),
                        _ => None,
                    })
                    .enumerate()
                    .filter_map(|(order, datum)| {
                        let raw_screen_loc = datum_field_or_initial(
                            self,
                            datum,
                            &FieldName::parse("screen_loc").expect("screen_loc field"),
                        )
                        .ok()
                        .and_then(|value| match value {
                            Value::Text(text) => Some(text.to_string()),
                            Value::Null => None,
                            value => Some(value.to_string()),
                        })
                        .unwrap_or_default();
                        let (map_control, screen_loc) = raw_screen_loc
                            .split_once(':')
                            .filter(|(prefix, coordinates)| {
                                !prefix.is_empty()
                                    && prefix.chars().all(|ch| {
                                        ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
                                    })
                                    && !matches!(
                                        prefix.trim().to_ascii_uppercase().as_str(),
                                        "TOP"
                                            | "BOTTOM"
                                            | "NORTH"
                                            | "SOUTH"
                                            | "LEFT"
                                            | "RIGHT"
                                            | "EAST"
                                            | "WEST"
                                            | "CENTER"
                                    )
                                    && coordinates.contains(',')
                            })
                            .map_or((None, raw_screen_loc.clone()), |(control, coordinates)| {
                                (Some(control.to_owned()), coordinates.to_owned())
                            });
                        self.local_client_appearance(datum, 0, &mut HashSet::new())
                            .map(|appearance| {
                                (
                                    order,
                                    LocalClientScreenAppearance {
                                        map_control,
                                        screen_loc,
                                        insertion: order,
                                        appearance,
                                    },
                                )
                            })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        screen.sort_by(|(left_order, left), (right_order, right)| {
            left.appearance
                .plane
                .total_cmp(&right.appearance.plane)
                .then_with(|| left.appearance.layer.total_cmp(&right.appearance.layer))
                .then_with(|| left_order.cmp(right_order))
        });
        let screen = screen
            .into_iter()
            .map(|(_, appearance)| appearance)
            .collect();
        LocalClientMapSnapshot {
            width,
            height,
            z,
            tiles,
            screen,
        }
    }

    /// Validates and queues a mouse proc on one atom in this client's screen list.
    pub fn queue_local_screen_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        target_index: u32,
        target_generation: u32,
        event: LocalScreenPointerEvent,
        location: &str,
        params: &str,
    ) -> Result<(), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let screen = datum_field_or_initial(
            self,
            client,
            &FieldName::parse("screen").expect("client screen field"),
        )
        .map_err(|error| error.to_string())?;
        let Value::List(screen) = screen else {
            return Err("client screen is not a list".into());
        };
        let target = self
            .heap
            .list(screen)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Datum(id) => Some(*id),
                _ => None,
            })
            .find(|id| id.index() == target_index && id.generation() == target_generation)
            .ok_or_else(|| "screen target is stale or not owned by session".to_owned())?;
        self.heap.datum(target).map_err(|error| error.to_string())?;
        self.queue_local_atom_pointer(module, client, mob, target, event, location, params)
    }

    /// Validates and queues a click on an atom rendered in the addressed map cell.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_local_map_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        target_index: u32,
        target_generation: u32,
        x: i32,
        y: i32,
        z: i32,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let mob = self
            .client
            .attached_mob(client)
            .ok_or_else(|| "local client is not attached to a mob".to_owned())?;
        let target = self
            .heap
            .datum_id_at_index(target_index)
            .filter(|id| id.generation() == target_generation)
            .ok_or_else(|| "map target is stale".to_owned())?;
        let datum = self.heap.datum(target).map_err(|error| error.to_string())?;
        if !is_atom_type_path(datum.type_path()) {
            return Err("map target is not an atom".to_owned());
        }
        let expected = (x as f32, y as f32, z as f32);
        if builtins::datum_coordinates(self, &Value::Datum(target)) != Some(expected) {
            return Err("map target is stale or outside the addressed cell".to_owned());
        }
        if self.turf_at(x, y, z).is_none() {
            return Err("map pointer cell has no materialized turf".to_owned());
        }
        self.queue_local_atom_pointer(
            module,
            client,
            mob,
            target,
            LocalScreenPointerEvent::Click,
            control,
            params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn queue_local_atom_pointer(
        &mut self,
        module: &Module,
        client: DatumId,
        mob: DatumId,
        target: DatumId,
        event: LocalScreenPointerEvent,
        control: &str,
        params: &str,
    ) -> Result<(), String> {
        let target_value = Value::Datum(target);
        let usr = Value::Datum(mob);
        let null_or_text = |value: &str| {
            if value.is_empty() {
                Value::Null
            } else {
                Value::text(value)
            }
        };
        let (receiver, method, arguments) = match event {
            LocalScreenPointerEvent::Click => (
                Value::Datum(client),
                "Click",
                vec![
                    target_value.clone(),
                    Value::Null,
                    null_or_text(control),
                    Value::text(params),
                ],
            ),
            LocalScreenPointerEvent::Entered | LocalScreenPointerEvent::Exited => {
                let location = datum_field_or_initial(
                    self,
                    target,
                    &FieldName::parse("loc").expect("atom loc field"),
                )
                .unwrap_or(Value::Null);
                (
                    target_value.clone(),
                    match event {
                        LocalScreenPointerEvent::Entered => "MouseEntered",
                        LocalScreenPointerEvent::Exited => "MouseExited",
                        LocalScreenPointerEvent::Click => unreachable!(),
                    },
                    vec![location, null_or_text(control), Value::text(params)],
                )
            }
        };
        let caller = ExecutionContext::new(receiver.clone(), usr.clone());
        let resolved = dynamic_call_target_named(module, self, &receiver, method, &caller, false)
            .or_else(|error| {
            // Small fixtures may omit BYOND's built-in `/client/Click`.
            // Preserve direct atom dispatch there while full worlds take
            // OpenDream's client interception path.
            if event != LocalScreenPointerEvent::Click {
                return Err(error);
            }
            let target_context = ExecutionContext::new(target_value.clone(), usr);
            dynamic_call_target_named(module, self, &target_value, "Click", &target_context, false)
        })?;
        let (procedure, context) = resolved;
        let program = module.resolve_procedure(procedure)?;
        let arguments =
            if matches!(event, LocalScreenPointerEvent::Click) && context.src == target_value {
                if program.parameter_names.len() <= 2 {
                    vec![null_or_text(control), Value::text(params)]
                } else {
                    vec![Value::Null, null_or_text(control), Value::text(params)]
                }
            } else {
                arguments
            };
        let frame = make_frame(procedure, program, &arguments, &context);
        schedule_frames(self, vec![frame], 0.0);
        Ok(())
    }

    pub(crate) fn local_client_appearance(
        &self,
        datum: DatumId,
        depth: usize,
        visited: &mut HashSet<DatumId>,
    ) -> Option<LocalClientAppearance> {
        if depth >= 16 || !visited.insert(datum) {
            return None;
        }
        let type_path = self.heap.datum(datum).ok()?.type_path().as_str().to_owned();
        let value =
            |name: &str| datum_field_or_initial(self, datum, &FieldName::parse(name).unwrap()).ok();
        let numeric = |name: &str, fallback: f32| {
            value(name)
                .and_then(|value| value.as_number())
                .unwrap_or(fallback)
        };
        let text = |name: &str| match value(name) {
            Some(Value::Text(text) | Value::File(text)) => Some(text.to_string()),
            Some(Value::Null) | None => None,
            Some(value) => Some(value.to_string()),
        };
        let icon = value("icon").and_then(|value| self.local_client_icon_resource(&value, 0));
        let mut nested = |name: &str| {
            let mut entries = match value(name) {
                Some(Value::List(list)) => self
                    .heap
                    .list(list)
                    .ok()
                    .map(|values| {
                        values
                            .positions()
                            .filter_map(|(_, value)| match value {
                                Value::Datum(datum) => Some(*datum),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                Some(Value::Datum(datum)) => vec![datum],
                _ => Vec::new(),
            }
            .into_iter()
            .enumerate()
            .filter_map(|(order, child)| {
                self.local_client_appearance(child, depth + 1, visited)
                    .map(|appearance| (order, appearance))
            })
            .collect::<Vec<_>>();
            entries.sort_by(|(left_order, left), (right_order, right)| {
                left.plane
                    .total_cmp(&right.plane)
                    .then_with(|| left.layer.total_cmp(&right.layer))
                    .then_with(|| left_order.cmp(right_order))
            });
            entries
                .into_iter()
                .map(|(_, appearance)| appearance)
                .collect()
        };
        let underlays = nested("underlays");
        let overlays = nested("overlays");
        visited.remove(&datum);
        Some(LocalClientAppearance {
            datum,
            type_path,
            icon,
            icon_state: text("icon_state"),
            dir: numeric("dir", 2.0) as i32,
            layer: numeric("layer", 0.0),
            plane: numeric("plane", 0.0),
            appearance_flags: numeric("appearance_flags", 0.0) as i32,
            mouse_opacity: numeric("mouse_opacity", 1.0) as i32,
            pixel_x: numeric("pixel_x", 0.0),
            pixel_y: numeric("pixel_y", 0.0),
            pixel_w: numeric("pixel_w", 0.0),
            pixel_z: numeric("pixel_z", 0.0),
            color: text("color"),
            alpha: numeric("alpha", 255.0),
            maptext: text("maptext"),
            maptext_width: numeric("maptext_width", 0.0),
            maptext_height: numeric("maptext_height", 0.0),
            maptext_x: numeric("maptext_x", 0.0),
            maptext_y: numeric("maptext_y", 0.0),
            underlays,
            overlays,
        })
    }

    pub(crate) fn local_client_icon_resource(&self, value: &Value, depth: usize) -> Option<String> {
        if depth >= 16 {
            return None;
        }
        match value {
            Value::File(path) | Value::Text(path) => Some(path.to_string()),
            Value::Datum(icon) => {
                let datum = self.heap.datum(*icon).ok()?;
                let path = datum.type_path().as_str();
                if path != "/icon" && !path.starts_with("/icon/") {
                    return None;
                }
                let backing =
                    datum_field_or_initial(self, *icon, &FieldName::parse("icon").unwrap()).ok()?;
                self.local_client_icon_resource(&backing, depth + 1)
            }
            _ => None,
        }
    }

    pub(crate) fn local_client_coordinates(&self, mob: DatumId) -> Result<(i32, i32, i32), String> {
        let loc = FieldName::parse("loc").unwrap();
        let Value::Datum(turf) =
            datum_field_or_initial(self, mob, &loc).map_err(|error| error.to_string())?
        else {
            return Err("controlled mob is not located on a turf".to_owned());
        };
        self.world_turfs
            .iter()
            .find_map(|(coordinate, candidate)| (*candidate == turf).then_some(*coordinate))
            .ok_or_else(|| {
                "controlled mob turf is absent from the authoritative world index".to_owned()
            })
    }
}

fn normalize_client_command_name(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .flat_map(char::to_lowercase)
        .collect()
}

fn split_client_command(command: &str) -> Result<(&str, &str), String> {
    let command = command.trim();
    if command.is_empty() {
        return Err("client command is empty".to_owned());
    }
    Ok(command
        .split_once(char::is_whitespace)
        .map_or((command, ""), |(name, arguments)| (name, arguments.trim())))
}

fn parse_client_command_arguments(arguments: &str) -> Result<Vec<String>, String> {
    let mut parsed = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for ch in arguments.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }
        if quoted && ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            quoted = !quoted;
            continue;
        }
        if ch.is_whitespace() && !quoted {
            if !current.is_empty() {
                parsed.push(std::mem::take(&mut current));
            }
            continue;
        }
        current.push(ch);
    }
    if escaped {
        current.push('\\');
    }
    if quoted {
        return Err("client command has an unterminated quote".to_owned());
    }
    if !current.is_empty() {
        parsed.push(current);
    }
    Ok(parsed)
}
