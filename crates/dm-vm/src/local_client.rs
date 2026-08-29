//! The in-process BYOND client model: connected sessions, map/appearance
//! transport snapshots, and the modal prompt/verb continuation machinery.
//!
//! The engine executes DM with no external display. A *local client* is the
//! authoritative in-VM session a host (`dm-lifecycle`) can attach, move,
//! receive map snapshots from, and answer prompts for. Prompt continuations
//! suspend the calling DM frame and re-schedule it when the host answers.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::bytecode::VerbParameterType;
use crate::{CallFrame, ExecutionState, OwnedContinuation, schedule_frames};
use dm_dmf::Diagnostic;
use dm_value::{DatumId, Value};

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
            if state.interactive_local_clients.contains(&datum) {
                Some(datum)
            } else {
                state.local_client_mobs.iter().find_map(|(client, mob)| {
                    (*mob == datum && state.interactive_local_clients.contains(client))
                        .then_some(*client)
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
    state.local_prompt_sequence = state.local_prompt_sequence.saturating_add(1);
    let id = state.local_prompt_sequence;
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
    state.pending_local_prompts.insert(id, prompt);
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
    for datum in [Some(client), state.local_client_mobs.get(&client).copied()]
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
    state
        .heap
        .datum(*datum)
        .map(|value| format!("{} [0x{:x}]", value.type_path().as_str(), datum.index() + 1))
        .unwrap_or_else(|_| value.to_string())
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
    state.local_prompt_sequence = state.local_prompt_sequence.saturating_add(1);
    let id = state.local_prompt_sequence;
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
    state.pending_local_prompts.insert(
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
