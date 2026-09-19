//! The first visible Dream64 local-client shell.

#![deny(missing_docs)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::Path,
};
use std::io::{Read, Write};

use dm_dmf::{
    ControlTree, ControlType, PixelRect, UiCommand, UiEvent, WEBVIEW2_BYOND_BRIDGE_BOOTSTRAP,
};
use dm_vm::ExecutionState;
use dm_world::WorldCoordinate;
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod gpu;
mod sprite;
mod launch;
mod layout;
mod render_cpu;
mod widget_state;
mod transport;
mod ui_commands;
mod render_map;
pub(crate) use render_map::*;
#[allow(unused_imports)]
pub(crate) use render_cpu::{
    PromptHit, client_prompt_hit, client_prompt_rect, draw_appearance_maptext_signed,
    draw_bitmap_text, draw_boot_status, draw_button_control, draw_client_prompt,
    draw_input_control, draw_label_control, draw_output_control, draw_panel,
    pixel_rect_contains, strip_output_markup, wrap_output_line,
};
pub(crate) use ui_commands::{
    BrowserUpdate, ClientPrompt, ClientPromptKind, InboundUiCommand, SoundUpdate, UiOwnerCommand,
    UiPresentation,
};
use launch::LaunchOptions;
use layout::{ClientLayout, control_has_type, dmf_false, dmf_truthy, parse_pair, resolve_pane_layout_in};
pub(crate) use transport::percent_decode_form;
use transport::{
    ClientPromptResponse, ClientTransport, LoopbackTransport, MapSnapshot, PendingRemoteTransport,
    RemoteTransport, WorldScene, parse_byond_url, quote_dmf_assignment,
    snapshot_has_lobby_screen, valid_byond_callback,
};
use widget_state::{
    ButtonState, InputState, LabelState, button_states_from_ui, input_states_from_ui,
    label_states_from_ui, take_input_submission,
};
#[cfg(test)]
use sprite::composite_tile;
use sprite::{Appearance, SpriteCache};

#[cfg(windows)]
use wry::{
    Rect, WebView, WebViewBuilder,
    dpi::{LogicalPosition, LogicalSize},
};
#[cfg(windows)]
use {
    dm_native_menu::muda::{Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu},
    winit::raw_window_handle::{HasWindowHandle, RawWindowHandle},
};

const LOCAL_SKIN: &str = "window \"main\"\n\
\telem \"main\"\n\
\t\ttype = MAIN\n\
\t\tsize = 1024x768\n\
\telem \"map\"\n\
\t\ttype = MAP\n\
\telem \"output\"\n\
\t\ttype = OUTPUT\n\
\telem \"browser\"\n\
\t\ttype = BROWSER\n\
\t\tpos = 690,74\n\
\t\tsize = 318x618\n";

pub(crate) const DEFAULT_RETAINED_OUTPUT_LINES: usize = 512;
const MAX_UI_OWNER_COMMANDS: usize = 256;
static NEXT_CLIENT_GENERATION: AtomicU64 = AtomicU64::new(1);
const AUDIO_BROWSER_CONTROL: &str = "__dream64_audio";

#[repr(u16)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClientReadiness {
    Transport = 1 << 0,
    LocalResources = 1 << 1,
    Skin = 1 << 2,
    ResourceManifest = 1 << 3,
    ResourcePayload = 1 << 4,
    MapRenderer = 1 << 5,
    WebViewEnvironment = 1 << 6,
    WebViewDocument = 1 << 7,
}

#[derive(Clone, Debug)]
struct GenerationTagged<T> {
    generation: u64,
    command: T,
}

#[derive(Clone, Debug)]
struct UiOwnerCommandQueue<T> {
    generation: u64,
    capacity: usize,
    commands: VecDeque<GenerationTagged<T>>,
    stale_dropped: u64,
    overflow_rejected: u64,
}

impl<T> UiOwnerCommandQueue<T> {
    fn new(generation: u64, capacity: usize) -> Self {
        Self {
            generation,
            capacity,
            commands: VecDeque::with_capacity(capacity),
            stale_dropped: 0,
            overflow_rejected: 0,
        }
    }

    fn advance_generation(&mut self, generation: u64) {
        self.generation = generation;
        let before = self.commands.len();
        self.commands
            .retain(|command| command.generation == generation);
        self.stale_dropped = self
            .stale_dropped
            .saturating_add((before - self.commands.len()) as u64);
    }

    fn push(&mut self, generation: u64, command: T) -> Result<(), T> {
        if generation != self.generation {
            self.stale_dropped = self.stale_dropped.saturating_add(1);
            return Err(command);
        }
        if self.commands.len() == self.capacity {
            self.overflow_rejected = self.overflow_rejected.saturating_add(1);
            return Err(command);
        }
        self.commands.push_back(GenerationTagged {
            generation,
            command,
        });
        Ok(())
    }

    fn pop_current(&mut self) -> Option<T> {
        while let Some(command) = self.commands.pop_front() {
            if command.generation == self.generation {
                return Some(command.command);
            }
            self.stale_dropped = self.stale_dropped.saturating_add(1);
        }
        None
    }
}

#[derive(Clone, Debug)]
struct ClientReadinessCoordinator {
    generation: u64,
    reached: u16,
    webview_required: bool,
    audio_available: bool,
    resources_sent: bool,
    interactive_sent: bool,
}

impl ClientReadinessCoordinator {
    fn new() -> Self {
        Self {
            generation: NEXT_CLIENT_GENERATION.fetch_add(1, Ordering::Relaxed),
            reached: 0,
            webview_required: false,
            audio_available: false,
            resources_sent: false,
            interactive_sent: false,
        }
    }

    fn rearm(&mut self) {
        *self = Self::new();
    }

    fn mark(&mut self, generation: u64, readiness: ClientReadiness) -> Result<(), &'static str> {
        if generation != self.generation {
            return Err("stale-client-generation");
        }
        self.reached |= readiness as u16;
        Ok(())
    }

    const fn has(&self, readiness: ClientReadiness) -> bool {
        self.reached & readiness as u16 != 0
    }

    fn resources_ready(&self) -> bool {
        self.has(ClientReadiness::Transport)
            && self.has(ClientReadiness::LocalResources)
            && self.has(ClientReadiness::Skin)
            && self.has(ClientReadiness::ResourceManifest)
            && self.has(ClientReadiness::ResourcePayload)
    }

    fn interactive_ready(&self) -> bool {
        self.resources_ready()
            && self.has(ClientReadiness::MapRenderer)
            && (!self.webview_required
                || (self.has(ClientReadiness::WebViewEnvironment)
                    && self.has(ClientReadiness::WebViewDocument)))
    }
}

#[cfg(windows)]
const EMPTY_BROWSER_DOCUMENT: &str = "<!doctype html><html><head></head><body></body></html>";

#[cfg(windows)]
struct BrowserAssetServer {
    origin: String,
    assets: std::sync::Arc<std::sync::RwLock<BTreeMap<String, Vec<u8>>>>,
    next_document: u64,
}

#[cfg(windows)]
struct NativeMenuBar {
    root: Menu,
    commands: BTreeMap<MenuId, String>,
    signature: String,
}

#[cfg(windows)]
impl NativeMenuBar {
    fn from_ui(ui: &dm_dmf::UiState) -> Result<Option<Self>, String> {
        let tree = ui.tree();
        let Some(section) = tree
            .auxiliary
            .iter()
            .find(|section| section.id.starts_with("menu:"))
        else {
            return Ok(None);
        };
        let namespace = section.id.strip_prefix("menu:").unwrap_or(&section.id);
        #[derive(Clone)]
        struct Entry {
            id: Option<String>,
            name: String,
            command: String,
            category: Option<String>,
            parent: Option<String>,
            index: Option<i32>,
            order: usize,
        }
        let effective = |id: Option<&str>, property: &str, fallback: Option<&str>| {
            id.and_then(|id| ui.winget(&format!("{namespace}.{id}"), property).ok())
                .or_else(|| fallback.map(str::to_owned))
                .filter(|value| !value.is_empty())
        };
        let mut entries = section
            .controls
            .iter()
            .enumerate()
            .map(|(order, control)| Entry {
                id: control.id.clone(),
                name: effective(control.id.as_deref(), "name", control.property("name"))
                    .unwrap_or_default(),
                command: effective(
                    control.id.as_deref(),
                    "command",
                    control.property("command"),
                )
                .unwrap_or_default(),
                category: effective(
                    control.id.as_deref(),
                    "category",
                    control.property("category"),
                ),
                parent: effective(control.id.as_deref(), "parent", control.property("parent")),
                index: effective(control.id.as_deref(), "index", control.property("index"))
                    .and_then(|value| value.parse().ok()),
                order,
            })
            .collect::<Vec<_>>();
        let static_ids = entries
            .iter()
            .filter_map(|entry| entry.id.clone())
            .collect::<BTreeSet<_>>();
        for id in ui
            .section_control_ids(namespace)
            .map_err(|error| format!("enumerate DMF menu: {error:?}"))?
            .into_iter()
            .filter(|id| !static_ids.contains(id))
        {
            let address = format!("{namespace}.{id}");
            let property = |name| {
                ui.winget(&address, name)
                    .ok()
                    .filter(|value| !value.is_empty())
            };
            let order = entries.len();
            entries.push(Entry {
                id: Some(id),
                name: property("name").unwrap_or_default(),
                command: property("command").unwrap_or_default(),
                category: property("category"),
                parent: property("parent"),
                index: property("index").and_then(|value| value.parse().ok()),
                order,
            });
        }
        let signature = entries
            .iter()
            .map(|entry| {
                format!(
                    "{:?}|{}|{}|{:?}|{:?}|{:?}",
                    entry.id, entry.name, entry.command, entry.category, entry.parent, entry.index
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let root = Menu::new();
        let mut commands = BTreeMap::new();
        let mut categories = entries
            .iter()
            .filter(|entry| {
                entry.category.is_none()
                    && entry
                        .parent
                        .as_deref()
                        .is_none_or(|parent| parent.eq_ignore_ascii_case(namespace))
                    && !entry.name.is_empty()
            })
            .cloned()
            .collect::<Vec<_>>();
        categories.sort_by_key(|entry| (entry.index.unwrap_or(entry.order as i32), entry.order));
        for category in categories {
            let label = percent_decode_form(&category.name).unwrap_or(category.name.clone());
            let category_key = category.id.as_deref().unwrap_or(&category.name);
            let mut children = entries
                .iter()
                .filter(|entry| {
                    entry.category.as_deref() == Some(category.name.as_str())
                        || entry.parent.as_deref() == Some(category_key)
                })
                .cloned()
                .collect::<Vec<_>>();
            children.sort_by_key(|entry| (entry.index.unwrap_or(entry.order as i32), entry.order));
            if children.is_empty() && !category.command.is_empty() {
                let item = MenuItem::new(label, true, None);
                commands.insert(item.id().clone(), category.command);
                root.append(&item).map_err(|error| error.to_string())?;
                continue;
            }
            let submenu = Submenu::new(label, true);
            for entry in children {
                let label = percent_decode_form(&entry.name).unwrap_or(entry.name.clone());
                if label.is_empty() {
                    submenu
                        .append(&PredefinedMenuItem::separator())
                        .map_err(|error| error.to_string())?;
                    continue;
                }
                let item = MenuItem::new(label, true, None);
                if !entry.command.is_empty() {
                    commands.insert(item.id().clone(), entry.command);
                }
                submenu.append(&item).map_err(|error| error.to_string())?;
            }
            root.append(&submenu).map_err(|error| error.to_string())?;
        }
        Ok(Some(Self {
            root,
            commands,
            signature,
        }))
    }

    fn install(&self, window: &Window) -> Result<(), String> {
        let handle = window.window_handle().map_err(|error| error.to_string())?;
        let RawWindowHandle::Win32(handle) = handle.as_raw() else {
            return Err("native Windows menu requires a Win32 window".to_owned());
        };
        dm_native_menu::install_for_hwnd(&self.root, handle.hwnd.get())
    }

    fn drain_commands(&self) -> Vec<String> {
        let mut commands = Vec::new();
        while let Ok(event) = MenuEvent::receiver().try_recv() {
            if let Some(command) = self.commands.get(event.id()) {
                commands.push(command.clone());
            }
        }
        commands
    }
}

#[cfg(windows)]
impl BrowserAssetServer {
    fn start() -> Result<Self, String> {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| format!("bind browser asset server: {error}"))?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        let assets = std::sync::Arc::new(std::sync::RwLock::new(BTreeMap::new()));
        let server_assets = assets.clone();
        std::thread::Builder::new()
            .name("dream64-browser-assets".to_owned())
            .spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    let mut request = [0_u8; 8192];
                    let Ok(length) = stream.read(&mut request) else {
                        continue;
                    };
                    let first_line = String::from_utf8_lossy(&request[..length])
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .to_owned();
                    let target = first_line
                        .split_ascii_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .split('?')
                        .next()
                        .unwrap_or("/")
                        .trim_start_matches('/');
                    let path = percent_decode_form(target).ok();
                    let body = path.as_ref().and_then(|path| {
                        server_assets
                            .read()
                            .ok()
                            .and_then(|assets| assets.get(path).cloned())
                    });
                    let (status, mime, body) = body.map_or_else(
                        || ("404 Not Found", "text/plain", b"not found".to_vec()),
                        |body| ("200 OK", browser_mime_type(path.as_deref().unwrap_or("")), body),
                    );
                    let header = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(header.as_bytes());
                    let _ = stream.write_all(&body);
                }
            })
            .map_err(|error| format!("start browser asset server: {error}"))?;
        Ok(Self {
            origin: format!("http://127.0.0.1:{}", address.port()),
            assets,
            next_document: 1,
        })
    }

    fn insert(&self, name: String, data: Vec<u8>) {
        if let Ok(mut assets) = self.assets.write() {
            assets.insert(name, data);
        }
    }

    fn contains(&self, name: &str) -> bool {
        self.assets
            .read()
            .is_ok_and(|assets| assets.contains_key(name))
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.origin, encode_url_path(path))
    }

    fn publish_document(&mut self, html: &str) -> String {
        let name = format!("browse_{}.html", self.next_document);
        self.next_document = self.next_document.saturating_add(1);
        self.insert(name.clone(), html.as_bytes().to_vec());
        self.url(&name)
    }
}

#[cfg(windows)]
fn encode_url_path(path: &str) -> String {
    let mut encoded = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.' | b'~') {
            encoded.push(char::from(byte));
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

#[cfg(windows)]
fn browser_mime_type(path: &str) -> &'static str {
    match path
        .rsplit('.')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "css" => "text/css",
        "html" | "htm" => "text/html; charset=utf-8",
        "js" => "application/javascript",
        "json" => "application/json",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        "jpg" | "jpeg" => "image/jpeg",
        "ogg" | "oga" => "audio/ogg",
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "mid" | "midi" => "audio/midi",
        "ttf" => "font/ttf",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "eot" => "application/vnd.ms-fontobject",
        _ => "application/octet-stream",
    }
}

/// Runs a visible local client using a supplied DMF skin or the development skin.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let launch = LaunchOptions::parse()?;
    let startup_snapshot_active = launch.startup_replay.is_some();
    let (skin_source, skin_label) = load_skin(launch.skin.as_deref())?;
    eprintln!("client-skin-loaded: {skin_label}");
    let (mut runtime, scene) = launch
        .world
        .as_deref()
        .map(|world| WorldScene::boot(world, launch.map.as_deref()))
        .transpose()?
        .map_or_else(
            || (ExecutionState::new(), None),
            |(runtime, scene)| (runtime, Some(scene)),
        );
    let event_loop = EventLoop::new()?;
    let context = Context::new(event_loop.owned_display_handle())?;
    let client = runtime.open_local_client(&skin_source)?;
    runtime.set_local_client_interactive(client, true)?;
    let (startup_snapshot, startup_ui_events) = if let Some(path) = launch.startup_replay.as_deref()
    {
        let mut splash = RemoteTransport::replay(path)?;
        splash.attach()?;
        let snapshot = splash.request_snapshot()?;
        let mut events = Vec::new();
        loop {
            let mut batch = splash.poll_ui_events()?;
            if batch.is_empty() {
                break;
            }
            events.append(&mut batch);
        }
        (Some(snapshot), events)
    } else {
        (None, Vec::new())
    };
    let mut transport = if let Some(scene) = scene {
        let mut transport = LoopbackTransport::new(scene);
        transport.attach(&mut runtime, client)?;
        ClientTransport::Offline(transport)
    } else if let Some(path) = launch.replay.as_deref() {
        let mut transport = RemoteTransport::replay(path)?;
        transport.attach()?;
        ClientTransport::Remote(transport)
    } else {
        // Create the native DMF shell immediately. The live server may still
        // be running world/New and bringing Master/subsystems online; attach
        // asynchronously once its loopback endpoint begins servicing clients.
        ClientTransport::Pending(PendingRemoteTransport {
            address: launch.connect,
            record: launch.record,
            next_attempt: std::time::Instant::now(),
            last_error: None,
        })
    };
    // Offline playback has no server tick or socket activity to provoke a
    // follow-up redraw. Preload its authoritative frame before entering the
    // Windows event loop so a maximized Wry window cannot remain forever on
    // the first empty surface. Live connections retain the staged shell-first
    // path while world/New and Master are still booting.
    let initial_snapshot = match &mut transport {
        ClientTransport::Remote(transport) if transport.is_replay() => {
            Some(transport.request_snapshot()?)
        }
        _ => startup_snapshot,
    };
    let initial_snapshot_stage = if initial_snapshot.is_some() { 2 } else { 0 };
    let mut layout = ClientLayout::from_tree(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui()
            .tree(),
    );
    let mut ui_presentation = UiPresentation::default();
    let mut startup_browser_updates = Vec::new();
    for (sequence, command) in startup_ui_events {
        let Some(session) = runtime.client_session_mut(client) else {
            break;
        };
        // Browser updates are loaded again from the live stream. Applying the
        // saved DMF commands here is enough to establish the correct lobby
        // panes before the native window is ever shown.
        if let Ok(Some(update)) = ui_presentation.apply(sequence, command, session, &mut layout) {
            startup_browser_updates.push(update);
        }
    }
    let input_states = input_states_from_ui(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui(),
        &layout,
    );
    let button_states = button_states_from_ui(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui(),
        &layout,
    );
    let label_states = label_states_from_ui(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui(),
        &layout,
    );
    let focused_input = None;
    let macro_bindings = MacroBindings::from_tree(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui()
            .tree(),
    );
    let (browser_message_sender, browser_messages) = std::sync::mpsc::channel();
    #[cfg(windows)]
    let browser_assets = BrowserAssetServer::start()?;
    #[cfg(windows)]
    for (name, data) in &ui_presentation.browser_resources {
        browser_assets.insert(name.clone(), data.clone());
    }
    let readiness = ClientReadinessCoordinator::new();
    let ui_owner_commands = UiOwnerCommandQueue::new(readiness.generation, MAX_UI_OWNER_COMMANDS);
    let mut application = LocalClient {
        context,
        surface: None,
        runtime,
        client,
        local_input_events: 0,
        inbound_ui_events: 0,
        transport,
        readiness,
        ui_owner_commands,
        snapshot: initial_snapshot,
        sprites: SpriteCache::default(),
        layout,
        ui_presentation,
        macro_bindings,
        modifiers: ModifiersState::empty(),
        snapshot_stage: initial_snapshot_stage,
        hud_snapshot_refreshed: false,
        startup_snapshot_active,
        startup_snapshot_visible: !startup_snapshot_active,
        next_startup_snapshot_refresh: std::time::Instant::now(),
        deferred_live_ui: Vec::new(),
        cursor_position: None,
        last_map_click: None,
        hovered_screen: None,
        dragging_main_splitter: false,
        next_screen_refresh: None,
        input_states,
        focused_input,
        button_states,
        label_states,
        active_prompt: None,
        pending_screenshot: None,
        startup_browser_updates,
        browser_message_sender,
        browser_messages,
        browser_generation: Arc::new(AtomicU64::new(0)),
        #[cfg(windows)]
        browsers: BTreeMap::new(),
        #[cfg(windows)]
        ready_browsers: BTreeSet::new(),
        #[cfg(windows)]
        pending_browser_scripts: BTreeMap::new(),
        #[cfg(windows)]
        native_menu: None,
        #[cfg(windows)]
        browser_assets,
    };
    event_loop.run_app(&mut application)?;
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct MacroBindings {
    sets: BTreeMap<String, BTreeMap<String, (String, String)>>,
}

impl MacroBindings {
    fn from_tree(tree: &ControlTree) -> Self {
        let mut bindings = Self::default();
        for section in &tree.auxiliary {
            let Some(set) = section.id.strip_prefix("macro:") else {
                continue;
            };
            for control in &section.controls {
                let Some(id) = control.id.as_deref() else {
                    continue;
                };
                let property = |name: &str| {
                    control
                        .properties
                        .iter()
                        .rev()
                        .find(|property| property.key.eq_ignore_ascii_case(name))
                        .map(|property| property.value.decoded.clone())
                };
                if let (Some(name), Some(command)) = (property("name"), property("command")) {
                    bindings
                        .sets
                        .entry(set.to_ascii_lowercase())
                        .or_default()
                        .insert(
                            id.to_ascii_lowercase(),
                            (name.to_ascii_uppercase(), command),
                        );
                }
            }
        }
        bindings
    }

    fn refresh_control(&mut self, ui: &dm_dmf::UiState, control: &str) {
        let control_id = control.rsplit('.').next().unwrap_or(control);
        let parent = ui
            .winget(control, "parent")
            .ok()
            .filter(|value| !value.is_empty());
        let existing_set = self.sets.iter().find_map(|(set, controls)| {
            controls
                .contains_key(&control_id.to_ascii_lowercase())
                .then(|| set.clone())
        });
        let Some(set) = parent.or(existing_set) else {
            return;
        };
        let Ok(name) = ui.winget(control, "name") else {
            return;
        };
        let Ok(command) = ui.winget(control, "command") else {
            return;
        };
        self.sets
            .entry(set.trim_start_matches("macro:").to_ascii_lowercase())
            .or_default()
            .insert(
                control_id.to_ascii_lowercase(),
                (name.to_ascii_uppercase(), command),
            );
    }

    fn command(&self, ui: &dm_dmf::UiState, key: &str) -> Option<String> {
        let active = active_macro_set(ui);
        self.sets
            .get(&active)?
            .values()
            .find_map(|(name, command)| name.eq_ignore_ascii_case(key).then(|| command.clone()))
    }
}

fn active_macro_set(ui: &dm_dmf::UiState) -> String {
    let main = ui.tree().windows.iter().find_map(|window| {
        window.controls.iter().find_map(|control| {
            (control.control_type == ControlType::Main)
                .then(|| {
                    control
                        .id
                        .as_deref()
                        .map(|id| format!("{}.{}", window.id, id))
                })
                .flatten()
        })
    });
    main.and_then(|control| ui.winget(&control, "macro").ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "default".to_owned())
        .trim_start_matches("macro:")
        .to_ascii_lowercase()
}

fn macro_key_name(code: KeyCode, modifiers: ModifiersState, pressed: bool) -> Option<String> {
    let key = match code {
        KeyCode::ShiftLeft | KeyCode::ShiftRight => "SHIFT".to_owned(),
        KeyCode::ControlLeft | KeyCode::ControlRight => "CTRL".to_owned(),
        KeyCode::AltLeft | KeyCode::AltRight => "ALT".to_owned(),
        KeyCode::ArrowUp => "NORTH".to_owned(),
        KeyCode::ArrowDown => "SOUTH".to_owned(),
        KeyCode::ArrowLeft => "WEST".to_owned(),
        KeyCode::ArrowRight => "EAST".to_owned(),
        KeyCode::Space => "SPACE".to_owned(),
        KeyCode::Enter => "RETURN".to_owned(),
        KeyCode::Escape => "ESCAPE".to_owned(),
        KeyCode::Tab => "TAB".to_owned(),
        _ => {
            let debug = format!("{code:?}");
            if let Some(letter) = debug.strip_prefix("Key") {
                letter.to_owned()
            } else if let Some(digit) = debug.strip_prefix("Digit") {
                digit.to_owned()
            } else {
                return None;
            }
        }
    };
    let is_modifier = matches!(
        code,
        KeyCode::ShiftLeft
            | KeyCode::ShiftRight
            | KeyCode::ControlLeft
            | KeyCode::ControlRight
            | KeyCode::AltLeft
            | KeyCode::AltRight
    );
    let mut parts = Vec::new();
    if !is_modifier {
        if modifiers.control_key() {
            parts.push("CTRL".to_owned());
        }
        if modifiers.alt_key() {
            parts.push("ALT".to_owned());
        }
        if modifiers.shift_key() {
            parts.push("SHIFT".to_owned());
        }
    }
    parts.push(key);
    if !pressed {
        parts.push("UP".to_owned());
    }
    Some(parts.join("+"))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MacroDispatch {
    server_command: Option<String>,
}

fn dispatch_macro(
    session: &mut dm_dmf::ClientSession,
    bindings: &MacroBindings,
    key: &str,
) -> Option<MacroDispatch> {
    let Some(command) = bindings.command(session.ui(), key) else {
        return None;
    };
    Some(dispatch_ui_command(session, command))
}

fn dispatch_ui_command(session: &mut dm_dmf::ClientSession, command: String) -> MacroDispatch {
    if let Some(winset) = command.strip_prefix(".winset ") {
        let winset = winset.trim().trim_matches('"');
        if apply_conditional_winset(session, winset) {
            return MacroDispatch {
                server_command: None,
            };
        }
        let assignment = winset.split_once(char::is_whitespace).map_or_else(
            || {
                let (left, value) = winset.split_once('=')?;
                let property = left.rfind('.')?;
                Some((
                    &left[..property],
                    format!("{}={value}", &left[property + 1..]),
                ))
            },
            |(control, parameters)| Some((control, parameters.to_owned())),
        );
        if let Some((control, parameters)) = assignment {
            let control = if let Some(short) = control.strip_prefix(':') {
                resolve_control_type(session.ui().tree(), short, ControlType::Map)
                    .unwrap_or_else(|_| short.to_owned())
            } else {
                control.to_owned()
            };
            let _ = session.apply_command(UiCommand::WinSet {
                control,
                parameters,
            });
            return MacroDispatch {
                server_command: None,
            };
        }
    }
    session.push_event(UiEvent::Command {
        command: command.clone(),
    });
    MacroDispatch {
        server_command: (!command.starts_with('.')).then_some(command),
    }
}

fn dispatch_button_command(
    session: &mut dm_dmf::ClientSession,
    control: &str,
    command: String,
) -> MacroDispatch {
    let pushbox = session
        .ui()
        .winget(control, "button-type")
        .is_ok_and(|value| value.eq_ignore_ascii_case("pushbox"));
    if pushbox {
        let checked = session
            .ui()
            .winget(control, "is-checked")
            .is_ok_and(|value| dmf_truthy(&value));
        let _ = session.apply_command(UiCommand::WinSet {
            control: control.to_owned(),
            parameters: format!("is-checked={}", !checked),
        });
    }
    dispatch_ui_command(session, command)
}

/// Applies BYOND's conditional winset form used by pushbox controls:
/// `control.property=value ? target.property=value : target.property=value`.
fn apply_conditional_winset(session: &mut dm_dmf::ClientSession, source: &str) -> bool {
    let mut applied = false;
    let mut remaining = source.trim();
    while let Some(question) = remaining.find('?') {
        let condition = remaining[..question].trim().trim_matches('"');
        let after_question = remaining[question + 1..].trim();
        let next = double_quote_separator(after_question);
        let (branches, rest) = next.map_or((after_question, ""), |index| {
            (&after_question[..index], &after_question[index + 2..])
        });
        let (if_true, if_false) = branches
            .split_once(':')
            .map_or((branches.trim(), ""), |(if_true, if_false)| {
                (if_true.trim(), if_false.trim())
            });
        let condition_true = condition
            .split_once('=')
            .and_then(|(left, expected)| {
                let property = left.rfind('.')?;
                session
                    .ui()
                    .winget(&left[..property], &left[property + 1..])
                    .ok()
                    .map(|actual| actual.eq_ignore_ascii_case(expected.trim_matches('"')))
            })
            .unwrap_or(false);
        let selected = if condition_true { if_true } else { if_false };
        if let Some((left, value)) = selected.trim().split_once('=')
            && let Some(property) = left.rfind('.')
        {
            let value = decode_conditional_winset_value(value);
            let _ = session.apply_command(UiCommand::WinSet {
                control: left[..property].trim().trim_matches('"').to_owned(),
                parameters: format!(
                    "{}=\"{}\"",
                    left[property + 1..].trim(),
                    value.replace('\\', "\\\\").replace('"', "\\\"")
                ),
            });
            applied = true;
        }
        remaining = rest.trim();
        if remaining.is_empty() {
            break;
        }
    }
    applied
}

fn decode_conditional_winset_value(source: &str) -> String {
    let mut value = source.trim();
    if let Some(unquoted) = value.strip_prefix('"') {
        value = unquoted;
    }
    let mut decoded = value.replace("\\\"", "\"").replace("\\\\", "\\");
    if decoded.ends_with('"') && !value.ends_with("\\\"") {
        decoded.pop();
    }
    decoded
}

fn double_quote_separator(source: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    (0..bytes.len().saturating_sub(1)).find(|index| {
        bytes[*index] == b'"'
            && bytes[*index + 1] == b'"'
            && (*index == 0 || bytes[*index - 1] != b'\\')
    })
}

#[cfg(test)]
fn materialize_browser_resources(
    html: &str,
    resources: &BTreeMap<String, Vec<u8>>,
    cache: &mut BTreeMap<String, String>,
) -> String {
    let mut stack = Vec::new();
    let html = rewrite_quoted_attribute(html, "src", "", resources, cache, &mut stack);
    let html = rewrite_quoted_attribute(&html, "href", "", resources, cache, &mut stack);
    rewrite_css_urls(&html, "", resources, cache, &mut stack)
}

#[cfg(test)]
fn rewrite_quoted_attribute(
    source: &str,
    attribute: &str,
    base: &str,
    resources: &BTreeMap<String, Vec<u8>>,
    cache: &mut BTreeMap<String, String>,
    stack: &mut Vec<String>,
) -> String {
    let needle = format!("{attribute}=");
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0;
    while let Some(offset) = find_ascii_case_insensitive(&source[cursor..], &needle) {
        let start = cursor + offset;
        let value_start = start + needle.len();
        output.push_str(&source[cursor..value_start]);
        let Some(quote) = source[value_start..]
            .chars()
            .next()
            .filter(|value| matches!(value, '\'' | '"'))
        else {
            cursor = value_start;
            continue;
        };
        output.push(quote);
        let content_start = value_start + quote.len_utf8();
        let Some(end_offset) = source[content_start..].find(quote) else {
            cursor = content_start;
            continue;
        };
        let content_end = content_start + end_offset;
        let reference = &source[content_start..content_end];
        output.push_str(
            &resource_data_uri(reference, base, resources, cache, stack)
                .unwrap_or_else(|| reference.to_owned()),
        );
        output.push(quote);
        cursor = content_end + quote.len_utf8();
    }
    output.push_str(&source[cursor..]);
    output
}

#[cfg(test)]
fn rewrite_css_urls(
    source: &str,
    base: &str,
    resources: &BTreeMap<String, Vec<u8>>,
    cache: &mut BTreeMap<String, String>,
    stack: &mut Vec<String>,
) -> String {
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0;
    while let Some(offset) = find_ascii_case_insensitive(&source[cursor..], "url(") {
        let start = cursor + offset;
        output.push_str(&source[cursor..start + 4]);
        let mut content_start = start + 4;
        while source
            .as_bytes()
            .get(content_start)
            .is_some_and(u8::is_ascii_whitespace)
        {
            output.push(source.as_bytes()[content_start] as char);
            content_start += 1;
        }
        let quote = source
            .as_bytes()
            .get(content_start)
            .copied()
            .filter(|value| matches!(*value, b'\'' | b'"'));
        if let Some(quote) = quote {
            output.push(quote as char);
            content_start += 1;
        }
        let terminator = quote.unwrap_or(b')') as char;
        let Some(end_offset) = source[content_start..].find(terminator) else {
            cursor = content_start;
            continue;
        };
        let content_end = content_start + end_offset;
        let reference = source[content_start..content_end].trim();
        output.push_str(
            &resource_data_uri(reference, base, resources, cache, stack)
                .unwrap_or_else(|| reference.to_owned()),
        );
        if let Some(quote) = quote {
            output.push(quote as char);
            let after_quote = content_end + 1;
            let Some(close_offset) = source[after_quote..].find(')') else {
                cursor = after_quote;
                continue;
            };
            output.push_str(&source[after_quote..after_quote + close_offset + 1]);
            cursor = after_quote + close_offset + 1;
        } else {
            output.push(')');
            cursor = content_end + 1;
        }
    }
    output.push_str(&source[cursor..]);
    output
}

#[cfg(test)]
fn resource_data_uri(
    reference: &str,
    base: &str,
    resources: &BTreeMap<String, Vec<u8>>,
    cache: &mut BTreeMap<String, String>,
    stack: &mut Vec<String>,
) -> Option<String> {
    let path = normalize_resource_path(base, reference)?;
    if let Some(uri) = cache.get(&path) {
        return Some(uri.clone());
    }
    if stack.iter().any(|entry| entry == &path) {
        return None;
    }
    let bytes = resources.get(&path)?;
    stack.push(path.clone());
    let mime = resource_mime(&path);
    let payload = if mime == "text/css" {
        let Ok(css) = std::str::from_utf8(bytes) else {
            stack.pop();
            return None;
        };
        let parent = Path::new(&path)
            .parent()
            .and_then(Path::to_str)
            .unwrap_or("")
            .replace('\\', "/");
        rewrite_css_urls(css, &parent, resources, cache, stack).into_bytes()
    } else {
        bytes.clone()
    };
    stack.pop();
    let uri = format!("data:{mime};base64,{}", encode_base64(&payload));
    cache.insert(path, uri.clone());
    Some(uri)
}

pub(crate) fn normalize_resource_path(base: &str, reference: &str) -> Option<String> {
    let reference = reference.trim().replace('\\', "/");
    if reference.is_empty()
        || reference.starts_with(['/', '#'])
        || reference.contains(':')
        || reference.contains(['?', '#'])
    {
        return None;
    }
    let mut parts = base
        .split('/')
        .filter(|part| !part.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    for part in reference.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            _ => parts.push(part.to_owned()),
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

#[cfg(test)]
fn resource_mime(path: &str) -> &'static str {
    match Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => "image/png",
        "gif" => "image/gif",
        "jpg" | "jpeg" => "image/jpeg",
        "svg" => "image/svg+xml",
        "css" => "text/css",
        "js" => "text/javascript",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "html" | "htm" => "text/html",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
fn find_ascii_case_insensitive(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .as_bytes()
        .windows(needle.len())
        .position(|window| window.eq_ignore_ascii_case(needle.as_bytes()))
}

#[cfg(test)]
fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let bits = (u32::from(chunk[0]) << 16)
            | (u32::from(*chunk.get(1).unwrap_or(&0)) << 8)
            | u32::from(*chunk.get(2).unwrap_or(&0));
        encoded.push(TABLE[((bits >> 18) & 63) as usize] as char);
        encoded.push(TABLE[((bits >> 12) & 63) as usize] as char);
        encoded.push(if chunk.len() > 1 {
            TABLE[((bits >> 6) & 63) as usize] as char
        } else {
            '='
        });
        encoded.push(if chunk.len() > 2 {
            TABLE[(bits & 63) as usize] as char
        } else {
            '='
        });
    }
    encoded
}

pub(crate) fn resolve_control_type(
    tree: &ControlTree,
    address: &str,
    expected: ControlType,
) -> Result<String, String> {
    let matches = tree
        .windows
        .iter()
        .flat_map(|window| {
            window.controls.iter().filter_map(move |node| {
                let id = node.id.as_deref()?;
                ((address == id || address == format!("{}.{}", window.id, id))
                    && node.control_type == expected)
                    .then(|| format!("{}.{}", window.id, id))
            })
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("{address} is not a {expected:?} control")),
        _ => Err(format!("ambiguous {expected:?} control {address}")),
    }
}

fn load_skin(path: Option<&Path>) -> Result<(String, String), std::io::Error> {
    let Some(path) = path else {
        return Ok((LOCAL_SKIN.to_owned(), "development skin".to_owned()));
    };
    let source = std::fs::read_to_string(path)?;
    Ok((source, path.display().to_string()))
}


struct LocalClient {
    context: Context<OwnedDisplayHandle>,
    surface: Option<ClientSurface>,
    runtime: ExecutionState,
    client: dm_value::DatumId,
    local_input_events: usize,
    inbound_ui_events: usize,
    transport: ClientTransport,
    readiness: ClientReadinessCoordinator,
    ui_owner_commands: UiOwnerCommandQueue<UiOwnerCommand>,
    snapshot: Option<MapSnapshot>,
    sprites: SpriteCache,
    layout: ClientLayout,
    ui_presentation: UiPresentation,
    macro_bindings: MacroBindings,
    modifiers: ModifiersState,
    snapshot_stage: u8,
    hud_snapshot_refreshed: bool,
    startup_snapshot_active: bool,
    startup_snapshot_visible: bool,
    next_startup_snapshot_refresh: std::time::Instant,
    deferred_live_ui: Vec<(u64, InboundUiCommand)>,
    cursor_position: Option<(u32, u32)>,
    last_map_click: Option<WorldCoordinate>,
    hovered_screen: Option<(u32, u32)>,
    dragging_main_splitter: bool,
    next_screen_refresh: Option<std::time::Instant>,
    input_states: BTreeMap<String, InputState>,
    focused_input: Option<String>,
    button_states: BTreeMap<String, ButtonState>,
    label_states: BTreeMap<String, LabelState>,
    active_prompt: Option<ClientPrompt>,
    pending_screenshot: Option<PathBuf>,
    startup_browser_updates: Vec<BrowserUpdate>,
    browser_message_sender: std::sync::mpsc::Sender<(u64, String, String)>,
    browser_messages: std::sync::mpsc::Receiver<(u64, String, String)>,
    browser_generation: Arc<AtomicU64>,
    #[cfg(windows)]
    browsers: BTreeMap<String, WebView>,
    #[cfg(windows)]
    ready_browsers: BTreeSet<String>,
    #[cfg(windows)]
    pending_browser_scripts: BTreeMap<String, Vec<String>>,
    #[cfg(windows)]
    native_menu: Option<NativeMenuBar>,
    #[cfg(windows)]
    browser_assets: BrowserAssetServer,
}

enum ClientSurface {
    Gpu(Box<gpu::GpuRenderer>),
    Cpu(Surface<OwnedDisplayHandle, Arc<Window>>),
}

impl ClientSurface {
    const fn is_gpu(&self) -> bool {
        matches!(self, Self::Gpu(_))
    }

    fn window(&self) -> &Window {
        match self {
            Self::Gpu(renderer) => renderer.window(),
            Self::Cpu(surface) => surface.window(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        match self {
            Self::Gpu(renderer) => {
                renderer.resize(width, height);
                Ok(())
            }
            Self::Cpu(surface) => {
                let Some(width) = NonZeroU32::new(width) else {
                    return Ok(());
                };
                let Some(height) = NonZeroU32::new(height) else {
                    return Ok(());
                };
                surface
                    .resize(width, height)
                    .map_err(|error| format!("resize CPU surface: {error}"))
            }
        }
    }

    fn present(
        &mut self,
        pixels: &[u32],
        dmi_sprites: &[gpu::DmiSpriteDraw],
        sprites: &[gpu::SpriteDraw],
    ) -> Result<(), String> {
        match self {
            Self::Gpu(renderer) => renderer.present(pixels, dmi_sprites, sprites),
            Self::Cpu(surface) => {
                let mut buffer = surface
                    .buffer_mut()
                    .map_err(|error| format!("draw CPU surface: {error}"))?;
                if buffer.len() != pixels.len() {
                    return Err(format!(
                        "CPU framebuffer has {} pixels; expected {}",
                        buffer.len(),
                        pixels.len()
                    ));
                }
                buffer.copy_from_slice(pixels);
                buffer
                    .present()
                    .map_err(|error| format!("present CPU surface: {error}"))
            }
        }
    }
}

impl LocalClient {
    fn main_splitter_hit(&self, point: (u32, u32)) -> bool {
        let layout = self.effective_layout();
        let divider = layout.map.x.saturating_add(layout.map.width);
        point.1 < layout.map.y.saturating_add(layout.map.height) && point.0.abs_diff(divider) <= 6
    }

    fn set_main_splitter_from_pointer(&mut self, pointer_x: u32) {
        let Some(surface) = &self.surface else {
            return;
        };
        let width = surface.window().inner_size().width.max(1);
        let percent = pointer_x
            .saturating_mul(100)
            .checked_div(width)
            .unwrap_or(50)
            .clamp(20, 80);
        let Some(session) = self.runtime.client_session_mut(self.client) else {
            return;
        };
        if let Err(error) = session.apply_command(UiCommand::WinSet {
            control: "mainwindow.split".to_owned(),
            parameters: format!("splitter={percent}"),
        }) {
            eprintln!("client-splitter-error: {error:?}");
            return;
        }
        self.layout.refresh_from_ui(session.ui());
        #[cfg(windows)]
        self.sync_browser_layout();
        if let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
    }

    fn drain_browser_messages(&mut self) {
        while let Ok((generation, control, message)) = self.browser_messages.try_recv() {
            if generation != self.readiness.generation {
                continue;
            }
            let Ok(message) = serde_json::from_str::<serde_json::Value>(&message) else {
                eprintln!("client-browser-message-error: invalid JSON");
                continue;
            };
            match message.get("kind").and_then(serde_json::Value::as_str) {
                Some("byond") => {
                    let Some(url) = message.get("url").and_then(serde_json::Value::as_str) else {
                        eprintln!("client-browser-message-error: BYOND URL is missing");
                        continue;
                    };
                    if let Err(error) = self.handle_browser_byond_url(&control, url) {
                        eprintln!(
                            "client-browser-call-error: control={control} url={url:?} error={error}"
                        );
                    }
                }
                Some("topic") => {
                    let Some(topic) = message.get("topic").and_then(serde_json::Value::as_str)
                    else {
                        eprintln!("client-browser-message-error: topic payload is missing");
                        continue;
                    };
                    if let Err(error) = self.transport.send_browser_topic(topic) {
                        eprintln!("client-browser-topic-error: {error}");
                    } else {
                        self.schedule_screen_refresh();
                    }
                }
                Some("ready") => {
                    #[cfg(windows)]
                    self.ready_browsers.insert(control.clone());
                    if control != AUDIO_BROWSER_CONTROL {
                        if let Err(error) = self
                            .readiness
                            .mark(generation, ClientReadiness::WebViewDocument)
                        {
                            eprintln!("client-readiness: {error}");
                        }
                        self.publish_client_readiness();
                    } else {
                        self.readiness.audio_available = true;
                    }
                    // `pending_browser_scripts`/`browsers` hold WebView2 state
                    // and are themselves `#[cfg(windows)]`; draining them has to
                    // carry the same gate or this arm cannot compile off Windows.
                    #[cfg(windows)]
                    {
                        let scripts = self
                            .pending_browser_scripts
                            .remove(&control)
                            .unwrap_or_default();
                        if let Some(browser) = self.browsers.get(&control) {
                            for script in scripts {
                                if let Err(error) = browser.evaluate_script(&script) {
                                    eprintln!(
                                        "client-browser-script-error: control={control} error={error}"
                                    );
                                }
                            }
                        }
                    }
                }
                Some(kind) => eprintln!("client-browser-message-unsupported: {kind}"),
                None => eprintln!("client-browser-message-error: kind is missing"),
            }
        }
    }

    fn handle_browser_byond_url(&mut self, browser_control: &str, url: &str) -> Result<(), String> {
        let (path, parameters) = parse_byond_url(url)?;
        match path.as_str() {
            "winset" => {
                if let Some(command) = parameters.get("command") {
                    let session = self
                        .runtime
                        .client_session_mut(self.client)
                        .ok_or("browser client session is missing")?;
                    session.push_event(UiEvent::Command {
                        command: command.clone(),
                    });
                    return self.transport.send_command(command);
                }
                let id = parameters
                    .get("id")
                    .or_else(|| parameters.get("element"))
                    .filter(|id| !id.is_empty())
                    .map_or_else(|| browser_control.to_owned(), Clone::clone);
                let assignments = parameters
                    .iter()
                    .filter(|(name, _)| !matches!(name.as_str(), "id" | "element" | "callback"))
                    .map(|(name, value)| format!("{name}={}", quote_dmf_assignment(value)))
                    .collect::<Vec<_>>()
                    .join(";");
                let session = self
                    .runtime
                    .client_session_mut(self.client)
                    .ok_or("browser client session is missing")?;
                session
                    .apply_command(UiCommand::WinSet {
                        control: id,
                        parameters: assignments,
                    })
                    .map_err(|error| format!("browser winset failed: {error:?}"))?;
                self.layout.refresh_from_ui(session.ui());
                self.sync_input_states();
                #[cfg(windows)]
                self.sync_browser_layout();
                if let Some(surface) = &self.surface {
                    surface.window().request_redraw();
                }
                Ok(())
            }
            "winget" => {
                let id = parameters
                    .get("id")
                    .filter(|id| !id.is_empty())
                    .map_or(browser_control, String::as_str);
                let property = parameters.get("property").map_or("*", String::as_str);
                let callback = parameters
                    .get("callback")
                    .filter(|callback| valid_byond_callback(callback))
                    .ok_or("browser winget callback is invalid")?;
                let session = self
                    .runtime
                    .client_session(self.client)
                    .ok_or("browser client session is missing")?;
                let values = if property == "*" || property.is_empty() {
                    session
                        .ui()
                        .winget_all(id)
                        .map_err(|error| format!("browser winget failed: {error:?}"))?
                } else {
                    let names = property
                        .split(',')
                        .map(str::trim)
                        .filter(|name| !name.is_empty());
                    let mut values = BTreeMap::new();
                    for name in names {
                        values.insert(
                            name.to_owned(),
                            session
                                .ui()
                                .winget(id, name)
                                .map_err(|error| format!("browser winget failed: {error:?}"))?,
                        );
                    }
                    values
                };
                #[cfg(windows)]
                if let Some(browser) = self.browsers.get(browser_control) {
                    let value =
                        serde_json::to_string(&values).map_err(|error| error.to_string())?;
                    browser
                        .evaluate_script(&format!("{callback}({value});"))
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            }
            _ => self.transport.send_browser_topic(url),
        }
    }

    fn effective_layout(&self) -> ClientLayout {
        self.surface.as_ref().map_or_else(
            || self.layout.clone(),
            |surface| {
                let size = surface.window().inner_size();
                let mut layout = self.layout.clone();
                if let Some(session) = self.runtime.client_session(self.client) {
                    layout.apply_resolved_panes_in(session.ui(), Some((size.width, size.height)));
                }
                layout
            },
        )
    }

    fn apply_inbound_ui(&mut self) -> bool {
        let Ok(mut events) = self.transport.poll_ui_events() else {
            return false;
        };
        if self.startup_snapshot_active && self.transport.is_live() {
            self.deferred_live_ui.append(&mut events);
            return false;
        }
        if !self.startup_snapshot_active && !self.deferred_live_ui.is_empty() {
            self.deferred_live_ui.append(&mut events);
            events = std::mem::take(&mut self.deferred_live_ui);
        }
        let inbound_before = self.inbound_ui_events;
        let previous_layout = self.layout.clone();
        let mut acknowledged_sequence = None;
        for (sequence, command) in events {
            let browser_resource = match &command {
                InboundUiCommand::BrowseResource { name, data } => {
                    normalize_resource_path("", name).map(|name| (name, data.clone()))
                }
                _ => None,
            };
            let changed_macro = match &command {
                InboundUiCommand::WinSet { control, .. } => Some(control.clone()),
                _ => None,
            };
            let Some(session) = self.runtime.client_session_mut(self.client) else {
                break;
            };
            let applied =
                match self
                    .ui_presentation
                    .apply(sequence, command, session, &mut self.layout)
                {
                    Ok(Some(update)) => {
                        let generation = self.readiness.generation;
                        if self
                            .ui_owner_commands
                            .push(generation, UiOwnerCommand::BrowserCommit(update))
                            .is_err()
                        {
                            eprintln!("client-ui-owner-queue-full: command=browser");
                            break;
                        }
                        true
                    }
                    Ok(None) => true,
                    Err(error) => {
                        eprintln!("client-ui: rejected event {sequence}: {error}");
                        false
                    }
                };
            self.inbound_ui_events = self.inbound_ui_events.saturating_add(1);
            if !applied {
                break;
            }
            acknowledged_sequence = Some(sequence);
            if let Some((name, data)) = browser_resource {
                let generation = self.readiness.generation;
                if self
                    .ui_owner_commands
                    .push(generation, UiOwnerCommand::ResourceCommit { name, data })
                    .is_err()
                {
                    eprintln!("client-ui-owner-queue-full: command=resource");
                    break;
                }
            }
            if let Some(control) = changed_macro
                && let Some(session) = self.runtime.client_session(self.client)
            {
                self.macro_bindings.refresh_control(session.ui(), &control);
            }
        }
        self.drain_ui_owner_commands();
        if let Some(sequence) = acknowledged_sequence {
            match self.transport.acknowledge_ui(sequence) {
                Ok(()) => {
                    self.mark_client_ready(ClientReadiness::ResourceManifest);
                    self.publish_client_readiness();
                }
                Err(error) => {
                    eprintln!("client-ui: acknowledgement {sequence} failed: {error}");
                }
            }
        }
        self.sync_input_states();
        if self.active_prompt.is_none() {
            self.active_prompt = self.ui_presentation.pending_prompts.pop_front();
            #[cfg(windows)]
            if self.active_prompt.is_some() {
                self.sync_browser_layout();
            }
        }
        if self.layout != previous_layout {
            if let Some(surface) = &self.surface {
                if (self.layout.window_width, self.layout.window_height)
                    != (previous_layout.window_width, previous_layout.window_height)
                {
                    let _ = surface
                        .window()
                        .request_inner_size(winit::dpi::LogicalSize::new(
                            f64::from(self.layout.window_width),
                            f64::from(self.layout.window_height),
                        ));
                }
                surface.window().request_redraw();
            }
            #[cfg(windows)]
            self.sync_browser_layout();
        }
        if self.inbound_ui_events != inbound_before
            && let Some(surface) = &self.surface
        {
            surface.window().set_title(&self.title());
        }
        self.inbound_ui_events != inbound_before
    }

    fn sync_input_states(&mut self) {
        let Some(session) = self.runtime.client_session(self.client) else {
            return;
        };
        let desired = input_states_from_ui(session.ui(), &self.layout);
        self.input_states
            .retain(|address, _| desired.contains_key(address));
        for (address, desired) in desired {
            match self.input_states.get_mut(&address) {
                Some(current) if current.command == desired.command => {}
                Some(current) => *current = desired,
                None => {
                    self.input_states.insert(address, desired);
                }
            }
        }
        self.button_states = button_states_from_ui(session.ui(), &self.layout);
        self.label_states = label_states_from_ui(session.ui(), &self.layout);
        if self
            .focused_input
            .as_ref()
            .is_some_and(|address| !self.input_states.contains_key(address))
        {
            self.focused_input = None;
        }
    }

    fn submit_focused_input(&mut self) {
        let Some(address) = self.focused_input.clone() else {
            return;
        };
        let no_command = self
            .runtime
            .client_session(self.client)
            .and_then(|session| session.ui().winget(&address, "no-command").ok())
            .is_some_and(|value| dmf_truthy(&value));
        let Some(state) = self.input_states.get_mut(&address) else {
            return;
        };
        if no_command {
            return;
        }
        let command = take_input_submission(state);
        if command.trim().is_empty() {
            return;
        }
        if let Some(session) = self.runtime.client_session_mut(self.client) {
            session.push_event(UiEvent::Command {
                command: command.clone(),
            });
        }
        self.local_input_events = self.local_input_events.saturating_add(1);
        if let Err(error) = self.transport.send_command(&command) {
            eprintln!("client-command-error: command={command:?} error={error}");
        }
    }

    fn handle_input_key(&mut self, event: &winit::event::KeyEvent) -> bool {
        if self.focused_input.is_none() {
            return false;
        }
        if event.state != ElementState::Pressed {
            return true;
        }
        match &event.logical_key {
            Key::Named(NamedKey::Enter) => self.submit_focused_input(),
            Key::Named(NamedKey::Backspace) => {
                if let Some(state) = self
                    .focused_input
                    .as_ref()
                    .and_then(|address| self.input_states.get_mut(address))
                {
                    state.text.pop();
                }
            }
            Key::Named(NamedKey::Escape) => self.focused_input = None,
            _ if !self.modifiers.control_key() && !self.modifiers.alt_key() => {
                if let Some(text) = &event.text
                    && let Some(state) = self
                        .focused_input
                        .as_ref()
                        .and_then(|address| self.input_states.get_mut(address))
                {
                    state
                        .text
                        .extend(text.chars().filter(|character| !character.is_control()));
                }
            }
            _ => {}
        }
        if let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
        true
    }

    fn finish_active_prompt(&mut self, response: ClientPromptResponse) {
        let Some(id) = self.active_prompt.as_ref().map(|prompt| prompt.id) else {
            return;
        };
        match self.transport.send_prompt_response(id, response) {
            Ok(()) => {
                self.active_prompt = self.ui_presentation.pending_prompts.pop_front();
                #[cfg(windows)]
                self.sync_browser_layout();
                if let Some(surface) = &self.surface {
                    surface.window().request_redraw();
                }
            }
            Err(error) => eprintln!("client-prompt-response-error: id={id} error={error}"),
        }
    }

    fn active_prompt_accept_response(&self) -> Option<ClientPromptResponse> {
        self.active_prompt
            .as_ref()
            .and_then(|prompt| match prompt.kind {
                ClientPromptKind::List | ClientPromptKind::Alert => (!prompt.choices.is_empty())
                    .then_some(ClientPromptResponse::Choice(prompt.selected)),
                ClientPromptKind::Number => prompt
                    .edit
                    .trim()
                    .parse::<f32>()
                    .ok()
                    .filter(|value| value.is_finite())
                    .map(ClientPromptResponse::Number),
                _ => Some(ClientPromptResponse::Text(prompt.edit.clone())),
            })
    }

    fn handle_prompt_key(&mut self, event: &winit::event::KeyEvent) -> bool {
        if self.active_prompt.is_none() {
            return false;
        }
        if event.state != ElementState::Pressed {
            return true;
        }
        match &event.logical_key {
            Key::Named(NamedKey::Escape) => {
                if self
                    .active_prompt
                    .as_ref()
                    .is_some_and(|prompt| prompt.can_cancel)
                {
                    self.finish_active_prompt(ClientPromptResponse::Null);
                }
            }
            Key::Named(NamedKey::ArrowUp) => {
                if let Some(prompt) = &mut self.active_prompt
                    && !prompt.choices.is_empty()
                {
                    prompt.selected = prompt.selected.saturating_sub(1);
                }
            }
            Key::Named(NamedKey::ArrowDown) => {
                if let Some(prompt) = &mut self.active_prompt
                    && !prompt.choices.is_empty()
                {
                    prompt.selected = (prompt.selected + 1).min(prompt.choices.len() - 1);
                }
            }
            Key::Named(NamedKey::Backspace) => {
                if let Some(prompt) = &mut self.active_prompt
                    && !matches!(
                        prompt.kind,
                        ClientPromptKind::List | ClientPromptKind::Alert
                    )
                {
                    prompt.edit.pop();
                }
            }
            Key::Named(NamedKey::Enter) => {
                let response = self.active_prompt_accept_response();
                if let Some(response) = response {
                    self.finish_active_prompt(response);
                }
            }
            Key::Character(text) => {
                if let Some(prompt) = &mut self.active_prompt
                    && !matches!(
                        prompt.kind,
                        ClientPromptKind::List | ClientPromptKind::Alert
                    )
                    && !self.modifiers.control_key()
                    && !self.modifiers.alt_key()
                {
                    prompt.edit.push_str(text);
                }
            }
            _ => {}
        }
        if let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
        true
    }

    fn refresh_snapshot(&mut self) {
        if self.transport.is_live() && !self.startup_snapshot_active && self.hud_snapshot_refreshed
        {
            return;
        }
        match self.transport.request_snapshot() {
            Ok(snapshot) => {
                if self.startup_snapshot_active
                    && self.transport.is_live()
                    && !snapshot_has_lobby_screen(&snapshot)
                {
                    eprintln!(
                        "client-startup-snapshot-retained: live_screen={}",
                        snapshot.screen.len()
                    );
                    return;
                }
                if self.startup_snapshot_active && self.transport.is_live() {
                    self.startup_snapshot_active = false;
                    eprintln!("client-startup-snapshot-replaced: live lobby is available");
                }
                // The first accepted live snapshot is already taken after the
                // authoritative UI attached. Do not immediately request the
                // same 65k-cell world a second time for the HUD.
                self.hud_snapshot_refreshed = true;
                let appearance_count = snapshot.appearances.values().map(Vec::len).sum::<usize>();
                let screen_count = snapshot.screen.len();
                eprintln!(
                    "client-snapshot-ready: cells={} appearances={} screen={} resources={}",
                    snapshot.cells.len(),
                    appearance_count,
                    screen_count,
                    snapshot.resources.len()
                );
                eprintln!(
                    "client-screen-resources: {}",
                    snapshot
                        .resources
                        .keys()
                        .map(|path| path.to_string_lossy())
                        .collect::<Vec<_>>()
                        .join(" | ")
                );
                for screen in &snapshot.screen {
                    if screen.appearances.iter().any(|appearance| {
                        appearance
                            .resource
                            .to_string_lossy()
                            .contains("background_monke.dmi")
                    }) {
                        eprintln!(
                            "client-screen-background-entry: loc={:?} appearances={} insertion={}",
                            screen.screen_loc,
                            screen.appearances.len(),
                            screen.insertion
                        );
                    }
                }
                let generation = self.readiness.generation;
                if self
                    .ui_owner_commands
                    .push(generation, UiOwnerCommand::MapCommit(snapshot))
                    .is_err()
                {
                    eprintln!("client-ui-owner-queue-full: command=map");
                    return;
                }
                self.drain_ui_owner_commands();
                if self.transport.is_live() {
                    // A reconnect to an already-running server may receive no
                    // new UI event batch. The accepted snapshot itself proves
                    // that its resource payload is installed, so advance both
                    // readiness phases here instead of waiting forever for an
                    // acknowledgement that will never be generated.
                    self.mark_client_ready(ClientReadiness::ResourceManifest);
                    self.mark_client_ready(ClientReadiness::ResourcePayload);
                    self.publish_client_readiness();
                }
                if let Some(surface) = &self.surface {
                    surface.window().set_title(&self.title());
                    surface.window().request_redraw();
                }
            }
            Err(error) => {
                eprintln!("client-snapshot-error: {error}");
                if let Some(surface) = &self.surface {
                    surface
                        .window()
                        .set_title(&format!("{} — snapshot error: {error}", self.title()));
                }
            }
        }
    }

    fn drain_ui_owner_commands(&mut self) {
        while let Some(command) = self.ui_owner_commands.pop_current() {
            match command {
                UiOwnerCommand::MapCommit(snapshot) => self.snapshot = Some(snapshot),
                UiOwnerCommand::BrowserCommit(update) => {
                    #[cfg(windows)]
                    self.apply_browser_update(update);
                    #[cfg(not(windows))]
                    let _ = update;
                }
                UiOwnerCommand::ResourceCommit { name, data } => {
                    #[cfg(windows)]
                    self.browser_assets.insert(name, data);
                    #[cfg(not(windows))]
                    let _ = (name, data);
                }
            }
        }
    }

    fn schedule_screen_refresh(&mut self) {
        self.next_screen_refresh =
            Some(std::time::Instant::now() + std::time::Duration::from_millis(150));
    }

    fn refresh_screen_snapshot(&mut self) {
        match self.transport.request_screen_snapshot() {
            Ok(update) => {
                if let Some(snapshot) = &mut self.snapshot {
                    snapshot.screen = update.screen;
                    snapshot.resources.extend(update.resources);
                }
                if let Some(surface) = &self.surface {
                    surface.window().request_redraw();
                }
            }
            Err(error) => eprintln!("client-screen-snapshot-error: {error}"),
        }
    }

    #[cfg(windows)]
    fn webview_bounds(rect: PixelRect) -> Rect {
        Rect {
            position: LogicalPosition::new(f64::from(rect.x), f64::from(rect.y)).into(),
            size: LogicalSize::new(f64::from(rect.width), f64::from(rect.height)).into(),
        }
    }

    #[cfg(windows)]
    fn sync_native_menu(&mut self) {
        let Some(surface) = &self.surface else { return };
        let Some(session) = self.runtime.client_session(self.client) else {
            return;
        };
        let Ok(Some(menu)) = NativeMenuBar::from_ui(session.ui()) else {
            return;
        };
        if self
            .native_menu
            .as_ref()
            .is_some_and(|current| current.signature == menu.signature)
        {
            return;
        }
        match menu.install(surface.window()) {
            Ok(()) => {
                eprintln!("client-native-menu: synchronized runtime entries");
                self.native_menu = Some(menu);
            }
            Err(error) => eprintln!("client-native-menu-error: {error}"),
        }
    }

    #[cfg(windows)]
    fn visible_browser_rects(&self) -> BTreeMap<String, PixelRect> {
        if self.active_prompt.is_some() {
            return BTreeMap::new();
        }
        let Some(surface) = &self.surface else {
            return BTreeMap::new();
        };
        let Some(session) = self.runtime.client_session(self.client) else {
            return BTreeMap::new();
        };
        let size = surface.window().inner_size();
        let mut browsers = resolve_pane_layout_in(session.ui(), Some((size.width, size.height)))
            .controls
            .into_iter()
            .filter(|(address, rect)| {
                rect.width > 0
                    && rect.height > 0
                    && control_has_type(session.ui().tree(), address, ControlType::Browser)
            })
            .collect::<BTreeMap<_, _>>();
        let ui = session.ui();
        let root_window = ui
            .tree()
            .windows
            .iter()
            .find(|window| {
                window.controls.iter().any(|control| {
                    control.control_type == ControlType::Main
                        && control.property("is-default").is_some_and(dmf_truthy)
                })
            })
            .map(|window| window.id.as_str());
        let mut window_ids = ui
            .tree()
            .windows
            .iter()
            .map(|window| window.id.clone())
            .collect::<Vec<_>>();
        window_ids.extend(ui.cloned_window_ids());
        for window_id in window_ids {
            if root_window == Some(window_id.as_str()) {
                continue;
            }
            let main_address = if ui.winexists(&window_id) {
                window_id.clone()
            } else {
                format!("{window_id}.{window_id}")
            };
            if ui
                .winget(&main_address, "is-pane")
                .is_ok_and(|value| dmf_truthy(&value))
                || ui
                    .winget(&main_address, "is-visible")
                    .is_ok_and(|value| dmf_false(&value))
            {
                continue;
            }
            let source_size = ui
                .winget(&main_address, "size")
                .ok()
                .and_then(|value| parse_pair(&value, 'x'))
                .unwrap_or((800, 600));
            let overlay_width = source_size.0.min(size.width.saturating_mul(9) / 10).max(1);
            let overlay_height = source_size.1.min(size.height.saturating_mul(9) / 10).max(1);
            let overlay = PixelRect {
                x: size.width.saturating_sub(overlay_width) / 2,
                y: size.height.saturating_sub(overlay_height) / 2,
                width: overlay_width,
                height: overlay_height,
            };
            let Ok(control_ids) = ui.section_control_ids(&window_id) else {
                continue;
            };
            for control_id in control_ids {
                let address = format!("{window_id}.{control_id}");
                if !ui.winexists_type(&address).eq_ignore_ascii_case("BROWSER")
                    || ui
                        .winget(&address, "is-visible")
                        .is_ok_and(|value| dmf_false(&value))
                {
                    continue;
                }
                let local_pos = ui
                    .winget(&address, "pos")
                    .ok()
                    .and_then(|value| parse_pair(&value, ','))
                    .unwrap_or((0, 0));
                let local_size = ui
                    .winget(&address, "size")
                    .ok()
                    .and_then(|value| parse_pair(&value, 'x'))
                    .unwrap_or(source_size);
                browsers.insert(
                    address,
                    PixelRect {
                        x: overlay.x
                            + local_pos.0.saturating_mul(overlay.width) / source_size.0.max(1),
                        y: overlay.y
                            + local_pos.1.saturating_mul(overlay.height) / source_size.1.max(1),
                        width: local_size.0.saturating_mul(overlay.width) / source_size.0.max(1),
                        height: local_size.1.saturating_mul(overlay.height) / source_size.1.max(1),
                    },
                );
            }
        }
        browsers
    }

    #[cfg(windows)]
    fn sync_browser_layout(&self) {
        let visible_rects = self.visible_browser_rects();
        for (control, browser) in &self.browsers {
            let rect = visible_rects.get(control).copied();
            let visible = rect.is_some();
            let _ = browser.set_visible(visible);
            if let Some(rect) = rect {
                let _ = browser.set_bounds(Self::webview_bounds(rect));
            }
        }
    }

    #[cfg(windows)]
    fn ensure_browser(&mut self, control: &str) {
        if self.browsers.contains_key(control) {
            return;
        }
        let browser_window = self
            .surface
            .as_ref()
            .expect("the browser has a parent window")
            .window();
        let bounds = Self::webview_bounds(PixelRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        let browser = WebViewBuilder::new()
            .with_html(EMPTY_BROWSER_DOCUMENT)
            .with_bounds(bounds)
            .with_visible(false)
            .with_initialization_script(WEBVIEW2_BYOND_BRIDGE_BOOTSTRAP)
            .with_navigation_handler({
                let sender = self.browser_message_sender.clone();
                let control = control.to_owned();
                let generation = Arc::clone(&self.browser_generation);
                move |url| {
                    if url.to_ascii_lowercase().starts_with("byond://") {
                        let message = serde_json::json!({ "kind": "byond", "url": url });
                        let _ = sender.send((
                            generation.load(Ordering::Acquire),
                            control.clone(),
                            message.to_string(),
                        ));
                        false
                    } else {
                        true
                    }
                }
            })
            .with_ipc_handler({
                let sender = self.browser_message_sender.clone();
                let control = control.to_owned();
                let generation = Arc::clone(&self.browser_generation);
                move |request| {
                    let _ = sender.send((
                        generation.load(Ordering::Acquire),
                        control.clone(),
                        request.body().clone(),
                    ));
                }
            })
            .build_as_child(&browser_window)
            .expect("WebView2 creates the Dream64 browser surface");
        browser
            .set_bounds(bounds)
            .expect("the browser applies its initial DMF rectangle");
        self.browsers.insert(control.to_owned(), browser);
        if control != AUDIO_BROWSER_CONTROL {
            self.readiness.webview_required = true;
            self.mark_client_ready(ClientReadiness::WebViewEnvironment);
            self.publish_client_readiness();
        } else {
            self.readiness.audio_available = true;
        }
        self.sync_browser_layout();
    }

    #[cfg(windows)]
    fn load_browser_html(&mut self, control: &str, html: &str) {
        self.ensure_browser(control);
        self.ready_browsers.remove(control);
        self.pending_browser_scripts.remove(control);
        let url = self.browser_assets.publish_document(html);
        if let Some(browser) = self.browsers.get(control) {
            if let Err(error) = browser.load_url(&url) {
                eprintln!("client-browser-navigation-error: control={control} error={error}");
            }
        }
    }

    #[cfg(windows)]
    fn load_browser_resource(&mut self, control: &str, path: &str) {
        self.ensure_browser(control);
        if !self.browser_assets.contains(path) {
            match self.transport.request_resource(path) {
                Ok(data) => self.browser_assets.insert(path.to_owned(), data),
                Err(error) => {
                    eprintln!(
                        "client-browser-resource-error: control={control} path={path:?} error={error}"
                    );
                    return;
                }
            }
        }
        self.ready_browsers.remove(control);
        self.pending_browser_scripts.remove(control);
        let url = self.browser_assets.url(path);
        if let Some(browser) = self.browsers.get(control)
            && let Err(error) = browser.load_url(&url)
        {
            eprintln!("client-browser-navigation-error: control={control} error={error}");
        }
    }

    #[cfg(windows)]
    fn execute_browser_script(&mut self, control: &str, script: &str) {
        self.ensure_browser(control);
        if !self.ready_browsers.contains(control) {
            self.pending_browser_scripts
                .entry(control.to_owned())
                .or_default()
                .push(script.to_owned());
            return;
        }
        if let Some(browser) = self.browsers.get(control)
            && let Err(error) = browser.evaluate_script(script)
        {
            eprintln!("client-browser-script-error: control={control} error={error}");
        }
    }

    #[cfg(windows)]
    fn apply_browser_update(&mut self, update: BrowserUpdate) {
        match update {
            BrowserUpdate::Link(url) => open_external_url(&url),
            BrowserUpdate::Html { control, html } => {
                self.load_browser_html(&control, &html);
            }
            BrowserUpdate::Resource { control, path } => {
                self.load_browser_resource(&control, &path);
            }
            BrowserUpdate::Script { control, script } => {
                self.execute_browser_script(&control, &script);
            }
            BrowserUpdate::Sound(sound) => self.apply_sound_update(&sound),
        }
    }

    #[cfg(windows)]
    fn apply_sound_update(&mut self, sound: &SoundUpdate) {
        let url = match sound.file.as_deref() {
            Some(path) => {
                let Some(path) = normalize_resource_path("", path) else {
                    eprintln!("client-sound-resource-error: invalid path {path:?}");
                    return;
                };
                if !self.browser_assets.contains(&path) {
                    match self.transport.request_resource(&path) {
                        Ok(data) => self.browser_assets.insert(path.clone(), data),
                        Err(error) => {
                            eprintln!("client-sound-resource-error: path={path:?} error={error}");
                            return;
                        }
                    }
                }
                Some(self.browser_assets.url(&path))
            }
            None => None,
        };
        self.ensure_browser(AUDIO_BROWSER_CONTROL);
        let url = serde_json::to_string(&url).expect("sound URL serializes");
        let channel = sound.channel;
        let repeat = sound.repeat;
        let volume = (sound.volume / 100.0).clamp(0.0, 1.0);
        let playback_rate = if sound.frequency > 0.0 {
            (sound.frequency / 44_100.0).clamp(0.0625, 16.0)
        } else {
            1.0
        };
        let pan = (sound.pan / 100.0).clamp(-1.0, 1.0);
        let script = format!(
            r#"(() => {{
const state = window.__dream64Audio ||= {{ context: null, channels: new Map() }};
const stop = entry => {{ if (!entry) return; entry.audio.pause(); entry.audio.src = ''; }};
const channel = {channel};
const url = {url};
if (url === null) {{
  if (channel === 0) {{ for (const entry of state.channels.values()) stop(entry); state.channels.clear(); }}
  else {{ stop(state.channels.get(channel)); state.channels.delete(channel); }}
  return;
}}
if (channel !== 0) {{ stop(state.channels.get(channel)); state.channels.delete(channel); }}
const audio = new Audio(url);
audio.loop = {repeat};
audio.volume = {volume};
audio.preservesPitch = false;
audio.playbackRate = {playback_rate};
const entry = {{ audio }};
try {{
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  state.context ||= new AudioContext();
  const source = state.context.createMediaElementSource(audio);
  const panner = state.context.createStereoPanner();
  panner.pan.value = {pan};
  source.connect(panner).connect(state.context.destination);
  entry.source = source; entry.panner = panner;
  state.context.resume();
}} catch (_) {{}}
if (channel !== 0) state.channels.set(channel, entry);
audio.addEventListener('ended', () => {{ if (channel !== 0 && state.channels.get(channel) === entry) state.channels.delete(channel); }});
audio.play().catch(error => console.error('Dream64 sound playback failed', error));
}})();"#
        );
        if let Some(browser) = self.browsers.get(AUDIO_BROWSER_CONTROL)
            && let Err(error) = browser.evaluate_script(&script)
        {
            eprintln!("client-sound-script-error: {error}");
        }
    }

    fn title(&self) -> String {
        self.runtime
            .client_session(self.client)
            .and_then(|session| {
                session.ui().tree().windows.iter().find_map(|window| {
                    let main = window.controls.iter().find(|control| {
                        control.control_type == ControlType::Main
                            && control.property("is-default").is_some_and(dmf_truthy)
                    })?;
                    let address = format!("{}.{}", window.id, main.id.as_deref()?);
                    session
                        .ui()
                        .winget(&address, "title")
                        .ok()
                        .filter(|title| !title.trim().is_empty())
                })
            })
            .unwrap_or_else(|| "Dream64".to_owned())
    }

    fn mark_client_ready(&mut self, readiness: ClientReadiness) {
        let generation = self.readiness.generation;
        if let Err(error) = self.readiness.mark(generation, readiness) {
            eprintln!("client-readiness: {error}");
        }
    }

    fn publish_client_readiness(&mut self) {
        if self.readiness.resources_ready() && !self.readiness.resources_sent {
            match self.transport.mark_resources_ready() {
                Ok(()) => self.readiness.resources_sent = true,
                Err(error) => eprintln!("client-resource-readiness-failed: {error}"),
            }
        }
        if self.readiness.interactive_ready() && !self.readiness.interactive_sent {
            match self.transport.mark_input_ready() {
                Ok(()) => self.readiness.interactive_sent = true,
                Err(error) => eprintln!("client-input-readiness-failed: {error}"),
            }
        }
    }

    fn resize(surface: &mut ClientSurface, width: u32, height: u32) {
        if let Err(error) = surface.resize(width, height) {
            eprintln!("client-surface-resize-error: {error}");
        }
    }

    fn redraw(
        surface: &mut ClientSurface,
        snapshot: Option<&MapSnapshot>,
        sprites: &mut SpriteCache,
        layout: ClientLayout,
        output_text: &BTreeMap<String, Vec<String>>,
        input_states: &BTreeMap<String, InputState>,
        focused_input: Option<&str>,
        button_states: &BTreeMap<String, ButtonState>,
        label_states: &BTreeMap<String, LabelState>,
        active_prompt: Option<&ClientPrompt>,
        transport_status: &str,
        screenshot_path: Option<&Path>,
    ) -> bool {
        if let Some(snapshot) = snapshot {
            for (path, bytes) in &snapshot.resources {
                sprites.insert(path.clone(), bytes);
            }
        }
        let size = surface.window().inner_size();
        let width = usize::try_from(size.width).expect("window width fits usize");
        let height = usize::try_from(size.height).expect("window height fits usize");
        let mut buffer = vec![0_u32; width.saturating_mul(height)];
        let mut gpu_sprites = Vec::new();
        let mut gpu_dmi_sprites = Vec::new();
        // Requested screenshots and diagnostic frame dumps are composed
        // through the CPU parity path so the PNG contains world and HUD
        // sprites, not only the base layer uploaded before the GPU batches.
        let frame_dump_requested = std::env::var_os("DREAM64_DUMP_FRAME").is_some();
        let gpu_enabled = surface.is_gpu() && screenshot_path.is_none() && !frame_dump_requested;
        // The supplied BYOND skin owns the full surface. Match OpenDream's
        // RobustToolbox window default and let resolved DMF controls paint it.
        buffer.fill(0xfff0f0f0);
        draw_map(
            &mut buffer,
            width,
            height,
            MapTransform::new(
                layout.map,
                layout.map_tile_size,
                layout.map_zoom,
                &layout.map_zoom_mode,
                layout.map_letterbox,
            ),
            snapshot,
            sprites,
            gpu_enabled.then_some((&mut gpu_dmi_sprites, &mut gpu_sprites)),
        );
        // Keep the DMF CHILD splitter visible on the native map side. The
        // browser pane is a child WebView and otherwise covers the boundary.
        let divider_x = layout.map.x.saturating_add(layout.map.width);
        draw_panel(
            &mut buffer,
            width,
            height,
            usize::try_from(divider_x.saturating_sub(3)).unwrap_or(0),
            usize::try_from(layout.map.y).unwrap_or(0),
            3,
            usize::try_from(layout.map.height).unwrap_or(0),
            0xff8a_8a8a,
        );
        if snapshot.is_none() {
            draw_boot_status(&mut buffer, width, height, layout.map, transport_status);
        }
        for (address, rect) in &layout.output_rects {
            draw_output_control(
                &mut buffer,
                width,
                height,
                *rect,
                output_text.get(address).map(Vec::as_slice).unwrap_or(&[]),
            );
        }
        for (address, rect) in &layout.input_rects {
            draw_input_control(
                &mut buffer,
                width,
                height,
                *rect,
                input_states
                    .get(address)
                    .map_or("", |state| state.text.as_str()),
                focused_input == Some(address.as_str()),
            );
        }
        for (address, rect) in &layout.button_rects {
            draw_button_control(
                &mut buffer,
                width,
                height,
                *rect,
                button_states
                    .get(address)
                    .map_or("", |button| button.text.as_str()),
                button_states
                    .get(address)
                    .is_some_and(|button| button.checked),
            );
        }
        for (address, rect) in &layout.label_rects {
            draw_label_control(
                &mut buffer,
                width,
                height,
                *rect,
                label_states
                    .get(address)
                    .map_or("", |label| label.text.as_str()),
            );
        }
        if let Some(prompt) = active_prompt {
            draw_client_prompt(&mut buffer, width, height, prompt);
        }
        if snapshot.is_some_and(|snapshot| !snapshot.screen.is_empty()) {
            maybe_dump_rendered_frame(&buffer, size.width, size.height);
        }
        if let Some(path) = screenshot_path {
            match write_rendered_frame(path, &buffer, size.width, size.height) {
                Ok(()) => eprintln!("client-screenshot: {}", path.display()),
                Err(error) => eprintln!("client-screenshot-error: {error}"),
            }
        }
        if let Err(error) = surface.present(&buffer, &gpu_dmi_sprites, &gpu_sprites) {
            eprintln!("client-surface-present-error: {error}");
            false
        } else {
            true
        }
    }
}

#[cfg(windows)]
fn open_external_url(url: &str) {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        eprintln!("client-link-error: unsupported URL {url:?}");
        return;
    }
    if let Err(error) = std::process::Command::new("rundll32.exe")
        .args(["url.dll,FileProtocolHandler", url])
        .spawn()
    {
        eprintln!("client-link-error: url={url:?} error={error}");
    }
}

fn maybe_dump_rendered_frame(buffer: &[u32], width: u32, height: u32) {
    static DUMPED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    let Ok(path) = std::env::var("DREAM64_DUMP_FRAME") else {
        return;
    };
    DUMPED.get_or_init(|| {
        let result = write_rendered_frame(Path::new(&path), buffer, width, height);
        match result {
            Ok(()) => eprintln!("client-frame-dump: {path}"),
            Err(error) => eprintln!("client-frame-dump-error: {error}"),
        }
    });
}

fn write_rendered_frame(
    path: &Path,
    buffer: &[u32],
    width: u32,
    height: u32,
) -> Result<(), String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("write PNG header: {error}"))?;
    let rgba = buffer
        .iter()
        .flat_map(|pixel| {
            let [blue, green, red, alpha] = pixel.to_le_bytes();
            [red, green, blue, alpha]
        })
        .collect::<Vec<_>>();
    writer
        .write_image_data(&rgba)
        .map_err(|error| format!("write PNG pixels: {error}"))
}

impl ApplicationHandler for LocalClient {
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(windows)]
        if let Some(menu) = &self.native_menu {
            for command in menu.drain_commands() {
                if command.eq_ignore_ascii_case(".quit") {
                    event_loop.exit();
                    return;
                }
                if command.eq_ignore_ascii_case(".reconnect") {
                    if let Err(error) = self.transport.reconnect() {
                        eprintln!("client-menu-command-error: command={command:?} error={error}");
                    }
                    continue;
                }
                if command
                    .split_ascii_whitespace()
                    .next()
                    .is_some_and(|verb| verb.eq_ignore_ascii_case(".screenshot"))
                {
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |duration| duration.as_secs());
                    self.pending_screenshot =
                        Some(PathBuf::from(format!("dream64-screenshot-{timestamp}.png")));
                    if let Some(surface) = &self.surface {
                        surface.window().request_redraw();
                    }
                    continue;
                }
                if let Err(error) = self.transport.send_command(&command) {
                    eprintln!("client-menu-command-error: command={command:?} error={error}");
                }
            }
        }
        self.drain_browser_messages();
        let previous_transport_status = self.transport.label().to_owned();
        if self.transport.try_connect() {
            self.readiness.rearm();
            self.ui_owner_commands
                .advance_generation(self.readiness.generation);
            self.browser_generation
                .store(self.readiness.generation, Ordering::Release);
            self.mark_client_ready(ClientReadiness::Transport);
            self.mark_client_ready(ClientReadiness::LocalResources);
            self.mark_client_ready(ClientReadiness::Skin);
            #[cfg(windows)]
            {
                let has_browser = self
                    .browsers
                    .keys()
                    .any(|control| control != AUDIO_BROWSER_CONTROL);
                let has_ready_document = self
                    .ready_browsers
                    .iter()
                    .any(|control| control != AUDIO_BROWSER_CONTROL);
                self.readiness.webview_required = has_browser;
                self.readiness.audio_available = self.browsers.contains_key(AUDIO_BROWSER_CONTROL);
                if has_browser {
                    self.mark_client_ready(ClientReadiness::WebViewEnvironment);
                }
                if has_ready_document {
                    self.mark_client_ready(ClientReadiness::WebViewDocument);
                }
            }
            if let Err(error) = self.transport.mark_skin_ready() {
                eprintln!("client-skin-readiness-failed: {error}");
            }
            self.startup_snapshot_visible = true;
            if self.startup_snapshot_active {
                // Replay and live streams both start sequence numbering at 1.
                // Preserve the rendered replay state, but accept the live
                // stream as a fresh authoritative sequence once it is ready.
                self.ui_presentation.last_sequence = 0;
            }
            self.snapshot_stage = 1;
            self.hud_snapshot_refreshed = false;
            if let Some(surface) = &self.surface {
                surface.window().request_redraw();
            }
        }
        if self.transport.label() != previous_transport_status
            && let Some(surface) = &self.surface
        {
            if self
                .transport
                .label()
                .contains("Starting world and subsystem controller")
            {
                self.startup_snapshot_visible = true;
            }
            surface.window().request_redraw();
        }
        // UI commands are produced asynchronously by `/client/New()` and later
        // server ticks. Poll at the server tick cadence, but repaint only when
        // the authoritative UI actually changed. A continuous full-window
        // redraw here consumed an entire CPU core while an idle lobby sat open.
        let received_ui = self.apply_inbound_ui();
        #[cfg(windows)]
        if received_ui {
            self.sync_native_menu();
            self.sync_browser_layout();
        }
        if received_ui && self.snapshot_stage >= 2 && !self.hud_snapshot_refreshed {
            self.hud_snapshot_refreshed = true;
            self.refresh_snapshot();
        }
        if received_ui && let Some(surface) = &self.surface {
            surface.window().request_redraw();
        }
        let now = std::time::Instant::now();
        if self.startup_snapshot_active
            && self.transport.is_live()
            && now >= self.next_startup_snapshot_refresh
        {
            self.next_startup_snapshot_refresh = now + std::time::Duration::from_secs(2);
            self.refresh_snapshot();
        }
        if self
            .next_screen_refresh
            .is_some_and(|deadline| now >= deadline)
        {
            self.next_screen_refresh = None;
            self.refresh_screen_snapshot();
        }
        if self.sprites.has_animations()
            && let Some(surface) = &self.surface
        {
            surface.window().request_redraw();
        }
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            std::time::Instant::now()
                + if self.sprites.has_animations() {
                    std::time::Duration::from_millis(50)
                } else {
                    std::time::Duration::from_millis(100)
                },
        ));
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default()
            .with_title(self.title())
            .with_maximized(true)
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(self.layout.window_width),
                f64::from(self.layout.window_height),
            ));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("the native Dream64 client window is created"),
        );
        #[cfg(windows)]
        if self.native_menu.is_none()
            && let Some(session) = self.runtime.client_session(self.client)
        {
            match NativeMenuBar::from_ui(session.ui()) {
                Ok(Some(menu)) => match menu.install(&window) {
                    Ok(()) => {
                        eprintln!("client-native-menu: installed");
                        self.native_menu = Some(menu);
                    }
                    Err(error) => eprintln!("client-native-menu-error: {error}"),
                },
                Ok(None) => {}
                Err(error) => eprintln!("client-native-menu-error: {error}"),
            }
        }
        let size = window.inner_size();
        let mut surface = match gpu::GpuRenderer::new(window.clone()) {
            Ok(renderer) => {
                eprintln!("client-renderer: wgpu adapter={}", renderer.adapter_label());
                ClientSurface::Gpu(Box::new(renderer))
            }
            Err(error) => {
                eprintln!("client-renderer-fallback: {error}");
                ClientSurface::Cpu(
                    Surface::new(&self.context, window)
                        .expect("the fallback client surface is created"),
                )
            }
        };
        Self::resize(&mut surface, size.width, size.height);
        // Some Windows/Wry combinations do not deliver a RedrawRequested
        // event for a request made while the maximized window is still being
        // resumed. Present once synchronously so offline replay cannot expose
        // an indefinitely black first frame. Later invalidations continue to
        // use the normal event-loop redraw path.
        let mut layout = self.layout.clone();
        if let Some(session) = self.runtime.client_session(self.client) {
            layout.apply_resolved_panes_in(session.ui(), Some((size.width, size.height)));
        }
        Self::redraw(
            &mut surface,
            self.startup_snapshot_visible
                .then_some(self.snapshot.as_ref())
                .flatten(),
            &mut self.sprites,
            layout,
            &self.ui_presentation.output_text,
            &self.input_states,
            self.focused_input.as_deref(),
            &self.button_states,
            &self.label_states,
            self.active_prompt.as_ref(),
            self.transport.label(),
            None,
        );
        self.surface = Some(surface);
        #[cfg(windows)]
        {
            let updates = std::mem::take(&mut self.startup_browser_updates);
            for update in updates {
                self.apply_browser_update(update);
            }
            self.sync_browser_layout();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if self
            .surface
            .as_ref()
            .is_none_or(|surface| surface.window().id() != window_id)
        {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(surface) = &mut self.surface {
                    Self::resize(surface, size.width, size.height);
                    surface.window().request_redraw();
                }
                #[cfg(windows)]
                self.sync_browser_layout();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if self.handle_prompt_key(&event) {
                    return;
                }
                if self.handle_input_key(&event) {
                    return;
                }
                if let PhysicalKey::Code(code) = event.physical_key {
                    let movement =
                        match code {
                            winit::keyboard::KeyCode::KeyW | winit::keyboard::KeyCode::ArrowUp => {
                                Some((0, 1))
                            }
                            winit::keyboard::KeyCode::KeyS
                            | winit::keyboard::KeyCode::ArrowDown => Some((0, -1)),
                            winit::keyboard::KeyCode::KeyA
                            | winit::keyboard::KeyCode::ArrowLeft => Some((-1, 0)),
                            winit::keyboard::KeyCode::KeyD
                            | winit::keyboard::KeyCode::ArrowRight => Some((1, 0)),
                            _ => None,
                        };
                    if event.state == ElementState::Pressed
                        && movement.is_some_and(|(dx, dy)| {
                            self.transport
                                .send_movement(&mut self.runtime, dx, dy)
                                .ok()
                                .flatten()
                                .is_some_and(|snapshot| {
                                    self.snapshot = Some(snapshot);
                                    true
                                })
                        })
                    {
                        if let Some(surface) = &self.surface {
                            surface.window().request_redraw();
                        }
                    }
                    let mut macro_server_command = None;
                    if let Some(session) = self.runtime.client_session_mut(self.client) {
                        if let Some(key) = macro_key_name(
                            code,
                            self.modifiers,
                            event.state == ElementState::Pressed,
                        ) {
                            macro_server_command =
                                dispatch_macro(session, &self.macro_bindings, &key)
                                    .and_then(|dispatch| dispatch.server_command);
                        }
                        session.push_event(UiEvent::Key {
                            key: format!("{code:?}"),
                            pressed: event.state == ElementState::Pressed,
                        });
                        self.local_input_events = self.local_input_events.saturating_add(1);
                        let title = self.title();
                        if let Some(surface) = &self.surface {
                            surface.window().set_title(&title);
                        }
                    }
                    if let Some(command) = macro_server_command
                        && let Err(error) = self.transport.send_command(&command)
                    {
                        eprintln!("client-command-error: command={command:?} error={error}");
                    }
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let point = (position.x.max(0.0) as u32, position.y.max(0.0) as u32);
                self.cursor_position = Some(point);
                if self.dragging_main_splitter {
                    self.set_main_splitter_from_pointer(point.0);
                    return;
                }
                let layout = self.effective_layout();
                let hit = self.snapshot.as_ref().and_then(|snapshot| {
                    screen_hit_at(
                        snapshot,
                        &mut self.sprites,
                        MapTransform::new(
                            layout.map,
                            layout.map_tile_size,
                            layout.map_zoom,
                            &layout.map_zoom_mode,
                            layout.map_letterbox,
                        ),
                        point.0,
                        point.1,
                    )
                });
                let next = hit.as_ref().map(|hit| (hit.0, hit.1));
                if next != self.hovered_screen {
                    if let Some(previous) = self.hovered_screen {
                        let _ = self.transport.send_screen_pointer(
                            previous,
                            "exited",
                            "",
                            &format!("mouse-x={};mouse-y={}", point.0, point.1),
                        );
                    }
                    if let Some((index, generation, screen_loc, control)) = &hit {
                        let _ = self.transport.send_screen_pointer(
                            (*index, *generation),
                            "entered",
                            control.as_deref().unwrap_or(""),
                            &format!(
                                "screen-loc={screen_loc};mouse-x={};mouse-y={}",
                                point.0, point.1
                            ),
                        );
                    }
                    self.hovered_screen = next;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                let point = self.cursor_position.unwrap_or_default();
                if button == MouseButton::Left && self.main_splitter_hit(point) {
                    self.dragging_main_splitter = true;
                    self.set_main_splitter_from_pointer(point.0);
                    return;
                }
                if let Some(prompt) = self.active_prompt.as_ref() {
                    let size = self
                        .surface
                        .as_ref()
                        .map(|surface| surface.window().inner_size())
                        .unwrap_or_default();
                    match client_prompt_hit(
                        usize::try_from(size.width).unwrap_or(0),
                        usize::try_from(size.height).unwrap_or(0),
                        prompt,
                        point,
                    ) {
                        Some(PromptHit::Choice(index)) => {
                            if let Some(prompt) = &mut self.active_prompt {
                                prompt.selected = index;
                            }
                        }
                        Some(PromptHit::Accept) => {
                            if let Some(response) = self.active_prompt_accept_response() {
                                self.finish_active_prompt(response);
                            }
                        }
                        Some(PromptHit::Cancel) => {
                            self.finish_active_prompt(ClientPromptResponse::Null);
                        }
                        None => {}
                    }
                    if let Some(surface) = &self.surface {
                        surface.window().request_redraw();
                    }
                    return;
                }
                let layout = self.effective_layout();
                self.focused_input = layout
                    .input_rects
                    .iter()
                    .find(|(_, rect)| pixel_rect_contains(**rect, point))
                    .map(|(address, _)| address.clone());
                let button_command = (button == MouseButton::Left)
                    .then(|| {
                        layout
                            .button_rects
                            .iter()
                            .find(|(_, rect)| pixel_rect_contains(**rect, point))
                            .and_then(|(address, _)| {
                                self.button_states
                                    .get(address)
                                    .map(|button| (address.clone(), button.command.clone()))
                            })
                            .filter(|(_, command)| !command.is_empty())
                    })
                    .flatten();
                if let Some((address, command)) = button_command {
                    let dispatch = self
                        .runtime
                        .client_session_mut(self.client)
                        .map(|session| dispatch_button_command(session, &address, command));
                    if let Some(command) = dispatch.and_then(|dispatch| dispatch.server_command)
                        && let Err(error) = self.transport.send_command(&command)
                    {
                        eprintln!("client-command-error: command={command:?} error={error}");
                    } else {
                        self.schedule_screen_refresh();
                    }
                    if let Some(session) = self.runtime.client_session(self.client) {
                        self.input_states = input_states_from_ui(session.ui(), &layout);
                        self.button_states = button_states_from_ui(session.ui(), &layout);
                    }
                }
                let screen_hit = self.snapshot.as_ref().and_then(|snapshot| {
                    screen_hit_at(
                        snapshot,
                        &mut self.sprites,
                        MapTransform::new(
                            layout.map,
                            layout.map_tile_size,
                            layout.map_zoom,
                            &layout.map_zoom_mode,
                            layout.map_letterbox,
                        ),
                        point.0,
                        point.1,
                    )
                });
                if let Some((index, generation, screen_loc, control)) = screen_hit {
                    let params = click_pointer_params(
                        &format!(
                            "screen-loc={screen_loc};mouse-x={};mouse-y={}",
                            point.0, point.1
                        ),
                        button,
                        self.modifiers,
                    );
                    if self
                        .transport
                        .send_screen_pointer(
                            (index, generation),
                            "click",
                            control.as_deref().unwrap_or("map"),
                            &params,
                        )
                        .is_ok()
                    {
                        self.schedule_screen_refresh();
                    }
                } else if let Some(hit) = self.snapshot.as_ref().and_then(|snapshot| {
                    map_hit_at(
                        snapshot,
                        &mut self.sprites,
                        MapTransform::new(
                            layout.map,
                            layout.map_tile_size,
                            layout.map_zoom,
                            &layout.map_zoom_mode,
                            layout.map_letterbox,
                        ),
                        point.0,
                        point.1,
                    )
                }) {
                    let params = click_pointer_params(&hit.2, button, self.modifiers);
                    if let Err(error) = self
                        .transport
                        .send_map_pointer(hit.0, hit.1, "map", &params)
                    {
                        eprintln!("map-pointer-error: {error}");
                    }
                }
                self.last_map_click = self.snapshot.as_ref().zip(self.cursor_position).and_then(
                    |(snapshot, (x, y))| {
                        let layout = self.effective_layout();
                        MapTransform::new(
                            layout.map,
                            layout.map_tile_size,
                            layout.map_zoom,
                            &layout.map_zoom_mode,
                            layout.map_letterbox,
                        )
                        .world_at(snapshot, x, y)
                    },
                );
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.dragging_main_splitter = false;
            }
            WindowEvent::RedrawRequested => {
                let received_ui = self.apply_inbound_ui();
                if received_ui && self.snapshot_stage >= 2 && !self.hud_snapshot_refreshed {
                    self.hud_snapshot_refreshed = true;
                    self.refresh_snapshot();
                }
                let snapshot = self
                    .startup_snapshot_visible
                    .then_some(self.snapshot.as_ref())
                    .flatten();
                let snapshot_present = snapshot.is_some();
                let layout = self.effective_layout();
                let screenshot = self.pending_screenshot.take();
                let presented = if let Some(surface) = &mut self.surface {
                    Self::redraw(
                        surface,
                        snapshot,
                        &mut self.sprites,
                        layout,
                        &self.ui_presentation.output_text,
                        &self.input_states,
                        self.focused_input.as_deref(),
                        &self.button_states,
                        &self.label_states,
                        self.active_prompt.as_ref(),
                        self.transport.label(),
                        screenshot.as_deref(),
                    )
                } else {
                    false
                };
                if presented && snapshot_present {
                    self.mark_client_ready(ClientReadiness::MapRenderer);
                    self.publish_client_readiness();
                }
                // Present the native DMF shell once before the potentially
                // expensive full snapshot/resource exchange. This guarantees
                // a visible responsive window during first-world loading.
                if self.snapshot_stage == 0 {
                    self.snapshot_stage = 1;
                    if let Some(surface) = &self.surface {
                        surface.window().request_redraw();
                    }
                } else if self.snapshot_stage == 1 {
                    self.snapshot_stage = 2;
                    self.refresh_snapshot();
                }
            }
            _ => {}
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use crate::transport::{
        ReplayReader, ReplayRecorder, ScreenAppearance, SnapshotResourceBudget,
        MAX_SNAPSHOT_RESOURCE_BYTES, RESOURCE_CHUNK_BYTES, decode_hex_bounded, encode_hex,
        normalize_screen_selector, parse_appearance_tree, parse_ui_event, read_frame, write_frame,
    };
    use dm_value::{DatumId, FieldName, TypePath, Value};

    fn launch_options(arguments: &[&str]) -> Result<LaunchOptions, String> {
        LaunchOptions::parse_from(arguments.iter().map(std::ffi::OsString::from))
    }

    fn mark_required_client_resources(coordinator: &mut ClientReadinessCoordinator) {
        let generation = coordinator.generation;
        for readiness in [
            ClientReadiness::Transport,
            ClientReadiness::LocalResources,
            ClientReadiness::Skin,
            ClientReadiness::ResourceManifest,
            ClientReadiness::ResourcePayload,
        ] {
            coordinator.mark(generation, readiness).unwrap();
        }
    }

    #[test]
    fn client_readiness_is_generation_bound_and_monotonic() {
        let mut coordinator = ClientReadinessCoordinator::new();
        let generation = coordinator.generation;
        coordinator
            .mark(generation, ClientReadiness::Transport)
            .unwrap();
        coordinator
            .mark(generation, ClientReadiness::Transport)
            .unwrap();
        assert!(coordinator.has(ClientReadiness::Transport));

        coordinator.rearm();
        assert_ne!(coordinator.generation, generation);
        assert!(!coordinator.has(ClientReadiness::Transport));
        assert_eq!(
            coordinator.mark(generation, ClientReadiness::Skin),
            Err("stale-client-generation")
        );
        assert!(!coordinator.has(ClientReadiness::Skin));
    }

    #[test]
    fn ui_owner_queue_preserves_fifo_and_rejects_overflow() {
        let mut queue = UiOwnerCommandQueue::new(7, 2);
        assert_eq!(queue.push(7, "map"), Ok(()));
        assert_eq!(queue.push(7, "browser"), Ok(()));
        assert_eq!(queue.push(7, "resource"), Err("resource"));
        assert_eq!(queue.pop_current(), Some("map"));
        assert_eq!(queue.pop_current(), Some("browser"));
        assert_eq!(queue.pop_current(), None);
        assert_eq!(queue.overflow_rejected, 1);
    }

    #[test]
    fn ui_owner_queue_drops_stale_generation_on_reconnect() {
        let mut queue = UiOwnerCommandQueue::new(11, 4);
        queue.push(11, 1).unwrap();
        queue.push(11, 2).unwrap();
        queue.advance_generation(12);
        assert_eq!(queue.pop_current(), None);
        assert_eq!(queue.stale_dropped, 2);
        assert_eq!(queue.push(11, 3), Err(3));
        assert_eq!(queue.push(12, 4), Ok(()));
        assert_eq!(queue.pop_current(), Some(4));
        assert_eq!(queue.stale_dropped, 3);
    }

    #[test]
    fn resource_hex_decode_rejects_size_before_allocating() {
        assert_eq!(decode_hex_bounded("00010203", 4), Ok(vec![0, 1, 2, 3]));
        assert_eq!(
            decode_hex_bounded("00010203", 3),
            Err("payload exceeds resource byte limit")
        );
        assert_eq!(
            decode_hex_bounded("0", 4),
            Err("hex payload has odd length")
        );
        assert_eq!(
            decode_hex_bounded("zz", 4),
            Err("hex payload contains an invalid digit")
        );
    }

    #[test]
    fn snapshot_resource_budget_checks_aggregate_without_truncation() {
        let mut budget = SnapshotResourceBudget::default();
        budget.reserve(MAX_SNAPSHOT_RESOURCE_BYTES).unwrap();
        assert_eq!(
            budget.reserve(1),
            Err("snapshot resource bytes exceed limit")
        );
        let mut overflow = SnapshotResourceBudget {
            count: 0,
            bytes: usize::MAX,
        };
        assert_eq!(
            overflow.reserve(1),
            Err("snapshot resource byte accounting overflow")
        );
    }

    #[test]
    fn client_readiness_waits_for_presented_map_but_not_audio() {
        let mut coordinator = ClientReadinessCoordinator::new();
        mark_required_client_resources(&mut coordinator);
        assert!(coordinator.resources_ready());
        assert!(!coordinator.interactive_ready());
        assert!(!coordinator.audio_available);

        let generation = coordinator.generation;
        coordinator
            .mark(generation, ClientReadiness::MapRenderer)
            .unwrap();
        assert!(coordinator.interactive_ready());
    }

    #[test]
    fn client_readiness_waits_for_required_webview_document() {
        let mut coordinator = ClientReadinessCoordinator::new();
        coordinator.webview_required = true;
        mark_required_client_resources(&mut coordinator);
        let generation = coordinator.generation;
        coordinator
            .mark(generation, ClientReadiness::MapRenderer)
            .unwrap();
        coordinator
            .mark(generation, ClientReadiness::WebViewEnvironment)
            .unwrap();
        assert!(!coordinator.interactive_ready());
        coordinator
            .mark(generation, ClientReadiness::WebViewDocument)
            .unwrap();
        assert!(coordinator.interactive_ready());
    }

    #[test]
    fn launch_options_accept_each_supported_mode() {
        let live = launch_options(&[
            "--skin",
            "lobby.dmf",
            "--connect",
            "127.0.0.1:55164",
            "--startup-replay",
            "startup.d64r",
            "--record-replay",
            "live.d64r",
        ])
        .expect("live launch options");
        assert_eq!(live.skin.as_deref(), Some(Path::new("lobby.dmf")));
        assert_eq!(live.connect.port(), 55_164);
        assert_eq!(
            live.startup_replay.as_deref(),
            Some(Path::new("startup.d64r"))
        );
        assert_eq!(live.record.as_deref(), Some(Path::new("live.d64r")));

        let replay = launch_options(&["lobby.dmf", "--replay", "lobby.d64r"])
            .expect("replay launch options");
        assert_eq!(replay.skin.as_deref(), Some(Path::new("lobby.dmf")));
        assert_eq!(replay.replay.as_deref(), Some(Path::new("lobby.d64r")));

        let offline = launch_options(&[
            "--world",
            "game.dme",
            "--map",
            "station.dmm",
            "--skin",
            "game.dmf",
        ])
        .expect("offline launch options");
        assert_eq!(offline.world.as_deref(), Some(Path::new("game.dme")));
        assert_eq!(offline.map.as_deref(), Some(Path::new("station.dmm")));
    }

    #[test]
    fn launch_options_reject_conflicting_modes() {
        for (arguments, expected) in [
            (vec!["--map", "station.dmm"], "--map requires --world"),
            (
                vec!["--record-replay", "a.d64r", "--replay", "b.d64r"],
                "--record-replay and --replay are mutually exclusive",
            ),
            (
                vec!["--startup-replay", "a.d64r", "--replay", "b.d64r"],
                "--replay and --startup-replay are mutually exclusive",
            ),
            (
                vec!["--world", "game.dme", "--replay", "a.d64r"],
                "replay recording/playback cannot be combined with --world",
            ),
            (
                vec!["--world", "game.dme", "--startup-replay", "a.d64r"],
                "--startup-replay cannot be combined with --world",
            ),
        ] {
            assert_eq!(launch_options(&arguments).unwrap_err(), expected);
        }
    }

    #[test]
    fn launch_options_accept_remote_server_address() {
        let options = launch_options(&["--connect", "192.0.2.10:51664"]).unwrap();
        assert_eq!(options.connect, "192.0.2.10:51664".parse().unwrap());
    }

    #[test]
    fn launch_options_reject_missing_values_duplicates_and_unknown_flags() {
        for (arguments, expected) in [
            (vec!["--skin"], "--skin requires a path"),
            (
                vec!["--skin", "one.dmf", "--skin", "two.dmf"],
                "--skin may only be specified once",
            ),
            (
                vec!["--connect", "127.0.0.1:1", "--connect", "127.0.0.1:2"],
                "--connect may only be specified once",
            ),
        ] {
            assert_eq!(launch_options(&arguments).unwrap_err(), expected);
        }
        assert!(
            launch_options(&["--bogus"])
                .unwrap_err()
                .contains("unknown client argument")
        );
    }

    fn opaque_png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("png header");
            writer
                .write_image_data(&vec![255; usize::try_from(width * height * 4).unwrap()])
                .expect("png pixels");
        }
        bytes
    }

    fn solid_png(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut pixels = Vec::with_capacity(usize::try_from(width * height * 4).unwrap());
        for _ in 0..width * height {
            pixels.extend_from_slice(&rgba);
        }
        {
            let mut encoder = png::Encoder::new(&mut bytes, width, height);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .expect("png header")
                .write_image_data(&pixels)
                .expect("png pixels");
        }
        bytes
    }

    fn pick_appearance(
        datum_index: u32,
        resource: PathBuf,
        layer: f32,
        mouse_opacity: i32,
        appearance_flags: i32,
    ) -> Appearance {
        Appearance {
            datum_index,
            datum_generation: 0,
            resource,
            state: String::new(),
            direction: 2,
            frame: 1,
            plane: 0.0,
            layer,
            appearance_flags,
            mouse_opacity,
            pixel_x: 0,
            pixel_y: 0,
            color: [255; 3],
            alpha: 255,
            maptext: None,
            maptext_width: 0,
            maptext_height: 0,
            maptext_x: 0,
            maptext_y: 0,
        }
    }

    #[test]
    fn picking_modes_and_pass_mouse_share_one_policy() {
        let pixels = [0x0000_0000, 0xffff_ffff];
        let mut appearance = pick_appearance(1, PathBuf::new(), 0.0, 1, 0);
        assert!(!appearance_hit_at(&appearance, &pixels, 2, 1, 0, 0));
        assert!(appearance_hit_at(&appearance, &pixels, 2, 1, 1, 0));
        appearance.mouse_opacity = 2;
        assert!(appearance_hit_at(&appearance, &pixels, 2, 1, 0, 0));
        appearance.mouse_opacity = 0;
        assert!(!appearance_hit_at(&appearance, &pixels, 2, 1, 1, 0));
        appearance.mouse_opacity = 2;
        appearance.appearance_flags = PASS_MOUSE_APPEARANCE_FLAG;
        assert!(!appearance_hit_at(&appearance, &pixels, 2, 1, 1, 0));
        assert!(!appearance_hit_at(&appearance, &pixels, 2, 1, 2, 0));
    }

    #[test]
    fn overlapping_world_sprite_transparency_falls_through_but_opaque_bounds_win() {
        let bottom_resource = PathBuf::from("pick-bottom.png");
        let top_resource = PathBuf::from("pick-top-transparent.png");
        let mut sprites = SpriteCache::default();
        sprites.insert(bottom_resource.clone(), &solid_png(1, 1, [255; 4]));
        sprites.insert(top_resource.clone(), &solid_png(1, 1, [255, 255, 255, 0]));
        let owner = WorldCoordinate { x: 1, y: 1, z: 1 };
        let transform = MapTransform::new(
            PixelRect {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
            32,
            1.0,
            "normal",
            false,
        );
        let make_snapshot = |top_opacity| MapSnapshot {
            center: owner,
            cells: BTreeMap::new(),
            turf_targets: BTreeMap::new(),
            appearances: BTreeMap::from([(
                (1, 1, 1),
                vec![
                    pick_appearance(10, bottom_resource.clone(), 1.0, 1, 0),
                    pick_appearance(20, top_resource.clone(), 2.0, top_opacity, 0),
                ],
            )]),
            screen: Vec::new(),
            resources: BTreeMap::new(),
        };
        let pixel_fallthrough = map_hit_at(&make_snapshot(1), &mut sprites, transform, 0, 31)
            .expect("bottom opaque pixel remains pickable");
        assert_eq!(pixel_fallthrough.0, (10, 0));
        let opaque_bounds = map_hit_at(&make_snapshot(2), &mut sprites, transform, 0, 31)
            .expect("opaque bounds pick transparent pixels");
        assert_eq!(opaque_bounds.0, (20, 0));
    }

    #[test]
    fn world_and_screen_items_apply_the_same_mouse_policy() {
        let resource = PathBuf::from("pick-parity-transparent.png");
        let mut sprites = SpriteCache::default();
        sprites.insert(resource.clone(), &solid_png(1, 1, [255, 255, 255, 0]));
        let owner = WorldCoordinate { x: 1, y: 1, z: 1 };
        let transform = MapTransform::new(
            PixelRect {
                x: 0,
                y: 0,
                width: 32,
                height: 32,
            },
            32,
            1.0,
            "normal",
            false,
        );
        for (mouse_opacity, expected) in [(0, false), (1, false), (2, true)] {
            let appearance = pick_appearance(30, resource.clone(), 1.0, mouse_opacity, 0);
            let snapshot = MapSnapshot {
                center: owner,
                cells: BTreeMap::new(),
                turf_targets: BTreeMap::new(),
                appearances: BTreeMap::from([((1, 1, 1), vec![appearance.clone()])]),
                screen: vec![ScreenAppearance {
                    datum_index: 30,
                    datum_generation: 0,
                    map_control: None,
                    screen_loc: "1,1".to_owned(),
                    type_path: "/atom/movable/screen/test".to_owned(),
                    insertion: 0,
                    appearances: vec![appearance],
                }],
                resources: BTreeMap::new(),
            };
            assert_eq!(
                map_hit_at(&snapshot, &mut sprites, transform, 0, 31).is_some(),
                expected
            );
            assert_eq!(
                screen_hit_at(&snapshot, &mut sprites, transform, 0, 0).is_some(),
                expected
            );
        }
    }

    #[test]
    fn retained_world_items_share_zoomed_bounds_with_cross_turf_picking() {
        let resource = PathBuf::from("wide.png");
        let mut sprites = SpriteCache::default();
        sprites.insert(resource.clone(), &opaque_png(64, 32));
        let appearance = Appearance {
            datum_index: 41,
            datum_generation: 2,
            resource,
            state: String::new(),
            direction: 2,
            frame: 1,
            plane: 0.0,
            layer: 1.0,
            appearance_flags: 0,
            mouse_opacity: 1,
            pixel_x: 5,
            pixel_y: 0,
            color: [255; 3],
            alpha: 255,
            maptext: None,
            maptext_width: 0,
            maptext_height: 0,
            maptext_x: 0,
            maptext_y: 0,
        };
        let owner = WorldCoordinate { x: 2, y: 1, z: 1 };
        let snapshot = MapSnapshot {
            center: owner,
            cells: BTreeMap::new(),
            turf_targets: BTreeMap::new(),
            appearances: BTreeMap::from([((owner.x, owner.y, owner.z), vec![appearance])]),
            screen: Vec::new(),
            resources: BTreeMap::new(),
        };
        let transform = MapTransform::new(
            PixelRect {
                x: 0,
                y: 0,
                width: 192,
                height: 64,
            },
            32,
            2.0,
            "normal",
            false,
        );
        let items = world_display_items(&snapshot, &mut sprites, transform);
        assert_eq!(items.len(), 1);
        assert_eq!(
            (items[0].origin_x, items[0].width, items[0].height),
            (74, 128, 64)
        );
        let gpu_items = gpu_world_display_items(&snapshot, &mut sprites, transform);
        assert_eq!(gpu_items.len(), 1);
        assert_eq!(gpu_items[0].draw.destination, [74.0, 0.0, 128.0, 64.0]);
        assert_eq!(gpu_items[0].draw.source, [0, 0, 64, 32]);
        assert_eq!(gpu_items[0].draw.tint, [255, 255, 255, 255]);
        // x=150 resolves to the turf east of the owner, but the wide sprite
        // visibly covers it. Picking must return the appearance and its owner.
        let hit = map_hit_at(&snapshot, &mut sprites, transform, 150, 20).expect("wide hit");
        assert_eq!(hit.0, (41, 2));
        assert_eq!(hit.1, owner);
    }

    #[test]
    fn native_world_sprite_spans_tiles_and_respects_pixel_offsets_and_map_clip() {
        // A 4x2 appearance owned by a 2x2 turf extends one full neighboring
        // turf to the right. +1,+1 shifts it right and upward in BYOND space.
        let sprite = vec![0xffff_0000; 8];
        let mut buffer = vec![0_u32; 8 * 6];
        let (x, y) = world_appearance_origin(2, 4, 2, 1, 1);
        assert_eq!((x, y), (3, 1));
        blit_sprite_clipped(
            &mut buffer,
            8,
            6,
            x,
            y,
            4,
            &sprite,
            PixelRect {
                x: 2,
                y: 1,
                width: 4,
                height: 4,
            },
        );
        assert_eq!(buffer[1 * 8 + 3], 0xffff_0000);
        assert_eq!(
            buffer[1 * 8 + 5],
            0xffff_0000,
            "spans beyond its 2px owner turf"
        );
        assert_eq!(
            buffer[1 * 8 + 6],
            0,
            "clips at the map control, not turf edge"
        );
        assert_eq!(buffer[3 * 8 + 3], 0, "positive pixel_y moves upward");
    }

    #[test]
    fn screen_loc_uses_byond_bottom_left_tiles_and_pixel_offsets() {
        let transform = MapTransform {
            clip: PixelRect {
                x: 0,
                y: 0,
                width: 320,
                height: 160,
            },
            origin_x: 10,
            origin_y: 20,
            tile: 32,
            columns: 10,
            rows: 5,
        };
        assert_eq!(screen_loc_pixels("1,1", transform, 32, 32), Some((10, 148)));
        assert_eq!(
            screen_loc_pixels("EAST:-4,NORTH:2", transform, 32, 32),
            Some((294, 18))
        );
        assert_eq!(
            screen_loc_pixels("RIGHT:-4,TOP:2", transform, 32, 32),
            Some((294, 18))
        );
        assert_eq!(
            screen_loc_pixels("CENTER,CENTER", transform, 32, 32),
            Some((138, 84))
        );
        assert_eq!(
            screen_loc_pixels("TOP,CENTER:-61", transform, 295, 145),
            Some((109, 20))
        );
        assert_eq!(
            screen_loc_pixels("\"TOP,CENTER:-61\"", transform, 295, 145),
            Some((109, 20))
        );
        assert_eq!(
            screen_loc_pixels("TOP:-87,CENTER", transform, 32, 32),
            Some((170, 107))
        );
        assert_eq!(
            screen_loc_pixels("TOP:-54,CENTER", transform, 32, 32),
            Some((170, 74))
        );
        assert_eq!(
            normalize_screen_selector(Some("TOP".into()), "-87,CENTER:+100".into()),
            (None, "TOP:-87,CENTER:+100".into())
        );
        assert_eq!(
            normalize_screen_selector(Some("map".into()), "1,1".into()),
            (Some("map".into()), "1,1".into())
        );
        assert!(screen_is_render_pipeline_helper(&ScreenAppearance {
            datum_index: 1,
            datum_generation: 0,
            map_control: None,
            screen_loc: "CENTER".into(),
            type_path: "/atom/movable/screen/plane_master/hud".into(),
            insertion: 0,
            appearances: Vec::new(),
        }));
        assert!(screen_is_render_pipeline_helper(&ScreenAppearance {
            datum_index: 3,
            datum_generation: 0,
            map_control: None,
            screen_loc: "WEST,SOUTH to EAST,NORTH".into(),
            type_path: "/atom/movable/screen/fullscreen/lighting_backdrop/unlit".into(),
            insertion: 0,
            appearances: Vec::new(),
        }));
        assert!(screen_is_render_pipeline_helper(&ScreenAppearance {
            datum_index: 2,
            datum_generation: 0,
            map_control: None,
            screen_loc: "CENTER-9,CENTER-7".into(),
            type_path: "/atom/movable/screen/click_catcher".into(),
            insertion: 0,
            appearances: Vec::new(),
        }));
        assert!(!screen_is_render_pipeline_helper(&ScreenAppearance {
            datum_index: 4,
            datum_generation: 0,
            map_control: None,
            screen_loc: "TOP:-126,CENTER:62".into(),
            type_path: "/atom/movable/screen/lobby/button/ready".into(),
            insertion: 0,
            appearances: Vec::new(),
        }));
    }

    #[test]
    fn empty_browse_resource_payload_decodes_without_dropping_ui_batch() {
        assert_eq!(
            parse_ui_event("U 7 browse_resource 746573742e706e67 -"),
            Ok((
                7,
                InboundUiCommand::BrowseResource {
                    name: "test.png".into(),
                    data: vec![],
                }
            ))
        );
    }

    #[test]
    fn prompt_wire_row_decodes_typed_modal_and_mouse_targets() {
        let (_, InboundUiCommand::Prompt(prompt)) = parse_ui_event(
            "U 8 prompt 12 list 1 43686f6f7365 526f6c65 456e67696e656572 456e67696e656572,446f63746f72",
        )
        .unwrap()
        else {
            panic!("prompt row must decode as a prompt")
        };
        assert_eq!(prompt.kind, ClientPromptKind::List);
        assert_eq!(prompt.title, "Choose");
        assert_eq!(prompt.message, "Role");
        assert_eq!(prompt.choices, ["Engineer", "Doctor"]);
        let rect = client_prompt_rect(800, 600);
        assert_eq!(
            client_prompt_hit(800, 600, &prompt, (rect.x + 20, rect.y + 106)),
            Some(PromptHit::Choice(1))
        );
        assert_eq!(
            client_prompt_hit(
                800,
                600,
                &prompt,
                (rect.x + rect.width - 50, rect.y + rect.height - 28),
            ),
            Some(PromptHit::Accept)
        );
    }

    #[test]
    fn sound_wire_row_decodes_channel_playback_update() {
        let (sequence, command) =
            parse_ui_event("U 46 sound 7 1 80 22050 -25 736f756e642f6c6f6262792e6f6767").unwrap();
        assert_eq!(sequence, 46);
        assert_eq!(
            command,
            InboundUiCommand::Sound(SoundUpdate {
                file: Some("sound/lobby.ogg".into()),
                channel: 7,
                repeat: true,
                volume: 80.0,
                frequency: 22050.0,
                pan: -25.0,
            })
        );
        let (_, stop) = parse_ui_event("U 47 sound 7 0 100 0 0 -").unwrap();
        assert!(matches!(
            stop,
            InboundUiCommand::Sound(SoundUpdate {
                file: None,
                channel: 7,
                ..
            })
        ));
    }

    /// Boot-independent smoke test for the production client contract. Run with:
    /// `cargo test -p dm-client offline_monkestation_skin_protocol2_and_sprite_fixture -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires the sibling Monkestation2.0 checkout"]
    fn offline_monkestation_skin_protocol2_and_sprite_fixture() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(3)
            .expect("dm-client must be inside the Dream64 workspace");
        let skin_path = workspace.join("Monkestation2.0/interface/skin.dmf");
        let source = std::fs::read_to_string(&skin_path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", skin_path.display()));
        let document = dm_dmf::parse(&source);
        assert!(
            document.diagnostics.is_empty(),
            "{:?}",
            document.diagnostics
        );
        let tree = ControlTree::from_document(&document);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut layout = ClientLayout::from_tree(session.ui().tree());
        let say_command = session
            .ui()
            .winget("inputbuttons.saybutton", "command")
            .unwrap();
        assert_eq!(
            dispatch_button_command(&mut session, "inputbuttons.saybutton", say_command),
            MacroDispatch {
                server_command: None
            }
        );
        assert_eq!(
            session.ui().winget("inputbuttons.saybutton", "is-checked"),
            Ok("true".into())
        );
        assert_eq!(
            session.ui().winget("inputwindow.input", "command"),
            Ok("!say \"".into())
        );
        assert_eq!(
            session.ui().winget("inputbuttons.mebutton", "is-checked"),
            Ok("false".into())
        );
        assert_eq!(
            layout.map,
            PixelRect {
                x: 0,
                y: 0,
                width: 320,
                height: 440,
            }
        );
        assert_eq!(layout.browser, None, "inactive browser pane stays hidden");
        assert_eq!(
            layout.output_rects["output_legacy.output"],
            PixelRect {
                x: 320,
                y: 227,
                width: 320,
                height: 206,
            }
        );
        session
            .apply_command(UiCommand::WinSet {
                control: "output_selector.legacy_output_selector".to_owned(),
                parameters: "left=output_browser".to_owned(),
            })
            .unwrap();
        layout.refresh_from_ui(session.ui());
        assert_eq!(
            layout.browser,
            Some(PixelRect {
                x: 320,
                y: 227,
                width: 320,
                height: 206,
            })
        );
        assert!(layout.output_rects.is_empty());
        let mut presentation = UiPresentation::default();

        let wire = |sequence: u64, kind: &str, first: &str, second: &[u8]| {
            format!(
                "U {sequence} {kind} {} {}",
                encode_hex(first.as_bytes()),
                encode_hex(second),
            )
        };
        let rows = [
            wire(
                1,
                "winset",
                "ShiftUp",
                b"command=.winset :map.right-click=true",
            ),
            wire(2, "output", "output", b"offline-ready"),
            wire(3, "browse_resource", "lobby.dmi", &[137, 80, 78, 71]),
            wire(
                4,
                "browse",
                "browseroutput",
                b"<h1>Offline lobby</h1><img src='lobby.dmi'>",
            ),
        ];
        for row in rows {
            let (sequence, event) = parse_ui_event(&row).expect("protocol-2 event parses");
            presentation
                .apply(sequence, event, &mut session, &mut layout)
                .expect("production skin accepts event");
        }
        assert_eq!(
            presentation.output_text["output_legacy.output"],
            ["offline-ready"]
        );
        assert!(presentation.browser_html["output_browser.browseroutput"].contains("lobby.dmi"));
        assert_eq!(
            presentation.browser_resources["lobby.dmi"],
            [137, 80, 78, 71]
        );
        assert_eq!(presentation.last_sequence, 4);

        let fixture_root =
            std::env::temp_dir().join(format!("dream64-offline-client-{}", std::process::id()));
        std::fs::create_dir_all(&fixture_root).unwrap();
        let dmi = fixture_root.join("tile.dmi");
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[255, 255, 255, 255])
                .unwrap();
        }
        std::fs::write(&dmi, bytes).unwrap();
        let appearance = |plane, layer, pixel_x, color, alpha| Appearance {
            datum_index: 1,
            datum_generation: 0,
            resource: dmi.clone(),
            state: String::new(),
            direction: 2,
            frame: 1,
            plane,
            layer,
            appearance_flags: 0,
            mouse_opacity: 1,
            pixel_x,
            pixel_y: 0,
            color,
            alpha,
            maptext: None,
            maptext_width: 0,
            maptext_height: 0,
            maptext_x: 0,
            maptext_y: 0,
        };
        let pixels = composite_tile(
            &mut SpriteCache::default(),
            &[
                appearance(0.0, 0.0, 0, [255, 0, 0], 255),
                appearance(0.0, 1.0, 0, [0, 0, 255], 128),
                appearance(1.0, 0.0, 1, [0, 255, 0], 255),
            ],
            2,
            1,
        )
        .expect("DMI/PNG appearances composite");
        assert_eq!(pixels[1], 0xff00_ff00, "plane and pixel offset are honored");
        assert_ne!(
            pixels[0], pixels[1],
            "layer tint/alpha composite separately"
        );
        std::fs::remove_dir_all(fixture_root).unwrap();
    }

    #[test]
    fn real_skin_style_ids_and_panes_resolve_main_lobby_geometry() {
        let document = dm_dmf::parse(concat!(
            "window \"mainwindow\"\n",
            "\telem \"mainwindow\"\n\t\ttype = MAIN\n\t\tpos = 281,0\n\t\tsize = 640x440\n\t\tis-default = true\n",
            "\telem \"split\"\n\t\ttype = CHILD\n\t\tpos = 0,0\n\t\tsize = 640x440\n\t\tanchor1 = 0,0\n\t\tanchor2 = 100,100\n\t\tleft = \"mapwindow\"\n\t\tright = \"info_and_buttons\"\n\t\tis-vert = true\n",
            "window \"mapwindow\"\n",
            "\telem \"mapwindow\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 640x480\n\t\tis-pane = true\n",
            "\telem \"map\"\n\t\ttype = MAP\n\t\tpos = 0,0\n\t\tsize = 640x480\n\t\tanchor1 = 0,0\n\t\tanchor2 = 100,100\n",
            "window \"info_and_buttons\"\n",
            "\telem \"info_and_buttons\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 640x440\n\t\tis-pane = true\n",
            "\telem \"info\"\n\t\ttype = CHILD\n\t\tpos = 0,0\n\t\tsize = 640x440\n\t\tanchor1 = 0,0\n\t\tanchor2 = 100,100\n\t\tleft = \"output_browser\"\n",
            "window \"output_browser\"\n",
            "\telem \"output_browser\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 640x456\n\t\tis-pane = true\n",
            "\telem \"browseroutput\"\n\t\ttype = BROWSER\n\t\tpos = 0,0\n\t\tsize = 640x456\n\t\tanchor1 = 0,0\n\t\tanchor2 = 100,100\n",
        ));
        assert!(document.diagnostics.is_empty());
        let tree = ControlTree::from_document(&document);
        let layout = ClientLayout::from_tree(&tree);
        assert_eq!((layout.window_width, layout.window_height), (640, 440));
        assert_eq!(
            layout.map,
            PixelRect {
                x: 0,
                y: 0,
                width: 320,
                height: 440
            }
        );
        assert_eq!(
            layout.browser,
            Some(PixelRect {
                x: 320,
                y: 0,
                width: 320,
                height: 440
            })
        );
        assert_eq!(
            layout.browser_control.as_deref(),
            Some("output_browser.browseroutput")
        );
        let mut ui = dm_dmf::UiState::new(tree);
        let maximized = resolve_pane_layout_in(&ui, Some((1_920, 1_030)));
        assert_eq!(
            maximized.root,
            PixelRect {
                x: 0,
                y: 0,
                width: 1_920,
                height: 1_030
            }
        );
        assert_eq!(
            maximized.controls["mapwindow.map"],
            PixelRect {
                x: 0,
                y: 0,
                width: 960,
                height: 1_030
            }
        );
        assert_eq!(
            maximized.controls["output_browser.browseroutput"],
            PixelRect {
                x: 960,
                y: 0,
                width: 960,
                height: 1_030
            }
        );
        ui.winset("mainwindow.split", "splitter=65")
            .expect("authored saved splitter control resolves");
        let saved = resolve_pane_layout_in(&ui, Some((1_920, 1_030)));
        assert_eq!(saved.controls["mapwindow.map"].width, 1_248);
        assert_eq!(saved.controls["output_browser.browseroutput"].x, 1_248);
        assert_eq!(saved.controls["output_browser.browseroutput"].width, 672);
    }

    #[test]
    fn ordered_wire_ui_events_route_to_typed_dmf_controls() {
        let document = dm_dmf::parse(concat!(
            "window \"main\"\n",
            "\telem \"main\"\n\t\ttype = MAIN\n\t\tsize = 640x440\n\t\tis-default = true\n",
            "\telem \"browser\"\n\t\ttype = BROWSER\n\t\tpos = 320,0\n\t\tsize = 320x300\n",
            "\telem \"log\"\n\t\ttype = OUTPUT\n\t\tpos = 320,300\n\t\tsize = 320x140\n",
        ));
        let tree = ControlTree::from_document(&document);
        let mut layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();

        let (sequence, browse) =
            parse_ui_event("U 7 browse 62726f77736572 3c623e6c6f6262793c2f623e").unwrap();
        assert_eq!(
            presentation
                .apply(sequence, browse, &mut session, &mut layout)
                .unwrap(),
            Some(BrowserUpdate::Html {
                control: "main.browser".to_owned(),
                html: "<b>lobby</b>".to_owned(),
            })
        );
        let (sequence, output) = parse_ui_event("U 8 output 6c6f67 7265616479").unwrap();
        assert_eq!(
            presentation
                .apply(sequence, output, &mut session, &mut layout)
                .unwrap(),
            None
        );
        assert_eq!(presentation.output_text["main.log"], ["ready"]);
        assert_eq!(presentation.last_sequence, 8);

        assert_eq!(
            presentation
                .apply(
                    9,
                    InboundUiCommand::Output {
                        control: Some("browser".to_owned()),
                        message: "html/typing_indicator.html".to_owned(),
                    },
                    &mut session,
                    &mut layout,
                )
                .unwrap(),
            Some(BrowserUpdate::Resource {
                control: "main.browser".to_owned(),
                path: "html/typing_indicator.html".to_owned(),
            })
        );
        assert_eq!(
            presentation
                .apply(
                    10,
                    InboundUiCommand::Output {
                        control: Some("browser:update".to_owned()),
                        message: "%7B%22type%22%3A%22ready%22%7D".to_owned(),
                    },
                    &mut session,
                    &mut layout,
                )
                .unwrap(),
            Some(BrowserUpdate::Script {
                control: "main.browser".to_owned(),
                script: "update(\"{\\\"type\\\":\\\"ready\\\"}\");".to_owned(),
            })
        );
        assert_eq!(
            presentation
                .apply(
                    11,
                    InboundUiCommand::Output {
                        control: Some("browser.browser:update".to_owned()),
                        message: "%7B%22type%22%3A%22update_stat%22%7D".to_owned(),
                    },
                    &mut session,
                    &mut layout,
                )
                .unwrap(),
            Some(BrowserUpdate::Script {
                control: "main.browser".to_owned(),
                script: "update(\"{\\\"type\\\":\\\"update_stat\\\"}\");".to_owned(),
            })
        );

        // A repeated drained record cannot be applied twice.
        let (_, duplicate) = parse_ui_event("U 8 output 6c6f67 6475706c6963617465").unwrap();
        presentation
            .apply(8, duplicate, &mut session, &mut layout)
            .unwrap();
        assert_eq!(presentation.output_text["main.log"], ["ready"]);
    }

    #[test]
    fn output_retention_tracks_the_effective_dmf_lines_property() {
        let document = dm_dmf::parse(concat!(
            "window \"main\"\n",
            "\telem \"main\"\n\t\ttype = MAIN\n\t\tis-default = true\n",
            "\telem \"log\"\n\t\ttype = OUTPUT\n\t\tlines = 2\n",
        ));
        let tree = ControlTree::from_document(&document);
        let mut layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();

        presentation
            .apply(
                1,
                InboundUiCommand::Output {
                    control: Some("log".to_owned()),
                    message: "one\ntwo\nthree".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        assert_eq!(presentation.output_text["main.log"], ["two", "three"]);

        presentation
            .apply(
                2,
                InboundUiCommand::WinSet {
                    control: "log".to_owned(),
                    parameters: "lines=1".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        presentation
            .apply(
                3,
                InboundUiCommand::Output {
                    control: Some("log".to_owned()),
                    message: "four".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        assert_eq!(presentation.output_text["main.log"], ["four"]);
    }

    #[test]
    fn winset_recomputes_native_layout_before_the_next_ui_event() {
        let document = dm_dmf::parse(concat!(
            "window \"main\"\n",
            "\telem \"main\"\n\t\ttype = MAIN\n\t\tsize = 640x440\n\t\tis-default = true\n",
            "\telem \"map\"\n\t\ttype = MAP\n\t\tpos = 0,0\n\t\tsize = 320x440\n",
            "\telem \"browser\"\n\t\ttype = BROWSER\n\t\tpos = 320,0\n\t\tsize = 320x440\n",
        ));
        let tree = ControlTree::from_document(&document);
        let mut layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();

        presentation
            .apply(
                1,
                InboundUiCommand::WinSet {
                    control: "main.map".to_owned(),
                    parameters: "pos=12,18;size=400x300".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        assert_eq!(
            layout.map,
            PixelRect {
                x: 12,
                y: 18,
                width: 400,
                height: 300,
            }
        );

        presentation
            .apply(
                2,
                InboundUiCommand::WinSet {
                    control: "main.browser".to_owned(),
                    parameters: "is-visible=false".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        assert_eq!(layout.browser, None);
        assert_eq!(layout.browser_control, None);
    }

    #[test]
    fn map_zoom_letterbox_and_click_transform_share_effective_dmf_properties() {
        let document = dm_dmf::parse(concat!(
            "window \"main\"\n",
            "\telem \"main\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 100x70\n\t\tis-default = true\n",
            "\telem \"map\"\n\t\ttype = MAP\n\t\tpos = 10,20\n\t\tsize = 100x70\n",
        ));
        let tree = ControlTree::from_document(&document);
        let mut session = dm_dmf::ClientSession::new(tree);
        session
            .apply_command(UiCommand::WinSet {
                control: "main.map".to_owned(),
                parameters: "tile-size=16;zoom=2;zoom-mode=normal;letterbox=true".to_owned(),
            })
            .unwrap();
        let mut layout = ClientLayout::from_tree(session.ui().tree());
        layout.refresh_from_ui(session.ui());
        assert_eq!(layout.map_tile_size, 16);
        assert_eq!(layout.map_zoom, 2.0);
        assert_eq!(layout.map_zoom_mode, "normal");
        assert!(layout.map_letterbox);

        let transform = MapTransform::new(
            layout.map,
            layout.map_tile_size,
            layout.map_zoom,
            &layout.map_zoom_mode,
            layout.map_letterbox,
        );
        assert_eq!(transform.tile, 32);
        assert_eq!((transform.origin_x, transform.origin_y), (12, 23));
        assert_eq!((transform.columns, transform.rows), (3, 2));
        let snapshot = MapSnapshot {
            center: WorldCoordinate { x: 10, y: 10, z: 2 },
            cells: BTreeMap::new(),
            turf_targets: BTreeMap::new(),
            appearances: BTreeMap::new(),
            screen: Vec::new(),
            resources: BTreeMap::new(),
        };
        assert_eq!(transform.world_at(&snapshot, 44, 55), Some(snapshot.center));
        assert_eq!(
            transform.world_at(&snapshot, 12, 23),
            Some(WorldCoordinate { x: 9, y: 11, z: 2 })
        );
        assert_eq!(transform.world_at(&snapshot, 10, 23), None);

        let mut pixels = vec![0xff01_0203; 140 * 100];
        draw_map(
            &mut pixels,
            140,
            100,
            transform,
            None,
            &mut SpriteCache::default(),
            None,
        );
        assert_eq!(pixels[23 * 140 + 10], 0xff00_0000, "letterbox bar is black");
        assert_eq!(
            pixels[23 * 140 + 9],
            0xff01_0203,
            "outside map is untouched"
        );
        assert_ne!(pixels[24 * 140 + 13], 0xff01_0203, "first tile is rendered");
        assert_eq!(
            click_pointer_params(
                "icon-x=4;icon-y=7;screen-loc=2:3,1:6",
                MouseButton::Right,
                ModifiersState::CONTROL | ModifiersState::SHIFT,
            ),
            "icon-x=4;icon-y=7;screen-loc=2:3,1:6;right=1;button=right;shift=1;ctrl=1"
        );
    }

    #[test]
    fn output_renderer_draws_latest_clipped_plaintext_inside_dmf_rectangle() {
        let width = 48;
        let height = 24;
        let untouched = 0xff55_6677;
        let mut buffer = vec![untouched; width * height];
        draw_output_control(
            &mut buffer,
            width,
            height,
            PixelRect {
                x: 4,
                y: 3,
                width: 36,
                height: 15,
            },
            &["old".to_owned(), "<b>NEW</b>".to_owned()],
        );
        assert_eq!(strip_output_markup("<b>A&amp;B</b>"), "A&B");
        assert!(buffer.iter().any(|pixel| *pixel == 0xff00_0000));
        assert_eq!(buffer[2 * width + 4], untouched, "top clip is untouched");
        assert_eq!(buffer[3 * width + 3], untouched, "left clip is untouched");
        assert_eq!(
            buffer[18 * width + 4],
            untouched,
            "bottom clip is untouched"
        );
        assert_eq!(buffer[3 * width + 40], untouched, "right clip is untouched");
    }

    #[test]
    fn browse_does_not_cross_the_output_control_boundary() {
        let document = dm_dmf::parse(
            "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\telem \"log\"\n\t\ttype = OUTPUT\n",
        );
        let tree = ControlTree::from_document(&document);
        let mut layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();
        presentation
            .apply(
                1,
                InboundUiCommand::Browse {
                    control: "log".to_owned(),
                    html: "not output".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        assert!(presentation.output_text.is_empty());
        assert_eq!(presentation.browser_html["log"], "not output");
    }

    #[test]
    fn browse_resources_remain_relative_for_the_loopback_asset_origin() {
        let document = dm_dmf::parse(
            "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 320x200\n\t\tis-default = true\n\telem \"browser\"\n\t\ttype = BROWSER\n\t\tpos = 0,0\n\t\tsize = 320x200\n",
        );
        let tree = ControlTree::from_document(&document);
        let mut layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();
        presentation
            .apply(
                1,
                InboundUiCommand::BrowseResource {
                    name: "logo.png".to_owned(),
                    data: vec![0, 1, 2],
                },
                &mut session,
                &mut layout,
            )
            .unwrap();
        let loaded = presentation
            .apply(
                2,
                InboundUiCommand::Browse {
                    control: "browser".to_owned(),
                    html: "<img src='logo.png'>".to_owned(),
                },
                &mut session,
                &mut layout,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            loaded,
            BrowserUpdate::Html {
                control: "main.browser".to_owned(),
                html: "<img src='logo.png'>".to_owned(),
            }
        );
        assert_eq!(presentation.browser_resources["logo.png"], vec![0, 1, 2]);
    }

    #[test]
    fn browser_resources_resolve_nested_css_images_fonts_and_reject_traversal() {
        let resources = BTreeMap::from([
            (
                "ui/css/site.css".to_owned(),
                b".logo{background:url('../img/logo.png')}@font-face{src:url(../fonts/ui.woff2)}"
                    .to_vec(),
            ),
            ("ui/img/logo.png".to_owned(), vec![0, 1, 2]),
            ("ui/fonts/ui.woff2".to_owned(), vec![3, 4, 5]),
        ]);
        let mut cache = BTreeMap::new();
        let image = "data:image/png;base64,AAEC";
        let font = "data:font/woff2;base64,AwQF";
        let rewritten_css =
            format!(".logo{{background:url('{image}')}}@font-face{{src:url({font})}}");
        let expected_css = format!(
            "data:text/css;base64,{}",
            encode_base64(rewritten_css.as_bytes())
        );
        let html = materialize_browser_resources(
            "<link href='ui/css/site.css'><img src=\"ui/img/logo.png\">",
            &resources,
            &mut cache,
        );
        assert_eq!(
            html,
            format!("<link href='{expected_css}'><img src=\"{image}\">")
        );
        assert_eq!(cache.len(), 3, "nested assets share cached data URIs");
        assert_eq!(normalize_resource_path("ui/css", "../../../secret"), None);
        assert_eq!(normalize_resource_path("", "../secret"), None);

        let document = dm_dmf::parse(
            "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 10x10\n\t\tis-default = true\n",
        );
        let tree = ControlTree::from_document(&document);
        let mut session = dm_dmf::ClientSession::new(tree.clone());
        let mut layout = ClientLayout::from_tree(&tree);
        let mut presentation = UiPresentation::default();
        assert!(
            presentation
                .apply(
                    1,
                    InboundUiCommand::BrowseResource {
                        name: "../secret".to_owned(),
                        data: vec![9],
                    },
                    &mut session,
                    &mut layout,
                )
                .is_err()
        );
        assert!(presentation.browser_resources.is_empty());
    }

    fn loopback_fixture() -> (ExecutionState, DatumId, LoopbackTransport) {
        let mut runtime = ExecutionState::new();
        let client = runtime
            .heap_mut()
            .allocate_datum(TypePath::parse("/client").unwrap());
        let player_mob = runtime
            .heap_mut()
            .allocate_datum(TypePath::parse("/mob/player").unwrap());
        let mut cells = BTreeMap::new();
        let mut turfs = BTreeMap::new();
        for (x, color) in [(1, 0xff11_2233), (2, 0xff44_5566)] {
            let turf = runtime
                .heap_mut()
                .allocate_datum(TypePath::parse("/turf/open").unwrap());
            for (name, value) in [("x", x), ("y", 1), ("z", 1)] {
                runtime
                    .heap_mut()
                    .set_datum_field(
                        turf,
                        FieldName::parse(name).unwrap(),
                        Value::number(value as f32),
                    )
                    .unwrap();
            }
            cells.insert((x, 1, 1), color);
            turfs.insert((x, 1, 1), turf);
        }
        let scene = WorldScene {
            cells,
            turfs,
            player: WorldCoordinate { x: 1, y: 1, z: 1 },
            player_mob,
            label: "loopback fixture".to_owned(),
        };
        (runtime, client, LoopbackTransport::new(scene))
    }

    #[test]
    fn loopback_attach_snapshot_and_wasd_movement_round_trip() {
        let (mut runtime, client, mut transport) = loopback_fixture();
        transport.attach(&mut runtime, client).unwrap();
        let initial = transport.request_snapshot(10, 7);
        assert_eq!(initial.center, WorldCoordinate { x: 1, y: 1, z: 1 });
        assert_eq!(initial.cells.len(), 2);

        let moved = transport
            .send_movement(&mut runtime, 1, 0)
            .expect("D movement reaches the adjacent turf");
        assert_eq!(moved.center, WorldCoordinate { x: 2, y: 1, z: 1 });
        assert!(transport.send_movement(&mut runtime, 1, 0).is_none());

        let mob = runtime
            .heap()
            .datum_field(client, &FieldName::parse("mob").unwrap())
            .unwrap();
        assert_eq!(mob, &Value::Datum(transport.scene.player_mob));
        assert_eq!(
            runtime
                .heap()
                .datum_field(transport.scene.player_mob, &FieldName::parse("x").unwrap())
                .unwrap(),
            &Value::number(2.0)
        );
    }

    #[test]
    fn unattached_loopback_rejects_input_but_allows_snapshot_request() {
        let (mut runtime, _client, mut transport) = loopback_fixture();
        assert_eq!(transport.request_snapshot(0, 0).cells.len(), 1);
        assert!(transport.send_movement(&mut runtime, 1, 0).is_none());
    }

    #[test]
    fn protocol_replay_round_trips_recorded_responses_and_idles_ui_polling() {
        let path =
            std::env::temp_dir().join(format!("dream64-client-replay-{}.d64r", std::process::id()));
        {
            let mut recorder = ReplayRecorder::create(&path).unwrap();
            recorder
                .record("attach", "ok attach client=c1 x=4 y=5 z=1")
                .unwrap();
            recorder
                .record("ui_events c1", "ok ui_events count=1\nU 1 output 6869 6f")
                .unwrap();
            recorder
                .record("resource c1 69636f6e2e646d69", "ok resource datahex=0102")
                .unwrap();
            recorder
                .record(
                    "client_command c1 4669782d537461742d50616e656c",
                    "ok client_command protocol=5 client=c1",
                )
                .unwrap();
            recorder
                .record(
                    "browser_topic c1 62796f6e643a2f2f3f616374696f6e3d7265616479",
                    "ok browser_topic protocol=4 client=c1",
                )
                .unwrap();
            recorder
                .record(
                    "screen_pointer c1 1:0 click - 6c6566743d31",
                    "ok screen_pointer protocol=3 client=c1",
                )
                .unwrap();
            recorder
                .record(
                    "map_pointer c1 1:0 4 5 1 6d61696e2e6d6170 6c6566743d31",
                    "ok map_pointer protocol=6 client=c1",
                )
                .unwrap();
            recorder
                .record("resource c1 69636f6e2e646d69", "ok resource datahex=0102")
                .unwrap();
        }
        let mut transport = RemoteTransport::replay(&path).unwrap();
        transport.attach().unwrap();
        transport.send_command("Fix-Stat-Panel").unwrap();
        transport
            .send_browser_topic("byond://?action=ready")
            .unwrap();
        transport
            .send_screen_pointer((1, 0), "click", "", "left=1")
            .unwrap();
        transport
            .send_map_pointer(
                (1, 0),
                WorldCoordinate { x: 4, y: 5, z: 1 },
                "main.map",
                "left=1",
            )
            .unwrap();
        assert_eq!(transport.send_movement(1, 0).unwrap(), None);
        assert_eq!(transport.poll_ui_events().unwrap().len(), 1);
        let mut replay = ReplayReader::open(&path).unwrap();
        assert_eq!(
            replay.exchange("attach").unwrap(),
            "ok attach client=c1 x=4 y=5 z=1"
        );
        assert!(replay.exchange("ui_events c1").unwrap().contains("count=1"));
        assert_eq!(
            replay.exchange("ui_events c1").unwrap(),
            "ok ui_events count=0\n"
        );
        assert_eq!(
            replay
                .exchange("browser_topic c1 62796f6e643a2f2f3f616374696f6e3d7265616479")
                .unwrap(),
            "ok browser_topic protocol=4 client=c1"
        );
        assert_eq!(
            replay
                .exchange("client_command c1 756e7265636f72646564")
                .unwrap(),
            "ok client_command replay=ignored\n"
        );
        for _ in 0..2 {
            assert_eq!(
                replay.exchange("resource c1 69636f6e2e646d69").unwrap(),
                "ok resource datahex=0102"
            );
        }
        assert_eq!(
            replay
                .exchange("resource c1 6d697373696e672e68746d6c")
                .unwrap(),
            "ok resource datahex="
        );
        assert!(replay.exchange("map_snapshot c1").is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn browser_byond_urls_decode_local_window_api_parameters() {
        let (path, parameters) = parse_byond_url(
            "byond://winset?element=output_selector.legacy_output_selector&left=output_browser",
        )
        .unwrap();
        assert_eq!(path, "winset");
        assert_eq!(
            parameters["element"],
            "output_selector.legacy_output_selector"
        );
        assert_eq!(parameters["left"], "output_browser");

        let (path, parameters) = parse_byond_url(
            "byond://winget?id=browseroutput&property=size%2Cview-size&callback=Byond.__callbacks__%5B12%5D",
        )
        .unwrap();
        assert_eq!(path, "winget");
        assert_eq!(parameters["property"], "size,view-size");
        assert!(valid_byond_callback(&parameters["callback"]));
        assert!(valid_byond_callback("checkoutput"));
        assert!(valid_byond_callback("Dream64.callbacks.checkoutput"));
        assert!(!valid_byond_callback("alert(1)"));
        assert!(!valid_byond_callback("checkoutput;alert(1)"));
    }

    #[cfg(windows)]
    #[test]
    fn native_menu_bar_preserves_dmf_categories_separators_and_commands() {
        let source = "menu \"menu\"\n\telem\n\t\tname = \"&File\"\n\t\tcommand = \"\"\n\telem\n\t\tname = \"&Reconnect\"\n\t\tcategory = \"&File\"\n\t\tcommand = \".reconnect\"\n\telem\n\t\tname = \"\"\n\t\tcategory = \"&File\"\n\t\tcommand = \"\"\n\telem\n\t\tname = \"&Quit\\tAlt-F4\"\n\t\tcategory = \"&File\"\n\t\tcommand = \".quit\"\n\telem \"help-menu\"\n\t\tname = \"&Help\"\n\t\tcommand = \"\"\n\telem \"Use Internet Routing Relay\"\n\t\tname = \"Internet Routing Relays\"\n\t\tcommand = \"internet-routing-relays\"\n";
        let tree = ControlTree::from_document(&dm_dmf::parse(source));
        let mut ui = dm_dmf::UiState::new(tree);
        ui.winset("/datum/verbs/menu/admin", "parent=menu;name=&Admin")
            .unwrap();
        ui.winset(
            "/client/proc/adminwho",
            "parent=/datum/verbs/menu/admin;name=Admin%20Who;command=adminwho",
        )
        .unwrap();
        ui.winset("help-menu", "index=1000").unwrap();
        let menu = NativeMenuBar::from_ui(&ui)
            .unwrap()
            .expect("DMF menu should become a native menu");

        assert_eq!(menu.root.items().len(), 4);
        assert_eq!(
            menu.commands.values().cloned().collect::<BTreeSet<_>>(),
            BTreeSet::from([
                ".quit".to_owned(),
                ".reconnect".to_owned(),
                "adminwho".to_owned(),
                "internet-routing-relays".to_owned()
            ])
        );
    }

    #[test]
    fn remote_prompt_event_and_typed_response_round_trip() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            for (expected, response) in [
                ("attach", "ok attach protocol=1 client=c1 x=1 y=1 z=1"),
                (
                    "ui_events c1",
                    "ok ui_events protocol=2 client=c1 count=1\nU 1 prompt 4 number 1 5469746c65 56616c75653f 3132 -\n",
                ),
                (
                    "prompt_response c1 4 number 17.5",
                    "ok prompt_response protocol=7 client=c1 id=4",
                ),
            ] {
                let request = String::from_utf8(read_frame(&mut stream).unwrap()).unwrap();
                assert_eq!(request, expected);
                write_frame(&mut stream, response.as_bytes()).unwrap();
            }
        });
        let mut transport = RemoteTransport::connect(address, None).unwrap();
        transport.attach().unwrap();
        let events = transport.poll_ui_events().unwrap();
        assert!(matches!(
            &events[0],
            (
                1,
                InboundUiCommand::Prompt(ClientPrompt {
                    id: 4,
                    kind: ClientPromptKind::Number,
                    default,
                    ..
                })
            ) if default == "12"
        ));
        transport
            .send_prompt_response(4, ClientPromptResponse::Number(17.5))
            .unwrap();
        server.join().unwrap();
    }

    #[test]
    fn remote_transport_attaches_requests_snapshot_and_sends_commands_and_cardinal_input() {
        let mut png_bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut png_bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgba);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&[255, 0, 0, 255])
                .unwrap();
        }
        let resource_response = format!(
            "ok resource_chunk protocol=3 pathhex=69636f6e2e646d69 offset=0 total={} eof=1 datahex={}",
            png_bytes.len(),
            encode_hex(&png_bytes)
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            for (expected, response) in [
                (
                    "attach",
                    "ok attach protocol=1 client=c1 mob=[0xd1] x=4 y=5 z=1",
                ),
                (
                    "client_command c1 6669782d63686174",
                    "ok client_command protocol=5 client=c1",
                ),
                (
                    "map_snapshot c1",
                    "ok map_snapshot protocol=2 width=8 height=8 z=1 tiles=2\nT 4 5 2f747572662f6f70656e 233131323233  1\nA 1:0 2f6f626a 69636f6e2e646d69 - 2 00000000 00000000 00000000 00000000 00000000 00000000 - 437f0000 0 0\nT 5 5 2f747572662f6f70656e2f666c6f6f72 -  0\n",
                ),
                (
                    "resource_chunk c1 69636f6e2e646d69 0 262144",
                    resource_response.as_str(),
                ),
                (
                    "map_pointer c1 1:0 4 5 1 6d61696e2e6d6170 6c6566743d31",
                    "ok map_pointer protocol=6 client=c1",
                ),
                ("move c1 east", "ok move x=5 y=5 z=1"),
                (
                    "map_snapshot c1",
                    "ok map_snapshot protocol=2 width=8 height=8 z=1 tiles=1\nT 5 5 2f747572662f6f70656e2f666c6f6f72 233434353536  0\n",
                ),
            ] {
                let request = String::from_utf8(read_frame(&mut stream).unwrap()).unwrap();
                assert_eq!(request, expected);
                write_frame(&mut stream, response.as_bytes()).unwrap();
            }
        });

        let mut transport = RemoteTransport::connect(address, None).unwrap();
        transport.attach().unwrap();
        transport.send_command("fix-chat").unwrap();
        let snapshot = transport.request_snapshot().unwrap();
        assert_eq!(snapshot.center, WorldCoordinate { x: 4, y: 5, z: 1 });
        assert_eq!(snapshot.cells.len(), 2);
        assert_eq!(
            snapshot.appearances.values().map(Vec::len).sum::<usize>(),
            1
        );
        let legacy = &snapshot.appearances[&(4, 5, 1)][0];
        assert_eq!(legacy.appearance_flags, 0);
        assert_eq!(legacy.mouse_opacity, 1);
        assert_eq!(snapshot.resources.len(), 1);
        transport
            .send_map_pointer(
                (1, 0),
                WorldCoordinate { x: 4, y: 5, z: 1 },
                "main.map",
                "left=1",
            )
            .unwrap();
        let moved = transport.send_movement(1, 0).unwrap().unwrap();
        assert_eq!(moved.center, WorldCoordinate { x: 5, y: 5, z: 1 });
        assert_eq!(moved.cells.len(), 1);
        server.join().unwrap();
    }

    #[test]
    fn snapshot_v4_decodes_narrow_appearance_mouse_policy() {
        let lines = [
            "A 1:0 2f6f626a 69636f6e2e646d69 - 2 00000000 00000000 00000000 00000000 00000000 00000000 - 437f0000 0 0 - 00000000 00000000 00000000 00000000 4096 2",
        ];
        let mut cursor = 0;
        let mut appearances = Vec::new();
        parse_appearance_tree(&lines, &mut cursor, &mut appearances).unwrap();
        assert_eq!(cursor, 1);
        assert_eq!(appearances.len(), 1);
        assert_eq!(appearances[0].appearance_flags, 4096);
        assert_eq!(appearances[0].mouse_opacity, 2);
    }

    #[test]
    fn remote_resource_transport_reassembles_bounded_chunks() {
        let bytes = (0..RESOURCE_CHUNK_BYTES as usize + 3)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let first = format!(
            "ok resource_chunk protocol=3 offset=0 total={} eof=0 datahex={}",
            bytes.len(),
            encode_hex(&bytes[..RESOURCE_CHUNK_BYTES as usize])
        );
        let second = format!(
            "ok resource_chunk protocol=3 offset={} total={} eof=1 datahex={}",
            RESOURCE_CHUNK_BYTES,
            bytes.len(),
            encode_hex(&bytes[RESOURCE_CHUNK_BYTES as usize..])
        );
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            for (expected, response) in [
                (
                    "attach".to_owned(),
                    "ok attach protocol=1 client=c1".to_owned(),
                ),
                (
                    "resource_chunk c1 69636f6e732f6c617267652e646d69 0 262144".to_owned(),
                    first,
                ),
                (
                    "resource_chunk c1 69636f6e732f6c617267652e646d69 262144 262144".to_owned(),
                    second,
                ),
            ] {
                let request = String::from_utf8(read_frame(&mut stream).unwrap()).unwrap();
                assert_eq!(request, expected);
                write_frame(&mut stream, response.as_bytes()).unwrap();
            }
        });
        let mut transport = RemoteTransport::connect(address, None).unwrap();
        transport.attach().unwrap();
        assert_eq!(
            transport.request_resource("icons/large.dmi").unwrap(),
            bytes
        );
        server.join().unwrap();
    }

    #[test]
    fn monk_shift_macros_apply_client_side_winset_on_down_and_up() {
        let document = dm_dmf::parse(
            "macro \"default\"\n\telem \"Shift\"\n\t\tname = \"SHIFT\"\n\t\tcommand = \".winset :map.right-click=false\"\n\telem \"ShiftUp\"\n\t\tname = \"SHIFT+UP\"\n\t\tcommand = \".winset :map.right-click=true\"\nwindow \"mapwindow\"\n\telem \"main\"\n\t\ttype = MAIN\n\t\tmacro = default\n\telem \"map\"\n\t\ttype = MAP\n\t\tright-click = true\n",
        );
        let tree = ControlTree::from_document(&document);
        let bindings = MacroBindings::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);

        assert_eq!(
            macro_key_name(KeyCode::ShiftLeft, ModifiersState::SHIFT, true).as_deref(),
            Some("SHIFT")
        );
        assert_eq!(
            dispatch_macro(&mut session, &bindings, "SHIFT"),
            Some(MacroDispatch {
                server_command: None
            })
        );
        assert_eq!(
            session.ui().winget("mapwindow.map", "right-click"),
            Ok("false".into())
        );
        assert_eq!(
            dispatch_macro(&mut session, &bindings, "SHIFT+UP"),
            Some(MacroDispatch {
                server_command: None
            })
        );
        assert_eq!(
            session.ui().winget("mapwindow.map", "right-click"),
            Ok("true".into())
        );
        assert!(session.take_events().is_empty());
    }

    #[test]
    fn runtime_default_binding_and_modified_key_up_emit_commands() {
        let document = dm_dmf::parse(
            "macro \"default\"\nwindow \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\t\tmacro = default\n",
        );
        let tree = ControlTree::from_document(&document);
        let mut bindings = MacroBindings::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        for (control, parameters) in [
            ("default-W", "parent=default;name=W;command=.north"),
            (
                "default-ctrl-shift-k-up",
                "parent=default;name=CTRL+SHIFT+K+UP;command=.modified-release",
            ),
        ] {
            session
                .apply_command(UiCommand::WinSet {
                    control: control.to_owned(),
                    parameters: parameters.to_owned(),
                })
                .unwrap();
            bindings.refresh_control(session.ui(), control);
        }

        assert_eq!(
            dispatch_macro(&mut session, &bindings, "W"),
            Some(MacroDispatch {
                server_command: None
            })
        );
        assert_eq!(
            macro_key_name(
                KeyCode::KeyK,
                ModifiersState::CONTROL | ModifiersState::SHIFT,
                false
            )
            .as_deref(),
            Some("CTRL+SHIFT+K+UP")
        );
        assert_eq!(
            dispatch_macro(&mut session, &bindings, "CTRL+SHIFT+K+UP"),
            Some(MacroDispatch {
                server_command: None
            })
        );
        assert_eq!(
            session.take_events(),
            vec![
                UiEvent::Command {
                    command: ".north".into()
                },
                UiEvent::Command {
                    command: ".modified-release".into()
                },
            ]
        );
    }

    #[test]
    fn input_submission_matches_opendream_prefix_and_prefill_rules() {
        let mut prefixed = InputState {
            command: "ooc ".into(),
            text: "hello station".into(),
        };
        assert_eq!(take_input_submission(&mut prefixed), "ooc hello station");
        assert!(prefixed.text.is_empty());

        let mut prefilled = InputState {
            command: "!say \"".into(),
            text: "say \"hello\"".into(),
        };
        assert_eq!(take_input_submission(&mut prefilled), "say \"hello\"");
        assert_eq!(prefilled.text, "say \"");
    }
}
