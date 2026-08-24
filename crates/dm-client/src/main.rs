//! The first visible Dream64 local-client shell.

#![deny(missing_docs)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    path::Path,
};
use std::{
    io::{Read, Write},
    net::{IpAddr, SocketAddr, TcpStream},
};

use dm_compiler::CompilerDatabase;
use dm_dmf::{
    ControlTree, ControlType, PixelRect, UiCommand, UiEvent, WEBVIEW2_BYOND_BRIDGE_BOOTSTRAP,
};
use dm_lifecycle::artifact::CompiledArtifact;
use dm_project::Project;
use dm_runtime::RuntimeImage;
use dm_value::{DatumId, FieldName, TypePath, Value};
use dm_vm::ExecutionState;
use dm_world::{AtomCategory, WorldCoordinate};
use font8x8::UnicodeFonts;
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{Window, WindowAttributes, WindowId};

mod gpu;
mod sprite;
#[cfg(test)]
use sprite::composite_tile;
use sprite::{Appearance, SpriteCache, composite_native, rasterize_world_appearance};

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

const DEFAULT_RETAINED_OUTPUT_LINES: usize = 512;

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
    let mut application = LocalClient {
        context,
        surface: None,
        runtime,
        client,
        local_input_events: 0,
        inbound_ui_events: 0,
        transport,
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

#[derive(Clone, Debug, PartialEq)]
struct ClientLayout {
    window_width: u32,
    window_height: u32,
    map: PixelRect,
    map_tile_size: u32,
    map_zoom: f32,
    map_zoom_mode: String,
    map_letterbox: bool,
    browser: Option<PixelRect>,
    browser_control: Option<String>,
    output_controls: Vec<String>,
    output_rects: BTreeMap<String, PixelRect>,
    input_rects: BTreeMap<String, PixelRect>,
    button_rects: BTreeMap<String, PixelRect>,
    label_rects: BTreeMap<String, PixelRect>,
}

impl ClientLayout {
    fn from_tree(tree: &ControlTree) -> Self {
        let controls = tree.windows.iter().flat_map(|window| &window.controls);
        let main = controls
            .clone()
            .find(|control| {
                control.control_type == ControlType::Main
                    && control.property("is-default").is_some_and(dmf_truthy)
                    && !control.property("is-visible").is_some_and(dmf_false)
            })
            .or_else(|| {
                controls
                    .clone()
                    .find(|control| control.control_type == ControlType::Main)
            });
        let main_rect = main
            .and_then(dm_dmf::ControlNode::pixel_rect)
            .unwrap_or(PixelRect {
                x: 0,
                y: 0,
                width: 1_024,
                height: 768,
            });
        let map = controls
            .clone()
            .find(|control| control.control_type == ControlType::Map)
            .and_then(dm_dmf::ControlNode::pixel_rect)
            .unwrap_or(PixelRect {
                x: 16,
                y: 74,
                width: 640,
                height: 528,
            });
        let browser_node = controls
            .clone()
            .filter(|control| control.control_type == ControlType::Browser)
            .filter(|control| !control.property("is-visible").is_some_and(dmf_false))
            .find(|control| {
                control.id.as_deref() == Some("browseroutput")
                    || control.property("is-default").is_some_and(dmf_truthy)
            })
            .or_else(|| {
                controls
                    .clone()
                    .find(|control| control.control_type == ControlType::Browser)
            });
        let browser = browser_node.and_then(dm_dmf::ControlNode::pixel_rect);
        let browser_control = browser_node.and_then(|node| qualified_control(tree, node));
        let output_controls = controls
            .filter(|control| control.control_type == ControlType::Output)
            .filter_map(|node| qualified_control(tree, node))
            .collect::<Vec<_>>();
        let output_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Output)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let input_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Input)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let button_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Button)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let label_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Label)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let mut layout = Self {
            window_width: main_rect.width,
            window_height: main_rect.height,
            map,
            map_tile_size: 32,
            map_zoom: 0.0,
            map_zoom_mode: "normal".to_owned(),
            map_letterbox: true,
            browser,
            browser_control,
            output_controls,
            output_rects,
            input_rects,
            button_rects,
            label_rects,
        };
        layout.apply_resolved_panes(&dm_dmf::UiState::new(tree.clone()));
        layout
    }

    fn refresh_from_ui(&mut self, ui: &dm_dmf::UiState) {
        self.apply_resolved_panes(ui);
    }

    fn apply_resolved_panes(&mut self, ui: &dm_dmf::UiState) {
        self.apply_resolved_panes_in(ui, None);
    }

    fn apply_resolved_panes_in(&mut self, ui: &dm_dmf::UiState, viewport: Option<(u32, u32)>) {
        let resolved = resolve_pane_layout_in(ui, viewport);
        self.window_width = resolved.root.width;
        self.window_height = resolved.root.height;
        if let Some((address, rect)) = resolved
            .controls
            .iter()
            .find(|(address, _)| control_has_type(ui.tree(), address, ControlType::Map))
        {
            self.map = *rect;
            self.map_tile_size = ui
                .winget(address, "tile-size")
                .ok()
                .filter(|value| !value.is_empty())
                .or_else(|| ui.winget(address, "icon-size").ok())
                .and_then(|value| value.parse().ok())
                .filter(|value| *value > 0)
                .unwrap_or(32);
            self.map_zoom = ui
                .winget(address, "zoom")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|value: &f32| value.is_finite() && *value >= 0.0)
                .unwrap_or(0.0);
            self.map_zoom_mode = ui
                .winget(address, "zoom-mode")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "normal".to_owned());
            self.map_letterbox = ui
                .winget(address, "letterbox")
                .ok()
                .filter(|value| !value.is_empty())
                .is_none_or(|value| !dmf_false(&value));
        }
        let browser = resolved
            .controls
            .iter()
            .find(|(address, _)| {
                control_has_type(ui.tree(), address, ControlType::Browser)
                    && (address.ends_with(".browseroutput")
                        || ui
                            .winget(address, "is-default")
                            .is_ok_and(|value| dmf_truthy(&value)))
            })
            .or_else(|| {
                resolved
                    .controls
                    .iter()
                    .find(|(address, _)| control_has_type(ui.tree(), address, ControlType::Browser))
            });
        self.browser_control = browser.map(|(address, _)| address.clone());
        self.browser = browser.map(|(_, rect)| *rect);
        self.output_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Output))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.input_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Input))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.button_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Button))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.label_rects = resolved
            .controls
            .into_iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Label))
            .collect();
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct InputState {
    command: String,
    text: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ButtonState {
    text: String,
    command: String,
    checked: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct LabelState {
    text: String,
}

fn input_states_from_ui(
    ui: &dm_dmf::UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, InputState> {
    layout
        .input_rects
        .keys()
        .map(|address| {
            let command = ui.winget(address, "command").unwrap_or_default();
            let text = command
                .strip_prefix('!')
                .map_or_else(String::new, str::to_owned);
            (address.clone(), InputState { command, text })
        })
        .collect()
}

fn take_input_submission(state: &mut InputState) -> String {
    let command = if state.command.starts_with('!') {
        state.text.clone()
    } else {
        format!("{}{}", state.command, state.text)
    };
    state.text = state
        .command
        .strip_prefix('!')
        .map_or_else(String::new, str::to_owned);
    command
}

fn button_states_from_ui(
    ui: &dm_dmf::UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, ButtonState> {
    layout
        .button_rects
        .keys()
        .map(|address| {
            (
                address.clone(),
                ButtonState {
                    text: ui.winget(address, "text").unwrap_or_default(),
                    command: ui.winget(address, "command").unwrap_or_default(),
                    checked: ui
                        .winget(address, "is-checked")
                        .is_ok_and(|value| dmf_truthy(&value)),
                },
            )
        })
        .collect()
}

fn label_states_from_ui(
    ui: &dm_dmf::UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, LabelState> {
    layout
        .label_rects
        .keys()
        .map(|address| {
            (
                address.clone(),
                LabelState {
                    text: ui.winget(address, "text").unwrap_or_default(),
                },
            )
        })
        .collect()
}

struct ResolvedPaneLayout {
    root: PixelRect,
    controls: BTreeMap<String, PixelRect>,
}

fn resolve_pane_layout_in(
    ui: &dm_dmf::UiState,
    viewport_size: Option<(u32, u32)>,
) -> ResolvedPaneLayout {
    let tree = ui.tree();
    let root_window = tree
        .windows
        .iter()
        .find(|window| {
            window.controls.iter().any(|control| {
                control.control_type == ControlType::Main
                    && ui
                        .winget(
                            &format!("{}.{}", window.id, control.id.as_deref().unwrap_or("")),
                            "is-default",
                        )
                        .is_ok_and(|value| dmf_truthy(&value))
            })
        })
        .or_else(|| tree.windows.first());
    let root = root_window
        .and_then(|window| {
            window
                .controls
                .iter()
                .find(|control| control.control_type == ControlType::Main)
        })
        .and_then(dm_dmf::ControlNode::pixel_rect)
        .unwrap_or(PixelRect {
            x: 0,
            y: 0,
            width: 1_024,
            height: 768,
        });
    let viewport = PixelRect {
        x: 0,
        y: 0,
        width: viewport_size.map_or(root.width, |size| size.0),
        height: viewport_size.map_or(root.height, |size| size.1),
    };
    let mut controls = BTreeMap::new();
    let mut active = Vec::new();
    if let Some(window) = root_window {
        resolve_pane_window(ui, &window.id, viewport, &mut controls, &mut active);
    }
    ResolvedPaneLayout {
        root: viewport,
        controls,
    }
}

fn resolve_pane_window(
    ui: &dm_dmf::UiState,
    window_id: &str,
    viewport: PixelRect,
    resolved: &mut BTreeMap<String, PixelRect>,
    active: &mut Vec<String>,
) {
    if active.iter().any(|entry| entry == window_id) {
        return;
    }
    let Some(window) = ui
        .tree()
        .windows
        .iter()
        .find(|window| window.id == window_id)
    else {
        return;
    };
    active.push(window_id.to_owned());
    let source = window
        .controls
        .iter()
        .find(|control| control.control_type == ControlType::Main)
        .and_then(dm_dmf::ControlNode::pixel_rect)
        .unwrap_or(PixelRect {
            x: 0,
            y: 0,
            width: viewport.width,
            height: viewport.height,
        });
    for control in &window.controls {
        if control.control_type == ControlType::Main {
            continue;
        }
        let Some(id) = control.id.as_deref() else {
            continue;
        };
        let address = format!("{window_id}.{id}");
        if ui
            .winget(&address, "is-visible")
            .is_ok_and(|value| dmf_false(&value))
        {
            continue;
        }
        let Some(local) = effective_pixel_rect(ui, &address, control.pixel_rect()) else {
            continue;
        };
        let rect = anchored_rect(ui, &address, local, source, viewport);
        if control
            .property("type")
            .is_some_and(|kind| kind.eq_ignore_ascii_case("CHILD"))
        {
            let left = ui.winget(&address, "left").unwrap_or_default();
            let right = ui.winget(&address, "right").unwrap_or_default();
            let vertical_value = ui.winget(&address, "is-vert").unwrap_or_default();
            let vertical = dmf_truthy(&vertical_value);
            let split = ui
                .winget(&address, "splitter")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(50)
                .min(100);
            let (first, second) = split_rect(rect, vertical, split, right.is_empty());
            if !left.is_empty() {
                resolve_pane_window(ui, &left, first, resolved, active);
            }
            if !right.is_empty() {
                resolve_pane_window(ui, &right, second, resolved, active);
            }
        } else {
            resolved.insert(address, rect);
        }
    }
    active.pop();
}

fn anchored_rect(
    ui: &dm_dmf::UiState,
    address: &str,
    base: PixelRect,
    source: PixelRect,
    target: PixelRect,
) -> PixelRect {
    let first = ui
        .winget(address, "anchor1")
        .ok()
        .and_then(|value| parse_anchor(&value));
    let second = ui
        .winget(address, "anchor2")
        .ok()
        .and_then(|value| parse_anchor(&value));
    let dx = i64::from(target.width) - i64::from(source.width);
    let dy = i64::from(target.height) - i64::from(source.height);
    let left = i64::from(base.x) + anchor_delta(dx, first.map(|value| value.0));
    let top = i64::from(base.y) + anchor_delta(dy, first.map(|value| value.1));
    let right = i64::from(base.x + base.width) + anchor_delta(dx, second.map(|value| value.0));
    let bottom = i64::from(base.y + base.height) + anchor_delta(dy, second.map(|value| value.1));
    PixelRect {
        x: target
            .x
            .saturating_add(u32::try_from(left.max(0)).unwrap_or(0)),
        y: target
            .y
            .saturating_add(u32::try_from(top.max(0)).unwrap_or(0)),
        width: u32::try_from((right - left).max(0)).unwrap_or(0),
        height: u32::try_from((bottom - top).max(0)).unwrap_or(0),
    }
}

fn parse_anchor(value: &str) -> Option<(i32, i32)> {
    let (x, y) = value.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

fn anchor_delta(delta: i64, anchor: Option<i32>) -> i64 {
    anchor
        .filter(|value| *value >= 0)
        .map_or(0, |value| delta * i64::from(value) / 100)
}

fn split_rect(
    rect: PixelRect,
    vertical: bool,
    percent: u32,
    only_one: bool,
) -> (PixelRect, PixelRect) {
    if only_one {
        return (
            rect,
            PixelRect {
                x: rect.x + rect.width,
                y: rect.y,
                width: 0,
                height: rect.height,
            },
        );
    }
    if vertical {
        let first = rect.width.saturating_mul(percent) / 100;
        (
            PixelRect {
                width: first,
                ..rect
            },
            PixelRect {
                x: rect.x + first,
                width: rect.width - first,
                ..rect
            },
        )
    } else {
        let first = rect.height.saturating_mul(percent) / 100;
        (
            PixelRect {
                height: first,
                ..rect
            },
            PixelRect {
                y: rect.y + first,
                height: rect.height - first,
                ..rect
            },
        )
    }
}

fn control_has_type(tree: &ControlTree, address: &str, expected: ControlType) -> bool {
    address
        .split_once('.')
        .and_then(|(window, control)| tree.control(window, control))
        .is_some_and(|control| control.control_type == expected)
}

fn effective_pixel_rect(
    ui: &dm_dmf::UiState,
    address: &str,
    fallback: Option<PixelRect>,
) -> Option<PixelRect> {
    let mut rect = fallback?;
    if let Ok(position) = ui.winget(address, "pos")
        && let Some((x, y)) = parse_pair(&position, ',')
    {
        rect.x = x;
        rect.y = y;
    }
    if let Ok(size) = ui.winget(address, "size")
        && let Some((width, height)) = parse_pair(&size, 'x')
    {
        rect.width = width;
        rect.height = height;
    }
    Some(rect)
}

fn parse_pair(value: &str, separator: char) -> Option<(u32, u32)> {
    let (left, right) = value.trim().split_once(separator)?;
    Some((left.trim().parse().ok()?, right.trim().parse().ok()?))
}

fn qualified_control(tree: &ControlTree, needle: &dm_dmf::ControlNode) -> Option<String> {
    let id = needle.id.as_deref()?;
    tree.windows.iter().find_map(|window| {
        window
            .controls
            .iter()
            .any(|node| std::ptr::eq(node, needle))
            .then(|| format!("{}.{}", window.id, id))
    })
}

#[derive(Clone, Debug, PartialEq)]
enum InboundUiCommand {
    WinSet {
        control: String,
        parameters: String,
    },
    Output {
        control: Option<String>,
        message: String,
    },
    BrowseResource {
        name: String,
        data: Vec<u8>,
    },
    Browse {
        control: String,
        html: String,
    },
    Prompt(ClientPrompt),
    Sound(SoundUpdate),
}

#[derive(Clone, Debug, PartialEq)]
struct SoundUpdate {
    file: Option<String>,
    channel: i32,
    repeat: bool,
    volume: f32,
    frequency: f32,
    pan: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ClientPromptKind {
    Text,
    Message,
    Number,
    Color,
    File,
    List,
    Alert,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClientPrompt {
    id: u64,
    kind: ClientPromptKind,
    title: String,
    message: String,
    default: String,
    choices: Vec<String>,
    can_cancel: bool,
    edit: String,
    selected: usize,
}

#[derive(Default)]
struct UiPresentation {
    last_sequence: u64,
    browser_html: BTreeMap<String, String>,
    output_text: BTreeMap<String, Vec<String>>,
    browser_resources: BTreeMap<String, Vec<u8>>,
    pending_prompts: VecDeque<ClientPrompt>,
}

#[derive(Debug, PartialEq)]
enum BrowserUpdate {
    Html { control: String, html: String },
    Resource { control: String, path: String },
    Script { control: String, script: String },
    Sound(SoundUpdate),
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

impl UiPresentation {
    fn apply(
        &mut self,
        sequence: u64,
        command: InboundUiCommand,
        session: &mut dm_dmf::ClientSession,
        layout: &mut ClientLayout,
    ) -> Result<Option<BrowserUpdate>, String> {
        if sequence <= self.last_sequence {
            return Ok(None);
        }
        let browser_update = match command {
            InboundUiCommand::WinSet {
                control,
                parameters,
            } => {
                session
                    .apply_command(UiCommand::WinSet {
                        control,
                        parameters,
                    })
                    .map_err(|error| format!("{error:?}"))?;
                // Winset updates authoritative UiState overrides. Re-resolve
                // native presentation geometry immediately so subsequent UI
                // events and the next redraw observe the same property state.
                layout.refresh_from_ui(session.ui());
                None
            }
            InboundUiCommand::Output { control, message } => {
                let control = control
                    .or_else(|| layout.output_controls.first().cloned())
                    .ok_or("skin has no OUTPUT control")?;
                if let Some((target, function)) = control.split_once(':')
                    && let Some(control) =
                        resolve_browser_output_target(session.ui().tree(), target)
                {
                    Some(BrowserUpdate::Script {
                        control,
                        script: browser_output_script(function, &message),
                    })
                } else {
                    match resolve_control_type(session.ui().tree(), &control, ControlType::Output) {
                        Ok(control) => {
                            let retained_lines = output_line_limit(session.ui(), &control);
                            let lines = self.output_text.entry(control).or_default();
                            lines.extend(message.lines().map(str::to_owned));
                            if lines.len() > retained_lines {
                                lines.drain(..lines.len() - retained_lines);
                            }
                            None
                        }
                        Err(output_error) => {
                            if let Ok(control) = resolve_control_type(
                                session.ui().tree(),
                                &control,
                                ControlType::Browser,
                            ) {
                                let path =
                                    normalize_resource_path("", &message).ok_or_else(|| {
                                        format!("invalid browser output resource path {message:?}")
                                    })?;
                                Some(BrowserUpdate::Resource { control, path })
                            } else {
                                return Err(output_error);
                            }
                        }
                    }
                }
            }
            InboundUiCommand::BrowseResource { name, data } => {
                let name = normalize_resource_path("", &name)
                    .ok_or_else(|| format!("invalid browser resource path {name:?}"))?;
                self.browser_resources.insert(name, data);
                None
            }
            InboundUiCommand::Browse { control, html } => {
                // BYOND calls this selector `window`: a skin control may be
                // qualified or unqualified, while an unknown value names a
                // future popup. Preserve popup documents without routing them
                // into the skin's embedded browser.
                let resolved =
                    resolve_control_type(session.ui().tree(), &control, ControlType::Browser).ok();
                let key = resolved.clone().unwrap_or(control);
                self.browser_html.insert(key, html.clone());
                resolved.map(|control| BrowserUpdate::Html { control, html })
            }
            InboundUiCommand::Prompt(mut prompt) => {
                prompt.edit.clone_from(&prompt.default);
                prompt.selected = prompt
                    .choices
                    .iter()
                    .position(|choice| choice == &prompt.default)
                    .unwrap_or(0);
                self.pending_prompts.push_back(prompt);
                None
            }
            InboundUiCommand::Sound(sound) => Some(BrowserUpdate::Sound(sound)),
        };
        self.last_sequence = sequence;
        Ok(browser_update)
    }
}

fn output_line_limit(ui: &dm_dmf::UiState, control: &str) -> usize {
    ui.winget(control, "lines")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|limit| *limit > 0)
        .unwrap_or(DEFAULT_RETAINED_OUTPUT_LINES)
}

fn resolve_browser_output_target(tree: &ControlTree, target: &str) -> Option<String> {
    resolve_control_type(tree, target, ControlType::Browser)
        .ok()
        .or_else(|| {
            // BYOND's embedded-browser output address is
            // `<control>.browser:<javascript function>`. The `.browser`
            // segment names the browser document, not a DMF control.
            let control = target.strip_suffix(".browser")?;
            resolve_control_type(tree, control, ControlType::Browser).ok()
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

fn normalize_resource_path(base: &str, reference: &str) -> Option<String> {
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

fn resolve_control_type(
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

fn dmf_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "yes" | "1"
    )
}

fn dmf_false(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "false" | "no" | "0"
    )
}

fn load_skin(path: Option<&Path>) -> Result<(String, String), std::io::Error> {
    let Some(path) = path else {
        return Ok((LOCAL_SKIN.to_owned(), "development skin".to_owned()));
    };
    let source = std::fs::read_to_string(path)?;
    Ok((source, path.display().to_string()))
}

#[derive(Debug)]
struct LaunchOptions {
    skin: Option<PathBuf>,
    world: Option<PathBuf>,
    map: Option<PathBuf>,
    connect: SocketAddr,
    record: Option<PathBuf>,
    replay: Option<PathBuf>,
    startup_replay: Option<PathBuf>,
}

impl LaunchOptions {
    fn parse() -> Result<Self, String> {
        let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
        if arguments.is_empty() {
            let address = prompt_for_server()?;
            Self::parse_from(["--connect".into(), address.to_string().into()])
        } else {
            Self::parse_from(arguments)
        }
    }

    fn parse_from(arguments: impl IntoIterator<Item = std::ffi::OsString>) -> Result<Self, String> {
        let mut options = Self {
            skin: None,
            world: None,
            map: None,
            connect: "127.0.0.1:51664"
                .parse()
                .expect("default loopback address is valid"),
            record: None,
            replay: None,
            startup_replay: None,
        };
        let mut connect_seen = false;
        let mut arguments = arguments.into_iter();
        while let Some(argument) = arguments.next() {
            match argument.to_string_lossy().as_ref() {
                "--skin" => set_path_option(
                    &mut options.skin,
                    next_path(&mut arguments, "--skin")?,
                    "--skin",
                )?,
                "--world" => set_path_option(
                    &mut options.world,
                    next_path(&mut arguments, "--world")?,
                    "--world",
                )?,
                "--map" => set_path_option(
                    &mut options.map,
                    next_path(&mut arguments, "--map")?,
                    "--map",
                )?,
                "--connect" => {
                    if std::mem::replace(&mut connect_seen, true) {
                        return Err("--connect may only be specified once".to_owned());
                    }
                    let value = arguments
                        .next()
                        .ok_or_else(|| "--connect requires an IP address and port".to_owned())?;
                    options.connect = value
                        .to_string_lossy()
                        .parse::<SocketAddr>()
                        .map_err(|error| format!("invalid --connect address: {error}"))?;
                }
                "--record-replay" => set_path_option(
                    &mut options.record,
                    next_path(&mut arguments, "--record-replay")?,
                    "--record-replay",
                )?,
                "--replay" => set_path_option(
                    &mut options.replay,
                    next_path(&mut arguments, "--replay")?,
                    "--replay",
                )?,
                "--startup-replay" => set_path_option(
                    &mut options.startup_replay,
                    next_path(&mut arguments, "--startup-replay")?,
                    "--startup-replay",
                )?,
                other if other.ends_with(".dmf") && options.skin.is_none() => {
                    options.skin = Some(PathBuf::from(argument));
                }
                other => {
                    return Err(format!(
                        "unknown client argument {other:?}; use [--connect 127.0.0.1:51664] [--startup-replay <file>] [--record-replay <file>] or --replay <file>, or --world <.dme> --map <.dmm> [--skin <.dmf>]"
                    ));
                }
            }
        }
        if options.map.is_some() && options.world.is_none() {
            return Err("--map requires --world".to_owned());
        }
        if options.record.is_some() && options.replay.is_some() {
            return Err("--record-replay and --replay are mutually exclusive".to_owned());
        }
        if options.replay.is_some() && options.startup_replay.is_some() {
            return Err("--replay and --startup-replay are mutually exclusive".to_owned());
        }
        if options.world.is_some() && (options.record.is_some() || options.replay.is_some()) {
            return Err("replay recording/playback cannot be combined with --world".to_owned());
        }
        if options.world.is_some() && options.startup_replay.is_some() {
            return Err("--startup-replay cannot be combined with --world".to_owned());
        }
        Ok(options)
    }
}

#[cfg(windows)]
fn prompt_for_server() -> Result<SocketAddr, String> {
    const SCRIPT: &str = r#"
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
$form = New-Object System.Windows.Forms.Form
$form.Text = 'Connect to Dream64'
$form.ClientSize = New-Object System.Drawing.Size(360, 165)
$form.FormBorderStyle = 'FixedDialog'
$form.StartPosition = 'CenterScreen'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true

$ipLabel = New-Object System.Windows.Forms.Label
$ipLabel.Text = 'Server IP address'
$ipLabel.Location = New-Object System.Drawing.Point(20, 18)
$ipLabel.AutoSize = $true
$form.Controls.Add($ipLabel)

$ip = New-Object System.Windows.Forms.TextBox
$ip.Location = New-Object System.Drawing.Point(20, 40)
$ip.Size = New-Object System.Drawing.Size(320, 23)
$form.Controls.Add($ip)

$portLabel = New-Object System.Windows.Forms.Label
$portLabel.Text = 'Port'
$portLabel.Location = New-Object System.Drawing.Point(20, 75)
$portLabel.AutoSize = $true
$form.Controls.Add($portLabel)

$port = New-Object System.Windows.Forms.NumericUpDown
$port.Location = New-Object System.Drawing.Point(20, 97)
$port.Size = New-Object System.Drawing.Size(120, 23)
$port.Minimum = 1
$port.Maximum = 65535
$port.Value = 51664
$form.Controls.Add($port)

$connect = New-Object System.Windows.Forms.Button
$connect.Text = 'Connect'
$connect.Location = New-Object System.Drawing.Point(178, 96)
$connect.Size = New-Object System.Drawing.Size(78, 26)
$connect.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.AcceptButton = $connect
$form.Controls.Add($connect)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Text = 'Cancel'
$cancel.Location = New-Object System.Drawing.Point(262, 96)
$cancel.Size = New-Object System.Drawing.Size(78, 26)
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancel
$form.Controls.Add($cancel)

$form.Add_Shown({ $ip.Focus() })
if ($form.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($ip.Text.Trim() + '|' + [int]$port.Value)
    exit 0
}
exit 1
"#;
    loop {
        let output = std::process::Command::new("powershell.exe")
            .args(["-NoProfile", "-STA", "-Command", SCRIPT])
            .output()
            .map_err(|error| format!("could not open connection window: {error}"))?;
        if !output.status.success() {
            return Err("connection cancelled".to_owned());
        }
        let response = String::from_utf8(output.stdout)
            .map_err(|_| "connection window returned invalid text".to_owned())?;
        let (ip, port) = response
            .trim()
            .split_once('|')
            .ok_or("connection window returned an invalid address")?;
        match (ip.parse::<IpAddr>(), port.parse::<u16>()) {
            (Ok(ip), Ok(port @ 1..)) => return Ok(SocketAddr::new(ip, port)),
            _ => {
                let _ = std::process::Command::new("powershell.exe")
                    .args([
                        "-NoProfile",
                        "-STA",
                        "-Command",
                        "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Enter a valid server IP address.', 'Dream64', 'OK', 'Warning')",
                    ])
                    .status();
            }
        }
    }
}

#[cfg(not(windows))]
fn prompt_for_server() -> Result<SocketAddr, String> {
    println!("Dream64 Server Connection\n");
    let ip = loop {
        let value = prompt_line("Server IP address: ")?;
        match value.parse::<IpAddr>() {
            Ok(ip) => break ip,
            Err(error) => eprintln!("Invalid IP address: {error}"),
        }
    };
    let port = loop {
        let value = prompt_line("Server port [51664]: ")?;
        if value.is_empty() {
            break 51_664;
        }
        match value.parse::<u16>() {
            Ok(0) => eprintln!("Port must be between 1 and 65535."),
            Ok(port) => break port,
            Err(error) => eprintln!("Invalid port: {error}"),
        }
    };
    Ok(SocketAddr::new(ip, port))
}

#[cfg(not(windows))]
fn prompt_line(label: &str) -> Result<String, String> {
    print!("{label}");
    std::io::stdout()
        .flush()
        .map_err(|error| format!("could not display connection prompt: {error}"))?;
    let mut value = String::new();
    std::io::stdin()
        .read_line(&mut value)
        .map_err(|error| format!("could not read connection prompt: {error}"))?;
    Ok(value.trim().to_owned())
}

fn set_path_option(slot: &mut Option<PathBuf>, value: PathBuf, flag: &str) -> Result<(), String> {
    if slot.replace(value).is_some() {
        return Err(format!("{flag} may only be specified once"));
    }
    Ok(())
}

fn next_path(
    arguments: &mut impl Iterator<Item = std::ffi::OsString>,
    flag: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} requires a path"))
}

struct WorldScene {
    cells: BTreeMap<(i32, i32, i32), u32>,
    turfs: BTreeMap<(i32, i32, i32), DatumId>,
    player: WorldCoordinate,
    player_mob: DatumId,
    label: String,
}

/// A transport-neutral map image returned to a connected client.
#[derive(Clone, Debug, PartialEq)]
struct MapSnapshot {
    center: WorldCoordinate,
    cells: BTreeMap<(i32, i32, i32), u32>,
    turf_targets: BTreeMap<(i32, i32, i32), (u32, u32)>,
    appearances: BTreeMap<(i32, i32, i32), Vec<Appearance>>,
    screen: Vec<ScreenAppearance>,
    resources: BTreeMap<PathBuf, Vec<u8>>,
}

fn snapshot_has_lobby_screen(snapshot: &MapSnapshot) -> bool {
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
struct ScreenAppearance {
    datum_index: u32,
    datum_generation: u32,
    map_control: Option<String>,
    screen_loc: String,
    type_path: String,
    insertion: usize,
    appearances: Vec<Appearance>,
}

/// In-process implementation of the client/server exchange. Keeping the
/// window behind this boundary makes a future TCP transport a codec swap
/// rather than another renderer implementation.
struct LoopbackTransport {
    scene: WorldScene,
    attached: Option<DatumId>,
}

enum ClientTransport {
    Pending(PendingRemoteTransport),
    Remote(RemoteTransport),
    Offline(LoopbackTransport),
}

struct PendingRemoteTransport {
    address: SocketAddr,
    record: Option<PathBuf>,
    next_attempt: std::time::Instant,
    last_error: Option<String>,
}

enum ClientPromptResponse {
    Null,
    Text(String),
    Number(f32),
    Choice(usize),
}

impl ClientTransport {
    fn try_connect(&mut self) -> bool {
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
                transport.send_readiness("skin_ready")?;
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

    fn request_snapshot(&mut self) -> Result<MapSnapshot, String> {
        match self {
            Self::Pending(_) => Err("server is still starting".to_owned()),
            Self::Remote(transport) => transport.request_snapshot(),
            Self::Offline(transport) => Ok(transport.request_snapshot(10, 7)),
        }
    }

    fn request_screen_snapshot(&mut self) -> Result<MapSnapshot, String> {
        match self {
            Self::Pending(_) => Err("server is still starting".to_owned()),
            Self::Remote(transport) => transport.request_screen_snapshot(),
            Self::Offline(transport) => Ok(transport.request_snapshot(10, 7)),
        }
    }

    fn send_movement(
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

    fn label(&self) -> &str {
        match self {
            Self::Pending(pending) => pending.last_error.as_deref().unwrap_or("Starting server"),
            Self::Remote(transport) => &transport.label,
            Self::Offline(transport) => transport.label(),
        }
    }

    fn poll_ui_events(&mut self) -> Result<Vec<(u64, InboundUiCommand)>, String> {
        match self {
            Self::Pending(_) => Ok(Vec::new()),
            Self::Remote(transport) => transport.poll_ui_events(),
            Self::Offline(_) => Ok(Vec::new()),
        }
    }

    fn acknowledge_ui(&mut self, sequence: u64) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.acknowledge_ui(sequence),
        }
    }

    fn mark_resources_ready(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.send_readiness("resources_ready"),
        }
    }

    fn mark_input_ready(&mut self) -> Result<(), String> {
        match self {
            Self::Pending(_) | Self::Offline(_) => Ok(()),
            Self::Remote(transport) => transport.send_readiness("input_ready"),
        }
    }

    fn send_screen_pointer(
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

    fn send_map_pointer(
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

    fn send_browser_topic(&mut self, topic: &str) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => transport.send_browser_topic(topic),
            Self::Offline(_) => Ok(()),
        }
    }

    fn send_command(&mut self, command: &str) -> Result<(), String> {
        match self {
            Self::Pending(_) => Ok(()),
            Self::Remote(transport) => transport.send_command(command),
            // Replays contain server-to-client state, not a live VM to mutate.
            Self::Offline(_) => Ok(()),
        }
    }

    fn reconnect(&mut self) -> Result<(), String> {
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

    fn request_resource(&mut self, path: &str) -> Result<Vec<u8>, String> {
        match self {
            Self::Pending(_) => Err(format!(
                "server is still starting; resource {path:?} unavailable"
            )),
            Self::Remote(transport) => transport.request_resource(path),
            Self::Offline(_) => Err(format!("offline world has no resource {path:?}")),
        }
    }

    fn send_prompt_response(
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

    fn is_live(&self) -> bool {
        matches!(
            self,
            Self::Remote(RemoteTransport {
                stream: ProtocolStream::Live(_),
                ..
            })
        )
    }
}

struct RemoteTransport {
    stream: ProtocolStream,
    address: Option<SocketAddr>,
    label: String,
    client_token: Option<String>,
    center: Option<WorldCoordinate>,
    recorder: Option<ReplayRecorder>,
    resource_cache: BTreeMap<String, Vec<u8>>,
}

const REPLAY_MAGIC: &[u8] = b"D64REPLAY\0\x01";

enum ProtocolStream {
    Live(TcpStream),
    Replay(ReplayReader),
}

struct ReplayRecorder {
    file: std::fs::File,
    recorded_resources: BTreeSet<String>,
}

impl ReplayRecorder {
    fn create(path: &Path) -> Result<Self, String> {
        let mut file = std::fs::File::create(path)
            .map_err(|error| format!("create replay {}: {error}", path.display()))?;
        file.write_all(REPLAY_MAGIC)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            file,
            recorded_resources: BTreeSet::new(),
        })
    }

    fn record(&mut self, command: &str, response: &str) -> Result<(), String> {
        if command.starts_with("resource ") && !self.recorded_resources.insert(command.to_owned()) {
            return Ok(());
        }
        write_replay_blob(&mut self.file, command.as_bytes())?;
        write_replay_blob(&mut self.file, response.as_bytes())?;
        self.file.flush().map_err(|error| error.to_string())
    }
}

struct ReplayReader {
    responses: BTreeMap<String, VecDeque<String>>,
    repeatable_resources: BTreeMap<String, String>,
}

impl ReplayReader {
    fn open(path: &Path) -> Result<Self, String> {
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

    fn exchange(&mut self, command: &str) -> Result<String, String> {
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
    fn is_replay(&self) -> bool {
        matches!(self.stream, ProtocolStream::Replay(_))
    }

    fn request_resource(&mut self, path: &str) -> Result<Vec<u8>, String> {
        if let Some(bytes) = self.resource_cache.get(path) {
            return Ok(bytes.clone());
        }
        let session = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?;
        let response = self.exchange(&format!(
            "resource {session} {}",
            encode_hex(path.as_bytes())
        ))?;
        require_ok(&response, "resource")?;
        let bytes = response_fields(&response)
            .get("datahex")
            .and_then(|value| decode_hex(value))
            .ok_or_else(|| format!("resource response omitted data for {path}"))?;
        self.resource_cache.insert(path.to_owned(), bytes.clone());
        Ok(bytes)
    }

    fn send_browser_topic(&mut self, topic: &str) -> Result<(), String> {
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

    fn send_command(&mut self, command: &str) -> Result<(), String> {
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

    fn send_prompt_response(
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

    fn send_screen_pointer(
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

    fn send_map_pointer(
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

    fn connect(address: SocketAddr, record: Option<&Path>) -> Result<Self, String> {
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

    fn replay(path: &Path) -> Result<Self, String> {
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

    fn attach(&mut self) -> Result<(), String> {
        let response = self.exchange("attach")?;
        require_ok(&response, "attach")?;
        let fields = response_fields(&response);
        self.client_token = fields.get("client").cloned();
        self.center = coordinate_fields(&fields);
        Ok(())
    }

    fn request_snapshot(&mut self) -> Result<MapSnapshot, String> {
        self.request_snapshot_command("map_snapshot")
    }

    fn request_screen_snapshot(&mut self) -> Result<MapSnapshot, String> {
        self.request_snapshot_command("screen_snapshot")
    }

    fn request_snapshot_command(&mut self, command: &str) -> Result<MapSnapshot, String> {
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
        for path in paths {
            let path_text = path.to_string_lossy();
            let data = self.request_resource(&path_text)?;
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

    fn send_movement(&mut self, dx: i32, dy: i32) -> Result<Option<MapSnapshot>, String> {
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

    fn poll_ui_events(&mut self) -> Result<Vec<(u64, InboundUiCommand)>, String> {
        let token = self
            .client_token
            .as_deref()
            .ok_or("server attach response did not identify the client")?
            .to_owned();
        let response = self.exchange(&format!("ui_events {token}"))?;
        require_ok(&response, "ui_events")?;
        response.lines().skip(1).map(parse_ui_event).collect()
    }

    fn acknowledge_ui(&mut self, sequence: u64) -> Result<(), String> {
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

    fn send_readiness(&mut self, command: &str) -> Result<(), String> {
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

    fn exchange(&mut self, command: &str) -> Result<String, String> {
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

fn parse_ui_event(line: &str) -> Result<(u64, InboundUiCommand), String> {
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

fn parse_byond_url(url: &str) -> Result<(String, BTreeMap<String, String>), String> {
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

fn browser_output_script(function: &str, value: &str) -> String {
    let arguments = value
        .split('&')
        .map(|part| percent_decode_form(part).unwrap_or_else(|_| part.to_owned()))
        .map(|part| serde_json::to_string(&part).expect("browser output text serializes as JSON"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{function}({arguments});")
}

fn percent_decode_form(value: &str) -> Result<String, String> {
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

fn quote_dmf_assignment(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn valid_byond_callback(callback: &str) -> bool {
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

fn write_frame(stream: &mut impl Write, payload: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(payload.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    stream.write_all(&len.to_be_bytes())?;
    stream.write_all(payload)
}

fn read_frame(stream: &mut impl Read) -> std::io::Result<Vec<u8>> {
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

fn parse_appearance_tree(
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
    if fields.first() != Some(&"A") || !matches!(fields.len(), 16 | 21) {
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

fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

impl LoopbackTransport {
    fn new(scene: WorldScene) -> Self {
        Self {
            scene,
            attached: None,
        }
    }

    fn attach(&mut self, runtime: &mut ExecutionState, client: DatumId) -> Result<(), String> {
        self.scene.attach_local_player(runtime, client)?;
        self.attached = Some(client);
        Ok(())
    }

    fn request_snapshot(&self, radius_x: i32, radius_y: i32) -> MapSnapshot {
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

    fn send_movement(
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

    fn label(&self) -> &str {
        &self.scene.label
    }
}

impl WorldScene {
    fn boot(world: &Path, requested_map: Option<&Path>) -> Result<(ExecutionState, Self), String> {
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

    fn attach_local_player(
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

    fn move_player(&mut self, runtime: &mut ExecutionState, dx: i32, dy: i32) -> bool {
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

    fn sync_player_fields(&mut self, runtime: &mut ExecutionState) -> Result<(), String> {
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

    fn turf_for_player(&self) -> DatumId {
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

struct LocalClient {
    context: Context<OwnedDisplayHandle>,
    surface: Option<ClientSurface>,
    runtime: ExecutionState,
    client: dm_value::DatumId,
    local_input_events: usize,
    inbound_ui_events: usize,
    transport: ClientTransport,
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
    browser_message_sender: std::sync::mpsc::Sender<(String, String)>,
    browser_messages: std::sync::mpsc::Receiver<(String, String)>,
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
        while let Ok((control, message)) = self.browser_messages.try_recv() {
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
                    self.ready_browsers.insert(control.clone());
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
                        #[cfg(windows)]
                        self.apply_browser_update(update);
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
            #[cfg(windows)]
            if let Some((name, data)) = browser_resource {
                self.browser_assets.insert(name, data);
            }
            if let Some(control) = changed_macro
                && let Some(session) = self.runtime.client_session(self.client)
            {
                self.macro_bindings.refresh_control(session.ui(), &control);
            }
        }
        if let Some(sequence) = acknowledged_sequence {
            match self.transport.acknowledge_ui(sequence) {
                Ok(()) => {
                    if let Err(error) = self.transport.mark_resources_ready() {
                        eprintln!("client-ui: resource readiness failed: {error}");
                    } else if self.snapshot.is_some()
                        && let Err(error) = self.transport.mark_input_ready()
                    {
                        eprintln!("client-input-readiness-failed: {error}");
                    }
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
                    let _ = surface.window().request_inner_size(LogicalSize::new(
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
                self.snapshot = Some(snapshot);
                if self.transport.is_live() {
                    // A reconnect to an already-running server may receive no
                    // new UI event batch. The accepted snapshot itself proves
                    // that its resource payload is installed, so advance both
                    // readiness phases here instead of waiting forever for an
                    // acknowledgement that will never be generated.
                    if let Err(error) = self.transport.mark_resources_ready() {
                        eprintln!("client-resource-readiness-failed: {error}");
                    } else if let Err(error) = self.transport.mark_input_ready() {
                        eprintln!("client-input-readiness-failed: {error}");
                    }
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
        resolve_pane_layout_in(session.ui(), Some((size.width, size.height)))
            .controls
            .into_iter()
            .filter(|(address, rect)| {
                rect.width > 0
                    && rect.height > 0
                    && control_has_type(session.ui().tree(), address, ControlType::Browser)
            })
            .collect()
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
                move |url| {
                    if url.to_ascii_lowercase().starts_with("byond://") {
                        let message = serde_json::json!({ "kind": "byond", "url": url });
                        let _ = sender.send((control.clone(), message.to_string()));
                        false
                    } else {
                        true
                    }
                }
            })
            .with_ipc_handler({
                let sender = self.browser_message_sender.clone();
                let control = control.to_owned();
                move |request| {
                    let _ = sender.send((control.clone(), request.body().clone()));
                }
            })
            .build_as_child(&browser_window)
            .expect("WebView2 creates the Dream64 browser surface");
        browser
            .set_bounds(bounds)
            .expect("the browser applies its initial DMF rectangle");
        self.browsers.insert(control.to_owned(), browser);
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
        const AUDIO_CONTROL: &str = "__dream64_audio";
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
        self.ensure_browser(AUDIO_CONTROL);
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
        if let Some(browser) = self.browsers.get(AUDIO_CONTROL)
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
    ) {
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
        }
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
                let layout = self.effective_layout();
                let screenshot = self.pending_screenshot.take();
                if let Some(surface) = &mut self.surface {
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
                    );
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

fn pixel_rect_contains(rect: PixelRect, point: (u32, u32)) -> bool {
    point.0 >= rect.x
        && point.1 >= rect.y
        && point.0 < rect.x.saturating_add(rect.width)
        && point.1 < rect.y.saturating_add(rect.height)
}

fn draw_panel(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    panel_width: usize,
    panel_height: usize,
    color: u32,
) {
    let end_y = y.saturating_add(panel_height).min(height);
    let end_x = x.saturating_add(panel_width).min(width);
    for row in y.min(height)..end_y {
        let start = row * width + x.min(width);
        let end = row * width + end_x;
        buffer[start..end].fill(color);
    }
}

fn draw_output_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    lines: &[String],
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xffffffff,
    );
    const GLYPH_ADVANCE: usize = 8;
    const LINE_ADVANCE: usize = 9;
    let columns = control_width.saturating_sub(4) / GLYPH_ADVANCE;
    let visible_lines = control_height.saturating_sub(4) / LINE_ADVANCE;
    if columns == 0 || visible_lines == 0 {
        return;
    }
    let rendered = lines
        .iter()
        .flat_map(|line| wrap_output_line(&strip_output_markup(line), columns))
        .collect::<Vec<_>>();
    let first = rendered.len().saturating_sub(visible_lines);
    for (row, line) in rendered[first..].iter().enumerate() {
        draw_bitmap_text(
            buffer,
            width,
            height,
            x + 2,
            y + 2 + row * LINE_ADVANCE,
            line,
            0xff000000,
            (x + control_width, y + control_height),
        );
    }
}

fn draw_input_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
    focused: bool,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        if focused { 0xff0078d7 } else { 0xff7a7a7a },
    );
    if control_width > 2 && control_height > 2 {
        draw_panel(
            buffer,
            width,
            height,
            x + 1,
            y + 1,
            control_width - 2,
            control_height - 2,
            0xffffffff,
        );
    }
    let columns = control_width.saturating_sub(6) / 8;
    let visible = text
        .chars()
        .rev()
        .take(columns)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 3,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xff000000,
        (x + control_width.saturating_sub(2), y + control_height),
    );
    if focused {
        let cursor_x = x + 3 + visible.chars().count() * 8;
        draw_panel(
            buffer,
            width,
            height,
            cursor_x.min(x + control_width.saturating_sub(2)),
            y + 3,
            1,
            control_height.saturating_sub(6),
            0xff000000,
        );
    }
}

fn draw_boot_status(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    status: &str,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    if control_width < 32 || control_height < 32 {
        return;
    }
    let columns = control_width.saturating_sub(48) / 8;
    let visible = status.chars().take(columns).collect::<String>();
    let text_x = x + 24;
    let text_y = y + control_height / 2;
    draw_bitmap_text(
        buffer,
        width,
        height,
        text_x,
        text_y.saturating_sub(16),
        "DREAM64 / MONKESTATION",
        0xff4fd8ff,
        (x + control_width, y + control_height),
    );
    draw_bitmap_text(
        buffer,
        width,
        height,
        text_x,
        text_y,
        &visible,
        0xffffffff,
        (x + control_width, y + control_height),
    );
}

fn draw_button_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
    checked: bool,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xff767676,
    );
    if control_width > 2 && control_height > 2 {
        draw_panel(
            buffer,
            width,
            height,
            x + 1,
            y + 1,
            control_width - 2,
            control_height - 2,
            if checked { 0xffc8c8c8 } else { 0xffe1e1e1 },
        );
    }
    let columns = control_width.saturating_sub(4) / 8;
    let visible = text.chars().take(columns).collect::<String>();
    let text_width = visible.chars().count() * 8;
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + control_width.saturating_sub(text_width) / 2,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xff000000,
        (x + control_width, y + control_height),
    );
}

fn draw_label_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xff222222,
    );
    let visible = text
        .chars()
        .take(control_width.saturating_sub(4) / 8)
        .collect::<String>();
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 2,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xffffffff,
        (x + control_width, y + control_height),
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PromptHit {
    Choice(usize),
    Accept,
    Cancel,
}

fn client_prompt_rect(width: usize, height: usize) -> PixelRect {
    let prompt_width = width.saturating_sub(32).min(560);
    let prompt_height = height.saturating_sub(32).min(260);
    PixelRect {
        x: u32::try_from(width.saturating_sub(prompt_width) / 2).unwrap_or(0),
        y: u32::try_from(height.saturating_sub(prompt_height) / 2).unwrap_or(0),
        width: u32::try_from(prompt_width).unwrap_or(0),
        height: u32::try_from(prompt_height).unwrap_or(0),
    }
}

fn client_prompt_hit(
    width: usize,
    height: usize,
    prompt: &ClientPrompt,
    point: (u32, u32),
) -> Option<PromptHit> {
    let rect = client_prompt_rect(width, height);
    let x = point.0;
    let y = point.1;
    if x < rect.x
        || y < rect.y
        || x >= rect.x.saturating_add(rect.width)
        || y >= rect.y.saturating_add(rect.height)
    {
        return None;
    }
    let local_x = x - rect.x;
    let local_y = y - rect.y;
    if !prompt.choices.is_empty() && (78..194).contains(&local_y) {
        let index = usize::try_from((local_y - 78) / 24).ok()?;
        if index < prompt.choices.len().min(5) {
            return Some(PromptHit::Choice(index));
        }
    }
    let footer_y = rect.height.saturating_sub(42);
    if local_y >= footer_y && local_y < footer_y + 28 {
        let width = rect.width;
        if local_x >= width - 116 && local_x < width - 16 {
            return Some(PromptHit::Accept);
        }
        if prompt.can_cancel && local_x >= width - 224 && local_x < width - 124 {
            return Some(PromptHit::Cancel);
        }
    }
    None
}

fn draw_client_prompt(buffer: &mut [u32], width: usize, height: usize, prompt: &ClientPrompt) {
    let rect = client_prompt_rect(width, height);
    let x = usize::try_from(rect.x).unwrap_or(0);
    let y = usize::try_from(rect.y).unwrap_or(0);
    let w = usize::try_from(rect.width).unwrap_or(0);
    let h = usize::try_from(rect.height).unwrap_or(0);
    draw_panel(buffer, width, height, x, y, w, h, 0xffd6dae2);
    if w > 4 && h > 4 {
        draw_panel(
            buffer,
            width,
            height,
            x + 2,
            y + 2,
            w - 4,
            h - 4,
            0xff20252d,
        );
    }
    draw_panel(buffer, width, height, x + 2, y + 2, w - 4, 28, 0xff343b47);
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 10,
        y + 12,
        &prompt
            .title
            .chars()
            .take(w.saturating_sub(20) / 8)
            .collect::<String>(),
        0xffffffff,
        (x + w - 6, y + 28),
    );
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 12,
        y + 45,
        &prompt
            .message
            .chars()
            .take(w.saturating_sub(24) / 8)
            .collect::<String>(),
        0xfff0f2f5,
        (x + w - 8, y + 66),
    );
    if prompt.choices.is_empty()
        && !matches!(
            prompt.kind,
            ClientPromptKind::List | ClientPromptKind::Alert
        )
    {
        draw_input_control(
            buffer,
            width,
            height,
            PixelRect {
                x: rect.x + 12,
                y: rect.y + 78,
                width: rect.width.saturating_sub(24),
                height: 30,
            },
            &prompt.edit,
            true,
        );
    } else if prompt.choices.is_empty() {
        draw_bitmap_text(
            buffer,
            width,
            height,
            x + 18,
            y + 86,
            "No available choices",
            0xffff8888,
            (x + w - 18, y + 108),
        );
    } else {
        for (index, choice) in prompt.choices.iter().take(5).enumerate() {
            let row_y = y + 78 + index * 24;
            draw_panel(
                buffer,
                width,
                height,
                x + 12,
                row_y,
                w.saturating_sub(24),
                22,
                if index == prompt.selected {
                    0xff5b4610
                } else {
                    0xff343b47
                },
            );
            draw_bitmap_text(
                buffer,
                width,
                height,
                x + 18,
                row_y + 7,
                choice,
                0xffffffff,
                (x + w - 18, row_y + 22),
            );
        }
    }
    let button_y = y + h.saturating_sub(42);
    if prompt.can_cancel {
        draw_button_control(
            buffer,
            width,
            height,
            PixelRect {
                x: rect.x + u32::try_from(w.saturating_sub(224)).unwrap_or(0),
                y: u32::try_from(button_y).unwrap_or(0),
                width: 100,
                height: 28,
            },
            "Cancel",
            false,
        );
    }
    draw_button_control(
        buffer,
        width,
        height,
        PixelRect {
            x: rect.x + u32::try_from(w.saturating_sub(116)).unwrap_or(0),
            y: u32::try_from(button_y).unwrap_or(0),
            width: 100,
            height: 28,
        },
        "OK",
        false,
    );
}

fn wrap_output_line(line: &str, columns: usize) -> Vec<String> {
    let characters = line.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return vec![String::new()];
    }
    characters
        .chunks(columns)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

fn strip_output_markup(message: &str) -> String {
    let mut plain = String::with_capacity(message.len());
    let mut in_tag = false;
    for character in message.chars() {
        match character {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            '&' if !in_tag => plain.push('&'),
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[allow(clippy::too_many_arguments)]
fn draw_bitmap_text(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    clip: (usize, usize),
) {
    for (column, character) in text.chars().enumerate() {
        let Some(glyph) = font8x8::BASIC_FONTS.get(character) else {
            continue;
        };
        let glyph_x = x + column * 8;
        for (row, bits) in glyph.iter().copied().enumerate() {
            let pixel_y = y + row;
            if pixel_y >= height || pixel_y >= clip.1 {
                continue;
            }
            for bit in 0..8 {
                let pixel_x = glyph_x + bit;
                if bits & (1 << bit) != 0 && pixel_x < width && pixel_x < clip.0 {
                    buffer[pixel_y * width + pixel_x] = color;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MapTransform {
    clip: PixelRect,
    origin_x: u32,
    origin_y: u32,
    tile: u32,
    columns: u32,
    rows: u32,
}

impl MapTransform {
    fn new(rect: PixelRect, tile_size: u32, zoom: f32, zoom_mode: &str, letterbox: bool) -> Self {
        let scale = if zoom > 0.0 {
            if zoom_mode.eq_ignore_ascii_case("normal") {
                zoom.round().max(1.0)
            } else {
                zoom
            }
        } else {
            1.0
        };
        let tile = ((tile_size.max(1) as f32) * scale).round().max(1.0) as u32;
        let columns = rect.width / tile;
        let rows = rect.height / tile;
        let used_width = columns * tile;
        let used_height = rows * tile;
        Self {
            clip: rect,
            origin_x: rect.x
                + if letterbox {
                    (rect.width - used_width) / 2
                } else {
                    0
                },
            origin_y: rect.y
                + if letterbox {
                    (rect.height - used_height) / 2
                } else {
                    0
                },
            tile,
            columns,
            rows,
        }
    }

    fn world_at(
        &self,
        snapshot: &MapSnapshot,
        screen_x: u32,
        screen_y: u32,
    ) -> Option<WorldCoordinate> {
        if screen_x < self.origin_x
            || screen_y < self.origin_y
            || screen_x >= self.origin_x + self.columns * self.tile
            || screen_y >= self.origin_y + self.rows * self.tile
        {
            return None;
        }
        let column = i32::try_from((screen_x - self.origin_x) / self.tile).ok()?;
        let row = i32::try_from((screen_y - self.origin_y) / self.tile).ok()?;
        let center_column = i32::try_from(self.columns / 2).ok()?;
        let center_row = i32::try_from(self.rows / 2).ok()?;
        Some(WorldCoordinate {
            x: snapshot.center.x + column - center_column,
            y: snapshot.center.y + center_row - row,
            z: snapshot.center.z,
        })
    }
}

struct WorldDisplayItem {
    datum: (u32, u32),
    owner: WorldCoordinate,
    origin_x: i32,
    origin_y: i32,
    width: u32,
    height: u32,
    pixels: Vec<u32>,
    appearance: Appearance,
    maptext_origin_x: i32,
    maptext_origin_y: i32,
}

fn world_display_items(
    snapshot: &MapSnapshot,
    sprites: &mut SpriteCache,
    transform: MapTransform,
) -> Vec<WorldDisplayItem> {
    let center_column = i32::try_from(transform.columns / 2).unwrap_or(0);
    let center_row = i32::try_from(transform.rows / 2).unwrap_or(0);
    let mut appearances = Vec::new();
    for row in 0..transform.rows {
        for column in 0..transform.columns {
            let owner = WorldCoordinate {
                x: snapshot.center.x + i32::try_from(column).unwrap_or(0) - center_column,
                y: snapshot.center.y + center_row - i32::try_from(row).unwrap_or(0),
                z: snapshot.center.z,
            };
            if let Some(items) = snapshot.appearances.get(&(owner.x, owner.y, owner.z)) {
                appearances.extend(
                    items
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(insertion, appearance)| (row, column, insertion, owner, appearance)),
                );
            }
        }
    }
    appearances.sort_by(|left, right| {
        left.4
            .plane
            .total_cmp(&right.4.plane)
            .then_with(|| left.4.layer.total_cmp(&right.4.layer))
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    appearances
        .into_iter()
        .filter_map(|(row, column, _, owner, appearance)| {
            let (native_width, native_height, native_pixels) =
                rasterize_world_appearance(sprites, &appearance).ok()?;
            if native_width == 0 || native_height == 0 {
                return None;
            }
            // DM appearance pixels are expressed in the conventional 32px
            // icon coordinate space. Apply the MAP's effective tile transform
            // to both bounds and offsets.
            let sprite_scale = transform.tile as f32 / 32.0;
            let width = ((native_width as f32) * sprite_scale).round().max(1.0) as u32;
            let height = ((native_height as f32) * sprite_scale).round().max(1.0) as u32;
            let pixels =
                scale_argb_nearest(&native_pixels, native_width, native_height, width, height);
            let cell_left = i32::try_from(transform.origin_x + column * transform.tile).ok()?;
            let cell_bottom =
                i32::try_from(transform.origin_y + (row + 1) * transform.tile).ok()?;
            let pixel_x = ((appearance.pixel_x as f32) * sprite_scale).round() as i32;
            let pixel_y = ((appearance.pixel_y as f32) * sprite_scale).round() as i32;
            let (origin_x, origin_y) =
                world_appearance_origin(cell_left, cell_bottom, height, pixel_x, pixel_y);
            Some(WorldDisplayItem {
                datum: (appearance.datum_index, appearance.datum_generation),
                owner,
                origin_x,
                origin_y,
                width,
                height,
                pixels,
                appearance,
                maptext_origin_x: cell_left,
                maptext_origin_y: i32::try_from(transform.origin_y + row * transform.tile).ok()?,
            })
        })
        .collect()
}

struct GpuWorldDisplayItem {
    draw: gpu::DmiSpriteDraw,
    appearance: Appearance,
    maptext_origin_x: i32,
    maptext_origin_y: i32,
}

fn gpu_world_display_items(
    snapshot: &MapSnapshot,
    sprites: &mut SpriteCache,
    transform: MapTransform,
) -> Vec<GpuWorldDisplayItem> {
    let center_column = i32::try_from(transform.columns / 2).unwrap_or(0);
    let center_row = i32::try_from(transform.rows / 2).unwrap_or(0);
    let mut appearances = Vec::new();
    for row in 0..transform.rows {
        for column in 0..transform.columns {
            let owner = WorldCoordinate {
                x: snapshot.center.x + i32::try_from(column).unwrap_or(0) - center_column,
                y: snapshot.center.y + center_row - i32::try_from(row).unwrap_or(0),
                z: snapshot.center.z,
            };
            if let Some(items) = snapshot.appearances.get(&(owner.x, owner.y, owner.z)) {
                appearances.extend(
                    items
                        .iter()
                        .cloned()
                        .enumerate()
                        .map(|(insertion, appearance)| (row, column, insertion, appearance)),
                );
            }
        }
    }
    appearances.sort_by(|left, right| {
        left.3
            .plane
            .total_cmp(&right.3.plane)
            .then_with(|| left.3.layer.total_cmp(&right.3.layer))
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    appearances
        .into_iter()
        .filter_map(|(row, column, _, appearance)| {
            if appearance.resource.as_os_str().is_empty() {
                return None;
            }
            let frame = sprites.gpu_frame(&appearance).ok()?;
            let sprite_scale = transform.tile as f32 / 32.0;
            let width = (frame.width as f32 * sprite_scale).round().max(1.0);
            let height = (frame.height as f32 * sprite_scale).round().max(1.0);
            let cell_left = i32::try_from(transform.origin_x + column * transform.tile).ok()?;
            let cell_bottom =
                i32::try_from(transform.origin_y + (row + 1) * transform.tile).ok()?;
            let pixel_x = (appearance.pixel_x as f32 * sprite_scale).round() as i32;
            let pixel_y = (appearance.pixel_y as f32 * sprite_scale).round() as i32;
            let (origin_x, origin_y) = world_appearance_origin(
                cell_left,
                cell_bottom,
                height.round() as u32,
                pixel_x,
                pixel_y,
            );
            Some(GpuWorldDisplayItem {
                draw: gpu::DmiSpriteDraw {
                    resource: frame.resource,
                    sheet_width: frame.sheet_width,
                    sheet_height: frame.sheet_height,
                    rgba: frame.rgba,
                    source: [frame.source_x, frame.source_y, frame.width, frame.height],
                    destination: [origin_x as f32, origin_y as f32, width, height],
                    tint: [
                        appearance.color[0],
                        appearance.color[1],
                        appearance.color[2],
                        appearance.alpha,
                    ],
                    clip: [
                        transform.clip.x,
                        transform.clip.y,
                        transform.clip.x.saturating_add(transform.clip.width),
                        transform.clip.y.saturating_add(transform.clip.height),
                    ],
                },
                appearance,
                maptext_origin_x: cell_left,
                maptext_origin_y: i32::try_from(transform.origin_y + row * transform.tile).ok()?,
            })
        })
        .collect()
}

fn scale_argb_nearest(
    source: &[u32],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Vec<u32> {
    if source_width == width && source_height == height {
        return source.to_vec();
    }
    let mut output = vec![0; usize::try_from(width.saturating_mul(height)).unwrap_or(0)];
    for y in 0..height {
        let source_y = y.saturating_mul(source_height) / height;
        for x in 0..width {
            let source_x = x.saturating_mul(source_width) / width;
            output[usize::try_from(y * width + x).unwrap()] =
                source[usize::try_from(source_y * source_width + source_x).unwrap()];
        }
    }
    output
}

fn draw_map(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    transform: MapTransform,
    snapshot: Option<&MapSnapshot>,
    sprites: &mut SpriteCache,
    mut gpu_batches: Option<(&mut Vec<gpu::DmiSpriteDraw>, &mut Vec<gpu::SpriteDraw>)>,
) {
    draw_panel(
        buffer,
        width,
        height,
        usize::try_from(transform.clip.x).unwrap_or(0),
        usize::try_from(transform.clip.y).unwrap_or(0),
        usize::try_from(transform.clip.width).unwrap_or(0),
        usize::try_from(transform.clip.height).unwrap_or(0),
        0xff00_0000,
    );
    let x = usize::try_from(transform.origin_x).unwrap_or(0);
    let y = usize::try_from(transform.origin_y).unwrap_or(0);
    let tile = usize::try_from(transform.tile).unwrap_or(1);
    let columns = usize::try_from(transform.columns).unwrap_or(0);
    let rows = usize::try_from(transform.rows).unwrap_or(0);
    let center_column = i32::try_from(columns / 2).expect("grid columns fit i32");
    let center_row = i32::try_from(rows / 2).expect("grid rows fit i32");
    for row in 0..rows {
        for column in 0..columns {
            let shade = snapshot
                .and_then(|snapshot| {
                    let x = snapshot.center.x
                        + i32::try_from(column).expect("grid column fits i32")
                        - center_column;
                    let y = snapshot.center.y + center_row
                        - i32::try_from(row).expect("grid row fits i32");
                    snapshot.cells.get(&(x, y, snapshot.center.z)).copied()
                })
                .unwrap_or_else(|| {
                    if snapshot.is_some() {
                        // Sparse snapshot coordinates still occupy the full
                        // BYOND viewport. Paint empty test cells instead of
                        // leaving most of the render pane as a black void.
                        if (row + column) % 2 == 0 {
                            0xff3b_3b3b
                        } else {
                            0xff36_3636
                        }
                    } else if (row + column) % 2 == 0 {
                        0xff31566d
                    } else {
                        0xff294a61
                    }
                });
            draw_panel(
                buffer,
                width,
                height,
                x + column * tile + 1,
                y + row * tile + 1,
                tile.saturating_sub(2),
                tile.saturating_sub(2),
                shade,
            );
        }
    }
    // World appearances are a viewport-wide ordered scene. BYOND icons are
    // anchored by their lower-left corner to the owning turf and may be much
    // larger than world.icon_size (title/splash turfs are a common example).
    if let (Some(snapshot), Some((dmi_sprites, _))) = (snapshot, gpu_batches.as_mut()) {
        for item in gpu_world_display_items(snapshot, sprites, transform) {
            draw_appearance_maptext_signed(
                buffer,
                width,
                height,
                item.maptext_origin_x,
                item.maptext_origin_y,
                std::slice::from_ref(&item.appearance),
            );
            dmi_sprites.push(item.draw);
        }
    } else if let Some(snapshot) = snapshot {
        for item in world_display_items(snapshot, sprites, transform) {
            draw_appearance_maptext_signed(
                buffer,
                width,
                height,
                item.maptext_origin_x,
                item.maptext_origin_y,
                std::slice::from_ref(&item.appearance),
            );
            blit_sprite_clipped(
                buffer,
                width,
                height,
                item.origin_x,
                item.origin_y,
                usize::try_from(item.width).unwrap_or(0),
                &item.pixels,
                transform.clip,
            );
        }
    }
    if !snapshot.is_some_and(|snapshot| !snapshot.screen.is_empty()) {
        draw_panel(
            buffer,
            width,
            height,
            x + tile * usize::try_from(center_column).unwrap_or(0) + tile / 6,
            y + tile * usize::try_from(center_row).unwrap_or(0) + tile / 6,
            tile.saturating_mul(2) / 3,
            tile.saturating_mul(2) / 3,
            0xffe6b85c,
        );
    }
    if let Some(snapshot) = snapshot {
        for screen in &snapshot.screen {
            if screen_is_render_pipeline_helper(screen) {
                continue;
            }
            match composite_native(sprites, &screen.appearances) {
                Ok((sprite_width, sprite_height, sprite)) => {
                    let Some((screen_x, screen_y)) = screen_loc_pixels(
                        &screen.screen_loc,
                        transform,
                        sprite_width,
                        sprite_height,
                    ) else {
                        continue;
                    };
                    if screen.appearances.iter().any(|appearance| {
                        appearance
                            .resource
                            .to_string_lossy()
                            .contains("background_monke.dmi")
                    }) {
                        static BACKGROUND_DIAGNOSTIC: std::sync::OnceLock<()> =
                            std::sync::OnceLock::new();
                        BACKGROUND_DIAGNOSTIC.get_or_init(|| {
                            let nonzero = sprite.iter().filter(|pixel| **pixel >> 24 != 0).count();
                            let visible = sprite
                                .iter()
                                .enumerate()
                                .filter(|(index, pixel)| {
                                    if **pixel >> 24 == 0 {
                                        return false;
                                    }
                                    let px = screen_x
                                        + i32::try_from(index % usize::try_from(sprite_width).unwrap_or(1))
                                            .unwrap_or(i32::MAX);
                                    let py = screen_y
                                        + i32::try_from(index / usize::try_from(sprite_width).unwrap_or(1))
                                            .unwrap_or(i32::MAX);
                                    px >= 0
                                        && py >= 0
                                        && usize::try_from(px).is_ok_and(|px| px < width)
                                        && usize::try_from(py).is_ok_and(|py| py < height)
                                })
                                .count();
                            eprintln!(
                                "client-screen-background: screen_loc={:?} native={}x{} origin={},{} nonzero={} visible={}",
                                screen.screen_loc,
                                sprite_width,
                                sprite_height,
                                screen_x,
                                screen_y,
                                nonzero,
                                visible
                            );
                        });
                    }
                    draw_appearance_maptext_signed(
                        buffer,
                        width,
                        height,
                        screen_x,
                        screen_y,
                        &screen.appearances,
                    );
                    if let Some((_, batch)) = gpu_batches.as_mut() {
                        batch.push(gpu::SpriteDraw {
                            x: screen_x,
                            y: screen_y,
                            width: sprite_width,
                            height: sprite_height,
                            pixels: sprite,
                            clip: [
                                0,
                                0,
                                u32::try_from(width).unwrap_or(u32::MAX),
                                u32::try_from(height).unwrap_or(u32::MAX),
                            ],
                        });
                    } else {
                        blit_sprite_signed(
                            buffer,
                            width,
                            height,
                            screen_x,
                            screen_y,
                            usize::try_from(sprite_width).unwrap_or(1),
                            &sprite,
                        );
                    }
                }
                Err(error) => eprintln!(
                    "client-sprite-error: screen_loc={:?} {error}",
                    screen.screen_loc
                ),
            }
        }
    }
}

fn draw_appearance_maptext_signed(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    origin_x: i32,
    origin_y: i32,
    appearances: &[Appearance],
) {
    for appearance in appearances {
        let Some(markup) = appearance.maptext.as_deref() else {
            continue;
        };
        let plain = strip_output_markup(
            &markup
                .replace("<br>", "\n")
                .replace("<br/>", "\n")
                .replace("<br />", "\n"),
        );
        let box_width = appearance.maptext_width.max(8);
        let box_height = appearance.maptext_height.max(8);
        let x = origin_x + appearance.pixel_x + appearance.maptext_x;
        let y = origin_y - appearance.pixel_y - appearance.maptext_y - box_height;
        let Ok(x) = usize::try_from(x.max(0)) else {
            continue;
        };
        let Ok(y) = usize::try_from(y.max(0)) else {
            continue;
        };
        let clip_x = x.saturating_add(usize::try_from(box_width).unwrap_or(8));
        let clip_y = y.saturating_add(usize::try_from(box_height).unwrap_or(8));
        for (line, text) in plain.lines().enumerate() {
            draw_bitmap_text(
                buffer,
                width,
                height,
                x,
                y.saturating_add(line * 10),
                text,
                0xffee_eeee,
                (clip_x, clip_y),
            );
        }
    }
}

fn screen_loc_pixels(
    screen_loc: &str,
    transform: MapTransform,
    _sprite_width: u32,
    sprite_height: u32,
) -> Option<(i32, i32)> {
    let selector = screen_loc.split(" to ").next()?.trim();
    let selector = selector
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(selector)
        .trim();
    let mut axes = selector.split(',');
    let first = axes.next()?.trim();
    let second = axes.next()?.trim();
    // Monk's large lobby art uses BYOND's row,column named point form.
    if matches!(
        first.to_ascii_uppercase().split(':').next(),
        Some("TOP" | "BOTTOM")
    ) {
        let horizontal = named_screen_axis(second, transform.columns, transform.tile)?;
        // BYOND named screen coordinates anchor the atom's icon origin at the
        // selected screen point. CENTER does not center the complete native
        // sprite bounds around that point. Large lobby art deliberately uses
        // CENTER plus a pixel offset chosen for its 32px icon origin; subtracting
        // half of the 295px composed background pushed the whole HUD offscreen.
        let x = horizontal;
        let y = if first.to_ascii_uppercase().starts_with("TOP") {
            // TOP offsets use BYOND's upward-positive screen axis. The client
            // framebuffer is downward-positive, so TOP:-87 is 87px below top.
            -screen_pixel_offset(first)
        } else {
            i32::try_from(transform.rows.checked_mul(transform.tile)?).ok()?
                - i32::try_from(sprite_height).ok()?
                - screen_pixel_offset(first)
        };
        return Some((
            i32::try_from(transform.origin_x).ok()?.checked_add(x)?,
            i32::try_from(transform.origin_y).ok()?.checked_add(y)?,
        ));
    }
    let parse_axis = |axis: &str, extent: u32| -> Option<i32> {
        let mut parts = axis.trim().split(':');
        let tile = match parts.next()?.trim().to_ascii_uppercase().as_str() {
            "WEST" | "SOUTH" | "LEFT" | "BOTTOM" => 1,
            "EAST" | "NORTH" | "RIGHT" | "TOP" => i32::try_from(extent).ok()?,
            "CENTER" => i32::try_from((extent + 1) / 2).ok()?,
            value => value.parse().ok()?,
        };
        let offset = parts
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        Some((tile - 1) * i32::try_from(transform.tile).ok()? + offset)
    };
    let x = parse_axis(first, transform.columns)?;
    let y_from_bottom = parse_axis(second, transform.rows)?;
    let tile = i32::try_from(transform.tile).ok()?;
    let viewport_height = i32::try_from(transform.rows).ok()?.checked_mul(tile)?;
    let y = viewport_height
        .checked_sub(y_from_bottom)?
        .checked_sub(tile)?;
    Some((
        i32::try_from(transform.origin_x).ok()?.checked_add(x)?,
        i32::try_from(transform.origin_y).ok()?.checked_add(y)?,
    ))
}

fn named_screen_prefix(prefix: &str) -> bool {
    matches!(
        prefix.trim().to_ascii_uppercase().as_str(),
        "TOP" | "BOTTOM" | "NORTH" | "SOUTH" | "LEFT" | "RIGHT" | "EAST" | "WEST" | "CENTER"
    )
}

fn normalize_screen_selector(
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

fn screen_is_render_pipeline_helper(screen: &ScreenAppearance) -> bool {
    screen
        .type_path
        .starts_with("/atom/movable/screen/plane_master")
        || screen
            .type_path
            .starts_with("/atom/movable/screen/click_catcher")
        || screen
            .type_path
            .starts_with("/atom/movable/render_plane_relay")
        // These full-screen white/black quads are inputs to TG's lighting
        // plane-master blend chain. Drawing them as ordinary HUD sprites
        // produces the stray 32px white square at screen_loc 1,1.
        || screen
            .type_path
            .starts_with("/atom/movable/screen/fullscreen/lighting_backdrop")
        || screen.appearances.iter().any(|appearance| {
            appearance.state.eq_ignore_ascii_case("not_ready")
                && appearance
                    .resource
                    .to_string_lossy()
                    .replace('\\', "/")
                    .ends_with("icons/hud/lobby/ready.dmi")
        })
}

fn screen_hit_at(
    snapshot: &MapSnapshot,
    sprites: &mut SpriteCache,
    transform: MapTransform,
    x: u32,
    y: u32,
) -> Option<(u32, u32, String, Option<String>)> {
    for screen in snapshot.screen.iter().rev() {
        if screen_is_render_pipeline_helper(screen) {
            continue;
        }
        let Ok((width, height, pixels)) = composite_native(sprites, &screen.appearances) else {
            continue;
        };
        let Some((origin_x, origin_y)) =
            screen_loc_pixels(&screen.screen_loc, transform, width, height)
        else {
            continue;
        };
        let local_x = i32::try_from(x).ok()?.checked_sub(origin_x)?;
        let local_y = i32::try_from(y).ok()?.checked_sub(origin_y)?;
        if local_x < 0
            || local_y < 0
            || local_x >= i32::try_from(width).ok()?
            || local_y >= i32::try_from(height).ok()?
        {
            continue;
        }
        let index = usize::try_from(local_y).ok()? * usize::try_from(width).ok()?
            + usize::try_from(local_x).ok()?;
        if pixels.get(index).is_some_and(|pixel| *pixel >> 24 != 0) {
            return Some((
                screen.datum_index,
                screen.datum_generation,
                screen.screen_loc.clone(),
                screen.map_control.clone(),
            ));
        }
    }
    None
}

fn map_hit_at(
    snapshot: &MapSnapshot,
    sprites: &mut SpriteCache,
    transform: MapTransform,
    x: u32,
    y: u32,
) -> Option<((u32, u32), WorldCoordinate, String)> {
    let coordinate = transform.world_at(snapshot, x, y)?;
    let screen_x = i32::try_from(x).ok()?;
    let screen_y = i32::try_from(y).ok()?;
    let display_hit = world_display_items(snapshot, sprites, transform)
        .into_iter()
        .rev()
        .find(|item| {
            let local_x = screen_x - item.origin_x;
            let local_y = screen_y - item.origin_y;
            if local_x < 0
                || local_y < 0
                || local_x >= i32::try_from(item.width).unwrap_or(i32::MAX)
                || local_y >= i32::try_from(item.height).unwrap_or(i32::MAX)
            {
                return false;
            }
            let index = usize::try_from(local_y).unwrap() * usize::try_from(item.width).unwrap()
                + usize::try_from(local_x).unwrap();
            item.pixels.get(index).is_some_and(|pixel| pixel >> 24 != 0)
        });
    let (target, target_coordinate) =
        display_hit
            .map(|item| (item.datum, item.owner))
            .or_else(|| {
                snapshot
                    .turf_targets
                    .get(&(coordinate.x, coordinate.y, coordinate.z))
                    .copied()
                    .map(|target| (target, coordinate))
            })?;
    let column = (x - transform.origin_x) / transform.tile + 1;
    let row = transform.rows - (y - transform.origin_y) / transform.tile;
    let local_x = (x - transform.origin_x) % transform.tile;
    let local_y = (y - transform.origin_y) % transform.tile;
    let icon_x = local_x + 1;
    let icon_y = transform.tile - local_y;
    let params = format!(
        "icon-x={icon_x};icon-y={icon_y};screen-loc={column}:{local_x},{row}:{}",
        transform.tile - local_y - 1
    );
    Some((target, target_coordinate, params))
}

fn click_pointer_params(base: &str, button: MouseButton, modifiers: ModifiersState) -> String {
    let mut fields = base
        .split(';')
        .filter(|field| !field.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    match button {
        MouseButton::Left => fields.extend(["left=1".to_owned(), "button=left".to_owned()]),
        MouseButton::Right => fields.extend(["right=1".to_owned(), "button=right".to_owned()]),
        MouseButton::Middle => fields.extend(["middle=1".to_owned(), "button=middle".to_owned()]),
        MouseButton::Back => fields.push("button=back".to_owned()),
        MouseButton::Forward => fields.push("button=forward".to_owned()),
        MouseButton::Other(number) => fields.push(format!("button={number}")),
    }
    if modifiers.shift_key() {
        fields.push("shift=1".to_owned());
    }
    if modifiers.control_key() {
        fields.push("ctrl=1".to_owned());
    }
    if modifiers.alt_key() {
        fields.push("alt=1".to_owned());
    }
    fields.join(";")
}

fn screen_pixel_offset(axis: &str) -> i32 {
    axis.split(':')
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

fn named_screen_axis(axis: &str, extent: u32, tile: u32) -> Option<i32> {
    let base = match axis.split(':').next()?.trim().to_ascii_uppercase().as_str() {
        "LEFT" | "WEST" | "BOTTOM" | "SOUTH" => 0,
        "CENTER" => i32::try_from(extent.checked_mul(tile)? / 2).ok()?,
        "RIGHT" | "EAST" | "TOP" | "NORTH" => i32::try_from(extent.checked_mul(tile)?).ok()?,
        value => (value.parse::<i32>().ok()? - 1) * i32::try_from(tile).ok()?,
    };
    Some(base + screen_pixel_offset(axis))
}

fn blit_sprite_signed(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    sprite_width: usize,
    sprite: &[u32],
) {
    for (index, pixel) in sprite.iter().copied().enumerate() {
        if pixel >> 24 == 0 {
            continue;
        }
        let destination_x = x + i32::try_from(index % sprite_width).unwrap_or(i32::MAX);
        let destination_y = y + i32::try_from(index / sprite_width).unwrap_or(i32::MAX);
        if destination_x >= 0
            && destination_y >= 0
            && usize::try_from(destination_x).is_ok_and(|x| x < width)
            && usize::try_from(destination_y).is_ok_and(|y| y < height)
        {
            buffer[usize::try_from(destination_y).unwrap() * width
                + usize::try_from(destination_x).unwrap()] = pixel;
        }
    }
}

fn blit_sprite_clipped(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: i32,
    y: i32,
    sprite_width: usize,
    sprite: &[u32],
    clip: PixelRect,
) {
    let clip_x = i32::try_from(clip.x).unwrap_or(i32::MAX);
    let clip_y = i32::try_from(clip.y).unwrap_or(i32::MAX);
    let clip_right = clip_x.saturating_add(i32::try_from(clip.width).unwrap_or(i32::MAX));
    let clip_bottom = clip_y.saturating_add(i32::try_from(clip.height).unwrap_or(i32::MAX));
    for (index, pixel) in sprite.iter().copied().enumerate() {
        if pixel >> 24 == 0 || sprite_width == 0 {
            continue;
        }
        let dx = x.saturating_add(i32::try_from(index % sprite_width).unwrap_or(i32::MAX));
        let dy = y.saturating_add(i32::try_from(index / sprite_width).unwrap_or(i32::MAX));
        if dx >= clip_x
            && dx < clip_right
            && dy >= clip_y
            && dy < clip_bottom
            && dx >= 0
            && dy >= 0
            && usize::try_from(dx).is_ok_and(|dx| dx < width)
            && usize::try_from(dy).is_ok_and(|dy| dy < height)
        {
            buffer[usize::try_from(dy).unwrap() * width + usize::try_from(dx).unwrap()] = pixel;
        }
    }
}

fn world_appearance_origin(
    cell_left: i32,
    cell_bottom: i32,
    sprite_height: u32,
    pixel_x: i32,
    pixel_y: i32,
) -> (i32, i32) {
    (
        cell_left.saturating_add(pixel_x),
        cell_bottom
            .saturating_sub(i32::try_from(sprite_height).unwrap_or(i32::MAX))
            .saturating_sub(pixel_y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn launch_options(arguments: &[&str]) -> Result<LaunchOptions, String> {
        LaunchOptions::parse_from(arguments.iter().map(std::ffi::OsString::from))
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
            "ok resource protocol=2 pathhex=69636f6e2e646d69 datahex={}",
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
                ("resource c1 69636f6e2e646d69", resource_response.as_str()),
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
