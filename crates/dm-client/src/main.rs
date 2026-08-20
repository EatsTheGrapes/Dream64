//! The first visible Dream64 local-client shell.

#![deny(missing_docs)]

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use std::{collections::BTreeMap, path::Path};
use std::{io::Read, io::Write, net::SocketAddr, net::TcpStream};

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
use softbuffer::{Context, Surface};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowAttributes, WindowId};

mod sprite;
use sprite::{Appearance, SpriteCache, composite_tile};

#[cfg(windows)]
use wry::{
    Rect, WebView, WebViewBuilder,
    dpi::{LogicalPosition, LogicalSize},
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

#[cfg(windows)]
const EMPTY_BROWSER_DOCUMENT: &str = "<!doctype html><html><head></head><body></body></html>";

/// Runs a visible local client using a supplied DMF skin or the development skin.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let launch = LaunchOptions::parse()?;
    let (skin_source, skin_label) = load_skin(launch.skin.as_deref())?;
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
    let transport = if let Some(scene) = scene {
        let mut transport = LoopbackTransport::new(scene);
        transport.attach(&mut runtime, client)?;
        ClientTransport::Offline(transport)
    } else {
        let mut transport = RemoteTransport::connect(launch.connect)?;
        transport.attach()?;
        ClientTransport::Remote(transport)
    };
    let layout = ClientLayout::from_tree(
        runtime
            .client_session(client)
            .expect("new client session exists")
            .ui()
            .tree(),
    );
    let mut application = LocalClient {
        context,
        surface: None,
        runtime,
        client,
        skin_label,
        queued_events: 0,
        transport,
        snapshot: None,
        sprites: SpriteCache::default(),
        layout,
        ui_presentation: UiPresentation::default(),
        snapshot_stage: 0,
        #[cfg(windows)]
        browser: None,
    };
    event_loop.run_app(&mut application)?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ClientLayout {
    window_width: u32,
    window_height: u32,
    map: PixelRect,
    browser: Option<PixelRect>,
    browser_control: Option<String>,
    output_controls: Vec<String>,
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
        let mut map = controls
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
            .find(|control| {
                control.id.as_deref() == Some("browseroutput")
                    || control.property("is-default").is_some_and(dmf_truthy)
            })
            .or_else(|| {
                controls
                    .clone()
                    .find(|control| control.control_type == ControlType::Browser)
            });
        let mut browser = browser_node.and_then(dm_dmf::ControlNode::pixel_rect);
        let browser_control = browser_node.and_then(|node| qualified_control(tree, node));
        let output_controls = controls
            .filter(|control| control.control_type == ControlType::Output)
            .filter_map(|node| qualified_control(tree, node))
            .collect();
        // Pane windows describe their children in pane-local coordinates. The
        // real mainwindow's first CHILD divides mapwindow and
        // info_and_buttons left/right; project full-width pane rectangles into
        // that native shell until general CHILD layout resolution lands.
        if tree.windows.len() > 1 && map.width >= main_rect.width {
            map.width = main_rect.width / 2;
            map.height = map.height.min(main_rect.height);
            if let Some(rect) = &mut browser {
                if rect.width >= main_rect.width {
                    rect.x = map.width;
                    rect.y = 0;
                    rect.width = main_rect.width.saturating_sub(map.width);
                    rect.height = rect.height.min(main_rect.height);
                }
            }
        }
        Self {
            window_width: main_rect.width,
            window_height: main_rect.height,
            map,
            browser,
            browser_control,
            output_controls,
        }
    }
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
}

#[derive(Default)]
struct UiPresentation {
    last_sequence: u64,
    browser_html: BTreeMap<String, String>,
    output_text: BTreeMap<String, Vec<String>>,
    browser_resources: BTreeMap<String, Vec<u8>>,
}

impl UiPresentation {
    fn apply(
        &mut self,
        sequence: u64,
        command: InboundUiCommand,
        session: &mut dm_dmf::ClientSession,
        layout: &ClientLayout,
    ) -> Result<Option<String>, String> {
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
                None
            }
            InboundUiCommand::Output { control, message } => {
                let control = control
                    .or_else(|| layout.output_controls.first().cloned())
                    .ok_or("skin has no OUTPUT control")?;
                let control =
                    resolve_control_type(session.ui().tree(), &control, ControlType::Output)?;
                self.output_text.entry(control).or_default().push(message);
                None
            }
            InboundUiCommand::BrowseResource { name, data } => {
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
                let html = materialize_browser_resources(&html, &self.browser_resources);
                let key = resolved.clone().unwrap_or(control);
                self.browser_html.insert(key, html.clone());
                layout
                    .browser_control
                    .as_ref()
                    .is_some_and(|shown| resolved.as_ref() == Some(shown))
                    .then_some(html)
            }
        };
        self.last_sequence = sequence;
        Ok(browser_update)
    }
}

fn materialize_browser_resources(html: &str, resources: &BTreeMap<String, Vec<u8>>) -> String {
    resources
        .iter()
        .fold(html.to_owned(), |document, (name, bytes)| {
            let mime = match Path::new(name)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "png" => "image/png",
                "gif" => "image/gif",
                "jpg" | "jpeg" => "image/jpeg",
                "css" => "text/css",
                "js" => "text/javascript",
                _ => "application/octet-stream",
            };
            let uri = format!("data:{mime};base64,{}", encode_base64(bytes));
            document
                .replace(&format!("\"{name}\""), &format!("\"{uri}\""))
                .replace(&format!("'{name}'"), &format!("'{uri}'"))
        })
}

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

struct LaunchOptions {
    skin: Option<PathBuf>,
    world: Option<PathBuf>,
    map: Option<PathBuf>,
    connect: SocketAddr,
}

impl LaunchOptions {
    fn parse() -> Result<Self, String> {
        let mut options = Self {
            skin: None,
            world: None,
            map: None,
            connect: "127.0.0.1:51664"
                .parse()
                .expect("default loopback address is valid"),
        };
        let mut arguments = std::env::args_os().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.to_string_lossy().as_ref() {
                "--skin" => options.skin = Some(next_path(&mut arguments, "--skin")?),
                "--world" => options.world = Some(next_path(&mut arguments, "--world")?),
                "--map" => options.map = Some(next_path(&mut arguments, "--map")?),
                "--connect" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| "--connect requires a loopback address".to_owned())?;
                    options.connect = value
                        .to_string_lossy()
                        .parse::<SocketAddr>()
                        .map_err(|error| format!("invalid --connect address: {error}"))?;
                    if !options.connect.ip().is_loopback() {
                        return Err("--connect must use a loopback address".to_owned());
                    }
                }
                other if other.ends_with(".dmf") && options.skin.is_none() => {
                    options.skin = Some(PathBuf::from(argument));
                }
                other => {
                    return Err(format!(
                        "unknown client argument {other:?}; use [--connect 127.0.0.1:51664] or --world <.dme> --map <.dmm> [--skin <.dmf>]"
                    ));
                }
            }
        }
        if options.map.is_some() && options.world.is_none() {
            return Err("--map requires --world".to_owned());
        }
        Ok(options)
    }
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
    appearances: BTreeMap<(i32, i32, i32), Vec<Appearance>>,
    resources: BTreeMap<PathBuf, Vec<u8>>,
}

/// In-process implementation of the client/server exchange. Keeping the
/// window behind this boundary makes a future TCP transport a codec swap
/// rather than another renderer implementation.
struct LoopbackTransport {
    scene: WorldScene,
    attached: Option<DatumId>,
}

enum ClientTransport {
    Remote(RemoteTransport),
    Offline(LoopbackTransport),
}

impl ClientTransport {
    fn request_snapshot(&mut self) -> Result<MapSnapshot, String> {
        match self {
            Self::Remote(transport) => transport.request_snapshot(),
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
            Self::Remote(transport) => transport.send_movement(dx, dy),
            Self::Offline(transport) => Ok(transport.send_movement(runtime, dx, dy)),
        }
    }

    fn label(&self) -> &str {
        match self {
            Self::Remote(transport) => &transport.label,
            Self::Offline(transport) => transport.label(),
        }
    }

    fn poll_ui_events(&mut self) -> Result<Vec<(u64, InboundUiCommand)>, String> {
        match self {
            Self::Remote(transport) => transport.poll_ui_events(),
            Self::Offline(_) => Ok(Vec::new()),
        }
    }
}

struct RemoteTransport {
    stream: TcpStream,
    label: String,
    client_token: Option<String>,
    center: Option<WorldCoordinate>,
}

impl RemoteTransport {
    fn connect(address: SocketAddr) -> Result<Self, String> {
        if !address.ip().is_loopback() {
            return Err("remote client transport is restricted to loopback".to_owned());
        }
        let stream =
            TcpStream::connect(address).map_err(|error| format!("connect {address}: {error}"))?;
        stream
            .set_nodelay(true)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            stream,
            label: format!("server {address}"),
            client_token: None,
            center: None,
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
        let token = self
            .client_token
            .as_deref()
            .ok_or_else(|| "server attach response did not identify the client".to_owned())?
            .to_owned();
        let response = self.exchange(&format!("map_snapshot {token}"))?;
        require_ok(&response, "map_snapshot")?;
        let header = response_fields(&response);
        let mut cells = BTreeMap::new();
        let mut appearances = BTreeMap::<(i32, i32, i32), Vec<Appearance>>::new();
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
            for _ in 0..appearance_count {
                parse_appearance_tree(&lines, &mut cursor, &mut flattened)?;
            }
            appearances.insert((x, y, inferred_z), flattened);
        }
        let center = self.center.unwrap_or_else(|| {
            let max_x = cells.keys().map(|(x, _, _)| *x).max().unwrap_or(1);
            let max_y = cells.keys().map(|(_, y, _)| *y).max().unwrap_or(1);
            WorldCoordinate {
                x: (max_x + 1) / 2,
                y: (max_y + 1) / 2,
                z: inferred_z,
            }
        });
        let mut resources = BTreeMap::new();
        let paths = appearances
            .iter()
            .filter(|((x, y, z), _)| {
                *z == center.z && (*x - center.x).abs() <= 10 && (*y - center.y).abs() <= 7
            })
            .flat_map(|(_, appearances)| appearances)
            .map(|appearance| appearance.resource.clone())
            .collect::<std::collections::BTreeSet<_>>();
        for path in paths {
            let path_text = path.to_string_lossy();
            let response = self.exchange(&format!(
                "resource {token} {}",
                encode_hex(path_text.as_bytes())
            ))?;
            require_ok(&response, "resource")?;
            let fields = response_fields(&response);
            let data = fields
                .get("datahex")
                .and_then(|value| decode_hex(value))
                .ok_or_else(|| format!("resource response omitted data for {path_text}"))?;
            resources.insert(path, data);
        }
        Ok(MapSnapshot {
            center,
            cells,
            appearances,
            resources,
        })
    }

    fn send_movement(&mut self, dx: i32, dy: i32) -> Result<Option<MapSnapshot>, String> {
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

    fn exchange(&mut self, command: &str) -> Result<String, String> {
        write_frame(&mut self.stream, command.as_bytes()).map_err(|error| error.to_string())?;
        let payload = read_frame(&mut self.stream).map_err(|error| error.to_string())?;
        String::from_utf8(payload).map_err(|_| "server response is not UTF-8".to_owned())
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
            data: decode_hex(fields[4]).ok_or("invalid browse resource bytes")?,
        },
        ("browse", 5) => InboundUiCommand::Browse {
            control: text(3)?,
            html: text(4)?,
        },
        _ => return Err(format!("unsupported UI event row: {line}")),
    };
    Ok((sequence, command))
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
    let line = lines.get(*cursor).ok_or("truncated appearance tree")?;
    *cursor += 1;
    let mut fields = line.split_ascii_whitespace().collect::<Vec<_>>();
    // Protocol 2 originally rendered Some("") as an empty whitespace token.
    // Live servers using that first codec therefore omit the icon-state column.
    // Normalize those rows while the explicit `~` representation rolls out.
    if fields.first() == Some(&"A") && fields.len() == 15 {
        fields.insert(4, "~");
    }
    if fields.first() != Some(&"A") || fields.len() != 16 {
        return Err("invalid appearance row".to_owned());
    }
    let underlays = fields[14]
        .parse::<usize>()
        .map_err(|_| "invalid underlay count")?;
    let overlays = fields[15]
        .parse::<usize>()
        .map_err(|_| "invalid overlay count")?;
    for _ in 0..underlays {
        parse_appearance_tree(lines, cursor, output)?;
    }
    if let Some(icon) = decode_optional_hex_text(fields[3]) {
        let numeric = |index| u32::from_str_radix(fields[index], 16).map(f32::from_bits);
        let color = decode_optional_hex_text(fields[12])
            .as_deref()
            .and_then(parse_snapshot_color)
            .map(|argb| [(argb >> 16) as u8, (argb >> 8) as u8, argb as u8])
            .unwrap_or([255; 3]);
        output.push(Appearance {
            resource: PathBuf::from(icon),
            state: decode_optional_hex_text(fields[4]).unwrap_or_default(),
            direction: fields[5].parse().unwrap_or(2),
            frame: 1,
            layer: numeric(6).unwrap_or(0.0),
            plane: numeric(7).unwrap_or(0.0),
            pixel_x: numeric(8).unwrap_or(0.0).round() as i32,
            pixel_y: numeric(9).unwrap_or(0.0).round() as i32,
            color,
            alpha: numeric(13).unwrap_or(255.0).clamp(0.0, 255.0).round() as u8,
        });
    }
    for _ in 0..overlays {
        parse_appearance_tree(lines, cursor, output)?;
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
            appearances: BTreeMap::new(),
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
    surface: Option<Surface<OwnedDisplayHandle, Rc<Window>>>,
    runtime: ExecutionState,
    client: dm_value::DatumId,
    skin_label: String,
    queued_events: usize,
    transport: ClientTransport,
    snapshot: Option<MapSnapshot>,
    sprites: SpriteCache,
    layout: ClientLayout,
    ui_presentation: UiPresentation,
    snapshot_stage: u8,
    #[cfg(windows)]
    browser: Option<WebView>,
}

impl LocalClient {
    fn apply_inbound_ui(&mut self) {
        let Ok(events) = self.transport.poll_ui_events() else {
            return;
        };
        for (sequence, command) in events {
            let Some(session) = self.runtime.client_session_mut(self.client) else {
                break;
            };
            match self
                .ui_presentation
                .apply(sequence, command, session, &self.layout)
            {
                Ok(Some(html)) =>
                {
                    #[cfg(windows)]
                    if let Some(browser) = &self.browser {
                        let _ = browser.load_html(&html);
                    }
                }
                Ok(None) => {}
                Err(error) => eprintln!("client-ui: rejected event {sequence}: {error}"),
            }
        }
    }

    #[cfg(windows)]
    fn browser_bounds(&self) -> Rect {
        let rect = self.layout.browser.unwrap_or(dm_dmf::PixelRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        });
        Rect {
            position: LogicalPosition::new(f64::from(rect.x), f64::from(rect.y)).into(),
            size: LogicalSize::new(f64::from(rect.width), f64::from(rect.height)).into(),
        }
    }

    fn title(&self) -> String {
        let world = self.transport.label();
        format!(
            "Dream64 local client — {} — {} — {} UI event(s)",
            self.skin_label, world, self.queued_events
        )
    }

    fn resize(surface: &mut Surface<OwnedDisplayHandle, Rc<Window>>, width: u32, height: u32) {
        let Some(width) = NonZeroU32::new(width) else {
            return;
        };
        let Some(height) = NonZeroU32::new(height) else {
            return;
        };
        surface
            .resize(width, height)
            .expect("the client surface resizes");
    }

    fn redraw(
        surface: &mut Surface<OwnedDisplayHandle, Rc<Window>>,
        snapshot: Option<&MapSnapshot>,
        sprites: &mut SpriteCache,
        layout: ClientLayout,
    ) {
        if let Some(snapshot) = snapshot {
            for (path, bytes) in &snapshot.resources {
                sprites.insert(path.clone(), bytes);
            }
        }
        let size = surface.window().inner_size();
        let width = usize::try_from(size.width).expect("window width fits usize");
        let height = usize::try_from(size.height).expect("window height fits usize");
        let mut buffer = surface
            .buffer_mut()
            .expect("the client surface is drawable");
        buffer.fill(0xff171a21);
        draw_panel(
            &mut buffer,
            width,
            height,
            16,
            16,
            width.saturating_sub(32),
            42,
            0xff252b36,
        );
        draw_panel(
            &mut buffer,
            width,
            height,
            16,
            74,
            width * 2 / 3 - 24,
            height.saturating_sub(166),
            0xff1d3447,
        );
        draw_map(
            &mut buffer,
            width,
            height,
            usize::try_from(layout.map.x).unwrap_or(0),
            usize::try_from(layout.map.y).unwrap_or(0),
            usize::try_from(layout.map.width)
                .unwrap_or(width)
                .min(width),
            usize::try_from(layout.map.height)
                .unwrap_or(height)
                .min(height),
            snapshot,
            sprites,
        );
        draw_panel(
            &mut buffer,
            width,
            height,
            width * 2 / 3 + 8,
            74,
            width / 3 - 24,
            height.saturating_sub(166),
            0xff272238,
        );
        draw_panel(
            &mut buffer,
            width,
            height,
            16,
            height.saturating_sub(76),
            width.saturating_sub(32),
            60,
            0xff20252d,
        );
        buffer.present().expect("the client surface presents");
    }
}

impl ApplicationHandler for LocalClient {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default()
            .with_title(self.title())
            .with_inner_size(winit::dpi::LogicalSize::new(
                f64::from(self.layout.window_width),
                f64::from(self.layout.window_height),
            ));
        let window = Rc::new(
            event_loop
                .create_window(attributes)
                .expect("the native Dream64 client window is created"),
        );
        let size = window.inner_size();
        let mut surface =
            Surface::new(&self.context, window).expect("the client surface is created");
        Self::resize(&mut surface, size.width, size.height);
        surface.window().request_redraw();
        self.surface = Some(surface);
        #[cfg(windows)]
        {
            let browser_window = self
                .surface
                .as_ref()
                .expect("the browser has a parent window")
                .window()
                .clone();
            let bounds = self.browser_bounds();
            let browser = WebViewBuilder::new()
                .with_html(EMPTY_BROWSER_DOCUMENT)
                .with_bounds(bounds)
                .with_initialization_script(WEBVIEW2_BYOND_BRIDGE_BOOTSTRAP)
                .build_as_child(&browser_window)
                .expect("WebView2 creates the Dream64 browser surface");
            browser
                .set_bounds(bounds)
                .expect("the browser applies its initial DMF rectangle");
            self.browser = Some(browser);
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
                if let Some(browser) = &self.browser {
                    browser
                        .set_bounds(self.browser_bounds())
                        .expect("the browser follows its DMF rectangle");
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
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
                    if let Some(session) = self.runtime.client_session_mut(self.client) {
                        session.push_event(UiEvent::Key {
                            key: format!("{code:?}"),
                            pressed: event.state == ElementState::Pressed,
                        });
                        self.queued_events = self.queued_events.saturating_add(1);
                        let title = self.title();
                        if let Some(surface) = &self.surface {
                            surface.window().set_title(&title);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.apply_inbound_ui();
                let snapshot = self.snapshot.as_ref();
                if let Some(surface) = &mut self.surface {
                    Self::redraw(surface, snapshot, &mut self.sprites, self.layout.clone());
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
                    match self.transport.request_snapshot() {
                        Ok(snapshot) => {
                            let appearance_count =
                                snapshot.appearances.values().map(Vec::len).sum::<usize>();
                            eprintln!(
                                "client-snapshot-ready: cells={} appearances={} resources={}",
                                snapshot.cells.len(),
                                appearance_count,
                                snapshot.resources.len()
                            );
                            self.snapshot = Some(snapshot);
                            if let Some(surface) = &self.surface {
                                surface.window().set_title(&format!(
                                    "{} — sprites {appearance_count} / resources {}",
                                    self.title(),
                                    self.snapshot
                                        .as_ref()
                                        .map_or(0, |snapshot| snapshot.resources.len())
                                ));
                                surface.window().request_redraw();
                            }
                        }
                        Err(error) => {
                            eprintln!("client-snapshot-error: {error}");
                            if let Some(surface) = &self.surface {
                                surface.window().set_title(&format!(
                                    "{} — snapshot error: {error}",
                                    self.title()
                                ));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
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

fn draw_map(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    grid_width: usize,
    grid_height: usize,
    snapshot: Option<&MapSnapshot>,
    sprites: &mut SpriteCache,
) {
    const TILE: usize = 32;
    let columns = grid_width / TILE;
    let rows = grid_height / TILE;
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
                    if (row + column) % 2 == 0 {
                        0xff31566d
                    } else {
                        0xff294a61
                    }
                });
            draw_panel(
                buffer,
                width,
                height,
                x + column * TILE + 1,
                y + row * TILE + 1,
                TILE.saturating_sub(2),
                TILE.saturating_sub(2),
                shade,
            );
            if let Some(snapshot) = snapshot {
                let cell_x = snapshot.center.x
                    + i32::try_from(column).expect("grid column fits i32")
                    - center_column;
                let cell_y =
                    snapshot.center.y + center_row - i32::try_from(row).expect("grid row fits i32");
                if let Some(appearances) =
                    snapshot
                        .appearances
                        .get(&(cell_x, cell_y, snapshot.center.z))
                {
                    match composite_tile(sprites, appearances, TILE as u32, TILE as u32) {
                        Ok(sprite) => blit_sprite(
                            buffer,
                            width,
                            height,
                            x + column * TILE,
                            y + row * TILE,
                            TILE,
                            &sprite,
                        ),
                        Err(error) => eprintln!(
                            "client-sprite-error: cell={cell_x},{cell_y},{} {error}",
                            snapshot.center.z
                        ),
                    }
                }
            }
        }
    }
    draw_panel(
        buffer,
        width,
        height,
        x + TILE * usize::try_from(center_column).unwrap_or(0) + 5,
        y + TILE * usize::try_from(center_row).unwrap_or(0) + 5,
        22,
        22,
        0xffe6b85c,
    );
}

fn blit_sprite(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    sprite_width: usize,
    sprite: &[u32],
) {
    for (index, pixel) in sprite.iter().copied().enumerate() {
        if pixel >> 24 == 0 {
            continue;
        }
        let destination_x = x + index % sprite_width;
        let destination_y = y + index / sprite_width;
        if destination_x < width && destination_y < height {
            buffer[destination_y * width + destination_x] = pixel;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn real_skin_style_ids_and_panes_resolve_main_lobby_geometry() {
        let document = dm_dmf::parse(concat!(
            "window \"mainwindow\"\n",
            "\telem \"mainwindow\"\n\t\ttype = MAIN\n\t\tpos = 281,0\n\t\tsize = 640x440\n\t\tis-default = true\n",
            "\telem \"split\"\n\t\ttype = CHILD\n\t\tpos = 0,0\n\t\tsize = 640x440\n\t\tleft = \"mapwindow\"\n\t\tright = \"info_and_buttons\"\n",
            "window \"mapwindow\"\n",
            "\telem \"mapwindow\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 640x480\n\t\tis-pane = true\n",
            "\telem \"map\"\n\t\ttype = MAP\n\t\tpos = 0,0\n\t\tsize = 640x480\n",
            "window \"output_browser\"\n",
            "\telem \"output_browser\"\n\t\ttype = MAIN\n\t\tpos = 0,0\n\t\tsize = 640x456\n\t\tis-pane = true\n",
            "\telem \"browseroutput\"\n\t\ttype = BROWSER\n\t\tpos = 0,0\n\t\tsize = 640x456\n",
        ));
        assert!(document.diagnostics.is_empty());
        let layout = ClientLayout::from_tree(&ControlTree::from_document(&document));
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
        let layout = ClientLayout::from_tree(&tree);
        let mut session = dm_dmf::ClientSession::new(tree);
        let mut presentation = UiPresentation::default();

        let (sequence, browse) =
            parse_ui_event("U 7 browse 62726f77736572 3c623e6c6f6262793c2f623e").unwrap();
        assert_eq!(
            presentation
                .apply(sequence, browse, &mut session, &layout)
                .unwrap()
                .as_deref(),
            Some("<b>lobby</b>")
        );
        let (sequence, output) = parse_ui_event("U 8 output 6c6f67 7265616479").unwrap();
        assert_eq!(
            presentation
                .apply(sequence, output, &mut session, &layout)
                .unwrap(),
            None
        );
        assert_eq!(presentation.output_text["main.log"], ["ready"]);
        assert_eq!(presentation.last_sequence, 8);

        // A repeated drained record cannot be applied twice.
        let (_, duplicate) = parse_ui_event("U 8 output 6c6f67 6475706c6963617465").unwrap();
        presentation
            .apply(8, duplicate, &mut session, &layout)
            .unwrap();
        assert_eq!(presentation.output_text["main.log"], ["ready"]);
    }

    #[test]
    fn browse_does_not_cross_the_output_control_boundary() {
        let document = dm_dmf::parse(
            "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\telem \"log\"\n\t\ttype = OUTPUT\n",
        );
        let tree = ControlTree::from_document(&document);
        let layout = ClientLayout::from_tree(&tree);
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
                &layout,
            )
            .unwrap();
        assert!(presentation.output_text.is_empty());
        assert_eq!(presentation.browser_html["log"], "not output");
    }

    #[test]
    fn browse_resources_are_registered_before_ordered_html_navigation() {
        let document = dm_dmf::parse(
            "window \"main\"\n\telem \"main\"\n\t\ttype = MAIN\n\telem \"browser\"\n\t\ttype = BROWSER\n",
        );
        let tree = ControlTree::from_document(&document);
        let layout = ClientLayout::from_tree(&tree);
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
                &layout,
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
                &layout,
            )
            .unwrap()
            .unwrap();
        assert_eq!(loaded, "<img src='data:image/png;base64,AAEC'>");
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
    fn remote_transport_attaches_requests_snapshot_and_sends_cardinal_input() {
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
                    "map_snapshot c1",
                    "ok map_snapshot protocol=2 width=8 height=8 z=1 tiles=2\nT 4 5 2f747572662f6f70656e 233131323233  1\nA 1:0 2f6f626a 69636f6e2e646d69 - 2 00000000 00000000 00000000 00000000 00000000 00000000 - 437f0000 0 0\nT 5 5 2f747572662f6f70656e2f666c6f6f72 -  0\n",
                ),
                ("resource c1 69636f6e2e646d69", resource_response.as_str()),
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

        let mut transport = RemoteTransport::connect(address).unwrap();
        transport.attach().unwrap();
        let snapshot = transport.request_snapshot().unwrap();
        assert_eq!(snapshot.center, WorldCoordinate { x: 4, y: 5, z: 1 });
        assert_eq!(snapshot.cells.len(), 2);
        assert_eq!(
            snapshot.appearances.values().map(Vec::len).sum::<usize>(),
            1
        );
        assert_eq!(snapshot.resources.len(), 1);
        let moved = transport.send_movement(1, 0).unwrap().unwrap();
        assert_eq!(moved.center, WorldCoordinate { x: 5, y: 5, z: 1 });
        assert_eq!(moved.cells.len(), 1);
        server.join().unwrap();
    }
}
