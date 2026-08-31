//! Text-protocol serialization for the loopback IPC: client-state lines, map
//! and screen snapshots, appearance trees, retained UI event batches, and the
//! sandboxed project-resource readers, plus the hex helpers they share.

use dm_vm::{ExecutionState, LocalClientMapSnapshot, LocalClientState, LocalClientUiEvent};

use super::MAX_RESOURCE_CHUNK_BYTES;

pub(super) fn format_state(
    kind: &str,
    session: &str,
    state: &LocalClientState,
    tick: u64,
) -> String {
    format!(
        "ok {kind} protocol=1 client={session} mob=[0xd{:06x}] tick={tick} x={} y={} z={}",
        state.mob.index() + 1,
        state.x,
        state.y,
        state.z
    )
}
pub(super) fn encode_snapshot(
    session: &str,
    tick: u64,
    center: (i32, i32, i32),
    snapshot: LocalClientMapSnapshot,
) -> String {
    let mut out = format!(
        "ok map_snapshot protocol=4 session={session} tick={tick} width={} height={} x={} y={} z={} tiles={} screen={}\n",
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
        "A {:x}:{:x} {} {} {} {} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {} {:08x} {} {} {} {:08x} {:08x} {:08x} {:08x} {} {}",
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
        appearance.appearance_flags,
        appearance.mouse_opacity,
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

pub(super) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

pub(super) fn unhex(value: &str) -> Result<Vec<u8>, String> {
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

pub(super) fn read_project_resource(state: &ExecutionState, path: &str) -> Result<Vec<u8>, String> {
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

#[derive(Debug, PartialEq)]
pub(super) struct ResourceChunk {
    pub(super) bytes: Vec<u8>,
    pub(super) total: u64,
    pub(super) eof: bool,
}

pub(super) fn read_project_resource_chunk(
    state: &ExecutionState,
    path: &str,
    offset: u64,
    length: u32,
) -> Result<ResourceChunk, String> {
    use std::{
        io::{Read as _, Seek as _, SeekFrom},
        path::{Component, Path},
    };
    if length == 0 || length > MAX_RESOURCE_CHUNK_BYTES {
        return Err("resource chunk length is out of range".to_owned());
    }
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
    let mut file = std::fs::File::open(target).map_err(|error| error.to_string())?;
    let total = file.metadata().map_err(|error| error.to_string())?.len();
    if offset > total {
        return Err("resource chunk offset exceeds file length".to_owned());
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| error.to_string())?;
    let remaining = total - offset;
    let take = remaining.min(u64::from(length)) as usize;
    let mut bytes = vec![0; take];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(ResourceChunk {
        bytes,
        total,
        eof: offset + take as u64 == total,
    })
}

pub(super) fn encode_retained_ui_events(
    session: &str,
    events: &[(u64, LocalClientUiEvent)],
) -> String {
    use std::fmt::Write as _;
    let mut output = format!(
        "ok ui_events protocol=6 client={session} count={}\n",
        events.len()
    );
    for (sequence, event) in events {
        match event.clone() {
            LocalClientUiEvent::Link { url } => {
                let _ = writeln!(output, "U {sequence} link {}", required_hex(url.as_bytes()));
            }
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
