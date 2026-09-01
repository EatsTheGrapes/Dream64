//! PNG-backed DMI decode/encode with BYOND `Description` metadata.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless
)]

use std::fmt;
use std::io::Cursor;

use crate::Rgba;
use crate::bitmap::{Frame, IconBitmap, IconState};

/// Error decoding or encoding a DMI.
#[derive(Debug)]
pub enum DmiError {
    Png(String),
    Metadata(String),
}

impl fmt::Display for DmiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Png(m) => write!(f, "DMI PNG error: {m}"),
            Self::Metadata(m) => write!(f, "DMI metadata error: {m}"),
        }
    }
}

impl std::error::Error for DmiError {}

#[derive(Default)]
struct RawState {
    name: String,
    dirs: u32,
    frames: u32,
    delays: Vec<f32>,
    loop_count: i64,
    rewind: bool,
    movement: bool,
    hotspot: Option<Vec<i64>>,
}

fn parse_description(text: &str) -> Result<Vec<RawState>, DmiError> {
    let mut states: Vec<RawState> = Vec::new();
    let mut current: Option<RawState> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(DmiError::Metadata(format!("invalid line {line:?}")));
        };
        let key = key.trim();
        let value = value.trim();
        let num = |v: &str| -> Result<i64, DmiError> {
            v.parse::<i64>()
                .or_else(|_| v.parse::<f64>().map(|f| f as i64))
                .map_err(|e| DmiError::Metadata(format!("bad {key} {value:?}: {e}")))
        };
        match key {
            // `width`/`height` come from the PNG IHDR + the pre-scan below.
            "version" | "md5" | "width" | "height" => {}
            "state" => {
                if let Some(previous) = current.take() {
                    states.push(previous);
                }
                let name = serde_json::from_str::<String>(value)
                    .unwrap_or_else(|_| value.trim_matches('"').to_owned());
                current = Some(RawState {
                    name,
                    dirs: 1,
                    frames: 1,
                    ..RawState::default()
                });
            }
            "dirs" => {
                if let Some(s) = current.as_mut() {
                    s.dirs = num(value)?.max(1) as u32;
                }
            }
            "frames" => {
                if let Some(s) = current.as_mut() {
                    s.frames = num(value)?.max(1) as u32;
                }
            }
            "delay" => {
                if let Some(s) = current.as_mut() {
                    s.delays = value
                        .split(',')
                        .map(str::trim)
                        .filter(|v| !v.is_empty())
                        .map(str::parse::<f32>)
                        .collect::<Result<_, _>>()
                        .map_err(|e| DmiError::Metadata(format!("bad delay {value:?}: {e}")))?;
                }
            }
            "loop" => {
                if let Some(s) = current.as_mut() {
                    s.loop_count = num(value)?;
                }
            }
            "rewind" => {
                if let Some(s) = current.as_mut() {
                    s.rewind = num(value)? != 0;
                }
            }
            "movement" => {
                if let Some(s) = current.as_mut() {
                    s.movement = num(value)? != 0;
                }
            }
            "hotspot" => {
                if let Some(s) = current.as_mut() {
                    s.hotspot = Some(
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|v| !v.is_empty())
                            .map(|v| v.parse::<f64>().map(|f| f as i64))
                            .collect::<Result<_, _>>()
                            .map_err(|e| DmiError::Metadata(format!("bad hotspot: {e}")))?,
                    );
                }
            }
            other => return Err(DmiError::Metadata(format!("unknown key {other:?}"))),
        }
    }
    if let Some(s) = current {
        states.push(s);
    }
    Ok(states)
}

/// Render an `IconBitmap` to a BYOND `Description` string.
pub(crate) fn build_description(bitmap: &IconBitmap) -> String {
    let mut out = String::from("# BEGIN DMI\nversion = 4.0\n");
    out.push_str(&format!("\twidth = {}\n", bitmap.width));
    out.push_str(&format!("\theight = {}\n", bitmap.height));
    for state in &bitmap.states {
        let name = serde_json::to_string(&state.name).unwrap_or_else(|_| "\"\"".to_owned());
        out.push_str(&format!("state = {name}\n"));
        out.push_str(&format!("\tdirs = {}\n", state.dirs.max(1)));
        out.push_str(&format!("\tframes = {}\n", state.frame_count.max(1)));
        if state.frame_count > 1 && !state.delays.is_empty() {
            let delays = state
                .delays
                .iter()
                .map(|d| format_num(f64::from(*d)))
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!("\tdelay = {delays}\n"));
        }
        if state.loop_count != 0 {
            out.push_str(&format!("\tloop = {}\n", state.loop_count));
        }
        if state.rewind {
            out.push_str("\trewind = 1\n");
        }
        if state.movement {
            out.push_str("\tmovement = 1\n");
        }
        if let Some(hotspot) = &state.hotspot {
            let hs = hotspot
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&format!("\thotspot = {hs}\n"));
        }
    }
    out.push_str("# END DMI\n");
    out
}

fn format_num(value: f64) -> String {
    if (value.fract()).abs() < f64::EPSILON {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Decode a PNG-backed DMI into an [`IconBitmap`].
pub(crate) fn decode(bytes: &[u8]) -> Result<IconBitmap, DmiError> {
    let transforms = png::Transformations::EXPAND | png::Transformations::STRIP_16;
    let mut decoder = png::Decoder::new(Cursor::new(bytes));
    decoder.set_transformations(transforms);
    let mut reader = decoder
        .read_info()
        .map_err(|e| DmiError::Png(e.to_string()))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let out = reader
        .next_frame(&mut buf)
        .map_err(|e| DmiError::Png(e.to_string()))?;
    let (img_w, img_h) = (out.width, out.height);
    let rgba = to_rgba(&buf[..out.buffer_size()], out.color_type, img_w, img_h)?;

    let info = reader.info();
    let mut description = None;
    for chunk in &info.compressed_latin1_text {
        if chunk.keyword == "Description" {
            description = chunk.get_text().ok();
        }
    }
    if description.is_none() {
        for chunk in &info.uncompressed_latin1_text {
            if chunk.keyword == "Description" {
                description = Some(chunk.text.clone());
            }
        }
    }

    let (mut cell_w, mut cell_h) = (img_w, img_h);
    let raw_states = match &description {
        Some(text) => {
            for line in text.lines() {
                let line = line.trim();
                if let Some(v) = line.strip_prefix("width = ") {
                    cell_w = v.trim().parse().unwrap_or(cell_w);
                } else if let Some(v) = line.strip_prefix("height = ") {
                    cell_h = v.trim().parse().unwrap_or(cell_h);
                }
            }
            parse_description(text)?
        }
        None => vec![RawState {
            name: String::new(),
            dirs: 1,
            frames: 1,
            ..RawState::default()
        }],
    };
    if cell_w == 0 || cell_h == 0 {
        return Err(DmiError::Metadata("zero cell size".into()));
    }
    let cols = (img_w / cell_w).max(1);

    let mut states = Vec::with_capacity(raw_states.len());
    let mut index = 0u32;
    for raw in raw_states {
        let count = raw.dirs * raw.frames;
        let mut cells = Vec::with_capacity(count as usize);
        for _ in 0..count {
            let cx = (index % cols) * cell_w;
            let cy = (index / cols) * cell_h;
            cells.push(extract(&rgba, img_w, img_h, cx, cy, cell_w, cell_h));
            index += 1;
        }
        states.push(IconState {
            name: raw.name,
            dirs: raw.dirs,
            frame_count: raw.frames,
            delays: raw.delays,
            loop_count: raw.loop_count,
            rewind: raw.rewind,
            movement: raw.movement,
            hotspot: raw.hotspot,
            cells,
        });
    }

    Ok(IconBitmap {
        width: cell_w,
        height: cell_h,
        states,
    })
}

fn to_rgba(
    buf: &[u8],
    color: png::ColorType,
    width: u32,
    height: u32,
) -> Result<Vec<Rgba>, DmiError> {
    let count = (width as usize) * (height as usize);
    let mut out = vec![[0u8; 4]; count];
    match color {
        png::ColorType::Rgba => {
            for (i, px) in out.iter_mut().enumerate() {
                px.copy_from_slice(&buf[i * 4..i * 4 + 4]);
            }
        }
        png::ColorType::Rgb => {
            for (i, px) in out.iter_mut().enumerate() {
                px[..3].copy_from_slice(&buf[i * 3..i * 3 + 3]);
                px[3] = 255;
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for (i, px) in out.iter_mut().enumerate() {
                let g = buf[i * 2];
                *px = [g, g, g, buf[i * 2 + 1]];
            }
        }
        png::ColorType::Grayscale => {
            for (i, px) in out.iter_mut().enumerate() {
                let g = buf[i];
                *px = [g, g, g, 255];
            }
        }
        png::ColorType::Indexed => {
            return Err(DmiError::Png("indexed PNG not expanded".into()));
        }
    }
    Ok(out)
}

fn extract(rgba: &[Rgba], img_w: u32, img_h: u32, ox: u32, oy: u32, w: u32, h: u32) -> Frame {
    let mut pixels = vec![[0u8; 4]; (w as usize) * (h as usize)];
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (ox + x, oy + y);
            if sx < img_w && sy < img_h {
                pixels[(y as usize) * (w as usize) + (x as usize)] =
                    rgba[(sy as usize) * (img_w as usize) + (sx as usize)];
            }
        }
    }
    Frame {
        width: w,
        height: h,
        pixels,
    }
}

/// Encode an [`IconBitmap`] to a PNG-backed DMI byte stream.
pub(crate) fn encode(bitmap: &IconBitmap) -> Result<Vec<u8>, DmiError> {
    let (w, h) = (bitmap.width.max(1), bitmap.height.max(1));
    let total: u32 = bitmap
        .states
        .iter()
        .map(|s| s.dirs.max(1) * s.frame_count.max(1))
        .sum::<u32>()
        .max(1);
    let cols = ((total as f64).sqrt().ceil() as u32).max(1);
    let rows = total.div_ceil(cols);
    let (img_w, img_h) = (cols * w, rows * h);

    let mut canvas = vec![0u8; (img_w as usize) * (img_h as usize) * 4];
    let mut index = 0u32;
    for state in &bitmap.states {
        let expected = (state.dirs.max(1) * state.frame_count.max(1)) as usize;
        for cell in state.cells.iter().take(expected) {
            blit(&mut canvas, img_w, index % cols * w, index / cols * h, cell);
            index += 1;
        }
        for _ in state.cells.len()..expected {
            index += 1;
        }
    }

    let description = build_description(bitmap);
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(Cursor::new(&mut bytes), img_w, img_h);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .add_ztxt_chunk("Description".to_owned(), description)
            .map_err(|e| DmiError::Png(e.to_string()))?;
        let mut writer = encoder
            .write_header()
            .map_err(|e| DmiError::Png(e.to_string()))?;
        writer
            .write_image_data(&canvas)
            .map_err(|e| DmiError::Png(e.to_string()))?;
        writer.finish().map_err(|e| DmiError::Png(e.to_string()))?;
    }
    Ok(bytes)
}

fn blit(canvas: &mut [u8], img_w: u32, ox: u32, oy: u32, cell: &Frame) {
    for y in 0..cell.height {
        for x in 0..cell.width {
            let (dx, dy) = (ox + x, oy + y);
            let di = ((dy as usize) * (img_w as usize) + (dx as usize)) * 4;
            canvas[di..di + 4].copy_from_slice(&cell.get(x, y));
        }
    }
}
