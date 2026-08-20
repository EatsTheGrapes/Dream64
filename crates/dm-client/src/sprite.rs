//! PNG-backed DMI decoding and deterministic appearance compositing.

use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

/// One server-owned appearance ready for client composition.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Appearance {
    pub resource: PathBuf,
    pub state: String,
    pub direction: u8,
    pub frame: u32,
    pub plane: f32,
    pub layer: f32,
    pub pixel_x: i32,
    pub pixel_y: i32,
    pub color: [u8; 3],
    pub alpha: u8,
}

/// Decoded RGBA sprite sheet and DMI state layout.
#[derive(Clone, Debug)]
pub(crate) struct DmiSheet {
    pub width: u32,
    pub height: u32,
    image_width: u32,
    rgba: Vec<u8>,
    states: Vec<DmiState>,
}

#[derive(Clone, Debug)]
struct DmiState {
    name: String,
    dirs: u32,
    frames: u32,
    first_cell: u32,
}

/// Resource cache shared by every rendered map frame.
#[derive(Default)]
pub(crate) struct SpriteCache {
    sheets: HashMap<PathBuf, Result<DmiSheet, String>>,
}

impl SpriteCache {
    pub(crate) fn insert(&mut self, path: PathBuf, bytes: &[u8]) {
        self.sheets
            .entry(path)
            .or_insert_with(|| DmiSheet::decode(bytes));
    }

    pub(crate) fn load(&mut self, path: &Path) -> Result<&DmiSheet, String> {
        let entry = self
            .sheets
            .entry(path.to_owned())
            .or_insert_with(|| DmiSheet::load(path));
        entry.as_ref().map_err(Clone::clone)
    }
}

impl DmiSheet {
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        Self::decode(&fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        let (description, ihdr) = png_description(bytes)?;
        let (width, height, states) = parse_description(description.as_deref(), ihdr.0, ihdr.1)?;
        let mut decoder = png::Decoder::new(Cursor::new(bytes));
        decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
        let mut reader = decoder.read_info().map_err(|error| error.to_string())?;
        let mut decoded = vec![0; reader.output_buffer_size()];
        let output = reader
            .next_frame(&mut decoded)
            .map_err(|error| error.to_string())?;
        let source = &decoded[..output.buffer_size()];
        let pixels = usize::try_from(output.width)
            .ok()
            .and_then(|width| {
                usize::try_from(output.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or("PNG dimensions overflow")?;
        let mut rgba = Vec::with_capacity(pixels * 4);
        match output.color_type {
            png::ColorType::Rgba => rgba.extend_from_slice(source),
            png::ColorType::Rgb => source
                .chunks_exact(3)
                .for_each(|pixel| rgba.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255])),
            png::ColorType::Grayscale => source
                .iter()
                .for_each(|value| rgba.extend_from_slice(&[*value, *value, *value, 255])),
            png::ColorType::GrayscaleAlpha => source.chunks_exact(2).for_each(|pixel| {
                rgba.extend_from_slice(&[pixel[0], pixel[0], pixel[0], pixel[1]])
            }),
            png::ColorType::Indexed => return Err("indexed PNG was not expanded".to_owned()),
        }
        Ok(Self {
            width,
            height,
            image_width: output.width,
            rgba,
            states,
        })
    }

    fn cell(&self, state: &str, direction: u8, frame: u32) -> Option<(u32, u32)> {
        let state = self
            .states
            .iter()
            .find(|candidate| candidate.name == state)
            .or_else(|| self.states.first())?;
        let dir = direction_index(direction, state.dirs);
        let frame = frame.saturating_sub(1).min(state.frames.saturating_sub(1));
        let cell = state.first_cell + frame * state.dirs + dir;
        let columns = self.image_width / self.width;
        (columns > 0).then_some((
            (cell % columns) * self.width,
            (cell / columns) * self.height,
        ))
    }
}

/// Composites appearances in BYOND plane/layer order into an ARGB tile buffer.
pub(crate) fn composite_tile(
    cache: &mut SpriteCache,
    appearances: &[Appearance],
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<u32>, String> {
    let mut ordered = appearances.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by(|(left_index, left), (right_index, right)| {
        left.plane
            .total_cmp(&right.plane)
            .then_with(|| left.layer.total_cmp(&right.layer))
            .then(left_index.cmp(right_index))
    });
    let mut output = vec![
        0_u32;
        usize::try_from(tile_width * tile_height)
            .map_err(|_| "tile dimensions overflow")?
    ];
    for (_, appearance) in ordered {
        let sheet = cache.load(&appearance.resource)?;
        let Some((source_x, source_y)) =
            sheet.cell(&appearance.state, appearance.direction, appearance.frame)
        else {
            continue;
        };
        for sy in 0..sheet.height {
            for sx in 0..sheet.width {
                let dx = i64::from(sx) + i64::from(appearance.pixel_x);
                // PNG scanlines and the client framebuffer are top-down;
                // BYOND's positive pixel_y moves an appearance upward.
                let dy = i64::from(sy) - i64::from(appearance.pixel_y);
                if dx < 0 || dy < 0 || dx >= i64::from(tile_width) || dy >= i64::from(tile_height) {
                    continue;
                }
                let source_index =
                    usize::try_from(((source_y + sy) * sheet.image_width + source_x + sx) * 4)
                        .unwrap();
                let source = &sheet.rgba[source_index..source_index + 4];
                let alpha = u32::from(source[3]) * u32::from(appearance.alpha) / 255;
                let tinted = [
                    u32::from(source[0]) * u32::from(appearance.color[0]) / 255,
                    u32::from(source[1]) * u32::from(appearance.color[1]) / 255,
                    u32::from(source[2]) * u32::from(appearance.color[2]) / 255,
                ];
                let destination_index = usize::try_from(dy).unwrap()
                    * usize::try_from(tile_width).unwrap()
                    + usize::try_from(dx).unwrap();
                output[destination_index] = blend_argb(output[destination_index], tinted, alpha);
            }
        }
    }
    Ok(output)
}

fn blend_argb(destination: u32, source: [u32; 3], alpha: u32) -> u32 {
    let inverse = 255 - alpha;
    let da = destination >> 24;
    let dr = destination >> 16 & 255;
    let dg = destination >> 8 & 255;
    let db = destination & 255;
    let out_a = alpha + da * inverse / 255;
    let blend = |source: u32, destination: u32| (source * alpha + destination * inverse) / 255;
    out_a << 24 | blend(source[0], dr) << 16 | blend(source[1], dg) << 8 | blend(source[2], db)
}

fn direction_index(direction: u8, dirs: u32) -> u32 {
    let index = match direction {
        1 => 1,
        4 => 2,
        8 => 3,
        _ => 0,
    };
    index.min(dirs.saturating_sub(1))
}

fn png_description(bytes: &[u8]) -> Result<(Option<String>, (u32, u32)), String> {
    if !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("resource is not PNG".to_owned());
    }
    let mut cursor = 8;
    let mut dimensions = None;
    let mut description = None;
    while cursor + 12 <= bytes.len() {
        let length = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        let end = cursor
            .checked_add(12 + length)
            .ok_or("PNG chunk overflow")?;
        if end > bytes.len() {
            return Err("truncated PNG chunk".to_owned());
        }
        let kind = &bytes[cursor + 4..cursor + 8];
        let data = &bytes[cursor + 8..cursor + 8 + length];
        if kind == b"IHDR" && data.len() >= 8 {
            dimensions = Some((
                u32::from_be_bytes(data[..4].try_into().unwrap()),
                u32::from_be_bytes(data[4..8].try_into().unwrap()),
            ));
        }
        if matches!(kind, b"tEXt" | b"zTXt") && data.starts_with(b"Description\0") {
            if kind == b"tEXt" {
                description = Some(String::from_utf8_lossy(&data[12..]).into_owned());
            } else if data.get(12) == Some(&0) {
                let mut decoder = flate2::read::ZlibDecoder::new(&data[13..]);
                let mut text = String::new();
                decoder
                    .read_to_string(&mut text)
                    .map_err(|error| error.to_string())?;
                description = Some(text);
            }
        }
        cursor = end;
    }
    Ok((description, dimensions.ok_or("PNG lacks IHDR")?))
}

fn parse_description(
    description: Option<&str>,
    image_width: u32,
    image_height: u32,
) -> Result<(u32, u32, Vec<DmiState>), String> {
    let Some(description) = description else {
        return Ok((
            image_width,
            image_height,
            vec![DmiState {
                name: String::new(),
                dirs: 1,
                frames: 1,
                first_cell: 0,
            }],
        ));
    };
    let mut width = image_width;
    let mut height = image_height;
    let mut raw = Vec::<(String, u32, u32)>::new();
    for line in description.lines().map(str::trim) {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            "width" => width = value.trim().parse().map_err(|_| "invalid DMI width")?,
            "height" => height = value.trim().parse().map_err(|_| "invalid DMI height")?,
            "state" => raw.push((value.trim().trim_matches('"').to_owned(), 1, 1)),
            "dirs" => {
                if let Some(state) = raw.last_mut() {
                    state.1 = value.trim().parse().map_err(|_| "invalid DMI dirs")?;
                }
            }
            "frames" => {
                if let Some(state) = raw.last_mut() {
                    state.2 = value.trim().parse().map_err(|_| "invalid DMI frames")?;
                }
            }
            _ => {}
        }
    }
    if raw.is_empty() {
        raw.push((String::new(), 1, 1));
    }
    let mut first = 0;
    let states = raw
        .into_iter()
        .map(|(name, dirs, frames)| {
            let state = DmiState {
                name,
                dirs: dirs.max(1),
                frames: frames.max(1),
                first_cell: first,
            };
            first += state.dirs * state.frames;
            state
        })
        .collect();
    Ok((width, height, states))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_png(path: &Path, rgba: [u8; 4]) {
        let file = fs::File::create(path).unwrap();
        let mut encoder = png::Encoder::new(file, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&rgba)
            .unwrap();
    }

    #[test]
    fn dmi_description_tracks_state_cell_offsets_dirs_and_frames() {
        let description = "# BEGIN DMI\nversion = 4.0\nwidth = 32\nheight = 32\nstate = \"idle\"\ndirs = 4\nframes = 2\nstate = \"walk\"\ndirs = 1\nframes = 3\n# END DMI\n";
        let (width, height, states) = parse_description(Some(description), 256, 32).unwrap();
        assert_eq!((width, height), (32, 32));
        assert_eq!(
            (states[0].first_cell, states[0].dirs, states[0].frames),
            (0, 4, 2)
        );
        assert_eq!(
            (states[1].first_cell, states[1].dirs, states[1].frames),
            (8, 1, 3)
        );
    }

    #[test]
    fn png_fixture_composites_plane_layer_tint_alpha_and_pixel_offset() {
        let root = std::env::temp_dir().join(format!("dream64-sprite-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let red = root.join("red.dmi");
        let white = root.join("white.dmi");
        fixture_png(&red, [255, 0, 0, 255]);
        fixture_png(&white, [255, 255, 255, 255]);
        let appearance = |resource: PathBuf, plane, layer, pixel_x, color, alpha| Appearance {
            resource,
            state: String::new(),
            direction: 2,
            frame: 1,
            plane,
            layer,
            pixel_x,
            pixel_y: 0,
            color,
            alpha,
        };
        let appearances = vec![
            appearance(red, 0.0, 0.0, 0, [255; 3], 255),
            appearance(white.clone(), 0.0, 1.0, 0, [0, 0, 255], 128),
            appearance(white, -1.0, 0.0, 1, [0, 255, 0], 255),
        ];
        let pixels = composite_tile(&mut SpriteCache::default(), &appearances, 2, 1).unwrap();
        assert_eq!(pixels[0] >> 24, 255);
        assert!(
            (pixels[0] & 255) >= 127,
            "upper blue layer must alpha blend"
        );
        assert!(
            (pixels[0] >> 16 & 255) >= 126,
            "lower red layer remains visible"
        );
        assert_eq!(pixels[1], 0xff00_ff00, "pixel_x offsets the green sprite");
        fs::remove_dir_all(root).unwrap();
    }
}
