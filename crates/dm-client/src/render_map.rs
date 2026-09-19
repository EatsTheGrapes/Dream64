//! Map rendering, hit testing, and screen/world display item management.

use dm_dmf::PixelRect;
use dm_world::WorldCoordinate;
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

use crate::gpu;
use crate::render_cpu::{draw_appearance_maptext_signed, draw_panel};
use crate::sprite::{self, Appearance, SpriteCache};
use crate::transport::{MapSnapshot, ScreenAppearance};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MapTransform {
    pub(crate) clip: PixelRect,
    pub(crate) origin_x: u32,
    pub(crate) origin_y: u32,
    pub(crate) tile: u32,
    pub(crate) columns: u32,
    pub(crate) rows: u32,
}

impl MapTransform {
    pub(crate) fn new(
        rect: PixelRect,
        tile_size: u32,
        zoom: f32,
        zoom_mode: &str,
        letterbox: bool,
    ) -> Self {
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

    pub(crate) fn world_at(
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

pub(crate) struct WorldDisplayItem {
    pub(crate) datum: (u32, u32),
    pub(crate) owner: WorldCoordinate,
    pub(crate) origin_x: i32,
    pub(crate) origin_y: i32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixels: Vec<u32>,
    pub(crate) appearance: Appearance,
    pub(crate) maptext_origin_x: i32,
    pub(crate) maptext_origin_y: i32,
}

pub(crate) fn world_display_items(
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
                sprite::rasterize_world_appearance(sprites, &appearance).ok()?;
            if native_width == 0 || native_height == 0 {
                return None;
            }
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

pub(crate) struct GpuWorldDisplayItem {
    pub(crate) draw: gpu::DmiSpriteDraw,
    pub(crate) appearance: Appearance,
    pub(crate) maptext_origin_x: i32,
    pub(crate) maptext_origin_y: i32,
}

pub(crate) fn gpu_world_display_items(
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

pub(crate) fn draw_map(
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
    let snapshot_bounds =
        snapshot.and_then(|snapshot| snapshot_bounds_at_z(snapshot, snapshot.center.z));
    for row in 0..rows {
        for column in 0..columns {
            let shade = snapshot
                .and_then(|snapshot| {
                    let x = snapshot.center.x
                        + i32::try_from(column).expect("grid column fits i32")
                        - center_column;
                    let y = snapshot.center.y + center_row
                        - i32::try_from(row).expect("grid row fits i32");
                    snapshot
                        .cells
                        .get(&(x, y, snapshot.center.z))
                        .copied()
                        .or_else(|| {
                            let (min_x, max_x, min_y, max_y) = snapshot_bounds?;
                            snapshot
                                .cells
                                .get(&(
                                    x.clamp(min_x, max_x),
                                    y.clamp(min_y, max_y),
                                    snapshot.center.z,
                                ))
                                .copied()
                        })
                })
                .unwrap_or_else(|| {
                    if snapshot.is_some() {
                        0xff3b_3b3b
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
            match sprite::composite_native(sprites, &screen.appearances) {
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

fn snapshot_bounds_at_z(snapshot: &MapSnapshot, z: i32) -> Option<(i32, i32, i32, i32)> {
    let mut coordinates = snapshot
        .cells
        .keys()
        .filter(|coordinate| coordinate.2 == z)
        .map(|coordinate| (coordinate.0, coordinate.1));
    let (first_x, first_y) = coordinates.next()?;
    Some(coordinates.fold(
        (first_x, first_x, first_y, first_y),
        |(min_x, max_x, min_y, max_y), (x, y)| {
            (min_x.min(x), max_x.max(x), min_y.min(y), max_y.max(y))
        },
    ))
}

pub(crate) fn screen_loc_pixels(
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
    if matches!(
        first.to_ascii_uppercase().split(':').next(),
        Some("TOP" | "BOTTOM")
    ) {
        let horizontal = named_screen_axis(second, transform.columns, transform.tile)?;
        let x = horizontal;
        let y = if first.to_ascii_uppercase().starts_with("TOP") {
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

pub(crate) fn screen_is_render_pipeline_helper(screen: &ScreenAppearance) -> bool {
    screen
        .type_path
        .starts_with("/atom/movable/screen/plane_master")
        || screen
            .type_path
            .starts_with("/atom/movable/screen/click_catcher")
        || screen
            .type_path
            .starts_with("/atom/movable/render_plane_relay")
        || screen
            .type_path
            .starts_with("/atom/movable/screen/fullscreen/lighting_backdrop")
}

pub(crate) const PASS_MOUSE_APPEARANCE_FLAG: i32 = 1 << 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MousePickPolicy {
    Transparent,
    PixelOpaque,
    Opaque,
}

impl MousePickPolicy {
    fn from_mouse_opacity(mouse_opacity: i32) -> Self {
        match mouse_opacity {
            0 => Self::Transparent,
            2 => Self::Opaque,
            _ => Self::PixelOpaque,
        }
    }
}

pub(crate) fn appearance_hit_at(
    appearance: &Appearance,
    pixels: &[u32],
    width: u32,
    height: u32,
    local_x: i32,
    local_y: i32,
) -> bool {
    if local_x < 0
        || local_y < 0
        || local_x >= i32::try_from(width).unwrap_or(i32::MAX)
        || local_y >= i32::try_from(height).unwrap_or(i32::MAX)
    {
        return false;
    }
    if appearance.appearance_flags & PASS_MOUSE_APPEARANCE_FLAG != 0 {
        return false;
    }
    let pixel_index = usize::try_from(local_y).unwrap() * usize::try_from(width).unwrap()
        + usize::try_from(local_x).unwrap();
    match MousePickPolicy::from_mouse_opacity(appearance.mouse_opacity) {
        MousePickPolicy::Transparent => false,
        MousePickPolicy::PixelOpaque => pixels
            .get(pixel_index)
            .is_some_and(|pixel| *pixel >> 24 != 0),
        MousePickPolicy::Opaque => true,
    }
}

pub(crate) fn screen_hit_at(
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
        let Ok((width, height, _pixels)) = sprite::composite_native(sprites, &screen.appearances)
        else {
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
        let hit = screen.appearances.iter().rev().any(|appearance| {
            let Ok((appearance_width, appearance_height, appearance_pixels)) =
                sprite::rasterize_world_appearance(sprites, appearance)
            else {
                return false;
            };
            appearance_hit_at(
                appearance,
                &appearance_pixels,
                appearance_width,
                appearance_height,
                local_x - appearance.pixel_x,
                local_y + appearance.pixel_y,
            )
        });
        if hit {
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

pub(crate) fn map_hit_at(
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
            appearance_hit_at(
                &item.appearance,
                &item.pixels,
                item.width,
                item.height,
                screen_x - item.origin_x,
                screen_y - item.origin_y,
            )
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

pub(crate) fn click_pointer_params(
    base: &str,
    button: MouseButton,
    modifiers: ModifiersState,
) -> String {
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

pub(crate) fn blit_sprite_clipped(
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

pub(crate) fn world_appearance_origin(
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
