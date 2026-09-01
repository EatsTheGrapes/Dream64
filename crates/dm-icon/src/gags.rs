//! Greyscale Asset Generation: composite a GAGS config's layer stacks into an
//! output DMI, one icon_state per top-level config key.
//!
//! Config shape (`code/datums/greyscale/json_configs/*.json`): the top-level
//! object maps output state name -> layer list. A layer is either an object
//! (`type`, `blend_mode`, `color_ids`, plus type-specific keys) or a nested
//! list (a group; the first entry's `blend_mode` drives how the group merges
//! into the stack).

use crate::bitmap::{Frame, IconBitmap, IconState};
use crate::blend::BlendMode;
use crate::{Rgba, parse_color};

/// Parse a BYOND GAGS colour string (`"#ff00ff#ffaa00"`) into a palette.
#[must_use]
pub fn parse_color_string(text: &str) -> Vec<Rgba> {
    text.split('#')
        .filter(|s| !s.is_empty())
        .filter_map(|s| parse_color(&format!("#{s}")))
        .collect()
}

/// Composite every output state defined by `config_json` using `template` as
/// the source DMI and `palette` as the 1-indexed colour list.
///
/// # Errors
/// Returns a message if the config JSON is not a valid GAGS document.
pub fn composite(
    config_json: &str,
    template: &IconBitmap,
    palette: &[Rgba],
) -> Result<IconBitmap, String> {
    let root: serde_json::Value =
        serde_json::from_str(config_json).map_err(|e| format!("invalid GAGS config: {e}"))?;
    let obj = root
        .as_object()
        .ok_or_else(|| "GAGS config root must be an object".to_owned())?;

    let mut output = IconBitmap {
        width: template.width,
        height: template.height,
        states: Vec::with_capacity(obj.len()),
    };

    for (state_name, layers) in obj {
        let layers = layers
            .as_array()
            .ok_or_else(|| format!("GAGS state {state_name:?} must be a layer list"))?;
        let generated = generate_group(layers, template, palette);
        let state = match generated {
            Some(bitmap) => finalize_state(state_name, bitmap, template),
            None => IconState {
                name: state_name.clone(),
                dirs: 1,
                frame_count: 1,
                delays: Vec::new(),
                loop_count: 0,
                rewind: false,
                movement: false,
                hotspot: None,
                cells: vec![Frame::transparent(template.width, template.height)],
            },
        };
        output.states.push(state);
    }

    if output.states.is_empty() {
        return Err("GAGS config produced no states".to_owned());
    }
    Ok(output)
}

fn finalize_state(name: &str, mut bitmap: IconBitmap, template: &IconBitmap) -> IconState {
    if (bitmap.width, bitmap.height) != (template.width, template.height) {
        bitmap.scale(template.width, template.height);
    }
    let mut state = bitmap.states.pop().unwrap_or_else(|| IconState {
        name: name.to_owned(),
        dirs: 1,
        frame_count: 1,
        delays: Vec::new(),
        loop_count: 0,
        rewind: false,
        movement: false,
        hotspot: None,
        cells: vec![Frame::transparent(template.width, template.height)],
    });
    state.name = String::from(name);
    state
}

/// Merge a layer group into a single bitmap.
fn generate_group(
    layers: &[serde_json::Value],
    template: &IconBitmap,
    palette: &[Rgba],
) -> Option<IconBitmap> {
    let mut accumulator: Option<IconBitmap> = None;
    for layer in layers {
        let (layer_icon, blend_mode) = if let Some(group) = layer.as_array() {
            let icon = generate_group(group, template, palette)?;
            let mode = group
                .first()
                .and_then(|l| l.get("blend_mode"))
                .and_then(serde_json::Value::as_str)
                .and_then(BlendMode::from_gags_name)
                .unwrap_or(BlendMode::Overlay);
            (Some(icon), mode)
        } else {
            let mode = layer
                .get("blend_mode")
                .and_then(serde_json::Value::as_str)
                .and_then(BlendMode::from_gags_name)
                .unwrap_or(BlendMode::Overlay);
            (generate_layer(layer, template, palette), mode)
        };

        let Some(layer_icon) = layer_icon else {
            continue;
        };
        match &mut accumulator {
            None => accumulator = Some(layer_icon),
            Some(acc) => acc.blend_icon(&layer_icon, blend_mode, 1, 1),
        }
    }
    accumulator
}

fn generate_layer(
    layer: &serde_json::Value,
    template: &IconBitmap,
    palette: &[Rgba],
) -> Option<IconBitmap> {
    let layer_type = layer.get("type").and_then(serde_json::Value::as_str)?;
    let color_ids: Vec<serde_json::Value> = layer
        .get("color_ids")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let resolved_colors: Vec<Rgba> = color_ids
        .iter()
        .filter_map(|id| {
            if let Some(n) = id.as_i64() {
                palette.get((n - 1).max(0) as usize).copied()
            } else {
                id.as_str().and_then(parse_color)
            }
        })
        .collect();

    match layer_type {
        "icon_state" => {
            let state_name = layer
                .get("icon_state")
                .and_then(serde_json::Value::as_str)?;
            let mut icon = template.select_state(state_name);
            if icon.states.is_empty() {
                return None;
            }
            if let Some(color) = resolved_colors.first() {
                icon.blend_color(*color, BlendMode::Multiply);
            }
            Some(icon)
        }
        // color_matrix / reference layers are not yet composited natively; they
        // contribute nothing rather than aborting bundle generation, so the
        // output DMI still carries every config-defined state name.
        // TODO(dm-icon): color_matrix + reference GAGS layers.
        _ => None,
    }
}
