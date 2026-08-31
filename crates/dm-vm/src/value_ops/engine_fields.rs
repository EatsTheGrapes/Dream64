//! Engine-root built-in field metadata.
//!
//! Split out of `value_ops`: the static tables of BYOND-defined field names
//! per engine root type (`/atom`, `/mob`, `/client`, ...) and the pure
//! lookups that resolve a concrete runtime path to its guaranteed engine
//! roots and their compile-time initial values.

use std::collections::BTreeMap;

use dm_value::{FieldName, TypePath, Value};

use crate::ExecutionState;

/// Returns the effective initial field catalog for an engine atom root.
///
/// `OpenDream` exposes `/atom` variables through `DreamObjectAtom` and its
/// engine-owned appearance state even when an object's concrete definition is
/// synthesized at runtime. Dream64 normally flattens those values into every
/// registered type. Legacy/native construction can produce a concrete path
/// absent from that catalog, so standard atom fields must fall back through
/// the guaranteed engine roots rather than becoming nonexistent.
pub(crate) fn engine_root_paths(runtime_type: &TypePath) -> &'static [&'static str] {
    let path = runtime_type.as_str();
    if path == "/world" {
        &["/world", "/datum"]
    } else if path == "/obj" || path.starts_with("/obj/") {
        &["/obj", "/atom/movable", "/atom", "/datum"]
    } else if path == "/mob" || path.starts_with("/mob/") {
        &["/mob", "/atom/movable", "/atom", "/datum"]
    } else if path == "/turf" || path.starts_with("/turf/") {
        &["/turf", "/atom", "/datum"]
    } else if path == "/area" || path.starts_with("/area/") {
        &["/area", "/atom", "/datum"]
    } else if path == "/atom/movable" || path.starts_with("/atom/movable/") {
        &["/atom/movable", "/atom", "/datum"]
    } else if path == "/atom" || path.starts_with("/atom/") {
        &["/atom", "/datum"]
    } else if path == "/image" || path.starts_with("/image/") {
        &["/image", "/datum"]
    } else if path == "/client" || path.starts_with("/client/") {
        &["/client", "/datum"]
    } else if path == "/particles" || path.starts_with("/particles/") {
        &["/particles", "/datum"]
    } else if path == "/sound" || path.starts_with("/sound/") {
        &["/sound", "/datum"]
    } else if path == "/datum" || path.starts_with("/datum/") {
        &["/datum"]
    } else {
        // Engine-owned datum kinds such as `/regex`, `/dm_filter`, `/matrix`,
        // and `/icon` do not necessarily appear beneath `/datum` in the
        // project's source tree, but they still expose BYOND's base datum
        // storage.
        &["/datum"]
    }
}

pub(crate) const ENGINE_DATUM_FIELDS: &[&str] = &["datum_flags", "tag"];
pub(crate) const ENGINE_ATOM_FIELDS: &[&str] = &[
    "alpha",
    "appearance",
    "appearance_flags",
    "blend_mode",
    "color",
    "contents",
    "density",
    "desc",
    "dir",
    "gender",
    "filters",
    "icon",
    "icon_state",
    "invisibility",
    "layer",
    "loc",
    "luminosity",
    "maptext",
    "maptext_height",
    "maptext_width",
    "maptext_x",
    "maptext_y",
    "mouse_opacity",
    "mouse_over_pointer",
    "name",
    "opacity",
    "overlays",
    "particles",
    "plane",
    "pixel_x",
    "pixel_y",
    "pixel_w",
    "pixel_z",
    "render_source",
    "render_target",
    "suffix",
    "text",
    "transform",
    "underlays",
    "vis_contents",
    "vis_locs",
    "vis_flags",
    "verbs",
    "x",
    "y",
    "z",
];
pub(crate) const ENGINE_MOVABLE_FIELDS: &[&str] = &[
    "animate_movement",
    "bound_height",
    "bound_width",
    "bound_x",
    "bound_y",
    "glide_size",
    "locs",
    "screen_loc",
    "step_x",
    "step_y",
    "step_size",
];
pub(crate) const ENGINE_MOB_FIELDS: &[&str] = &[
    "ckey",
    "client",
    "eye",
    "key",
    "perspective",
    "see_in_dark",
    "see_infrared",
    "see_invisible",
    "sight",
];
pub(crate) const ENGINE_WORLD_FIELDS: &[&str] = &["maxx", "maxy", "maxz"];
pub(crate) const ENGINE_CLIENT_FIELDS: &[&str] = &[
    "address",
    "ckey",
    "computer_id",
    "connection",
    "byond_build",
    "byond_version",
    "control_freak",
    "dir",
    "eye",
    "gender",
    "fps",
    "inactivity",
    "key",
    "mob",
    "mouse_pointer_icon",
    "perspective",
    "pixel_w",
    "pixel_x",
    "pixel_y",
    "pixel_z",
    "screen",
    "statobj",
    "view",
];
pub(crate) const ENGINE_IMAGE_FIELDS: &[&str] = &[
    "alpha",
    "appearance",
    "appearance_flags",
    "blend_mode",
    "color",
    "dir",
    "icon",
    "icon_state",
    "layer",
    "loc",
    "name",
    "overlays",
    "plane",
    "pixel_x",
    "pixel_y",
    "pixel_w",
    "pixel_z",
    "transform",
    "underlays",
    "vis_contents",
];
pub(crate) const ENGINE_PARTICLE_FIELDS: &[&str] = &[
    "color",
    "width",
    "height",
    "count",
    "spawning",
    "bound1",
    "bound2",
    "gravity",
    "gradient",
    "color_change",
    "transform",
    "icon",
    "icon_state",
    "lifespan",
    "fadein",
    "fade",
    "position",
    "velocity",
    "scale",
    "grow",
    "rotation",
    "spin",
    "friction",
    "drift",
];
pub(crate) const ENGINE_SOUND_FIELDS: &[&str] = &[
    "file",
    "repeat",
    "wait",
    "channel",
    "volume",
    "frequency",
    "pan",
    "offset",
];

pub(crate) fn engine_owner_field_names(owner: &str) -> &'static [&'static str] {
    match owner {
        "/datum" => ENGINE_DATUM_FIELDS,
        "/atom" => ENGINE_ATOM_FIELDS,
        "/atom/movable" => ENGINE_MOVABLE_FIELDS,
        "/mob" => ENGINE_MOB_FIELDS,
        "/world" => ENGINE_WORLD_FIELDS,
        "/client" => ENGINE_CLIENT_FIELDS,
        "/image" => ENGINE_IMAGE_FIELDS,
        "/particles" => ENGINE_PARTICLE_FIELDS,
        "/sound" => ENGINE_SOUND_FIELDS,
        _ => &[],
    }
}

pub(crate) fn engine_owner_initial_value(owner: &str, field: &FieldName) -> Option<Value> {
    let name = field.as_str();
    if !engine_owner_field_names(owner).contains(&name) {
        return None;
    }
    let value = match owner {
        "/datum" => match name {
            "datum_flags" => Value::number(0.0),
            _ => Value::Null,
        },
        "/atom" => match name {
            "alpha" => Value::number(255.0),
            "dir" => Value::number(2.0),
            "gender" => Value::text("neuter"),
            "layer" | "mouse_opacity" => Value::number(1.0),
            "maptext_height" | "maptext_width" => Value::number(32.0),
            "appearance_flags" | "blend_mode" | "density" | "invisibility" | "luminosity"
            | "maptext_x" | "maptext_y" | "opacity" | "plane" | "pixel_x" | "pixel_y"
            | "pixel_w" | "pixel_z" | "vis_flags" | "x" | "y" | "z" => Value::number(0.0),
            _ => Value::Null,
        },
        "/atom/movable" => match name {
            "bound_height" | "bound_width" | "step_size" => Value::number(32.0),
            "animate_movement" | "bound_x" | "bound_y" | "glide_size" | "step_x" | "step_y" => {
                Value::number(0.0)
            }
            _ => Value::Null,
        },
        "/mob" => match name {
            "see_in_dark" => Value::number(2.0),
            "perspective" | "see_infrared" | "see_invisible" | "sight" => Value::number(0.0),
            _ => Value::Null,
        },
        "/world" => Value::number(0.0),
        "/client" => match name {
            "dir" => Value::number(2.0),
            "gender" => Value::text("neuter"),
            "fps" => Value::number(10.0),
            "view" => Value::number(5.0),
            "control_freak" | "inactivity" | "perspective" | "pixel_w" | "pixel_x" | "pixel_y"
            | "pixel_z" => Value::number(0.0),
            _ => Value::Null,
        },
        "/image" => match name {
            "alpha" => Value::number(255.0),
            "dir" => Value::number(2.0),
            "appearance_flags" | "blend_mode" | "layer" | "plane" | "pixel_x" | "pixel_y"
            | "pixel_w" | "pixel_z" => Value::number(0.0),
            _ => Value::Null,
        },
        "/particles" => Value::Null,
        "/sound" => match name {
            "volume" => Value::number(100.0),
            "frequency" | "pan" => Value::number(0.0),
            _ => Value::Null,
        },
        _ => return None,
    };
    Some(value)
}

pub(crate) fn engine_builtin_initial_value(
    runtime_type: &TypePath,
    field: &FieldName,
) -> Option<Value> {
    engine_root_paths(runtime_type)
        .iter()
        .find_map(|owner| engine_owner_initial_value(owner, field))
}

pub(crate) fn engine_builtin_initial_fields(runtime_type: &TypePath) -> BTreeMap<FieldName, Value> {
    let mut fields = BTreeMap::new();
    for owner in engine_root_paths(runtime_type).iter().rev() {
        for name in engine_owner_field_names(owner) {
            let field = FieldName::parse(name).expect("engine field name is valid");
            if let Some(value) = engine_owner_initial_value(owner, &field) {
                fields.insert(field, value);
            }
        }
    }
    fields
}

pub(crate) fn engine_root_initial_value<'a>(
    state: &'a ExecutionState,
    runtime_type: &TypePath,
    field: &FieldName,
) -> Option<&'a Value> {
    engine_root_paths(runtime_type).iter().find_map(|root| {
        TypePath::parse(root)
            .ok()
            .and_then(|root| state.initial_values.get(&root))
            .and_then(|values| values.get(field))
    })
}

pub(crate) fn engine_root_initial_field_maps<'a>(
    state: &'a ExecutionState,
    runtime_type: &TypePath,
) -> impl DoubleEndedIterator<Item = &'a BTreeMap<FieldName, Value>> {
    engine_root_paths(runtime_type).iter().filter_map(|root| {
        TypePath::parse(root)
            .ok()
            .and_then(|root| state.initial_values.get(&root))
    })
}

pub(crate) fn initial_value_or_engine_root(
    state: &ExecutionState,
    runtime_type: &TypePath,
    field: &FieldName,
) -> Option<Value> {
    state.effective_initial_value(runtime_type, field)
}
