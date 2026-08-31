//! `qdel`/`del`/`qdel_value`/`unregister_runtime_datum`, `typecacheof`, image
//! construction, and `mutable_appearance` snapshotting shared with the native
//! value layer.

use dm_value::{DatumId, FieldName, TypePath, Value};

use super::{
    ExecutionState, builtin_contents_field, builtin_loc_field, icons::icon_backing_resource,
    synchronize_moved_atom_contents,
};
pub(super) fn qdel_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Ok(Value::Null);
    }
    for argument in arguments {
        let argument = state.heap.canonicalize_value(argument);
        qdel_value(&argument, state).map_err(|error| format!("qdel failed: {error}"))?;
    }
    Ok(Value::Null)
}

pub(super) fn del_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match &state.heap.canonicalize_value(&arguments[0]) {
        Value::Null => {}
        Value::Datum(datum) => {
            unregister_runtime_datum(state, *datum)?;
            state
                .heap_mut()
                .destroy_datum(*datum)
                .map_err(|error| format!("del failed: {error}"))?;
        }
        Value::List(list) => {
            state.associative_lists.remove(list);
            state
                .heap_mut()
                .destroy_list(*list)
                .map_err(|error| format!("del failed: {error}"))?;
        }
        value => return Err(format!("del cannot delete {value}")),
    }
    Ok(Value::Null)
}

pub(super) fn qdel_value(value: &Value, state: &mut ExecutionState) -> Result<(), String> {
    match &state.heap.canonicalize_value(value) {
        Value::Null => Ok(()),
        Value::Number(_)
        | Value::Text(_)
        | Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_) => Ok(()),
        Value::Datum(datum) => {
            unregister_runtime_datum(state, *datum)?;
            state
                .heap_mut()
                .destroy_datum(*datum)
                .map_err(|error| error.to_string())
                .map(|_| ())
        }
        Value::List(list) => {
            let entries = state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            for entry in entries {
                qdel_value(&entry, state)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn unregister_runtime_datum(
    state: &mut ExecutionState,
    datum: DatumId,
) -> Result<(), String> {
    let loc = builtin_loc_field();
    let old_loc = state
        .heap
        .datum_field(datum, loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(loc) => Some(*loc),
            _ => None,
        });
    synchronize_moved_atom_contents(state, datum, old_loc, None)?;

    let world = FieldName::parse("world").expect("built-in world global");
    let contents = builtin_contents_field();
    let world_contents = state
        .global(&world)
        .and_then(|value| match value {
            Value::Datum(world) => Some(*world),
            _ => None,
        })
        .and_then(|world| state.heap.datum_field(world, contents).ok())
        .and_then(|value| match value {
            Value::List(list) => Some(*list),
            _ => None,
        });
    if let Some(list) = world_contents {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(datum));
    }
    Ok(())
}

pub(super) fn typecacheof_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let target = arguments
        .first()
        .ok_or_else(|| "typecacheof requires a base type".to_owned())?;
    let raw_targets = match target {
        Value::List(list) => state
            .heap()
            .list(*list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>(),
        target => vec![target.clone()],
    };
    let targets = raw_targets
        .iter()
        .filter_map(|target| match target {
            // DM's typesof(null) contributes no paths. This matters for helper
            // lists which deliberately contain conditional/null entries.
            Value::Null => None,
            Value::TypePath(path) => Some(Ok(path.clone())),
            Value::ModifiedTypePath(path) => Some(Ok(path.base().clone())),
            Value::Text(text) => Some(
                TypePath::parse(text)
                    .map_err(|_| format!("typecacheof requires type paths, received {target}")),
            ),
            _ => Some(Err(format!(
                "typecacheof requires type paths, received {target}"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let paths = {
        let mut paths = std::collections::BTreeSet::new();
        for target in targets {
            paths.insert(target.clone());
            paths.extend(
                state
                    .type_paths()
                    .filter(|path| {
                        *path == &target
                            || path.as_str().starts_with(&format!("{}/", target.as_str()))
                    })
                    .cloned(),
            );
        }
        paths
    };

    let result = state.heap_mut().allocate_list();
    let list = state
        .heap_mut()
        .list_mut(result)
        .map_err(|error| error.to_string())?;

    for path in paths {
        let _ = list.set_key(Value::TypePath(path), Value::number(1.0));
    }
    Ok(Value::List(result))
}

pub(super) fn image_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let image_path = TypePath::parse("/image").expect("\"/image\" is a canonical BYOND type path");
    let image = state.heap_mut().allocate_datum(image_path.clone());
    state.seed_native_datum_defaults(image, &image_path)?;

    // DreamObjectImage.Initialize starts with a complete cloned appearance,
    // not an icon resource whose value happens to be the source object. This
    // distinction is observable in getFlatIcon(image(layer_image)): the new
    // image must expose layer_image.icon, icon_state, offsets, and nested
    // appearances while remaining independently mutable.
    for (name, value) in [
        ("alpha", Value::number(255.0)),
        ("appearance", Value::Null),
        ("appearance_flags", Value::number(0.0)),
        ("blend_mode", Value::number(0.0)),
        ("color", Value::Null),
        ("desc", Value::Null),
        ("dir", Value::number(2.0)),
        ("filters", Value::Null),
        ("glide_size", Value::number(0.0)),
        ("icon", Value::Null),
        ("icon_state", Value::Null),
        ("invisibility", Value::number(0.0)),
        ("layer", Value::number(0.0)),
        ("loc", Value::Null),
        ("maptext", Value::Null),
        ("maptext_height", Value::number(32.0)),
        ("maptext_width", Value::number(32.0)),
        ("maptext_x", Value::number(0.0)),
        ("maptext_y", Value::number(0.0)),
        ("mouse_drag_pointer", Value::Null),
        ("mouse_drop_pointer", Value::Null),
        ("mouse_drop_zone", Value::number(0.0)),
        ("mouse_opacity", Value::number(1.0)),
        ("mouse_over_pointer", Value::Null),
        ("name", Value::Null),
        ("opacity", Value::number(0.0)),
        ("overlays", Value::Null),
        ("plane", Value::number(0.0)),
        ("pixel_w", Value::number(0.0)),
        ("pixel_x", Value::number(0.0)),
        ("pixel_y", Value::number(0.0)),
        ("pixel_z", Value::number(0.0)),
        ("render_source", Value::Null),
        ("render_target", Value::Null),
        ("transform", Value::Null),
        ("underlays", Value::Null),
        ("vis_contents", Value::Null),
    ] {
        state
            .heap_mut()
            .set_datum_field(
                image,
                FieldName::parse(name).expect("image field name"),
                value,
            )
            .map_err(|error| error.to_string())?;
    }

    for name in ["overlays", "underlays", "vis_contents", "filters"] {
        let list = state.heap_mut().allocate_list();
        state
            .heap_mut()
            .set_datum_field(
                image,
                FieldName::parse(name).expect("image list field name"),
                Value::List(list),
            )
            .map_err(|error| error.to_string())?;
    }

    if let Some(source) = arguments.first() {
        copy_image_appearance(source, image, state)?;
    }

    if let Some(Value::Datum(location)) = arguments.get(1) {
        state
            .heap_mut()
            .set_datum_field(
                image,
                FieldName::parse("loc").expect("field name loc"),
                Value::Datum(*location),
            )
            .map_err(|error| error.to_string())?;
    }
    for (index, name) in ["icon_state", "layer", "dir", "pixel_x", "pixel_y"]
        .into_iter()
        .enumerate()
    {
        let Some(value) = arguments.get(index + 2) else {
            break;
        };
        // Optional nulls preserve the copied appearance. In particular,
        // image(existing_image) must not reset its icon state or layer.
        if matches!(value, Value::Null) {
            continue;
        }
        if name == "dir" && !value.as_number().is_some_and(|value| value > 0.0) {
            continue;
        }
        state
            .heap_mut()
            .set_datum_field(
                image,
                FieldName::parse(name).expect("image override field name"),
                value.clone(),
            )
            .map_err(|error| error.to_string())?;
    }

    Ok(Value::Datum(image))
}

const IMAGE_APPEARANCE_SCALARS: [&str; 31] = [
    "alpha",
    "appearance_flags",
    "blend_mode",
    "color",
    "desc",
    "dir",
    "glide_size",
    "icon",
    "icon_state",
    "invisibility",
    "layer",
    "maptext",
    "maptext_height",
    "maptext_width",
    "maptext_x",
    "maptext_y",
    "mouse_drag_pointer",
    "mouse_drop_pointer",
    "mouse_drop_zone",
    "mouse_opacity",
    "mouse_over_pointer",
    "name",
    "opacity",
    "plane",
    "pixel_w",
    "pixel_x",
    "pixel_y",
    "pixel_z",
    "render_source",
    "render_target",
    "transform",
];

pub(crate) fn is_appearance_source(path: &TypePath) -> bool {
    let path = path.as_str();
    path == "/image"
        || path.starts_with("/image/")
        || path == "/mutable_appearance"
        || path.starts_with("/mutable_appearance/")
        || ["/atom", "/area", "/turf", "/obj", "/mob"]
            .into_iter()
            .any(|root| path == root || path.starts_with(&format!("{root}/")))
}

pub(crate) fn copy_image_appearance(
    source: &Value,
    destination: DatumId,
    state: &mut ExecutionState,
) -> Result<(), String> {
    if let Value::Datum(icon) = source
        && state.heap().datum(*icon).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/icon" || path.starts_with("/icon/")
        })
    {
        let resource = icon_backing_resource(source, state, 0)?;
        state
            .heap_mut()
            .set_datum_field(
                destination,
                FieldName::parse("icon").expect("image icon field"),
                resource,
            )
            .map_err(|error| error.to_string())?;
        return Ok(());
    }
    let Value::Datum(source_datum) = source else {
        // A resource value creates a fresh appearance with that resource as
        // its icon. Invalid scalar values follow BYOND/OpenDream by producing
        // the default appearance instead of storing the scalar as `icon`.
        if matches!(source, Value::File(_)) {
            state
                .heap_mut()
                .set_datum_field(
                    destination,
                    FieldName::parse("icon").expect("image icon field"),
                    source.clone(),
                )
                .map_err(|error| error.to_string())?;
        }
        return Ok(());
    };
    let mut source = *source_datum;
    let source_path = state
        .heap()
        .datum(source)
        .map_err(|error| error.to_string())?
        .type_path()
        .clone();
    if !is_appearance_source(&source_path) {
        return Ok(());
    }

    // Dream64 currently represents BYOND's first-class `appearance` value as
    // an image-shaped datum. Honor it when present so image(atom) observes a
    // previously assigned complete appearance rather than the atom's stale
    // declaration fields.
    if let Ok(Value::Datum(appearance)) = state.heap().datum_field(
        source,
        &FieldName::parse("appearance").expect("appearance field"),
    ) && state
        .heap()
        .datum(*appearance)
        .is_ok_and(|datum| is_appearance_source(datum.type_path()))
    {
        source = *appearance;
    }

    let mut copied = Vec::new();
    for name in IMAGE_APPEARANCE_SCALARS {
        let field = FieldName::parse(name).expect("appearance scalar field");
        if let Ok(value) = super::datum_field_or_initial(state, source, &field) {
            copied.push((field, value));
        }
    }
    // MutableAppearance.GetCopy copies each visual collection into an
    // independent container while retaining the contained appearance/atom
    // identities. ValueHeap::copy_list has precisely those shallow-copy
    // semantics.
    for name in ["overlays", "underlays", "vis_contents", "filters"] {
        let field = FieldName::parse(name).expect("appearance list field");
        if let Ok(Value::List(list)) = super::datum_field_or_initial(state, source, &field) {
            let copy = state
                .heap_mut()
                .copy_list(list)
                .map_err(|error| error.to_string())?;
            copied.push((field, Value::List(copy)));
        }
    }
    for (field, value) in copied {
        state
            .heap_mut()
            .set_datum_field(destination, field, value)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn appearance_snapshot_builtin(
    source: DatumId,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let appearance = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mutable_appearance").expect("built-in appearance path"));
    copy_image_appearance(&Value::Datum(source), appearance, state)?;
    Ok(Value::Datum(appearance))
}
