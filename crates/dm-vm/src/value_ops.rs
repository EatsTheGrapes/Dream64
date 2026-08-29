//! Value-model semantics shared by the interpreter front doors and the
//! `execution` engine: canonical forms, comparison, truthiness, stack
//! micro-operations, scalar arithmetic, and the datum/list/vector/matrix
//! operators plus engine-root field access.

use smallvec::SmallVec;
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, OnceLock};

use crate::ExecutionState;
use crate::MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES;
use crate::MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE;
use crate::MAX_INSTANCE_INITIALIZER_PLAN_CACHE_ENTRIES;
use crate::RuntimeError;
use crate::SavefileState;
use crate::boot_trace_enabled;
use crate::builtins;
use crate::builtins::{execute_standard_builtin, is_subtype};
use crate::bytecode::{
    CompoundAssignmentOperator, InstanceInitializer, Module, ProcedureId, TypePredicateKind,
};
use crate::compile::EXPANDED_ARGUMENT_COUNT;
use crate::execute_module_in_context;
use dm_value::{DatumId, FieldName, ListId, TypePath, Value, ValueError, ValueHeap};

/// Entry-frame object context retained across a procedure call chain.
#[derive(Clone, Debug, PartialEq)]
pub struct ExecutionContext {
    pub(crate) src: Value,
    pub(crate) usr: Value,
}

impl ExecutionContext {
    /// Creates a context with explicit `src` and `usr` values.
    #[must_use]
    pub const fn new(src: Value, usr: Value) -> Self {
        Self { src, usr }
    }

    /// Returns the current source object.
    #[must_use]
    pub const fn src(&self) -> &Value {
        &self.src
    }

    /// Returns the current user object.
    #[must_use]
    pub const fn usr(&self) -> &Value {
        &self.usr
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            src: Value::Null,
            usr: Value::Null,
        }
    }
}

const DM_BIT_MASK: u32 = (1 << 24) - 1;

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn dm_u24(value: f32) -> u32 {
    (value.trunc() as i64 as u32) & DM_BIT_MASK
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn bitwise_binary(
    left: f32,
    right: f32,
    operation: impl FnOnce(u32, u32) -> u32,
) -> f32 {
    (operation(dm_u24(left), dm_u24(right)) & DM_BIT_MASK) as f32
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn bitwise_not(value: f32) -> f32 {
    ((!dm_u24(value)) & DM_BIT_MASK) as f32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
pub(crate) fn bitwise_shift(left: f32, right: f32, operation: impl FnOnce(u32, u32) -> u32) -> f32 {
    let count = right.trunc().max(0.0) as u32;
    if count >= 24 {
        return 0.0;
    }
    (operation(dm_u24(left), count) & DM_BIT_MASK) as f32
}

#[inline]
pub(crate) fn dm_list_length_number(length: usize) -> f32 {
    // A list's physical length is already a non-negative integer. Rust's
    // integer-to-binary32 conversion has the same correctly rounded result as
    // formatting that integer as decimal and parsing it back, without an
    // allocation and decimal conversion on every `.len` in map construction.
    length as f32
}

#[inline]
pub(crate) fn dm_list_resize_length(length: f32) -> usize {
    // The caller has already rejected negatives and non-finite values. Float
    // casts truncate and saturate at usize::MAX, matching the prior
    // `trunc().to_string().parse().unwrap_or(usize::MAX)` behavior.
    length.trunc() as usize
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn integer_remainder(left: f32, right: f32) -> f32 {
    let left = left.trunc() as i32;
    let right = right.trunc() as i32;
    if right == 0 {
        f32::NAN
    } else {
        (left % right) as f32
    }
}

pub(crate) fn fractional_remainder(left: f32, right: f32) -> f32 {
    if right == 0.0 {
        f32::NAN
    } else {
        right * (left / right).fract()
    }
}

pub(crate) fn compare_values(
    left: &Value,
    right: &Value,
) -> Result<Option<std::cmp::Ordering>, String> {
    match (left, right) {
        (Value::Text(left), Value::Text(right)) => Ok(Some(left.as_ref().cmp(right.as_ref()))),
        (Value::Null | Value::Number(_), Value::Null | Value::Number(_)) => {
            let left = match left {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            let right = match right {
                Value::Null => 0.0,
                Value::Number(number) => number.to_f32(),
                _ => unreachable!(),
            };
            Ok(left.partial_cmp(&right))
        }
        _ => Err(format!(
            "comparison requires two numbers or two text values, received {left} and {right}"
        )),
    }
}

pub(crate) fn values_equivalent(
    left: &Value,
    right: &Value,
    heap: &ValueHeap,
) -> Result<bool, String> {
    let left = heap.canonicalize_value(left);
    let right = heap.canonicalize_value(right);
    if let (Value::Datum(left), Value::Datum(right)) = (&left, &right)
        && is_matrix_datum(*left, heap)
        && is_matrix_datum(*right, heap)
    {
        return Ok(matrix_components(*left, heap)? == matrix_components(*right, heap)?);
    }
    let (Value::List(left_id), Value::List(right_id)) = (&left, &right) else {
        return Ok(left.semantic_eq(&right));
    };
    let left = heap.list(*left_id).map_err(|error| error.to_string())?;
    let right = heap.list(*right_id).map_err(|error| error.to_string())?;
    if left.len() != right.len() {
        return Ok(false);
    }
    for index in 1..=left.len() {
        let left_key = left.get(index).map_err(|error| error.to_string())?;
        let right_key = right.get(index).map_err(|error| error.to_string())?;
        if !values_equivalent(left_key, right_key, heap)? {
            return Ok(false);
        }
        let left_assoc = left.get_key(left_key).cloned().unwrap_or(Value::Null);
        let right_assoc = right.get_key(right_key).cloned().unwrap_or(Value::Null);
        if !values_equivalent(&left_assoc, &right_assoc, heap)? {
            return Ok(false);
        }
    }
    Ok(true)
}

pub(crate) const VECTOR_FIELDS: [&str; 3] = ["x", "y", "z"];

pub(crate) fn is_vector_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/vector")
}

pub(crate) fn is_icon_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/icon")
}

pub(crate) fn clone_icon_datum(icon: DatumId, heap: &mut ValueHeap) -> Result<DatumId, String> {
    if !is_icon_datum(icon, heap) {
        return Err("icon clone requires an /icon datum".to_owned());
    }
    let fields = heap
        .datum_fields(icon)
        .map_err(|error| error.to_string())?
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<Vec<_>>();
    let clone = heap.allocate_datum(TypePath::parse("/icon").expect("icon path is valid"));
    for (name, value) in fields {
        let value = match value {
            Value::List(list) => {
                Value::List(heap.copy_list(list).map_err(|error| error.to_string())?)
            }
            value => value,
        };
        heap.set_datum_field(clone, name, value)
            .map_err(|error| error.to_string())?;
    }
    Ok(clone)
}

pub(crate) fn icon_dimension(icon: DatumId, name: &str, heap: &ValueHeap) -> f32 {
    heap.datum_field(
        icon,
        &FieldName::parse(name).expect("internal icon dimension field is valid"),
    )
    .ok()
    .and_then(Value::as_number)
    .unwrap_or(32.0)
}

pub(crate) fn icon_number(
    argument: Option<&Value>,
    method: &str,
    name: &str,
) -> Result<f32, String> {
    argument
        .and_then(Value::as_number)
        .ok_or_else(|| format!("icon.{method} requires numeric {name}"))
}

pub(crate) fn record_icon_operation(
    icon: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    let operation = heap.allocate_list();
    {
        let values = heap
            .list_mut(operation)
            .map_err(|error| error.to_string())?;
        values.add(Value::text(method));
        for argument in arguments {
            values.add(argument.clone());
        }
    }
    let field = FieldName::parse("_dream64_icon_operations")
        .expect("internal icon operation field is valid");
    let operations = if let Ok(Value::List(operations)) = heap.datum_field(icon, &field) {
        *operations
    } else {
        let operations = heap.allocate_list();
        heap.set_datum_field(icon, field, Value::List(operations))
            .map_err(|error| error.to_string())?;
        operations
    };
    heap.list_mut(operations)
        .map_err(|error| error.to_string())?
        .add(Value::List(operation));
    Ok(())
}

pub(crate) fn execute_icon_method(
    icon: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let width_field = FieldName::parse("_dream64_width").expect("internal icon width is valid");
    let height_field = FieldName::parse("_dream64_height").expect("internal icon height is valid");
    match method {
        "Width" if arguments.is_empty() => {
            Ok(Value::number(icon_dimension(icon, "_dream64_width", heap)))
        }
        "Height" if arguments.is_empty() => {
            Ok(Value::number(icon_dimension(icon, "_dream64_height", heap)))
        }
        "Scale" if (1..=2).contains(&arguments.len()) => {
            let width = icon_number(arguments.first(), method, "width")?;
            let height = icon_number(arguments.get(1), method, "height").unwrap_or(width);
            heap.set_datum_field(icon, width_field, Value::number(width))
                .map_err(|error| error.to_string())?;
            heap.set_datum_field(icon, height_field, Value::number(height))
                .map_err(|error| error.to_string())?;
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        "Crop" if arguments.len() == 4 => {
            let x1 = icon_number(arguments.first(), method, "x1")?;
            let y1 = icon_number(arguments.get(1), method, "y1")?;
            let x2 = icon_number(arguments.get(2), method, "x2")?;
            let y2 = icon_number(arguments.get(3), method, "y2")?;
            heap.set_datum_field(icon, width_field, Value::number((x2 - x1).abs() + 1.0))
                .map_err(|error| error.to_string())?;
            heap.set_datum_field(icon, height_field, Value::number((y2 - y1).abs() + 1.0))
                .map_err(|error| error.to_string())?;
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        "Shift" | "DrawBox" | "Insert" | "Flip" | "SwapColor"
            if (method == "Shift" && (2..=3).contains(&arguments.len()))
                || (method == "DrawBox" && (1..=5).contains(&arguments.len()))
                || (method == "Insert" && (1..=6).contains(&arguments.len()))
                || (method == "Flip" && arguments.len() == 1)
                || (method == "SwapColor" && arguments.len() == 2) =>
        {
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        "Turn" if arguments.len() == 1 => {
            record_icon_operation(icon, method, arguments, heap)?;
            Ok(Value::Null)
        }
        // Pixel decoding is renderer/resource-provider work. BYOND yields null
        // for a transparent or unavailable pixel, which is the truthful
        // headless result while retaining every mutating operation above.
        "GetPixel" if (2..=5).contains(&arguments.len()) => Ok(Value::Null),
        _ => Err(format!(
            "icon.{method} received unsupported arguments ({})",
            arguments.len()
        )),
    }
}

pub(crate) fn apply_icon_map_colors(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !matches!(arguments.len(), 4 | 5 | 12 | 20) {
        return Err(format!(
            "icon.MapColors requires 4, 5, 12, or 20 arguments, received {}",
            arguments.len()
        ));
    }
    let matrix = heap.allocate_list();
    for value in arguments {
        heap.list_mut(matrix)
            .map_err(|error| error.to_string())?
            .add(value.clone());
    }
    heap.set_datum_field(
        icon,
        FieldName::parse("_dream64_color_matrix").expect("headless icon field is valid"),
        Value::List(matrix),
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

pub(crate) fn apply_icon_blend(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !(1..=4).contains(&arguments.len()) {
        return Err(format!(
            "icon.Blend requires an icon/color and up to mode, x, y; received {} arguments",
            arguments.len()
        ));
    }
    let history_field =
        FieldName::parse("_dream64_blends").expect("headless icon blend field is valid");
    let history = if let Ok(Value::List(history)) = heap.datum_field(icon, &history_field) {
        *history
    } else {
        let history = heap.allocate_list();
        heap.set_datum_field(icon, history_field, Value::List(history))
            .map_err(|error| error.to_string())?;
        history
    };
    let operation = heap.allocate_list();
    for value in [
        arguments[0].clone(),
        arguments.get(1).cloned().unwrap_or(Value::number(0.0)),
        arguments.get(2).cloned().unwrap_or(Value::number(1.0)),
        arguments.get(3).cloned().unwrap_or(Value::number(1.0)),
    ] {
        heap.list_mut(operation)
            .map_err(|error| error.to_string())?
            .add(value);
    }
    heap.list_mut(history)
        .map_err(|error| error.to_string())?
        .add(Value::List(operation));
    Ok(())
}

pub(crate) fn apply_icon_set_intensity(
    icon: DatumId,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    if !(1..=3).contains(&arguments.len()) {
        return Err(format!(
            "icon.SetIntensity requires r and optional g and b, received {} arguments",
            arguments.len()
        ));
    }
    let red = arguments[0]
        .as_number()
        .ok_or_else(|| "icon.SetIntensity red component must be numeric".to_owned())?;
    let green = arguments
        .get(1)
        .unwrap_or(&arguments[0])
        .as_number()
        .ok_or_else(|| "icon.SetIntensity green component must be numeric".to_owned())?;
    let blue = arguments
        .get(2)
        .unwrap_or(&arguments[0])
        .as_number()
        .ok_or_else(|| "icon.SetIntensity blue component must be numeric".to_owned())?;
    apply_icon_map_colors(
        icon,
        &[
            Value::number(red),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(green),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(blue),
            Value::number(0.0),
            Value::number(0.0),
            Value::number(0.0),
        ],
        heap,
    )
}

pub(crate) fn vector_components(datum: DatumId, heap: &ValueHeap) -> Result<[f32; 3], String> {
    if !is_vector_datum(datum, heap) {
        return Err("vector operation requires a /vector datum".to_owned());
    }
    let mut values = [0.0; 3];
    for (index, name) in VECTOR_FIELDS.iter().enumerate() {
        let field = FieldName::parse(name).expect("vector field is valid");
        values[index] = heap
            .datum_field(datum, &field)
            .map_err(|error| error.to_string())?
            .as_number()
            .unwrap_or(0.0);
    }
    Ok(values)
}

pub(crate) fn write_vector(
    datum: DatumId,
    values: [f32; 3],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    for (name, value) in VECTOR_FIELDS.into_iter().zip(values) {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("vector field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    let magnitude = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    for (name, value) in [("len", 3.0), ("size", magnitude)] {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("vector metadata field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn allocate_vector(values: [f32; 3], heap: &mut ValueHeap) -> Result<DatumId, String> {
    let datum = heap.allocate_datum(TypePath::parse("/vector").expect("vector path is valid"));
    write_vector(datum, values, heap)?;
    Ok(datum)
}

pub(crate) fn construct_vector(
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<DatumId, String> {
    if arguments.len() > 3 {
        return Err("vector accepts at most three arguments".to_owned());
    }
    let mut values = [0.0; 3];
    for (index, value) in arguments.iter().enumerate() {
        values[index] = value.as_number().unwrap_or(0.0);
    }
    allocate_vector(values, heap)
}

pub(crate) fn vector_zip(
    left: DatumId,
    right: DatumId,
    heap: &ValueHeap,
    operation: impl Fn(f32, f32) -> f32,
) -> Result<[f32; 3], String> {
    let left = vector_components(left, heap)?;
    let right = vector_components(right, heap)?;
    Ok(std::array::from_fn(|index| {
        operation(left[index], right[index])
    }))
}

pub(crate) fn execute_vector_binary(
    operator: &str,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = vector_components(datum, heap)?;
    let right_values = match right {
        Value::Datum(other) if is_vector_datum(*other, heap) => vector_components(*other, heap)?,
        value => [value.as_number().unwrap_or(0.0); 3],
    };
    let values = match operator {
        "*" => std::array::from_fn(|index| left[index] * right_values[index]),
        "/" => std::array::from_fn(|index| left[index] / right_values[index]),
        _ => return Err(format!("unsupported vector operator {operator}")),
    };
    Ok(Value::Datum(allocate_vector(values, heap)?))
}

pub(crate) fn execute_vector_compound(
    operator: CompoundAssignmentOperator,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = vector_components(datum, heap)?;
    let right_values = match right {
        Value::Datum(other) if is_vector_datum(*other, heap) => vector_components(*other, heap)?,
        value => [value.as_number().unwrap_or(0.0); 3],
    };
    let values = match operator {
        CompoundAssignmentOperator::Add => {
            std::array::from_fn(|index| left[index] + right_values[index])
        }
        CompoundAssignmentOperator::Subtract => {
            std::array::from_fn(|index| left[index] - right_values[index])
        }
        CompoundAssignmentOperator::Multiply => {
            std::array::from_fn(|index| left[index] * right_values[index])
        }
        CompoundAssignmentOperator::Divide => {
            std::array::from_fn(|index| left[index] / right_values[index])
        }
        _ => return Err("unsupported vector compound operator".to_owned()),
    };
    write_vector(datum, values, heap)?;
    Ok(Value::Datum(datum))
}

pub(crate) fn execute_vector_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let current = vector_components(datum, heap)?;
    match method.to_ascii_lowercase().as_str() {
        "dot" => {
            let Some(Value::Datum(other)) = arguments.first() else {
                return Err("vector.Dot requires a vector".to_owned());
            };
            let other = vector_components(*other, heap)?;
            Ok(Value::number(
                current.iter().zip(other).map(|(a, b)| a * b).sum::<f32>(),
            ))
        }
        "interpolate" => {
            let Some(Value::Datum(other)) = arguments.first() else {
                return Err("vector.Interpolate requires a vector and factor".to_owned());
            };
            let other = vector_components(*other, heap)?;
            let factor = arguments.get(1).and_then(Value::as_number).unwrap_or(0.0);
            let values = std::array::from_fn(|index| {
                current[index] + (other[index] - current[index]) * factor
            });
            Ok(Value::Datum(allocate_vector(values, heap)?))
        }
        "normalize" => {
            let magnitude = current
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let values = if magnitude == 0.0 {
                current
            } else {
                current.map(|value| value / magnitude)
            };
            write_vector(datum, values, heap)?;
            Ok(Value::Datum(datum))
        }
        _ => Err(format!("unknown /vector procedure {method:?}")),
    }
}

pub(crate) const MATRIX_FIELDS: [&str; 6] = ["a", "b", "c", "d", "e", "f"];

pub(crate) fn matrix_numeric(value: &Value) -> f32 {
    value.as_number().unwrap_or(0.0)
}

pub(crate) fn is_matrix_datum(datum: DatumId, heap: &ValueHeap) -> bool {
    heap.datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/matrix")
}

pub(crate) fn matrix_components(datum: DatumId, heap: &ValueHeap) -> Result<[f32; 6], String> {
    if !is_matrix_datum(datum, heap) {
        return Err("matrix operation requires a /matrix datum".to_owned());
    }
    let mut values = [0.0; 6];
    for (index, name) in MATRIX_FIELDS.iter().enumerate() {
        let field = FieldName::parse(name).expect("matrix field is valid");
        values[index] = matrix_numeric(heap.datum_field(datum, &field).map_err(|e| e.to_string())?);
    }
    Ok(values)
}

pub(crate) fn write_matrix(
    datum: DatumId,
    values: [f32; 6],
    heap: &mut ValueHeap,
) -> Result<(), String> {
    for (name, value) in MATRIX_FIELDS.into_iter().zip(values) {
        heap.set_datum_field(
            datum,
            FieldName::parse(name).expect("matrix field is valid"),
            Value::number(value),
        )
        .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn allocate_matrix(values: [f32; 6], heap: &mut ValueHeap) -> Result<DatumId, String> {
    let datum = heap.allocate_datum(TypePath::parse("/matrix").expect("matrix path is valid"));
    write_matrix(datum, values, heap)?;
    Ok(datum)
}

/// Applies constructor state owned by BYOND's engine types. These arguments
/// are not a project-defined `New()` call: contextual `var/icon/I = new(...)`
/// must retain the same resource fields as the `icon(...)` builtin.
pub(crate) fn initialize_engine_resource(
    state: &mut ExecutionState,
    datum: DatumId,
    type_path: &TypePath,
    arguments: &[Value],
) -> Result<(), String> {
    let path = type_path.as_str();
    if path == "/image"
        || path.starts_with("/image/")
        || path == "/mutable_appearance"
        || path.starts_with("/mutable_appearance/")
    {
        if let Some(source) = arguments.first() {
            // `/image` and `/mutable_appearance` are engine-backed appearance
            // objects. OpenDream's DreamObjectImage.Initialize copies the
            // complete source appearance before the DM `New()` proc runs.
            // This is observably different from storing the source datum in
            // `.icon`: Monk's decal smoothing constructs `pic = new(image)`
            // and later reuses `pic.icon` and `pic.icon_state`.
            builtins::copy_image_appearance(source, datum, state)?;
        }
        return Ok(());
    }

    let fields: &[&str] = match type_path.as_str() {
        "/icon" => &["icon", "icon_state", "dir", "frame", "moving"],
        _ => return Ok(()),
    };
    for (name, value) in fields.iter().zip(arguments) {
        state
            .heap
            .set_datum_field(
                datum,
                FieldName::parse(name).expect("engine resource field is valid"),
                value.clone(),
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(crate) fn allocate_or_replace_engine_datum(
    state: &mut ExecutionState,
    type_path: TypePath,
    arguments: &[Value],
) -> Result<DatumId, String> {
    let path = type_path.as_str();
    let is_turf = path == "/turf" || path.starts_with("/turf/");
    if is_turf
        && let Some(Value::Datum(existing)) = arguments.first()
        && state.heap.datum(*existing).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/turf" || path.starts_with("/turf/")
        })
    {
        initialize_existing_datum(state, *existing, type_path.clone(), true, true)?;
        initialize_engine_resource(state, *existing, &type_path, arguments)?;
        return Ok(*existing);
    }
    let datum = allocate_initialized_datum(state, type_path.clone())?;
    initialize_engine_resource(state, datum, &type_path, arguments)?;
    state
        .heap
        .compact_datum_layout(datum)
        .map_err(|error| error.to_string())?;
    Ok(datum)
}

pub(crate) fn runtime_initial_field_value(
    state: &mut ExecutionState,
    type_path: &TypePath,
    field: &FieldName,
) -> Result<Value, String> {
    if let Some(value) = state
        .initial_field_value_cache
        .get(type_path)
        .and_then(|fields| fields.get(field))
    {
        return Ok(value.clone());
    }
    let catalog_value = state
        .effective_initial_value(type_path, field)
        .unwrap_or(Value::Null);
    if !matches!(catalog_value, Value::Null) {
        cache_runtime_initial_field_value(state, type_path, field, &catalog_value);
        return Ok(catalog_value);
    }

    let mut current = Some(type_path.clone());
    let mut has_runtime_default = false;
    while let Some(path) = current {
        has_runtime_default |= state
            .instance_initializers
            .get(&path)
            .is_some_and(|initializers| {
                initializers.iter().any(|initializer| match initializer {
                    InstanceInitializer::Constant {
                        field: candidate, ..
                    }
                    | InstanceInitializer::Program {
                        field: candidate, ..
                    } => candidate == field,
                })
            });
        current = state.type_parent(&path).cloned();
    }
    if !has_runtime_default {
        cache_runtime_initial_field_value(state, type_path, field, &Value::Null);
        return Ok(Value::Null);
    }

    let prototype = if let Some(prototype) = state.initial_prototypes.get(type_path).copied() {
        prototype
    } else {
        // Publish the identity before running initializer programs so a
        // self-referential type-scope read terminates against the partially
        // initialized prototype, just as an object-tree prototype does.
        let prototype = state.heap.allocate_datum(type_path.clone());
        state
            .initial_prototypes
            .insert(type_path.clone(), prototype);
        state
            .initial_prototypes_initializing
            .insert(type_path.clone());
        let initialization =
            initialize_existing_datum(state, prototype, type_path.clone(), false, false);
        state.initial_prototypes_initializing.remove(type_path);
        initialization?;
        // This datum is the engine's hidden object-tree prototype used to
        // evaluate runtime instance defaults for type-scoped reads. It is not
        // an instantiated atom and must never become visible through `world`
        // or `world.contents`. Ordinary atom initialization registers every
        // atom, so undo that registration for this synthetic identity.
        if is_atom_type_path(type_path) {
            let contents = FieldName::parse("contents").expect("built-in contents field");
            let world_contents = state
                .global(&FieldName::parse("world").expect("built-in world global"))
                .and_then(|value| match value {
                    Value::Datum(world) => state.heap.datum_field(*world, &contents).ok(),
                    _ => None,
                })
                .and_then(|value| match value {
                    Value::List(list) => Some(*list),
                    _ => None,
                });
            if let Some(list) = world_contents {
                state
                    .heap
                    .list_mut(list)
                    .map_err(|error| error.to_string())?
                    .remove_first(&Value::Datum(prototype));
            }
        }
        prototype
    };
    let value = state
        .heap
        .datum_field(prototype, field)
        .cloned()
        .unwrap_or(Value::Null);
    if !state.initial_prototypes_initializing.contains(type_path) {
        cache_runtime_initial_field_value(state, type_path, field, &value);
    }
    Ok(value)
}

pub(crate) fn cache_runtime_initial_field_value(
    state: &mut ExecutionState,
    type_path: &TypePath,
    field: &FieldName,
    value: &Value,
) {
    if state.initial_field_value_cache_entries >= MAX_EFFECTIVE_INITIAL_VALUE_CACHE_ENTRIES {
        return;
    }
    let fields = state
        .initial_field_value_cache
        .entry(type_path.clone())
        .or_default();
    if fields.len() < MAX_EFFECTIVE_INITIAL_VALUE_CACHE_FIELDS_PER_TYPE
        && fields.insert(field.clone(), value.clone()).is_none()
    {
        state.initial_field_value_cache_entries += 1;
    }
}

pub(crate) fn allocate_initialized_datum(
    state: &mut ExecutionState,
    type_path: TypePath,
) -> Result<DatumId, String> {
    let datum = state.heap.allocate_datum(type_path.clone());
    initialize_existing_datum(state, datum, type_path, false, true)?;
    Ok(datum)
}

pub(crate) fn instance_initializer_plan(
    state: &mut ExecutionState,
    type_path: &TypePath,
) -> Arc<[InstanceInitializer]> {
    if let Some(plan) = state.instance_initializer_plans.get(type_path) {
        return Arc::clone(plan);
    }
    let mut hierarchy = Vec::new();
    let mut current = Some(type_path.clone());
    while let Some(path) = current {
        hierarchy.push(path.clone());
        current = state.type_parent(&path).cloned();
    }
    hierarchy.reverse();
    let plan = hierarchy
        .into_iter()
        .flat_map(|path| {
            state
                .instance_initializers
                .get(&path)
                .into_iter()
                .flatten()
                .cloned()
        })
        .collect::<Arc<[_]>>();
    if state.instance_initializer_plans.len() < MAX_INSTANCE_INITIALIZER_PLAN_CACHE_ENTRIES {
        state
            .instance_initializer_plans
            .insert(type_path.clone(), Arc::clone(&plan));
    }
    plan
}

pub(crate) fn initialize_existing_datum(
    state: &mut ExecutionState,
    datum: DatumId,
    type_path: TypePath,
    preserve_cell: bool,
    compact_defaults: bool,
) -> Result<(), String> {
    if compact_defaults && preserve_cell {
        state.compact_default_datums.insert(datum);
    } else {
        state.compact_default_datums.remove(&datum);
    }
    // `/regex`, `/icon`, and the other engine-owned datum roots do not need
    // to appear in the project's object tree. They still inherit BYOND's
    // built-in `/datum` storage. Seed that storage at the allocation boundary
    // so both project types and engine roots expose it before declaration
    // defaults and `New` run.
    if !compact_defaults {
        for (name, value) in [("datum_flags", Value::number(0.0)), ("tag", Value::Null)] {
            let field = FieldName::parse(name).expect("built-in datum field is valid");
            if state.heap.datum_field(datum, &field).is_err() {
                state
                    .heap
                    .set_datum_field(datum, field, value)
                    .map_err(|error| error.to_string())?;
            }
        }
    }
    let initial_values = if compact_defaults {
        BTreeMap::new()
    } else {
        state.inherited_initial_values(&type_path)
    };
    let is_atom = is_atom_type_path(&type_path);
    if preserve_cell {
        let fields = state
            .heap
            .datum_fields(datum)
            .map_err(|error| error.to_string())?
            .map(|(name, _)| name.clone())
            .filter(|name| !is_map_cell_structural_field(name))
            .collect::<Vec<_>>();
        for field in fields {
            state
                .heap
                .delete_datum_field(datum, &field)
                .map_err(|error| error.to_string())?;
        }
        state
            .heap
            .set_datum_type_path(datum, type_path.clone())
            .map_err(|error| error.to_string())?;
    }
    for (name, value) in initial_values {
        if preserve_cell && is_map_cell_structural_field(&name) {
            continue;
        }
        state
            .heap
            .set_datum_field(datum, name, value)
            .map_err(|error| error.to_string())?;
    }
    if type_path.as_str() == "/client" || type_path.as_str().starts_with("/client/") {
        for name in ["images", "screen", "verbs"] {
            let field = FieldName::parse(name).expect("built-in client list field is valid");
            if !matches!(state.heap.datum_field(datum, &field), Ok(Value::List(_))) {
                let list = state.heap.allocate_list();
                state
                    .heap
                    .set_datum_field(datum, field, Value::List(list))
                    .map_err(|error| error.to_string())?;
            }
        }
    }
    let plans = instance_initializer_plan(state, &type_path);
    let initializer_module = state.instance_initializer_module.clone();
    for initializer in plans.iter().cloned() {
        let (field, value) = match initializer {
            InstanceInitializer::Constant { field, value } => {
                // Retyping a live map cell must retain its engine-owned spatial
                // identity. These slots predate initializer execution, so their
                // mere presence cannot be used as evidence that an earlier
                // initializer wrote them.
                if preserve_cell && is_map_cell_structural_field(&field) {
                    continue;
                }
                // Compact engine turfs read untouched scalar defaults from the
                // shared type catalog. A later constant action only needs an
                // owned slot when an earlier runtime initializer actually
                // wrote that field (including through a side effect).
                if compact_defaults && state.heap.datum_field(datum, &field).is_err() {
                    continue;
                }
                (field, value)
            }
            InstanceInitializer::Program { field, entry } => {
                let module = initializer_module
                    .as_ref()
                    .ok_or_else(|| "runtime instance initializer module is absent".to_owned())?;
                let value = execute_module_in_context(
                    module,
                    entry,
                    &[],
                    state,
                    &ExecutionContext::new(Value::Datum(datum), Value::Null),
                )
                .map_err(|error| error.to_string())?;
                (field, value)
            }
        };
        state
            .heap
            .set_datum_field(datum, field, value)
            .map_err(|error| error.to_string())?;
    }
    if is_atom {
        let contents = FieldName::parse("contents").expect("built-in contents field");
        if preserve_cell {
            state
                .heap
                .compact_datum_layout(datum)
                .map_err(|error| error.to_string())?;
            return Ok(());
        }
        let world = FieldName::parse("world").expect("built-in world global");
        let world_contents = state
            .global(&world)
            .and_then(|value| match value {
                Value::Datum(world) => Some(*world),
                _ => None,
            })
            .and_then(|world| state.heap.datum_field(world, &contents).ok())
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            });
        if let Some(list) = world_contents {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::Datum(datum));
        }
    }
    state
        .heap
        .compact_datum_layout(datum)
        .map_err(|error| error.to_string())?;
    Ok(())
}

pub(crate) fn is_atom_type_path(path: &TypePath) -> bool {
    let path = path.as_str();
    ["/atom", "/area", "/turf", "/obj", "/mob"]
        .into_iter()
        .any(|root| {
            path == root
                || path
                    .strip_prefix(root)
                    .is_some_and(|rest| rest.starts_with('/'))
        })
}

pub(crate) fn is_appearance_type_path(path: &TypePath) -> bool {
    let path = path.as_str();
    path == "/image"
        || path.starts_with("/image/")
        || path == "/mutable_appearance"
        || path.starts_with("/mutable_appearance/")
}

pub(crate) fn is_turf_type_path(path: &TypePath) -> bool {
    path.as_str() == "/turf" || path.as_str().starts_with("/turf/")
}

pub(crate) fn is_area_type_path(path: &TypePath) -> bool {
    path.as_str() == "/area" || path.as_str().starts_with("/area/")
}

/// BYOND exposes an area's spatial coordinates through one of its contained
/// turfs. Areas do not own a single map cell, so the indexed world mapping is
/// the authoritative source when no explicit coordinate override exists.
pub(crate) fn area_coordinate_field(
    state: &ExecutionState,
    area: DatumId,
    field: &FieldName,
) -> Option<Value> {
    let coordinate = state
        .world_areas
        .iter()
        .find_map(|(coordinate, candidate)| (*candidate == area).then_some(*coordinate))?;
    let component = match field.as_str() {
        "x" => coordinate.0,
        "y" => coordinate.1,
        "z" => coordinate.2,
        _ => return None,
    };
    Some(Value::number(component as f32))
}

pub(crate) fn is_map_cell_structural_field(field: &FieldName) -> bool {
    matches!(field.as_str(), "x" | "y" | "z" | "loc" | "contents")
}

/// Returns whether a datum field needs type-specific engine behavior instead
/// of the sparse instance/default/shared lookup used by ordinary DM members.
pub(crate) fn datum_field_requires_special_read(
    runtime_type: &TypePath,
    field: &FieldName,
) -> bool {
    let path = runtime_type.as_str();
    (path == "/savefile" || path.starts_with("/savefile/"))
        || matches!(
            field.as_str(),
            "type"
                | "parent_type"
                | "appearance"
                | "transform"
                | "x"
                | "y"
                | "z"
                | "contents"
                | "filters"
                | "overlays"
                | "underlays"
                | "verbs"
                | "vis_contents"
                | "vis_locs"
                | "locs"
        )
}

pub(crate) fn lazy_atom_list_field(
    state: &mut ExecutionState,
    datum: DatumId,
    field: &FieldName,
) -> Result<Option<Value>, String> {
    enum ListKind {
        Contents,
        Locs,
        Ordinary,
        VisContents,
        VisLocs,
    }
    let kind = match field.as_str() {
        "contents" => ListKind::Contents,
        "locs" => ListKind::Locs,
        "filters" | "overlays" | "underlays" | "verbs" => ListKind::Ordinary,
        "vis_contents" => ListKind::VisContents,
        "vis_locs" => ListKind::VisLocs,
        _ => return Ok(None),
    };
    let runtime_type = state
        .heap
        .datum(datum)
        .map_err(|error| error.to_string())?
        .type_path();
    let appearance_list = is_appearance_type_path(runtime_type)
        && matches!(field.as_str(), "filters" | "overlays" | "underlays");
    if !is_atom_type_path(runtime_type) && !appearance_list {
        return Ok(None);
    }
    let list = match kind {
        ListKind::Contents => state.ensure_contents(datum)?,
        ListKind::Locs => return movable_locs(state, datum).map(Some),
        ListKind::VisContents => state.ensure_visibility_list(datum, true)?,
        ListKind::VisLocs => state.ensure_visibility_list(datum, false)?,
        ListKind::Ordinary => {
            if let Ok(Value::List(list)) = state.heap.datum_field(datum, field) {
                *list
            } else {
                let list = state.heap.allocate_list();
                state
                    .heap
                    .set_datum_field(datum, field.clone(), Value::List(list))
                    .map_err(|error| error.to_string())?;
                list
            }
        }
    };
    Ok(Some(Value::List(list)))
}

/// Materializes BYOND's read-only `atom/movable.locs` view.
///
/// `locs` contains every turf overlapped by the movable's pixel bounds. Most
/// movables occupy only `loc`; multi-tile machinery expands the view through
/// `bound_*`. A fresh engine view avoids retaining one heap list on every
/// movable that base game initialization happens to inspect.
pub(crate) fn movable_locs(state: &mut ExecutionState, datum: DatumId) -> Result<Value, String> {
    let runtime_type = state
        .heap
        .datum(datum)
        .map_err(|error| error.to_string())?
        .type_path()
        .clone();
    if !builtins::is_subtype(
        state,
        &runtime_type,
        &TypePath::parse("/atom/movable").expect("built-in movable path is valid"),
    ) {
        return Ok(Value::Null);
    }

    let list = state.heap.allocate_list();
    let loc = datum_field_or_initial(
        state,
        datum,
        &FieldName::parse("loc").expect("built-in loc field is valid"),
    )
    .unwrap_or(Value::Null);
    let Value::Datum(loc_datum) = loc else {
        return Ok(Value::List(list));
    };
    let loc_type = state
        .heap
        .datum(loc_datum)
        .map_err(|error| error.to_string())?
        .type_path();
    if !is_turf_type_path(loc_type) {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(loc_datum));
        return Ok(Value::List(list));
    }

    let scalar = |state: &ExecutionState, name: &str, fallback: f32| {
        datum_field_or_initial(
            state,
            datum,
            &FieldName::parse(name).expect("built-in movable bounds field is valid"),
        )
        .ok()
        .and_then(|value| value.as_number())
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
    };
    let bound_x = scalar(state, "bound_x", 0.0);
    let bound_y = scalar(state, "bound_y", 0.0);
    let bound_width = scalar(state, "bound_width", 32.0).max(1.0);
    let bound_height = scalar(state, "bound_height", 32.0).max(1.0);
    let icon_size = 32.0;
    let min_dx = (bound_x / icon_size).floor() as i32;
    let min_dy = (bound_y / icon_size).floor() as i32;
    let max_dx = ((bound_x + bound_width - 1.0) / icon_size).floor() as i32;
    let max_dy = ((bound_y + bound_height - 1.0) / icon_size).floor() as i32;

    let Some((base_x, base_y, base_z)) =
        builtins::datum_coordinates(state, &Value::Datum(loc_datum))
    else {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(loc_datum));
        return Ok(Value::List(list));
    };
    let (base_x, base_y, base_z) = (base_x as i32, base_y as i32, base_z as i32);
    let mut locations = Vec::new();
    for dx in min_dx..=max_dx {
        for dy in min_dy..=max_dy {
            if let Some(turf) = state.turf_at(base_x + dx, base_y + dy, base_z) {
                locations.push(turf);
            }
        }
    }
    if locations.is_empty() {
        locations.push(loc_datum);
    }
    let values = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for location in locations {
        values.add(Value::Datum(location));
    }
    Ok(Value::List(list))
}

pub(crate) fn world_contents_iteration_snapshot(
    state: &mut ExecutionState,
    contents: ListId,
) -> Result<ListId, String> {
    let values = state
        .heap
        .list(contents)
        .map_err(|error| error.to_string())?
        .positions()
        .map(|(_, value)| value.clone())
        .collect::<Vec<_>>();
    let mob = TypePath::parse("/mob").expect("built-in mob path is valid");
    let movable = TypePath::parse("/atom/movable").expect("built-in movable path is valid");
    let area = TypePath::parse("/area").expect("built-in area path is valid");
    let turf = TypePath::parse("/turf").expect("built-in turf path is valid");
    // World iteration has a fixed category order but remains stable inside
    // each category. A stable sort recalculated several parent walks for each
    // comparison (O(N log N)) over the complete loaded world just before
    // SSatoms. Classify each value exactly once instead.
    let mut buckets: [Vec<Value>; 5] = std::array::from_fn(|_| Vec::new());
    for value in values {
        let Value::Datum(datum) = &value else {
            buckets[4].push(value);
            continue;
        };
        let Ok(datum) = state.heap.datum(*datum) else {
            buckets[4].push(value);
            continue;
        };
        let path = datum.type_path();
        let category = if is_subtype(state, path, &mob) {
            0
        } else if is_subtype(state, path, &movable) {
            1
        } else if is_subtype(state, path, &area) {
            2
        } else if is_subtype(state, path, &turf) {
            3
        } else {
            4
        };
        buckets[category].push(value);
    }

    let snapshot = state.heap.allocate_list();
    for value in buckets.into_iter().flatten() {
        state
            .heap
            .list_mut(snapshot)
            .expect("newly allocated world iteration snapshot is live")
            .add(value);
    }
    Ok(snapshot)
}

pub(crate) fn atom_contents_iteration_snapshot(
    state: &mut ExecutionState,
    owner: DatumId,
    contents: ListId,
) -> Result<ListId, String> {
    let loc = FieldName::parse("loc").expect("built-in loc field is valid");
    let values = state
        .heap
        .list(contents)
        .map_err(|error| error.to_string())?
        .positions()
        .filter_map(|(_, value)| {
            let Value::Datum(member) = value else {
                return None;
            };
            let actual_owner = state
                .heap
                .datum_field(*member, &loc)
                .ok()
                .and_then(|value| match value {
                    Value::Datum(location) => Some(*location),
                    _ => None,
                });
            (actual_owner == Some(owner)).then(|| value.clone())
        })
        .collect::<Vec<_>>();
    let snapshot = state.heap.allocate_list();
    for value in values {
        state
            .heap
            .list_mut(snapshot)
            .expect("newly allocated atom contents snapshot is live")
            .add(value);
    }
    Ok(snapshot)
}

pub(crate) fn construct_matrix(
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<DatumId, String> {
    match arguments {
        [] | [Value::Null] => {
            return allocate_matrix([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], heap);
        }
        [Value::Datum(source)] if is_matrix_datum(*source, heap) => {
            return allocate_matrix(matrix_components(*source, heap)?, heap);
        }
        [a, b, c, d, e, f] => {
            return allocate_matrix([a, b, c, d, e, f].map(matrix_numeric), heap);
        }
        _ => {}
    }
    let mode_value = arguments
        .last()
        .and_then(Value::as_number)
        .ok_or_else(|| "matrix operation mode must be numeric".to_owned())?
        as i32;
    let mode = mode_value & 127;
    let modify = mode_value & 128 != 0;
    let source = arguments.first().and_then(|value| match value {
        Value::Datum(datum) if is_matrix_datum(*datum, heap) => Some(*datum),
        _ => None,
    });
    let mut values = source.map_or(Ok([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), |datum| {
        matrix_components(datum, heap)
    })?;
    match mode {
        0 => {
            let source = source.ok_or_else(|| "MATRIX_COPY requires a matrix".to_owned())?;
            values = matrix_components(source, heap)?;
        }
        4 => {
            let determinant = values[0] * values[4] - values[1] * values[3];
            if determinant == 0.0 {
                return Err("cannot invert a singular matrix".to_owned());
            }
            values = [
                values[4] / determinant,
                -values[1] / determinant,
                (values[1] * values[5] - values[4] * values[2]) / determinant,
                -values[3] / determinant,
                values[0] / determinant,
                (values[3] * values[2] - values[0] * values[5]) / determinant,
            ];
        }
        5 => {
            let offset = usize::from(source.is_some());
            let radians = matrix_numeric(&arguments[offset]).to_radians();
            let mut cosine = radians.cos();
            let mut sine = radians.sin();
            if cosine.abs() < 1.0e-6 {
                cosine = 0.0;
            }
            if sine.abs() < 1.0e-6 {
                sine = 0.0;
            }
            values = matrix_product(values, [cosine, sine, 0.0, -sine, cosine, 0.0]);
        }
        6 => {
            let offset = usize::from(source.is_some());
            let x = matrix_numeric(&arguments[offset]);
            let y = if arguments.len() - offset >= 3 {
                matrix_numeric(&arguments[offset + 1])
            } else {
                x
            };
            values = matrix_product(values, [x, 0.0, 0.0, 0.0, y, 0.0]);
        }
        7 => {
            let offset = usize::from(source.is_some());
            let x = matrix_numeric(&arguments[offset]);
            let y = if arguments.len() - offset >= 3 {
                matrix_numeric(&arguments[offset + 1])
            } else {
                x
            };
            values[2] += x;
            values[5] += y;
        }
        _ => return Err(format!("unknown matrix operation mode {mode}")),
    }
    if modify {
        let datum = source.ok_or_else(|| "MATRIX_MODIFY requires a matrix".to_owned())?;
        write_matrix(datum, values, heap)?;
        Ok(datum)
    } else {
        allocate_matrix(values, heap)
    }
}

pub(crate) fn execute_matrix_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let current = matrix_components(datum, heap)?;
    let updated = match method.to_ascii_lowercase().as_str() {
        "add" | "subtract" => {
            let Value::Datum(other) = arguments.first().unwrap_or(&Value::Null) else {
                return Err(format!("matrix.{method} requires a matrix"));
            };
            let other = matrix_components(*other, heap)?;
            let sign = if method.eq_ignore_ascii_case("add") {
                1.0
            } else {
                -1.0
            };
            std::array::from_fn(|index| current[index] + sign * other[index])
        }
        "multiply" => match arguments.first().unwrap_or(&Value::Null) {
            Value::Null => current,
            Value::Datum(other) if is_matrix_datum(*other, heap) => {
                matrix_product(current, matrix_components(*other, heap)?)
            }
            value => current.map(|component| component * matrix_numeric(value)),
        },
        "scale" => {
            let x = arguments.first().map_or(0.0, matrix_numeric);
            let y = arguments.get(1).map_or(x, matrix_numeric);
            [
                current[0] * x,
                current[1] * x,
                current[2] * x,
                current[3] * y,
                current[4] * y,
                current[5] * y,
            ]
        }
        "translate" => {
            let Some(x) = arguments.first().and_then(Value::as_number) else {
                return Ok(Value::Datum(datum));
            };
            let y = arguments.get(1).and_then(Value::as_number).unwrap_or(x);
            [
                current[0],
                current[1],
                current[2] + x,
                current[3],
                current[4],
                current[5] + y,
            ]
        }
        "turn" => {
            let degrees = arguments.first().map_or(0.0, matrix_numeric).to_radians();
            let mut cosine = degrees.cos();
            let mut sine = degrees.sin();
            if cosine.abs() < 1.0e-6 {
                cosine = 0.0;
            }
            if sine.abs() < 1.0e-6 {
                sine = 0.0;
            }
            let rotation = [cosine, sine, 0.0, -sine, cosine, 0.0];
            matrix_product(current, rotation)
        }
        "invert" => {
            let determinant = current[0] * current[4] - current[1] * current[3];
            if determinant == 0.0 {
                return Err("cannot invert a singular matrix".to_owned());
            }
            [
                current[4] / determinant,
                -current[1] / determinant,
                (current[1] * current[5] - current[4] * current[2]) / determinant,
                -current[3] / determinant,
                current[0] / determinant,
                (current[3] * current[2] - current[0] * current[5]) / determinant,
            ]
        }
        _ => return Err(format!("unknown /matrix procedure {method:?}")),
    };
    write_matrix(datum, updated, heap)?;
    Ok(Value::Datum(datum))
}

pub(crate) fn matrix_product(left: [f32; 6], right: [f32; 6]) -> [f32; 6] {
    [
        left[0] * right[0] + left[3] * right[1],
        left[1] * right[0] + left[4] * right[1],
        left[2] * right[0] + left[5] * right[1] + right[2],
        left[0] * right[3] + left[3] * right[4],
        left[1] * right[3] + left[4] * right[4],
        left[2] * right[3] + left[5] * right[4] + right[5],
    ]
}

pub(crate) fn execute_matrix_binary(
    operator: &str,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = matrix_components(datum, heap)?;
    let result = match operator {
        "*" => match right {
            Value::Datum(other) if is_matrix_datum(*other, heap) => {
                matrix_product(left, matrix_components(*other, heap)?)
            }
            value => left.map(|component| component * matrix_numeric(value)),
        },
        "/" => {
            let divisor = matrix_numeric(right);
            left.map(|component| component / divisor)
        }
        _ => return Err("unsupported binary matrix operator".to_owned()),
    };
    allocate_matrix(result, heap).map(Value::Datum)
}

pub(crate) fn execute_matrix_compound(
    operator: CompoundAssignmentOperator,
    datum: DatumId,
    right: &Value,
    heap: &mut ValueHeap,
) -> Result<Value, String> {
    let left = matrix_components(datum, heap)?;
    let updated = match operator {
        CompoundAssignmentOperator::Add | CompoundAssignmentOperator::Subtract => {
            let Value::Datum(other) = right else {
                return Err("matrix addition/subtraction requires another matrix".to_owned());
            };
            let other = matrix_components(*other, heap)?;
            let sign = if matches!(operator, CompoundAssignmentOperator::Add) {
                1.0
            } else {
                -1.0
            };
            std::array::from_fn(|index| left[index] + sign * other[index])
        }
        CompoundAssignmentOperator::Multiply => match right {
            Value::Datum(other) if is_matrix_datum(*other, heap) => {
                matrix_product(left, matrix_components(*other, heap)?)
            }
            value => left.map(|component| component * matrix_numeric(value)),
        },
        CompoundAssignmentOperator::Divide => {
            let divisor = matrix_numeric(right);
            left.map(|component| component / divisor)
        }
        _ => return Err("unsupported compound matrix operator".to_owned()),
    };
    write_matrix(datum, updated, heap)?;
    Ok(Value::Datum(datum))
}

pub(crate) fn validate_jump(target: usize, instruction_count: usize) -> Result<(), String> {
    if target > instruction_count {
        return Err(format!("invalid jump target {target}"));
    }
    Ok(())
}

pub(crate) fn values_equal(heap: &ValueHeap, left: &Value, right: &Value) -> bool {
    let left = heap.canonicalize_value(left);
    let right = heap.canonicalize_value(right);
    left.semantic_eq(&right)
}

pub(crate) fn canonicalize_value(heap: &ValueHeap, value: &Value) -> Value {
    heap.canonicalize_value(value)
}

pub(crate) fn canonicalize_owned_value(heap: &ValueHeap, value: Value) -> Value {
    match value {
        Value::Datum(datum) if heap.datum(datum).is_err() => Value::Null,
        Value::List(list) if heap.list(list).is_err() => Value::Null,
        value => value,
    }
}

pub(crate) fn replace_text_builtin(
    arguments: &[Value],
    exact: bool,
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<Value, String> {
    // BYOND's haystack and needle parameters are text-typed: a non-text
    // haystack returns null, while a non-text needle is the empty string.
    // Replacement is deliberately different and uses normal DM stringification
    // (for example the numeric constant 90 becomes "90").
    let Value::Text(source) = &arguments[0] else {
        return Ok(Value::Null);
    };
    let source = source.to_string();
    let needle = match &arguments[1] {
        Value::Text(text) => text.to_string(),
        _ => String::new(),
    };
    let replacement = stringify_dm_value(&arguments[2], heap)?;
    if needle.is_empty() {
        if replacement.is_empty() {
            return Ok(Value::text(source));
        }
        return replace_empty_needle(&source, &replacement, arguments, character_indices)
            .map(Value::text);
    }

    let (start, end) = replacement_bounds(&source, arguments, character_indices)?;
    let prefix = &source[..start];
    let target = &source[start..end];
    let suffix = &source[end..];
    let replaced = if exact {
        target.replace(&needle, &replacement)
    } else {
        replace_text_ascii_insensitive(target, &needle, &replacement)
    };
    Ok(Value::text(format!("{prefix}{replaced}{suffix}")))
}

pub(crate) fn stringify_dm_value(value: &Value, heap: &ValueHeap) -> Result<String, String> {
    match value {
        Value::Null => Ok(String::new()),
        Value::Text(text) | Value::File(text) => Ok(text.to_string()),
        Value::Number(number) => Ok(number.to_f32().to_string()),
        Value::TypePath(path) => Ok(path.to_string()),
        Value::ModifiedTypePath(path) => Ok(path.base().to_string()),
        Value::Datum(datum) => {
            let datum = heap.datum(*datum).map_err(|error| error.to_string())?;
            let name = FieldName::parse("name").expect("built-in datum name is valid");
            if let Ok(Value::Text(name)) = datum.field(&name) {
                Ok(name.to_string())
            } else {
                Ok(datum.type_path().to_string())
            }
        }
        Value::List(_) => Ok("/list".to_owned()),
    }
}

pub(crate) fn replace_empty_needle(
    source: &str,
    replacement: &str,
    arguments: &[Value],
    character_indices: bool,
) -> Result<String, String> {
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let mut start = signed_text_index(arguments.get(3), 1)?;
    if start == 0 {
        return Ok(source.to_owned());
    }
    if start < 0 {
        start = limit.saturating_add(start).max(1);
    }
    let mut end = signed_text_index(arguments.get(4), 0)?;
    if end <= 0 {
        end = limit.saturating_add(end).max(start);
    }
    let mut start = usize::try_from(start.clamp(1, limit)).unwrap_or(usize::MAX);
    let mut end = usize::try_from(end.clamp(1, limit)).unwrap_or(usize::MAX);
    if start == 1 {
        start = 2;
    }
    end = end.min(logical_length);

    let mut output = String::with_capacity(source.len().saturating_add(replacement.len()));
    for (zero_based, character) in source.chars().enumerate() {
        output.push(character);
        let position = zero_based.saturating_add(1);
        if position >= start.saturating_sub(1) && position < end {
            output.push_str(replacement);
        }
    }
    Ok(output)
}

pub(crate) fn replace_text_regex(
    module: &Module,
    state: &mut ExecutionState,
    regex: DatumId,
    arguments: &[Value],
    character_indices: bool,
    caller_context: &ExecutionContext,
) -> Result<Value, String> {
    let Value::Text(source) = &arguments[0] else {
        return Ok(Value::Null);
    };
    let source = source.to_string();
    let field = |name| FieldName::parse(name).expect("regex field is valid");
    let pattern = state
        .heap()
        .datum_field(regex, &field("_dream64_pattern"))
        .map_err(|error| error.to_string())?
        .clone();
    let pattern = builtin_text(&pattern, &state.heap, "regex pattern")?;
    let flags = state
        .heap()
        .datum_field(regex, &field("flags"))
        .ok()
        .and_then(|value| match value {
            Value::Text(text) => Some(text.as_ref()),
            _ => None,
        })
        .unwrap_or("")
        .to_owned();
    let global = flags.contains('g');
    let (start, end) = replacement_bounds(&source, arguments, character_indices)?;
    let prefix = source[..start].to_owned();
    let suffix = source[end..].to_owned();
    let mut target = source[start..end].to_owned();
    let replacement_proc = matches!(arguments[2], Value::TypePath(_)).then(|| {
        dynamic_call_target(
            module,
            state,
            &Value::Null,
            &arguments[2],
            caller_context,
            true,
        )
    });
    let replacement_text = if replacement_proc.is_none() {
        Some(builtin_text(
            &arguments[2],
            &state.heap,
            "replacetext replacement",
        )?)
    } else {
        None
    };
    let replacement_proc = replacement_proc.transpose()?;

    let mut cursor = 0;
    loop {
        let Some((begin, finish, captures)) =
            builtins::regex_search(&pattern, &flags, &target, cursor, target.len())?
        else {
            break;
        };
        let replacement = if let Some((procedure, context)) = &replacement_proc {
            let mut callback_arguments = Vec::with_capacity(captures.len() + 1);
            callback_arguments.push(Value::text(&target[begin..finish]));
            callback_arguments.extend(
                captures
                    .into_iter()
                    .map(|capture| capture.map_or(Value::Null, Value::text)),
            );
            let value =
                execute_module_in_context(module, *procedure, &callback_arguments, state, context)
                    .map_err(|error| error.to_string())?;
            match value {
                Value::Null => String::new(),
                Value::Text(text) => text.to_string(),
                Value::Number(number) => number.to_f32().to_string(),
                Value::TypePath(path) => path.to_string(),
                value => format!("{value}"),
            }
        } else {
            replacement_text.clone().unwrap_or_default()
        };
        target.replace_range(begin..finish, &replacement);
        cursor = begin.saturating_add(replacement.len().max(1));
        if !global {
            break;
        }
    }
    Ok(Value::text(format!("{prefix}{target}{suffix}")))
}

pub(crate) fn copy_text_builtin(
    arguments: &[Value],
    character_indices: bool,
    heap: &ValueHeap,
) -> Result<String, String> {
    let source = builtin_text(&arguments[0], heap, "copytext text")?;
    let logical_length = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = signed_text_index(arguments.get(1), 1)?;
    let end = signed_text_index(arguments.get(2), 0)?;
    let start = resolve_text_position(start, logical_length);
    let end = if end == 0 {
        logical_length.saturating_add(1)
    } else {
        resolve_text_position(end, logical_length)
    };
    if end <= start {
        return Ok(String::new());
    }
    let start = start.saturating_sub(1);
    let end = end.saturating_sub(1);
    let (start, end) = if character_indices {
        (
            character_offset(&source, start),
            character_offset(&source, end),
        )
    } else {
        (
            previous_char_boundary(&source, start),
            previous_char_boundary(&source, end),
        )
    };
    Ok(source[start..end].to_owned())
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "DM text positions are integralized from binary32 at the language boundary"
)]
pub(crate) fn signed_text_index(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => {
            let number = number.to_f32();
            if !number.is_finite() {
                return Ok(default);
            }
            Ok(number.trunc() as i64)
        }
        Some(value) => Err(format!(
            "copytext bounds require a number, received {value}"
        )),
    }
}

pub(crate) fn resolve_text_position(position: i64, logical_length: usize) -> usize {
    let limit = i64::try_from(logical_length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

pub(crate) fn builtin_text(
    value: &Value,
    heap: &ValueHeap,
    context: &str,
) -> Result<String, String> {
    match value {
        Value::Text(text) => Ok(String::from(text.as_ref())),
        Value::Datum(datum)
            if heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .to_string()
                == "/regex" =>
        {
            Err(format!("{context} regex matching is not yet supported"))
        }
        _ => Err(format!("{context} requires text, received {value}")),
    }
}

/// Advances the fixed-seed headless random stream and returns a unit interval
/// sample. Keeping it in [`ExecutionState`] makes repeated calls vary while
/// fresh headless worlds remain reproducible.
#[allow(
    clippy::cast_precision_loss,
    reason = "the upper 24 generator bits deliberately map onto the binary32 unit interval"
)]
pub(crate) fn deterministic_unit(state: &mut u64) -> f32 {
    if *state == 0 {
        *state = 0x9e37_79b9_7f4a_7c15;
    }
    *state ^= *state << 7;
    *state ^= *state >> 9;
    *state ^= *state << 8;
    let high = (*state >> 40) as u32;
    high as f32 / 16_777_216.0
}

/// Implements DM's `round(A)` and `round(A, B)` forms for scalar numbers.
///
/// The single-argument form is the historical floor operation.  With a
/// non-zero multiple, BYOND chooses the nearest multiple; an exact halfway
/// value goes toward positive infinity, as in `floor(A / B + 0.5)`.  A zero
/// multiple follows the legacy floor form rather than dividing by zero.
pub(crate) fn round_builtin(arguments: &[Value]) -> Result<f32, String> {
    let value = match &arguments[0] {
        Value::Null => 0.0,
        value => value
            .as_number()
            .ok_or_else(|| format!("round requires a number, received {value}"))?,
    };
    if arguments.len() == 1 {
        return Ok(value.floor());
    }
    let multiple = arguments[1].as_number().ok_or_else(|| {
        format!(
            "round multiple requires a number, received {}",
            arguments[1]
        )
    })?;
    if multiple == 0.0 {
        return Ok(value.floor());
    }
    // The sign of a multiple does not alter its set of multiples.  Using its
    // magnitude also preserves BYOND's increasing-number-line tie rule.
    Ok((value / multiple.abs() + 0.5).floor() * multiple.abs())
}

pub(crate) fn random_integer(arguments: &[Value], state: &mut u64) -> Result<f32, String> {
    let bounds = arguments
        .iter()
        .map(|value| {
            value
                .as_number()
                .ok_or_else(|| format!("rand requires numbers, received {value}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (low, high) = match bounds.as_slice() {
        [] => return Ok(deterministic_unit(state)),
        [high] => (0.0, *high),
        [low, high] => (*low, *high),
        _ => return Err("rand accepts zero, one, or two bounds".to_owned()),
    };
    let mut low = low.ceil();
    let mut high = high.floor();
    if !low.is_finite() || !high.is_finite() {
        return Err(format!("invalid rand range {low} through {high}"));
    }
    if high < low {
        std::mem::swap(&mut low, &mut high);
    }
    Ok(low + (deterministic_unit(state) * (high - low + 1.0)).floor())
}

pub(crate) fn roll_dice(arguments: &[Value], state: &mut u64) -> Result<f32, String> {
    let (count, sides, offset) = match arguments {
        [Value::Text(dice)] => {
            let dice = dice.trim();
            let (count, remainder) = dice
                .split_once(['d', 'D'])
                .ok_or_else(|| format!("invalid dice expression {dice:?}"))?;
            let sign = remainder
                .char_indices()
                .skip(1)
                .find(|(_, character)| matches!(character, '+' | '-'))
                .map(|(index, _)| index);
            let (sides, offset) = sign.map_or((remainder, "0"), |index| remainder.split_at(index));
            (
                count
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice count {count:?}"))?,
                sides
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice sides {sides:?}"))?,
                offset
                    .parse::<i32>()
                    .map_err(|_| format!("invalid dice offset {offset:?}"))?,
            )
        }
        [sides] => (
            1,
            sides
                .as_number()
                .ok_or_else(|| format!("roll requires a number or dice text, received {sides}"))?
                .trunc() as i32,
            0,
        ),
        [count, sides] => (
            count
                .as_number()
                .ok_or_else(|| format!("roll count requires a number, received {count}"))?
                .trunc() as i32,
            sides
                .as_number()
                .ok_or_else(|| format!("roll sides requires a number, received {sides}"))?
                .trunc() as i32,
            0,
        ),
        _ => return Err("roll requires one or two arguments".to_owned()),
    };
    if count < 0 || sides < 1 {
        return Err(format!("invalid dice dimensions {count}d{sides}"));
    }
    let mut total = offset as f32;
    for _ in 0..count {
        total += random_integer(&[Value::number(1.0), Value::number(sides as f32)], state)?;
    }
    Ok(total)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the unit sample is non-negative and strictly below one, yielding a valid list offset"
)]
pub(crate) fn pick_value(
    values: &[Value],
    weighted: &[bool],
    heap: &ValueHeap,
    state: &mut u64,
) -> Result<Value, String> {
    if weighted.len() == 1
        && !weighted[0]
        && let [Value::List(list)] = values
    {
        let list = heap.list(*list).map_err(|error| error.to_string())?;
        if list.is_empty() {
            return Ok(Value::Null);
        }
        let index = (deterministic_unit(state) * list.len() as f32).floor() as usize + 1;
        return list.get(index).cloned().map_err(|error| error.to_string());
    }
    let mut cursor = 0;
    let mut entries = Vec::with_capacity(weighted.len());
    let mut total = 0.0_f32;
    for is_weighted in weighted {
        let weight = if *is_weighted {
            let value = values
                .get(cursor)
                .ok_or_else(|| "invalid pick weights".to_owned())?;
            cursor += 1;
            value
                .as_number()
                .ok_or_else(|| format!("pick weight requires a number, received {value}"))?
                .max(0.0)
        } else {
            1.0
        };
        let candidate = values
            .get(cursor)
            .ok_or_else(|| "invalid pick candidates".to_owned())?
            .clone();
        cursor += 1;
        total += weight;
        entries.push((weight, candidate));
    }
    if total <= 0.0 {
        return Ok(Value::Null);
    }
    let mut point = deterministic_unit(state) * total;
    for (weight, candidate) in entries {
        if point < weight {
            return Ok(candidate);
        }
        point -= weight;
    }
    Ok(Value::Null)
}

/// Returns BYOND's legacy `length()` result for the runtime values accepted
/// by the headless VM. Text uses byte length because regular DM text indices
/// are byte indices; `_char` builtins are the explicit character-indexed API.
pub(crate) fn builtin_length(value: &Value, heap: &ValueHeap) -> Result<f32, String> {
    let length = match value {
        Value::Null => 0,
        Value::Text(text) => text.len(),
        Value::List(list) => heap.list(*list).map_err(|error| error.to_string())?.len(),
        // BYOND's legacy length() is also used as a cheap list/text probe.
        // Values outside those two families (including type paths) report 0
        // rather than raising a runtime error.
        _ => 0,
    };
    length
        .to_string()
        .parse::<f32>()
        .map_err(|error| format!("length cannot be represented as binary32: {error}"))
}

/// Produces the opaque reference text used by DM's `ref()` builtin.
///
/// BYOND reserves the `0xe...` range for list references; preserving that
/// convention matters to code which uses a list reference as an associative
/// key. Dream64's headless heap keeps identities live for the execution, so
/// its monotonic slot identity is sufficient for a stable reference.
pub(crate) fn ref_builtin(value: &Value) -> Value {
    let reference = match value {
        Value::Datum(datum) => format!("[0xd{:06x}]", datum.index() + 1),
        Value::List(list) => format!("[0xe{:06x}]", list.index() + 1),
        // `ref` identifies runtime heap objects; scalar values have no
        // object identity and therefore cannot yield a usable reference.
        Value::Null
        | Value::Number(_)
        | Value::Text(_)
        | Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_) => return Value::Null,
    };
    Value::text(reference)
}

/// Resolves BYOND's `get_step(atom_or_turf, direction)` against live turfs.
///
/// BYOND directions are bit flags: north/south affect Y, east/west affect X,
/// and up/down affect Z, so combined values naturally move on multiple axes.
/// A source atom's materialized coordinates are sufficient even when its
/// `loc` has not been explicitly connected in the headless world model.
pub(crate) fn spatial_field_names() -> &'static (FieldName, FieldName, FieldName, FieldName) {
    static FIELDS: OnceLock<(FieldName, FieldName, FieldName, FieldName)> = OnceLock::new();
    FIELDS.get_or_init(|| {
        (
            FieldName::parse("x").expect("built-in coordinate field is valid"),
            FieldName::parse("y").expect("built-in coordinate field is valid"),
            FieldName::parse("z").expect("built-in coordinate field is valid"),
            FieldName::parse("loc").expect("built-in loc field is valid"),
        )
    })
}

pub(crate) fn get_step_builtin(
    source: &Value,
    direction: &Value,
    state: &ExecutionState,
) -> Result<Value, String> {
    let Value::Datum(source) = source else {
        return Ok(Value::Null);
    };
    // BYOND's spatial builtins accept atoms, not arbitrary `/datum` values.
    // `get_turf(component)` is commonly used defensively and returns null;
    // it must not probe synthetic x/y/z fields on the component datum.
    let source_datum = state
        .heap()
        .datum(*source)
        .map_err(|error| error.to_string())?;
    if !is_atom_type_path(source_datum.type_path()) {
        return Ok(Value::Null);
    }
    let Some(direction) = direction.as_number() else {
        return Ok(Value::Null);
    };
    let Some(direction) = dm_direction_bits(direction) else {
        return Ok(Value::Null);
    };
    // Unknown bits do not name a world direction.  `get_step(source, 0)` is
    // useful as a normalized `get_turf` and returns the containing turf.
    let (x, y, z, loc) = spatial_field_names();
    let original_source = *source;
    let mut coordinate_source = original_source;
    let mut current = original_source;
    // Atom containment is normally only one or two links deep. Keep cycle
    // protection inline for that overwhelmingly common case; SmallVec spills
    // safely for pathological/deep loc chains without changing semantics.
    let mut visited = SmallVec::<[DatumId; 8]>::new();
    while !visited.contains(&current) {
        visited.push(current);
        let datum = state
            .heap()
            .datum(current)
            .map_err(|error| error.to_string())?;
        if is_turf_type_path(datum.type_path()) {
            coordinate_source = current;
            break;
        }
        let Ok(Value::Datum(parent)) = datum_field_or_initial(state, current, loc) else {
            break;
        };
        current = parent;
    }
    let coordinate = |field: &FieldName| -> Result<f32, String> {
        datum_field_or_initial(state, coordinate_source, field)
            .map_err(|error| error.to_string())?
            .as_number()
            .ok_or_else(|| format!("get_step source coordinate {field} is not numeric"))
    };
    let source_x = coordinate(x)?;
    let source_y = coordinate(y)?;
    let source_z = coordinate(z)?;
    let target_x = source_x + f32::from(u8::from(direction & 4 != 0))
        - f32::from(u8::from(direction & 8 != 0));
    let target_y = source_y + f32::from(u8::from(direction & 1 != 0))
        - f32::from(u8::from(direction & 2 != 0));
    let target_z = source_z + f32::from(u8::from(direction & 16 != 0))
        - f32::from(u8::from(direction & 32 != 0));
    let Some((target_x, target_y, target_z)) = dm_world_coordinate(target_x)
        .zip(dm_world_coordinate(target_y))
        .zip(dm_world_coordinate(target_z))
        .map(|((x, y), z)| (x, y, z))
    else {
        return Ok(Value::Null);
    };
    // `(0, 0, 0)` is the default coordinate triplet for atoms that are not
    // inside the world. It must never resolve to an unindexed `/turf`
    // prototype in lightweight states: BYOND world coordinates are 1-based.
    if target_x < 1 || target_y < 1 || target_z < 1 {
        return Ok(Value::Null);
    }
    if !state.world_turfs.is_empty() {
        return Ok(state
            .turf_at(target_x, target_y, target_z)
            .map_or(Value::Null, Value::Datum));
    }
    // Standalone VM callers can provide a heap without a canonical world
    // geometry index. Preserve that lightweight embedding contract without
    // making production world lookups fall back to an O(all datums) scan.
    for (datum, candidate) in state.heap().datums() {
        let path = candidate.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            continue;
        }
        if candidate
            .field(x)
            .is_ok_and(|value| value.as_number() == Some(target_x as f32))
            && candidate
                .field(y)
                .is_ok_and(|value| value.as_number() == Some(target_y as f32))
            && candidate
                .field(z)
                .is_ok_and(|value| value.as_number() == Some(target_z as f32))
        {
            return Ok(Value::Datum(datum));
        }
    }
    Ok(Value::Null)
}

#[inline]
pub(crate) fn dm_world_coordinate(value: f32) -> Option<i32> {
    (value.is_finite()
        && value.fract() == 0.0
        && value >= i32::MIN as f32
        // `i32::MAX as f32` rounds upward to 2^31, which parse::<i32> rejects.
        && value < 2_147_483_648.0)
        .then_some(value as i32)
}

#[inline]
pub(crate) fn dm_direction_bits(direction: f32) -> Option<i16> {
    // Direction masks are a tiny closed integer domain. Avoid formatting the
    // number as decimal and parsing it back on every topology/step query.
    (direction.is_finite() && direction.fract() == 0.0 && (0.0..=63.0).contains(&direction))
        .then_some(direction as i16)
}

pub(crate) fn direction_towards_builtin(
    source: &Value,
    target: &Value,
    state: &ExecutionState,
) -> Result<Value, String> {
    let (Some((source_x, source_y, source_z)), Some((target_x, target_y, target_z))) = (
        builtins::datum_coordinates(state, source),
        builtins::datum_coordinates(state, target),
    ) else {
        return Ok(Value::number(0.0));
    };
    if source_z != target_z {
        return Ok(Value::number(0.0));
    }
    let dx = target_x - source_x;
    let dy = target_y - source_y;
    let mut direction = 0_u8;
    if dy > 0.0 {
        direction |= 1;
    } else if dy < 0.0 {
        direction |= 2;
    }
    if dx > 0.0 {
        direction |= 4;
    } else if dx < 0.0 {
        direction |= 8;
    }
    Ok(Value::number(f32::from(direction)))
}

/// Resolves BYOND's `block()` over materialized headless turfs.
pub(crate) fn block_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let list = state.heap.allocate_list();
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");

    let datum_coordinates = |value: &Value, heap: &ValueHeap| -> Option<(f32, f32, f32)> {
        let Value::Datum(datum) = value else {
            return None;
        };
        let datum = heap.datum(*datum).ok()?;
        let path = datum.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            return None;
        }
        Some((
            datum.field(&x).ok()?.as_number()?,
            datum.field(&y).ok()?.as_number()?,
            datum.field(&z).ok()?.as_number()?,
        ))
    };
    let numeric = |value: &Value| value.as_number().filter(|number| number.is_finite());

    let (start, end) = match arguments {
        [start, end] => {
            let Some(start) = datum_coordinates(start, &state.heap) else {
                return Ok(Value::List(list));
            };
            let Some(end) = datum_coordinates(end, &state.heap) else {
                return Ok(Value::List(list));
            };
            (start, end)
        }
        [start_x, start_y, start_z, rest @ ..] if rest.len() <= 3 => {
            let (Some(start_x), Some(start_y), Some(start_z)) =
                (numeric(start_x), numeric(start_y), numeric(start_z))
            else {
                return Ok(Value::List(list));
            };
            let end_x = rest
                .first()
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_x);
            let end_y = rest
                .get(1)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_y);
            let end_z = rest
                .get(2)
                .filter(|value| !matches!(value, Value::Null))
                .and_then(numeric)
                .unwrap_or(start_z);
            ((start_x, start_y, start_z), (end_x, end_y, end_z))
        }
        _ => return Err("block requires two turfs or three through six coordinates".to_owned()),
    };

    // Accept either corner ordering while preserving the inclusive rectangular
    // volume described by the two endpoints. This is important to movement
    // code whose source/destination order naturally changes with direction.
    let low = (start.0.min(end.0), start.1.min(end.1), start.2.min(end.2));
    let high = (start.0.max(end.0), start.1.max(end.1), start.2.max(end.2));
    let matching = if state.world_turfs.is_empty() {
        // Standalone VM callers can construct a synthetic heap without ever
        // materializing world geometry. Preserve that useful fallback while
        // keeping real worlds on their authoritative coordinate index.
        state
            .heap
            .datums()
            .filter_map(|(datum, candidate)| {
                let path = candidate.type_path().as_str();
                if path != "/turf" && !path.starts_with("/turf/") {
                    return None;
                }
                let candidate_x = candidate.field(&x).ok()?.as_number()?;
                let candidate_y = candidate.field(&y).ok()?.as_number()?;
                let candidate_z = candidate.field(&z).ok()?.as_number()?;
                (candidate_x >= low.0
                    && candidate_x <= high.0
                    && candidate_y >= low.1
                    && candidate_y <= high.1
                    && candidate_z >= low.2
                    && candidate_z <= high.2)
                    .then_some(datum)
            })
            .collect::<Vec<_>>()
    } else {
        let integer_bounds = |low: f32, high: f32| -> Option<(i32, i32)> {
            let low = low.ceil();
            let high = high.floor();
            if low > high || high < i32::MIN as f32 || low > i32::MAX as f32 {
                return None;
            }
            #[allow(clippy::cast_possible_truncation)]
            Some((low as i32, high as i32))
        };
        let (Some((low_x, high_x)), Some((low_y, high_y)), Some((low_z, high_z))) = (
            integer_bounds(low.0, high.0),
            integer_bounds(low.1, high.1),
            integer_bounds(low.2, high.2),
        ) else {
            return Ok(Value::List(list));
        };
        let axis_len = |low: i32, high: i32| {
            u128::try_from(i64::from(high) - i64::from(low) + 1)
                .expect("ordered i32 bounds have a positive span")
        };
        let volume = axis_len(low_x, high_x)
            .saturating_mul(axis_len(low_y, high_y))
            .saturating_mul(axis_len(low_z, high_z));
        let direct_limit = (state.world_turfs.len() as u128)
            .saturating_mul(2)
            .max(4_096);
        if volume <= direct_limit {
            let mut matching = Vec::new();
            for z in low_z..=high_z {
                for y in low_y..=high_y {
                    for x in low_x..=high_x {
                        if let Some(turf) = state.turf_at(x, y, z) {
                            matching.push(turf);
                        }
                    }
                }
            }
            matching
        } else {
            // Avoid walking an attacker-sized coordinate cuboid. Filtering
            // the compact index is still linear only in materialized turfs,
            // then restore block()'s z/y/x coordinate order.
            let mut matching = state
                .world_turfs
                .iter()
                .filter(|((x, y, z), _)| {
                    *x >= low_x
                        && *x <= high_x
                        && *y >= low_y
                        && *y <= high_y
                        && *z >= low_z
                        && *z <= high_z
                })
                .map(|(coordinate, datum)| (*coordinate, *datum))
                .collect::<Vec<_>>();
            matching.sort_unstable_by_key(|((x, y, z), _)| (*z, *y, *x));
            matching.into_iter().map(|(_, datum)| datum).collect()
        }
    };
    let result = state
        .heap
        .list_mut(list)
        .expect("a newly allocated list handle must be live");
    result.extend_positional(matching.into_iter().map(Value::Datum));
    Ok(Value::List(list))
}

/// Resolves BYOND's `range()` over the materialized headless world.
///
/// The regular `range(distance, center)` spelling and BYOND's accepted
/// reversed `range(center, distance)` spelling are both supported.  With one
/// argument, the current procedure's `src` is the center.  BYOND range is a
/// square tile radius (Chebyshev distance), not Euclidean distance. BYOND emits
/// the center tile first, followed by the remaining tiles in x/y order; each
/// turf is followed by its unique area (on first encounter) and live contents.
pub(crate) fn range_builtin(
    arguments: &[Value],
    src: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let null_center = Value::Null;
    let (distance, center) = match arguments {
        [distance] => (distance.as_number(), src),
        [first, second] => match (first.as_number(), second.as_number()) {
            (Some(distance), None) => (Some(distance), second),
            (None, Some(distance)) => (Some(distance), first),
            // A number cannot be a materialized map location.  Keeping this
            // an empty result mirrors BYOND's non-loc center behavior while
            // avoiding a fabricated coordinate.
            _ => (None, &null_center),
        },
        _ => return Err("range accepts one or two arguments".to_owned()),
    };
    let list = state.heap.allocate_list();
    let Some(distance) = distance else {
        return Ok(Value::List(list));
    };
    if !distance.is_finite() || distance < 0.0 {
        return Ok(Value::List(list));
    }
    let Some((center_x, center_y, center_z)) = builtins::datum_coordinates(state, center) else {
        return Ok(Value::List(list));
    };
    let x = FieldName::parse("x").expect("built-in coordinate field is valid");
    let y = FieldName::parse("y").expect("built-in coordinate field is valid");
    let z = FieldName::parse("z").expect("built-in coordinate field is valid");
    let distance = distance.floor();
    let matching = if state.world_turfs.is_empty() {
        // Standalone VM fixtures can supply coordinate-bearing atoms without
        // constructing canonical world geometry. Retain the historical scan
        // for those synthetic states only.
        state
            .heap
            .datums()
            .filter_map(|(datum, candidate)| {
                let path = candidate.type_path().as_str();
                if path == "/area" || path.starts_with("/area/") {
                    return None;
                }
                let candidate_x = candidate.field(&x).ok()?.as_number()?;
                let candidate_y = candidate.field(&y).ok()?.as_number()?;
                let candidate_z = candidate.field(&z).ok()?.as_number()?;
                (candidate_z.total_cmp(&center_z).is_eq()
                    && (candidate_x - center_x).abs() <= distance
                    && (candidate_y - center_y).abs() <= distance)
                    .then_some(datum)
            })
            .collect::<Vec<_>>()
    } else {
        let coordinate = |value: f32| -> Option<i32> {
            (value.is_finite()
                && value.fract() == 0.0
                && value >= i32::MIN as f32
                && value <= i32::MAX as f32)
                .then_some({
                    #[allow(clippy::cast_possible_truncation)]
                    {
                        value as i32
                    }
                })
        };
        let (Some(center_x), Some(center_y), Some(center_z)) = (
            coordinate(center_x),
            coordinate(center_y),
            coordinate(center_z),
        ) else {
            return Ok(Value::List(list));
        };
        #[allow(clippy::cast_possible_truncation)]
        let distance = distance.min(i32::MAX as f32) as i32;
        let low_x = center_x.saturating_sub(distance);
        let high_x = center_x.saturating_add(distance);
        let low_y = center_y.saturating_sub(distance);
        let high_y = center_y.saturating_add(distance);
        let axis_len = |low: i32, high: i32| {
            u128::try_from(i64::from(high) - i64::from(low) + 1)
                .expect("ordered i32 bounds have a positive span")
        };
        let area = axis_len(low_x, high_x).saturating_mul(axis_len(low_y, high_y));
        let direct_limit = (state.world_turfs.len() as u128)
            .saturating_mul(2)
            .max(4_096);
        let mut tiles = if area <= direct_limit {
            let mut tiles = Vec::new();
            for x in low_x..=high_x {
                for y in low_y..=high_y {
                    if let Some(turf) = state.turf_at(x, y, center_z) {
                        tiles.push(((x, y, center_z), turf));
                    }
                }
            }
            tiles
        } else {
            state
                .world_turfs
                .iter()
                .filter(|((x, y, z), _)| {
                    *z == center_z && *x >= low_x && *x <= high_x && *y >= low_y && *y <= high_y
                })
                .map(|(coordinate, turf)| (*coordinate, *turf))
                .collect::<Vec<_>>()
        };
        let center_coordinate = (center_x, center_y, center_z);
        if let Some(index) = tiles
            .iter()
            .position(|(coordinate, _)| *coordinate == center_coordinate)
        {
            let center = tiles.remove(index);
            tiles.insert(0, center);
        }

        let contents = FieldName::parse("contents").expect("built-in contents field");
        let mut seen_areas = HashSet::new();
        let mut matching = Vec::new();
        for (coordinate, turf) in tiles {
            matching.push(turf);
            if let Some(area) = state.world_areas.get(&coordinate).copied()
                && seen_areas.insert(area)
            {
                matching.push(area);
            }
            if let Ok(Value::List(contents)) = state.heap.datum_field(turf, &contents) {
                matching.extend(
                    state
                        .heap
                        .list(*contents)
                        .map_err(|error| error.to_string())?
                        .positions()
                        .filter_map(|(_, value)| match value {
                            Value::Datum(datum) => Some(*datum),
                            _ => None,
                        }),
                );
            }
        }
        matching
    };
    let result = state
        .heap
        .list_mut(list)
        .expect("a newly allocated list handle must be live");
    for datum in matching {
        result.add(Value::Datum(datum));
    }
    Ok(Value::List(list))
}

/// Resolves BYOND's `typesof()` selector against the runtime's canonical type
/// catalog. The selected path itself is always present even for a deliberately
/// partial headless catalog, matching the inclusive nature of `typesof`.
pub(crate) fn typesof_builtin(
    value: &Value,
    heap: &ValueHeap,
    catalog: &std::collections::BTreeSet<TypePath>,
) -> Result<Vec<TypePath>, String> {
    let selector = match value {
        // BYOND 516 filters null selectors out. This matters for helper
        // routines that expand a caller-provided list of roots one at a time.
        Value::Null => return Ok(Vec::new()),
        Value::TypePath(path) => path.clone(),
        Value::Text(text) => {
            let Ok(path) = TypePath::parse(text) else {
                return Ok(Vec::new());
            };
            if !catalog.contains(&path) {
                return Ok(Vec::new());
            }
            path
        }
        Value::Datum(datum) => heap
            .datum(*datum)
            .map_err(|error| error.to_string())?
            .type_path()
            .clone(),
        value => {
            return Err(format!(
                "typesof requires a type path or datum, received {value}"
            ));
        }
    };
    // Canonical descendant paths occupy one contiguous lexical range in the
    // BTreeSet. Start at the selector in O(log N) and stop at the first
    // sibling instead of rescanning the complete project type catalog for
    // every `typesof()`/`subtypesof()` helper call.
    let descendant_prefix = format!("{}/", selector.as_str());
    let mut paths = catalog
        .range(selector.clone()..)
        .take_while(|path| {
            path.as_str() == selector.as_str() || path.as_str().starts_with(&descendant_prefix)
        })
        .cloned()
        .collect::<Vec<_>>();
    if !catalog.contains(&selector) {
        paths.insert(0, selector);
    }
    Ok(paths)
}

/// Evaluates the small family of BYOND value/type classifiers that need heap
/// access for datum runtime paths. `isloc` is variadic, unlike the other
/// simple classifiers: all its supplied values must be atoms.
pub(crate) fn type_predicate_builtin(
    kind: TypePredicateKind,
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<bool, String> {
    let heap = &state.heap;
    let value = arguments
        .first()
        .ok_or_else(|| "type predicate requires a value".to_owned())?;
    let value = heap.canonicalize_value(value);
    match kind {
        TypePredicateKind::IsNull => Ok(matches!(value, Value::Null)),
        TypePredicateKind::IsNum => Ok(matches!(value, Value::Number(_))),
        TypePredicateKind::IsPath => {
            let Value::TypePath(candidate) = value else {
                return Ok(false);
            };
            let Some(target) = arguments.get(1) else {
                return Ok(true);
            };
            let Value::TypePath(target) = target else {
                return Ok(false);
            };
            Ok(is_subtype(state, &candidate, target))
        }
        TypePredicateKind::IsList => Ok(matches!(value, Value::List(_))),
        TypePredicateKind::IsMovable => {
            let target = TypePath::parse("/atom/movable").expect("built-in movable path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsTurf => {
            let target = TypePath::parse("/turf").expect("built-in turf path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsIcon => match value {
            Value::File(path) => Ok(matches!(
                std::path::Path::new(path.as_ref())
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
                    .map(str::to_ascii_lowercase)
                    .as_deref(),
                Some("dmi" | "bmp" | "png" | "jpg" | "gif")
            )),
            Value::Datum(datum) => {
                let path = heap
                    .datum(datum)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                Ok(path == "/icon" || path.starts_with("/icon/"))
            }
            _ => Ok(false),
        },
        TypePredicateKind::IsLoc => {
            let target = TypePath::parse("/atom").expect("built-in atom path is valid");
            Ok(arguments.iter().all(|value| {
                let Value::Datum(datum) = value else {
                    return false;
                };
                heap.datum(*datum)
                    .is_ok_and(|datum| is_subtype(state, datum.type_path(), &target))
            }))
        }
        TypePredicateKind::IsType => {
            let Some(target) = arguments.get(1) else {
                // BYOND lists are first-class datum-like runtime values for
                // the unqualified `istype(value)` predicate. Project code
                // commonly uses this shape to validate a typed list argument
                // before constructing an owner datum (for example admin rank
                // lists during world startup).
                return Ok(matches!(value, Value::Datum(_) | Value::List(_)));
            };
            let Value::TypePath(target) = target else {
                return Ok(false);
            };
            if let Value::List(list) = value {
                return Ok(target.as_str() == "/list"
                    || (target.as_str() == "/alist" && state.is_associative_list(list)));
            }
            let candidate = match value {
                Value::Datum(datum) => heap
                    .datum(datum)
                    .map_err(|error| error.to_string())?
                    .type_path(),
                _ => return Ok(false),
            };
            Ok(is_subtype(state, candidate, target))
        }
    }
}

pub(crate) fn replacement_bounds(
    source: &str,
    arguments: &[Value],
    character_indices: bool,
) -> Result<(usize, usize), String> {
    let index_limit = if character_indices {
        source.chars().count()
    } else {
        source.len()
    };
    let start = optional_text_index(arguments.get(3), 1)?;
    let end = optional_text_index(arguments.get(4), 0)?;
    // BYOND text positions are 1-based and the end is exclusive; zero end
    // extends through the whole remaining text.
    let start = start.clamp(1, index_limit.saturating_add(1));
    let end = if end == 0 {
        index_limit.saturating_add(1)
    } else {
        end.clamp(start, index_limit.saturating_add(1))
    };
    if character_indices {
        Ok((
            character_offset(source, start.saturating_sub(1)),
            character_offset(source, end.saturating_sub(1)),
        ))
    } else {
        // Legacy DM indices count UTF-8 bytes. Clamp inward to valid Rust
        // boundaries instead of manufacturing invalid text slices.
        Ok((
            previous_char_boundary(source, start.saturating_sub(1)),
            previous_char_boundary(source, end.saturating_sub(1)),
        ))
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "DM text positions are non-negative integral binary32 values"
)]
pub(crate) fn optional_text_index(value: Option<&Value>, default: usize) -> Result<usize, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => Ok(number.to_f32().max(0.0) as usize),
        Some(value) => Err(format!(
            "replacetext bounds require a number, received {value}"
        )),
    }
}

pub(crate) fn character_offset(text: &str, character_index: usize) -> usize {
    text.char_indices()
        .nth(character_index)
        .map_or(text.len(), |(byte, _)| byte)
}

pub(crate) fn previous_char_boundary(text: &str, mut byte_index: usize) -> usize {
    byte_index = byte_index.min(text.len());
    while byte_index > 0 && !text.is_char_boundary(byte_index) {
        byte_index -= 1;
    }
    byte_index
}

pub(crate) fn replace_text_ascii_insensitive(
    target: &str,
    needle: &str,
    replacement: &str,
) -> String {
    if !needle.is_ascii() {
        // DM's Unicode case folding is more involved than Rust's simple
        // lowercase mapping. Preserve deterministic exact text for the rare
        // non-ASCII fallback rather than corrupting byte offsets.
        return target.replace(needle, replacement);
    }
    let needle_lower = needle.to_ascii_lowercase();
    let bytes = target.as_bytes();
    let mut output = String::with_capacity(target.len());
    let mut cursor = 0;
    while cursor < target.len() {
        let remaining = &target[cursor..];
        if remaining.len() >= needle.len()
            && remaining.as_bytes()[..needle.len()]
                .iter()
                .map(u8::to_ascii_lowercase)
                .eq(needle_lower.bytes())
        {
            output.push_str(replacement);
            cursor += needle.len();
        } else {
            let width = char::from(bytes[cursor]).len_utf8();
            output.push_str(&target[cursor..cursor + width]);
            cursor += width;
        }
    }
    output
}

pub(crate) fn runtime_truthy(heap: &ValueHeap, value: &Value) -> Result<bool, String> {
    heap.truthy(&canonicalize_value(heap, value))
        .map_err(|error| error.to_string())
}

pub(crate) fn logical_or_empty_list_field(
    state: &mut ExecutionState,
    receiver: Value,
    name: &FieldName,
) -> Result<Value, String> {
    let receiver = state.heap().canonicalize_value(&receiver);
    let current = match &receiver {
        Value::TypePath(path) => match name.as_str() {
            "type" => Value::TypePath(path.clone()),
            "parent_type" => state
                .type_parent(path)
                .cloned()
                .map_or(Value::Null, Value::TypePath),
            _ => state
                .initial_value(path, name)
                .cloned()
                .unwrap_or(Value::Null),
        },
        Value::ModifiedTypePath(path) => match name.as_str() {
            "type" => Value::TypePath(path.base().clone()),
            "parent_type" => state
                .type_parent(path.base())
                .cloned()
                .map_or(Value::Null, Value::TypePath),
            _ => path
                .overrides()
                .iter()
                .rev()
                .find(|(field, _)| field == name)
                .map(|(_, value)| value.clone())
                .or_else(|| state.initial_value(path.base(), name).cloned())
                .unwrap_or(Value::Null),
        },
        Value::List(list) if name.as_str() == "len" => Value::number(
            state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .len() as f32,
        ),
        Value::Datum(datum) => {
            let runtime_type = state
                .heap
                .datum(*datum)
                .map_err(|error| error.to_string())?
                .type_path()
                .clone();
            if name.as_str() == "type" {
                Value::TypePath(runtime_type)
            } else if name.as_str() == "parent_type" {
                state
                    .type_parent(&runtime_type)
                    .cloned()
                    .map_or(Value::Null, Value::TypePath)
            } else if name.as_str() == "appearance"
                && builtins::is_appearance_source(&runtime_type)
                && matches!(
                    datum_field_or_initial(state, *datum, name),
                    Ok(Value::Null) | Err(ValueError::MissingField(_))
                )
            {
                builtins::appearance_snapshot_builtin(*datum, state)?
            } else if name.as_str() == "transform"
                && builtins::is_appearance_source(&runtime_type)
                && matches!(
                    datum_field_or_initial(state, *datum, name),
                    Ok(Value::Null) | Err(ValueError::MissingField(_))
                )
            {
                Value::Datum(allocate_matrix(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    &mut state.heap,
                )?)
            } else if let Some(value) = lazy_atom_list_field(state, *datum, name)? {
                value
            } else if runtime_type.as_str() == "/savefile"
                || runtime_type.as_str().starts_with("/savefile/")
            {
                match name.as_str() {
                    "cd" => Value::text(
                        savefile_current_directory(&state.savefiles.entry(*datum).or_default().cd)
                            .to_owned(),
                    ),
                    "eof" => {
                        let savefile = state.savefiles.entry(*datum).or_default();
                        let path = savefile_current_directory(&savefile.cd);
                        Value::number(if savefile.entries.contains_key(path) {
                            0.0
                        } else {
                            1.0
                        })
                    }
                    "dir" => {
                        let children =
                            savefile_directory_entries(state.savefiles.entry(*datum).or_default());
                        let list = state.heap.allocate_list();
                        let values = state
                            .heap
                            .list_mut(list)
                            .map_err(|error| error.to_string())?;
                        for child in children {
                            values.add(Value::text(child));
                        }
                        Value::List(list)
                    }
                    _ => state
                        .heap
                        .datum_field(*datum, name)
                        .cloned()
                        .map_err(|error| error.to_string())?,
                }
            } else {
                datum_field_or_shared(state, *datum, name).map_err(|error| error.to_string())?
            }
        }
        Value::Null => return Err("field read received null".to_owned()),
        value => return Err(format!("field read requires a datum, received {value}")),
    };
    if runtime_truthy(&state.heap, &current)? {
        return Ok(current);
    }

    let value = Value::List(state.heap.allocate_list());
    match receiver {
        Value::Datum(datum) => {
            assign_datum_or_shared_field(state, datum, name.clone(), value.clone())?;
        }
        Value::List(list) if name.as_str() == "len" => {
            let visibility_before = state
                .is_visibility_list(list)
                .then(|| state.visibility_members(list))
                .transpose()?;
            if state.is_associative_list(list) {
                // Assigning a non-number to len coerces to zero for both list
                // kinds, which is the only alist length assignment allowed.
            }
            state
                .heap
                .list_mut(list)
                .and_then(|values| values.resize(0))
                .map_err(|error| error.to_string())?;
            if let Some(before) = visibility_before {
                state.normalize_and_synchronize_visibility_list(list, &before)?;
            }
        }
        Value::Null => return Err("field write received null".to_owned()),
        value => {
            return Err(format!(
                "field write requires a datum or list.len, received {value}"
            ));
        }
    }
    Ok(value)
}

pub(crate) fn logical_or_empty_list_index(
    state: &mut ExecutionState,
    receiver: Value,
    key: Value,
) -> Result<Value, String> {
    let receiver = state.heap().canonicalize_value(&receiver);
    let list = match &receiver {
        Value::List(list) => *list,
        Value::Text(text) => {
            let index = value_to_list_index(&key)?;
            let current = text
                .chars()
                .nth(index - 1)
                .map_or(Value::Null, |character| Value::text(character.to_string()));
            if runtime_truthy(&state.heap, &current)? {
                return Ok(current);
            }
            let _allocated_rhs = Value::List(state.heap.allocate_list());
            return Err(format!("list assignment received {receiver}"));
        }
        Value::Datum(savefile)
            if state.heap.datum(*savefile).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/savefile" || path.starts_with("/savefile/")
            }) =>
        {
            let rendered_key = match &key {
                Value::Text(key) => key.to_string(),
                value => value.to_string(),
            };
            let resolved = savefile_resolve_path(
                &state.savefiles.entry(*savefile).or_default().cd,
                &rendered_key,
            );
            let entry = state
                .heap
                .allocate_datum(TypePath::parse("/savefile/entry").unwrap());
            state.savefile_entries.insert(entry, (*savefile, resolved));
            return Ok(Value::Datum(entry));
        }
        value => return Err(format!("list index operation received {value}")),
    };
    let current = if state.global_vars_proxy == Some(list) {
        match &key {
            Value::Text(name) => FieldName::parse(name)
                .ok()
                .and_then(|name| state.global(&name).cloned())
                .unwrap_or(Value::Null),
            _ => read_list_value(&state.heap, list, &key, false).unwrap_or(Value::Null),
        }
    } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
        match &key {
            Value::Text(name) => {
                let field = FieldName::parse(name).ok();
                if let Some(value) = field
                    .as_ref()
                    .map(|field| lazy_atom_list_field(state, datum, field))
                    .transpose()?
                    .flatten()
                {
                    value
                } else {
                    field
                        .as_ref()
                        .and_then(|field| datum_shared_storage(state, datum, field))
                        .and_then(|storage| state.global(&storage).cloned())
                        .or_else(|| {
                            field
                                .and_then(|field| datum_field_or_initial(state, datum, &field).ok())
                        })
                        .unwrap_or(Value::Null)
                }
            }
            _ => read_list_value(&state.heap, list, &key, false).unwrap_or(Value::Null),
        }
    } else {
        match read_list_value(&state.heap, list, &key, state.is_associative_list(list)) {
            Ok(value) => value,
            Err(ValueError::MissingKey) => Value::Null,
            Err(error) => return Err(error.to_string()),
        }
    };
    if runtime_truthy(&state.heap, &current)? {
        return Ok(current);
    }

    let value = Value::List(state.heap.allocate_list());
    if state.is_visibility_list(list) {
        return Err("cannot write to an index of a visibility relationship list".to_owned());
    }
    if state.global_vars_proxy == Some(list) {
        let Value::Text(name) = key else {
            return Err("global.vars writes require a text key".to_owned());
        };
        let name = FieldName::parse(&name).map_err(|error| error.to_string())?;
        state.set_global(name, value.clone());
    } else if let Some(datum) = state.datum_vars_proxies.get(&list).copied() {
        write_datum_vars(state, datum, list, key, value.clone())?;
    } else {
        let associative = state.is_associative_list(list);
        write_list_value(&mut state.heap, list, key, value.clone(), associative)
            .map_err(|error| error.to_string())?;
    }
    Ok(value)
}

pub(crate) fn locate_in_container(
    search: &Value,
    container: &Value,
    state: &ExecutionState,
) -> Result<Value, String> {
    let list = match container {
        Value::List(list) => Some(*list),
        Value::Datum(datum) => state
            .heap()
            .datum_field(
                *datum,
                &FieldName::parse("contents").expect("built-in contents field is valid"),
            )
            .ok()
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            }),
        _ => None,
    };
    let Some(list) = list else {
        return Ok(Value::Null);
    };
    let values = state.heap().list(list).map_err(|error| error.to_string())?;
    for (_, candidate) in values.positions() {
        let matches = match search {
            Value::TypePath(target) => match candidate {
                Value::Datum(datum) => state
                    .heap()
                    .datum(*datum)
                    .is_ok_and(|datum| builtins::is_subtype(state, datum.type_path(), target)),
                Value::TypePath(candidate) => builtins::is_subtype(state, candidate, target),
                Value::ModifiedTypePath(candidate) => {
                    builtins::is_subtype(state, candidate.base(), target)
                }
                _ => false,
            },
            Value::Text(reference) if reference.starts_with("[0x") => {
                ref_builtin(candidate).semantic_eq(search)
            }
            _ => candidate.semantic_eq(search),
        };
        if matches {
            return Ok(candidate.clone());
        }
    }
    Ok(Value::Null)
}

pub(crate) fn locate_single(search: &Value, state: &ExecutionState) -> Value {
    match search {
        Value::Null => Value::Null,
        Value::Datum(datum) => {
            if state.heap().datum(*datum).is_ok() {
                Value::Datum(*datum)
            } else {
                Value::Null
            }
        }
        Value::List(list) => {
            if state.heap().list(*list).is_ok() {
                Value::List(*list)
            } else {
                Value::Null
            }
        }
        Value::TypePath(target) => state
            .heap()
            .datums()
            .find(|(_, datum)| builtins::is_subtype(state, datum.type_path(), target))
            .map_or(Value::Null, |(datum, _)| Value::Datum(datum)),
        Value::ModifiedTypePath(target) => state
            .heap()
            .datums()
            .find(|(_, datum)| builtins::is_subtype(state, datum.type_path(), target.base()))
            .map_or(Value::Null, |(datum, _)| Value::Datum(datum)),
        Value::Text(text) => {
            if let Some(reference) = parse_heap_reference(text) {
                return match reference {
                    HeapReference::Datum(index) => state
                        .heap()
                        .datum_id_at_index(index)
                        .map_or(Value::Null, Value::Datum),
                    HeapReference::List(index) => state
                        .heap()
                        .list_id_at_index(index)
                        .map_or(Value::Null, Value::List),
                };
            }
            let tag = FieldName::parse("tag").expect("built-in datum tag field is valid");
            state
                .heap()
                .datums()
                .find(|(_, datum)| {
                    datum
                        .field(&tag)
                        .is_ok_and(|candidate| candidate.semantic_eq(search))
                })
                .map_or(Value::Null, |(datum, _)| Value::Datum(datum))
        }
        Value::Number(_) | Value::File(_) => Value::Null,
    }
}

pub(crate) enum HeapReference {
    Datum(u32),
    List(u32),
}

pub(crate) fn parse_heap_reference(text: &str) -> Option<HeapReference> {
    let body = text.strip_prefix("[0x")?.strip_suffix(']')?;
    if body.len() < 2 {
        return None;
    }
    let (kind, digits) = body.split_at(1);
    let encoded = u32::from_str_radix(digits, 16).ok()?;
    let index = encoded.checked_sub(1)?;
    match kind {
        "d" => Some(HeapReference::Datum(index)),
        "e" => Some(HeapReference::List(index)),
        _ => None,
    }
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static DYNAMIC_LOOKUP_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(crate) fn dynamic_call_target(
    module: &Module,
    state: &mut ExecutionState,
    receiver: &Value,
    selector: &Value,
    caller_context: &ExecutionContext,
    null_receiver_is_global: bool,
) -> Result<(ProcedureId, ExecutionContext), String> {
    let selector = match selector {
        Value::Text(selector) => selector.as_ref(),
        Value::TypePath(selector) => selector.as_str(),
        _ => {
            return Err(format!(
                "call procedure selector must be text or a type path, received {selector}"
            ));
        }
    };

    dynamic_call_target_named(
        module,
        state,
        receiver,
        selector,
        caller_context,
        null_receiver_is_global,
    )
}

pub(crate) fn dynamic_call_target_named(
    module: &Module,
    state: &mut ExecutionState,
    receiver: &Value,
    selector: &str,
    caller_context: &ExecutionContext,
    null_receiver_is_global: bool,
) -> Result<(ProcedureId, ExecutionContext), String> {
    let (base_path, context) = match receiver {
        Value::Null if null_receiver_is_global => (None, caller_context.clone()),
        Value::Null => {
            return Err("cannot call a procedure on null".to_owned());
        }
        Value::Datum(datum) => {
            let Ok(record) = state.heap().datum(*datum) else {
                return Err("cannot call a procedure on null".to_owned());
            };
            (
                Some(record.type_path().clone()),
                ExecutionContext::new(Value::Datum(*datum), caller_context.usr.clone()),
            )
        }
        Value::List(list) if state.heap().list(*list).is_err() => {
            return Err("cannot call a procedure on null".to_owned());
        }
        Value::TypePath(path) => (Some(path.clone()), caller_context.clone()),
        _ => {
            return Err(format!(
                "call receiver must be a datum, type path, or null, received {receiver}"
            ));
        }
    };
    let relative_selector = selector.trim_start_matches('/');
    let (selector_namespace, selector_path) = relative_selector.strip_prefix("proc/").map_or_else(
        || {
            relative_selector
                .strip_prefix("verb/")
                .map_or((None, relative_selector), |path| (Some("verb"), path))
        },
        |path| (Some("proc"), path),
    );
    let selector_cache_key = selector_namespace.map_or_else(
        || std::borrow::Cow::Borrowed(selector_path),
        |namespace| std::borrow::Cow::Owned(format!("{namespace}/{selector_path}")),
    );
    let absolute = selector.starts_with('/');
    if !absolute
        && let Some(base_path) = &base_path
        && let Some(cached) = state
            .dynamic_receiver_targets
            .get(&(module.identity.0, base_path.clone()))
            .and_then(|selectors| selectors.get(selector_cache_key.as_ref()))
    {
        return cached.map_or_else(
            || {
                Err(format!(
                    "dynamic call could not resolve procedure {:?}",
                    format!("{}/proc/{selector_path}", base_path.as_str())
                ))
            },
            |procedure| Ok((procedure, context)),
        );
    }
    let requested = if absolute {
        selector.to_owned()
    } else if base_path.is_none() {
        format!("/proc/{selector_path}")
    } else {
        format!(
            "{}/proc/{selector_path}",
            base_path.as_ref().expect("receiver path exists").as_str()
        )
    };

    let find_candidate = |candidate: &str| {
        #[cfg(test)]
        DYNAMIC_LOOKUP_PROBES.set(DYNAMIC_LOOKUP_PROBES.get() + 1);
        module.dynamic_names.get(candidate).copied()
    };

    // Absolute/global procedure references already identify their namespace.
    // Preserve their historical exact/lexical lookup. Ordinary receiver calls,
    // however, must follow the runtime object tree: DM parent_type can point
    // outside the receiver's lexical path (notably /obj -> /atom).
    let resolved = if absolute || base_path.is_none() {
        let mut candidate = requested.clone();
        loop {
            if let Some(procedure) = find_candidate(&candidate) {
                break Some(procedure);
            }
            let Some(segment) = candidate.rfind("/proc/") else {
                break None;
            };
            let owner = &candidate[..segment];
            if owner == "/proc" || owner.is_empty() {
                break None;
            }
            let Some(parent_end) = owner.rfind('/') else {
                break None;
            };
            candidate = format!("{}/proc/{selector_path}", &owner[..parent_end]);
        }
    } else {
        let mut current = base_path.clone();
        let mut visited = std::collections::BTreeSet::new();
        let mut owners = Vec::new();
        while let Some(owner) = current {
            if !visited.insert(owner.clone()) {
                break;
            }
            owners.push(owner.clone());
            current = state
                .type_parents
                .get(&owner)
                .cloned()
                .flatten()
                .or_else(|| {
                    let parent = owner.as_str().rsplit_once('/')?.0;
                    (!parent.is_empty())
                        .then(|| TypePath::parse(parent).ok())
                        .flatten()
                });
        }

        // An unqualified member call searches the proc namespace through the
        // complete runtime parent chain before considering verbs. Besides
        // matching DM's proc preference, this avoids probing a nonexistent
        // verb at every inheritance level on ordinary hot-path calls.
        let explicit_namespace = selector_namespace.map(|namespace| [namespace]);
        let namespaces = explicit_namespace
            .as_ref()
            .map_or(&["proc", "verb"][..], |namespace| &namespace[..]);
        let mut found = None;
        'namespace: for namespace in namespaces {
            for owner in &owners {
                let candidate = format!("{}/{namespace}/{selector_path}", owner.as_str());
                if let Some(procedure) = find_candidate(&candidate) {
                    found = Some(procedure);
                    break 'namespace;
                }
            }
        }
        found
    };
    if !absolute && let Some(base_path) = base_path {
        state
            .dynamic_receiver_targets
            .entry((module.identity.0, base_path))
            .or_default()
            .insert(selector_cache_key.into_owned(), resolved);
    }
    resolved.map_or_else(
        || {
            Err(format!(
                "dynamic call could not resolve procedure {requested:?}"
            ))
        },
        |procedure| Ok((procedure, context)),
    )
}

pub(crate) fn dynamic_call_target_named_at_callsite(
    module: &Module,
    state: &mut ExecutionState,
    receiver: &Value,
    selector: &str,
    caller_context: &ExecutionContext,
    null_receiver_is_global: bool,
    callsite: Option<(ProcedureId, usize)>,
) -> Result<(ProcedureId, ExecutionContext), String> {
    let Some((caller, instruction)) = callsite else {
        return dynamic_call_target_named(
            module,
            state,
            receiver,
            selector,
            caller_context,
            null_receiver_is_global,
        );
    };
    let Value::Datum(datum) = receiver else {
        return dynamic_call_target_named(
            module,
            state,
            receiver,
            selector,
            caller_context,
            null_receiver_is_global,
        );
    };
    let receiver_type = state
        .heap
        .datum(*datum)
        .map_err(|_| "cannot call a procedure on null".to_owned())?
        .type_path()
        .clone();
    let key = (
        module.identity.0,
        caller,
        instruction,
        receiver_type.storage_identity(),
    );
    if let Some((cached_type, target)) = state.dynamic_callsite_targets.get(&key)
        && cached_type == &receiver_type
    {
        return Ok((
            *target,
            ExecutionContext::new(Value::Datum(*datum), caller_context.usr.clone()),
        ));
    }
    let (target, context) = dynamic_call_target_named(
        module,
        state,
        receiver,
        selector,
        caller_context,
        null_receiver_is_global,
    )?;
    state
        .dynamic_callsite_targets
        .insert(key, (receiver_type, target));
    Ok((target, context))
}

pub(crate) fn hascall_builtin(
    module: &Module,
    state: &ExecutionState,
    receiver: &Value,
    selector: &Value,
) -> bool {
    let selector = match selector {
        Value::Text(selector) => selector.as_ref(),
        Value::TypePath(selector) => selector.as_str(),
        _ => return false,
    };
    let selector = selector
        .trim_end_matches('/')
        .rsplit("/proc/")
        .next()
        .unwrap_or(selector)
        .trim_start_matches("proc/")
        .trim_start_matches('/');
    if selector.is_empty() || selector.contains('/') {
        return false;
    }

    if matches!(receiver, Value::List(_)) {
        return matches!(
            selector.to_ascii_lowercase().as_str(),
            "add"
                | "copy"
                | "cut"
                | "find"
                | "insert"
                | "join"
                | "remove"
                | "removeall"
                | "splice"
                | "swap"
        );
    }

    let path = match receiver {
        Value::Datum(datum) => match state.heap.datum(*datum) {
            Ok(datum) => datum.type_path().clone(),
            Err(_) => return false,
        },
        Value::TypePath(path) => path.clone(),
        _ => return false,
    };
    let native = match path.as_str() {
        "/list" => matches!(
            selector.to_ascii_lowercase().as_str(),
            "add"
                | "copy"
                | "cut"
                | "find"
                | "insert"
                | "join"
                | "remove"
                | "removeall"
                | "splice"
                | "swap"
        ),
        "/regex" => selector.eq_ignore_ascii_case("Find"),
        "/matrix" => matches!(
            selector.to_ascii_lowercase().as_str(),
            "add"
                | "subtract"
                | "multiply"
                | "scale"
                | "translate"
                | "turn"
                | "invert"
                | "interpolate"
        ),
        "/vector" => matches!(
            selector.to_ascii_lowercase().as_str(),
            "dot" | "interpolate" | "cross" | "magnitude" | "normalize" | "copy"
        ),
        "/icon" => matches!(
            selector.to_ascii_lowercase().as_str(),
            "mapcolors"
                | "blend"
                | "setintensity"
                | "scale"
                | "crop"
                | "shift"
                | "width"
                | "height"
                | "drawbox"
                | "insert"
                | "getpixel"
                | "turn"
                | "flip"
                | "swapcolor"
        ),
        path if path == "/savefile" || path.starts_with("/savefile/") => {
            selector.eq_ignore_ascii_case("ExportText")
        }
        _ => false,
    };
    if native {
        return true;
    }

    let mut current = Some(path);
    while let Some(owner) = current {
        let requested = format!("{}/proc/{selector}", owner.as_str());
        // Project code overwhelmingly uses the declaration's canonical
        // selector spelling. Resolve that common path through the module's
        // effective procedure index instead of scanning every linked body for
        // every ancestor. Preserve the historical case-insensitive fallback
        // for reflective callers that supply a differently cased string.
        if module.effective_procedure_id(&requested).is_some() {
            return true;
        }
        if module.paths.iter().any(|candidate| {
            candidate
                .split_once('@')
                .map_or(candidate.as_str(), |(path, _)| path)
                .eq_ignore_ascii_case(&requested)
        }) {
            return true;
        }
        current = state
            .type_parents
            .get(&owner)
            .cloned()
            .flatten()
            .or_else(|| {
                let parent = owner.as_str().rsplit_once('/')?.0;
                (!parent.is_empty())
                    .then(|| TypePath::parse(parent).ok())
                    .flatten()
            });
    }
    false
}

pub(crate) fn execute_del(
    module: &Module,
    arguments: &[Value],
    state: &mut ExecutionState,
    caller_context: &ExecutionContext,
) -> Result<Value, RuntimeError> {
    let Some(value) = arguments.first() else {
        return Ok(Value::Null);
    };
    let Value::Datum(datum) = value else {
        return execute_standard_builtin("del", arguments, state).map_err(|message| RuntimeError {
            message,
            instruction: 0,
            source_span: None,
            call_stack: Vec::new(),
        });
    };

    // A Del() body is allowed to delete its own src. In that case invalidate
    // the handle immediately, while the outer deletion remains responsible for
    // treating its eventual stale finalization as success.
    if !state.deleting_datums.insert(*datum) {
        let _ = state.heap_mut().destroy_datum(*datum);
        return Ok(Value::Null);
    }

    let receiver = Value::Datum(*datum);
    let hook = dynamic_call_target(
        module,
        state,
        &receiver,
        &Value::text("Del"),
        caller_context,
        false,
    );
    let hook_result = match hook {
        Ok((procedure, context)) => {
            execute_module_in_context(module, procedure, &[], state, &context).map(|_| ())
        }
        Err(_) => Ok(()),
    };

    // BYOND invalidates the object after Del() regardless of its return value.
    // Runtime failure likewise must not resurrect a half-cleaned-up datum.
    let _ = state.heap_mut().destroy_datum(*datum);
    state.deleting_datums.remove(datum);
    hook_result.map(|()| Value::Null)
}

pub(crate) fn constructor_target_if_present(
    module: &Module,
    state: &mut ExecutionState,
    datum: DatumId,
    caller_context: &ExecutionContext,
) -> Option<(ProcedureId, ExecutionContext)> {
    let receiver = Value::Datum(datum);
    let selector = Value::text("New");
    let traced_type = boot_trace_enabled()
        .then(|| {
            state
                .heap
                .datum(datum)
                .ok()
                .map(|datum| datum.type_path().clone())
        })
        .flatten()
        .filter(|path| path.as_str().starts_with("/datum/controller/subsystem/"));
    let (constructor, context) =
        match dynamic_call_target(module, state, &receiver, &selector, caller_context, false) {
            Ok(target) => target,
            Err(error) => {
                if let Some(path) = traced_type {
                    eprintln!("boot-vm: subsystem-constructor-missing type={path} reason={error}");
                }
                return None;
            }
        };
    if let Some(path) = traced_type {
        eprintln!(
            "boot-vm: subsystem-constructor type={} procedure={}",
            path,
            module.procedure_path(constructor).unwrap_or("<unknown>")
        );
    }
    Some((constructor, context))
}

pub(crate) fn construct_sized_list(
    arguments: &[Value],
    heap: &mut ValueHeap,
) -> Result<ListId, String> {
    fn dimension(value: &Value) -> Result<usize, String> {
        let number = value
            .as_number()
            .ok_or_else(|| format!("list dimension must be numeric, received {value}"))?;
        if !number.is_finite() || number < 0.0 || number.fract() != 0.0 {
            return Err(format!(
                "list dimension must be a non-negative integer, received {number}"
            ));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let size = number as usize;
        if size as f32 != number {
            return Err(format!("list dimension is too large: {number}"));
        }
        Ok(size)
    }

    let list = heap.allocate_list();
    let Some((first, remaining)) = arguments.split_first() else {
        return Ok(list);
    };
    let size = dimension(first)?;
    if remaining.is_empty() {
        heap.list_mut(list)
            .map_err(|error| error.to_string())?
            .resize(size)
            .map_err(|error| error.to_string())?;
    } else {
        for _ in 0..size {
            let child = construct_sized_list(remaining, heap)?;
            heap.list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::List(child));
        }
    }
    Ok(list)
}

pub(crate) fn read_list_value(
    heap: &ValueHeap,
    list: ListId,
    key: &Value,
    associative: bool,
) -> Result<Value, ValueError> {
    let values = heap.list(list)?;
    if matches!(key, Value::Number(_)) && !associative {
        let index = value_to_list_index(key).map_err(ValueError::InvalidListIndex)?;
        values
            .get(index)
            .map(|value| canonicalize_value(heap, value))
    } else {
        values
            .get_key(key)
            .map(|value| canonicalize_value(heap, value))
    }
}

pub(crate) fn savefile_export_value(value: &Value) -> String {
    match value {
        // Headless rendering does not own PNG pixels. Preserve BYOND's
        // base64-shaped savefile contract with a deterministic payload so
        // callers can cache and transport the result.
        Value::Datum(_) => "ZHJlYW02NA==".to_owned(),
        Value::Text(value) => value.to_string(),
        value => value.to_string(),
    }
}

pub(crate) fn savefile_current_directory(cd: &str) -> &str {
    if cd.is_empty() { "/" } else { cd }
}

pub(crate) fn savefile_resolve_path(cd: &str, path: &str) -> String {
    let joined = if path.starts_with('/') {
        path.to_owned()
    } else {
        format!(
            "{}/{}",
            savefile_current_directory(cd).trim_end_matches('/'),
            path
        )
    };
    let parts = joined
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .fold(Vec::new(), |mut parts, part| {
            if part == ".." {
                parts.pop();
            } else {
                parts.push(part);
            }
            parts
        });
    if parts.is_empty() {
        "/".to_owned()
    } else {
        format!("/{}", parts.join("/"))
    }
}

pub(crate) fn savefile_directory_entries(savefile: &SavefileState) -> Vec<String> {
    let directory = savefile_current_directory(&savefile.cd);
    let prefix = if directory == "/" {
        "/".to_owned()
    } else {
        format!("{}/", directory.trim_end_matches('/'))
    };
    let mut children = savefile
        .entries
        .keys()
        .filter_map(|path| path.strip_prefix(&prefix))
        .filter_map(|remainder| remainder.split('/').next())
        .filter(|child| !child.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    children.sort();
    children.dedup();
    children
}

pub(crate) fn datum_shared_storage(
    state: &ExecutionState,
    datum: DatumId,
    field: &FieldName,
) -> Option<FieldName> {
    let path = state.heap.datum(datum).ok()?.type_path();
    state.shared_fields.get(path)?.get(field).cloned()
}

/// Reads one instance field, falling back to the immutable effective type
/// default whenever the declared slot was not materialized on the heap.
///
/// Bulk map objects deliberately use this sparse representation, but the same
/// rule is also required for engine-created or legacy allocation paths: a DM
/// variable declared anywhere in the effective hierarchy remains readable at
/// its initial value. Stale handles and genuinely unknown fields remain errors.
pub(crate) fn datum_field_or_initial(
    state: &ExecutionState,
    datum: DatumId,
    field: &FieldName,
) -> Result<Value, ValueError> {
    let record = state.heap.datum(datum)?;
    datum_field_or_initial_record(state, record, field)
}

pub(crate) fn datum_field_or_initial_record(
    state: &ExecutionState,
    record: &dm_value::Datum,
    field: &FieldName,
) -> Result<Value, ValueError> {
    if let Some(value) = record.field_optional(field) {
        return Ok(value.clone());
    }
    initial_value_or_engine_root(state, record.type_path(), field)
        .ok_or_else(|| ValueError::MissingField(field.clone()))
}

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

/// Reads an ordinary datum member, falling back to inherited type-static
/// storage only when no instance member of that name exists. BYOND permits
/// shared/static variables to be accessed through dynamically typed datum
/// expressions, so compile-time binding is an optimization rather than a
/// correctness requirement.
pub(crate) fn datum_field_or_shared(
    state: &ExecutionState,
    datum: DatumId,
    field: &FieldName,
) -> Result<Value, ValueError> {
    let record = state.heap.datum(datum)?;
    datum_field_or_shared_record(state, record, field)
}

pub(crate) fn datum_field_or_shared_record(
    state: &ExecutionState,
    record: &dm_value::Datum,
    field: &FieldName,
) -> Result<Value, ValueError> {
    match datum_field_or_initial_record(state, record, field) {
        Ok(value) => Ok(value),
        Err(error @ ValueError::MissingField(_)) => {
            let Some(storage) = state
                .shared_fields
                .get(record.type_path())
                .and_then(|fields| fields.get(field))
                .cloned()
            else {
                return Err(error);
            };
            Ok(state.global(&storage).cloned().unwrap_or(Value::Null))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn assign_datum_or_shared_field(
    state: &mut ExecutionState,
    datum: DatumId,
    field: FieldName,
    value: Value,
) -> Result<(), String> {
    if matches!(
        datum_field_or_initial(state, datum, &field),
        Err(ValueError::MissingField(_))
    ) && let Some(storage) = datum_shared_storage(state, datum, &field)
    {
        state.set_global(storage, value);
        return Ok(());
    }
    assign_datum_field(state, datum, field, value)
}

pub(crate) fn assign_datum_field(
    state: &mut ExecutionState,
    datum: DatumId,
    field: FieldName,
    value: Value,
) -> Result<(), String> {
    if matches!(field.as_str(), "vis_contents" | "vis_locs") {
        let is_vis_contents = field.as_str() == "vis_contents";
        let replacement = match &value {
            Value::Null => Vec::new(),
            Value::Datum(member) => vec![*member],
            Value::List(source) => state
                .heap
                .list(*source)
                .map_err(|error| error.to_string())?
                .positions()
                .filter_map(|(_, value)| match value {
                    Value::Datum(member) => Some(*member),
                    Value::Null => None,
                    _ => None,
                })
                .collect(),
            value => {
                return Err(format!(
                    "{} assignment requires an atom, list, or null, received {value}",
                    field.as_str()
                ));
            }
        };
        for member in &replacement {
            let is_atom = state.heap.datum(*member).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/atom"
                    || path.starts_with("/atom/")
                    || path == "/obj"
                    || path.starts_with("/obj/")
                    || path == "/mob"
                    || path.starts_with("/mob/")
                    || path == "/turf"
                    || path.starts_with("/turf/")
                    || path == "/area"
                    || path.starts_with("/area/")
            });
            if !is_atom {
                return Err(format!("{} can only contain atoms", field.as_str()));
            }
        }
        let list = state.ensure_visibility_list(datum, is_vis_contents)?;
        let before = state.visibility_members(list)?;
        {
            let target = state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?;
            target.resize(0).map_err(|error| error.to_string())?;
            for member in replacement {
                let value = Value::Datum(member);
                if !target.contains(&value) {
                    target.add(value);
                }
            }
        }
        state.synchronize_visibility_list(list, &before)?;
        return Ok(());
    }
    let is_savefile = state.heap.datum(datum).is_ok_and(|datum| {
        let path = datum.type_path().as_str();
        path == "/savefile" || path.starts_with("/savefile/")
    });
    if is_savefile && field.as_str() == "cd" {
        let requested = match &value {
            Value::Text(value) => value.as_ref(),
            value => return Err(format!("savefile.cd requires text, received {value}")),
        };
        let current = state.savefiles.entry(datum).or_default().cd.clone();
        state.savefiles.entry(datum).or_default().cd = savefile_resolve_path(&current, requested);
    }
    let is_world = state
        .heap
        .datum(datum)
        .is_ok_and(|datum| datum.type_path().as_str() == "/world");
    if field.as_str() == "loc" {
        let is_image = state.heap.datum(datum).is_ok_and(|datum| {
            let path = datum.type_path().as_str();
            path == "/image"
                || path.starts_with("/image/")
                || path == "/mutable_appearance"
                || path.starts_with("/mutable_appearance/")
        });
        let old_loc = state
            .heap
            .datum_field(datum, &field)
            .ok()
            .and_then(|value| match value {
                Value::Datum(loc) => Some(*loc),
                _ => None,
            });
        let new_loc = match &value {
            Value::Datum(loc) => Some(*loc),
            Value::Null => None,
            value => {
                return Err(format!(
                    "loc assignment requires a datum or null, received {value}"
                ));
            }
        };
        let is_movable = state
            .heap
            .datum(datum)
            .is_ok_and(|datum| builtins::is_movable_path(datum.type_path().as_str()));
        let new_loc_is_turf = new_loc.is_some_and(|loc| {
            state.heap.datum(loc).is_ok_and(|datum| {
                let path = datum.type_path().as_str();
                path == "/turf" || path.starts_with("/turf/")
            })
        });
        if is_movable && new_loc_is_turf {
            builtins::move_movable_to_turf(state, datum, new_loc.expect("turf loc exists"))?;
            return Ok(());
        }
        if is_movable && let Some(new_loc) = new_loc {
            builtins::move_movable_to_atom(state, datum, new_loc)?;
            return Ok(());
        }
        // `/image.loc` is only the visual context used for client rendering.
        // Images are not physical atoms/movables and must never enter the
        // target's `contents` list. OpenDream stores this in DreamObjectImage's
        // private `_loc` without calling DreamObjectMovable.SetLoc.
        if !is_image && old_loc != new_loc {
            builtins::synchronize_moved_atom_contents(state, datum, old_loc, new_loc)?;
        }
    }
    if is_world && matches!(field.as_str(), "maxx" | "maxy" | "maxz") {
        let requested = value
            .as_number()
            .ok_or_else(|| format!("world.{} requires a numeric value", field.as_str()))?;
        if !requested.is_finite()
            || requested.fract() != 0.0
            || requested < 1.0
            || requested > i32::MAX as f32
        {
            return Err(format!(
                "world.{} must be a positive integer, received {requested}",
                field.as_str()
            ));
        }
        #[allow(clippy::cast_possible_truncation)]
        let requested = requested as i32;
        let mut dimensions = (
            state.world_dimension(datum, "maxx")?,
            state.world_dimension(datum, "maxy")?,
            state.world_dimension(datum, "maxz")?,
        );
        match field.as_str() {
            "maxx" => dimensions.0 = requested,
            "maxy" => dimensions.1 = requested,
            "maxz" => dimensions.2 = requested,
            _ => unreachable!(),
        }
        state.resize_world_geometry(datum, dimensions)?;
    }
    let client_mob_assignment =
        if field.as_str() == "mob" && state.client_sessions.contains_key(&datum) {
            match &value {
                Value::Datum(mob) => {
                    let path = state
                        .heap
                        .datum(*mob)
                        .map_err(|error| error.to_string())?
                        .type_path()
                        .as_str();
                    if path != "/mob" && !path.starts_with("/mob/") {
                        return Err("client.mob assignment requires a /mob or null".to_owned());
                    }
                    Some(Some(*mob))
                }
                Value::Null => Some(None),
                _ => return Err("client.mob assignment requires a /mob or null".to_owned()),
            }
        } else {
            None
        };
    state
        .heap
        .set_datum_field(datum, field.clone(), value.clone())
        .map_err(|error| error.to_string())?;
    // A live BYOND connection follows `client.mob` across lobby, observer,
    // character, and respawn handoffs. Keep the host-side attachment in lockstep
    // with the DM field so snapshots, clicks, movement, and verb dispatch never
    // continue targeting the previous mob.
    if let Some(client_mob_assignment) = client_mob_assignment {
        match client_mob_assignment {
            Some(mob) => {
                state.local_client_mobs.insert(datum, mob);
                if let Some(module) = state.instance_initializer_module.clone() {
                    state.populate_local_verb_inventory(&module, mob)?;
                }
            }
            None => {
                state.local_client_mobs.remove(&datum);
            }
        }
    }
    if is_world {
        let reciprocal = match (field.as_str(), value.as_number()) {
            ("tick_lag", Some(value)) if value.is_finite() && value > 0.0 => {
                Some(("fps", 10.0 / value))
            }
            ("fps", Some(value)) if value.is_finite() && value > 0.0 => {
                Some(("tick_lag", 10.0 / value))
            }
            _ => None,
        };
        if let Some((field, reciprocal)) = reciprocal {
            let _ = state.heap.set_datum_field(
                datum,
                FieldName::parse(field).expect("built-in world timing field"),
                Value::number(reciprocal),
            );
        }
    }
    Ok(())
}

pub(crate) fn write_datum_vars(
    state: &mut ExecutionState,
    datum: DatumId,
    list: ListId,
    key: Value,
    value: Value,
) -> Result<(), String> {
    let Value::Text(name) = &key else {
        return Err("datum.vars writes require a text key".to_owned());
    };
    let field = FieldName::parse(name).map_err(|error| error.to_string())?;
    if let Some(storage) = datum_shared_storage(state, datum, &field) {
        state.set_global(storage, value.clone());
    } else {
        assign_datum_field(state, datum, field, value.clone())?;
    }
    write_list_value(&mut state.heap, list, key, value, false).map_err(|error| error.to_string())
}

pub(crate) fn write_list_value(
    heap: &mut ValueHeap,
    list: ListId,
    key: Value,
    value: Value,
    associative: bool,
) -> Result<(), ValueError> {
    let values = heap.list_mut(list)?;
    if matches!(key, Value::Number(_)) && !associative {
        let index = value_to_list_index(&key).map_err(ValueError::InvalidListIndex)?;
        // BYOND's list-index assignment grows by one when targeting
        // `list.len + 1`. OpenDream's opcode path likewise calls SetValue with
        // `allowGrowth: true`; helpers such as orange() use this as append.
        if index == values.len() + 1 {
            values.add(value);
        } else {
            values.set(index, value)?;
        }
    } else {
        values.set_key(key, value);
    }
    Ok(())
}

pub(crate) fn value_to_list_index(value: &Value) -> Result<usize, String> {
    let Some(number) = value.as_number() else {
        return Err(format!("list index must be numeric, received {value}"));
    };
    let index = number.trunc();
    if !index.is_finite() || index < 1.0 {
        return Err(format!(
            "list index must truncate to a positive number, received {number}"
        ));
    }
    if index >= i32::MAX as f32 {
        return Err(format!("list index {number} exceeds the BYOND index range"));
    }
    Ok(index as usize)
}

pub(crate) fn indexed_text_character(text: &str, index: usize) -> Value {
    let character = if text.is_ascii() {
        text.as_bytes()
            .get(index.saturating_sub(1))
            .copied()
            .map(char::from)
    } else {
        text.chars().nth(index.saturating_sub(1))
    };
    character.map_or(Value::Null, |value| Value::text(value.to_string()))
}

pub(crate) fn pop(stack: &mut SmallVec<[Value; 8]>) -> Result<Value, String> {
    stack
        .pop()
        .ok_or_else(|| "bytecode stack underflow".to_owned())
}

pub(crate) fn pop_builtin_arguments(
    stack: &mut SmallVec<[Value; 8]>,
    count: usize,
) -> SmallVec<[Value; 8]> {
    let start = stack.len() - count;
    stack.drain(start..).collect()
}

pub(crate) fn allocate_dm_array(heap: &mut ValueHeap, sizes: &[usize], depth: usize) -> ListId {
    let list = heap.allocate_list();
    for _ in 0..sizes.get(depth).copied().unwrap_or(0) {
        let value = if depth + 1 < sizes.len() {
            Value::List(allocate_dm_array(heap, sizes, depth + 1))
        } else {
            Value::Null
        };
        heap.list_mut(list)
            .expect("new array list is live")
            .add(value);
    }
    list
}

pub(crate) fn execute_animate(
    names: &[Option<String>],
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let explicit_target = names
        .iter()
        .zip(arguments)
        .find(|(name, _)| name.is_none())
        .map(|(_, value)| value.clone());
    let target = explicit_target
        .clone()
        .or_else(|| state.last_animation_target.clone());
    if let Some(target) = explicit_target {
        state.last_animation_target = Some(target);
    }
    let Some(Value::Datum(target)) = target else {
        // Rendering-only calls against null or unsupported client-side values
        // have no persistent effect in a headless world.
        return Ok(Value::Null);
    };

    const CONTROL_ARGUMENTS: &[&str] = &[
        "time",
        "loop",
        "easing",
        "flags",
        "delay",
        "tag",
        "command",
        "appearance",
        "var_list",
        "object",
    ];
    for (name, value) in names.iter().zip(arguments) {
        let Some(name) = name else { continue };
        if CONTROL_ARGUMENTS.contains(&name.to_ascii_lowercase().as_str()) {
            if name.eq_ignore_ascii_case("var_list") {
                let Value::List(list) = value else { continue };
                let fields = state
                    .heap
                    .list(*list)
                    .map_err(|error| error.to_string())?
                    .associations()
                    .filter_map(|(key, value)| match key {
                        Value::Text(key) => {
                            FieldName::parse(key).ok().map(|key| (key, value.clone()))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let datum = state
                    .heap
                    .datum_mut(target)
                    .map_err(|error| error.to_string())?;
                for (field, value) in fields {
                    datum.set_field(field, value);
                }
            }
            continue;
        }
        let field = FieldName::parse(name).map_err(|error| error.to_string())?;
        state
            .heap
            .datum_mut(target)
            .map_err(|error| error.to_string())?
            .set_field(field, value.clone());
    }
    Ok(Value::Null)
}

pub(crate) fn scalar_number_string(value: Value) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

pub(crate) fn mutate_scalar_value(
    value: Value,
    delta: i8,
    prefix: bool,
) -> Result<(Value, Value), String> {
    let old_result = value.clone();
    let old_number = match value {
        Value::Null | Value::Text(_) => 0.0,
        Value::Number(number) => number.to_f32(),
        value => {
            return Err(format!(
                "increment/decrement requires a scalar value, received {value}"
            ));
        }
    };
    let updated = Value::number(old_number + f32::from(delta));
    let result = if prefix { updated.clone() } else { old_result };
    Ok((result, updated))
}

pub(crate) fn execute_scalar_add(left: Value, right: Value) -> Result<Value, String> {
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(Value::number(left.to_f32() + right.to_f32()))
        }
        (Value::Null, Value::Number(right)) => Ok(Value::number(right.to_f32())),
        (Value::Number(left), Value::Null) => Ok(Value::number(left.to_f32())),
        (Value::Null, Value::Null) => Ok(Value::number(0.0)),
        // A declaration-only `/list` variable begins as null, and BYOND's
        // `field += list(value)` idiom initializes it to that list. Logging
        // queues and many SS13 lazy collections depend on this coercion.
        (Value::Null, right @ Value::List(_)) => Ok(right),
        (Value::Null, Value::Text(right)) => Ok(Value::Text(right)),
        (Value::Text(left), Value::Null) => Ok(Value::Text(left)),
        (Value::Text(left), Value::Text(right)) => Ok(Value::text(format!("{left}{right}"))),
        (Value::Null, right) => Ok(right),
        (left, Value::Null) => Ok(left),
        (left, right) => Err(format!(
            "addition requires compatible DM values, received {left} and {right}"
        )),
    }
}

pub(crate) fn execute_scalar_compound_assignment(
    operator: CompoundAssignmentOperator,
    left: Value,
    right: Value,
) -> Result<Value, String> {
    if matches!(operator, CompoundAssignmentOperator::Add)
        && matches!((&left, &right), (Value::Null, Value::List(_)))
    {
        return Ok(right);
    }
    if matches!(operator, CompoundAssignmentOperator::Add)
        && matches!(
            (&left, &right),
            (Value::Text(_) | Value::Null, Value::Text(_)) | (Value::Text(_), Value::Null)
        )
    {
        return execute_scalar_add(left, right);
    }
    if matches!(operator, CompoundAssignmentOperator::Add)
        && (matches!(&left, Value::Null) || matches!(&right, Value::Null))
    {
        return execute_scalar_add(left, right);
    }
    let left = scalar_number_string(left)?;
    let right = scalar_number_string(right)?;
    let value = match operator {
        CompoundAssignmentOperator::Add => left + right,
        CompoundAssignmentOperator::Subtract => left - right,
        CompoundAssignmentOperator::Multiply => left * right,
        CompoundAssignmentOperator::Divide => left / right,
        CompoundAssignmentOperator::Remainder => integer_remainder(left, right),
        CompoundAssignmentOperator::FractionalRemainder => fractional_remainder(left, right),
        CompoundAssignmentOperator::BitAnd => bitwise_binary(left, right, |a, b| a & b),
        CompoundAssignmentOperator::BitOr => bitwise_binary(left, right, |a, b| a | b),
        CompoundAssignmentOperator::BitXor => bitwise_binary(left, right, |a, b| a ^ b),
        CompoundAssignmentOperator::ShiftLeft => {
            bitwise_shift(left, right, |left, right| left << right)
        }
        CompoundAssignmentOperator::ShiftRight => {
            bitwise_shift(left, right, |left, right| left >> right)
        }
    };
    Ok(Value::number(value))
}

pub(crate) fn pop_number(stack: &mut SmallVec<[Value; 8]>) -> Result<f32, String> {
    let value = pop(stack)?;
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        value => Err(format!("numeric operation received {value}")),
    }
}

/// Resolves a compact static call count or the count marker produced by an
/// immediately preceding `arglist()` expansion.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn runtime_argument_count(
    stack: &mut SmallVec<[Value; 8]>,
    encoded: u16,
) -> Result<usize, String> {
    if encoded != EXPANDED_ARGUMENT_COUNT {
        return Ok(usize::from(encoded));
    }
    let value = stack
        .pop()
        .ok_or_else(|| "bytecode stack underflow".to_owned())?;
    let Value::Number(number) = value else {
        return Err("expanded call argument count is not numeric".to_owned());
    };
    let count = number.to_f32();
    if !count.is_finite() || count < 0.0 || count > f32::from(u16::MAX) || count.fract() != 0.0 {
        return Err("expanded call argument count is invalid".to_owned());
    }
    Ok(count as usize)
}

/// Places `image()` arguments into `/image/New(icon, loc, icon_state, layer,
/// dir, pixel_x, pixel_y)` slots.
///
/// BYOND has one constructor-specific positional rule: when the second
/// argument is text it is `icon_state`, not `loc`, and every following
/// positional argument shifts over one slot. `OpenDream` performs the same
/// reordering for keyed calls before `DreamObjectImage.Initialize` observes
/// them. Keeping it at the VM call boundary means the native image constructor
/// can apply copied appearance state first and explicit overrides second.
pub(crate) fn order_image_arguments(
    arguments: &[Value],
    argument_names: &[Option<String>],
) -> Result<Vec<Value>, String> {
    const NAMES: [&str; 7] = [
        "icon",
        "loc",
        "icon_state",
        "layer",
        "dir",
        "pixel_x",
        "pixel_y",
    ];
    if arguments.len() != argument_names.len() {
        return Err("image argument metadata does not match its value count".to_owned());
    }
    let mut ordered = vec![Value::Null; NAMES.len().max(arguments.len())];
    let mut skipping_location = false;
    for (index, (name, value)) in argument_names.iter().zip(arguments).enumerate() {
        let destination = if let Some(name) = name {
            NAMES
                .iter()
                .position(|candidate| candidate == name)
                .ok_or_else(|| format!("image has no argument named {name:?}"))?
        } else {
            if index == 1 && matches!(value, Value::Text(_)) {
                skipping_location = true;
            }
            index + usize::from(skipping_location)
        };
        if destination >= ordered.len() {
            return Err("image has too many constructor arguments".to_owned());
        }
        ordered[destination] = value.clone();
    }
    Ok(ordered)
}
