//! Engine geometry / appearance datum operators: `/icon`, `/vector`, and
//! `/matrix`.
//!
//! Split out of `value_ops`: the constructors, field readers/writers, and
//! arithmetic (binary, compound-assignment, and method dispatch) for BYOND's
//! built-in geometry datum types, which share nothing with the scalar/list
//! value semantics in the parent module.

use crate::bytecode::CompoundAssignmentOperator;
use dm_value::{DatumId, FieldName, TypePath, Value, ValueHeap};

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
