//! `/datum/...` resource construction and the `generator` distribution
//! procedures sharing the kernel PRNG seed.

use std::fmt::Write;

use dm_value::{DatumId, FieldName, TypePath, Value};

use crate::{allocate_vector, vector_components};

use super::{ExecutionState, parse_hex_color, value_text};
pub(super) fn resource_datum_builtin(
    path: &str,
    fields: &[&str],
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let path = TypePath::parse(path).map_err(|error| error.to_string())?;
    let datum = state.heap_mut().allocate_datum(path.clone());
    state.seed_native_datum_defaults(datum, &path)?;
    for (field, value) in fields.iter().zip(arguments) {
        state
            .heap_mut()
            .set_datum_field(
                datum,
                FieldName::parse(field).map_err(|error| error.to_string())?,
                value.clone(),
            )
            .map_err(|error| error.to_string())?;
    }
    Ok(Value::Datum(datum))
}

fn generator_field(
    generator: DatumId,
    name: &str,
    state: &ExecutionState,
) -> Result<Value, String> {
    super::datum_field_or_initial(
        state,
        generator,
        &FieldName::parse(name).expect("generator field is valid"),
    )
    .map_err(|error| error.to_string())
}

fn generator_distribution_sample(
    low: f32,
    high: f32,
    distribution: i32,
    state: &mut ExecutionState,
) -> f32 {
    let (low, high) = if low <= high {
        (low, high)
    } else {
        (high, low)
    };
    if low == high {
        return low;
    }
    let unit = super::deterministic_unit(&mut state.random_state);
    let factor = match distribution {
        1 => {
            // OpenDream models NORMAL_RAND with a normal distribution whose
            // finite interval spans six standard deviations, then clamps the
            // rare tails. Box-Muller keeps that contract deterministic here.
            let second = super::deterministic_unit(&mut state.random_state);
            let normal = (-2.0 * unit.max(f32::MIN_POSITIVE).ln()).sqrt()
                * (std::f32::consts::TAU * second).cos();
            return ((low + high) * 0.5 + normal * (high - low) / 6.0).clamp(low, high);
        }
        2 => unit.sqrt(),
        3 => unit.cbrt(),
        _ => unit,
    };
    low + factor * (high - low)
}

fn generator_vector_components(value: &Value, state: &ExecutionState) -> [f32; 3] {
    match value {
        Value::Datum(datum) => vector_components(*datum, state.heap()).unwrap_or([0.0; 3]),
        Value::List(list) => {
            let Ok(list) = state.heap().list(*list) else {
                return [0.0; 3];
            };
            std::array::from_fn(|index| {
                list.get(index + 1)
                    .ok()
                    .and_then(Value::as_number)
                    .unwrap_or(0.0)
            })
        }
        _ => [0.0; 3],
    }
}

pub(super) fn generator_rand_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Value::Datum(generator) = arguments[0] else {
        return Err("generator.Rand requires a /generator receiver".to_owned());
    };
    let path = state
        .heap()
        .datum(generator)
        .map_err(|error| error.to_string())?
        .type_path();
    if path.as_str() != "/generator" && !path.as_str().starts_with("/generator/") {
        return Err("generator.Rand requires a /generator receiver".to_owned());
    }

    let kind = generator_field(generator, "type", state)?;
    let kind = value_text(&kind)
        .ok_or_else(|| format!("invalid generator type {kind}"))?
        .to_owned();
    let low = generator_field(generator, "a", state).unwrap_or(Value::number(0.0));
    let high = generator_field(generator, "b", state).unwrap_or(Value::number(1.0));
    let distribution = generator_field(generator, "rand", state)
        .ok()
        .and_then(|value| value.as_number())
        .unwrap_or(0.0) as i32;

    match kind.as_str() {
        "num" => {
            let low = low.as_number().unwrap_or(0.0);
            let high = high.as_number().unwrap_or(1.0);
            Ok(Value::number(generator_distribution_sample(
                low,
                high,
                distribution,
                state,
            )))
        }
        "vector" | "box" => {
            let low = generator_vector_components(&low, state);
            let high = generator_vector_components(&high, state);
            let values = if kind == "vector" {
                let factor = generator_distribution_sample(0.0, 1.0, distribution, state);
                std::array::from_fn(|index| low[index] + (high[index] - low[index]) * factor)
            } else {
                std::array::from_fn(|index| {
                    generator_distribution_sample(low[index], high[index], distribution, state)
                })
            };
            allocate_vector(values, state.heap_mut()).map(Value::Datum)
        }
        "circle" | "sphere" => {
            let low = low.as_number().unwrap_or(0.0);
            let high = high.as_number().unwrap_or(1.0);
            let radius = generator_distribution_sample(low, high, distribution, state);
            let theta = super::deterministic_unit(&mut state.random_state) * std::f32::consts::TAU;
            let values = if kind == "circle" {
                [theta.cos() * radius, theta.sin() * radius, 0.0]
            } else {
                let phi = super::deterministic_unit(&mut state.random_state) * std::f32::consts::PI;
                [
                    theta.cos() * phi.sin() * radius,
                    theta.sin() * phi.sin() * radius,
                    phi.cos() * radius,
                ]
            };
            allocate_vector(values, state.heap_mut()).map(Value::Datum)
        }
        "square" | "cube" => {
            let low = generator_vector_components(&low, state).map(f32::abs);
            let high = generator_vector_components(&high, state).map(f32::abs);
            let mut values = std::array::from_fn(|index| {
                generator_distribution_sample(-high[index], high[index], distribution, state)
            });
            if values[0].abs() < low[0] {
                let sign = if super::deterministic_unit(&mut state.random_state) < 0.5 {
                    -1.0
                } else {
                    1.0
                };
                values[1] =
                    sign * generator_distribution_sample(low[1], high[1], distribution, state);
            }
            if kind == "cube" && values[1].abs() < low[1] {
                let sign = if super::deterministic_unit(&mut state.random_state) < 0.5 {
                    -1.0
                } else {
                    1.0
                };
                values[2] =
                    sign * generator_distribution_sample(low[2], high[2], distribution, state);
            } else if kind == "square" {
                values[2] = 0.0;
            }
            allocate_vector(values, state.heap_mut()).map(Value::Datum)
        }
        "color" => {
            let low_text = value_text(&low).unwrap_or("#000000");
            let high_text = value_text(&high).unwrap_or("#ffffff");
            let low = parse_hex_color(low_text)
                .ok_or_else(|| format!("invalid generator color {low_text:?}"))?;
            let high = parse_hex_color(high_text)
                .ok_or_else(|| format!("invalid generator color {high_text:?}"))?;
            let factor = generator_distribution_sample(0.0, 1.0, distribution, state);
            let alpha = low.len() == 4 || high.len() == 4;
            let component = |values: &[u8], index: usize, default: u8| {
                f32::from(values.get(index).copied().unwrap_or(default))
            };
            let components = (0..usize::from(3 + u8::from(alpha)))
                .map(|index| {
                    let left = component(&low, index, 255);
                    let right = component(&high, index, 255);
                    (left + (right - left) * factor).round().clamp(0.0, 255.0) as u8
                })
                .collect::<Vec<_>>();
            let mut output = String::from("#");
            for component in components {
                write!(output, "{component:02x}").expect("writing to a string cannot fail");
            }
            Ok(Value::text(output))
        }
        _ => Err(format!("invalid generator type {kind:?}")),
    }
}
