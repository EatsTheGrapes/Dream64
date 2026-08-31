//! Parallel atmos difference-snapshotting shared between `native` acceleration
//! and the `_dream64_atmos_*` standard builtins, plus small shared datum-field
//! helpers (`loc`, `contents`, coordinates) used across cluster modules.

use dm_value::{DatumId, FieldName, ListId, Value};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use crate::worker_lane::{
    AtmosCompareResult, AtmosCompareSnapshot, AtmosGasSample, compare_atmos_batch,
};

use super::{ExecutionState, number};
pub(super) fn atmos_field(name: &str) -> FieldName {
    FieldName::parse(name).expect("atmos built-in field is valid")
}

pub(super) fn builtin_loc_field() -> &'static FieldName {
    static FIELD: OnceLock<FieldName> = OnceLock::new();
    FIELD.get_or_init(|| FieldName::parse("loc").expect("built-in loc field is valid"))
}

pub(super) fn builtin_contents_field() -> &'static FieldName {
    static FIELD: OnceLock<FieldName> = OnceLock::new();
    FIELD.get_or_init(|| FieldName::parse("contents").expect("built-in contents field is valid"))
}

pub(super) fn builtin_coordinate_fields() -> &'static [FieldName; 3] {
    static FIELDS: OnceLock<[FieldName; 3]> = OnceLock::new();
    FIELDS.get_or_init(|| {
        ["x", "y", "z"]
            .map(|name| FieldName::parse(name).expect("built-in coordinate field is valid"))
    })
}

pub(super) fn atmos_setup_differences(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let (Value::List(difference_check), Value::List(active_turfs)) = (&arguments[0], &arguments[1])
    else {
        return Err("atmos difference batch requires turf and active lists".to_owned());
    };
    let thresholds = (
        number(&arguments[2], "minimum moles delta")?,
        number(&arguments[3], "minimum air ratio")?,
        number(&arguments[4], "minimum temperature delta")?,
    );
    let turfs = state
        .heap()
        .list(*difference_check)
        .map_err(|e| e.to_string())?
        .positions()
        .filter_map(|(_, value)| match value {
            Value::Datum(id) => Some(*id),
            _ => None,
        })
        .collect::<Vec<_>>();
    let positions = turfs
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect::<HashMap<_, _>>();
    let adjacent_field = atmos_field("atmos_adjacent_turfs");
    let air_field = atmos_field("air");
    let cycle_field = atmos_field("current_cycle");
    let mut owners = Vec::new();
    let mut jobs = Vec::new();
    for (potential_index, potential) in turfs.iter().copied().enumerate() {
        let Ok(Value::Datum(potential_air)) =
            crate::datum_field_or_initial(state, potential, &air_field)
        else {
            continue;
        };
        let Ok(Value::List(adjacent)) =
            crate::datum_field_or_initial(state, potential, &adjacent_field)
        else {
            continue;
        };
        let enemies = state
            .heap()
            .list(adjacent)
            .map_err(|e| e.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Datum(id) => Some(*id),
                _ => None,
            })
            .collect::<Vec<_>>();
        for enemy in enemies {
            if positions
                .get(&enemy)
                .is_some_and(|index| *index < potential_index)
            {
                continue;
            }
            if crate::datum_field_or_initial(state, enemy, &cycle_field)
                .ok()
                .and_then(|v| v.as_number())
                == Some(f32::NEG_INFINITY)
            {
                continue;
            }
            let Ok(Value::Datum(enemy_air)) =
                crate::datum_field_or_initial(state, enemy, &air_field)
            else {
                continue;
            };
            jobs.push(atmos_compare_snapshot(
                state,
                potential_air,
                enemy_air,
                thresholds,
            )?);
            owners.push((potential_index, potential, enemy));
        }
    }
    let workers = std::env::var("DREAM64_ATMOS_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    let results = compare_atmos_batch(&jobs, workers);
    let excited_field = atmos_field("excited");
    let mut activated = HashSet::new();
    for ((potential_index, potential, enemy), result) in owners.into_iter().zip(results) {
        if matches!(result, AtmosCompareResult::Compatible) || !activated.insert(potential_index) {
            continue;
        }
        for turf in [potential, enemy] {
            let excited = crate::datum_field_or_initial(state, turf, &excited_field)
                .ok()
                .is_some_and(|v| crate::runtime_truthy(state.heap(), &v).unwrap_or(false));
            if !excited {
                state
                    .heap_mut()
                    .set_datum_field(turf, excited_field.clone(), Value::number(1.0))
                    .map_err(|e| e.to_string())?;
                state
                    .heap_mut()
                    .list_mut(*active_turfs)
                    .map_err(|e| e.to_string())?
                    .add(Value::Datum(turf));
            }
        }
    }
    for turf in turfs {
        state
            .heap_mut()
            .set_datum_field(turf, cycle_field.clone(), Value::number(f32::NEG_INFINITY))
            .map_err(|e| e.to_string())?;
    }
    Ok(Value::Null)
}

fn atmos_compare_snapshot(
    state: &ExecutionState,
    cached: DatumId,
    sample: DatumId,
    thresholds: (f32, f32, f32),
) -> Result<AtmosCompareSnapshot, String> {
    let gases_field = atmos_field("gases");
    let temperature_field = atmos_field("temperature");
    let Ok(Value::List(cached_gases)) = crate::datum_field_or_initial(state, cached, &gases_field)
    else {
        return Err("cached gases is not a list".to_owned());
    };
    let Ok(Value::List(sample_gases)) = crate::datum_field_or_initial(state, sample, &gases_field)
    else {
        return Err("sample gases is not a list".to_owned());
    };
    let cached_values = state.heap().list(cached_gases).map_err(|e| e.to_string())?;
    let sample_values = state.heap().list(sample_gases).map_err(|e| e.to_string())?;
    let mut keys = cached_values
        .positions()
        .map(|(_, key)| key.clone())
        .collect::<Vec<_>>();
    for (_, key) in sample_values.positions() {
        if !keys.iter().any(|candidate| candidate.semantic_eq(key)) {
            keys.push(key.clone());
        }
    }
    let gas_value = |list: ListId, key: &Value| {
        state
            .heap()
            .list(list)
            .ok()
            .and_then(|v| v.get_key(key).ok())
            .and_then(|v| match v {
                Value::List(id) => state.heap().list(*id).ok(),
                _ => None,
            })
            .and_then(|v| v.get(1).ok())
            .and_then(Value::as_number)
            .unwrap_or(0.0)
    };
    let gases = keys
        .into_iter()
        .map(|key| AtmosGasSample {
            id: Arc::from(key.to_string()),
            cached: gas_value(cached_gases, &key),
            sample: gas_value(sample_gases, &key),
        })
        .collect();
    let temperature = |datum| {
        crate::datum_field_or_initial(state, datum, &temperature_field)
            .ok()
            .and_then(|v| v.as_number())
            .unwrap_or(0.0)
    };
    Ok(AtmosCompareSnapshot {
        gases,
        cached_temperature: temperature(cached),
        sample_temperature: temperature(sample),
        minimum_moles_delta: thresholds.0,
        minimum_air_ratio: thresholds.1,
        minimum_temperature_delta: thresholds.2,
    })
}
