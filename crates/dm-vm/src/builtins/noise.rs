//! `rust_g` cellular and Poisson noise generators exposed as `_rust_g`*
//! procedures, hosted in the kernel thread for forward-compatible parallellism.

use std::collections::HashSet;

use dm_value::Value;

use super::{ExecutionState, strict_text};
pub(super) fn rust_g_cellular_noise(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    fn parse_number(value: &Value, state: &ExecutionState, name: &str) -> Result<f64, String> {
        let text = strict_text(value, state, name)?;
        text.parse::<f64>()
            .map_err(|error| format!("cnoise_generate {name} is invalid: {error}"))
    }

    fn parse_dimension(value: &Value, state: &ExecutionState, name: &str) -> Result<usize, String> {
        let number = parse_number(value, state, name)?;
        if !number.is_finite() || number.fract() != 0.0 || !(1.0..=4_096.0).contains(&number) {
            return Err(format!(
                "cnoise_generate {name} must be a whole number from 1 through 4096"
            ));
        }
        Ok(number as usize)
    }

    fn parse_count(value: &Value, state: &ExecutionState, name: &str) -> Result<usize, String> {
        let number = parse_number(value, state, name)?;
        if !number.is_finite() || number.fract() != 0.0 || !(0.0..=4_096.0).contains(&number) {
            return Err(format!(
                "cnoise_generate {name} must be a whole number from 0 through 4096"
            ));
        }
        Ok(number as usize)
    }

    fn parse_neighbour_limit(
        value: &Value,
        state: &ExecutionState,
        name: &str,
    ) -> Result<u8, String> {
        let number = parse_number(value, state, name)?;
        if !number.is_finite() || number.fract() != 0.0 || !(0.0..=8.0).contains(&number) {
            return Err(format!(
                "cnoise_generate {name} must be a whole number from 0 through 8"
            ));
        }
        Ok(number as u8)
    }

    let percentage = parse_number(&arguments[0], state, "percentage")?;
    if !percentage.is_finite() || !(0.0..=100.0).contains(&percentage) {
        return Err("cnoise_generate percentage must be from 0 through 100".to_owned());
    }
    let smoothing = parse_count(&arguments[1], state, "smoothing_iterations")?;
    let birth_limit = parse_neighbour_limit(&arguments[2], state, "birth_limit")?;
    let death_limit = parse_neighbour_limit(&arguments[3], state, "death_limit")?;
    let width = parse_dimension(&arguments[4], state, "width")?;
    let height = parse_dimension(&arguments[5], state, "height")?;
    let cells = width
        .checked_mul(height)
        .ok_or_else(|| "cnoise_generate dimensions overflow the host index range".to_owned())?;
    if cells > 16_777_216 {
        return Err("cnoise_generate dimensions exceed the 16,777,216-cell limit".to_owned());
    }

    let mut current = Vec::with_capacity(cells);
    for _ in 0..cells {
        current.push(
            f64::from(super::deterministic_unit(&mut state.random_state)) * 100.0 < percentage,
        );
    }
    let mut next = vec![false; cells];
    for _ in 0..smoothing {
        for y in 0..height {
            for x in 0..width {
                let mut neighbours = 0_u8;
                for dy in -1_isize..=1 {
                    for dx in -1_isize..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        if nx >= 0
                            && ny >= 0
                            && nx < width as isize
                            && ny < height as isize
                            && current[ny as usize * width + nx as usize]
                        {
                            neighbours = neighbours.saturating_add(1);
                        }
                    }
                }
                let index = y * width + x;
                next[index] = if current[index] {
                    neighbours >= death_limit
                } else {
                    neighbours > birth_limit
                };
            }
        }
        std::mem::swap(&mut current, &mut next);
    }

    let mut output = String::with_capacity(cells);
    output.extend(
        current
            .into_iter()
            .map(|closed| if closed { '1' } else { '0' }),
    );
    Ok(Value::text(output))
}

pub(super) fn rust_g_poisson_noise(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    use fast_poisson::Poisson2D;

    let parse = |index: usize, name: &str| {
        strict_text(&arguments[index], state, name)
            .map_err(|error| format!("noise_poisson_map {name} is invalid: {error}"))
    };
    let seed = parse(0, "seed")?
        .parse::<u64>()
        .map_err(|error| format!("noise_poisson_map seed is invalid: {error}"))?;
    let width = parse(1, "width")?
        .parse::<usize>()
        .map_err(|error| format!("noise_poisson_map width is invalid: {error}"))?;
    let height = parse(2, "height")?
        .parse::<usize>()
        .map_err(|error| format!("noise_poisson_map height is invalid: {error}"))?;
    let radius = parse(3, "radius")?
        .parse::<f32>()
        .map_err(|error| format!("noise_poisson_map radius is invalid: {error}"))?;
    let cells = width
        .checked_mul(height)
        .ok_or_else(|| "noise_poisson_map dimensions overflow the host index range".to_owned())?;
    if cells > 16_777_216 {
        return Err("noise_poisson_map dimensions exceed the 16,777,216-cell limit".to_owned());
    }

    // Keep this construction identical to rust-g's poissonnoise export. The
    // iterator yields floating points; rust-g truncates both coordinates and
    // then collapses the set into a row-major binary string.
    let points: HashSet<(usize, usize)> = Poisson2D::new()
        .with_dimensions([width as f32, height as f32], radius)
        .with_seed(seed)
        .iter()
        .map(|[x, y]| (x as usize, y as usize))
        .collect();
    let mut output = String::with_capacity(cells);
    for y in 0..height {
        for x in 0..width {
            output.push(if points.contains(&(x, y)) { '1' } else { '0' });
        }
    }
    Ok(Value::text(output))
}
