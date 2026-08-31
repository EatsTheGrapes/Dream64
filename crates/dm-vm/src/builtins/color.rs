//! RGB/HSV/HSL color construction, `gradient`, `rgb2num`, and the
//! `time2text`/`civil_from_days_since_2000` calendar conversions.

use std::fmt::Write;

use dm_value::Value;

use super::{ExecutionState, number, strict_text};
fn color_byte(value: &Value, context: &str) -> Result<u8, String> {
    Ok(number(value, context)?.round().clamp(0.0, 255.0) as u8)
}

pub(super) fn rgb_builtin(arguments: &[Value]) -> Result<Value, String> {
    let r = color_byte(&arguments[0], "rgb red")?;
    let g = color_byte(&arguments[1], "rgb green")?;
    let b = color_byte(&arguments[2], "rgb blue")?;
    // The fifth positional argument is color space. RGB is the native/default
    // space; conversion of alternate spaces is kept explicit rather than
    // silently producing the wrong color.
    if arguments.len() == 5 && arguments[4].as_number().is_some_and(|space| space != 0.0) {
        return Err("rgb alternate color spaces are not implemented".to_owned());
    }
    if let Some(alpha) = arguments.get(3) {
        Ok(Value::text(format!(
            "#{r:02x}{g:02x}{b:02x}{:02x}",
            color_byte(alpha, "rgb alpha")?
        )))
    } else {
        Ok(Value::text(format!("#{r:02x}{g:02x}{b:02x}")))
    }
}

pub(crate) fn parse_hex_color(text: &str) -> Option<Vec<u8>> {
    let hex = text.strip_prefix('#')?;
    let expanded = match hex.len() {
        3 | 4 => hex.chars().flat_map(|c| [c, c]).collect::<String>(),
        6 | 8 => hex.to_owned(),
        _ => return None,
    };
    (0..expanded.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&expanded[index..index + 2], 16).ok())
        .collect()
}

pub(super) fn rgb2num_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    // BYOND applies rgb2num's documented default white color when the color
    // argument is null. OpenDream's conformance fixture explicitly verifies
    // rgb2num(null) == rgb2num("#fff").
    let text = if arguments[0] == Value::Null {
        "#FFFFFF".to_owned()
    } else {
        strict_text(&arguments[0], state, "rgb2num")?
    };
    let components =
        parse_hex_color(&text).ok_or_else(|| format!("rgb2num invalid color {text:?}"))?;
    let space = arguments.get(1).and_then(Value::as_number).unwrap_or(0.0);
    let converted = match space as i32 {
        0 => components[..3]
            .iter()
            .map(|component| f32::from(*component))
            .collect::<Vec<_>>(),
        1 | 2 => {
            let red = f32::from(components[0]) / 255.0;
            let green = f32::from(components[1]) / 255.0;
            let blue = f32::from(components[2]) / 255.0;
            let maximum = red.max(green).max(blue);
            let minimum = red.min(green).min(blue);
            let delta = maximum - minimum;
            let hue = if delta == 0.0 {
                0.0
            } else if maximum == red {
                60.0 * ((green - blue) / delta).rem_euclid(6.0)
            } else if maximum == green {
                60.0 * ((blue - red) / delta + 2.0)
            } else {
                60.0 * ((red - green) / delta + 4.0)
            };
            if space as i32 == 1 {
                vec![
                    hue,
                    if maximum == 0.0 {
                        0.0
                    } else {
                        delta / maximum * 100.0
                    },
                    maximum * 100.0,
                ]
            } else {
                let lightness = f32::midpoint(maximum, minimum);
                vec![
                    hue,
                    if delta == 0.0 {
                        0.0
                    } else {
                        delta / (1.0 - (2.0 * lightness - 1.0).abs()) * 100.0
                    },
                    lightness * 100.0,
                ]
            }
        }
        _ => return Err(format!("rgb2num invalid color space {space}")),
    };
    let id = state.heap.allocate_list();
    let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
    for component in converted {
        list.add(Value::number(component));
    }
    if let Some(alpha) = components.get(3) {
        list.add(Value::number(f32::from(*alpha)));
    }
    Ok(Value::List(id))
}

pub(super) fn gradient_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let mut index = number(arguments.last().expect("gradient arity"), "gradient index")?;
    let items = &arguments[..arguments.len() - 1];
    let mut stops = Vec::new();
    let mut looping = false;
    if items
        .first()
        .is_some_and(|value| value.as_number().is_some())
    {
        let mut cursor = 0;
        while cursor + 1 < items.len() {
            let Some(position) = items[cursor].as_number() else {
                break;
            };
            if !matches!(items[cursor + 1], Value::Text(_)) {
                break;
            }
            stops.push((position, &items[cursor + 1]));
            cursor += 2;
        }
        looping = items[cursor..]
            .iter()
            .any(|value| matches!(value, Value::Text(text) if text.eq_ignore_ascii_case("loop")));
    } else {
        let colors = items
            .iter()
            .filter(|value| matches!(value, Value::Text(_)))
            .collect::<Vec<_>>();
        let divisor = colors.len().saturating_sub(1).max(1) as f32;
        stops.extend(
            colors
                .into_iter()
                .enumerate()
                .map(|(i, color)| (i as f32 / divisor, color)),
        );
    }
    if stops.len() < 2 {
        return Err("gradient requires at least two color stops".to_owned());
    }
    let first = stops[0].0;
    let last = stops[stops.len() - 1].0;
    if looping && last > first {
        index = (index - first).rem_euclid(last - first) + first;
    }
    let segment = stops
        .windows(2)
        .position(|pair| index <= pair[1].0)
        .unwrap_or(stops.len() - 2);
    let (left_at, left_value) = stops[segment];
    let (right_at, right_value) = stops[segment + 1];
    let amount = if right_at == left_at {
        0.0
    } else {
        (index - left_at) / (right_at - left_at)
    };
    let left = parse_hex_color(&strict_text(left_value, state, "gradient color")?)
        .ok_or_else(|| "gradient requires hexadecimal colors".to_owned())?;
    let right = parse_hex_color(&strict_text(right_value, state, "gradient color")?)
        .ok_or_else(|| "gradient requires hexadecimal colors".to_owned())?;
    let count = left.len().max(right.len());
    let mut output = String::from("#");
    for component in 0..count {
        let a = f32::from(*left.get(component).unwrap_or(&255));
        let b = f32::from(*right.get(component).unwrap_or(&255));
        write!(output, "{:02x}", (a + (b - a) * amount).round() as u8).unwrap();
    }
    Ok(Value::text(output))
}

pub(super) fn time2text_builtin(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Value, String> {
    let ticks = number(&arguments[0], "time2text timestamp")? as i64;
    let format = arguments.get(1).map_or_else(
        || Ok("DDD MMM DD hh:mm:ss YYYY".to_owned()),
        |value| strict_text(value, state, "time2text format"),
    )?;
    let timezone = arguments.get(2).and_then(Value::as_number).unwrap_or(0.0);
    let seconds = ticks.div_euclid(10) + (timezone * 3600.0) as i64;
    let days = seconds.div_euclid(86_400);
    let day_seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days_since_2000(days);
    let weekdays = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"];
    let weekday_names = [
        "Saturday",
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ];
    let months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let hour = day_seconds / 3600;
    let minute = day_seconds / 60 % 60;
    let second = day_seconds % 60;
    let mut out = format;
    for (token, value) in [
        ("YYYY", format!("{year:04}")),
        ("Month", month_names[month - 1].to_owned()),
        ("DDD", weekdays[days.rem_euclid(7) as usize].to_owned()),
        ("Day", weekday_names[days.rem_euclid(7) as usize].to_owned()),
        ("MMM", months[month - 1].to_owned()),
        ("YY", format!("{:02}", year % 100)),
        ("MM", format!("{month:02}")),
        ("DD", format!("{day:02}")),
        ("hh", format!("{hour:02}")),
        ("mm", format!("{minute:02}")),
        ("ss", format!("{second:02}")),
    ] {
        out = out.replace(token, &value);
    }
    Ok(Value::text(out))
}

fn civil_from_days_since_2000(days: i64) -> (i64, usize, i64) {
    // Howard Hinnant's civil date algorithm, offset from 1970 to 2000.
    let z = days + 10_957 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month as usize, day)
}
