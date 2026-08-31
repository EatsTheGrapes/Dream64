//! Dream Maker legacy `text()` template form, `\[` interpolation,
//! roman numerals, `\proper`/`\gender` macros and suffix handling.

use dm_value::{FieldName, Value};

use crate::{
    TEXT_MACRO_A, TEXT_MACRO_A_UPPER, TEXT_MACRO_IMPROPER, TEXT_MACRO_OBJECT, TEXT_MACRO_ORDINAL,
    TEXT_MACRO_PLURAL, TEXT_MACRO_POSSESSIVE, TEXT_MACRO_POSSESSIVE_ADJECTIVE,
    TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER, TEXT_MACRO_POSSESSIVE_UPPER, TEXT_MACRO_PROPER,
    TEXT_MACRO_REFLEXIVE, TEXT_MACRO_ROMAN, TEXT_MACRO_ROMAN_UPPER, TEXT_MACRO_SUBJECT,
    TEXT_MACRO_SUBJECT_UPPER, TEXT_MACRO_THE, TEXT_MACRO_THE_UPPER,
};

use super::{ExecutionState, runtime_text, value_text};

pub(super) fn text_template(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(Value::Text(template)) = arguments.first() else {
        return Err("text() expected a string as its first argument".to_owned());
    };
    let mut values = arguments[1..].iter();
    let mut output = String::with_capacity(template.len());
    let mut characters = template.chars().peekable();
    let mut holes = 0_usize;
    let mut pending_prefix = None;
    let mut previous_value = None;
    let mut previous_output_start = 0_usize;
    while let Some(character) = characters.next() {
        if matches!(
            character,
            TEXT_MACRO_THE
                | TEXT_MACRO_THE_UPPER
                | TEXT_MACRO_A
                | TEXT_MACRO_A_UPPER
                | TEXT_MACRO_PROPER
                | TEXT_MACRO_IMPROPER
                | TEXT_MACRO_ROMAN
                | TEXT_MACRO_ROMAN_UPPER
        ) {
            pending_prefix = Some(character);
            continue;
        }
        if matches!(
            character,
            TEXT_MACRO_ORDINAL
                | TEXT_MACRO_PLURAL
                | TEXT_MACRO_SUBJECT
                | TEXT_MACRO_SUBJECT_UPPER
                | TEXT_MACRO_POSSESSIVE_ADJECTIVE
                | TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER
                | TEXT_MACRO_OBJECT
                | TEXT_MACRO_REFLEXIVE
                | TEXT_MACRO_POSSESSIVE
                | TEXT_MACRO_POSSESSIVE_UPPER
        ) {
            apply_text_suffix(
                character,
                previous_value,
                previous_output_start,
                &mut output,
                state,
            )?;
            continue;
        }
        if character != '[' {
            output.push(character);
            continue;
        }

        let mut lookahead = characters.clone();
        let mut whitespace = String::new();
        while lookahead.peek().is_some_and(|value| value.is_whitespace()) {
            whitespace.push(lookahead.next().expect("peeked whitespace exists"));
        }
        if lookahead.next() != Some(']') {
            output.push('[');
            continue;
        }
        for _ in 0..=whitespace.chars().count() {
            characters.next();
        }
        let value = values
            .next()
            .ok_or_else(|| "text() has fewer arguments than template holes".to_owned())?;
        previous_output_start = output.len();
        output.push_str(&format_text_interpolation(
            value,
            pending_prefix.take(),
            state,
        )?);
        previous_value = Some(value);
        holes += 1;
    }
    if values.next().is_some() {
        return Err(format!(
            "text() has more arguments than template holes ({holes})"
        ));
    }
    Ok(Value::text(output))
}

fn is_text_format_marker(character: char) -> bool {
    matches!(
        character,
        TEXT_MACRO_THE
            | TEXT_MACRO_THE_UPPER
            | TEXT_MACRO_A
            | TEXT_MACRO_A_UPPER
            | TEXT_MACRO_PROPER
            | TEXT_MACRO_IMPROPER
            | TEXT_MACRO_ROMAN
            | TEXT_MACRO_ROMAN_UPPER
            | TEXT_MACRO_ORDINAL
            | TEXT_MACRO_PLURAL
            | TEXT_MACRO_SUBJECT
            | TEXT_MACRO_SUBJECT_UPPER
            | TEXT_MACRO_POSSESSIVE_ADJECTIVE
            | TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER
            | TEXT_MACRO_OBJECT
            | TEXT_MACRO_REFLEXIVE
            | TEXT_MACRO_POSSESSIVE
            | TEXT_MACRO_POSSESSIVE_UPPER
    )
}

fn text_macro_visible(value: &Value, state: &ExecutionState) -> Result<String, String> {
    Ok(runtime_text(value, state, "text() interpolation")?
        .chars()
        .filter(|character| !is_text_format_marker(*character))
        .collect())
}

fn text_macro_is_proper(value: &Value, state: &ExecutionState) -> Result<bool, String> {
    let raw = runtime_text(value, state, "text() article")?;
    if raw.starts_with(TEXT_MACRO_PROPER) {
        return Ok(true);
    }
    if raw.starts_with(TEXT_MACRO_IMPROPER) {
        return Ok(false);
    }
    let Some(first) = raw
        .chars()
        .find(|character| !is_text_format_marker(*character))
    else {
        return Ok(true);
    };
    Ok(first.is_whitespace() || first.is_uppercase())
}

fn format_text_interpolation(
    value: &Value,
    prefix: Option<char>,
    state: &ExecutionState,
) -> Result<String, String> {
    let visible = text_macro_visible(value, state)?;
    let Some(prefix) = prefix else {
        return Ok(visible);
    };
    match prefix {
        TEXT_MACRO_THE | TEXT_MACRO_THE_UPPER => {
            if text_macro_is_proper(value, state)? {
                Ok(visible)
            } else {
                let article = if prefix == TEXT_MACRO_THE_UPPER {
                    "The "
                } else {
                    "the "
                };
                Ok(format!("{article}{visible}"))
            }
        }
        TEXT_MACRO_A | TEXT_MACRO_A_UPPER => {
            if text_macro_is_proper(value, state)? {
                return Ok(visible);
            }
            let plural = value_gender(value, state).as_deref() == Some("plural");
            let vowel = visible.chars().next().is_some_and(|character| {
                matches!(character.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
            });
            let article = match (prefix == TEXT_MACRO_A_UPPER, plural, vowel) {
                (true, true, _) => "Some ",
                (false, true, _) => "some ",
                (true, false, true) => "An ",
                (false, false, true) => "an ",
                (true, false, false) => "A ",
                (false, false, false) => "a ",
            };
            Ok(format!("{article}{visible}"))
        }
        TEXT_MACRO_ROMAN | TEXT_MACRO_ROMAN_UPPER => {
            Ok(value.as_number().map_or_else(String::new, |number| {
                roman_text(number, prefix == TEXT_MACRO_ROMAN_UPPER)
            }))
        }
        // `\\proper` and `\\improper` are metadata markers when stored in a
        // literal name. During runtime text() formatting BYOND consumes them.
        TEXT_MACRO_PROPER | TEXT_MACRO_IMPROPER => Ok(visible),
        _ => Ok(visible),
    }
}

fn roman_text(number: f32, upper: bool) -> String {
    if number.is_nan() {
        return "-".to_owned();
    }
    if number.is_infinite() {
        return if number.is_sign_negative() {
            "-inf"
        } else {
            "inf"
        }
        .to_owned();
    }
    let mut value = number.trunc() as i64;
    let mut output = String::new();
    if value < 0 {
        output.push('-');
        value = value.saturating_abs();
    }
    for (amount, lower, upper_character) in [
        (1000, 'm', 'M'),
        (500, 'd', 'D'),
        (100, 'c', 'C'),
        (50, 'l', 'L'),
        (10, 'x', 'X'),
        (5, 'v', 'V'),
        (1, 'i', 'I'),
    ] {
        while value >= amount {
            value -= amount;
            output.push(if upper { upper_character } else { lower });
        }
    }
    output
}

fn value_gender(value: &Value, state: &ExecutionState) -> Option<String> {
    let Value::Datum(datum) = value else {
        return None;
    };
    super::datum_field_or_initial(
        state,
        *datum,
        &FieldName::parse("gender").expect("gender field name is valid"),
    )
    .ok()
    .as_ref()
    .and_then(value_text)
    .map(str::to_owned)
}

fn apply_text_suffix(
    suffix: char,
    previous: Option<&Value>,
    previous_output_start: usize,
    output: &mut String,
    state: &ExecutionState,
) -> Result<(), String> {
    let Some(previous) = previous else {
        return Ok(());
    };
    match suffix {
        TEXT_MACRO_ORDINAL => {
            output.truncate(previous_output_start);
            let integer = previous.as_number().map_or(0_i64, |number| number as i64);
            output.push_str(&integer.to_string());
            output.push_str(match integer {
                1 => "st",
                2 => "nd",
                3 => "rd",
                _ => "th",
            });
        }
        TEXT_MACRO_PLURAL => {
            if previous.as_number() != Some(1.0) {
                output.push('s');
            }
        }
        _ => {
            let Some(gender) = value_gender(previous, state) else {
                return Ok(());
            };
            let index = match gender.as_str() {
                "male" => 0,
                "female" => 1,
                "plural" => 2,
                "neuter" => 3,
                _ => return Ok(()),
            };
            let words: [&[&str; 4]; 8] = [
                &["he", "she", "they", "it"],
                &["He", "She", "They", "It"],
                &["his", "her", "their", "its"],
                &["His", "Her", "Their", "Its"],
                &["him", "her", "them", "it"],
                &["himself", "herself", "themself", "itself"],
                &["his", "hers", "theirs", "its"],
                &["His", "Hers", "Theirs", "Its"],
            ];
            let family = match suffix {
                TEXT_MACRO_SUBJECT => 0,
                TEXT_MACRO_SUBJECT_UPPER => 1,
                TEXT_MACRO_POSSESSIVE_ADJECTIVE => 2,
                TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER => 3,
                TEXT_MACRO_OBJECT => 4,
                TEXT_MACRO_REFLEXIVE => 5,
                TEXT_MACRO_POSSESSIVE => 6,
                TEXT_MACRO_POSSESSIVE_UPPER => 7,
                _ => return Ok(()),
            };
            output.push_str(words[family][index]);
        }
    }
    Ok(())
}
