//! Native implementations of documented BYOND global procedures.
//!
//! These routines are deliberately runtime primitives rather than injected DM
//! source when their behavior depends on host state, type metadata, or precise
//! text-indexing semantics.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::unnecessary_wraps,
    reason = "DM uses binary32 numbers for integer/index boundaries and native builtin dispatch shares a Result ABI"
)]

use std::fs;
use std::fs::OpenOptions;
use std::io::Write as _;

use std::time::{SystemTime, UNIX_EPOCH};

use dm_value::{DatumId, FieldName, Value};
use smallvec::SmallVec;

use super::{CompoundAssignmentOperator, ExecutionState, NativeWalk, NativeWalkKind};

mod atmos;
// `native` acceleration and sibling clusters reach the shared datum-field
// helpers and the atmos difference builtin through the root re-exports.
#[cfg(test)]
use self::atmos::atmos_field;
use self::atmos::{
    atmos_setup_differences, builtin_contents_field, builtin_coordinate_fields, builtin_loc_field,
};

mod lists;
// The interpreter, native acceleration, and value-operation layers call into
// the list-cluster helpers directly through `crate::builtins::*`.
pub(super) use self::lists::{
    execute_list_binary_operator, execute_list_compound_operator, execute_list_method,
    is_movable_path, move_movable_to_atom, move_movable_to_turf, move_turf_to_area,
};

pub(crate) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
        "_dream64_atmos_setup_differences" => (5, 5),
        "abs" | "ceil" | "floor" | "fract" | "trunc" | "sign" | "sqrt" | "sin" | "cos" | "tan"
        | "arcsin" | "arccos" | "length_char" | "lowertext" | "uppertext" | "trimtext"
        | "ascii2text" | "text2path" | "isinf" | "isnan" | "ckey" | "fexists" | "file2text"
        | "lentext" | "list2params" | "params2list" | "file" | "html_encode" | "html_decode"
        | "isfile" | "fdel" | "del" | "rand_seed" | "link" | "ckeyEx" | "refcount" | "issaved" => {
            (1, 1)
        }
        "run" => (1, 3),
        "flist" => (0, 1),
        "fcopy_rsc" | "REGEX_QUOTE" | "REGEX_QUOTE_REPLACEMENT" => (1, 1),
        "browse" => (1, 2),
        "browse_rsc" | "ftp" => (1, 2),
        "winset" => (2, 3),
        "winget" => (2, 3),
        "winexists" => (2, 2),
        "alert" => (1, 6),
        "input" => (0, 4),
        "FLOOR" => (2, 2),
        "fcopy" => (2, 2),
        "text2file" => (2, 3),
        "json_decode" | "md5" => (0, 1),
        "json_encode" => (0, 2),
        "log" | "arctan" | "text2ascii" | "text2ascii_char" | "text2num" => (1, 2),
        "image" => (0, 7),
        "qdel" | "typecacheof" | "icon" => (0, 5),
        "view" => (0, 2),
        "orange" => (1, 2),
        "oview" | "viewers" | "oviewers" | "hearers" | "ohearers" => (0, 2),
        "step" => (2, 3),
        "step_towards" => (2, 2),
        "step_to" | "get_step_to" | "get_step_away" => (2, 3),
        "step_away" => (2, 3),
        "step_rand" | "get_step_rand" => (1, 2),
        "walk" => (2, 4),
        "walk_towards" => (2, 4),
        "walk_to" | "walk_away" => (2, 5),
        "walk_rand" => (1, 3),
        "bounds_dist" => (2, 2),
        "winshow" => (2, 3),
        "winclone" => (3, 3),
        "shell" => (1, 2),
        "sound" => (0, 7),
        "icon_states" => (1, 2),
        "newlist" => (0, usize::MAX),
        "min" | "max" => (0, usize::MAX),
        "clamp" | "lerp" => (3, 3),
        "cmptext" | "cmptextEx" | "sorttext" | "sorttextEx" | "sortText" | "addtext" => {
            (0, usize::MAX)
        }
        "text" => (1, usize::MAX),
        "num2text" => (1, 3),
        "time2text" => (1, 3),
        "rgb2num" => (1, 2),
        "rgb" => (3, 5),
        "gradient" => (2, usize::MAX),
        "generator" => (1, 4),
        "findtext"
        | "findtextEx"
        | "findtext_char"
        | "findtextEx_char"
        | "findlasttext"
        | "findlasttextEx"
        | "findlasttext_char"
        | "findlasttextEx_char"
        | "jointext" => (2, 4),
        "splittext" | "splittext_char" => (2, 5),
        "spantext" | "spantext_char" | "nonspantext" | "nonspantext_char" => (2, 3),
        "splicetext" | "splicetext_char" => (4, 4),
        "astype" => (1, 2),
        "get_dist" | "turn" | "flick" | "output" => (2, 2),
        "values_cut_over" | "values_cut_under" => (2, 3),
        "values_dot" => (2, 2),
        "values_product" | "values_sum" => (1, 1),
        "_dream64_world_profile" => (1, 3),
        "_dream64_world_get_config" => (1, 2),
        "_dream64_world_set_config" => (3, 3),
        "_dream64_world_open_port" => (2, 2),
        "_dream64_generator_rand" => (1, 1),
        "_dream64_icon_swap_color" => (3, 3),
        _ => return None,
    })
}

pub(super) fn execute_standard_builtin(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    execute_standard_builtin_with_usr(name, arguments, state, &Value::Null)
}

pub(super) fn execute_standard_builtin_with_usr(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
) -> Result<Value, String> {
    match name {
        "_dream64_atmos_setup_differences" => atmos_setup_differences(arguments, state),
        // Headless input has no interactive client. Preserve BYOND's supplied
        // default (the fourth positional argument), or null when absent.
        "input" => Ok(arguments.get(3).cloned().unwrap_or(Value::Null)),
        "text" => text_template(arguments, state),
        "newlist" => newlist_builtin(arguments, state),
        "qdel" => qdel_builtin(arguments, state),
        "del" => del_builtin(arguments, state),
        // `rand_seed()` resets the same per-world stream consumed by rand(),
        // prob(), pick(), roll(), and random-direction fallbacks.
        "rand_seed" => {
            let seed = number(&arguments[0], "rand_seed")?.trunc() as i64;
            state.random_state = seed as u64;
            Ok(Value::Null)
        }
        "abs" => unary_number(arguments, f32::abs),
        "ceil" => unary_number(arguments, f32::ceil),
        "floor" => unary_number(arguments, f32::floor),
        "fract" => unary_number(arguments, f32::fract),
        "trunc" => unary_number(arguments, f32::trunc),
        "sign" => unary_number(arguments, |value| {
            if value > 0.0 {
                1.0
            } else if value < 0.0 {
                -1.0
            } else {
                value
            }
        }),
        "sqrt" => unary_number(arguments, f32::sqrt),
        "sin" => unary_number(arguments, |value| value.to_radians().sin()),
        "cos" => unary_number(arguments, |value| value.to_radians().cos()),
        "tan" => unary_number(arguments, |value| value.to_radians().tan()),
        "arcsin" => inverse_trig(arguments, f32::asin),
        "arccos" => inverse_trig(arguments, f32::acos),
        "arctan" => arctan_builtin(arguments),
        "log" => log_builtin(arguments),
        "clamp" => clamp_builtin(arguments, state),
        "lerp" => lerp_builtin(arguments),
        "min" => extrema_builtin(arguments, state, false),
        "max" => extrema_builtin(arguments, state, true),
        "length_char" => length_char(arguments, state),
        "lowertext" => text_case_map(arguments, str::to_lowercase),
        "uppertext" => text_case_map(arguments, str::to_uppercase),
        "trimtext" if matches!(arguments[0], Value::Null) => Ok(Value::Null),
        "trimtext" => text_map(arguments, state, |value| value.trim().to_owned()),
        "fcopy_rsc" => fcopy_rsc(arguments, state),
        "link" => headless_link(arguments, state, usr),
        "run" => Ok(Value::Null),
        "issaved" => Ok(Value::number(1.0)),
        "REGEX_QUOTE" => regex_quote(arguments, state, false),
        "REGEX_QUOTE_REPLACEMENT" => regex_quote(arguments, state, true),
        "browse" => headless_browse(arguments, state),
        "browse_rsc" => headless_transfer("browse_rsc", arguments, state),
        "ftp" => headless_transfer("ftp", arguments, state),
        "winset" => headless_winset(arguments, state, usr),
        "winshow" => headless_winshow(arguments, state),
        "winclone" => headless_winclone(arguments, state),
        "winget" => headless_winget(arguments, state),
        "winexists" => headless_winexists(arguments, state),
        "alert" => headless_alert(arguments),
        "FLOOR" => floor_multiple(arguments),
        "typecacheof" => typecacheof_builtin(arguments, state),
        "ascii2text" => ascii2text(arguments),
        "text2ascii" => text2ascii(arguments, state, false),
        "text2ascii_char" => text2ascii(arguments, state, true),
        "text2num" => text2num(arguments, state),
        "text2path" => text2path(arguments, state),
        "isinf" => numeric_classifier(arguments, f32::is_infinite),
        "isnan" => numeric_classifier(arguments, f32::is_nan),
        "cmptext" => cmptext(arguments, state, false),
        "cmptextEx" => cmptext(arguments, state, true),
        "findtext" => findtext(arguments, state, false, false, false),
        "findtextEx" => findtext(arguments, state, true, false, false),
        "findtext_char" => findtext(arguments, state, false, true, false),
        "findtextEx_char" => findtext(arguments, state, true, true, false),
        "findlasttext" => findtext(arguments, state, false, false, true),
        "findlasttextEx" => findtext(arguments, state, true, false, true),
        "findlasttext_char" => findtext(arguments, state, false, true, true),
        "findlasttextEx_char" => findtext(arguments, state, true, true, true),
        "splittext" => splittext(arguments, state, false),
        "splittext_char" => splittext(arguments, state, true),
        "jointext" => jointext(arguments, state),
        "addtext" => addtext(arguments, state),
        "spantext" => spantext(arguments, state, false, true),
        "spantext_char" => spantext(arguments, state, true, true),
        "nonspantext" => spantext(arguments, state, false, false),
        "nonspantext_char" => spantext(arguments, state, true, false),
        "splicetext" => splicetext(arguments, state, false),
        "splicetext_char" => splicetext(arguments, state, true),
        "get_dist" => get_dist(arguments, state),
        "turn" => turn(arguments, state),
        "astype" => astype(arguments, state),
        // `flick()` temporarily changes only the client-rendered icon state;
        // the atom's persistent `icon_state` is deliberately untouched.
        "flick" => Ok(Value::Null),
        "output" => resource_datum_builtin("/output", &["message", "control"], arguments, state),
        "values_cut_over" => values_cut(arguments, state, true),
        "values_cut_under" => values_cut(arguments, state, false),
        "values_dot" => values_dot(arguments, state),
        "values_product" => values_fold(arguments, state, true),
        "values_sum" => values_fold(arguments, state, false),
        "_dream64_world_profile" => world_profile(arguments, state),
        "_dream64_world_get_config" => world_get_config(arguments, state),
        "_dream64_world_set_config" => world_set_config(arguments, state),
        "_dream64_world_open_port" => world_open_port(arguments, state),
        "ckey" => ckey(arguments, state),
        "ckeyEx" => ckey_ex(arguments, state),
        "refcount" => Ok(Value::number(match arguments.first() {
            Some(Value::Datum(_) | Value::List(_)) => 1.0,
            _ => 0.0,
        })),
        "fexists" => fexists(arguments, state),
        "file2text" => file2text(arguments, state),
        "isfile" => Ok(Value::number(f32::from(matches!(
            arguments[0],
            Value::File(_)
        )))),
        "fdel" => fdel(arguments, state),
        "flist" => flist(arguments, state),
        "fcopy" => fcopy(arguments, state),
        "text2file" => text2file(arguments, state),
        "html_encode" => html_encode(arguments, state),
        "html_decode" => html_decode(arguments, state),
        "rgb" => rgb_builtin(arguments),
        "rgb2num" => rgb2num_builtin(arguments, state),
        "gradient" => gradient_builtin(arguments, state),
        "generator" => {
            resource_datum_builtin("/generator", &["type", "a", "b", "rand"], arguments, state)
        }
        "time2text" => time2text_builtin(arguments, state),
        "orange" => orange_builtin(arguments, state, usr),
        "view" => spatial_query(arguments, state, usr, false, false),
        "oview" => spatial_query(arguments, state, usr, false, true),
        "viewers" | "hearers" => spatial_query(arguments, state, usr, true, false),
        "ohearers" | "oviewers" => spatial_query(arguments, state, usr, true, true),
        "step" => step_builtin(arguments, state),
        "step_towards" => step_towards_builtin(arguments, state),
        "step_to" => step_to_builtin(arguments, state),
        "get_step_to" => get_step_to_builtin(arguments, state),
        "step_away" => step_away_builtin(arguments, state),
        "get_step_away" => get_step_away_builtin(arguments, state),
        "step_rand" => step_rand_builtin(arguments, state),
        "get_step_rand" => get_step_rand_builtin(arguments, state),
        "walk" | "walk_towards" | "walk_to" | "walk_away" | "walk_rand" => {
            start_native_walk(name, arguments, state)
        }
        "bounds_dist" => bounds_dist_builtin(arguments, state),
        "shell" => Ok(Value::Null),
        "file" => match &arguments[0] {
            Value::Text(path) | Value::File(path) => Ok(Value::file(path.clone())),
            value => Err(format!("file() requires a resource path, received {value}")),
        },
        "lentext" => lentext(arguments, state),
        "sorttext" => sorttext(arguments, state, false),
        "sorttextEx" | "sortText" => sorttext(arguments, state, true),
        "num2text" => num2text(arguments),
        "list2params" => list2params(arguments, state),
        "params2list" => params2list(arguments, state),
        "json_decode" => json_decode_builtin(arguments, state),
        "json_encode" => json_encode_builtin(arguments, state),
        "md5" => md5_builtin(arguments),
        "image" => image_builtin(arguments, state),
        "icon" => icon_builtin(arguments, state),
        "sound" => resource_datum_builtin(
            "/sound",
            &[
                "file",
                "repeat",
                "wait",
                "channel",
                "volume",
                "frequency",
                "pan",
            ],
            arguments,
            state,
        ),
        "icon_states" => icon_states_builtin(arguments, state),
        "_dream64_generator_rand" => generator_rand_builtin(arguments, state),
        "_dream64_icon_swap_color" => icon_swap_color_builtin(arguments, state),
        _ => Err(format!("unknown native DM builtin {name:?}")),
    }
}

pub(super) fn execute_external_call(
    library: &Value,
    function: &Value,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let library = strict_text(library, state, "external library")?;
    let function = strict_text(function, state, "external function")?;
    let filename = std::path::Path::new(&library)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(&library)
        .to_ascii_lowercase();
    if matches!(filename.as_str(), "memorystats.dll" | "libmemorystats.so") {
        return match (function.as_str(), arguments) {
            // byond-memorystats exposes DreamDaemon's allocator report through
            // this single zero-argument export. Dream64 cannot truthfully
            // manufacture BYOND's internal prototype/object counters, so keep
            // their established text schema at zero and append the real host
            // process resident set when the platform can provide it safely.
            ("memory_stats", []) => Ok(Value::text(headless_memory_stats_report())),
            _ => Err(format!(
                "external call {library}::{function} requires an installed host bridge"
            )),
        };
    }
    if matches!(
        filename.as_str(),
        "dreamluau" | "dreamluau.dll" | "libdreamluau.so"
    ) {
        let export = function.strip_prefix("byond:").unwrap_or(&function);
        return match export {
            // DreamLuaU's process-wide configuration and cleanup calls have no
            // observable DM return value. Headless Dream64 does not embed a
            // Luau VM, but these hooks must remain safe and idempotent so
            // ordinary datum destruction and SS13 startup can proceed.
            "set_usr" | "set_execution_limit_millis" | "set_execution_limit_secs"
                if arguments.len() == 1 =>
            {
                Ok(Value::Null)
            }
            "clear_execution_limit" if arguments.is_empty() => Ok(Value::Null),
            "set_new_wrapper"
            | "set_var_get_wrapper"
            | "set_var_set_wrapper"
            | "set_object_call_wrapper"
            | "set_global_call_wrapper"
            | "set_print_wrapper"
                if arguments.len() == 1 =>
            {
                Ok(Value::Null)
            }
            "collect_garbage" | "kill_state" | "clear_ref_userdata" if arguments.len() == 1 => {
                Ok(Value::Null)
            }
            "kill_sleeping_thread" | "kill_yielded_thread" if arguments.len() == 2 => {
                Ok(Value::Null)
            }
            // No Luau frames exist in headless mode.
            "get_traceback" if arguments.len() == 1 => Ok(Value::Null),
            _ => Err(format!(
                "external call {library}::{function} requires an installed host bridge"
            )),
        };
    }
    if !matches!(
        filename.as_str(),
        "rust_g" | "rust_g.dll" | "librust_g.so" | "librust_g64.so"
    ) {
        return Err(format!(
            "external call {library}::{function} requires an installed host bridge"
        ));
    }
    match function.as_str() {
        "get_version" if arguments.is_empty() => {
            Ok(Value::text(concat!(env!("CARGO_PKG_VERSION"), "-dream64")))
        }
        "unix_timestamp" if arguments.is_empty() => {
            let seconds = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| format!("unix_timestamp failed: {error}"))?
                .as_secs();
            Ok(Value::text(seconds.to_string()))
        }
        "formatted_timestamp" if matches!(arguments.len(), 1 | 2) => {
            let format = strict_text(&arguments[0], state, "formatted_timestamp format")?;
            let offset_hours = arguments
                .get(1)
                .map(|value| {
                    value
                        .as_number()
                        .or_else(|| value.to_string().parse::<f32>().ok())
                        .ok_or_else(|| "formatted_timestamp offset must be numeric".to_owned())
                })
                .transpose()?
                .unwrap_or(0.0);
            let unix_millis = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|error| format!("formatted_timestamp failed: {error}"))?
                .as_millis();
            let unix_millis = i64::try_from(unix_millis).unwrap_or(i64::MAX);
            Ok(Value::text(format_unix_timestamp(
                unix_millis,
                &format,
                offset_hours,
            )))
        }
        "rg_git_revparse" if arguments.len() == 1 => {
            let revision = strict_text(&arguments[0], state, "rg_git_revparse revision")?;
            validate_git_revision(&revision)?;
            run_git_bridge(
                state,
                &["rev-parse", "--verify", "--end-of-options", &revision],
            )
        }
        "rg_git_commit_date" if matches!(arguments.len(), 1 | 2) => {
            let revision = strict_text(&arguments[0], state, "rg_git_commit_date revision")?;
            validate_git_revision(&revision)?;
            let format = arguments
                .get(1)
                .map(|value| strict_text(value, state, "rg_git_commit_date format"))
                .transpose()?
                .unwrap_or_else(|| "%F".to_owned());
            validate_git_date_format(&format)?;
            let date = format!("--date=format:{format}");
            run_git_bridge(
                state,
                &[
                    "log",
                    "-1",
                    "--format=%ad",
                    &date,
                    "--end-of-options",
                    &revision,
                ],
            )
        }
        "rg_git_commit_date_head" if matches!(arguments.len(), 0 | 1) => {
            let format = arguments
                .first()
                .map(|value| strict_text(value, state, "rg_git_commit_date_head format"))
                .transpose()?
                .unwrap_or_else(|| "%F".to_owned());
            validate_git_date_format(&format)?;
            let date = format!("--date=format:{format}");
            run_git_bridge(
                state,
                &[
                    "log",
                    "-1",
                    "--format=%ad",
                    &date,
                    "--end-of-options",
                    "HEAD",
                ],
            )
        }
        "hash_string" if arguments.len() == 2 => rust_g_hash_string(arguments, state),
        "hash_file" if arguments.len() == 2 => rust_g_hash_file(arguments, state),
        "url_encode" if arguments.len() == 1 => rust_g_url_encode(arguments, state),
        "url_decode" if arguments.len() == 1 => rust_g_url_decode(arguments, state),
        "json_is_valid" if arguments.len() == 1 => {
            let input = strict_text(&arguments[0], state, "json_is_valid input")?;
            Ok(Value::text(
                if serde_json::from_str::<serde_json::Value>(&input).is_ok() {
                    "true"
                } else {
                    "false"
                },
            ))
        }
        "cnoise_generate" if arguments.len() == 6 => rust_g_cellular_noise(arguments, state),
        "noise_poisson_map" if arguments.len() == 4 => rust_g_poisson_noise(arguments, state),
        "log_write" if arguments.len() == 3 => {
            let path = relaxed_resolved_file_path(&arguments[..1], state, "log_write")?;
            let text = strict_text(&arguments[1], state, "log_write text")?;
            let format_internally =
                strict_text(&arguments[2], state, "log_write format")?.eq_ignore_ascii_case("true");
            let parent = path
                .parent()
                .ok_or_else(|| "log_write path has no parent".to_owned())?;
            fs::create_dir_all(parent)
                .map_err(|error| format!("log_write failed to create parent: {error}"))?;
            let root = state
                .project_root()
                .ok_or_else(|| "log_write requires a configured project root".to_owned())?
                .canonicalize()
                .map_err(|error| format!("log_write project root is unavailable: {error}"))?;
            let parent = parent
                .canonicalize()
                .map_err(|error| format!("log_write parent directory is unavailable: {error}"))?;
            if !parent.starts_with(root) {
                return Err("log_write path escapes the project root".to_owned());
            }
            let path = parent.join(
                path.file_name()
                    .ok_or_else(|| "log_write path is invalid".to_owned())?,
            );
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|error| format!("log_write failed: {error}"))?;
            file.write_all(text.as_bytes())
                .map_err(|error| format!("log_write failed: {error}"))?;
            if format_internally && !text.ends_with('\n') {
                file.write_all(b"\n")
                    .map_err(|error| format!("log_write failed: {error}"))?;
            }
            Ok(Value::Null)
        }
        "log_close_all" if arguments.is_empty() => Ok(Value::Null),
        "file_write" | "file_append" if arguments.len() == 2 => {
            let text = strict_text(&arguments[0], state, function.as_str())?;
            let path = relaxed_resolved_file_path(&arguments[1..], state, function.as_str())?;
            let parent = path
                .parent()
                .ok_or_else(|| format!("{function} path has no parent"))?;
            fs::create_dir_all(parent)
                .map_err(|error| format!("{function} failed to create parent: {error}"))?;
            // Re-resolve after creation so a raced or pre-existing symlink can
            // never redirect the eventual write outside the project root.
            let root = state
                .project_root()
                .ok_or_else(|| format!("{function} requires a configured project root"))?
                .canonicalize()
                .map_err(|error| format!("{function} project root is unavailable: {error}"))?;
            let parent = parent
                .canonicalize()
                .map_err(|error| format!("{function} parent directory is unavailable: {error}"))?;
            if !parent.starts_with(root) {
                return Err(format!("{function} path escapes the project root"));
            }
            let path = parent.join(
                path.file_name()
                    .ok_or_else(|| format!("{function} path is invalid"))?,
            );
            if function == "file_write" {
                fs::write(path, text).map_err(|error| format!("file_write failed: {error}"))?;
            } else {
                let mut file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .map_err(|error| format!("file_append failed: {error}"))?;
                file.write_all(text.as_bytes())
                    .map_err(|error| format!("file_append failed: {error}"))?;
            }
            // rust-g's void file helpers yield BYOND null on success.
            Ok(Value::Null)
        }
        "file_exists" if arguments.len() == 1 => {
            let path = resolved_file_path(arguments, state, "file_exists")?;
            Ok(Value::text(if path.exists() { "true" } else { "false" }))
        }
        "file_read" if arguments.len() == 1 => {
            let path = resolved_file_path(arguments, state, "file_read")?;
            match fs::read_to_string(path) {
                Ok(text) => Ok(Value::text(text)),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
                Err(error) => Err(format!("file_read failed: {error}")),
            }
        }
        "toml_file_to_json" if arguments.len() == 1 => {
            let result = resolved_file_path(arguments, state, "toml_file_to_json")
                .and_then(|path| fs::read_to_string(path).map_err(|error| error.to_string()))
                .and_then(|source| parse_toml_document(&source));
            let (success, content) = match result {
                Ok(document) => (
                    true,
                    serde_json::to_string(&document).map_err(|error| error.to_string())?,
                ),
                Err(error) => (false, error),
            };
            Ok(Value::text(
                serde_json::json!({ "success": success, "content": content }).to_string(),
            ))
        }
        "time_reset" if arguments.len() == 1 => {
            let name = strict_text(&arguments[0], state, "time_reset")?;
            state.reset_external_timer(name);
            Ok(Value::Null)
        }
        "time_milliseconds" if arguments.len() == 1 => {
            let name = strict_text(&arguments[0], state, "time_milliseconds")?;
            Ok(Value::text(
                state.external_timer_milliseconds(&name).to_string(),
            ))
        }
        "time_microseconds" if arguments.len() == 1 => {
            let name = strict_text(&arguments[0], state, "time_microseconds")?;
            Ok(Value::text(
                state.external_timer_microseconds(&name).to_string(),
            ))
        }
        "iconforge_load_gags_config" if arguments.len() == 3 => {
            iconforge_load_gags_config(arguments, state)
        }
        "iconforge_load_gags_config_async" if arguments.len() == 3 => {
            let result = iconforge_load_gags_config(arguments, state)?;
            Ok(Value::text(
                state.enqueue_iconforge_job(owned_value_text(result)),
            ))
        }
        "iconforge_gags" if arguments.len() == 3 => iconforge_gags(arguments, state),
        "iconforge_gags_async" if arguments.len() == 3 => {
            let result = iconforge_gags(arguments, state)?;
            Ok(Value::text(
                state.enqueue_iconforge_job(owned_value_text(result)),
            ))
        }
        "iconforge_check" if arguments.len() == 1 => {
            let id = strict_text(&arguments[0], state, "iconforge_check job id")?;
            Ok(Value::text(state.poll_iconforge_job(&id).unwrap_or_else(
                || format!("IconForge error: Unknown job ID '{id}'"),
            )))
        }
        "iconforge_cleanup" if arguments.is_empty() => {
            // rust-g's cleanup releases rendered icon/image caches. Loaded
            // GAGS configurations and outstanding async jobs remain valid.
            // The headless VM has no rendered cache to release here.
            Ok(Value::Null)
        }
        "iconforge_cache_valid" if arguments.len() == 3 => Ok(Value::text(
            r#"{"result":"0","fail_reason":"Dream64 headless cache has no rendered spritesheet"}"#,
        )),
        "iconforge_cache_valid_async" if arguments.len() == 3 => {
            let result = r#"{"result":"0","fail_reason":"Dream64 headless cache has no rendered spritesheet"}"#;
            Ok(Value::text(state.enqueue_iconforge_job(result.to_owned())))
        }
        "iconforge_generate" if arguments.len() == 6 => Ok(Value::text(
            r#"{"sizes":{},"sprites":{},"dmi_hashes":{},"sprites_hash":"","error":null,"headless":true}"#,
        )),
        "iconforge_generate_async" if arguments.len() == 6 => {
            let result = r#"{"sizes":{},"sprites":{},"dmi_hashes":{},"sprites_hash":"","error":null,"headless":true}"#;
            Ok(Value::text(state.enqueue_iconforge_job(result.to_owned())))
        }
        "iconforge_generate_headless" if arguments.len() == 3 => {
            let file_path = strict_text(&arguments[0], state, "iconforge_generate_headless path")?;
            Ok(Value::text(
                serde_json::json!({
                    "file_path": file_path,
                    "width": serde_json::Value::Null,
                    "height": serde_json::Value::Null,
                    "error": "Dream64 headless mode skipped icon rendering",
                })
                .to_string(),
            ))
        }
        "sql_connect_pool" if arguments.len() == 1 => Ok(Value::text(
            serde_json::json!({
                "status": "err",
                "data": "Dream64 headless SQL host is unavailable",
            })
            .to_string(),
        )),
        "sql_connected" if arguments.len() == 1 => Ok(Value::text(r#"{"status":"offline"}"#)),
        "sql_disconnect_pool" if arguments.len() == 1 => Ok(Value::Null),
        "sql_query_blocking" if arguments.len() == 3 => Ok(Value::text(
            r#"{"status":"offline","data":"Dream64 headless SQL host is unavailable"}"#,
        )),
        "sql_query_async" if arguments.len() == 3 => {
            let result =
                r#"{"status":"offline","data":"Dream64 headless SQL host is unavailable"}"#;
            Ok(Value::text(state.enqueue_sql_job(result.to_owned())))
        }
        "sql_check_query" if arguments.len() == 1 => {
            let id = strict_text(&arguments[0], state, "sql_check_query job id")?;
            Ok(Value::text(
                state
                    .poll_sql_job(&id)
                    .unwrap_or_else(|| "NO SUCH JOB".to_owned()),
            ))
        }
        "dmi_read_metadata" if arguments.len() == 1 => {
            let requested = strict_text(&arguments[0], state, "dmi_read_metadata path")?;
            let resolved = relaxed_resolved_file_path(arguments, state, "dmi_read_metadata path")?;
            let metadata = read_dmi_metadata(&resolved).unwrap_or_else(|error| DmiMetadata {
                width: 32,
                height: 32,
                states: Vec::new(),
                error: Some(format!(
                    "Failed to read DMI '{requested}' (resolved to '{}') - {error}",
                    resolved.display()
                )),
            });
            Ok(Value::text(metadata.to_json().to_string()))
        }
        "dmi_icon_states" if arguments.len() == 1 => {
            let resolved = relaxed_resolved_file_path(arguments, state, "dmi_icon_states path")?;
            let states = read_dmi_metadata(&resolved)
                .map(|metadata| {
                    metadata
                        .states
                        .into_iter()
                        .map(|state| state.name)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            Ok(Value::text(serde_json::json!(states).to_string()))
        }
        "dmi_strip_metadata" if arguments.len() == 1 => Ok(Value::text("")),
        "dmi_resize_png" if arguments.len() == 4 => Ok(Value::text("")),
        "dmi_inject_metadata" if arguments.len() == 2 => Ok(Value::text("")),
        _ => Err(format!(
            "external call {library}::{function} requires an installed host bridge"
        )),
    }
}

fn headless_memory_stats_report() -> String {
    let resident = current_process_resident_bytes()
        .map_or_else(|| "unavailable".to_owned(), format_memory_size);
    format!(
        "Server mem usage:\n\
prototypes:\n\
\tobj: 0 B (0)\n\
\tmob: 0 B (0)\n\
\tproc: 0 B (0)\n\
\tstr: 0 B (0)\n\
\tappearance: 0 B (0)\n\
\tfilter: 0 B (0)\n\
\tid array: 0 B (0)\n\
\tmap: 0 B (0,0,0)\n\
objects:\n\
\tmobs: 0 B (0)\n\
\tobjs: 0 B (0)\n\
\tdatums: 0 B (0)\n\
\timages: 0 B (0)\n\
\tlists: 0 B (0)\n\
\tprocs: 0 B (0)\n\
Dream64 host:\n\
\tresident: {resident}"
    )
}

fn format_memory_size(bytes: u64) -> String {
    const KIB: u64 = 1_024;
    const MIB: u64 = KIB * KIB;
    const GIB: u64 = MIB * KIB;
    if bytes >= GIB {
        format!("{:.2} GB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KB", bytes as f64 / KIB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(windows)]
fn current_process_resident_bytes() -> Option<u64> {
    // There is no safe standard-library API for the Windows working set, and
    // this crate deliberately forbids unsafe host calls. Spawning PowerShell
    // or tasklist here delayed Monkestation startup by 0.5-1.3 seconds. Keep
    // the compatibility report truthful and non-blocking; process telemetry
    // belongs in the lifecycle host where it can be sampled asynchronously.
    None
}

#[cfg(unix)]
fn current_process_resident_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let resident_kib = status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_ascii_whitespace()
            .next()?
            .parse::<u64>()
            .ok()
    })?;
    resident_kib.checked_mul(1_024)
}

#[cfg(not(any(windows, unix)))]
fn current_process_resident_bytes() -> Option<u64> {
    None
}

fn newlist_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let result = state.heap.allocate_list();
    for argument in arguments {
        let Value::TypePath(path) = argument else {
            return Err("newlist() arguments must be type paths".to_owned());
        };
        let datum = state.heap.allocate_datum(path.clone());
        state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(datum));
    }
    Ok(Value::List(result))
}

fn value_text(value: &Value) -> Option<&str> {
    match value {
        Value::Text(text) => Some(text),
        _ => None,
    }
}

fn strict_text(value: &Value, state: &ExecutionState, context: &str) -> Result<String, String> {
    match value {
        // File/resource values retain their own runtime type for `isfile`, but
        // BYOND text-consuming and filesystem APIs observe their path text.
        Value::Text(text) | Value::File(text) => Ok(text.to_string()),
        _ => Err(format!(
            "{context} requires text, received {}",
            runtime_text(value, state, context)?
        )),
    }
}

fn text_map(
    arguments: &[Value],
    state: &ExecutionState,
    operation: impl FnOnce(&str) -> String,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text builtin")?;
    Ok(Value::text(operation(&text)))
}

fn text_case_map(
    arguments: &[Value],
    operation: impl FnOnce(&str) -> String,
) -> Result<Value, String> {
    let Some(value) = arguments.first() else {
        return Ok(Value::Null);
    };
    let Value::Text(text) = value else {
        // BYOND's lowertext()/uppertext() leave non-text values unchanged.
        // This matters for helpers such as uppertext(dir2text(0)), where the
        // inner proc returns numeric NONE rather than a string.
        return Ok(value.clone());
    };
    Ok(Value::text(operation(text)))
}
pub(super) fn datum_coordinates(state: &ExecutionState, value: &Value) -> Option<(f32, f32, f32)> {
    let Value::Datum(original) = value else {
        return None;
    };
    if state
        .heap
        .datum(*original)
        .is_ok_and(|datum| super::is_area_type_path(datum.type_path()))
    {
        return state
            .world_areas
            .iter()
            .find_map(|(coordinate, area)| (*area == *original).then_some(*coordinate))
            .map(|(x, y, z)| (x as f32, y as f32, z as f32));
    }
    // BYOND spatial builtins use the containing turf for objects nested in
    // mobs, items, closets, and other movable containers. Their own x/y/z
    // fields may still contain zero or a stale former turf coordinate. Follow
    // loc links exactly as get_step(atom, 0) does, while retaining the
    // original datum as the fallback for lightweight uncontained fixtures.
    let loc = builtin_loc_field();
    let mut coordinate_source = *original;
    let mut current = *original;
    let mut visited = SmallVec::<[DatumId; 8]>::new();
    while !visited.contains(&current) {
        visited.push(current);
        let datum = state.heap.datum(current).ok()?;
        if super::is_turf_type_path(datum.type_path()) {
            coordinate_source = current;
            break;
        }
        let Ok(Value::Datum(parent)) = super::datum_field_or_initial(state, current, loc) else {
            break;
        };
        current = parent;
    }
    let coordinate = |field: &FieldName| {
        super::datum_field_or_initial(state, coordinate_source, field)
            .ok()?
            .as_number()
    };
    let [x, y, z] = builtin_coordinate_fields();
    Some((coordinate(x)?, coordinate(y)?, coordinate(z)?))
}

mod spatial;
// The interpreter, native-movement, and value-operation layers call into
// the spatial-cluster helpers directly through `crate::builtins::*`, and the
// dispatcher below addresses the leaf procedures unqualified.
#[cfg(test)]
use self::spatial::indexed_spatial_candidates;
pub(super) use self::spatial::{advance_native_walks, is_subtype, synchronize_moved_atom_contents};
// The dispatcher addresses the icon/DMI leaf procedures unqualified, and the
// resource-baked icon lookup is shared with the file/image clusters.
mod icons;
#[cfg(test)]
pub(super) use self::icons::DMI_METADATA_PHYSICAL_READS;
pub(super) use self::icons::{DmiMetadata, icon_states_builtin, read_dmi_metadata};
use self::icons::{fcopy_rsc, icon_builtin, icon_swap_color_builtin};
// The dispatcher addresses the noise/forge/world/generator leaf procedures
// unqualified; `resource_datum_builtin` is also shared with the icon builder.
mod noise;
use self::noise::{rust_g_cellular_noise, rust_g_poisson_noise};
mod forge;
use self::forge::{
    format_unix_timestamp, iconforge_gags, iconforge_load_gags_config, owned_value_text,
    parse_toml_document, run_git_bridge, validate_git_date_format, validate_git_revision,
};
mod world;
use self::world::{world_get_config, world_open_port, world_profile, world_set_config};
mod generator;
use self::generator::{generator_rand_builtin, resource_datum_builtin};
mod text_template;
use self::text_template::text_template;
mod file;
pub(super) use self::file::{execute_output, relaxed_resolved_file_path, resolved_file_path};
use self::file::{fcopy, fdel, fexists, file2text, flist, html_decode, html_encode, text2file};
mod color;
pub(super) use self::color::parse_hex_color;
use self::color::{gradient_builtin, rgb_builtin, rgb2num_builtin, time2text_builtin};
mod values;
use self::values::{values_cut, values_dot, values_fold};
mod crypto_hash;
use self::crypto_hash::{
    md5_builtin, rust_g_hash_file, rust_g_hash_string, rust_g_url_decode, rust_g_url_encode,
};
mod json;
use self::json::{json_decode_builtin, json_encode_builtin};
mod text_arith;
pub(super) use self::text_arith::params2list;
use self::text_arith::{lentext, list2params, num2text, sorttext};
mod numeric;
use self::numeric::{
    arctan_builtin, clamp_builtin, extrema_builtin, inverse_trig, lerp_builtin, log_builtin,
    unary_number,
};
pub(super) use self::numeric::{number, runtime_text, truthy};
mod qdel_appearance;
pub(super) use self::qdel_appearance::{
    appearance_snapshot_builtin, copy_image_appearance, is_appearance_source,
};
use self::qdel_appearance::{del_builtin, image_builtin, qdel_builtin, typecacheof_builtin};
mod ui;
use self::spatial::{
    astype, bounds_dist_builtin, ckey, ckey_ex, get_dist, get_step_away_builtin,
    get_step_rand_builtin, get_step_to_builtin, orange_builtin, spatial_query, start_native_walk,
    step_away_builtin, step_builtin, step_rand_builtin, step_to_builtin, step_towards_builtin,
    turn,
};
pub(super) use self::ui::local_client_for_value;
mod text_ops;
use self::text_ops::{
    addtext, ascii2text, cmptext, findtext, jointext, length_char, numeric_classifier, spantext,
    splicetext, splittext, text2ascii, text2num, text2path,
};
pub(super) use self::text_ops::{execute_regex_method, is_regex_datum, regex_search};
use self::ui::{
    floor_multiple, headless_alert, headless_browse, headless_link, headless_transfer,
    headless_winclone, headless_winexists, headless_winget, headless_winset, headless_winshow,
    regex_quote,
};
// Names bridged from the crate root so the extracted clusters keep addressing
// the native value-layer helpers through `super::*`.
use crate::{
    allocate_matrix, clone_icon_datum, datum_field_or_initial, deterministic_unit,
    execute_icon_method, get_step_builtin, is_atom_type_path, is_icon_datum, is_matrix_datum,
    is_turf_type_path, matrix_components, matrix_product,
};
#[cfg(test)]
mod tests;
