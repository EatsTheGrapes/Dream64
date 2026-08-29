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
            Some(Value::Datum(_)) | Some(Value::List(_)) => 1.0,
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
        .map(format_memory_size)
        .unwrap_or_else(|| "unavailable".to_owned());
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
pub(super) use self::icons::{DmiMetadata, read_dmi_metadata};
use self::icons::{fcopy_rsc, icon_builtin, icon_states_builtin, icon_swap_color_builtin};
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
mod json_md5_tests {
    use super::*;

    #[test]
    fn byond_letter_class_matches_multiline_admin_rank_blocks() {
        let source = "Name = Host\nInclude = @ ADMIN BAN\nExclude = FUN\nEdit =\n";
        let pattern = r"^Name\s*=\s*(.+?)\s*\n+Include\s*=\s*([\l @]*?)\s*\n+Exclude\s*=\s*([\l @]*?)\s*\n+Edit\s*=\s*([\l @]*?)\s*\n*$";
        let found = regex_search(pattern, "gm", source, 0, source.len())
            .expect("BYOND regex should compile")
            .expect("rank block should match");
        assert_eq!(found.2[0].as_deref(), Some("Host"));
        assert_eq!(found.2[1].as_deref(), Some("@ ADMIN BAN"));
        assert_eq!(found.2[2].as_deref(), Some("FUN"));
        assert_eq!(found.2[3].as_deref(), Some(""));
    }

    #[test]
    fn text_interpolation_uses_byond_list_display_name() {
        let mut state = ExecutionState::new();
        let positional = state.heap.allocate_list();
        state
            .heap
            .list_mut(positional)
            .unwrap()
            .add(Value::number(1.0));
        let associative = state.heap.allocate_list();
        state
            .heap
            .list_mut(associative)
            .unwrap()
            .set_key(Value::text("a"), Value::number(3.0));

        assert_eq!(
            text_template(
                &[
                    Value::text("plain=|[]| assoc=|[]|"),
                    Value::List(positional),
                    Value::List(associative),
                ],
                &state,
            )
            .unwrap(),
            Value::text("plain=|/list| assoc=|/list|")
        );
    }

    fn encoded(value: Value, state: &ExecutionState) -> String {
        let Value::Text(text) = json_encode_builtin(&[value], state).expect("JSON should encode")
        else {
            panic!("json_encode must return text");
        };
        text.to_string()
    }

    #[test]
    fn json_encodes_dm_scalars_and_special_numbers() {
        let state = ExecutionState::new();
        assert_eq!(encoded(Value::Null, &state), "null");
        assert_eq!(encoded(Value::number(7.0), &state), "7");
        assert_eq!(encoded(Value::number(15.5), &state), "15.5");
        assert_eq!(encoded(Value::text("A\nB"), &state), r#""A\nB""#);
        assert_eq!(
            encoded(Value::number(f32::NAN), &state),
            r#"{"__number__":"NaN"}"#
        );
        assert_eq!(
            encoded(Value::number(f32::INFINITY), &state),
            r#"{"__number__":"Infinity"}"#
        );
    }

    #[test]
    fn json_encodes_positional_associative_and_pretty_lists() {
        let mut state = ExecutionState::new();
        let positional = state.heap.allocate_list();
        state
            .heap
            .list_mut(positional)
            .unwrap()
            .add(Value::number(1.0));
        state
            .heap
            .list_mut(positional)
            .unwrap()
            .add(Value::text("two"));
        assert_eq!(encoded(Value::List(positional), &state), r#"[1,"two"]"#);

        let associative = state.heap.allocate_list();
        state
            .heap
            .list_mut(associative)
            .unwrap()
            .set_key(Value::text("name"), Value::text("fridge"));
        state
            .heap
            .list_mut(associative)
            .unwrap()
            .set_key(Value::text("power"), Value::number(12.0));
        assert_eq!(
            encoded(Value::List(associative), &state),
            r#"{"name":"fridge","power":12}"#
        );
        let Value::Text(pretty) =
            json_encode_builtin(&[Value::List(associative), Value::number(1.0)], &state).unwrap()
        else {
            panic!("pretty JSON must be text");
        };
        assert!(pretty.contains('\n'));
    }

    #[test]
    fn json_decodes_arrays_objects_booleans_and_special_numbers() {
        let mut state = ExecutionState::new();
        let decoded =
            json_decode_builtin(&[Value::text(r#"{"a":[true,null,2.5]}"#)], &mut state).unwrap();
        assert_eq!(encoded(decoded, &state), r#"{"a":[1,null,2.5]}"#);
        let special =
            json_decode_builtin(&[Value::text(r#"{"__number__":"-Infinity"}"#)], &mut state)
                .unwrap();
        assert!(special.as_number().unwrap().is_infinite());
        assert!(special.as_number().unwrap().is_sign_negative());
    }

    #[test]
    fn md5_hashes_text_bytes_and_rejects_non_text_values() {
        assert_eq!(
            md5_builtin(&[Value::text("md5_test")]).unwrap(),
            Value::text("c74318b61a3024520c466f828c043c79")
        );
        assert_eq!(md5_builtin(&[Value::number(5.0)]).unwrap(), Value::Null);
        assert_eq!(md5_builtin(&[]).unwrap(), Value::Null);
        assert_eq!(encoded(Value::Null, &ExecutionState::new()), "null");
    }
}

#[cfg(test)]
mod color_text_file_tests {
    use super::*;
    use dm_value::TypePath;
    use std::process::Command;

    #[test]
    fn rgb_round_trips_short_and_alpha_hex_colors() {
        let mut state = ExecutionState::new();
        assert_eq!(
            rgb_builtin(&[Value::number(255.0), Value::number(128.0), Value::Null]).unwrap(),
            Value::text("#ff8000")
        );
        let Value::List(parts) = rgb2num_builtin(&[Value::text("#5af8")], &mut state).unwrap()
        else {
            panic!("rgb2num must return a list")
        };
        let parts = state.heap.list(parts).unwrap();
        assert_eq!(parts.get(1), Ok(&Value::number(85.0)));
        assert_eq!(parts.get(2), Ok(&Value::number(170.0)));
        assert_eq!(parts.get(3), Ok(&Value::number(255.0)));
        assert_eq!(parts.get(4), Ok(&Value::number(136.0)));
    }

    #[test]
    fn rgb2num_converts_hsv_and_hsl_like_opendream() {
        let mut state = ExecutionState::new();
        for (space, expected) in [
            (1.0, [291.70734, 56.164383, 85.882355]),
            (2.0, [291.70734, 63.07692, 61.764706]),
        ] {
            let Value::List(parts) =
                rgb2num_builtin(&[Value::text("#ca60db"), Value::number(space)], &mut state)
                    .unwrap()
            else {
                panic!("rgb2num must return a list")
            };
            let parts = state.heap.list(parts).unwrap();
            for (index, expected) in expected.into_iter().enumerate() {
                let actual = parts.get(index + 1).unwrap().as_number().unwrap();
                assert!(
                    (actual - expected).abs() < 0.0001,
                    "component {index}: {actual}"
                );
            }
        }
    }

    #[test]
    fn rgb2num_treats_null_as_default_white_like_byond_and_opendream() {
        let mut state = ExecutionState::new();
        let Value::List(parts) = rgb2num_builtin(&[Value::Null], &mut state).unwrap() else {
            panic!("rgb2num must return a list")
        };
        let parts = state.heap.list(parts).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts.get(1), Ok(&Value::number(255.0)));
        assert_eq!(parts.get(2), Ok(&Value::number(255.0)));
        assert_eq!(parts.get(3), Ok(&Value::number(255.0)));

        let Value::List(parts) =
            rgb2num_builtin(&[Value::Null, Value::number(2.0)], &mut state).unwrap()
        else {
            panic!("rgb2num must return a list")
        };
        let parts = state.heap.list(parts).unwrap();
        assert_eq!(parts.get(1), Ok(&Value::number(0.0)));
        assert_eq!(parts.get(2), Ok(&Value::number(0.0)));
        assert_eq!(parts.get(3), Ok(&Value::number(100.0)));
    }

    #[test]
    fn gradient_interpolates_rgb_components() {
        let mut state = ExecutionState::new();
        assert_eq!(
            gradient_builtin(
                &[
                    Value::text("#ff0000"),
                    Value::text("#000000"),
                    Value::number(0.2)
                ],
                &mut state
            )
            .unwrap(),
            Value::text("#cc0000")
        );
        assert_eq!(
            gradient_builtin(
                &[
                    Value::number(0.0),
                    Value::text("#ff0000"),
                    Value::number(1.0),
                    Value::text("#000000"),
                    Value::text("loop"),
                    Value::number(0.2),
                ],
                &mut state,
            )
            .unwrap(),
            Value::text("#cc0000")
        );
    }

    #[test]
    fn html_entities_round_trip_without_double_decoding() {
        let state = ExecutionState::new();
        let encoded = html_encode(&[Value::text("<&\"'>")], &state).unwrap();
        assert_eq!(encoded, Value::text("&lt;&amp;&quot;&#39;&gt;"));
        assert_eq!(
            html_decode(&[encoded], &state).unwrap(),
            Value::text("<&\"'>")
        );
    }

    #[test]
    fn realtime_epoch_and_timezone_format_deterministically() {
        let state = ExecutionState::new();
        assert_eq!(
            time2text_builtin(
                &[
                    Value::number(0.0),
                    Value::text("YYYY-MM-DD hh:mm:ss"),
                    Value::number(0.0)
                ],
                &state
            )
            .unwrap(),
            Value::text("2000-01-01 00:00:00")
        );
        assert_eq!(
            time2text_builtin(
                &[
                    Value::number(0.0),
                    Value::text("hh:mm"),
                    Value::number(-5.0)
                ],
                &state
            )
            .unwrap(),
            Value::text("19:00")
        );
    }

    #[test]
    fn filesystem_builtins_and_output_stay_inside_project_root() {
        let root = std::env::temp_dir().join(format!("dream64-vm-files-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("data/logs/nested")).unwrap();
        fs::create_dir_all(root.join("html/changelogs/archive")).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());

        assert_eq!(
            text2file(
                &[Value::text("first"), Value::text("data/logs/runtime.log")],
                &state
            )
            .unwrap(),
            Value::number(1.0)
        );
        execute_output(
            &Value::text("data/logs/runtime.log"),
            &Value::text("second"),
            &mut state,
        )
        .unwrap();
        assert_eq!(
            file2text(&[Value::text("data/logs/runtime.log")], &state).unwrap(),
            Value::text("firstsecond\n")
        );
        assert_eq!(
            fcopy(
                &[
                    Value::text("data/logs/runtime.log"),
                    Value::text("data/logs/copy.log")
                ],
                &state
            )
            .unwrap(),
            Value::number(1.0)
        );
        let Value::List(files) = flist(&[Value::text("data/logs")], &mut state).unwrap() else {
            panic!("flist should return a list");
        };
        let files = state
            .heap()
            .list(files)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(files.len(), 3);
        assert!(files.contains(&Value::text("nested/")));
        assert_eq!(
            file2text(&[Value::file("data/logs/nested/")], &state),
            Ok(Value::Null),
        );
        assert_eq!(
            fexists(&[Value::text("data/logs/runtime.log")], &state),
            Ok(Value::number(1.0))
        );
        assert_eq!(
            fexists(&[Value::text("data/not-created/deeper/dummy.sav")], &state),
            Ok(Value::number(0.0))
        );
        assert_eq!(
            fexists(
                &[Value::text("config/../html/changelogs/archive/2000-01.yml")],
                &state
            ),
            Ok(Value::number(0.0))
        );
        assert_eq!(
            file2text(
                &[Value::text("data/not-created/deeper/missing.txt")],
                &state
            ),
            Ok(Value::Null)
        );
        assert!(fexists(&[Value::text("../outside")], &state).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn file_writes_create_missing_destination_directories_like_byond() {
        let root = std::env::temp_dir().join(format!(
            "dream64-vm-write-parents-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("source.txt"), "payload").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());

        assert_eq!(
            fcopy(
                &[
                    Value::text("source.txt"),
                    Value::text("tmp/md5asfile/deep/copied.txt"),
                ],
                &state,
            )
            .unwrap(),
            Value::number(1.0),
        );
        assert_eq!(
            fs::read_to_string(root.join("tmp/md5asfile/deep/copied.txt")).unwrap(),
            "payload",
        );
        assert_eq!(
            fcopy(
                &[
                    Value::text("missing/source.txt"),
                    Value::text("tmp/missing-copy.txt"),
                ],
                &state,
            )
            .unwrap(),
            Value::number(0.0),
            "a missing source is an ordinary failed copy, not a runtime error",
        );
        assert_eq!(
            fcopy(&[Value::Null, Value::text("tmp/null-copy.txt")], &state),
            Ok(Value::number(0.0)),
            "BYOND reports an invalid/null source as an unsuccessful copy",
        );
        assert!(!root.join("tmp/null-copy.txt").exists());
        assert_eq!(
            text2file(
                &[
                    Value::text("written"),
                    Value::text("generated/nested/value.txt"),
                ],
                &state,
            )
            .unwrap(),
            Value::number(1.0),
        );
        assert_eq!(
            fs::read_to_string(root.join("generated/nested/value.txt")).unwrap(),
            "written",
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fdel_trailing_slash_removes_a_nonempty_directory_tree() {
        let root = std::env::temp_dir().join(format!(
            "dream64-vm-fdel-tree-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("tmp/nested")).unwrap();
        fs::write(root.join("tmp/nested/value.txt"), "value").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());

        assert_eq!(
            fdel(&[Value::text("tmp/")], &state).unwrap(),
            Value::number(1.0)
        );
        assert!(!root.join("tmp").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn text2file_appends_by_default_and_reports_io_failure() {
        let root = std::env::temp_dir().join(format!(
            "dream64-vm-text2file-contract-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("data")).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());

        assert_eq!(
            text2file(&[Value::text("one"), Value::text("data/value.txt")], &state).unwrap(),
            Value::number(1.0)
        );
        assert_eq!(
            text2file(&[Value::text("two"), Value::text("data/value.txt")], &state).unwrap(),
            Value::number(1.0)
        );
        assert_eq!(
            fs::read_to_string(root.join("data/value.txt")).unwrap(),
            "onetwo"
        );
        assert_eq!(
            text2file(&[Value::text("bad"), Value::text("data")], &state).unwrap(),
            Value::number(0.0)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_formatted_timestamp_matches_logger_shape_and_offset() {
        let unix_millis = 946_684_800_123;
        assert_eq!(
            format_unix_timestamp(unix_millis, "%Y-%m-%d %H:%M:%S%.3f %z", 0.0),
            "2000-01-01 00:00:00.123 +0000"
        );
        assert_eq!(
            format_unix_timestamp(unix_millis, "%F %T", -8.0),
            "1999-12-31 16:00:00"
        );
    }

    #[test]
    fn rust_g_logging_family_appends_formats_and_closes() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rustg-log-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");
        for (text, formatted) in [("raw\n", "false"), ("readable", "true")] {
            assert_eq!(
                execute_external_call(
                    &library,
                    &Value::text("log_write"),
                    &[
                        Value::text("data/logs/round/runtime.log"),
                        Value::text(text),
                        Value::text(formatted),
                    ],
                    &mut state,
                ),
                Ok(Value::Null)
            );
        }
        assert_eq!(
            fs::read_to_string(root.join("data/logs/round/runtime.log")).unwrap(),
            "raw\nreadable\n"
        );
        assert_eq!(
            execute_external_call(&library, &Value::text("log_close_all"), &[], &mut state,),
            Ok(Value::Null)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_file_bridge_overwrites_appends_creates_and_rejects_traversal() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-files-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("data")).unwrap();
        fs::create_dir_all(root.join("config")).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");
        assert_eq!(
            execute_external_call(&library, &Value::text("get_version"), &[], &mut state),
            Ok(Value::text(concat!(env!("CARGO_PKG_VERSION"), "-dream64")))
        );

        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_write"),
                &[Value::text("first"), Value::text("data/runtime.log")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "first"
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_exists"),
                &[Value::text("data/runtime.log")],
                &mut state,
            ),
            Ok(Value::text("true"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_read"),
                &[Value::text("data/runtime.log")],
                &mut state,
            ),
            Ok(Value::text("first"))
        );
        // Plexora compares this exact rust-g text result to `"true"` before
        // attempting to read its legacy config.
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_exists"),
                &[Value::text("config/plexora.json")],
                &mut state,
            ),
            Ok(Value::text("false"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("file_read"),
                &[Value::text("data/missing.log")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        execute_external_call(
            &library,
            &Value::text("file_append"),
            &[Value::text("+second"), Value::text("data/runtime.log")],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "first+second"
        );
        execute_external_call(
            &library,
            &Value::text("file_write"),
            &[Value::text("replacement"), Value::text("data/runtime.log")],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("data/runtime.log")).unwrap(),
            "replacement"
        );
        execute_external_call(
            &library,
            &Value::text("file_write"),
            &[
                Value::text("header\n"),
                Value::text("data/logs/2026/08/10/round-start/secret/game.log.json"),
            ],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            fs::read_to_string(root.join("data/logs/2026/08/10/round-start/secret/game.log.json"))
                .unwrap(),
            "header\n",
            "SetupLogs creates its entire dated/category directory tree"
        );
        assert!(
            execute_external_call(
                &library,
                &Value::text("file_write"),
                &[Value::text("escape"), Value::text("../escape.log")],
                &mut state,
            )
            .is_err()
        );
        for function in ["file_exists", "file_read"] {
            assert!(
                execute_external_call(
                    &library,
                    &Value::text(function),
                    &[Value::text("../escape.log")],
                    &mut state,
                )
                .is_err()
            );
        }
        let outside = std::env::temp_dir().join(format!(
            "dream64-rust-g-outside-{}-{:?}.log",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::write(&outside, "outside").unwrap();
        let link = root.join("data/linked.log");
        #[cfg(windows)]
        let linked = std::os::windows::fs::symlink_file(&outside, &link);
        #[cfg(unix)]
        let linked = std::os::unix::fs::symlink(&outside, &link);
        if linked.is_ok() {
            assert!(
                execute_external_call(
                    &library,
                    &Value::text("file_write"),
                    &[Value::text("escape"), Value::text("data/linked.log")],
                    &mut state,
                )
                .is_err(),
                "an existing symlink may not redirect writes outside the project root"
            );
            assert_eq!(fs::read_to_string(&outside).unwrap(), "outside");
        }
        assert!(execute_external_call(&library, &Value::text("unknown"), &[], &mut state).is_err());
        fs::remove_dir_all(root).unwrap();
        fs::remove_file(outside).unwrap();
    }

    #[test]
    fn dreamluau_headless_cleanup_and_configuration_are_safe_but_strict() {
        let mut state = ExecutionState::new();
        let library = Value::text("dreamluau.dll");
        let object = Value::Null;

        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("byond:clear_ref_userdata"),
                std::slice::from_ref(&object),
                &mut state,
            ),
            Ok(Value::Null),
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("byond:set_execution_limit_secs"),
                &[Value::number(5.0)],
                &mut state,
            ),
            Ok(Value::Null),
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("byond:get_traceback"),
                &[Value::number(1.0)],
                &mut state,
            ),
            Ok(Value::Null),
        );
        assert!(
            execute_external_call(
                &library,
                &Value::text("byond:clear_ref_userdata"),
                &[],
                &mut state,
            )
            .is_err()
        );
        assert!(
            execute_external_call(&library, &Value::text("byond:unknown"), &[], &mut state,)
                .is_err()
        );
    }

    #[test]
    fn memorystats_bridge_preserves_monk_report_shape_and_rejects_unknown_exports() {
        let mut state = ExecutionState::new();
        let library = Value::text("memorystats.dll");
        let Value::Text(report) =
            execute_external_call(&library, &Value::text("memory_stats"), &[], &mut state).unwrap()
        else {
            panic!("memory_stats must return text");
        };
        assert!(report.starts_with("Server mem usage:\nprototypes:\n"));
        assert!(report.contains("\nobjects:\n"));
        assert!(report.contains("\nDream64 host:\n\tresident: "));
        assert!(
            execute_external_call(
                &library,
                &Value::text("memory_stats"),
                &[Value::Null],
                &mut state,
            )
            .is_err()
        );
        assert!(
            execute_external_call(&library, &Value::text("unknown"), &[], &mut state,).is_err()
        );
    }

    #[test]
    fn rust_g_iconforge_async_jobs_poll_and_preserve_gags_error_contracts() {
        let root = std::env::temp_dir().join(format!(
            "dream64-iconforge-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("icons")).unwrap();
        fs::write(root.join("icons/base.dmi"), b"headless fixture").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");
        let job = execute_external_call(
            &library,
            &Value::text("iconforge_load_gags_config_async"),
            &[
                Value::text("/datum/greyscale_config/test"),
                Value::text("{\"layers\":[]}"),
                Value::text("icons/base.dmi"),
            ],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("iconforge_check"),
                std::slice::from_ref(&job),
                &mut state,
            ),
            Ok(Value::text("NO RESULTS YET"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("iconforge_check"),
                &[job],
                &mut state,
            ),
            Ok(Value::text("OK"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("iconforge_gags"),
                &[
                    Value::text("/datum/greyscale_config/test"),
                    Value::text("#ffffff"),
                    Value::text("tmp/gags/test.dmi"),
                ],
                &mut state,
            ),
            Ok(Value::text("OK"))
        );
        assert!(root.join("tmp/gags/test.dmi").is_file());
        assert_eq!(
            fs::read(root.join("tmp/gags/test.dmi")).unwrap(),
            b"headless fixture",
            "headless GAGS output must remain a valid copy of the source DMI rather than an empty placeholder",
        );
        let generated = execute_external_call(
            &library,
            &Value::text("iconforge_generate"),
            &[
                Value::text("data/spritesheets/"),
                Value::text("startup"),
                Value::text("{}"),
                Value::text("0"),
                Value::text("0"),
                Value::text("1"),
            ],
            &mut state,
        )
        .unwrap();
        let generated: serde_json::Value =
            serde_json::from_str(&owned_value_text(generated)).unwrap();
        assert_eq!(generated["error"], serde_json::Value::Null);
        assert_eq!(generated["headless"], true);
        assert!(
            generated["sizes"]
                .as_object()
                .is_some_and(|sizes| sizes.is_empty())
        );
        let missing = execute_external_call(
            &library,
            &Value::text("iconforge_load_gags_config"),
            &[
                Value::text("/datum/greyscale_config/missing"),
                Value::text("{}"),
                Value::text("icons/missing.dmi"),
            ],
            &mut state,
        )
        .unwrap();
        assert!(
            owned_value_text(missing)
                .starts_with("IconForge error: Failed to open DMI 'icons/missing.dmi'")
        );
        assert!(
            execute_external_call(&library, &Value::text("iconforge_unknown"), &[], &mut state,)
                .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_sql_bridge_fails_offline_without_aborting_async_pollers() {
        let library = Value::text("rust_g");
        let mut state = ExecutionState::new();
        let connection = execute_external_call(
            &library,
            &Value::text("sql_connect_pool"),
            &[Value::text("{}")],
            &mut state,
        )
        .unwrap();
        let Value::Text(connection) = connection else {
            panic!("SQL connection result should be JSON text");
        };
        let decoded: serde_json::Value = serde_json::from_str(&connection).unwrap();
        assert_eq!(decoded["status"], "err");

        let job = execute_external_call(
            &library,
            &Value::text("sql_query_async"),
            &[
                Value::text("missing"),
                Value::text("SELECT 1"),
                Value::text("[]"),
            ],
            &mut state,
        )
        .unwrap();
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("sql_check_query"),
                std::slice::from_ref(&job),
                &mut state,
            ),
            Ok(Value::text("NO RESULTS YET"))
        );
        let result = execute_external_call(
            &library,
            &Value::text("sql_check_query"),
            &[job],
            &mut state,
        )
        .unwrap();
        let Value::Text(result) = result else {
            panic!("SQL query result should be JSON text");
        };
        let decoded: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(decoded["status"], "offline");
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("sql_check_query"),
                &[Value::text("unknown")],
                &mut state,
            ),
            Ok(Value::text("NO SUCH JOB"))
        );
    }

    #[test]
    fn rust_g_dmi_metadata_degrades_missing_render_resources_to_empty_metadata() {
        let root = std::env::temp_dir().join(format!(
            "dream64-dmi-metadata-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let result = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("dmi_read_metadata"),
            &[Value::text("missing/nested/icon.dmi")],
            &mut state,
        )
        .unwrap();
        let Value::Text(result) = result else {
            panic!("DMI metadata should be JSON text");
        };
        let decoded: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(decoded["width"], 32);
        assert_eq!(decoded["height"], 32);
        assert_eq!(decoded["states"], serde_json::json!([]));
        assert!(
            decoded["headless_error"]
                .as_str()
                .unwrap()
                .contains("missing/nested/icon.dmi")
        );
    }

    #[test]
    fn rust_g_dmi_metadata_reads_png_description_states_and_dimensions() {
        use flate2::Compression;
        use flate2::write::ZlibEncoder;

        let root = std::env::temp_dir().join(format!(
            "dream64-dmi-description-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("icons")).unwrap();
        let description = concat!(
            "# BEGIN DMI\n",
            "version = 4.0\n",
            "width = 480\n",
            "height = 480\n",
            "state = \"cloak\"\n",
            "dirs = 1\n",
            "frames = 1\n",
            "state = \"admin\"\n",
            "dirs = 4\n",
            "frames = 2\n",
            "delay = 1,2\n",
            "# END DMI\n",
        );
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(description.as_bytes()).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        let mut push_chunk = |kind: &[u8; 4], data: &[u8]| {
            png.extend_from_slice(&(data.len() as u32).to_be_bytes());
            png.extend_from_slice(kind);
            png.extend_from_slice(data);
            png.extend_from_slice(&[0; 4]);
        };
        let mut header = Vec::new();
        header.extend_from_slice(&960u32.to_be_bytes());
        header.extend_from_slice(&960u32.to_be_bytes());
        header.extend_from_slice(&[8, 6, 0, 0, 0]);
        push_chunk(b"IHDR", &header);
        let mut text = b"Description\0\0".to_vec();
        text.extend_from_slice(&compressed);
        push_chunk(b"zTXt", &text);
        push_chunk(b"IEND", &[]);
        let dmi_path = root.join("icons/test.dmi");
        fs::write(&dmi_path, png).unwrap();

        for _ in 0..100 {
            let metadata = read_dmi_metadata(&dmi_path).unwrap();
            assert_eq!((metadata.width, metadata.height), (480, 480));
        }
        let physical_reads = DMI_METADATA_PHYSICAL_READS
            .get()
            .unwrap()
            .lock()
            .unwrap()
            .get(&dmi_path)
            .copied()
            .unwrap_or(0);
        assert_eq!(
            physical_reads, 1,
            "unchanged DMI metadata should be parsed once across repeated greyscale/icon queries"
        );

        let mut state = ExecutionState::new();
        state.set_project_root(root);
        let result = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("dmi_read_metadata"),
            &[Value::text("icons/test.dmi")],
            &mut state,
        )
        .unwrap();
        let Value::Text(result) = result else {
            panic!("DMI metadata should be JSON text");
        };
        let decoded: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(decoded["width"], 480);
        assert_eq!(decoded["height"], 480);
        assert_eq!(decoded["states"][0]["name"], "cloak");
        assert_eq!(decoded["states"][1]["name"], "admin");
        assert_eq!(decoded["states"][1]["dirs"], 4);
        assert_eq!(decoded["states"][1]["frames"], 2);
        let icon_states =
            execute_standard_builtin("icon_states", &[Value::text("icons/test.dmi")], &mut state)
                .unwrap();
        let Value::List(icon_states) = icon_states else {
            panic!("icon_states should return a list");
        };
        assert_eq!(
            state
                .heap()
                .list(icon_states)
                .unwrap()
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>(),
            vec![Value::text("cloak"), Value::text("admin")],
        );
        let icon = execute_standard_builtin(
            "icon",
            &[Value::file("icons/test.dmi"), Value::text("cloak")],
            &mut state,
        )
        .unwrap();
        let Value::Datum(icon) = icon else {
            panic!("icon() should return an icon datum");
        };
        assert_eq!(
            state
                .heap()
                .datum_field(icon, &FieldName::parse("_dream64_width").unwrap()),
            Ok(&Value::number(480.0)),
        );
        assert_eq!(
            state
                .heap()
                .datum_field(icon, &FieldName::parse("_dream64_height").unwrap()),
            Ok(&Value::number(480.0)),
        );

        let mut changed = fs::read(&dmi_path).unwrap();
        changed.push(0);
        fs::write(&dmi_path, changed).unwrap();
        assert_eq!(read_dmi_metadata(&dmi_path).unwrap().width, 480);
        assert_eq!(
            DMI_METADATA_PHYSICAL_READS
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .get(&dmi_path)
                .copied(),
            Some(2),
            "length-changing replacement must invalidate cached metadata"
        );
    }

    #[test]
    fn text2num_passes_numbers_and_null_like_byond_516() {
        let state = ExecutionState::new();
        assert_eq!(
            text2num(&[Value::number(-2.5)], &state),
            Ok(Value::number(-2.5)),
        );
        assert_eq!(text2num(&[Value::Null], &state), Ok(Value::Null));
        assert_eq!(
            text2num(&[Value::text("12x")], &state),
            Ok(Value::number(12.0)),
        );
        assert_eq!(text2num(&[Value::text("bad")], &state), Ok(Value::Null),);
    }

    #[test]
    #[ignore = "local allocation-focused release benchmark"]
    fn text2num_borrowed_text_release_benchmark() {
        let input = Value::text("  -12345.75 trailing map constant");
        let iterations = 2_000_000;
        let parse = |text: &str| {
            let text = text.trim_start();
            let bytes = text.as_bytes();
            let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
            let mut saw_digit = false;
            let mut saw_dot = false;
            while let Some(byte) = bytes.get(end).copied() {
                if byte.is_ascii_digit() {
                    saw_digit = true;
                    end += 1;
                } else if byte == b'.' && !saw_dot {
                    saw_dot = true;
                    end += 1;
                } else {
                    break;
                }
            }
            saw_digit.then(|| text[..end].parse::<f32>().ok()).flatten()
        };

        let old_started = std::time::Instant::now();
        for _ in 0..iterations {
            let Value::Text(text) = std::hint::black_box(&input) else {
                unreachable!()
            };
            let owned = text.to_string();
            std::hint::black_box(parse(&owned));
        }
        let old = old_started.elapsed();

        let borrowed_started = std::time::Instant::now();
        for _ in 0..iterations {
            let Value::Text(text) = std::hint::black_box(&input) else {
                unreachable!()
            };
            std::hint::black_box(parse(text));
        }
        let borrowed = borrowed_started.elapsed();
        eprintln!(
            "text2num iterations={iterations} owned={old:?} borrowed={borrowed:?} speedup={:.2}x",
            old.as_secs_f64() / borrowed.as_secs_f64()
        );
    }

    #[test]
    fn text2path_returns_null_for_non_text_and_resolves_valid_text_like_byond_516() {
        let mut state = ExecutionState::new();
        let path = TypePath::parse("/datum/reagent/toxin/carpotoxin").unwrap();
        state.set_type_paths([path.clone()]);
        assert_eq!(
            text2path(&[Value::TypePath(path.clone())], &state),
            Ok(Value::Null),
        );
        assert_eq!(text2path(&[Value::Null], &state), Ok(Value::Null));
        assert_eq!(text2path(&[Value::number(5.0)], &state), Ok(Value::Null));
        let datum = state.heap_mut().allocate_datum(path.clone());
        assert_eq!(text2path(&[Value::Datum(datum)], &state), Ok(Value::Null));
        assert_eq!(
            text2path(&[Value::text(path.as_str())], &state),
            Ok(Value::TypePath(path)),
        );
        assert_eq!(
            text2path(&[Value::text("/datum/not_real")], &state),
            Ok(Value::Null),
        );
    }

    #[test]
    #[ignore = "release-only TGM text2path lookup benchmark"]
    fn tgm_text2path_catalog_lookup_benchmark() {
        const PATHS: usize = 10_000;
        const ROUNDS: usize = 2_000;
        let catalog = (0..PATHS)
            .map(|index| TypePath::parse(&format!("/obj/generated/path_{index:05}")).unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        let needle = "/obj/generated/path_09999";
        let run = |indexed: bool| {
            let started = std::time::Instant::now();
            for _ in 0..ROUNDS {
                let found = if indexed {
                    catalog.get(needle)
                } else {
                    catalog.iter().find(|path| path.as_str() == needle)
                };
                std::hint::black_box(found);
            }
            started.elapsed()
        };
        let linear = run(false);
        let indexed = run(true);
        eprintln!(
            "tgm-text2path paths={PATHS} rounds={ROUNDS} linear_ms={} indexed_ms={} speedup={:.2}",
            linear.as_millis(),
            indexed.as_millis(),
            linear.as_secs_f64() / indexed.as_secs_f64(),
        );
        assert!(indexed < linear);
    }

    #[test]
    fn rust_g_cellular_noise_is_bounded_row_major_and_binary() {
        let library = Value::text("rust_g");
        let function = Value::text("cnoise_generate");
        let arguments = [
            Value::text("45"),
            Value::text("3"),
            Value::text("4"),
            Value::text("3"),
            Value::text("4"),
            Value::text("3"),
        ];
        let mut first_state = ExecutionState::new();
        let first = execute_external_call(&library, &function, &arguments, &mut first_state)
            .expect("documented cellular-noise call should succeed");
        let Value::Text(first) = first else {
            panic!("cellular noise must return text")
        };
        assert_eq!(first.len(), 12);
        assert!(first.bytes().all(|byte| matches!(byte, b'0' | b'1')));

        assert_eq!(
            execute_external_call(
                &library,
                &function,
                &[
                    Value::text("0"),
                    Value::text("1"),
                    Value::text("4"),
                    Value::text("3"),
                    Value::text("5"),
                    Value::text("4"),
                ],
                &mut ExecutionState::new(),
            ),
            Ok(Value::text("0".repeat(20))),
            "rust-g ignores out-of-bounds neighbours instead of closing map edges"
        );

        let mut second_state = ExecutionState::new();
        assert_eq!(
            execute_external_call(&library, &function, &arguments, &mut second_state),
            Ok(Value::text(first)),
            "equal headless random streams must produce equal row-major maps"
        );
        assert!(
            execute_external_call(
                &library,
                &function,
                &[
                    Value::text("45"),
                    Value::text("3"),
                    Value::text("4"),
                    Value::text("3"),
                    Value::text("0"),
                    Value::text("3"),
                ],
                &mut ExecutionState::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn rust_g_poisson_noise_matches_station_row_major_contract() {
        let library = Value::text("rust_g");
        let function = Value::text("noise_poisson_map");
        let arguments = [
            Value::text("1337"),
            Value::text("32"),
            Value::text("24"),
            Value::text("6"),
        ];
        let first =
            execute_external_call(&library, &function, &arguments, &mut ExecutionState::new())
                .expect("documented Poisson-noise call should succeed");
        let Value::Text(first) = first else {
            panic!("Poisson noise must return text")
        };
        assert_eq!(first.len(), 32 * 24);
        assert!(first.bytes().all(|byte| matches!(byte, b'0' | b'1')));
        assert!(first.contains('1'));
        assert!(first.contains('0'));
        assert_eq!(
            execute_external_call(&library, &function, &arguments, &mut ExecutionState::new(),),
            Ok(Value::text(first)),
            "the explicit rust-g seed must produce a stable station sample",
        );
    }

    #[test]
    fn rust_g_git_bridge_resolves_head_formats_dates_and_rejects_unsafe_revisions() {
        if Command::new("git").arg("--version").output().is_err() {
            return;
        }
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-git-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let git = |arguments: &[&str]| {
            Command::new("git")
                .args(arguments)
                .current_dir(&root)
                .output()
                .unwrap()
        };
        assert!(git(&["init", "--quiet"]).status.success());
        assert!(
            git(&["config", "user.name", "Dream64 Test"])
                .status
                .success()
        );
        assert!(
            git(&["config", "user.email", "dream64@example.invalid"])
                .status
                .success()
        );
        fs::write(root.join("tracked.txt"), "fixture").unwrap();
        assert!(git(&["add", "tracked.txt"]).status.success());
        let status = Command::new("git")
            .args(["commit", "--quiet", "-m", "fixture"])
            .current_dir(&root)
            .env("GIT_AUTHOR_DATE", "2020-01-02T03:04:05Z")
            .env("GIT_COMMITTER_DATE", "2020-01-02T03:04:05Z")
            .status()
            .unwrap();
        assert!(status.success());

        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");
        let Value::Text(head) = execute_external_call(
            &library,
            &Value::text("rg_git_revparse"),
            &[Value::text("HEAD")],
            &mut state,
        )
        .unwrap() else {
            panic!("HEAD should resolve to text");
        };
        assert_eq!(head.len(), 40);
        assert!(head.chars().all(|character| character.is_ascii_hexdigit()));
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("rg_git_revparse"),
                &[Value::text("refs/heads/does-not-exist")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("rg_git_commit_date"),
                &[Value::text("HEAD"), Value::text("%F")],
                &mut state,
            ),
            Ok(Value::text("2020-01-02"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("rg_git_commit_date_head"),
                &[Value::text("%F")],
                &mut state,
            ),
            Ok(Value::text("2020-01-02"))
        );
        for unsafe_revision in [
            "--help",
            "../../outside",
            "HEAD;status",
            "HEAD refs/heads/x",
        ] {
            assert!(
                execute_external_call(
                    &library,
                    &Value::text("rg_git_revparse"),
                    &[Value::text(unsafe_revision)],
                    &mut state,
                )
                .is_err()
            );
        }
        assert!(
            execute_external_call(
                &library,
                &Value::text("rg_git_commit_date"),
                &[Value::text("HEAD"), Value::text("%F\n--pretty=%s")],
                &mut state,
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_toml_bridge_returns_double_encoded_config_envelope() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-toml-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("config")).unwrap();
        fs::write(root.join("config/settings.toml"), "# config\n[shared]\n\"# phrase\" = \"blocked # text\"\nenabled = true\nweights = [1, -2, 3.5]\n[server.network]\nport = 1337\n[[relay]]\nid = \"east\"\naddress = \"byond://east:{port}\"\n[[relay]]\nid = \"direct\"\naddress = \"byond://direct:{port}\"\n").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let Value::Text(envelope) = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("toml_file_to_json"),
            &[Value::text("config/settings.toml")],
            &mut state,
        )
        .unwrap() else {
            panic!("TOML bridge should return text")
        };
        let envelope: serde_json::Value = serde_json::from_str(&envelope).unwrap();
        assert_eq!(envelope["success"], true);
        let document: serde_json::Value =
            serde_json::from_str(envelope["content"].as_str().unwrap()).unwrap();
        assert_eq!(document["shared"]["# phrase"], "blocked # text");
        assert_eq!(document["shared"]["weights"][2], 3.5);
        assert_eq!(document["server"]["network"]["port"], 1337);
        assert_eq!(document["relay"][1]["id"], "direct");

        let Value::Text(missing) = execute_external_call(
            &Value::text("rust_g"),
            &Value::text("toml_file_to_json"),
            &[Value::text("config/missing.toml")],
            &mut state,
        )
        .unwrap() else {
            unreachable!()
        };
        let missing: serde_json::Value = serde_json::from_str(&missing).unwrap();
        assert_eq!(missing["success"], false);
        assert!(!missing["content"].as_str().unwrap().is_empty());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rust_g_named_timers_reset_and_return_numeric_text() {
        let mut state = ExecutionState::new();
        let library = Value::text("rust_g");
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("time_reset"),
                &[Value::text("subsystem")],
                &mut state,
            ),
            Ok(Value::Null)
        );
        let Value::Text(milliseconds) = execute_external_call(
            &library,
            &Value::text("time_milliseconds"),
            &[Value::text("subsystem")],
            &mut state,
        )
        .unwrap() else {
            panic!("timer should return numeric text")
        };
        assert!(milliseconds.parse::<f64>().is_ok());
        let Value::Text(microseconds) = execute_external_call(
            &library,
            &Value::text("time_microseconds"),
            &[Value::text("subsystem")],
            &mut state,
        )
        .unwrap() else {
            panic!("timer should return numeric text")
        };
        assert!(microseconds.parse::<f64>().is_ok());
    }

    #[test]
    fn rust_g_url_codec_matches_ref_tags_and_form_encoding() {
        let mut state = ExecutionState::new();
        let library = Value::text("rust_g");
        // Monkestation's REF() wraps this result in literal brackets when a
        // datum opts into tag-backed references. Spaces use `+`; Unicode is
        // encoded bytewise as UTF-8; URL-reserved characters are escaped.
        let tag = "suicide: Résumé /?x=1+2&[]#%";
        let encoded = "suicide%3A+R%C3%A9sum%C3%A9+%2F%3Fx%3D1%2B2%26%5B%5D%23%25";
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("url_encode"),
                &[Value::text(tag)],
                &mut state,
            ),
            Ok(Value::text(encoded)),
        );
        assert_eq!(
            format!("[{encoded}]"),
            "[suicide%3A+R%C3%A9sum%C3%A9+%2F%3Fx%3D1%2B2%26%5B%5D%23%25]",
            "REF() keeps the encoded rust-g payload inside literal brackets",
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("url_decode"),
                &[Value::text(encoded)],
                &mut state,
            ),
            Ok(Value::text(tag)),
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("url_decode"),
                &[Value::text("a+b=c%20d&e%23f=g;%2b=%zz")],
                &mut state,
            ),
            Ok(Value::text("a b=c d&e#f=g;+=%zz")),
            "decode treats plus as space and leaves malformed escapes intact",
        );
        assert!(
            execute_external_call(
                &library,
                &Value::text("url_encode_extra"),
                &[Value::text(tag)],
                &mut state,
            )
            .unwrap_err()
            .contains("installed host bridge"),
            "nearby unknown exports must remain strict",
        );
    }

    #[test]
    fn rust_g_startup_hash_and_json_utility_family_is_real_and_sandboxed() {
        let root = std::env::temp_dir().join(format!(
            "dream64-rust-g-utilities-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("asset.css"), b"startup").unwrap();
        let mut state = ExecutionState::new();
        state.set_project_root(root.clone());
        let library = Value::text("rust_g");
        let expected = Value::text(format!("{:x}", md5::compute(b"startup")));
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("hash_string"),
                &[Value::text("md5"), Value::text("startup")],
                &mut state,
            ),
            Ok(expected.clone())
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("hash_file"),
                &[Value::text("md5"), Value::text("asset.css")],
                &mut state,
            ),
            Ok(expected.clone())
        );
        let icon = state
            .heap_mut()
            .allocate_datum(TypePath::parse("/icon").unwrap());
        state
            .heap_mut()
            .set_datum_field(
                icon,
                FieldName::parse("icon").unwrap(),
                Value::file("asset.css"),
            )
            .unwrap();
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("hash_file"),
                &[Value::text("md5"), Value::Datum(icon)],
                &mut state,
            ),
            Ok(expected),
            "rust-g observes an icon's backing resource path at the native call boundary",
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("json_is_valid"),
                &[Value::text("{\"ready\":true}")],
                &mut state,
            ),
            Ok(Value::text("true"))
        );
        assert_eq!(
            execute_external_call(
                &library,
                &Value::text("json_is_valid"),
                &[Value::text("{broken")],
                &mut state,
            ),
            Ok(Value::text("false"))
        );
        assert!(
            execute_external_call(
                &library,
                &Value::text("hash_file"),
                &[Value::text("md5"), Value::text("../outside")],
                &mut state,
            )
            .is_err()
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod atmos_batch_tests {
    use super::*;
    use dm_value::TypePath;

    fn mixture(state: &mut ExecutionState, oxygen: f32) -> DatumId {
        let mixture = state
            .heap
            .allocate_datum(TypePath::parse("/datum/gas_mixture").unwrap());
        let gases = state.heap.allocate_list();
        let values = state.heap.allocate_list();
        state
            .heap
            .list_mut(values)
            .unwrap()
            .add(Value::number(oxygen));
        state
            .heap
            .list_mut(gases)
            .unwrap()
            .set_key(Value::text("o2"), Value::List(values));
        state
            .heap
            .set_datum_field(mixture, atmos_field("gases"), Value::List(gases))
            .unwrap();
        state
            .heap
            .set_datum_field(mixture, atmos_field("temperature"), Value::number(293.15))
            .unwrap();
        mixture
    }

    #[test]
    fn native_atmos_batch_snapshots_workers_and_commits_in_turf_order() {
        let mut state = ExecutionState::new();
        let first = state
            .heap
            .allocate_datum(TypePath::parse("/turf/open/test").unwrap());
        let second = state
            .heap
            .allocate_datum(TypePath::parse("/turf/open/test").unwrap());
        let first_air = mixture(&mut state, 10.0);
        let second_air = mixture(&mut state, 1.0);
        for (turf, air, neighbor) in [(first, first_air, second), (second, second_air, first)] {
            let adjacent = state.heap.allocate_list();
            state
                .heap
                .list_mut(adjacent)
                .unwrap()
                .add(Value::Datum(neighbor));
            for (field, value) in [
                (atmos_field("air"), Value::Datum(air)),
                (atmos_field("atmos_adjacent_turfs"), Value::List(adjacent)),
                (atmos_field("current_cycle"), Value::number(0.0)),
                (atmos_field("excited"), Value::number(0.0)),
            ] {
                state.heap.set_datum_field(turf, field, value).unwrap();
            }
        }
        let difference = state.heap.allocate_list();
        state
            .heap
            .list_mut(difference)
            .unwrap()
            .add(Value::Datum(first));
        state
            .heap
            .list_mut(difference)
            .unwrap()
            .add(Value::Datum(second));
        let active = state.heap.allocate_list();
        assert_eq!(
            atmos_setup_differences(
                &[
                    Value::List(difference),
                    Value::List(active),
                    Value::number(0.01),
                    Value::number(0.001),
                    Value::number(4.0),
                ],
                &mut state,
            ),
            Ok(Value::Null),
        );
        assert_eq!(
            state
                .heap
                .list(active)
                .unwrap()
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>(),
            [Value::Datum(first), Value::Datum(second)],
        );
        for turf in [first, second] {
            assert_eq!(
                super::super::datum_field_or_initial(&state, turf, &atmos_field("current_cycle"))
                    .unwrap(),
                Value::number(f32::NEG_INFINITY),
            );
        }
    }
}

#[cfg(test)]
mod spatial_tests {
    use super::*;
    use dm_value::TypePath;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn list_union_does_not_propagate_destroyed_object_handles_as_null() {
        let mut state = ExecutionState::new();
        let live = state
            .heap
            .allocate_datum(TypePath::parse("/mob/live").unwrap());
        let destroyed = state
            .heap
            .allocate_datum(TypePath::parse("/mob/destroyed").unwrap());
        let source = state.heap.allocate_list();
        {
            let source = state.heap.list_mut(source).unwrap();
            source.add(Value::Datum(live));
            source.add(Value::Null);
            source.add(Value::Datum(destroyed));
        }
        state.heap.destroy_datum(destroyed).unwrap();

        let result = state.heap.allocate_list();
        execute_list_compound_operator(
            CompoundAssignmentOperator::BitOr,
            result,
            &Value::List(source),
            &mut state,
        )
        .unwrap();

        let values = state
            .heap
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert_eq!(values, [Value::Datum(live), Value::Null]);
        assert!(
            values
                .iter()
                .all(|value| !value.semantic_eq(&Value::Datum(destroyed)))
        );
    }

    fn place(state: &mut ExecutionState, path: &str, x: f32, y: f32) -> dm_value::DatumId {
        let id = state.heap.allocate_datum(TypePath::parse(path).unwrap());
        for (name, value) in [("x", x), ("y", y), ("z", 1.0)] {
            state
                .heap
                .set_datum_field(id, FieldName::parse(name).unwrap(), Value::number(value))
                .unwrap();
        }
        id
    }

    fn place_world_turf(state: &mut ExecutionState, x: i32, y: i32) -> DatumId {
        let turf = place(state, "/turf/open", x as f32, y as f32);
        state
            .ensure_contents(turf)
            .expect("indexed fixture turf contents should materialize");
        state.world_turfs.insert((x, y, 1), turf);
        turf
    }

    #[test]
    fn indexed_view_uses_authoritative_cell_membership_without_rewalking_coordinates() {
        let mut state = ExecutionState::new();
        let center = place_world_turf(&mut state, 5, 5);
        let east = place_world_turf(&mut state, 6, 5);
        let member = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/indexed").unwrap());
        move_movable_to_turf(&mut state, member, east).unwrap();
        // x/y/z on a contained movable are derived from loc. Bogus direct
        // fields must not cause a second geometry pass to reject it.
        for (name, value) in [("x", 900.0), ("y", 901.0), ("z", 7.0)] {
            state
                .heap
                .set_datum_field(
                    member,
                    FieldName::parse(name).unwrap(),
                    Value::number(value),
                )
                .unwrap();
        }

        assert!(
            spatial_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(center)],
                false,
                false,
            )
            .contains(&member)
        );
        assert!(
            !spatial_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(east)],
                false,
                true,
            )
            .contains(&member)
        );
    }

    #[test]
    #[ignore = "release-only bounded spatial-index benchmark"]
    fn benchmark_indexed_view_skips_redundant_coordinate_resolution() {
        let mut state = ExecutionState::new();
        let side = 31_i32;
        for x in 1..=side {
            for y in 1..=side {
                let turf = place_world_turf(&mut state, x, y);
                for _ in 0..3 {
                    let member = state
                        .heap
                        .allocate_datum(TypePath::parse("/obj/effect/view_bench").unwrap());
                    move_movable_to_turf(&mut state, member, turf).unwrap();
                }
            }
        }
        let center = state.turf_at(16, 16, 1).unwrap();
        let iterations = 2_000;

        let optimized_started = Instant::now();
        for _ in 0..iterations {
            black_box(
                spatial_query(
                    &[Value::number(7.0), Value::Datum(center)],
                    &mut state,
                    &Value::Null,
                    false,
                    false,
                )
                .unwrap(),
            );
        }
        let optimized = optimized_started.elapsed();

        let reference_started = Instant::now();
        for _ in 0..iterations {
            let candidates = indexed_spatial_candidates(&state, 16.0, 16.0, 1.0, 7.0, 7.0, false);
            let output = state.heap.allocate_list();
            for id in &candidates {
                if let Some((x, y, z)) = datum_coordinates(&state, &Value::Datum(*id))
                    && z == 1.0
                    && (x - 16.0).abs() <= 7.0
                    && (y - 16.0).abs() <= 7.0
                {
                    state.heap.list_mut(output).unwrap().add(Value::Datum(*id));
                }
            }
            black_box(output);
        }
        let redundant_validation = reference_started.elapsed();
        eprintln!(
            "indexed-view-benchmark iterations={iterations} optimized_ms={} reference_ms={}",
            optimized.as_millis(),
            redundant_validation.as_millis(),
        );
    }

    fn spatial_result(
        state: &mut ExecutionState,
        arguments: &[Value],
        mobs_only: bool,
        exclude_center: bool,
    ) -> Vec<DatumId> {
        let Value::List(result) =
            spatial_query(arguments, state, &Value::Null, mobs_only, exclude_center).unwrap()
        else {
            panic!("spatial query must return a list")
        };
        state
            .heap
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| match value {
                Value::Datum(datum) => *datum,
                value => panic!("spatial query returned non-datum {value}"),
            })
            .collect()
    }

    fn orange_result(state: &mut ExecutionState, arguments: &[Value], usr: &Value) -> Vec<DatumId> {
        let Value::List(result) =
            execute_standard_builtin_with_usr("orange", arguments, state, usr).unwrap()
        else {
            panic!("orange must return a list")
        };
        state
            .heap
            .list(result)
            .unwrap()
            .positions()
            .map(|(_, value)| match value {
                Value::Datum(datum) => *datum,
                value => panic!("orange returned non-datum {value}"),
            })
            .collect()
    }

    #[test]
    fn view_families_filter_distance_center_and_mob_type() {
        let mut state = ExecutionState::new();
        let center = place(&mut state, "/turf/open", 5.0, 5.0);
        place(&mut state, "/mob/living", 6.0, 5.0);
        place(&mut state, "/obj/item", 6.0, 6.0);
        place(&mut state, "/mob/living", 9.0, 5.0);
        let Value::List(view) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            &Value::Null,
            false,
            false,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(view).unwrap().len(), 3);
        let Value::List(viewers) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            &Value::Null,
            true,
            false,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(viewers).unwrap().len(), 1);
        let Value::List(oview) = spatial_query(
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
            &Value::Null,
            false,
            true,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(oview).unwrap().len(), 2);

        let Value::List(oviewers) = execute_standard_builtin(
            "oviewers",
            &[Value::number(1.0), Value::Datum(center)],
            &mut state,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(state.heap.list(oviewers).unwrap().len(), 1);
    }

    #[test]
    fn view_families_default_to_usr_and_accept_arguments_in_either_order() {
        let mut state = ExecutionState::new();
        let world = state
            .heap
            .allocate_datum(TypePath::parse("/world").unwrap());
        state
            .heap
            .set_datum_field(world, FieldName::parse("view").unwrap(), Value::number(2.0))
            .unwrap();
        state.set_global(FieldName::parse("world").unwrap(), Value::Datum(world));
        let center = place(&mut state, "/mob/living", 5.0, 5.0);
        place(&mut state, "/mob/living", 7.0, 5.0);
        place(&mut state, "/mob/living", 8.0, 5.0);

        let Value::List(defaulted) =
            execute_standard_builtin_with_usr("viewers", &[], &mut state, &Value::Datum(center))
                .unwrap()
        else {
            panic!()
        };
        assert_eq!(state.heap.list(defaulted).unwrap().len(), 2);

        for arguments in [
            vec![Value::number(1.0), Value::Datum(center)],
            vec![Value::Datum(center), Value::number(1.0)],
        ] {
            let Value::List(result) =
                execute_standard_builtin_with_usr("viewers", &arguments, &mut state, &Value::Null)
                    .unwrap()
            else {
                panic!()
            };
            assert_eq!(state.heap.list(result).unwrap().len(), 1);
        }

        let Value::List(no_usr) =
            execute_standard_builtin_with_usr("viewers", &[], &mut state, &Value::Null).unwrap()
        else {
            panic!()
        };
        assert!(state.heap.list(no_usr).unwrap().is_empty());
    }

    #[test]
    fn indexed_view_bounds_direct_contents_filters_and_excludes_nested_inventory() {
        let mut state = ExecutionState::new();
        let center = place_world_turf(&mut state, 5, 5);
        let near = place_world_turf(&mut state, 6, 5);
        let far = place_world_turf(&mut state, 8, 5);
        let area = state
            .heap
            .allocate_datum(TypePath::parse("/area/station").unwrap());
        state.world_areas.insert((6, 5, 1), area);

        let container = state
            .heap
            .allocate_datum(TypePath::parse("/obj/structure/closet").unwrap());
        move_movable_to_turf(&mut state, container, near).unwrap();
        state.ensure_contents(container).unwrap();
        let nested_mob = state
            .heap
            .allocate_datum(TypePath::parse("/mob/living/nested").unwrap());
        move_movable_to_atom(&mut state, nested_mob, container).unwrap();
        let direct_object = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/direct").unwrap());
        move_movable_to_turf(&mut state, direct_object, near).unwrap();
        let center_mob = state
            .heap
            .allocate_datum(TypePath::parse("/mob/living/center").unwrap());
        move_movable_to_turf(&mut state, center_mob, center).unwrap();

        // A coordinate-bearing atom that is not a member of any bounded
        // turf must not leak in merely because it occupies a heap slot.
        let unrelated = place(&mut state, "/obj/item/unrelated", 6.0, 5.0);
        let far_mob = state
            .heap
            .allocate_datum(TypePath::parse("/mob/living/far").unwrap());
        move_movable_to_turf(&mut state, far_mob, far).unwrap();

        // Corrupt duplicate contents entries are tolerated without returning
        // the same atom twice.
        let near_contents = state.ensure_contents(near).unwrap();
        state
            .heap
            .list_mut(near_contents)
            .unwrap()
            .add(Value::Datum(container));

        let arguments = [Value::number(1.0), Value::Datum(center)];
        assert_eq!(
            spatial_result(&mut state, &arguments, false, false),
            vec![center, center_mob, near, container, direct_object]
        );
        assert_eq!(
            spatial_result(&mut state, &arguments, true, false),
            vec![center_mob]
        );
        assert_eq!(
            spatial_result(&mut state, &arguments, false, true),
            vec![near, container, direct_object]
        );
        assert_eq!(
            spatial_result(&mut state, &arguments, true, true),
            Vec::<DatumId>::new()
        );

        for absent in [far, area, nested_mob, unrelated, far_mob] {
            assert!(!spatial_result(&mut state, &arguments, false, false).contains(&absent));
        }
    }

    #[test]
    fn indexed_view_uses_center_then_concentric_spiral_and_contents_order() {
        let mut state = ExecutionState::new();
        let southwest = place_world_turf(&mut state, 4, 4);
        let west = place_world_turf(&mut state, 4, 5);
        let northwest = place_world_turf(&mut state, 4, 6);
        let south = place_world_turf(&mut state, 5, 4);
        let center = place_world_turf(&mut state, 5, 5);
        let north = place_world_turf(&mut state, 5, 6);
        let southeast = place_world_turf(&mut state, 6, 4);
        let east = place_world_turf(&mut state, 6, 5);
        let northeast = place_world_turf(&mut state, 6, 6);
        let first = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/first").unwrap());
        let second = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/second").unwrap());
        move_movable_to_turf(&mut state, first, center).unwrap();
        move_movable_to_turf(&mut state, second, center).unwrap();

        assert_eq!(
            spatial_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(center)],
                false,
                false,
            ),
            vec![
                center, first, second, southwest, west, northwest, south, north, southeast, east,
                northeast,
            ]
        );
    }

    #[test]
    fn indexed_view_respects_rectangular_text_bounds_and_stale_members() {
        let mut state = ExecutionState::new();
        let center = place_world_turf(&mut state, 5, 5);
        let vertical = place_world_turf(&mut state, 5, 6);
        let horizontal = place_world_turf(&mut state, 6, 5);
        let stale = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/stale").unwrap());
        move_movable_to_turf(&mut state, stale, vertical).unwrap();
        state.heap.destroy_datum(stale).unwrap();

        assert_eq!(
            spatial_result(
                &mut state,
                &[Value::text("1x3"), Value::Datum(center)],
                false,
                false,
            ),
            vec![center, vertical]
        );
        assert!(
            !spatial_result(
                &mut state,
                &[Value::text("1x3"), Value::Datum(center)],
                false,
                false,
            )
            .contains(&horizontal)
        );
    }

    #[test]
    fn indexed_view_uses_non_turf_centers_and_live_direct_membership() {
        let mut state = ExecutionState::new();
        let old_turf = place_world_turf(&mut state, 10, 10);
        let new_turf = place_world_turf(&mut state, 11, 10);
        let center = state
            .heap
            .allocate_datum(TypePath::parse("/mob/living/center").unwrap());
        move_movable_to_turf(&mut state, center, old_turf).unwrap();
        state.ensure_contents(center).unwrap();
        let inventory = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/inventory").unwrap());
        move_movable_to_atom(&mut state, inventory, center).unwrap();
        let moving = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/moving").unwrap());
        move_movable_to_turf(&mut state, moving, old_turf).unwrap();
        move_movable_to_turf(&mut state, moving, new_turf).unwrap();

        let centered = [Value::number(0.0), Value::Datum(center)];
        assert_eq!(
            spatial_result(&mut state, &centered, false, false),
            vec![old_turf, center]
        );
        assert!(spatial_result(&mut state, &centered, false, true).is_empty());
        assert!(!spatial_result(&mut state, &centered, false, false).contains(&inventory));
        assert!(!spatial_result(&mut state, &centered, false, false).contains(&moving));

        assert_eq!(
            spatial_result(
                &mut state,
                &[Value::number(0.0), Value::Datum(new_turf)],
                false,
                false,
            ),
            vec![new_turf, moving]
        );
    }

    #[test]
    fn orange_compiles_to_one_native_standard_builtin_instruction() {
        use dm_syntax::parse;

        let syntax = parse("/proc/run(center)\n\treturn orange(3, center)\n").unwrap();
        let module = crate::compile_module(&syntax.definitions).unwrap();
        let entry = module.procedure_id("/proc/run").unwrap();
        let program = module.procedure(entry).unwrap();
        assert!(program.instructions.iter().any(|instruction| matches!(
            instruction,
            crate::Instruction::StandardBuiltin {
                name,
                argument_count: 2,
                ..
            } if name == "orange"
        )));
        assert!(!program.instructions.iter().any(|instruction| matches!(
            instruction,
            crate::Instruction::Call { .. } | crate::Instruction::CallDynamic { .. }
        )));
    }

    #[test]
    fn indexed_orange_preserves_range_order_and_direct_loc_exclusions() {
        let mut state = ExecutionState::new();
        let center = place_world_turf(&mut state, 5, 5);
        let west = place_world_turf(&mut state, 4, 5);
        let center_area = state
            .heap
            .allocate_datum(TypePath::parse("/area/center").unwrap());
        let west_area = state
            .heap
            .allocate_datum(TypePath::parse("/area/west").unwrap());
        let loc = FieldName::parse("loc").unwrap();
        for area in [center_area, west_area] {
            state
                .heap
                .set_datum_field(area, loc.clone(), Value::Null)
                .unwrap();
        }
        state.world_areas.insert((5, 5, 1), center_area);
        state.world_areas.insert((4, 5, 1), west_area);

        let center_object = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/center").unwrap());
        move_movable_to_turf(&mut state, center_object, center).unwrap();
        let west_object = state
            .heap
            .allocate_datum(TypePath::parse("/obj/item/west").unwrap());
        move_movable_to_turf(&mut state, west_object, west).unwrap();

        // Same coordinates are insufficient: indexed orange must never inspect
        // or return an unrelated atom outside the turf membership graph.
        let unrelated = place(&mut state, "/obj/item/unrelated", 4.0, 5.0);
        state
            .heap
            .set_datum_field(unrelated, loc.clone(), Value::Null)
            .unwrap();

        let before_lists = state.heap.live_list_count();
        assert_eq!(
            orange_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(center)],
                &Value::Null,
            ),
            vec![center_area, west, west_area, west_object]
        );
        assert_eq!(
            state.heap.live_list_count(),
            before_lists + 1,
            "native orange should allocate only its output list"
        );
        assert!(
            !orange_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(center)],
                &Value::Null,
            )
            .contains(&unrelated)
        );
        assert_eq!(
            orange_result(&mut state, &[Value::number(0.0)], &Value::Datum(center)),
            vec![center_area],
            "the omitted second argument defaults to usr"
        );
        assert_eq!(
            orange_result(
                &mut state,
                &[Value::Datum(center), Value::number(1.0)],
                &Value::Null,
            ),
            vec![center_area, west, west_area, west_object],
            "orange accepts center and distance in reversed order"
        );
    }

    #[test]
    fn synthetic_orange_fallback_filters_non_atoms_center_and_direct_children() {
        let mut state = ExecutionState::new();
        let center = place(&mut state, "/turf/open", 3.0, 3.0);
        let loc = FieldName::parse("loc").unwrap();
        state
            .heap
            .set_datum_field(center, loc.clone(), Value::Null)
            .unwrap();
        let neighbor = place(&mut state, "/obj/item/neighbor", 4.0, 3.0);
        state
            .heap
            .set_datum_field(neighbor, loc.clone(), Value::Null)
            .unwrap();
        let direct_child = place(&mut state, "/obj/item/child", 3.0, 3.0);
        state
            .heap
            .set_datum_field(direct_child, loc.clone(), Value::Datum(center))
            .unwrap();
        let non_atom = place(&mut state, "/datum/coordinates", 4.0, 3.0);
        state
            .heap
            .set_datum_field(non_atom, loc, Value::Null)
            .unwrap();

        assert_eq!(
            orange_result(
                &mut state,
                &[Value::number(1.0), Value::Datum(center)],
                &Value::Null,
            ),
            vec![neighbor]
        );
    }

    #[test]
    fn headless_ui_retains_window_and_resource_transport_state() {
        let mut state = ExecutionState::new();
        let client = state
            .heap
            .allocate_datum(TypePath::parse("/client").unwrap());
        assert_eq!(
            execute_standard_builtin(
                "winset",
                &[
                    Value::Datum(client),
                    Value::text("mapwindow"),
                    Value::text("size=640x480;focus=true"),
                ],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            execute_standard_builtin(
                "winget",
                &[
                    Value::Datum(client),
                    Value::text("mapwindow"),
                    Value::text("size"),
                ],
                &mut state,
            ),
            Ok(Value::text("640x480"))
        );

        for (builtin, resource, name) in [
            ("browse_rsc", "icons/a.dmi", "a.dmi"),
            ("ftp", "data/report.txt", "report.txt"),
        ] {
            let event = execute_standard_builtin(
                builtin,
                &[Value::text(resource), Value::text(name)],
                &mut state,
            )
            .unwrap();
            execute_output(&Value::Datum(client), &event, &mut state).unwrap();
        }
        let Value::List(events) = state
            .heap
            .datum_field(client, &FieldName::parse("_dream64_output_events").unwrap())
            .unwrap()
        else {
            panic!("headless client output should retain transport events")
        };
        assert_eq!(state.heap.list(*events).unwrap().len(), 2);
        for (index, kind) in [(1, "browse_rsc"), (2, "ftp")] {
            let Value::List(event) = state.heap.list(*events).unwrap().get(index).unwrap() else {
                panic!("transport event should be an associative descriptor")
            };
            assert_eq!(
                state
                    .heap
                    .list(*event)
                    .unwrap()
                    .get_key(&Value::text("kind")),
                Ok(&Value::text(kind))
            );
        }
    }

    #[test]
    fn registered_client_sessions_receive_authoritative_window_builtins() {
        let mut state = ExecutionState::new();
        let client = state
            .open_local_client(
                "window \"main\"\n\telem \"status\"\n\t\ttype = LABEL\n\t\tis-visible = true\n",
            )
            .expect("a valid local skin should create a client");

        assert_eq!(
            execute_standard_builtin(
                "winset",
                &[
                    Value::Datum(client),
                    Value::text("main.status"),
                    Value::text("text=connected"),
                ],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            execute_standard_builtin(
                "winget",
                &[
                    Value::Datum(client),
                    Value::text("main.status"),
                    Value::text("text"),
                ],
                &mut state,
            ),
            Ok(Value::text("connected"))
        );
        assert_eq!(
            execute_standard_builtin(
                "winshow",
                &[
                    Value::Datum(client),
                    Value::text("main.status"),
                    Value::number(0.0)
                ],
                &mut state,
            ),
            Ok(Value::Null)
        );
        assert_eq!(
            state
                .client_session(client)
                .expect("registered session should remain attached")
                .ui()
                .winget("main.status", "is-visible"),
            Ok("false".to_owned())
        );
        assert_eq!(
            execute_standard_builtin(
                "winexists",
                &[Value::Datum(client), Value::text("main.status")],
                &mut state,
            ),
            Ok(Value::text("LABEL"))
        );
    }

    #[test]
    fn local_client_creation_rejects_an_invalid_skin_before_allocating_a_session() {
        let mut state = ExecutionState::new();

        let error = state
            .open_local_client("window missing-quotes\n")
            .expect_err("malformed DMF should not create a local client");
        let unrelated_client = state
            .heap
            .allocate_datum(TypePath::parse("/client").unwrap());

        assert!(!error.diagnostics.is_empty());
        assert!(state.client_session_mut(unrelated_client).is_none());
    }

    #[test]
    fn step_moves_to_a_materialized_neighbor_and_reports_failure() {
        let mut state = ExecutionState::new();
        let origin = place(&mut state, "/turf/open", 2.0, 2.0);
        let east = place(&mut state, "/turf/open", 3.0, 2.0);
        let mob = place(&mut state, "/mob/living", 2.0, 2.0);
        let west_area = place(&mut state, "/area/west", 0.0, 0.0);
        let east_area = place(&mut state, "/area/east", 0.0, 0.0);
        let contents = FieldName::parse("contents").unwrap();
        for datum in [origin, east, west_area, east_area] {
            let list = state.heap.allocate_list();
            state
                .heap
                .set_datum_field(datum, contents.clone(), Value::List(list))
                .unwrap();
        }
        state
            .heap
            .list_mut(match state.heap.datum_field(origin, &contents).unwrap() {
                Value::List(list) => *list,
                _ => unreachable!(),
            })
            .unwrap()
            .add(Value::Datum(mob));
        for (datum, loc) in [(origin, west_area), (east, east_area), (mob, origin)] {
            state
                .heap
                .set_datum_field(datum, FieldName::parse("loc").unwrap(), Value::Datum(loc))
                .unwrap();
        }
        let west_contents = match state.heap.datum_field(west_area, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        let east_contents = match state.heap.datum_field(east_area, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        state
            .heap
            .list_mut(west_contents)
            .unwrap()
            .add(Value::Datum(mob));
        assert_eq!(
            step_builtin(&[Value::Datum(mob), Value::number(4.0)], &mut state).unwrap(),
            Value::number(1.0)
        );
        assert_eq!(
            state
                .heap
                .datum(mob)
                .unwrap()
                .field(&FieldName::parse("loc").unwrap()),
            Ok(&Value::Datum(east))
        );
        let origin_contents = match state.heap.datum_field(origin, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        let east_turf_contents = match state.heap.datum_field(east, &contents).unwrap() {
            Value::List(list) => *list,
            _ => unreachable!(),
        };
        assert!(
            !state
                .heap
                .list(origin_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            state
                .heap
                .list(east_turf_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            !state
                .heap
                .list(west_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            state
                .heap
                .list(east_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert_eq!(
            step_builtin(&[Value::Datum(mob), Value::number(4.0)], &mut state).unwrap(),
            Value::number(0.0)
        );
        del_builtin(&[Value::Datum(mob)], &mut state).unwrap();
        assert!(
            !state
                .heap
                .list(east_turf_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert!(
            !state
                .heap
                .list(east_contents)
                .unwrap()
                .contains(&Value::Datum(mob))
        );
        assert_ne!(origin, east);
    }
}
