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

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read as _, Write as _};
use std::path::Component;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use dm_value::{DatumId, FieldName, ListId, TypePath, Value};

use super::{
    CompoundAssignmentOperator, ExecutionState, NativeWalk, NativeWalkKind, compare_values,
};

pub(super) fn standard_builtin_arity(name: &str) -> Option<(usize, usize)> {
    Some(match name {
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
        // `link()` wraps a URL for BYOND's output operator. Headless output
        // retains the URL value itself, which is sufficient to preserve the
        // observable redirect payload without a client browser.
        "link" => Ok(arguments.first().cloned().unwrap_or(Value::Null)),
        "run" => Ok(Value::Null),
        "issaved" => Ok(Value::number(1.0)),
        "REGEX_QUOTE" => regex_quote(arguments, state, false),
        "REGEX_QUOTE_REPLACEMENT" => regex_quote(arguments, state, true),
        "browse" => headless_browse(arguments, state),
        "browse_rsc" => headless_transfer("browse_rsc", arguments, state),
        "ftp" => headless_transfer("ftp", arguments, state),
        "winset" => headless_winset(arguments, state),
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
            state.clear_iconforge();
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
    let pid = std::process::id().to_string();
    let query = format!("(Get-Process -Id {pid}).WorkingSet64");
    if let Ok(output) = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", &query])
        .output()
        && output.status.success()
        && let Ok(bytes) = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<u64>()
    {
        return Some(bytes);
    }

    // PowerShell can be removed on a deliberately minimal Windows host. Keep
    // the inbox task-list tool as a best-effort fallback; failure simply makes
    // the report say that the host aggregate is unavailable.
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout);
    let memory_kib = line
        .trim()
        .trim_end_matches('"')
        .rsplit_once("\",\"")?
        .1
        .chars()
        .filter(char::is_ascii_digit)
        .collect::<String>()
        .parse::<u64>()
        .ok()?;
    memory_kib.checked_mul(1_024)
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

fn rust_g_cellular_noise(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

fn rust_g_poisson_noise(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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

#[derive(Debug)]
struct DmiMetadata {
    width: u32,
    height: u32,
    states: Vec<DmiState>,
    error: Option<String>,
}

impl DmiMetadata {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "width": self.width,
            "height": self.height,
            "states": self.states.iter().map(DmiState::to_json).collect::<Vec<_>>(),
            "headless_error": self.error,
        })
    }
}

#[derive(Debug)]
struct DmiState {
    name: String,
    dirs: u32,
    frames: u32,
    delay: Vec<f64>,
    loop_value: i64,
    rewind: i64,
    movement: i64,
    hotspot: Option<Vec<i64>>,
}

impl DmiState {
    fn new(name: String) -> Self {
        Self {
            name,
            dirs: 1,
            frames: 1,
            delay: Vec::new(),
            loop_value: 0,
            rewind: 0,
            movement: 0,
            hotspot: None,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "dirs": self.dirs,
            "frames": self.frames,
            "delay": self.delay,
            "loop": self.loop_value,
            "rewind": self.rewind,
            "movement": self.movement,
            "hotspot": self.hotspot,
        })
    }
}

fn read_dmi_metadata(path: &std::path::Path) -> Result<DmiMetadata, String> {
    let png = fs::read(path).map_err(|error| error.to_string())?;
    if !png.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("resource is not a PNG-backed DMI".to_owned());
    }
    let mut cursor = 8usize;
    let mut image_width = None;
    let mut image_height = None;
    let mut description = None;
    while cursor.checked_add(12).is_some_and(|end| end <= png.len()) {
        let length = u32::from_be_bytes(
            png[cursor..cursor + 4]
                .try_into()
                .map_err(|_| "invalid PNG chunk length")?,
        ) as usize;
        let chunk_type = &png[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start
            .checked_add(length)
            .ok_or("PNG chunk length overflow")?;
        let chunk_end = data_end.checked_add(4).ok_or("PNG chunk CRC overflow")?;
        if chunk_end > png.len() {
            return Err("truncated PNG chunk".to_owned());
        }
        let data = &png[data_start..data_end];
        match chunk_type {
            b"IHDR" if data.len() >= 8 => {
                image_width = Some(u32::from_be_bytes(data[0..4].try_into().unwrap()));
                image_height = Some(u32::from_be_bytes(data[4..8].try_into().unwrap()));
            }
            b"tEXt" => {
                if let Some(separator) = data.iter().position(|byte| *byte == 0)
                    && &data[..separator] == b"Description"
                {
                    description = Some(
                        String::from_utf8(data[separator + 1..].to_vec())
                            .map_err(|error| error.to_string())?,
                    );
                }
            }
            b"zTXt" => {
                if let Some(separator) = data.iter().position(|byte| *byte == 0)
                    && &data[..separator] == b"Description"
                {
                    let method = *data
                        .get(separator + 1)
                        .ok_or("DMI zTXt chunk lacks compression method")?;
                    if method != 0 {
                        return Err(format!("unsupported DMI zTXt compression method {method}"));
                    }
                    let compressed = data
                        .get(separator + 2..)
                        .ok_or("DMI zTXt chunk lacks compressed data")?;
                    let mut decoder = flate2::read::ZlibDecoder::new(compressed);
                    let mut decoded = String::new();
                    decoder
                        .read_to_string(&mut decoded)
                        .map_err(|error| error.to_string())?;
                    description = Some(decoded);
                }
            }
            _ => {}
        }
        cursor = chunk_end;
        if description.is_some() && image_width.is_some() {
            break;
        }
    }
    let image_width = image_width.ok_or("PNG is missing IHDR width")?;
    let image_height = image_height.ok_or("PNG is missing IHDR height")?;
    match description {
        Some(description) => parse_dmi_description(&description, image_width, image_height),
        None => Ok(DmiMetadata {
            width: image_width,
            height: image_height,
            states: vec![DmiState::new(String::new())],
            error: None,
        }),
    }
}

fn icon_states_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let mut resource = match arguments.first().cloned().unwrap_or(Value::Null) {
        Value::Datum(datum) => super::datum_field_or_initial(
            state,
            datum,
            &FieldName::parse("icon").expect("icon field name is valid"),
        )
        .map_err(|error| error.to_string())?,
        value => value,
    };
    if let Value::Datum(_) = resource {
        resource = icon_backing_resource(&resource, state, 0)?;
    }
    let requested = match resource {
        Value::File(path) => path.to_string(),
        Value::Text(path) => path.to_string(),
        Value::Null => {
            return Err("icon_states resource requires text, received null".to_owned());
        }
        value => {
            return Err(format!(
                "icon_states resource requires text, received {value}"
            ));
        }
    };
    let resolved = relaxed_resolved_file_path(
        &[Value::text(requested.clone())],
        state,
        "icon_states resource",
    )?;
    let metadata = read_dmi_metadata(&resolved).map_err(|error| {
        format!(
            "icon_states failed for resource {requested:?} resolved to '{}': {error}",
            resolved.display()
        )
    })?;
    let list = state.heap_mut().allocate_list();
    let values = state
        .heap_mut()
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for icon_state in metadata.states {
        values.add(Value::text(icon_state.name));
    }
    Ok(Value::List(list))
}

fn parse_dmi_description(
    description: &str,
    image_width: u32,
    image_height: u32,
) -> Result<DmiMetadata, String> {
    let mut metadata = DmiMetadata {
        width: image_width,
        height: image_height,
        states: Vec::new(),
        error: None,
    };
    let mut state = None;
    for raw_line in description.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("invalid DMI metadata line {line:?}"));
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "version" => {}
            "width" => {
                metadata.width = value
                    .parse()
                    .map_err(|error| format!("invalid DMI width {value:?}: {error}"))?;
            }
            "height" => {
                metadata.height = value
                    .parse()
                    .map_err(|error| format!("invalid DMI height {value:?}: {error}"))?;
            }
            "state" => {
                if let Some(previous) = state.take() {
                    metadata.states.push(previous);
                }
                let name = serde_json::from_str::<String>(value)
                    .unwrap_or_else(|_| value.trim_matches('"').to_owned());
                state = Some(DmiState::new(name));
            }
            "dirs" => {
                if let Some(state) = state.as_mut() {
                    state.dirs = value
                        .parse()
                        .map_err(|error| format!("invalid DMI dirs {value:?}: {error}"))?;
                }
            }
            "frames" => {
                if let Some(state) = state.as_mut() {
                    state.frames = value
                        .parse()
                        .map_err(|error| format!("invalid DMI frames {value:?}: {error}"))?;
                }
            }
            "delay" => {
                if let Some(state) = state.as_mut() {
                    state.delay = value
                        .split(',')
                        .map(str::trim)
                        .map(str::parse)
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| format!("invalid DMI delay {value:?}: {error}"))?;
                }
            }
            "loop" | "rewind" | "movement" => {
                if let Some(state) = state.as_mut() {
                    let parsed = value
                        .parse()
                        .map_err(|error| format!("invalid DMI {key} {value:?}: {error}"))?;
                    match key {
                        "loop" => state.loop_value = parsed,
                        "rewind" => state.rewind = parsed,
                        _ => state.movement = parsed,
                    }
                }
            }
            "hotspot" => {
                if let Some(state) = state.as_mut() {
                    state.hotspot = Some(
                        value
                            .split(',')
                            .map(str::trim)
                            .map(str::parse)
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|error| format!("invalid DMI hotspot {value:?}: {error}"))?,
                    );
                }
            }
            _ => return Err(format!("unsupported DMI metadata key {key:?}")),
        }
    }
    if let Some(state) = state {
        metadata.states.push(state);
    }
    Ok(metadata)
}

fn format_unix_timestamp(unix_millis: i64, format: &str, offset_hours: f32) -> String {
    let offset_seconds = (offset_hours * 3_600.0).round() as i64;
    let local_millis = unix_millis.saturating_add(offset_seconds.saturating_mul(1_000));
    let days = local_millis.div_euclid(86_400_000);
    let day_millis = local_millis.rem_euclid(86_400_000);
    let (year, month, day) = civil_date_from_unix_days(days);
    let hour = day_millis / 3_600_000;
    let minute = day_millis / 60_000 % 60;
    let second = day_millis / 1_000 % 60;
    let millis = day_millis % 1_000;
    let offset_sign = if offset_seconds < 0 { '-' } else { '+' };
    let offset_abs = offset_seconds.abs();
    let offset = format!(
        "{offset_sign}{:02}{:02}",
        offset_abs / 3_600,
        offset_abs / 60 % 60
    );
    let literal_percent = "\u{0}";
    format
        .replace("%%", literal_percent)
        .replace("%.3f", &format!(".{millis:03}"))
        .replace("%F", &format!("{year:04}-{month:02}-{day:02}"))
        .replace("%T", &format!("{hour:02}:{minute:02}:{second:02}"))
        .replace("%Y", &format!("{year:04}"))
        .replace("%m", &format!("{month:02}"))
        .replace("%d", &format!("{day:02}"))
        .replace("%H", &format!("{hour:02}"))
        .replace("%M", &format!("{minute:02}"))
        .replace("%S", &format!("{second:02}"))
        .replace("%z", &offset)
        .replace(literal_percent, "%")
}

fn civil_date_from_unix_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

fn owned_value_text(value: Value) -> String {
    match value {
        Value::Text(text) => text.to_string(),
        Value::Null => String::new(),
        value => value.to_string(),
    }
}

fn iconforge_load_gags_config(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let config_path = strict_text(&arguments[0], state, "iconforge config path")?;
    let config_json = strict_text(&arguments[1], state, "iconforge config JSON")?;
    if let Err(error) = serde_json::from_str::<serde_json::Value>(&config_json) {
        return Ok(Value::text(format!(
            "IconForge error: Failed to parse config for '{config_path}': {error}"
        )));
    }
    let icon_path_text = strict_text(&arguments[2], state, "iconforge icon path")?;
    let icon_path = resolved_file_path(&arguments[2..3], state, "iconforge icon path")?;
    if let Err(error) = fs::metadata(&icon_path) {
        return Ok(Value::text(format!(
            "IconForge error: Failed to open DMI '{icon_path_text}' (resolved to '{}') - {error}",
            icon_path.display()
        )));
    }
    state.load_iconforge_gags_config(config_path, icon_path);
    Ok(Value::text("OK"))
}

fn iconforge_gags(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let config_path = strict_text(&arguments[0], state, "iconforge config path")?;
    if !state.has_iconforge_gags_config(&config_path) {
        return Ok(Value::text(format!(
            "IconForge error: Provided config_path {config_path} has not been loaded by iconforge_load_gags_config!"
        )));
    }
    let output_text = strict_text(&arguments[2], state, "iconforge output path")?;
    let relative = std::path::Path::new(&output_text);
    if relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err("iconforge output path escapes the project root".to_owned());
    }
    let root = state
        .project_root()
        .ok_or_else(|| "iconforge output path requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("iconforge project root is unavailable: {error}"))?;
    let mut parent = root.clone();
    for component in relative
        .parent()
        .into_iter()
        .flat_map(std::path::Path::components)
    {
        let Component::Normal(component) = component else {
            continue;
        };
        parent.push(component);
        if !parent.exists() {
            fs::create_dir(&parent).map_err(|error| {
                format!("IconForge error: Failed to create output directory: {error}")
            })?;
        }
        let resolved = parent.canonicalize().map_err(|error| {
            format!("IconForge error: Failed to resolve output directory: {error}")
        })?;
        if !resolved.starts_with(&root) {
            return Err("iconforge output path escapes the project root".to_owned());
        }
        parent = resolved;
    }
    let output = resolved_file_path(&arguments[2..3], state, "iconforge output path")?;
    let source = state
        .iconforge_gags_source(&config_path)
        .ok_or_else(|| format!("IconForge error: Config {config_path} lost its source DMI"))?;
    fs::copy(source, &output).map_err(|error| {
        format!(
            "IconForge error: Failed to create headless output '{}' from '{}': {error}",
            output.display(),
            source.display()
        )
    })?;
    Ok(Value::text("OK"))
}

fn validate_git_revision(revision: &str) -> Result<(), String> {
    if revision.is_empty()
        || revision.len() > 256
        || revision.starts_with('-')
        || revision.contains("..")
        || revision.contains("//")
        || revision.chars().any(|character| {
            !(character.is_ascii_alphanumeric()
                || matches!(character, '/' | '.' | '_' | '-' | '^' | '~'))
        })
    {
        return Err("git revision contains unsafe syntax".to_owned());
    }
    Ok(())
}

fn validate_git_date_format(format: &str) -> Result<(), String> {
    if format.is_empty()
        || format.len() > 128
        || format
            .chars()
            .any(|character| character.is_control() || matches!(character, '\0' | '\r' | '\n'))
    {
        return Err("git date format contains unsafe syntax".to_owned());
    }
    Ok(())
}

fn run_git_bridge(state: &ExecutionState, arguments: &[&str]) -> Result<Value, String> {
    let root = state
        .project_root()
        .ok_or_else(|| "git bridge requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("git bridge project root is unavailable: {error}"))?;
    let output = Command::new("git")
        .args(arguments)
        .current_dir(root)
        .output()
        .map_err(|error| format!("git bridge failed to start: {error}"))?;
    if !output.status.success() {
        return Ok(Value::Null);
    }
    let text = String::from_utf8(output.stdout)
        .map_err(|error| format!("git bridge returned non-UTF-8 output: {error}"))?;
    Ok(Value::text(text.trim_end_matches(['\r', '\n']).to_owned()))
}

fn parse_toml_document(source: &str) -> Result<serde_json::Value, String> {
    let mut root = serde_json::Map::new();
    let mut context = Vec::<String>::new();
    for (line_index, raw) in source.lines().enumerate() {
        let line = strip_toml_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("[[") && line.ends_with("]]") {
            context = parse_toml_key_path(&line[2..line.len() - 2])?;
            let (parent, leaf) = context
                .split_last()
                .ok_or_else(|| "empty array-table name".to_owned())?;
            let object = toml_object_at(&mut root, leaf)?;
            let array = object
                .entry(parent.clone())
                .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                .as_array_mut()
                .ok_or_else(|| {
                    format!(
                        "line {}: array table conflicts with a value",
                        line_index + 1
                    )
                })?;
            array.push(serde_json::Value::Object(serde_json::Map::new()));
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            context = parse_toml_key_path(&line[1..line.len() - 1])?;
            toml_object_at(&mut root, &context)?;
            continue;
        }
        let (key, value) = split_toml_assignment(line)
            .ok_or_else(|| format!("line {}: expected key = value", line_index + 1))?;
        let mut path = context.clone();
        path.extend(parse_toml_key_path(key)?);
        let (leaf, parents) = path
            .split_last()
            .ok_or_else(|| format!("line {}: empty key", line_index + 1))?;
        let object = toml_object_at(&mut root, parents)?;
        if object
            .insert(leaf.clone(), parse_toml_value(value)?)
            .is_some()
        {
            return Err(format!("line {}: duplicate key {leaf}", line_index + 1));
        }
    }
    Ok(serde_json::Value::Object(root))
}

fn toml_object_at<'a>(
    root: &'a mut serde_json::Map<String, serde_json::Value>,
    path: &[String],
) -> Result<&'a mut serde_json::Map<String, serde_json::Value>, String> {
    let mut object = root;
    for segment in path {
        let mut value = object
            .entry(segment.clone())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        if let serde_json::Value::Array(array) = value {
            value = array
                .last_mut()
                .ok_or_else(|| format!("array table {segment} has no current entry"))?;
        }
        object = value
            .as_object_mut()
            .ok_or_else(|| format!("table {segment} conflicts with a value"))?;
    }
    Ok(object)
}

fn strip_toml_comment(line: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
        } else if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if character == '#' && quote.is_none() {
            return &line[..index];
        }
    }
    line
}

fn split_toml_assignment(line: &str) -> Option<(&str, &str)> {
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, character) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
        } else if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if quote.is_none() {
            match character {
                '[' | '{' => depth += 1,
                ']' | '}' => depth = depth.saturating_sub(1),
                '=' if depth == 0 => return Some((line[..index].trim(), line[index + 1..].trim())),
                _ => {}
            }
        }
    }
    None
}

fn parse_toml_key_path(source: &str) -> Result<Vec<String>, String> {
    split_toml_items(source, '.')
        .into_iter()
        .map(|part| parse_toml_key(part.trim()))
        .collect()
}

fn parse_toml_key(source: &str) -> Result<String, String> {
    if source.starts_with('"') {
        serde_json::from_str(source).map_err(|error| format!("invalid quoted key: {error}"))
    } else if source.starts_with('\'') && source.ends_with('\'') && source.len() >= 2 {
        Ok(source[1..source.len() - 1].to_owned())
    } else if !source.is_empty()
        && source
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    {
        Ok(source.to_owned())
    } else {
        Err(format!("invalid TOML key {source:?}"))
    }
}

fn parse_toml_value(source: &str) -> Result<serde_json::Value, String> {
    let source = source.trim();
    if source.starts_with('"') {
        return serde_json::from_str::<String>(source)
            .map(serde_json::Value::String)
            .map_err(|error| format!("invalid string: {error}"));
    }
    if source.starts_with('\'') && source.ends_with('\'') && source.len() >= 2 {
        return Ok(serde_json::Value::String(
            source[1..source.len() - 1].to_owned(),
        ));
    }
    if source.starts_with('[') && source.ends_with(']') {
        let inner = &source[1..source.len() - 1];
        return split_toml_items(inner, ',')
            .into_iter()
            .filter(|item| !item.trim().is_empty())
            .map(parse_toml_value)
            .collect::<Result<Vec<_>, _>>()
            .map(serde_json::Value::Array);
    }
    match source {
        "true" => return Ok(serde_json::Value::Bool(true)),
        "false" => return Ok(serde_json::Value::Bool(false)),
        _ => {}
    }
    let number = source.replace('_', "");
    if let Ok(integer) = number.parse::<i64>() {
        return Ok(serde_json::Value::Number(integer.into()));
    }
    if let Ok(float) = number.parse::<f64>() {
        return serde_json::Number::from_f64(float)
            .map(serde_json::Value::Number)
            .ok_or_else(|| format!("non-finite TOML number {source}"));
    }
    Err(format!("unsupported TOML value {source:?}"))
}

fn split_toml_items(source: &str, delimiter: char) -> Vec<&str> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut quote = None;
    let mut escaped = false;
    let mut depth = 0usize;
    for (index, character) in source.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if quote == Some('"') && character == '\\' {
            escaped = true;
            continue;
        }
        if matches!(character, '"' | '\'') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if quote.is_none() {
            match character {
                '[' | '{' => depth += 1,
                ']' | '}' => depth = depth.saturating_sub(1),
                _ if character == delimiter && depth == 0 => {
                    items.push(&source[start..index]);
                    start = index + character.len_utf8();
                }
                _ => {}
            }
        }
    }
    items.push(&source[start..]);
    items
}

fn world_profile(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let format = arguments
        .get(2)
        .and_then(value_text)
        .or_else(|| arguments.get(1).and_then(value_text));
    if format == Some("json") {
        return Ok(Value::text("[]"));
    }
    let columns: &[&str] = if arguments.get(1).and_then(value_text) == Some("sendmaps") {
        &["name", "value", "calls"]
    } else {
        &["name", "self", "total", "real", "over", "calls"]
    };
    let list = state.heap_mut().allocate_list();
    for column in columns {
        state
            .heap_mut()
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::text(*column));
    }
    Ok(Value::List(list))
}

fn world_get_config(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let config_set = strict_text(&arguments[0], state, "world.GetConfig config set")?;
    let config_set = config_set.rsplit('/').next().unwrap_or(&config_set);
    match config_set {
        "env" => {
            let Some(name) = arguments.get(1).and_then(value_text) else {
                return Ok(Value::Null);
            };
            Ok(match state.environment_override(name) {
                Some(Some(value)) => value.clone(),
                Some(None) => Value::Null,
                None => std::env::var(name).map_or(Value::Null, Value::text),
            })
        }
        "ban" | "keyban" | "ipban" | "admin" => Ok(Value::List(state.heap_mut().allocate_list())),
        _ => Err(format!("unknown world configuration set {config_set:?}")),
    }
}

fn world_set_config(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let config_set = strict_text(&arguments[0], state, "world.SetConfig config set")?;
    let config_set = config_set.rsplit('/').next().unwrap_or(&config_set);
    match config_set {
        "env" => {
            let name = strict_text(&arguments[1], state, "world.SetConfig parameter")?;
            let value = value_text(&arguments[2]).map(Value::text);
            state.set_environment_override(name, value);
        }
        "ban" | "keyban" | "ipban" | "admin" => {}
        _ => return Err(format!("unknown world configuration set {config_set:?}")),
    }
    Ok(Value::Null)
}

fn value_text(value: &Value) -> Option<&str> {
    match value {
        Value::Text(text) => Some(text),
        _ => None,
    }
}

fn world_open_port(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Value::Datum(world) = arguments[0] else {
        return Err("world.OpenPort requires a world datum receiver".to_owned());
    };
    let port = arguments[1]
        .as_number()
        .ok_or_else(|| "world.OpenPort requires a numeric port".to_owned())?;
    state
        .heap_mut()
        .set_datum_field(
            world,
            FieldName::parse("port").expect("built-in world field is valid"),
            Value::number(port),
        )
        .map_err(|error| error.to_string())?;
    Ok(Value::number(1.0))
}

fn resource_datum_builtin(
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
        Value::Datum(datum) => super::vector_components(*datum, state.heap()).unwrap_or([0.0; 3]),
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

fn generator_rand_builtin(
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
            super::allocate_vector(values, state.heap_mut()).map(Value::Datum)
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
            super::allocate_vector(values, state.heap_mut()).map(Value::Datum)
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
            super::allocate_vector(values, state.heap_mut()).map(Value::Datum)
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

fn icon_swap_color_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Value::Datum(icon) = arguments[0] else {
        return Err("icon.SwapColor requires an /icon receiver".to_owned());
    };
    if !super::is_icon_datum(icon, state.heap()) {
        return Err("icon.SwapColor requires an /icon receiver".to_owned());
    }
    super::execute_icon_method(icon, "SwapColor", &arguments[1..], state.heap_mut())
}

/// Constructs BYOND's mutable `/icon` value.
///
/// An existing `/icon` is a copy-constructor input, not the backing resource
/// stored in the new icon's `icon` field. OpenDream's `DreamObjectIcon`
/// mirrors BYOND by copying its complete `DreamIcon` here. This is observable
/// in tg-derived `getFlatIcon()`, which starts every render with
/// `flat_template = icon(file); flat = icon(flat_template)` and then mutates
/// `flat` independently.
fn icon_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if let Some(Value::Datum(source)) = arguments.first()
        && super::is_icon_datum(*source, &state.heap)
    {
        return super::clone_icon_datum(*source, &mut state.heap).map(Value::Datum);
    }

    let icon = resource_datum_builtin(
        "/icon",
        &["icon", "icon_state", "dir", "frame", "moving"],
        arguments,
        state,
    )?;

    // A BYOND /icon owns the dimensions of the selected DMI frame, not the
    // engine's 32x32 fallback. Large canvas DMIs (Monkestation's holomap is
    // 480x480) immediately observe this through Width()/Height(). Keep the
    // constructor permissive for synthetic/missing headless resources, but
    // seed exact metadata whenever the backing resource is available.
    if let (Value::Datum(icon), Some(Value::File(_) | Value::Text(_))) = (&icon, arguments.first())
        && let Ok(resolved) =
            relaxed_resolved_file_path(&arguments[..1], state, "icon constructor resource")
        && let Ok(metadata) = read_dmi_metadata(&resolved)
    {
        state
            .heap_mut()
            .set_datum_field(
                *icon,
                FieldName::parse("_dream64_width").expect("internal icon width is valid"),
                Value::number(metadata.width as f32),
            )
            .map_err(|error| error.to_string())?;
        state
            .heap_mut()
            .set_datum_field(
                *icon,
                FieldName::parse("_dream64_height").expect("internal icon height is valid"),
                Value::number(metadata.height as f32),
            )
            .map_err(|error| error.to_string())?;
    }

    Ok(icon)
}

fn fcopy_rsc(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(value) = arguments.first() else {
        return Ok(Value::Null);
    };
    match value {
        Value::File(path) | Value::Text(path) => Ok(Value::file(path.clone())),
        Value::Null => Ok(Value::Null),
        Value::Datum(_) => icon_backing_resource(value, state, 0),
        value => Err(format!(
            "fcopy_rsc requires a file, path, or icon, received {value}"
        )),
    }
}

/// OpenDream, matching BYOND, keeps `/icon` objects distinct from resources:
/// `isfile(icon)` is false, while `fcopy_rsc(icon)` materializes an icon
/// resource. Dream64's headless renderer retains the constructor's backing
/// resource instead of rasterizing pixels, so unwrap that backing resource
/// (including icons cloned from other icons) into the first-class `File`
/// value used by filesystem/resource builtins.
fn icon_backing_resource(
    value: &Value,
    state: &ExecutionState,
    depth: usize,
) -> Result<Value, String> {
    if depth >= 64 {
        return Err("fcopy_rsc encountered a cyclic icon resource".to_owned());
    }
    let Value::Datum(icon) = value else {
        return Err(format!("fcopy_rsc requires an icon, received {value}"));
    };
    let datum = state.heap.datum(*icon).map_err(|error| error.to_string())?;
    let path = datum.type_path().as_str();
    if path != "/icon" && !path.starts_with("/icon/") {
        return Err(format!("fcopy_rsc requires an icon, received {value}"));
    }
    let field = FieldName::parse("icon").expect("built-in icon field is valid");
    match datum.field(&field) {
        Ok(Value::File(path)) | Ok(Value::Text(path)) => Ok(Value::file(path.clone())),
        Ok(Value::Datum(backing)) => {
            icon_backing_resource(&Value::Datum(*backing), state, depth + 1)
        }
        Ok(Value::Null) | Err(_) => Ok(Value::Null),
        Ok(value) => Err(format!(
            "fcopy_rsc icon has an unsupported backing resource {value}"
        )),
    }
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

fn values_cut(
    arguments: &[Value],
    state: &mut ExecutionState,
    over: bool,
) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let threshold = number(&arguments[1], "values_cut threshold")?;
    let inclusive = arguments.get(2).is_some_and(truthy);
    let snapshot = list_operator_snapshot(list, state)?;
    let mut removed = 0_usize;
    for entry in snapshot {
        let remove = entry
            .associated
            .as_ref()
            .and_then(Value::as_number)
            .map_or(true, |value| {
                if over {
                    value > threshold || (inclusive && value == threshold)
                } else {
                    value < threshold || (inclusive && value == threshold)
                }
            });
        if remove
            && state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_key(&entry.key)
                .or_else(|| state.heap.list_mut(list).ok()?.remove_last(&entry.key))
                .is_some()
        {
            removed += 1;
        }
    }
    Ok(Value::number(removed as f32))
}

fn values_dot(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let (Value::List(left), Value::List(right)) = (&arguments[0], &arguments[1]) else {
        return Ok(Value::number(0.0));
    };
    let left = state.heap.list(*left).map_err(|error| error.to_string())?;
    let right = state.heap.list(*right).map_err(|error| error.to_string())?;
    let total = left.positions().fold(0.0, |total, (_, key)| {
        let Some(left_value) = left.get_key(key).ok().and_then(Value::as_number) else {
            return total;
        };
        let Some(right_value) = right.get_key(key).ok().and_then(Value::as_number) else {
            return total;
        };
        total + left_value * right_value
    });
    Ok(Value::number(total))
}

fn values_fold(
    arguments: &[Value],
    state: &ExecutionState,
    product: bool,
) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    let mut values = list
        .positions()
        .filter_map(|(_, key)| list.get_key(key).ok().and_then(Value::as_number));
    let result = if product {
        values
            .next()
            .map_or(0.0, |first| values.fold(first, |a, b| a * b))
    } else {
        values.sum()
    };
    Ok(Value::number(result))
}

/// Implements Dream Maker's legacy `text()` template form. Empty bracket
/// expressions in the literal template consume the following arguments in
/// order; whitespace inside a hole is ignored. Escaped brackets remain text.
fn text_template(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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
            super::TEXT_MACRO_THE
                | super::TEXT_MACRO_THE_UPPER
                | super::TEXT_MACRO_A
                | super::TEXT_MACRO_A_UPPER
                | super::TEXT_MACRO_PROPER
                | super::TEXT_MACRO_IMPROPER
                | super::TEXT_MACRO_ROMAN
                | super::TEXT_MACRO_ROMAN_UPPER
        ) {
            pending_prefix = Some(character);
            continue;
        }
        if matches!(
            character,
            super::TEXT_MACRO_ORDINAL
                | super::TEXT_MACRO_PLURAL
                | super::TEXT_MACRO_SUBJECT
                | super::TEXT_MACRO_SUBJECT_UPPER
                | super::TEXT_MACRO_POSSESSIVE_ADJECTIVE
                | super::TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER
                | super::TEXT_MACRO_OBJECT
                | super::TEXT_MACRO_REFLEXIVE
                | super::TEXT_MACRO_POSSESSIVE
                | super::TEXT_MACRO_POSSESSIVE_UPPER
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
        for _ in 0..whitespace.chars().count() + 1 {
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
        super::TEXT_MACRO_THE
            | super::TEXT_MACRO_THE_UPPER
            | super::TEXT_MACRO_A
            | super::TEXT_MACRO_A_UPPER
            | super::TEXT_MACRO_PROPER
            | super::TEXT_MACRO_IMPROPER
            | super::TEXT_MACRO_ROMAN
            | super::TEXT_MACRO_ROMAN_UPPER
            | super::TEXT_MACRO_ORDINAL
            | super::TEXT_MACRO_PLURAL
            | super::TEXT_MACRO_SUBJECT
            | super::TEXT_MACRO_SUBJECT_UPPER
            | super::TEXT_MACRO_POSSESSIVE_ADJECTIVE
            | super::TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER
            | super::TEXT_MACRO_OBJECT
            | super::TEXT_MACRO_REFLEXIVE
            | super::TEXT_MACRO_POSSESSIVE
            | super::TEXT_MACRO_POSSESSIVE_UPPER
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
    if raw.starts_with(super::TEXT_MACRO_PROPER) {
        return Ok(true);
    }
    if raw.starts_with(super::TEXT_MACRO_IMPROPER) {
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
        super::TEXT_MACRO_THE | super::TEXT_MACRO_THE_UPPER => {
            if text_macro_is_proper(value, state)? {
                Ok(visible)
            } else {
                let article = if prefix == super::TEXT_MACRO_THE_UPPER {
                    "The "
                } else {
                    "the "
                };
                Ok(format!("{article}{visible}"))
            }
        }
        super::TEXT_MACRO_A | super::TEXT_MACRO_A_UPPER => {
            if text_macro_is_proper(value, state)? {
                return Ok(visible);
            }
            let plural = value_gender(value, state).as_deref() == Some("plural");
            let vowel = visible.chars().next().is_some_and(|character| {
                matches!(character.to_ascii_lowercase(), 'a' | 'e' | 'i' | 'o' | 'u')
            });
            let article = match (prefix == super::TEXT_MACRO_A_UPPER, plural, vowel) {
                (true, true, _) => "Some ",
                (false, true, _) => "some ",
                (true, false, true) => "An ",
                (false, false, true) => "an ",
                (true, false, false) => "A ",
                (false, false, false) => "a ",
            };
            Ok(format!("{article}{visible}"))
        }
        super::TEXT_MACRO_ROMAN | super::TEXT_MACRO_ROMAN_UPPER => {
            Ok(value.as_number().map_or_else(String::new, |number| {
                roman_text(number, prefix == super::TEXT_MACRO_ROMAN_UPPER)
            }))
        }
        // `\\proper` and `\\improper` are metadata markers when stored in a
        // literal name. During runtime text() formatting BYOND consumes them.
        super::TEXT_MACRO_PROPER | super::TEXT_MACRO_IMPROPER => Ok(visible),
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
        super::TEXT_MACRO_ORDINAL => {
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
        super::TEXT_MACRO_PLURAL => {
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
                super::TEXT_MACRO_SUBJECT => 0,
                super::TEXT_MACRO_SUBJECT_UPPER => 1,
                super::TEXT_MACRO_POSSESSIVE_ADJECTIVE => 2,
                super::TEXT_MACRO_POSSESSIVE_ADJECTIVE_UPPER => 3,
                super::TEXT_MACRO_OBJECT => 4,
                super::TEXT_MACRO_REFLEXIVE => 5,
                super::TEXT_MACRO_POSSESSIVE => 6,
                super::TEXT_MACRO_POSSESSIVE_UPPER => 7,
                _ => return Ok(()),
            };
            output.push_str(words[family][index]);
        }
    }
    Ok(())
}

fn md5_builtin(arguments: &[Value]) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Ok(Value::Null);
    };
    Ok(Value::text(format!("{:x}", md5::compute(text.as_bytes()))))
}

fn rust_g_hash_string(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let algorithm = strict_text(&arguments[0], state, "hash_string algorithm")?;
    let text = strict_text(&arguments[1], state, "hash_string text")?;
    match algorithm.to_ascii_lowercase().as_str() {
        "md5" => Ok(Value::text(format!("{:x}", md5::compute(text.as_bytes())))),
        algorithm => Err(format!(
            "hash_string algorithm {algorithm:?} is unavailable in the Dream64 host"
        )),
    }
}

fn rust_g_hash_file(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let algorithm = strict_text(&arguments[0], state, "hash_file algorithm")?;
    if !algorithm.eq_ignore_ascii_case("md5") {
        return Err(format!(
            "hash_file algorithm {algorithm:?} is unavailable in the Dream64 host"
        ));
    }
    let path = relaxed_resolved_file_path(&arguments[1..], state, "hash_file path")?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("hash_file failed to read '{}': {error}", path.display()))?;
    Ok(Value::text(format!("{:x}", md5::compute(bytes))))
}

fn rust_g_url_encode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "url_encode input")?;
    let mut encoded = String::with_capacity(text.len());
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in text.bytes() {
        match byte {
            b' ' => encoded.push('+'),
            b'*' | b'-' | b'.' | b'0'..=b'9' | b'A'..=b'Z' | b'_' | b'a'..=b'z' => {
                encoded.push(char::from(byte));
            }
            _ => {
                encoded.push('%');
                encoded.push(char::from(HEX[usize::from(byte >> 4)]));
                encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
            }
        }
    }
    Ok(Value::text(encoded))
}

fn rust_g_url_decode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "url_decode input")?;
    let source = text.as_bytes();
    let mut decoded = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        if source[index] == b'+' {
            decoded.push(b' ');
            index += 1;
            continue;
        }
        if source[index] == b'%'
            && index + 2 < source.len()
            && let (Some(high), Some(low)) =
                (hex_nibble(source[index + 1]), hex_nibble(source[index + 2]))
        {
            decoded.push((high << 4) | low);
            index += 3;
            continue;
        }
        decoded.push(source[index]);
        index += 1;
    }
    Ok(Value::text(String::from_utf8_lossy(&decoded).into_owned()))
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn json_encode_builtin(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let pretty = arguments
        .get(1)
        .and_then(Value::as_number)
        .is_some_and(|flags| flags.trunc() as i32 & 1 != 0);
    let value = arguments.first().unwrap_or(&Value::Null);
    let json = json_value_from_dm(value, state, 0)?;
    let encoded = if pretty {
        serde_json::to_string_pretty(&json)
    } else {
        serde_json::to_string(&json)
    }
    .map_err(|error| format!("json_encode failed: {error}"))?;
    Ok(Value::text(encoded))
}

fn json_value_from_dm(
    value: &Value,
    state: &ExecutionState,
    depth: usize,
) -> Result<serde_json::Value, String> {
    if depth >= 20 {
        return Ok(serde_json::Value::Null);
    }
    match value {
        Value::Null => Ok(serde_json::Value::Null),
        Value::Number(number) => {
            let number = number.to_f32();
            if number.is_finite() {
                let json_number =
                    number
                        .to_string()
                        .parse::<serde_json::Number>()
                        .map_err(|error| {
                            format!("json_encode cannot encode number {number}: {error}")
                        })?;
                Ok(serde_json::Value::Number(json_number))
            } else {
                let spelling = if number.is_nan() {
                    "NaN"
                } else if number.is_sign_positive() {
                    "Infinity"
                } else {
                    "-Infinity"
                };
                let mut object = serde_json::Map::new();
                object.insert(
                    "__number__".to_owned(),
                    serde_json::Value::String(spelling.to_owned()),
                );
                Ok(serde_json::Value::Object(object))
            }
        }
        Value::Text(text) | Value::File(text) => Ok(serde_json::Value::String(text.to_string())),
        Value::TypePath(path) => Ok(serde_json::Value::String(path.to_string())),
        Value::ModifiedTypePath(path) => Ok(serde_json::Value::String(path.base().to_string())),
        Value::Datum(_) => Ok(serde_json::Value::String(runtime_text(
            value,
            state,
            "json_encode datum",
        )?)),
        Value::List(id) => {
            let list = state.heap.list(*id).map_err(|error| error.to_string())?;
            let entries = list
                .positions()
                .map(|(_, key)| Ok((key.clone(), list.get_key(key).ok().cloned())))
                .collect::<Result<Vec<_>, String>>()?;
            if list.associative_len() == 0 {
                entries
                    .into_iter()
                    .map(|(value, _)| json_value_from_dm(&value, state, depth + 1))
                    .collect::<Result<Vec<_>, _>>()
                    .map(serde_json::Value::Array)
            } else {
                let mut object = serde_json::Map::new();
                for (key, associated) in entries {
                    let key = runtime_text(&key, state, "json_encode list key")?;
                    let value = associated.map_or(Ok(serde_json::Value::Null), |value| {
                        json_value_from_dm(&value, state, depth + 1)
                    })?;
                    object.insert(key, value);
                }
                Ok(serde_json::Value::Object(object))
            }
        }
    }
}

fn json_decode_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Some(Value::Text(text)) = arguments.first() else {
        return Err("json_decode requires text".to_owned());
    };
    let json: serde_json::Value = serde_json::from_str(text).map_err(|error| {
        let preview = text.chars().take(256).collect::<String>();
        format!("json_decode failed for {preview:?}: {error}")
    })?;
    dm_value_from_json(&json, state)
}

fn dm_value_from_json(
    json: &serde_json::Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match json {
        serde_json::Value::Null => Ok(Value::Null),
        serde_json::Value::Bool(value) => Ok(Value::number(f32::from(*value))),
        serde_json::Value::Number(value) => {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("json_decode invalid number {value}"))?
                as f32;
            if !number.is_finite() {
                return Err(format!("json_decode number is outside DM's range: {value}"));
            }
            Ok(Value::number(number))
        }
        serde_json::Value::String(value) => Ok(Value::text(value.clone())),
        serde_json::Value::Array(values) => {
            let decoded = values
                .iter()
                .map(|value| dm_value_from_json(value, state))
                .collect::<Result<Vec<_>, _>>()?;
            let id = state.heap.allocate_list();
            let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
            for value in decoded {
                list.add(value);
            }
            Ok(Value::List(id))
        }
        serde_json::Value::Object(object) => {
            if object.len() == 1 {
                if let Some(serde_json::Value::String(number)) = object.get("__number__") {
                    let value = match number.as_str() {
                        "NaN" => f32::NAN,
                        "Infinity" => f32::INFINITY,
                        "-Infinity" => f32::NEG_INFINITY,
                        _ => number.parse::<f32>().map_err(|_| {
                            format!("json_decode invalid special number {number:?}")
                        })?,
                    };
                    return Ok(Value::number(value));
                }
            }
            let decoded = object
                .iter()
                .map(|(key, value)| {
                    dm_value_from_json(value, state).map(|value| (Value::text(key.clone()), value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let id = state.heap.allocate_list();
            let list = state.heap.list_mut(id).map_err(|error| error.to_string())?;
            for (key, value) in decoded {
                list.set_key(key, value);
            }
            Ok(Value::List(id))
        }
    }
}

fn lentext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "lentext")?;
    Ok(Value::number(text.len() as f32))
}

fn sorttext(arguments: &[Value], state: &ExecutionState, exact: bool) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Ok(Value::number(0.0));
    }
    let values = arguments
        .iter()
        // BYOND sorttext is a comparator over each value's text
        // representation. It accepts null (""), numbers, type paths, and
        // datums; tg/Monk relies on this while sorting associative type
        // catalogs whose optional display key can be null.
        .map(|value| runtime_text(value, state, "sorttext"))
        .collect::<Result<Vec<_>, _>>()?;
    let compare = |left: &str, right: &str| {
        if exact {
            left.cmp(right)
        } else {
            left.to_lowercase().cmp(&right.to_lowercase())
        }
    };
    let ascending = values
        .windows(2)
        .all(|pair| compare(&pair[0], &pair[1]).is_lt());
    let descending = values
        .windows(2)
        .all(|pair| compare(&pair[0], &pair[1]).is_gt());
    Ok(Value::number(if ascending {
        1.0
    } else if descending {
        -1.0
    } else {
        0.0
    }))
}

fn num2text(arguments: &[Value]) -> Result<Value, String> {
    let value = number(&arguments[0], "num2text")?;
    if arguments.len() == 3 {
        let digits = number(&arguments[1], "num2text digits")?.trunc().max(0.0) as usize;
        let radix = number(&arguments[2], "num2text radix")?.trunc() as u32;
        if !(2..=36).contains(&radix) {
            return Err(format!("num2text radix {radix} is outside 2..=36"));
        }
        let negative = value.is_sign_negative();
        let mut integer = value.abs().trunc() as u32;
        let alphabet = b"0123456789abcdefghijklmnopqrstuvwxyz";
        let mut encoded = Vec::new();
        loop {
            encoded.push(alphabet[(integer % radix) as usize] as char);
            integer /= radix;
            if integer == 0 {
                break;
            }
        }
        while encoded.len() < digits {
            encoded.push('0');
        }
        if negative {
            encoded.push('-');
        }
        encoded.reverse();
        return Ok(Value::text(encoded.into_iter().collect::<String>()));
    }
    let sigfig = arguments.get(1).map_or(Ok(6_usize), |value| {
        number(value, "num2text sigfig").map(|value| value.trunc().max(1.0) as usize)
    })?;
    let plain = value.to_string();
    let significant_digits = plain.chars().filter(char::is_ascii_digit).count();
    if significant_digits <= sigfig || value == 0.0 {
        return Ok(Value::text(plain));
    }
    Ok(Value::text(format!(
        "{:.*e}",
        sigfig.saturating_sub(1),
        value
    )))
}

fn form_encode(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                output.push(char::from(byte));
            }
            b' ' => output.push('+'),
            _ => write!(&mut output, "%{byte:02X}").expect("writing to a String cannot fail"),
        }
    }
    output
}

fn form_decode(value: &str) -> Result<String, String> {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[index + 1..index + 3])
                    .map_err(|error| error.to_string())?;
                let byte = u8::from_str_radix(hex, 16)
                    .map_err(|_| format!("invalid parameter escape %{hex}"))?;
                output.push(byte);
                index += 3;
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8(output).map_err(|error| format!("parameter text is not UTF-8: {error}"))
}

fn list2params(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list_id) = arguments[0] else {
        return Err(format!(
            "list2params requires a list, received {}",
            arguments[0]
        ));
    };
    let list = state
        .heap
        .list(list_id)
        .map_err(|error| error.to_string())?;
    let mut pairs = Vec::with_capacity(list.len());
    for (_, key) in list.positions() {
        let key_text = runtime_text(key, state, "list2params key")?;
        let encoded_key = form_encode(&key_text);
        if let Ok(associated) = list.get_key(key) {
            let value_text = runtime_text(associated, state, "list2params value")?;
            pairs.push(format!("{encoded_key}={}", form_encode(&value_text)));
        } else {
            pairs.push(encoded_key);
        }
    }
    Ok(Value::text(pairs.join("&")))
}

fn params2list(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let params = strict_text(&arguments[0], state, "params2list")?;
    let result = state.heap.allocate_list();
    for part in params.split(['&', ';']) {
        if part.is_empty() {
            continue;
        }
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        let key = Value::text(form_decode(key)?);
        let value = Value::text(form_decode(value)?);
        state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?
            .set_key(key, value);
    }
    Ok(Value::List(result))
}

fn unary_number(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
    let value = number(&arguments[0], "numeric builtin")?;
    Ok(Value::number(operation(value)))
}

fn extrema_builtin(
    arguments: &[Value],
    state: &ExecutionState,
    maximum: bool,
) -> Result<Value, String> {
    let values = if let [Value::List(list)] = arguments {
        state
            .heap()
            .list(*list)
            .map_err(|error| error.to_string())?
            .positions()
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>()
    } else {
        arguments.to_vec()
    };
    let Some(mut result) = values.first().cloned() else {
        return Ok(Value::Null);
    };
    for value in values.iter().skip(1) {
        let ordering = compare_values(value, &result)?;
        if ordering.is_some_and(|ordering| {
            if maximum {
                ordering.is_gt()
            } else {
                ordering.is_lt()
            }
        }) {
            result.clone_from(value);
        }
    }
    Ok(result)
}

fn fallback_number(value: &Value) -> f32 {
    match value {
        Value::Number(number) => number.to_f32(),
        Value::Null
        | Value::Text(_)
        | Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_)
        | Value::Datum(_)
        | Value::List(_) => 0.0,
    }
}

fn inverse_trig(arguments: &[Value], operation: impl FnOnce(f32) -> f32) -> Result<Value, String> {
    let value = fallback_number(&arguments[0]);
    let value = if (-1.0..=1.0).contains(&value) {
        operation(value).to_degrees()
    } else {
        0.0
    };
    Ok(Value::number(value))
}

fn arctan_builtin(arguments: &[Value]) -> Result<Value, String> {
    let first = fallback_number(&arguments[0]);
    let value = if arguments.len() == 1 {
        first.atan().to_degrees()
    } else {
        let second = fallback_number(&arguments[1]);
        second.atan2(first).to_degrees()
    };
    Ok(Value::number(value))
}

fn number(value: &Value, context: &str) -> Result<f32, String> {
    match value {
        Value::Null => Ok(0.0),
        Value::Number(number) => Ok(number.to_f32()),
        _ => Err(format!("{context} requires a number, received {value}")),
    }
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Number(number) => number.to_f32() != 0.0,
        Value::Text(text) => !text.is_empty(),
        Value::File(_)
        | Value::TypePath(_)
        | Value::ModifiedTypePath(_)
        | Value::Datum(_)
        | Value::List(_) => true,
    }
}

fn log_builtin(arguments: &[Value]) -> Result<Value, String> {
    let value = if arguments.len() == 1 {
        number(&arguments[0], "log")?.ln()
    } else {
        let base = number(&arguments[0], "log base")?;
        let value = number(&arguments[1], "log value")?;
        value.log(base)
    };
    Ok(Value::number(value))
}

fn lerp_builtin(arguments: &[Value]) -> Result<Value, String> {
    let start = number(&arguments[0], "lerp start")?;
    let end = number(&arguments[1], "lerp end")?;
    let factor = number(&arguments[2], "lerp factor")?;
    Ok(Value::number(start + (end - start) * factor))
}

/// Implements BYOND's scalar and list `clamp(value, low, high)` forms.
/// Bounds are interchangeable. List input produces a new positional list and
/// skips nonnumeric entries, matching Dream Maker's observable behavior.
fn clamp_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let mut low = number(&arguments[1], "clamp lower bound")?;
    let mut high = number(&arguments[2], "clamp upper bound")?;
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    if let Value::List(list) = arguments[0] {
        let clamped = state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .positions()
            .filter_map(|(_, value)| match value {
                Value::Number(number) => Some(Value::number(number.to_f32().clamp(low, high))),
                _ => None,
            })
            .collect::<Vec<_>>();
        let result = state.heap.allocate_list();
        let list = state
            .heap
            .list_mut(result)
            .map_err(|error| error.to_string())?;
        for value in clamped {
            list.add(value);
        }
        Ok(Value::List(result))
    } else {
        let value = number(&arguments[0], "clamp value")?;
        Ok(Value::number(value.clamp(low, high)))
    }
}

fn runtime_text(value: &Value, state: &ExecutionState, _context: &str) -> Result<String, String> {
    match value {
        Value::Text(text) | Value::File(text) => Ok(text.to_string()),
        Value::Null => Ok(String::new()),
        Value::Number(number) => {
            let number = number.to_f32();
            Ok(if number.is_nan() {
                "nan".to_owned()
            } else {
                number.to_string()
            })
        }
        Value::TypePath(path) => Ok(path.to_string()),
        Value::ModifiedTypePath(path) => Ok(path.base().to_string()),
        Value::Datum(datum) => {
            let datum = state
                .heap
                .datum(*datum)
                .map_err(|error| error.to_string())?;
            let name = FieldName::parse("name").expect("built-in datum name is valid");
            if let Ok(Value::Text(name)) = datum.field(&name) {
                return Ok(name.to_string());
            }
            Ok(datum.type_path().to_string())
        }
        // BYOND exposes lists as the engine datum display name rather than
        // joining their contents. Verified on 516.1680 for both positional
        // and associative lists: `"[L]"` is exactly `/list`.
        Value::List(_) => Ok("/list".to_owned()),
    }
}

fn qdel_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() {
        return Ok(Value::Null);
    }
    for argument in arguments {
        qdel_value(argument, state).map_err(|error| format!("qdel failed: {error}"))?;
    }
    Ok(Value::Null)
}

fn del_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    match &arguments[0] {
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

fn qdel_value(value: &Value, state: &mut ExecutionState) -> Result<(), String> {
    match value {
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

fn unregister_runtime_datum(state: &mut ExecutionState, datum: DatumId) -> Result<(), String> {
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let old_loc = state
        .heap
        .datum_field(datum, &loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(loc) => Some(*loc),
            _ => None,
        });
    synchronize_moved_atom_contents(state, datum, old_loc, None)?;

    let world = FieldName::parse("world").expect("built-in world global");
    let contents = FieldName::parse("contents").expect("built-in contents field");
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
            .remove_first(&Value::Datum(datum));
    }
    Ok(())
}

fn typecacheof_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

fn image_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

pub(super) fn is_appearance_source(path: &TypePath) -> bool {
    let path = path.as_str();
    path == "/image"
        || path.starts_with("/image/")
        || path == "/mutable_appearance"
        || path.starts_with("/mutable_appearance/")
        || ["/atom", "/area", "/turf", "/obj", "/mob"]
            .into_iter()
            .any(|root| path == root || path.starts_with(&format!("{root}/")))
}

pub(super) fn copy_image_appearance(
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

pub(super) fn appearance_snapshot_builtin(
    source: DatumId,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let appearance = state
        .heap_mut()
        .allocate_datum(TypePath::parse("/mutable_appearance").expect("built-in appearance path"));
    copy_image_appearance(&Value::Datum(source), appearance, state)?;
    Ok(Value::Datum(appearance))
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

fn regex_quote(
    arguments: &[Value],
    state: &ExecutionState,
    replacement: bool,
) -> Result<Value, String> {
    let text = arguments
        .first()
        .map(|value| runtime_text(value, state, "REGEX_QUOTE argument"))
        .transpose()?
        .unwrap_or_default();
    if replacement {
        return Ok(Value::text(text.replace('$', "$$")));
    }
    let mut quoted = String::with_capacity(text.len());
    for character in text.chars() {
        if matches!(
            character,
            '\\' | '.' | '^' | '$' | '|' | '?' | '*' | '+' | '(' | ')' | '[' | ']' | '{' | '}'
        ) {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    Ok(Value::text(quoted))
}

fn headless_browse(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let descriptor = state.heap.allocate_list();
    let list = state
        .heap
        .list_mut(descriptor)
        .expect("new browse descriptor is live");
    list.set_key(
        Value::text("body"),
        arguments.first().cloned().unwrap_or(Value::Null),
    );
    list.set_key(
        Value::text("options"),
        arguments.get(1).cloned().unwrap_or(Value::Null),
    );
    Ok(Value::List(descriptor))
}

fn headless_transfer(
    kind: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let descriptor = state.heap.allocate_list();
    state.mark_associative_list(descriptor);
    let list = state
        .heap
        .list_mut(descriptor)
        .map_err(|error| error.to_string())?;
    list.set_key(Value::text("kind"), Value::text(kind));
    list.set_key(
        Value::text("resource"),
        arguments.first().cloned().unwrap_or(Value::Null),
    );
    list.set_key(
        Value::text("name"),
        arguments.get(1).cloned().unwrap_or(Value::Null),
    );
    Ok(Value::List(descriptor))
}

fn headless_winset(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        // BYOND accepts null when no client is available; a headless server
        // has no window to mutate in that case.
        return Ok(Value::Null);
    };
    let field = FieldName::parse("_dream64_winset").expect("headless UI field is valid");
    let settings = match state.heap.datum_field(*client, &field) {
        Ok(Value::List(settings)) => *settings,
        _ => {
            let settings = state.heap.allocate_list();
            state
                .heap
                .set_datum_field(*client, field, Value::List(settings))
                .map_err(|error| error.to_string())?;
            settings
        }
    };
    let control = arguments.get(1).cloned().unwrap_or(Value::Null);
    let params = arguments.get(2).cloned().unwrap_or(Value::Null);
    state
        .heap
        .list_mut(settings)
        .map_err(|error| error.to_string())?
        .set_key(control, params);
    Ok(Value::Null)
}

fn headless_winshow(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        return Ok(Value::Null);
    };
    let field = FieldName::parse("_dream64_winshow").expect("headless UI field");
    let settings = match state.heap.datum_field(*client, &field) {
        Ok(Value::List(list)) => *list,
        _ => {
            let list = state.heap.allocate_list();
            state.mark_associative_list(list);
            state
                .heap
                .set_datum_field(*client, field, Value::List(list))
                .map_err(|e| e.to_string())?;
            list
        }
    };
    state
        .heap
        .list_mut(settings)
        .map_err(|e| e.to_string())?
        .set_key(
            arguments.get(1).cloned().unwrap_or(Value::Null),
            arguments
                .get(2)
                .cloned()
                .unwrap_or_else(|| Value::number(1.0)),
        );
    Ok(Value::Null)
}

fn headless_winclone(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let [Value::Datum(client), source, destination] = arguments else {
        return Ok(Value::number(0.0));
    };
    let field = FieldName::parse("_dream64_winset").expect("headless UI field");
    let Some(Value::List(settings)) = state.heap.datum_field(*client, &field).ok().cloned() else {
        return Ok(Value::number(0.0));
    };
    let value = state
        .heap
        .list(settings)
        .map_err(|e| e.to_string())?
        .get_key(source)
        .ok()
        .cloned();
    let Some(value) = value else {
        return Ok(Value::number(0.0));
    };
    state
        .heap
        .list_mut(settings)
        .map_err(|e| e.to_string())?
        .set_key(destination.clone(), value);
    Ok(Value::number(1.0))
}

fn headless_winget(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let (client, control, property) = match arguments {
        [Value::Datum(client), control, property] => (*client, control, property),
        _ => return Ok(Value::text("")),
    };
    let Some(Value::List(settings)) = state
        .heap
        .datum_field(
            client,
            &FieldName::parse("_dream64_winset").expect("headless UI field is valid"),
        )
        .ok()
    else {
        return Ok(Value::text(""));
    };
    let Some(control) = value_text(control) else {
        return Ok(Value::text(""));
    };
    let Some(property) = value_text(property) else {
        return Ok(Value::text(""));
    };
    let settings = state
        .heap
        .list(*settings)
        .map_err(|error| error.to_string())?;
    let Ok(value) = settings.get_key(&Value::text(control)) else {
        return Ok(Value::text(""));
    };
    let Some(parameters) = value_text(value) else {
        return Ok(Value::text(""));
    };
    let value = parameters.split(';').find_map(|entry| {
        let (name, value) = entry.split_once('=')?;
        (name.trim() == property).then(|| value.trim().to_owned())
    });
    Ok(Value::text(value.unwrap_or_default()))
}

fn headless_winexists(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(Value::Datum(client)) = arguments.first() else {
        return Ok(Value::number(0.0));
    };
    let control = arguments.get(1).cloned().unwrap_or(Value::Null);
    let exists = state
        .heap
        .datum_field(
            *client,
            &FieldName::parse("_dream64_winset").expect("headless UI field is valid"),
        )
        .ok()
        .and_then(|value| match value {
            Value::List(settings) => state.heap.list(*settings).ok(),
            _ => None,
        })
        .is_some_and(|settings| settings.get_key(&control).is_ok());
    Ok(Value::number(if exists { 1.0 } else { 0.0 }))
}

/// A headless server cannot display BYOND's modal alert window. Select the
/// first offered button, which is the deterministic analogue of accepting the
/// dialog's default action. Both documented call forms are accepted:
/// `alert(usr, message, title, button1, ...)` and the implicit-usr form.
fn headless_alert(arguments: &[Value]) -> Result<Value, String> {
    let explicit_usr =
        arguments.len() >= 4 && matches!(arguments.first(), Some(Value::Datum(_) | Value::Null));
    let button = arguments
        .get(if explicit_usr { 3 } else { 2 })
        .filter(|value| !matches!(value, Value::Null))
        .cloned()
        .unwrap_or_else(|| Value::text("Ok"));
    Ok(button)
}

fn floor_multiple(arguments: &[Value]) -> Result<Value, String> {
    let value = arguments[0]
        .as_number()
        .ok_or_else(|| format!("FLOOR value must be numeric, received {}", arguments[0]))?;
    let multiple = arguments[1]
        .as_number()
        .ok_or_else(|| format!("FLOOR multiple must be numeric, received {}", arguments[1]))?;
    if multiple == 0.0 {
        return Ok(Value::number(0.0));
    }
    Ok(Value::number((value / multiple).floor() * multiple))
}

fn length_char(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let length = match &arguments[0] {
        Value::Null => 0,
        Value::Text(text) => text.chars().count(),
        Value::List(list) => state
            .heap
            .list(*list)
            .map_err(|error| error.to_string())?
            .len(),
        value => {
            return Err(format!(
                "length_char requires text or a list, received {value}"
            ));
        }
    };
    Ok(Value::number(length as f32))
}

fn ascii2text(arguments: &[Value]) -> Result<Value, String> {
    let value = number(&arguments[0], "ascii2text")?;
    if !value.is_finite() {
        return Ok(Value::Null);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let codepoint = value.trunc().max(0.0) as u32;
    Ok(char::from_u32(codepoint)
        .map_or(Value::Null, |character| Value::text(character.to_string())))
}

fn logical_length(text: &str, character_indices: bool) -> usize {
    if character_indices {
        text.chars().count()
    } else {
        text.len()
    }
}

#[allow(clippy::cast_possible_truncation)]
fn signed_position(value: Option<&Value>, default: i64) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) => Ok(number.to_f32().trunc() as i64),
        Some(value) => Err(format!("text position requires a number, received {value}")),
    }
}

fn resolve_position(position: i64, length: usize) -> usize {
    let limit = i64::try_from(length)
        .unwrap_or(i64::MAX - 1)
        .saturating_add(1);
    let position = if position < 0 {
        limit.saturating_add(position)
    } else {
        position
    };
    usize::try_from(position.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn byte_offset(text: &str, logical_position_zero_based: usize, character_indices: bool) -> usize {
    if character_indices {
        text.char_indices()
            .nth(logical_position_zero_based)
            .map_or(text.len(), |(offset, _)| offset)
    } else {
        let mut offset = logical_position_zero_based.min(text.len());
        while offset > 0 && !text.is_char_boundary(offset) {
            offset -= 1;
        }
        offset
    }
}

fn text2ascii(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2ascii")?;
    let length = logical_length(&text, character_indices);
    let position = resolve_position(signed_position(arguments.get(1), 1)?, length);
    if position > length {
        return Ok(Value::number(0.0));
    }
    if character_indices {
        let value = text.chars().nth(position - 1).map_or(0, u32::from);
        Ok(Value::number(value as f32))
    } else {
        Ok(Value::number(f32::from(text.as_bytes()[position - 1])))
    }
}

fn text2num(arguments: &[Value], _state: &ExecutionState) -> Result<Value, String> {
    let text = match &arguments[0] {
        Value::Number(number) => return Ok(Value::Number(*number)),
        Value::Null => return Ok(Value::Null),
        Value::Text(text) => text.to_string(),
        _ => return Ok(Value::Null),
    };
    let radix = if let Some(radix) = arguments.get(1) {
        number(radix, "text2num radix")?.trunc() as i32
    } else {
        10
    };
    if !(2..=36).contains(&radix) {
        return Ok(Value::Null);
    }
    let text = text.trim_start();
    if radix == 10 {
        let bytes = text.as_bytes();
        let mut end = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
        let mut saw_digit = false;
        let mut saw_dot = false;
        let mut saw_exp = false;
        while let Some(byte) = bytes.get(end).copied() {
            if byte.is_ascii_digit() {
                saw_digit = true;
                end += 1;
            } else if byte == b'.' && !saw_dot && !saw_exp {
                saw_dot = true;
                end += 1;
            } else if matches!(byte, b'e' | b'E') && saw_digit && !saw_exp {
                saw_exp = true;
                end += 1;
                if matches!(bytes.get(end), Some(b'+' | b'-')) {
                    end += 1;
                }
            } else {
                break;
            }
        }
        if !saw_digit {
            return Ok(Value::Null);
        }
        return text[..end]
            .parse::<f32>()
            .map(Value::number)
            .or(Ok(Value::Null));
    }
    let mut chars = text.char_indices();
    let mut sign = 1_i64;
    let mut start = 0;
    if let Some((_, first)) = chars.next() {
        if first == '-' {
            sign = -1;
            start = 1;
        } else if first == '+' {
            start = 1;
        }
    }
    let mut end = start;
    for (offset, character) in text[start..].char_indices() {
        if character.to_digit(radix as u32).is_none() {
            break;
        }
        end = start + offset + character.len_utf8();
    }
    if end == start {
        return Ok(Value::Null);
    }
    let integer = i64::from_str_radix(&text[start..end], radix as u32).ok();
    Ok(integer.map_or(Value::Null, |value| Value::number((value * sign) as f32)))
}

fn text2path(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    // BYOND 516 returns null for every non-text input, including an already
    // resolved type path. Only the textual spelling participates in lookup.
    let Value::Text(text) = &arguments[0] else {
        return Ok(Value::Null);
    };
    Ok(state
        .type_paths
        .iter()
        .find(|path| path.as_str() == text.as_ref())
        .cloned()
        .map_or(Value::Null, Value::TypePath))
}

fn numeric_classifier(
    arguments: &[Value],
    predicate: impl FnOnce(f32) -> bool,
) -> Result<Value, String> {
    Ok(Value::number(f32::from(match &arguments[0] {
        Value::Number(number) => predicate(number.to_f32()),
        _ => false,
    })))
}

fn cmptext(arguments: &[Value], state: &ExecutionState, exact: bool) -> Result<Value, String> {
    let first = strict_text(&arguments[0], state, "cmptext")?;
    for value in &arguments[1..] {
        let value = strict_text(value, state, "cmptext")?;
        let matches = if exact {
            first == value
        } else {
            first.eq_ignore_ascii_case(&value) || first.to_lowercase() == value.to_lowercase()
        };
        if !matches {
            return Ok(Value::number(0.0));
        }
    }
    Ok(Value::number(1.0))
}

fn text_region(text: &str, start: i64, end: i64, character_indices: bool) -> (usize, usize, usize) {
    let length = logical_length(text, character_indices);
    let start = resolve_position(start, length);
    let end = if end == 0 {
        length.saturating_add(1)
    } else {
        resolve_position(end, length)
    };
    let (start, end) = if end < start {
        (end, start)
    } else {
        (start, end)
    };
    let start_byte = byte_offset(text, start.saturating_sub(1), character_indices);
    let end_byte = byte_offset(text, end.saturating_sub(1), character_indices);
    (start_byte, end_byte, start)
}

fn find_match(text: &str, needle: &str, exact: bool, reverse: bool) -> Option<usize> {
    let matches_at = |offset: usize| {
        let tail = &text[offset..];
        if exact {
            tail.starts_with(needle)
        } else {
            tail.get(..needle.len())
                .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
                || tail.to_lowercase().starts_with(&needle.to_lowercase())
        }
    };
    let mut offsets = text
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(text.len()));
    if reverse {
        offsets
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .find(|offset| matches_at(*offset))
    } else {
        offsets.find(|offset| matches_at(*offset))
    }
}

fn findtext(
    arguments: &[Value],
    state: &mut ExecutionState,
    exact: bool,
    character_indices: bool,
    reverse: bool,
) -> Result<Value, String> {
    // BYOND text searches accept null as an empty text value. This is
    // observable when `file2text()` returns null for a directory entry from
    // `flist()`: map readers probe the result with `findtext()` before their
    // regex loop and must receive a normal no-match rather than a runtime.
    let haystack = match &arguments[0] {
        Value::Null => String::new(),
        Value::Text(text) | Value::File(text) => text.to_string(),
        // BYOND's text-search family returns a normal no-match for a
        // non-text haystack instead of raising a runtime. Monkestation's
        // immune system relies on this when an older call site passes its
        // `/datum/blood_type` singleton directly to `findtext()`.
        _ => return Ok(Value::number(0.0)),
    };
    if let Value::Datum(regex) = arguments[1] {
        let start = signed_position(arguments.get(2), 1)?.max(1) as usize;
        let end = signed_position(arguments.get(3), 0)?;
        let end = if end <= 0 {
            haystack.len() + 1
        } else {
            end as usize
        };
        return regex_find(regex, &haystack, start, end, false, false, state);
    }
    let needle = strict_text(&arguments[1], state, "findtext needle")?;
    if reverse {
        let length = logical_length(&haystack, character_indices);
        let start = signed_position(arguments.get(2), 0)?;
        let start = if start == 0 {
            length.saturating_add(1)
        } else {
            resolve_position(start, length)
        };
        let end = resolve_position(signed_position(arguments.get(3), 1)?, length);
        if start < end {
            return Ok(Value::number(0.0));
        }
        let region_start = byte_offset(&haystack, end.saturating_sub(1), character_indices);
        let region_end = byte_offset(&haystack, start.saturating_sub(1), character_indices);
        let region = &haystack[region_start..region_end];
        let Some(found) = find_match(region, &needle, exact, true) else {
            return Ok(Value::number(0.0));
        };
        let byte = region_start + found;
        let position = if character_indices {
            haystack[..byte].chars().count() + 1
        } else {
            byte + 1
        };
        return Ok(Value::number(position as f32));
    }
    let start = signed_position(arguments.get(2), 1)?;
    let end = signed_position(arguments.get(3), 0)?;
    let (region_start, region_end, _) = text_region(&haystack, start, end, character_indices);
    let region = &haystack[region_start..region_end];
    let Some(found) = find_match(region, &needle, exact, false) else {
        return Ok(Value::number(0.0));
    };
    let byte = region_start + found;
    let position = if character_indices {
        haystack[..byte].chars().count() + 1
    } else {
        byte + 1
    };
    Ok(Value::number(position as f32))
}

pub(super) fn is_regex_datum(datum: DatumId, state: &ExecutionState) -> bool {
    state
        .heap()
        .datum(datum)
        .is_ok_and(|value| value.type_path().as_str() == "/regex")
}

pub(super) fn execute_regex_method(
    datum: DatumId,
    method: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if method != "Find" || arguments.is_empty() || arguments.len() > 3 {
        return Err(format!("unknown or invalid /regex procedure {method:?}"));
    }
    // `/regex.Find()` applies the same null-to-empty text coercion as the
    // global text-search procedures. In particular, this lets a parsed-map
    // reader finish cleanly after `file2text()` rejected a directory.
    let haystack = if matches!(arguments[0], Value::Null) {
        String::new()
    } else {
        strict_text(&arguments[0], state, "regex.Find haystack")?
    };
    let supplied_start = arguments
        .get(1)
        .is_some_and(|value| !matches!(value, Value::Null));
    let start = signed_position(arguments.get(1), 1)?.max(1) as usize;
    let end = signed_position(arguments.get(2), 0)?;
    let end = if end <= 0 {
        haystack.len() + 1
    } else {
        end as usize
    };
    regex_find(datum, &haystack, start, end, true, !supplied_start, state)
}

fn regex_find(
    datum: DatumId,
    haystack: &str,
    requested_start: usize,
    requested_end: usize,
    method_call: bool,
    use_global_cursor: bool,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let field = |name| FieldName::parse(name).expect("regex field is valid");
    let pattern = state
        .heap()
        .datum_field(datum, &field("_dream64_pattern"))
        .map_err(|error| error.to_string())?
        .clone();
    let pattern = strict_text(&pattern, state, "regex pattern")?;
    let flags = state
        .heap()
        .datum_field(datum, &field("flags"))
        .ok()
        .and_then(value_text)
        .unwrap_or("")
        .to_owned();
    let global = flags.contains('g');
    let previous = state
        .heap()
        .datum_field(datum, &field("_dream64_haystack"))
        .ok()
        .and_then(value_text);
    let cursor = state
        .heap()
        .datum_field(datum, &field("_dream64_cursor"))
        .ok()
        .and_then(Value::as_number)
        .unwrap_or(0.0) as usize;
    let start = if global && use_global_cursor && previous == Some(haystack) && cursor > 0 {
        cursor
    } else {
        requested_start.saturating_sub(1)
    };
    let end = requested_end.saturating_sub(1).min(haystack.len());
    let found = regex_search(&pattern, &flags, haystack, start.min(end), end)?;
    let Some((begin, finish, captures)) = found else {
        if global {
            for (name, value) in [
                ("next", Value::Null),
                ("_dream64_cursor", Value::number(0.0)),
            ] {
                state
                    .heap_mut()
                    .set_datum_field(datum, field(name), value)
                    .map_err(|e| e.to_string())?;
            }
        }
        if method_call {
            state
                .heap_mut()
                .set_datum_field(datum, field("text"), Value::text(haystack))
                .map_err(|e| e.to_string())?;
        }
        return Ok(Value::number(0.0));
    };
    let groups = state.heap_mut().allocate_list();
    for capture in captures {
        state
            .heap_mut()
            .list_mut(groups)
            .map_err(|error| error.to_string())?
            .add(capture.map_or(Value::Null, Value::text));
    }
    let next = if finish > begin {
        finish
    } else {
        finish.saturating_add(1)
    };
    let mut fields = vec![
        ("match", Value::text(&haystack[begin..finish])),
        ("index", Value::number((begin + 1) as f32)),
        ("group", Value::List(groups)),
        ("_dream64_cursor", Value::number(next as f32)),
        ("_dream64_haystack", Value::text(haystack)),
    ];
    if global {
        fields.push(("next", Value::number((next + 1) as f32)));
    }
    if method_call {
        fields.push(("text", Value::text(haystack)));
    }
    for (name, value) in fields {
        state
            .heap_mut()
            .set_datum_field(datum, field(name), value)
            .map_err(|e| e.to_string())?;
    }
    Ok(Value::number((begin + 1) as f32))
}

pub(super) fn regex_search(
    pattern: &str,
    flags: &str,
    haystack: &str,
    start: usize,
    end: usize,
) -> Result<Option<(usize, usize, Vec<Option<String>>)>, String> {
    let pattern = translate_byond_regex_pattern(pattern);
    let case_insensitive = flags.contains('i');
    let multi_line = flags.contains('m');
    type RegexCache = HashMap<(String, bool, bool), Arc<fancy_regex::Regex>>;
    static REGEX_CACHE: OnceLock<Mutex<RegexCache>> = OnceLock::new();
    let key = (pattern.clone(), case_insensitive, multi_line);
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let regex = cache
        .lock()
        .map_err(|_| "regex cache lock is poisoned".to_owned())?
        .get(&key)
        .cloned();
    let regex = if let Some(regex) = regex {
        regex
    } else {
        let mut builder = fancy_regex::RegexBuilder::new(&pattern);
        builder
            .case_insensitive(case_insensitive)
            .multi_line(multi_line);
        let regex = Arc::new(
            builder
                .build()
                .map_err(|error| format!("invalid regex {pattern:?}: {error}"))?,
        );
        cache
            .lock()
            .map_err(|_| "regex cache lock is poisoned".to_owned())?
            .insert(key, Arc::clone(&regex));
        regex
    };
    let captures = regex
        .captures_from_pos(haystack, start)
        .map_err(|error| format!("regex match failed for {pattern:?}: {error}"))?;
    let Some(captures) = captures else {
        return Ok(None);
    };
    let Some(whole) = captures.get(0) else {
        return Ok(None);
    };
    if whole.end() > end {
        return Ok(None);
    }
    let groups = (1..captures.len())
        .map(|index| {
            captures
                .get(index)
                .map(|capture| capture.as_str().to_owned())
        })
        .collect();
    Ok(Some((whole.start(), whole.end(), groups)))
}

fn translate_byond_regex_pattern(pattern: &str) -> String {
    let mut translated = String::with_capacity(pattern.len());
    let mut characters = pattern.chars().peekable();
    let mut in_character_class = false;
    while let Some(character) = characters.next() {
        match character {
            '[' => {
                in_character_class = true;
                translated.push(character);
            }
            ']' => {
                in_character_class = false;
                translated.push(character);
            }
            '\\' if characters.peek() == Some(&'l') => {
                characters.next();
                if in_character_class {
                    translated.push_str("A-Za-z");
                } else {
                    translated.push_str("[A-Za-z]");
                }
            }
            _ => translated.push(character),
        }
    }
    translated
}

fn splittext(
    arguments: &[Value],
    state: &mut ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    if matches!(arguments[0], Value::Null) {
        return Ok(Value::List(state.heap.allocate_list()));
    }
    let text = strict_text(&arguments[0], state, "splittext text")?;
    let start = signed_position(arguments.get(2), 1)?;
    let end = signed_position(arguments.get(3), 0)?;
    let include_delimiters = arguments.get(4).is_some_and(truthy);
    let (region_start, region_end, _) = text_region(&text, start, end, character_indices);
    let target = &text[region_start..region_end];
    let list = state.heap.allocate_list();
    let mut output = Vec::new();
    if let Value::Datum(regex) = arguments[1] {
        if !is_regex_datum(regex, state) {
            return Ok(Value::List(list));
        }
        let field = |name| FieldName::parse(name).expect("regex field is valid");
        let pattern = state
            .heap()
            .datum_field(regex, &field("_dream64_pattern"))
            .map_err(|error| error.to_string())?
            .clone();
        let pattern = strict_text(&pattern, state, "splittext regex delimiter")?;
        let flags = state
            .heap()
            .datum_field(regex, &field("flags"))
            .ok()
            .and_then(value_text)
            .unwrap_or("")
            .to_owned();
        let mut segment_start = region_start;
        let mut search_start = region_start;
        while search_start <= region_end {
            let Some((found, finish, captures)) =
                regex_search(&pattern, &flags, &text, search_start, region_end)?
            else {
                break;
            };
            output.push(text[segment_start..found].to_owned());
            if include_delimiters {
                output.push(text[found..finish].to_owned());
            } else {
                output.extend(captures.into_iter().flatten());
            }
            segment_start = finish;
            search_start = if finish > found {
                finish
            } else {
                finish
                    + text[finish..region_end]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1)
            };
        }
        output.push(text[segment_start..region_end].to_owned());
    } else {
        let delimiter = strict_text(&arguments[1], state, "splittext delimiter")?;
        if delimiter.is_empty() {
            output.extend(target.chars().map(|character| character.to_string()));
        } else {
            let mut cursor = 0;
            while let Some(found) = target[cursor..].find(&delimiter) {
                let found = cursor + found;
                output.push(target[cursor..found].to_owned());
                if include_delimiters {
                    output.push(delimiter.clone());
                }
                cursor = found + delimiter.len();
            }
            output.push(target[cursor..].to_owned());
        }
    }
    // BYOND applies Start/End only to the matching region. Text outside that
    // region remains attached to the first and last split elements.
    if output.is_empty() {
        output.push(text.clone());
    } else {
        output[0].insert_str(0, &text[..region_start]);
        output
            .last_mut()
            .expect("split output exists")
            .push_str(&text[region_end..]);
    }
    let entries = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for item in output {
        entries.add(Value::text(item));
    }
    Ok(Value::List(list))
}

fn jointext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Value::List(list) = arguments[0] else {
        return Err(format!(
            "jointext requires a list, received {}",
            arguments[0]
        ));
    };
    let glue = runtime_text(&arguments[1], state, "jointext glue")?;
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    let length = list.len();
    let start = resolve_position(signed_position(arguments.get(2), 1)?, length);
    let end_arg = signed_position(arguments.get(3), 0)?;
    let end = if end_arg == 0 {
        length.saturating_add(1)
    } else {
        resolve_position(end_arg, length)
    };
    let mut items = Vec::new();
    for index in start..end.min(length.saturating_add(1)) {
        let value = list.get(index).map_err(|error| error.to_string())?;
        items.push(runtime_text(value, state, "jointext item")?);
    }
    Ok(Value::text(items.join(&glue)))
}

fn addtext(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let mut output = String::new();
    for value in arguments {
        output.push_str(&strict_text(value, state, "addtext")?);
    }
    Ok(Value::text(output))
}

fn spantext(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
    matching: bool,
) -> Result<Value, String> {
    let haystack = strict_text(&arguments[0], state, "spantext haystack")?;
    let needles = strict_text(&arguments[1], state, "spantext needles")?;
    let length = logical_length(&haystack, character_indices);
    let start = resolve_position(signed_position(arguments.get(2), 1)?, length);
    let start_byte = byte_offset(&haystack, start.saturating_sub(1), character_indices);
    let mut count = 0usize;
    for character in haystack[start_byte..].chars() {
        let contains = needles.contains(character);
        if contains != matching {
            break;
        }
        count += if character_indices {
            1
        } else {
            character.len_utf8()
        };
    }
    Ok(Value::number(count as f32))
}

fn splicetext(
    arguments: &[Value],
    state: &ExecutionState,
    character_indices: bool,
) -> Result<Value, String> {
    let source = strict_text(&arguments[0], state, "splicetext text")?;
    let start = signed_position(arguments.get(1), 1)?;
    let end = signed_position(arguments.get(2), 0)?;
    let replacement = strict_text(&arguments[3], state, "splicetext replacement")?;
    let (start, end, _) = text_region(&source, start, end, character_indices);
    Ok(Value::text(format!(
        "{}{}{}",
        &source[..start],
        replacement,
        &source[end..]
    )))
}

pub(super) fn datum_coordinates(state: &ExecutionState, value: &Value) -> Option<(f32, f32, f32)> {
    let Value::Datum(original) = value else {
        return None;
    };
    // BYOND spatial builtins use the containing turf for objects nested in
    // mobs, items, closets, and other movable containers. Their own x/y/z
    // fields may still contain zero or a stale former turf coordinate. Follow
    // loc links exactly as get_step(atom, 0) does, while retaining the
    // original datum as the fallback for lightweight uncontained fixtures.
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let mut coordinate_source = *original;
    let mut current = *original;
    let mut visited = HashSet::new();
    while visited.insert(current) {
        let datum = state.heap.datum(current).ok()?;
        if super::is_turf_type_path(datum.type_path()) {
            coordinate_source = current;
            break;
        }
        let Ok(Value::Datum(parent)) = super::datum_field_or_initial(state, current, &loc) else {
            break;
        };
        current = parent;
    }
    let coordinate = |name: &str| {
        super::datum_field_or_initial(
            state,
            coordinate_source,
            &FieldName::parse(name).expect("coordinate name is valid"),
        )
        .ok()?
        .as_number()
    };
    Some((coordinate("x")?, coordinate("y")?, coordinate("z")?))
}

/// Adds one turf followed by its direct movable contents, matching the cell
/// ordering used by BYOND/OpenDream's view enumeration. Inventory descendants
/// are not members of the surrounding turf cell and therefore remain hidden.
fn append_spatial_cell(
    state: &ExecutionState,
    turf: DatumId,
    expected_coordinate: (i32, i32, i32),
    seen: &mut HashSet<DatumId>,
    output: &mut Vec<DatumId>,
) {
    let Some((x, y, z)) = datum_coordinates(state, &Value::Datum(turf)) else {
        return;
    };
    let (expected_x, expected_y, expected_z) = expected_coordinate;
    if (x, y, z) != (expected_x as f32, expected_y as f32, expected_z as f32)
        || !state
            .heap
            .datum(turf)
            .is_ok_and(|datum| super::is_turf_type_path(datum.type_path()))
        || !seen.insert(turf)
    {
        return;
    }
    output.push(turf);

    let contents = FieldName::parse("contents").expect("built-in contents field");
    let members = state
        .heap
        .datum_field(turf, &contents)
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok(),
            _ => None,
        })
        .map(|list| {
            list.positions()
                .filter_map(|(_, value)| match value {
                    Value::Datum(member) => Some(*member),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for member in members {
        let Ok(datum) = state.heap.datum(member) else {
            continue;
        };
        let path = datum.type_path().as_str();
        if (path == "/area" || path.starts_with("/area/")) || !seen.insert(member) {
            continue;
        }
        output.push(member);
    }
}

fn spiral_order_key(delta_x: i32, delta_y: i32, distance_x: i32, distance_y: i32) -> (u32, u64) {
    let radius = delta_x.unsigned_abs().max(delta_y.unsigned_abs());
    if radius == 0 {
        return (0, 0);
    }
    let radius_i32 = i32::try_from(radius).expect("coordinate delta radius fits i32");
    let vertical_radius = radius_i32.min(distance_y);
    let left_count = if radius_i32 <= distance_x {
        u64::from(vertical_radius.unsigned_abs()) * 2 + 1
    } else {
        0
    };
    if radius_i32 <= distance_x && delta_x == -radius_i32 {
        return (
            radius,
            u64::from((delta_y + vertical_radius).unsigned_abs()),
        );
    }

    let interior_low = (-radius_i32 + 1).max(-distance_x);
    let interior_high = (radius_i32 - 1).min(distance_x);
    let interior_count = if radius_i32 <= distance_y && interior_low <= interior_high {
        u64::from((interior_high - interior_low + 1).unsigned_abs()) * 2
    } else {
        0
    };
    if radius_i32 <= distance_y
        && (delta_y == -radius_i32 || delta_y == radius_i32)
        && delta_x >= interior_low
        && delta_x <= interior_high
    {
        let top = u64::from(delta_y == radius_i32);
        return (
            radius,
            left_count + u64::from((delta_x - interior_low).unsigned_abs()) * 2 + top,
        );
    }

    (
        radius,
        left_count + interior_count + u64::from((delta_y + vertical_radius).unsigned_abs()),
    )
}

fn indexed_spatial_candidates(
    state: &ExecutionState,
    center_x: f32,
    center_y: f32,
    center_z: f32,
    distance_x: f32,
    distance_y: f32,
) -> Vec<DatumId> {
    let integral_coordinate = |value: f32| -> Option<i32> {
        (value.is_finite()
            && value.fract() == 0.0
            && value >= i32::MIN as f32
            && value <= i32::MAX as f32)
            .then(|| value as i32)
    };
    let (Some(center_x), Some(center_y), Some(center_z)) = (
        integral_coordinate(center_x),
        integral_coordinate(center_y),
        integral_coordinate(center_z),
    ) else {
        return Vec::new();
    };
    let distance_x = distance_x.min(i32::MAX as f32) as i32;
    let distance_y = distance_y.min(i32::MAX as f32) as i32;
    let low_x = center_x.saturating_sub(distance_x);
    let high_x = center_x.saturating_add(distance_x);
    let low_y = center_y.saturating_sub(distance_y);
    let high_y = center_y.saturating_add(distance_y);

    let axis_len = |low: i32, high: i32| {
        u128::try_from(i64::from(high) - i64::from(low) + 1)
            .expect("ordered i32 bounds have a positive span")
    };
    let area = axis_len(low_x, high_x).saturating_mul(axis_len(low_y, high_y));
    let direct_limit = (state.world_turfs.len() as u128)
        .saturating_mul(2)
        .max(4_096);
    let ordered_turfs = if area <= direct_limit {
        let mut turfs = Vec::new();
        if let Some(turf) = state.turf_at(center_x, center_y, center_z) {
            turfs.push(((center_x, center_y), turf));
        }
        for radius in 1..=distance_x.max(distance_y) {
            let vertical_radius = radius.min(distance_y);
            if radius <= distance_x {
                let x = center_x.saturating_sub(radius);
                for delta_y in -vertical_radius..=vertical_radius {
                    if let Some(turf) = state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                    {
                        turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                    }
                }
            }
            if radius <= distance_y {
                let low_delta_x = (-radius + 1).max(-distance_x);
                let high_delta_x = (radius - 1).min(distance_x);
                for delta_x in low_delta_x..=high_delta_x {
                    let x = center_x.saturating_add(delta_x);
                    for delta_y in [-radius, radius] {
                        if let Some(turf) =
                            state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                        {
                            turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                        }
                    }
                }
            }
            if radius <= distance_x {
                let x = center_x.saturating_add(radius);
                for delta_y in -vertical_radius..=vertical_radius {
                    if let Some(turf) = state.turf_at(x, center_y.saturating_add(delta_y), center_z)
                    {
                        turfs.push(((x, center_y.saturating_add(delta_y)), turf));
                    }
                }
            }
        }
        turfs
    } else {
        let mut turfs = state
            .world_turfs
            .iter()
            .filter_map(|((x, y, z), turf)| {
                (*z == center_z && *x >= low_x && *x <= high_x && *y >= low_y && *y <= high_y)
                    .then_some(((*x, *y), *turf))
            })
            .collect::<Vec<_>>();
        turfs.sort_unstable_by_key(|((x, y), _)| {
            spiral_order_key(*x - center_x, *y - center_y, distance_x, distance_y)
        });
        turfs
    };

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for ((x, y), turf) in ordered_turfs {
        append_spatial_cell(state, turf, (x, y, center_z), &mut seen, &mut candidates);
    }
    candidates
}

fn append_orange_candidate(
    state: &mut ExecutionState,
    output: ListId,
    candidate: DatumId,
    center: &Value,
    loc: &FieldName,
) -> Result<(), String> {
    let datum = state
        .heap
        .datum(candidate)
        .map_err(|error| error.to_string())?;
    if !super::is_atom_type_path(datum.type_path()) || Value::Datum(candidate).semantic_eq(center) {
        return Ok(());
    }
    let candidate_loc =
        super::datum_field_or_initial(state, candidate, loc).map_err(|error| error.to_string())?;
    if candidate_loc.semantic_eq(center) {
        return Ok(());
    }
    state
        .heap
        .list_mut(output)
        .map_err(|error| error.to_string())?
        .add(Value::Datum(candidate));
    Ok(())
}

/// Native form of BYOND's `orange()` using the same indexed cell order as
/// `range()`, but filtering directly into the result. The semantic builtin's
/// historical DM body materialized an intermediate range list and then kept
/// every atom except the center and atoms whose direct `loc` was the center.
fn orange_builtin(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
) -> Result<Value, String> {
    let output = state.heap.allocate_list();
    let Some(first) = arguments.first() else {
        return Err("orange requires one or two arguments".to_owned());
    };
    let second = arguments.get(1).unwrap_or(usr);
    let (distance, center) = if let Some(distance) = first.as_number() {
        (Some(distance), second)
    } else {
        (second.as_number(), first)
    };
    let Some(distance) = distance else {
        return Ok(Value::List(output));
    };
    if !distance.is_finite() || distance < 0.0 {
        return Ok(Value::List(output));
    }
    let Some((center_x, center_y, center_z)) = datum_coordinates(state, center) else {
        return Ok(Value::List(output));
    };
    let distance = distance.floor();
    let loc = FieldName::parse("loc").expect("built-in loc field");

    if state.world_turfs.is_empty() {
        // Geometry-free fixtures retain range()'s historical arena scan and
        // direct coordinate fields. Production worlds never enter this path.
        let x = FieldName::parse("x").expect("built-in coordinate field");
        let y = FieldName::parse("y").expect("built-in coordinate field");
        let z = FieldName::parse("z").expect("built-in coordinate field");
        let candidates = state
            .heap
            .datums()
            .filter_map(|(candidate, datum)| {
                let path = datum.type_path().as_str();
                if path == "/area" || path.starts_with("/area/") {
                    return None;
                }
                let candidate_x = datum.field(&x).ok()?.as_number()?;
                let candidate_y = datum.field(&y).ok()?.as_number()?;
                let candidate_z = datum.field(&z).ok()?.as_number()?;
                (candidate_z.total_cmp(&center_z).is_eq()
                    && (candidate_x - center_x).abs() <= distance
                    && (candidate_y - center_y).abs() <= distance)
                    .then_some(candidate)
            })
            .collect::<Vec<_>>();
        for candidate in candidates {
            append_orange_candidate(state, output, candidate, center, &loc)?;
        }
        return Ok(Value::List(output));
    }

    let integral_coordinate = |value: f32| -> Option<i32> {
        (value.is_finite()
            && value.fract() == 0.0
            && value >= i32::MIN as f32
            && value <= i32::MAX as f32)
            .then(|| value as i32)
    };
    let (Some(center_x), Some(center_y), Some(center_z)) = (
        integral_coordinate(center_x),
        integral_coordinate(center_y),
        integral_coordinate(center_z),
    ) else {
        return Ok(Value::List(output));
    };
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
        let center_tile = tiles.remove(index);
        tiles.insert(0, center_tile);
    }

    let contents = FieldName::parse("contents").expect("built-in contents field");
    let mut seen_areas = HashSet::new();
    for (coordinate, turf) in tiles {
        append_orange_candidate(state, output, turf, center, &loc)?;
        if let Some(area) = state.world_areas.get(&coordinate).copied()
            && seen_areas.insert(area)
        {
            append_orange_candidate(state, output, area, center, &loc)?;
        }
        let members = state
            .heap
            .datum_field(turf, &contents)
            .ok()
            .and_then(|value| match value {
                Value::List(list) => state.heap.list(*list).ok(),
                _ => None,
            })
            .map(|list| {
                list.positions()
                    .filter_map(|(_, value)| match value {
                        Value::Datum(member) => Some(*member),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        for member in members {
            append_orange_candidate(state, output, member, center, &loc)?;
        }
    }
    Ok(Value::List(output))
}

fn spatial_query(
    arguments: &[Value],
    state: &mut ExecutionState,
    usr: &Value,
    mobs_only: bool,
    exclude_center: bool,
) -> Result<Value, String> {
    let default_distance = state
        .global(&FieldName::parse("world").expect("built-in world global"))
        .and_then(|world| match world {
            Value::Datum(world) => state
                .heap
                .datum_field(
                    *world,
                    &FieldName::parse("view").expect("built-in world view field"),
                )
                .ok(),
            _ => None,
        })
        .and_then(Value::as_number)
        .filter(|distance| distance.is_finite() && *distance >= 0.0)
        .unwrap_or(5.0)
        .floor();
    let mut distance_x = default_distance;
    let mut distance_y = default_distance;
    let mut center = usr.clone();
    for argument in arguments {
        match argument {
            Value::Null => {}
            Value::Datum(id) => {
                let datum = state.heap.datum(*id).map_err(|error| error.to_string())?;
                let atom = TypePath::parse("/atom").expect("built-in atom path");
                if !is_subtype(state, datum.type_path(), &atom) {
                    return Err(format!(
                        "spatial query center requires an atom, received {argument}"
                    ));
                }
                center = argument.clone();
            }
            Value::Number(value) => {
                distance_x = value.to_f32().floor();
                distance_y = distance_x;
            }
            Value::Text(value) => {
                let (width, height) = value
                    .split_once('x')
                    .or_else(|| value.split_once('X'))
                    .ok_or_else(|| {
                        format!("spatial query distance requires a number or view size, received {argument}")
                    })?;
                let width = width.trim().parse::<u32>().map_err(|_| {
                    format!("spatial query distance has an invalid width: {argument}")
                })?;
                let height = height.trim().parse::<u32>().map_err(|_| {
                    format!("spatial query distance has an invalid height: {argument}")
                })?;
                distance_x = (width / 2) as f32;
                distance_y = (height / 2) as f32;
            }
            _ => {
                return Err(format!(
                    "spatial query requires an atom and optional distance, received {argument}"
                ));
            }
        }
    }
    let output = state.heap.allocate_list();
    let Some((center_x, center_y, center_z)) = datum_coordinates(state, &center) else {
        return Ok(Value::List(output));
    };
    if !distance_x.is_finite() || distance_x < 0.0 || !distance_y.is_finite() || distance_y < 0.0 {
        return Ok(Value::List(output));
    }
    let candidates = if state.world_turfs.is_empty() {
        // Lightweight standalone fixtures may supply coordinate-bearing atoms
        // without constructing canonical world geometry.
        state.heap.datums().map(|(id, _)| id).collect::<Vec<_>>()
    } else {
        indexed_spatial_candidates(state, center_x, center_y, center_z, distance_x, distance_y)
    };
    let matching = candidates
        .into_iter()
        .filter_map(|id| {
            let datum = state.heap.datum(id).ok()?;
            let path = datum.type_path().as_str();
            if path == "/area" || path.starts_with("/area/") {
                return None;
            }
            if mobs_only && path != "/mob" && !path.starts_with("/mob/") {
                return None;
            }
            let (x, y, z) = datum_coordinates(state, &Value::Datum(id))?;
            if exclude_center && x == center_x && y == center_y && z == center_z {
                return None;
            }
            (z == center_z
                && (x - center_x).abs() <= distance_x
                && (y - center_y).abs() <= distance_y)
                .then_some(id)
        })
        .collect::<Vec<_>>();
    let list = state
        .heap
        .list_mut(output)
        .map_err(|error| error.to_string())?;
    for datum in matching {
        list.add(Value::Datum(datum));
    }
    Ok(Value::List(output))
}

fn step_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let Value::Datum(atom) = arguments[0] else {
        return Ok(Value::number(0.0));
    };
    let direction = number(&arguments[1], "step direction")? as i16;
    if direction & !15 != 0 {
        return Ok(Value::number(0.0));
    }
    let Some((x, y, z)) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(0.0));
    };
    let target = (
        x + f32::from(u8::from(direction & 4 != 0)) - f32::from(u8::from(direction & 8 != 0)),
        y + f32::from(u8::from(direction & 1 != 0)) - f32::from(u8::from(direction & 2 != 0)),
        z,
    );
    let turf = state.heap.datums().find_map(|(id, datum)| {
        let path = datum.type_path().as_str();
        if path != "/turf" && !path.starts_with("/turf/") {
            return None;
        }
        let coordinate = |name: &str| {
            datum
                .field(&FieldName::parse(name).expect("coordinate field"))
                .ok()?
                .as_number()
        };
        ((coordinate("x")?, coordinate("y")?, coordinate("z")?) == target).then_some(id)
    });
    let Some(turf) = turf else {
        return Ok(Value::number(0.0));
    };
    let loc_name = FieldName::parse("loc").expect("movement field");
    let old_loc = state
        .heap
        .datum_field(atom, &loc_name)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    if old_loc != Some(turf) {
        synchronize_moved_atom_contents(state, atom, old_loc, Some(turf))?;
    }
    for (name, value) in [
        ("x", Value::number(target.0)),
        ("y", Value::number(target.1)),
        ("z", Value::number(target.2)),
        ("loc", Value::Datum(turf)),
    ] {
        state
            .heap
            .set_datum_field(atom, FieldName::parse(name).expect("movement field"), value)
            .map_err(|error| error.to_string())?;
    }
    Ok(Value::number(1.0))
}

fn direction_between(source: &Value, target: &Value, state: &ExecutionState, away: bool) -> i16 {
    let (Some((sx, sy, sz)), Some((tx, ty, tz))) = (
        datum_coordinates(state, source),
        datum_coordinates(state, target),
    ) else {
        return 0;
    };
    if sz != tz {
        return 0;
    }
    let (dx, dy) = if away {
        (sx - tx, sy - ty)
    } else {
        (tx - sx, ty - sy)
    };
    let mut direction = 0;
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
    direction
}

fn step_towards_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let direction = direction_between(&arguments[0], &arguments[1], state, false);
    step_builtin(
        &[arguments[0].clone(), Value::number(f32::from(direction))],
        state,
    )
}

fn within_minimum_distance(arguments: &[Value], state: &ExecutionState) -> bool {
    let minimum = arguments.get(2).and_then(Value::as_number).unwrap_or(0.0);
    minimum > 0.0
        && matches!(
            (datum_coordinates(state, &arguments[0]), datum_coordinates(state, &arguments[1])),
            (Some(left), Some(right))
                if left.2 == right.2
                    && (left.0 - right.0).abs().max((left.1 - right.1).abs()) <= minimum
        )
}

fn step_to_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if within_minimum_distance(arguments, state) {
        return Ok(Value::number(0.0));
    }
    step_towards_builtin(arguments, state)
}

fn get_step_to_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if within_minimum_distance(arguments, state) {
        return Ok(Value::Null);
    }
    let direction = direction_between(&arguments[0], &arguments[1], state, false);
    super::get_step_builtin(&arguments[0], &Value::number(f32::from(direction)), state)
}

fn step_away_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if let Some(maximum) = arguments.get(2).and_then(Value::as_number)
        && maximum > 0.0
        && let (Some(source), Some(target)) = (
            datum_coordinates(state, &arguments[0]),
            datum_coordinates(state, &arguments[1]),
        )
        && (source.0 - target.0)
            .abs()
            .max((source.1 - target.1).abs())
            .max((source.2 - target.2).abs())
            > maximum
    {
        return Ok(Value::number(0.0));
    }
    let direction = direction_between(&arguments[0], &arguments[1], state, true);
    step_builtin(
        &[arguments[0].clone(), Value::number(f32::from(direction))],
        state,
    )
}

fn get_step_away_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let direction = direction_between(&arguments[0], &arguments[1], state, true);
    super::get_step_builtin(&arguments[0], &Value::number(f32::from(direction)), state)
}

fn step_rand_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let directions = [1_i16, 2, 4, 8];
    let index = (super::deterministic_unit(&mut state.random_state) * 4.0) as usize;
    step_builtin(
        &[
            arguments[0].clone(),
            Value::number(f32::from(directions[index.min(3)])),
        ],
        state,
    )
}

fn get_step_rand_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let directions = [1_i16, 2, 4, 8];
    let index = (super::deterministic_unit(&mut state.random_state) * 4.0) as usize;
    super::get_step_builtin(
        &arguments[0],
        &Value::number(f32::from(directions[index.min(3)])),
        state,
    )
}

fn walk_movable(value: &Value, state: &ExecutionState) -> Option<DatumId> {
    let Value::Datum(datum) = value else {
        return None;
    };
    let path = state.heap().datum(*datum).ok()?.type_path();
    let movable = TypePath::parse("/atom/movable").expect("movable path is valid");
    is_subtype(state, path, &movable).then_some(*datum)
}

fn walk_target(value: Option<&Value>, state: &ExecutionState) -> Option<DatumId> {
    let Value::Datum(datum) = value? else {
        return None;
    };
    let path = state.heap().datum(*datum).ok()?.type_path();
    let atom = TypePath::parse("/atom").expect("atom path is valid");
    is_subtype(state, path, &atom).then_some(*datum)
}

fn walk_lag(value: Option<&Value>) -> u64 {
    value
        .and_then(Value::as_number)
        .filter(|lag| lag.is_finite() && *lag > 0.0)
        .map_or(1, |lag| lag.trunc() as u64)
        .max(1)
}

fn start_native_walk(
    name: &str,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    let Some(movable) = arguments
        .first()
        .and_then(|value| walk_movable(value, state))
    else {
        return Ok(Value::Null);
    };

    let (kind, lag) = match name {
        "walk" => {
            let direction = arguments.get(1).and_then(Value::as_number).unwrap_or(0.0) as i16;
            if direction == 0 {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            }
            (
                NativeWalkKind::Direction(direction),
                walk_lag(arguments.get(2)),
            )
        }
        "walk_rand" => (NativeWalkKind::Random, walk_lag(arguments.get(1))),
        "walk_towards" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (NativeWalkKind::Towards(target), walk_lag(arguments.get(2)))
        }
        "walk_to" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (
                NativeWalkKind::To {
                    target,
                    minimum: arguments.get(2).and_then(Value::as_number).unwrap_or(0.0),
                },
                walk_lag(arguments.get(3)),
            )
        }
        "walk_away" => {
            let Some(target) = walk_target(arguments.get(1), state) else {
                state.native_walks.remove(&movable);
                return Ok(Value::Null);
            };
            (
                NativeWalkKind::Away {
                    target,
                    maximum: arguments.get(2).and_then(Value::as_number).unwrap_or(5.0),
                },
                walk_lag(arguments.get(3)),
            )
        }
        _ => return Err(format!("unknown native walk procedure {name:?}")),
    };

    let sequence = state.scheduler_sequence;
    state.scheduler_sequence = state.scheduler_sequence.saturating_add(1);
    state.native_walks.insert(
        movable,
        NativeWalk {
            due_tick: state.scheduler_tick.saturating_add(lag),
            sequence,
            lag,
            kind,
        },
    );
    Ok(Value::Null)
}

fn native_walk_step(movable: DatumId, kind: &NativeWalkKind, state: &mut ExecutionState) -> bool {
    if state.heap().datum(movable).is_err() {
        return false;
    }
    let movable = Value::Datum(movable);
    match *kind {
        NativeWalkKind::Direction(direction) => {
            step_builtin(&[movable, Value::number(f32::from(direction))], state).is_ok()
        }
        NativeWalkKind::Random => step_rand_builtin(&[movable], state).is_ok(),
        NativeWalkKind::Towards(target) => {
            state.heap().datum(target).is_ok()
                && step_towards_builtin(&[movable, Value::Datum(target)], state).is_ok()
        }
        NativeWalkKind::To { target, minimum } => {
            if state.heap().datum(target).is_err() {
                return false;
            }
            let arguments = [movable, Value::Datum(target), Value::number(minimum)];
            if within_minimum_distance(&arguments, state) {
                return false;
            }
            step_to_builtin(&arguments, state).is_ok()
        }
        NativeWalkKind::Away { target, maximum } => {
            if state.heap().datum(target).is_err() {
                return false;
            }
            let arguments = [movable, Value::Datum(target), Value::number(maximum)];
            if maximum > 0.0
                && let (Some(source), Some(target)) = (
                    datum_coordinates(state, &arguments[0]),
                    datum_coordinates(state, &arguments[1]),
                )
                && (source.0 - target.0)
                    .abs()
                    .max((source.1 - target.1).abs())
                    .max((source.2 - target.2).abs())
                    > maximum
            {
                return false;
            }
            step_away_builtin(&arguments, state).is_ok()
        }
    }
}

pub(super) fn advance_native_walks(state: &mut ExecutionState) {
    let now = state.scheduler_tick;
    let mut due = state
        .native_walks
        .iter()
        .filter(|(_, walk)| walk.due_tick <= now)
        .map(|(movable, walk)| (*movable, walk.due_tick, walk.sequence))
        .collect::<Vec<_>>();
    due.sort_unstable_by_key(|(_, due_tick, sequence)| (*due_tick, *sequence));

    for (movable, _, _) in due {
        let Some(mut walk) = state.native_walks.remove(&movable) else {
            continue;
        };
        let mut active = true;
        while active && walk.due_tick <= now {
            active = native_walk_step(movable, &walk.kind, state);
            walk.due_tick = walk.due_tick.saturating_add(walk.lag);
        }
        if active {
            state.native_walks.insert(movable, walk);
        }
    }
}

fn bounds_dist_builtin(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(left) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    let Some(right) = datum_coordinates(state, &arguments[1]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    if left.2 != right.2 {
        return Ok(Value::number(f32::INFINITY));
    }
    let dimension = |value: &Value, name: &str| {
        let Value::Datum(datum) = value else {
            return 32.0;
        };
        let Ok(field) = FieldName::parse(name) else {
            return 32.0;
        };
        super::datum_field_or_initial(state, *datum, &field)
            .ok()
            .as_ref()
            .and_then(Value::as_number)
            .unwrap_or(32.0)
    };
    let horizontal = (right.0 - left.0).abs() * 32.0
        - (dimension(&arguments[0], "bound_width") + dimension(&arguments[1], "bound_width")) / 2.0;
    let vertical = (right.1 - left.1).abs() * 32.0
        - (dimension(&arguments[0], "bound_height") + dimension(&arguments[1], "bound_height"))
            / 2.0;
    Ok(Value::number(horizontal.max(vertical)))
}

pub(super) fn synchronize_moved_atom_contents(
    state: &mut ExecutionState,
    atom: DatumId,
    old_loc: Option<DatumId>,
    new_loc: Option<DatumId>,
) -> Result<(), String> {
    let contents = FieldName::parse("contents").expect("built-in contents field");
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let enclosing_area = |state: &ExecutionState, turf: DatumId| {
        if !state
            .heap
            .datum(turf)
            .is_ok_and(|datum| super::is_turf_type_path(datum.type_path()))
        {
            return None;
        }
        state
            .heap
            .datum_field(turf, &loc)
            .ok()
            .and_then(|value| match value {
                Value::Datum(area) => Some(*area),
                _ => None,
            })
    };
    let old_area = old_loc.and_then(|turf| enclosing_area(state, turf));
    let new_area = new_loc.and_then(|turf| enclosing_area(state, turf));
    let contents_list = |state: &ExecutionState, container: DatumId| {
        state
            .heap
            .datum_field(container, &contents)
            .ok()
            .and_then(|value| match value {
                Value::List(list) => Some(*list),
                _ => None,
            })
    };
    if let Some(old_loc) = old_loc
        && let Some(list) = contents_list(state, old_loc)
    {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .remove_first(&Value::Datum(atom));
    }
    if let Some(list) = new_loc.and_then(|container| contents_list(state, container)) {
        state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::Datum(atom));
    }
    if old_area != new_area {
        if let Some(list) = old_area.and_then(|area| contents_list(state, area)) {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_first(&Value::Datum(atom));
        }
        if let Some(list) = new_area.and_then(|area| contents_list(state, area)) {
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .add(Value::Datum(atom));
        }
    }
    Ok(())
}

fn get_dist(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if matches!((&arguments[0], &arguments[1]), (Value::Datum(left), Value::Datum(right)) if left == right)
    {
        return Ok(Value::number(-1.0));
    }
    let Some(left) = datum_coordinates(state, &arguments[0]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    let Some(right) = datum_coordinates(state, &arguments[1]) else {
        return Ok(Value::number(f32::INFINITY));
    };
    Ok(Value::number(
        (left.0 - right.0)
            .abs()
            .max((left.1 - right.1).abs())
            .max((left.2 - right.2).abs()),
    ))
}

pub(super) fn is_subtype(state: &ExecutionState, candidate: &TypePath, target: &TypePath) -> bool {
    if candidate == target {
        return true;
    }
    if let (Some(candidate), Some(target)) = (
        state.subtype_interval(candidate),
        state.subtype_interval(target),
    ) {
        return target.0 <= candidate.0 && candidate.1 <= target.1;
    }
    let mut current = candidate.clone();
    for _ in 0..512 {
        let parent = if let Some(parent) = state.type_parents.get(&current) {
            parent.clone()
        } else {
            fallback_parent(&current)
        };
        let Some(parent) = parent else {
            return false;
        };
        if &parent == target {
            return true;
        }
        if parent == current {
            return false;
        }
        current = parent;
    }
    false
}

fn fallback_parent(path: &TypePath) -> Option<TypePath> {
    let path = path.as_str();
    let explicit = match path {
        "/obj" | "/mob" => Some("/atom/movable"),
        "/area" | "/turf" | "/atom/movable" => Some("/atom"),
        "/atom" => Some("/datum"),
        _ => None,
    };
    if let Some(parent) = explicit {
        return TypePath::parse(parent).ok();
    }
    if let Some(index) = path.rfind('/') {
        if index > 0 {
            return TypePath::parse(&path[..index]).ok();
        }
    }
    TypePath::parse("/datum").ok()
}

fn astype(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let Some(target) = arguments.get(1) else {
        // The one-argument form gets its constraint from contextual DM type
        // information; lowering has already validated that context.
        return Ok(arguments[0].clone());
    };
    let Value::TypePath(target) = target else {
        return Ok(Value::Null);
    };
    let candidate = match &arguments[0] {
        Value::Datum(datum) => state
            .heap
            .datum(*datum)
            .map_err(|error| error.to_string())?
            .type_path(),
        Value::TypePath(path) => path,
        _ => return Ok(Value::Null),
    };
    Ok(if is_subtype(state, candidate, target) {
        arguments[0].clone()
    } else {
        Value::Null
    })
}

fn turn(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    if let Value::Datum(icon) = &arguments[0]
        && super::is_icon_datum(*icon, &state.heap)
    {
        let angle = arguments[1].as_number().unwrap_or(0.0);
        let cloned = super::clone_icon_datum(*icon, &mut state.heap)?;
        super::execute_icon_method(cloned, "Turn", &[Value::number(angle)], &mut state.heap)?;
        return Ok(Value::Datum(cloned));
    }
    if let Value::Datum(matrix) = &arguments[0]
        && super::is_matrix_datum(*matrix, &state.heap)
    {
        let angle = number(&arguments[1], "turn angle")?.to_radians();
        let mut cosine = angle.cos();
        let mut sine = angle.sin();
        if cosine.abs() < 1.0e-6 {
            cosine = 0.0;
        }
        if sine.abs() < 1.0e-6 {
            sine = 0.0;
        }
        let rotated = super::matrix_product(
            super::matrix_components(*matrix, &state.heap)?,
            [cosine, sine, 0.0, -sine, cosine, 0.0],
        );
        return super::allocate_matrix(rotated, &mut state.heap).map(Value::Datum);
    }
    const DIRECTIONS: [i32; 8] = [1, 9, 8, 10, 2, 6, 4, 5];
    let direction = number(&arguments[0], "turn direction")?.trunc() as i32;
    let angle = number(&arguments[1], "turn angle")?;
    let steps = (angle / 45.0).trunc() as i32;
    if steps == 0 {
        return Ok(Value::number(direction as f32));
    }
    let index = DIRECTIONS
        .iter()
        .position(|candidate| *candidate == direction);
    let index = index.unwrap_or_else(|| {
        let sample = super::deterministic_unit(&mut state.random_state);
        (sample * DIRECTIONS.len() as f32).floor() as usize % DIRECTIONS.len()
    });
    let rotated = (index as i32 + steps).rem_euclid(DIRECTIONS.len() as i32) as usize;
    Ok(Value::number(DIRECTIONS[rotated] as f32))
}

fn ckey(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let key = runtime_text(&arguments[0], state, "ckey")?;
    Ok(Value::text(
        key.chars()
            .filter(|character| character.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>(),
    ))
}

fn ckey_ex(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let key = runtime_text(&arguments[0], state, "ckeyEx")?;
    Ok(Value::text(
        key.chars()
            .filter(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '@' | '_' | '-')
            })
            .collect::<String>(),
    ))
}

#[derive(Clone)]
struct ListOperatorEntry {
    key: Value,
    associated: Option<Value>,
}

fn list_operator_snapshot(
    list: ListId,
    state: &ExecutionState,
) -> Result<Vec<ListOperatorEntry>, String> {
    let list = state.heap.list(list).map_err(|error| error.to_string())?;
    Ok(list
        .positions()
        .map(|(_, key)| {
            let associated = list.get_key(key).ok().cloned();
            ListOperatorEntry {
                key: key.clone(),
                associated,
            }
        })
        .collect())
}

fn add_operator_entry(
    list: ListId,
    entry: ListOperatorEntry,
    state: &mut ExecutionState,
    only_if_absent: bool,
) -> Result<(), String> {
    let preserve_existing_key = state.is_associative_list(list);
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    if (only_if_absent || preserve_existing_key) && target.contains(&entry.key) {
        return Ok(());
    }
    if let Some(associated) = entry.associated {
        target.set_key(entry.key, associated);
    } else {
        target.add(entry.key);
    }
    Ok(())
}

fn remove_all_operator_matches(
    list: ListId,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0;
    while state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .remove_last(value)
        .is_some()
    {
        removed += 1;
    }
    Ok(removed)
}

fn operator_rhs_entries(
    value: &Value,
    state: &ExecutionState,
) -> Result<Vec<ListOperatorEntry>, String> {
    if let Value::List(list) = value {
        list_operator_snapshot(*list, state)
    } else {
        Ok(vec![ListOperatorEntry {
            key: value.clone(),
            associated: None,
        }])
    }
}

pub(super) fn execute_list_binary_operator(
    operator: &str,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    match operator {
        "+" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, false)?;
            }
            Ok(Value::List(result))
        }
        "-" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let keys = operator_rhs_entries(right, state)?
                .into_iter()
                .map(|entry| entry.key)
                .collect::<Vec<_>>();
            state
                .heap
                .list_mut(result)
                .map_err(|error| error.to_string())?
                .subtract_entries(&keys)
                .map_err(|error| error.to_string())?;
            Ok(Value::List(result))
        }
        "|" => {
            let result = state.heap.allocate_list();
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            for entry in list_operator_snapshot(left, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(result, entry, state, true)?;
            }
            Ok(Value::List(result))
        }
        "&" => {
            let result = state
                .heap
                .copy_list(left)
                .map_err(|error| error.to_string())?;
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(result, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(result, &entry.key, state)?;
                }
            }
            Ok(Value::List(result))
        }
        "^" => {
            let result = state.heap.allocate_list();
            if state.is_associative_list(left) {
                state.mark_associative_list(result);
            }
            let left_entries = list_operator_snapshot(left, state)?;
            let right_entries = operator_rhs_entries(right, state)?;
            for entry in &left_entries {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry.clone(), state, true)?;
                }
            }
            for entry in right_entries {
                if !left_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    add_operator_entry(result, entry, state, true)?;
                }
            }
            Ok(Value::List(result))
        }
        _ => Err(format!("unsupported /list binary operator {operator:?}")),
    }
}

pub(super) fn execute_list_compound_operator(
    operator: CompoundAssignmentOperator,
    left: ListId,
    right: &Value,
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if !matches!(right, Value::List(_)) {
        let incremental = match operator {
            CompoundAssignmentOperator::Add => {
                state.mutate_vis_contents_scalar(left, right, true)?
            }
            CompoundAssignmentOperator::Subtract => {
                state.mutate_vis_contents_scalar(left, right, false)?
            }
            _ => None,
        };
        if incremental.is_some() {
            return Ok(Value::List(left));
        }
    }
    let visibility_before = state
        .is_visibility_list(left)
        .then(|| state.visibility_members(left))
        .transpose()?;
    match operator {
        CompoundAssignmentOperator::Add => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, false)?;
            }
        }
        CompoundAssignmentOperator::Subtract => {
            let keys = operator_rhs_entries(right, state)?
                .into_iter()
                .map(|entry| entry.key)
                .collect::<Vec<_>>();
            state
                .heap
                .list_mut(left)
                .map_err(|error| error.to_string())?
                .subtract_entries(&keys)
                .map_err(|error| error.to_string())?;
        }
        CompoundAssignmentOperator::BitOr => {
            for entry in operator_rhs_entries(right, state)? {
                add_operator_entry(left, entry, state, true)?;
            }
        }
        CompoundAssignmentOperator::BitAnd => {
            let right_entries = operator_rhs_entries(right, state)?;
            let snapshot = list_operator_snapshot(left, state)?;
            for entry in snapshot {
                if !right_entries
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                }
            }
        }
        CompoundAssignmentOperator::BitXor => {
            let right_entries = operator_rhs_entries(right, state)?;
            let original = list_operator_snapshot(left, state)?;
            for entry in right_entries {
                if original
                    .iter()
                    .any(|candidate| candidate.key.semantic_eq(&entry.key))
                {
                    remove_all_operator_matches(left, &entry.key, state)?;
                } else {
                    add_operator_entry(left, entry, state, true)?;
                }
            }
        }
        CompoundAssignmentOperator::Multiply
        | CompoundAssignmentOperator::Divide
        | CompoundAssignmentOperator::Remainder
        | CompoundAssignmentOperator::FractionalRemainder
        | CompoundAssignmentOperator::ShiftLeft
        | CompoundAssignmentOperator::ShiftRight => {
            return Err(format!(
                "operator {operator:?} is not defined for a BYOND list"
            ));
        }
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(left, &before)?;
    }
    Ok(Value::List(left))
}

pub(super) fn execute_list_method(
    name: &str,
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Option<Result<Value, String>> {
    let alist = state.is_associative_list(list);
    Some(match name {
        "Add" => list_add(list, arguments, state),
        "Copy" if alist && !arguments.is_empty() => {
            Err("alist.Copy does not accept range arguments".to_owned())
        }
        "Copy" => list_copy(list, arguments, state),
        "Cut" => list_cut(list, arguments, state),
        "Find" => list_find(list, arguments, state),
        "Insert" if alist => Err("alist.Insert is not supported".to_owned()),
        "Insert" => list_insert(list, arguments, state),
        "Join" => list_join(list, arguments, state),
        "Remove" => list_remove(list, arguments, state, false),
        "RemoveAll" => list_remove(list, arguments, state, true),
        "Splice" if alist => Err("alist.Splice is not supported".to_owned()),
        "Splice" => list_splice(list, arguments, state),
        "Swap" if alist => Err("alist.Swap is not supported".to_owned()),
        "Swap" => list_swap(list, arguments, state),
        _ => return None,
    })
}

fn list_integer(value: Option<&Value>, default: i64, context: &str) -> Result<i64, String> {
    match value {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(number)) if number.to_f32().is_finite() => {
            Ok(number.to_f32().trunc() as i64)
        }
        Some(value) => Err(format!(
            "{context} requires a numeric index, received {value}"
        )),
    }
}

fn list_boundary(value: i64, len: usize, zero_is_end: bool) -> Result<usize, String> {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 {
        if zero_is_end { limit } else { 1 }
    } else {
        value
    };
    if value < 1 || value > limit {
        return Err(format!("list index {value} is outside 1 through {limit}"));
    }
    usize::try_from(value).map_err(|error| format!("list index is not representable: {error}"))
}

fn splice_boundary(value: i64, len: usize, zero_is_end: bool) -> usize {
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let value = if value == 0 && zero_is_end {
        limit
    } else if value < 0 {
        limit.saturating_add(value)
    } else {
        value
    };
    usize::try_from(value.clamp(1, limit)).unwrap_or(usize::MAX)
}

fn flattened_list_arguments(
    arguments: &[Value],
    state: &ExecutionState,
) -> Result<Vec<Value>, String> {
    let mut values = Vec::new();
    for argument in arguments {
        if let Value::List(list) = argument {
            let snapshot = state
                .heap
                .list(*list)
                .map_err(|error| error.to_string())?
                .positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>();
            values.extend(snapshot);
        } else {
            values.push(argument.clone());
        }
    }
    Ok(values)
}

fn list_add(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err("list.Add requires at least one item".to_owned());
    }
    if let [value] = arguments
        && !matches!(value, Value::List(_))
        && state
            .mutate_vis_contents_scalar(list, value, true)?
            .is_some()
    {
        return Ok(Value::Null);
    }
    let values = flattened_list_arguments(arguments, state)?;
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    if let Some(owner) = state.contents_owner(list) {
        let owner_path = state
            .heap
            .datum(owner)
            .map_err(|error| error.to_string())?
            .type_path()
            .as_str()
            .to_owned();
        if owner_path == "/area" || owner_path.starts_with("/area/") {
            for value in values {
                let Value::Datum(turf) = value else {
                    return Err("area.contents.Add requires a turf".to_owned());
                };
                let path = state
                    .heap
                    .datum(turf)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                if path != "/turf" && !path.starts_with("/turf/") {
                    return Err("area.contents.Add requires a turf".to_owned());
                }
                move_turf_to_area(state, turf, owner)?;
            }
            return Ok(Value::Null);
        }
        if owner_path == "/turf" || owner_path.starts_with("/turf/") {
            for value in values {
                let Value::Datum(movable) = value else {
                    return Err("turf.contents.Add requires a movable atom".to_owned());
                };
                let path = state
                    .heap
                    .datum(movable)
                    .map_err(|error| error.to_string())?
                    .type_path()
                    .as_str();
                if !is_movable_path(path) {
                    return Err("turf.contents.Add requires a movable atom".to_owned());
                }
                move_movable_to_turf(state, movable, owner)?;
            }
            return Ok(Value::Null);
        }
    }
    let associative_only = state.is_associative_list(list);
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        if associative_only {
            if !target.contains(&value) {
                target.set_key(value, Value::Null);
            }
        } else {
            target.add(value);
        }
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

pub(super) fn is_movable_path(path: &str) -> bool {
    path == "/obj"
        || path.starts_with("/obj/")
        || path == "/mob"
        || path.starts_with("/mob/")
        || path == "/atom/movable"
        || path.starts_with("/atom/movable/")
}

pub(super) fn move_movable_to_turf(
    state: &mut ExecutionState,
    movable: DatumId,
    turf: DatumId,
) -> Result<(), String> {
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let old_loc = state
        .heap
        .datum_field(movable, &loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    if old_loc != Some(turf) {
        synchronize_moved_atom_contents(state, movable, old_loc, Some(turf))?;
    }
    let coordinates =
        ["x", "y", "z"].map(|name| FieldName::parse(name).expect("built-in coordinate field"));
    let values = coordinates
        .iter()
        .map(|field| {
            state
                .heap
                .datum_field(turf, field)
                .cloned()
                .unwrap_or(Value::Null)
        })
        .collect::<Vec<_>>();
    state
        .heap
        .set_datum_field(movable, loc, Value::Datum(turf))
        .map_err(|error| error.to_string())?;
    for (field, value) in coordinates.into_iter().zip(values) {
        state
            .heap
            .set_datum_field(movable, field, value)
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

pub(super) fn move_movable_to_atom(
    state: &mut ExecutionState,
    movable: DatumId,
    location: DatumId,
) -> Result<(), String> {
    let location_is_turf = state.heap.datum(location).is_ok_and(|datum| {
        let path = datum.type_path().as_str();
        path == "/turf" || path.starts_with("/turf/")
    });
    if location_is_turf {
        return move_movable_to_turf(state, movable, location);
    }

    let loc = FieldName::parse("loc").expect("built-in loc field");
    let old_loc = state
        .heap
        .datum_field(movable, &loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(datum) => Some(*datum),
            _ => None,
        });
    if old_loc != Some(location) {
        synchronize_moved_atom_contents(state, movable, old_loc, Some(location))?;
    }
    state
        .heap
        .set_datum_field(movable, loc, Value::Datum(location))
        .map_err(|error| error.to_string())?;
    Ok(())
}

fn move_turf_to_area(
    state: &mut ExecutionState,
    turf: DatumId,
    new_area: DatumId,
) -> Result<(), String> {
    let loc = FieldName::parse("loc").expect("built-in loc field");
    let contents = FieldName::parse("contents").expect("built-in contents field");
    let old_area = state
        .heap
        .datum_field(turf, &loc)
        .ok()
        .and_then(|value| match value {
            Value::Datum(area) => Some(*area),
            _ => None,
        });
    if old_area == Some(new_area) {
        return Ok(());
    }
    let contained = state
        .heap
        .datum_field(turf, &contents)
        .ok()
        .and_then(|value| match value {
            Value::List(list) => state.heap.list(*list).ok(),
            _ => None,
        })
        .map(|list| {
            list.positions()
                .map(|(_, value)| value.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if let Some(old_area) = old_area
        && let Ok(Value::List(list)) = state.heap.datum_field(old_area, &contents)
    {
        let list = *list;
        let values = std::iter::once(Value::Datum(turf)).chain(contained.iter().cloned());
        let target = state
            .heap
            .list_mut(list)
            .map_err(|error| error.to_string())?;
        for value in values {
            target.remove_first(&value);
        }
    }
    let new_contents = state.ensure_contents(new_area)?;
    {
        let target = state
            .heap
            .list_mut(new_contents)
            .map_err(|error| error.to_string())?;
        for value in std::iter::once(Value::Datum(turf)).chain(contained) {
            if !target.contains(&value) {
                target.add(value);
            }
        }
    }
    state
        .heap
        .set_datum_field(turf, loc, Value::Datum(new_area))
        .map_err(|error| error.to_string())?;
    state.note_turf_area(turf, new_area);
    Ok(())
}

fn list_copy(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Copy accepts Start and End only".to_owned());
    }
    state.refresh_vars_proxy(list)?;
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let start = list_boundary(
        list_integer(arguments.first(), 1, "list.Copy Start")?,
        len,
        false,
    )?;
    let end = list_boundary(
        list_integer(arguments.get(1), 0, "list.Copy End")?,
        len,
        true,
    )?;
    let copy = source
        .copy_range(start, end)
        .map_err(|error| error.to_string())?;
    let result = state.heap.allocate_list();
    *state
        .heap
        .list_mut(result)
        .map_err(|error| error.to_string())? = copy;
    if state.is_associative_list(list) {
        state.mark_associative_list(result);
    }
    Ok(Value::List(result))
}

fn list_cut(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 {
        return Err("list.Cut accepts Start and End only".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let raw_start = list_integer(arguments.first(), 1, "list.Cut Start")?;
    if raw_start < 0 {
        return Err("list.Cut Start cannot be negative".to_owned());
    }
    let start = list_boundary(
        raw_start.min(i64::try_from(len + 1).unwrap_or(i64::MAX)),
        len,
        false,
    )?;
    let raw_end = list_integer(arguments.get(1), 0, "list.Cut End")?;
    if raw_end < 0 {
        return Err("list.Cut End cannot be negative".to_owned());
    }
    let end = if raw_end == 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        list_boundary(raw_end, len, true)?
    };
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

fn list_find(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if arguments.is_empty() || arguments.len() > 3 {
        return Err("list.Find requires Elem and optional Start/End".to_owned());
    }
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let raw_start = list_integer(arguments.get(1), 1, "list.Find Start")
        .unwrap_or(1)
        .max(1);
    let start = usize::try_from(raw_start)
        .unwrap_or(usize::MAX)
        .min(len.saturating_add(1));
    let raw_end = list_integer(arguments.get(2), 0, "list.Find End").unwrap_or(0);
    let end = if raw_end <= 0 || raw_end > i64::try_from(len + 1).unwrap_or(i64::MAX) {
        len + 1
    } else {
        usize::try_from(raw_end).unwrap_or(len + 1)
    };
    let found = source
        .find_position(&arguments[0], start.max(1), end.max(1))
        .map_err(|error| error.to_string())?;
    Ok(Value::number(found as f32))
}

fn list_insert(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() < 2 {
        return Err("list.Insert requires Index and at least one item".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let raw = list_integer(arguments.first(), 0, "list.Insert Index")?;
    let mut index = if raw <= 0 {
        len + 1
    } else {
        usize::try_from(raw).map_err(|error| format!("list.Insert index is invalid: {error}"))?
    };
    if index > len + 1 {
        return Err(format!("list.Insert index {index} exceeds {}", len + 1));
    }
    let values = flattened_list_arguments(&arguments[1..], state)?;
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for value in values {
        target
            .insert(index, value)
            .map_err(|error| error.to_string())?;
        index += 1;
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::number(index as f32))
}

fn list_join(list: ListId, arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    if arguments.len() > 3 {
        return Err("list.Join accepts optional Glue, Start, and End".to_owned());
    }
    // BYOND declares Glue as a string but permits it to be omitted. OpenDream
    // observes the missing slot as null and TryGetValueAsString consequently
    // supplies an empty separator. Monkestation relies on this exact shape in
    // `generate_icon_key().Join()` while building human preview appearances.
    let glue = arguments.first().map_or_else(
        || Ok(String::new()),
        |value| runtime_text(value, state, "list.Join Glue"),
    )?;
    let source = state.heap.list(list).map_err(|error| error.to_string())?;
    let len = source.len();
    let limit = i64::try_from(len).unwrap_or(i64::MAX - 1).saturating_add(1);
    let mut start = list_integer(arguments.get(1), 1, "list.Join Start").unwrap_or(1);
    let mut end = list_integer(arguments.get(2), 0, "list.Join End").unwrap_or(0);
    if end <= 0 {
        end = end.saturating_add(limit);
    }
    if start < 0 {
        start = start.saturating_add(limit);
    }
    if start == 0 || start >= end {
        return Ok(Value::text(""));
    }
    let start = usize::try_from(start.max(1))
        .unwrap_or(usize::MAX)
        .min(len + 1);
    let end = usize::try_from(end.max(1))
        .unwrap_or(usize::MAX)
        .min(len + 1);
    let mut values = Vec::new();
    for index in start..end {
        values.push(runtime_text(
            source.get(index).map_err(|error| error.to_string())?,
            state,
            "list.Join item",
        )?);
    }
    Ok(Value::text(values.join(&glue)))
}

fn list_remove_once(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<usize, String> {
    let mut removed = 0usize;
    for argument in arguments {
        if matches!(argument, Value::List(candidate) if *candidate == list) {
            let len = state
                .heap
                .list(list)
                .map_err(|error| error.to_string())?
                .len();
            state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .resize(0)
                .map_err(|error| error.to_string())?;
            removed += len;
            break;
        }
        let values = flattened_list_arguments(std::slice::from_ref(argument), state)?;
        for value in values {
            if state
                .heap
                .list_mut(list)
                .map_err(|error| error.to_string())?
                .remove_last(&value)
                .is_some()
            {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

fn list_remove(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
    all: bool,
) -> Result<Value, String> {
    if arguments.is_empty() {
        return Err(if all {
            "list.RemoveAll requires at least one item"
        } else {
            "list.Remove requires at least one item"
        }
        .to_owned());
    }
    if let [value] = arguments
        && !matches!(value, Value::List(_))
        && let Some(removed) = state.mutate_vis_contents_scalar(list, value, false)?
    {
        return Ok(Value::number(f32::from(removed)));
    }
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    let result = if all {
        let mut total = 0usize;
        loop {
            let removed = list_remove_once(list, arguments, state)?;
            total += removed;
            if removed == 0 {
                break;
            }
        }
        Value::number(total as f32)
    } else {
        Value::number(f32::from(list_remove_once(list, arguments, state)? > 0))
    };
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(result)
}

fn list_splice(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() > 2 && arguments.len() < 3 {
        return Err("invalid list.Splice arguments".to_owned());
    }
    let len = state
        .heap
        .list(list)
        .map_err(|error| error.to_string())?
        .len();
    let mut start = splice_boundary(
        list_integer(arguments.first(), 1, "list.Splice Start")?,
        len,
        false,
    );
    let mut end = splice_boundary(
        list_integer(arguments.get(1), 0, "list.Splice End")?,
        len,
        true,
    );
    if end < start {
        std::mem::swap(&mut start, &mut end);
    }
    let visibility_before = state
        .is_visibility_list(list)
        .then(|| state.visibility_members(list))
        .transpose()?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .cut_range(start, end)
        .map_err(|error| error.to_string())?;
    if arguments.len() <= 2 {
        if let Some(before) = visibility_before {
            state.normalize_and_synchronize_visibility_list(list, &before)?;
        }
        return Ok(Value::Null);
    }
    let values = flattened_list_arguments(&arguments[2..], state)?;
    let index = start.min(
        state
            .heap
            .list(list)
            .map_err(|error| error.to_string())?
            .len()
            + 1,
    );
    let target = state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?;
    for (offset, value) in values.into_iter().enumerate() {
        target
            .insert(index + offset, value)
            .map_err(|error| error.to_string())?;
    }
    if let Some(before) = visibility_before {
        state.normalize_and_synchronize_visibility_list(list, &before)?;
    }
    Ok(Value::Null)
}

fn list_swap(
    list: ListId,
    arguments: &[Value],
    state: &mut ExecutionState,
) -> Result<Value, String> {
    if arguments.len() != 2 {
        return Err("list.Swap requires exactly two indices".to_owned());
    }
    let first = list_integer(arguments.first(), 0, "list.Swap Index1")?;
    let second = list_integer(arguments.get(1), 0, "list.Swap Index2")?;
    let first = usize::try_from(first).map_err(|_| "list.Swap Index1 is invalid".to_owned())?;
    let second = usize::try_from(second).map_err(|_| "list.Swap Index2 is invalid".to_owned())?;
    state
        .heap
        .list_mut(list)
        .map_err(|error| error.to_string())?
        .swap(first, second)
        .map_err(|error| error.to_string())?;
    Ok(Value::Null)
}

fn resolved_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<PathBuf, String> {
    let path = strict_text(&arguments[0], state, context)?;
    let path = PathBuf::from(path);
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?;
    let root = root
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let candidate = if path.is_absolute() {
        path
    } else {
        if path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err(format!("{context} path escapes the project root"));
        }
        root.join(path)
    };
    let existing = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|error| error.to_string())?
    } else {
        let parent = candidate
            .parent()
            .ok_or_else(|| format!("{context} path has no parent"))?;
        let parent = parent
            .canonicalize()
            .map_err(|error| format!("{context} parent directory is unavailable: {error}"))?;
        parent.join(
            candidate
                .file_name()
                .ok_or_else(|| format!("{context} path is invalid"))?,
        )
    };
    if !existing.starts_with(&root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(existing)
}

fn relaxed_resolved_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<PathBuf, String> {
    let raw = strict_text(&arguments[0], state, context)?;
    let relative = PathBuf::from(raw);
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let candidate = if relative.is_absolute() {
        relative
    } else {
        if relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        }) {
            return Err(format!("{context} path escapes the project root"));
        }
        root.join(relative)
    };
    let mut ancestor = candidate.as_path();
    let resolved_ancestor = loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break ancestor.canonicalize().map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| format!("{context} path has no existing ancestor"))?;
            }
            Err(error) => return Err(format!("{context} path is unavailable: {error}")),
        }
    };
    if !resolved_ancestor.starts_with(&root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(candidate)
}

fn prepare_write_file_path(
    arguments: &[Value],
    state: &ExecutionState,
    context: &str,
) -> Result<Option<PathBuf>, String> {
    let candidate = relaxed_resolved_file_path(arguments, state, context)?;
    let parent = candidate
        .parent()
        .ok_or_else(|| format!("{context} path has no parent"))?;
    // BYOND creates every missing destination directory for its file-writing
    // builtins. Keep I/O failures as an ordinary false result for callers
    // such as fcopy()/text2file(), while retaining containment failures as
    // runtime errors rather than allowing a symlink escape.
    if fs::create_dir_all(parent).is_err() {
        return Ok(None);
    }
    let root = state
        .project_root()
        .ok_or_else(|| format!("{context} requires a configured project root"))?
        .canonicalize()
        .map_err(|error| format!("{context} project root is unavailable: {error}"))?;
    let parent = match parent.canonicalize() {
        Ok(parent) => parent,
        Err(_) => return Ok(None),
    };
    if !parent.starts_with(root) {
        return Err(format!("{context} path escapes the project root"));
    }
    Ok(Some(
        parent.join(
            candidate
                .file_name()
                .ok_or_else(|| format!("{context} path is invalid"))?,
        ),
    ))
}

fn fexists(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let raw = strict_text(&arguments[0], state, "fexists")?;
    let relative = PathBuf::from(raw);
    let root = state
        .project_root()
        .ok_or_else(|| "fexists requires a configured project root".to_owned())?
        .canonicalize()
        .map_err(|error| format!("fexists project root is unavailable: {error}"))?;
    let invalid_relative_root = !relative.is_absolute()
        && relative
            .components()
            .any(|component| matches!(component, Component::RootDir | Component::Prefix(_)));
    if invalid_relative_root {
        return Err("fexists path escapes the project root".to_owned());
    }
    let path = if relative.is_absolute() {
        relative
    } else {
        let mut contained = PathBuf::new();
        for component in relative.components() {
            match component {
                Component::CurDir => {}
                Component::Normal(segment) => contained.push(segment),
                Component::ParentDir => {
                    if !contained.pop() {
                        return Err("fexists path escapes the project root".to_owned());
                    }
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err("fexists path escapes the project root".to_owned());
                }
            }
        }
        root.join(contained)
    };

    // A missing intermediate directory is an ordinary negative existence
    // result in BYOND. Canonicalize the nearest existing ancestor so that the
    // relaxed lookup still rejects symlink and absolute-path escapes.
    let mut ancestor = path.as_path();
    let resolved_ancestor = loop {
        match fs::symlink_metadata(ancestor) {
            Ok(_) => break ancestor.canonicalize().map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                ancestor = ancestor
                    .parent()
                    .ok_or_else(|| "fexists path has no existing ancestor".to_owned())?;
            }
            Err(error) => return Err(format!("fexists path is unavailable: {error}")),
        }
    };
    if !resolved_ancestor.starts_with(&root) {
        return Err("fexists path escapes the project root".to_owned());
    }
    Ok(Value::number(f32::from(path.exists())))
}

fn file2text(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    // A contained path may have multiple nonexistent parent components. BYOND
    // reports a missing file as null; resolve its nearest existing ancestor
    // only to enforce root/symlink containment, then let the read return
    // NotFound normally.
    let path = relaxed_resolved_file_path(arguments, state, "file2text")?;
    // BYOND resources may name a directory (notably entries returned by
    // `flist()`). A directory is not readable file content, so `file2text()`
    // returns null instead of surfacing the host OS' access-denied/is-directory
    // error. OpenDream follows the same contract by only loading resource data
    // when `File.Exists(path)` is true.
    if !path.is_file() {
        return Ok(Value::Null);
    }
    match fs::read_to_string(path) {
        Ok(text) => Ok(Value::text(text)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Value::Null),
        Err(error) => Err(format!("file2text failed: {error}")),
    }
}

fn fdel(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let raw = strict_text(&arguments[0], state, "fdel")?;
    let directory = raw.ends_with('/') || raw.ends_with('\\');
    let path = resolved_file_path(arguments, state, "fdel")?;
    let result = if directory {
        // BYOND treats a trailing slash as explicit authorization to remove
        // the entire directory tree, including nested files/directories.
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    Ok(Value::number(f32::from(result.is_ok())))
}

fn text2file(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "text2file text")?;
    let Some(path) = prepare_write_file_path(&arguments[1..], state, "text2file")? else {
        return Ok(Value::number(0.0));
    };
    // BYOND appends by default. A false optional compatibility flag requests
    // replacement, matching the existing extended arity accepted here.
    let append = arguments.get(2).is_none_or(truthy);
    let mut options = OpenOptions::new();
    options
        .create(true)
        .write(true)
        .append(append)
        .truncate(!append);
    let result = options
        .open(path)
        .and_then(|mut file| file.write_all(text.as_bytes()));
    Ok(Value::number(f32::from(result.is_ok())))
}

fn fcopy(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let mut source = arguments
        .first()
        .cloned()
        .ok_or_else(|| "fcopy source requires text, received null".to_owned())?;
    if let Value::Datum(_) = source {
        source = icon_backing_resource(&source, state, 0)?;
    }
    let source = match source {
        Value::Text(_) | Value::File(_) => {
            relaxed_resolved_file_path(&[source], state, "fcopy source")?
        }
        Value::Null => return Err("fcopy source requires text, received null".to_owned()),
        value => {
            return Err(format!(
                "fcopy source requires text, received {}",
                runtime_text(&value, state, "fcopy source")?
            ));
        }
    };
    let Some(destination) = prepare_write_file_path(&arguments[1..], state, "fcopy destination")?
    else {
        return Ok(Value::number(0.0));
    };
    Ok(Value::number(f32::from(
        fs::copy(source, destination).is_ok(),
    )))
}

fn flist(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
    let fallback = [Value::text(".")];
    let path = resolved_file_path(
        if arguments.is_empty() {
            &fallback
        } else {
            arguments
        },
        state,
        "flist",
    )?;
    let list = state.heap_mut().allocate_list();
    let mut names = fs::read_dir(path)
        .map_err(|error| format!("flist failed: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| {
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                name.push('/');
            }
            name
        })
        .collect::<Vec<_>>();
    names.sort();
    for name in names {
        state
            .heap_mut()
            .list_mut(list)
            .map_err(|error| error.to_string())?
            .add(Value::text(name));
    }
    Ok(Value::List(list))
}

pub(super) fn execute_output(
    target: &Value,
    value: &Value,
    state: &mut ExecutionState,
) -> Result<(), String> {
    if let Value::Datum(target) = target {
        let field = FieldName::parse("_dream64_output_events")
            .expect("headless output event field is valid");
        let events = match state.heap.datum_field(*target, &field) {
            Ok(Value::List(events)) => *events,
            _ => {
                let events = state.heap.allocate_list();
                state
                    .heap
                    .set_datum_field(*target, field, Value::List(events))
                    .map_err(|error| error.to_string())?;
                events
            }
        };
        state
            .heap
            .list_mut(events)
            .map_err(|error| error.to_string())?
            .add(value.clone());
        return Ok(());
    }
    let Value::Text(_) = target else {
        return Ok(());
    };
    let path = prepare_write_file_path(std::slice::from_ref(target), state, "output")?
        .ok_or_else(|| "output failed to create destination parent".to_owned())?;
    let text = runtime_text(value, state, "output value")?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("output failed: {error}"))?;
    writeln!(file, "{text}").map_err(|error| format!("output failed: {error}"))
}

fn html_encode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "html_encode")?;
    let mut output = String::with_capacity(text.len());
    for character in text.chars() {
        output.push_str(match character {
            '&' => "&amp;",
            '<' => "&lt;",
            '>' => "&gt;",
            '"' => "&quot;",
            '\'' => "&#39;",
            _ => {
                output.push(character);
                continue;
            }
        });
    }
    Ok(Value::text(output))
}

fn html_decode(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
    let text = strict_text(&arguments[0], state, "html_decode")?;
    Ok(Value::text(
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&#x27;", "'")
            .replace("&amp;", "&"),
    ))
}

fn color_byte(value: &Value, context: &str) -> Result<u8, String> {
    Ok(number(value, context)?.round().clamp(0.0, 255.0) as u8)
}

fn rgb_builtin(arguments: &[Value]) -> Result<Value, String> {
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

fn parse_hex_color(text: &str) -> Option<Vec<u8>> {
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

fn rgb2num_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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
                let lightness = (maximum + minimum) / 2.0;
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

fn gradient_builtin(arguments: &[Value], state: &mut ExecutionState) -> Result<Value, String> {
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

fn time2text_builtin(arguments: &[Value], state: &ExecutionState) -> Result<Value, String> {
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
        fs::write(root.join("icons/test.dmi"), png).unwrap();

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
            Ok(expected)
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
mod spatial_tests {
    use super::*;

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
