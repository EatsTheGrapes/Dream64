//! Command-line surface for `dream64-server`: the subcommand set and the
//! ready-world mode selected from process environment variables.

use std::env;
use std::ffi::OsString;
use std::path::PathBuf;

use dm_project::ProjectDefines;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Command {
    Compile,
    Plan,
    Boot,
    Sweep,
    SweepClosure,
    LobbyPreflight,
    LobbyPreview,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProductionReadyWorldIdentity {
    pub(crate) random_seed: u64,
    pub(crate) deployment_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ReadyWorldMode {
    Disabled,
    Development,
    Prewarm(ProductionReadyWorldIdentity),
    Activate(ProductionReadyWorldIdentity),
}

impl ReadyWorldMode {
    pub(crate) const fn production_identity(&self) -> Option<&ProductionReadyWorldIdentity> {
        match self {
            Self::Prewarm(identity) | Self::Activate(identity) => Some(identity),
            Self::Disabled | Self::Development => None,
        }
    }

    pub(crate) const fn writes_snapshot(&self) -> bool {
        matches!(self, Self::Development | Self::Prewarm(_))
    }
}

pub(crate) fn parse_ready_world_mode(
    prewarm: bool,
    activate: bool,
    development: bool,
    disabled: bool,
    random_seed: Option<&str>,
    deployment_id: Option<&str>,
) -> Result<ReadyWorldMode, String> {
    if disabled {
        return Ok(ReadyWorldMode::Disabled);
    }
    if prewarm && activate {
        return Err(
            "DREAM64_PREWARM_READY_WORLD and DREAM64_ACTIVATE_READY_WORLD are mutually exclusive"
                .to_owned(),
        );
    }
    if !prewarm && !activate {
        return Ok(if development {
            ReadyWorldMode::Development
        } else {
            ReadyWorldMode::Disabled
        });
    }
    let random_seed = random_seed
        .ok_or_else(|| "production ready-world mode requires DREAM64_RANDOM_SEED".to_owned())?
        .parse::<u64>()
        .map_err(|_| "DREAM64_RANDOM_SEED must be a nonzero u64".to_owned())?;
    if random_seed == 0 {
        return Err("DREAM64_RANDOM_SEED must be a nonzero u64".to_owned());
    }
    let deployment_id = deployment_id
        .map(str::trim)
        .filter(|identity| !identity.is_empty())
        .ok_or_else(|| "production ready-world mode requires DREAM64_DEPLOYMENT_ID".to_owned())?
        .to_owned();
    let identity = ProductionReadyWorldIdentity {
        random_seed,
        deployment_id,
    };
    Ok(if prewarm {
        ReadyWorldMode::Prewarm(identity)
    } else {
        ReadyWorldMode::Activate(identity)
    })
}

pub(crate) fn ready_world_mode_from_environment() -> Result<ReadyWorldMode, String> {
    let enabled = |name| env::var(name).is_ok_and(|value| value.trim() == "1");
    parse_ready_world_mode(
        enabled("DREAM64_PREWARM_READY_WORLD"),
        enabled("DREAM64_ACTIVATE_READY_WORLD"),
        env::var_os("DREAM64_ENABLE_READY_WORLD_CACHE").is_some(),
        env::var_os("DREAM64_DISABLE_READY_CACHE").is_some(),
        env::var("DREAM64_RANDOM_SEED").ok().as_deref(),
        env::var("DREAM64_DEPLOYMENT_ID").ok().as_deref(),
    )
}

/// Parses the trailing `dream64-server` arguments after the subcommand and
/// `<world.dme>`: at most one optional `map.dmm` plus any number of BYOND/gcc
/// style `-D NAME[=VALUE]` / `--define NAME[=VALUE]` preprocessor defines.
///
/// # Errors
///
/// Returns a message when a define flag is missing its value, a define name is
/// invalid, or an unexpected extra positional argument is present.
pub(crate) fn parse_trailing_arguments(
    arguments: impl IntoIterator<Item = OsString>,
) -> Result<(Option<PathBuf>, ProjectDefines), String> {
    let mut map = None;
    let mut defines = ProjectDefines::new();
    let mut arguments = arguments.into_iter();
    while let Some(argument) = arguments.next() {
        let text = argument.to_string_lossy();
        if text == "-D" || text == "--define" {
            let spec = arguments
                .next()
                .ok_or_else(|| "-D/--define requires a NAME[=VALUE] argument".to_owned())?;
            defines
                .push_spec(&spec.to_string_lossy())
                .map_err(|error| error.to_string())?;
        } else if let Some(spec) = text.strip_prefix("--define=") {
            defines.push_spec(spec).map_err(|error| error.to_string())?;
        } else if text.starts_with("-D") && text.len() > 2 {
            defines
                .push_spec(&text["-D".len()..])
                .map_err(|error| error.to_string())?;
        } else if map.is_none() && !text.starts_with('-') {
            map = Some(PathBuf::from(&argument));
        } else {
            return Err(format!("unexpected argument {text:?}"));
        }
    }
    Ok((map, defines))
}

pub(crate) const fn progress_label(command: Command) -> &'static str {
    match command {
        Command::Compile => "compile-progress",
        Command::LobbyPreflight => "lobby-preflight-progress",
        Command::LobbyPreview => "lobby-preview-progress",
        _ => "boot-progress",
    }
}

#[cfg(test)]
mod tests {
    use super::{ProductionReadyWorldIdentity, ReadyWorldMode, parse_ready_world_mode};

    #[test]
    fn production_ready_world_modes_require_exclusive_complete_identity() {
        assert!(parse_ready_world_mode(true, true, false, false, Some("7"), Some("blue")).is_err());
        assert!(parse_ready_world_mode(true, false, false, false, None, Some("blue")).is_err());
        assert!(
            parse_ready_world_mode(true, false, false, false, Some("0"), Some("blue")).is_err()
        );
        assert!(parse_ready_world_mode(true, false, false, false, Some("7"), Some(" ")).is_err());
        assert_eq!(
            parse_ready_world_mode(true, false, true, false, Some("7"), Some(" blue ")).unwrap(),
            ReadyWorldMode::Prewarm(ProductionReadyWorldIdentity {
                random_seed: 7,
                deployment_id: "blue".to_owned(),
            })
        );
        assert_eq!(
            parse_ready_world_mode(false, true, false, false, Some("9"), Some("green")).unwrap(),
            ReadyWorldMode::Activate(ProductionReadyWorldIdentity {
                random_seed: 9,
                deployment_id: "green".to_owned(),
            })
        );
        assert_eq!(
            parse_ready_world_mode(true, true, true, true, None, None).unwrap(),
            ReadyWorldMode::Disabled,
            "the explicit disable switch overrides every cache mode"
        );
        assert_eq!(
            parse_ready_world_mode(false, false, true, false, None, None).unwrap(),
            ReadyWorldMode::Development
        );
    }
}
