//! Command-line surface for `dream64-server`: the subcommand set.

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
