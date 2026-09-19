use std::collections::{BTreeMap, VecDeque};

use dm_dmf::{ControlType, UiCommand};

use crate::DEFAULT_RETAINED_OUTPUT_LINES;
use crate::layout::ClientLayout;
use crate::{normalize_resource_path, percent_decode_form, resolve_control_type};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InboundUiCommand {
    Link {
        url: String,
    },
    WinSet {
        control: String,
        parameters: String,
    },
    Output {
        control: Option<String>,
        message: String,
    },
    BrowseResource {
        name: String,
        data: Vec<u8>,
    },
    Browse {
        control: String,
        html: String,
    },
    Prompt(ClientPrompt),
    Sound(SoundUpdate),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SoundUpdate {
    pub(crate) file: Option<String>,
    pub(crate) channel: i32,
    pub(crate) repeat: bool,
    pub(crate) volume: f32,
    pub(crate) frequency: f32,
    pub(crate) pan: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ClientPromptKind {
    Text,
    Message,
    Number,
    Color,
    File,
    List,
    Alert,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ClientPrompt {
    pub(crate) id: u64,
    pub(crate) kind: ClientPromptKind,
    pub(crate) title: String,
    pub(crate) message: String,
    pub(crate) default: String,
    pub(crate) choices: Vec<String>,
    pub(crate) can_cancel: bool,
    pub(crate) edit: String,
    pub(crate) selected: usize,
}

#[derive(Default)]
pub(crate) struct UiPresentation {
    pub(crate) last_sequence: u64,
    pub(crate) browser_html: BTreeMap<String, String>,
    pub(crate) output_text: BTreeMap<String, Vec<String>>,
    pub(crate) browser_resources: BTreeMap<String, Vec<u8>>,
    pub(crate) pending_prompts: VecDeque<ClientPrompt>,
}

#[derive(Debug, PartialEq)]
pub(crate) enum BrowserUpdate {
    Link(String),
    Html { control: String, html: String },
    Resource { control: String, path: String },
    Script { control: String, script: String },
    Sound(SoundUpdate),
}

#[derive(Debug)]
pub(crate) enum UiOwnerCommand {
    MapCommit(crate::MapSnapshot),
    BrowserCommit(BrowserUpdate),
    ResourceCommit { name: String, data: Vec<u8> },
}

fn output_line_limit(ui: &dm_dmf::UiState, control: &str) -> usize {
    ui.winget(control, "lines")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|limit| *limit > 0)
        .unwrap_or(DEFAULT_RETAINED_OUTPUT_LINES)
}

fn resolve_browser_output_target(ui: &dm_dmf::UiState, target: &str) -> Option<String> {
    resolve_ui_control_type(ui, target, ControlType::Browser)
        .ok()
        .or_else(|| {
            // BYOND's embedded-browser output address is
            // `<control>.browser:<javascript function>`. The `.browser`
            // segment names the browser document, not a DMF control.
            let control = target.strip_suffix(".browser")?;
            resolve_ui_control_type(ui, control, ControlType::Browser).ok()
        })
}

fn resolve_ui_control_type(
    ui: &dm_dmf::UiState,
    address: &str,
    expected: ControlType,
) -> Result<String, String> {
    if address.contains('.')
        && ui
            .winexists_type(address)
            .eq_ignore_ascii_case(expected_name(expected))
    {
        return Ok(address.to_owned());
    }
    let mut window_ids = ui
        .tree()
        .windows
        .iter()
        .map(|window| window.id.clone())
        .collect::<Vec<_>>();
    window_ids.extend(ui.cloned_window_ids());
    let matches = window_ids
        .into_iter()
        .flat_map(|window| {
            ui.section_control_ids(&window)
                .unwrap_or_default()
                .into_iter()
                .map(move |control| (window.clone(), control))
        })
        .filter_map(|(window, control)| {
            let qualified = format!("{window}.{control}");
            (control == address
                && ui
                    .winexists_type(&qualified)
                    .eq_ignore_ascii_case(expected_name(expected)))
            .then_some(qualified)
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(format!("{address} is not a {expected:?} control")),
        _ => Err(format!("ambiguous {expected:?} control {address}")),
    }
}

const fn expected_name(expected: ControlType) -> &'static str {
    match expected {
        ControlType::Main => "MAIN",
        ControlType::Map => "MAP",
        ControlType::Browser => "BROWSER",
        ControlType::Input => "INPUT",
        ControlType::Output => "OUTPUT",
        ControlType::Label => "LABEL",
        ControlType::Button => "BUTTON",
        ControlType::Unknown => "UNKNOWN",
    }
}

fn browser_output_script(function: &str, value: &str) -> String {
    let arguments = value
        .split('&')
        .map(|part| percent_decode_form(part).unwrap_or_else(|_| part.to_owned()))
        .map(|part| serde_json::to_string(&part).expect("browser output text serializes as JSON"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{function}({arguments});")
}

impl UiPresentation {
    pub(crate) fn apply(
        &mut self,
        sequence: u64,
        command: InboundUiCommand,
        session: &mut dm_dmf::ClientSession,
        layout: &mut ClientLayout,
    ) -> Result<Option<BrowserUpdate>, String> {
        if sequence <= self.last_sequence {
            return Ok(None);
        }
        let browser_update = match command {
            InboundUiCommand::Link { url } => Some(BrowserUpdate::Link(url)),
            InboundUiCommand::WinSet {
                control,
                parameters,
            } => {
                session
                    .apply_command(UiCommand::WinSet {
                        control,
                        parameters,
                    })
                    .map_err(|error| format!("{error:?}"))?;
                // Winset updates authoritative UiState overrides. Re-resolve
                // native presentation geometry immediately so subsequent UI
                // events and the next redraw observe the same property state.
                layout.refresh_from_ui(session.ui());
                None
            }
            InboundUiCommand::Output { control, message } => {
                let control = control
                    .or_else(|| layout.output_controls.first().cloned())
                    .ok_or("skin has no OUTPUT control")?;
                if let Some((target, function)) = control.split_once(':')
                    && let Some(control) = resolve_browser_output_target(session.ui(), target)
                {
                    Some(BrowserUpdate::Script {
                        control,
                        script: browser_output_script(function, &message),
                    })
                } else {
                    match resolve_control_type(session.ui().tree(), &control, ControlType::Output) {
                        Ok(control) => {
                            let retained_lines = output_line_limit(session.ui(), &control);
                            let lines = self.output_text.entry(control).or_default();
                            lines.extend(message.lines().map(str::to_owned));
                            if lines.len() > retained_lines {
                                lines.drain(..lines.len() - retained_lines);
                            }
                            None
                        }
                        Err(output_error) => {
                            if let Ok(control) = resolve_ui_control_type(
                                session.ui(),
                                &control,
                                ControlType::Browser,
                            ) {
                                let path =
                                    normalize_resource_path("", &message).ok_or_else(|| {
                                        format!("invalid browser output resource path {message:?}")
                                    })?;
                                Some(BrowserUpdate::Resource { control, path })
                            } else {
                                return Err(output_error);
                            }
                        }
                    }
                }
            }
            InboundUiCommand::BrowseResource { name, data } => {
                let name = normalize_resource_path("", &name)
                    .ok_or_else(|| format!("invalid browser resource path {name:?}"))?;
                self.browser_resources.insert(name, data);
                None
            }
            InboundUiCommand::Browse { control, html } => {
                // BYOND calls this selector `window`: a skin control may be
                // qualified or unqualified. An unknown value creates an
                // implicit top-level browser window from the skin popup
                // template, which TGUI immediately probes with winexists().
                if !control.is_empty() && !session.ui().winexists(&control) {
                    session
                        .ensure_browser_window(&control)
                        .map_err(|error| format!("browse window creation failed: {error:?}"))?;
                }
                let resolved =
                    resolve_ui_control_type(session.ui(), &control, ControlType::Browser).ok();
                let key = resolved.clone().unwrap_or(control);
                self.browser_html.insert(key, html.clone());
                resolved.map(|control| BrowserUpdate::Html { control, html })
            }
            InboundUiCommand::Prompt(mut prompt) => {
                prompt.edit.clone_from(&prompt.default);
                prompt.selected = prompt
                    .choices
                    .iter()
                    .position(|choice| choice == &prompt.default)
                    .unwrap_or(0);
                self.pending_prompts.push_back(prompt);
                None
            }
            InboundUiCommand::Sound(sound) => Some(BrowserUpdate::Sound(sound)),
        };
        self.last_sequence = sequence;
        Ok(browser_update)
    }
}
