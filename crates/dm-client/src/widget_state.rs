use std::collections::BTreeMap;

use dm_dmf::UiState;

use crate::layout::{ClientLayout, dmf_truthy};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct InputState {
    pub(crate) command: String,
    pub(crate) text: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct ButtonState {
    pub(crate) text: String,
    pub(crate) command: String,
    pub(crate) checked: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct LabelState {
    pub(crate) text: String,
}

pub(crate) fn input_states_from_ui(
    ui: &UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, InputState> {
    layout
        .input_rects
        .keys()
        .map(|address| {
            let command = ui.winget(address, "command").unwrap_or_default();
            let text = command
                .strip_prefix('!')
                .map_or_else(String::new, str::to_owned);
            (address.clone(), InputState { command, text })
        })
        .collect()
}

pub(crate) fn take_input_submission(state: &mut InputState) -> String {
    let command = if state.command.starts_with('!') {
        state.text.clone()
    } else {
        format!("{}{}", state.command, state.text)
    };
    state.text = state
        .command
        .strip_prefix('!')
        .map_or_else(String::new, str::to_owned);
    command
}

pub(crate) fn button_states_from_ui(
    ui: &UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, ButtonState> {
    layout
        .button_rects
        .keys()
        .map(|address| {
            (
                address.clone(),
                ButtonState {
                    text: ui.winget(address, "text").unwrap_or_default(),
                    command: ui.winget(address, "command").unwrap_or_default(),
                    checked: ui
                        .winget(address, "is-checked")
                        .is_ok_and(|value| dmf_truthy(&value)),
                },
            )
        })
        .collect()
}

pub(crate) fn label_states_from_ui(
    ui: &UiState,
    layout: &ClientLayout,
) -> BTreeMap<String, LabelState> {
    layout
        .label_rects
        .keys()
        .map(|address| {
            (
                address.clone(),
                LabelState {
                    text: ui.winget(address, "text").unwrap_or_default(),
                },
            )
        })
        .collect()
}
