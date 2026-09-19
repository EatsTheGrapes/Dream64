use std::collections::BTreeMap;

use dm_dmf::{ControlTree, ControlType, PixelRect, UiState};

pub(crate) fn dmf_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "true" | "yes" | "1"
    )
}

pub(crate) fn dmf_false(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "false" | "no" | "0"
    )
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ClientLayout {
    pub(crate) window_width: u32,
    pub(crate) window_height: u32,
    pub(crate) map: PixelRect,
    pub(crate) map_tile_size: u32,
    pub(crate) map_zoom: f32,
    pub(crate) map_zoom_mode: String,
    pub(crate) map_letterbox: bool,
    pub(crate) browser: Option<PixelRect>,
    pub(crate) browser_control: Option<String>,
    pub(crate) output_controls: Vec<String>,
    pub(crate) output_rects: BTreeMap<String, PixelRect>,
    pub(crate) input_rects: BTreeMap<String, PixelRect>,
    pub(crate) button_rects: BTreeMap<String, PixelRect>,
    pub(crate) label_rects: BTreeMap<String, PixelRect>,
}

impl ClientLayout {
    pub(crate) fn from_tree(tree: &ControlTree) -> Self {
        let controls = tree.windows.iter().flat_map(|window| &window.controls);
        let main = controls
            .clone()
            .find(|control| {
                control.control_type == ControlType::Main
                    && control.property("is-default").is_some_and(dmf_truthy)
                    && !control.property("is-visible").is_some_and(dmf_false)
            })
            .or_else(|| {
                controls
                    .clone()
                    .find(|control| control.control_type == ControlType::Main)
            });
        let main_rect = main
            .and_then(dm_dmf::ControlNode::pixel_rect)
            .unwrap_or(PixelRect {
                x: 0,
                y: 0,
                width: 1_024,
                height: 768,
            });
        let map = controls
            .clone()
            .find(|control| control.control_type == ControlType::Map)
            .and_then(dm_dmf::ControlNode::pixel_rect)
            .unwrap_or(PixelRect {
                x: 16,
                y: 74,
                width: 640,
                height: 528,
            });
        let browser_node = controls
            .clone()
            .filter(|control| control.control_type == ControlType::Browser)
            .filter(|control| !control.property("is-visible").is_some_and(dmf_false))
            .find(|control| {
                control.id.as_deref() == Some("browseroutput")
                    || control.property("is-default").is_some_and(dmf_truthy)
            })
            .or_else(|| {
                controls
                    .clone()
                    .find(|control| control.control_type == ControlType::Browser)
            });
        let browser = browser_node.and_then(dm_dmf::ControlNode::pixel_rect);
        let browser_control = browser_node.and_then(|node| qualified_control(tree, node));
        let output_controls = controls
            .filter(|control| control.control_type == ControlType::Output)
            .filter_map(|node| qualified_control(tree, node))
            .collect::<Vec<_>>();
        let output_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Output)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let input_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Input)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let button_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Button)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let label_rects = tree
            .windows
            .iter()
            .flat_map(|window| &window.controls)
            .filter(|control| control.control_type == ControlType::Label)
            .filter_map(|node| Some((qualified_control(tree, node)?, node.pixel_rect()?)))
            .collect();
        let mut layout = Self {
            window_width: main_rect.width,
            window_height: main_rect.height,
            map,
            map_tile_size: 32,
            map_zoom: 0.0,
            map_zoom_mode: "normal".to_owned(),
            map_letterbox: true,
            browser,
            browser_control,
            output_controls,
            output_rects,
            input_rects,
            button_rects,
            label_rects,
        };
        layout.apply_resolved_panes(&UiState::new(tree.clone()));
        layout
    }

    pub(crate) fn refresh_from_ui(&mut self, ui: &UiState) {
        self.apply_resolved_panes(ui);
    }

    fn apply_resolved_panes(&mut self, ui: &UiState) {
        self.apply_resolved_panes_in(ui, None);
    }

    pub(crate) fn apply_resolved_panes_in(&mut self, ui: &UiState, viewport: Option<(u32, u32)>) {
        let resolved = resolve_pane_layout_in(ui, viewport);
        self.window_width = resolved.root.width;
        self.window_height = resolved.root.height;
        if let Some((address, rect)) = resolved
            .controls
            .iter()
            .find(|(address, _)| control_has_type(ui.tree(), address, ControlType::Map))
        {
            self.map = *rect;
            self.map_tile_size = ui
                .winget(address, "tile-size")
                .ok()
                .filter(|value| !value.is_empty())
                .or_else(|| ui.winget(address, "icon-size").ok())
                .and_then(|value| value.parse().ok())
                .filter(|value| *value > 0)
                .unwrap_or(32);
            self.map_zoom = ui
                .winget(address, "zoom")
                .ok()
                .and_then(|value| value.parse().ok())
                .filter(|value: &f32| value.is_finite() && *value >= 0.0)
                .unwrap_or(0.0);
            self.map_zoom_mode = ui
                .winget(address, "zoom-mode")
                .ok()
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "normal".to_owned());
            self.map_letterbox = ui
                .winget(address, "letterbox")
                .ok()
                .filter(|value| !value.is_empty())
                .is_none_or(|value| !dmf_false(&value));
        }
        let browser = resolved
            .controls
            .iter()
            .find(|(address, _)| {
                control_has_type(ui.tree(), address, ControlType::Browser)
                    && (address.ends_with(".browseroutput")
                        || ui
                            .winget(address, "is-default")
                            .is_ok_and(|value| dmf_truthy(&value)))
            })
            .or_else(|| {
                resolved
                    .controls
                    .iter()
                    .find(|(address, _)| control_has_type(ui.tree(), address, ControlType::Browser))
            });
        self.browser_control = browser.map(|(address, _)| address.clone());
        self.browser = browser.map(|(_, rect)| *rect);
        self.output_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Output))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.input_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Input))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.button_rects = resolved
            .controls
            .iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Button))
            .map(|(address, rect)| (address.clone(), *rect))
            .collect();
        self.label_rects = resolved
            .controls
            .into_iter()
            .filter(|(address, _)| control_has_type(ui.tree(), address, ControlType::Label))
            .collect();
    }
}

pub(crate) struct ResolvedPaneLayout {
    pub(crate) root: PixelRect,
    pub(crate) controls: BTreeMap<String, PixelRect>,
}

pub(crate) fn resolve_pane_layout_in(
    ui: &UiState,
    viewport_size: Option<(u32, u32)>,
) -> ResolvedPaneLayout {
    let tree = ui.tree();
    let root_window = tree
        .windows
        .iter()
        .find(|window| {
            window.controls.iter().any(|control| {
                control.control_type == ControlType::Main
                    && ui
                        .winget(
                            &format!("{}.{}", window.id, control.id.as_deref().unwrap_or("")),
                            "is-default",
                        )
                        .is_ok_and(|value| dmf_truthy(&value))
            })
        })
        .or_else(|| tree.windows.first());
    let root = root_window
        .and_then(|window| {
            window
                .controls
                .iter()
                .find(|control| control.control_type == ControlType::Main)
        })
        .and_then(dm_dmf::ControlNode::pixel_rect)
        .unwrap_or(PixelRect {
            x: 0,
            y: 0,
            width: 1_024,
            height: 768,
        });
    let viewport = PixelRect {
        x: 0,
        y: 0,
        width: viewport_size.map_or(root.width, |size| size.0),
        height: viewport_size.map_or(root.height, |size| size.1),
    };
    let mut controls = BTreeMap::new();
    let mut active = Vec::new();
    if let Some(window) = root_window {
        resolve_pane_window(ui, &window.id, viewport, &mut controls, &mut active);
    }
    ResolvedPaneLayout {
        root: viewport,
        controls,
    }
}

fn resolve_pane_window(
    ui: &UiState,
    window_id: &str,
    viewport: PixelRect,
    resolved: &mut BTreeMap<String, PixelRect>,
    active: &mut Vec<String>,
) {
    if active.iter().any(|entry| entry == window_id) {
        return;
    }
    let Some(window) = ui
        .tree()
        .windows
        .iter()
        .find(|window| window.id == window_id)
    else {
        return;
    };
    active.push(window_id.to_owned());
    let source = window
        .controls
        .iter()
        .find(|control| control.control_type == ControlType::Main)
        .and_then(dm_dmf::ControlNode::pixel_rect)
        .unwrap_or(PixelRect {
            x: 0,
            y: 0,
            width: viewport.width,
            height: viewport.height,
        });
    for control in &window.controls {
        if control.control_type == ControlType::Main {
            continue;
        }
        let Some(id) = control.id.as_deref() else {
            continue;
        };
        let address = format!("{window_id}.{id}");
        if ui
            .winget(&address, "is-visible")
            .is_ok_and(|value| dmf_false(&value))
        {
            continue;
        }
        let Some(local) = effective_pixel_rect(ui, &address, control.pixel_rect()) else {
            continue;
        };
        let rect = anchored_rect(ui, &address, local, source, viewport);
        if control
            .property("type")
            .is_some_and(|kind| kind.eq_ignore_ascii_case("CHILD"))
        {
            let left = ui.winget(&address, "left").unwrap_or_default();
            let right = ui.winget(&address, "right").unwrap_or_default();
            let vertical_value = ui.winget(&address, "is-vert").unwrap_or_default();
            let vertical = dmf_truthy(&vertical_value);
            let split = ui
                .winget(&address, "splitter")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(50)
                .min(100);
            let (first, second) = split_rect(rect, vertical, split, right.is_empty());
            if !left.is_empty() {
                resolve_pane_window(ui, &left, first, resolved, active);
            }
            if !right.is_empty() {
                resolve_pane_window(ui, &right, second, resolved, active);
            }
        } else {
            resolved.insert(address, rect);
        }
    }
    active.pop();
}

fn anchored_rect(
    ui: &UiState,
    address: &str,
    base: PixelRect,
    source: PixelRect,
    target: PixelRect,
) -> PixelRect {
    let first = ui
        .winget(address, "anchor1")
        .ok()
        .and_then(|value| parse_anchor(&value));
    let second = ui
        .winget(address, "anchor2")
        .ok()
        .and_then(|value| parse_anchor(&value));
    let dx = i64::from(target.width) - i64::from(source.width);
    let dy = i64::from(target.height) - i64::from(source.height);
    let left = i64::from(base.x) + anchor_delta(dx, first.map(|value| value.0));
    let top = i64::from(base.y) + anchor_delta(dy, first.map(|value| value.1));
    let right = i64::from(base.x + base.width) + anchor_delta(dx, second.map(|value| value.0));
    let bottom = i64::from(base.y + base.height) + anchor_delta(dy, second.map(|value| value.1));
    PixelRect {
        x: target
            .x
            .saturating_add(u32::try_from(left.max(0)).unwrap_or(0)),
        y: target
            .y
            .saturating_add(u32::try_from(top.max(0)).unwrap_or(0)),
        width: u32::try_from((right - left).max(0)).unwrap_or(0),
        height: u32::try_from((bottom - top).max(0)).unwrap_or(0),
    }
}

fn parse_anchor(value: &str) -> Option<(i32, i32)> {
    let (x, y) = value.split_once(',')?;
    Some((x.trim().parse().ok()?, y.trim().parse().ok()?))
}

fn anchor_delta(delta: i64, anchor: Option<i32>) -> i64 {
    anchor
        .filter(|value| *value >= 0)
        .map_or(0, |value| delta * i64::from(value) / 100)
}

fn split_rect(
    rect: PixelRect,
    vertical: bool,
    percent: u32,
    only_one: bool,
) -> (PixelRect, PixelRect) {
    if only_one {
        return (
            rect,
            PixelRect {
                x: rect.x + rect.width,
                y: rect.y,
                width: 0,
                height: rect.height,
            },
        );
    }
    if vertical {
        let first = rect.width.saturating_mul(percent) / 100;
        (
            PixelRect {
                width: first,
                ..rect
            },
            PixelRect {
                x: rect.x + first,
                width: rect.width - first,
                ..rect
            },
        )
    } else {
        let first = rect.height.saturating_mul(percent) / 100;
        (
            PixelRect {
                height: first,
                ..rect
            },
            PixelRect {
                y: rect.y + first,
                height: rect.height - first,
                ..rect
            },
        )
    }
}

pub(crate) fn control_has_type(tree: &ControlTree, address: &str, expected: ControlType) -> bool {
    address
        .split_once('.')
        .and_then(|(window, control)| tree.control(window, control))
        .is_some_and(|control| control.control_type == expected)
}

fn effective_pixel_rect(
    ui: &UiState,
    address: &str,
    fallback: Option<PixelRect>,
) -> Option<PixelRect> {
    let mut rect = fallback?;
    if let Ok(position) = ui.winget(address, "pos")
        && let Some((x, y)) = parse_pair(&position, ',')
    {
        rect.x = x;
        rect.y = y;
    }
    if let Ok(size) = ui.winget(address, "size")
        && let Some((width, height)) = parse_pair(&size, 'x')
    {
        rect.width = width;
        rect.height = height;
    }
    Some(rect)
}

pub(crate) fn parse_pair(value: &str, separator: char) -> Option<(u32, u32)> {
    let (left, right) = value.trim().split_once(separator)?;
    Some((left.trim().parse().ok()?, right.trim().parse().ok()?))
}

fn qualified_control(tree: &ControlTree, needle: &dm_dmf::ControlNode) -> Option<String> {
    let id = needle.id.as_deref()?;
    tree.windows.iter().find_map(|window| {
        window
            .controls
            .iter()
            .any(|node| std::ptr::eq(node, needle))
            .then(|| format!("{}.{}", window.id, id))
    })
}
