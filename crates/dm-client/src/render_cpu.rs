use dm_dmf::PixelRect;
use font8x8::UnicodeFonts;

use crate::sprite::Appearance;
use crate::ui_commands::{ClientPrompt, ClientPromptKind};

pub(crate) fn pixel_rect_contains(rect: PixelRect, point: (u32, u32)) -> bool {
    point.0 >= rect.x
        && point.1 >= rect.y
        && point.0 < rect.x.saturating_add(rect.width)
        && point.1 < rect.y.saturating_add(rect.height)
}

pub(crate) fn draw_panel(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    panel_width: usize,
    panel_height: usize,
    color: u32,
) {
    let end_y = y.saturating_add(panel_height).min(height);
    let end_x = x.saturating_add(panel_width).min(width);
    for row in y.min(height)..end_y {
        let start = row * width + x.min(width);
        let end = row * width + end_x;
        buffer[start..end].fill(color);
    }
}

pub(crate) fn draw_output_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    lines: &[String],
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xffffffff,
    );
    const GLYPH_ADVANCE: usize = 8;
    const LINE_ADVANCE: usize = 9;
    let columns = control_width.saturating_sub(4) / GLYPH_ADVANCE;
    let visible_lines = control_height.saturating_sub(4) / LINE_ADVANCE;
    if columns == 0 || visible_lines == 0 {
        return;
    }
    let rendered = lines
        .iter()
        .flat_map(|line| wrap_output_line(&strip_output_markup(line), columns))
        .collect::<Vec<_>>();
    let first = rendered.len().saturating_sub(visible_lines);
    for (row, line) in rendered[first..].iter().enumerate() {
        draw_bitmap_text(
            buffer,
            width,
            height,
            x + 2,
            y + 2 + row * LINE_ADVANCE,
            line,
            0xff000000,
            (x + control_width, y + control_height),
        );
    }
}

pub(crate) fn draw_input_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
    focused: bool,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        if focused { 0xff0078d7 } else { 0xff7a7a7a },
    );
    if control_width > 2 && control_height > 2 {
        draw_panel(
            buffer,
            width,
            height,
            x + 1,
            y + 1,
            control_width - 2,
            control_height - 2,
            0xffffffff,
        );
    }
    let columns = control_width.saturating_sub(6) / 8;
    let visible = text
        .chars()
        .rev()
        .take(columns)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 3,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xff000000,
        (x + control_width.saturating_sub(2), y + control_height),
    );
    if focused {
        let cursor_x = x + 3 + visible.chars().count() * 8;
        draw_panel(
            buffer,
            width,
            height,
            cursor_x.min(x + control_width.saturating_sub(2)),
            y + 3,
            1,
            control_height.saturating_sub(6),
            0xff000000,
        );
    }
}

pub(crate) fn draw_boot_status(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    status: &str,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    if control_width < 32 || control_height < 32 {
        return;
    }
    let columns = control_width.saturating_sub(48) / 8;
    let visible = status.chars().take(columns).collect::<String>();
    let text_x = x + 24;
    let text_y = y + control_height / 2;
    draw_bitmap_text(
        buffer,
        width,
        height,
        text_x,
        text_y.saturating_sub(16),
        "DREAM64 / MONKESTATION",
        0xff4fd8ff,
        (x + control_width, y + control_height),
    );
    draw_bitmap_text(
        buffer,
        width,
        height,
        text_x,
        text_y,
        &visible,
        0xffffffff,
        (x + control_width, y + control_height),
    );
}

pub(crate) fn draw_button_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
    checked: bool,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xff767676,
    );
    if control_width > 2 && control_height > 2 {
        draw_panel(
            buffer,
            width,
            height,
            x + 1,
            y + 1,
            control_width - 2,
            control_height - 2,
            if checked { 0xffc8c8c8 } else { 0xffe1e1e1 },
        );
    }
    let columns = control_width.saturating_sub(4) / 8;
    let visible = text.chars().take(columns).collect::<String>();
    let text_width = visible.chars().count() * 8;
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + control_width.saturating_sub(text_width) / 2,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xff000000,
        (x + control_width, y + control_height),
    );
}

pub(crate) fn draw_label_control(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    rect: PixelRect,
    text: &str,
) {
    let x = usize::try_from(rect.x).unwrap_or(0).min(width);
    let y = usize::try_from(rect.y).unwrap_or(0).min(height);
    let control_width = usize::try_from(rect.width)
        .unwrap_or(0)
        .min(width.saturating_sub(x));
    let control_height = usize::try_from(rect.height)
        .unwrap_or(0)
        .min(height.saturating_sub(y));
    draw_panel(
        buffer,
        width,
        height,
        x,
        y,
        control_width,
        control_height,
        0xff222222,
    );
    let visible = text
        .chars()
        .take(control_width.saturating_sub(4) / 8)
        .collect::<String>();
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 2,
        y + control_height.saturating_sub(8) / 2,
        &visible,
        0xffffffff,
        (x + control_width, y + control_height),
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PromptHit {
    Choice(usize),
    Accept,
    Cancel,
}

pub(crate) fn client_prompt_rect(width: usize, height: usize) -> PixelRect {
    let prompt_width = width.saturating_sub(32).min(560);
    let prompt_height = height.saturating_sub(32).min(260);
    PixelRect {
        x: u32::try_from(width.saturating_sub(prompt_width) / 2).unwrap_or(0),
        y: u32::try_from(height.saturating_sub(prompt_height) / 2).unwrap_or(0),
        width: u32::try_from(prompt_width).unwrap_or(0),
        height: u32::try_from(prompt_height).unwrap_or(0),
    }
}

pub(crate) fn client_prompt_hit(
    width: usize,
    height: usize,
    prompt: &ClientPrompt,
    point: (u32, u32),
) -> Option<PromptHit> {
    let rect = client_prompt_rect(width, height);
    let x = point.0;
    let y = point.1;
    if x < rect.x
        || y < rect.y
        || x >= rect.x.saturating_add(rect.width)
        || y >= rect.y.saturating_add(rect.height)
    {
        return None;
    }
    let local_x = x - rect.x;
    let local_y = y - rect.y;
    if !prompt.choices.is_empty() && (78..194).contains(&local_y) {
        let index = usize::try_from((local_y - 78) / 24).ok()?;
        if index < prompt.choices.len().min(5) {
            return Some(PromptHit::Choice(index));
        }
    }
    let footer_y = rect.height.saturating_sub(42);
    if local_y >= footer_y && local_y < footer_y + 28 {
        let width = rect.width;
        if local_x >= width - 116 && local_x < width - 16 {
            return Some(PromptHit::Accept);
        }
        if prompt.can_cancel && local_x >= width - 224 && local_x < width - 124 {
            return Some(PromptHit::Cancel);
        }
    }
    None
}

pub(crate) fn draw_client_prompt(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    prompt: &ClientPrompt,
) {
    let rect = client_prompt_rect(width, height);
    let x = usize::try_from(rect.x).unwrap_or(0);
    let y = usize::try_from(rect.y).unwrap_or(0);
    let w = usize::try_from(rect.width).unwrap_or(0);
    let h = usize::try_from(rect.height).unwrap_or(0);
    draw_panel(buffer, width, height, x, y, w, h, 0xffd6dae2);
    if w > 4 && h > 4 {
        draw_panel(
            buffer,
            width,
            height,
            x + 2,
            y + 2,
            w - 4,
            h - 4,
            0xff20252d,
        );
    }
    draw_panel(buffer, width, height, x + 2, y + 2, w - 4, 28, 0xff343b47);
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 10,
        y + 12,
        &prompt
            .title
            .chars()
            .take(w.saturating_sub(20) / 8)
            .collect::<String>(),
        0xffffffff,
        (x + w - 6, y + 28),
    );
    draw_bitmap_text(
        buffer,
        width,
        height,
        x + 12,
        y + 45,
        &prompt
            .message
            .chars()
            .take(w.saturating_sub(24) / 8)
            .collect::<String>(),
        0xfff0f2f5,
        (x + w - 8, y + 66),
    );
    if prompt.choices.is_empty()
        && !matches!(
            prompt.kind,
            ClientPromptKind::List | ClientPromptKind::Alert
        )
    {
        draw_input_control(
            buffer,
            width,
            height,
            PixelRect {
                x: rect.x + 12,
                y: rect.y + 78,
                width: rect.width.saturating_sub(24),
                height: 30,
            },
            &prompt.edit,
            true,
        );
    } else if prompt.choices.is_empty() {
        draw_bitmap_text(
            buffer,
            width,
            height,
            x + 18,
            y + 86,
            "No available choices",
            0xffff8888,
            (x + w - 18, y + 108),
        );
    } else {
        for (index, choice) in prompt.choices.iter().take(5).enumerate() {
            let row_y = y + 78 + index * 24;
            draw_panel(
                buffer,
                width,
                height,
                x + 12,
                row_y,
                w.saturating_sub(24),
                22,
                if index == prompt.selected {
                    0xff5b4610
                } else {
                    0xff343b47
                },
            );
            draw_bitmap_text(
                buffer,
                width,
                height,
                x + 18,
                row_y + 7,
                choice,
                0xffffffff,
                (x + w - 18, row_y + 22),
            );
        }
    }
    let button_y = y + h.saturating_sub(42);
    if prompt.can_cancel {
        draw_button_control(
            buffer,
            width,
            height,
            PixelRect {
                x: rect.x + u32::try_from(w.saturating_sub(224)).unwrap_or(0),
                y: u32::try_from(button_y).unwrap_or(0),
                width: 100,
                height: 28,
            },
            "Cancel",
            false,
        );
    }
    draw_button_control(
        buffer,
        width,
        height,
        PixelRect {
            x: rect.x + u32::try_from(w.saturating_sub(116)).unwrap_or(0),
            y: u32::try_from(button_y).unwrap_or(0),
            width: 100,
            height: 28,
        },
        "OK",
        false,
    );
}

pub(crate) fn wrap_output_line(line: &str, columns: usize) -> Vec<String> {
    let characters = line.chars().collect::<Vec<_>>();
    if characters.is_empty() {
        return vec![String::new()];
    }
    characters
        .chunks(columns)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

pub(crate) fn strip_output_markup(message: &str) -> String {
    let mut plain = String::with_capacity(message.len());
    let mut in_tag = false;
    for character in message.chars() {
        match character {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            '&' if !in_tag => plain.push('&'),
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_bitmap_text(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    text: &str,
    color: u32,
    clip: (usize, usize),
) {
    for (column, character) in text.chars().enumerate() {
        let Some(glyph) = font8x8::BASIC_FONTS.get(character) else {
            continue;
        };
        let glyph_x = x + column * 8;
        for (row, bits) in glyph.iter().copied().enumerate() {
            let pixel_y = y + row;
            if pixel_y >= height || pixel_y >= clip.1 {
                continue;
            }
            for bit in 0..8 {
                let pixel_x = glyph_x + bit;
                if bits & (1 << bit) != 0 && pixel_x < width && pixel_x < clip.0 {
                    buffer[pixel_y * width + pixel_x] = color;
                }
            }
        }
    }
}

pub(crate) fn draw_appearance_maptext_signed(
    buffer: &mut [u32],
    width: usize,
    height: usize,
    origin_x: i32,
    origin_y: i32,
    appearances: &[Appearance],
) {
    for appearance in appearances {
        let Some(markup) = appearance.maptext.as_deref() else {
            continue;
        };
        let plain = strip_output_markup(
            &markup
                .replace("<br>", "\n")
                .replace("<br/>", "\n")
                .replace("<br />", "\n"),
        );
        let box_width = appearance.maptext_width.max(8);
        let box_height = appearance.maptext_height.max(8);
        let x = origin_x + appearance.pixel_x + appearance.maptext_x;
        let y = origin_y - appearance.pixel_y - appearance.maptext_y - box_height;
        let Ok(x) = usize::try_from(x.max(0)) else {
            continue;
        };
        let Ok(y) = usize::try_from(y.max(0)) else {
            continue;
        };
        let clip_x = x.saturating_add(usize::try_from(box_width).unwrap_or(8));
        let clip_y = y.saturating_add(usize::try_from(box_height).unwrap_or(8));
        for (line, text) in plain.lines().enumerate() {
            draw_bitmap_text(
                buffer,
                width,
                height,
                x,
                y.saturating_add(line * 10),
                text,
                0xffee_eeee,
                (clip_x, clip_y),
            );
        }
    }
}
