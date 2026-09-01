//! DMI codec, raster-op, and GAGS integration tests.
#![allow(
    clippy::cast_possible_truncation,
    clippy::manual_let_else,
    clippy::many_single_char_names
)]

use dm_icon::gags::{self, parse_color_string};
use dm_icon::{BlendMode, Frame, IconBitmap, IconState, parse_color};

const BLANK_32: &[u8] = include_bytes!("fixtures/blank_32.dmi");
const MASKS: &[u8] = include_bytes!("fixtures/masks.dmi");
const ORACLE_TEMPLATE: &[u8] = include_bytes!("../../../fixtures/oracle/icon_ops/template.dmi");
const ORACLE_EXPECTED: &str =
    include_str!("../../../fixtures/oracle/icon_ops/expected-byond-516.1680.txt");

/// Format a pixel the way BYOND's `icon.GetPixel` does: `""` when fully
/// transparent, `#rrggbb` when opaque, `#rrggbbaa` otherwise.
fn byond_pixel(p: Option<[u8; 4]>) -> String {
    match p {
        None => String::new(),
        Some([r, g, b, 255]) => format!("#{r:02x}{g:02x}{b:02x}"),
        Some([r, g, b, a]) => format!("#{r:02x}{g:02x}{b:02x}{a:02x}"),
    }
}

fn solid_state(name: &str, w: u32, h: u32, color: [u8; 4], dirs: u32, frames: u32) -> IconState {
    let cell = Frame {
        width: w,
        height: h,
        pixels: vec![color; (w * h) as usize],
    };
    IconState {
        name: name.to_owned(),
        dirs,
        frame_count: frames,
        delays: if frames > 1 {
            vec![1.0; frames as usize]
        } else {
            Vec::new()
        },
        loop_count: 0,
        rewind: false,
        movement: false,
        hotspot: None,
        cells: vec![cell; (dirs * frames) as usize],
    }
}

#[test]
fn decodes_byond_authored_blank() {
    let bmp = IconBitmap::from_dmi_bytes(BLANK_32).expect("decode blank");
    assert_eq!((bmp.width, bmp.height), (32, 32));
    assert_eq!(bmp.state_names(), vec!["nothing".to_owned()]);
    let state = &bmp.states[0];
    assert_eq!((state.dirs, state.frame_count), (1, 1));
    assert_eq!(state.cells.len(), 1);
    assert_eq!(state.cells[0].pixels.len(), 32 * 32);
}

#[test]
fn decodes_multi_state_sheet_with_animation_and_layout() {
    let bmp = IconBitmap::from_dmi_bytes(MASKS).expect("decode masks");
    assert_eq!((bmp.width, bmp.height), (32, 32));
    // First state is the 4-frame idle animation.
    let first = &bmp.states[0];
    assert_eq!(first.name, "");
    assert_eq!((first.dirs, first.frame_count), (1, 4));
    assert_eq!(first.cells.len(), 4);
    assert_eq!(first.delays, vec![1.2, 1.2, 1.2, 1.2]);
    // Known states used by the bandana GAGS config exist.
    assert!(bmp.state("bandana_cloth").is_some());
    assert!(bmp.state("bandana_cloth_up").is_some());
    // Every cell is a full 32x32 tile.
    for state in &bmp.states {
        assert_eq!(state.cells.len() as u32, state.dirs * state.frame_count);
        for cell in &state.cells {
            assert_eq!((cell.width, cell.height), (32, 32));
        }
    }
}

#[test]
fn round_trips_metadata_and_pixels() {
    let mut src = IconBitmap {
        width: 4,
        height: 3,
        states: vec![
            solid_state("", 4, 3, [10, 20, 30, 255], 1, 2),
            solid_state("walk", 4, 3, [1, 2, 3, 128], 4, 1),
        ],
    };
    // Give one pixel a distinct value to prove cell extraction is positional.
    src.states[0].cells[1].pixels[5] = [99, 88, 77, 66];

    let bytes = src.to_dmi_bytes().expect("encode");
    let back = IconBitmap::from_dmi_bytes(&bytes).expect("decode");

    assert_eq!(back.width, 4);
    assert_eq!(back.height, 3);
    assert_eq!(back.state_names(), vec![String::new(), "walk".to_owned()]);
    assert_eq!(back.states[0].frame_count, 2);
    assert_eq!(back.states[0].delays, vec![1.0, 1.0]);
    assert_eq!(back.states[1].dirs, 4);
    assert_eq!(back.states[0].cells[1].pixels[5], [99, 88, 77, 66]);
    assert_eq!(back.states[1].cells[0].pixels[0], [1, 2, 3, 128]);
}

#[test]
fn blend_multiply_colorizes() {
    let mut bmp = IconBitmap {
        width: 1,
        height: 1,
        states: vec![solid_state("", 1, 1, [200, 200, 200, 255], 1, 1)],
    };
    bmp.blend_color(parse_color("#ff8000").unwrap(), BlendMode::Multiply);
    let px = bmp.states[0].cells[0].pixels[0];
    assert_eq!(px[0], 200); // 200 * 255 / 255
    assert_eq!(px[1], 100); // 200 * 128 / 255 ~ 100
    assert_eq!(px[2], 0);
    assert_eq!(px[3], 255);
}

#[test]
fn blend_overlay_respects_alpha() {
    let mut bmp = IconBitmap {
        width: 1,
        height: 1,
        states: vec![solid_state("", 1, 1, [0, 0, 0, 255], 1, 1)],
    };
    bmp.blend_color([255, 255, 255, 128], BlendMode::Overlay);
    let px = bmp.states[0].cells[0].pixels[0];
    assert_eq!(px[3], 255);
    assert!((120..=140).contains(&px[0]), "got {}", px[0]);
}

#[test]
fn scale_and_crop_change_dimensions() {
    let mut bmp = IconBitmap {
        width: 2,
        height: 2,
        states: vec![solid_state("", 2, 2, [1, 2, 3, 255], 1, 1)],
    };
    bmp.scale(4, 4);
    assert_eq!((bmp.width, bmp.height), (4, 4));
    assert_eq!(bmp.states[0].cells[0].pixels.len(), 16);
    bmp.crop(1, 1, 2, 2);
    assert_eq!((bmp.width, bmp.height), (2, 2));
}

#[test]
fn turn_ninety_swaps_axes_exactly() {
    let mut bmp = IconBitmap {
        width: 2,
        height: 1,
        states: vec![solid_state("", 2, 1, [0, 0, 0, 0], 1, 1)],
    };
    bmp.states[0].cells[0].pixels[0] = [255, 0, 0, 255];
    bmp.turn(90.0);
    assert_eq!((bmp.width, bmp.height), (1, 2));
}

#[test]
fn insert_adds_state_at_canvas_size() {
    let mut bundle = IconBitmap {
        width: 4,
        height: 4,
        states: vec![solid_state("base", 4, 4, [0, 0, 0, 0], 1, 1)],
    };
    let piece = IconBitmap {
        width: 2,
        height: 2,
        states: vec![solid_state("x", 2, 2, [9, 9, 9, 255], 1, 1)],
    };
    bundle.insert(&piece, "hat");
    assert_eq!(
        bundle.state_names(),
        vec!["base".to_owned(), "hat".to_owned()]
    );
    let hat = bundle.state("hat").unwrap();
    assert_eq!((hat.cells[0].width, hat.cells[0].height), (4, 4));
}

#[test]
fn gags_bandana_produces_config_state_names() {
    let config = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../Monkestation2.0/code/datums/greyscale/json_configs/bandana.json"
    ));
    let config = match config {
        Ok(text) => text,
        Err(_) => return, // Monkestation tree not checked out beside Dream64
    };
    let template = IconBitmap::from_dmi_bytes(MASKS).expect("decode masks template");
    let palette = parse_color_string("#4477aa");
    let out = gags::composite(&config, &template, &palette).expect("composite bandana");

    let mut names = out.state_names();
    names.sort();
    assert_eq!(names, vec!["bandana".to_owned(), "bandana_up".to_owned()]);
    assert_eq!((out.width, out.height), (32, 32));
    // Colorized output must not be entirely transparent.
    let any_opaque = out.states[0].cells[0].pixels.iter().any(|p| p[3] > 0);
    assert!(any_opaque, "bandana layer composited to nothing");

    // And it must survive a DMI round-trip with the same state set.
    let bytes = out.to_dmi_bytes().expect("encode gags output");
    let back = IconBitmap::from_dmi_bytes(&bytes).expect("decode gags output");
    let mut back_names = back.state_names();
    back_names.sort();
    assert_eq!(back_names, names);
}

#[test]
fn matches_byond_icon_ops_oracle() {
    let mut expected = std::collections::HashMap::new();
    for line in ORACLE_EXPECTED.lines() {
        if let Some((k, v)) = line.split_once('=') {
            expected.insert(k.to_owned(), v.to_owned());
        }
    }
    let mut got: Vec<(String, String)> = Vec::new();
    let mut emit = |k: &str, v: String| got.push((k.to_owned(), v));

    let full = IconBitmap::from_dmi_bytes(ORACLE_TEMPLATE).expect("decode oracle template");
    let base = full.select_state("box");
    let px = |b: &IconBitmap, x: u32, y: u32| byond_pixel(b.get_pixel(x, y, None, 0, 1));

    emit("base_dims", format!("{}x{}", base.width, base.height));
    emit("base_states", full.state_names().join(","));
    emit("base_fill", px(&base, 4, 4));
    emit("base_hole", px(&base, 4, 28));

    let mut mult = base.clone();
    mult.blend_color(parse_color("#4080c0").unwrap(), BlendMode::Multiply);
    emit("multiply_fill", px(&mult, 4, 4));

    let mut added = base.clone();
    added.blend_color(parse_color("#202020").unwrap(), BlendMode::Add);
    emit("add_fill", px(&added, 4, 4));

    let mut over = base.clone();
    over.blend_color(parse_color("#ffffff80").unwrap(), BlendMode::Overlay);
    emit("overlay_fill", px(&over, 4, 4));

    let mut scaled = base.clone();
    scaled.scale(64, 64);
    emit("scaled_dims", format!("{}x{}", scaled.width, scaled.height));
    emit("scaled_fill", px(&scaled, 8, 8));
    emit("scaled_hole", px(&scaled, 8, 56));

    let mut cropped = base.clone();
    cropped.crop(1, 1, 16, 16);
    emit(
        "cropped_dims",
        format!("{}x{}", cropped.width, cropped.height),
    );
    emit("cropped_fill", px(&cropped, 1, 1));
    emit("cropped_far", px(&cropped, 16, 16));

    let mut flipped = base.clone();
    flipped.flip(1); // NORTH
    emit("flip_low", px(&flipped, 4, 4));
    emit("flip_high", px(&flipped, 4, 28));

    let mut turned = base.clone();
    turned.turn(90.0);
    emit("turn_dims", format!("{}x{}", turned.width, turned.height));

    let mut swapped = base.clone();
    swapped.swap_color(
        parse_color("#808080").unwrap(),
        parse_color("#ff00ff").unwrap(),
    );
    emit("swap_fill", px(&swapped, 4, 4));

    for (key, value) in got {
        assert_eq!(
            expected.get(&key).map(String::as_str),
            Some(value.as_str()),
            "oracle mismatch for {key}"
        );
    }
}

#[test]
fn parse_color_forms() {
    assert_eq!(parse_color("#fff"), Some([255, 255, 255, 255]));
    assert_eq!(parse_color("#ff0000"), Some([255, 0, 0, 255]));
    assert_eq!(parse_color("#00000080"), Some([0, 0, 0, 128]));
    assert_eq!(parse_color("nope"), None);
    assert_eq!(parse_color_string("#112233#445566").len(), 2);
}
