//! Pixel-level `/icon` compositing for Dream64.
//!
//! This crate keeps every image codec and raster operation out of `dm-vm`. It
//! provides:
//!
//! * [`IconBitmap`] / [`IconState`] – an in-memory DMI (states, each a grid of
//!   `dirs * frames` RGBA8 cells).
//! * DMI decode/encode ([`IconBitmap::from_dmi_bytes`], [`IconBitmap::to_dmi_bytes`])
//!   that round-trips BYOND's `Description` `zTXt` metadata.
//! * The BYOND `/icon` raster ops (`Blend`, `Insert`, `Scale`, `Crop`, `Flip`,
//!   `Turn`, `SwapColor`, `MapColors`, `GetPixel`, `DrawBox`, `Shift`).
//! * [`gags`] – Greyscale Asset Generation: composite a config's layer stacks
//!   into an output DMI.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::format_push_string,
    clippy::doc_markdown
)]

mod bitmap;
mod blend;
mod dmi;
pub mod gags;

pub use bitmap::{Frame, IconBitmap, IconState};
pub use blend::BlendMode;
pub use dmi::DmiError;

/// A straight-alpha RGBA8 pixel.
pub type Rgba = [u8; 4];

/// Parse a BYOND colour string (`#rgb`, `#rgba`, `#rrggbb`, `#rrggbbaa`, or a
/// bare CSS-ish name for the handful BYOND accepts) into straight-alpha RGBA8.
#[must_use]
pub fn parse_color(text: &str) -> Option<Rgba> {
    let text = text.trim();
    let hex = text.strip_prefix('#')?;
    let bytes = hex.as_bytes();
    let nib = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    match bytes.len() {
        3 | 4 => {
            let mut out = [255u8; 4];
            for (i, &c) in bytes.iter().enumerate() {
                let v = nib(c)?;
                out[i] = v << 4 | v;
            }
            if bytes.len() == 3 {
                out[3] = 255;
            }
            Some(out)
        }
        6 | 8 => {
            let mut out = [255u8; 4];
            for i in 0..bytes.len() / 2 {
                out[i] = nib(bytes[i * 2])? << 4 | nib(bytes[i * 2 + 1])?;
            }
            if bytes.len() == 6 {
                out[3] = 255;
            }
            Some(out)
        }
        _ => None,
    }
}
