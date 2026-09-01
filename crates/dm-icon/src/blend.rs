//! BYOND `ICON_*` blend functions over straight-alpha RGBA8 pixels.

use crate::Rgba;

/// BYOND blend function selector. Numeric values match BYOND's `ICON_*`
/// constants (`ICON_ADD == 0` … `ICON_UNDERLAY == 6`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Add,
    Subtract,
    Multiply,
    Overlay,
    And,
    Or,
    Underlay,
}

impl BlendMode {
    /// Map a BYOND numeric blend constant.
    #[must_use]
    pub fn from_byond(value: i64) -> Option<Self> {
        Some(match value {
            0 => Self::Add,
            1 => Self::Subtract,
            2 => Self::Multiply,
            3 => Self::Overlay,
            4 => Self::And,
            5 => Self::Or,
            6 => Self::Underlay,
            _ => return None,
        })
    }

    /// Map a GAGS `blend_mode` config string.
    #[must_use]
    pub fn from_gags_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "add" => Self::Add,
            "subtract" => Self::Subtract,
            "multiply" => Self::Multiply,
            "overlay" => Self::Overlay,
            "and" => Self::And,
            "or" => Self::Or,
            "underlay" => Self::Underlay,
            _ => return None,
        })
    }
}

fn clamp_add(a: u8, b: u8) -> u8 {
    a.saturating_add(b)
}

fn clamp_sub(a: u8, b: u8) -> u8 {
    a.saturating_sub(b)
}

fn mul(a: u8, b: u8) -> u8 {
    ((u16::from(a) * u16::from(b) + 127) / 255) as u8
}

/// `over` alpha composite: `top` drawn onto `bottom`, straight alpha.
#[must_use]
pub fn over(top: Rgba, bottom: Rgba) -> Rgba {
    let sa = f32::from(top[3]) / 255.0;
    let da = f32::from(bottom[3]) / 255.0;
    let oa = sa + da * (1.0 - sa);
    if oa <= f32::EPSILON {
        return [0, 0, 0, 0];
    }
    let mut out = [0u8; 4];
    for i in 0..3 {
        let s = f32::from(top[i]) / 255.0;
        let d = f32::from(bottom[i]) / 255.0;
        let v = (s * sa + d * da * (1.0 - sa)) / oa;
        out[i] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out[3] = (oa * 255.0).round().clamp(0.0, 255.0) as u8;
    out
}

/// Apply `mode` blending `src` (the icon/colour being blended in) onto `dst`
/// (the icon receiving the operation). Returns the new `dst` pixel.
#[must_use]
pub fn blend_pixel(dst: Rgba, src: Rgba, mode: BlendMode) -> Rgba {
    match mode {
        BlendMode::Overlay => over(src, dst),
        BlendMode::Underlay => over(dst, src),
        BlendMode::Add => {
            // BYOND scales the added colour by the source coverage.
            let sa = src[3];
            [
                clamp_add(dst[0], mul(src[0], sa)),
                clamp_add(dst[1], mul(src[1], sa)),
                clamp_add(dst[2], mul(src[2], sa)),
                clamp_add(dst[3], src[3]),
            ]
        }
        BlendMode::Subtract => {
            let sa = src[3];
            [
                clamp_sub(dst[0], mul(src[0], sa)),
                clamp_sub(dst[1], mul(src[1], sa)),
                clamp_sub(dst[2], mul(src[2], sa)),
                clamp_sub(dst[3], src[3]),
            ]
        }
        BlendMode::Multiply => {
            // Colour channels multiply; where src is transparent it acts as
            // white (identity) so a `#rrggbb` colour leaves alpha untouched.
            let sa = f32::from(src[3]) / 255.0;
            let lerp_mul = |d: u8, s: u8| -> u8 {
                let m = f32::from(mul(d, s));
                let base = f32::from(d);
                (base + (m - base) * sa).round().clamp(0.0, 255.0) as u8
            };
            [
                lerp_mul(dst[0], src[0]),
                lerp_mul(dst[1], src[1]),
                lerp_mul(dst[2], src[2]),
                mul(dst[3], src[3]),
            ]
        }
        BlendMode::And => {
            if src[3] == 0 || dst[3] == 0 {
                return [0, 0, 0, 0];
            }
            [
                mul(dst[0], src[0]),
                mul(dst[1], src[1]),
                mul(dst[2], src[2]),
                dst[3].min(src[3]),
            ]
        }
        BlendMode::Or => {
            if src[3] == 0 {
                return dst;
            }
            if dst[3] == 0 {
                return src;
            }
            [
                clamp_add(dst[0], src[0]),
                clamp_add(dst[1], src[1]),
                clamp_add(dst[2], src[2]),
                dst[3].max(src[3]),
            ]
        }
    }
}
