//! In-memory DMI model and the BYOND `/icon` raster operations.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::doc_markdown
)]

use crate::blend::{BlendMode, blend_pixel};
use crate::dmi;
use crate::{DmiError, Rgba};

/// A single icon cell: one direction of one animation frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Row-major, `width * height` straight-alpha RGBA8 pixels.
    pub pixels: Vec<Rgba>,
}

impl Frame {
    #[must_use]
    pub fn transparent(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![[0, 0, 0, 0]; (width as usize) * (height as usize)],
        }
    }

    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> Rgba {
        if x >= self.width || y >= self.height {
            return [0, 0, 0, 0];
        }
        self.pixels[(y as usize) * (self.width as usize) + (x as usize)]
    }

    fn set(&mut self, x: u32, y: u32, value: Rgba) {
        if x < self.width && y < self.height {
            let w = self.width as usize;
            self.pixels[(y as usize) * w + (x as usize)] = value;
        }
    }
}

/// One DMI icon_state: metadata plus `dirs * frame_count` cells.
#[derive(Clone, Debug, PartialEq)]
pub struct IconState {
    pub name: String,
    pub dirs: u32,
    pub frame_count: u32,
    pub delays: Vec<f32>,
    pub loop_count: i64,
    pub rewind: bool,
    pub movement: bool,
    pub hotspot: Option<Vec<i64>>,
    /// `dirs * frame_count` cells, ordered frame-major then direction.
    pub cells: Vec<Frame>,
}

impl IconState {
    #[must_use]
    pub fn cell(&self, frame: u32, dir: u32) -> &Frame {
        &self.cells[(frame as usize) * (self.dirs as usize) + (dir as usize)]
    }

    fn cell_mut(&mut self, frame: u32, dir: u32) -> &mut Frame {
        let dirs = self.dirs as usize;
        &mut self.cells[(frame as usize) * dirs + (dir as usize)]
    }
}

/// A full DMI: canvas dimensions and an ordered list of icon_states.
#[derive(Clone, Debug, PartialEq)]
pub struct IconBitmap {
    pub width: u32,
    pub height: u32,
    pub states: Vec<IconState>,
}

impl IconBitmap {
    /// Decode a PNG-backed DMI.
    ///
    /// # Errors
    /// Returns [`DmiError`] for malformed PNG data or DMI `Description` text.
    pub fn from_dmi_bytes(bytes: &[u8]) -> Result<Self, DmiError> {
        dmi::decode(bytes)
    }

    /// Encode to a PNG-backed DMI with a BYOND `Description` `zTXt` chunk.
    ///
    /// # Errors
    /// Returns [`DmiError`] if PNG encoding fails.
    pub fn to_dmi_bytes(&self) -> Result<Vec<u8>, DmiError> {
        dmi::encode(self)
    }

    #[must_use]
    pub fn state(&self, name: &str) -> Option<&IconState> {
        self.states.iter().find(|s| s.name == name)
    }

    #[must_use]
    pub fn state_names(&self) -> Vec<String> {
        self.states.iter().map(|s| s.name.clone()).collect()
    }

    /// Reduce this bitmap to a single icon_state (BYOND `icon(icon, state)`).
    /// A missing state yields an empty (transparent) bitmap of the same size.
    #[must_use]
    pub fn select_state(&self, name: &str) -> Self {
        // `state(name)` matched on `s.name == name`, so the clone is already
        // named correctly.
        let states = self
            .state(name)
            .cloned()
            .map(|s| vec![s])
            .unwrap_or_default();
        Self {
            width: self.width,
            height: self.height,
            states,
        }
    }

    fn for_each_cell(&mut self, mut f: impl FnMut(&mut Frame)) {
        for state in &mut self.states {
            for cell in &mut state.cells {
                f(cell);
            }
        }
    }

    /// `icon.GetPixel(x, y, icon_state, dir, frame)` – 1-based coordinates with
    /// y counted from the bottom, matching BYOND. Returns `None` for a fully
    /// transparent pixel.
    #[must_use]
    pub fn get_pixel(
        &self,
        x: u32,
        y: u32,
        state: Option<&str>,
        dir: u32,
        frame: u32,
    ) -> Option<Rgba> {
        let state = match state {
            Some(name) => self.state(name)?,
            None => self.states.first()?,
        };
        if x == 0 || y == 0 || x > self.width || y > self.height {
            return None;
        }
        let dir = dir.min(state.dirs.saturating_sub(1));
        let frame = frame
            .saturating_sub(1)
            .min(state.frame_count.saturating_sub(1));
        let cell = state.cell(frame, dir);
        let px = cell.get(x - 1, self.height - y);
        if px[3] == 0 { None } else { Some(px) }
    }

    /// `icon.Scale(width, height)` – nearest-neighbour resample of every cell.
    pub fn scale(&mut self, new_w: u32, new_h: u32) {
        let (ow, oh) = (self.width, self.height);
        if new_w == 0 || new_h == 0 || (ow, oh) == (new_w, new_h) {
            return;
        }
        self.for_each_cell(|cell| {
            let mut out = Frame::transparent(new_w, new_h);
            for y in 0..new_h {
                let sy = ((y as u64 * oh as u64) / new_h as u64) as u32;
                for x in 0..new_w {
                    let sx = ((x as u64 * ow as u64) / new_w as u64) as u32;
                    out.set(x, y, cell.get(sx.min(ow - 1), sy.min(oh - 1)));
                }
            }
            *cell = out;
        });
        self.width = new_w;
        self.height = new_h;
    }

    /// `icon.Crop(x1, y1, x2, y2)` – 1-based inclusive, y from the bottom.
    /// The canvas may grow (padding with transparency) as in BYOND.
    pub fn crop(&mut self, x1: i32, y1: i32, x2: i32, y2: i32) {
        let (lx, hx) = (x1.min(x2), x1.max(x2));
        let (ly, hy) = (y1.min(y2), y1.max(y2));
        let new_w = (hx - lx + 1).max(1) as u32;
        let new_h = (hy - ly + 1).max(1) as u32;
        let oh = self.height as i32;
        self.for_each_cell(|cell| {
            let mut out = Frame::transparent(new_w, new_h);
            for oy in 0..new_h {
                // destination row `oy` (top-down) maps to BYOND y = hy - oy
                let by = hy - oy as i32;
                let src_y = oh - by;
                for ox in 0..new_w {
                    let bx = lx + ox as i32;
                    let src_x = bx - 1;
                    if src_x >= 0
                        && src_y >= 0
                        && src_x < cell.width as i32
                        && src_y < cell.height as i32
                    {
                        out.set(ox, oy, cell.get(src_x as u32, src_y as u32));
                    }
                }
            }
            *cell = out;
        });
        self.width = new_w;
        self.height = new_h;
    }

    /// `icon.Flip(dir)` – NORTH/SOUTH flip vertically, EAST/WEST horizontally.
    pub fn flip(&mut self, dir: i64) {
        let horizontal = matches!(dir, 4 | 8); // EAST | WEST
        self.for_each_cell(|cell| {
            let mut out = Frame::transparent(cell.width, cell.height);
            for y in 0..cell.height {
                for x in 0..cell.width {
                    let (sx, sy) = if horizontal {
                        (cell.width - 1 - x, y)
                    } else {
                        (x, cell.height - 1 - y)
                    };
                    out.set(x, y, cell.get(sx, sy));
                }
            }
            *cell = out;
        });
    }

    /// `icon.Turn(angle)` – exact for multiples of 90°, nearest-neighbour
    /// rotation about the centre otherwise.
    pub fn turn(&mut self, angle: f64) {
        let normalized = ((angle % 360.0) + 360.0) % 360.0;
        let (ow, oh) = (self.width, self.height);
        let quarter = (normalized / 90.0).round();
        let exact = (normalized - quarter * 90.0).abs() < 1.0e-6;
        if exact {
            let steps = (quarter as i64).rem_euclid(4);
            if steps == 0 {
                return;
            }
            let (nw, nh) = if steps % 2 == 1 { (oh, ow) } else { (ow, oh) };
            self.for_each_cell(|cell| {
                let mut out = Frame::transparent(nw, nh);
                for y in 0..cell.height {
                    for x in 0..cell.width {
                        let px = cell.get(x, y);
                        let (dx, dy) = match steps {
                            1 => (y, cell.width - 1 - x),
                            2 => (cell.width - 1 - x, cell.height - 1 - y),
                            _ => (cell.height - 1 - y, x),
                        };
                        out.set(dx, dy, px);
                    }
                }
                *cell = out;
            });
            self.width = nw;
            self.height = nh;
            return;
        }
        let radians = normalized.to_radians();
        let (sin, cos) = radians.sin_cos();
        let cx = f64::from(ow) / 2.0;
        let cy = f64::from(oh) / 2.0;
        self.for_each_cell(|cell| {
            let mut out = Frame::transparent(cell.width, cell.height);
            for y in 0..cell.height {
                for x in 0..cell.width {
                    let rx = f64::from(x) + 0.5 - cx;
                    let ry = f64::from(y) + 0.5 - cy;
                    let sx = rx * cos + ry * sin + cx - 0.5;
                    let sy = -rx * sin + ry * cos + cy - 0.5;
                    let sxi = sx.round();
                    let syi = sy.round();
                    if sxi >= 0.0
                        && syi >= 0.0
                        && sxi < f64::from(cell.width)
                        && syi < f64::from(cell.height)
                    {
                        out.set(x, y, cell.get(sxi as u32, syi as u32));
                    }
                }
            }
            *cell = out;
        });
    }

    /// `icon.Shift(dir, offset, wrap)` – translate every cell.
    pub fn shift(&mut self, dir: i64, offset: i32, wrap: bool) {
        let (mut dx, mut dy) = (0i32, 0i32);
        match dir {
            1 => dy = offset,  // NORTH (up, +y bottom-origin -> -row)
            2 => dy = -offset, // SOUTH
            4 => dx = offset,  // EAST
            8 => dx = -offset, // WEST
            _ => {}
        }
        self.for_each_cell(|cell| {
            let mut out = Frame::transparent(cell.width, cell.height);
            let (w, h) = (cell.width as i32, cell.height as i32);
            for y in 0..h {
                for x in 0..w {
                    // screen row: y from top; bottom-origin shift => subtract dy
                    let mut sx = x - dx;
                    let mut sy = y + dy;
                    if wrap {
                        sx = sx.rem_euclid(w);
                        sy = sy.rem_euclid(h);
                    }
                    if sx >= 0 && sy >= 0 && sx < w && sy < h {
                        out.set(x as u32, y as u32, cell.get(sx as u32, sy as u32));
                    }
                }
            }
            *cell = out;
        });
    }

    /// `icon.DrawBox(rgb, x1, y1, x2, y2)` – filled rectangle, 1-based, y from
    /// the bottom. `None` clears (transparent) as BYOND does.
    pub fn draw_box(&mut self, color: Option<Rgba>, x1: i32, y1: i32, x2: i32, y2: i32) {
        let (lx, hx) = (x1.min(x2).max(1), x1.max(x2));
        let (ly, hy) = (y1.min(y2).max(1), y1.max(y2));
        let h = self.height as i32;
        let value = color.unwrap_or([0, 0, 0, 0]);
        self.for_each_cell(|cell| {
            for by in ly..=hy {
                let sy = h - by;
                for bx in lx..=hx {
                    let sx = bx - 1;
                    if sx >= 0 && sy >= 0 && sx < cell.width as i32 && sy < cell.height as i32 {
                        cell.set(sx as u32, sy as u32, value);
                    }
                }
            }
        });
    }

    /// `icon.SwapColor(old, new)` – replace every exactly-matching pixel.
    pub fn swap_color(&mut self, old: Rgba, new: Rgba) {
        self.for_each_cell(|cell| {
            for px in &mut cell.pixels {
                if *px == old {
                    *px = new;
                }
            }
        });
    }

    /// `icon.MapColors(...)` with a 4x3 / 4x4 / 4x5 colour matrix expressed as a
    /// flat, column-major (BYOND `rr,rg,rb,ra, gr,...`) list of coefficients in
    /// 0..1 space. Accepts 12, 16, or 20 entries (with optional constant row).
    pub fn map_colors(&mut self, matrix: &[f32]) {
        // Normalise to a 4x5 matrix: rows = output R,G,B,A; cols = in R,G,B,A,1.
        let mut m = [[0.0f32; 5]; 4];
        match matrix.len() {
            // rr,rg,rb, gr,gg,gb, br,bg,bb  (3x3, RGB only)
            9 => {
                for r in 0..3 {
                    for c in 0..3 {
                        m[r][c] = matrix[r * 3 + c];
                    }
                }
                m[3][3] = 1.0;
            }
            // 4x4 with alpha
            16 => {
                for r in 0..4 {
                    for c in 0..4 {
                        m[r][c] = matrix[r * 4 + c];
                    }
                }
            }
            // 4x5 with constant column
            20 => {
                for r in 0..4 {
                    for c in 0..5 {
                        m[r][c] = matrix[r * 5 + c];
                    }
                }
            }
            // 4x3: R,G,B,A inputs -> R,G,B outputs, alpha preserved
            12 => {
                for r in 0..3 {
                    for c in 0..4 {
                        m[r][c] = matrix[r * 4 + c];
                    }
                }
                m[3][3] = 1.0;
            }
            _ => return,
        }
        self.for_each_cell(|cell| {
            for px in &mut cell.pixels {
                let inp = [
                    f32::from(px[0]) / 255.0,
                    f32::from(px[1]) / 255.0,
                    f32::from(px[2]) / 255.0,
                    f32::from(px[3]) / 255.0,
                    1.0,
                ];
                let mut out = [0.0f32; 4];
                for (r, row) in m.iter().enumerate() {
                    out[r] = row.iter().zip(inp).map(|(a, b)| a * b).sum();
                }
                for (i, v) in out.iter().enumerate() {
                    px[i] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
                }
            }
        });
    }

    /// `icon.Blend(rgb, function, x, y)` – blend a solid colour into every cell.
    pub fn blend_color(&mut self, color: Rgba, mode: BlendMode) {
        self.for_each_cell(|cell| {
            for px in &mut cell.pixels {
                *px = blend_pixel(*px, color, mode);
            }
        });
    }

    /// `icon.Blend(icon, function, x, y)` – blend another icon in. The other
    /// icon's first state / matching cell count is used; 1-based `x`,`y` shift
    /// the overlay with a bottom-left origin.
    pub fn blend_icon(&mut self, other: &IconBitmap, mode: BlendMode, x: i32, y: i32) {
        let Some(src_state) = other.states.first() else {
            return;
        };
        let dx = x - 1;
        let dy = y - 1;
        let self_h = self.height as i32;
        let other_h = other.height as i32;
        for state in &mut self.states {
            for frame in 0..state.frame_count {
                for dir in 0..state.dirs {
                    let src_frame = frame.min(src_state.frame_count.saturating_sub(1));
                    let src_dir = dir.min(src_state.dirs.saturating_sub(1));
                    let src = src_state.cell(src_frame, src_dir);
                    let dst = state.cell_mut(frame, dir);
                    for oy in 0..src.height as i32 {
                        for ox in 0..src.width as i32 {
                            // convert to bottom-origin, apply shift, back to top-origin
                            let by = (other_h - 1 - oy) + dy;
                            let tx = ox + dx;
                            let ty = self_h - 1 - by;
                            if tx >= 0 && ty >= 0 && tx < dst.width as i32 && ty < dst.height as i32
                            {
                                let s = src.get(ox as u32, oy as u32);
                                let d = dst.get(tx as u32, ty as u32);
                                dst.set(tx as u32, ty as u32, blend_pixel(d, s, mode));
                            }
                        }
                    }
                }
            }
        }
    }

    /// `icon.Insert(new_icon, icon_state, dir, frame, moving, delay)` – add or
    /// replace an icon_state from another bitmap's first state.
    pub fn insert(&mut self, other: &IconBitmap, state_name: &str) {
        if other.states.is_empty() {
            return;
        }
        // Resample the source onto our canvas if the dimensions differ.
        let mut resampled;
        let source: &IconBitmap = if (other.width, other.height) == (self.width, self.height) {
            other
        } else {
            resampled = other.clone();
            resampled.scale(self.width, self.height);
            &resampled
        };
        let mut src = source.states[0].clone();
        src.name = String::from(state_name);
        if let Some(existing) = self.states.iter_mut().find(|s| s.name == state_name) {
            *existing = src;
        } else {
            self.states.push(src);
        }
    }

    /// Flatten to a single RGBA8 image of the first state's first cell,
    /// compositing over an opaque-free background. Handy for tests.
    #[must_use]
    pub fn flatten_first(&self) -> Option<Frame> {
        let state = self.states.first()?;
        Some(state.cell(0, 0).clone())
    }
}
