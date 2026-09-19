//! A built-in proportional bitmap font, so an image can be labeled without a
//! font file, a system font service or a third-party crate.
//!
//! The glyphs are drawn in `tools/gen_font.py` and packed into
//! [`font_data`](super::font_data). Each occupies a cell nine rows tall: rows
//! 0-6 carry the cap height and rows 7-8 the descenders, with the baseline
//! directly under row 6. Widths vary per glyph (`i` is one column and `M` is
//! five), which reads far better at label sizes than a fixed pitch.
//!
//! Text is drawn at an integer `scale`. For smooth edges, draw into a
//! framebuffer at two or three times the final size and
//! [`downsample`](super::framebuffer::Framebuffer::downsample) it, which is
//! what `scripts/catalog_sheet.py` does.

use super::font_data::{BASELINE, BITMAP, FIRST, LAST, ROWS, WIDTHS};
use super::framebuffer::{BlendMode, Framebuffer, Rect};

/// Blank columns between two glyphs, before scaling.
pub const TRACKING: u32 = 1;
/// Blank rows between two lines, before scaling.
pub const LEADING: u32 = 2;

/// Height of one line of text at `scale`, in pixels.
#[inline]
pub fn line_height(scale: u32) -> u32 {
    ROWS as u32 * scale
}

/// Distance from the top of a cell down to the baseline, in pixels.
#[inline]
pub fn baseline(scale: u32) -> u32 {
    BASELINE as u32 * scale
}

/// The glyph index for `ch`, substituting `?` for anything outside the table.
#[inline]
fn slot(ch: char) -> usize {
    let c = u32::from(ch);
    let c = if (u32::from(FIRST)..=u32::from(LAST)).contains(&c) {
        c as u8
    } else {
        b'?'
    };
    (c - FIRST) as usize
}

/// Ink columns of one glyph, before scaling.
#[inline]
fn glyph_width(ch: char) -> u32 {
    u32::from(WIDTHS[slot(ch)])
}

/// Width of a single line of `text` at `scale`, in pixels.
///
/// Trailing tracking is not counted, so two strings drawn back to back with
/// this as the offset sit exactly one tracking column apart.
pub fn text_width(text: &str, scale: u32) -> u32 {
    let mut w = 0u32;
    let mut first = true;
    for ch in text.chars() {
        if ch == '\n' {
            break;
        }
        if !first {
            w += TRACKING;
        }
        w += glyph_width(ch);
        first = false;
    }
    w * scale
}

/// Width of the widest line in `text`, in pixels.
pub fn text_block_width(text: &str, scale: u32) -> u32 {
    text.split('\n')
        .map(|l| text_width(l, scale))
        .max()
        .unwrap_or(0)
}

/// Height of `text` in pixels, counting its newlines.
pub fn text_block_height(text: &str, scale: u32) -> u32 {
    let lines = text.split('\n').count() as u32;
    lines * line_height(scale) + lines.saturating_sub(1) * LEADING * scale
}

impl Framebuffer {
    /// Draw `text` with its cell's top-left corner at `(x, y)`.
    ///
    /// `scale` magnifies each ink column into a `scale`x`scale` block. Returns
    /// the width of the text drawn, so captions can be chained.
    pub fn draw_text(
        &mut self,
        x: i64,
        y: i64,
        text: &str,
        rgba: [u8; 4],
        scale: u32,
        mode: BlendMode,
    ) -> u32 {
        if scale == 0 {
            return 0;
        }
        let mut widest = 0u32;
        let mut pen_y = y;
        for line in text.split('\n') {
            let mut pen_x = x;
            for ch in line.chars() {
                let s = slot(ch);
                let w = u32::from(WIDTHS[s]);
                let rows = &BITMAP[s * ROWS..(s + 1) * ROWS];
                for (row, bits) in rows.iter().enumerate() {
                    if *bits == 0 {
                        continue;
                    }
                    for col in 0..w {
                        if bits & (0x80 >> col) != 0 {
                            self.fill_rect(
                                Rect {
                                    x: pen_x + i64::from(col * scale),
                                    y: pen_y + (row as i64) * i64::from(scale),
                                    w: scale,
                                    h: scale,
                                },
                                rgba,
                                mode,
                            );
                        }
                    }
                }
                pen_x += i64::from((w + TRACKING) * scale);
            }
            widest = widest.max(text_width(line, scale));
            pen_y += i64::from(line_height(scale) + LEADING * scale);
        }
        widest
    }

    /// Draw `text` centered horizontally on `cx`, with its cell top at `y`.
    pub fn draw_text_centered(
        &mut self,
        cx: i64,
        y: i64,
        text: &str,
        rgba: [u8; 4],
        scale: u32,
        mode: BlendMode,
    ) -> u32 {
        let mut pen_y = y;
        let mut widest = 0;
        for line in text.split('\n') {
            let w = text_width(line, scale);
            self.draw_text(cx - i64::from(w) / 2, pen_y, line, rgba, scale, mode);
            widest = widest.max(w);
            pen_y += i64::from(line_height(scale) + LEADING * scale);
        }
        widest
    }
}

/// Shorten `text` to fit `max_width` pixels at `scale`, ending in an ellipsis.
///
/// Returns `text` unchanged when it already fits, and an empty string when not
/// even the ellipsis does.
pub fn ellipsize(text: &str, max_width: u32, scale: u32) -> String {
    const DOTS: &str = "...";
    if text_width(text, scale) <= max_width {
        return text.to_string();
    }
    let dots = text_width(DOTS, scale);
    if dots > max_width {
        return String::new();
    }
    let mut out = String::new();
    for ch in text.chars() {
        let mut trial = out.clone();
        trial.push(ch);
        if text_width(&trial, scale) + (TRACKING * scale) + dots > max_width {
            break;
        }
        out = trial;
    }
    out.push_str(DOTS);
    out
}
