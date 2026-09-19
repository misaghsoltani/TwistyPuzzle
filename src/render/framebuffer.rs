//! RGBA8 framebuffers and blitting.
//!
//! Pixels are stored as tightly packed `RGBA` bytes, the layout NumPy, Pillow
//! and most GUI toolkits expect, so a frame can be handed to Python without
//! copying or reordering.
//!
//! Compositing happens on sRGB-encoded values, matching a WebGL framebuffer:
//! the fragment shader encodes to sRGB and the blend unit then operates on
//! those bytes.

use rayon::prelude::*;

/// How a source is combined with the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Overwrite the destination, alpha included.
    Copy,
    /// `dst = src*a + dst*(1-a)`: ordinary source-over alpha, with
    /// non-premultiplied alpha.
    Over,
    /// `dst = dst + src*a`: additive.
    Additive,
}

/// A tightly packed RGBA8 image.
#[derive(Clone)]
pub struct Framebuffer {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// A rectangle clipped against both surfaces by [`Framebuffer::blit`].
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: u32,
    pub h: u32,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Framebuffer {
        Framebuffer {
            width,
            height,
            data: vec![0; (width as usize * height as usize) * 4],
        }
    }

    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Framebuffer {
        let mut fb = Framebuffer::new(width, height);
        fb.fill(rgba);
        fb
    }

    /// Wrap existing RGBA8 bytes.
    pub fn from_raw(width: u32, height: u32, data: Vec<u8>) -> Option<Framebuffer> {
        if data.len() != width as usize * height as usize * 4 {
            return None;
        }
        Some(Framebuffer {
            width,
            height,
            data,
        })
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.width
    }
    #[inline]
    pub fn height(&self) -> u32 {
        self.height
    }
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
    /// Bytes per row.
    #[inline]
    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.data[i],
            self.data[i + 1],
            self.data[i + 2],
            self.data[i + 3],
        ]
    }

    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, rgba: [u8; 4]) {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        self.data[i..i + 4].copy_from_slice(&rgba);
    }

    /// Fill the whole surface with one color.
    pub fn fill(&mut self, rgba: [u8; 4]) {
        if rgba[0] == rgba[1] && rgba[1] == rgba[2] && rgba[2] == rgba[3] {
            self.data.fill(rgba[0]);
            return;
        }
        // Build one row, then replicate it: much faster than per-pixel stores.
        let stride = self.stride();
        let (first, rest) = self.data.split_at_mut(stride);
        for px in first.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        for row in rest.chunks_exact_mut(stride) {
            row.copy_from_slice(first);
        }
    }

    pub fn fill_rect(&mut self, rect: Rect, rgba: [u8; 4], mode: BlendMode) {
        let Some((dx, dy, w, h)) = clip(rect, self.width, self.height) else {
            return;
        };
        let stride = self.stride();
        for row in 0..h {
            let base = (dy + row) as usize * stride + dx as usize * 4;
            for col in 0..w as usize {
                let i = base + col * 4;
                blend_pixel(&mut self.data[i..i + 4], rgba, mode);
            }
        }
    }

    /// Copy a rectangle of `src` onto `self` at `(dx, dy)`.
    ///
    /// The source rectangle is clipped to `src` and the destination to `self`,
    /// so out-of-range coordinates are harmless. Rows are moved with
    /// `copy_from_slice` when the mode is [`BlendMode::Copy`], which compiles
    /// to a `memcpy` per row.
    pub fn blit(&mut self, src: &Framebuffer, src_rect: Rect, dx: i64, dy: i64, mode: BlendMode) {
        // Clip the source rectangle to the source surface.
        let Some((sx, sy, sw, sh)) = clip(src_rect, src.width, src.height) else {
            return;
        };
        // Shift the destination by however much the source was clipped.
        let dx = dx + (i64::from(sx) - src_rect.x);
        let dy = dy + (i64::from(sy) - src_rect.y);

        // Clip the destination.
        let Some((dxc, dyc, w, h)) = clip(
            Rect {
                x: dx,
                y: dy,
                w: sw,
                h: sh,
            },
            self.width,
            self.height,
        ) else {
            return;
        };
        let skip_x = (i64::from(dxc) - dx) as u32;
        let skip_y = (i64::from(dyc) - dy) as u32;
        let sx = sx + skip_x;
        let sy = sy + skip_y;

        let s_stride = src.stride();
        let d_stride = self.stride();
        for row in 0..h {
            let s = (sy + row) as usize * s_stride + sx as usize * 4;
            let d = (dyc + row) as usize * d_stride + dxc as usize * 4;
            let n = w as usize * 4;
            match mode {
                BlendMode::Copy => {
                    self.data[d..d + n].copy_from_slice(&src.data[s..s + n]);
                },
                _ => {
                    for col in 0..w as usize {
                        let sp = [
                            src.data[s + col * 4],
                            src.data[s + col * 4 + 1],
                            src.data[s + col * 4 + 2],
                            src.data[s + col * 4 + 3],
                        ];
                        blend_pixel(&mut self.data[d + col * 4..d + col * 4 + 4], sp, mode);
                    }
                },
            }
        }
    }

    /// Box-filter downsample by an integer factor.
    ///
    /// This is the supersampling resolve: rendering at `factor` times the
    /// output size and averaging gives the same coverage as multisampling,
    /// because every fragment in this scene is flat-shaded.
    pub fn downsample(&self, factor: u32) -> Framebuffer {
        assert!(factor >= 1, "downsample factor must be positive");
        if factor == 1 {
            return self.clone();
        }
        let ow = self.width / factor;
        let oh = self.height / factor;
        let mut out = Framebuffer::new(ow, oh);
        let n = factor * factor;
        let half = n / 2;
        let src = &self.data;
        let sw = self.width as usize;
        out.data
            .par_chunks_mut(ow as usize * 4)
            .enumerate()
            .for_each(|(oy, row)| {
                for ox in 0..ow as usize {
                    let mut acc = [0u32; 4];
                    for sy in 0..factor as usize {
                        let base = ((oy * factor as usize + sy) * sw + ox * factor as usize) * 4;
                        for sx in 0..factor as usize {
                            let i = base + sx * 4;
                            acc[0] += u32::from(src[i]);
                            acc[1] += u32::from(src[i + 1]);
                            acc[2] += u32::from(src[i + 2]);
                            acc[3] += u32::from(src[i + 3]);
                        }
                    }
                    for c in 0..4 {
                        // Round half up, as a resolve does.
                        row[ox * 4 + c] = ((acc[c] + half) / n) as u8;
                    }
                }
            });
        out
    }

    /// Nearest-neighbor scale into a new surface.
    pub fn scaled(&self, width: u32, height: u32) -> Framebuffer {
        let mut out = Framebuffer::new(width, height);
        if width == 0 || height == 0 || self.width == 0 || self.height == 0 {
            return out;
        }
        let sw = u64::from(self.width);
        let sh = u64::from(self.height);
        let stride = self.stride();
        out.data
            .par_chunks_mut(width as usize * 4)
            .enumerate()
            .for_each(|(y, row)| {
                let sy = (y as u64 * sh / u64::from(height)).min(sh - 1) as usize;
                let sbase = sy * stride;
                for x in 0..width as usize {
                    let sx = (x as u64 * sw / u64::from(width)).min(sw - 1) as usize;
                    let i = sbase + sx * 4;
                    row[x * 4..x * 4 + 4].copy_from_slice(&self.data[i..i + 4]);
                }
            });
        out
    }

    /// Encode as a PNG.
    pub fn to_png(&self) -> Vec<u8> {
        crate::render::png::encode_rgba(self.width, self.height, &self.data)
    }
}

#[inline]
fn clip(rect: Rect, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let x0 = rect.x.max(0);
    let y0 = rect.y.max(0);
    let x1 = (rect.x + i64::from(rect.w)).min(i64::from(width));
    let y1 = (rect.y + i64::from(rect.h)).min(i64::from(height));
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some((x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

#[inline]
fn blend_pixel(dst: &mut [u8], src: [u8; 4], mode: BlendMode) {
    match mode {
        BlendMode::Copy => dst.copy_from_slice(&src),
        BlendMode::Over => {
            let a = u32::from(src[3]);
            if a == 0 {
                return;
            }
            if a == 255 {
                dst.copy_from_slice(&src);
                return;
            }
            let ia = 255 - a;
            for c in 0..3 {
                // Rounded 8-bit lerp.
                let v = u32::from(src[c]) * a + u32::from(dst[c]) * ia + 127;
                dst[c] = ((v + (v >> 8)) >> 8) as u8;
            }
            let out_a = a * 255 + u32::from(dst[3]) * ia + 127;
            dst[3] = ((out_a + (out_a >> 8)) >> 8) as u8;
        },
        BlendMode::Additive => {
            let a = u32::from(src[3]);
            for c in 0..3 {
                let v = u32::from(src[c]) * a + 127;
                let add = (v + (v >> 8)) >> 8;
                dst[c] = (u32::from(dst[c]) + add).min(255) as u8;
            }
            dst[3] = (u32::from(dst[3]) + a).min(255) as u8;
        },
    }
}
