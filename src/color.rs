//! Color in a linear-sRGB working space.
//!
//! Components are stored **linear**, while `from_hex` and `get_hex` convert to
//! and from sRGB (the 0xRRGGBB packing standard across software).
//! `set_hsl` writes its result straight into the working space, so hues chosen
//! that way are linear values and not sRGB ones.

use crate::color_tables::{LINEAR_THRESHOLD, SRGB_TO_LINEAR};

/// The sRGB transfer function, for an 8-bit component.
///
/// Read from a table rather than recomputed, so the value
/// is identical on every platform. See [`crate::color_tables`].
#[inline]
pub fn srgb_byte_to_linear(k: u8) -> f64 {
    SRGB_TO_LINEAR[k as usize]
}

/// Encode a linear value as an 8-bit sRGB component, clamping to `[0, 1]`.
///
/// Equivalent to `round(clamp(LinearToSRGB(c) * 255, 0, 255))`, but resolved by
/// binary search over precomputed thresholds: exact, platform-independent, and
/// faster than a `pow` per channel.
#[inline]
pub fn linear_to_srgb_byte(c: f64) -> u8 {
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(c > LINEAR_THRESHOLD[0]) {
        // Negated deliberately: this also catches NaN, which the caller
        // would clamp to 0.
        return 0;
    }
    let mut lo = 0usize; // LINEAR_THRESHOLD[lo] <= c
    let mut hi = LINEAR_THRESHOLD.len(); // c < threshold[hi] (virtual)
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if c >= LINEAR_THRESHOLD[mid] {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + 1) as u8
}

/// The inverse transfer function, continuous, for callers that want the
/// float rather than the quantized byte. Uses the host `powf`, whereas the rendering
/// and `get_hex` paths use [`linear_to_srgb_byte`] instead.
#[inline]
pub fn linear_to_srgb(c: f64) -> f64 {
    if c < 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(0.41666) - 0.055
    }
}

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[inline]
fn euclidean_modulo(n: f64, m: f64) -> f64 {
    ((n % m) + m) % m
}

fn hue2rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 {
        t += 1.0;
    }
    if t > 1.0 {
        t -= 1.0;
    }
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 1.0 / 2.0 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * 6.0 * (2.0 / 3.0 - t);
    }
    p
}

/// Linear-sRGB color.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };

    /// From a packed `0xRRGGBB`, interpreted as sRGB.
    pub fn from_hex(hex: u32) -> Color {
        Color {
            r: srgb_byte_to_linear(((hex >> 16) & 255) as u8),
            g: srgb_byte_to_linear(((hex >> 8) & 255) as u8),
            b: srgb_byte_to_linear((hex & 255) as u8),
        }
    }

    /// `setHSL(h, s, l)` with the default (working-space) color space.
    pub fn from_hsl(h: f64, s: f64, l: f64) -> Color {
        let h = euclidean_modulo(h, 1.0);
        let s = clamp(s, 0.0, 1.0);
        let l = clamp(l, 0.0, 1.0);
        if s == 0.0 {
            return Color { r: l, g: l, b: l };
        }
        let p = if l <= 0.5 {
            l * (1.0 + s)
        } else {
            l + s - l * s
        };
        let q = 2.0 * l - p;
        Color {
            r: hue2rgb(q, p, h + 1.0 / 3.0),
            g: hue2rgb(q, p, h),
            b: hue2rgb(q, p, h - 1.0 / 3.0),
        }
    }

    /// The color as three 8-bit sRGB components, as written to a framebuffer.
    #[inline]
    pub fn to_srgb_bytes(&self) -> [u8; 3] {
        [
            linear_to_srgb_byte(self.r),
            linear_to_srgb_byte(self.g),
            linear_to_srgb_byte(self.b),
        ]
    }

    /// `getHex()`: converts back to sRGB and packs into `0xRRGGBB`.
    pub fn get_hex(&self) -> u32 {
        let f = |c: f64| u32::from(linear_to_srgb_byte(c));
        f(self.r) * 65536 + f(self.g) * 256 + f(self.b)
    }

    pub fn get_hex_string(&self) -> String {
        format!("{:06x}", self.get_hex())
    }
}
