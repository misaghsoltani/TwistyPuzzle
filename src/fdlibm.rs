//! Transcendental functions that give the same answer everywhere.
//!
//! The host C library is not usable here. Platform `libm` implementations
//! disagree with each other in the last place (on macOS, for instance,
//! `acos(0.5)` differs from the value most other platforms produce by one unit
//! in the last place, which is enough to turn an angle of `120` degrees into
//! `120.00000000000001`). A puzzle's coordinates would then depend on the
//! machine that drew it.
//!
//! These are transcriptions of Sun's fdlibm, which is correctly rounded to
//! within the documented bound and, more importantly here, is deterministic.
//! `sqrt` is exempt because IEEE 754 requires it to be correctly rounded, so
//! the hardware instruction already agrees everywhere, and `pow` is exempt
//! because the color pipeline, its only caller, uses the exact lookup
//! tables in [`crate::color_tables`] instead.
//!
//! Derived from fdlibm, Copyright (C) 1993 by Sun Microsystems, Inc.
//! Developed at SunSoft, a Sun Microsystems, Inc. business. Permission to use,
//! copy, modify, and distribute this software is freely granted, provided that
//! this notice is preserved.

// This file is a transcription, so the lints that ask it to look like ordinary
// Rust are asking it to stop being checkable against its source.
//
// The three bit-twiddling helpers are `#define`s in the original and must not
// become calls, while the constants that read like `FRAC_1_SQRT_2` are fdlibm's own
// decimal spellings, and substituting Rust's would substitute a different
// double.
#![allow(clippy::inline_always, clippy::approx_constant)]

#[inline(always)]
fn hi(x: f64) -> u32 {
    (x.to_bits() >> 32) as u32
}

#[inline(always)]
fn lo(x: f64) -> u32 {
    x.to_bits() as u32
}

#[inline(always)]
fn with_lo(x: f64, l: u32) -> f64 {
    f64::from_bits((x.to_bits() & 0xffff_ffff_0000_0000) | u64::from(l))
}

const PIO2_HI: f64 = f64::from_bits(0x3FF9_21FB_5444_2D18);
const PIO2_LO: f64 = f64::from_bits(0x3C91_A626_3314_5C07);
const PI: f64 = f64::from_bits(0x4009_21FB_5444_2D18);

const PS0: f64 = f64::from_bits(0x3FC5_5555_5555_5555);
const PS1: f64 = f64::from_bits(0xBFD4_D612_03EB_6F7D);
const PS2: f64 = f64::from_bits(0x3FC9_C155_0E88_4455);
const PS3: f64 = f64::from_bits(0xBFA4_8228_B568_8F3B);
const PS4: f64 = f64::from_bits(0x3F49_EFE0_7501_B288);
const PS5: f64 = f64::from_bits(0x3F02_3DE1_0DFD_F709);
const QS1: f64 = f64::from_bits(0xC003_3A27_1C8A_2D4B);
const QS2: f64 = f64::from_bits(0x4000_2AE5_9C59_8AC8);
const QS3: f64 = f64::from_bits(0xBFE6_066C_1B8D_0159);
const QS4: f64 = f64::from_bits(0x3FB3_B8C5_B12E_9282);

#[inline]
fn acos_r(z: f64) -> f64 {
    let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
    let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
    p / q
}

/// `Math.acos`: fdlibm `__ieee754_acos`.
pub fn acos(x: f64) -> f64 {
    let hx = hi(x) as i32;
    let ix = hx & 0x7fff_ffff;
    if ix >= 0x3ff0_0000 {
        // |x| >= 1
        if ((ix - 0x3ff0_0000) as u32 | lo(x)) == 0 {
            return if hx > 0 { 0.0 } else { PI + 2.0 * PIO2_LO };
        }
        return f64::NAN;
    }
    if ix < 0x3fe0_0000 {
        // |x| < 0.5
        if ix <= 0x3c60_0000 {
            // |x| < 2**-57
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let r = acos_r(z);
        return PIO2_HI - (x - (PIO2_LO - x * r));
    }
    if hx < 0 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let r = acos_r(z);
        let s = z.sqrt();
        let w = r * s - PIO2_LO;
        PI - 2.0 * (s + w)
    } else {
        // x > 0.5
        let z = (1.0 - x) * 0.5;
        let s = z.sqrt();
        let df = with_lo(s, 0);
        let c = (z - df * df) / (s + df);
        let r = acos_r(z);
        let w = r * s + c;
        2.0 * (df + w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acos_matches_v8() {
        // Expected values as raw bit patterns, so a comparison cannot be
        // softened by a decimal literal that does not round-trip.
        let cases: &[(f64, u64)] = &[
            (0.5, 0x3ff0_c152_382d_7366),
            (-0.5, 0x4000_c152_382d_7366),
            (0.0, 0x3ff9_21fb_5444_2d18),
            (1.0, 0x0000_0000_0000_0000),
            (-1.0, 0x4009_21fb_5444_2d18),
            (0.25, 0x3ff5_1700_e0c1_4b25),
            (0.9, 0x3fdc_dd9f_8f92_2e98),
            (-0.7071067811865476, 0x4002_d97c_7f33_21d2),
            (0.9999999, 0x3f3d_4eff_c851_e7f2),
            (1e-20, 0x3ff9_21fb_5444_2d18),
            (-0.999999, 0x4009_1f15_dfb8_2062),
            (0.7071067811865476, 0x3fe9_21fb_5444_2d18),
            (0.3333333333333333, 0x3ff3_b202_8082_e8d4),
        ];
        for &(x, bits) in cases {
            assert_eq!(acos(x).to_bits(), bits, "acos({x})");
        }
    }
}
