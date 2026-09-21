//! Building a puzzle out of planes.

use indexmap::IndexMap;

use crate::color::Color;
use crate::exact::qq_nothing;
use crate::math::ExactPlane;
use crate::num::ring::RingOps;
use crate::piece::{cube_polygeometry, slice_polygeometry, PolyGeometry};
use crate::Result;

/// The first six colors are the original Rubik's Cube colors, while later ones are
/// spread around the hue circle by the golden ratio so they contrast with the
/// colors already used.
///
/// <https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/>
pub fn get_color(i: usize) -> Color {
    const BASIC: [u32; 6] = [0xFFFFFF, 0xC41E3A, 0x009E60, 0x0051BA, 0xFF5800, 0xFFD500];
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    if i < 6 {
        Color::from_hex(BASIC[i])
    } else {
        Color::from_hsl((i as f64 / phi) % 1.0, 1.0, 0.5)
    }
}

pub fn cut_color() -> Color {
    Color::from_hex(0x666666)
}

/// Intersect the backs of `faces`, yielding the puzzle shell.
///
/// Only correct for convex polyhedra. Starts from a cube far larger than any
/// puzzle and slices it down.
pub fn make_shell(faces: &[ExactPlane]) -> Result<PolyGeometry> {
    let mut g = cube_polygeometry(&qq_nothing().from_int(1000), cut_color(), true)?;
    for (i, face) in faces.iter().enumerate() {
        let (_front, back) = slice_polygeometry(&g, face, get_color(i), false)?;
        g = back;
    }
    Ok(g)
}

/// Apply every cut plane to every piece.
///
/// Cuts sharing a normal are applied deepest-first, and the back half is set
/// aside so each subsequent cut only has to slice the remaining front.
pub fn make_cuts(cuts: &[ExactPlane], pieces: Vec<PolyGeometry>) -> Result<Vec<PolyGeometry>> {
    let mut pieces = pieces;

    // Insertion-ordered: the order these are visited sets the order cuts are
    // applied, which is geometry, not float noise. Keyed by the normal's memo
    // form instead of its printed one.
    let mut planes: IndexMap<String, Vec<ExactPlane>> = IndexMap::new();
    let mut key = String::new();
    for p in cuts {
        key.clear();
        p.normal.write_key(&mut key);
        if let Some(v) = planes.get_mut(&key) {
            v.push(p.clone());
        } else {
            planes.insert(key.clone(), vec![p.clone()]);
        }
    }

    for (_, mut ps) in planes {
        let mut backpieces: Vec<PolyGeometry> = Vec::new();
        // Descending by constant, using the pinned sort: comparing algebraic
        // numbers can refine the shared field, so comparison order matters.
        crate::sort::sort_by(&mut ps, |a, b| Ok(-a.constant.compare(&b.constant)?))?;
        for cut in &ps {
            let mut frontpieces: Vec<PolyGeometry> = Vec::new();
            for piece in &pieces {
                let (frontpiece, backpiece) = slice_polygeometry(piece, cut, cut_color(), true)?;
                if !frontpiece.faces.is_empty() && !frontpiece.faces.iter().all(|f| f.interior) {
                    frontpieces.push(frontpiece);
                }
                if !backpiece.faces.is_empty() && !backpiece.faces.iter().all(|f| f.interior) {
                    backpieces.push(backpiece);
                }
            }
            pieces = frontpieces;
        }
        pieces.extend(backpieces);
    }

    Ok(pieces)
}
