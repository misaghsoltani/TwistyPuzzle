//! Cuts, stops and moves.
//!
//! The `planes` and `stops` maps are iterated, so they are insertion-ordered
//! (`IndexMap`), where the order decides which stop a turn lands on. See
//! `SEMANTICS.md` §2. The two caches are lookup-only and may be hashed.
//!
//! Every key here comes from `write_key`, which names a value by its field's
//! identity and its polynomial. Deliberately not the `Display` form: that
//! spells the field out as `Q(root of .. in [lower,upper])`, and those bounds
//! are the field's *current* isolating interval, which `refine()` narrows as
//! the computation proceeds. A key built from them is both large and moving,
//! so entries stop being found shortly after they are inserted.

use core::fmt::Write as _;
use std::collections::HashMap;

use indexmap::IndexMap;

use crate::exact::AlgebraicNumber;
use crate::math::{ExactPlane, ExactQuaternion, Quat};
use crate::num::ring::Elem;
use crate::piece::PolyGeometry;
use crate::Result;

pub struct Puzzle {
    pub pieces: Vec<PolyGeometry>,
    pub global_rot: Quat,
    extent_cache: HashMap<String, (AlgebraicNumber, AlgebraicNumber)>,
    side_cache: HashMap<String, i32>,
}

impl Puzzle {
    pub fn new(pieces: Vec<PolyGeometry>) -> Puzzle {
        Puzzle {
            pieces,
            global_rot: Quat::IDENTITY,
            extent_cache: HashMap::new(),
            side_cache: HashMap::new(),
        }
    }
}

impl Default for Puzzle {
    fn default() -> Self {
        Puzzle::new(Vec::new())
    }
}

#[derive(Clone)]
pub struct Cut {
    pub plane: ExactPlane,
    pub front: Vec<usize>,
    pub back: Vec<usize>,
}

impl Cut {
    pub fn new(plane: ExactPlane, front: Vec<usize>, back: Vec<usize>) -> Cut {
        Cut { plane, front, back }
    }

    pub fn neg(&self) -> Cut {
        Cut::new(self.plane.neg(), self.back.clone(), self.front.clone())
    }
}

/// Which side of `plane` a piece lies on: +1 front, -1 back, 0 straddling.
///
/// `plane_str`, `normal_str` and `rot_str` are the `Display` forms of the
/// plane, its normal, and the piece's rotation. They make up the cache keys
/// and are passed in already rendered: the plane is fixed across the inner
/// loop and a piece's rotation is fixed across the whole call, so building
/// them here would re-stringify an entire number field per piece per plane.
fn get_side(
    puzzle: &mut Puzzle,
    piece_num: usize,
    plane: &ExactPlane,
    plane_str: &str,
    normal_str: &str,
    rot_str: &str,
) -> Result<i32> {
    let side_key = cache_key(plane_str, piece_num, rot_str);
    if let Some(&s) = puzzle.side_cache.get(&side_key) {
        return Ok(s);
    }

    let extent_key = cache_key(normal_str, piece_num, rot_str);
    let (xmin, xmax) = if let Some(v) = puzzle.extent_cache.get(&extent_key) {
        v.clone()
    } else {
        // Rotate the axis backward instead of rotating every vertex.
        let piece = &puzzle.pieces[piece_num];
        let axis = piece.rot.conj().apply(&plane.normal)?;
        let mut xmin: Option<AlgebraicNumber> = None;
        let mut xmax: Option<AlgebraicNumber> = None;
        for v in &piece.vertices {
            let x = axis.dot(v)?;
            if xmin
                .as_ref()
                .map(|m| x.compare(m))
                .transpose()?
                .unwrap_or(-1)
                < 0
            {
                xmin = Some(x.clone());
            }
            if xmax
                .as_ref()
                .map(|m| x.compare(m))
                .transpose()?
                .unwrap_or(1)
                > 0
            {
                xmax = Some(x);
            }
        }
        let v = (
            xmin.expect("a piece always has at least one vertex"),
            xmax.expect("a piece always has at least one vertex"),
        );
        puzzle.extent_cache.insert(extent_key, v.clone());
        v
    };

    let d = Elem::neg(&plane.constant);
    let side = if d.compare(&xmin)? <= 0 {
        1
    } else if d.compare(&xmax)? >= 0 {
        -1
    } else {
        0
    };
    puzzle.side_cache.insert(side_key, side);
    Ok(side)
}

/// `${prefix}/${piece},${rot}`: the cache key, spelled out.
fn cache_key(prefix: &str, piece_num: usize, rot_str: &str) -> String {
    let mut k = String::with_capacity(prefix.len() + rot_str.len() + 10);
    k.push_str(prefix);
    k.push('/');
    let _ = write!(k, "{piece_num}");
    k.push(',');
    k.push_str(rot_str);
    k
}

/// The `Display` form of each piece's rotation, computed once per call.
fn rotation_strings(puzzle: &Puzzle, piece_nums: &[usize]) -> Vec<Option<String>> {
    let mut out = vec![None; puzzle.pieces.len()];
    for &p in piece_nums {
        if out[p].is_none() {
            let mut k = String::new();
            puzzle.pieces[p].rot.write_key(&mut k);
            out[p] = Some(k);
        }
    }
    out
}

/// Every plane that touches but does not intersect the given pieces.
pub fn find_cuts(puzzle: &mut Puzzle, piece_nums: Option<&[usize]>) -> Result<Vec<Cut>> {
    let piece_nums: Vec<usize> = match piece_nums {
        Some(p) => p.to_vec(),
        None => (0..puzzle.pieces.len()).collect(),
    };

    // Every exterior face normal is a potential axis.
    let mut planes: IndexMap<String, ExactPlane> = IndexMap::new();
    // One reused buffer: most faces of most pieces name a plane that has
    // already been seen, and only a new one needs an owned key.
    let mut key = String::new();
    for piece in &puzzle.pieces {
        for face in &piece.faces {
            if face.interior {
                let normal = piece.rot.apply(&face.plane.normal)?;
                let plane = ExactPlane::new(normal, face.plane.constant.clone()).canonicalize()?;
                key.clear();
                plane.write_key(&mut key);
                if !planes.contains_key(key.as_str()) {
                    planes.insert(key.clone(), plane);
                }
            }
        }
    }

    let plane_list: Vec<(String, ExactPlane)> = planes.into_iter().collect();
    let rot_strs = rotation_strings(puzzle, &piece_nums);
    let mut cuts: Vec<Cut> = Vec::new();
    for (plane_str, plane) in plane_list {
        let mut normal_str = String::new();
        plane.normal.write_key(&mut normal_str);
        let mut is_cut = true;
        let mut front: Vec<usize> = Vec::new();
        let mut back: Vec<usize> = Vec::new();
        for &p in &piece_nums {
            let rot_str = rot_strs[p].as_deref().unwrap_or_default();
            let side = get_side(puzzle, p, &plane, &plane_str, &normal_str, rot_str)?;
            match side.cmp(&0) {
                core::cmp::Ordering::Less => back.push(p),
                core::cmp::Ordering::Greater => front.push(p),
                core::cmp::Ordering::Equal => {
                    is_cut = false;
                    break;
                },
            }
        }
        if is_cut && !front.is_empty() && !back.is_empty() {
            cuts.push(Cut::new(plane, front, back));
        }
    }
    Ok(cuts)
}

/// Every rotation about `cut` that leaves the puzzle in a fully cut state,
/// ordered by increasing angle.
pub fn find_stops(puzzle: &mut Puzzle, cut: &Cut) -> Result<Vec<ExactQuaternion>> {
    let c = cut.plane.normal.clone();
    let front_cuts = find_cuts(puzzle, Some(&cut.front))?;
    let back_cuts = find_cuts(puzzle, Some(&cut.back))?;

    // All rotation angles that form a total cut.
    let mut stops: IndexMap<String, ExactQuaternion> = IndexMap::new();
    for half1 in &front_cuts {
        for half2 in &back_cuts {
            let (p1, p2) = (&half1.plane.normal, &half2.plane.normal);
            let (d1, d2) = (&half1.plane.constant, &half2.plane.constant);
            let h1 = c.dot(p1)?;
            let h2 = c.dot(p2)?;

            // Skip planes parallel to the cut.
            if Elem::equals(
                &Elem::mul(&h1, &h1)?,
                &Elem::mul(&c.dot(&c)?, &p1.dot(p1)?)?,
            ) {
                continue;
            }

            // Can half1.plane be rotated onto half2.plane (or its negation)?
            if Elem::equals(d1, d2) && Elem::equals(&h1, &h2) {
                let rot = ExactQuaternion::from_axis_points(&c, p1, p2)?;
                let mut k = String::new();
                rot.write_key(&mut k);
                stops.insert(k, rot);
            }
            if Elem::equals(&Elem::neg(d1), d2) && Elem::equals(&Elem::neg(&h1), &h2) {
                let rot = ExactQuaternion::from_axis_points(&c, p1, &p2.neg())?;
                let mut k = String::new();
                rot.write_key(&mut k);
                stops.insert(k, rot);
            }
        }
    }

    let mut ret: Vec<(AlgebraicNumber, ExactQuaternion)> = Vec::new();
    for rot in stops.into_values() {
        ret.push((rot.pseudo_angle()?, rot));
    }
    crate::sort::sort_by(&mut ret, |a, b| a.0.compare(&b.0))?;
    Ok(ret.into_iter().map(|x| x.1).collect())
}

/// Apply `rot` to the front half of `cut`.
///
/// Piece 0 is immovable: a move that would turn it instead turns the rest of
/// the puzzle the other way and accumulates the difference into `global_rot`.
pub fn make_move(puzzle: &mut Puzzle, cut: &Cut, rot: &ExactQuaternion) -> Result<()> {
    let mut rot = rot.clone();
    let mut cut = cut.clone();
    if cut.front.contains(&0) {
        puzzle.global_rot = puzzle.global_rot.mul(&rot.to_f64()?).normalize();
        rot = rot.conj();
        cut = cut.neg();
    }

    for &p in &cut.front {
        let new_rot = rot.mul(&puzzle.pieces[p].rot)?.pseudo_normalize()?;
        puzzle.pieces[p].rot = new_rot;
    }
    Ok(())
}
