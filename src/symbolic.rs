//! Numerical and symbolic state representations.
//!
//! The simulator's own notion of state is geometric: every piece carries its
//! exact vertices and an exact rotation. That is the right representation for
//! turning and drawing a puzzle and the wrong one for learning on it, where
//! what is wanted is a small integer array.
//!
//! This module supplies the usual one: a **sticker array**. Each exterior face
//! of the *solved* puzzle defines a slot, slots are numbered once, and a state
//! is the vector saying which sticker sits in each slot. For example, for the
//! 3x3x3: 54 slots, six colors, the solved array running `0,0,0,...,1,1,1,...`.
//!
//! It does not describe every puzzle, and says so rather than guessing when it
//! cannot. A puzzle that *jumbles* (such as a Radiolarian, a jumble prism, or the Big
//! Chop) has legal turns that leave pieces where no piece sits when the puzzle
//! is solved, so there is no fixed set of slots to number. Thirty-six of the
//! eighty-five cataloged puzzles do not jumble (see [`jumbles`] and
//! [`NON_JUMBLING`]).
//!
//! Two arrays are offered, because the two codebases differ in what they store:
//!
//! * [`StickerMap::colors`]: the color in each slot, `0..color_count`.
//! * [`StickerMap::permutation`]: which *slot* the sticker in each slot came
//!   from, `0..sticker_count`. It distinguishes states that the color array
//!   cannot (two identically colored stickers swapped).
//!
//! # Matching a sticker to its slot
//!
//! A sticker is located by the exact sum of its rotated vertices. The key for
//! that sum is built from the *coefficient vectors* of its coordinates over the
//! field's power basis, never from [`Display`](core::fmt::Display): a field
//! prints its own isolating interval, and `refine()` narrows that interval as a
//! computation proceeds, so the printed form of one unchanging number changes
//! over time. Coefficient vectors are canonical within a field, so two values
//! are equal exactly when their keys are.
//!
//! # When the table is built
//!
//! Numbering the slots requires the solved puzzle, so the table is built either
//! from a puzzle that has not moved yet or from a fresh build of the same
//! recipe. It is built lazily, on first use, and never by the drawing or
//! turning paths: deciding a sign can refine the shared field, and how far that
//! field has been refined is visible in the coordinates the renderer produces.

use core::fmt::Write as _;
use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::error::{Error, Result};
use crate::exact::AlgebraicNumber;
use crate::math::{ExactPlane, ExactVector3};
use crate::movement::Cut;
use crate::movement::Puzzle;
use crate::num::ring::Elem;
use crate::piece::{PolyFace, PolyGeometry};

/* -------------------------------------------------------------------------- */
/*  Exact keys                                                                */
/* -------------------------------------------------------------------------- */

/// Append the canonical text of one exact number: its coefficients over the
/// field's power basis, each as `numerator/denominator`.
///
/// Deliberately not `Display`, which embeds the field's isolating interval and
/// therefore changes as `refine()` narrows it.
fn write_exact(n: &AlgebraicNumber, out: &mut String) {
    for c in &n.poly.coeffs {
        let _ = write!(out, "{}/{},", c.n, c.d);
    }
    out.push(';');
}

/// A key identifying an exact point, tagged with `arity` so that two faces can
/// never collide on the strength of their vertex sums alone.
fn point_key(v: &ExactVector3, arity: usize) -> String {
    let mut s = String::with_capacity(64);
    let _ = write!(s, "{arity}:");
    write_exact(&v.x, &mut s);
    write_exact(&v.y, &mut s);
    write_exact(&v.z, &mut s);
    s
}

/// A key identifying an exact rotation, for the per-piece slot cache.
fn rotation_key(q: &crate::math::ExactQuaternion) -> String {
    let mut s = String::with_capacity(64);
    write_exact(&q.x, &mut s);
    write_exact(&q.y, &mut s);
    write_exact(&q.z, &mut s);
    write_exact(&q.w, &mut s);
    s
}

/// True for a quaternion that rotates nothing, whatever its magnitude.
fn is_identity(q: &crate::math::ExactQuaternion) -> bool {
    q.x.is_zero() && q.y.is_zero() && q.z.is_zero() && !q.w.is_zero()
}

/// Every vertex of a piece, in world coordinates.
fn world_vertices(p: &PolyGeometry) -> Result<Vec<ExactVector3>> {
    if is_identity(&p.rot) {
        return Ok(p.vertices.clone());
    }
    p.vertices.iter().map(|v| p.rot.apply(v)).collect()
}

/// The sum of a face's world vertices: its centroid, undivided.
fn face_sum(world: &[ExactVector3], face: &PolyFace) -> Result<ExactVector3> {
    let mut acc = world[face.vertices[0]].clone();
    for &i in &face.vertices[1..] {
        acc = acc.add(&world[i])?;
    }
    Ok(acc)
}

/* -------------------------------------------------------------------------- */
/*  The sticker map                                                           */
/* -------------------------------------------------------------------------- */

/// The solved layout of a puzzle's stickers: which slots exist, what color
/// each holds when solved, and which face of which piece is at home in it.
pub struct StickerMap {
    /// Slot index for the key of each solved sticker position.
    index: HashMap<String, u32>,
    /// Color index of the sticker at home in each slot.
    home_colors: Vec<u16>,
    /// Distinct colors, packed `0xRRGGBB`, in order of first appearance.
    palette: Vec<u32>,
    /// `home[piece][k]` is the slot occupied when solved by the piece's `k`-th
    /// exterior face, and `face_of[piece][k]` is that face's index.
    home: Vec<Vec<u32>>,
    face_of: Vec<Vec<u32>>,
}

impl StickerMap {
    /// Number the slots of a solved puzzle.
    ///
    /// # Errors
    ///
    /// If two stickers of the solved puzzle claim the same position, which
    /// would make the numbering ambiguous and every state read wrong.
    pub fn build(puzzle: &Puzzle) -> Result<StickerMap> {
        // Collect first, number afterward. Slots are numbered in color order
        // so that the solved array reads `0,0,...,1,1,...`, which is what
        // `arange(54) // 9` gives for the 3x3x3.
        let mut found: Vec<(String, u16, usize, usize)> = Vec::new();
        let mut palette: Vec<u32> = Vec::new();
        for (p, piece) in puzzle.pieces.iter().enumerate() {
            let world = world_vertices(piece)?;
            for (fi, face) in piece.faces.iter().enumerate() {
                if face.interior {
                    continue;
                }
                let hex = face.color.get_hex();
                let ci = if let Some(i) = palette.iter().position(|&c| c == hex) {
                    i
                } else {
                    palette.push(hex);
                    palette.len() - 1
                };
                let ci = u16::try_from(ci)
                    .map_err(|_| Error::Other("puzzle has too many colors".into()))?;
                found.push((
                    point_key(&face_sum(&world, face)?, face.vertices.len()),
                    ci,
                    p,
                    fi,
                ));
            }
        }
        u32::try_from(found.len())
            .map_err(|_| Error::Other("puzzle has too many stickers".into()))?;
        found.sort_by_key(|a| (a.1, a.2, a.3));

        let n = puzzle.pieces.len();
        let mut map = StickerMap {
            index: HashMap::with_capacity(found.len()),
            home_colors: Vec::with_capacity(found.len()),
            palette,
            home: vec![Vec::new(); n],
            face_of: vec![Vec::new(); n],
        };
        for (slot, (key, ci, p, fi)) in found.into_iter().enumerate() {
            let slot = slot as u32;
            if map.index.insert(key, slot).is_some() {
                return Err(Error::State(
                    "two stickers of the solved puzzle share a position, so slots cannot be \
                     numbered"
                        .into(),
                ));
            }
            map.home_colors.push(ci);
            map.home[p].push(slot);
            map.face_of[p].push(fi as u32);
        }
        Ok(map)
    }

    /// How many stickers, which sets how long every state vector is.
    pub fn sticker_count(&self) -> usize {
        self.home_colors.len()
    }

    /// How many distinct colors the puzzle's stickers take.
    pub fn color_count(&self) -> usize {
        self.palette.len()
    }

    /// The distinct sticker colors, packed `0xRRGGBB`, indexed by color.
    pub fn palette(&self) -> &[u32] {
        &self.palette
    }

    /// The color in each slot when the puzzle is solved.
    pub fn solved_colors(&self) -> &[u16] {
        &self.home_colors
    }

    /// The slot of every exterior face of the solved puzzle, and that face's
    /// index within its piece.
    pub fn home_slots(&self, piece: usize) -> (&[u32], &[u32]) {
        (&self.home[piece], &self.face_of[piece])
    }

    /// Where the sticker in each slot came from: `out[slot]` is the slot that
    /// sticker occupies when the puzzle is solved.
    ///
    /// # Errors
    ///
    /// If a sticker has left the lattice of solved positions, which a puzzle
    /// that jumbles can do and a doctrinaire one cannot.
    pub fn permutation(&self, puzzle: &Puzzle, cache: &mut SlotCache) -> Result<Vec<u32>> {
        let mut out = vec![u32::MAX; self.home_colors.len()];
        for (p, piece) in puzzle.pieces.iter().enumerate() {
            let now = cache.slots_of(self, piece, p)?;
            for (k, &slot) in now.iter().enumerate() {
                out[slot as usize] = self.home[p][k];
            }
        }
        if let Some(i) = out.iter().position(|&x| x == u32::MAX) {
            return Err(Error::State(format!(
                "slot {i} is empty: the puzzle's stickers no longer sit on the solved lattice"
            )));
        }
        Ok(out)
    }

    /// The color in each slot.
    ///
    /// The array a network is fed, with one categorical value per sticker.
    ///
    /// # Errors
    ///
    /// As [`permutation`](Self::permutation).
    pub fn colors(&self, puzzle: &Puzzle, cache: &mut SlotCache) -> Result<Vec<u16>> {
        let perm = self.permutation(puzzle, cache)?;
        Ok(perm
            .into_iter()
            .map(|h| self.home_colors[h as usize])
            .collect())
    }

    /// Whether every sticker is the color it should be.
    ///
    /// Colors, not identities: two identically colored stickers may be swapped.
    ///
    /// # Errors
    ///
    /// As [`permutation`](Self::permutation).
    pub fn is_solved(&self, puzzle: &Puzzle, cache: &mut SlotCache) -> Result<bool> {
        Ok(self.colors(puzzle, cache)? == self.home_colors)
    }

    /// The slot each exterior face of `piece` occupies now.
    fn slots_now(&self, piece: &PolyGeometry, p: usize) -> Result<Vec<u32>> {
        let world = world_vertices(piece)?;
        let mut out = Vec::with_capacity(self.home[p].len());
        for &fi in &self.face_of[p] {
            let face = &piece.faces[fi as usize];
            let key = point_key(&face_sum(&world, face)?, face.vertices.len());
            let slot = self.index.get(&key).copied().ok_or_else(|| {
                Error::State(format!(
                    "face {fi} of piece {p} is not at any solved sticker position; this puzzle \
                     does not return to its own lattice"
                ))
            })?;
            out.push(slot);
        }
        Ok(out)
    }
}

/// Memoizes the slots a piece's faces occupy, keyed by the piece's exact rotation.
///
/// A piece takes few distinct orientations, so after a short warm up reading a
/// whole state costs one hash lookup per piece rather than a rotation per
/// vertex.
#[derive(Default)]
pub struct SlotCache {
    entries: HashMap<(usize, String), Arc<[u32]>>,
}

impl SlotCache {
    /// Create an empty slot cache.
    pub fn new() -> SlotCache {
        SlotCache::default()
    }

    /// Forget everything, used when the map itself is rebuilt.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn slots_of(&mut self, map: &StickerMap, piece: &PolyGeometry, p: usize) -> Result<Arc<[u32]>> {
        if is_identity(&piece.rot) {
            return Ok(Arc::from(map.home[p].as_slice()));
        }
        let key = (p, rotation_key(&piece.rot));
        if let Some(v) = self.entries.get(&key) {
            return Ok(Arc::clone(v));
        }
        let v: Arc<[u32]> = Arc::from(map.slots_now(piece, p)?.as_slice());
        self.entries.insert(key, Arc::clone(&v));
        Ok(v)
    }
}

/* -------------------------------------------------------------------------- */
/*  Actions                                                                   */
/* -------------------------------------------------------------------------- */

/// The names of a puzzle's turns, and the exact planes they turn about.
///
/// Grips are re-derived after every move, so an action cannot be a grip index:
/// it is a plane, matched exactly against the grips the puzzle currently has.
/// A grip that has gone (a layer locked by a bandaged state) makes the
/// action unavailable rather than turning some other layer by accident.
pub struct ActionTable {
    names: Vec<String>,
    planes: Vec<ExactPlane>,
}

impl ActionTable {
    /// Name the grips of a solved puzzle.
    ///
    /// Grips sharing an axis share a letter and are numbered outward in the
    /// order the simulator lists them, and an axis with a single grip is the letter
    /// alone. Turning a grip the way its arrow points is the bare name, and the
    /// other way is the name with a `'`.
    ///
    /// # Errors
    ///
    /// If a grip's plane has a zero normal, which cannot happen for a plane
    /// that came out of the cut finder.
    pub fn build(grips: &[Cut]) -> Result<ActionTable> {
        let mut axes: IndexMap<String, Vec<usize>> = IndexMap::new();
        for (i, g) in grips.iter().enumerate() {
            axes.entry(direction_key(&g.plane.normal)?)
                .or_default()
                .push(i);
        }
        let mut names = vec![String::new(); grips.len()];
        for (ai, members) in axes.values().enumerate() {
            let letter = axis_letter(ai);
            if members.len() == 1 {
                names[members[0]] = letter;
            } else {
                for (li, &m) in members.iter().enumerate() {
                    names[m] = format!("{letter}{}", li + 1);
                }
            }
        }
        Ok(ActionTable {
            names,
            planes: grips.iter().map(|g| g.plane.clone()).collect(),
        })
    }

    /// How many grips the solved puzzle has.
    pub fn grip_count(&self) -> usize {
        self.names.len()
    }

    /// How many actions: two per grip, one each way.
    pub fn action_count(&self) -> usize {
        self.names.len() * 2
    }

    /// The name of grip `i`.
    pub fn grip_name(&self, i: usize) -> Option<&str> {
        self.names.get(i).map(String::as_str)
    }

    /// Every grip name, in order.
    pub fn grip_names(&self) -> &[String] {
        &self.names
    }

    /// The name of action `i`, where action `2k` turns grip `k` forward and
    /// action `2k + 1` turns it back.
    pub fn action_name(&self, i: usize) -> Option<String> {
        let name = self.names.get(i / 2)?;
        Some(if i % 2 == 0 {
            name.clone()
        } else {
            format!("{name}'")
        })
    }

    /// Every action name, in index order.
    pub fn action_names(&self) -> Vec<String> {
        (0..self.action_count())
            .map(|i| self.action_name(i).unwrap_or_default())
            .collect()
    }

    /// The action index of a move name such as `"A"` or `"B2'"`.
    pub fn action_index(&self, name: &str) -> Option<usize> {
        let (base, dir) = split_direction(name);
        let g = self.names.iter().position(|n| n == base)?;
        Some(g * 2 + usize::from(dir < 0))
    }

    /// The plane action `i` turns about, and which way.
    pub fn action_plane(&self, i: usize) -> Option<(&ExactPlane, i32)> {
        let plane = self.planes.get(i / 2)?;
        Some((plane, if i % 2 == 0 { 1 } else { -1 }))
    }

    /// The name of the grip turning about `plane`, if this puzzle has one.
    ///
    /// The inverse of [`locate`](Self::locate): it turns a grip the simulator
    /// is holding now back into the name the solved puzzle gave it.
    pub fn name_of_plane(&self, plane: &ExactPlane) -> Option<&str> {
        let i = self.planes.iter().position(|p| same_plane(p, plane))?;
        self.names.get(i).map(String::as_str)
    }

    /// Where the grip this action turns sits in `grips` right now, if it is
    /// still there.
    pub fn locate(&self, i: usize, grips: &[Cut]) -> Option<usize> {
        let plane = self.planes.get(i / 2)?;
        grips.iter().position(|g| same_plane(&g.plane, plane))
    }
}

/// Exact equality of two planes, component by component.
///
/// Structural, so it neither refines the shared field nor depends on how far it
/// has already been refined.
pub fn same_plane(a: &ExactPlane, b: &ExactPlane) -> bool {
    Elem::equals(&a.normal.x, &b.normal.x)
        && Elem::equals(&a.normal.y, &b.normal.y)
        && Elem::equals(&a.normal.z, &b.normal.z)
        && Elem::equals(&a.constant, &b.constant)
}

/// A key for a direction that ignores length but not sense, so that two cuts
/// through parallel planes share an axis and two facing the opposite way do
/// not.
fn direction_key(v: &ExactVector3) -> Result<String> {
    let c = if v.x.is_zero() {
        if v.y.is_zero() {
            &v.z
        } else {
            &v.y
        }
    } else {
        &v.x
    };
    if c.is_zero() {
        return Err(Error::State("a grip has a degenerate axis".into()));
    }
    let unit = ExactVector3::new(
        Elem::div(&v.x, c)?,
        Elem::div(&v.y, c)?,
        Elem::div(&v.z, c)?,
    );
    let sign = if c.sign()? < 0 { '-' } else { '+' };
    let mut s = String::with_capacity(64);
    s.push(sign);
    s.push_str(&point_key(&unit, 0));
    Ok(s)
}

/// `A`, `B`, ... `Z`, `AA`, `AB`, ... (spreadsheet column names).
fn axis_letter(mut i: usize) -> String {
    let mut out = Vec::new();
    loop {
        out.push(b'A' + (i % 26) as u8);
        if i < 26 {
            break;
        }
        i = i / 26 - 1;
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

/// Split a trailing `'` off a move name, returning the direction it means.
fn split_direction(name: &str) -> (&str, i32) {
    match name.strip_suffix('\'') {
        Some(base) => (base, -1),
        None => (name, 1),
    }
}

/// One move of a sequence: which action, and how many times.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    pub action: usize,
    pub repeat: u32,
}

/// Parse a whitespace-separated move sequence such as `"A B2' A"`.
///
/// A name may carry a repeat count, so `A2` is `A` twice and `A2'` is `A'`
/// twice. Counts bind tighter than the `'`, which is how cube notation reads.
///
/// # Errors
///
/// If a name is not one of the puzzle's grips, or its repeat count is not a
/// positive number.
pub fn parse_moves(table: &ActionTable, text: &str) -> Result<Vec<Move>> {
    let mut out = Vec::new();
    for word in text.split_whitespace() {
        out.push(parse_move(table, word)?);
    }
    Ok(out)
}

/// Parse a single move such as `"B2'"`.
///
/// # Errors
///
/// As [`parse_moves`].
pub fn parse_move(table: &ActionTable, word: &str) -> Result<Move> {
    let (base, dir) = split_direction(word);
    if let Some(a) = table.action_index(word) {
        return Ok(Move {
            action: a,
            repeat: 1,
        });
    }
    // `<name><count>`: peel the trailing digits off and try again, but only if
    // what is left is itself a grip: `A12` is grip `A1` twice, not grip `A` a
    // dozen times, when both grips exist.
    let digits = base.len() - base.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    for take in 1..=digits {
        let split = base.len() - take;
        let (head, tail) = base.split_at(split);
        let Some(g) = table.names.iter().position(|n| n == head) else {
            continue;
        };
        let repeat: u32 = tail
            .parse()
            .map_err(|_| Error::Parse(format!("move '{word}' has an unreadable repeat count")))?;
        if repeat == 0 {
            return Err(Error::Parse(format!("move '{word}' repeats zero times")));
        }
        return Ok(Move {
            action: g * 2 + usize::from(dir < 0),
            repeat,
        });
    }
    Err(Error::Parse(format!(
        "'{word}' is not a move of this puzzle; its grips are {}",
        table.names.join(", ")
    )))
}

/* -------------------------------------------------------------------------- */
/*  Which puzzles a sticker array can describe                                */
/* -------------------------------------------------------------------------- */

/// Whether any turn of `recipe` moves a sticker off the solved lattice.
///
/// A puzzle *jumbles* when a legal turn leaves a piece somewhere no piece sits
/// when the puzzle is solved (the Radiolarians, the jumble prisms, the Big
/// Chop). For those there is no fixed set of sticker slots to number, so there
/// is no sticker array either, and [`StickerMap::colors`] reports which sticker
/// left rather than inventing a slot for it. Thirty-six of the eighty-five
/// cataloged puzzles do not jumble, and [`NON_JUMBLING`] lists them.
///
/// This builds and turns a puzzle of its own, so it disturbs nothing, and it is
/// a probe rather than a proof: it tries every single turn from solved and then
/// a forty-move walk.
///
/// # Errors
///
/// If the recipe does not describe a puzzle.
pub fn jumbles(recipe: &str) -> Result<bool> {
    let mut probe = crate::simulator::Simulator::from_query(recipe)?;
    let count = probe.symbolic()?.actions.action_count();
    if count == 0 {
        return Ok(false);
    }
    for a in 0..count {
        if probe.apply_action(a)? {
            if probe.stickers().is_err() {
                return Ok(true);
            }
            if !probe.apply_action(a ^ 1)? || probe.stickers().is_err() {
                return Ok(true);
            }
        }
    }
    // SplitMix64, so the walk is the same every time this is asked.
    let mut state: u64 = 0x5DEE_CE66_D1CE_4005;
    for _ in 0..40 {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let a = ((z ^ (z >> 31)) as usize).rem_euclid(count);
        if probe.apply_action(a)? && probe.stickers().is_err() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The cataloged recipes that do not jumble, and so have a sticker array.
///
/// Derived by [`jumbles`] and pinned here so that asking the question costs
/// nothing, and `every_non_jumbling_entry_is_listed` re-derives it.
pub const NON_JUMBLING: &[&str] = &[
    "?shell=T$1&cut=T$0",
    "?shell=T$1&cut=T$-1/3",
    "?shell=T$1&cut=T$-1/3&cut=T$-5/3",
    "?shell=T$1&cut=T$-7/12",
    "?shell=T$1&cut=T$-2&cut=T$-1&cut=T$0",
    "?shell=C$1&cut=C$0",
    "?shell=C$1&cut=C$1/3",
    "?shell=C$1&cut=C$0&cut=C$1/2",
    "?shell=C$1&cut=C$1/5&cut=C$3/5",
    "?shell=C$1&cut=C$0&cut=C$1/3&cut=C$2/3",
    "?shell=C$1&cut=C$1/7&cut=C$3/7&cut=C$5/7",
    "?shell=C$sqrt(2)/3&cut=O$0",
    "?shell=C$sqrt(2)/3&cut=O$1/6",
    "?shell=C$sqrt(2)/3&cut=O$1/3",
    "?shell=C$sqrt(2)/3&cut=O$1/2",
    "?shell=O$1&cut=O$0",
    "?shell=O$1&cut=O$1/3",
    "?shell=O$1&cut=O$1/2",
    "?shell=O$1&cut=O$0&cut=O$1/2",
    "?shell=O$1&cut=O$1/5&cut=O$3/5",
    "?shell=O$sqrt(2)/2&cut=C$0",
    "?shell=O$sqrt(2)/2&cut=C$1/3",
    "?shell=O$sqrt(2)/2&cut=C$1/3&cut=C$2/3",
    "?shell=D$1&cut=D$0",
    "?shell=D$1&cut=D$sqrt(5)-2",
    "?shell=D$1&cut=D$1/sqrt(5)",
    "?shell=D$1&cut=D$2/(sqrt(5)+1)",
    "?shell=D$1&cut=I$(5-sqrt(5))/2",
    "?shell=I$sqrt(5)+2&cut=D$0",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/10",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/6",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/4",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/2",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/2+1/2",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/2+2/3",
    "?shell=I$sqrt(5)+2&cut=D$(sqrt(5)+1)/2+2/3&cut=D$(sqrt(5)+1)/2+4/3",
];

/* -------------------------------------------------------------------------- */
/*  Moves as permutations                                                     */
/* -------------------------------------------------------------------------- */

/// Every move of a puzzle, as a permutation of its sticker slots.
///
/// Turning a puzzle geometrically is expensive: a move re-derives the whole cut
/// structure, which for a 3x3x3 is about two milliseconds. For a puzzle whose
/// moves are fixed permutations (which is every puzzle that does not jumble),
/// the same turn is a gather, and a state can be stepped in nanoseconds.
///
/// Each move is applied once to the solved puzzle, the resulting slot permutation
/// is recorded, and the move is taken back.
///
/// `new[i] = old[moves[a][i]]`.
pub struct PermutationTable {
    moves: Vec<Vec<u32>>,
    solved: Vec<u16>,
}

impl PermutationTable {
    /// Derive the table, or report that this puzzle has no such table.
    ///
    /// `None` means a move was not available, or a state could not be read, or
    /// undoing a move did not restore the puzzle, each of which says the
    /// moves are not fixed permutations, and that the geometry has to be
    /// turned move by move instead.
    ///
    /// # Errors
    ///
    /// If turning the puzzle fails outright.
    pub fn build(sim: &mut crate::simulator::Simulator) -> Result<Option<PermutationTable>> {
        let view = sim.symbolic()?;
        let n = view.actions.action_count();
        let Ok(solved) = sim.stickers() else {
            return Ok(None);
        };
        let mut moves = Vec::with_capacity(n);
        for a in 0..n {
            if !sim.apply_action(a)? {
                return Ok(None);
            }
            let Ok(ids) = sim.sticker_ids() else {
                return Ok(None);
            };
            // A layer that locks after one turn means availability depends on
            // the state, so a fixed table would be a lie. Cheap to ask: the
            // grips have just been re-derived anyway.
            if !sim.action_mask()?.iter().all(|&ok| ok) {
                sim.undo()?;
                return Ok(None);
            }
            moves.push(ids);
            if !sim.undo()? || sim.stickers().as_deref() != Ok(solved.as_slice()) {
                return Ok(None);
            }
        }
        Ok(Some(PermutationTable { moves, solved }))
    }

    /// How many moves the table covers.
    pub fn action_count(&self) -> usize {
        self.moves.len()
    }

    /// How long a state vector is.
    pub fn sticker_count(&self) -> usize {
        self.solved.len()
    }

    /// The colors of a solved puzzle.
    pub fn solved(&self) -> &[u16] {
        &self.solved
    }

    /// The permutation move `a` applies.
    pub fn permutation(&self, a: usize) -> Option<&[u32]> {
        self.moves.get(a).map(Vec::as_slice)
    }

    /// Every permutation, in action order.
    pub fn permutations(&self) -> &[Vec<u32>] {
        &self.moves
    }

    /// Check the table against the geometry over a walk of `steps` moves.
    ///
    /// The table is derived from single moves out of the solved state, which
    /// only shows that each move *is* a permutation there. This turns the
    /// puzzle both ways at once and requires them to agree, which is what
    /// makes the fast path a checked property rather than an assumption.
    ///
    /// Leaves the puzzle where it found it.
    ///
    /// # Errors
    ///
    /// If turning the puzzle fails.
    pub fn verify(
        &self,
        sim: &mut crate::simulator::Simulator,
        steps: usize,
        seed: u64,
    ) -> Result<bool> {
        let mut state = self.solved.clone();
        let mut scratch = Vec::with_capacity(state.len());
        let mut rng = seed | 1;
        let mut made = 0usize;
        let mut ok = true;
        for _ in 0..steps {
            rng = rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let a = ((z ^ (z >> 31)) as usize).rem_euclid(self.moves.len());
            if !sim.apply_action(a)? {
                ok = false;
                break;
            }
            made += 1;
            self.apply(&mut state, a, &mut scratch)?;
            if sim.stickers().as_deref() != Ok(state.as_slice()) {
                ok = false;
                break;
            }
        }
        for _ in 0..made {
            sim.undo()?;
        }
        Ok(ok)
    }

    /// Apply move `a` to `state`, in place.
    ///
    /// # Errors
    ///
    /// If `a` is not a move of this puzzle.
    pub fn apply(&self, state: &mut [u16], a: usize, scratch: &mut Vec<u16>) -> Result<()> {
        let perm = self
            .moves
            .get(a)
            .ok_or_else(|| Error::Range(format!("action {a} out of range")))?;
        scratch.clear();
        scratch.extend_from_slice(state);
        for (i, &from) in perm.iter().enumerate() {
            state[i] = scratch[from as usize];
        }
        Ok(())
    }
}

/* -------------------------------------------------------------------------- */
/*  Ground atoms                                                              */
/* -------------------------------------------------------------------------- */

/// The name of the predicate relating a slot to the color it holds.
pub const COLOR_PREDICATE: &str = "color";

/// A state as ground atoms: `color(s<slot>, c<color>)` for every slot.
///
/// An atom is a predicate followed by its arguments, all of them strings),
/// so a state can be handed to a solver or compared against a partial goal
/// without going through the array.
pub fn ground_atoms(colors: &[u16]) -> Vec<(String, String, String)> {
    colors
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            (
                COLOR_PREDICATE.to_string(),
                format!("s{i}"),
                format!("c{c}"),
            )
        })
        .collect()
}
