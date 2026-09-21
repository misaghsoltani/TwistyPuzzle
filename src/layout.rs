//! A puzzle's stickers numbered face by face, and the moves over them.
//!
//! The simulator numbers a puzzle's sticker slots in the order its geometry
//! produces them, which is a good numbering in general and an unfamiliar one
//! for a cube in particular. The literature on cubes numbers *facelets*
//! instead: six faces in a fixed order, each read as a grid, so that a solved
//! cube is `0,0,0,...,1,1,1,...` and a move is a permutation anyone can check
//! against a cube in their hands.
//!
//! [`Layout`] is that numbering, for any puzzle. Where a standard convention
//! exists it is used, and where none does the puzzle's own order is, because
//! `SEMANTICS.md` §2 already makes that order canonical: slots are numbered in
//! color order, so a solved state reads `0,0,...,1,1,...` across all puzzles.
//!
//! # A cube
//!
//! Derived from the puzzle's geometry instead of assumed:
//!
//! * **Faces** are numbered `U, D, L, R, B, F`, with `U` at `+y`, `R` at `+x`
//!   and `F` at `+z`.
//! * **Within a face**, facelets are read row-major, rows running along `ROW`
//!   and columns along `COL`, where `ROW x COL` is the outward normal:
//!   `U (+x, -z)`, `D (+x, +z)`, `L (+z, +y)`, `R (-z, +y)`, `B (-x, +y)`,
//!   `F (+x, +y)`.
//! * **Moves** start with the quarter turns of the six faces, face-major in
//!   that same face order, `2k` turning face `k` counterclockwise seen from
//!   outside and `2k + 1` clockwise, named `U-1` and `U1`. The plain names `U`
//!   and `U'` mean the same two. Any further turn the puzzle has (such as an inner
//!   slice of a 4x4x4) follows them, under the puzzle's own name with
//!   a colon in front of it, because a cube names its grips `A` to `F` too
//!   and `F1` would otherwise mean two different turns.
//!
//! Every one of those is a convention, and `tests/layout.rs` checks this
//! module against recorded permutations instead of against itself.
//!
//! # Any other puzzle
//!
//! * **Faces** are the puzzle's sticker colors, in the puzzle's own order.
//! * **Facelets** are its slots, in the puzzle's own order: the identity, so
//!   that a state means the same thing on both sides of the translation.
//! * **Moves** are the puzzle's actions, in its own order and under its own
//!   names.
//!
//! # In either case
//!
//! A state is either the **color** in each facelet, where the color of a
//! facelet is the face it belongs to when solved, so the goal of a 3x3x3 is
//! `arange(54) // 9`. Or the **identity** of the sticker in it, where the goal
//! is `arange(54)`. The second distinguishes states that identical colors cannot differentiate.
//!
//! # What it is for
//!
//! Two things. A state can be handed to and taken from anything that speaks
//! the usual layout, without either side knowing how the other numbers its
//! slots. And, because the moves are permutations over a small array, a
//! [`Layout`] is a complete engine for its puzzle on its own: [`next_states`]
//! steps a whole batch of states without a simulator anywhere in sight.
//!
//! The conventions above are a contract, not an implementation detail:
//! `SEMANTICS.md` §12 states them and says what may not change.
//!
//! [`next_states`]: Layout::next_states

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::Error;
use crate::simulator::Simulator;
use crate::symbolic::PermutationTable;
use crate::Result;

/// A cube's faces, in the order they are numbered.
pub const FACES: [&str; 6] = ["U", "D", "L", "R", "B", "F"];

/// Each face's outward normal, in face order.
const NORMAL: [[f64; 3]; 6] = [
    [0.0, 1.0, 0.0],  // U
    [0.0, -1.0, 0.0], // D
    [-1.0, 0.0, 0.0], // L
    [1.0, 0.0, 0.0],  // R
    [0.0, 0.0, -1.0], // B
    [0.0, 0.0, 1.0],  // F
];

/// The direction a face's rows run in.
const ROW: [[f64; 3]; 6] = [
    [1.0, 0.0, 0.0],  // U
    [1.0, 0.0, 0.0],  // D
    [0.0, 0.0, 1.0],  // L
    [0.0, 0.0, -1.0], // R
    [-1.0, 0.0, 0.0], // B
    [1.0, 0.0, 0.0],  // F
];

/// The direction a face's columns run in. `ROW x COL` is the outward normal.
const COL: [[f64; 3]; 6] = [
    [0.0, 0.0, -1.0], // U
    [0.0, 0.0, 1.0],  // D
    [0.0, 1.0, 0.0],  // L
    [0.0, 1.0, 0.0],  // R
    [0.0, 1.0, 0.0],  // B
    [0.0, 1.0, 0.0],  // F
];

/// One character per face, for a puzzle whose faces have no letters.
const GLYPHS: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// How far a coordinate may drift from where the lattice puts it and still be
/// recognized. The coordinates come from exact arithmetic rounded once to the
/// nearest double, so the real error is at the last bit. This is loose enough
/// not to care and tight enough that two facelets can never be confused.
const EPS: f64 = 1.0e-9;

/// What may sit in a facelet: a face number, a color, or a sticker identity.
///
/// The width is the caller's to choose, because it is the caller who knows how
/// many distinct values there will be (`SEMANTICS.md` §13). A face of a 3x3x3 fits a byte and a
/// slot of a 20x20x20 does not, and permuting an array does not care either
/// way: the same code runs at every width, so nothing is paid for the reach of
/// a puzzle nobody built.
pub trait Cell: Copy + Send + Sync + 'static {
    /// Zero, which is a face, a color and a slot that all exist.
    const ZERO: Self;
    /// How many distinct values this width holds, which is what a caller
    /// asking for it has to fit inside.
    const SPAN: u64;
    /// The value `v`, which the caller has already bounded.
    fn from_index(v: usize) -> Self;
    /// What it is worth, for indexing.
    fn index(self) -> usize;
}

macro_rules! cell_for {
    ($($t:ty),*) => {$(
        impl Cell for $t {
            const ZERO: Self = 0;
            const SPAN: u64 = 1u64 << (8 * std::mem::size_of::<$t>());
            #[inline]
            fn from_index(v: usize) -> Self {
                v as $t
            }
            #[inline]
            fn index(self) -> usize {
                self as usize
            }
        }
    )*};
}

cell_for!(u8, u16, u32);

/// The grid a cube's faces are read as.
#[derive(Clone, Copy)]
struct Grid {
    /// Facelets along one side of a face.
    size: usize,
    /// Half the cube's edge, in puzzle coordinates.
    half: f64,
}

/// A puzzle's stickers, numbered face by face, and the moves over them.
pub struct Layout {
    /// The puzzle this describes, as a canonical query. Kept so that a
    /// caller handing a layout and a batch to the same call can be told when
    /// the two are not the same puzzle, instead of quietly getting an array
    /// of the right length and the wrong meaning.
    recipe: String,
    /// Set for a cube, which is read as a grid and turned by named faces.
    grid: Option<Grid>,
    /// How many faces there are: colors, which for a cube is six.
    face_count: usize,
    /// `facelet_of_slot[s]` is where the simulator's slot `s` sits in the
    /// face-by-face numbering, and `slot_of_facelet` is the way back. Both
    /// are the identity for a puzzle read in its own order.
    facelet_of_slot: Vec<u32>,
    slot_of_facelet: Vec<u32>,
    /// The face each of the puzzle's own sticker colors belongs to, and back.
    face_of_color: Vec<u16>,
    color_of_face: Vec<u16>,
    /// How many facelets each face carries. Equal for a cube, and not in
    /// general: a prism's square sides and triangular ends are both faces.
    face_sizes: Vec<u32>,
    /// What each face is called.
    face_names: Vec<String>,
    /// The face each facelet belongs to when solved.
    goal: Vec<u32>,
    /// `moves[m][i] = j`: after move `m`, facelet `i` holds what facelet `j`
    /// held.
    moves: Vec<Vec<u32>>,
    names: Vec<String>,
    /// The simulator's own action index for each move, when it has one.
    actions: Vec<Option<u32>>,
    /// The move that undoes each move, when this layout has one.
    inverse: Vec<Option<u32>>,
}

impl Layout {
    /// Work out the layout of a puzzle, from the puzzle itself.
    ///
    /// # Errors
    ///
    /// If the puzzle jumbles, and so has no fixed permutation for a move
    /// (unless it is a cube, whose face turns are geometry and hold whatever
    /// the rest of it does).
    pub fn build(sim: &mut Simulator) -> Result<Layout> {
        let recipe = sim.query()?;
        let table = PermutationTable::build(sim)?;
        let view = sim.symbolic()?;
        let at = view.stickers.positions();
        let count = at.len();
        let colors = view.stickers.color_count();
        let solved = view.stickers.solved_colors();

        let grid = cube_grid(at, colors);
        let mut layout = match grid {
            Some(grid) => Layout::on_a_cube(grid, at, solved, colors)?,
            None => Layout::as_numbered(count, solved, colors),
        };
        layout.recipe = recipe;
        layout.derive_moves(&view.actions, table.as_ref())?;
        layout.derive_inverses();
        Ok(layout)
    }

    /// The layout of a cube: six named faces, each read as a grid.
    fn on_a_cube(grid: Grid, at: &[[f64; 3]], solved: &[u16], colors: usize) -> Result<Layout> {
        let count = at.len();
        let per_face = grid.size * grid.size;
        let mut facelet_of_slot = vec![u32::MAX; count];
        let mut slot_of_facelet = vec![u32::MAX; count];
        for (slot, &p) in at.iter().enumerate() {
            // `cube_grid` has already placed every sticker, so this cannot
            // fail. It is written out instead of unwrapped so that a change
            // to one of the two is caught by the other.
            let f = locate(p, grid.size, grid.half)
                .ok_or_else(|| Error::State(format!("sticker {slot} is not on the face of a cube")))?;
            if slot_of_facelet[f] != u32::MAX {
                return Err(Error::State(format!(
                    "two stickers fall on facelet {f}, so this is not a cube"
                )));
            }
            facelet_of_slot[slot] = f as u32;
            slot_of_facelet[f] = slot as u32;
        }

        // Each face carries one color and each color one face, or an array
        // whose values are face numbers would not describe the puzzle.
        let mut face_of_color = vec![u16::MAX; colors];
        let mut color_of_face = vec![u16::MAX; 6];
        for (slot, &c) in solved.iter().enumerate() {
            let face = (facelet_of_slot[slot] as usize / per_face) as u16;
            let held = &mut face_of_color[c as usize];
            if *held != u16::MAX && *held != face {
                return Err(Error::State(format!(
                    "color {c} is on more than one face, so a face cannot stand for a color"
                )));
            }
            *held = face;
            color_of_face[face as usize] = c;
        }

        Ok(Layout {
            recipe: String::new(),
            grid: Some(grid),
            face_count: 6,
            facelet_of_slot,
            slot_of_facelet,
            face_of_color,
            color_of_face,
            face_sizes: vec![per_face as u32; 6],
            face_names: FACES.iter().map(|&s| s.to_string()).collect(),
            goal: (0..count).map(|i| (i / per_face) as u32).collect(),
            moves: Vec::new(),
            names: Vec::new(),
            actions: Vec::new(),
            inverse: Vec::new(),
        })
    }

    /// The layout of everything else: the puzzle's own numbering, kept.
    ///
    /// Slots are already numbered in color order (`SEMANTICS.md` §2), so
    /// grouping them by face is what the puzzle has already done and the
    /// translation is the identity. Saying it in full anyway, instead of
    /// special-casing it away, is what lets everything below be written once.
    fn as_numbered(count: usize, solved: &[u16], colors: usize) -> Layout {
        let mut face_sizes = vec![0u32; colors];
        for &c in solved {
            face_sizes[c as usize] += 1;
        }
        Layout {
            recipe: String::new(),
            grid: None,
            face_count: colors,
            facelet_of_slot: (0..count as u32).collect(),
            slot_of_facelet: (0..count as u32).collect(),
            face_of_color: (0..colors as u16).collect(),
            color_of_face: (0..colors as u16).collect(),
            face_sizes,
            face_names: (0..colors).map(|f| f.to_string()).collect(),
            goal: solved.iter().map(|&c| u32::from(c)).collect(),
            moves: Vec::new(),
            names: Vec::new(),
            actions: Vec::new(),
            inverse: Vec::new(),
        }
    }

    /// Every move, as a permutation of the facelets.
    ///
    /// A cube's six faces come first, in the standard order, worked out from
    /// the geometry and then *matched* against the puzzle's own actions:
    /// requiring the two to agree is what makes the layout a description of
    /// this puzzle instead of an assumption about it. Whatever the puzzle can do
    /// besides follows, under its own name.
    fn derive_moves(&mut self, actions: &crate::symbolic::ActionTable, table: Option<&PermutationTable>) -> Result<()> {
        let mut taken = vec![false; actions.action_count()];
        if self.grid.is_some() {
            self.derive_face_turns()?;
            if let Some(table) = table {
                self.actions = vec![None; self.moves.len()];
                for (a, slots) in table.permutations().iter().enumerate() {
                    let as_facelets = self.slots_to_facelets(slots);
                    for (m, perm) in self.moves.iter().enumerate() {
                        if *perm == as_facelets {
                            self.actions[m] = Some(a as u32);
                            taken[a] = true;
                        }
                    }
                }
            } else {
                self.actions = vec![None; self.moves.len()];
            }
        } else if table.is_none() {
            return Err(Error::State(
                "this puzzle's moves are not fixed permutations of its stickers (it jumbles, \
                 or a layer of it locks), so there is nothing to number face by face"
                    .into(),
            ));
        }

        let names = actions.action_names();
        // A cube numbers its own grips `A` to `F` as well, so `F1` would name
        // two different turns of a 4x4x4 (the face and the slice behind it).
        // A turn the cube conventions do not name therefore keeps the
        // puzzle's name with a colon in front of it, which no face turn's
        // name can start with, so that every move in a layout is named once.
        let own = if self.grid.is_some() { ":" } else { "" };
        if let Some(table) = table {
            for (a, slots) in table.permutations().iter().enumerate() {
                if taken[a] {
                    continue;
                }
                let perm = self.slots_to_facelets(slots);
                self.moves.push(perm);
                self.names.push(format!("{own}{}", names[a]));
                self.actions.push(Some(a as u32));
            }
        }
        Ok(())
    }

    /// A cube's twelve quarter turns, as permutations of the facelets.
    fn derive_face_turns(&mut self) -> Result<()> {
        let grid = self.grid.expect("a cube");
        let count = self.facelet_count();
        let cell = 2.0 * grid.half / grid.size as f64;
        for face in 0..6 {
            for &sign in &[-1.0f64, 1.0] {
                let n = NORMAL[face];
                let mut perm: Vec<u32> = (0..count as u32).collect();
                for (i, slot) in perm.iter_mut().enumerate() {
                    let p = self.position(i);
                    // The outer layer on this face, which is what a face turn
                    // takes with it.
                    if dot(p, n) <= grid.half - cell {
                        continue;
                    }
                    // Where the facelet now at `i` came from: the turn runs
                    // one way, so reading it backward runs the other.
                    let from = quarter_turn(n, sign, p);
                    *slot = locate(from, grid.size, grid.half).ok_or_else(|| {
                        Error::State(format!("turning face {} sent facelet {i} off the cube", FACES[face]))
                    })? as u32;
                }
                self.moves.push(perm);
                let dir = if sign < 0.0 { -1 } else { 1 };
                self.names.push(format!("{}{dir}", FACES[face]));
            }
        }
        Ok(())
    }

    /// Which move undoes which, worked out instead of assumed.
    ///
    /// A cube's quarter turns come in pairs, so `m ^ 1` would do for them, and
    /// nothing else here is that tidy: a puzzle numbers its actions as it
    /// likes, and a macro-move is undone by another macro only if the caller
    /// happened to include it. Composing each move with each candidate would
    /// be quadratic over the thousands of macros a caller may ask for, so
    /// every permutation is indexed once and each move's inverse looked up.
    fn derive_inverses(&mut self) {
        let mut seen: HashMap<&[u32], u32> = HashMap::with_capacity(self.moves.len());
        for (m, perm) in self.moves.iter().enumerate() {
            seen.entry(perm.as_slice()).or_insert(m as u32);
        }
        let k = self.facelet_count();
        let mut back = vec![0u32; k];
        self.inverse = self
            .moves
            .iter()
            .enumerate()
            .map(|(m, perm)| {
                for (i, &from) in perm.iter().enumerate() {
                    back[from as usize] = i as u32;
                }
                // A move that undoes itself (a half turn, or one that moves
                // nothing) answers with itself instead of whichever
                // other move happens to have the same permutation.
                if back == *perm {
                    return Some(m as u32);
                }
                seen.get(back.as_slice()).copied()
            })
            .collect();
    }

    /// A permutation of slots, rewritten as one of facelets.
    fn slots_to_facelets(&self, perm: &[u32]) -> Vec<u32> {
        (0..perm.len())
            .map(|f| {
                let slot = self.slot_of_facelet[f] as usize;
                self.facelet_of_slot[perm[slot] as usize]
            })
            .collect()
    }

    /// Where facelet `i` sits in space, for a cube.
    fn position(&self, i: usize) -> [f64; 3] {
        let grid = self.grid.expect("a cube");
        let per_face = grid.size * grid.size;
        let (face, rest) = (i / per_face, i % per_face);
        let (r, c) = (rest / grid.size, rest % grid.size);
        let cell = 2.0 * grid.half / grid.size as f64;
        let a = -grid.half + (r as f64 + 0.5) * cell;
        let b = -grid.half + (c as f64 + 0.5) * cell;
        let (n, row, col) = (NORMAL[face], ROW[face], COL[face]);
        [
            n[0] * grid.half + row[0] * a + col[0] * b,
            n[1] * grid.half + row[1] * a + col[1] * b,
            n[2] * grid.half + row[2] * a + col[2] * b,
        ]
    }

    /* --- what it is ------------------------------------------------------ */

    /// How many facelets a face has along one side, for a cube, and `None`
    /// for a puzzle whose faces are not grids.
    pub fn size(&self) -> Option<usize> {
        self.grid.map(|g| g.size)
    }

    /// The puzzle this describes, as a canonical query string.
    pub fn recipe(&self) -> &str {
        &self.recipe
    }

    /// Whether this is a cube, read by the conventions cubes are read by.
    pub fn is_cube(&self) -> bool {
        self.grid.is_some()
    }

    /// How many facelets there are, which is how long a state is.
    pub fn facelet_count(&self) -> usize {
        self.goal.len()
    }

    /// How many faces there are, which is how many values a facelet can hold.
    pub fn face_count(&self) -> usize {
        self.face_count
    }

    /// How many colors the puzzle has, which is how many values a sticker
    /// array read out of it can hold. The same as the number of faces, since
    /// a face is a color, and named separately because the two are different
    /// things to count.
    pub fn color_count(&self) -> usize {
        self.face_of_color.len()
    }

    /// How many facelets each face carries.
    pub fn face_sizes(&self) -> &[u32] {
        &self.face_sizes
    }

    /// What each face is called.
    pub fn face_names(&self) -> &[String] {
        &self.face_names
    }

    /// How many moves there are.
    pub fn move_count(&self) -> usize {
        self.moves.len()
    }

    /// Every move's name, in move order.
    pub fn move_names(&self) -> &[String] {
        &self.names
    }

    /// Every move, as a permutation of the facelets.
    pub fn moves(&self) -> &[Vec<u32>] {
        &self.moves
    }

    /// The simulator's own action index for each move, where it has one.
    pub fn actions(&self) -> &[Option<u32>] {
        &self.actions
    }

    /// Where each of the simulator's slots sits in the face numbering.
    pub fn facelet_of_slot(&self) -> &[u32] {
        &self.facelet_of_slot
    }

    /// Which slot each facelet is, the other way around.
    pub fn slot_of_facelet(&self) -> &[u32] {
        &self.slot_of_facelet
    }

    /// The inverse of each move, where present in this layout.
    pub fn inverses(&self) -> &[Option<u32>] {
        &self.inverse
    }

    /// The inverse action for each move in `moves`.
    ///
    /// # Errors
    ///
    /// If a move is out of range, or this layout does not contain the move
    /// that would undo it, which only a hand-picked set of macro-moves can
    /// fail to.
    pub fn inverse_moves(&self, moves: &[u32]) -> Result<Vec<u32>> {
        moves
            .iter()
            .map(|&m| {
                let held = self
                    .inverse
                    .get(m as usize)
                    .ok_or_else(|| Error::Range(format!("move {m} out of range (have {})", self.moves.len())))?;
                held.ok_or_else(|| Error::Range(format!("nothing in this layout undoes {}", self.names[m as usize])))
            })
            .collect()
    }

    /// The move a name stands for.
    ///
    /// A cube's faces answer to `"U1"`, `"U-1"`, `"U"` and `"U'"`, where `U`
    /// and `U1` are the clockwise quarter turn seen from outside the face.
    /// Every other move answers to the name it is listed under.
    pub fn move_index(&self, name: &str) -> Option<usize> {
        let name = name.trim();
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return Some(i);
        }
        self.grid?;
        let (face, rest) = name.split_at(name.chars().next()?.len_utf8());
        let f = FACES.iter().position(|&x| x == face)?;
        match rest {
            "" | "1" | "+1" => Some(f * 2 + 1),
            "'" | "-1" => Some(f * 2),
            _ => None,
        }
    }

    /// The same layout with a different set of moves: each of `sequences`
    /// composed into one permutation.
    ///
    /// Layouts can be configured with macro-moves instead of single turns (for example,
    /// all combinations of length three, expanding twelve base moves into 1,728 macro-moves).
    /// Composing permutations allows layout operations to apply directly over macro actions:
    /// `next_states` steps by a macro, `trajectories` walks in them, and the facelet numbering is untouched.
    ///
    /// The new moves are named by the moves they are made of. A macro is not
    /// one turn of the puzzle, so none of them is one of its actions, and
    /// [`inverse_moves`](Self::inverse_moves) answers only for the macros
    /// whose undoing is also in the set.
    ///
    /// # Errors
    ///
    /// If a sequence names a move this layout does not have, or there are no
    /// sequences at all.
    pub fn with_moves(&self, sequences: &[Vec<u32>]) -> Result<Layout> {
        if sequences.is_empty() {
            return Err(Error::Range("a layout needs at least one move".into()));
        }
        let k = self.facelet_count();
        let mut moves = Vec::with_capacity(sequences.len());
        let mut names = Vec::with_capacity(sequences.len());
        for parts in sequences {
            let mut perm: Vec<u32> = (0..k as u32).collect();
            let mut name = String::new();
            for &m in parts {
                let step = self
                    .moves
                    .get(m as usize)
                    .ok_or_else(|| Error::Range(format!("move {m} out of range (have {})", self.moves.len())))?;
                // `perm` says where each facelet's contents came from, so
                // composing means following the earlier permutation through
                // the later one.
                perm = step.iter().map(|&i| perm[i as usize]).collect();
                if !name.is_empty() {
                    name.push(' ');
                }
                name.push_str(&self.names[m as usize]);
            }
            moves.push(perm);
            names.push(if name.is_empty() { ".".into() } else { name });
        }
        let mut next = Layout {
            recipe: self.recipe.clone(),
            grid: self.grid,
            face_count: self.face_count,
            facelet_of_slot: self.facelet_of_slot.clone(),
            slot_of_facelet: self.slot_of_facelet.clone(),
            face_of_color: self.face_of_color.clone(),
            color_of_face: self.color_of_face.clone(),
            face_sizes: self.face_sizes.clone(),
            face_names: self.face_names.clone(),
            goal: self.goal.clone(),
            actions: vec![None; moves.len()],
            inverse: Vec::new(),
            moves,
            names,
        };
        next.derive_inverses();
        Ok(next)
    }

    /* --- the goal -------------------------------------------------------- */

    /// The face each facelet belongs to when solved, which for a 3x3x3 is
    /// `arange(54) // 9`.
    pub fn goal_colors<C: Cell>(&self) -> Vec<C> {
        self.goal.iter().map(|&f| C::from_index(f as usize)).collect()
    }

    /// The sticker each facelet holds when solved, which is the facelet
    /// itself: `arange(n)`.
    pub fn goal_ids<I: Cell>(&self) -> Vec<I> {
        (0..self.facelet_count()).map(I::from_index).collect()
    }

    /* --- translating states ---------------------------------------------- */

    /// Rewrite states from the simulator's slot order into face order, with
    /// each color replaced by the face it belongs to.
    ///
    /// `states` is `rows * facelet_count` colors as the simulator numbers them.
    /// The result is the same shape, in face order.
    ///
    /// # Errors
    ///
    /// If `states` is not a whole number of rows, or names a color this
    /// puzzle does not have.
    pub fn to_facelets<C: Cell>(&self, states: &[C]) -> Result<Vec<C>> {
        let k = self.facelet_count();
        Self::check(states.len(), k)?;
        if let Some(bad) = states.iter().find(|c| c.index() >= self.face_of_color.len()) {
            return Err(Error::Range(format!(
                "{} is not one of this puzzle's {} colors",
                bad.index(),
                self.face_of_color.len()
            )));
        }
        let mut out = vec![C::ZERO; states.len()];
        out.par_chunks_mut(k).zip(states.par_chunks(k)).for_each(|(dst, src)| {
            for (f, d) in dst.iter_mut().enumerate() {
                let slot = self.slot_of_facelet[f] as usize;
                *d = C::from_index(self.face_of_color[src[slot].index()] as usize);
            }
        });
        Ok(out)
    }

    /// Rewrite sticker identities from slot order into face order, with each
    /// identity named by the facelet it belongs to.
    ///
    /// # Errors
    ///
    /// As [`to_facelets`](Self::to_facelets), for slots.
    pub fn to_facelet_ids<I: Cell>(&self, ids: &[I]) -> Result<Vec<I>> {
        let k = self.facelet_count();
        Self::check(ids.len(), k)?;
        if let Some(bad) = ids.iter().find(|i| i.index() >= k) {
            return Err(Error::Range(format!(
                "{} is not one of this puzzle's {k} slots",
                bad.index()
            )));
        }
        let mut out = vec![I::ZERO; ids.len()];
        out.par_chunks_mut(k).zip(ids.par_chunks(k)).for_each(|(dst, src)| {
            for (f, d) in dst.iter_mut().enumerate() {
                let slot = self.slot_of_facelet[f] as usize;
                *d = I::from_index(self.facelet_of_slot[src[slot].index()] as usize);
            }
        });
        Ok(out)
    }

    /// Rewrite states in face order back into the simulator's slot order and
    /// its own colors: converts externally defined state representations into native simulator slot order.
    ///
    /// # Errors
    ///
    /// If `facelets` is not a whole number of rows, or names a face that does
    /// not exist.
    pub fn from_facelets<C: Cell>(&self, facelets: &[C]) -> Result<Vec<C>> {
        let k = self.facelet_count();
        Self::check(facelets.len(), k)?;
        if let Some(bad) = facelets.iter().find(|f| f.index() >= self.face_count) {
            return Err(Error::Range(format!(
                "{} is not one of this puzzle's {} faces",
                bad.index(),
                self.face_count
            )));
        }
        let mut out = vec![C::ZERO; facelets.len()];
        out.par_chunks_mut(k)
            .zip(facelets.par_chunks(k))
            .for_each(|(dst, src)| {
                for (f, &face) in src.iter().enumerate() {
                    let slot = self.slot_of_facelet[f] as usize;
                    dst[slot] = C::from_index(self.color_of_face[face.index()] as usize);
                }
            });
        Ok(out)
    }

    /* --- turning, with no puzzle in sight --------------------------------- */

    /// Turn a batch of states, one move each, entirely in the array.
    ///
    /// `states` is `rows * facelet_count` values and `moves` is one move per
    /// row. The values may be colors or identities: a move is a permutation of
    /// the facelets and does not care what is sitting in them.
    ///
    /// # Errors
    ///
    /// If the shapes disagree, or a move is out of range.
    pub fn next_states<C: Cell>(&self, states: &[C], moves: &[u32]) -> Result<Vec<C>> {
        let k = self.facelet_count();
        Self::check(states.len(), k)?;
        let rows = states.len() / k;
        if moves.len() != rows {
            return Err(Error::Range(format!("got {} moves for {rows} states", moves.len())));
        }
        self.check_moves(moves)?;
        let mut out = vec![C::ZERO; states.len()];
        out.par_chunks_mut(k)
            .zip(states.par_chunks(k))
            .zip(moves.par_iter())
            .for_each(|((dst, src), &m)| {
                for (d, &from) in dst.iter_mut().zip(&self.moves[m as usize]) {
                    *d = src[from as usize];
                }
            });
        Ok(out)
    }

    /// Walk `steps.len()` states away from solved and keep the whole path.
    ///
    /// Each row takes `steps[row]` turns, avoiding immediate inversion of the
    /// preceding move. Returns the states after every turn, laid out
    /// `rows * (steps + 1) * facelet_count` with the solved state first, and
    /// the moves that produced them, `rows * steps`. Rows asked for fewer
    /// steps than the longest simply stop, repeating their last state and
    /// leaving [`move_count`](Self::move_count) (one past the last real
    /// move) in place of a move, so the block stays rectangular.
    ///
    /// Generates complete trajectory data: initial states, terminal goal states,
    /// intermediate states, and applied action sequences.
    ///
    /// # Errors
    ///
    /// If this layout has no moves to walk in.
    pub fn trajectories<C: Cell>(&self, steps: &[u32], seed: u64) -> Result<(Vec<C>, Vec<u32>)> {
        let k = self.facelet_count();
        let rows = steps.len();
        let longest = steps.iter().copied().max().unwrap_or(0) as usize;
        let count = self.moves.len();
        if count == 0 && longest > 0 {
            return Err(Error::Range(
                "this puzzle has no moves, so there is nowhere to walk to".into(),
            ));
        }
        let goal: Vec<C> = self.goal_colors();
        let mut states = vec![C::ZERO; rows * (longest + 1) * k];
        let mut taken = vec![count as u32; rows * longest.max(1)];
        states
            .par_chunks_mut((longest + 1) * k)
            .zip(taken.par_chunks_mut(longest.max(1)))
            .zip(steps.par_iter())
            .enumerate()
            .for_each(|(row, ((path, picks), &depth))| {
                let mut rng = start_of(seed, row);
                path[..k].copy_from_slice(&goal);
                let mut last = u32::MAX;
                for t in 0..longest {
                    let (before, after) = path.split_at_mut((t + 1) * k);
                    let src = &before[t * k..];
                    let dst = &mut after[..k];
                    if (t as u32) < depth {
                        let m = self.pick(&mut rng, last);
                        for (d, &from) in dst.iter_mut().zip(&self.moves[m]) {
                            *d = src[from as usize];
                        }
                        picks[t] = m as u32;
                        last = m as u32;
                    } else {
                        dst.copy_from_slice(src);
                    }
                }
            });
        Ok((states, taken))
    }

    /// Walk states away from solved, keeping only where each ends up.
    ///
    /// The same walk [`trajectories`](Self::trajectories) makes, without the
    /// path: start states are frequently sampled in large batches and at
    /// deep horizons, avoiding the memory overhead of storing full trajectory
    /// histories when only terminal states are required.
    ///
    /// # Errors
    ///
    /// As [`trajectories`](Self::trajectories).
    pub fn walk<C: Cell>(&self, steps: &[u32], seed: u64) -> Result<Vec<C>> {
        let k = self.facelet_count();
        let count = self.moves.len();
        if count == 0 && steps.iter().any(|&d| d > 0) {
            return Err(Error::Range(
                "this puzzle has no moves, so there is nowhere to walk to".into(),
            ));
        }
        let goal: Vec<C> = self.goal_colors();
        let mut out = vec![C::ZERO; steps.len() * k];
        out.par_chunks_mut(k.max(1))
            .zip(steps.par_iter())
            .enumerate()
            .for_each_init(
                || vec![C::ZERO; k],
                |scratch, (row, (state, &depth))| {
                    state.copy_from_slice(&goal);
                    if count == 0 {
                        return;
                    }
                    // Seeded exactly as `trajectories` seeds its rows, so the
                    // same seed and depths give the same states either way.
                    let mut rng = start_of(seed, row);
                    let mut last = u32::MAX;
                    for _ in 0..depth {
                        let m = self.pick(&mut rng, last);
                        scratch.copy_from_slice(state);
                        for (d, &from) in state.iter_mut().zip(&self.moves[m]) {
                            *d = scratch[from as usize];
                        }
                        last = m as u32;
                    }
                },
            );
        Ok(out)
    }

    /* --- reading a state -------------------------------------------------- */

    /// A batch of states as indicator bytes, `rows * facelet_count *
    /// face_count`.
    ///
    /// # Errors
    ///
    /// As [`to_facelets`](Self::to_facelets).
    pub fn one_hot<C: Cell>(&self, facelets: &[C]) -> Result<Vec<u8>> {
        let k = self.facelet_count();
        Self::check(facelets.len(), k)?;
        let c = self.face_count;
        let mut out = vec![0u8; facelets.len() * c];
        out.par_chunks_mut(k * c)
            .zip(facelets.par_chunks(k))
            .for_each(|(dst, src)| {
                for (i, face) in src.iter().enumerate() {
                    if face.index() < c {
                        dst[i * c + face.index()] = 1;
                    }
                }
            });
        Ok(out)
    }

    /// Whether each state is solved: every facelet holding its own face.
    ///
    /// # Errors
    ///
    /// As [`to_facelets`](Self::to_facelets).
    pub fn solved<C: Cell>(&self, facelets: &[C]) -> Result<Vec<bool>> {
        let k = self.facelet_count();
        Self::check(facelets.len(), k)?;
        Ok(facelets
            .par_chunks(k)
            .map(|row| row.iter().zip(&self.goal).all(|(v, &f)| v.index() == f as usize))
            .collect())
    }

    /// One state written out, a line per face, as a net would read.
    pub fn text<C: Cell>(&self, facelets: &[C]) -> String {
        let width = self.grid.map_or(usize::MAX, |g| g.size);
        let mut out = String::new();
        let mut at = 0usize;
        for (face, name) in self.face_names.iter().enumerate() {
            if face > 0 {
                out.push('\n');
            }
            out.push_str(name);
            out.push(' ');
            let size = self.face_sizes[face] as usize;
            for i in 0..size {
                if i > 0 && i % width == 0 {
                    out.push('/');
                }
                let v = facelets.get(at + i).map_or(usize::MAX, |c| c.index());
                out.push(self.glyph(v));
            }
            at += size;
        }
        out
    }

    /// The character a face is drawn with in [`text`](Self::text).
    fn glyph(&self, face: usize) -> char {
        if self.grid.is_some() {
            return FACES.get(face).map_or('?', |f| f.as_bytes()[0] as char);
        }
        GLYPHS.get(face).map_or('?', |&b| b as char)
    }

    /// Select a move that does not invert the preceding move.
    fn pick(&self, state: &mut u64, last: u32) -> usize {
        let count = self.moves.len();
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let a = ((z ^ (z >> 31)) as usize).rem_euclid(count);
        // Which move undoes the last one is a property of the puzzle, not of
        // how its moves happen to be numbered, so it is looked up.
        let undo = if last == u32::MAX {
            None
        } else {
            self.inverse[last as usize]
        };
        if count > 2 && undo == Some(a as u32) {
            (a + 1) % count
        } else {
            a
        }
    }

    fn check(len: usize, k: usize) -> Result<()> {
        if k == 0 || len % k != 0 {
            return Err(Error::Range(format!(
                "a state of this puzzle is {k} values, and {len} is not a whole number of them"
            )));
        }
        Ok(())
    }

    fn check_moves(&self, moves: &[u32]) -> Result<()> {
        if let Some(&bad) = moves.iter().find(|&&m| m as usize >= self.moves.len()) {
            return Err(Error::Range(format!(
                "move {bad} out of range (have {})",
                self.moves.len()
            )));
        }
        Ok(())
    }
}

/// Where a row's walk starts, from the seed and which row it is.
fn start_of(seed: u64, row: usize) -> u64 {
    seed.wrapping_add(row as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1
}

/// The grid a puzzle's stickers form, if they form a cube's.
///
/// Six colors, a square number of stickers on each of six axis-aligned faces,
/// and every sticker on a lattice point of that grid. Anything else is read in
/// the puzzle's own order instead, which is why this answers `None` rather
/// than refusing.
fn cube_grid(at: &[[f64; 3]], colors: usize) -> Option<Grid> {
    let count = at.len();
    if count == 0 || colors != 6 || count % 6 != 0 {
        return None;
    }
    let per_face = count / 6;
    let size = (per_face as f64).sqrt().round() as usize;
    if size == 0 || size * size != per_face {
        return None;
    }
    let half = at.iter().flat_map(|p| p.iter().map(|v| v.abs())).fold(0.0f64, f64::max);
    if half <= EPS {
        return None;
    }
    let mut seen = vec![false; count];
    for &p in at {
        let f = locate(p, size, half)?;
        if seen[f] {
            return None;
        }
        seen[f] = true;
    }
    Some(Grid { size, half })
}

/// Which facelet a point on the cube's surface is, or `None` if it is not one.
fn locate(p: [f64; 3], size: usize, half: f64) -> Option<usize> {
    let cell = 2.0 * half / size as f64;
    for face in 0..6 {
        if (dot(p, NORMAL[face]) - half).abs() > EPS {
            continue;
        }
        let a = (dot(p, ROW[face]) + half - cell * 0.5) / cell;
        let b = (dot(p, COL[face]) + half - cell * 0.5) / cell;
        let (r, c) = (a.round(), b.round());
        if (a - r).abs() > 1.0e-6 || (b - c).abs() > 1.0e-6 {
            return None;
        }
        let (r, c) = (r as isize, c as isize);
        if r < 0 || c < 0 || r >= size as isize || c >= size as isize {
            return None;
        }
        return Some(face * size * size + r as usize * size + c as usize);
    }
    None
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Turn `p` a quarter turn about the unit axis `a`, counterclockwise for
/// `sign = 1` and clockwise for `sign = -1` seen from the `+a` side.
///
/// Written as `sign * (a x p) + a (a . p)` instead of through a trigonometric
/// matrix, because the axes here are the coordinate axes and this makes a
/// quarter turn a signed swap of coordinates: exact, with no `sin` of a right
/// angle rounding to something that is not zero.
fn quarter_turn(a: [f64; 3], sign: f64, p: [f64; 3]) -> [f64; 3] {
    let c = cross(a, p);
    let d = dot(a, p);
    [sign * c[0] + a[0] * d, sign * c[1] + a[1] * d, sign * c[2] + a[2] * d]
}
