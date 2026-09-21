//! Many puzzles at once: building, turning, scrambling and drawing them
//! together, across every core, without going back to the caller in between.
//!
//! Each puzzle owns its number field outright, so two puzzles share no state
//! and can be built on different threads, which is what [`build_many`] and
//! [`render_many`] do.
//!
//! Within a single puzzle the numbers share one isolating interval, so the
//! work is not split across threads there. Nothing about the arithmetic
//! forbids it, because a conversion refines to the correctly rounded double
//! either way (`SEMANTICS.md` §1, §9), but nothing is structured for it
//! either.
//!
//! # Why a batch is not a list of puzzles
//!
//! [`PuzzleBatch`] keeps many states of *one* puzzle and steps them together.
//! Two things make that worth having, and both are about not paying per
//! puzzle for what can be paid once:
//!
//! * **The moves are shared.** Every copy of one recipe numbers its slots the
//!   same way and turns by the same permutations, so the geometry is derived
//!   once for the batch instead of once for each of its members. A batch of
//!   a thousand 3x3x3s costs one build, not a thousand.
//! * **A turn stops being geometry.** For a puzzle whose moves are fixed
//!   permutations of its slots (every puzzle that does not jumble), a state is
//!   an array and a move is a gather: nanoseconds, against the two
//!   milliseconds re-deriving the cut structure costs. A whole batch of turns
//!   is then one call with the interpreter released, run across every core.
//!
//! A puzzle that does jumble, or one whose layers can lock, has no such table
//! and falls back to turning each copy through its own geometry. The two paths
//! present the same interface and [`PuzzleBatch::is_tabular`] says which is in
//! use.

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::Error;
use crate::layout::Cell;
use crate::render::camera::Camera;
use crate::render::lut::StickerLut;
use crate::render::Framebuffer;
use crate::simulator::{Simulator, SymbolicView};
use crate::symbolic::PermutationTable;
use crate::Result;

/// How a batch of puzzles should be prepared and photographed.
#[derive(Clone, Copy, Debug)]
pub struct RenderSpec {
    pub width: u32,
    pub height: u32,
    pub background: [u8; 4],
    /// Supersampling factor, where 1 disables antialiasing.
    pub supersample: u32,
    pub show_arrows: bool,
    pub show_edges: bool,
    /// Random moves to apply before rendering.
    pub scramble: u32,
    /// Seed for those moves. Each puzzle is seeded with `seed + its index`, so
    /// a batch is reproducible however the work happens to be scheduled.
    pub seed: Option<u64>,
    /// Camera distance, where `None` keeps the default (12 units).
    pub distance: Option<f64>,
    /// Degrees to swing the camera sideways from the head-on view.
    pub yaw: f64,
    /// Degrees to raise the camera above the head-on view.
    pub pitch: f64,
}

impl Default for RenderSpec {
    fn default() -> Self {
        RenderSpec {
            width: 512,
            height: 512,
            background: [255, 255, 255, 255],
            supersample: 2,
            show_arrows: false,
            show_edges: true,
            scramble: 0,
            seed: None,
            distance: None,
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

/// Build every recipe, in parallel, keeping the order of the input.
pub fn build_many(recipes: &[String]) -> Vec<Result<Simulator>> {
    recipes.par_iter().map(|r| Simulator::from_query(r)).collect()
}

/// Build, scramble and render every recipe, in parallel.
///
/// One puzzle per task instead of one image band per task: the puzzles are
/// the independent unit, and the renderer is itself parallel, so nesting the
/// two lets rayon keep every core busy on a batch of any shape.
pub fn render_many(recipes: &[String], spec: &RenderSpec) -> Vec<Result<Framebuffer>> {
    recipes
        .par_iter()
        .enumerate()
        .map(|(i, recipe)| render_one(recipe, spec, i as u64))
        .collect()
}

fn render_one(recipe: &str, spec: &RenderSpec, index: u64) -> Result<Framebuffer> {
    let mut sim = Simulator::from_query(recipe)?;
    apply_spec(&mut sim, spec);
    if spec.scramble > 0 {
        sim.seed(spec.seed.unwrap_or(0).wrapping_add(index));
        sim.scramble(spec.scramble);
        sim.settle()?;
    }
    Ok(sim.render(spec.width, spec.height))
}

/// Put a spec's view settings on a simulator.
fn apply_spec(sim: &mut Simulator, spec: &RenderSpec) {
    {
        let opts = sim.options_mut();
        opts.background = spec.background;
        opts.supersample = spec.supersample.max(1);
        opts.draw_arrows = spec.show_arrows;
        opts.draw_edges = spec.show_edges;
    }
    if spec.yaw != 0.0 || spec.pitch != 0.0 {
        sim.look_from(spec.yaw, spec.pitch, spec.distance.unwrap_or(12.0));
    } else if let Some(d) = spec.distance {
        sim.set_distance(d);
    }
}

/// Worker threads available to the batch functions.
///
/// This is rayon's global pool, shared with the renderer's band-parallel
/// rasterizer. Set `RAYON_NUM_THREADS` to change it.
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/* -------------------------------------------------------------------------- */
/*  Batched state                                                             */
/* -------------------------------------------------------------------------- */

/// What one step did to one puzzle.
#[derive(Clone, Copy, Debug)]
pub struct StepOutcome {
    /// Whether the move could be made at all.
    pub applied: bool,
    /// Whether the puzzle is now solved.
    pub solved: bool,
}

/// Where a batch keeps its states.
enum Backend {
    /// Every move is a fixed permutation of the slots, so a state is an array
    /// of slot numbers and a turn is a gather. Boxed because it carries a
    /// whole simulator and the other variant carries a pointer.
    Table(Tabular),
    /// Some layer locks, or the puzzle jumbles, so each copy has to be turned
    /// through its own geometry.
    Geometry(Vec<Simulator>),
}

/// A table, at the narrowest width its puzzle's slots fit in.
///
/// A state is one slot number per sticker, and almost every puzzle has fewer
/// than 256 stickers, so a byte holds one. Stepping a batch is memory-bound
/// (since a gather reads a permutation and a state and writes a state), so the
/// width is most of what a move costs, and a 3x3x3 stepped as bytes moves a
/// quarter of the memory it would as words. A puzzle large enough to need
/// them still gets words. Nothing is capped, only narrowed
/// (`SEMANTICS.md` §13).
///
/// Boxed because each carries a whole simulator and the other backend
/// carries a pointer.
enum Tabular {
    N8(Box<Table<u8>>),
    N16(Box<Table<u16>>),
    N32(Box<Table<u32>>),
}

/// Run the same code over whichever width a table happens to be.
macro_rules! tabular {
    ($t:expr, |$x:ident| $body:expr) => {
        match $t {
            Tabular::N8($x) => $body,
            Tabular::N16($x) => $body,
            Tabular::N32($x) => $body,
        }
    };
}

/// The array-and-gather representation.
struct Table<I: Cell> {
    /// A solved copy, kept for what the arrays cannot answer: the picture.
    model: Simulator,
    /// `states[r * k + i]` is the slot the sticker now in slot `i` of row `r`
    /// started in. Identities instead of colors, because colors follow from
    /// identities and not the other way around: two stickers of one color can
    /// be swapped, and only the identities say so.
    states: Vec<I>,
    /// A second buffer, so a gather reads one and writes the other instead of
    /// copying a row aside first.
    scratch: Vec<I>,
    /// The permutations, flattened to `action_count * k` so one move is one
    /// contiguous run.
    perms: Vec<I>,
    solved: Vec<u16>,
}

impl<I: Cell> Table<I> {
    /// A table of `count` solved copies.
    fn new(model: Simulator, table: &PermutationTable, count: usize, k: usize) -> Table<I> {
        let mut perms = Vec::with_capacity(table.action_count() * k);
        for p in table.permutations() {
            perms.extend(p.iter().map(|&v| I::from_index(v as usize)));
        }
        let mut states = Vec::with_capacity(count * k);
        for _ in 0..count {
            states.extend((0..k).map(I::from_index));
        }
        Table {
            model,
            scratch: states.clone(),
            states,
            perms,
            solved: table.solved().to_vec(),
        }
    }
}

/// Many states of one puzzle, stepped together.
///
///     >>> # (the Python side of this is `twistypuzzle.PuzzleBatch`)
pub struct PuzzleBatch {
    recipe: String,
    view: Arc<SymbolicView>,
    backend: Backend,
    /// One generator per row, so a scramble is reproducible however the batch
    /// is scheduled.
    rng: Vec<u64>,
    /// The last action taken by each row, so a walk does not step back over
    /// itself. `NONE` when there has not been one.
    last: Vec<u32>,
    /// Every move each row has made since it was last solved, in order.
    ///
    /// Four bytes a move, which is nothing beside the states, and it is the
    /// only way back to the geometry: a tabular batch keeps arrays, so the
    /// only way to put a *puzzle* into a row's position is to turn one
    /// through the same moves.
    history: Vec<Vec<u32>>,
    /// The drawing tables, most recently used first, and what each was built
    /// for. More than one because an image observation is often two views of
    /// the same puzzle, and alternating between them must not rebuild a table
    /// every frame.
    luts: Vec<(LutKey, StickerLut)>,
}

/// How many drawing tables a batch keeps before dropping the least recently
/// used. Enough for a handful of viewpoints, few enough that a table for a
/// large frame is not kept by accident.
const LUT_CACHE: usize = 4;

/// How a batch's frames should come out.
#[derive(Clone, Copy, Debug)]
pub struct FrameSpec {
    pub width: u32,
    pub height: u32,
    /// 3 for RGB, 4 to keep the alpha channel.
    pub channels: usize,
    /// Channels before rows and columns, `(n, c, h, w)`, as a convolutional
    /// network wants them, instead of `(n, h, w, c)`.
    pub channels_first: bool,
}

impl FrameSpec {
    pub fn new(width: u32, height: u32) -> FrameSpec {
        FrameSpec {
            width,
            height,
            channels: 4,
            channels_first: false,
        }
    }

    /// Values in one frame.
    fn len(&self) -> usize {
        self.width as usize * self.height as usize * self.channels
    }
}

/// No action yet, in the slot where one would be recorded.
const NONE: u32 = u32::MAX;

/// What a drawing table was built for. Anything here changing means the table
/// no longer describes the picture and has to be built again.
#[derive(Clone, Copy, PartialEq)]
struct LutKey {
    width: u32,
    height: u32,
    supersample: u32,
    background: [u8; 4],
    line_width: u64,
    draw_edges: bool,
    draw_pieces: bool,
    camera: [u64; 10],
}

impl PuzzleBatch {
    /// Build `count` copies of `recipe`.
    ///
    /// The geometry is derived once: if the puzzle's moves are fixed
    /// permutations, one puzzle is built and the rest of the batch is arrays,
    /// and only a puzzle that has no such table costs one build per copy.
    ///
    /// # Errors
    ///
    /// If the recipe does not build, or the puzzle has no sticker numbering.
    pub fn build(recipe: &str, count: usize) -> Result<PuzzleBatch> {
        if count == 0 {
            return Err(Error::Range("a batch needs at least one puzzle".into()));
        }
        let mut model = Simulator::from_query(recipe)?;
        let view = model.symbolic()?;
        let table = PermutationTable::build(&mut model)?;
        let k = view.stickers.sticker_count();

        // A batch draws states, not controls. The arrows are a thing to click
        // on, they are blended instead of written, and a drawing table
        // cannot express them, so neither backend draws them and the two
        // agree about what a frame of a state looks like.
        model.options_mut().draw_arrows = false;
        let backend = if let Some(table) = table {
            // The narrowest width that names every slot. A state and a
            // permutation are both slot numbers, so both narrow together.
            Backend::Table(if k <= 1 << 8 {
                Tabular::N8(Box::new(Table::new(model, &table, count, k)))
            } else if k <= 1 << 16 {
                Tabular::N16(Box::new(Table::new(model, &table, count, k)))
            } else {
                Tabular::N32(Box::new(Table::new(model, &table, count, k)))
            })
        } else {
            {
                let recipes = vec![recipe.to_string(); count];
                let mut sims: Vec<Simulator> = build_many(&recipes).into_iter().collect::<Result<Vec<_>>>()?;
                // One numbering, built once and shared: every copy of one
                // recipe numbers its slots identically.
                for s in &mut sims {
                    s.adopt_symbolic(Arc::clone(&view));
                    s.options_mut().draw_arrows = false;
                }
                Backend::Geometry(sims)
            }
        };

        Ok(PuzzleBatch {
            recipe: recipe.to_string(),
            view,
            backend,
            rng: (0..count).map(|i| 0x2545_F491_4F6C_DD1D ^ i as u64).collect(),
            last: vec![NONE; count],
            history: vec![Vec::new(); count],
            luts: Vec::new(),
        })
    }

    /// How many puzzles the batch holds.
    pub fn len(&self) -> usize {
        self.rng.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rng.is_empty()
    }

    /// The recipe every copy was built from.
    pub fn recipe(&self) -> &str {
        &self.recipe
    }

    /// The shared sticker numbering and move names.
    pub fn view(&self) -> &Arc<SymbolicView> {
        &self.view
    }

    /// Whether moves are being applied as permutations instead of by turning
    /// geometry, which is the difference between nanoseconds and milliseconds
    /// a move.
    pub fn is_tabular(&self) -> bool {
        matches!(self.backend, Backend::Table(_))
    }

    pub fn sticker_count(&self) -> usize {
        self.view.stickers.sticker_count()
    }

    pub fn action_count(&self) -> usize {
        self.view.actions.action_count()
    }

    pub fn color_count(&self) -> usize {
        self.view.stickers.color_count()
    }

    /// The color in each slot when the puzzle is solved.
    pub fn solved_colors(&self) -> &[u16] {
        self.view.stickers.solved_colors()
    }

    /// The puzzles themselves, for a batch that keeps geometry.
    ///
    /// `None` for a tabular batch, which keeps one solved puzzle and arrays.
    pub fn sims_mut(&mut self) -> Option<&mut [Simulator]> {
        match &mut self.backend {
            Backend::Geometry(sims) => Some(sims),
            Backend::Table(_) => None,
        }
    }

    /// The solved puzzle a tabular batch draws from, for setting the camera
    /// and the view options a rendered frame will use.
    pub fn model_mut(&mut self) -> Option<&mut Simulator> {
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| Some(&mut t.model)),
            Backend::Geometry(_) => None,
        }
    }

    /// Every move as a permutation of the sticker slots, laid out
    /// `action_count * sticker_count`, or `None` for a batch that turns
    /// geometry because its puzzle has no such table.
    ///
    /// `new[i] = old[perm[a * k + i]]`. This is what the batch itself steps
    /// with. It is offered because an agent that wants to look a move ahead
    /// without taking it needs the same thing.
    pub fn permutations(&self) -> Option<Vec<u32>> {
        match &self.backend {
            Backend::Table(t) => Some(tabular!(t, |t| t.perms.iter().map(|v| v.index() as u32).collect())),
            Backend::Geometry(_) => None,
        }
    }

    /// The moves one row has made since it was last solved.
    ///
    /// Turning a freshly built puzzle through them puts it in that row's
    /// position, which is the way from an array back to geometry.
    ///
    /// It grows by four bytes a move and is emptied by
    /// [`solve`](Self::solve), [`reset`](Self::reset) and their selective
    /// forms. A loop that only ever steps, as one generating data might,
    /// should solve now and then or the history is the one thing in a batch
    /// that grows without bound.
    ///
    /// # Errors
    ///
    /// If `i` is not a row of this batch.
    pub fn history(&self, i: usize) -> Result<&[u32]> {
        let n = self.len();
        self.history
            .get(i)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::Range(format!("puzzle {i} out of range (have {n})")))
    }

    /// How many moves each row has made since it was last solved.
    pub fn move_counts(&self) -> Vec<usize> {
        self.history.iter().map(Vec::len).collect()
    }

    /// Where the camera this batch draws from is: its position, what it
    /// looks at, which way is up, and its field of view in degrees.
    ///
    /// Offered so that a caller who moves the camera can put it back. Drawing
    /// two viewpoints of the same batch is an ordinary thing to want, and a
    /// caller that has to remember where the camera started in order to do it
    /// is a caller that will forget.
    pub fn camera(&self) -> Option<&Camera> {
        match &self.backend {
            Backend::Table(t) => tabular!(t, |t| Some(t.model.camera())),
            Backend::Geometry(sims) => sims.first().map(Simulator::camera),
        }
    }

    /// Apply a view setting to whatever this batch draws with: the one
    /// solved puzzle a tabular batch paints from, or every copy of a batch
    /// that keeps geometry.
    ///
    /// The drawing tables are kept: each records what it was built for, and a
    /// frame checks that before using one. Keeping them is what lets a caller
    /// alternate between two viewpoints without rebuilding a table each time.
    pub fn configure(&mut self, f: &dyn Fn(&mut Simulator)) {
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| f(&mut t.model)),
            Backend::Geometry(sims) => {
                for s in sims.iter_mut() {
                    f(s);
                }
            },
        }
    }

    /// Seed row `i`'s scrambler.
    ///
    /// # Errors
    ///
    /// If `i` is not a row of this batch.
    pub fn seed(&mut self, i: usize, seed: u64) -> Result<()> {
        let n = self.len();
        let slot = self
            .rng
            .get_mut(i)
            .ok_or_else(|| Error::Range(format!("puzzle {i} out of range (have {n})")))?;
        *slot = seed;
        self.last[i] = NONE;
        Ok(())
    }

    /// Seed every row, giving row `i` `seed + i` so that one number fixes the
    /// whole batch while no two rows walk together.
    pub fn seed_all(&mut self, seed: u64) {
        for (i, slot) in self.rng.iter_mut().enumerate() {
            *slot = seed.wrapping_add(i as u64);
        }
        self.last.fill(NONE);
    }

    /* --- turning -------------------------------------------------------- */

    /// Turn every puzzle once.
    ///
    /// `actions[i]` is the move puzzle `i` makes. An action naming a layer
    /// that cannot turn leaves that puzzle alone and reports `applied: false`,
    /// which only a batch that keeps geometry can produce.
    ///
    /// # Errors
    ///
    /// If `actions` is the wrong length, an action is out of range, or a turn
    /// fails.
    pub fn step(&mut self, actions: &[u32]) -> Result<Vec<StepOutcome>> {
        let n = self.len();
        if actions.len() != n {
            return Err(Error::Range(format!("got {} actions for {n} puzzles", actions.len())));
        }
        let count = self.action_count();
        if let Some(&bad) = actions.iter().find(|&&a| a as usize >= count) {
            return Err(Error::Range(format!("action {bad} out of range (have {count})")));
        }
        let (last, history) = (&mut self.last, &mut self.history);
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let k = t.solved.len();
                let (perms, solved) = (&t.perms, &t.solved);
                t.scratch
                    .par_chunks_mut(k.max(1))
                    .zip(t.states.par_chunks(k.max(1)))
                    .zip(actions.par_iter())
                    .for_each(|((dst, src), &a)| {
                        let perm = &perms[a as usize * k..][..k];
                        for (d, from) in dst.iter_mut().zip(perm) {
                            *d = src[from.index()];
                        }
                    });
                std::mem::swap(&mut t.states, &mut t.scratch);
                last.copy_from_slice(actions);
                for (row, &a) in history.iter_mut().zip(actions) {
                    row.push(a);
                }
                Ok(t.states
                    .par_chunks(k.max(1))
                    .map(|row| StepOutcome {
                        applied: true,
                        solved: is_home(row, solved),
                    })
                    .collect())
            }),
            Backend::Geometry(sims) => sims
                .par_iter_mut()
                .zip(actions.par_iter())
                .zip(last.par_iter_mut())
                .zip(history.par_iter_mut())
                .map(|(((sim, &a), last), past)| {
                    let applied = sim.apply_action(a as usize)?;
                    if applied {
                        *last = a;
                        past.push(a);
                    }
                    Ok(StepOutcome {
                        applied,
                        // A state that cannot be read is one where a sticker
                        // has left the solved lattice, which a jumbling
                        // puzzle's turns do. That is not an answer of "we do
                        // not know": a sticker off the lattice is certainly
                        // not at home, so the puzzle is certainly not solved.
                        // Reading the state itself still says what happened.
                        solved: sim.is_solved().unwrap_or(false),
                    })
                })
                .collect(),
        }
    }

    /// Undo the last move of the rows `which` selects.
    ///
    /// Returns whether each row had a move to undo. A row that has not
    /// moved since it was last solved has not.
    ///
    /// # Errors
    ///
    /// If `which` is the wrong length, or a turn fails.
    pub fn undo(&mut self, which: &[bool]) -> Result<Vec<bool>> {
        let n = self.len();
        if which.len() != n {
            return Err(Error::Range(format!("got {} flags for {n} puzzles", which.len())));
        }
        let mut undone = vec![false; n];
        let (history, last) = (&mut self.history, &mut self.last);
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let k = t.solved.len().max(1);
                let perms = &t.perms;
                t.states
                    .par_chunks_mut(k)
                    .zip(history.par_iter_mut())
                    .zip(last.par_iter_mut())
                    .zip(which.par_iter())
                    .zip(undone.par_iter_mut())
                    .for_each_init(
                        || vec![Cell::ZERO; k],
                        |scratch, ((((row, past), last), &go), done)| {
                            if !go {
                                return;
                            }
                            let Some(a) = past.pop() else { return };
                            // The permutation read backward, instead of the
                            // move that is meant to be this one's opposite:
                            // this undoes what was done whatever the action
                            // table calls the other direction.
                            let perm = &perms[a as usize * k..][..k];
                            scratch.copy_from_slice(row);
                            for (i, to) in perm.iter().enumerate() {
                                row[to.index()] = scratch[i];
                            }
                            *last = past.last().copied().unwrap_or(NONE);
                            *done = true;
                        },
                    );
                Ok(undone)
            }),
            Backend::Geometry(sims) => {
                sims.par_iter_mut()
                    .zip(history.par_iter_mut())
                    .zip(last.par_iter_mut())
                    .zip(which.par_iter())
                    .zip(undone.par_iter_mut())
                    .try_for_each(|((((sim, past), last), &go), done)| {
                        if !go || past.is_empty() {
                            return Ok(());
                        }
                        if sim.undo()? {
                            past.pop();
                            *last = past.last().copied().unwrap_or(NONE);
                            *done = true;
                        }
                        Ok(())
                    })?;
                Ok(undone)
            },
        }
    }

    /// Make a written sequence of moves, such as `"A B2' A'"`, in every
    /// puzzle.
    ///
    /// Returns how many turns each puzzle was asked for, counting repeats. A
    /// turn that a puzzle cannot make leaves that one alone, as a step does.
    ///
    /// # Errors
    ///
    /// If a name is not one of this puzzle's moves, or a turn fails.
    pub fn apply(&mut self, moves: &str) -> Result<usize> {
        let view = Arc::clone(&self.view);
        let parsed = crate::symbolic::parse_moves(&view.actions, moves)?;
        let n = self.len();
        let mut made = 0usize;
        for m in parsed {
            let actions = vec![m.action as u32; n];
            for _ in 0..m.repeat {
                self.step(&actions)?;
                made += 1;
            }
        }
        Ok(made)
    }

    /* --- putting a state in ---------------------------------------------- */

    /// Put states that did not come from this batch into it.
    ///
    /// `ids` is `len() * sticker_count()` slot numbers, laid out as
    /// [`sticker_ids`](Self::sticker_ids) gives them: for each slot, the slot
    /// the sticker now in it started in. A state read out of one batch, saved
    /// to a file, or produced by a [`Layout`](crate::layout::Layout) with no
    /// puzzle in sight goes back in this way, which is what makes the batch
    /// something a solver can drive instead of only walking.
    ///
    /// Each row must be an arrangement of the puzzle's stickers (every slot
    /// named exactly once), and that is checked. Whether the arrangement is
    /// reachable from solved via valid turns is not validated during ingestion, as
    /// reachability is undecidable in general. Configurations that cannot be reached
    /// from solved are retained, rendered, and retrieved without error, and will not
    /// reach a solved state.
    ///
    /// The history is emptied, because the moves a row had made no longer
    /// lead to where it now is.
    ///
    /// # Errors
    ///
    /// If `ids` is the wrong length, a row is not an arrangement of the
    /// stickers, or this batch keeps geometry instead of arrays, which
    /// cannot be put into a given state at all.
    pub fn set_states<I: Cell>(&mut self, ids: &[I]) -> Result<()> {
        let (n, k) = (self.len(), self.sticker_count());
        if k == 0 || ids.len() != n * k {
            return Err(Error::Range(format!(
                "a state of this puzzle is {k} slots and this batch holds {n} of them, so {} \
                 values is not a state each",
                ids.len()
            )));
        }
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                if let Some(row) = ids
                    .par_chunks(k)
                    .enumerate()
                    .find_map_any(|(row, src)| (!is_arrangement(src, k)).then_some(row))
                {
                    return Err(Error::Range(format!(
                        "row {row} is not an arrangement of this puzzle's {k} stickers"
                    )));
                }
                t.states
                    .par_iter_mut()
                    .zip(ids.par_iter())
                    .for_each(|(dst, src)| *dst = Cell::from_index(src.index()));
                Ok(())
            }),
            Backend::Geometry(_) => Err(Error::State(
                "this batch turns geometry instead of arrays, because its puzzle's moves are \
                 not fixed permutations, so there is no arrangement to put it into"
                    .into(),
            )),
        }?;
        self.forget();
        Ok(())
    }

    /// The same, from colors instead of identities.
    ///
    /// `colors` is `len() * sticker_count()` colors, laid out as
    /// [`observations`](Self::observations) gives them. Colors do not
    /// determine identities (two stickers of one color may be swapped and
    /// nothing in the array says so), so stickers of each color are
    /// assigned in slot order, representing an indistinguishable configuration
    /// for callers inspecting only facelet colors.
    ///
    /// # Errors
    ///
    /// As [`set_states`](Self::set_states), and if a row does not have the
    /// same number of each color as the solved puzzle.
    pub fn set_colors<C: Cell>(&mut self, colors: &[C]) -> Result<()> {
        let (n, k) = (self.len(), self.sticker_count());
        if k == 0 || colors.len() != n * k {
            return Err(Error::Range(format!(
                "a state of this puzzle is {k} colors and this batch holds {n} of them, so {} \
                 values is not a state each",
                colors.len()
            )));
        }
        let solved = self.view.stickers.solved_colors();
        let count = self.color_count();
        // Locations of each color's stickers in the solved state, in slot
        // order: canonical slot identities to assign.
        let mut home: Vec<Vec<u32>> = vec![Vec::new(); count];
        for (slot, &c) in solved.iter().enumerate() {
            home[c as usize].push(slot as u32);
        }
        let mut ids = vec![0u32; colors.len()];
        let bad = ids
            .par_chunks_mut(k)
            .zip(colors.par_chunks(k))
            .enumerate()
            .find_map_any(|(row, (dst, src))| {
                let mut at = vec![0usize; count];
                for (d, c) in dst.iter_mut().zip(src) {
                    let c = c.index();
                    if c >= count {
                        return Some(row);
                    }
                    let Some(&slot) = home[c].get(at[c]) else {
                        return Some(row);
                    };
                    at[c] += 1;
                    *d = slot;
                }
                None
            });
        if let Some(row) = bad {
            return Err(Error::Range(format!(
                "row {row} does not have the colors of this puzzle: every state of it has as \
                 many stickers of each color as the solved one"
            )));
        }
        self.set_states(&ids)
    }

    /// Forget where every row has been, which a state put in from outside
    /// makes true of all of them.
    fn forget(&mut self) {
        for row in &mut self.history {
            row.clear();
        }
        self.last.fill(NONE);
    }

    /// Solve every puzzle, without scrambling it again.
    ///
    /// # Errors
    ///
    /// If a puzzle that keeps geometry cannot be unwound or rebuilt.
    pub fn solve(&mut self) -> Result<()> {
        let which = vec![true; self.len()];
        self.solve_where(&which)
    }

    /// Solve the rows `which` selects.
    ///
    /// # Errors
    ///
    /// As [`solve`](Self::solve).
    pub fn solve_where(&mut self, which: &[bool]) -> Result<()> {
        let n = self.len();
        if which.len() != n {
            return Err(Error::Range(format!("got {} flags for {n} puzzles", which.len())));
        }
        for (i, &go) in which.iter().enumerate() {
            if go {
                self.last[i] = NONE;
                self.history[i].clear();
            }
        }
        let view = Arc::clone(&self.view);
        let recipe = self.recipe.clone();
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let k = t.solved.len().max(1);
                t.states.par_chunks_mut(k).zip(which.par_iter()).for_each(|(row, &go)| {
                    if go {
                        for (i, slot) in row.iter_mut().enumerate() {
                            *slot = Cell::from_index(i);
                        }
                    }
                });
                Ok(())
            }),
            Backend::Geometry(sims) => sims.par_iter_mut().zip(which.par_iter()).try_for_each(|(sim, &go)| {
                if go {
                    rebuild_solved(sim, &recipe, &view)?;
                }
                Ok(())
            }),
        }
    }

    /// Solve every puzzle and walk each `depth` random turns away again.
    ///
    /// Initializes via a random walk from the solved goal instead of sampling
    /// uniformly from the state space. Uniform sampling yields configurations
    /// near the puzzle's maximum diameter, providing poor training signal for
    /// reinforcement learning algorithms, whereas a random walk of length $k$
    /// ensures solvability within at most $k$ moves by construction.
    ///
    /// # Errors
    ///
    /// If a turn or an unwind fails.
    pub fn reset(&mut self, depth: u32) -> Result<()> {
        let which = vec![true; self.len()];
        self.reset_where(&which, depth)
    }

    /// Solve one puzzle and scramble just that one, leaving the rest alone.
    ///
    /// # Errors
    ///
    /// As [`reset`](Self::reset), or if `i` is not a row of this batch.
    pub fn reset_one(&mut self, i: usize, depth: u32) -> Result<()> {
        let n = self.len();
        if i >= n {
            return Err(Error::Range(format!("puzzle {i} out of range (have {n})")));
        }
        let mut which = vec![false; n];
        which[i] = true;
        self.reset_where(&which, depth)
    }

    /// Solve the rows `which` selects and walk each `depth` turns away.
    ///
    /// Vectorized environment autoreset in a single invocation: completed
    /// episodes are reset while active episodes remain unaffected, parallelized
    /// across all available cores.
    ///
    /// # Errors
    ///
    /// As [`reset`](Self::reset).
    pub fn reset_where(&mut self, which: &[bool], depth: u32) -> Result<()> {
        self.solve_where(which)?;
        let depths: Vec<u32> = which.iter().map(|&go| u32::from(go) * depth).collect();
        self.scramble(&depths)
    }

    /// Walk each puzzle `depths[i]` random turns from wherever it stands.
    ///
    /// A walk avoids immediately inverting the preceding move, preventing
    /// length reduction in the resulting scramble trajectory.
    ///
    /// # Errors
    ///
    /// If `depths` is the wrong length, or a turn fails.
    pub fn scramble(&mut self, depths: &[u32]) -> Result<()> {
        let n = self.len();
        if depths.len() != n {
            return Err(Error::Range(format!("got {} depths for {n} puzzles", depths.len())));
        }
        let count = self.action_count();
        if count == 0 {
            return Ok(());
        }
        let (last, rng, history) = (&mut self.last, &mut self.rng, &mut self.history);
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let k = t.solved.len().max(1);
                let perms = &t.perms;
                t.states
                    .par_chunks_mut(k)
                    .zip(rng.par_iter_mut())
                    .zip(last.par_iter_mut())
                    .zip(history.par_iter_mut())
                    .zip(depths.par_iter())
                    .for_each_init(
                        || vec![Cell::ZERO; k],
                        |scratch, ((((row, rng), last), past), &depth)| {
                            for _ in 0..depth {
                                let a = pick(rng, count, *last);
                                let perm = &perms[a * k..][..k];
                                scratch.copy_from_slice(row);
                                for (d, from) in row.iter_mut().zip(perm) {
                                    *d = scratch[from.index()];
                                }
                                *last = a as u32;
                                past.push(a as u32);
                            }
                        },
                    );
                Ok(())
            }),
            Backend::Geometry(sims) => sims
                .par_iter_mut()
                .zip(rng.par_iter_mut())
                .zip(last.par_iter_mut())
                .zip(history.par_iter_mut())
                .zip(depths.par_iter())
                .try_for_each(|((((sim, rng), last), past), &depth)| walk_geometry(sim, rng, last, past, depth, count)),
        }
    }

    /* --- reading -------------------------------------------------------- */

    /// Every puzzle's sticker array, laid out one after another.
    ///
    /// `len() * sticker_count()` values, matching the standard `(n, k)`
    /// observation array layout. The integer width is caller-selected: almost
    /// every puzzle has few enough colors for a byte, and the few that do not
    /// are read at `u16` or wider instead of refused.
    ///
    /// # Errors
    ///
    /// If a puzzle's state cannot be read, which a jumbling puzzle's cannot,
    /// or if integer type `C` cannot represent the number of unique colors.
    pub fn observations<C: Cell>(&mut self) -> Result<Vec<C>> {
        let colors = self.color_count();
        if colors as u64 > C::SPAN {
            return Err(Error::Range(format!(
                "this puzzle has {colors} colors, which do not fit the {} values a \
                 {}-byte observation has room for",
                C::SPAN,
                C::SPAN.trailing_zeros() / 8
            )));
        }
        let k = self.sticker_count();
        let mut out = vec![C::ZERO; self.len() * k];
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let solved = &t.solved;
                out.par_chunks_mut(k.max(1))
                    .zip(t.states.par_chunks(k.max(1)))
                    .for_each(|(row, ids)| {
                        for (dst, id) in row.iter_mut().zip(ids) {
                            *dst = C::from_index(solved[id.index()] as usize);
                        }
                    });
            }),
            Backend::Geometry(sims) => {
                out.par_chunks_mut(k.max(1))
                    .zip(sims.par_iter_mut())
                    .try_for_each(|(row, sim)| {
                        let colors = sim.stickers()?;
                        for (dst, &c) in row.iter_mut().zip(colors.iter()) {
                            *dst = C::from_index(c as usize);
                        }
                        Ok(())
                    })?;
            },
        }
        Ok(out)
    }

    /// Where the sticker in each slot of each puzzle started: the states as
    /// permutations, `len() * sticker_count()` of them.
    ///
    /// Distinguishes states with identical color configurations, such as swapped identical facelets.
    ///
    /// # Errors
    ///
    /// If a puzzle's state cannot be read.
    pub fn sticker_ids<I: Cell>(&mut self) -> Result<Vec<I>> {
        let k = self.view.stickers.sticker_count();
        if k as u64 > I::SPAN {
            return Err(Error::Range(format!(
                "this puzzle has {k} slots, which do not fit the {} a slot number of this width \
                 has room for",
                I::SPAN
            )));
        }
        match &mut self.backend {
            Backend::Table(t) => Ok(tabular!(t, |t| t
                .states
                .iter()
                .map(|v| I::from_index(v.index()))
                .collect())),
            Backend::Geometry(sims) => {
                let mut out = vec![I::ZERO; sims.len() * k];
                out.par_chunks_mut(k.max(1))
                    .zip(sims.par_iter_mut())
                    .try_for_each(|(row, sim)| {
                        for (d, &v) in row.iter_mut().zip(sim.sticker_ids()?.iter()) {
                            *d = I::from_index(v as usize);
                        }
                        Ok(())
                    })?;
                Ok(out)
            },
        }
    }

    /// Every puzzle's state as indicator bytes: `len() * sticker_count() *
    /// color_count()` of them, one per (slot, color) pair.
    ///
    /// Categorical one-hot indicator encoding, expanded natively within
    /// a single contiguous buffer.
    ///
    /// # Errors
    ///
    /// As [`observations`](Self::observations).
    pub fn one_hot(&mut self) -> Result<Vec<u8>> {
        let k = self.sticker_count();
        let c = self.color_count();
        // Widest here on purpose: the indicator block this expands into is
        // `color_count` times the size of the colors it came from, so the
        // colors are never what costs anything.
        let colors = self.observations::<u32>()?;
        let mut out = vec![0u8; self.len() * k * c];
        out.par_chunks_mut((k * c).max(1))
            .zip(colors.par_chunks(k.max(1)))
            .for_each(|(row, state)| {
                for (i, &color) in state.iter().enumerate() {
                    row[i * c + color as usize] = 1;
                }
            });
        Ok(out)
    }

    /// Which moves each puzzle can make, laid out one after another.
    ///
    /// # Errors
    ///
    /// If a puzzle's grips cannot be derived.
    pub fn action_masks(&mut self) -> Result<Vec<bool>> {
        let n = self.action_count();
        match &mut self.backend {
            // A table is only derived for a puzzle whose layers all turn and
            // keep turning, which the table's own construction checks rather
            // than assumes.
            Backend::Table(_) => Ok(vec![true; self.len() * n]),
            Backend::Geometry(sims) => {
                let mut out = vec![false; sims.len() * n];
                out.par_chunks_mut(n.max(1))
                    .zip(sims.par_iter_mut())
                    .try_for_each(|(row, sim)| {
                        row.copy_from_slice(&sim.action_mask()?);
                        Ok(())
                    })?;
                Ok(out)
            },
        }
    }

    /// Whether each puzzle is solved: every sticker the color it should be.
    ///
    /// A puzzle whose stickers have left the solved lattice, which only a
    /// jumbling puzzle's turns can do, is not solved: no state it cannot be
    /// read out of is one where every sticker is at home.
    ///
    /// # Errors
    ///
    /// If a puzzle cannot be inspected at all.
    pub fn solved(&mut self) -> Result<Vec<bool>> {
        match &mut self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let k = t.solved.len().max(1);
                let solved = &t.solved;
                Ok(t.states.par_chunks(k).map(|row| is_home(row, solved)).collect())
            }),
            // As in `step`: a sticker off the lattice is not at home.
            Backend::Geometry(sims) => Ok(sims
                .par_iter_mut()
                .map(|sim| sim.is_solved().unwrap_or(false))
                .collect()),
        }
    }

    /* --- drawing -------------------------------------------------------- */

    /// Draw every puzzle, across every core, as `(n, height, width, 4)`
    /// RGBA bytes.
    ///
    /// # Errors
    ///
    /// If a puzzle cannot be drawn.
    pub fn render(&mut self, width: u32, height: u32) -> Result<Vec<u8>> {
        self.render_with(FrameSpec::new(width, height))
    }

    /// Draw every puzzle the way `spec` asks for.
    ///
    /// A tabular batch precomputes the mapping from visible facelet slots to
    /// pixel coordinates, reducing subsequent rendering to direct table lookups.
    /// Precomputed tables are cached and reused across frames, allowing
    /// alternation between cached viewpoints without recomputing the raster
    /// lookup table.
    ///
    /// # Errors
    ///
    /// If a puzzle cannot be drawn.
    pub fn render_with(&mut self, spec: FrameSpec) -> Result<Vec<u8>> {
        let rows = self.len();
        let mut out = vec![0u8; rows * spec.len()];
        self.paint_into(spec, &mut out, |dst, v| *dst = v)?;
        Ok(out)
    }

    /// Rendered frames normalized to floating-point values in `[0.0, 1.0]`,
    /// formatted for vision-based learning models.
    ///
    /// # Errors
    ///
    /// As [`render_with`](Self::render_with).
    pub fn render_float(&mut self, spec: FrameSpec) -> Result<Vec<f32>> {
        let rows = self.len();
        let mut out = vec![0f32; rows * spec.len()];
        self.paint_into(spec, &mut out, |dst, v| *dst = f32::from(v) / 255.0)?;
        Ok(out)
    }

    /// Draw every puzzle into `out`, one value per channel per pixel, through
    /// `put` so that the same rearranging serves bytes and real numbers both.
    fn paint_into<T: Send + Sync>(
        &mut self,
        spec: FrameSpec,
        out: &mut [T],
        put: impl Fn(&mut T, u8) + Send + Sync,
    ) -> Result<()> {
        if spec.channels != 3 && spec.channels != 4 {
            return Err(Error::Range(format!(
                "a frame has 3 or 4 channels, not {}",
                spec.channels
            )));
        }
        let (w, h) = (spec.width as usize, spec.height as usize);
        let rgba = spec.len().max(1);

        if self.is_tabular() {
            self.ensure_table(spec.width, spec.height)?;
        }
        match &self.backend {
            Backend::Table(t) => tabular!(t, |t| {
                let lut = &self.luts[0].1;
                let k = t.solved.len().max(1);
                let solved = &t.solved;
                let palette = lut.palette();
                out.par_chunks_mut(rgba).zip(t.states.par_chunks(k)).for_each_init(
                    || (vec![0u16; k], vec![0u8; w * h * 4]),
                    |(colors, pixels), (dst, ids)| {
                        for (c, id) in colors.iter_mut().zip(ids) {
                            *c = solved[id.index()];
                        }
                        lut.paint(colors, palette, pixels);
                        write_frame(dst, pixels, spec, &put);
                    },
                );
                Ok(())
            }),
            Backend::Geometry(sims) => {
                out.par_chunks_mut(rgba).zip(sims.par_iter()).for_each(|(dst, sim)| {
                    let fb = sim.render(spec.width, spec.height);
                    write_frame(dst, fb.as_bytes(), spec, &put);
                });
                Ok(())
            },
        }
    }

    /// Draw states that did not come from this batch.
    ///
    /// `colors` contains `rows * sticker_count` color indices, and the rows need
    /// not originate from this batch: external state trajectories can be
    /// rendered directly without constructing individual puzzle instances.
    /// The batch's own states are untouched.
    ///
    /// # Errors
    ///
    /// If `colors` is not a whole number of states, or this batch keeps
    /// geometry instead of a drawing table.
    pub fn render_states<C: Cell>(&mut self, colors: &[C], spec: FrameSpec) -> Result<Vec<u8>> {
        let k = self.sticker_count();
        if k == 0 || colors.len() % k != 0 {
            return Err(Error::Range(format!(
                "a state of this puzzle is {k} colors, and {} is not a whole number of them",
                colors.len()
            )));
        }
        let rows = colors.len() / k;
        self.ensure_table(spec.width, spec.height)?;
        let lut = &self.luts[0].1;
        let frame = spec.len();
        let (w, h) = (spec.width as usize, spec.height as usize);
        let mut out = vec![0u8; rows * frame];
        let palette = lut.palette();
        out.par_chunks_mut(frame.max(1))
            .zip(colors.par_chunks(k))
            .for_each_init(
                || vec![0u8; w * h * 4],
                |pixels, (dst, state)| {
                    lut.paint(state, palette, pixels);
                    write_frame(dst, pixels, spec, |d, v| *d = v);
                },
            );
        Ok(out)
    }

    /// Make sure the drawing table for this size is the first one held.
    ///
    /// Separate from using it so that the borrow ends before the states are
    /// read: building a table needs the model puzzle, and painting needs the
    /// states, and the two live in the same place.
    fn ensure_table(&mut self, width: u32, height: u32) -> Result<()> {
        let Backend::Table(t) = &mut self.backend else {
            return Err(Error::State(
                "this puzzle has no drawing table: its stickers do not stay on the solved \
                 lattice, so which one a pixel shows depends on the state"
                    .into(),
            ));
        };
        let model = tabular!(t, |t| &mut t.model);
        let key = lut_key(model, width, height);
        if let Some(i) = self.luts.iter().position(|(k, _)| *k == key) {
            self.luts[..=i].rotate_right(1);
            return Ok(());
        }
        let lut = model.sticker_lut(width, height)?;
        self.luts.insert(0, (key, lut));
        self.luts.truncate(LUT_CACHE);
        Ok(())
    }

    /// Forget the drawing tables, so the next frame builds a fresh one.
    ///
    /// Only needed when something a table depends on is changed through a
    /// route this batch cannot see.
    pub fn invalidate_lut(&mut self) {
        self.luts.clear();
    }
}

/// Copy one painted RGBA frame into the layout a caller asked for.
fn write_frame<T>(dst: &mut [T], pixels: &[u8], spec: FrameSpec, put: impl Fn(&mut T, u8)) {
    let (w, h) = (spec.width as usize, spec.height as usize);
    for y in 0..h {
        for x in 0..w {
            let at = (y * w + x) * 4;
            for c in 0..spec.channels {
                if at + c >= pixels.len() {
                    continue;
                }
                let to = if spec.channels_first {
                    c * w * h + y * w + x
                } else {
                    (y * w + x) * spec.channels + c
                };
                put(&mut dst[to], pixels[at + c]);
            }
        }
    }
}

/// Every setting a drawing table depends on, as one comparable value.
fn lut_key(sim: &Simulator, width: u32, height: u32) -> LutKey {
    let opts = sim.options();
    let c = sim.camera();
    LutKey {
        width,
        height,
        supersample: opts.supersample.max(1),
        background: opts.background,
        line_width: opts.line_width.to_bits(),
        draw_edges: opts.draw_edges,
        draw_pieces: opts.draw_pieces,
        camera: [
            c.position.x.to_bits(),
            c.position.y.to_bits(),
            c.position.z.to_bits(),
            c.target.x.to_bits(),
            c.target.y.to_bits(),
            c.target.z.to_bits(),
            c.up.x.to_bits(),
            c.up.y.to_bits(),
            c.up.z.to_bits(),
            c.fov.to_bits(),
        ],
    }
}

/// Whether a row names every slot of the puzzle exactly once.
fn is_arrangement<I: Cell>(row: &[I], k: usize) -> bool {
    let mut seen = vec![false; k];
    for v in row {
        let i = v.index();
        if i >= k || seen[i] {
            return false;
        }
        seen[i] = true;
    }
    true
}

/// Whether every sticker of one row is the color it should be.
fn is_home<I: Cell>(ids: &[I], solved: &[u16]) -> bool {
    ids.iter().enumerate().all(|(i, id)| solved[id.index()] == solved[i])
}

/// Select an action that does not invert the preceding move.
///
/// Action `2k` and action `2k + 1` turn one grip opposite ways, so `a ^ 1` is
/// the move that would undo `a`. A clash is nudged to the next grip rather
/// than redrawn, which keeps a walk of `depth` moves exactly that long.
fn pick(state: &mut u64, count: usize, last: u32) -> usize {
    let a = next_below(state, count);
    if count > 2 && last != NONE && a as u32 == (last ^ 1) {
        (a + 2) % count
    } else {
        a
    }
}

/// One puzzle's random walk, through its geometry.
fn walk_geometry(
    sim: &mut Simulator,
    rng: &mut u64,
    last: &mut u32,
    past: &mut Vec<u32>,
    depth: u32,
    count: usize,
) -> Result<()> {
    let mut made = 0u32;
    // Bounded: a puzzle whose layers all lock would otherwise spin here.
    for _ in 0..depth.saturating_mul(8) {
        if made == depth {
            break;
        }
        let a = pick(rng, count, *last);
        if sim.apply_action(a)? {
            *last = a as u32;
            past.push(a as u32);
            made += 1;
        }
    }
    Ok(())
}

/// Put one puzzle back to solved, by unwinding if that works and by building
/// it again if a locked layer stops the unwind part-way.
fn rebuild_solved(sim: &mut Simulator, recipe: &str, view: &Arc<SymbolicView>) -> Result<()> {
    if !sim.restore()? {
        let mut fresh = Simulator::from_query(recipe)?;
        fresh.adopt_symbolic(Arc::clone(view));
        *sim = fresh;
    }
    Ok(())
}

/// SplitMix64, advanced in place.
fn next_below(state: &mut u64, n: usize) -> usize {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as usize).rem_euclid(n)
}
