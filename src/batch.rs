//! Building and rendering many puzzles at once, in parallel.
//!
//! Each puzzle owns its number field outright, so two puzzles share no state
//! and can be built on different threads, which is what these functions do.
//!
//! Within a single puzzle the numbers share one isolating interval, so the
//! work is not split across threads here. Nothing about the arithmetic forbids
//! it, because a conversion refines to the correctly rounded double either way
//! (`SEMANTICS.md` §1, §9), but nothing is structured for it either.

use std::sync::Arc;

use rayon::prelude::*;

use crate::error::Error;
use crate::render::Framebuffer;
use crate::simulator::{Simulator, SymbolicView};
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
    recipes
        .par_iter()
        .map(|r| Simulator::from_query(r))
        .collect()
}

/// Build, scramble and render every recipe, in parallel.
///
/// One puzzle per task rather than one image band per task: the puzzles are
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
    if spec.scramble > 0 {
        sim.seed(spec.seed.unwrap_or(0).wrapping_add(index));
        sim.scramble(spec.scramble);
        sim.settle()?;
    }
    Ok(sim.render(spec.width, spec.height))
}

/// Worker threads available to the batch functions.
///
/// This is rayon's global pool, shared with the renderer's band-parallel
/// rasterizer. Set `RAYON_NUM_THREADS` to change it.
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

/* -------------------------------------------------------------------------- */
/*  Batched stepping                                                          */
/* -------------------------------------------------------------------------- */

/// Many copies of one puzzle, stepped together.
///
/// The point of keeping them in Rust rather than in a list of Python objects is
/// that a whole batch of turns is one call: the interpreter is released once,
/// the turns run across every core, and the states come back as a single block
/// of bytes that NumPy can view without copying. A vectorized environment built
/// on this does not pay per-environment interpreter overhead at all.
pub struct PuzzleBatch {
    sims: Vec<Simulator>,
    view: Arc<SymbolicView>,
    /// One generator per puzzle, so a scramble is reproducible however the
    /// batch is scheduled.
    rng: Vec<u64>,
    /// The last action taken by each puzzle, so a scramble does not walk back
    /// over itself.
    last: Vec<Option<usize>>,
}

/// What one step did to one puzzle.
pub struct StepOutcome {
    /// Whether the move could be made at all.
    pub applied: bool,
    /// Whether the puzzle is now solved.
    pub solved: bool,
}

impl PuzzleBatch {
    /// Build `count` copies of `recipe`, across every core.
    ///
    /// # Errors
    ///
    /// If the recipe does not build, or the puzzle has no sticker numbering.
    pub fn build(recipe: &str, count: usize) -> Result<PuzzleBatch> {
        if count == 0 {
            return Err(Error::Range("a batch needs at least one puzzle".into()));
        }
        let recipes = vec![recipe.to_string(); count];
        let mut sims: Vec<Simulator> = build_many(&recipes)
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        // One numbering, built once and shared: every copy of one recipe numbers
        // its slots identically, so building it `count` times would be waste.
        let view = sims[0].symbolic()?;
        for s in &mut sims {
            s.adopt_symbolic(Arc::clone(&view));
        }
        Ok(PuzzleBatch {
            rng: (0..count)
                .map(|i| 0x2545_F491_4F6C_DD1D ^ i as u64)
                .collect(),
            last: vec![None; count],
            sims,
            view,
        })
    }

    pub fn len(&self) -> usize {
        self.sims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sims.is_empty()
    }

    pub fn view(&self) -> &Arc<SymbolicView> {
        &self.view
    }

    pub fn sticker_count(&self) -> usize {
        self.view.stickers.sticker_count()
    }

    pub fn action_count(&self) -> usize {
        self.view.actions.action_count()
    }

    /// The puzzles themselves, for anything the batch does not do itself.
    pub fn sims_mut(&mut self) -> &mut [Simulator] {
        &mut self.sims
    }

    /// Seed puzzle `i`'s scrambler.
    pub fn seed(&mut self, i: usize, seed: u64) {
        self.rng[i] = seed;
        self.last[i] = None;
    }

    /// Solve puzzle `i` and walk `depth` random turns away from solved.
    ///
    /// A walk from the goal rather than a uniform random state: a state drawn
    /// uniformly is almost always at the far end of the puzzle and teaches a
    /// learner nothing, where a walk of known length is reachable by
    /// construction.
    ///
    /// # Errors
    ///
    /// If a turn or an unwind fails.
    pub fn reset_one(&mut self, i: usize, depth: u32) -> Result<()> {
        let n = self.action_count();
        if !self.sims[i].restore()? {
            // A locked layer stopped the unwind, so start from a clean build.
            let query = self.sims[i].query()?;
            let mut fresh = Simulator::from_query(&query)?;
            fresh.adopt_symbolic(Arc::clone(&self.view));
            self.sims[i] = fresh;
        }
        self.last[i] = None;
        if n == 0 {
            return Ok(());
        }
        let mut made = 0u32;
        // Bounded: a puzzle whose layers all lock would otherwise spin here.
        for _ in 0..depth.saturating_mul(8) {
            if made == depth {
                break;
            }
            let a = next_below(&mut self.rng[i], n);
            if self.last[i] == Some(a ^ 1) {
                continue;
            }
            if self.sims[i].apply_action(a)? {
                self.last[i] = Some(a);
                made += 1;
            }
        }
        Ok(())
    }

    /// Solve every puzzle and scramble each to `depth`, across every core.
    ///
    /// # Errors
    ///
    /// As [`reset_one`](Self::reset_one).
    pub fn reset_all(&mut self, depth: u32) -> Result<()> {
        let view = Arc::clone(&self.view);
        let n = view.actions.action_count();
        self.sims
            .par_iter_mut()
            .zip(self.rng.par_iter_mut())
            .zip(self.last.par_iter_mut())
            .try_for_each(|((sim, rng), last)| reset_sim(sim, rng, last, depth, n, &view))
    }

    /// Turn every puzzle once, across every core.
    ///
    /// `actions[i]` is the move puzzle `i` makes. An action naming a layer that
    /// cannot turn leaves that puzzle alone and reports `applied: false`.
    ///
    /// # Errors
    ///
    /// If `actions` is the wrong length, or a turn fails.
    pub fn step(&mut self, actions: &[u32]) -> Result<Vec<StepOutcome>> {
        if actions.len() != self.sims.len() {
            return Err(Error::Range(format!(
                "got {} actions for {} puzzles",
                actions.len(),
                self.sims.len()
            )));
        }
        let n = self.action_count();
        self.sims
            .par_iter_mut()
            .zip(actions.par_iter())
            .zip(self.last.par_iter_mut())
            .map(|((sim, &a), last)| {
                let a = a as usize;
                if a >= n {
                    return Err(Error::Range(format!("action {a} out of range (have {n})")));
                }
                let applied = sim.apply_action(a)?;
                if applied {
                    *last = Some(a);
                }
                Ok(StepOutcome {
                    applied,
                    solved: sim.is_solved()?,
                })
            })
            .collect()
    }

    /// Every puzzle's sticker array, laid out one after another.
    ///
    /// `out` is `len() * sticker_count()` bytes, puzzle by puzzle, which is
    /// exactly the `(n, k)` `uint8` array a learner wants.
    ///
    /// # Errors
    ///
    /// If a puzzle's state cannot be read, which a jumbling puzzle's cannot.
    pub fn observations(&mut self) -> Result<Vec<u8>> {
        let k = self.sticker_count();
        let mut out = vec![0u8; self.sims.len() * k];
        out.par_chunks_mut(k)
            .zip(self.sims.par_iter_mut())
            .try_for_each(|(row, sim)| {
                let colors = sim.stickers()?;
                for (dst, &c) in row.iter_mut().zip(colors.iter()) {
                    *dst = c as u8;
                }
                Ok(())
            })?;
        Ok(out)
    }

    /// Which moves each puzzle can make, laid out one after another.
    ///
    /// # Errors
    ///
    /// If a puzzle's grips cannot be derived.
    pub fn action_masks(&mut self) -> Result<Vec<bool>> {
        let n = self.action_count();
        let mut out = vec![false; self.sims.len() * n];
        out.par_chunks_mut(n)
            .zip(self.sims.par_iter_mut())
            .try_for_each(|(row, sim)| {
                row.copy_from_slice(&sim.action_mask()?);
                Ok(())
            })?;
        Ok(out)
    }

    /// Whether each puzzle is solved.
    ///
    /// # Errors
    ///
    /// As [`observations`](Self::observations).
    pub fn solved(&mut self) -> Result<Vec<bool>> {
        self.sims.par_iter_mut().map(Simulator::is_solved).collect()
    }
}

/// One puzzle's reset, written free so rayon can hold each borrow separately.
fn reset_sim(
    sim: &mut Simulator,
    rng: &mut u64,
    last: &mut Option<usize>,
    depth: u32,
    n: usize,
    view: &Arc<SymbolicView>,
) -> Result<()> {
    if !sim.restore()? {
        let query = sim.query()?;
        let mut fresh = Simulator::from_query(&query)?;
        fresh.adopt_symbolic(Arc::clone(view));
        *sim = fresh;
    }
    *last = None;
    if n == 0 {
        return Ok(());
    }
    let mut made = 0u32;
    for _ in 0..depth.saturating_mul(8) {
        if made == depth {
            break;
        }
        let a = next_below(rng, n);
        if *last == Some(a ^ 1) {
            continue;
        }
        if sim.apply_action(a)? {
            *last = Some(a);
            made += 1;
        }
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
