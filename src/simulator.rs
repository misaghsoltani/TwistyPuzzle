//! The interactive simulator: everything an application needs around a
//! puzzle, with no window system in sight.
//!
//! A [`Simulator`] owns a puzzle, its renderable meshes, the grips and their
//! arrows, a camera with trackball controls, and the state of an in-progress
//! move. Drive it by feeding pointer events and calling [`Simulator::frame`],
//! which hands back a finished image.

use std::sync::Arc;

use crate::builder::{build, BuiltPuzzle};
use crate::color::Color;
use crate::error::{Error, Result};
use crate::math::{ExactQuaternion, Quat, Vec3};
use crate::movement::{find_cuts, find_stops, make_move, Cut, Puzzle};
use crate::parse::{generate_query, parse_query, PuzzleRecipe};
use crate::render::camera::{Camera, Mat4};
use crate::render::scene::{
    arrow_instances, render_frame, ArrowInstance, PuzzleMeshes, SceneOptions,
};
use crate::render::trackball::{Pointer, TrackballControls};
use crate::render::Framebuffer;
use crate::symbolic::{ActionTable, SlotCache, StickerMap};

/// Angular speed of a move animation, in radians per second.
pub const RAD_PER_SEC: f64 = std::f64::consts::TAU;

/// Which way to turn a grip: the direction an arrow points, or the exact stop
/// to land on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Turn {
    /// `+1` or `-1`: the nearest stop that is not where the grip already sits.
    Direction(i32),
    /// An index into [`Simulator::stops`].
    Stop(usize),
}

/// A move being animated.
struct ActiveMove {
    /// Milliseconds the animation should take.
    duration: f64,
    /// Milliseconds elapsed.
    elapsed: f64,
    pieces: Vec<usize>,
    from_quat: Vec<Quat>,
    step_quat: Vec<Quat>,
    angle: f64,
}

/// A small, seedable PRNG, so scrambles are reproducible.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 bits of mantissa, matching `Math.random`'s range.
        ((z >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }
}

pub struct Simulator {
    recipe: PuzzleRecipe,
    puzzle: Puzzle,
    meshes: PuzzleMeshes,
    scale: f64,
    field: String,

    /// Turnable cuts, in arrow order: `grips[i / 2]` with direction
    /// `(i % 2) * 2 - 1` is what arrow `i` activates.
    grips: Vec<Cut>,
    arrows: Vec<ArrowInstance>,
    hovered_arrow: Option<usize>,
    pressed_arrow: Option<usize>,

    /// World orientation of each piece, as the renderer sees it.
    piece_quats: Vec<Quat>,

    camera: Camera,
    controls: TrackballControls,
    options: SceneOptions,

    active: Option<ActiveMove>,
    pending_scramble: u32,
    last_random: Option<(usize, i32)>,
    rng: Rng,
    move_count: u64,

    /// The sticker numbering and the move names, built on first use.
    ///
    /// Deliberately not built up front: naming an axis decides signs, deciding
    /// a sign can refine the shared number field, and how far that field has
    /// been refined is visible in the coordinates the renderer produces. A
    /// puzzle that is only drawn is drawn exactly as it would have been.
    symbolic: Option<Arc<SymbolicView>>,
    slot_cache: SlotCache,

    /// The plane and rotation of every turn made, newest last, so a move can
    /// be taken back exactly, without knowing what named it.
    history: Vec<(crate::math::ExactPlane, ExactQuaternion)>,
}

/// A puzzle's sticker numbering and its move names, which together turn its
/// geometry into the integer arrays and move strings a solver wants.
pub struct SymbolicView {
    pub stickers: StickerMap,
    pub actions: ActionTable,
}

impl Simulator {
    /// Build a simulator from a recipe query string such as
    /// `"?shell=C$1&cut=C$1/3"`.
    pub fn from_query(query: &str) -> Result<Simulator> {
        let recipe = parse_query(query)?;
        Simulator::from_recipe(recipe)
    }

    pub fn from_recipe(recipe: PuzzleRecipe) -> Result<Simulator> {
        let BuiltPuzzle {
            puzzle,
            scale,
            field,
            ..
        } = build(&recipe)?;
        let meshes = PuzzleMeshes::build(&puzzle)?;
        let mut sim = Simulator {
            recipe,
            puzzle,
            meshes,
            scale,
            field,
            grips: Vec::new(),
            arrows: Vec::new(),
            hovered_arrow: None,
            pressed_arrow: None,
            piece_quats: Vec::new(),
            camera: Camera::default(),
            controls: TrackballControls::new(),
            options: SceneOptions::default(),
            active: None,
            pending_scramble: 0,
            last_random: None,
            rng: Rng(0x2545_F491_4F6C_DD1D),
            move_count: 0,
            symbolic: None,
            slot_cache: SlotCache::new(),
            history: Vec::new(),
        };
        sim.snap_piece_quats();
        sim.refresh_grips()?;
        Ok(sim)
    }

    /* --- accessors --------------------------------------------------- */

    pub fn puzzle(&self) -> &Puzzle {
        &self.puzzle
    }
    /// Mutable access, for algorithms that memoize into the puzzle's caches.
    pub fn puzzle_mut(&mut self) -> &mut Puzzle {
        &mut self.puzzle
    }
    pub fn recipe(&self) -> &PuzzleRecipe {
        &self.recipe
    }
    pub fn query(&self) -> Result<String> {
        generate_query(&self.recipe)
    }
    /// The number field every coordinate lives in.
    pub fn field(&self) -> &str {
        &self.field
    }
    pub fn piece_count(&self) -> usize {
        self.puzzle.pieces.len()
    }
    /// Reciprocal of the shell circumradius: the factor that fits the puzzle
    /// in a unit sphere.
    pub fn scale(&self) -> f64 {
        self.scale
    }
    pub fn grips(&self) -> &[Cut] {
        &self.grips
    }
    pub fn grip_count(&self) -> usize {
        self.grips.len()
    }
    pub fn arrow_count(&self) -> usize {
        self.arrows.len()
    }
    pub fn hovered_arrow(&self) -> Option<usize> {
        self.hovered_arrow
    }
    pub fn is_animating(&self) -> bool {
        self.active.is_some() || self.pending_scramble > 0
    }
    pub fn moves_made(&self) -> u64 {
        self.move_count
    }
    pub fn camera(&self) -> &Camera {
        &self.camera
    }
    /// Look at the puzzle from `yaw` and `pitch` degrees off the default
    /// head-on view, `distance` units out.
    ///
    /// `(0, 0)` is the default, head-on camera, and a little of each
    /// turns a flat-looking silhouette into a solid.
    pub fn look_from(&mut self, yaw_deg: f64, pitch_deg: f64, distance: f64) {
        let (sy, cy) = yaw_deg.to_radians().sin_cos();
        let (sp, cp) = pitch_deg.to_radians().sin_cos();
        let cam = self.camera_mut();
        cam.position = Vec3::new(distance * cp * sy, distance * sp, distance * cp * cy);
        cam.up = Vec3::new(0.0, 1.0, 0.0);
    }

    /// Move the camera to `d` units from the origin, keeping its direction.
    pub fn set_distance(&mut self, d: f64) {
        let p = self.camera().position;
        let len = p.length();
        let s = if len == 0.0 { 0.0 } else { d / len };
        self.camera_mut().position = p.scale(s);
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }
    pub fn options(&self) -> &SceneOptions {
        &self.options
    }
    pub fn options_mut(&mut self) -> &mut SceneOptions {
        &mut self.options
    }
    pub fn controls_mut(&mut self) -> &mut TrackballControls {
        &mut self.controls
    }

    /// Seed the scramble PRNG.
    pub fn seed(&mut self, seed: u64) {
        self.rng = Rng(seed);
        self.last_random = None;
    }

    /* --- grips and arrows -------------------------------------------- */

    /// Recompute the turnable cuts and the arrows that drive them.
    ///
    /// Normals point away from the origin, so a cut through the center gets an
    /// arrow both ways.
    fn refresh_grips(&mut self) -> Result<()> {
        let grips = turnable_cuts(&mut self.puzzle)?;
        self.arrows = arrow_instances(&grips, self.puzzle.global_rot)?;
        self.grips = grips;
        self.hovered_arrow = None;
        Ok(())
    }

    /// Set every piece's rendered orientation from its exact rotation.
    fn snap_piece_quats(&mut self) {
        let g = self.puzzle.global_rot;
        self.piece_quats = self
            .puzzle
            .pieces
            .iter()
            .map(|p| g.mul(&p.rot.to_f64().unwrap_or(Quat::IDENTITY)))
            .collect();
    }

    /* --- pointer ------------------------------------------------------ */

    /// Move the pointer, in normalized device coordinates: `x` and `y` in
    /// `[-1, 1]` with `y` pointing up.
    pub fn pointer_move(&mut self, p: Pointer) {
        if self.controls.is_dragging() {
            self.controls.pointer_move(p);
        }
        self.hovered_arrow = self.pick_arrow(p);
    }

    /// Press the pointer. Returns `true` if an arrow was pressed, in which
    /// case the camera is not dragged.
    pub fn pointer_down(&mut self, p: Pointer) -> bool {
        self.hovered_arrow = self.pick_arrow(p);
        self.pressed_arrow = self.hovered_arrow;
        if self.pressed_arrow.is_none() {
            self.controls.pointer_down(p);
        }
        self.pressed_arrow.is_some()
    }

    /// Release the pointer, activating an arrow if the press and release were
    /// on the same one. Returns the move that was started, if any.
    pub fn pointer_up(&mut self, p: Pointer) -> Result<Option<(usize, i32)>> {
        self.controls.pointer_up();
        self.hovered_arrow = self.pick_arrow(p);
        let hit = match (self.hovered_arrow, self.pressed_arrow.take()) {
            (Some(a), Some(b)) if a == b => a,
            _ => return Ok(None),
        };
        let ci = hit / 2;
        let dir = (hit % 2) as i32 * 2 - 1;
        self.begin_move(ci, dir)?;
        Ok(Some((ci, dir)))
    }

    /// Clear any hovered arrow state, for example when the pointer exits the
    /// viewport.
    pub fn clear_hover(&mut self) {
        self.hovered_arrow = None;
    }

    /// The arrow under the pointer, if any.
    ///
    /// A ray from the camera
    /// through the pointer, tested against every arrow triangle on both sides,
    /// nearest hit wins.
    pub fn pick_arrow(&self, p: Pointer) -> Option<usize> {
        if !self.options.draw_arrows {
            return None;
        }
        let (origin, dir) = self.ray(p);
        let mut best: Option<(f64, usize)> = None;
        for (i, a) in self.arrows.iter().enumerate() {
            for tri in &self.meshes.arrow.triangles {
                let v: [Vec3; 3] = [
                    transform(&a.model, tri[0]),
                    transform(&a.model, tri[1]),
                    transform(&a.model, tri[2]),
                ];
                if let Some(t) = ray_triangle(origin, dir, &v) {
                    if best.is_none_or(|(bt, _)| t < bt) {
                        best = Some((t, i));
                    }
                }
            }
        }
        best.map(|(_, i)| i)
    }

    /// `Raycaster.setFromCamera` for a perspective camera.
    fn ray(&self, p: Pointer) -> (Vec3, Vec3) {
        let origin = self.camera.position;
        let inv_proj = self.camera.projection_matrix().invert();
        let cam_world = self.camera.view_matrix().invert();
        let a = inv_proj.transform_point4(Vec3::new(p.x, p.y, 0.5));
        let a = Vec3::new(a[0] / a[3], a[1] / a[3], a[2] / a[3]);
        let b = cam_world.transform_point4(a);
        let world = Vec3::new(b[0] / b[3], b[1] / b[3], b[2] / b[3]);
        (origin, world.sub(&origin).normalize())
    }

    /* --- moves --------------------------------------------------------- */

    /// Start turning grip `ci` in direction `dir` (`+1` or `-1`).
    ///
    /// Picks the first non-zero stop in that direction. If the grip cannot
    /// turn at all, animates a full revolution and comes back.
    pub fn begin_move(&mut self, ci: usize, dir: i32) -> Result<()> {
        self.begin_turn(ci, Turn::Direction(dir))
    }

    /// Turn grip `ci` to the stop `stop` of [`Simulator::stops`], whatever
    /// direction that lies in.
    ///
    /// The finest control there is over a move: `begin_move` picks the nearest
    /// stop that is not where the puzzle already stands, and this picks any of
    /// them.
    pub fn begin_move_to(&mut self, ci: usize, stop: usize) -> Result<()> {
        self.begin_turn(ci, Turn::Stop(stop))
    }

    /// Start turning grip `ci`, either the way `turn` points or to the stop it
    /// names.
    pub fn begin_turn(&mut self, ci: usize, turn: Turn) -> Result<()> {
        if ci >= self.grips.len() {
            return Err(Error::Range(format!(
                "grip index {ci} out of range (have {})",
                self.grips.len()
            )));
        }
        if self.active.is_some() {
            self.end_move();
        }
        let cut = self.grips[ci].clone();
        let rots = find_stops(&mut self.puzzle, &cut)?;
        let dir = match turn {
            Turn::Direction(d) => d,
            Turn::Stop(_) => 0,
        };

        let (rot, mut angle) = match turn {
            Turn::Stop(k) => {
                let r = rots.get(k).ok_or_else(|| {
                    Error::Range(format!(
                        "stop index {k} out of range (grip {ci} has {})",
                        rots.len()
                    ))
                })?;
                (r.clone(), r.approx_angle()?)
            },
            Turn::Direction(_) if rots.is_empty() => {
                // No move possible, so spin a full turn and come back.
                (ExactQuaternion::identity(), std::f64::consts::TAU)
            },
            Turn::Direction(_) => {
                let zi = if dir < 0 {
                    let mut zi = rots.len() - 1;
                    while rots[zi].pseudo_angle()?.is_zero() && zi > 0 {
                        zi -= 1;
                    }
                    zi
                } else {
                    let mut zi = 0;
                    while rots[zi].pseudo_angle()?.is_zero() && zi < rots.len() - 1 {
                        zi += 1;
                    }
                    zi
                };
                (rots[zi].clone(), rots[zi].approx_angle()?)
            },
        };
        if dir < 0 {
            while angle >= 0.0 {
                angle -= std::f64::consts::TAU;
            }
        }
        if dir > 0 {
            while angle <= 0.0 {
                angle += std::f64::consts::TAU;
            }
        }

        // A one-radian rotation about the axis, and slerping to it by `angle`
        // avoids the ambiguity of an exactly 180-degree target.
        let axis = cut.plane.normal.to_f64()?.normalize();
        let urot = Quat::from_axis_angle(&axis, 1.0);

        let mut from_quat = Vec::with_capacity(cut.front.len());
        let mut step_quat = Vec::with_capacity(cut.front.len());
        for &i in &cut.front {
            let prot = self.puzzle.pieces[i].rot.to_f64()?;
            from_quat.push(self.puzzle.global_rot.mul(&prot));
            step_quat.push(self.puzzle.global_rot.mul(&urot).mul(&prot));
        }

        self.active = Some(ActiveMove {
            duration: angle.abs() / RAD_PER_SEC * 1000.0,
            elapsed: 0.0,
            pieces: cut.front.clone(),
            from_quat,
            step_quat,
            angle,
        });

        make_move(&mut self.puzzle, &cut, &rot)?;
        self.move_count += 1;
        self.history.push((cut.plane.clone(), rot.clone()));
        self.refresh_grips()?;
        Ok(())
    }

    /// Take the last turn back, returning `false` if there is none to take.
    ///
    /// The move is recorded by the plane it turned about and the exact rotation
    /// it applied, so undoing it applies that rotation's inverse, which is
    /// right whether the turn was taken by direction or to a named stop. The
    /// plane, rather than the grip index, because the grips are re-derived
    /// after every turn and their order need not hold.
    pub fn undo(&mut self) -> Result<bool> {
        let Some((plane, rot)) = self.history.pop() else {
            return Ok(false);
        };
        let Some(gi) = self
            .grips
            .iter()
            .position(|g| crate::symbolic::same_plane(&g.plane, &plane))
        else {
            // Put it back: the layer is locked, so the move cannot be undone
            // yet, and losing the record would make it unrecoverable.
            self.history.push((plane, rot));
            return Ok(false);
        };
        self.end_move();
        let cut = self.grips[gi].clone();
        make_move(&mut self.puzzle, &cut, &rot.conj())?;
        self.move_count += 1;
        self.refresh_grips()?;
        self.snap_piece_quats();
        Ok(true)
    }

    /// Take every turn back, leaving the puzzle solved.
    ///
    /// Cheaper than building it again: a turn is undone by one exact rotation,
    /// where a rebuild re-derives the whole geometry.
    ///
    /// Returns `false` if a locked layer stopped it part-way, in which case the
    /// puzzle is as far back as it could get and the rest of the history is
    /// still there.
    pub fn restore(&mut self) -> Result<bool> {
        while !self.history.is_empty() {
            if !self.undo()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// How many turns have been made and not taken back.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// The plane and the rotation of each turn made, oldest first.
    pub fn history(&self) -> &[(crate::math::ExactPlane, ExactQuaternion)] {
        &self.history
    }

    /// Finish the current move immediately, snapping to the exact orientation.
    pub fn end_move(&mut self) {
        self.active = None;
        self.snap_piece_quats();
    }

    /// Queue `n` random moves, applied one per frame.
    pub fn scramble(&mut self, n: u32) {
        self.pending_scramble += n;
    }

    /// Finish the move in flight and apply every queued scramble move at once,
    /// leaving the puzzle at rest.
    ///
    /// The same end state `advance` reaches, without the animation: it takes
    /// the moves through the same `move_random`/`end_move` pair, in the same
    /// order.
    pub fn settle(&mut self) -> Result<()> {
        self.end_move();
        while self.pending_scramble > 0 {
            self.pending_scramble -= 1;
            self.move_random()?;
            self.end_move();
        }
        Ok(())
    }

    /// Cancel any pending scramble moves and finish the move in flight.
    pub fn cancel_scramble(&mut self) {
        self.pending_scramble = 0;
        self.end_move();
    }

    /// Apply one random move, avoiding immediately undoing the previous one.
    pub fn move_random(&mut self) -> Result<()> {
        if self.grips.is_empty() {
            return Ok(());
        }
        let mut ci = (self.rng.next_f64() * self.grips.len() as f64).floor() as usize;
        let mut dir = (self.rng.next_f64() * 2.0).floor() as i32 * 2 - 1;
        while self.last_random == Some((ci, -dir)) {
            ci = (self.rng.next_f64() * self.grips.len() as f64).floor() as usize;
            dir = (self.rng.next_f64() * 2.0).floor() as i32 * 2 - 1;
        }
        self.last_random = Some((ci, dir));
        self.begin_move(ci.min(self.grips.len() - 1), dir)
    }

    /// Advance animation and scrambling by `dt_ms` milliseconds.
    pub fn advance(&mut self, dt_ms: f64) -> Result<()> {
        if let Some(m) = self.active.as_mut() {
            m.elapsed += dt_ms;
            let ti = if m.duration > 0.0 {
                m.elapsed / m.duration
            } else {
                2.0
            };
            if ti > 1.0 {
                self.end_move();
            } else {
                let (pieces, from, step, angle) = (&m.pieces, &m.from_quat, &m.step_quat, m.angle);
                for (k, &i) in pieces.iter().enumerate() {
                    self.piece_quats[i] = Quat::slerp(&from[k], &step[k], ti * angle);
                }
            }
        } else if self.pending_scramble > 0 {
            self.pending_scramble -= 1;
            self.move_random()?;
        }
        self.controls.update(&mut self.camera);
        Ok(())
    }

    /// Render the current state.
    pub fn render(&self, width: u32, height: u32) -> Framebuffer {
        let mut arrows = self.arrows.clone();
        if let Some(h) = self.hovered_arrow {
            if let Some(a) = arrows.get_mut(h) {
                a.highlighted = true;
            }
        }
        let mut camera = self.camera;
        camera.aspect = if height == 0 {
            1.0
        } else {
            f64::from(width) / f64::from(height)
        };
        render_frame(
            &self.meshes,
            &self.piece_quats,
            self.scale,
            &arrows,
            &camera,
            width,
            height,
            &self.options,
        )
    }

    /// Advance by `dt_ms` and render, the usual per-frame call.
    pub fn frame(&mut self, dt_ms: f64, width: u32, height: u32) -> Result<Framebuffer> {
        self.advance(dt_ms)?;
        Ok(self.render(width, height))
    }

    /// Every rotation grip `ci` can be turned to, in increasing angle, as the
    /// exact quaternions the mover uses.
    pub fn stops(&mut self, ci: usize) -> Result<Vec<ExactQuaternion>> {
        let cut = self
            .grips
            .get(ci)
            .ok_or_else(|| {
                Error::Range(format!(
                    "grip index {ci} out of range (have {})",
                    self.grips.len()
                ))
            })?
            .clone();
        find_stops(&mut self.puzzle, &cut)
    }

    /* --- symbolic and numerical state ---------------------------------- */

    /// The sticker numbering and move names for this puzzle, built on first
    /// use and kept thereafter.
    ///
    /// Numbering the slots needs the solved puzzle. If this one has already
    /// turned, the recipe is built a second time to recover it, so the first
    /// call on a scrambled puzzle costs a build.
    pub fn symbolic(&mut self) -> Result<Arc<SymbolicView>> {
        if let Some(v) = &self.symbolic {
            return Ok(Arc::clone(v));
        }
        let view = if self.move_count == 0 {
            Arc::new(SymbolicView {
                stickers: StickerMap::build(&self.puzzle)?,
                actions: ActionTable::build(&self.grips)?,
            })
        } else {
            let mut solved = build(&self.recipe)?.puzzle;
            let grips = turnable_cuts(&mut solved)?;
            Arc::new(SymbolicView {
                stickers: StickerMap::build(&solved)?,
                actions: ActionTable::build(&grips)?,
            })
        };
        self.symbolic = Some(Arc::clone(&view));
        Ok(view)
    }

    /// Adopt a sticker numbering built elsewhere.
    ///
    /// Every puzzle built from one recipe numbers its slots the same way, so a
    /// batch of them can share a single table instead of each building its own.
    /// The caller is responsible for the two recipes being the same.
    pub fn adopt_symbolic(&mut self, view: Arc<SymbolicView>) {
        self.symbolic = Some(view);
        self.slot_cache.clear();
    }

    /// The color in each sticker slot: the state as a network sees it.
    pub fn stickers(&mut self) -> Result<Vec<u16>> {
        let view = self.symbolic()?;
        view.stickers.colors(&self.puzzle, &mut self.slot_cache)
    }

    /// Where the sticker in each slot belongs: the state as a permutation.
    pub fn sticker_ids(&mut self) -> Result<Vec<u32>> {
        let view = self.symbolic()?;
        view.stickers
            .permutation(&self.puzzle, &mut self.slot_cache)
    }

    /// Whether every sticker is the color it should be.
    pub fn is_solved(&mut self) -> Result<bool> {
        let view = self.symbolic()?;
        view.stickers.is_solved(&self.puzzle, &mut self.slot_cache)
    }

    /// Turn the grip an action names, without animating.
    ///
    /// Returns `false` if the grip is not turnable in the puzzle's current
    /// state, having changed nothing.
    pub fn apply_action(&mut self, action: usize) -> Result<bool> {
        let view = self.symbolic()?;
        if action >= view.actions.action_count() {
            return Err(Error::Range(format!(
                "action {action} out of range (have {})",
                view.actions.action_count()
            )));
        }
        let Some(gi) = view.actions.locate(action, &self.grips) else {
            return Ok(false);
        };
        let dir = if action % 2 == 0 { 1 } else { -1 };
        self.begin_move(gi, dir)?;
        self.end_move();
        Ok(true)
    }

    /// Which of this puzzle's actions can be taken right now.
    pub fn action_mask(&mut self) -> Result<Vec<bool>> {
        let view = self.symbolic()?;
        Ok((0..view.actions.action_count())
            .map(|a| view.actions.locate(a, &self.grips).is_some())
            .collect())
    }

    /// Background color used when rendering.
    pub fn set_background(&mut self, color: Color, alpha: f64) {
        let [r, g, b] = color.to_srgb_bytes();
        self.options.background = [r, g, b, (alpha.clamp(0.0, 1.0) * 255.0).round() as u8];
    }
}

/// Every cut a puzzle can be turned about: one that does not pass through
/// the center turns one way, and one that does turns both.
///
/// Normals point away from the origin.
pub fn turnable_cuts(puzzle: &mut Puzzle) -> Result<Vec<Cut>> {
    let mut grips: Vec<Cut> = Vec::new();
    for cut in find_cuts(puzzle, None)? {
        let s = cut.plane.constant.sign()?;
        if s <= 0 {
            grips.push(cut.clone());
        }
        if s >= 0 {
            grips.push(cut.neg());
        }
    }
    crate::sort::sort_by(&mut grips, |a, b| {
        b.plane.constant.compare(&a.plane.constant)
    })?;
    Ok(grips)
}

#[inline]
fn transform(m: &Mat4, v: Vec3) -> Vec3 {
    let p = m.transform_point4(v);
    Vec3::new(p[0], p[1], p[2])
}

/// Moller-Trumbore ray/triangle intersection, hitting both faces.
///
/// Returns the ray parameter of the hit, if it is in front of the origin.
fn ray_triangle(origin: Vec3, dir: Vec3, v: &[Vec3; 3]) -> Option<f64> {
    const EPS: f64 = 1e-12;
    let e1 = v[1].sub(&v[0]);
    let e2 = v[2].sub(&v[0]);
    let p = dir.cross(&e2);
    let det = e1.dot(&p);
    if det.abs() < EPS {
        return None;
    }
    let inv_det = 1.0 / det;
    let t = origin.sub(&v[0]);
    let u = t.dot(&p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = t.cross(&e1);
    let w = dir.dot(&q) * inv_det;
    if w < 0.0 || u + w > 1.0 {
        return None;
    }
    let dist = e2.dot(&q) * inv_det;
    if dist > EPS {
        Some(dist)
    } else {
        None
    }
}
