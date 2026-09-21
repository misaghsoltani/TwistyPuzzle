//! State visualization via precomputed pixel lookup tables instead of rasterization.
//!
//! Non-jumbling puzzles preserve their geometric envelope across all states: pieces
//! occupy identical spatial positions regardless of scramble configuration, with
//! only the facet colors varying. Consequently, under a fixed camera viewpoint
//! and frame resolution, the sticker slot projected onto each pixel is invariant to
//! puzzle state. Precomputing this mapping reduces frame rendering to a single
//! table lookup per pixel.
//!
//! This enables high-throughput rendering across large batches: whereas software
//! rasterization of a 3x3x3 puzzle requires evaluating thousands of triangle-pixel
//! coverage tests, table lookup requires only memory loads and stores. This allows
//! image-based reinforcement learning pipelines to generate visual observations
//! with minimal computational overhead.
//!
//! # How close the pixels are
//!
//! The scene has one ambient light and Lambert materials, so shading collapses
//! to a scale and an sRGB encode (`SEMANTICS.md` §7) and every fragment of a
//! face is the same byte triple. Nothing interpolates across a triangle.
//! Building the table therefore means running the same rasterizer over the
//! same geometry with each triangle painted its slot's number instead of its
//! color: the depth test picks the same winner, and the winner's identity is
//! what lands in the buffer. Supersampling resolves afterward, over the
//! colors, exactly as the ordinary path resolves over its own. On a solved
//! puzzle the two agree to the byte.
//!
//! On a scrambled one they agree on every pixel but a seam. A piece that has
//! turned occupies the same cell and shows the same surface, but it carries
//! its own triangulation and outline segments around with it, so a pixel lying
//! exactly on a boundary can fall to one side under the table's geometry and
//! to the other under the state's. Measured over the catalog that is a handful
//! of pixels in a frame (under half a percent, always on a boundary, never a
//! whole sticker). A batch is self-consistent regardless, since every frame in
//! it comes from the one table.
//!
//! `SEMANTICS.md` §10 states the conditions under which a table may be used at
//! all. In short: the puzzle must be at rest, its stickers must stay on the
//! solved lattice, and the arrows must be off, because they are blended instead
//! of opaque and their coverage is not a function of any slot.

use rayon::prelude::*;

use super::camera::Camera;
use super::framebuffer::Framebuffer;
use super::raster::{Cull, DrawCall, RenderTarget};
use super::scene::{PuzzleMeshes, SceneOptions};
use crate::layout::Cell;
use crate::math::{Quat, Vec3};
use crate::render::camera::Mat4;

/// Which sticker slot each pixel of a frame shows.
///
/// Built for one puzzle, one camera and one frame size. Change any of them and
/// the table no longer describes the picture, so a batch keeps the table it
/// built and rebuilds when asked for a different size.
pub struct StickerLut {
    width: u32,
    height: u32,
    supersample: u32,
    /// One entry per supersampled pixel: a slot number below
    /// [`sticker_count`](Self::sticker_count), or `sticker_count + i` for the
    /// `i`-th fixed color. At the narrowest width that names all of them
    /// (`SEMANTICS.md` §13), which for a 512-pixel frame drawn at two samples
    /// a side is a megabyte instead of four.
    ids: Ids,
    /// Colors that do not depend on the state: the background, the outlines,
    /// and the inside of a piece, which is only ever seen through a gap.
    literals: Vec<[u8; 4]>,
    sticker_count: usize,
    /// The RGBA each sticker color is drawn in, taken from the solved
    /// puzzle's own faces instead of recomputed, so a painted sticker is the
    /// byte triple the rasterizer would have written for it.
    palette: Vec<[u8; 4]>,
}

impl StickerLut {
    /// Work out which slot each pixel shows.
    ///
    /// `slot_of(piece, face)` gives the slot that face of that piece occupies
    /// when the puzzle is solved, or `None` for a face that carries no sticker.
    /// `piece_quats` and `scale` are the ones the ordinary renderer would use.
    ///
    /// Arrows are not drawn: they are blended over the frame instead of
    /// written into it, so their pixels are a mixture of a fixed color and
    /// whatever is behind, which no table over slots can express.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        meshes: &PuzzleMeshes,
        piece_quats: &[Quat],
        scale: f64,
        camera: &Camera,
        width: u32,
        height: u32,
        opts: &SceneOptions,
        solved_colors: &[u16],
        slot_of: &dyn Fn(usize, u32) -> Option<u32>,
    ) -> StickerLut {
        let sticker_count = solved_colors.len();
        let ss = opts.supersample.max(1);
        let mut literals: Vec<[u8; 4]> = vec![opts.background];
        let base = sticker_count as u32;

        // The number a pixel will hold, written into the color channels the
        // rasterizer copies. Every draw here is opaque, so `BlendMode::Copy`
        // moves the four bytes across untouched.
        let tag = |literals: &mut Vec<[u8; 4]>, color: [u8; 4]| -> [u8; 4] {
            let i = literals.iter().position(|&c| c == color).unwrap_or_else(|| {
                literals.push(color);
                literals.len() - 1
            });
            (base + i as u32).to_le_bytes()
        };

        // One color table per piece, parallel to its triangles. The solved
        // puzzle shows each slot its own color, so the same walk collects
        // what each color is drawn in.
        let colors = solved_colors.iter().copied().max().map_or(0, |m| m as usize + 1);
        let mut palette: Vec<[u8; 4]> = vec![[0, 0, 0, 255]; colors];
        let mut tags: Vec<Vec<[u8; 4]>> = Vec::with_capacity(meshes.pieces.len());
        for (p, mesh) in meshes.pieces.iter().enumerate() {
            let mut row = Vec::with_capacity(mesh.triangles.len());
            for (t, &face) in mesh.faces.iter().enumerate() {
                row.push(match slot_of(p, face) {
                    Some(slot) => {
                        if let Some(&c) = solved_colors.get(slot as usize) {
                            palette[c as usize] = mesh.colors[t];
                        }
                        slot.to_le_bytes()
                    },
                    None => tag(&mut literals, mesh.colors[t]),
                });
            }
            tags.push(row);
        }
        let edge_tag = tag(&mut literals, [0, 0, 0, 255]);
        let clear = (base).to_le_bytes();

        let mut target = RenderTarget::new(width * ss, height * ss);
        target.clear(clear);
        let scale_v = Vec3::new(scale, scale, scale);
        let mut calls: Vec<DrawCall<'_>> = Vec::with_capacity(meshes.pieces.len() * 2);
        for (i, mesh) in meshes.pieces.iter().enumerate() {
            let q = piece_quats.get(i).copied().unwrap_or(Quat::IDENTITY);
            let model = Mat4::compose(Vec3::default(), q, scale_v);
            if opts.draw_pieces {
                calls.push(DrawCall {
                    model,
                    triangles: &mesh.triangles,
                    colors: &tags[i],
                    cull: Cull::Back,
                    ..Default::default()
                });
            }
            if opts.draw_edges {
                calls.push(DrawCall {
                    model,
                    lines: &mesh.edges,
                    line_color: edge_tag,
                    line_width: opts.line_width * f64::from(ss),
                    cull: Cull::None,
                    polygon_offset: Some((-1.0, -4.0)),
                    ..Default::default()
                });
            }
        }
        super::raster::draw(&mut target, &calls, &camera.view_projection());

        let bytes = target.color.as_bytes();
        let named = sticker_count + literals.len();
        let each = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let ids = if named <= 1 << 8 {
            Ids::N8(each.map(|v| v as u8).collect())
        } else if named <= 1 << 16 {
            Ids::N16(each.map(|v| v as u16).collect())
        } else {
            Ids::N32(each.collect())
        };
        StickerLut {
            width,
            height,
            supersample: ss,
            ids,
            literals,
            sticker_count,
            palette,
        }
    }

    /// The RGBA each sticker color is drawn in, ready for
    /// [`paint`](Self::paint). Taken from the puzzle's own faces instead of
    /// recomputed, so a painted sticker is the byte triple the rasterizer
    /// would have written for it.
    pub fn palette(&self) -> &[[u8; 4]] {
        &self.palette
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn supersample(&self) -> u32 {
        self.supersample
    }

    /// How many slots the table was built over, which is how long a state
    /// handed to [`paint`](Self::paint) must be.
    pub fn sticker_count(&self) -> usize {
        self.sticker_count
    }

    /// Bytes one frame takes: `width * height * 4`.
    pub fn frame_bytes(&self) -> usize {
        self.width as usize * self.height as usize * 4
    }

    /// Paint one state into `out`, which must be [`frame_bytes`](Self::frame_bytes) long.
    ///
    /// `colors[i]` is the color now in slot `i` and `palette[c]` is the RGBA
    /// that color is drawn in, already shaded.
    pub fn paint<C: Cell>(&self, colors: &[C], palette: &[[u8; 4]], out: &mut [u8]) {
        let ss = self.supersample as usize;
        let w = self.width as usize;
        let sw = w * ss;
        let n = (ss * ss) as u32;
        let half = n / 2;
        let base = self.sticker_count;
        let lookup = |id: usize| -> [u8; 4] {
            if id < base {
                let c = colors.get(id).copied().unwrap_or(C::ZERO).index();
                palette.get(c).copied().unwrap_or([0, 0, 0, 255])
            } else {
                self.literals.get(id - base).copied().unwrap_or([0, 0, 0, 255])
            }
        };
        // The width the table happens to be is chosen once, here, and not
        // once a pixel.
        match &self.ids {
            Ids::N8(v) => resolve(v, &lookup, out, w, ss, sw, n, half),
            Ids::N16(v) => resolve(v, &lookup, out, w, ss, sw, n, half),
            Ids::N32(v) => resolve(v, &lookup, out, w, ss, sw, n, half),
        }
    }

    /// Paint one state into a fresh frame.
    pub fn frame<C: Cell>(&self, colors: &[C], palette: &[[u8; 4]]) -> Framebuffer {
        let mut out = Framebuffer::new(self.width, self.height);
        self.paint(colors, palette, out.as_bytes_mut());
        out
    }

    /// Paint a whole batch, one frame after another, across every core.
    ///
    /// `states` is `rows * sticker_count` colors and `out` is
    /// `rows * frame_bytes()` bytes.
    pub fn paint_many<C: Cell>(&self, states: &[C], palette: &[[u8; 4]], out: &mut [u8]) {
        let k = self.sticker_count.max(1);
        out.par_chunks_mut(self.frame_bytes())
            .zip(states.par_chunks(k))
            .for_each(|(frame, colors)| self.paint(colors, palette, frame));
    }
}

/// One entry per supersampled pixel, at the narrowest width that names every
/// slot and every fixed color of the puzzle the table was built for.
enum Ids {
    N8(Vec<u8>),
    N16(Vec<u16>),
    N32(Vec<u32>),
}

/// Paint the table's entries into a frame, resolving the supersamples.
///
/// Split out of [`StickerLut::paint`] so that the width of an entry is a type
/// parameter instead of a branch inside the loop.
#[allow(clippy::too_many_arguments, reason = "every one of them is a loop bound")]
fn resolve<I: Cell>(
    ids: &[I],
    lookup: &impl Fn(usize) -> [u8; 4],
    out: &mut [u8],
    w: usize,
    ss: usize,
    sw: usize,
    n: u32,
    half: u32,
) {
    if ss == 1 {
        for (px, id) in out.chunks_exact_mut(4).zip(ids.iter()) {
            px.copy_from_slice(&lookup(id.index()));
        }
        return;
    }
    // The same box filter the ordinary renderer resolves with, applied to the
    // colors the table names instead of a rendered surface.
    for (oy, row) in out.chunks_exact_mut(w * 4).enumerate() {
        for ox in 0..w {
            let mut acc = [0u32; 4];
            for sy in 0..ss {
                let start = (oy * ss + sy) * sw + ox * ss;
                for sx in 0..ss {
                    let px = lookup(ids[start + sx].index());
                    for c in 0..4 {
                        acc[c] += u32::from(px[c]);
                    }
                }
            }
            for c in 0..4 {
                row[ox * 4 + c] = ((acc[c] + half) / n) as u8;
            }
        }
    }
}
