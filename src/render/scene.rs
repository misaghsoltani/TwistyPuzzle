//! Turning a puzzle into draw calls: geometry, edges, arrows and the full
//! frame.

use indexmap::IndexMap;

use super::camera::{Camera, Mat4};
use super::framebuffer::{BlendMode, Framebuffer};
use super::raster::{draw_with, Cull, DrawCall, DrawScratch, RenderTarget};
use crate::color::{linear_to_srgb_byte, Color};
use crate::math::{Quat, Vec3};
use crate::movement::{Cut, Puzzle};
use crate::Result;

/// Intensity of the scene's single `AmbientLight`.
pub const AMBIENT_INTENSITY: f64 = 1.5;

/// Convert a linear vertex color to the byte the fragment shader would write.
///
/// With only an ambient light and a Lambert material, the whole shading
/// pipeline collapses to a scale and an sRGB encode, as detailed in `SEMANTICS.md` §7.
#[inline]
pub fn shade(color: Color, alpha: f64) -> [u8; 4] {
    [
        linear_to_srgb_byte(color.r * AMBIENT_INTENSITY),
        linear_to_srgb_byte(color.g * AMBIENT_INTENSITY),
        linear_to_srgb_byte(color.b * AMBIENT_INTENSITY),
        (alpha.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// Triangles and per-triangle colors for one piece, plus its outline edges.
pub struct PieceMesh {
    pub triangles: Vec<[Vec3; 3]>,
    pub colors: Vec<[u8; 4]>,
    pub edges: Vec<[Vec3; 2]>,
    /// Which face of the piece each triangle came from, since a face of more
    /// than three sides becomes a fan of them. What a sticker is drawn on is a
    /// face, not a triangle, so anything that colors a frame by sticker needs
    /// the way back.
    pub faces: Vec<u32>,
}

/// Every edge between two faces that meet at more than `threshold_angle`.
///
/// Keeps an edge when it is on the silhouette (used by one triangle only) or
/// when the two triangles sharing it differ in normal by more than
/// `threshold_angle` degrees. Vertices are matched by a rounded string key,
/// before hashing, so coincident vertices merge whatever route produced them.
pub fn edges_geometry(triangles: &[[Vec3; 3]], threshold_angle: f64) -> Vec<[Vec3; 2]> {
    const PRECISION: f64 = 1.0e4;

    #[derive(Clone)]
    struct Edge {
        v0: Vec3,
        v1: Vec3,
        normal: Vec3,
        live: bool,
    }

    let threshold_dot = threshold_angle.to_radians().cos();
    // Insertion-ordered, because the order the leftovers come out in decides
    // the order the edges are drawn.
    let mut edge_data: IndexMap<(i64, i64, i64, i64, i64, i64), Edge> = IndexMap::new();
    let mut out: Vec<[Vec3; 2]> = Vec::new();

    let key = |v: Vec3| -> (i64, i64, i64) {
        (
            (v.x * PRECISION).round() as i64,
            (v.y * PRECISION).round() as i64,
            (v.z * PRECISION).round() as i64,
        )
    };

    for tri in triangles {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        // Triangle.getNormal: normalize((c - b) x (a - b))
        let normal = c.sub(&b).cross(&a.sub(&b)).normalize();
        let h = [key(a), key(b), key(c)];
        if h[0] == h[1] || h[1] == h[2] || h[2] == h[0] {
            continue; // degenerate
        }
        let verts = [a, b, c];
        for j in 0..3 {
            let jn = (j + 1) % 3;
            let (k0, k1) = (h[j], h[jn]);
            let fwd = (k0.0, k0.1, k0.2, k1.0, k1.1, k1.2);
            let rev = (k1.0, k1.1, k1.2, k0.0, k0.1, k0.2);
            if let Some(e) = edge_data.get_mut(&rev) {
                if e.live {
                    if normal.dot(&e.normal) <= threshold_dot {
                        out.push([verts[j], verts[jn]]);
                    }
                    e.live = false;
                    continue;
                }
                continue;
            }
            edge_data.entry(fwd).or_insert(Edge {
                v0: verts[j],
                v1: verts[jn],
                normal,
                live: true,
            });
        }
    }

    // Unmatched edges are silhouette edges and are always kept.
    for e in edge_data.values() {
        if e.live {
            out.push([e.v0, e.v1]);
        }
    }
    out
}

/// Build the renderable mesh for one piece.
///
/// Positions pass through `f32`, as they do in a `Float32BufferAttribute`, so
/// vertex merging in [`edges_geometry`] sees values it can compare.
pub fn piece_mesh(piece: &crate::piece::PolyGeometry) -> Result<PieceMesh> {
    let mut triangles = Vec::new();
    let mut colors = Vec::new();
    let mut faces = Vec::new();
    for (fi, pf) in piece.faces.iter().enumerate() {
        let vs = &pf.vertices;
        if vs.len() < 3 {
            continue;
        }
        let color = shade(pf.color, 1.0);
        let v0 = to_f32(piece.vertices[vs[0]].to_f64()?);
        for i in 1..vs.len() - 1 {
            let vcur = to_f32(piece.vertices[vs[i]].to_f64()?);
            let vnext = to_f32(piece.vertices[vs[i + 1]].to_f64()?);
            triangles.push([v0, vcur, vnext]);
            colors.push(color);
            faces.push(fi as u32);
        }
    }
    let edges = edges_geometry(&triangles, 1.0);
    Ok(PieceMesh {
        triangles,
        colors,
        edges,
        faces,
    })
}

#[inline]
fn to_f32(v: Vec3) -> Vec3 {
    Vec3::new(f64::from(v.x as f32), f64::from(v.y as f32), f64::from(v.z as f32))
}

/// The arrow overlay drawn on each grip.
pub struct ArrowMesh {
    pub triangles: Vec<[Vec3; 3]>,
}

/// The arrow profile, as a closed polygon.
///
/// A ring segment of unit radius with an arrowhead, built from the same
/// arc sampling of 12 divisions per curve, so the silhouette is smooth
/// without being expensive. The interior triangulation is unobservable: the
/// arrow is flat-filled, so any correct triangulation paints the same pixels.
fn arrow_outline() -> Vec<[f64; 2]> {
    const ARROW_WIDTH: f64 = 0.75;
    const ARROW_HEAD_WIDTH: f64 = 1.5;
    const ARROW_HEAD_LENGTH: f64 = 0.75;
    let tail_angle = 90.0f64.to_radians();
    let head_angle = -45.0f64.to_radians() + ARROW_HEAD_LENGTH / 2.0;
    let tip_angle = -45.0f64.to_radians() - ARROW_HEAD_LENGTH / 2.0;

    // `EllipseCurve.getPoints(divisions)` yields `divisions + 1` samples.
    let arc = |r: f64, a0: f64, a1: f64, clockwise: bool, out: &mut Vec<[f64; 2]>| {
        const DIVISIONS: usize = 12;
        let two_pi = std::f64::consts::TAU;
        let mut delta = (a1 - a0) % two_pi;
        if delta.abs() < f64::EPSILON {
            delta = if (a1 - a0).abs() > f64::EPSILON { two_pi } else { 0.0 };
        }
        if clockwise && delta > 0.0 {
            delta -= two_pi;
        } else if !clockwise && delta < 0.0 {
            delta += two_pi;
        }
        for i in 0..=DIVISIONS {
            let t = i as f64 / DIVISIONS as f64;
            let angle = a0 + t * delta;
            out.push([r * angle.cos(), r * angle.sin()]);
        }
    };

    let mut pts: Vec<[f64; 2]> = Vec::new();
    arc(1.0 + ARROW_WIDTH / 2.0, tail_angle, head_angle, true, &mut pts);
    let polar = |r: f64, t: f64| [r * t.cos(), r * t.sin()];
    pts.push(polar(1.0 + ARROW_HEAD_WIDTH / 2.0, head_angle));
    pts.push(polar(1.0, tip_angle));
    pts.push(polar(1.0 - ARROW_HEAD_WIDTH / 2.0, head_angle));
    arc(1.0 - ARROW_WIDTH / 2.0, head_angle, tail_angle, false, &mut pts);
    pts
}

/// Ear-clipping triangulation of a simple polygon.
fn triangulate(poly: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let area2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let signed: f64 = (0..n)
        .map(|i| {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    let mut idx: Vec<usize> = (0..n).collect();
    if signed < 0.0 {
        idx.reverse();
    }

    let mut out = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0;
    while idx.len() > 3 && guard < 4 * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
            let (a, b, c) = (poly[ia], poly[ib], poly[ic]);
            if area2(a, b, c) <= 0.0 {
                continue; // reflex
            }
            let contains = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let p = poly[j];
                area2(a, b, p) >= 0.0 && area2(b, c, p) >= 0.0 && area2(c, a, p) >= 0.0
            });
            if contains {
                continue;
            }
            out.push([ia, ib, ic]);
            idx.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

/// The extruded arrow solid: front cap, back cap and side walls.
///
/// The flat shape extruded 0.2 along its normal, with no bevel.
pub fn arrow_mesh() -> ArrowMesh {
    const DEPTH: f64 = 0.2;
    let poly = arrow_outline();
    let faces = triangulate(&poly);
    let mut triangles = Vec::with_capacity(faces.len() * 2 + poly.len() * 2);
    for f in &faces {
        let p = |i: usize| Vec3::new(poly[i][0], poly[i][1], 0.0);
        triangles.push([p(f[0]), p(f[1]), p(f[2])]);
    }
    for f in &faces {
        let p = |i: usize| Vec3::new(poly[i][0], poly[i][1], DEPTH);
        triangles.push([p(f[2]), p(f[1]), p(f[0])]);
    }
    let n = poly.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let a = Vec3::new(poly[i][0], poly[i][1], 0.0);
        let b = Vec3::new(poly[j][0], poly[j][1], 0.0);
        let c = Vec3::new(poly[j][0], poly[j][1], DEPTH);
        let d = Vec3::new(poly[i][0], poly[i][1], DEPTH);
        triangles.push([a, b, c]);
        triangles.push([a, c, d]);
    }
    ArrowMesh { triangles }
}

/// One arrow placed in the scene.
#[derive(Clone, Copy, Debug)]
pub struct ArrowInstance {
    /// Model matrix placing the unit arrow.
    pub model: Mat4,
    /// True when the pointer is over this arrow.
    pub highlighted: bool,
}

/// Placement of the arrows for a set of grips.
///
/// Each grip contributes two arrows (the second mirrored in x for the reverse
/// direction), pushed in the order `activate_arrow` decodes with
/// `ci = i / 2`, and `dir = (i % 2) * 2 - 1`.
pub fn arrow_instances(grips: &[Cut], global_rot: Quat) -> Result<Vec<ArrowInstance>> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out = Vec::with_capacity(grips.len() * 2);
    for cut in grips {
        // Captured once, as `let h = String(cut.plane.normal)` does in the
        // reference. It cannot be recomputed after the arrow is placed:
        // `to_f64` refines the field, and a field's isolating interval is part
        // of how its numbers print, so the second string would name a
        // different key.
        let mut h = String::new();
        cut.plane.normal.write_key(&mut h);
        let d = *counts.entry(h.clone()).or_insert(0);
        let normal = cut.plane.normal.to_f64()?.normalize().apply_quat(&global_rot);
        let rot = Quat::from_unit_vectors(&Vec3::new(0.0, 0.0, 1.0), &normal);
        // `arrow.position.z = 1.25 + 0.25 * d`, then rotated onto the axis.
        let position = Vec3::new(0.0, 0.0, 1.25 + 0.25 * d as f64).apply_quat(&rot);
        let base = Mat4::compose(position, rot, Vec3::new(0.2, 0.2, 0.2));
        out.push(ArrowInstance {
            model: base,
            highlighted: false,
        });
        let mirrored = Mat4::compose(position, rot, Vec3::new(-0.2, 0.2, 0.2));
        out.push(ArrowInstance {
            model: mirrored,
            highlighted: false,
        });
        counts.insert(h, d + 1);
    }
    Ok(out)
}

/// Everything needed to draw one frame.
#[derive(Clone, Copy)]
pub struct SceneOptions {
    /// Background color, as RGBA bytes.
    pub background: [u8; 4],
    /// Supersampling factor, where 2 matches 4x multisampling closely enough
    /// that flat-shaded coverage is identical.
    pub supersample: u32,
    /// Edge line width in device pixels. WebGL ignores `linewidth` on
    /// essentially every platform, so these are 1-pixel lines.
    pub line_width: f64,
    pub draw_edges: bool,
    pub draw_arrows: bool,
    pub draw_pieces: bool,
}

impl Default for SceneOptions {
    fn default() -> Self {
        SceneOptions {
            background: [0, 0, 0, 0],
            supersample: 2,
            line_width: 1.0,
            draw_edges: true,
            draw_arrows: true,
            draw_pieces: true,
        }
    }
}

/// Cached, transform-independent geometry for a puzzle.
pub struct PuzzleMeshes {
    pub pieces: Vec<PieceMesh>,
    pub arrow: ArrowMesh,
}

impl PuzzleMeshes {
    pub fn build(puzzle: &Puzzle) -> Result<PuzzleMeshes> {
        let mut pieces = Vec::with_capacity(puzzle.pieces.len());
        for p in &puzzle.pieces {
            pieces.push(piece_mesh(p)?);
        }
        Ok(PuzzleMeshes {
            pieces,
            arrow: arrow_mesh(),
        })
    }
}

/// Buffers a frame needs that do not change between frames.
///
/// A frame at 1280x1280 with 2x supersampling rasterizes into 26 MB of color
/// and 26 MB of depth, and an interactive window throws both away sixty times
/// a second unless something holds on to them. This does. It is passed as an
/// argument instead of stored as a field so that the thing being drawn stays
/// shareable: [`crate::batch::PuzzleBatch`] draws its geometry backend from
/// several threads at once, each with a scratch of its own.
///
/// One scratch may be used for any size, camera or puzzle in turn. It grows to
/// the largest frame asked of it and keeps that much.
#[derive(Default)]
pub struct FrameScratch {
    target: RenderTarget,
    palette: ArrowPalette,
    work: DrawScratch,
}

impl FrameScratch {
    pub fn new() -> FrameScratch {
        FrameScratch::default()
    }

    /// Bytes currently held, for a caller that wants to account for them.
    pub fn capacity(&self) -> usize {
        self.target.capacity()
            + self.work.capacity()
            + (self.palette.plain.capacity() + self.palette.lit.capacity()) * 4
    }
}

/// The two flat colors every arrow is drawn in, indexed per triangle.
///
/// They depend on nothing but the background and the arrow mesh, so they are
/// built when one of those changes and not once a frame.
#[derive(Default)]
struct ArrowPalette {
    background: [u8; 4],
    plain: Vec<[u8; 4]>,
    lit: Vec<[u8; 4]>,
}

impl ArrowPalette {
    fn refresh(&mut self, background: [u8; 4], triangles: usize) {
        if self.background == background && self.plain.len() == triangles {
            return;
        }
        let lum =
            0.299 * f64::from(background[0]) + 0.587 * f64::from(background[1]) + 0.114 * f64::from(background[2]);
        let (plain, lit) = if lum > 128.0 {
            (
                shade(Color::from_hex(0x001E_293B), 0.45),
                shade(Color::from_hex(0x0025_63EB), 0.9),
            )
        } else {
            (
                shade(Color::from_hex(0x00FF_FFFF), 0.3),
                shade(Color::from_hex(0x00FF_FFCC), 0.9),
            )
        };
        self.background = background;
        self.plain.clear();
        self.plain.resize(triangles, plain);
        self.lit.clear();
        self.lit.resize(triangles, lit);
    }
}

/// Whether an arrow is drawn lit: the one the pointer is over, or one the
/// caller has already marked.
#[inline]
fn lit_up(arrow: &ArrowInstance, index: usize, hovered: Option<usize>) -> bool {
    arrow.highlighted || hovered == Some(index)
}

/// Render a frame into `out`, which is `width * height` RGBA8 pixels.
///
/// `piece_quats` gives the world orientation of each piece (the animated value
/// during a move, `global_rot * piece.rot` otherwise), and `scale` is the factor
/// that fits the puzzle in a unit sphere. `hovered` lights one arrow without
/// the caller having to copy the instances to say so.
///
/// This is [`render_frame`] without its allocations: the rasterizer's buffers
/// come from `scratch` and the finished pixels go straight where the caller
/// wants them, which for a window is the surface it is about to show.
///
/// # Errors
///
/// If `out` is not exactly `width * height * 4` bytes.
#[allow(
    clippy::too_many_arguments,
    reason = "a frame is described by this many things, and grouping them would only move the list"
)]
pub fn render_frame_into(
    scratch: &mut FrameScratch,
    meshes: &PuzzleMeshes,
    piece_quats: &[Quat],
    scale: f64,
    arrows: &[ArrowInstance],
    hovered: Option<usize>,
    camera: &Camera,
    width: u32,
    height: u32,
    opts: &SceneOptions,
    out: &mut [u8],
) -> Result<()> {
    let want = width as usize * height as usize * 4;
    if out.len() != want {
        return Err(crate::Error::Range(format!(
            "a {width}x{height} frame is {want} bytes, got {}",
            out.len()
        )));
    }
    let ss = opts.supersample.max(1);
    let target = &mut scratch.target;
    target.resize(width * ss, height * ss);
    target.clear(opts.background);
    let vp = camera.view_projection();

    let scale_v = Vec3::new(scale, scale, scale);
    let mut calls: Vec<DrawCall<'_>> = Vec::with_capacity(meshes.pieces.len() * 2 + arrows.len());

    // Opaque pass: faces, then their edges. The faces carry a polygon offset
    // so the edges, drawn at the same depth, win the depth test.
    for (i, mesh) in meshes.pieces.iter().enumerate() {
        let q = piece_quats.get(i).copied().unwrap_or(Quat::IDENTITY);
        let model = Mat4::compose(Vec3::default(), q, scale_v);
        if opts.draw_pieces {
            calls.push(DrawCall {
                model,
                triangles: &mesh.triangles,
                colors: &mesh.colors,
                cull: Cull::Back,
                ..Default::default()
            });
        }
        if opts.draw_edges {
            calls.push(DrawCall {
                model,
                lines: &mesh.edges,
                line_color: [0, 0, 0, 255],
                line_width: opts.line_width * f64::from(ss),
                // A line's quad winds one way or the other depending on the
                // segment's screen direction, so culling must be off.
                cull: Cull::None,
                // An edge lies exactly in the plane of the face it borders, so
                // it needs a nudge toward the viewer to win the depth test.
                // The alternative is to push the *faces* back with
                // `polygonOffset`. Pulling the lines forward instead is
                // equivalent and cannot be destabilized by sliver triangles.
                polygon_offset: Some((-1.0, -4.0)),
                ..Default::default()
            });
        }
    }

    // Transparent pass: arrows, blended over the opaque image. The two
    // color tables are built once and shared by every arrow, since `colors`
    // is indexed per triangle.
    let n_arrow_tris = meshes.arrow.triangles.len();
    scratch.palette.refresh(opts.background, n_arrow_tris);
    let (plain, lit) = (&scratch.palette.plain, &scratch.palette.lit);
    if opts.draw_arrows {
        for (i, a) in arrows.iter().enumerate() {
            if !lit_up(a, i, hovered) {
                calls.push(DrawCall {
                    model: a.model,
                    triangles: &meshes.arrow.triangles,
                    colors: plain,
                    cull: Cull::None,
                    blend: BlendMode::Over,
                    ..Default::default()
                });
            }
        }
        for (i, a) in arrows.iter().enumerate() {
            if lit_up(a, i, hovered) {
                calls.push(DrawCall {
                    model: a.model,
                    triangles: &meshes.arrow.triangles,
                    colors: lit,
                    cull: Cull::None,
                    blend: BlendMode::Over,
                    ..Default::default()
                });
            }
        }
    }

    draw_with(&mut scratch.work, target, &calls, &vp);
    if ss > 1 {
        target.color.downsample_into(ss, out);
    } else {
        out.copy_from_slice(target.color.as_bytes());
    }
    Ok(())
}

/// Render a frame into a framebuffer of its own.
///
/// [`render_frame_into`] with the buffers allocated and thrown away around it:
/// what a caller drawing one frame wants, and what a caller drawing a stream of
/// them should not use.
///
/// No arrow is lit except one the caller has already marked `highlighted`.
/// Lighting the arrow under the pointer is [`render_frame_into`]'s `hovered`,
/// which is what [`crate::simulator::Simulator::render`] goes through.
#[allow(
    clippy::too_many_arguments,
    reason = "a frame is described by this many things, and grouping them would only move the list"
)]
pub fn render_frame(
    meshes: &PuzzleMeshes,
    piece_quats: &[Quat],
    scale: f64,
    arrows: &[ArrowInstance],
    camera: &Camera,
    width: u32,
    height: u32,
    opts: &SceneOptions,
) -> Framebuffer {
    let mut fb = Framebuffer::new(width, height);
    let mut scratch = FrameScratch::new();
    render_frame_into(
        &mut scratch,
        meshes,
        piece_quats,
        scale,
        arrows,
        None,
        camera,
        width,
        height,
        opts,
        fb.as_bytes_mut(),
    )
    .expect("a framebuffer of the size asked for");
    fb
}
