//! A tiled, parallel triangle and line rasterizer with a depth buffer.
//!
//! Geometry is transformed once, binned into screen tiles, then each tile is
//! rasterized independently. Tiles touch disjoint pixels, so the work
//! parallelizes without locking, and the result does not depend on how it was
//! scheduled: within a tile, primitives are always visited in submission
//! order, so depth ties break deterministically.
//!
//! Coverage uses edge functions with a consistent sign test, so an edge shared
//! by two triangles is rasterized exactly once, preventing cracks and
//! double-blending of translucent geometry.

use rayon::prelude::*;

use super::camera::Mat4;
use super::framebuffer::{BlendMode, Framebuffer};
use crate::math::Vec3;

/// Which side of a triangle to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    /// Front faces only.
    Back,
    /// Back faces only.
    Front,
    /// Both.
    None,
}

/// A batch of primitives sharing render state and a model matrix.
pub struct DrawCall<'a> {
    /// Model matrix applied to this call's geometry.
    pub model: Mat4,
    /// Triangle vertices in model space, three per triangle.
    pub triangles: &'a [[Vec3; 3]],
    /// One sRGB color per triangle.
    pub colors: &'a [[u8; 4]],
    /// Line endpoints in model space, two per segment.
    pub lines: &'a [[Vec3; 2]],
    pub line_color: [u8; 4],
    /// Line width in device pixels.
    pub line_width: f64,
    pub cull: Cull,
    pub blend: BlendMode,
    pub depth_write: bool,
    pub depth_test: bool,
    /// `polygonOffsetFactor` and `polygonOffsetUnits`, as in OpenGL.
    pub polygon_offset: Option<(f32, f32)>,
}

impl Default for DrawCall<'_> {
    fn default() -> Self {
        DrawCall {
            model: Mat4::IDENTITY,
            triangles: &[],
            colors: &[],
            lines: &[],
            line_color: [0, 0, 0, 255],
            line_width: 1.0,
            cull: Cull::Back,
            blend: BlendMode::Copy,
            depth_write: true,
            depth_test: true,
            polygon_offset: None,
        }
    }
}

#[derive(Clone, Copy)]
struct TriState {
    blend: BlendMode,
    depth_write: bool,
    depth_test: bool,
    polygon_offset: Option<(f32, f32)>,
}

/// A triangle projected into screen space and ready to rasterize.
///
/// Coordinates are kept in `f64`. The edge functions are stepped
/// incrementally across a scanline, and in `f32` the accumulated rounding is
/// enough to flip a sign next to a shared edge, which opens visible cracks
/// between adjacent triangles.
#[derive(Clone, Copy)]
struct ScreenTri {
    /// `(x, y, z)` in pixels and normalized depth.
    v: [[f64; 3]; 3],
    color: [u8; 4],
    /// Index into the per-call render-state table.
    state: u32,
}

/// Height, in rows, of one parallel rasterization band.
const BAND: u32 = 32;

/// Largest depth slope the polygon-offset term will compensate for.
const SLOPE_LIMIT: f64 = 1.0e-5;

/// Color plus depth: the target of a render pass.
pub struct RenderTarget {
    pub color: Framebuffer,
    pub depth: Vec<f32>,
}

impl Default for RenderTarget {
    fn default() -> RenderTarget {
        RenderTarget::new(0, 0)
    }
}

impl RenderTarget {
    pub fn new(width: u32, height: u32) -> RenderTarget {
        RenderTarget {
            color: Framebuffer::new(width, height),
            depth: vec![f32::INFINITY; width as usize * height as usize],
        }
    }

    /// Make this the given size, keeping the buffers when they already are.
    ///
    /// The contents are not preserved and are not cleared either: every caller
    /// clears to a background immediately afterward, and zeroing first would
    /// be writing the whole surface twice.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if self.width() == width && self.height() == height {
            return;
        }
        self.color.resize(width, height);
        self.depth.resize(width as usize * height as usize, f32::INFINITY);
    }

    /// Bytes held by the two buffers.
    pub fn capacity(&self) -> usize {
        self.color.capacity() + self.depth.capacity() * 4
    }

    pub fn clear(&mut self, rgba: [u8; 4]) {
        self.color.fill(rgba);
        self.depth.fill(f32::INFINITY);
    }

    pub fn width(&self) -> u32 {
        self.color.width()
    }

    pub fn height(&self) -> u32 {
        self.color.height()
    }
}

#[inline]
fn project(mvp: &Mat4, p: Vec3, w: f64, h: f64) -> Option<[f64; 3]> {
    let c = mvp.transform_point4(p);
    if c[3] <= 1e-9 {
        return None;
    }
    let inv = 1.0 / c[3];
    Some([
        (c[0] * inv + 1.0) * 0.5 * w,
        (1.0 - c[1] * inv) * 0.5 * h,
        (c[2] * inv + 1.0) * 0.5,
    ])
}

/// Clip a triangle against the near plane, emitting a fan of screen-space
/// triangles (none if it lies entirely behind the camera).
fn clip_near(mvp: &Mat4, tri: &[Vec3; 3], w: f64, h: f64, out: &mut Vec<[[f64; 3]; 3]>) {
    const EPS: f64 = 1e-9;
    let c: [[f64; 4]; 3] = [
        mvp.transform_point4(tri[0]),
        mvp.transform_point4(tri[1]),
        mvp.transform_point4(tri[2]),
    ];
    let dist = |p: &[f64; 4]| p[2] + p[3];
    let all_in = c.iter().all(|p| p[3] > EPS && dist(p) >= 0.0);
    if all_in {
        let mut v = [[0.0f64; 3]; 3];
        for i in 0..3 {
            match project(mvp, tri[i], w, h) {
                Some(p) => v[i] = p,
                None => return,
            }
        }
        out.push(v);
        return;
    }
    if c.iter().all(|p| dist(p) < 0.0) {
        return;
    }

    // Sutherland-Hodgman against `z + w >= 0` in clip space.
    let mut poly: Vec<[f64; 4]> = Vec::with_capacity(4);
    for i in 0..3 {
        let a = c[i];
        let b = c[(i + 1) % 3];
        let (da, db) = (dist(&a), dist(&b));
        if da >= 0.0 {
            poly.push(a);
        }
        if (da >= 0.0) != (db >= 0.0) {
            let t = da / (da - db);
            poly.push([
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            ]);
        }
    }
    if poly.len() < 3 {
        return;
    }
    let to_screen = |p: &[f64; 4]| -> [f64; 3] {
        let inv = 1.0 / p[3].max(EPS);
        [
            (p[0] * inv + 1.0) * 0.5 * w,
            (1.0 - p[1] * inv) * 0.5 * h,
            (p[2] * inv + 1.0) * 0.5,
        ]
    };
    let s: Vec<[f64; 3]> = poly.iter().map(to_screen).collect();
    for i in 1..s.len() - 1 {
        out.push([s[0], s[i], s[i + 1]]);
    }
}

#[inline]
fn edge(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Pixel bounding box clipped to the surface, or `None` if fully outside.
fn bbox(t: &ScreenTri, width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 {
        return None;
    }
    let fminx = t.v[0][0].min(t.v[1][0]).min(t.v[2][0]);
    let fminy = t.v[0][1].min(t.v[1][1]).min(t.v[2][1]);
    let fmaxx = t.v[0][0].max(t.v[1][0]).max(t.v[2][0]);
    let fmaxy = t.v[0][1].max(t.v[1][1]).max(t.v[2][1]);
    // Negated deliberately: a NaN coordinate fails `>=` and is rejected here,
    // where `fmaxx < 0.0` would let it through.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(fmaxx >= 0.0) || !(fmaxy >= 0.0) {
        return None;
    }
    if fminx > f64::from(width - 1) || fminy > f64::from(height - 1) {
        return None;
    }
    Some((
        fminx.floor().max(0.0) as u32,
        fminy.floor().max(0.0) as u32,
        (fmaxx.ceil() as u32).min(width - 1),
        (fmaxy.ceil() as u32).min(height - 1),
    ))
}

/// Rasterize `calls` into `target`, in order.
/// The working lists a [`draw`] builds, kept so a second call can refill them
/// instead of allocating them again.
///
/// A frame's triangles, its per-call render states and one index list per
/// horizontal band: a 2560-row surface has eighty bands, so eighty vectors are
/// grown from nothing on every call that does not reuse these.
#[derive(Default)]
pub struct DrawScratch {
    states: Vec<TriState>,
    tris: Vec<ScreenTri>,
    clip: Vec<[[f64; 3]; 3]>,
    bins: Vec<Vec<u32>>,
}

impl DrawScratch {
    /// Bytes held by the working lists.
    pub fn capacity(&self) -> usize {
        self.states.capacity() * size_of::<TriState>()
            + self.tris.capacity() * size_of::<ScreenTri>()
            + self.clip.capacity() * size_of::<[[f64; 3]; 3]>()
            + self.bins.iter().map(|b| b.capacity() * 4).sum::<usize>()
    }
}

pub fn draw(target: &mut RenderTarget, calls: &[DrawCall<'_>], view_projection: &Mat4) {
    draw_with(&mut DrawScratch::default(), target, calls, view_projection);
}

/// [`draw`], reusing the caller's working lists.
pub fn draw_with(work: &mut DrawScratch, target: &mut RenderTarget, calls: &[DrawCall<'_>], view_projection: &Mat4) {
    let width = target.width();
    let height = target.height();
    if width == 0 || height == 0 || calls.is_empty() {
        return;
    }
    let (wf, hf) = (f64::from(width), f64::from(height));

    let DrawScratch {
        states,
        tris,
        clip: scratch,
        bins,
    } = work;
    states.clear();
    states.extend(calls.iter().map(|c| TriState {
        blend: c.blend,
        depth_write: c.depth_write,
        depth_test: c.depth_test,
        polygon_offset: c.polygon_offset,
    }));

    // 1. Transform every primitive to screen space.
    tris.clear();
    for (call_index, call) in calls.iter().enumerate() {
        let mvp = view_projection.mul(&call.model);
        let state = call_index as u32;

        for (ti, tri) in call.triangles.iter().enumerate() {
            scratch.clear();
            clip_near(&mvp, tri, wf, hf, scratch);
            let color = call.colors.get(ti).copied().unwrap_or([255, 255, 255, 255]);
            for v in scratch.iter() {
                let area = edge(v[0], v[1], v[2]);
                // Screen y points down, so a counterclockwise winding in world
                // space gives a negative signed area here.
                let keep = match call.cull {
                    Cull::Back => area < 0.0,
                    Cull::Front => area > 0.0,
                    Cull::None => area != 0.0,
                };
                if keep {
                    tris.push(ScreenTri { v: *v, color, state });
                }
            }
        }

        // Lines become screen-space quads of the requested pixel width.
        if !call.lines.is_empty() {
            let hw = call.line_width * 0.5;
            for seg in call.lines {
                let (Some(a), Some(b)) = (project(&mvp, seg[0], wf, hf), project(&mvp, seg[1], wf, hf)) else {
                    continue;
                };
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-12 {
                    continue;
                }
                let (nx, ny) = (-dy / len * hw, dx / len * hw);
                let p0 = [a[0] + nx, a[1] + ny, a[2]];
                let p1 = [a[0] - nx, a[1] - ny, a[2]];
                let p2 = [b[0] - nx, b[1] - ny, b[2]];
                let p3 = [b[0] + nx, b[1] + ny, b[2]];
                for v in [[p0, p1, p2], [p0, p2, p3]] {
                    if edge(v[0], v[1], v[2]) != 0.0 {
                        tris.push(ScreenTri {
                            v,
                            color: call.line_color,
                            state,
                        });
                    }
                }
            }
        }
    }
    if tris.is_empty() {
        return;
    }

    // 2. Bin into horizontal bands. Bands are disjoint row ranges, so they
    // can be handed to `par_chunks_mut` and rasterized without any sharing.
    let bands = height.div_ceil(BAND) as usize;
    bins.truncate(bands);
    bins.resize_with(bands, Vec::new);
    for b in bins.iter_mut() {
        b.clear();
    }
    for (i, t) in tris.iter().enumerate() {
        let Some((_, miny, _, maxy)) = bbox(t, width, height) else {
            continue;
        };
        for band in (miny / BAND)..=(maxy / BAND) {
            bins[band as usize].push(i as u32);
        }
    }

    // 3. Rasterize bands in parallel.
    let stride = width as usize * 4;
    let rows = BAND as usize;
    target
        .color
        .as_bytes_mut()
        .par_chunks_mut(stride * rows)
        .zip(target.depth.par_chunks_mut(width as usize * rows))
        .enumerate()
        .for_each(|(band, (color, depth))| {
            let list = &bins[band];
            if list.is_empty() {
                return;
            }
            let y0 = band as u32 * BAND;
            let y1 = (y0 + BAND).min(height);
            for &ti in list {
                let t = &tris[ti as usize];
                raster_tri(t, &states[t.state as usize], y0, y1, width, height, depth, color);
            }
        });
}

#[allow(clippy::too_many_arguments)]
fn raster_tri(
    t: &ScreenTri,
    st: &TriState,
    band_y0: u32,
    band_y1: u32,
    width: u32,
    height: u32,
    depth: &mut [f32],
    color: &mut [u8],
) {
    let Some((bx0, by0, bx1, by1)) = bbox(t, width, height) else {
        return;
    };
    let x0 = bx0;
    let y0 = by0.max(band_y0);
    let x1 = bx1;
    let y1 = by1.min(band_y1 - 1);
    if x0 > x1 || y0 > y1 {
        return;
    }

    let (a, b, c) = (t.v[0], t.v[1], t.v[2]);
    let area = edge(a, b, c);
    if area == 0.0 {
        return;
    }
    let inv_area = 1.0 / area;
    let positive = area > 0.0;

    // Depth bias: OpenGL's `factor * slope + units * r`.
    let bias: f64 = match st.polygon_offset {
        Some((factor, units)) => {
            let d1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let d2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let det = d1[0] * d2[1] - d1[1] * d2[0];
            let m = if det.abs() < 1e-6 {
                // A sliver triangle has an almost undefined screen-space
                // gradient, where OpenGL would compute a huge slope here and push
                // the polygon far enough back for hidden geometry to show
                // through, so treat it as flat.
                0.0
            } else {
                let dzdx = (d1[2] * d2[1] - d2[2] * d1[1]) / det;
                let dzdy = (d2[2] * d1[0] - d1[2] * d2[0]) / det;
                // Clamp for the same reason: one pixel of depth is the most
                // slope compensation that can ever be wanted.
                dzdx.abs().max(dzdy.abs()).min(SLOPE_LIMIT)
            };
            // `r`, the smallest resolvable depth difference, for an f32 buffer.
            f64::from(factor) * m + f64::from(units) * 1e-6
        },
        None => 0.0,
    };

    // Edge functions are affine in the pixel center, so step them incrementally
    // instead of recomputing three cross products per pixel.
    let px0 = f64::from(x0) + 0.5;
    let py0 = f64::from(y0) + 0.5;
    let p0 = [px0, py0, 0.0];
    let (mut row0, mut row1, mut row2) = (edge(b, c, p0), edge(c, a, p0), edge(a, b, p0));
    // w0 = edge(b, c, p), so dw0/dx = b.y - c.y and dw0/dy = c.x - b.x.
    let dx0 = b[1] - c[1];
    let dx1 = c[1] - a[1];
    let dx2 = a[1] - b[1];
    let dy0 = c[0] - b[0];
    let dy1 = a[0] - c[0];
    let dy2 = b[0] - a[0];

    for y in y0..=y1 {
        let (mut w0, mut w1, mut w2) = (row0, row1, row2);
        let base = (y - band_y0) as usize * width as usize;
        for x in x0..=x1 {
            let inside = if positive {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if inside {
                let z = ((w0 * a[2] + w1 * b[2] + w2 * c[2]) * inv_area + bias) as f32;
                let idx = base + x as usize;
                if !st.depth_test || z < depth[idx] {
                    if st.depth_write {
                        depth[idx] = z;
                    }
                    let o = idx * 4;
                    blend_into(&mut color[o..o + 4], t.color, st.blend);
                }
            }
            w0 += dx0;
            w1 += dx1;
            w2 += dx2;
        }
        row0 += dy0;
        row1 += dy1;
        row2 += dy2;
    }
}

#[inline]
fn blend_into(dst: &mut [u8], src: [u8; 4], mode: BlendMode) {
    match mode {
        BlendMode::Copy => dst.copy_from_slice(&src),
        BlendMode::Over => {
            let a = u32::from(src[3]);
            if a == 255 {
                dst.copy_from_slice(&src);
                return;
            }
            if a == 0 {
                return;
            }
            let ia = 255 - a;
            for i in 0..3 {
                let v = u32::from(src[i]) * a + u32::from(dst[i]) * ia + 127;
                dst[i] = ((v + (v >> 8)) >> 8) as u8;
            }
            let oa = a * 255 + u32::from(dst[3]) * ia + 127;
            dst[3] = ((oa + (oa >> 8)) >> 8) as u8;
        },
        BlendMode::Additive => {
            let a = u32::from(src[3]);
            for i in 0..3 {
                let v = u32::from(src[i]) * a + 127;
                dst[i] = (u32::from(dst[i]) + ((v + (v >> 8)) >> 8)).min(255) as u8;
            }
            dst[3] = (u32::from(dst[3]) + a).min(255) as u8;
        },
    }
}
