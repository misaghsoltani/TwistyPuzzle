//! Convex polyhedral pieces and exact plane slicing.
//!
//! Map choices here are normative instead of incidental: `edge_index`
//! iterates in insertion order (`IndexMap`) and `cutgraph` in ascending key
//! order (`BTreeMap`). `cutgraph` order decides which vertex starts the
//! traversal, and therefore the winding (and thus the outward normal) of
//! every generated cut face. See `SEMANTICS.md` §2.

use std::collections::BTreeMap;

use indexmap::IndexMap;

use crate::color::Color;
use crate::exact::AlgebraicNumber;
use crate::math::{ExactPlane, ExactQuaternion, ExactVector3};
use crate::num::ring::Elem;
use crate::Result;

#[derive(Clone)]
pub struct PolyFace {
    pub vertices: Vec<usize>,
    pub plane: ExactPlane,
    pub color: Color,
    pub interior: bool,
}

#[derive(Clone)]
pub struct PolyGeometry {
    pub vertices: Vec<ExactVector3>,
    pub faces: Vec<PolyFace>,
    pub rot: ExactQuaternion,
}

impl PolyGeometry {
    pub fn new(vertices: Vec<ExactVector3>, faces: Vec<PolyFace>) -> PolyGeometry {
        PolyGeometry {
            vertices,
            faces,
            rot: ExactQuaternion::identity(),
        }
    }

    pub fn empty() -> PolyGeometry {
        PolyGeometry::new(Vec::new(), Vec::new())
    }

    /// Triangulated, flat-shaded vertex buffers, equivalent to the
    /// buffers the renderer takes: a triangle fan per face
    /// carrying the face normal and color on every vertex.
    pub fn to_buffers(&self) -> Result<TriangleBuffers> {
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut colors: Vec<f32> = Vec::new();
        for pf in &self.faces {
            let vs = &pf.vertices;
            let n = pf.plane.to_f64()?.normal;
            for i in 1..vs.len().saturating_sub(1) {
                let v0 = self.vertices[vs[0]].to_f64()?;
                let vcur = self.vertices[vs[i]].to_f64()?;
                let vnext = self.vertices[vs[i + 1]].to_f64()?;
                for v in [v0, vcur, vnext] {
                    positions.extend_from_slice(&[v.x as f32, v.y as f32, v.z as f32]);
                    normals.extend_from_slice(&[n.x as f32, n.y as f32, n.z as f32]);
                    colors.extend_from_slice(&[pf.color.r as f32, pf.color.g as f32, pf.color.b as f32, 1.0]);
                }
            }
        }
        Ok(TriangleBuffers {
            positions,
            normals,
            colors,
        })
    }
}

pub struct TriangleBuffers {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub colors: Vec<f32>,
}

/// An axis-aligned cube of half-width `d`.
pub fn cube_polygeometry(d: &AlgebraicNumber, color: Color, interior: bool) -> Result<PolyGeometry> {
    let mut g = PolyGeometry::empty();
    let nd = Elem::neg(d);
    for z in [&nd, d] {
        for y in [&nd, d] {
            for x in [&nd, d] {
                g.vertices.push(ExactVector3::new(x.clone(), y.clone(), z.clone()));
            }
        }
    }
    let one = d.field.from_vector_i64(&[1])?;
    let make_plane = |g: &PolyGeometry, x: usize, y: usize, z: usize| -> Result<ExactPlane> {
        let u = g.vertices[y].sub(&g.vertices[x])?;
        let v = g.vertices[z].sub(&g.vertices[y])?;
        Ok(ExactPlane::new(u.cross(&v)?, one.clone()))
    };
    for i in [1usize, 2, 4] {
        let j = if i < 4 { i * 2 } else { 1 };
        let k = if j < 4 { j * 2 } else { 1 };
        let p0 = make_plane(&g, 0, k, j + k)?;
        g.faces.push(PolyFace {
            vertices: vec![0, k, j + k, j],
            plane: p0,
            color,
            interior,
        });
        let p1 = make_plane(&g, i, j, i + j + k)?;
        g.faces.push(PolyFace {
            vertices: vec![i, i + j, i + j + k, i + k],
            plane: p1,
            color,
            interior,
        });
    }
    Ok(g)
}

/// Slice a `PolyGeometry` along a plane into `(front, back)`, where either may be empty.
///
/// Unlike a generic mesh slicer this preserves face colors and closes both
/// halves with a newly created face.
///
/// * `color`: color for any newly created face
/// * `interior`: whether the new face on the back piece is interior
pub fn slice_polygeometry(
    geometry: &PolyGeometry,
    plane: &ExactPlane,
    color: Color,
    interior: bool,
) -> Result<(PolyGeometry, PolyGeometry)> {
    let mut vertices: Vec<ExactVector3> = geometry.vertices.clone();
    let mut sides: Vec<i32> = Vec::with_capacity(vertices.len());
    for v in &vertices {
        sides.push(plane.side(v)?);
    }

    if sides.iter().all(|&s| s >= 0) {
        return Ok((geometry.clone(), PolyGeometry::empty()));
    }
    if sides.iter().all(|&s| s <= 0) {
        return Ok((PolyGeometry::empty(), geometry.clone()));
    }

    let mut front = PolyGeometry::empty();
    let mut frontmap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut back = PolyGeometry::empty();
    let mut backmap: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    // Cache of edge intersections, keyed by the edge's endpoints.
    let mut cross_index: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();

    for face in &geometry.faces {
        // Slice the face into frontpoints and backpoints.
        let mut frontpoints: Vec<usize> = Vec::new();
        let mut backpoints: Vec<usize> = Vec::new();
        let mut prev = face.vertices[face.vertices.len() - 1];
        for &cur in &face.vertices {
            if sides[cur] * sides[prev] < 0 {
                // Opposite sides: add the intersection point as a new vertex.
                let h = if sides[cur] > 0 { (prev, cur) } else { (cur, prev) };
                let c = if let Some(&c) = cross_index.get(&h) {
                    c
                } else {
                    let point = plane.intersect_line(&vertices[prev], &vertices[cur])?;
                    vertices.push(point);
                    sides.push(0);
                    let c = vertices.len() - 1;
                    cross_index.insert(h, c);
                    c
                };
                frontpoints.push(c);
                backpoints.push(c);
            }
            if sides[cur] >= 0 {
                frontpoints.push(cur);
            }
            if sides[cur] <= 0 {
                backpoints.push(cur);
            }
            prev = cur;
        }
        if frontpoints.len() >= 3 {
            for p in &mut frontpoints {
                let v = *p;
                let idx = *frontmap.entry(v).or_insert_with(|| {
                    front.vertices.push(vertices[v].clone());
                    front.vertices.len() - 1
                });
                *p = idx;
            }
            front.faces.push(PolyFace {
                vertices: frontpoints,
                plane: face.plane.clone(),
                color: face.color,
                interior: face.interior,
            });
        }
        if backpoints.len() >= 3 {
            for p in &mut backpoints {
                let v = *p;
                let idx = *backmap.entry(v).or_insert_with(|| {
                    back.vertices.push(vertices[v].clone());
                    back.vertices.len() - 1
                });
                *p = idx;
            }
            back.faces.push(PolyFace {
                vertices: backpoints,
                plane: face.plane.clone(),
                color: face.color,
                interior: face.interior,
            });
        }
    }

    close_polyhedron(&mut back, plane, color, interior);
    close_polyhedron(&mut front, &plane.neg(), color, true);

    Ok((front, back))
}

/// Find the missing face of a polyhedron and add it.
///
/// The edges of the missing face are the ones that appear exactly once. They
/// form an undirected graph which is traversed to recover the cut polygon in
/// counterclockwise order.
fn close_polyhedron(geometry: &mut PolyGeometry, plane: &ExactPlane, color: Color, interior: bool) {
    // Insertion-ordered, and the order is observable: see the module note.
    let mut edge_index: IndexMap<(usize, usize), Vec<(usize, usize)>> = IndexMap::new();
    for face in &geometry.faces {
        let mut prev = face.vertices[face.vertices.len() - 1];
        for &cur in &face.vertices {
            let h = if prev < cur { (prev, cur) } else { (cur, prev) };
            edge_index.entry(h).or_default().push((prev, cur));
            prev = cur;
        }
    }

    // Ascending key order, which decides the winding of the new face.
    let mut cutgraph: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for edges in edge_index.values() {
        crate::console_assert!(edges.len() <= 2, "close_polyhedron: edge used more than twice");
        if edges.len() == 1 {
            let (a, b) = edges[0];
            cutgraph.entry(b).or_default().push(a);
        }
    }

    let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut n_visited = 0usize;
    let vs: Vec<usize> = cutgraph.keys().copied().collect();

    // Traverse cutgraph to put the points in counterclockwise order.
    while n_visited < vs.len() {
        let mut cutpoints: Vec<usize> = Vec::new();
        let mut v = vs.iter().copied().find(|u| !visited.contains(u));
        while let Some(cur) = v {
            cutpoints.push(cur);
            visited.insert(cur);
            n_visited += 1;
            v = cutgraph
                .get(&cur)
                .and_then(|ns| ns.iter().copied().find(|u| !visited.contains(u)));
        }
        if cutpoints.is_empty() {
            eprintln!("not all cut points visited");
            break;
        }
        geometry.faces.push(PolyFace {
            vertices: cutpoints,
            plane: plane.clone(),
            color,
            interior,
        });
    }
}
