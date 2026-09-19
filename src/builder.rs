//! Turning a recipe into a puzzle: expand the shapes into planes, promote
//! every coefficient into one common number field, then slice.

use crate::exact::{promote, AlgebraicNumber};
use crate::make::{make_cuts, make_shell};
use crate::math::{ExactPlane, ExactVector3};
use crate::movement::Puzzle;
use crate::parse::{eval_expr, PuzzleRecipe, Shape};
use crate::piece::PolyGeometry;
use crate::polyhedra::polyhedron;
use crate::Result;

pub struct BuiltPuzzle {
    pub puzzle: Puzzle,
    pub shell: PolyGeometry,
    /// Reciprocal of the shell's circumradius: the factor that scales the
    /// puzzle to fit a unit sphere.
    pub scale: f64,
    /// The number field every coordinate ended up in.
    pub field: String,
}

/// Expand shapes into planes, evaluating every coefficient exactly.
pub fn shapes_to_planes(shapes: &[Shape]) -> Result<Vec<ExactPlane>> {
    let mut pplanes: Vec<Shape> = Vec::new();
    for s in shapes {
        match s {
            Shape::Plane { .. } => pplanes.push(s.clone()),
            Shape::Polyhedron { name, d } => pplanes.extend(polyhedron(name, d)?),
        }
    }

    let mut eplanes: Vec<ExactPlane> = Vec::with_capacity(pplanes.len());
    for pp in &pplanes {
        if let Shape::Plane { a, b, c, d } = pp {
            eplanes.push(ExactPlane::new(
                ExactVector3::new(eval_expr(a)?, eval_expr(b)?, eval_expr(c)?),
                eval_expr(d)?,
            ));
        }
    }
    Ok(eplanes)
}

/// Build the puzzle described by `recipe`.
///
/// Every coordinate of every plane is promoted into a single common number
/// field first, so all later comparisons are exact.
pub fn build(recipe: &PuzzleRecipe) -> Result<BuiltPuzzle> {
    let mut shell_planes = shapes_to_planes(&recipe.shell)?;
    let mut cut_planes = shapes_to_planes(&recipe.cuts)?;

    let mut all_numbers: Vec<AlgebraicNumber> = Vec::new();
    for plane in shell_planes.iter().chain(cut_planes.iter()) {
        all_numbers.push(plane.normal.x.clone());
        all_numbers.push(plane.normal.y.clone());
        all_numbers.push(plane.normal.z.clone());
        all_numbers.push(plane.constant.clone());
    }
    promote(&mut all_numbers)?;
    let field = all_numbers
        .first()
        .map_or_else(|| "(no planes)".to_string(), |n| n.field.to_string());

    let mut i = 0;
    for plane in shell_planes.iter_mut().chain(cut_planes.iter_mut()) {
        plane.normal.x = all_numbers[i].clone();
        plane.normal.y = all_numbers[i + 1].clone();
        plane.normal.z = all_numbers[i + 2].clone();
        plane.constant = all_numbers[i + 3].clone();
        i += 4;
    }

    let shell = make_shell(&shell_planes)?;
    let puzzle = Puzzle::new(make_cuts(&cut_planes, vec![shell.clone()])?);

    // Find the circumradius, which the renderer scales to 1.
    let mut r = 0.0f64;
    for v in &shell.vertices {
        let l = v.to_f64()?.length();
        if l > r {
            r = l;
        }
    }

    Ok(BuiltPuzzle {
        puzzle,
        shell,
        scale: 1.0 / r,
        field,
    })
}
