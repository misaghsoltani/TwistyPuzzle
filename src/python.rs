//! Python bindings.
//!
//! The module is built with PyO3 against the stable ABI (`abi3-py310`), so one
//! wheel per platform serves every Python from 3.10 onward.
//!
//! Long-running work (building a puzzle, finding stops, rendering a frame)
//! releases the GIL, so a caller can drive several puzzles from threads.

use pyo3::exceptions::{
    PyArithmeticError, PyIndexError, PyRuntimeError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::error::Error;
use crate::layout::Cell;
use crate::math::Vec3;
use crate::render::camera::Camera;
use crate::render::framebuffer::{BlendMode, Rect};
use crate::render::scene::SceneOptions;
use crate::render::trackball::Pointer;

pyo3::create_exception!(
    twistypuzzle._native,
    PuzzleError,
    pyo3::exceptions::PyException,
    "Base class for errors raised by the twistypuzzle extension."
);

fn to_py(e: Error) -> PyErr {
    // Moving the message out instead of cloning it: an error is built once
    // and converted once, so there is nothing left to borrow it for.
    match e {
        Error::Range(m) | Error::Division(m) if m.contains("Division by zero") => PyZeroDivisionError::new_err(m),
        Error::Range(m) => PyValueError::new_err(m),
        Error::Division(m) => PyArithmeticError::new_err(m),
        Error::Type(m) => PyTypeError::new_err(m),
        Error::Parse(m) | Error::State(m) => PuzzleError::new_err(m),
        Error::Other(m) => PyRuntimeError::new_err(m),
    }
}

type R<T> = Result<T, PyErr>;

/// Turn a recipe query string or a catalog name into a recipe.
///
/// Ten catalog entries share the placeholder name "Unknown", so a name is
/// not always an identifier. Recipes are, so an ambiguous name says which ones
/// it could have meant instead of picking one.
fn resolve_recipe(spec: &str) -> R<String> {
    if spec.starts_with('?') {
        return Ok(spec.to_string());
    }
    let mut matches = crate::catalog::find_all(spec);
    let entry = matches
        .next()
        .ok_or_else(|| PyValueError::new_err(format!("no cataloged puzzle named {spec:?}")))?;
    if let Some(second) = matches.next() {
        let mut recipes = vec![entry.recipe.to_string(), second.recipe.to_string()];
        recipes.extend(matches.map(|e| e.recipe.to_string()));
        return Err(PyValueError::new_err(format!(
            "{} cataloged puzzles are named {spec:?}, build one by recipe instead: {}",
            recipes.len(),
            recipes.join(", ")
        )));
    }
    Ok(entry.recipe.to_string())
}

/* -------------------------------------------------------------------------- */
/*  Exact arithmetic                                                          */
/* -------------------------------------------------------------------------- */

/// An exact rational number.
#[pyclass(name = "Fraction", module = "twistypuzzle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyFraction {
    inner: crate::num::fraction::Fraction,
}

#[pymethods]
impl PyFraction {
    #[new]
    #[pyo3(signature = (numerator, denominator = 1))]
    fn new(numerator: i64, denominator: i64) -> R<PyFraction> {
        let inner = crate::num::fraction::Fraction::new(
            crate::num::int::Int::from_i64(numerator),
            crate::num::int::Int::from_i64(denominator),
            true,
        )
        .map_err(to_py)?;
        Ok(PyFraction { inner })
    }

    /// Numerator, in lowest terms.
    #[getter]
    fn numerator(&self) -> String {
        self.inner.n.to_string()
    }

    /// Denominator, in lowest terms.
    #[getter]
    fn denominator(&self) -> String {
        self.inner.d.to_string()
    }

    fn __add__(&self, other: &PyFraction) -> R<PyFraction> {
        use crate::num::ring::Elem;
        Ok(PyFraction {
            inner: Elem::add(&self.inner, &other.inner).map_err(to_py)?,
        })
    }
    fn __sub__(&self, other: &PyFraction) -> R<PyFraction> {
        use crate::num::ring::Elem;
        Ok(PyFraction {
            inner: Elem::sub(&self.inner, &other.inner).map_err(to_py)?,
        })
    }
    fn __mul__(&self, other: &PyFraction) -> R<PyFraction> {
        use crate::num::ring::Elem;
        Ok(PyFraction {
            inner: Elem::mul(&self.inner, &other.inner).map_err(to_py)?,
        })
    }
    fn __truediv__(&self, other: &PyFraction) -> R<PyFraction> {
        use crate::num::ring::Elem;
        Ok(PyFraction {
            inner: Elem::div(&self.inner, &other.inner).map_err(to_py)?,
        })
    }
    fn __neg__(&self) -> PyFraction {
        use crate::num::ring::Elem;
        PyFraction {
            inner: Elem::neg(&self.inner),
        }
    }
    fn __abs__(&self) -> PyFraction {
        PyFraction {
            inner: self.inner.abs(),
        }
    }
    /// The value as a float, correctly rounded to nearest.
    fn __float__(&self) -> f64 {
        self.inner.to_f64_nearest()
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> String {
        format!("Fraction({}, {})", self.inner.n, self.inner.d)
    }
    fn __richcmp__(&self, other: &PyFraction, op: pyo3::pyclass::CompareOp) -> bool {
        let c = self.inner.compare(&other.inner);
        op.matches(c.cmp(&0))
    }
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.n.hash(&mut h);
        self.inner.d.hash(&mut h);
        h.finish()
    }
}

/// A real algebraic number, held exactly as an element of a number field.
///
/// Arithmetic is exact: comparisons and signs are always decided correctly, no
/// matter how close two values are.
#[pyclass(name = "Real", module = "twistypuzzle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyReal {
    inner: crate::exact::AlgebraicNumber,
}

#[pymethods]
impl PyReal {
    /// Construct from an integer.
    #[staticmethod]
    fn from_int(n: i64) -> PyReal {
        use crate::num::ring::RingOps;
        PyReal {
            inner: crate::exact::qq_nothing().from_int(n),
        }
    }

    /// Parse and evaluate an expression such as `"(1+sqrt(5))/2"`.
    #[staticmethod]
    fn parse(expr: &str) -> R<PyReal> {
        let e = crate::parse::parse_expr(expr).map_err(to_py)?;
        Ok(PyReal {
            inner: crate::parse::eval_expr(&e).map_err(to_py)?,
        })
    }

    /// The `k`-th root, in an extended field if necessary.
    fn root(&self, k: i64) -> R<PyReal> {
        let k = crate::num::int::Int::from_i64(k);
        Ok(PyReal {
            inner: crate::exact::root(&self.inner, &k).map_err(to_py)?,
        })
    }

    /// Sign: `-1`, `0` or `+1`, decided exactly.
    fn sign(&self) -> R<i32> {
        self.inner.sign().map_err(to_py)
    }

    /// A rational interval that certainly contains this number.
    fn interval(&self) -> R<(PyFraction, PyFraction)> {
        let (lo, hi) = self.inner.interval().map_err(to_py)?;
        Ok((PyFraction { inner: lo }, PyFraction { inner: hi }))
    }

    /// The minimal polynomial of the field's generator, as a string.
    #[getter]
    fn field(&self) -> String {
        self.inner.field.to_string()
    }

    fn __add__(&self, other: &PyReal) -> R<PyReal> {
        self.binop(other, crate::num::ring::Elem::add)
    }
    fn __sub__(&self, other: &PyReal) -> R<PyReal> {
        self.binop(other, crate::num::ring::Elem::sub)
    }
    fn __mul__(&self, other: &PyReal) -> R<PyReal> {
        self.binop(other, crate::num::ring::Elem::mul)
    }
    fn __truediv__(&self, other: &PyReal) -> R<PyReal> {
        self.binop(other, crate::num::ring::Elem::div)
    }
    fn __neg__(&self) -> PyReal {
        PyReal {
            inner: crate::num::ring::Elem::neg(&self.inner),
        }
    }
    fn __abs__(&self) -> R<PyReal> {
        Ok(PyReal {
            inner: self.inner.abs().map_err(to_py)?,
        })
    }
    /// The value as a float, correct to the last bit a double can hold.
    fn __float__(&self) -> R<f64> {
        self.inner.to_number().map_err(to_py)
    }
    fn __str__(&self) -> String {
        self.inner.to_string()
    }
    fn __repr__(&self) -> R<String> {
        Ok(format!("Real({})", self.inner.to_number().map_err(to_py)?))
    }
    fn __richcmp__(&self, other: &PyReal, op: pyo3::pyclass::CompareOp) -> R<bool> {
        let mut xs = vec![self.inner.clone(), other.inner.clone()];
        crate::exact::promote(&mut xs).map_err(to_py)?;
        let c = xs[0].compare(&xs[1]).map_err(to_py)?;
        Ok(op.matches(c.cmp(&0)))
    }
}

impl PyReal {
    fn binop(
        &self,
        other: &PyReal,
        f: impl Fn(
            &crate::exact::AlgebraicNumber,
            &crate::exact::AlgebraicNumber,
        ) -> crate::Result<crate::exact::AlgebraicNumber>,
    ) -> R<PyReal> {
        // Both operands must live in a common field before they can combine.
        let mut xs = vec![self.inner.clone(), other.inner.clone()];
        crate::exact::promote(&mut xs).map_err(to_py)?;
        Ok(PyReal {
            inner: f(&xs[0], &xs[1]).map_err(to_py)?,
        })
    }
}

/* -------------------------------------------------------------------------- */
/*  Recipes                                                                   */
/* -------------------------------------------------------------------------- */

/// The text of an exact expression, from whatever Python handed over.
///
/// A ratio is written as text (such as `"1/3"`) instead of as a float, because a
/// float is not the number it is spelled: `1/3` has no `double`, and a puzzle
/// cut at 0.3333333333333333 is a different puzzle from one cut at a third.
fn expr_text(v: &Bound<'_, PyAny>) -> R<String> {
    if let Ok(f) = v.extract::<PyFraction>() {
        return Ok(format!("{}/{}", f.inner.n, f.inner.d));
    }
    if let Ok(s) = v.extract::<String>() {
        return Ok(s);
    }
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i.to_string());
    }
    Err(PyTypeError::new_err(format!(
        "expected an expression as text, an integer or a Fraction, not {}",
        v.get_type().name()?
    )))
}

/// Parse an expression, reporting where it came from.
fn expr_of(v: &Bound<'_, PyAny>) -> R<crate::parse::Expr> {
    let text = expr_text(v)?;
    crate::parse::parse_expr(&text).map_err(to_py)
}

/// One shell or one cut: a named polyhedron, or a bare plane.
///
/// Every size and coefficient is an exact expression, so `"sqrt(5)/5"` means
/// what it says and not the nearest double to it.
///
///     >>> import twistypuzzle as tp
///     >>> tp.Shape.polyhedron("C", "1/3")
///     Shape.polyhedron("C", "1/3")
///     >>> str(tp.Shape.plane(1, 0, 0, "1/2"))
///     '1,0,0$1/2'
#[pyclass(name = "Shape", module = "twistypuzzle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyShape {
    inner: crate::parse::Shape,
}

#[pymethods]
impl PyShape {
    /// A polyhedron by code (`"C"` for a cube, `"D"` for a dodecahedron)
    /// scaled so its faces sit `size` from the center.
    ///
    /// `polyhedra()` lists every code.
    #[staticmethod]
    #[pyo3(signature = (name, size = None))]
    fn polyhedron(name: &str, size: Option<&Bound<'_, PyAny>>) -> R<PyShape> {
        let d = match size {
            Some(v) => expr_of(v)?,
            None => crate::parse::parse_expr("1").map_err(to_py)?,
        };
        if name.contains(['$', ',', '&', '=']) {
            return Err(PyValueError::new_err(format!("{name:?} is not a polyhedron code")));
        }
        Ok(PyShape {
            inner: crate::parse::Shape::Polyhedron {
                name: name.to_string(),
                d,
            },
        })
    }

    /// The plane `a*x + b*y + c*z = d`.
    #[staticmethod]
    fn plane(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>, c: &Bound<'_, PyAny>, d: &Bound<'_, PyAny>) -> R<PyShape> {
        Ok(PyShape {
            inner: crate::parse::Shape::Plane {
                a: expr_of(a)?,
                b: expr_of(b)?,
                c: expr_of(c)?,
                d: expr_of(d)?,
            },
        })
    }

    /// Read a shape back from its text, as it appears in a recipe.
    #[staticmethod]
    fn parse(text: &str) -> R<PyShape> {
        Ok(PyShape {
            inner: crate::parse::parse_shape(text).map_err(to_py)?,
        })
    }

    /// `"polyhedron"` or `"plane"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            crate::parse::Shape::Polyhedron { .. } => "polyhedron",
            crate::parse::Shape::Plane { .. } => "plane",
        }
    }

    /// The polyhedron's code, or `None` for a plane.
    #[getter]
    fn name(&self) -> Option<String> {
        match &self.inner {
            crate::parse::Shape::Polyhedron { name, .. } => Some(name.clone()),
            crate::parse::Shape::Plane { .. } => None,
        }
    }

    /// The offset: how far a polyhedron's faces, or the plane itself, sit from
    /// the center, as the text of an exact expression.
    #[getter]
    fn size(&self) -> R<String> {
        let d = match &self.inner {
            crate::parse::Shape::Polyhedron { d, .. } | crate::parse::Shape::Plane { d, .. } => d,
        };
        crate::parse::generate_expr(d, 10).map_err(to_py)
    }

    /// A plane's `(a, b, c)`, or `None` for a polyhedron.
    #[getter]
    fn coefficients(&self) -> R<Option<(String, String, String)>> {
        match &self.inner {
            crate::parse::Shape::Polyhedron { .. } => Ok(None),
            crate::parse::Shape::Plane { a, b, c, .. } => Ok(Some((
                crate::parse::generate_expr(a, 10).map_err(to_py)?,
                crate::parse::generate_expr(b, 10).map_err(to_py)?,
                crate::parse::generate_expr(c, 10).map_err(to_py)?,
            ))),
        }
    }

    /// The exact offset, evaluated.
    fn offset(&self) -> R<PyReal> {
        let d = match &self.inner {
            crate::parse::Shape::Polyhedron { d, .. } | crate::parse::Shape::Plane { d, .. } => d,
        };
        Ok(PyReal {
            inner: crate::parse::eval_expr(d).map_err(to_py)?,
        })
    }

    fn __str__(&self) -> R<String> {
        crate::parse::generate_shape(&self.inner).map_err(to_py)
    }

    fn __repr__(&self) -> R<String> {
        Ok(match &self.inner {
            crate::parse::Shape::Polyhedron { name, .. } => {
                format!("Shape.polyhedron({name:?}, {:?})", self.size()?)
            },
            crate::parse::Shape::Plane { .. } => {
                let (a, b, c) = self.coefficients()?.expect("a plane always has coefficients");
                format!("Shape.plane({a:?}, {b:?}, {c:?}, {:?})", self.size()?)
            },
        })
    }

    fn __richcmp__(&self, other: &PyShape, op: pyo3::pyclass::CompareOp) -> R<bool> {
        match op {
            pyo3::pyclass::CompareOp::Eq => Ok(self.inner == other.inner),
            pyo3::pyclass::CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyTypeError::new_err("shapes are not ordered")),
        }
    }

    fn __hash__(&self) -> R<u64> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.__str__()?.hash(&mut h);
        Ok(h.finish())
    }
}

/// A puzzle's shell and its cuts: everything needed to build one.
///
/// The same thing a recipe query string says, with each part addressable.
///
///     >>> import twistypuzzle as tp
///     >>> r = tp.Recipe([tp.Shape.polyhedron("C")], [tp.Shape.polyhedron("C", "1/3")])
///     >>> r.query
///     '?shell=C$1&cut=C$1/3'
///     >>> r == tp.Recipe.parse("?shell=C$1&cut=C$1/3")
///     True
#[pyclass(name = "Recipe", module = "twistypuzzle", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyRecipe {
    inner: crate::parse::PuzzleRecipe,
}

#[pymethods]
impl PyRecipe {
    #[new]
    #[pyo3(signature = (shell = Vec::new(), cuts = Vec::new()))]
    fn new(shell: Vec<PyShape>, cuts: Vec<PyShape>) -> PyRecipe {
        PyRecipe {
            inner: crate::parse::PuzzleRecipe {
                shell: shell.into_iter().map(|s| s.inner).collect(),
                cuts: cuts.into_iter().map(|s| s.inner).collect(),
            },
        }
    }

    /// Read a recipe query string, such as `"?shell=C$1&cut=C$1/3"`.
    #[staticmethod]
    fn parse(query: &str) -> R<PyRecipe> {
        Ok(PyRecipe {
            inner: crate::parse::parse_query(query).map_err(to_py)?,
        })
    }

    /// The recipe of a cataloged puzzle.
    ///
    /// Ten catalog entries share the placeholder name `"Unknown"`, so an
    /// ambiguous name says which recipes it could have meant instead of
    /// picking one.
    #[staticmethod]
    fn named(name: &str) -> R<PyRecipe> {
        PyRecipe::parse(&resolve_recipe(name)?)
    }

    /// The shapes the puzzle is carved from.
    #[getter]
    fn shell(&self) -> Vec<PyShape> {
        self.inner.shell.iter().map(|s| PyShape { inner: s.clone() }).collect()
    }

    /// The shapes that cut it.
    #[getter]
    fn cuts(&self) -> Vec<PyShape> {
        self.inner.cuts.iter().map(|s| PyShape { inner: s.clone() }).collect()
    }

    /// The recipe as a query string.
    #[getter]
    fn query(&self) -> R<String> {
        crate::parse::generate_query(&self.inner).map_err(to_py)
    }

    /// This recipe with `shell` in place of its own.
    fn with_shell(&self, shell: Vec<PyShape>) -> PyRecipe {
        PyRecipe::new(shell, self.cuts())
    }

    /// This recipe with `cuts` in place of its own.
    fn with_cuts(&self, cuts: Vec<PyShape>) -> PyRecipe {
        PyRecipe::new(self.shell(), cuts)
    }

    /// This recipe with more cuts added.
    fn adding_cuts(&self, cuts: Vec<PyShape>) -> PyRecipe {
        let mut all = self.cuts();
        all.extend(cuts);
        PyRecipe::new(self.shell(), all)
    }

    /// Build the puzzle this recipe describes.
    fn build(&self, py: Python<'_>) -> R<PyPuzzle> {
        PyPuzzle::from_recipe(py, self)
    }

    fn __str__(&self) -> R<String> {
        self.query()
    }

    fn __repr__(&self) -> R<String> {
        Ok(format!("Recipe.parse({:?})", self.query()?))
    }

    fn __richcmp__(&self, other: &PyRecipe, op: pyo3::pyclass::CompareOp) -> R<bool> {
        match op {
            pyo3::pyclass::CompareOp::Eq => Ok(self.inner == other.inner),
            pyo3::pyclass::CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(PyTypeError::new_err("recipes are not ordered")),
        }
    }

    fn __hash__(&self) -> R<u64> {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.query()?.hash(&mut h);
        Ok(h.finish())
    }
}

/// Whatever names a puzzle (a query string, a catalog name, or a `Recipe`),
/// resolved to a query string.
fn recipe_query(v: &Bound<'_, PyAny>) -> R<String> {
    if let Ok(r) = v.extract::<PyRecipe>() {
        return r.query();
    }
    if let Ok(s) = v.extract::<String>() {
        return resolve_recipe(&s);
    }
    Err(PyTypeError::new_err(format!(
        "expected a recipe query string, a catalog name or a Recipe, not {}",
        v.get_type().name()?
    )))
}

/// One turnable layer of a puzzle, as it stands right now.
///
/// A snapshot, not a handle: the grips are re-derived after every turn, so a
/// grip taken before a move may not be there after one. Turn it through the
/// puzzle it came from, by index.
#[pyclass(name = "Grip", module = "twistypuzzle", frozen, get_all)]
pub struct PyGrip {
    /// Where this grip sits in `Puzzle.grips()`.
    pub index: usize,
    /// The axis it turns about, as a unit vector.
    pub axis: (f64, f64, f64),
    /// How far the cutting plane is from the center.
    pub offset: f64,
    /// The exact plane, as text.
    pub plane: String,
    /// How many pieces turn with it.
    pub piece_count: usize,
    /// How many stay behind.
    pub other_count: usize,
}

#[pymethods]
impl PyGrip {
    fn __repr__(&self) -> String {
        format!(
            "<Grip {} axis=({:.3}, {:.3}, {:.3}) offset={:.3} pieces={}>",
            self.index, self.axis.0, self.axis.1, self.axis.2, self.offset, self.piece_count
        )
    }
}

/* -------------------------------------------------------------------------- */
/*  Images                                                                    */
/* -------------------------------------------------------------------------- */

/// An RGBA8 image.
///
/// The pixel data is tightly packed, so `numpy.asarray(image)` gives a
/// zero-copy `(height, width, 4)` view of `uint8`.
#[pyclass(name = "Image", module = "twistypuzzle")]
pub struct PyImage {
    inner: crate::render::Framebuffer,
}

fn blend_mode(name: &str) -> R<BlendMode> {
    match name {
        "copy" => Ok(BlendMode::Copy),
        "over" => Ok(BlendMode::Over),
        "add" | "additive" => Ok(BlendMode::Additive),
        other => Err(PyValueError::new_err(format!(
            "unknown blend mode {other:?}, expected 'copy', 'over' or 'add'"
        ))),
    }
}

#[pymethods]
impl PyImage {
    #[new]
    #[pyo3(signature = (width, height, color = (0, 0, 0, 0)))]
    fn new(width: u32, height: u32, color: (u8, u8, u8, u8)) -> PyImage {
        PyImage {
            inner: crate::render::Framebuffer::filled(width, height, [color.0, color.1, color.2, color.3]),
        }
    }

    /// Wrap existing tightly packed RGBA8 bytes.
    #[staticmethod]
    fn from_bytes(width: u32, height: u32, data: Vec<u8>) -> R<PyImage> {
        crate::render::Framebuffer::from_raw(width, height, data)
            .map(|inner| PyImage { inner })
            .ok_or_else(|| PyValueError::new_err("data length must be width * height * 4"))
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }
    /// Bytes per row.
    #[getter]
    fn stride(&self) -> usize {
        self.inner.stride()
    }

    /// A copy of the pixel data as `bytes`.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.as_bytes())
    }

    /// Encode as PNG.
    fn to_png<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let png = py.detach(|| self.inner.to_png());
        PyBytes::new(py, &png)
    }

    /// Write a PNG to `path`.
    fn save(&self, py: Python<'_>, path: &str) -> R<()> {
        let png = py.detach(|| self.inner.to_png());
        std::fs::write(path, png).map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// NumPy interface: a zero-copy `(height, width, 4)` `uint8` view.
    ///
    /// NumPy keeps this object alive through the array's `base`, so the view
    /// stays valid.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item(
            "shape",
            PyTuple::new(py, [self.inner.height() as usize, self.inner.width() as usize, 4])?,
        )?;
        d.set_item("typestr", "|u1")?;
        d.set_item("version", 3)?;
        d.set_item(
            "data",
            PyTuple::new(py, [self.inner.as_bytes().as_ptr() as usize, 0usize])?,
        )?;
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.inner.as_bytes().len()
    }

    /// The RGBA value at `(x, y)`.
    fn pixel(&self, x: u32, y: u32) -> R<(u8, u8, u8, u8)> {
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(PyIndexError::new_err("pixel out of range"));
        }
        let p = self.inner.pixel(x, y);
        Ok((p[0], p[1], p[2], p[3]))
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: (u8, u8, u8, u8)) -> R<()> {
        if x >= self.inner.width() || y >= self.inner.height() {
            return Err(PyIndexError::new_err("pixel out of range"));
        }
        self.inner.set_pixel(x, y, [color.0, color.1, color.2, color.3]);
        Ok(())
    }

    /// Fill the whole image with one color.
    fn fill(&mut self, color: (u8, u8, u8, u8)) {
        self.inner.fill([color.0, color.1, color.2, color.3]);
    }

    /// Fill a rectangle.
    #[pyo3(signature = (x, y, width, height, color, mode = "copy"))]
    fn fill_rect(&mut self, x: i64, y: i64, width: u32, height: u32, color: (u8, u8, u8, u8), mode: &str) -> R<()> {
        self.inner.fill_rect(
            Rect {
                x,
                y,
                w: width,
                h: height,
            },
            [color.0, color.1, color.2, color.3],
            blend_mode(mode)?,
        );
        Ok(())
    }

    /// Copy a rectangle of `src` onto this image at `(x, y)`.
    ///
    /// Both rectangles are clipped, so out-of-range coordinates are harmless.
    /// `mode` is `"copy"`, `"over"` (source-over alpha) or `"add"`.
    #[pyo3(signature = (src, x = 0, y = 0, src_x = 0, src_y = 0, src_width = None, src_height = None, mode = "copy"))]
    #[allow(clippy::too_many_arguments)]
    fn blit(
        &mut self,
        py: Python<'_>,
        src: &PyImage,
        x: i64,
        y: i64,
        src_x: i64,
        src_y: i64,
        src_width: Option<u32>,
        src_height: Option<u32>,
        mode: &str,
    ) -> R<()> {
        let rect = Rect {
            x: src_x,
            y: src_y,
            w: src_width.unwrap_or(src.inner.width()),
            h: src_height.unwrap_or(src.inner.height()),
        };
        let mode = blend_mode(mode)?;
        py.detach(|| self.inner.blit(&src.inner, rect, x, y, mode));
        Ok(())
    }

    /// Box-filter downsample by an integer factor.
    /// Draw `text` with the top-left of its cell at `(x, y)`.
    ///
    /// The font is built in, so this works the same on every platform with no
    /// font file and no system font service. `scale` magnifies each pixel of
    /// a glyph. For smooth edges, draw into an image `n` times the size and
    /// `downsample(n)` it. Returns the width drawn, so captions can be
    /// chained. `\n` starts a new line.
    #[pyo3(signature = (x, y, text, color = (0, 0, 0, 255), scale = 1, mode = "over", center = false))]
    #[allow(clippy::too_many_arguments)]
    fn draw_text(
        &mut self,
        x: i64,
        y: i64,
        text: &str,
        color: (u8, u8, u8, u8),
        scale: u32,
        mode: &str,
        center: bool,
    ) -> R<u32> {
        let rgba = [color.0, color.1, color.2, color.3];
        let mode = blend_mode(mode)?;
        Ok(if center {
            self.inner.draw_text_centered(x, y, text, rgba, scale, mode)
        } else {
            self.inner.draw_text(x, y, text, rgba, scale, mode)
        })
    }

    /// The `(width, height)` in pixels that `draw_text` would cover.
    #[staticmethod]
    #[pyo3(signature = (text, scale = 1))]
    fn text_size(text: &str, scale: u32) -> (u32, u32) {
        (
            crate::render::font::text_block_width(text, scale),
            crate::render::font::text_block_height(text, scale),
        )
    }

    /// Shorten `text` with an ellipsis until it fits `max_width` pixels.
    #[staticmethod]
    #[pyo3(signature = (text, max_width, scale = 1))]
    fn fit_text(text: &str, max_width: u32, scale: u32) -> String {
        crate::render::font::ellipsize(text, max_width, scale)
    }

    fn downsample(&self, py: Python<'_>, factor: u32) -> R<PyImage> {
        if factor == 0 {
            return Err(PyValueError::new_err("factor must be positive"));
        }
        Ok(PyImage {
            inner: py.detach(|| self.inner.downsample(factor)),
        })
    }

    /// Nearest-neighbor resize.
    fn resized(&self, py: Python<'_>, width: u32, height: u32) -> PyImage {
        PyImage {
            inner: py.detach(|| self.inner.scaled(width, height)),
        }
    }

    fn copy(&self) -> PyImage {
        PyImage {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("<Image {}x{} RGBA8>", self.inner.width(), self.inner.height())
    }
}

/* -------------------------------------------------------------------------- */
/*  Puzzle                                                                    */
/* -------------------------------------------------------------------------- */

/// A twisty puzzle: its pieces, its turnable cuts, and its state.
///
/// Build one from a recipe query string, a catalog name, a [`Recipe`], or the
/// shell and cuts spelled out:
///
///     >>> import twistypuzzle as tp
///     >>> tp.Puzzle("?shell=C$1&cut=C$1/3").piece_count
///     26
///     >>> tp.Puzzle("Rubik's Cube (3x3x3)").piece_count
///     26
///     >>> tp.Puzzle(shell=tp.Shape.polyhedron("C"),
///     ...           cuts=tp.Shape.polyhedron("C", "1/3")).piece_count
///     26
///
/// Every view setting is a keyword on the constructor as well as a property, so
/// a puzzle can be configured in one call or adjusted one knob at a time.
#[pyclass(name = "Puzzle", module = "twistypuzzle")]
pub struct PyPuzzle {
    sim: crate::simulator::Simulator,
}

/// The view settings that can be given at construction, set as properties, or
/// passed to a single `render`.
struct ViewOverrides {
    background: Option<(u8, u8, u8, u8)>,
    supersample: Option<u32>,
    show_arrows: Option<bool>,
    show_edges: Option<bool>,
    show_pieces: Option<bool>,
    line_width: Option<f64>,
    fov: Option<f64>,
    distance: Option<f64>,
    yaw: Option<f64>,
    pitch: Option<f64>,
}

impl ViewOverrides {
    /// Apply these settings, returning prior settings to allow subsequent restoration.
    fn apply(&self, sim: &mut crate::simulator::Simulator) -> (SceneOptions, Camera) {
        let saved = (*sim.options(), *sim.camera());
        if let Some(c) = self.background {
            sim.options_mut().background = [c.0, c.1, c.2, c.3];
        }
        if let Some(n) = self.supersample {
            sim.options_mut().supersample = n.max(1);
        }
        if let Some(b) = self.show_arrows {
            sim.options_mut().draw_arrows = b;
        }
        if let Some(b) = self.show_edges {
            sim.options_mut().draw_edges = b;
        }
        if let Some(b) = self.show_pieces {
            sim.options_mut().draw_pieces = b;
        }
        if let Some(w) = self.line_width {
            sim.options_mut().line_width = w;
        }
        if let Some(f) = self.fov {
            sim.camera_mut().fov = f;
        }
        // Yaw and pitch move the camera, so a distance given alongside them is
        // part of the same placement instead of a separate one.
        if self.yaw.is_some() || self.pitch.is_some() {
            let d = self.distance.unwrap_or_else(|| sim.camera().position.length());
            sim.look_from(self.yaw.unwrap_or(0.0), self.pitch.unwrap_or(0.0), d);
        } else if let Some(d) = self.distance {
            sim.set_distance(d);
        }
        saved
    }

    fn is_empty(&self) -> bool {
        self.background.is_none()
            && self.supersample.is_none()
            && self.show_arrows.is_none()
            && self.show_edges.is_none()
            && self.show_pieces.is_none()
            && self.line_width.is_none()
            && self.fov.is_none()
            && self.distance.is_none()
            && self.yaw.is_none()
            && self.pitch.is_none()
    }
}

#[pymethods]
impl PyPuzzle {
    #[new]
    #[pyo3(signature = (
        recipe = None,
        *,
        shell = None,
        cuts = None,
        background = None,
        supersample = None,
        show_arrows = None,
        show_edges = None,
        show_pieces = None,
        line_width = None,
        fov = None,
        distance = None,
        yaw = None,
        pitch = None,
        seed = None,
        scramble = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        recipe: Option<&Bound<'_, PyAny>>,
        shell: Option<&Bound<'_, PyAny>>,
        cuts: Option<&Bound<'_, PyAny>>,
        background: Option<(u8, u8, u8, u8)>,
        supersample: Option<u32>,
        show_arrows: Option<bool>,
        show_edges: Option<bool>,
        show_pieces: Option<bool>,
        line_width: Option<f64>,
        fov: Option<f64>,
        distance: Option<f64>,
        yaw: Option<f64>,
        pitch: Option<f64>,
        seed: Option<u64>,
        scramble: Option<u32>,
    ) -> R<PyPuzzle> {
        let query = match (recipe, shell, cuts) {
            (Some(r), None, None) => recipe_query(r)?,
            (None, shell, cuts) => {
                let shell = shapes_arg(shell, "shell")?;
                let cuts = shapes_arg(cuts, "cuts")?;
                if shell.is_empty() {
                    return Err(PyTypeError::new_err("give a recipe, or a shell and its cuts"));
                }
                PyRecipe::new(shell, cuts).query()?
            },
            (Some(_), _, _) => {
                return Err(PyTypeError::new_err(
                    "give either a recipe or a shell and cuts, not both",
                ));
            },
        };
        let sim = py
            .detach(move || crate::simulator::Simulator::from_query(&query))
            .map_err(to_py)?;
        let mut p = PyPuzzle { sim };
        let view = ViewOverrides {
            background,
            supersample,
            show_arrows,
            show_edges,
            show_pieces,
            line_width,
            fov,
            distance,
            yaw,
            pitch,
        };
        view.apply(&mut p.sim);
        if let Some(s) = seed {
            p.sim.seed(s);
        }
        if let Some(n) = scramble {
            p.sim.scramble(n);
            py.detach(|| p.sim.settle()).map_err(to_py)?;
        }
        Ok(p)
    }

    /// Build one of the cataloged puzzles by name.
    #[staticmethod]
    fn named(py: Python<'_>, name: &str) -> R<PyPuzzle> {
        let query = resolve_recipe(name)?;
        let sim = py
            .detach(move || crate::simulator::Simulator::from_query(&query))
            .map_err(to_py)?;
        Ok(PyPuzzle { sim })
    }

    /// Build the puzzle a [`Recipe`] describes.
    #[staticmethod]
    fn from_recipe(py: Python<'_>, recipe: &PyRecipe) -> R<PyPuzzle> {
        let query = recipe.query()?;
        let sim = py
            .detach(move || crate::simulator::Simulator::from_query(&query))
            .map_err(to_py)?;
        Ok(PyPuzzle { sim })
    }

    /* --- what it is ----------------------------------------------------- */

    #[getter]
    fn piece_count(&self) -> usize {
        self.sim.piece_count()
    }

    /// The number field all coordinates live in.
    #[getter]
    fn field(&self) -> String {
        self.sim.field().to_string()
    }

    /// The factor that scales the puzzle to fit a unit sphere.
    #[getter]
    fn scale(&self) -> f64 {
        self.sim.scale()
    }

    /// The shell and cuts this puzzle was built from.
    #[getter]
    fn recipe(&self) -> PyRecipe {
        PyRecipe {
            inner: self.sim.recipe().clone(),
        }
    }

    /// The recipe as a query string.
    #[getter]
    fn query(&self) -> R<String> {
        self.sim.query().map_err(to_py)
    }

    /// The geometry of piece `i`: vertices, faces, colors and flags.
    ///
    /// Takes the puzzle exclusively: converting a coordinate to a float can
    /// narrow the isolating interval that every number in this puzzle shares.
    /// See [`render`](Self::render).
    fn piece<'py>(&mut self, py: Python<'py>, index: usize) -> R<Bound<'py, PyDict>> {
        let p = self
            .sim
            .puzzle()
            .pieces
            .get(index)
            .ok_or_else(|| PyIndexError::new_err("piece index out of range"))?;
        let d = PyDict::new(py);
        let mut verts: Vec<(f64, f64, f64)> = Vec::with_capacity(p.vertices.len());
        for v in &p.vertices {
            let v = v.to_f64().map_err(to_py)?;
            verts.push((v.x, v.y, v.z));
        }
        d.set_item("vertices", verts)?;
        let faces = PyList::empty(py);
        for f in &p.faces {
            let fd = PyDict::new(py);
            fd.set_item("vertices", f.vertices.clone())?;
            fd.set_item("color", f.color.get_hex())?;
            fd.set_item("interior", f.interior)?;
            faces.append(fd)?;
        }
        d.set_item("faces", faces)?;
        d.set_item("rotation", p.rot.to_string())?;
        Ok(d)
    }

    /* --- grips and turning ---------------------------------------------- */

    /// Number of turnable grips. Grip `i` turns with `turn(i, +1)` or
    /// `turn(i, -1)`.
    #[getter]
    fn grip_count(&self) -> usize {
        self.sim.grip_count()
    }

    /// Every turnable layer, as it stands now.
    ///
    /// Takes the puzzle exclusively: an axis is reported as floats, and turning
    /// an exact coordinate into a float can refine the field they share.
    fn grips(&mut self) -> R<Vec<PyGrip>> {
        let mut out = Vec::with_capacity(self.sim.grip_count());
        for i in 0..self.sim.grip_count() {
            let cut = self.sim.grips()[i].clone();
            let p = cut.plane.to_f64().map_err(to_py)?;
            out.push(PyGrip {
                index: i,
                axis: (p.normal.x, p.normal.y, p.normal.z),
                offset: -p.constant,
                plane: cut.plane.to_string(),
                piece_count: cut.front.len(),
                other_count: cut.back.len(),
            });
        }
        Ok(out)
    }

    /// The exact plane of grip `i`, as a string.
    fn grip(&self, index: usize) -> R<String> {
        self.sim
            .grips()
            .get(index)
            .map(|c| c.plane.to_string())
            .ok_or_else(|| PyIndexError::new_err("grip index out of range"))
    }

    /// The stop angles available on grip `i`, in degrees.
    fn stops(&mut self, py: Python<'_>, index: usize) -> R<Vec<f64>> {
        py.detach(|| {
            let rots = self.sim.stops(index)?;
            rots.iter()
                .map(|r| Ok(r.approx_angle()? / std::f64::consts::PI * 180.0))
                .collect::<crate::Result<Vec<f64>>>()
        })
        .map_err(to_py)
    }

    /// Turn grip `index` in direction `direction` (`+1` or `-1`), `repeat`
    /// times.
    ///
    /// The move is applied immediately, and the animation, if any, is what
    /// `frame` interpolates.
    #[pyo3(signature = (index, direction = 1, *, repeat = 1))]
    fn turn(&mut self, py: Python<'_>, index: usize, direction: i32, repeat: u32) -> R<()> {
        py.detach(|| {
            for k in 0..repeat {
                self.sim.begin_move(index, direction)?;
                if k + 1 < repeat {
                    self.sim.end_move();
                }
            }
            Ok(())
        })
        .map_err(to_py)
    }

    /// Turn grip `index` to stop `stop` of `stops(index)`.
    ///
    /// The finest control there is: `turn` takes the nearest stop in a
    /// direction, and this takes whichever one is named.
    fn turn_to(&mut self, py: Python<'_>, index: usize, stop: usize) -> R<()> {
        py.detach(|| self.sim.begin_move_to(index, stop)).map_err(to_py)
    }

    /// Invert the most recent turn, returning `False` if the history is empty
    /// or the layer is locked.
    fn undo(&mut self, py: Python<'_>) -> R<bool> {
        py.detach(|| self.sim.undo()).map_err(to_py)
    }

    /// The number of applied turns in the current history.
    #[getter]
    fn history_length(&self) -> usize {
        self.sim.history_len()
    }

    /// Finish the move in flight and apply every queued scramble move,
    /// leaving the puzzle at rest.
    fn settle(&mut self, py: Python<'_>) -> R<()> {
        py.detach(|| self.sim.settle()).map_err(to_py)
    }

    /// Queue `count` random moves, applied by `advance` or `settle`.
    #[pyo3(signature = (count = 10))]
    fn scramble(&mut self, count: u32) {
        self.sim.scramble(count);
    }

    /// Apply one random move now.
    fn move_random(&mut self, py: Python<'_>) -> R<()> {
        py.detach(|| self.sim.move_random()).map_err(to_py)
    }

    /// Seed the scramble generator, for reproducible sequences.
    fn seed(&mut self, seed: u64) {
        self.sim.seed(seed);
    }

    /// How many moves have been applied.
    #[getter]
    fn move_count(&self) -> u64 {
        self.sim.moves_made()
    }

    /// Invert every applied turn, restoring the puzzle to the solved state.
    ///
    /// Computationally faster than [`reset`](Self::reset): applied turns are
    /// inverted via exact rotations without re-deriving the 3D geometry.
    /// Returns `False` if a locked layer halts inversion partway.
    fn restore(&mut self, py: Python<'_>) -> R<bool> {
        py.detach(|| self.sim.restore()).map_err(to_py)
    }

    /// Reconstruct the puzzle from its recipe, preserving camera and view settings.
    ///
    /// Always succeeds by performing full re-derivation. [`restore`](Self::restore)
    /// provides faster restoration when the puzzle has only undergone valid turns.
    fn reset(&mut self, py: Python<'_>) -> R<()> {
        let query = self.sim.query().map_err(to_py)?;
        let options = *self.sim.options();
        let camera = *self.sim.camera();
        let mut fresh = py
            .detach(move || crate::simulator::Simulator::from_query(&query))
            .map_err(to_py)?;
        *fresh.options_mut() = options;
        *fresh.camera_mut() = camera;
        self.sim = fresh;
        Ok(())
    }

    /* --- symbolic and numerical state ----------------------------------- */

    /// The color in each sticker slot: discrete numerical state representation.
    ///
    /// Slots are indexed over the canonical solved state.
    ///
    /// Raises `PuzzleError` for a puzzle that jumbles, which has no fixed set
    /// of discrete slots (`jumbles()` identifies such puzzles).
    fn stickers(&mut self, py: Python<'_>) -> R<Vec<u16>> {
        py.detach(|| self.sim.stickers()).map_err(to_py)
    }

    /// Where the sticker in each slot belongs: the state as a permutation.
    ///
    /// Solved, this is `range(sticker_count)`.
    fn sticker_ids(&mut self, py: Python<'_>) -> R<Vec<u32>> {
        py.detach(|| self.sim.sticker_ids()).map_err(to_py)
    }

    /// Where each sticker slot sits on the solved puzzle, as `(x, y, z)`.
    ///
    /// Slots do not move, so this describes the puzzle and not its state: it
    /// says which way slot `i` faces and where on that face it sits.
    /// Coordinates are before the scale that fits the puzzle in a unit sphere.
    fn sticker_positions(&mut self, py: Python<'_>) -> R<Vec<(f64, f64, f64)>> {
        let at = py.detach(|| self.sim.sticker_positions()).map_err(to_py)?;
        Ok(at.into_iter().map(|p| (p[0], p[1], p[2])).collect())
    }

    /// The color in each slot when the puzzle is solved: the goal state.
    #[getter]
    fn solved_stickers(&mut self, py: Python<'_>) -> R<Vec<u16>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.stickers.solved_colors().to_vec())
    }

    /// How many sticker slots the puzzle has.
    #[getter]
    fn sticker_count(&mut self, py: Python<'_>) -> R<usize> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.stickers.sticker_count())
    }

    /// How many distinct sticker colors it has.
    #[getter]
    fn color_count(&mut self, py: Python<'_>) -> R<usize> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.stickers.color_count())
    }

    /// The sticker colors themselves, packed `0xRRGGBB`, indexed by color.
    #[getter]
    fn palette(&mut self, py: Python<'_>) -> R<Vec<u32>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.stickers.palette().to_vec())
    }

    /// Whether every sticker is the color it should be.
    #[getter]
    fn is_solved(&mut self, py: Python<'_>) -> R<bool> {
        py.detach(|| self.sim.is_solved()).map_err(to_py)
    }

    /// The state as ground atoms, `("color", "s<slot>", "c<color>")`.
    ///
    /// A predicate and its arguments, all text, so a state can be
    /// compared against a partial goal without going
    /// through the array.
    fn ground_atoms(&mut self, py: Python<'_>) -> R<Vec<(String, String, String)>> {
        let colors = py.detach(|| self.sim.stickers()).map_err(to_py)?;
        Ok(crate::symbolic::ground_atoms(&colors))
    }

    /// Every move of this puzzle as a permutation of its sticker slots, or
    /// `None` if it has none.
    ///
    /// `new[i] = old[perm[a][i]]`. Each move is applied once to the solved
    /// puzzle, the resulting permutation recorded, and the move inverted.
    ///
    /// Geometric rotation requires approximately two milliseconds per move to
    /// evaluate cut geometry, whereas array permutation requires nanoseconds.
    /// Precomputing the table requires two turns per action once.
    ///
    /// `None` means the moves are not fixed permutations (such as a puzzle that
    /// jumbles, or one whose layers lock), and that the geometry has to be
    /// turned move by move instead. Deriving it leaves the puzzle as it found
    /// it.
    fn action_permutations(&mut self, py: Python<'_>) -> R<Option<PyArray>> {
        let table = py
            .detach(|| crate::symbolic::PermutationTable::build(&mut self.sim))
            .map_err(to_py)?;
        Ok(table.map(|t| {
            let (a, k) = (t.action_count(), t.sticker_count());
            let flat: Vec<u32> = t.permutations().concat();
            PyArray::ints(&flat, k.max(1), vec![a, k])
        }))
    }

    /// Every move this puzzle has, in action-index order.
    ///
    /// Action `2k` turns grip `k` the way its arrow points and `2k + 1` turns
    /// it back, where the names run `"A"`, `"A'"`, `"B"`, ... with a number after the
    /// letter when an axis has more than one layer.
    #[getter]
    fn actions(&mut self, py: Python<'_>) -> R<Vec<String>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.actions.action_names())
    }

    /// How many moves it has: two per grip of the solved puzzle.
    #[getter]
    fn action_count(&mut self, py: Python<'_>) -> R<usize> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.actions.action_count())
    }

    /// The names of the grips themselves, without a direction.
    #[getter]
    fn grip_names(&mut self, py: Python<'_>) -> R<Vec<String>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.actions.grip_names().to_vec())
    }

    /// The index of a move named like `"A"` or `"B2'"`, or `None`.
    fn action_index(&mut self, py: Python<'_>, name: &str) -> R<Option<usize>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.actions.action_index(name))
    }

    /// Which grip a move turns right now, or `None` if that layer is locked.
    fn action_grip(&mut self, py: Python<'_>, action: usize) -> R<Option<usize>> {
        let view = py.detach(|| self.sim.symbolic()).map_err(to_py)?;
        Ok(view.actions.locate(action, self.sim.grips()))
    }

    /// Make the move with index `action`, without animating.
    ///
    /// Returns `False`, having changed nothing, if that layer cannot turn in
    /// the puzzle's current state.
    fn apply_action(&mut self, py: Python<'_>, action: usize) -> R<bool> {
        py.detach(|| self.sim.apply_action(action)).map_err(to_py)
    }

    /// Which moves can be made right now, one flag per action.
    fn action_mask(&mut self, py: Python<'_>) -> R<Vec<bool>> {
        py.detach(|| self.sim.action_mask()).map_err(to_py)
    }

    /// Make a sequence of moves written out, such as `"A B2' A'"`.
    ///
    /// Returns how many turns were made, counting repeats.
    fn apply(&mut self, py: Python<'_>, moves: &str) -> R<usize> {
        py.detach(|| {
            let view = self.sim.symbolic()?;
            let parsed = crate::symbolic::parse_moves(&view.actions, moves)?;
            let mut made = 0usize;
            for m in parsed {
                for _ in 0..m.repeat {
                    if !self.sim.apply_action(m.action)? {
                        let name = view
                            .actions
                            .action_name(m.action)
                            .unwrap_or_else(|| m.action.to_string());
                        return Err(Error::State(format!(
                            "move '{name}' cannot be made: that layer is locked"
                        )));
                    }
                    made += 1;
                }
            }
            Ok(made)
        })
        .map_err(to_py)
    }

    /* --- view ------------------------------------------------------------ */

    /// Advance animation and camera by `dt_ms` milliseconds.
    #[pyo3(signature = (dt_ms = 16.0))]
    fn advance(&mut self, py: Python<'_>, dt_ms: f64) -> R<()> {
        py.detach(|| self.sim.advance(dt_ms)).map_err(to_py)
    }

    /// Render the current state.
    ///
    /// Any view setting can be overridden for this one frame. What is not
    /// given is left as the puzzle has it, and nothing given here sticks.
    ///
    /// This takes the puzzle exclusively even though it changes nothing you
    /// can observe through the API, because it is not actually a read: turning
    /// an exact coordinate into a float narrows the isolating interval that
    /// every number in this puzzle shares, and rendering fills the puzzle's
    /// memo caches. The pixels do not depend on the interleaving
    /// (`SEMANTICS.md` §1), but the mutation is real, so two threads rendering
    /// the *same* puzzle would be a data race. Python's borrow check turns
    /// that into a `RuntimeError` instead of letting it happen quietly.
    /// Render one puzzle per thread, or use `render_many`, which does exactly
    /// that.
    #[pyo3(signature = (
        width = 512,
        height = 512,
        *,
        background = None,
        supersample = None,
        show_arrows = None,
        show_edges = None,
        show_pieces = None,
        line_width = None,
        fov = None,
        distance = None,
        yaw = None,
        pitch = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn render(
        &mut self,
        py: Python<'_>,
        width: u32,
        height: u32,
        background: Option<(u8, u8, u8, u8)>,
        supersample: Option<u32>,
        show_arrows: Option<bool>,
        show_edges: Option<bool>,
        show_pieces: Option<bool>,
        line_width: Option<f64>,
        fov: Option<f64>,
        distance: Option<f64>,
        yaw: Option<f64>,
        pitch: Option<f64>,
    ) -> PyImage {
        let view = ViewOverrides {
            background,
            supersample,
            show_arrows,
            show_edges,
            show_pieces,
            line_width,
            fov,
            distance,
            yaw,
            pitch,
        };
        PyImage {
            inner: py.detach(|| {
                if view.is_empty() {
                    return self.sim.render(width, height);
                }
                let (options, camera) = view.apply(&mut self.sim);
                let fb = self.sim.render(width, height);
                *self.sim.options_mut() = options;
                *self.sim.camera_mut() = camera;
                fb
            }),
        }
    }

    /// Advance by `dt_ms` and render: the usual per-frame call.
    #[pyo3(signature = (dt_ms = 16.0, width = 512, height = 512))]
    fn frame(&mut self, py: Python<'_>, dt_ms: f64, width: u32, height: u32) -> R<PyImage> {
        let fb = py.detach(|| self.sim.frame(dt_ms, width, height)).map_err(to_py)?;
        Ok(PyImage { inner: fb })
    }

    /// True while a move is animating.
    #[getter]
    fn animating(&self) -> bool {
        self.sim.is_animating()
    }

    /// Background color, as an RGBA tuple.
    #[getter]
    fn background(&self) -> (u8, u8, u8, u8) {
        let c = self.sim.options().background;
        (c[0], c[1], c[2], c[3])
    }

    #[setter]
    fn set_background(&mut self, color: (u8, u8, u8, u8)) {
        self.sim.options_mut().background = [color.0, color.1, color.2, color.3];
    }

    /// Supersampling factor used when rendering (1 disables antialiasing).
    #[getter]
    fn supersample(&self) -> u32 {
        self.sim.options().supersample
    }

    #[setter]
    fn set_supersample(&mut self, factor: u32) {
        self.sim.options_mut().supersample = factor.max(1);
    }

    /// Whether the arrows that drive the turns are drawn.
    #[getter]
    fn show_arrows(&self) -> bool {
        self.sim.options().draw_arrows
    }

    #[setter]
    fn set_show_arrows(&mut self, on: bool) {
        self.sim.options_mut().draw_arrows = on;
    }

    /// Whether piece outlines are drawn.
    #[getter]
    fn show_edges(&self) -> bool {
        self.sim.options().draw_edges
    }

    #[setter]
    fn set_show_edges(&mut self, on: bool) {
        self.sim.options_mut().draw_edges = on;
    }

    /// Whether the pieces themselves are drawn. Turning this off leaves the arrows alone.
    #[getter]
    fn show_pieces(&self) -> bool {
        self.sim.options().draw_pieces
    }

    #[setter]
    fn set_show_pieces(&mut self, on: bool) {
        self.sim.options_mut().draw_pieces = on;
    }

    /// Width of the outlines, in device pixels.
    #[getter]
    fn line_width(&self) -> f64 {
        self.sim.options().line_width
    }

    #[setter]
    fn set_line_width(&mut self, w: f64) {
        self.sim.options_mut().line_width = w;
    }

    /// Field of view, in degrees.
    #[getter]
    fn fov(&self) -> f64 {
        self.sim.camera().fov
    }

    #[setter]
    fn set_fov(&mut self, fov: f64) {
        self.sim.camera_mut().fov = fov;
    }

    /// Distance from the camera to the puzzle.
    #[getter]
    fn distance(&self) -> f64 {
        self.sim.camera().position.length()
    }

    #[setter]
    fn set_distance(&mut self, d: f64) {
        self.sim.set_distance(d);
    }

    /// Where the camera is.
    #[getter]
    fn camera_position(&self) -> (f64, f64, f64) {
        let p = self.sim.camera().position;
        (p.x, p.y, p.z)
    }

    #[setter]
    fn set_camera_position(&mut self, p: (f64, f64, f64)) {
        self.sim.camera_mut().position = Vec3::new(p.0, p.1, p.2);
    }

    /// Which way is up for the camera.
    #[getter]
    fn camera_up(&self) -> (f64, f64, f64) {
        let p = self.sim.camera().up;
        (p.x, p.y, p.z)
    }

    #[setter]
    fn set_camera_up(&mut self, p: (f64, f64, f64)) {
        self.sim.camera_mut().up = Vec3::new(p.0, p.1, p.2);
    }

    /// What the camera looks at.
    #[getter]
    fn camera_target(&self) -> (f64, f64, f64) {
        let p = self.sim.camera().target;
        (p.x, p.y, p.z)
    }

    #[setter]
    fn set_camera_target(&mut self, p: (f64, f64, f64)) {
        self.sim.camera_mut().target = Vec3::new(p.0, p.1, p.2);
    }

    /// Look at the puzzle from `yaw` and `pitch` degrees off head-on.
    ///
    /// `(0, 0)` is the default, head-on camera. A little of each shows the
    /// puzzle as a solid instead of a silhouette.
    #[pyo3(signature = (yaw, pitch, distance = None))]
    fn look_from(&mut self, yaw: f64, pitch: f64, distance: Option<f64>) {
        let d = distance.unwrap_or_else(|| self.sim.camera().position.length());
        self.sim.look_from(yaw, pitch, d);
    }

    /// Orbit the camera by a normalized drag delta.
    fn orbit(&mut self, dx: f64, dy: f64) {
        self.sim.controls_mut().pointer_down(Pointer { x: 0.0, y: 0.0 });
        self.sim.controls_mut().pointer_move(Pointer { x: dx, y: dy });
        let mut cam = *self.sim.camera();
        self.sim.controls_mut().update(&mut cam);
        *self.sim.camera_mut() = cam;
        self.sim.controls_mut().pointer_up();
    }

    /* --- pointer -------------------------------------------------------- */

    /// Move the pointer. `x` and `y` are normalized to `[-1, 1]` with `y` up.
    fn pointer_move(&mut self, x: f64, y: f64) {
        self.sim.pointer_move(Pointer { x, y });
    }

    /// Press the pointer, returning `True` if it landed on an arrow.
    fn pointer_down(&mut self, x: f64, y: f64) -> bool {
        self.sim.pointer_down(Pointer { x, y })
    }

    /// Release the pointer, activating an arrow if one was pressed.
    ///
    /// Returns the `(grip, direction)` that was turned, or `None`.
    fn pointer_up(&mut self, py: Python<'_>, x: f64, y: f64) -> R<Option<(usize, i32)>> {
        py.detach(|| self.sim.pointer_up(Pointer { x, y })).map_err(to_py)
    }

    /// Index of the arrow under the pointer, or `None`.
    fn arrow_at(&self, x: f64, y: f64) -> Option<usize> {
        self.sim.pick_arrow(Pointer { x, y })
    }

    /// Index of the arrow the pointer is over, or `None`.
    #[getter]
    fn hovered_arrow(&self) -> Option<usize> {
        self.sim.hovered_arrow()
    }

    /// Clear any active arrow hover state.
    fn clear_hover(&mut self) {
        self.sim.clear_hover();
    }

    /// How many arrows there are: two per grip.
    #[getter]
    fn arrow_count(&self) -> usize {
        self.sim.arrow_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "<Puzzle {} pieces, {} grips>",
            self.sim.piece_count(),
            self.sim.grip_count()
        )
    }
}

/// One shape or several, however the caller spelled it.
fn shapes_arg(v: Option<&Bound<'_, PyAny>>, what: &str) -> R<Vec<PyShape>> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    if let Ok(one) = v.extract::<PyShape>() {
        return Ok(vec![one]);
    }
    if let Ok(text) = v.extract::<String>() {
        return Ok(vec![PyShape::parse(&text)?]);
    }
    let mut out = Vec::new();
    let items = v
        .try_iter()
        .map_err(|_| PyTypeError::new_err(format!("{what} must be a Shape, its text, or a sequence of either")))?;
    for item in items {
        let item = item?;
        if let Ok(one) = item.extract::<PyShape>() {
            out.push(one);
        } else if let Ok(text) = item.extract::<String>() {
            out.push(PyShape::parse(&text)?);
        } else {
            return Err(PyTypeError::new_err(format!(
                "{what} must hold Shapes or their text, not {}",
                item.get_type().name()?
            )));
        }
    }
    Ok(out)
}

/* -------------------------------------------------------------------------- */
/*  Module functions                                                          */
/* -------------------------------------------------------------------------- */

/// Every cataloged puzzle, as `(name, family, kind, recipe)` tuples.
#[pyfunction]
fn catalog() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    crate::catalog::CATALOG.to_vec()
}

/// Every supported polyhedron, as `(code, name)` pairs.
#[pyfunction]
fn polyhedra() -> Vec<(&'static str, &'static str)> {
    crate::polyhedra::shapes().collect()
}

/// Parse an arithmetic expression and evaluate it exactly.
#[pyfunction]
fn evaluate(expr: &str) -> R<PyReal> {
    PyReal::parse(expr)
}

/// Factor a rational polynomial given by its coefficients, lowest degree
/// first. Returns each irreducible factor's coefficients.
#[pyfunction]
fn factor_polynomial(py: Python<'_>, coefficients: Vec<i64>) -> R<Vec<Vec<String>>> {
    py.detach(move || {
        let p = crate::num::polynomial::polynomial_i64(&coefficients);
        let fs = crate::num::factoring::factor(&p)?;
        Ok(fs
            .iter()
            .map(|f| f.coeffs.iter().map(ToString::to_string).collect())
            .collect())
    })
    .map_err(to_py)
}

/* -------------------------------------------------------------------------- */
/*  Batch operations                                                          */
/* -------------------------------------------------------------------------- */

/// Build several puzzles at once, one per worker thread.
///
/// Each entry is a recipe query string or a catalog name. The interpreter is
/// released for the whole batch, so on a free-threaded build other Python
/// threads keep running and on a GIL build they do too.
#[pyfunction]
fn build_many(py: Python<'_>, recipes: Vec<Bound<'_, PyAny>>) -> R<Vec<PyPuzzle>> {
    let (resolved, labels) = recipe_queries(recipes)?;
    let built = py.detach(|| crate::batch::build_many(&resolved));
    let mut out = Vec::with_capacity(built.len());
    for (i, r) in built.into_iter().enumerate() {
        out.push(PyPuzzle {
            sim: r.map_err(|e| annotate(e, &labels[i]))?,
        });
    }
    Ok(out)
}

/// Build, optionally scramble, and render several puzzles at once.
///
/// Equivalent to calling `render_puzzle` on each entry, but the whole batch
/// runs across every core with the interpreter released. Puzzle `i` is seeded
/// with `seed + i`, so a scrambled batch is reproducible however the work is
/// scheduled.
#[pyfunction]
#[pyo3(signature = (
    recipes,
    width = 512,
    height = 512,
    *,
    background = (255, 255, 255, 255),
    supersample = 2,
    show_arrows = false,
    show_edges = true,
    scramble = 0,
    seed = None,
    distance = None,
    yaw = 0.0,
    pitch = 0.0,
))]
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn render_many(
    py: Python<'_>,
    recipes: Vec<Bound<'_, PyAny>>,
    width: u32,
    height: u32,
    background: (u8, u8, u8, u8),
    supersample: u32,
    show_arrows: bool,
    show_edges: bool,
    scramble: u32,
    seed: Option<u64>,
    distance: Option<f64>,
    yaw: f64,
    pitch: f64,
) -> R<Vec<PyImage>> {
    let (resolved, labels) = recipe_queries(recipes)?;
    let spec = crate::batch::RenderSpec {
        width,
        height,
        background: [background.0, background.1, background.2, background.3],
        supersample,
        show_arrows,
        show_edges,
        scramble,
        seed,
        distance,
        yaw,
        pitch,
    };
    let frames = py.detach(|| crate::batch::render_many(&resolved, &spec));
    let mut out = Vec::with_capacity(frames.len());
    for (i, r) in frames.into_iter().enumerate() {
        out.push(PyImage {
            inner: r.map_err(|e| annotate(e, &labels[i]))?,
        });
    }
    Ok(out)
}

/// Resolve a list of recipes, keeping how each was written so that a failure
/// can name the entry the caller gave instead of the query it became.
fn recipe_queries(recipes: Vec<Bound<'_, PyAny>>) -> R<(Vec<String>, Vec<String>)> {
    let mut resolved = Vec::with_capacity(recipes.len());
    let mut labels = Vec::with_capacity(recipes.len());
    for r in recipes {
        resolved.push(recipe_query(&r)?);
        labels.push(r.str().map_or_else(|_| "?".into(), |s| s.to_string()));
    }
    Ok((resolved, labels))
}

/// Say which puzzle failed: a batch error with no name is hard to act on.
fn annotate(e: Error, spec: &str) -> PyErr {
    let err = to_py(e);
    Python::attach(|py| {
        let msg = err.value(py).to_string();
        PyErr::from_type(err.get_type(py).clone(), format!("{spec}: {msg}"))
    })
}

/// Worker threads the batch functions and the rasterizer will use.
///
/// This is rayon's global pool. Set `RAYON_NUM_THREADS` before the first call
/// into the extension to change it.
#[pyfunction]
fn thread_count() -> usize {
    crate::batch::thread_count()
}

/// A contiguous memory buffer formatted for zero-copy NumPy inspection.
///
/// All batch query methods return an instance of `Array`: states, masks,
/// indicator vectors, and rendered frames. To avoid third-party runtime dependencies,
/// the core package exports this wrapper instead of returning NumPy ndarrays or
/// allocating nested Python lists. Memory is owned by a single contiguous allocation
/// that `numpy.asarray` wraps directly in place.
///
///     >>> import numpy as np                        # doctest: +SKIP
///     >>> a = np.asarray(batch.observations())      # doctest: +SKIP
///     >>> a.shape                                   # doctest: +SKIP
///     (64, 54)
///
/// Without NumPy, `tolist()` and indexing read the same numbers.
///
/// Each array owns its own buffer and nothing else writes to it, so a view
/// remains valid and immutable for as long as the array is alive, regardless of
/// subsequent simulator steps.
#[pyclass(name = "Array", module = "twistypuzzle")]
pub struct PyArray {
    data: Vec<u8>,
    shape: Vec<usize>,
    typestr: &'static str,
    itemsize: usize,
}

/// `<` or `>` for this machine, as the array interface spells it.
const ORDER: &str = if cfg!(target_endian = "little") { "<" } else { ">" };

/// Bytes per element for whole numbers below `bound`.
///
/// One byte up to 256 distinct values, two up to 65536, four beyond. Nothing
/// here is ever signed and nothing is ever wider than four bytes: a slot
/// number is checked against `u32` when the puzzle is built.
///
/// `bound` is the puzzle's, never the data's (`SEMANTICS.md` §13), so that
/// the element type is a property of the puzzle and a caller can write its
/// `dtype` down before seeing a state.
const fn width_for(bound: usize) -> usize {
    if bound <= 1 << 8 {
        1
    } else if bound <= 1 << 16 {
        2
    } else {
        4
    }
}

/// Run `$body` once, with `$C` naming the narrowest element type that holds
/// every value below `$bound`.
///
/// A closure cannot be generic, so this is a macro: the body is written once
/// and compiled at each of the three widths, and the one that runs is chosen
/// when the call is made. It is what lets the same array code serve a 3x3x3,
/// whose faces and slots both fit a byte, and a puzzle whose slots need four.
macro_rules! with_cell {
    ($bound:expr, |$c:ident| $body:block) => {
        match width_for($bound) {
            1 => {
                type $c = u8;
                $body
            },
            2 => {
                type $c = u16;
                $body
            },
            _ => {
                type $c = u32;
                $body
            },
        }
    };
}

impl PyArray {
    /// One byte per element: sticker colors, indicator vectors, pixels.
    fn bytes(data: Vec<u8>, shape: Vec<usize>) -> PyArray {
        PyArray {
            data,
            shape,
            typestr: "|u1",
            itemsize: 1,
        }
    }

    /// One flag per element, which NumPy reads as `bool`.
    fn flags(data: Vec<bool>, shape: Vec<usize>) -> PyArray {
        PyArray {
            data: data.into_iter().map(u8::from).collect(),
            shape,
            typestr: "|b1",
            itemsize: 1,
        }
    }

    /// Whole numbers, in the narrowest width that holds all of them.
    ///
    /// `bound` is one past the largest value the caller can produce (the
    /// number of slots for a slot number, of moves for a move, of colors for
    /// a color), not the largest value that happens to be in `data`.
    /// A puzzle's arrays all have the same element type whatever state it is
    /// in, which is what lets a caller write `dtype` down once.
    fn ints<C: Cell>(data: &[C], bound: usize, shape: Vec<usize>) -> PyArray {
        match width_for(bound) {
            1 => PyArray {
                data: data.iter().map(|v| v.index() as u8).collect(),
                shape,
                typestr: "|u1",
                itemsize: 1,
            },
            2 => {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for v in data {
                    bytes.extend_from_slice(&(v.index() as u16).to_ne_bytes());
                }
                PyArray {
                    data: bytes,
                    shape,
                    typestr: if ORDER == "<" { "<u2" } else { ">u2" },
                    itemsize: 2,
                }
            },
            _ => {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for v in data {
                    bytes.extend_from_slice(&(v.index() as u32).to_ne_bytes());
                }
                PyArray {
                    data: bytes,
                    shape,
                    typestr: if ORDER == "<" { "<u4" } else { ">u4" },
                    itemsize: 4,
                }
            },
        }
    }

    /// Real numbers, for the encodings that are not integers.
    fn floats(data: &[f32], shape: Vec<usize>) -> PyArray {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for &v in data {
            bytes.extend_from_slice(&v.to_ne_bytes());
        }
        PyArray {
            data: bytes,
            shape,
            typestr: if ORDER == "<" { "<f4" } else { ">f4" },
            itemsize: 4,
        }
    }

    fn count(&self) -> usize {
        self.data.len() / self.itemsize
    }

    /// The `i`-th element, as the Python number its type calls for.
    ///
    /// Every type this array can have is named here instead of leaving it to a
    /// catch-all: an element read at the wrong width is not an error anywhere,
    /// it is a wrong number, so a type nobody handled must not fall through to
    /// a type somebody did.
    fn element<'py>(&self, py: Python<'py>, i: usize) -> R<Bound<'py, PyAny>> {
        let at = i * self.itemsize;
        let b = &self.data[at..at + self.itemsize];
        match self.typestr {
            "|b1" => (b[0] != 0).into_bound_py_any(py),
            "|u1" => b[0].into_bound_py_any(py),
            "<u2" | ">u2" => u16::from_ne_bytes([b[0], b[1]]).into_bound_py_any(py),
            "<u4" | ">u4" => u32::from_ne_bytes([b[0], b[1], b[2], b[3]]).into_bound_py_any(py),
            _ => f32::from_ne_bytes([b[0], b[1], b[2], b[3]]).into_bound_py_any(py),
        }
    }

    /// `count` elements from `start`, as a list.
    fn run<'py>(&self, py: Python<'py>, start: usize, count: usize) -> R<Bound<'py, PyList>> {
        let at = start * self.itemsize;
        let bytes = &self.data[at..at + count * self.itemsize];
        let words = || {
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        };
        Ok(match self.typestr {
            "|b1" => PyList::new(py, bytes.iter().map(|&b| b != 0))?,
            "|u1" => PyList::new(py, bytes.iter().copied())?,
            "<u2" | ">u2" => PyList::new(py, bytes.chunks_exact(2).map(|c| u16::from_ne_bytes([c[0], c[1]])))?,
            "<u4" | ">u4" => PyList::new(py, words())?,
            _ => PyList::new(py, words().map(f32::from_bits))?,
        })
    }

    /// How many elements one step along the first axis covers.
    fn row_len(&self) -> usize {
        self.shape.iter().skip(1).product()
    }
}

#[pymethods]
impl PyArray {
    /// The shape, outermost axis first.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }

    /// How each element is encoded, in NumPy's own notation: `"|b1"` for a
    /// flag, `"<f4"` for a real number, and `"|u1"`, `"<u2"` or `"<u4"` for a
    /// whole number, whichever of the three is narrow enough to hold every
    /// value this puzzle can produce.
    #[getter]
    fn typestr(&self) -> &'static str {
        self.typestr
    }

    /// Bytes per element.
    #[getter]
    fn itemsize(&self) -> usize {
        self.itemsize
    }

    /// How many elements there are in total.
    #[getter]
    fn size(&self) -> usize {
        self.count()
    }

    /// NumPy interface: a zero-copy view of this array's own buffer.
    ///
    /// NumPy keeps this object alive through the array's `base`, so the view
    /// stays valid.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("shape", PyTuple::new(py, &self.shape)?)?;
        d.set_item("typestr", self.typestr)?;
        d.set_item("version", 3)?;
        d.set_item("data", PyTuple::new(py, [self.data.as_ptr() as usize, 0usize])?)?;
        Ok(d)
    }

    /// The length of the first axis, as `len()` means for a NumPy array.
    fn __len__(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    /// Row `i` for an array of more than one axis, or element `i` for a flat
    /// one. Negative indices count from the end.
    fn __getitem__<'py>(&self, py: Python<'py>, index: isize) -> R<Bound<'py, PyAny>> {
        let n = self.__len__() as isize;
        let i = if index < 0 { index + n } else { index };
        if i < 0 || i >= n {
            return Err(PyIndexError::new_err("index out of range"));
        }
        let i = i as usize;
        if self.shape.len() <= 1 {
            return self.element(py, i);
        }
        let width = self.row_len();
        Ok(self.run(py, i * width, width)?.into_any())
    }

    /// Every element, in order, as one flat list.
    fn tolist<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        self.run(py, 0, self.count())
    }

    /// A copy of the raw bytes.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.data)
    }

    fn __repr__(&self) -> String {
        let dims: Vec<String> = self.shape.iter().map(ToString::to_string).collect();
        format!(
            "<Array shape=({}{}) dtype={}>",
            dims.join(", "),
            if self.shape.len() == 1 { "," } else { "" },
            self.typestr
        )
    }
}

/// Parallel simulator for multiple instances of a twisty puzzle.
///
/// Designed for vectorized reinforcement learning and parallel search, avoiding
/// the overhead of individual Python interpreter transitions per environment.
/// Batched operations execute concurrently across threads with the Python GIL
/// released.
///
/// Performance benefits from shared precomputation: 3D geometry is derived once
/// for all instances of a recipe. For non-jumbling puzzles with fixed move
/// permutations, transitions are computed via array gather operations instead of
/// re-evaluating 3D geometry, achieving microsecond-scale batch transitions.
/// Frame rendering similarly uses precomputed pixel-to-slot mapping tables.
///
/// Jumbling puzzles or puzzles with dynamic layer locking simulate individual
/// 3D geometry per instance. Both execution models expose an identical interface.
/// The `is_tabular` property indicates whether permutation gathering is active.
///
///     >>> import twistypuzzle as tp
///     >>> b = tp.PuzzleBatch("Rubik's Cube (3x3x3)", 4, seed=0)
///     >>> b.reset(scramble=5)
///     >>> b.observations().shape
///     (4, 54)
///     >>> solved, applied = b.step([0, 1, 2, 3])
///     >>> len(solved)
///     4
#[pyclass(name = "PuzzleBatch", module = "twistypuzzle")]
pub struct PyPuzzleBatch {
    inner: crate::batch::PuzzleBatch,
}

impl PyPuzzleBatch {
    /// Refuse a puzzle number this batch does not have, the way Python
    /// refuses an index a sequence does not have.
    fn check(&self, index: usize) -> R<()> {
        let n = self.inner.len();
        if index >= n {
            return Err(PyIndexError::new_err(format!("puzzle {index} out of range (have {n})")));
        }
        Ok(())
    }

    /// Read a per-row selection: a list of flags, a list of row numbers, or
    /// `None` for every row.
    fn rows_of(&self, which: Option<&Bound<'_, PyAny>>) -> R<Vec<bool>> {
        let n = self.inner.len();
        let Some(v) = which else {
            return Ok(vec![true; n]);
        };
        if let Ok(flags) = v.extract::<Vec<bool>>() {
            if flags.len() != n {
                return Err(PyValueError::new_err(format!(
                    "got {} flags for {n} puzzles",
                    flags.len()
                )));
            }
            return Ok(flags);
        }
        let indices: Vec<usize> = v
            .extract()
            .map_err(|_| PyTypeError::new_err("expected a list of flags, a list of indices, or None"))?;
        let mut out = vec![false; n];
        for i in indices {
            if i >= n {
                return Err(PyIndexError::new_err(format!("puzzle {i} out of range (have {n})")));
            }
            out[i] = true;
        }
        Ok(out)
    }

    /// Read a depth given either once for the whole batch or once per row.
    fn depths_of(&self, depth: &Bound<'_, PyAny>) -> R<Vec<u32>> {
        let n = self.inner.len();
        if let Ok(one) = depth.extract::<u32>() {
            return Ok(vec![one; n]);
        }
        let many: Vec<u32> = depth
            .extract()
            .map_err(|_| PyTypeError::new_err("expected a number of moves, or one per puzzle"))?;
        if many.len() != n {
            return Err(PyValueError::new_err(format!(
                "got {} depths for {n} puzzles",
                many.len()
            )));
        }
        Ok(many)
    }
}

#[pymethods]
impl PyPuzzleBatch {
    /// Build `count` copies of one puzzle.
    ///
    /// Every view setting the constructor of `Puzzle` takes is taken here
    /// too, and applies to every frame the batch draws. All but one: a batch
    /// never draws the arrows. They are interactive controls instead of part
    /// of a state, and are blended over a frame instead of written into
    /// it, and a drawing table cannot express them, so leaving them out is
    /// what lets both kinds of batch agree about what a state looks like.
    #[new]
    #[pyo3(signature = (
        recipe,
        count,
        *,
        seed = None,
        scramble = None,
        background = None,
        supersample = None,
        show_edges = None,
        show_pieces = None,
        line_width = None,
        fov = None,
        distance = None,
        yaw = None,
        pitch = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        recipe: &Bound<'_, PyAny>,
        count: usize,
        seed: Option<u64>,
        scramble: Option<u32>,
        background: Option<(u8, u8, u8, u8)>,
        supersample: Option<u32>,
        show_edges: Option<bool>,
        show_pieces: Option<bool>,
        line_width: Option<f64>,
        fov: Option<f64>,
        distance: Option<f64>,
        yaw: Option<f64>,
        pitch: Option<f64>,
    ) -> R<PyPuzzleBatch> {
        let query = recipe_query(recipe)?;
        let view = ViewOverrides {
            background,
            supersample,
            // A batch draws states, not controls: the arrows are a thing to
            // click on, and no frame of a batch has them.
            show_arrows: Some(false),
            show_edges,
            show_pieces,
            line_width,
            fov,
            distance,
            yaw,
            pitch,
        };
        py.detach(|| {
            let mut inner = crate::batch::PuzzleBatch::build(&query, count)?;
            if !view.is_empty() {
                inner.configure(&|sim| {
                    view.apply(sim);
                });
            }
            if let Some(s) = seed {
                inner.seed_all(s);
            }
            if let Some(depth) = scramble {
                inner.reset(depth)?;
            }
            Ok(PyPuzzleBatch { inner })
        })
        .map_err(to_py)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The recipe every copy was built from.
    #[getter]
    fn query(&self) -> String {
        self.inner.recipe().to_string()
    }

    /// Whether moves are being applied as permutations instead of by turning
    /// geometry, which is the difference between microseconds and
    /// milliseconds a move, and whether frames are painted or rasterized.
    #[getter]
    fn is_tabular(&self) -> bool {
        self.inner.is_tabular()
    }

    /// How many sticker slots each puzzle has.
    #[getter]
    fn sticker_count(&self) -> usize {
        self.inner.sticker_count()
    }

    /// How many moves each puzzle has.
    #[getter]
    fn action_count(&self) -> usize {
        self.inner.action_count()
    }

    /// How many distinct sticker colors each puzzle has.
    #[getter]
    fn color_count(&self) -> usize {
        self.inner.color_count()
    }

    /// Every move name, in action-index order.
    #[getter]
    fn actions(&self) -> Vec<String> {
        self.inner.view().actions.action_names()
    }

    /// The names of the grips themselves, without a direction.
    #[getter]
    fn grip_names(&self) -> Vec<String> {
        self.inner.view().actions.grip_names().to_vec()
    }

    /// The index of a move named like `"A"` or `"B2'"`, or `None`.
    fn action_index(&self, name: &str) -> Option<usize> {
        self.inner.view().actions.action_index(name)
    }

    /// The colors of a solved puzzle: the goal every copy is working toward.
    #[getter]
    fn solved_stickers(&self) -> Vec<u16> {
        self.inner.view().stickers.solved_colors().to_vec()
    }

    /// The sticker colors themselves, packed `0xRRGGBB`, indexed by color.
    #[getter]
    fn palette(&self) -> Vec<u32> {
        self.inner.view().stickers.palette().to_vec()
    }

    /* --- seeding and scrambling ----------------------------------------- */

    /// Seed one puzzle's scrambler.
    fn seed(&mut self, index: usize, seed: u64) -> R<()> {
        self.check(index)?;
        self.inner.seed(index, seed).map_err(to_py)
    }

    /// Seed every puzzle, giving puzzle `i` `seed + i`, so one number fixes
    /// the whole batch and no two copies walk together.
    fn seed_all(&mut self, seed: u64) {
        self.inner.seed_all(seed);
    }

    /// Reset all puzzle instances to solved, then apply `scramble` pseudo-random turns.
    ///
    /// Scrambling via bounded random walks guarantees that each state is solvable
    /// in at most `scramble` steps, providing structured difficulty curricula for
    /// learning algorithms.
    #[pyo3(signature = (*, scramble = 0))]
    fn reset(&mut self, py: Python<'_>, scramble: u32) -> R<()> {
        py.detach(|| self.inner.reset(scramble)).map_err(to_py)
    }

    /// Solve one puzzle and scramble just that one, leaving the rest alone.
    #[pyo3(signature = (index, *, scramble = 0))]
    fn reset_one(&mut self, py: Python<'_>, index: usize, scramble: u32) -> R<()> {
        self.check(index)?;
        py.detach(|| self.inner.reset_one(index, scramble)).map_err(to_py)
    }

    /// Solve and re-scramble the copies `which` picks out.
    ///
    /// `which` is a flag per puzzle or a list of indices. This is a whole
    /// vectorized autoreset in one call: the episodes that finished start
    /// again while the rest are untouched.
    #[pyo3(signature = (which = None, *, scramble = 0))]
    fn reset_where(&mut self, py: Python<'_>, which: Option<&Bound<'_, PyAny>>, scramble: u32) -> R<()> {
        let rows = self.rows_of(which)?;
        py.detach(|| self.inner.reset_where(&rows, scramble)).map_err(to_py)
    }

    /// Solve the copies `which` picks out, without scrambling them again.
    #[pyo3(signature = (which = None))]
    fn solve(&mut self, py: Python<'_>, which: Option<&Bound<'_, PyAny>>) -> R<()> {
        let rows = self.rows_of(which)?;
        py.detach(|| self.inner.solve_where(&rows)).map_err(to_py)
    }

    /// Apply random walk moves from current configurations.
    ///
    /// `depth` is specified as a uniform scalar or per-puzzle sequence. Random walks
    /// do not immediately invert the preceding turn.
    fn scramble(&mut self, py: Python<'_>, depth: &Bound<'_, PyAny>) -> R<()> {
        let depths = self.depths_of(depth)?;
        py.detach(|| self.inner.scramble(&depths)).map_err(to_py)
    }

    /* --- turning --------------------------------------------------------- */

    /// Apply one action per puzzle instance.
    ///
    /// Returns `(solved, applied)`: whether each puzzle is now solved, and
    /// whether its move could be made at all. An action targeting an invalid or
    /// locked layer is ignored without raising an exception, allowing policies
    /// to operate without explicit action masking. Only geometric simulation
    /// detects blocked or locked layers.
    ///
    /// `actions` is one move per puzzle, as an array or a sequence. Passing an array
    /// allows zero-copy reading of actions across large batches.
    fn step(&mut self, py: Python<'_>, actions: &Bound<'_, PyAny>) -> R<(PyArray, PyArray)> {
        let n = self.inner.len();
        let actions = values_of(actions)?;
        let out = py.detach(|| self.inner.step(&actions)).map_err(to_py)?;
        let solved: Vec<bool> = out.iter().map(|o| o.solved).collect();
        let applied: Vec<bool> = out.iter().map(|o| o.applied).collect();
        Ok((PyArray::flags(solved, vec![n]), PyArray::flags(applied, vec![n])))
    }

    /// Every move as a permutation of the sticker slots, as an
    /// `(action_count, sticker_count)` block, or `None` for a puzzle whose
    /// moves are not fixed permutations.
    ///
    /// `new[i] = old[perm[a][i]]`. Used internally for batched transitions,
    /// and accessible for lookahead search without modifying simulator state.
    #[getter]
    fn permutations(&self) -> Option<PyArray> {
        let (a, k) = (self.inner.action_count(), self.inner.sticker_count());
        self.inner
            .permutations()
            .map(|flat| PyArray::ints(&flat, k, vec![a, k]))
    }

    /// The moves one puzzle has made since it was last solved.
    ///
    /// Turning a freshly built `Puzzle` through them puts it in that copy's
    /// position, which is the way from an array back to something that can be
    /// taken apart and looked at.
    fn history(&self, index: usize) -> R<Vec<u32>> {
        self.check(index)?;
        self.inner.history(index).map(<[u32]>::to_vec).map_err(to_py)
    }

    /// How many moves each puzzle has made since it was last solved.
    #[getter]
    fn move_counts(&self) -> Vec<usize> {
        self.inner.move_counts()
    }

    /// Invert the most recent turn of the puzzle instances specified by `which`.
    ///
    /// `which` accepts a boolean mask, index list, or `None` (all instances).
    /// Returns a boolean mask indicating which instances had an active history
    /// to invert.
    #[pyo3(signature = (which = None))]
    fn undo(&mut self, py: Python<'_>, which: Option<&Bound<'_, PyAny>>) -> R<PyArray> {
        let rows = self.rows_of(which)?;
        let n = self.inner.len();
        let done = py.detach(|| self.inner.undo(&rows)).map_err(to_py)?;
        Ok(PyArray::flags(done, vec![n]))
    }

    /// Make a written sequence of moves, such as `"A B2' A'"`, in every
    /// puzzle.
    ///
    /// Returns how many turns each puzzle was asked for, counting repeats. A
    /// turn a puzzle cannot make leaves that one alone, as a step does.
    fn apply(&mut self, py: Python<'_>, moves: &str) -> R<usize> {
        py.detach(|| self.inner.apply(moves)).map_err(to_py)
    }

    /* --- reading --------------------------------------------------------- */

    /// Put states that did not come from this batch into it.
    ///
    /// `states` is one state or a block of them, holding either the colors
    /// `observations` gives or the identities `sticker_ids` gives, selected
    /// with `ids=`. One state is put into every puzzle, a block is one
    /// state each.
    ///
    /// This is the way in for a state that came from somewhere else: read
    /// from a file, produced by a `Layout` with no puzzle in sight, or handed
    /// over by a solver. Every row must be a state of this puzzle (the same
    /// stickers, or the same count of each color), and that is checked.
    /// Whether the puzzle can be *turned* into it is not, because deciding
    /// that is the puzzle's whole difficulty. Unreachable configurations are
    /// preserved, rendered, and inspected faithfully without achieving the solved state.
    ///
    /// Each puzzle's history is emptied: the moves it had made no longer lead
    /// to where it now is.
    #[pyo3(signature = (states, *, ids = false))]
    fn set_states(&mut self, py: Python<'_>, states: &Bound<'_, PyAny>, ids: bool) -> R<()> {
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        let bound = if ids { k.max(1) } else { self.inner.color_count().max(1) };
        with_cell!(bound, |C| {
            let mut src: Vec<C> = cells_of(states)?;
            // One state for all of them is written out here, so that the
            // check and the write are the same code either way.
            if k > 0 && src.len() == k && n > 1 {
                src = src.repeat(n);
            }
            py.detach(|| {
                if ids {
                    self.inner.set_states(&src)
                } else {
                    self.inner.set_colors(&src)
                }
            })
            .map_err(to_py)
        })
    }

    /// Put states written in this puzzle's face-by-face numbering into it.
    ///
    /// `facelets` is one state or a block of them, as `facelets(layout)`
    /// gives them. The way in for a state that speaks the usual layout and
    /// knows nothing of how this puzzle numbers its slots.
    fn set_facelets(&mut self, py: Python<'_>, layout: &PyLayout, facelets: &Bound<'_, PyAny>) -> R<()> {
        self.same_puzzle(layout)?;
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        let bound = self.inner.color_count().max(layout.inner.face_count()).max(1);
        with_cell!(bound, |C| {
            let mut src: Vec<C> = cells_of(facelets)?;
            if k > 0 && src.len() == k && n > 1 {
                src = src.repeat(n);
            }
            py.detach(|| {
                let colors = layout.inner.from_facelets(&src)?;
                self.inner.set_colors(&colors)
            })
            .map_err(to_py)
        })
    }

    /// Observation matrix of shape `(n, sticker_count)` containing integer color IDs
    /// for all puzzle instances.
    fn observations(&mut self, py: Python<'_>) -> R<PyArray> {
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        let bound = self.inner.color_count().max(1);
        with_cell!(bound, |C| {
            let data: Vec<C> = py.detach(|| self.inner.observations()).map_err(to_py)?;
            Ok(PyArray::ints(&data, bound, vec![n, k]))
        })
    }

    /// Where the sticker in each slot started, as an `(n, sticker_count)`
    /// block of slot numbers: the states as permutations.
    ///
    /// Distinguishes states that identical colors cannot differentiate, such as
    /// permutations between identically colored pieces. Solved rows equal `range(sticker_count)`.
    fn sticker_ids(&mut self, py: Python<'_>) -> R<PyArray> {
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        with_cell!(k.max(1), |I| {
            let data: Vec<I> = py.detach(|| self.inner.sticker_ids()).map_err(to_py)?;
            Ok(PyArray::ints(&data, k.max(1), vec![n, k]))
        })
    }

    /// Every state as indicator bytes, `(n, sticker_count * color_count)`.
    ///
    /// Formatted for neural network inputs, and materialized directly within native
    /// memory in a single pass.
    fn one_hot(&mut self, py: Python<'_>) -> R<PyArray> {
        let n = self.inner.len();
        let width = self.inner.sticker_count() * self.inner.color_count();
        let data = py.detach(|| self.inner.one_hot()).map_err(to_py)?;
        Ok(PyArray::bytes(data, vec![n, width]))
    }

    /// Which moves each puzzle can make, as an `(n, action_count)` block of
    /// flags.
    fn action_masks(&mut self, py: Python<'_>) -> R<PyArray> {
        let (n, a) = (self.inner.len(), self.inner.action_count());
        let data = py.detach(|| self.inner.action_masks()).map_err(to_py)?;
        Ok(PyArray::flags(data, vec![n, a]))
    }

    /// Whether each puzzle is solved.
    fn solved(&mut self, py: Python<'_>) -> R<PyArray> {
        let n = self.inner.len();
        let data = py.detach(|| self.inner.solved()).map_err(to_py)?;
        Ok(PyArray::flags(data, vec![n]))
    }

    /* --- drawing --------------------------------------------------------- */

    /// Draw every puzzle, as an `(n, height, width, 4)` block of RGBA bytes.
    ///
    /// A tabular batch works out which slot each pixel shows once and then
    /// paints each frame by looking it up, which is what makes a frame per
    /// puzzle per step affordable: tens of microseconds for a whole batch of
    /// small frames instead of a rasterization each. The table is kept until
    /// the size, the camera or the view settings change.
    ///
    /// A painted frame is the frame the rasterizer draws, exactly so for a
    /// solved puzzle. On a scrambled one a piece that has turned carries its
    /// own outline around with it, so a pixel sitting exactly on a seam can
    /// fall either way: under half a percent of them, never a whole sticker,
    /// and the same every time for a given state.
    /// `channels = 3` drops the alpha channel, `channels_first` outputs
    /// `(n, c, h, w)` for convolutional architectures, and `dtype = "f4"`
    /// normalizes channel values to `[0.0, 1.0]`.
    #[pyo3(signature = (width = 256, height = 256, *, channels = 4, channels_first = false, dtype = "u1"))]
    fn render(
        &mut self,
        py: Python<'_>,
        width: u32,
        height: u32,
        channels: usize,
        channels_first: bool,
        dtype: &str,
    ) -> R<PyArray> {
        let n = self.inner.len();
        let spec = crate::batch::FrameSpec {
            width,
            height,
            channels,
            channels_first,
        };
        let shape = if channels_first {
            vec![n, channels, height as usize, width as usize]
        } else {
            vec![n, height as usize, width as usize, channels]
        };
        match dtype {
            "u1" | "uint8" => {
                let data = py.detach(|| self.inner.render_with(spec)).map_err(to_py)?;
                Ok(PyArray::bytes(data, shape))
            },
            "f4" | "float32" => {
                let data = py.detach(|| self.inner.render_float(spec)).map_err(to_py)?;
                Ok(PyArray::floats(&data, shape))
            },
            other => Err(PyValueError::new_err(format!(
                "dtype must be \"u1\" or \"f4\", not {other:?}"
            ))),
        }
    }

    /// Refuse a layout that describes a different puzzle.
    ///
    /// Two puzzles of the same size have arrays of the same length, so
    /// nothing about the shapes would catch the mix-up: the answer would
    /// simply mean something else, and for `set_facelets` it would be
    /// written into the batch.
    fn same_puzzle(&self, layout: &PyLayout) -> R<()> {
        let (mine, theirs) = (self.inner.recipe(), layout.inner.recipe());
        if mine != theirs {
            return Err(PyValueError::new_err(format!(
                "this layout describes {theirs}, and this batch holds {mine}"
            )));
        }
        Ok(())
    }

    /// Every puzzle's state in its face-by-face numbering.
    ///
    /// The same states `observations` gives, re-indexed and recolored by a
    /// `Layout`, so they can be handed to anything that speaks the usual
    /// layout of this puzzle.
    fn facelets(&mut self, py: Python<'_>, layout: &PyLayout) -> R<PyArray> {
        self.same_puzzle(layout)?;
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        let bound = self.inner.color_count().max(layout.inner.face_count()).max(1);
        with_cell!(bound, |C| {
            let colors: Vec<C> = py.detach(|| self.inner.observations()).map_err(to_py)?;
            let out = py.detach(|| layout.inner.to_facelets(&colors)).map_err(to_py)?;
            Ok(PyArray::ints(&out, bound, vec![n, k]))
        })
    }

    /// The same, for the identity of the sticker in each facelet.
    fn facelet_ids(&mut self, py: Python<'_>, layout: &PyLayout) -> R<PyArray> {
        self.same_puzzle(layout)?;
        let (n, k) = (self.inner.len(), self.inner.sticker_count());
        with_cell!(k.max(1), |I| {
            let ids: Vec<I> = py.detach(|| self.inner.sticker_ids()).map_err(to_py)?;
            let out = py.detach(|| layout.inner.to_facelet_ids(&ids)).map_err(to_py)?;
            Ok(PyArray::ints(&out, k.max(1), vec![n, k]))
        })
    }

    /// Draw states that did not come from this batch.
    ///
    /// `states` is one sticker array or a block of them, as `observations`
    /// gives them. The rows need not be this batch's own: a solver holding
    /// states of its own can have pictures of them without building a puzzle
    /// for each, and this batch's own states are untouched.
    #[pyo3(signature = (states, width = 256, height = 256, *, channels = 4, channels_first = false, dtype = "u1"))]
    #[allow(clippy::too_many_arguments, reason = "the signature is the Python one")]
    fn render_states(
        &mut self,
        py: Python<'_>,
        states: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        channels: usize,
        channels_first: bool,
        dtype: &str,
    ) -> R<PyArray> {
        let colors: Vec<u32> = values_of(states)?;
        let k = self.inner.sticker_count().max(1);
        let rows = colors.len() / k;
        let spec = crate::batch::FrameSpec {
            width,
            height,
            channels,
            channels_first,
        };
        let shape = if channels_first {
            vec![rows, channels, height as usize, width as usize]
        } else {
            vec![rows, height as usize, width as usize, channels]
        };
        let bytes = py.detach(|| self.inner.render_states(&colors, spec)).map_err(to_py)?;
        match dtype {
            "u1" | "uint8" => Ok(PyArray::bytes(bytes, shape)),
            "f4" | "float32" => {
                let scaled: Vec<f32> = bytes.iter().map(|&v| f32::from(v) / 255.0).collect();
                Ok(PyArray::floats(&scaled, shape))
            },
            other => Err(PyValueError::new_err(format!(
                "dtype must be \"u1\" or \"f4\", not {other:?}"
            ))),
        }
    }

    /// Render each puzzle instance into an individual [`Image`](PyImage) object.
    #[pyo3(signature = (width = 256, height = 256))]
    fn frames(&mut self, py: Python<'_>, width: u32, height: u32) -> R<Vec<PyImage>> {
        let data = py.detach(|| self.inner.render(width, height)).map_err(to_py)?;
        let frame = width as usize * height as usize * 4;
        data.chunks(frame.max(1))
            .map(|bytes| {
                crate::render::Framebuffer::from_raw(width, height, bytes.to_vec())
                    .map(|inner| PyImage { inner })
                    .ok_or_else(|| PyValueError::new_err("frame size does not match"))
            })
            .collect()
    }

    /* --- view settings ---------------------------------------------------- */

    /// Where the camera this batch draws from is: `(position, target, up,
    /// fov)`, with the three vectors as `(x, y, z)` and the field of view in
    /// degrees.
    ///
    /// What `configure` takes, so a caller who moves the camera to draw
    /// another viewpoint can put it back:
    ///
    ///     >>> where_it_was = batch.camera                  # doctest: +SKIP
    ///     >>> batch.configure(camera_position=elsewhere)   # doctest: +SKIP
    ///     >>> other = batch.render(32, 32)                 # doctest: +SKIP
    ///     >>> batch.configure(camera_position=where_it_was[0],
    ///     ...                 camera_target=where_it_was[1],
    ///     ...                 camera_up=where_it_was[2])   # doctest: +SKIP
    #[getter]
    fn camera(&self) -> Option<CameraView> {
        self.inner.camera().map(|c| {
            (
                (c.position.x, c.position.y, c.position.z),
                (c.target.x, c.target.y, c.target.z),
                (c.up.x, c.up.y, c.up.z),
                c.fov,
            )
        })
    }

    /// Change any of the view settings, for every frame drawn from now on.
    ///
    /// The same keywords the constructor takes.
    #[pyo3(signature = (
        *,
        background = None,
        supersample = None,
        show_edges = None,
        show_pieces = None,
        line_width = None,
        fov = None,
        distance = None,
        yaw = None,
        pitch = None,
        camera_position = None,
        camera_target = None,
        camera_up = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn configure(
        &mut self,
        background: Option<(u8, u8, u8, u8)>,
        supersample: Option<u32>,
        show_edges: Option<bool>,
        show_pieces: Option<bool>,
        line_width: Option<f64>,
        fov: Option<f64>,
        distance: Option<f64>,
        yaw: Option<f64>,
        pitch: Option<f64>,
        camera_position: Option<(f64, f64, f64)>,
        camera_target: Option<(f64, f64, f64)>,
        camera_up: Option<(f64, f64, f64)>,
    ) {
        let view = ViewOverrides {
            background,
            supersample,
            show_arrows: None,
            show_edges,
            show_pieces,
            line_width,
            fov,
            distance,
            yaw,
            pitch,
        };
        let placed = camera_position.is_some() || camera_target.is_some() || camera_up.is_some();
        if view.is_empty() && !placed {
            return;
        }
        self.inner.configure(&|sim| {
            view.apply(sim);
            // After the view settings, so that an explicit camera wins over a
            // yaw and a pitch given in the same call.
            let c = sim.camera_mut();
            if let Some(p) = camera_position {
                c.position = Vec3::new(p.0, p.1, p.2);
            }
            if let Some(p) = camera_target {
                c.target = Vec3::new(p.0, p.1, p.2);
            }
            if let Some(p) = camera_up {
                c.up = Vec3::new(p.0, p.1, p.2);
            }
        });
    }

    fn __repr__(&self) -> String {
        format!(
            "<PuzzleBatch {} of {} stickers={} actions={} {}>",
            self.inner.len(),
            self.inner.recipe(),
            self.inner.sticker_count(),
            self.inner.action_count(),
            if self.inner.is_tabular() {
                "tabular"
            } else {
                "geometric"
            },
        )
    }
}

/// Read whole numbers from whatever Python handed over.
///
/// An `Array` of this package's own, anything exposing a C-contiguous buffer
/// (a NumPy array of any unsigned or signed integer width, `bytes`, an
/// `array.array`), or a sequence, flat or one row per state. The buffer path
/// matters: a batch of a hundred thousand states read element by element
/// through the interpreter costs more than every move made over it, and read
/// as a block costs one pass over memory.
///
/// The values are narrowed to `C` as they are read, and a value too large for
/// it is refused instead of wrapped.
fn cells_of<C: Cell>(v: &Bound<'_, PyAny>) -> R<Vec<C>> {
    if let Ok(a) = v.extract::<PyRef<'_, PyArray>>() {
        return match a.typestr {
            "|u1" => a.data.iter().map(|&b| narrow(u32::from(b))).collect(),
            "<u2" | ">u2" => a
                .data
                .chunks_exact(2)
                .map(|c| narrow(u32::from(u16::from_ne_bytes([c[0], c[1]]))))
                .collect(),
            "<u4" | ">u4" => a
                .data
                .chunks_exact(4)
                .map(|c| narrow(u32::from_ne_bytes([c[0], c[1], c[2], c[3]])))
                .collect(),
            other => Err(PyTypeError::new_err(format!(
                "expected an array of whole numbers, not one of {other}"
            ))),
        };
    }
    if let Some(flat) = from_array_interface(v)? {
        return flat.into_iter().map(narrow).collect();
    }
    if let Ok(flat) = v.extract::<Vec<u32>>() {
        return flat.into_iter().map(narrow).collect();
    }
    v.extract::<Vec<Vec<u32>>>()
        .map_err(|_| PyTypeError::new_err("expected an array, a buffer, or a sequence of whole numbers"))?
        .into_iter()
        .flatten()
        .map(narrow)
        .collect()
}

/// The same, as plain `u32`, for a caller that will choose its width after
/// seeing what came in.
fn values_of(v: &Bound<'_, PyAny>) -> R<Vec<u32>> {
    cells_of::<u32>(v)
}

/// `v` at a narrower width, or a refusal if it does not fit.
#[inline]
fn narrow<C: Cell>(v: u32) -> R<C> {
    if u64::from(v) >= C::SPAN {
        return Err(PyValueError::new_err(format!(
            "{v} does not fit the {} values this puzzle's arrays hold",
            C::SPAN
        )));
    }
    Ok(C::from_index(v as usize))
}

/// Whole numbers read in one go out of a NumPy array or anything else that
/// describes itself the same way, or `None` for an object that does not.
///
/// Reading a hundred thousand states element by element through the
/// interpreter costs more than every move that will be made over them, and
/// `__array_interface__` is the arrangement NumPy already offers for not
/// doing that: it says how the block is encoded, and `bytes()` hands the
/// block over in one copy made in C. Nothing here reads the caller's memory
/// directly, so an object that describes itself wrongly gets a refusal and
/// not a fault.
fn from_array_interface(v: &Bound<'_, PyAny>) -> R<Option<Vec<u32>>> {
    let Ok(iface) = v.getattr("__array_interface__") else {
        return Ok(None);
    };
    let Ok(dict) = iface.cast::<PyDict>() else {
        return Ok(None);
    };
    // Strides mean the block is not laid out row after row, and `bytes()`
    // would copy something other than what the caller sees.
    if dict.get_item("strides")?.is_some_and(|s| !s.is_none()) {
        return Ok(None);
    }
    let Some(kind) = dict.get_item("typestr")? else {
        return Ok(None);
    };
    let kind: String = kind.extract()?;
    let py = v.py();
    let raw = py.get_type::<PyBytes>().call1((v,))?.extract::<Vec<u8>>()?;
    let (order, rest) = kind.split_at(1);
    let (sign, width) = rest.split_at(1);
    let width: usize = width
        .parse()
        .map_err(|_| PyTypeError::new_err(format!("expected an array of whole numbers, not one of {kind}")))?;
    if (sign != "u" && sign != "i") || !matches!(width, 1 | 2 | 4 | 8) {
        return Err(PyTypeError::new_err(format!(
            "expected an array of whole numbers, not one of {kind}"
        )));
    }
    let big = order == ">";
    let mut out = Vec::with_capacity(raw.len() / width);
    for chunk in raw.chunks_exact(width) {
        let mut word = [0u8; 8];
        if big {
            word[8 - width..].copy_from_slice(chunk);
        } else {
            word[..width].copy_from_slice(chunk);
        }
        let value = if big {
            u64::from_be_bytes(word)
        } else {
            u64::from_le_bytes(word)
        };
        // A signed array is read as the numbers it holds, and a negative one
        // is not a sticker, a face or a move.
        if sign == "i" && top_bit(chunk, big) {
            return Err(PyValueError::new_err(
                "a sticker, a face and a move are all counted from zero, and this array holds a \
                 negative number",
            ));
        }
        out.push(
            u32::try_from(value)
                .map_err(|_| PyValueError::new_err("this array holds a number too large for a sticker slot"))?,
        );
    }
    Ok(Some(out))
}

/// Whether a two's-complement number of these bytes is negative.
fn top_bit(chunk: &[u8], big: bool) -> bool {
    let top = if big { chunk[0] } else { chunk[chunk.len() - 1] };
    top & 0x80 != 0
}

/// How many distinct values an array that came in already knows how to hold.
///
/// An `Array` says so in its own type. Anything else is read at the narrowest
/// width that holds what is actually in it, which for a state means the
/// number of faces or of slots, whichever the caller sent.
fn span_of(v: &Bound<'_, PyAny>, values: &[u32]) -> usize {
    if let Ok(a) = v.extract::<PyRef<'_, PyArray>>() {
        return match a.itemsize {
            1 => 1 << 8,
            2 => 1 << 16,
            // Four bytes. A float is four bytes too, and lands here, which is
            // harmless only because `cells_of` matches on the type name and
            // has already refused one by the time anything asks how wide it
            // is. Widen that refusal before widening this arm.
            _ => usize::MAX,
        };
    }
    values.iter().copied().max().map_or(1, |m| m as usize + 1)
}

/// A puzzle's stickers numbered face by face, and the moves over them.
///
/// The simulator numbers slots in the order its geometry produces them. Where
/// a puzzle has a numbering the world already agrees on, this is the
/// translation between the two, worked out from the puzzle's own geometry.
/// Where it has none, this is the puzzle's own order, which is already
/// canonical, so that everything below works the same way for every puzzle.
///
/// **A cube.** Faces are numbered `U, D, L, R, B, F`, with `U` at `+y`, `R`
/// at `+x` and `F` at `+z`. Facelets within a face are read row-major, rows
/// running along `ROW` and columns along `COL`, where `ROW x COL` is the
/// outward normal: `U (+x, -z)`, `D (+x, +z)`, `L (+z, +y)`, `R (-z, +y)`,
/// `B (-x, +y)`, `F (+x, +y)`. The first twelve moves are the face quarter
/// turns, face-major in the same order, `2k` turning face `k` counterclockwise
/// seen from outside and `2k + 1` clockwise: `U-1`, `U1`, `D-1`, `D1`,
/// `L-1`, ... The names `U` and `U'` mean `U1` and `U-1`. A cube with more
/// layers than three can also turn its inner slices, and those follow, under
/// the puzzle's own names, each with a colon in front of it: a cube names its
/// grips `A` to `F` as well, so `F1` would otherwise mean two turns of a
/// 4x4x4.
///
/// **Anything else.** Faces are the puzzle's colors, facelets are its slots
/// and moves are its actions, each in the puzzle's own order and under its
/// own names.
///
/// **State representation.** State can be represented as an array of facelet
/// colors (where the goal on a 3x3x3 is `arange(54) // 9`), or as sticker
/// permutations (where the goal is `arange(54)`). The permutation formulation
/// distinguishes states that identical colors cannot differentiate.
///
/// Because moves are discrete permutations of an index array, a `Layout` operates
/// as an independent permutation engine: `next_states` evaluates transitions
/// across batches without invoking 3D geometric simulation.
///
///     >>> import twistypuzzle as tp
///     >>> c = tp.Layout()
///     >>> c.size, c.facelet_count, c.move_names[:4]
///     (3, 54, ['U-1', 'U1', 'D-1', 'D1'])
///     >>> goal = c.goal_colors
///     >>> after = c.next_states(goal, c.move_index("R"))
///     >>> after.tolist()[:9]
///     [0, 0, 0, 0, 0, 0, 5, 5, 5]
///     >>> m = tp.Layout("Megaminx")
///     >>> m.is_cube, m.face_count, m.facelet_count
///     (False, 12, 132)
#[pyclass(name = "Layout", module = "twistypuzzle")]
pub struct PyLayout {
    inner: crate::layout::Layout,
}

impl PyLayout {
    /// How many values a face or a color of this puzzle can be: what decides
    /// how wide its state arrays are.
    fn cell_bound(&self) -> usize {
        self.inner.face_count().max(self.inner.color_count())
    }
}

#[pymethods]
impl PyLayout {
    /// Work out the layout of a puzzle, from the puzzle itself.
    ///
    /// Raises `PuzzleError` for a puzzle whose moves are not fixed
    /// permutations of its stickers: one that jumbles, whose turns leave a
    /// sticker where none sits when it is solved, or one with a layer that
    /// locks, where which moves are available depends on the state.
    #[new]
    #[pyo3(signature = (recipe = None))]
    fn new(py: Python<'_>, recipe: Option<&Bound<'_, PyAny>>) -> R<PyLayout> {
        let query = match recipe {
            Some(v) => recipe_query(v)?,
            None => "?shell=C$1&cut=C$1/3".to_string(),
        };
        py.detach(|| {
            let mut sim = crate::simulator::Simulator::from_query(&query)?;
            Ok(PyLayout {
                inner: crate::layout::Layout::build(&mut sim)?,
            })
        })
        .map_err(to_py)
    }

    /// How many facelets a face has along one side, for a cube: 3 for a
    /// 3x3x3. `None` for a puzzle whose faces are not square grids.
    #[getter]
    fn size(&self) -> Option<usize> {
        self.inner.size()
    }

    /// Whether this puzzle is a cube, and so read by the conventions cubes
    /// are read by.
    #[getter]
    fn is_cube(&self) -> bool {
        self.inner.is_cube()
    }

    /// How many facelets there are, which is how long one state is.
    #[getter]
    fn facelet_count(&self) -> usize {
        self.inner.facelet_count()
    }

    /// How many faces there are, which is how many values a facelet holds.
    #[getter]
    fn face_count(&self) -> usize {
        self.inner.face_count()
    }

    /// How many facelets each face carries. Equal for a cube, and not in
    /// general: a prism's sides and its ends are both faces.
    #[getter]
    fn face_sizes(&self) -> PyArray {
        let sizes = self.inner.face_sizes();
        PyArray::ints(sizes, self.inner.facelet_count() + 1, vec![self.inner.face_count()])
    }

    /// How many moves there are.
    #[getter]
    fn move_count(&self) -> usize {
        self.inner.move_count()
    }

    /// The faces, in the order they are numbered.
    #[getter]
    fn faces(&self) -> Vec<String> {
        self.inner.face_names().to_vec()
    }

    /// Every move's name, in move order.
    #[getter]
    fn move_names(&self) -> Vec<String> {
        self.inner.move_names().to_vec()
    }

    /// Every move as a permutation of the facelets, `(move_count,
    /// facelet_count)`: `new[i] = old[perm[m][i]]`.
    #[getter]
    fn moves(&self) -> PyArray {
        let k = self.inner.facelet_count();
        let flat: Vec<u32> = self.inner.moves().concat();
        PyArray::ints(&flat, k.max(1), vec![self.inner.move_count(), k])
    }

    /// The puzzle's own action index for each move, or `None` where it has no
    /// single action for it.
    #[getter]
    fn actions(&self) -> Vec<Option<u32>> {
        self.inner.actions().to_vec()
    }

    /// The move that undoes each move, or `None` where this layout has none.
    #[getter]
    fn inverses(&self) -> Vec<Option<u32>> {
        self.inner.inverses().to_vec()
    }

    /// Where each of the puzzle's sticker slots sits in the face numbering.
    #[getter]
    fn facelet_of_slot(&self) -> PyArray {
        let k = self.inner.facelet_count();
        PyArray::ints(self.inner.facelet_of_slot(), k.max(1), vec![k])
    }

    /// Which slot each facelet is, the other way around.
    #[getter]
    fn slot_of_facelet(&self) -> PyArray {
        let k = self.inner.facelet_count();
        PyArray::ints(self.inner.slot_of_facelet(), k.max(1), vec![k])
    }

    /// The face each facelet belongs to when solved, which for a 3x3x3 is
    /// `arange(54) // 9`.
    #[getter]
    fn goal_colors(&self) -> PyArray {
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        with_cell!(bound, |C| {
            let goal: Vec<C> = self.inner.goal_colors();
            PyArray::ints(&goal, bound, vec![k])
        })
    }

    /// The sticker a solved puzzle has in each facelet, which is the facelet
    /// itself: `arange(n)`.
    #[getter]
    fn goal_ids(&self) -> PyArray {
        let k = self.inner.facelet_count();
        with_cell!(k.max(1), |I| {
            let goal: Vec<I> = self.inner.goal_ids();
            PyArray::ints(&goal, k.max(1), vec![k])
        })
    }

    /// The same layout with a different set of moves: each of `sequences`
    /// composed into one.
    ///
    /// Reinforcement learning agents and search algorithms frequently utilize
    /// macro-moves instead of single turns (for example, all sequences of length 3,
    /// expanding 12 primitive actions into 1,728 macro-actions).
    /// All other layout properties remain unchanged. The `next_states` method steps via
    /// the specified macro permutations.
    ///
    ///     >>> import itertools, twistypuzzle as tp
    ///     >>> triples = tp.Layout().with_moves(list(itertools.product(range(12), repeat=3)))
    ///     >>> triples.move_count
    ///     1728
    ///
    /// Macro-moves are composite operations instead of atomic puzzle actions.
    /// The `inverse_moves` method returns results for those macro-moves whose inverses
    /// are also present within the set.
    #[allow(clippy::needless_pass_by_value, reason = "pyo3 extracts an owned Vec")]
    fn with_moves(&self, py: Python<'_>, sequences: Vec<Vec<u32>>) -> R<PyLayout> {
        py.detach(|| {
            Ok(PyLayout {
                inner: self.inner.with_moves(&sequences)?,
            })
        })
        .map_err(to_py)
    }

    /// The move a name stands for.
    ///
    /// A cube's faces answer to `"U1"`, `"U-1"`, `"U"` and `"U'"`, where `U`
    /// and `U1` are the clockwise quarter turn seen from outside the face.
    /// Every other move answers to the name it is listed under.
    fn move_index(&self, name: &str) -> Option<usize> {
        self.inner.move_index(name)
    }

    /// The puzzle's own action index for each of `moves`.
    ///
    /// What to hand `PuzzleBatch.step` to make these moves on real puzzles.
    fn moves_to_actions(&self, moves: Vec<u32>) -> R<Vec<u32>> {
        moves
            .into_iter()
            .map(|m| {
                self.inner
                    .actions()
                    .get(m as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| PyValueError::new_err(format!("move {m} has no action of this puzzle")))
            })
            .collect()
    }

    /// Return the inverse move index for each move in `moves`.
    ///
    /// Computed directly from permutation inverses instead of index arithmetic,
    /// supporting both primitive moves and composite macro-moves. Raises
    /// `ValueError` if an inverse permutation is absent from the configured move set.
    fn inverse_moves(&self, moves: &Bound<'_, PyAny>) -> R<PyArray> {
        let m = values_of(moves)?;
        let back = self.inner.inverse_moves(&m).map_err(to_py)?;
        let n = back.len();
        Ok(PyArray::ints(&back, self.inner.move_count().max(1), vec![n]))
    }

    /* --- translating ------------------------------------------------------ */

    /// Rewrite states from the puzzle's slot order into face order, with each
    /// color replaced by the face it belongs to.
    fn to_facelets(&self, py: Python<'_>, states: &Bound<'_, PyAny>) -> R<PyArray> {
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        with_cell!(bound, |C| {
            let src: Vec<C> = cells_of(states)?;
            let out = py.detach(|| self.inner.to_facelets(&src)).map_err(to_py)?;
            Ok(PyArray::ints(&out, bound, shape_of(src.len(), k)))
        })
    }

    /// Rewrite sticker identities from slot order into face order.
    fn to_facelet_ids(&self, py: Python<'_>, ids: &Bound<'_, PyAny>) -> R<PyArray> {
        let k = self.inner.facelet_count();
        with_cell!(k.max(1), |I| {
            let src: Vec<I> = cells_of(ids)?;
            let out = py.detach(|| self.inner.to_facelet_ids(&src)).map_err(to_py)?;
            Ok(PyArray::ints(&out, k.max(1), shape_of(src.len(), k)))
        })
    }

    /// Rewrite states in face order back into the puzzle's slot order and its
    /// own colors: the way in for a state that came from somewhere else.
    fn from_facelets(&self, py: Python<'_>, facelets: &Bound<'_, PyAny>) -> R<PyArray> {
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        with_cell!(bound, |C| {
            let src: Vec<C> = cells_of(facelets)?;
            let out = py.detach(|| self.inner.from_facelets(&src)).map_err(to_py)?;
            Ok(PyArray::ints(&out, bound, shape_of(src.len(), k)))
        })
    }

    /* --- turning ---------------------------------------------------------- */

    /// Turn a batch of states, one move each, entirely in the array.
    ///
    /// `states` is one state or a block of them, holding colors or identities
    /// as you please: a move permutes the facelets and does not care what is
    /// sitting in them. `moves` is one move, or one per state. The result
    /// holds what it was given, at the width it was given it in.
    fn next_states(&self, py: Python<'_>, states: &Bound<'_, PyAny>, moves: &Bound<'_, PyAny>) -> R<PyArray> {
        let k = self.inner.facelet_count();
        let raw = values_of(states)?;
        let bound = span_of(states, &raw);
        let rows = raw.len() / k.max(1);
        let m = if let Ok(one) = moves.extract::<u32>() {
            vec![one; rows]
        } else {
            // Read as a block, like the states: a move per state, for a
            // hundred thousand states, is not a thing to pull through the
            // interpreter one at a time.
            values_of(moves).map_err(|_| PyTypeError::new_err("expected a move, or one per state"))?
        };
        let shape = shape_of(raw.len(), k);
        with_cell!(bound, |C| {
            let src: Vec<C> = raw.iter().map(|&v| C::from_index(v as usize)).collect();
            let out = py.detach(|| self.inner.next_states(&src, &m)).map_err(to_py)?;
            Ok(PyArray::ints(&out, bound, shape))
        })
    }

    /// Walk states away from solved and keep the whole path.
    ///
    /// `steps` is how far each walk goes: one number for all of them,
    /// `(low, high)` to draw a length for each, or one number each. Returns
    /// `(states, moves)`: the state after every turn, shaped `(rows, steps +
    /// 1, facelet_count)` with the solved state first, and the moves that
    /// produced them, shaped `(rows, steps)`. A walk never immediately takes
    /// back the move it has just made, so a walk of `k` moves is solvable in
    /// at most `k`. Rows given fewer steps than the longest stop early and
    /// repeat their last state, so the block stays rectangular. The moves
    /// those rows did not make are left as `move_count`, which is one past
    /// the last real move and is how a short row is recognized.
    ///
    /// One call gives an episode generator everything it needs: the start
    /// states are the last state of each walk, the goal is the first, and the
    /// moves between them are the solution read backward.
    #[pyo3(signature = (steps, count = None, *, seed = 0))]
    fn trajectories(
        &self,
        py: Python<'_>,
        steps: &Bound<'_, PyAny>,
        count: Option<usize>,
        seed: u64,
    ) -> R<(PyArray, PyArray)> {
        let each = depths_from(steps, count.unwrap_or(1), seed)?;
        let rows = each.len();
        let longest = each.iter().copied().max().unwrap_or(0) as usize;
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        let moved = self.inner.move_count() + 1;
        with_cell!(bound, |C| {
            let (states, moves): (Vec<C>, Vec<u32>) =
                py.detach(|| self.inner.trajectories(&each, seed)).map_err(to_py)?;
            Ok((
                PyArray::ints(&states, bound, vec![rows, longest + 1, k]),
                PyArray::ints(&moves, moved, vec![rows, longest.max(1)]),
            ))
        })
    }

    /// Walk `count` states `depth` moves away from solved, keeping only where
    /// each ended up.
    ///
    /// The usual way to make start states: a state drawn uniformly is almost
    /// always as far from solved as a state can be, where a walk of known
    /// length is solvable in at most that many moves by construction.
    ///
    /// `depth` is one number, `(low, high)` to draw a length for each state,
    /// or one number per state. Only the state each walk ends at is kept, so
    /// this costs what it returns however deep the walks are, which
    /// `trajectories` does not.
    #[pyo3(signature = (count, depth, *, seed = 0))]
    fn scrambled(&self, py: Python<'_>, count: usize, depth: &Bound<'_, PyAny>, seed: u64) -> R<PyArray> {
        let each = depths_from(depth, count, seed)?;
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        with_cell!(bound, |C| {
            let states: Vec<C> = py.detach(|| self.inner.walk(&each, seed)).map_err(to_py)?;
            Ok(PyArray::ints(&states, bound, vec![count, k]))
        })
    }

    /* --- encodings -------------------------------------------------------- */

    /// States as indicator bytes, one per (facelet, face) pair.
    fn one_hot(&self, py: Python<'_>, facelets: &Bound<'_, PyAny>) -> R<PyArray> {
        let (k, bound) = (self.inner.facelet_count(), self.cell_bound());
        let faces = self.inner.face_count();
        with_cell!(bound, |C| {
            let src: Vec<C> = cells_of(facelets)?;
            let out = py.detach(|| self.inner.one_hot(&src)).map_err(to_py)?;
            let rows = src.len() / k.max(1);
            Ok(PyArray::bytes(
                out,
                if src.len() == k {
                    vec![k * faces]
                } else {
                    vec![rows, k * faces]
                },
            ))
        })
    }

    /// Whether each state is solved: every facelet holding its own face.
    fn solved(&self, py: Python<'_>, facelets: &Bound<'_, PyAny>) -> R<PyArray> {
        let bound = self.cell_bound();
        with_cell!(bound, |C| {
            let src: Vec<C> = cells_of(facelets)?;
            let out = py.detach(|| self.inner.solved(&src)).map_err(to_py)?;
            let rows = out.len();
            Ok(PyArray::flags(out, vec![rows]))
        })
    }

    /// One state written out, a line per face.
    fn text(&self, facelets: &Bound<'_, PyAny>) -> R<String> {
        let bound = self.cell_bound();
        with_cell!(bound, |C| {
            let src: Vec<C> = cells_of(facelets)?;
            Ok(self.inner.text(&src))
        })
    }

    fn __repr__(&self) -> String {
        match self.inner.size() {
            Some(n) => format!(
                "<Layout cube {n}x{n}x{n} facelets={} moves={}>",
                self.inner.facelet_count(),
                self.inner.move_count()
            ),
            None => format!(
                "<Layout faces={} facelets={} moves={}>",
                self.inner.face_count(),
                self.inner.facelet_count(),
                self.inner.move_count()
            ),
        }
    }
}

/// Read a walk length given as one number, as `(low, high)` to draw from, or
/// as one number for each of `count` walks.
fn depths_from(depth: &Bound<'_, PyAny>, count: usize, seed: u64) -> R<Vec<u32>> {
    if let Ok(one) = depth.extract::<u32>() {
        return Ok(vec![one; count]);
    }
    if let Ok((low, high)) = depth.extract::<(u32, u32)>() {
        if high < low {
            return Err(PyValueError::new_err(format!(
                "a scramble range runs from low to high, and {low} to {high} does not"
            )));
        }
        // Drawn here instead of in the caller so that one seed fixes the
        // whole thing: the lengths and the walks that use them.
        let span = u64::from(high - low) + 1;
        let mut state = seed ^ 0x51ED_2701_A5C6_1B3D;
        return Ok((0..count)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                low + ((z ^ (z >> 31)) % span) as u32
            })
            .collect());
    }
    let many: Vec<u32> = values_of(depth)
        .map_err(|_| PyTypeError::new_err("expected a number of moves, a (low, high) range, or one per state"))?;
    if many.len() != count {
        return Err(PyValueError::new_err(format!(
            "got {} depths for {count} states",
            many.len()
        )));
    }
    Ok(many)
}

/// Where the camera is: position, target, up, and field of view in degrees.
type CameraView = ((f64, f64, f64), (f64, f64, f64), (f64, f64, f64), f64);

/// `(k,)` for a single state and `(rows, k)` for a block of them.
fn shape_of(len: usize, k: usize) -> Vec<usize> {
    if k == 0 || len == k {
        vec![len]
    } else {
        vec![len / k, k]
    }
}

/// Whether any turn of a puzzle moves a sticker off the solved lattice.
///
/// A puzzle that jumbles (such as a Radiolarian, a jumble prism, or the Big Chop) has
/// legal turns that leave pieces where no piece sits when it is solved, so
/// there is no fixed set of sticker slots to number and no sticker array.
/// Thirty-six of the eighty-five cataloged puzzles do not jumble, and
/// `non_jumbling()` lists them.
#[pyfunction]
fn jumbles(py: Python<'_>, recipe: &Bound<'_, PyAny>) -> R<bool> {
    let query = recipe_query(recipe)?;
    py.detach(move || crate::symbolic::jumbles(&query)).map_err(to_py)
}

/// The cataloged recipes that do not jumble, and so have a sticker array.
#[pyfunction]
fn non_jumbling() -> Vec<&'static str> {
    crate::symbolic::NON_JUMBLING.to_vec()
}

/* -------------------------------------------------------------------------- */
/*  Module                                                                    */
/* -------------------------------------------------------------------------- */

/// `gil_used = false` tells CPython this module does not need the global
/// interpreter lock. Without it a free-threaded interpreter re-enables the GIL
/// the moment the module is imported, which would undo the point of the build.
/// Nothing here relies on the GIL for synchronization: the shared state is a
/// number field, and its only mutable part sits behind its own lock.
#[pymodule(gil_used = false)]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("PuzzleError", m.py().get_type::<PuzzleError>())?;
    m.add_class::<PyFraction>()?;
    m.add_class::<PyReal>()?;
    m.add_class::<PyImage>()?;
    m.add_class::<PyPuzzle>()?;
    m.add_class::<PyShape>()?;
    m.add_class::<PyRecipe>()?;
    m.add_class::<PyGrip>()?;
    m.add_class::<PyPuzzleBatch>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyLayout>()?;
    m.add_function(wrap_pyfunction!(catalog, m)?)?;
    m.add_function(wrap_pyfunction!(polyhedra, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(factor_polynomial, m)?)?;
    m.add_function(wrap_pyfunction!(build_many, m)?)?;
    m.add_function(wrap_pyfunction!(render_many, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    m.add_function(wrap_pyfunction!(jumbles, m)?)?;
    m.add_function(wrap_pyfunction!(non_jumbling, m)?)?;
    // True when this build of the extension runs on a free-threaded
    // interpreter, i.e. one that never takes a global interpreter lock.
    m.add("free_threaded", cfg!(Py_GIL_DISABLED))?;
    Ok(())
}
