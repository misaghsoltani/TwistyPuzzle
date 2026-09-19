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
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyTuple};

use crate::error::Error;
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
    // Moving the message out rather than cloning it: an error is built once
    // and converted once, so there is nothing left to borrow it for.
    match e {
        Error::Range(m) | Error::Division(m) if m.contains("Division by zero") => {
            PyZeroDivisionError::new_err(m)
        },
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
/// it could have meant rather than picking one.
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
            "{} cataloged puzzles are named {spec:?}; build one by recipe instead: {}",
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
/// A ratio is written as text (such as `"1/3"`) rather than as a float, because a
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
            return Err(PyValueError::new_err(format!(
                "{name:?} is not a polyhedron code"
            )));
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
    fn plane(
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        c: &Bound<'_, PyAny>,
        d: &Bound<'_, PyAny>,
    ) -> R<PyShape> {
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
                let (a, b, c) = self
                    .coefficients()?
                    .expect("a plane always has coefficients");
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
    /// ambiguous name says which recipes it could have meant rather than
    /// picking one.
    #[staticmethod]
    fn named(name: &str) -> R<PyRecipe> {
        PyRecipe::parse(&resolve_recipe(name)?)
    }

    /// The shapes the puzzle is carved from.
    #[getter]
    fn shell(&self) -> Vec<PyShape> {
        self.inner
            .shell
            .iter()
            .map(|s| PyShape { inner: s.clone() })
            .collect()
    }

    /// The shapes that cut it.
    #[getter]
    fn cuts(&self) -> Vec<PyShape> {
        self.inner
            .cuts
            .iter()
            .map(|s| PyShape { inner: s.clone() })
            .collect()
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
            "unknown blend mode {other:?}; expected 'copy', 'over' or 'add'"
        ))),
    }
}

#[pymethods]
impl PyImage {
    #[new]
    #[pyo3(signature = (width, height, color = (0, 0, 0, 0)))]
    fn new(width: u32, height: u32, color: (u8, u8, u8, u8)) -> PyImage {
        PyImage {
            inner: crate::render::Framebuffer::filled(
                width,
                height,
                [color.0, color.1, color.2, color.3],
            ),
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
            PyTuple::new(
                py,
                [self.inner.height() as usize, self.inner.width() as usize, 4],
            )?,
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
        self.inner
            .set_pixel(x, y, [color.0, color.1, color.2, color.3]);
        Ok(())
    }

    /// Fill the whole image with one color.
    fn fill(&mut self, color: (u8, u8, u8, u8)) {
        self.inner.fill([color.0, color.1, color.2, color.3]);
    }

    /// Fill a rectangle.
    #[pyo3(signature = (x, y, width, height, color, mode = "copy"))]
    fn fill_rect(
        &mut self,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
        color: (u8, u8, u8, u8),
        mode: &str,
    ) -> R<()> {
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
        format!(
            "<Image {}x{} RGBA8>",
            self.inner.width(),
            self.inner.height()
        )
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
    /// Apply these settings, handing back what they replaced so it can be put
    /// back afterward.
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
        // part of the same placement rather than a separate one.
        if self.yaw.is_some() || self.pitch.is_some() {
            let d = self
                .distance
                .unwrap_or_else(|| sim.camera().position.length());
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
                    return Err(PyTypeError::new_err(
                        "give a recipe, or a shell and its cuts",
                    ));
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
        py.detach(|| self.sim.begin_move_to(index, stop))
            .map_err(to_py)
    }

    /// Take the last turn back, returning `False` if there is none to take or
    /// the layer is locked.
    fn undo(&mut self, py: Python<'_>) -> R<bool> {
        py.detach(|| self.sim.undo()).map_err(to_py)
    }

    /// How many turns have been made and not taken back.
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

    /// Take every turn back, leaving the puzzle solved.
    ///
    /// Far cheaper than [`reset`](Self::reset): a turn is undone by one exact
    /// rotation, where a rebuild re-derives the whole geometry. Returns
    /// `False` if a locked layer stopped it part-way, in which case the puzzle
    /// is as far back as it could get and `reset` is the way home.
    fn restore(&mut self, py: Python<'_>) -> R<bool> {
        py.detach(|| self.sim.restore()).map_err(to_py)
    }

    /// Build the puzzle again from its recipe, keeping the camera and view
    /// settings.
    ///
    /// Always works, and always costs a build. [`restore`](Self::restore) is
    /// the cheap way back for a puzzle that has only been turned.
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

    /// The color in each sticker slot: the state as a network sees it.
    ///
    /// Slots are numbered over the solved puzzle.
    ///
    /// Raises `PuzzleError` for a puzzle that jumbles, which has no fixed set
    /// of slots to number (`jumbles()` says which those are).
    fn stickers(&mut self, py: Python<'_>) -> R<Vec<u16>> {
        py.detach(|| self.sim.stickers()).map_err(to_py)
    }

    /// Where the sticker in each slot belongs: the state as a permutation.
    ///
    /// Solved, this is `range(sticker_count)`.
    fn sticker_ids(&mut self, py: Python<'_>) -> R<Vec<u32>> {
        py.detach(|| self.sim.sticker_ids()).map_err(to_py)
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
    /// puzzle, the permutation recorded, and the move taken back.
    ///
    /// Turning the geometry costs about two milliseconds a move, because it
    /// re-derives the whole cut structure, whereas a gather costs nanoseconds. Deriving
    /// the table costs two turns per move, once.
    ///
    /// `None` means the moves are not fixed permutations (such as a puzzle that
    /// jumbles, or one whose layers lock), and that the geometry has to be
    /// turned move by move instead. Deriving it leaves the puzzle as it found
    /// it.
    fn action_permutations(&mut self, py: Python<'_>) -> R<Option<Vec<Vec<u32>>>> {
        py.detach(|| {
            Ok(crate::symbolic::PermutationTable::build(&mut self.sim)?
                .map(|t| t.permutations().to_vec()))
        })
        .map_err(to_py)
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
    /// that into a `RuntimeError` rather than letting it happen quietly.
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
        let fb = py
            .detach(|| self.sim.frame(dt_ms, width, height))
            .map_err(to_py)?;
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
        self.sim.camera_mut().position = crate::math::Vec3::new(p.0, p.1, p.2);
    }

    /// Which way is up for the camera.
    #[getter]
    fn camera_up(&self) -> (f64, f64, f64) {
        let p = self.sim.camera().up;
        (p.x, p.y, p.z)
    }

    #[setter]
    fn set_camera_up(&mut self, p: (f64, f64, f64)) {
        self.sim.camera_mut().up = crate::math::Vec3::new(p.0, p.1, p.2);
    }

    /// What the camera looks at.
    #[getter]
    fn camera_target(&self) -> (f64, f64, f64) {
        let p = self.sim.camera().target;
        (p.x, p.y, p.z)
    }

    #[setter]
    fn set_camera_target(&mut self, p: (f64, f64, f64)) {
        self.sim.camera_mut().target = crate::math::Vec3::new(p.0, p.1, p.2);
    }

    /// Look at the puzzle from `yaw` and `pitch` degrees off head-on.
    ///
    /// `(0, 0)` is the default, head-on camera. A little of each shows the
    /// puzzle as a solid rather than a silhouette.
    #[pyo3(signature = (yaw, pitch, distance = None))]
    fn look_from(&mut self, yaw: f64, pitch: f64, distance: Option<f64>) {
        let d = distance.unwrap_or_else(|| self.sim.camera().position.length());
        self.sim.look_from(yaw, pitch, d);
    }

    /// Orbit the camera by a normalized drag delta.
    fn orbit(&mut self, dx: f64, dy: f64) {
        self.sim
            .controls_mut()
            .pointer_down(Pointer { x: 0.0, y: 0.0 });
        self.sim
            .controls_mut()
            .pointer_move(Pointer { x: dx, y: dy });
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
        py.detach(|| self.sim.pointer_up(Pointer { x, y }))
            .map_err(to_py)
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
    let items = v.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "{what} must be a Shape, its text, or a sequence of either"
        ))
    })?;
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
/// can name the entry the caller gave rather than the query it became.
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

/// Many copies of one puzzle, turned together.
///
/// Built for vectorized reinforcement learning, where the cost of a step is
/// dominated by crossing into the interpreter once per environment. A whole
/// batch of turns is one call here: the interpreter is released for all of it,
/// the turns run across every core, and the states come back as one block of
/// bytes that NumPy views without copying.
///
///     >>> import twistypuzzle as tp
///     >>> b = tp.PuzzleBatch("Rubik's Cube (3x3x3)", 4)
///     >>> b.reset(scramble=5)
///     >>> len(b.observations()) == len(b) * b.sticker_count
///     True
///     >>> _ = b.step([0, 1, 2, 3])
#[pyclass(name = "PuzzleBatch", module = "twistypuzzle")]
pub struct PyPuzzleBatch {
    inner: crate::batch::PuzzleBatch,
}

#[pymethods]
impl PyPuzzleBatch {
    /// Build `count` copies of one puzzle.
    #[new]
    #[pyo3(signature = (recipe, count, *, seed = None))]
    fn new(
        py: Python<'_>,
        recipe: &Bound<'_, PyAny>,
        count: usize,
        seed: Option<u64>,
    ) -> R<PyPuzzleBatch> {
        let query = recipe_query(recipe)?;
        let mut inner = py
            .detach(move || crate::batch::PuzzleBatch::build(&query, count))
            .map_err(to_py)?;
        if let Some(s) = seed {
            for i in 0..inner.len() {
                // Distinct streams, derived from one seed, so a batch is
                // reproducible as a whole and no two copies scramble alike.
                inner.seed(
                    i,
                    s.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                );
            }
        }
        Ok(PyPuzzleBatch { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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
        self.inner.view().stickers.color_count()
    }

    /// Every move name, in action-index order.
    #[getter]
    fn actions(&self) -> Vec<String> {
        self.inner.view().actions.action_names()
    }

    /// The colors of a solved puzzle: the goal every copy is working toward.
    #[getter]
    fn solved_stickers(&self) -> Vec<u16> {
        self.inner.view().stickers.solved_colors().to_vec()
    }

    /// Seed one puzzle's scrambler.
    fn seed(&mut self, index: usize, seed: u64) -> R<()> {
        if index >= self.inner.len() {
            return Err(PyIndexError::new_err("puzzle index out of range"));
        }
        self.inner.seed(index, seed);
        Ok(())
    }

    /// Solve every puzzle, then walk `scramble` random turns away from solved.
    ///
    /// A walk from the goal rather than a state drawn uniformly: a uniform
    /// state is almost always as far from solved as a state can be, and teaches
    /// a learner nothing, where a walk of known length is reachable by
    /// construction.
    #[pyo3(signature = (*, scramble = 0))]
    fn reset(&mut self, py: Python<'_>, scramble: u32) -> R<()> {
        py.detach(|| self.inner.reset_all(scramble)).map_err(to_py)
    }

    /// Solve one puzzle and scramble just that one, leaving the rest alone.
    #[pyo3(signature = (index, *, scramble = 0))]
    fn reset_one(&mut self, py: Python<'_>, index: usize, scramble: u32) -> R<()> {
        if index >= self.inner.len() {
            return Err(PyIndexError::new_err("puzzle index out of range"));
        }
        py.detach(|| self.inner.reset_one(index, scramble))
            .map_err(to_py)
    }

    /// Turn every puzzle once.
    ///
    /// Returns `(solved, applied)`: whether each puzzle is now solved, and
    /// whether its move could be made at all. A move naming a layer that
    /// cannot turn leaves that puzzle alone rather than raising, so a learner
    /// need not know the mask to act.
    fn step(&mut self, py: Python<'_>, actions: Vec<u32>) -> R<(Vec<bool>, Vec<bool>)> {
        let out = py
            .detach(move || self.inner.step(&actions))
            .map_err(to_py)?;
        Ok((
            out.iter().map(|o| o.solved).collect(),
            out.iter().map(|o| o.applied).collect(),
        ))
    }

    /// Every puzzle's sticker array, one after another, as `len * sticker_count`
    /// bytes.
    ///
    /// A `bytearray`, so `numpy.frombuffer(...).reshape(n, k)` is a writable
    /// view of it and nothing is copied on the way out.
    fn observations<'py>(&mut self, py: Python<'py>) -> R<Bound<'py, PyByteArray>> {
        let bytes = py.detach(|| self.inner.observations()).map_err(to_py)?;
        Ok(PyByteArray::new(py, &bytes))
    }

    /// Which moves each puzzle can make, one after another, as
    /// `len * action_count` flags.
    fn action_masks(&mut self, py: Python<'_>) -> R<Vec<bool>> {
        py.detach(|| self.inner.action_masks()).map_err(to_py)
    }

    /// Whether each puzzle is solved.
    fn solved(&mut self, py: Python<'_>) -> R<Vec<bool>> {
        py.detach(|| self.inner.solved()).map_err(to_py)
    }

    /// Draw every puzzle, across every core.
    #[pyo3(signature = (width = 256, height = 256, *, background = None, supersample = None,
                        show_arrows = None, show_edges = None, yaw = None, pitch = None,
                        distance = None))]
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
        yaw: Option<f64>,
        pitch: Option<f64>,
        distance: Option<f64>,
    ) -> Vec<PyImage> {
        let view = ViewOverrides {
            background,
            supersample,
            show_arrows,
            show_edges,
            show_pieces: None,
            line_width: None,
            fov: None,
            distance,
            yaw,
            pitch,
        };
        py.detach(|| {
            use rayon::prelude::*;
            self.inner
                .sims_mut()
                .par_iter_mut()
                .map(|sim| {
                    let (options, camera) = view.apply(sim);
                    let fb = sim.render(width, height);
                    *sim.options_mut() = options;
                    *sim.camera_mut() = camera;
                    PyImage { inner: fb }
                })
                .collect()
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "<PuzzleBatch {} puzzles, {} stickers, {} actions>",
            self.inner.len(),
            self.inner.sticker_count(),
            self.inner.action_count()
        )
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
    py.detach(move || crate::symbolic::jumbles(&query))
        .map_err(to_py)
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
