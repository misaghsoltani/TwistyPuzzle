//! Face planes of the supported polyhedra.
//!
//! The tables in [`polyhedra_data`] are exact: every plane is a rational
//! combination of radicals, simplified symbolically once and written down,
//! instead of computed numerically at run time. They are data, not code, and
//! are not meant to be edited by hand.
//!
//! Encoding (prefix notation):
//!
//! | token | meaning |
//! |---|---|
//! | `S` | the `scale` placeholder supplied by the caller |
//! | `A x y` | `x + y` |
//! | `M x y` | `x * y` |
//! | `P x y` | `x ^ y` |
//! | `n<num>/<den>;` | a rational literal |

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::error::{Error, Result};
use crate::num::fraction::Fraction;
use crate::num::int::Int;
use crate::parse::{Expr, Shape};
use crate::polyhedra_data::SHAPES;

struct Decoder<'a> {
    bytes: &'a [u8],
    i: usize,
}

impl Decoder<'_> {
    fn expr(&mut self) -> Result<Expr> {
        let tag = *self
            .bytes
            .get(self.i)
            .ok_or_else(|| Error::Other("polyhedron table truncated".into()))?;
        self.i += 1;
        match tag {
            b'S' => Ok(Expr::op("$scale", vec![])),
            b'A' | b'M' | b'P' => {
                let a = self.expr()?;
                let b = self.expr()?;
                let op = match tag {
                    b'A' => "+",
                    b'M' => "*",
                    _ => "^",
                };
                Ok(Expr::op(op, vec![a, b]))
            },
            b'n' => {
                let start = self.i;
                while self.i < self.bytes.len() && self.bytes[self.i] != b';' {
                    self.i += 1;
                }
                let s = core::str::from_utf8(&self.bytes[start..self.i])
                    .map_err(|_| Error::Other("polyhedron table is not valid UTF-8".into()))?;
                self.i += 1; // skip ';'
                let (n, d) = s
                    .split_once('/')
                    .ok_or_else(|| Error::Other("malformed rational in table".into()))?;
                let n = Int::parse(n).ok_or_else(|| Error::Other("malformed numerator in table".into()))?;
                let d = Int::parse(d).ok_or_else(|| Error::Other("malformed denominator in table".into()))?;
                Ok(Expr::num(Fraction::new(n, d, true)?))
            },
            other => Err(Error::Other(format!("unexpected byte {other:?} in polyhedron table"))),
        }
    }
}

/// Replace every `$scale` placeholder with the caller's expression.
fn substitute(x: &Expr, scale: &Expr) -> Expr {
    if x.op == "$scale" {
        return scale.clone();
    }
    Expr {
        op: x.op.clone(),
        args: x.args.iter().map(|a| substitute(a, scale)).collect(),
        val: x.val.clone(),
    }
}

fn index() -> &'static HashMap<&'static str, usize> {
    static INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    INDEX.get_or_init(|| SHAPES.iter().enumerate().map(|(i, s)| (s.0, i)).collect())
}

/// Decoded plane templates, cached per shape so a table is parsed at most once.
fn templates(i: usize) -> Result<&'static Vec<[Expr; 4]>> {
    static CACHE: OnceLock<Vec<OnceLock<Vec<[Expr; 4]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| SHAPES.iter().map(|_| OnceLock::new()).collect());
    if let Some(v) = cache[i].get() {
        return Ok(v);
    }
    let mut dec = Decoder {
        bytes: SHAPES[i].2.as_bytes(),
        i: 0,
    };
    let mut planes: Vec<[Expr; 4]> = Vec::new();
    while dec.i < dec.bytes.len() {
        planes.push([dec.expr()?, dec.expr()?, dec.expr()?, dec.expr()?]);
    }
    let _ = cache[i].set(planes);
    Ok(cache[i].get().expect("just populated"))
}

/// The face planes of polyhedron `code`, scaled by `scale`.
///
/// Expand a polyhedron code into its face planes, erroring for
/// an unknown code.
pub fn polyhedron(code: &str, scale: &Expr) -> Result<Vec<Shape>> {
    let i = *index()
        .get(code)
        .ok_or_else(|| Error::Other(format!("unknown polyhedron: '{code}'")))?;
    Ok(templates(i)?
        .iter()
        .map(|[a, b, c, d]| Shape::Plane {
            a: substitute(a, scale),
            b: substitute(b, scale),
            c: substitute(c, scale),
            d: substitute(d, scale),
        })
        .collect())
}

/// Every supported `(code, name)` pair, in table order.
pub fn shapes() -> impl Iterator<Item = (&'static str, &'static str)> {
    SHAPES.iter().map(|s| (s.0, s.1))
}

/// Whether `code` names a supported polyhedron.
pub fn has(code: &str) -> bool {
    index().contains_key(code)
}
