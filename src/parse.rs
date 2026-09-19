//! Expression and recipe parsing.
//!
//! The grammar:
//!
//! ```text
//! E -> E + T | E - T | T
//! T -> T * F | T / F | F
//! F -> - F | P ^ F | P
//! P -> <integer> | ( E ) | <function> ( E ) | <constant>
//! ```

use crate::error::{Error, Result};
use crate::exact::{promote, qq_nothing, root, AlgebraicNumber};
use crate::num::fraction::Fraction;
use crate::num::int::Int;
use crate::num::ring::{power, Elem};

#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub op: String,
    pub args: Vec<Expr>,
    pub val: Option<Fraction>,
}

impl Expr {
    pub fn op(op: &str, args: Vec<Expr>) -> Expr {
        Expr {
            op: op.to_string(),
            args,
            val: None,
        }
    }
    pub fn num(val: Fraction) -> Expr {
        Expr {
            op: "num".to_string(),
            args: Vec::new(),
            val: Some(val),
        }
    }
}

// A `Plane` carries four expressions where a `Polyhedron` carries one, so the
// variants differ in size by a few hundred bytes. Boxing the larger one would
// put an indirection in front of every coefficient for no benefit: a recipe
// holds a handful of shapes, read once when the puzzle is built.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    Polyhedron { name: String, d: Expr },
    Plane { a: Expr, b: Expr, c: Expr, d: Expr },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PuzzleRecipe {
    pub shell: Vec<Shape>,
    pub cuts: Vec<Shape>,
}

/// Evaluate an expression into an algebraic number.
pub fn eval_expr(x: &Expr) -> Result<AlgebraicNumber> {
    let mut args: Vec<AlgebraicNumber> = Vec::with_capacity(x.args.len());
    for a in &x.args {
        args.push(eval_expr(a)?);
    }

    if x.op == "^" {
        let base = args[0].clone();
        if args[1].poly.degree != 0 {
            return Err(Error::Parse("exponent must be a fraction".into()));
        }
        let y = args[1].poly.coeffs[0].clone();
        if args[0].sign()? < 0 && !y.d.is_one() {
            return Err(Error::Parse(
                "exponent of negative number must be an integer".into(),
            ));
        }
        let mut z = power(&base.field, &base, &y.n)?;
        if !y.d.is_one() {
            z = root(&z, &y.d)?;
        }
        return Ok(z);
    }

    if args.len() == 2 {
        promote(&mut args)?;
    }

    match x.op.as_str() {
        "+" => Elem::add(&args[0], &args[1]),
        "-" => Elem::sub(&args[0], &args[1]),
        "*" => Elem::mul(&args[0], &args[1]),
        "/" => Elem::div(&args[0], &args[1]),
        "neg" => Ok(Elem::neg(&args[0])),
        "num" => {
            let v = x
                .val
                .clone()
                .ok_or_else(|| Error::Parse("num without a value".into()))?;
            qq_nothing().from_vector(vec![v])
        },
        "sqrt" => {
            if x.args.len() != 1 {
                return Err(Error::Parse("sqrt must have an argument".into()));
            }
            root(&args[0], &Int::from_i64(2))
        },
        "cbrt" => {
            if x.args.len() != 1 {
                return Err(Error::Parse("cbrt must have an argument".into()));
            }
            root(&args[0], &Int::from_i64(3))
        },
        other => Err(Error::Parse(format!("unknown operation '{other}'"))),
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Sym(char),
    Ident(String),
    Num(Fraction),
}

/// Tokenize, by the rule:
/// `\s*(?:([+\-*/^()])|([A-Za-z_][A-Za-z_0-9]*)|(\d+)(\.\d+)?)\s*`
fn tokenize(s: &str) -> Result<Vec<Token>> {
    let cs: Vec<char> = s.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0usize;
    while i < cs.len() {
        let start = i;
        while i < cs.len() && cs[i].is_whitespace() {
            i += 1;
        }
        if i >= cs.len() {
            // Trailing whitespace alone does not match the sticky pattern.
            return Err(Error::Parse(format!(
                "Unexpected character <code>{}</code>",
                cs[start]
            )));
        }
        let c = cs[i];
        if matches!(c, '+' | '-' | '*' | '/' | '^' | '(' | ')') {
            tokens.push(Token::Sym(c));
            i += 1;
        } else if c.is_ascii_alphabetic() || c == '_' {
            let b = i;
            while i < cs.len() && (cs[i].is_ascii_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            tokens.push(Token::Ident(cs[b..i].iter().collect()));
        } else if c.is_ascii_digit() {
            let b = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            let int_part: String = cs[b..i].iter().collect();
            let mut n: i64 = int_part.parse().map_err(|_| {
                Error::Parse(format!("Number out of range: <code>{int_part}</code>"))
            })?;
            let mut d: i64 = 1;
            // Optional fractional part `(\.\d+)`.
            if i < cs.len() && cs[i] == '.' && i + 1 < cs.len() && cs[i + 1].is_ascii_digit() {
                let fb = i + 1;
                i += 1;
                while i < cs.len() && cs[i].is_ascii_digit() {
                    i += 1;
                }
                let frac: String = cs[fb..i].iter().collect();
                d = 10i64.pow(frac.len() as u32);
                n *= d;
                n += frac.parse::<i64>().map_err(|_| {
                    Error::Parse(format!("Number out of range: <code>{frac}</code>"))
                })?;
            }
            tokens.push(Token::Num(Fraction::of(n, d)));
        } else {
            return Err(Error::Parse(format!(
                "Unexpected character <code>{c}</code>"
            )));
        }
        while i < cs.len() && cs[i].is_whitespace() {
            i += 1;
        }
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    i: usize,
}

impl Parser {
    fn n(&self) -> usize {
        self.tokens.len()
    }

    fn parse_start(&mut self) -> Result<Expr> {
        let mut x = self.parse_term()?;
        while self.i < self.n() && matches!(self.tokens[self.i], Token::Sym('+' | '-')) {
            let Token::Sym(op) = self.tokens[self.i] else {
                unreachable!("the loop condition just matched a symbol")
            };
            self.i += 1;
            let y = self.parse_term()?;
            x = Expr::op(&op.to_string(), vec![x, y]);
        }
        Ok(x)
    }

    fn parse_term(&mut self) -> Result<Expr> {
        let mut x = self.parse_factor()?;
        while self.i < self.n() && matches!(self.tokens[self.i], Token::Sym('*' | '/')) {
            let Token::Sym(op) = self.tokens[self.i] else {
                unreachable!("the loop condition just matched a symbol")
            };
            self.i += 1;
            let y = self.parse_factor()?;
            x = Expr::op(&op.to_string(), vec![x, y]);
        }
        Ok(x)
    }

    fn parse_factor(&mut self) -> Result<Expr> {
        if self.i == self.n() {
            return Err(Error::Parse("Unexpected end of expression".into()));
        }
        if self.tokens[self.i] == Token::Sym('-') {
            self.i += 1;
            let x = self.parse_factor()?;
            return Ok(Expr::op("neg", vec![x]));
        }
        let x = self.parse_primary()?;
        if self.i < self.n() && self.tokens[self.i] == Token::Sym('^') {
            self.i += 1;
            let y = self.parse_factor()?;
            Ok(Expr::op("^", vec![x, y]))
        } else {
            Ok(x)
        }
    }

    fn parse_primary(&mut self) -> Result<Expr> {
        if self.i == self.n() {
            return Err(Error::Parse("Unexpected end of expression".into()));
        }
        match self.tokens[self.i].clone() {
            Token::Sym('(') => {
                self.i += 1;
                let x = self.parse_start()?;
                self.expect(')')?;
                Ok(x)
            },
            Token::Num(f) => {
                self.i += 1;
                Ok(Expr::num(f))
            },
            Token::Ident(id) => {
                self.i += 1;
                if self.i < self.n() && self.tokens[self.i] == Token::Sym('(') {
                    self.i += 1;
                    let x = self.parse_start()?;
                    self.expect(')')?;
                    Ok(Expr::op(&id, vec![x]))
                } else {
                    Ok(Expr::op(&id, vec![]))
                }
            },
            t @ Token::Sym(_) => Err(Error::Parse(format!(
                "Unexpected <code>{}</code>",
                token_str(&t)
            ))),
        }
    }

    fn expect(&mut self, tok: char) -> Result<()> {
        if self.i >= self.n() {
            Err(Error::Parse(format!(
                "Expected <code>{tok}</code>, but reached end of expression"
            )))
        } else if self.tokens[self.i] != Token::Sym(tok) {
            Err(Error::Parse(format!(
                "Expected <code>{}</code>, but found <code>{}</code>",
                tok,
                token_str(&self.tokens[self.i])
            )))
        } else {
            self.i += 1;
            Ok(())
        }
    }
}

fn token_str(t: &Token) -> String {
    match t {
        Token::Sym(c) => c.to_string(),
        // Identifiers carry a `$` sentinel, which is how a recipe distinguishes a name from a number.
        Token::Ident(s) => format!("${s}"),
        Token::Num(f) => f.to_string(),
    }
}

pub fn parse_expr(s: &str) -> Result<Expr> {
    let tokens = tokenize(s)?;
    let mut p = Parser { tokens, i: 0 };
    let x = p.parse_start()?;
    if p.i < p.n() {
        return Err(Error::Parse(format!(
            "Expected end of string, but found <code>{}</code>",
            token_str(&p.tokens[p.i])
        )));
    }
    Ok(x)
}

/// Parse one shape, such as `"C$1"` or `"1,0,0$1/3"`.
///
/// # Errors
///
/// If the text is not a shape.
pub fn parse_shape(s: &str) -> Result<Shape> {
    let parts: Vec<&str> = s.split('$').collect();
    if parts.len() != 2 {
        return Err(Error::Parse("expected $".into()));
    }
    let d = parse_expr(parts[1])?;
    let head = parts[0];
    // Every comma-separated field is parsed before deciding what the shape is, so a malformed field is an error even for a polyhedron.
    let coeffs: Vec<Expr> = head
        .split(',')
        .map(parse_expr)
        .collect::<Result<Vec<_>>>()?;
    if coeffs.len() == 3 {
        Ok(Shape::Plane {
            a: coeffs[0].clone(),
            b: coeffs[1].clone(),
            c: coeffs[2].clone(),
            d,
        })
    } else {
        Ok(Shape::Polyhedron {
            name: head.to_string(),
            d,
        })
    }
}

/// `decodeURIComponent`
fn decode_uri_component(s: &str) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len() {
                return Err(Error::Parse("URI malformed".into()));
            }
            let hex = core::str::from_utf8(&bytes[i + 1..i + 3])
                .map_err(|_| Error::Parse("URI malformed".into()))?;
            let v =
                u8::from_str_radix(hex, 16).map_err(|_| Error::Parse("URI malformed".into()))?;
            out.push(v);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| Error::Parse("URI malformed".into()))
}

/// Parse a `?shell=...&cut=...` query string into a recipe.
pub fn parse_query(s: &str) -> Result<PuzzleRecipe> {
    let mut ret = PuzzleRecipe::default();
    let s = decode_uri_component(s)?;
    if s.is_empty() {
        return Ok(ret);
    }
    if !s.starts_with('?') {
        return Err(Error::Parse("expected ?".into()));
    }
    let s = &s[1..];
    for kvstring in s.split('&') {
        let kv: Vec<&str> = kvstring.split('=').collect();
        if kv.len() != 2 {
            return Err(Error::Parse("expected exactly one =".into()));
        }
        match kv[0] {
            "shell" => ret.shell.push(parse_shape(kv[1])?),
            "cut" => ret.cuts.push(parse_shape(kv[1])?),
            k => eprintln!("warning: ignoring unknown key {k}"),
        }
    }
    Ok(ret)
}

/// Render an expression back to source text, inserting the minimum
/// parentheses needed at the given precedence.
pub fn generate_expr(x: &Expr, prec: i32) -> Result<String> {
    let s = match x.op.as_str() {
        "+" | "-" => {
            let s = format!(
                "{}{}{}",
                generate_expr(&x.args[0], 6)?,
                x.op,
                generate_expr(&x.args[1], 5)?
            );
            if prec < 6 {
                format!("({s})")
            } else {
                s
            }
        },
        "*" | "/" => {
            let s = format!(
                "{}{}{}",
                generate_expr(&x.args[0], 4)?,
                x.op,
                generate_expr(&x.args[1], 3)?
            );
            if prec < 4 {
                format!("({s})")
            } else {
                s
            }
        },
        "^" => {
            let s = format!(
                "{}{}{}",
                generate_expr(&x.args[0], 1)?,
                x.op,
                generate_expr(&x.args[1], 2)?
            );
            if prec < 2 {
                format!("({s})")
            } else {
                s
            }
        },
        "neg" => {
            let s = format!("-{}", generate_expr(&x.args[0], 2)?);
            if prec < 2 {
                format!("({s})")
            } else {
                s
            }
        },
        "num" => match &x.val {
            Some(v) => v.to_string(),
            None => "undefined".to_string(),
        },
        "sqrt" => format!("sqrt({})", generate_expr(&x.args[0], 10)?),
        other => {
            return Err(Error::Parse(format!(
                "generateExpr: invalid operation '{other}'"
            )));
        },
    };
    Ok(s)
}

/// Render a shape back to the text `parse_shape` reads.
///
/// # Errors
///
/// If the shape holds an expression that cannot be written.
pub fn generate_shape(s: &Shape) -> Result<String> {
    match s {
        Shape::Polyhedron { name, d } => Ok(format!("{}${}", name, generate_expr(d, 10)?)),
        Shape::Plane { a, b, c, d } => Ok(format!(
            "{},{},{}${}",
            generate_expr(a, 10)?,
            generate_expr(b, 10)?,
            generate_expr(c, 10)?,
            generate_expr(d, 10)?
        )),
    }
}

/// Render a recipe back to a `?shell=...&cut=...` query string.
pub fn generate_query(p: &PuzzleRecipe) -> Result<String> {
    let mut ret: Vec<String> = Vec::new();
    for s in &p.shell {
        ret.push(format!("shell={}", generate_shape(s)?));
    }
    for s in &p.cuts {
        ret.push(format!("cut={}", generate_shape(s)?));
    }
    Ok(format!("?{}", ret.join("&")))
}
