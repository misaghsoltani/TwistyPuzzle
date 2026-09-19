//! Ring traits and the concrete `Integer` / `IntegerMod` rings.

use core::cmp::Ordering;
use core::fmt;

use super::int::Int;
use crate::error::{Error, Result};

/// `interface RingElement<E>`.
///
/// Both the in-place and the returning form of each operation are required:
/// they are *not* interchangeable. `AlgebraicNumber::iadd` skips the reduction
/// modulo the field polynomial that `add` performs, and callers depend on
/// that difference. See `SEMANTICS.md` §6.
pub trait Elem: Clone + Sized + fmt::Display {
    /// Set to `true` by `Polynomial`, which `Polynomial::to_string`
    /// parenthesizes when it appears as a coefficient.
    const IS_POLY: bool = false;

    fn equals(&self, y: &Self) -> bool;

    /// Whether this is the additive identity of its ring.
    ///
    /// Equivalent to `self.equals(&ring.zero())`, but needs no ring and no
    /// temporary: `Polynomial::trim` asks this after every operation.
    fn is_zero(&self) -> bool;

    fn iadd(&mut self, y: &Self) -> Result<()>;
    fn add(&self, y: &Self) -> Result<Self>;
    fn isub(&mut self, y: &Self) -> Result<()>;
    fn sub(&self, y: &Self) -> Result<Self>;
    fn neg(&self) -> Self;
    fn imul(&mut self, y: &Self) -> Result<()>;
    fn mul(&self, y: &Self) -> Result<Self>;
    /// Field division, which may fail in a ring that is not a field.
    fn idiv(&mut self, y: &Self) -> Result<()>;
    fn div(&self, y: &Self) -> Result<Self>;
    fn inv(&self) -> Result<Self>;
}

/// `interface Euclidean<E>`
pub trait Euclidean: Elem {
    fn euclidean(&self) -> Int;
    fn divmod(&self, y: &Self) -> Result<(Self, Self)>;
    fn floordiv(&self, y: &Self) -> Result<Self> {
        Ok(self.divmod(y)?.0)
    }
    fn modulo(&self, y: &Self) -> Result<Self> {
        Ok(self.divmod(y)?.1)
    }
}

/// `interface Ring<E>`
pub trait RingOps<E>: Clone {
    fn zero(&self) -> E;
    fn one(&self) -> E;
    fn from_int(&self, n: i64) -> E;
}

/* -------------------------------------------------------------------------- */
/*  Integer / ZZ                                                              */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug)]
pub struct Integer {
    pub n: Int,
}

impl Integer {
    pub fn new(n: Int) -> Integer {
        Integer { n }
    }
    pub fn from_i64(n: i64) -> Integer {
        Integer {
            n: Int::from_i64(n),
        }
    }
    pub fn to_number(&self) -> f64 {
        self.n.to_f64()
    }
    pub fn sign(&self) -> i32 {
        self.n.sign()
    }
    pub fn abs(&self) -> Integer {
        Integer { n: self.n.abs() }
    }
    pub fn compare(&self, y: &Integer) -> i32 {
        match self.n.cmp(&y.n) {
            Ordering::Equal => 0,
            Ordering::Less => -1,
            Ordering::Greater => 1,
        }
    }
}

impl Elem for Integer {
    #[inline]
    fn is_zero(&self) -> bool {
        self.n.is_zero()
    }

    fn equals(&self, y: &Integer) -> bool {
        self.n.eq(&y.n)
    }
    fn iadd(&mut self, y: &Integer) -> Result<()> {
        self.n = self.n.add(&y.n);
        Ok(())
    }
    fn add(&self, y: &Integer) -> Result<Integer> {
        Ok(Integer {
            n: self.n.add(&y.n),
        })
    }
    fn isub(&mut self, y: &Integer) -> Result<()> {
        self.n = self.n.sub(&y.n);
        Ok(())
    }
    fn sub(&self, y: &Integer) -> Result<Integer> {
        Ok(Integer {
            n: self.n.sub(&y.n),
        })
    }
    fn neg(&self) -> Integer {
        Integer { n: self.n.neg() }
    }
    fn imul(&mut self, y: &Integer) -> Result<()> {
        self.n = self.n.mul(&y.n);
        Ok(())
    }
    fn mul(&self, y: &Integer) -> Result<Integer> {
        Ok(Integer {
            n: self.n.mul(&y.n),
        })
    }
    fn inv(&self) -> Result<Integer> {
        if self.n.eq(&Int::from_i64(1)) || self.n.eq(&Int::from_i64(-1)) {
            Ok(self.clone())
        } else {
            Err(Error::Division(
                "Multiplicative inverse does not exist".into(),
            ))
        }
    }
    fn idiv(&mut self, y: &Integer) -> Result<()> {
        if y.n.is_zero() {
            // The range check comes before the divisibility test, so a zero
            // modulus is reported as a range error rather than a division one.
            return Err(Error::Range("Division by zero".into()));
        }
        if self.n.rem(&y.n).is_zero() {
            self.n = self.n.div(&y.n);
            Ok(())
        } else {
            Err(Error::Division("Not divisible".into()))
        }
    }
    fn div(&self, y: &Integer) -> Result<Integer> {
        let mut c = self.clone();
        Elem::idiv(&mut c, y)?;
        Ok(c)
    }
}

impl Euclidean for Integer {
    fn euclidean(&self) -> Int {
        self.n.abs()
    }
    /// Floored division: a non-positive divisor is rejected, and then
    /// corrects a negative remainder.
    fn divmod(&self, y: &Integer) -> Result<(Integer, Integer)> {
        if y.n.sign() <= 0 {
            return Err(Error::Division("Division by zero".into()));
        }
        let (mut q, mut r) = self.n.div_rem(&y.n);
        if r.sign() < 0 {
            q = q.sub(&Int::from_i64(1));
            r = r.add(&y.n);
        }
        Ok((Integer { n: q }, Integer { n: r }))
    }
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.n)
    }
}

impl PartialEq for Integer {
    fn eq(&self, other: &Self) -> bool {
        self.n.eq(&other.n)
    }
}
impl Eq for Integer {}

#[derive(Clone, Copy, Debug, Default)]
pub struct ZZ;

impl RingOps<Integer> for ZZ {
    fn zero(&self) -> Integer {
        Integer::from_i64(0)
    }
    fn one(&self) -> Integer {
        Integer::from_i64(1)
    }
    fn from_int(&self, n: i64) -> Integer {
        Integer::from_i64(n)
    }
}

/* -------------------------------------------------------------------------- */
/*  IntegerMod / Z/mZ                                                         */
/* -------------------------------------------------------------------------- */

#[derive(Clone, Debug)]
pub struct IntegerMod {
    pub n: Int,
    pub m: Int,
}

impl IntegerMod {
    pub fn new(n: Int, m: Int) -> IntegerMod {
        let mut x = IntegerMod { n, m };
        x.reduce();
        x
    }

    fn reduce(&mut self) {
        // ((n % m) + m) % m
        self.n = self.n.rem(&self.m).add(&self.m).rem(&self.m);
    }

    fn check(&self, y: &IntegerMod) -> Result<()> {
        if self.m.eq(&y.m) {
            Ok(())
        } else {
            Err(Error::Type("Numbers have different moduli".into()))
        }
    }
}

impl Elem for IntegerMod {
    /// `n` is kept in `[0, m)` by `reduce`, so a zero residue is a zero `n`.
    #[inline]
    fn is_zero(&self) -> bool {
        self.n.is_zero()
    }

    fn equals(&self, y: &IntegerMod) -> bool {
        // `check` would error. Equality is only ever asked of
        // same-modulus values, so mismatches degrade to `false`.
        self.m.eq(&y.m) && self.n.eq(&y.n)
    }
    fn iadd(&mut self, y: &IntegerMod) -> Result<()> {
        self.check(y)?;
        self.n = self.n.add(&y.n);
        self.reduce();
        Ok(())
    }
    fn add(&self, y: &IntegerMod) -> Result<IntegerMod> {
        let mut c = self.clone();
        Elem::iadd(&mut c, y)?;
        Ok(c)
    }
    fn isub(&mut self, y: &IntegerMod) -> Result<()> {
        self.check(y)?;
        self.n = self.n.sub(&y.n);
        self.reduce();
        Ok(())
    }
    fn sub(&self, y: &IntegerMod) -> Result<IntegerMod> {
        let mut c = self.clone();
        Elem::isub(&mut c, y)?;
        Ok(c)
    }
    fn neg(&self) -> IntegerMod {
        IntegerMod::new(self.n.neg(), self.m.clone())
    }
    fn imul(&mut self, y: &IntegerMod) -> Result<()> {
        self.check(y)?;
        self.n = self.n.mul(&y.n);
        self.reduce();
        Ok(())
    }
    fn mul(&self, y: &IntegerMod) -> Result<IntegerMod> {
        let mut c = self.clone();
        Elem::imul(&mut c, y)?;
        Ok(c)
    }
    fn idiv(&mut self, y: &IntegerMod) -> Result<()> {
        self.check(y)?;
        let (r, s, _t) = Int::extended_gcd(&y.n, &self.m);
        if r.is_zero() {
            return Err(Error::Division("Not divisible".into()));
        }
        if self.n.rem(&r).is_zero() {
            self.n = s.mul(&self.n.div(&r));
            self.reduce();
            Ok(())
        } else {
            Err(Error::Division("Not divisible".into()))
        }
    }
    fn div(&self, y: &IntegerMod) -> Result<IntegerMod> {
        let mut c = self.clone();
        Elem::idiv(&mut c, y)?;
        Ok(c)
    }
    fn inv(&self) -> Result<IntegerMod> {
        let (r, s, _t) = Int::extended_gcd(&self.n, &self.m);
        if r.is_one() {
            Ok(IntegerMod::new(s, self.m.clone()))
        } else {
            Err(Error::Division(
                "Multiplicative inverse does not exist".into(),
            ))
        }
    }
}

impl fmt::Display for IntegerMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.n)
    }
}

impl PartialEq for IntegerMod {
    fn eq(&self, other: &Self) -> bool {
        Elem::equals(self, other)
    }
}
impl Eq for IntegerMod {}

#[derive(Clone, Debug)]
pub struct IntegersMod {
    pub m: Int,
}

impl IntegersMod {
    pub fn new(m: Int) -> IntegersMod {
        IntegersMod { m }
    }
}

impl RingOps<IntegerMod> for IntegersMod {
    fn zero(&self) -> IntegerMod {
        IntegerMod::new(Int::from_i64(0), self.m.clone())
    }
    fn one(&self) -> IntegerMod {
        IntegerMod::new(Int::from_i64(1), self.m.clone())
    }
    fn from_int(&self, n: i64) -> IntegerMod {
        IntegerMod::new(Int::from_i64(n), self.m.clone())
    }
}

/* -------------------------------------------------------------------------- */
/*  Generic ring algorithms                                                   */
/* -------------------------------------------------------------------------- */

/// `power`: repeated squaring (Cohen, Algorithm 1.2.1).
pub fn power<E: Elem, R: RingOps<E>>(k: &R, g: &E, n: &Int) -> Result<E> {
    if n.sign() < 0 {
        return Err(Error::Range(String::new()));
    }
    let two = Int::from_i64(2);
    let one = Int::from_i64(1);
    let mut y = k.one();
    let mut z = g.clone();
    let mut m = n.clone();
    while m.sign() > 0 {
        if m.rem(&two).is_one() {
            y = y.mul(&z)?;
            m = m.sub(&one);
        } else {
            z = z.mul(&z)?;
            m = m.div(&two);
        }
    }
    Ok(y)
}

/// `power_mod`
pub fn power_mod<E: Euclidean, R: RingOps<E>>(k: &R, g: &E, n: &Int, q: &E) -> Result<E> {
    let two = Int::from_i64(2);
    let one = Int::from_i64(1);
    let mut y = k.one();
    let mut z = g.clone();
    let mut m = n.clone();
    while m.sign() > 0 {
        if m.rem(&two).is_one() {
            y = y.mul(&z)?.modulo(q)?;
            m = m.sub(&one);
        } else {
            z = z.mul(&z)?.modulo(q)?;
            m = m.div(&two);
        }
    }
    Ok(y)
}

/// `product`
pub fn product<E: Elem, R: RingOps<E>>(k: &R, xs: &[E]) -> Result<E> {
    let mut prod = k.one();
    for x in xs {
        prod = prod.mul(x)?;
    }
    Ok(prod)
}

/// `gcd` over a Euclidean domain.
pub fn gcd<E: Euclidean, R: RingOps<E>>(k: &R, x: &E, y: &E) -> Result<E> {
    let (mut x, mut y) = if x.euclidean().cmp(&y.euclidean()) == Ordering::Less {
        (y.clone(), x.clone())
    } else {
        (x.clone(), y.clone())
    };
    let zero = k.zero();
    while !y.equals(&zero) {
        let (_, r) = x.divmod(&y)?;
        x = y;
        y = r;
    }
    Ok(x)
}

/// `extended_gcd`: returns `(r, s, t)` with `r = x*s + y*t`.
pub fn extended_gcd<E: Euclidean, R: RingOps<E>>(k: &R, x: &E, y: &E) -> Result<(E, E, E)> {
    let mut swap = false;
    let (x, y) = if x.euclidean().cmp(&y.euclidean()) == Ordering::Less {
        swap = true;
        (y.clone(), x.clone())
    } else {
        (x.clone(), y.clone())
    };

    let (mut r0, mut r1) = (x, y);
    let (mut s0, mut s1) = (k.one(), k.zero());
    let (mut t0, mut t1) = (k.zero(), k.one());

    let zero = k.zero();
    while !r1.equals(&zero) {
        let (q, r2) = r0.divmod(&r1)?;
        r0 = r1;
        r1 = r2;
        let s2 = s0.sub(&q.mul(&s1)?)?;
        s0 = s1;
        s1 = s2;
        let t2 = t0.sub(&q.mul(&t1)?)?;
        t0 = t1;
        t1 = t2;
    }

    if swap {
        core::mem::swap(&mut s0, &mut t0);
    }
    Ok((r0, s0, t0))
}
