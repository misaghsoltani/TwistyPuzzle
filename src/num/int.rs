//! Arbitrary-precision integers with an inline fast path.
//!
//! Nearly every value that flows through the exact-arithmetic core (fraction
//! numerators, polynomial coefficients, and modular residues) is small, so this
//! type keeps values in an `i64` for as long as it can and promotes to
//! [`BigInt`] only on overflow. The two representations are observationally
//! identical: division truncates toward zero and remainder takes the sign of
//! the dividend, whichever is in use.
//!
//! Building with the `bigint-only` feature disables the fast path entirely, so
//! the test suite can prove that claim rather than assert it.

use core::cmp::Ordering;
use core::fmt;
use std::borrow::Cow;
use std::sync::Arc;

use num_bigint::BigInt;
use num_integer::Integer as _;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// An integer that is either inline or heap-allocated.
///
/// The wide variant is behind an [`Arc`] for two reasons. It keeps the enum at
/// two words instead of five, which matters because `Fraction` holds two of
/// these and polynomials hold vectors of `Fraction`, and it makes cloning a
/// wide value a reference-count bump rather than a copy of its digits.
/// Sharing is safe without any copy-on-write dance because `Int` is immutable:
/// every operation here returns a new value and none mutates in place.
#[derive(Clone, Debug)]
pub enum Int {
    Small(i64),
    Big(Arc<BigInt>),
}

#[inline]
fn wrap_small(n: i64) -> Int {
    #[cfg(feature = "bigint-only")]
    {
        Int::Big(Arc::new(BigInt::from(n)))
    }
    #[cfg(not(feature = "bigint-only"))]
    {
        Int::Small(n)
    }
}

/// Stein's binary GCD on two non-negative values.
#[inline]
fn binary_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            break;
        }
    }
    a << shift
}

/// Demote a `BigInt` back to the inline representation when it fits.
#[inline]
fn shrink(b: BigInt) -> Int {
    #[cfg(feature = "bigint-only")]
    {
        Int::Big(Arc::new(b))
    }
    #[cfg(not(feature = "bigint-only"))]
    {
        match b.to_i64() {
            Some(n) => Int::Small(n),
            None => Int::Big(Arc::new(b)),
        }
    }
}

impl Int {
    pub const ZERO: Int = Int::Small(0);

    #[inline]
    pub fn from_i64(n: i64) -> Int {
        wrap_small(n)
    }

    #[inline]
    pub fn from_big(b: BigInt) -> Int {
        shrink(b)
    }

    /// The value as a `BigInt`, borrowed when it already is one.
    ///
    /// Every wide operation below goes through this rather than
    /// [`to_bigint`](Self::to_bigint), so promoting an inline value costs one
    /// allocation and an already-wide value costs none.
    #[inline]
    fn big(&self) -> Cow<'_, BigInt> {
        match self {
            Int::Small(n) => Cow::Owned(BigInt::from(*n)),
            Int::Big(b) => Cow::Borrowed(b),
        }
    }

    pub fn parse(s: &str) -> Option<Int> {
        s.parse::<i64>()
            .map(wrap_small)
            .ok()
            .or_else(|| s.parse::<BigInt>().ok().map(shrink))
    }

    #[inline]
    pub fn to_bigint(&self) -> BigInt {
        match self {
            Int::Small(n) => BigInt::from(*n),
            Int::Big(b) => (**b).clone(),
        }
    }

    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Int::Small(n) => Some(*n),
            Int::Big(b) => b.to_i64(),
        }
    }

    /// Exact when `|self| < 2^53`, which is the only case any caller relies
    /// on for reproducible output (see `Fraction::to_number`).
    #[inline]
    pub fn to_f64(&self) -> f64 {
        match self {
            Int::Small(n) => *n as f64,
            Int::Big(b) => b.to_f64().unwrap_or(f64::NAN),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        match self {
            Int::Small(n) => *n == 0,
            Int::Big(b) => b.is_zero(),
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        match self {
            Int::Small(n) => *n == 1,
            Int::Big(b) => b.is_one(),
        }
    }

    /// `bigutils.sign`
    #[inline]
    pub fn sign(&self) -> i32 {
        match self {
            Int::Small(n) => i32::from(*n > 0) - i32::from(*n < 0),
            Int::Big(b) => match b.sign() {
                num_bigint::Sign::Plus => 1,
                num_bigint::Sign::NoSign => 0,
                num_bigint::Sign::Minus => -1,
            },
        }
    }

    /// `bigutils.abs`
    #[inline]
    pub fn abs(&self) -> Int {
        match self {
            Int::Small(n) => match n.checked_abs() {
                Some(v) => wrap_small(v),
                None => shrink(BigInt::from(*n).abs()),
            },
            Int::Big(b) => shrink(b.abs()),
        }
    }

    #[inline]
    pub fn neg(&self) -> Int {
        match self {
            Int::Small(n) => match n.checked_neg() {
                Some(v) => wrap_small(v),
                None => shrink(-BigInt::from(*n)),
            },
            Int::Big(b) => shrink(-&**b),
        }
    }

    #[inline]
    pub fn add(&self, y: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if let Some(v) = a.checked_add(*b) {
                return wrap_small(v);
            }
        }
        shrink(&*self.big() + &*y.big())
    }

    #[inline]
    pub fn sub(&self, y: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if let Some(v) = a.checked_sub(*b) {
                return wrap_small(v);
            }
        }
        shrink(&*self.big() - &*y.big())
    }

    #[inline]
    pub fn mul(&self, y: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if let Some(v) = a.checked_mul(*b) {
                return wrap_small(v);
            }
        }
        shrink(&*self.big() * &*y.big())
    }

    /// Truncating division: the quotient rounds toward zero.
    ///
    /// # Panics
    /// Panics on division by zero, though every caller checks first, exactly as the
    /// reference does.
    #[inline]
    pub fn div(&self, y: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if *b != 0 {
                if let Some(v) = a.checked_div(*b) {
                    return wrap_small(v);
                }
            }
        }
        shrink(&*self.big() / &*y.big())
    }

    /// Remainder taking the sign of the dividend.
    #[inline]
    pub fn rem(&self, y: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if *b != 0 {
                if let Some(v) = a.checked_rem(*b) {
                    return wrap_small(v);
                }
            }
        }
        shrink(&*self.big() % &*y.big())
    }

    #[inline]
    pub fn div_rem(&self, y: &Int) -> (Int, Int) {
        if let (Int::Small(a), Int::Small(b)) = (self, y) {
            if *b != 0 && !(*a == i64::MIN && *b == -1) {
                return (wrap_small(a / b), wrap_small(a % b));
            }
        }
        let (q, r) = self.big().div_rem(&y.big());
        (shrink(q), shrink(r))
    }

    /// `x ** n` for `n >= 0`.
    pub fn pow(&self, n: u32) -> Int {
        if let Int::Small(a) = self {
            // Cheap guard: |a|^n only stays in range for small bases/exponents.
            if let Some(v) = a.checked_pow(n) {
                return wrap_small(v);
            }
        }
        shrink(num_traits::Pow::pow(self.to_bigint(), n))
    }

    /// `bigutils.gcd`: Euclid's algorithm on absolute values.
    ///
    /// The inline case uses Stein's binary algorithm instead: the greatest
    /// common divisor of two non-negative integers is unique, so this computes
    /// the same number Euclid's does, using shifts and subtractions in place of
    /// the hardware divide. `Fraction::reduce` calls this more often than any
    /// other routine here, and on this machine a 64-bit `idiv` is an order of
    /// magnitude slower than the shift it replaces.
    pub fn gcd(&self, other: &Int) -> Int {
        if let (Int::Small(a), Int::Small(b)) = (self, other) {
            if let (Some(x), Some(y)) = (a.checked_abs(), b.checked_abs()) {
                // The result never exceeds `max(x, y)`, so it always fits.
                return wrap_small(binary_gcd(x as u64, y as u64) as i64);
            }
        }
        let mut x = self.abs();
        let mut y = other.abs();
        if x.cmp(&y) == Ordering::Less {
            core::mem::swap(&mut x, &mut y);
        }
        while !y.is_zero() {
            let r = x.rem(&y);
            x = y;
            y = r;
        }
        x
    }

    /// `bigutils.extended_gcd`: returns `(r, s, t)` with `r = a*s + b*t`.
    pub fn extended_gcd(a: &Int, b: &Int) -> (Int, Int, Int) {
        let (mut r0, mut r1) = (a.clone(), b.clone());
        let (mut s0, mut s1) = (Int::from_i64(1), Int::from_i64(0));
        let (mut t0, mut t1) = (Int::from_i64(0), Int::from_i64(1));
        while !r1.is_zero() {
            let (q, r2) = r0.div_rem(&r1);
            r0 = r1;
            r1 = r2;
            let s2 = s0.sub(&q.mul(&s1));
            s0 = s1;
            s1 = s2;
            let t2 = t0.sub(&q.mul(&t1));
            t0 = t1;
            t1 = t2;
        }
        (r0, s0, t0)
    }

    /// `bigutils.sqrt`: Newton iteration yielding an *upper bound* on √n.
    pub fn sqrt_upper(&self) -> Int {
        let one = Int::from_i64(1);
        let two = Int::from_i64(2);
        let mut s_prev = Int::from_i64(-1);
        let mut s = self.clone();
        while s.sign() > 0 && s.cmp(&s_prev) != Ordering::Equal {
            s_prev = s.clone();
            // s = (s*s + n - 1)/(2*s) + 1
            s = s.mul(&s).add(self).sub(&one).div(&two.mul(&s)).add(&one);
        }
        s
    }

    /// `bigutils.factorial`
    pub fn factorial(n: &Int) -> Int {
        let one = Int::from_i64(1);
        let mut nfac = one.clone();
        let mut i = Int::from_i64(2);
        while i.cmp(n) != Ordering::Greater {
            nfac = nfac.mul(&i);
            i = i.add(&one);
        }
        nfac
    }

    /// `bigutils.comb`
    pub fn comb(n: &Int, k: &Int) -> Int {
        let nf = Int::factorial(n);
        let kf = Int::factorial(k);
        let nkf = Int::factorial(&n.sub(k));
        nf.div(&kf.mul(&nkf))
    }

    #[inline]
    pub fn cmp(&self, other: &Int) -> Ordering {
        match (self, other) {
            (Int::Small(a), Int::Small(b)) => a.cmp(b),
            (Int::Big(a), Int::Big(b)) => a.cmp(b),
            _ => self.big().cmp(&other.big()),
        }
    }

    #[inline]
    pub fn eq(&self, other: &Int) -> bool {
        match (self, other) {
            (Int::Small(a), Int::Small(b)) => a == b,
            (Int::Big(a), Int::Big(b)) => a == b,
            _ => self.big() == other.big(),
        }
    }

    /// Number of bits in `|self|`, used by the deterministic RNG.
    pub fn bits(&self) -> u64 {
        match self {
            Int::Small(n) => u64::from(64 - n.unsigned_abs().leading_zeros()),
            Int::Big(b) => b.bits(),
        }
    }
}

impl PartialEq for Int {
    fn eq(&self, other: &Self) -> bool {
        Int::eq(self, other)
    }
}
impl Eq for Int {}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}
impl Ord for Int {
    fn cmp(&self, other: &Self) -> Ordering {
        Int::cmp(self, other)
    }
}

impl core::hash::Hash for Int {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        // Small and Big must hash alike for equal values.
        match self {
            Int::Small(n) => n.hash(state),
            Int::Big(b) => match b.to_i64() {
                Some(n) => n.hash(state),
                None => b.hash(state),
            },
        }
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Int::Small(n) => write!(f, "{n}"),
            Int::Big(b) => write!(f, "{b}"),
        }
    }
}

impl From<i64> for Int {
    fn from(n: i64) -> Int {
        Int::from_i64(n)
    }
}
impl From<i32> for Int {
    fn from(n: i32) -> Int {
        Int::from_i64(i64::from(n))
    }
}
impl From<BigInt> for Int {
    fn from(b: BigInt) -> Int {
        Int::from_big(b)
    }
}

/// `2**k` computed exactly, including
/// gradual underflow to subnormals, flush to zero below `2^-1074`, and
/// overflow to infinity above `2^1023`.
pub fn pow2(k: i32) -> f64 {
    if k > 1023 {
        f64::INFINITY
    } else if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else if k >= -1074 {
        f64::from_bits(1u64 << (k + 1074))
    } else {
        0.0
    }
}

/// Endless prime sequence, mirroring `bigutils.primes`.
///
/// The trial-division schedule (including its early `break` once
/// `prev*prev > cur`) is preserved so the primes are produced in the same
/// order they are consumed.
pub struct Primes {
    prevs: Vec<Int>,
    cur: Int,
}

impl Primes {
    pub fn new() -> Primes {
        Primes {
            prevs: Vec::new(),
            cur: Int::from_i64(2),
        }
    }
}

impl Default for Primes {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for Primes {
    type Item = Int;
    fn next(&mut self) -> Option<Int> {
        let one = Int::from_i64(1);
        loop {
            let cur = self.cur.clone();
            self.cur = self.cur.add(&one);
            let mut is_prime = true;
            for prev in &self.prevs {
                if prev.mul(prev).cmp(&cur) == Ordering::Greater {
                    break;
                }
                if cur.rem(prev).is_zero() {
                    is_prime = false;
                    break;
                }
            }
            if is_prime {
                self.prevs.push(cur.clone());
                return Some(cur);
            }
        }
    }
}
