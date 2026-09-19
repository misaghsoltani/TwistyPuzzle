//! Exact rational arithmetic.
//!
//! The in-place operations carry a `reduce` flag, because an unreduced
//! fraction prints differently (`"2/4"`) and those strings are used as
//! dictionary keys during piece enumeration, making reduction status observable.
//! See `SEMANTICS.md` §2 and §5.

use core::cmp::Ordering;
use core::fmt;

use super::int::{pow2, Int};
use super::ring::{Elem, RingOps};
use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct Fraction {
    pub n: Int,
    pub d: Int,
}

impl Fraction {
    /// `new Fraction(n, d, reduce)`
    pub fn new(n: Int, d: Int, reduce: bool) -> Result<Fraction> {
        let (n, d) = match d.sign() {
            -1 => (n.neg(), d.neg()),
            0 => return Err(Error::Range("Division by zero".into())),
            _ => (n, d),
        };
        let mut f = Fraction { n, d };
        if reduce {
            f.reduce();
        }
        Ok(f)
    }

    /// `fraction(n, d)`: the reducing convenience constructor.
    pub fn of(n: i64, d: i64) -> Fraction {
        Fraction::new(Int::from_i64(n), Int::from_i64(d), true)
            .expect("fraction(): zero denominator")
    }

    /// `fraction(n)`
    pub fn int(n: i64) -> Fraction {
        Fraction {
            n: Int::from_i64(n),
            d: Int::from_i64(1),
        }
    }

    pub fn zero() -> Fraction {
        Fraction::int(0)
    }

    pub fn one() -> Fraction {
        Fraction::int(1)
    }

    /// Divide out the common factor, leaving the value unchanged.
    ///
    /// The three shortcuts are identities, not approximations: `gcd(n, 1)` and
    /// a gcd of one both leave the pair untouched, and `gcd(0, d) == d` sends
    /// `0/d` to `0/1`. Whole numbers (`d == 1`) are by far the most common
    /// case and now cost a single comparison.
    pub fn reduce(&mut self) -> &mut Self {
        if self.d.is_one() {
            return self;
        }
        if self.n.is_zero() {
            self.d = Int::from_i64(1);
            return self;
        }
        let g = self.n.gcd(&self.d);
        if g.is_one() {
            return self;
        }
        self.n = self.n.div(&g);
        self.d = self.d.div(&g);
        self
    }

    pub fn is_zero(&self) -> bool {
        self.n.is_zero()
    }

    pub fn ineg(&mut self) {
        self.n = self.n.neg();
    }

    /// `iadd(y, reduce)`
    pub fn iadd_r(&mut self, y: &Fraction, reduce: bool) {
        self.n = self.n.mul(&y.d).add(&self.d.mul(&y.n));
        self.d = self.d.mul(&y.d);
        if reduce {
            self.reduce();
        }
    }

    /// `imul(y, reduce)`
    pub fn imul_r(&mut self, y: &Fraction, reduce: bool) {
        self.n = self.n.mul(&y.n);
        self.d = self.d.mul(&y.d);
        if reduce {
            self.reduce();
        }
    }

    pub fn iinv(&mut self) -> Result<()> {
        match self.n.sign() {
            1 => core::mem::swap(&mut self.n, &mut self.d),
            -1 => {
                let (n, d) = (self.n.clone(), self.d.clone());
                self.n = d.neg();
                self.d = n.neg();
            },
            _ => return Err(Error::Range("Division by zero".into())),
        }
        Ok(())
    }

    /// The value as the nearest `f64`, ties to even.
    ///
    /// The only conversion there is: correctly rounded, and the one every
    /// rendered coordinate is built on.
    pub fn to_f64_nearest(&self) -> f64 {
        /// Bits in an `f64` significand, including the implicit one.
        const P: i32 = 53;

        if self.n.is_zero() {
            return 0.0;
        }
        let sign = if self.sign() < 0 { -1.0 } else { 1.0 };
        let (mut a, mut b) = (self.n.abs(), self.d.abs());

        // Scale until b/2 <= a < b, so a/b lands in [1/2, 1).
        let mut e: i32 = 0;
        let two = Int::from_i64(2);
        while two.mul(&a).cmp(&b) == Ordering::Less {
            a = a.mul(&two);
            e -= 1;
        }
        while a.cmp(&b) != Ordering::Less {
            b = b.mul(&two);
            e += 1;
        }

        // a * 2^53 / b lies in [2^52, 2^53), so `q` has exactly 53 bits and
        // converts to f64 without loss, as does q + 1, at 2^53.
        let (q, r) = a.mul(&two.pow(P as u32)).div_rem(&b);
        let twice_r = r.mul(&two);
        let q = match twice_r.cmp(&b) {
            Ordering::Greater => q.add(&Int::from_i64(1)),
            // A tie goes to the even neighbor.
            Ordering::Equal if !q.rem(&two).is_zero() => q.add(&Int::from_i64(1)),
            _ => q,
        };
        sign * q.to_f64() * pow2(e - P)
    }

    pub fn sign(&self) -> i32 {
        self.n.sign()
    }

    /// `compare(y)`
    pub fn compare(&self, y: &Fraction) -> i32 {
        self.n.mul(&y.d).sub(&y.n.mul(&self.d)).sign()
    }

    pub fn cmp_ord(&self, y: &Fraction) -> Ordering {
        match self.compare(y) {
            -1 => Ordering::Less,
            1 => Ordering::Greater,
            _ => Ordering::Equal,
        }
    }

    pub fn iabs(&mut self) {
        self.n = self.n.abs();
    }

    pub fn abs(&self) -> Fraction {
        let mut c = self.clone();
        c.iabs();
        c
    }

    /// `middle(y)`: a fraction between `self` and `y` whose denominator is a
    /// power of two.
    pub fn middle(&self, y: &Fraction) -> Fraction {
        let x = self;
        if x.equals(y) {
            return x.clone();
        }
        let two = Int::from_i64(2);
        let mut d = Int::from_i64(1);
        loop {
            let bound = Fraction::new(Int::from_i64(1), d.clone(), true)
                .expect("power of two is never zero");
            let mut diff = y.clone();
            let _ = Elem::isub(&mut diff, x);
            if diff.abs().compare(&bound) <= 0 {
                d = d.mul(&two);
            } else {
                break;
            }
        }
        let mut m = x.clone();
        m.iadd_r(y, true);
        let half_d = Fraction::new(d.clone(), two.clone(), true).expect("2 != 0");
        m.imul_r(&half_d, true);
        let half_sign =
            Fraction::new(Int::from_i64(i64::from(m.sign())), two.clone(), true).expect("2 != 0");
        m.iadd_r(&half_sign, true);
        let n = m.n.div(&m.d);
        Fraction::new(n, d, true).expect("power of two is never zero")
    }
}

impl Elem for Fraction {
    #[inline]
    fn is_zero(&self) -> bool {
        self.n.is_zero()
    }

    fn equals(&self, y: &Fraction) -> bool {
        // Denominators are always positive (`new` normalizes the sign and
        // every in-place operation multiplies two positive denominators), so
        // both shortcuts below give exactly the answer the cross-multiplication
        // gives, without its two `Int` multiplications. `Polynomial::trim`
        // compares a coefficient against zero after every polynomial
        // operation, which makes this one of the hottest predicates here.
        if self.n.is_zero() || y.n.is_zero() {
            return self.n.is_zero() && y.n.is_zero();
        }
        if self.d.eq(&y.d) {
            return self.n.eq(&y.n);
        }
        self.n.mul(&y.d).eq(&self.d.mul(&y.n))
    }

    fn iadd(&mut self, y: &Fraction) -> Result<()> {
        self.iadd_r(y, true);
        Ok(())
    }

    fn add(&self, y: &Fraction) -> Result<Fraction> {
        let mut c = self.clone();
        c.iadd_r(y, true);
        Ok(c)
    }

    fn isub(&mut self, y: &Fraction) -> Result<()> {
        self.n = self.n.mul(&y.d).sub(&self.d.mul(&y.n));
        self.d = self.d.mul(&y.d);
        self.reduce();
        Ok(())
    }

    fn sub(&self, y: &Fraction) -> Result<Fraction> {
        let mut c = self.clone();
        Elem::isub(&mut c, y)?;
        Ok(c)
    }

    fn neg(&self) -> Fraction {
        let mut c = self.clone();
        c.ineg();
        c
    }

    fn imul(&mut self, y: &Fraction) -> Result<()> {
        self.imul_r(y, true);
        Ok(())
    }

    fn mul(&self, y: &Fraction) -> Result<Fraction> {
        let mut c = self.clone();
        c.imul_r(y, true);
        Ok(c)
    }

    fn inv(&self) -> Result<Fraction> {
        let mut c = self.clone();
        c.iinv()?;
        Ok(c)
    }

    fn idiv(&mut self, y: &Fraction) -> Result<()> {
        // Dividing by one still reduces (`Polynomial::divmod` leans on that),
        // but the two multiplications it would do are both by one.
        if y.n.is_one() && y.d.is_one() {
            self.reduce();
            return Ok(());
        }
        match y.n.sign() {
            1 => {
                self.n = self.n.mul(&y.d);
                self.d = self.d.mul(&y.n);
            },
            -1 => {
                self.n = self.n.mul(&y.d.neg());
                self.d = self.d.mul(&y.n.neg());
            },
            _ => return Err(Error::Range("Division by zero".into())),
        }
        self.reduce();
        Ok(())
    }

    fn div(&self, y: &Fraction) -> Result<Fraction> {
        let mut c = self.clone();
        Elem::idiv(&mut c, y)?;
        Ok(c)
    }
}

impl fmt::Display for Fraction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d.is_one() {
            write!(f, "{}", self.n)
        } else {
            write!(f, "{}/{}", self.n, self.d)
        }
    }
}

impl PartialEq for Fraction {
    fn eq(&self, other: &Self) -> bool {
        Elem::equals(self, other)
    }
}
impl Eq for Fraction {}

/// `QQ`: the field of rationals.
#[derive(Clone, Copy, Debug, Default)]
pub struct QQ;

impl RingOps<Fraction> for QQ {
    fn zero(&self) -> Fraction {
        Fraction::new(Int::from_i64(0), Int::from_i64(1), false).unwrap()
    }
    fn one(&self) -> Fraction {
        Fraction::new(Int::from_i64(1), Int::from_i64(1), false).unwrap()
    }
    fn from_int(&self, n: i64) -> Fraction {
        Fraction::new(Int::from_i64(n), Int::from_i64(1), false).unwrap()
    }
}
