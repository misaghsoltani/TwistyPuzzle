//! Univariate polynomials over an arbitrary ring.

use core::cmp::Ordering;
use core::fmt;
use core::fmt::Write as _;

use super::fraction::{Fraction, QQ};
use super::int::Int;
use super::ring::{power, Elem, Euclidean, RingOps, ZZ};
use crate::error::{Error, Result};

const SUPERSCRIPTS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

/// `coeffs[i]` is the coefficient on `x^i`, and `degree == coeffs.len() - 1`
/// always holds because the constructor trims trailing zeros.
#[derive(Clone, Debug)]
pub struct Polynomial<E: Elem, R: RingOps<E>> {
    pub coeff_ring: R,
    pub coeffs: Vec<E>,
    pub degree: isize,
}

/// Return type of `pseudo_divmod`: the multiplier `p`, the quotient, and the
/// remainder, satisfying `p * dividend == quotient * divisor + remainder`.
type PseudoDivision<E, R> = (E, Polynomial<E, R>, Polynomial<E, R>);

impl<E: Elem, R: RingOps<E>> Polynomial<E, R> {
    pub fn new(coeff_ring: R, coeffs: Vec<E>) -> Polynomial<E, R> {
        let degree = coeffs.len() as isize - 1;
        let mut p = Polynomial {
            coeff_ring,
            coeffs,
            degree,
        };
        p.trim();
        p
    }

    pub fn trim(&mut self) -> &mut Self {
        while self.coeffs.last().is_some_and(Elem::is_zero) {
            self.coeffs.pop();
            self.degree -= 1;
        }
        self
    }

    /// `lc()`: the leading coefficient. `None` for the zero polynomial, which
    /// has none and which no caller asks for.
    pub fn lc(&self) -> Option<&E> {
        if self.degree < 0 {
            None
        } else {
            Some(&self.coeffs[self.degree as usize])
        }
    }

    pub fn is_zero(&self) -> bool {
        self.degree == -1
    }

    pub fn monic(&self) -> Result<Polynomial<E, R>> {
        let lc = self
            .lc()
            .ok_or_else(|| Error::Range("monic() of zero polynomial".into()))?
            .clone();
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for c in &self.coeffs {
            coeffs.push(c.div(&lc)?);
        }
        Ok(Polynomial::new(self.coeff_ring.clone(), coeffs))
    }

    pub fn map<E1: Elem, R1: RingOps<E1>>(
        &self,
        k1: R1,
        mut f: impl FnMut(&E) -> Result<E1>,
    ) -> Result<Polynomial<E1, R1>> {
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        for c in &self.coeffs {
            coeffs.push(f(c)?);
        }
        Ok(Polynomial::new(k1, coeffs))
    }

    /// `eval(arg)`: Horner's rule.
    pub fn eval(&self, arg: &E) -> Result<E> {
        let mut sum = self.coeff_ring.zero();
        for i in (0..=self.degree.max(-1)).rev() {
            if i < 0 {
                break;
            }
            sum.imul(arg)?;
            sum.iadd(&self.coeffs[i as usize])?;
        }
        Ok(sum)
    }

    /// `ismul(b)`: multiply every coefficient by a scalar.
    pub fn ismul(&mut self, b: &E) -> Result<()> {
        for c in &mut self.coeffs {
            c.imul(b)?;
        }
        self.trim();
        Ok(())
    }

    pub fn smul(&self, b: &E) -> Result<Polynomial<E, R>> {
        let mut c = self.clone();
        c.ismul(b)?;
        Ok(c)
    }

    pub fn isdiv(&mut self, b: &E) -> Result<()> {
        for c in &mut self.coeffs {
            c.idiv(b)?;
        }
        self.trim();
        Ok(())
    }

    pub fn sdiv(&self, b: &E) -> Result<Polynomial<E, R>> {
        let mut c = self.clone();
        c.isdiv(b)?;
        Ok(c)
    }

    /// `pseudo_divmod(b)`: returns `(p, q, r)` with `p*self == q*b + r`.
    pub fn pseudo_divmod(&self, b: &Polynomial<E, R>) -> Result<PseudoDivision<E, R>> {
        let m = self.degree;
        let n = b.degree;
        let mut p = self.coeff_ring.one();
        if n == -1 {
            return Err(Error::Division("Division by zero".into()));
        }
        let bn = b.coeffs[n as usize].clone();
        // c[0..k-1] is the remainder so far, c[k..m] the quotient so far.
        let mut c: Vec<E> = self.coeffs.clone();
        let mut k = m;
        while k >= n {
            for i in 0..=m {
                if i != k {
                    c[i as usize].imul(&bn)?;
                }
            }
            p.imul(&bn)?;
            for i in 0..n {
                let t = c[k as usize].mul(&b.coeffs[i as usize])?;
                c[(k - n + i) as usize].isub(&t)?;
            }
            k -= 1;
        }
        let lo = (n as usize).min(c.len());
        let hi = ((m + 1).max(0) as usize).min(c.len());
        let quot: Vec<E> = if lo <= hi { c[lo..hi].to_vec() } else { Vec::new() };
        let rem: Vec<E> = c[..lo].to_vec();
        Ok((
            p,
            Polynomial::new(self.coeff_ring.clone(), quot),
            Polynomial::new(self.coeff_ring.clone(), rem),
        ))
    }

    /// `modulo`, consuming the dividend.
    ///
    /// Same answer as [`Euclidean::modulo`], but the common case (a dividend
    /// that already has lower degree than the divisor) reuses the coefficient
    /// vector instead of copying it. Every `AlgebraicNumber` is reduced modulo
    /// its field polynomial on construction, so this is the single most
    /// frequently taken path in the crate.
    pub fn into_modulo(mut self, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
        if b.degree == -1 {
            return Err(Error::Division("Division by zero".into()));
        }
        if self.degree < b.degree {
            let one = self.coeff_ring.one();
            for c in &mut self.coeffs {
                c.idiv(&one)?;
            }
            self.trim();
            return Ok(self);
        }
        Ok(self.divmod(b)?.1)
    }

    pub fn derivative(&self) -> Result<Polynomial<E, R>> {
        let mut coeffs = Vec::with_capacity(self.degree.max(0) as usize);
        for i in 1..=self.degree.max(0) {
            if i > self.degree {
                break;
            }
            let mut t = self.coeff_ring.from_int(i as i64);
            t.imul(&self.coeffs[i as usize])?;
            coeffs.push(t);
        }
        Ok(Polynomial::new(self.coeff_ring.clone(), coeffs))
    }
}

impl<E: Elem, R: RingOps<E>> Elem for Polynomial<E, R> {
    const IS_POLY: bool = true;

    #[inline]
    fn is_zero(&self) -> bool {
        self.degree == -1
    }

    fn equals(&self, b: &Polynomial<E, R>) -> bool {
        if self.degree != b.degree {
            return false;
        }
        for i in 0..=self.degree.max(-1) {
            if i < 0 {
                break;
            }
            if !self.coeffs[i as usize].equals(&b.coeffs[i as usize]) {
                return false;
            }
        }
        true
    }

    fn iadd(&mut self, b: &Polynomial<E, R>) -> Result<()> {
        for i in 0..b.coeffs.len() {
            if i < self.coeffs.len() {
                self.coeffs[i].iadd(&b.coeffs[i])?;
            } else {
                self.coeffs.push(b.coeffs[i].clone());
                self.degree += 1;
            }
        }
        self.trim();
        Ok(())
    }

    fn add(&self, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
        let mut c = self.clone();
        Elem::iadd(&mut c, b)?;
        Ok(c)
    }

    fn neg(&self) -> Polynomial<E, R> {
        let coeffs = self.coeffs.iter().map(Elem::neg).collect();
        Polynomial::new(self.coeff_ring.clone(), coeffs)
    }

    /// The obvious spelling is `iadd(neg(b))`. Subtracting coefficient
    /// by coefficient is the same arithmetic, because `neg` never introduces or
    /// removes a zero coefficient (so the two loops visit the same indices),
    /// and it does not materialize the negated polynomial.
    fn isub(&mut self, b: &Polynomial<E, R>) -> Result<()> {
        for i in 0..b.coeffs.len() {
            if i < self.coeffs.len() {
                self.coeffs[i].isub(&b.coeffs[i])?;
            } else {
                self.coeffs.push(Elem::neg(&b.coeffs[i]));
                self.degree += 1;
            }
        }
        self.trim();
        Ok(())
    }

    fn sub(&self, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
        let mut c = self.clone();
        Elem::isub(&mut c, b)?;
        Ok(c)
    }

    fn mul(&self, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
        let m = self.degree;
        let n = b.degree;
        let mut coeffs: Vec<E> = Vec::with_capacity(((m + n + 1).max(0)) as usize);
        let mut k: isize = 0;
        while k <= m + n {
            let mut ck = self.coeff_ring.zero();
            let mut i = (k - n).max(0);
            while i <= k.min(m) {
                // Intermediate fractions are deliberately left unreduced here (see note above).
                let t = self.coeffs[i as usize].mul(&b.coeffs[(k - i) as usize])?;
                ck.iadd(&t)?;
                i += 1;
            }
            coeffs.push(ck);
            k += 1;
        }
        Ok(Polynomial::new(self.coeff_ring.clone(), coeffs))
    }

    fn imul(&mut self, b: &Polynomial<E, R>) -> Result<()> {
        let c = Elem::mul(self, b)?;
        self.coeffs = c.coeffs;
        self.degree = c.degree;
        Ok(())
    }

    fn div(&self, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
        if b.degree == 0 {
            // Special faster case.
            let b0 = b.coeffs[0].clone();
            let mut coeffs = Vec::with_capacity(self.coeffs.len());
            for x in &self.coeffs {
                coeffs.push(x.div(&b0)?);
            }
            Ok(Polynomial::new(self.coeff_ring.clone(), coeffs))
        } else {
            let (q, r) = self.divmod(b)?;
            if r.degree > -1 {
                return Err(Error::Division("not divisible".into()));
            }
            Ok(q)
        }
    }

    fn idiv(&mut self, b: &Polynomial<E, R>) -> Result<()> {
        let c = Elem::div(self, b)?;
        self.coeffs = c.coeffs;
        self.degree = c.degree;
        Ok(())
    }

    fn inv(&self) -> Result<Polynomial<E, R>> {
        Err(Error::Other("not implemented".into()))
    }
}

impl<E: Elem, R: RingOps<E>> Euclidean for Polynomial<E, R> {
    fn euclidean(&self) -> Int {
        Int::from_i64(self.degree as i64)
    }

    fn divmod(&self, b: &Polynomial<E, R>) -> Result<(Polynomial<E, R>, Polynomial<E, R>)> {
        if b.degree == -1 {
            return Err(Error::Division("Division by zero".into()));
        }
        if self.degree < b.degree {
            // Nothing to divide: the quotient is zero and the remainder is
            // `self`. `pseudo_divmod` would reach the same answer, but only
            // after copying the coefficients three times. This is the common
            // case, because reducing an element modulo its own field
            // polynomial is done on every construction.
            //
            // The pseudo-divisor is one here, so the scaling `pseudo_divmod`
            // would apply is exactly the per-coefficient normalization below.
            let one = self.coeff_ring.one();
            let mut coeffs = self.coeffs.clone();
            for c in &mut coeffs {
                c.idiv(&one)?;
            }
            return Ok((
                Polynomial::new(self.coeff_ring.clone(), Vec::new()),
                Polynomial::new(self.coeff_ring.clone(), coeffs),
            ));
        }
        let (p, mut q, mut r) = self.pseudo_divmod(b)?;
        q.isdiv(&p)?;
        r.isdiv(&p)?;
        Ok((q, r))
    }
}

impl<E: Elem, R: RingOps<E>> fmt::Display for Polynomial<E, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.degree == -1 {
            return write!(f, "0");
        }
        // These strings are dictionary keys during piece enumeration (see
        // `SEMANTICS.md` §2), so this runs once per face of every piece and is
        // written to avoid both temporaries and the formatting machinery.
        let one = self.coeff_ring.one();
        for i in (0..=self.degree).rev() {
            let c = &self.coeffs[i as usize];
            if Elem::is_zero(c) {
                continue;
            }
            if i < self.degree {
                f.write_str(" + ")?;
            }

            if !c.equals(&one) || i == 0 {
                if E::IS_POLY {
                    f.write_char('(')?;
                    fmt::Display::fmt(c, f)?;
                    f.write_char(')')?;
                } else {
                    fmt::Display::fmt(c, f)?;
                }
            }

            if i >= 1 {
                f.write_char('x')?;
            }
            if i > 1 {
                // Exponent digits, most significant first, without allocating.
                let mut digits = [0u8; 20];
                let mut n = i;
                let mut len = 0;
                while n > 0 {
                    digits[len] = (n % 10) as u8;
                    n /= 10;
                    len += 1;
                }
                for d in digits[..len].iter().rev() {
                    f.write_char(SUPERSCRIPTS[*d as usize])?;
                }
            }
        }
        Ok(())
    }
}

impl<E: Elem, R: RingOps<E>> PartialEq for Polynomial<E, R> {
    fn eq(&self, other: &Self) -> bool {
        Elem::equals(self, other)
    }
}
impl<E: Elem, R: RingOps<E>> Eq for Polynomial<E, R> {}

/// `Polynomials<E>`: the ring of polynomials over a coefficient ring.
#[derive(Clone, Debug)]
pub struct Polynomials<E: Elem, R: RingOps<E>> {
    pub coeff_ring: R,
    _marker: core::marker::PhantomData<fn() -> E>,
}

impl<E: Elem, R: RingOps<E>> Polynomials<E, R> {
    pub fn new(coeff_ring: R) -> Self {
        Polynomials {
            coeff_ring,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<E: Elem, R: RingOps<E>> RingOps<Polynomial<E, R>> for Polynomials<E, R> {
    fn zero(&self) -> Polynomial<E, R> {
        Polynomial::new(self.coeff_ring.clone(), vec![])
    }
    fn one(&self) -> Polynomial<E, R> {
        Polynomial::new(self.coeff_ring.clone(), vec![self.coeff_ring.one()])
    }
    fn from_int(&self, n: i64) -> Polynomial<E, R> {
        Polynomial::new(self.coeff_ring.clone(), vec![self.coeff_ring.from_int(n)])
    }
}

pub type QQx = Polynomials<Fraction, QQ>;
pub type ZZx = Polynomials<crate::num::ring::Integer, ZZ>;
pub type PolyQ = Polynomial<Fraction, QQ>;

pub fn qq_x() -> QQx {
    Polynomials::new(QQ)
}
pub fn zz_x() -> ZZx {
    Polynomials::new(ZZ)
}

/// `polynomial(coeffs)`
pub fn polynomial(coeffs: Vec<Fraction>) -> PolyQ {
    Polynomial::new(QQ, coeffs)
}

pub fn polynomial_i64(coeffs: &[i64]) -> PolyQ {
    polynomial(coeffs.iter().map(|&c| Fraction::int(c)).collect())
}

fn neg_one_pow<E: Elem, R: RingOps<E>>(k: &R, e: isize) -> E {
    k.from_int(if e.rem_euclid(2) == 0 { 1 } else { -1 })
}

/// Subresultant GCD (`polynomial.ts` `gcd`).
pub fn gcd<E: Elem, R: RingOps<E>>(a: &Polynomial<E, R>, b: &Polynomial<E, R>) -> Result<Polynomial<E, R>> {
    let k = a.coeff_ring.clone();
    let zero: Polynomial<E, R> = Polynomial::new(k.clone(), vec![]);

    let (a, b) = if a.degree < b.degree {
        (b.clone(), a.clone())
    } else {
        (a.clone(), b.clone())
    };
    let (mut r0, mut r1) = (a, b);
    let mut psi = k.from_int(-1);
    let mut d: isize = r0.degree - r1.degree;
    let mut beta = neg_one_pow(&k, d + 1);
    let mut gamma: Option<E> = r1.lc().cloned();

    while !Elem::equals(&r1, &zero) {
        let g_cur = gamma.clone().expect("r1 is non-zero, so it has a leading coefficient");
        let g = power(&k, &g_cur, &Int::from_i64((d + 1) as i64))?;
        let (_q, mut r2) = r0.smul(&g)?.divmod(&r1)?;
        r2.isdiv(&beta)?;
        r0 = r1;
        r1 = r2;

        let mut g = power(&k, &g_cur.neg(), &Int::from_i64(d as i64))?;
        if d > 1 {
            let p = power(&k, &psi, &Int::from_i64((d - 1) as i64))?;
            g.idiv(&p)?;
        } else if d < 1 {
            let p = power(&k, &psi, &Int::from_i64((1 - d) as i64))?;
            g.imul(&p)?;
        }
        psi = g;
        d = r0.degree - r1.degree;
        let mut nb = g_cur.neg();
        let p = power(&k, &psi, &Int::from_i64(d as i64))?;
        nb.imul(&p)?;
        beta = nb;
        gamma = r1.lc().cloned();
    }
    Ok(r0)
}

/// `resultant(a, b)`
pub fn resultant<E: Elem, R: RingOps<E>>(a: &Polynomial<E, R>, b: &Polynomial<E, R>) -> Result<E> {
    let k = a.coeff_ring.clone();
    let mut res = k.one();
    if a.degree < b.degree {
        let s = neg_one_pow(&k, a.degree * b.degree);
        res.imul(&s)?;
    }
    let g = gcd(a, b)?;
    if g.degree == 0 {
        res.imul(&g.coeffs[0])?;
        Ok(res)
    } else {
        Ok(k.zero())
    }
}

/// Number of sign changes in the Sturm sequence at `arg`.
fn sturm_sequence(p0: &PolyQ, arg: &Fraction) -> Result<i32> {
    let mut p0 = p0.clone();
    let mut p1 = p0.derivative()?;
    let mut prev_sign = p0.eval(arg)?.sign();
    let mut changes = 0;
    while p1.degree >= 0 {
        let cur_sign = p1.eval(arg)?.sign();
        if cur_sign != 0 {
            if prev_sign != 0 && cur_sign != prev_sign {
                changes += 1;
            }
            prev_sign = cur_sign;
        }
        let next = Elem::neg(&p0.modulo(&p1)?);
        p0 = p1;
        p1 = next;
    }
    Ok(changes)
}

/// Number of real roots in the closed interval `[lower, upper]`.
pub fn count_roots(p: &PolyQ, lower: &Fraction, upper: &Fraction) -> Result<i32> {
    let at_lower = i32::from(p.eval(lower)?.sign() == 0);
    Ok(at_lower + sturm_sequence(p, lower)? - sturm_sequence(p, upper)?)
}

/// `compare` on degrees, as used by the Euclidean `gcd` driver.
pub fn degree_cmp<E: Elem, R: RingOps<E>>(a: &Polynomial<E, R>, b: &Polynomial<E, R>) -> Ordering {
    a.degree.cmp(&b.degree)
}
