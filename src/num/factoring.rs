//! Factorization of rational polynomials over ℚ.
//!
//! Berlekamp-Zassenhaus: make squarefree and primitive, factor mod a small
//! prime (distinct-degree + Cantor-Zassenhaus), Hensel-lift to a prime power,
//! then recombine.

use super::fraction::{Fraction, QQ};
use super::int::{Int, Primes};
use super::polynomial::{gcd as poly_gcd, Polynomial, Polynomials};
use super::ring::{
    extended_gcd, gcd as ring_gcd, power_mod, product, Elem, Euclidean, Integer, IntegerMod, IntegersMod, RingOps, ZZ,
};
use crate::console_assert;
use crate::error::{Error, Result};
use core::cell::Cell;

type PolyQ = Polynomial<Fraction, QQ>;
type PolyZ = Polynomial<Integer, ZZ>;
type PolyFp = Polynomial<IntegerMod, IntegersMod>;

fn fp_x(p: &Int) -> Polynomials<IntegerMod, IntegersMod> {
    Polynomials::new(IntegersMod::new(p.clone()))
}

/* -------------------------------------------------------------------------- */
/*  Deterministic replacement for `Math.random`                               */
/* -------------------------------------------------------------------------- */

thread_local! {
    /// Only the *order* in which
    /// Cantor-Zassenhaus discovers factors depends on those draws (the factor
    /// set does not), and `exact::extend` (the sole caller of `factor`) selects
    /// its factor by root-counting instead of position. A deterministic
    /// stream therefore preserves observable behavior while making runs
    /// reproducible.
    static RNG_STATE: Cell<u64> = const { Cell::new(0x2545_F491_4F6C_DD1D) };
}

fn next_bit() -> bool {
    RNG_STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 33) & 1 == 1
    })
}

/// `bigutils.random`: a uniform integer in `[0, n)`, built bit by bit.
pub fn random(n: &Int) -> Int {
    let two = Int::from_i64(2);
    let mut r = Int::from_i64(0);
    let mut m = n.clone();
    while m.sign() > 0 {
        r = r.mul(&two);
        if next_bit() {
            r = r.add(&Int::from_i64(1));
        }
        m = m.div(&two);
    }
    r.rem(n)
}

/* -------------------------------------------------------------------------- */

/// Bound on the coefficients of factors of degree `<= deg(P)/2`.
fn factor_bound(p: &PolyZ) -> Result<Int> {
    let two = Int::from_i64(2);
    let one = Int::from_i64(1);
    let d = Int::from_i64(p.degree as i64).div(&two);
    let mut n = Int::from_i64(0);
    for c in &p.coeffs {
        n = n.add(&c.n.mul(&c.n));
    }
    n = n.sqrt_upper();
    let lc = p
        .lc()
        .ok_or_else(|| Error::Other("factor_bound of zero polynomial".into()))?;
    let a = Int::comb(&d.sub(&one), &d.div(&two).sub(&one)).mul(&lc.n);
    let b = Int::comb(&d.sub(&one), &d.div(&two)).mul(&n);
    Ok(a.add(&b))
}

/// `combinations(xs, k)`: every `k`-subset, lexicographically.
///
/// Driven by an internal callback so the sequence stays lazy, as the
/// generator would be. The `i > xs.len()` guard replaces a
/// non-terminating recursion that would otherwise be resolved by overflowing
/// the stack, it is unreachable for `k <= xs.len()`.
fn combinations<T: Clone>(
    xs: &[T],
    k: isize,
    i: usize,
    prefix: &mut Vec<T>,
    f: &mut impl FnMut(Vec<T>) -> Result<()>,
) -> Result<()> {
    if i > xs.len() {
        return Ok(());
    }
    if i as isize + k == xs.len() as isize {
        let mut out = prefix.clone();
        out.extend_from_slice(&xs[i..]);
        return f(out);
    }
    combinations(xs, k, i + 1, prefix, f)?;
    if k > 0 {
        prefix.push(xs[i].clone());
        let r = combinations(xs, k - 1, i + 1, prefix, f);
        prefix.pop();
        r?;
    }
    Ok(())
}

/// Public wrapper collecting every `k`-subset (used by the test-suite port).
pub fn combinations_vec<T: Clone>(xs: &[T], k: isize) -> Vec<Vec<T>> {
    let mut out = Vec::new();
    let mut prefix = Vec::new();
    let _ = combinations(xs, k, 0, &mut prefix, &mut |c| {
        out.push(c);
        Ok(())
    });
    out
}

/// Make `A` squarefree and primitive, returning an integer polynomial.
fn factor_squarefree(a: &PolyQ) -> Result<PolyZ> {
    let qq_x = Polynomials::new(QQ);
    let g = ring_gcd(&qq_x, a, &a.derivative()?)?;
    let a_sf = Elem::div(a, &g.monic()?)?;

    let mut n_gcd = a_sf.coeffs[0].n.clone();
    let mut d_lcm = a_sf.coeffs[0].d.clone();
    for c in &a_sf.coeffs {
        n_gcd = n_gcd.gcd(&c.n);
        d_lcm = d_lcm.mul(&c.d).div(&d_lcm.gcd(&c.d));
    }
    a_sf.map(ZZ, |c| Ok(Integer::new(c.n.mul(&d_lcm).div(&c.d).div(&n_gcd))))
}

/// Split `A` into `A0..Ad`, where `Ai` is the product of the irreducible
/// factors of degree `i` (Cohen, Algorithm 3.4.3).
fn factor_distinct_degrees(a: &PolyFp, p: &Int) -> Result<Vec<PolyFp>> {
    let fp = IntegersMod::new(p.clone());
    let fpx = fp_x(p);
    let one = Polynomial::new(fp.clone(), vec![fp.one()]);
    let x = Polynomial::new(fp.clone(), vec![fp.zero(), fp.one()]);
    let mut factors = vec![one.clone()];
    let mut v = a.clone();
    let mut w = x.clone();
    let mut d: isize = 1;
    while 2 * d <= a.degree {
        w = power_mod(&fpx, &w, p, &v)?; // ≡ x^(p^d) mod V
        let a_d = ring_gcd(&fpx, &Elem::sub(&w, &x)?, &v)?;
        factors.push(a_d.clone());
        v = Elem::div(&v, &a_d)?;
        d += 1;
    }
    while (factors.len() as isize) < v.degree {
        factors.push(one.clone());
    }
    factors.push(v.clone());
    console_assert!(
        Elem::equals(&product(&fpx, &factors)?, a),
        "factor_distinct_degrees: product of factors != A"
    );
    Ok(factors)
}

/// Split a squarefree `A` whose factors all have degree `d` (Cohen, 3.4.6).
fn factor_cantor_zassenhaus(a: &PolyFp, d: isize, p: &Int) -> Result<Vec<PolyFp>> {
    let fp = IntegersMod::new(p.clone());
    let fpx = fp_x(p);
    let one = Polynomial::new(fp.clone(), vec![fp.one()]);
    if p.eq(&Int::from_i64(2)) {
        return Err(Error::Range("p=2 not implemented".into()));
    }
    if a.degree == d {
        return Ok(vec![a.clone()]);
    }
    for _t in 0..1000 {
        // A random monic polynomial of degree 2d-1.
        let mut coeffs: Vec<IntegerMod> = Vec::new();
        for _i in 0..(2 * d - 1) {
            coeffs.push(IntegerMod::new(random(p), p.clone()));
        }
        coeffs.push(IntegerMod::new(Int::from_i64(1), p.clone()));
        let t_poly = Polynomial::new(fp.clone(), coeffs);
        // A non-trivial factor of A with probability close to 1/2.
        let exp = p.pow(d as u32).sub(&Int::from_i64(1)).div(&Int::from_i64(2));
        let pm = power_mod(&fpx, &t_poly, &exp, a)?;
        let b1 = ring_gcd(&fpx, a, &Elem::sub(&pm, &one)?)?.monic()?;
        if 0 < b1.degree && b1.degree < a.degree {
            let (b2, _) = a.divmod(&b1)?;
            let mut factors = factor_cantor_zassenhaus(&b1, d, p)?;
            factors.extend(factor_cantor_zassenhaus(&b2, d, p)?);
            console_assert!(
                Elem::equals(&product(&fpx, &factors)?, a),
                "factor_cantor_zassenhaus: product of factors != A"
            );
            return Ok(factors);
        }
    }
    Err(Error::Other("This (probably) shouldn't happen".into()))
}

/// Factor a squarefree polynomial in `F_p[x]`, ignoring multiplicities and
/// constant factors.
fn factor_mod(a: &PolyFp, p: &Int) -> Result<Vec<PolyFp>> {
    let factors_ddf = factor_distinct_degrees(a, p)?;
    let mut factors_modp: Vec<PolyFp> = Vec::new();
    for (d, f) in factors_ddf.iter().enumerate() {
        if f.degree != 0 {
            factors_modp.extend(factor_cantor_zassenhaus(f, d as isize, p)?);
        }
    }
    factors_modp.iter().map(Polynomial::monic).collect()
}

/* --- modulus-juggling helpers ------------------------------------------- */

fn to_zz(a: &PolyFp) -> Result<PolyZ> {
    a.map(ZZ, |c| Ok(Integer::new(c.n.clone())))
}

fn to_fp(a: &PolyZ, p: &Int) -> Result<PolyFp> {
    a.map(IntegersMod::new(p.clone()), |c| {
        Ok(IntegerMod::new(c.n.clone(), p.clone()))
    })
}

fn divmod_mod(a: &PolyZ, b: &PolyZ, p: &Int) -> Result<(PolyZ, PolyZ)> {
    let (q_p, r_p) = to_fp(a, p)?.divmod(&to_fp(b, p)?)?;
    Ok((to_zz(&q_p)?, to_zz(&r_p)?))
}

/// Lift a factorization `C = A_p * B_p (mod p)` to `mod p^e`
/// (Cohen, Algorithms 3.5.5 and 3.5.6).
fn hensel_lift_two(c: &PolyZ, a_p: &PolyFp, b_p: &PolyFp, p_in: &Int, e: &Int) -> Result<(PolyFp, PolyFp)> {
    let mut p = p_in.clone();
    let fpx = fp_x(&p);
    let (lc, u_p, v_p) = extended_gcd(&fpx, a_p, b_p)?;
    console_assert!(lc.degree == 0, "{} is not constant", lc);
    let (u_p, v_p) = (Elem::div(&u_p, &lc)?, Elem::div(&v_p, &lc)?);

    // Moduli change often, so the work happens in Z.
    let (mut a, mut b) = (to_zz(a_p)?, to_zz(b_p)?);
    let (mut u, mut v) = (to_zz(&u_p)?, to_zz(&v_p)?);
    let one: PolyZ = Polynomial::new(ZZ, vec![ZZ.one()]);

    let mut q = p.clone();
    let e_u32 = e.to_i64().ok_or_else(|| Error::Other("exponent too large".into()))? as u32;
    let q_final = p_in.pow(e_u32);

    // Overshoots to p^(2^k) where 2^k >= e.
    while q.cmp(&q_final) == core::cmp::Ordering::Less {
        let pint = Integer::new(p.clone());
        q = p.mul(&p); // lifting from p to q

        let f = Elem::sub(c, &Elem::mul(&a, &b)?)?.sdiv(&pint)?;
        let (t, a0) = divmod_mod(&Elem::mul(&v, &f)?, &a, &p)?;
        let b0 = Elem::add(&Elem::mul(&u, &f)?, &Elem::mul(&b, &t)?)?.map(ZZ, |c| c.modulo(&pint))?;
        let (a_new, b_new) = (Elem::add(&a, &a0.smul(&pint)?)?, Elem::add(&b, &b0.smul(&pint)?)?);
        a = a_new;
        b = b_new;

        let g = Elem::sub(&Elem::sub(&one, &Elem::mul(&u, &a)?)?, &Elem::mul(&v, &b)?)?.sdiv(&pint)?;
        let (s, v0) = divmod_mod(&Elem::mul(&v, &g)?, &a, &p)?;
        let u0 = Elem::add(&Elem::mul(&u, &g)?, &Elem::mul(&b, &s)?)?;
        let u0 = u0.map(ZZ, |c| c.modulo(&pint))?;
        let (u_new, v_new) = (Elem::add(&u, &u0.smul(&pint)?)?, Elem::add(&v, &v0.smul(&pint)?)?);
        u = u_new;
        v = v_new;

        p = q.clone();
    }
    Ok((to_fp(&a, &q_final)?, to_fp(&b, &q_final)?))
}

/// Lift a full factorization mod `p` to mod `p^e`
/// (von zur Gathen, Algorithm 15.17).
fn hensel_lift(c: &PolyZ, c_factors: &[PolyFp], p: &Int, e: &Int) -> Result<Vec<PolyFp>> {
    let e_u32 = e.to_i64().ok_or_else(|| Error::Other("exponent too large".into()))? as u32;
    if c_factors.len() == 1 {
        return Ok(vec![to_fp(c, &p.pow(e_u32))?.monic()?]);
    }
    let fpx = fp_x(p);
    let split = c_factors.len() / 2;
    let (a_factors, b_factors) = c_factors.split_at(split);
    let a_p = product(&fpx, a_factors)?;
    let lc = c
        .lc()
        .ok_or_else(|| Error::Other("hensel_lift of zero polynomial".into()))?;
    let b_p = product(&fpx, b_factors)?.smul(&IntegerMod::new(lc.n.clone(), p.clone()))?;
    let (a_q, b_q) = hensel_lift_two(c, &a_p, &b_p, p, e)?;
    let (a, b) = (to_zz(&a_q)?, to_zz(&b_q)?);
    let mut out = hensel_lift(&a, a_factors, p, e)?;
    out.extend(hensel_lift(&b, b_factors, p, e)?);
    Ok(out)
}

/// Recombine the lifted factors (Cohen, Algorithm 3.5.7, step 5).
///
/// Assumes the coefficients of the factors of `U` lie in `[-p/2, p/2)`.
fn combine_factors(u_in: &PolyZ, factors_modp: &[PolyFp], p: &Int) -> Result<Vec<PolyZ>> {
    let fpx = fp_x(p);
    let mut factors: Vec<PolyZ> = Vec::new();
    let mut u = u_in.clone();
    let mut d: isize = 1;
    let two = Int::from_i64(2);
    while d <= u.degree / 2 {
        let mut prefix = Vec::new();
        // `u` and `factors` are mutated as combinations are visited, exactly as a loop over a live generator would.
        let mut err: Option<Error> = None;
        combinations(factors_modp, d, 0, &mut prefix, &mut |comb| {
            if err.is_some() {
                return Ok(());
            }
            let mut step = || -> Result<()> {
                let lc = u
                    .lc()
                    .ok_or_else(|| Error::Other("combine_factors: zero polynomial".into()))?
                    .clone();
                let v = product(&fpx, &comb)?.smul(&IntegerMod::new(lc.n.clone(), p.clone()))?;

                // Move from F_p[x] to Z[x], using the fact that the
                // coefficients must lie in [-p/2, p/2).
                let mut v_z = v.map(ZZ, |c| {
                    Ok(Integer::new(if two.mul(&c.n).cmp(p) == core::cmp::Ordering::Less {
                        c.n.clone()
                    } else {
                        c.n.sub(p)
                    }))
                })?;
                let r = match u.smul(&lc)?.modulo(&v_z) {
                    Ok(r) => r,
                    Err(Error::Division(_)) => return Ok(()),
                    Err(e) => return Err(e),
                };
                if r.degree == -1 {
                    let mut n_gcd = v_z.coeffs[0].n.clone();
                    for c in &v_z.coeffs {
                        n_gcd = n_gcd.gcd(&c.n);
                    }
                    v_z = v_z.sdiv(&Integer::new(n_gcd))?;
                    factors.push(v_z.clone());
                    u = Elem::div(&u, &v_z)?;
                }
                Ok(())
            };
            if let Err(e) = step() {
                err = Some(e);
            }
            Ok(())
        })?;
        if let Some(e) = err {
            return Err(e);
        }
        d += 1;
    }
    if u.degree > 0 {
        factors.push(u);
    }
    Ok(factors)
}

/// The smallest odd prime `p` (not dividing `lc(U)`) for which `U` stays
/// squarefree mod `p`.
fn choose_prime(u: &PolyZ) -> Result<(Int, PolyFp)> {
    let one = Int::from_i64(1);
    let two = Int::from_i64(2);
    let lc = u
        .lc()
        .ok_or_else(|| Error::Other("choose_prime of zero polynomial".into()))?;
    for p in Primes::new() {
        let fpx = fp_x(&p);
        if p.eq(&two) {
            continue;
        }
        if !p.gcd(&lc.n).eq(&one) {
            continue;
        }
        let u_modp = to_fp(u, &p)?;
        if ring_gcd(&fpx, &u_modp, &u_modp.derivative()?)?.degree == 0 {
            return Ok((p, u_modp));
        }
    }
    Err(Error::Other("this shouldn't happen".into()))
}

/// `factor(A)`: the irreducible factors of a rational polynomial.
pub fn factor(a: &PolyQ) -> Result<Vec<PolyQ>> {
    // Make squarefree and primitive.
    let u = factor_squarefree(a)?;

    // Choose p such that U is squarefree mod p.
    let (p, u_modp) = choose_prime(&u)?;

    // Factor mod p.
    let factors_p = factor_mod(&u_modp, &p)?;

    // Choose exponent e.
    let b = factor_bound(&u)?;
    let lc = u.lc().ok_or_else(|| Error::Other("factor of zero polynomial".into()))?;
    let limit = Int::from_i64(2).mul(&lc.n).mul(&b);
    let mut e = Int::from_i64(1);
    let one = Int::from_i64(1);
    while p.pow(e.to_i64().unwrap_or(i64::MAX) as u32).cmp(&limit) != core::cmp::Ordering::Greater {
        e = e.add(&one);
    }

    // Lift to mod p^e.
    let factors_q = hensel_lift(&u, &factors_p, &p, &e)?;

    // Final combination, converted back to rationals.
    let q = p.pow(e.to_i64().unwrap_or(i64::MAX) as u32);
    combine_factors(&u, &factors_q, &q)?
        .iter()
        .map(|f| f.map(QQ, |c| Fraction::new(c.n.clone(), Int::from_i64(1), true)))
        .collect()
}

/// Suppress the unused-import warning for `poly_gcd`, kept for symmetry with the module layout.
#[allow(dead_code)]
fn _unused(a: &PolyQ, b: &PolyQ) -> Result<PolyQ> {
    poly_gcd(a, b)
}
