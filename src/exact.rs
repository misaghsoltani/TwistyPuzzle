//! Algebraic number fields ℚ(θ).
//!
//! Two properties of this representation are load-bearing and deliberate
//! (`SEMANTICS.md` §3 and §1):
//!
//! * **Field identity is pointer identity.** `check_same_field` and
//!   `extend`/`promote` compare fields with [`Arc::ptr_eq`], never
//!   structurally. Two fields with the same minimal polynomial are still
//!   different fields here, because their isolating intervals are separate
//!   mutable state and mixing numbers between them would mix that state.
//! * **`refine()` mutates state shared by every number in the field.** The
//!   isolating interval narrows over the lifetime of a computation, so
//!   `to_number` depends on call history. The interior mutability is the
//!   point, not an accident.

use std::sync::{Arc, OnceLock, RwLock};

use crate::error::{Error, Result};
use crate::num::factoring::factor;
use crate::num::fraction::{Fraction, QQ};
use crate::num::int::Int;
use crate::num::polynomial::{
    count_roots, gcd as poly_gcd, polynomial, qq_x, resultant, PolyQ, Polynomial, Polynomials, QQx,
};
use crate::num::ring::{extended_gcd, power, Elem, Euclidean, RingOps};

/// Bivariate: a polynomial whose coefficients are polynomials over ℚ.
pub type Poly2 = Polynomial<PolyQ, QQx>;
/// Trivariate, used only as scaffolding inside [`extend`].
pub type Poly3 = Polynomial<Poly2, Polynomials<PolyQ, QQx>>;

/// The half of a field that never changes once it is built.
///
/// Splitting it out of the lock matters: `degree`, `poly` and `powers` are
/// read on every single arithmetic operation, and a shared `RwLock` would
/// make each of those a synchronized access for no reason. Only the
/// isolating interval moves, and only [`refine`](AlgebraicNumberField::refine)
/// moves it.
pub struct FieldInner {
    pub degree: isize,
    /// Minimal polynomial of θ.
    pub poly: PolyQ,
    /// Powers of θ up to `2*degree`, reduced mod `poly` (Cohen), precomputed
    /// to speed up multiplication.
    pub powers: Vec<PolyQ>,
    /// The isolating interval, and the powers of its endpoints. This is the
    /// only mutable state in a field, and `refine()` narrows it in place for
    /// every number that shares the field, which is the point (`SEMANTICS.md`
    /// §1), not an oversight.
    pub interval: RwLock<FieldInterval>,
}

/// The isolating interval `[lower, upper]` around θ, with `powers_lower[i]`
/// and `powers_upper[i]` bounding θ^i.
pub struct FieldInterval {
    pub lower: Fraction,
    pub upper: Fraction,
    pub powers_lower: Vec<Fraction>,
    pub powers_upper: Vec<Fraction>,
}

/// ℚ(θ) where θ is a root of a polynomial with rational coefficients.
#[derive(Clone)]
pub struct AlgebraicNumberField(pub Arc<FieldInner>);

impl AlgebraicNumberField {
    pub fn new(poly: PolyQ, lower: Fraction, upper: Fraction) -> Result<AlgebraicNumberField> {
        let degree = poly.degree;
        let x = polynomial(vec![Fraction::int(0), Fraction::int(1)]);

        let mut powers: Vec<PolyQ> = vec![polynomial(vec![Fraction::int(1)])];
        for i in 1..degree {
            let prev = powers[(i - 1) as usize].clone();
            powers.push(Elem::mul(&prev, &x)?);
        }
        for i in degree..=(2 * degree) {
            let prev = powers[(i - 1) as usize].clone();
            powers.push(Elem::mul(&prev, &x)?.modulo(&poly)?);
        }

        if count_roots(&poly, &lower, &upper)? != 1 {
            return Err(Error::Other("Interval must contain exactly one root".into()));
        }

        let inner = FieldInner {
            degree,
            poly,
            powers,
            interval: RwLock::new(FieldInterval {
                lower,
                upper,
                powers_lower: Vec::new(),
                powers_upper: Vec::new(),
            }),
        };
        let field = AlgebraicNumberField(Arc::new(inner));
        field.precompute_interval_powers();
        Ok(field)
    }

    /// True when the two handles refer to the *same* field object.
    #[inline]
    pub fn is(&self, other: &AlgebraicNumberField) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// A stable identifier for this field, for use in cache keys.
    ///
    /// Field identity is pointer identity (two fields with the same minimal
    /// polynomial are still different fields), so the address is exactly the
    /// right thing and is stable for as long as the `Arc` lives.
    #[inline]
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    #[inline]
    pub fn degree(&self) -> isize {
        self.0.degree
    }

    /// The minimal polynomial of θ. Borrowed, not cloned: this is read on
    /// every construction of an [`AlgebraicNumber`].
    #[inline]
    pub fn poly(&self) -> &PolyQ {
        &self.0.poly
    }

    pub fn bounds(&self) -> (Fraction, Fraction) {
        let g = self.0.interval.read().expect("field lock poisoned");
        (g.lower.clone(), g.upper.clone())
    }

    pub fn equals(&self, other: &AlgebraicNumberField) -> Result<bool> {
        if self.is(other) {
            return Ok(true);
        }
        if !Elem::equals(&self.0.poly, &other.0.poly) {
            return Ok(false);
        }
        let poly = &self.0.poly;
        let (lo_a, hi_a) = self.bounds();
        let (lo_b, hi_b) = other.bounds();
        let lower = if lo_a.compare(&lo_b) > 0 { lo_a } else { lo_b };
        let upper = if hi_a.compare(&hi_b) < 0 { hi_a } else { hi_b };
        if lower.compare(&upper) > 0 {
            return Ok(false);
        }
        Ok(count_roots(poly, &lower, &upper)? == 1)
    }

    pub fn precompute_interval_powers(&self) {
        let mut g = self.0.interval.write().expect("field lock poisoned");
        let (lower, upper, degree) = (g.lower.clone(), g.upper.clone(), self.0.degree);
        let mut pl = vec![Fraction::int(1)];
        let mut pu = vec![Fraction::int(1)];
        for i in 1..=degree.max(0) {
            let i = i as usize;
            let mut vals = vec![
                Elem::mul(&pl[i - 1], &lower).expect("rational multiplication cannot fail"),
                Elem::mul(&pl[i - 1], &upper).expect("rational multiplication cannot fail"),
                Elem::mul(&pu[i - 1], &lower).expect("rational multiplication cannot fail"),
                Elem::mul(&pu[i - 1], &upper).expect("rational multiplication cannot fail"),
            ];
            // The pinned sort, for consistency with every other sort here.
            crate::sort::sort_by_infallible(&mut vals, |a: &Fraction, b: &Fraction| a.compare(b));
            pl.push(vals[0].clone());
            pu.push(vals[3].clone());
        }
        g.powers_lower = pl;
        g.powers_upper = pu;
    }

    /// Halve the isolating interval around θ.
    pub fn refine(&self) -> Result<()> {
        REFINE_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let (lower, upper) = self.bounds();
        let poly = &self.0.poly;
        let mid = lower.middle(&upper);
        let sl = poly.eval(&lower)?.sign();
        let sm = poly.eval(&mid)?.sign();
        let su = poly.eval(&upper)?.sign();
        {
            let mut g = self.0.interval.write().expect("field lock poisoned");
            if sl == 0 {
                g.upper = g.lower.clone();
            } else if su == 0 {
                g.lower = g.upper.clone();
            } else if sm == 0 {
                g.lower = mid.clone();
                g.upper = mid;
            } else if sm == sl {
                g.lower = mid;
            } else {
                g.upper = mid;
            }
        }
        self.precompute_interval_powers();
        Ok(())
    }

    pub fn from_vector(&self, coeffs: Vec<Fraction>) -> Result<AlgebraicNumber> {
        AlgebraicNumber::new(self.clone(), polynomial(coeffs))
    }

    pub fn from_vector_i64(&self, coeffs: &[i64]) -> Result<AlgebraicNumber> {
        self.from_vector(coeffs.iter().map(|&c| Fraction::int(c)).collect())
    }
}

impl RingOps<AlgebraicNumber> for AlgebraicNumberField {
    fn zero(&self) -> AlgebraicNumber {
        AlgebraicNumber::new(self.clone(), polynomial(vec![])).expect("zero is always reducible")
    }
    fn one(&self) -> AlgebraicNumber {
        AlgebraicNumber::new(self.clone(), polynomial(vec![Fraction::int(1)])).expect("one is always reducible")
    }
    fn from_int(&self, n: i64) -> AlgebraicNumber {
        AlgebraicNumber::new(self.clone(), polynomial(vec![Fraction::int(n)])).expect("a constant is always reducible")
    }
}

impl core::fmt::Display for AlgebraicNumberField {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let g = self.0.interval.read().expect("field lock poisoned");
        write!(f, "Q(root of {} in [{},{}])", self.0.poly, g.lower, g.upper)
    }
}

/// `algebraicNumberField(poly, lower, upper)`
pub fn algebraic_number_field(poly: PolyQ, lower: Fraction, upper: Fraction) -> Result<AlgebraicNumberField> {
    AlgebraicNumberField::new(poly, lower, upper)
}

/// Number of `refine()` calls so far, for diagnosing divergence from the
/// caller. Refinement is a side effect shared by every
/// number in a field, so its call count is observable.
pub static REFINE_COUNT: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

/// Read and reset the refinement counter.
pub fn take_refine_count() -> usize {
    REFINE_COUNT.swap(0, core::sync::atomic::Ordering::Relaxed)
}

static QQ_NOTHING: OnceLock<AlgebraicNumberField> = OnceLock::new();

/// `QQ_nothing`: ℚ presented as the degree-one field ℚ(1).
///
/// A single shared instance, because its identity participates in
/// `check_same_field`'s degree-one fast path.
pub fn qq_nothing() -> &'static AlgebraicNumberField {
    QQ_NOTHING.get_or_init(|| {
        AlgebraicNumberField::new(
            polynomial(vec![Fraction::int(-1), Fraction::int(1)]),
            Fraction::int(1),
            Fraction::int(1),
        )
        .expect("QQ_nothing is well-formed")
    })
}

/* -------------------------------------------------------------------------- */

#[derive(Clone)]
pub struct AlgebraicNumber {
    pub field: AlgebraicNumberField,
    pub poly: PolyQ,
}

impl AlgebraicNumber {
    pub fn new(field: AlgebraicNumberField, poly: PolyQ) -> Result<AlgebraicNumber> {
        let poly = poly.into_modulo(&field.0.poly)?;
        Ok(AlgebraicNumber { field, poly })
    }

    pub fn is_zero(&self) -> bool {
        self.poly.degree == -1
    }

    /// `check_same_field`: returns the field the operation should happen in.
    pub fn check_same_field(a: &AlgebraicNumber, b: &AlgebraicNumber) -> Result<AlgebraicNumberField> {
        if a.field.equals(&b.field)? {
            return Ok(a.field.clone());
        }
        if a.field.degree() == 1 {
            return Ok(b.field.clone());
        }
        if b.field.degree() == 1 {
            return Ok(a.field.clone());
        }
        Err(Error::Range(format!(
            "AlgebraicNumbers must have same field ({} != {})",
            a.field, b.field
        )))
    }

    /// The value as an `f64`, correctly rounded.
    ///
    /// Refines the field until both ends of the isolating interval round to
    /// the same double, which is then the double the value rounds to as well:
    /// rounding to nearest is monotone, so everything between two ends that
    /// agree rounds where they do. The answer therefore does not depend on how
    /// much refining happened to have been done already, which is what lets
    /// the rest of the crate treat comparison order as an implementation
    /// detail.
    ///
    /// Refinement is shared through the field, so the work is done once per
    /// field instead of once per call.
    pub fn to_number(&self) -> Result<f64> {
        let k = &self.field;
        let (mut lo, mut hi) = self.interval()?;
        for _ in 0..256 {
            let (l, h) = (lo.to_f64_nearest(), hi.to_f64_nearest());
            if l == h {
                return Ok(l);
            }
            k.refine()?;
            let iv = self.interval()?;
            lo = iv.0;
            hi = iv.1;
        }
        Ok(lo.to_f64_nearest())
    }

    pub fn compare(&self, b: &AlgebraicNumber) -> Result<i32> {
        Elem::sub(self, b)?.sign()
    }

    /// Interval arithmetic evaluation of `self` from the field's current
    /// bounds on θ.
    pub fn interval(&self) -> Result<(Fraction, Fraction)> {
        let p = &self.poly;
        // Borrowed, not cloned: `interval` runs inside every `sign` loop.
        let g = self.field.0.interval.read().expect("field lock poisoned");
        let (powers_lower, powers_upper) = (&g.powers_lower, &g.powers_upper);
        let mut l = Fraction::int(0);
        let mut u = Fraction::int(0);
        for i in 0..=p.degree.max(-1) {
            if i < 0 {
                break;
            }
            let i = i as usize;
            let s = p.coeffs[i].sign();
            if s > 0 {
                l.iadd_r(&Elem::mul(&p.coeffs[i], &powers_lower[i])?, false);
                u.iadd_r(&Elem::mul(&p.coeffs[i], &powers_upper[i])?, false);
            } else if s < 0 {
                l.iadd_r(&Elem::mul(&p.coeffs[i], &powers_upper[i])?, false);
                u.iadd_r(&Elem::mul(&p.coeffs[i], &powers_lower[i])?, false);
            }
        }
        l.reduce();
        u.reduce();
        Ok((l, u))
    }

    pub fn sign(&self) -> Result<i32> {
        let k = &self.field;
        if self.poly.degree == -1 {
            return Ok(0);
        }
        let (mut val_l, mut val_u) = self.interval()?;
        while val_l.sign() != val_u.sign() {
            k.refine()?;
            let iv = self.interval()?;
            val_l = iv.0;
            val_u = iv.1;
        }
        Ok(val_l.sign())
    }

    pub fn abs(&self) -> Result<AlgebraicNumber> {
        if self.sign()? < 0 {
            Ok(Elem::neg(self))
        } else {
            Ok(self.clone())
        }
    }
}

impl Elem for AlgebraicNumber {
    #[inline]
    fn is_zero(&self) -> bool {
        self.poly.degree == -1
    }

    fn equals(&self, b: &AlgebraicNumber) -> bool {
        // `check_same_field` would error, but every caller here has
        // already established compatible fields.
        match AlgebraicNumber::check_same_field(self, b) {
            Ok(_) => Elem::equals(&self.poly, &b.poly),
            Err(_) => false,
        }
    }

    /// Note: unlike `add`, this does **not** re-reduce modulo the field
    /// polynomial. See `SEMANTICS.md` §6.
    fn iadd(&mut self, b: &AlgebraicNumber) -> Result<()> {
        self.field = AlgebraicNumber::check_same_field(self, b)?;
        Elem::iadd(&mut self.poly, &b.poly)
    }

    fn add(&self, b: &AlgebraicNumber) -> Result<AlgebraicNumber> {
        let k = AlgebraicNumber::check_same_field(self, b)?;
        AlgebraicNumber::new(k, Elem::add(&self.poly, &b.poly)?)
    }

    fn neg(&self) -> AlgebraicNumber {
        AlgebraicNumber {
            field: self.field.clone(),
            poly: Elem::neg(&self.poly),
        }
    }

    fn isub(&mut self, b: &AlgebraicNumber) -> Result<()> {
        self.field = AlgebraicNumber::check_same_field(self, b)?;
        Elem::isub(&mut self.poly, &b.poly)
    }

    fn sub(&self, b: &AlgebraicNumber) -> Result<AlgebraicNumber> {
        let k = AlgebraicNumber::check_same_field(self, b)?;
        AlgebraicNumber::new(k, Elem::sub(&self.poly, &b.poly)?)
    }

    /// Reduction via the precomputed powers of θ (Cohen), instead of a
    /// polynomial division.
    ///
    /// The straightforward version multiplies the two polynomials, then folds the terms
    /// above θ^(d-1) back down. This walks the product one coefficient at a
    /// time and folds each as it appears, which computes the same numbers in
    /// the same way without materializing the product. Folding a zero
    /// coefficient leaves the accumulator untouched down to its unreduced
    /// numerator and denominator, so skipping those is not an approximation,
    /// and it is also what keeps the `powers` index inside the table when an
    /// unreduced operand pushes the product's nominal degree past it.
    fn mul(&self, b: &AlgebraicNumber) -> Result<AlgebraicNumber> {
        let k = AlgebraicNumber::check_same_field(self, b)?;
        // No lock is taken: `degree` and `powers` are immutable for the life
        // of the field, and this is the hottest path in the library.
        let kdeg = k.0.degree;
        let powers = &k.0.powers;
        let (pa, pb) = (&self.poly, &b.poly);
        let (m, n) = (pa.degree, pb.degree);

        let mut coeffs: Vec<Fraction> = vec![QQ.zero(); kdeg.max(0) as usize];
        let mut t = 0isize;
        while t <= m + n {
            // The product's coefficient on x^t, reduced exactly as
            // `Polynomial::mul` reduces it.
            let mut ct = QQ.zero();
            let mut i = (t - n).max(0);
            while i <= t.min(m) {
                let term = Elem::mul(&pa.coeffs[i as usize], &pb.coeffs[(t - i) as usize])?;
                ct.iadd(&term)?;
                i += 1;
            }
            if t < kdeg {
                coeffs[t as usize] = ct;
            } else if !ct.is_zero() {
                let pw = &powers[t as usize];
                for j in 0..=pw.degree.max(-1) {
                    if j < 0 {
                        break;
                    }
                    let term = Elem::mul(&pw.coeffs[j as usize], &ct)?;
                    coeffs[j as usize].iadd_r(&term, false);
                }
            }
            t += 1;
        }
        for c in &mut coeffs {
            c.reduce();
        }
        AlgebraicNumber::new(k, polynomial(coeffs))
    }

    fn imul(&mut self, b: &AlgebraicNumber) -> Result<()> {
        let c = Elem::mul(self, b)?;
        self.field = c.field;
        self.poly = c.poly;
        Ok(())
    }

    fn inv(&self) -> Result<AlgebraicNumber> {
        let (r, _s, t) = extended_gcd(&qq_x(), &self.field.0.poly, &self.poly)?;
        if r.degree > 0 {
            return Err(Error::Range(format!("Division by zero: {self}")));
        }
        AlgebraicNumber::new(self.field.clone(), Elem::div(&t, &r)?)
    }

    fn div(&self, b: &AlgebraicNumber) -> Result<AlgebraicNumber> {
        Elem::mul(self, &Elem::inv(b)?)
    }

    fn idiv(&mut self, b: &AlgebraicNumber) -> Result<()> {
        let inv = Elem::inv(b)?;
        Elem::imul(self, &inv)
    }
}

impl AlgebraicNumber {
    /// Append a form that identifies this value, for memo keys.
    ///
    /// Not `Display`: the printed form spells the field out as
    /// `Q(root of {poly} in [{lower},{upper}])`, and those bounds are the
    /// field's *current* isolating interval. They are enormous once refined,
    /// and they move, so a key built from them both costs a great deal to
    /// build and stops matching the entry it was meant to find. The field's
    /// identity is its address, which says the same thing in a word.
    pub fn write_key(&self, out: &mut String) {
        use core::fmt::Write as _;
        let _ = write!(out, "{:x}#{}", self.field.id(), self.poly);
    }
}

impl core::fmt::Display for AlgebraicNumber {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} in {}", self.poly, self.field)
    }
}

impl PartialEq for AlgebraicNumber {
    fn eq(&self, other: &Self) -> bool {
        Elem::equals(self, other)
    }
}
impl Eq for AlgebraicNumber {}

/* -------------------------------------------------------------------------- */
/*  Field construction                                                        */
/* -------------------------------------------------------------------------- */

fn poly2(coeffs: Vec<Vec<Fraction>>) -> Poly2 {
    Polynomial::new(qq_x(), coeffs.into_iter().map(polynomial).collect())
}

/// Given the minimal polynomial `B` of β over K, find the minimal polynomial
/// of β over ℚ.
pub fn normal(b: &Polynomial<AlgebraicNumber, AlgebraicNumberField>) -> Result<PolyQ> {
    let a = b.coeff_ring.poly();

    // Change α into the inner variable, transposing inner and outer.
    let mut b_coeffs: Vec<Vec<Fraction>> = Vec::new();
    for i in 0..=a.degree.max(-1) {
        if i < 0 {
            break;
        }
        b_coeffs.push(
            b.coeffs
                .iter()
                .map(|c| {
                    if i <= c.poly.degree {
                        c.poly.coeffs[i as usize].clone()
                    } else {
                        Fraction::int(0)
                    }
                })
                .collect(),
        );
    }

    // Res_x(A(x), B(x,y))
    let a_bivar = poly2(a.coeffs.iter().map(|c| vec![c.clone()]).collect());
    let mut b_normal = resultant(&a_bivar, &poly2(b_coeffs))?;

    // Make squarefree.
    let g = poly_gcd(&b_normal, &b_normal.derivative()?)?;
    Elem::idiv(&mut b_normal, &g)?;
    b_normal.monic()
}

/// Given ℚ(α) and ℚ(β), construct ℚ(α,β).
///
/// Returns `(field, alpha, beta)` where `alpha` and `beta` are the primitive
/// elements of the two input fields, represented in the returned field.
pub fn extend(
    q_alpha: &AlgebraicNumberField,
    q_beta: &AlgebraicNumberField,
) -> Result<(AlgebraicNumberField, AlgebraicNumber, AlgebraicNumber)> {
    if q_alpha.equals(q_beta)? {
        return Ok((
            q_alpha.clone(),
            q_alpha.from_vector_i64(&[0, 1])?,
            q_beta.from_vector_i64(&[0, 1])?,
        ));
    }

    let a = q_alpha.poly();
    let b = q_beta.poly();

    // The primitive element has the form kα+β for some k.
    let mut k: i64 = 1;
    let c_mult: PolyQ;
    loop {
        let qq_x_y: Polynomials<PolyQ, QQx> = Polynomials::new(qq_x());
        let a_bivar = poly2(a.coeffs.iter().map(|c| vec![c.clone()]).collect());
        let b_bivar: Poly3 = b.map(qq_x_y, |c| Ok(poly2(vec![vec![c.clone()]])))?;
        let beta = poly2(vec![vec![Fraction::int(0), Fraction::int(1)], vec![Fraction::int(-k)]]); // z - kx
        let cand = resultant(&a_bivar, &b_bivar.eval(&beta)?)?; // Res_x(A(x), B(z-kx))
                                                                // A squarefree C_mult means the conjugates of kα+β are distinct, so
                                                                // kα+β is a primitive element of ℚ(α,β).
        if poly_gcd(&cand, &cand.derivative()?)?.degree == 0 {
            c_mult = cand;
            break;
        }
        k += 1;
    }

    // Pick the factor of C_mult having kα+β as a root: the minimal polynomial.
    let c_factors = factor(&c_mult)?;
    let mut c: Option<PolyQ> = None;
    let kf = Fraction::int(k);
    let interval = |q_a: &AlgebraicNumberField, q_b: &AlgebraicNumberField| -> Result<(Fraction, Fraction)> {
        let (al, au) = q_a.bounds();
        let (bl, bu) = q_b.bounds();
        let mut lower = Elem::mul(&al, &kf)?;
        lower.iadd_r(&bl, true);
        let mut upper = Elem::mul(&au, &kf)?;
        upper.iadd_r(&bu, true);
        Ok((lower, upper))
    };
    let (mut lower, mut upper) = interval(q_alpha, q_beta)?;
    loop {
        let mut total = 0;
        for factor in &c_factors {
            let count = count_roots(factor, &lower, &upper)?;
            total += count;
            if count > 0 {
                c = Some(factor.clone());
            }
        }
        if total == 1 {
            break;
        }
        q_alpha.refine()?;
        q_beta.refine()?;
        let iv = interval(q_alpha, q_beta)?;
        lower = iv.0;
        upper = iv.1;
    }
    let c = c.ok_or_else(|| Error::Other("Could not construct C (this shouldn't happen)".into()))?;

    let q_gamma = AlgebraicNumberField::new(c.clone(), lower, upper)?;

    // Choose the new field and represent α and β in it.
    let field;
    let alpha_vec;
    let beta_vec;

    let bv = extend_helper(
        q_alpha,
        b,
        &c,
        vec![vec![Fraction::int(0), Fraction::int(k)], vec![Fraction::int(1)]],
    )?;
    if let Some(bv) = bv {
        field = q_alpha.clone();
        alpha_vec = Some(q_alpha.from_vector_i64(&[0, 1])?);
        beta_vec = Some(bv);
    } else {
        let av = extend_helper(
            q_beta,
            a,
            &c,
            vec![vec![Fraction::int(0), Fraction::int(1)], vec![Fraction::int(k)]],
        )?;
        if let Some(av) = av {
            field = q_beta.clone();
            alpha_vec = Some(av);
            beta_vec = Some(q_beta.from_vector_i64(&[0, 1])?);
        } else {
            field = q_gamma.clone();
            alpha_vec = extend_helper(
                &q_gamma,
                a,
                b,
                vec![vec![Fraction::int(0), Fraction::int(1)], vec![Fraction::int(-k)]],
            )?;
            beta_vec = extend_helper(
                &q_gamma,
                b,
                a,
                vec![vec![Fraction::int(0), Fraction::of(1, k)], vec![Fraction::of(-1, k)]],
            )?;
        }
    }
    match (alpha_vec, beta_vec) {
        (Some(av), Some(bv)) => Ok((field, av, bv)),
        _ => Err(Error::Other(
            "Could not represent alpha and beta (this shouldn't happen)".into(),
        )),
    }
}

/// Test whether the roots of `B` lie in `Q_alpha`, representing β there as
/// `gcd(B(y), C(g(α, y)))`.
fn extend_helper(
    q_alpha: &AlgebraicNumberField,
    b: &PolyQ,
    c: &PolyQ,
    g: Vec<Vec<Fraction>>,
) -> Result<Option<AlgebraicNumber>> {
    let poly_alpha = |coeffs: Vec<Vec<Fraction>>| -> Result<Polynomial<AlgebraicNumber, AlgebraicNumberField>> {
        let mut cs = Vec::with_capacity(coeffs.len());
        for c in coeffs {
            cs.push(q_alpha.from_vector(c)?);
        }
        Ok(Polynomial::new(q_alpha.clone(), cs))
    };
    let q_alpha_x: Polynomials<AlgebraicNumber, AlgebraicNumberField> = Polynomials::new(q_alpha.clone());
    let b_alpha = poly_alpha(b.coeffs.iter().map(|c| vec![c.clone()]).collect())?;
    let c_alpha_x = c.map(q_alpha_x, |cc| poly_alpha(vec![vec![cc.clone()]]))?;
    let gamma = poly_alpha(g)?;
    let b_factor = poly_gcd(&b_alpha, &c_alpha_x.eval(&gamma)?)?;
    if b_factor.degree == 1 {
        Ok(Some(Elem::neg(&b_factor.monic()?.coeffs[0])))
    } else {
        Ok(None)
    }
}

/// Promote a list of algebraic numbers so they all live in the same field.
pub fn promote(xs: &mut [AlgebraicNumber]) -> Result<()> {
    for i in 1..xs.len() {
        let (field, alpha, beta) = extend(&xs[i - 1].field, &xs[i].field)?;
        if !field.is(&xs[i - 1].field) {
            for x in &mut xs[..i] {
                let mapped = x.poly.map(field.clone(), |c| field.from_vector(vec![c.clone()]))?;
                *x = mapped.eval(&alpha)?;
            }
        }
        if !field.is(&xs[i].field) {
            let mapped = xs[i].poly.map(field.clone(), |c| field.from_vector(vec![c.clone()]))?;
            xs[i] = mapped.eval(&beta)?;
        }
    }
    Ok(())
}

/// The `k`-th root of `x`, possibly in a larger number field.
pub fn root(x: &AlgebraicNumber, k: &Int) -> Result<AlgebraicNumber> {
    let field = x.field.clone();
    let mut coeffs = vec![Elem::neg(x)];
    let mut i = Int::from_i64(0);
    let km1 = k.sub(&Int::from_i64(1));
    while i.cmp(&km1) == core::cmp::Ordering::Less {
        coeffs.push(field.from_int(0));
        i = i.add(&Int::from_i64(1));
    }
    coeffs.push(field.from_int(1));
    let poly = normal(&Polynomial::new(field, coeffs))?;
    let mut lower = Fraction::int(0);
    // Open question: what if x is negative?
    let mut upper = Elem::add(&x.interval()?.1, &Fraction::int(1))?;
    while count_roots(&poly, &lower, &upper)? > 1 {
        let mid = lower.middle(&upper);
        let p = power(&QQ, &mid, k)?;
        if qq_nothing().from_vector(vec![p])?.compare(x)? < 0 {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    let new_field = AlgebraicNumberField::new(poly, lower, upper)?;
    new_field.from_vector_i64(&[0, 1])
}
