//! Unit tests for the exact-arithmetic core: fractions, polynomials,
//! factorization and algebraic number fields.

use twistypuzzle::exact::{algebraic_number_field, extend, normal, qq_nothing, root, AlgebraicNumber};
use twistypuzzle::num::factoring::{combinations_vec, factor};
use twistypuzzle::num::fraction::Fraction;
use twistypuzzle::num::int::{Int, Primes};
use twistypuzzle::num::polynomial::{count_roots, polynomial_i64, resultant, Polynomial};
use twistypuzzle::num::ring::{gcd, power_mod, Elem, Euclidean, Integer, IntegersMod, RingOps, ZZ};

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 5e-3
}

#[test]
fn primes_and_sqrt() {
    let ps: Vec<String> = Primes::new()
        .take_while(|p| p.cmp(&Int::from_i64(10)) != std::cmp::Ordering::Greater)
        .map(|p| p.to_string())
        .collect();
    assert_eq!(ps, vec!["2", "3", "5", "7"]);
    assert_eq!(Int::from_i64(0).sqrt_upper().to_string(), "0");
    assert_eq!(Int::from_i64(1).sqrt_upper().to_string(), "1");
    assert_ne!(
        Int::from_i64(9999).sqrt_upper().cmp(&Int::from_i64(100)),
        std::cmp::Ordering::Less
    );
    assert_ne!(
        Int::from_i64(10001).sqrt_upper().cmp(&Int::from_i64(101)),
        std::cmp::Ordering::Less
    );
}

#[test]
fn fraction_to_string_and_number() {
    let cases: &[(i64, i64, &str)] = &[
        (-1, 1, "-1"),
        (0, 1, "0"),
        (1, 1, "1"),
        (-1, 2, "-1/2"),
        (1, 2, "1/2"),
        (-2, 3, "-2/3"),
        (2, 3, "2/3"),
        (-3, 4, "-3/4"),
        (3, 4, "3/4"),
        (-2, 2, "-1"),
        (0, 2, "0"),
        (2, 2, "1"),
        (-2, 4, "-1/2"),
        (2, 4, "1/2"),
    ];
    for &(n, d, s) in cases {
        let f = Fraction::of(n, d);
        assert_eq!(f.to_string(), s, "toString({n}/{d})");
        assert!(close(f.to_f64_nearest(), n as f64 / d as f64), "toNumber({n}/{d})");
    }
}

#[test]
fn fraction_arithmetic() {
    let vals: &[(i64, i64)] = &[(-1, 1), (0, 1), (1, 1), (-1, 2), (1, 2), (-2, 3), (2, 3), (3, 4)];
    for &(an, ad) in vals {
        for &(bn, bd) in vals {
            let (a, b) = (Fraction::of(an, ad), Fraction::of(bn, bd));
            let (af, bf) = (an as f64 / ad as f64, bn as f64 / bd as f64);
            assert!(close(Elem::add(&a, &b).unwrap().to_f64_nearest(), af + bf));
            assert!(close(Elem::sub(&a, &b).unwrap().to_f64_nearest(), af - bf));
            assert!(close(Elem::mul(&a, &b).unwrap().to_f64_nearest(), af * bf));
            if bn != 0 {
                assert!(close(Elem::div(&a, &b).unwrap().to_f64_nearest(), af / bf));
            } else {
                assert!(Elem::div(&a, &b).is_err());
            }
            assert_eq!(Elem::equals(&a, &b), af == bf);
        }
    }
}

#[test]
fn integer_and_integer_mod() {
    let int = |n: i64| Integer::from_i64(n);
    for (a, want) in [
        (0, 0),
        (1, 1),
        (9, 9),
        (10, 0),
        (11, 1),
        (-1, 9),
        (-9, 1),
        (-10, 0),
        (-11, 9),
    ] {
        assert_eq!(int(a).modulo(&int(10)).unwrap(), int(want), "{a} mod 10");
    }
    let m10 = IntegersMod::new(Int::from_i64(10));
    let md = |n: i64| m10.from_int(n);
    assert_eq!(Elem::add(&md(5), &md(7)).unwrap(), md(2));
    assert_eq!(Elem::sub(&md(5), &md(7)).unwrap(), md(8));
    assert_eq!(Elem::mul(&md(5), &md(7)).unwrap(), md(5));
    assert_eq!(Elem::div(&md(1), &md(3)).unwrap(), md(7));
    assert!(Elem::div(&md(1), &md(2)).is_err());
    assert_eq!(Elem::inv(&md(3)).unwrap(), md(7));
    assert!(Elem::inv(&md(2)).is_err());

    assert_eq!(power_mod(&ZZ, &int(2), &Int::from_i64(0), &int(100)).unwrap(), int(1));
    assert_eq!(power_mod(&ZZ, &int(2), &Int::from_i64(10), &int(100)).unwrap(), int(24));
    assert_eq!(gcd(&ZZ, &int(100), &int(128)).unwrap(), int(4));
}

#[test]
fn polynomial_to_string() {
    let cases: &[(&[i64], &str)] = &[
        (&[], "0"),
        (&[1], "1"),
        (&[-1], "-1"),
        (&[0, 1], "x"),
        (&[1, 1], "x + 1"),
        (&[-1, 1], "x + -1"),
        (&[1, -1], "-1x + 1"),
        (&[-1, -1], "-1x + -1"),
        (&[0, 2], "2x"),
        (&[0, 0, 1], "x²"),
        (&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], "x¹⁰"),
        (&[0], "0"),
        (&[1, 0], "1"),
    ];
    for &(c, s) in cases {
        assert_eq!(polynomial_i64(c).to_string(), s);
    }
}

#[test]
fn polynomial_divmod_and_roots() {
    let polys: &[&[i64]] = &[&[], &[1], &[-1], &[0, 1], &[1, 1], &[-1, 1], &[0, 2], &[0, 0, 1]];
    for a in polys {
        for b in polys {
            let (pa, pb) = (polynomial_i64(a), polynomial_i64(b));
            if pb.degree >= 0 {
                let (q, r) = pa.divmod(&pb).unwrap();
                let back = Elem::add(&Elem::mul(&pb, &q).unwrap(), &r).unwrap();
                assert!(Elem::equals(&back, &pa), "divmod {pa} / {pb}");
            } else {
                assert!(pa.divmod(&pb).is_err());
            }
        }
    }

    // count_roots over (x+3)x(x-3)
    let roots = [-3i64, 0, 3];
    let mut poly = polynomial_i64(&[1]);
    for r in roots {
        poly = Elem::mul(&poly, &polynomial_i64(&[-r, 1])).unwrap();
    }
    let points = [-4i64, -3, -2, -1, 0, 1, 2, 3, 4];
    for i in 0..points.len() {
        for j in i..points.len() {
            let want = roots.iter().filter(|&&r| points[i] <= r && r <= points[j]).count() as i32;
            let got = count_roots(&poly, &Fraction::int(points[i]), &Fraction::int(points[j])).unwrap();
            assert_eq!(got, want, "count_roots [{},{}]", points[i], points[j]);
        }
    }
}

#[test]
fn resultant_is_the_product_of_the_second_poly_over_the_first_roots() {
    let mk = |rs: &[i64]| {
        let mut p = polynomial_i64(&[1]);
        for &r in rs {
            p = Elem::mul(&p, &polynomial_i64(&[-r, 1])).unwrap();
        }
        p
    };
    let p1 = mk(&[1, 3, 5, 7]);
    let p2 = mk(&[2, 4, 6, 8]);
    assert_eq!(resultant(&p1, &p2).unwrap().to_string(), "212625");
    assert_eq!(resultant(&p1, &p1).unwrap().to_string(), "0");
}

#[test]
fn combinations_count() {
    assert_eq!(combinations_vec(&[1, 2, 3, 4, 5], 3).len(), 10);
}

#[test]
fn factor_matches_reference() {
    let cases: &[(&[i64], &[&[i64]])] = &[
        (&[2, 3, 1], &[&[1, 1], &[2, 1]]),
        (
            &[112, 368, 681, 933, 1395, 1692, 1775, 1482, 1335, 959, 577, 156, 135],
            &[&[4, 9, 8, 1, 3], &[4, 3, 1, 3, 5], &[7, 2, 9, 2, 9]],
        ),
        (
            &[7, 16, 44, 48, 49, 38, 43, 5, 9],
            &[&[1, 1, 4, 0, 1], &[7, 9, 7, 5, 9]],
        ),
        (
            &[105, 263, 731, 1252, 1878, 2395, 2487, 2186, 1615, 954, 382, 96, 8],
            &[&[7, 4, 6, 8, 1], &[3, 4, 9, 5, 2], &[5, 3, 6, 6, 4]],
        ),
    ];
    for (coeffs, want) in cases {
        let mut got: Vec<String> = factor(&polynomial_i64(coeffs))
            .unwrap()
            .iter()
            .map(ToString::to_string)
            .collect();
        let mut expect: Vec<String> = want.iter().map(|cs| polynomial_i64(cs).to_string()).collect();
        got.sort();
        expect.sort();
        assert_eq!(got, expect, "factor({coeffs:?})");
    }
}

/// ℚ(√2,√3), as used throughout `exact.test.ts`.
fn k_2_3() -> twistypuzzle::exact::AlgebraicNumberField {
    algebraic_number_field(
        polynomial_i64(&[1, 0, -10, 0, 1]),
        Fraction::of(3146264, 1000000),
        Fraction::of(3146265, 1000000),
    )
    .unwrap()
}

#[test]
fn algebraic_number_arithmetic() {
    let k = k_2_3();
    let vals: Vec<(Vec<Fraction>, f64)> = vec![
        (vec![Fraction::int(0); 4], 0.0),
        (
            vec![Fraction::int(1), Fraction::int(0), Fraction::int(0), Fraction::int(0)],
            1.0,
        ),
        (
            vec![Fraction::int(-1), Fraction::int(0), Fraction::int(0), Fraction::int(0)],
            -1.0,
        ),
        (
            vec![
                Fraction::int(0),
                Fraction::of(-9, 2),
                Fraction::int(0),
                Fraction::of(1, 2),
            ],
            std::f64::consts::SQRT_2,
        ),
        (
            vec![
                Fraction::int(0),
                Fraction::of(-9, 4),
                Fraction::int(0),
                Fraction::of(1, 4),
            ],
            std::f64::consts::SQRT_2 / 2.0,
        ),
        (
            vec![
                Fraction::int(0),
                Fraction::of(11, 2),
                Fraction::int(0),
                Fraction::of(-1, 2),
            ],
            3f64.sqrt(),
        ),
    ];
    for (c, want) in &vals {
        let x = k.from_vector(c.clone()).unwrap();
        assert!(close(x.to_number().unwrap(), *want), "toNumber {} != {}", x.poly, want);
        assert_eq!(x.sign().unwrap(), want.partial_cmp(&0.0).map(|o| o as i32).unwrap());
    }
    for (ac, av) in &vals {
        for (bc, bv) in &vals {
            let a = k.from_vector(ac.clone()).unwrap();
            let b = k.from_vector(bc.clone()).unwrap();
            assert!(close(Elem::add(&a, &b).unwrap().to_number().unwrap(), av + bv));
            assert!(close(Elem::sub(&a, &b).unwrap().to_number().unwrap(), av - bv));
            assert!(close(Elem::mul(&a, &b).unwrap().to_number().unwrap(), av * bv));
            if b.is_zero() {
                assert!(Elem::div(&a, &b).is_err());
            } else {
                assert!(close(Elem::div(&a, &b).unwrap().to_number().unwrap(), av / bv));
            }
        }
    }
}

#[test]
fn normal_of_x_squared_minus_sqrt2() {
    let k = k_2_3();
    let sqrt2 = k
        .from_vector(vec![
            Fraction::int(0),
            Fraction::of(-9, 2),
            Fraction::int(0),
            Fraction::of(1, 2),
        ])
        .unwrap();
    let p = Polynomial::new(
        k.clone(),
        vec![
            Elem::neg(&sqrt2),
            k.from_vector(vec![]).unwrap(),
            k.from_vector_i64(&[1]).unwrap(),
        ],
    );
    assert!(Elem::equals(&normal(&p).unwrap(), &polynomial_i64(&[-2, 0, 0, 0, 1])));
}

#[test]
fn extend_covers_every_pair() {
    let fields = vec![
        qq_nothing().clone(),
        algebraic_number_field(
            polynomial_i64(&[-2, 0, 1]),
            Fraction::of(1414213, 1000000),
            Fraction::of(1414214, 1000000),
        )
        .unwrap(),
        algebraic_number_field(
            polynomial_i64(&[-3, 0, 1]),
            Fraction::of(1732050, 1000000),
            Fraction::of(1732051, 1000000),
        )
        .unwrap(),
        algebraic_number_field(
            polynomial_i64(&[-5, 0, 1]),
            Fraction::of(2236067, 1000000),
            Fraction::of(2236068, 1000000),
        )
        .unwrap(),
        k_2_3(),
        algebraic_number_field(
            polynomial_i64(&[4, 0, -16, 0, 1]),
            Fraction::of(3968118, 1000000),
            Fraction::of(3968119, 1000000),
        )
        .unwrap(),
    ];
    for qa in &fields {
        for qb in &fields {
            let (qg, alpha, beta) = extend(qa, qb).unwrap();
            // Each primitive element must be a root of its own minimal polynomial.
            let pa = qa.poly().map(qg.clone(), |c| qg.from_vector(vec![c.clone()])).unwrap();
            assert!(pa.eval(&alpha).unwrap().is_zero(), "alpha is not a root");
            let pb = qb.poly().map(qg.clone(), |c| qg.from_vector(vec![c.clone()])).unwrap();
            assert!(pb.eval(&beta).unwrap().is_zero(), "beta is not a root");
        }
    }
}

#[test]
fn pentagon_constants_land_in_the_expected_field() {
    let num = |x: i64| qq_nothing().from_int(x);
    let sq = |x: &AlgebraicNumber| root(x, &Int::from_i64(2)).unwrap();

    // C0 = sqrt(10*(5-sqrt(5)))/20
    let c0 = Elem::div(
        &sq(&Elem::mul(&num(10), &Elem::sub(&num(5), &sq(&num(5))).unwrap()).unwrap()),
        &num(20),
    )
    .unwrap();
    let c1 = Elem::div(
        &sq(&Elem::mul(
            &num(5),
            &Elem::add(&num(5), &Elem::mul(&num(2), &sq(&num(5))).unwrap()).unwrap(),
        )
        .unwrap()),
        &num(10),
    )
    .unwrap();
    let c2 = Elem::div(&Elem::add(&num(1), &sq(&num(5))).unwrap(), &num(4)).unwrap();
    let c3 = Elem::div(
        &sq(&Elem::mul(&num(10), &Elem::add(&num(5), &sq(&num(5))).unwrap()).unwrap()),
        &num(10),
    )
    .unwrap();

    let (k0, _, _) = extend(qq_nothing(), &c0.field).unwrap();
    let (k1, _, _) = extend(&k0, &c1.field).unwrap();
    let (k2, _, _) = extend(&k1, &c2.field).unwrap();
    let (k3, _, _) = extend(&k2, &c3.field).unwrap();
    assert!(
        Elem::equals(k3.poly(), &polynomial_i64(&[2000, 0, -100, 0, 1])),
        "{}",
        k3.poly()
    );
}

/// Every conversion to `f64` is correctly rounded: `Fraction::to_f64_nearest`
/// and, built on it, `AlgebraicNumber::to_number`. There is one routine and it
/// is the accurate one.
#[test]
fn precise_conversion_is_correctly_rounded() {
    // Rationals whose nearest double the truncating division
    // undershoots, alongside ones where the two agree.
    for (n, d) in [
        (1i64, 3i64),
        (2, 3),
        (1, 7),
        (-1, 3),
        (355, 113),
        (1, 10),
        (7, 8),
        (0, 1),
        (-22, 7),
        (1, 1_000_003),
    ] {
        let f = Fraction::new(Int::from_i64(n), Int::from_i64(d), true).unwrap();
        assert_eq!(f.to_f64_nearest(), n as f64 / d as f64, "{n}/{d} rounded wrong");
    }

    // Algebraic numbers, against the platform's own libm. These agree to the
    // last bit because both are correctly rounded, not because libm is being
    // trusted for the value.
    let cases: [(&str, f64); 6] = [
        ("sqrt(2)", std::f64::consts::SQRT_2),
        ("sqrt(3)", 1.732_050_807_568_877_2),
        ("(1+sqrt(5))/2", 1.618_033_988_749_895),
        ("sqrt(5)/3", 0.745_355_992_499_929_9),
        ("-sqrt(2)", -std::f64::consts::SQRT_2),
        ("sqrt(2)*sqrt(3)", 2.449_489_742_783_178),
    ];
    for (expr, want) in cases {
        let e = twistypuzzle::parse::parse_expr(expr).unwrap();
        let x = twistypuzzle::parse::eval_expr(&e).unwrap();
        assert_eq!(x.to_number().unwrap(), want, "{expr}");
    }

    // Refining for precision must not touch the shared field, because the
    // own `to_number` has to answer identically before and after.
    let e = twistypuzzle::parse::parse_expr("sqrt(2)").unwrap();
    let x = twistypuzzle::parse::eval_expr(&e).unwrap();
    let before = x.to_number().unwrap();
    let bounds_before = x.field.bounds();
    let _ = x.to_number().unwrap();
    assert_eq!(x.to_number().unwrap(), before);
    let bounds_after = x.field.bounds();
    assert_eq!(bounds_before.0.to_string(), bounds_after.0.to_string());
    assert_eq!(bounds_before.1.to_string(), bounds_after.1.to_string());

    // Exact zero stays exact, and the cheap routine stays truncating.
    let zero = twistypuzzle::parse::eval_expr(&twistypuzzle::parse::parse_expr("sqrt(2)-sqrt(2)").unwrap()).unwrap();
    assert_eq!(zero.to_number().unwrap(), 0.0);
    assert_eq!(
        Fraction::new(Int::from_i64(1), Int::from_i64(3), true)
            .unwrap()
            .to_f64_nearest(),
        1.0 / 3.0
    );
}
