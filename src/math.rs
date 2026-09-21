//! Exact 3-D geometry: vectors, planes and quaternions over an algebraic
//! number field.
//!
//! The floating-point projections (`to_f64`, `to_plane_f64`, `to_quat_f64`)
//! normalize as they convert, because that is what the renderer downstream
//! expects of them.

use crate::exact::{qq_nothing, AlgebraicNumber};
use crate::num::ring::{Elem, RingOps};
use crate::{console_assert, Result};

/// A double-precision 3-vector.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    pub fn normalize(&self) -> Vec3 {
        let l = self.length();
        if l == 0.0 {
            Vec3::default()
        } else {
            Vec3::new(self.x / l, self.y / l, self.z / l)
        }
    }
    pub fn dot(&self, o: &Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn sub(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn scale(&self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn distance_to(&self, o: &Vec3) -> f64 {
        self.sub(o).length()
    }
    /// `Vector3.applyQuaternion`
    pub fn apply_quat(&self, q: &Quat) -> Vec3 {
        let (x, y, z) = (self.x, self.y, self.z);
        let (qx, qy, qz, qw) = (q.x, q.y, q.z, q.w);
        let tx = 2.0 * (qy * z - qz * y);
        let ty = 2.0 * (qz * x - qx * z);
        let tz = 2.0 * (qx * y - qy * x);
        Vec3::new(
            x + qw * tx + qy * tz - qz * ty,
            y + qw * ty + qz * tx - qx * tz,
            z + qw * tz + qx * ty - qy * tx,
        )
    }
}

/// A double-precision quaternion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Default for Quat {
    fn default() -> Self {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Quat {
        Quat { x, y, z, w }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    pub fn normalize(&self) -> Quat {
        let l = self.length();
        if l == 0.0 {
            Quat::IDENTITY
        } else {
            let l = 1.0 / l;
            Quat::new(self.x * l, self.y * l, self.z * l, self.w * l)
        }
    }

    /// `a.multiply(b)`: Hamilton product, three's argument order.
    pub fn mul(&self, b: &Quat) -> Quat {
        let (ax, ay, az, aw) = (self.x, self.y, self.z, self.w);
        let (bx, by, bz, bw) = (b.x, b.y, b.z, b.w);
        Quat::new(
            ax * bw + aw * bx + ay * bz - az * by,
            ay * bw + aw * by + az * bx - ax * bz,
            az * bw + aw * bz + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    }

    /// `setFromAxisAngle(axis, angle)`: `axis` must be normalized.
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Quat {
        let half = angle / 2.0;
        let s = half.sin();
        Quat::new(axis.x * s, axis.y * s, axis.z * s, half.cos())
    }

    /// `setFromUnitVectors(from, to)`
    pub fn from_unit_vectors(from: &Vec3, to: &Vec3) -> Quat {
        const EPS: f64 = 0.000001;
        let mut r = from.dot(to) + 1.0;
        let q = if r < EPS {
            r = 0.0;
            if from.x.abs() > from.z.abs() {
                Quat::new(-from.y, from.x, 0.0, r)
            } else {
                Quat::new(0.0, -from.z, from.y, r)
            }
        } else {
            let c = from.cross(to);
            Quat::new(c.x, c.y, c.z, r)
        };
        q.normalize()
    }

    /// `slerpQuaternions(a, b, t)`
    pub fn slerp(a: &Quat, b: &Quat, t: f64) -> Quat {
        if t == 0.0 {
            return *a;
        }
        if t == 1.0 {
            return *b;
        }
        let (x, y, z, w) = (a.x, a.y, a.z, a.w);
        let mut cos_half_theta = w * b.w + x * b.x + y * b.y + z * b.z;
        let (bx, by, bz, bw) = if cos_half_theta < 0.0 {
            cos_half_theta = -cos_half_theta;
            (-b.x, -b.y, -b.z, -b.w)
        } else {
            (b.x, b.y, b.z, b.w)
        };
        if cos_half_theta >= 1.0 {
            return Quat::new(x, y, z, w);
        }
        let sqr_sin_half_theta = 1.0 - cos_half_theta * cos_half_theta;
        if sqr_sin_half_theta <= f64::EPSILON {
            let s = 1.0 - t;
            let q = Quat::new(s * x + t * bx, s * y + t * by, s * z + t * bz, s * w + t * bw);
            return q.normalize();
        }
        let sin_half_theta = sqr_sin_half_theta.sqrt();
        let half_theta = sin_half_theta.atan2(cos_half_theta);
        let ratio_a = ((1.0 - t) * half_theta).sin() / sin_half_theta;
        let ratio_b = (t * half_theta).sin() / sin_half_theta;
        Quat::new(
            x * ratio_a + bx * ratio_b,
            y * ratio_a + by * ratio_b,
            z * ratio_a + bz * ratio_b,
            w * ratio_a + bw * ratio_b,
        )
    }
}

/// A double-precision plane, always stored normalized.
#[derive(Clone, Copy, Debug)]
pub struct PlaneF64 {
    pub normal: Vec3,
    pub constant: f64,
}

impl PlaneF64 {
    pub fn distance_to_point(&self, p: &Vec3) -> f64 {
        self.normal.dot(p) + self.constant
    }
}

/* -------------------------------------------------------------------------- */
/*  Exact types                                                               */
/* -------------------------------------------------------------------------- */

#[derive(Clone)]
pub struct ExactVector3 {
    pub x: AlgebraicNumber,
    pub y: AlgebraicNumber,
    pub z: AlgebraicNumber,
}

impl ExactVector3 {
    pub fn new(x: AlgebraicNumber, y: AlgebraicNumber, z: AlgebraicNumber) -> ExactVector3 {
        ExactVector3 { x, y, z }
    }

    pub fn to_f64(&self) -> Result<Vec3> {
        Ok(Vec3::new(self.x.to_number()?, self.y.to_number()?, self.z.to_number()?))
    }

    /// Sign of the first non-zero component, used to canonicalize planes.
    pub fn pseudo_sign(&self) -> Result<i32> {
        if self.x.sign()? != 0 {
            self.x.sign()
        } else if self.y.sign()? != 0 {
            self.y.sign()
        } else {
            self.z.sign()
        }
    }

    pub fn equals(&self, other: &ExactVector3) -> bool {
        Elem::equals(&self.x, &other.x) && Elem::equals(&self.y, &other.y) && Elem::equals(&self.z, &other.z)
    }

    pub fn add(&self, o: &ExactVector3) -> Result<ExactVector3> {
        Ok(ExactVector3::new(
            Elem::add(&self.x, &o.x)?,
            Elem::add(&self.y, &o.y)?,
            Elem::add(&self.z, &o.z)?,
        ))
    }

    pub fn sub(&self, o: &ExactVector3) -> Result<ExactVector3> {
        Ok(ExactVector3::new(
            Elem::sub(&self.x, &o.x)?,
            Elem::sub(&self.y, &o.y)?,
            Elem::sub(&self.z, &o.z)?,
        ))
    }

    pub fn neg(&self) -> ExactVector3 {
        ExactVector3::new(Elem::neg(&self.x), Elem::neg(&self.y), Elem::neg(&self.z))
    }

    pub fn scale(&self, o: &AlgebraicNumber) -> Result<ExactVector3> {
        Ok(ExactVector3::new(
            Elem::mul(&self.x, o)?,
            Elem::mul(&self.y, o)?,
            Elem::mul(&self.z, o)?,
        ))
    }

    pub fn dot(&self, o: &ExactVector3) -> Result<AlgebraicNumber> {
        let a = Elem::mul(&self.x, &o.x)?;
        let b = Elem::mul(&self.y, &o.y)?;
        let c = Elem::mul(&self.z, &o.z)?;
        Elem::add(&Elem::add(&a, &b)?, &c)
    }

    pub fn cross(&self, o: &ExactVector3) -> Result<ExactVector3> {
        Ok(ExactVector3::new(
            Elem::sub(&Elem::mul(&self.y, &o.z)?, &Elem::mul(&self.z, &o.y)?)?,
            Elem::sub(&Elem::mul(&self.z, &o.x)?, &Elem::mul(&self.x, &o.z)?)?,
            Elem::sub(&Elem::mul(&self.x, &o.y)?, &Elem::mul(&self.y, &o.x)?)?,
        ))
    }
}

impl ExactVector3 {
    /// Append a memo-key form. See [`AlgebraicNumber::write_key`].
    pub fn write_key(&self, out: &mut String) {
        out.push('[');
        self.x.write_key(out);
        out.push(',');
        self.y.write_key(out);
        out.push(',');
        self.z.write_key(out);
        out.push(']');
    }
}

impl core::fmt::Display for ExactVector3 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{},{},{}]", self.x, self.y, self.z)
    }
}

/// A plane whose normal need *not* be normalized, unlike its
/// floating-point counterpart: normalizing would need a square root, which
/// the field may not contain.
#[derive(Clone)]
pub struct ExactPlane {
    pub normal: ExactVector3,
    pub constant: AlgebraicNumber,
}

impl ExactPlane {
    pub fn new(normal: ExactVector3, constant: AlgebraicNumber) -> ExactPlane {
        ExactPlane { normal, constant }
    }

    pub fn to_f64(&self) -> Result<PlaneF64> {
        let n = self.normal.to_f64()?;
        let c = self.constant.to_number()?;
        // Normalized on the way out, which is what the renderer expects.
        let inv = 1.0 / n.length();
        Ok(PlaneF64 {
            normal: n.scale(inv),
            constant: c * inv,
        })
    }

    pub fn neg(&self) -> ExactPlane {
        ExactPlane::new(self.normal.neg(), Elem::neg(&self.constant))
    }

    /// Which side of the plane `v` lies on: +1 front, -1 back, 0 on it.
    pub fn side(&self, v: &ExactVector3) -> Result<i32> {
        Elem::add(&self.normal.dot(v)?, &self.constant)?.sign()
    }

    /// Intersection of the plane with the line through `a` and `b`.
    pub fn intersect_line(&self, a: &ExactVector3, b: &ExactVector3) -> Result<ExactVector3> {
        // n-x + d = 0, x = a + (b-a)t  =>  t = -(n-a + d) / n-(b-a)
        let ab = b.sub(a)?;
        let t = Elem::div(
            &Elem::add(&self.normal.dot(a)?, &self.constant)?,
            &self.normal.dot(&ab)?,
        )?;
        a.sub(&ab.scale(&t)?)
    }

    /// The "absolute value" of a plane, so it compares equal to its negation.
    pub fn canonicalize(&self) -> Result<ExactPlane> {
        if self.normal.pseudo_sign()? < 0 {
            Ok(self.neg())
        } else {
            Ok(self.clone())
        }
    }
}

impl ExactPlane {
    /// Append a memo-key form. See [`AlgebraicNumber::write_key`].
    pub fn write_key(&self, out: &mut String) {
        out.push('[');
        self.normal.write_key(out);
        out.push(',');
        self.constant.write_key(out);
        out.push(']');
    }
}

impl core::fmt::Display for ExactPlane {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{},{}]", self.normal, self.constant)
    }
}

#[derive(Clone)]
pub struct ExactQuaternion {
    pub x: AlgebraicNumber,
    pub y: AlgebraicNumber,
    pub z: AlgebraicNumber,
    pub w: AlgebraicNumber,
}

impl ExactQuaternion {
    pub fn new(x: AlgebraicNumber, y: AlgebraicNumber, z: AlgebraicNumber, w: AlgebraicNumber) -> ExactQuaternion {
        ExactQuaternion { x, y, z, w }
    }

    pub fn identity() -> ExactQuaternion {
        let zero = qq_nothing().from_int(0);
        let one = qq_nothing().from_int(1);
        ExactQuaternion::new(zero.clone(), zero.clone(), zero, one)
    }

    /// Given an axis `k` (not necessarily normalized) and `x`, `y` with
    /// `‖x‖ = ‖y‖` and `k-x = k-y`, the rotation about `k` taking `x` to `y`.
    pub fn from_axis_points(k: &ExactVector3, x: &ExactVector3, y: &ExactVector3) -> Result<ExactQuaternion> {
        let kxy = k.dot(&x.cross(y)?)?;
        let _kx = k.dot(x)?; // = k-y
        let xy = x.dot(y)?;
        if kxy.is_zero() {
            if xy.sign()? > 0 {
                Ok(ExactQuaternion::identity()) // 0 degrees
            } else {
                // 180 degrees
                Ok(ExactQuaternion::new(
                    k.x.clone(),
                    k.y.clone(),
                    k.z.clone(),
                    qq_nothing().from_int(0),
                ))
            }
        } else {
            let w = Elem::div(&kxy, &Elem::sub(&x.dot(x)?, &xy)?)?;
            Ok(ExactQuaternion::new(k.x.clone(), k.y.clone(), k.z.clone(), w))
        }
    }

    pub fn to_f64(&self) -> Result<Quat> {
        Ok(Quat::new(
            self.x.to_number()?,
            self.y.to_number()?,
            self.z.to_number()?,
            self.w.to_number()?,
        )
        .normalize())
    }

    pub fn equals(&self, o: &ExactQuaternion) -> bool {
        Elem::equals(&self.x, &o.x)
            && Elem::equals(&self.y, &o.y)
            && Elem::equals(&self.z, &o.z)
            && Elem::equals(&self.w, &o.w)
    }

    pub fn norm_squared(&self) -> Result<AlgebraicNumber> {
        let (x, y, z, w) = (&self.x, &self.y, &self.z, &self.w);
        let xx = Elem::mul(x, x)?;
        let yy = Elem::mul(y, y)?;
        let zz = Elem::mul(z, z)?;
        let ww = Elem::mul(w, w)?;
        Elem::add(&ww, &Elem::add(&Elem::add(&xx, &yy)?, &zz)?)
    }

    pub fn approx_angle(&self) -> Result<f64> {
        let x = self.x.to_number()?;
        let y = self.y.to_number()?;
        let z = self.z.to_number()?;
        let w = self.w.to_number()?;
        // fdlibm's acos, not the host libm's: see `crate::fdlibm`.
        Ok(crate::fdlibm::acos(w / (x * x + y * y + z * z + w * w).sqrt()) * 2.0)
    }

    /// A monotone but cheap stand-in for the rotation angle, used for sorting.
    pub fn pseudo_angle(&self) -> Result<AlgebraicNumber> {
        let one = qq_nothing().from_int(1);
        let w = &self.w;
        let t = Elem::div(&Elem::mul(w, &w.abs()?)?, &self.norm_squared()?)?;
        Elem::sub(&one, &t)
    }

    pub fn mul(&self, b: &ExactQuaternion) -> Result<ExactQuaternion> {
        let a = self;
        let t = |p: &AlgebraicNumber, q: &AlgebraicNumber| Elem::mul(p, q);
        Ok(ExactQuaternion::new(
            Elem::sub(
                &Elem::add(&Elem::add(&t(&a.w, &b.x)?, &t(&a.x, &b.w)?)?, &t(&a.y, &b.z)?)?,
                &t(&a.z, &b.y)?,
            )?,
            Elem::add(
                &Elem::add(&Elem::sub(&t(&a.w, &b.y)?, &t(&a.x, &b.z)?)?, &t(&a.y, &b.w)?)?,
                &t(&a.z, &b.x)?,
            )?,
            Elem::add(
                &Elem::sub(&Elem::add(&t(&a.w, &b.z)?, &t(&a.x, &b.y)?)?, &t(&a.y, &b.x)?)?,
                &t(&a.z, &b.w)?,
            )?,
            Elem::sub(
                &Elem::sub(&Elem::sub(&t(&a.w, &b.w)?, &t(&a.x, &b.x)?)?, &t(&a.y, &b.y)?)?,
                &t(&a.z, &b.z)?,
            )?,
        ))
    }

    pub fn scale(&self, b: &AlgebraicNumber) -> Result<ExactQuaternion> {
        Ok(ExactQuaternion::new(
            Elem::mul(&self.x, b)?,
            Elem::mul(&self.y, b)?,
            Elem::mul(&self.z, b)?,
            Elem::mul(&self.w, b)?,
        ))
    }

    pub fn conj(&self) -> ExactQuaternion {
        ExactQuaternion::new(
            Elem::neg(&self.x),
            Elem::neg(&self.y),
            Elem::neg(&self.z),
            self.w.clone(),
        )
    }

    pub fn inv(&self) -> Result<ExactQuaternion> {
        self.conj().scale(&Elem::inv(&self.norm_squared()?)?)
    }

    pub fn apply(&self, v: &ExactVector3) -> Result<ExactVector3> {
        let zero = qq_nothing().from_int(0);
        let vq = ExactQuaternion::new(v.x.clone(), v.y.clone(), v.z.clone(), zero);
        let vr = self.mul(&vq)?.mul(&self.inv()?)?;
        console_assert!(
            vr.w.is_zero(),
            "ExactQuaternion::apply: rotated vector has a w component"
        );
        Ok(ExactVector3::new(vr.x, vr.y, vr.z))
    }

    /// Scale so that the L1 norm is 1 and `w >= 0`, keeping coefficients small
    /// without needing a square root.
    pub fn pseudo_normalize(&self) -> Result<ExactQuaternion> {
        let (x, y, z, w) = (&self.x, &self.y, &self.z, &self.w);
        let mut n = Elem::add(&Elem::add(&Elem::add(&w.abs()?, &x.abs()?)?, &y.abs()?)?, &z.abs()?)?;
        if w.sign()? < 0 {
            n = Elem::neg(&n);
        }
        self.scale(&Elem::inv(&n)?)
    }
}

impl ExactQuaternion {
    /// Append a memo-key form. See [`AlgebraicNumber::write_key`].
    pub fn write_key(&self, out: &mut String) {
        out.push('[');
        self.x.write_key(out);
        out.push(',');
        self.y.write_key(out);
        out.push(',');
        self.z.write_key(out);
        out.push(',');
        self.w.write_key(out);
        out.push(']');
    }
}

impl core::fmt::Display for ExactQuaternion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "[{},{},{},{}]", self.x, self.y, self.z, self.w)
    }
}
