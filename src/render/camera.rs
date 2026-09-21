//! Matrices and the perspective camera, with the conventions the rest of the
//! renderer assumes: column-major storage and a right-handed view space.

use crate::math::{Quat, Vec3};

/// A 4x4 matrix in column-major order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// `e[column * 4 + row]`
    pub e: [f64; 16],
}

impl Default for Mat4 {
    fn default() -> Self {
        Mat4::IDENTITY
    }
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        e: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };

    /// `a * b`, applying `b` first.
    pub fn mul(&self, b: &Mat4) -> Mat4 {
        let a = &self.e;
        let bb = &b.e;
        let mut e = [0.0f64; 16];
        for c in 0..4 {
            for r in 0..4 {
                e[c * 4 + r] =
                    a[r] * bb[c * 4] + a[4 + r] * bb[c * 4 + 1] + a[8 + r] * bb[c * 4 + 2] + a[12 + r] * bb[c * 4 + 3];
            }
        }
        Mat4 { e }
    }

    /// Transform a point, returning homogeneous `(x, y, z, w)`.
    #[inline]
    pub fn transform_point4(&self, v: Vec3) -> [f64; 4] {
        let e = &self.e;
        [
            e[0] * v.x + e[4] * v.y + e[8] * v.z + e[12],
            e[1] * v.x + e[5] * v.y + e[9] * v.z + e[13],
            e[2] * v.x + e[6] * v.y + e[10] * v.z + e[14],
            e[3] * v.x + e[7] * v.y + e[11] * v.z + e[15],
        ]
    }

    /// `compose(position, quaternion, scale)`
    pub fn compose(position: Vec3, q: Quat, scale: Vec3) -> Mat4 {
        let (x, y, z, w) = (q.x, q.y, q.z, q.w);
        let (x2, y2, z2) = (x + x, y + y, z + z);
        let (xx, xy, xz) = (x * x2, x * y2, x * z2);
        let (yy, yz, zz) = (y * y2, y * z2, z * z2);
        let (wx, wy, wz) = (w * x2, w * y2, w * z2);
        let (sx, sy, sz) = (scale.x, scale.y, scale.z);
        Mat4 {
            e: [
                (1.0 - (yy + zz)) * sx,
                (xy + wz) * sx,
                (xz - wy) * sx,
                0.0,
                (xy - wz) * sy,
                (1.0 - (xx + zz)) * sy,
                (yz + wx) * sy,
                0.0,
                (xz + wy) * sz,
                (yz - wx) * sz,
                (1.0 - (xx + yy)) * sz,
                0.0,
                position.x,
                position.y,
                position.z,
                1.0,
            ],
        }
    }

    /// General inverse, by cofactor expansion.
    ///
    /// Returns the zero matrix for a singular input instead of erroring.
    pub fn invert(&self) -> Mat4 {
        let m = &self.e;
        let (n11, n21, n31, n41) = (m[0], m[1], m[2], m[3]);
        let (n12, n22, n32, n42) = (m[4], m[5], m[6], m[7]);
        let (n13, n23, n33, n43) = (m[8], m[9], m[10], m[11]);
        let (n14, n24, n34, n44) = (m[12], m[13], m[14], m[15]);

        let t11 =
            n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        let t12 =
            n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        let t13 =
            n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        let t14 =
            n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
        if det == 0.0 {
            return Mat4 { e: [0.0; 16] };
        }
        let d = 1.0 / det;
        let mut e = [0.0f64; 16];
        e[0] = t11 * d;
        e[1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44
            - n21 * n33 * n44)
            * d;
        e[2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44
            + n21 * n32 * n44)
            * d;
        e[3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43
            - n21 * n32 * n43)
            * d;
        e[4] = t12 * d;
        e[5] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44
            + n11 * n33 * n44)
            * d;
        e[6] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44
            - n11 * n32 * n44)
            * d;
        e[7] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43
            + n11 * n32 * n43)
            * d;
        e[8] = t13 * d;
        e[9] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44
            - n11 * n23 * n44)
            * d;
        e[10] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44
            + n11 * n22 * n44)
            * d;
        e[11] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43
            - n11 * n22 * n43)
            * d;
        e[12] = t14 * d;
        e[13] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34
            + n11 * n23 * n34)
            * d;
        e[14] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34
            - n11 * n22 * n34)
            * d;
        e[15] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33
            + n11 * n22 * n33)
            * d;
        Mat4 { e }
    }

    /// `makePerspective(left, right, top, bottom, near, far)` for the WebGL
    /// coordinate system (clip-space z in `[-1, 1]`).
    pub fn perspective(left: f64, right: f64, top: f64, bottom: f64, near: f64, far: f64) -> Mat4 {
        let x = 2.0 * near / (right - left);
        let y = 2.0 * near / (top - bottom);
        let a = (right + left) / (right - left);
        let b = (top + bottom) / (top - bottom);
        let c = -(far + near) / (far - near);
        let d = -2.0 * far * near / (far - near);
        Mat4 {
            e: [x, 0.0, 0.0, 0.0, 0.0, y, 0.0, 0.0, a, b, c, -1.0, 0.0, 0.0, d, 0.0],
        }
    }

    /// The world-to-camera matrix for an eye looking at `target`.
    ///
    /// This is the inverse of the camera's world matrix, built directly from
    /// the orthonormal basis instead of composing and inverting.
    pub fn look_at_view(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let mut z = eye.sub(&target);
        if z.dot(&z) == 0.0 {
            z.z = 1.0;
        }
        z = z.normalize();
        let mut x = up.cross(&z);
        if x.dot(&x) == 0.0 {
            // `up` and `z` are parallel: nudge, so the cross product exists.
            let mut zz = z;
            if up.z.abs() == 1.0 {
                zz.x += 0.0001;
            } else {
                zz.z += 0.0001;
            }
            z = zz.normalize();
            x = up.cross(&z);
        }
        x = x.normalize();
        let y = z.cross(&x);
        // Rotation transposed, translation negated.
        Mat4 {
            e: [
                x.x,
                y.x,
                z.x,
                0.0,
                x.y,
                y.y,
                z.y,
                0.0,
                x.z,
                y.z,
                z.z,
                0.0,
                -x.dot(&eye),
                -y.dot(&eye),
                -z.dot(&eye),
                1.0,
            ],
        }
    }
}

/// A perspective camera, with the trackball's eye/up state folded in.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Vertical field of view, in degrees.
    pub fov: f64,
    pub aspect: f64,
    pub near: f64,
    pub far: f64,
    pub zoom: f64,
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
}

impl Default for Camera {
    /// A 15° vertical field of view at `z = 12`, which frames a unit sphere with a little room.
    fn default() -> Camera {
        Camera {
            fov: 15.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            zoom: 1.0,
            position: Vec3::new(0.0, 0.0, 12.0),
            target: Vec3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

impl Camera {
    pub fn projection_matrix(&self) -> Mat4 {
        let top = self.near * (self.fov.to_radians() * 0.5).tan() / self.zoom;
        let height = 2.0 * top;
        let width = self.aspect * height;
        let left = -0.5 * width;
        Mat4::perspective(left, left + width, top, top - height, self.near, self.far)
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_view(self.position, self.target, self.up)
    }

    /// World-to-clip.
    pub fn view_projection(&self) -> Mat4 {
        self.projection_matrix().mul(&self.view_matrix())
    }
}
