//! Minimal column-major linear algebra for the engine.
//!
//! Matrices are stored column-major as `[f32; 16]` so they can be handed
//! directly to WGSL (`mat4x4<f32>`) with no transpose. Element `(col, row)`
//! lives at index `col * 4 + row`.

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub fn sub(self, o: Vec3) -> Vec3 {
        vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn normalize(self) -> Vec3 {
        let len = self.dot(self).sqrt();
        if len <= f32::EPSILON {
            vec3(0.0, 0.0, 0.0)
        } else {
            let inv = 1.0 / len;
            vec3(self.x * inv, self.y * inv, self.z * inv)
        }
    }
}

/// Column-major 4x4 matrix.
#[derive(Clone, Copy, Debug)]
pub struct Mat4(pub [f32; 16]);

impl Mat4 {
    pub const fn identity() -> Mat4 {
        Mat4([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ])
    }

    /// `self * rhs` (both column-major).
    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let a = &self.0;
        let b = &rhs.0;
        let mut out = [0.0f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += a[k * 4 + row] * b[col * 4 + k];
                }
                out[col * 4 + row] = sum;
            }
        }
        Mat4(out)
    }

    pub fn translation(t: Vec3) -> Mat4 {
        let mut m = Mat4::identity();
        m.0[12] = t.x;
        m.0[13] = t.y;
        m.0[14] = t.z;
        m
    }

    pub fn scale(s: Vec3) -> Mat4 {
        let mut m = Mat4::identity();
        m.0[0] = s.x;
        m.0[5] = s.y;
        m.0[10] = s.z;
        m
    }

    /// Rotation about an arbitrary axis (normalized internally), angle in radians.
    pub fn rotation(axis: Vec3, angle: f32) -> Mat4 {
        Mat4::rotation_unit_axis(axis.normalize(), angle)
    }

    /// Rotation about an axis that is **already unit length** — skips the
    /// normalize (sqrt + divides). This is the per-instance hot path: axes are
    /// normalized once at scene build, not re-normalized every frame.
    pub fn rotation_unit_axis(a: Vec3, angle: f32) -> Mat4 {
        let (s, c) = angle.sin_cos();
        let t = 1.0 - c;
        let (x, y, z) = (a.x, a.y, a.z);
        Mat4([
            t * x * x + c,
            t * x * y + s * z,
            t * x * z - s * y,
            0.0,
            t * x * y - s * z,
            t * y * y + c,
            t * y * z + s * x,
            0.0,
            t * x * z + s * y,
            t * y * z - s * x,
            t * z * z + c,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ])
    }

    /// Compose `translation(pos) * rotation(axis, angle) * scale(s)` directly
    /// into 16 floats, avoiding two full 4x4 multiplies per call. This is the
    /// hot path when rebuilding thousands of instance transforms each frame.
    /// `axis` must already be unit length (instances normalize at scene build).
    pub fn trs(pos: Vec3, axis: Vec3, angle: f32, s: f32) -> Mat4 {
        let mut m = Mat4::rotation_unit_axis(axis, angle);
        // Scale the rotation basis columns (uniform scale).
        m.0[0] *= s;
        m.0[1] *= s;
        m.0[2] *= s;
        m.0[4] *= s;
        m.0[5] *= s;
        m.0[6] *= s;
        m.0[8] *= s;
        m.0[9] *= s;
        m.0[10] *= s;
        // Inject translation.
        m.0[12] = pos.x;
        m.0[13] = pos.y;
        m.0[14] = pos.z;
        m
    }

    /// Right-handed look-at view matrix.
    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
        let f = center.sub(eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Mat4([
            s.x,
            u.x,
            -f.x,
            0.0,
            s.y,
            u.y,
            -f.y,
            0.0,
            s.z,
            u.z,
            -f.z,
            0.0,
            -s.dot(eye),
            -u.dot(eye),
            f.dot(eye),
            1.0,
        ])
    }

    /// Perspective projection with a WebGPU-style depth range of [0, 1].
    pub fn perspective(fovy_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fovy_radians * 0.5).tan();
        let mut m = Mat4([0.0; 16]);
        m.0[0] = f / aspect;
        m.0[5] = f;
        m.0[10] = far / (near - far);
        m.0[11] = -1.0;
        m.0[14] = (far * near) / (near - far);
        m
    }
}
