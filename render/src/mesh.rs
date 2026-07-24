//! Static mesh geometry, generated in WASM and uploaded once by JS.
//!
//! Vertex layout is interleaved `position: vec3, normal: vec3` — a 24-byte
//! stride that matches the vertex buffer layout declared on the JS side.

/// One cube face: 4 vertices (pos + normal), CCW when viewed from outside.
fn face(out: &mut Vec<f32>, quad: [[f32; 3]; 4], normal: [f32; 3]) {
    for v in quad.iter() {
        out.extend_from_slice(v);
        out.extend_from_slice(&normal);
    }
}

/// Returns interleaved vertex data for a unit cube centered at the origin.
pub fn cube_vertices() -> Vec<f32> {
    let mut v = Vec::with_capacity(24 * 6);
    let p = 0.5;
    // +X
    face(&mut v, [[p, -p, -p], [p, -p, p], [p, p, p], [p, p, -p]], [1.0, 0.0, 0.0]);
    // -X
    face(&mut v, [[-p, -p, p], [-p, -p, -p], [-p, p, -p], [-p, p, p]], [-1.0, 0.0, 0.0]);
    // +Y
    face(&mut v, [[-p, p, -p], [p, p, -p], [p, p, p], [-p, p, p]], [0.0, 1.0, 0.0]);
    // -Y
    face(&mut v, [[-p, -p, p], [p, -p, p], [p, -p, -p], [-p, -p, -p]], [0.0, -1.0, 0.0]);
    // +Z
    face(&mut v, [[-p, -p, p], [-p, p, p], [p, p, p], [p, -p, p]], [0.0, 0.0, 1.0]);
    // -Z
    face(&mut v, [[p, -p, -p], [p, p, -p], [-p, p, -p], [-p, -p, -p]], [0.0, 0.0, -1.0]);
    v
}

/// Returns 36 indices (two triangles per face) for the cube above.
pub fn cube_indices() -> Vec<u32> {
    let mut idx = Vec::with_capacity(36);
    for f in 0..6u32 {
        let b = f * 4;
        idx.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    idx
}

/// A scalable rectangle (flat grid) in the local XY plane, centered at the
/// origin, facing +Z. `width`/`height` scale it in X/Y; `nx`/`ny` are the
/// **vertex counts** along each axis (clamped to ≥ 2), so the sheet is made of
/// `(nx-1) * (ny-1)` quads. Same interleaved `position, normal` layout as the
/// cube/sphere, wound CCW when viewed from +Z. Handy as a ground plane, a wall,
/// or the render mesh for a cloth grid of the same vertex counts.
pub fn plane(width: f32, height: f32, nx: u32, ny: u32) -> (Vec<f32>, Vec<u32>) {
    let nx = nx.max(2);
    let ny = ny.max(2);
    let normal = [0.0f32, 0.0, 1.0];

    let mut verts = Vec::with_capacity((nx * ny * 6) as usize);
    for j in 0..ny {
        // v runs 0..1 across the height; centered so the rectangle straddles 0.
        let ty = j as f32 / (ny - 1) as f32;
        let y = (ty - 0.5) * height;
        for i in 0..nx {
            let tx = i as f32 / (nx - 1) as f32;
            let x = (tx - 0.5) * width;
            verts.extend_from_slice(&[x, y, 0.0]);
            verts.extend_from_slice(&normal);
        }
    }

    let mut idx = Vec::with_capacity(((nx - 1) * (ny - 1) * 6) as usize);
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let a = j * nx + i; // this row
            let b = a + nx; // next row
            // Two CCW triangles per quad (front face toward +Z).
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    (verts, idx)
}

/// A unit-radius UV sphere centered at the origin, same interleaved
/// `position, normal` layout as the cube (normal == position for a unit sphere).
/// `rings` = latitude stacks, `sectors` = longitude slices. Faces wind CCW when
/// viewed from outside, matching the cube so the same back-face-culled pipeline
/// draws it. Used by the physics ballpit; the cube demo never calls this.
pub fn uv_sphere(rings: u32, sectors: u32) -> (Vec<f32>, Vec<u32>) {
    use core::f32::consts::{PI, TAU};
    let rings = rings.max(2);
    let sectors = sectors.max(3);
    let stride = sectors + 1;

    let mut verts = Vec::with_capacity(((rings + 1) * stride * 6) as usize);
    for i in 0..=rings {
        let phi = (i as f32 / rings as f32) * PI; // 0 (north pole) .. PI (south)
        let (sp, cp) = phi.sin_cos();
        for j in 0..=sectors {
            let theta = (j as f32 / sectors as f32) * TAU;
            let (st, ct) = theta.sin_cos();
            // Position on the unit sphere; the normal is the same vector.
            let (x, y, z) = (sp * ct, cp, sp * st);
            verts.extend_from_slice(&[x, y, z, x, y, z]);
        }
    }

    let mut idx = Vec::with_capacity((rings * sectors * 6) as usize);
    for i in 0..rings {
        for j in 0..sectors {
            let a = i * stride + j;
            let b = a + stride;
            // Wound so the geometric normal points outward (CCW from outside).
            idx.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    (verts, idx)
}
