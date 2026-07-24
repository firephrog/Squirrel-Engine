//! Soft bodies + cloth — a small XPBD (Extended Position Based Dynamics) solver.
//!
//! Rapier is a rigid-body solver and has no cloth / deformable support, so this
//! is our own, kept deliberately separate from the Rapier pipeline: a soft body
//! is just a bag of point masses (`Particle`) wired together by distance
//! constraints (`Constraint`). XPBD is chosen over a raw mass-spring system
//! because it is unconditionally stable (no stiff-spring explosion — see the
//! same numerical caution the aero forces take) and its `compliance` parameter
//! maps cleanly onto "how stretchy is this", independent of timestep.
//!
//! Both shapes the engine exposes are the *same* `SoftBody` type, differing only
//! in how their particles + constraints are generated:
//!   * `build_cloth`    — a scalable rectangle of `nu * nv` particles (a sheet).
//!   * `build_soft_box` — a solid `nx * ny * nz` lattice (a squishy block).
//!
//! Collisions are resolved one-way as position projections against the pit
//! (floor + walls) and the rigid spheres (balls + player) read from Rapier each
//! tick, so cloth drapes over the pile and the player without us having to feed
//! forces back into Rapier's solver.

use rapier3d::prelude::*;

/// One point mass. XPBD integrates position directly; `prev` is last substep's
/// position (velocity is derived from `pos - prev`). `inv_mass == 0` pins it.
#[derive(Clone)]
struct Particle {
    pos: Vector,
    prev: Vector,
    vel: Vector,
    inv_mass: Real,
}

/// A distance constraint holding two particles at `rest` length. `compliance`
/// is inverse stiffness (0 = rigid); `lambda` is the XPBD Lagrange multiplier,
/// reset every substep.
#[derive(Clone)]
struct Constraint {
    a: usize,
    b: usize,
    rest: Real,
    compliance: Real,
    lambda: Real,
}

/// A deformable body: particles + the constraints that bind them, plus the
/// customizable simulation parameters (all set at creation — see `build_cloth`
/// / `build_soft_box`).
pub struct SoftBody {
    particles: Vec<Particle>,
    constraints: Vec<Constraint>,
    /// Surface triangle list (local particle indices) so the renderer can draw
    /// the body as a real deforming mesh instead of loose points: the full grid
    /// for cloth, the six outer faces for a soft box.
    indices: Vec<u32>,
    /// Shape-matching rest offsets (each particle relative to the rest centre of
    /// mass). Non-empty only for bodies that use shape matching (soft boxes);
    /// cloth leaves it empty and relies on distance constraints alone.
    rest: Vec<Vector>,
    /// Per-substep pull toward the best-fit rigid rest pose (0 = off). This is
    /// what lets a soft box recover after a big squish instead of staying
    /// crumpled: distance constraints alone cannot prevent inversion (a mirrored
    /// cell has identical edge lengths), but matching a rigid rest pose can.
    shape_stiffness: Real,
    /// Warm-started rotation (quaternion x,y,z,w) for the shape-match solve.
    rot: [f32; 4],
    /// Velocity drag per second (0 = none). Keeps a sheet from ringing forever.
    damping: Real,
    /// Constraint-solver iterations per substep (higher = stiffer / less stretch).
    iterations: u32,
    /// XPBD substeps per tick (higher = more accurate, more stable, costlier).
    substeps: u32,
    /// Collision thickness: particles keep this clearance from surfaces.
    particle_radius: Real,
    /// Sliding friction at contacts, 0 (ice) .. 1 (grippy).
    friction: Real,
    /// Sphere radius the renderer draws for each particle.
    render_radius: f32,
    /// Cosmetic tint seed handed to JS (unused by the sim itself).
    pub color_seed: f32,
    /// Per-object physics toggle (matches the rigid-body `enabled` flag). When
    /// off, `solve` is a no-op so the body freezes exactly where it is.
    pub enabled: bool,
}

impl SoftBody {
    pub fn particle_count(&self) -> usize {
        self.particles.len()
    }

    pub fn render_radius(&self) -> f32 {
        self.render_radius
    }

    /// The surface triangle list (local particle indices) for mesh rendering.
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Append this body's particle positions to a packed `[x, y, z, radius]`
    /// render buffer (the layout the scene turns into draw instances).
    pub fn pack_into(&self, out: &mut Vec<f32>) {
        for p in &self.particles {
            out.extend_from_slice(&[p.pos.x, p.pos.y, p.pos.z, self.render_radius]);
        }
    }

    /// Advance the body by one tick of `dt`, colliding against the pit `bounds`
    /// (`[min_x, max_x, min_z, max_z, floor_y, wall_top]`) and the rigid
    /// `spheres` (`(center, radius)`). No-op while disabled.
    ///
    /// `contacts` is parallel to `spheres`: for each sphere the solver sums the
    /// push it received from this body (direction, depth, and the soft surface's
    /// own velocity) so the caller can bounce the rigid body back — the two-way
    /// coupling that turns a pinned bed into a trampoline. Because the bed is
    /// pinned to the pit, the reaction momentum flows into the pins (the ground),
    /// so the caller may apply restitution without a matching push on the bed.
    pub fn solve(
        &mut self,
        gravity: Vector,
        dt: Real,
        spheres: &[(Vector, Real)],
        contacts: &mut [RigidContact],
        bounds: Option<[Real; 6]>,
    ) {
        if !self.enabled || dt <= 0.0 {
            return;
        }
        let sub = self.substeps.max(1);
        let sdt = dt / sub as Real;
        let drag = (1.0 - self.damping * sdt).clamp(0.0, 1.0);

        for _ in 0..sub {
            // 1. Integrate: apply gravity + drag, then predict positions.
            for p in &mut self.particles {
                if p.inv_mass == 0.0 {
                    p.prev = p.pos;
                    continue;
                }
                p.vel += gravity * sdt;
                p.vel *= drag;
                p.prev = p.pos;
                p.pos += p.vel * sdt;
            }

            // 2. Solve distance constraints (XPBD), reset multipliers first.
            for c in &mut self.constraints {
                c.lambda = 0.0;
            }
            let inv_sdt2 = 1.0 / (sdt * sdt);
            for _ in 0..self.iterations {
                for ci in 0..self.constraints.len() {
                    let (a, b, rest, compliance, lambda) = {
                        let c = &self.constraints[ci];
                        (c.a, c.b, c.rest, c.compliance, c.lambda)
                    };
                    let wa = self.particles[a].inv_mass;
                    let wb = self.particles[b].inv_mass;
                    let w = wa + wb;
                    if w == 0.0 {
                        continue;
                    }
                    let delta = self.particles[a].pos - self.particles[b].pos;
                    let len = delta.length();
                    if len < 1.0e-9 {
                        continue;
                    }
                    let dir = delta / len;
                    let cval = len - rest;
                    let alpha = compliance * inv_sdt2;
                    let dlambda = (-cval - alpha * lambda) / (w + alpha);
                    self.constraints[ci].lambda = lambda + dlambda;
                    let corr = dir * dlambda;
                    self.particles[a].pos += corr * wa;
                    self.particles[b].pos -= corr * wb;
                }
            }

            // 2b. Shape matching: pull toward the best-fit rigid rest pose so a
            // soft box recovers its shape and can never stay inverted/crumpled.
            if self.shape_stiffness > 0.0 && !self.rest.is_empty() {
                self.shape_match(self.shape_stiffness);
            }

            // 3. Collisions: project particles out of the pit + rigid spheres,
            //    accumulating the reaction on each sphere for the trampoline pass.
            for p in &mut self.particles {
                if p.inv_mass == 0.0 {
                    continue;
                }
                resolve_collisions(
                    p,
                    self.particle_radius,
                    self.friction,
                    spheres,
                    contacts,
                    sdt,
                    bounds,
                );
            }

            // 4. Derive velocity from the net position change this substep.
            let inv_sdt = 1.0 / sdt;
            for p in &mut self.particles {
                if p.inv_mass == 0.0 {
                    p.vel = Vector::ZERO;
                    continue;
                }
                p.vel = (p.pos - p.prev) * inv_sdt;
            }
        }
    }
}

impl SoftBody {
    /// Pull every particle a fraction `stiffness` toward the best-fit rigid
    /// transform of the rest shape (Müller et al. shape matching). The rotation
    /// is extracted from the covariance of current vs. rest offsets, warm-started
    /// from last step's rotation for speed + stability. Because the target is a
    /// *rigid* pose (proper rotation, no reflection), the body always relaxes
    /// back to its rest shape and can never lock into an inverted/crumpled state.
    fn shape_match(&mut self, stiffness: Real) {
        let n = self.particles.len();
        if n == 0 {
            return;
        }
        // Current centre of mass (uniform particle weights).
        let mut com = Vector::ZERO;
        for p in &self.particles {
            com += p.pos;
        }
        com /= n as Real;

        // Covariance A = Σ (x_i - com) ⊗ rest_i, stored as its three columns.
        let mut a0 = Vector::ZERO;
        let mut a1 = Vector::ZERO;
        let mut a2 = Vector::ZERO;
        for i in 0..n {
            let p = self.particles[i].pos - com;
            let q = self.rest[i];
            a0 += p * q.x;
            a1 += p * q.y;
            a2 += p * q.z;
        }

        let rot = extract_rotation(a0, a1, a2, self.rot);
        self.rot = rot;

        for i in 0..n {
            if self.particles[i].inv_mass == 0.0 {
                continue;
            }
            let goal = com + quat_rotate(rot, self.rest[i]);
            let cur = self.particles[i].pos;
            self.particles[i].pos += (goal - cur) * stiffness;
        }
    }
}

/// One soft-body → rigid sphere contact, summed over a tick. The caller uses it
/// to bounce the sphere (ball / player) off the soft surface: a pinned bed then
/// behaves like a springy floor (a trampoline). The push momentum is absorbed by
/// the bed's pins, so no equal-and-opposite force is fed back into the bed.
#[derive(Clone, Copy)]
pub struct RigidContact {
    /// Summed push direction on the sphere (away from the surface), weighted by
    /// penetration. Normalize for the average contact normal.
    pub normal: Vector,
    /// Deepest penetration seen this tick, in metres.
    pub pen: Real,
    /// Summed soft-surface velocity at the contacts — averaged by `count`, this
    /// lets a rebounding bed launch the sphere instead of merely cushioning it.
    pub bed_vel: Vector,
    /// Number of particle contacts summed in (0 = the sphere touched nothing).
    pub count: u32,
}

impl RigidContact {
    pub fn zero() -> Self {
        RigidContact { normal: Vector::ZERO, pen: 0.0, bed_vel: Vector::ZERO, count: 0 }
    }
}

/// Project one particle out of every surface it penetrates, applying friction by
/// bleeding off the tangential part of the motion it made this substep, and
/// record the reaction on each rigid sphere into `contacts` (parallel to
/// `spheres`) for the two-way trampoline bounce.
fn resolve_collisions(
    p: &mut Particle,
    r: Real,
    friction: Real,
    spheres: &[(Vector, Real)],
    contacts: &mut [RigidContact],
    sdt: Real,
    bounds: Option<[Real; 6]>,
) {
    // Pit floor + walls.
    if let Some([min_x, max_x, min_z, max_z, floor_y, _wall_top]) = bounds {
        if p.pos.y - r < floor_y {
            p.pos.y = floor_y + r;
            apply_friction(p, Vector::new(0.0, 1.0, 0.0), friction);
        }
        if p.pos.x - r < min_x {
            p.pos.x = min_x + r;
        } else if p.pos.x + r > max_x {
            p.pos.x = max_x - r;
        }
        if p.pos.z - r < min_z {
            p.pos.z = min_z + r;
        } else if p.pos.z + r > max_z {
            p.pos.z = max_z - r;
        }
    }

    // Rigid spheres (balls + player): push the particle to the surface (one-way,
    // so the bed deforms around the body) and log the reaction so the world can
    // push the body back the other way (the trampoline effect).
    let inv_sdt = 1.0 / sdt;
    for (j, &(center, radius)) in spheres.iter().enumerate() {
        let d = p.pos - center;
        let min_dist = radius + r;
        let dist2 = d.length_squared();
        if dist2 < min_dist * min_dist && dist2 > 1.0e-12 {
            let dist = dist2.sqrt();
            let n = d / dist; // sphere centre → particle
            let pen = min_dist - dist;

            // Reaction on the sphere points *away* from the particle (−n), so a
            // bed cradling a ball from below pushes the ball up. Weight the
            // summed normal by penetration; carry the particle's velocity so a
            // springing bed can fling the body.
            let vel = (p.pos - p.prev) * inv_sdt;
            let c = &mut contacts[j];
            c.normal -= n * pen;
            c.bed_vel += vel;
            c.pen = c.pen.max(pen);
            c.count += 1;

            p.pos = center + n * min_dist;
            apply_friction(p, n, friction);
        }
    }
}

/// Remove `friction` of the tangential motion the particle made this substep,
/// so contacts grip instead of sliding freely. `n` is the (unit) contact normal.
fn apply_friction(p: &mut Particle, n: Vector, friction: Real) {
    if friction <= 0.0 {
        return;
    }
    let motion = p.pos - p.prev;
    let normal_part = n * motion.dot(n);
    let tangential = motion - normal_part;
    // Pull `prev` toward `pos` along the tangent so the derived velocity loses
    // that fraction of its sliding component.
    p.prev += tangential * friction.clamp(0.0, 1.0);
}

/// Build a cloth sheet: a scalable rectangle of `nu * nv` particles spanning
/// `width` along `u` and `height` along `v` from `origin` (a corner). Structural
/// (grid), shear (diagonal) and bend (skip-one) constraints hold its shape.
/// `pin_mode`: 0 = free, 1 = pin the `v == 0` edge, 2 = pin the two `v == 0`
/// corners, 3 = pin all four corners, 4 = pin the whole border (a taut bed /
/// trampoline).
#[allow(clippy::too_many_arguments)]
pub fn build_cloth(
    origin: Vector,
    u: Vector,
    v: Vector,
    width: Real,
    height: Real,
    nu: u32,
    nv: u32,
    total_mass: Real,
    compliance: Real,
    bend_compliance: Real,
    damping: Real,
    particle_radius: Real,
    render_radius: f32,
    friction: Real,
    iterations: u32,
    substeps: u32,
    pin_mode: u32,
    color_seed: f32,
) -> SoftBody {
    let nu = nu.max(2);
    let nv = nv.max(2);
    let un = safe_normalize(u);
    let vn = safe_normalize(v);
    let du = width / (nu - 1) as Real;
    let dv = height / (nv - 1) as Real;

    let count = (nu * nv) as usize;
    let inv_mass = if total_mass > 0.0 {
        count as Real / total_mass
    } else {
        1.0
    };
    let idx = |i: u32, j: u32| (j * nu + i) as usize;

    let mut particles = Vec::with_capacity(count);
    for j in 0..nv {
        for i in 0..nu {
            let pos = origin + un * (du * i as Real) + vn * (dv * j as Real);
            let pinned = match pin_mode {
                1 => j == 0,
                2 => j == 0 && (i == 0 || i == nu - 1),
                3 => (i == 0 || i == nu - 1) && (j == 0 || j == nv - 1),
                // Whole border pinned — a taut bed (trampoline) held on all sides.
                4 => i == 0 || i == nu - 1 || j == 0 || j == nv - 1,
                _ => false,
            };
            particles.push(Particle {
                pos,
                prev: pos,
                vel: Vector::ZERO,
                inv_mass: if pinned { 0.0 } else { inv_mass },
            });
        }
    }

    let mut constraints = Vec::new();
    let mut add = |a: usize, b: usize, comp: Real, ps: &[Particle]| {
        let rest = (ps[a].pos - ps[b].pos).length();
        constraints.push(Constraint { a, b, rest, compliance: comp, lambda: 0.0 });
    };
    for j in 0..nv {
        for i in 0..nu {
            let a = idx(i, j);
            // Structural: right + down neighbors.
            if i + 1 < nu {
                add(a, idx(i + 1, j), compliance, &particles);
            }
            if j + 1 < nv {
                add(a, idx(i, j + 1), compliance, &particles);
            }
            // Shear: both diagonals of each quad.
            if i + 1 < nu && j + 1 < nv {
                add(a, idx(i + 1, j + 1), compliance, &particles);
                add(idx(i + 1, j), idx(i, j + 1), compliance, &particles);
            }
            // Bend: skip-one neighbors, softer, to resist folding.
            if i + 2 < nu {
                add(a, idx(i + 2, j), bend_compliance, &particles);
            }
            if j + 2 < nv {
                add(a, idx(i, j + 2), bend_compliance, &particles);
            }
        }
    }

    // Surface triangles: two per grid quad. The mesh is drawn double-sided, so
    // the winding is cosmetic (normals are recomputed from geometry each frame).
    let mut indices = Vec::with_capacity(((nu - 1) * (nv - 1) * 6) as usize);
    for j in 0..nv - 1 {
        for i in 0..nu - 1 {
            let a = idx(i, j) as u32;
            let b = idx(i, j + 1) as u32;
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }

    SoftBody {
        particles,
        constraints,
        indices,
        // Cloth is a thin sheet: distance/bend constraints alone hold it, and a
        // rigid rest pose would fight the drape, so shape matching stays off.
        rest: Vec::new(),
        shape_stiffness: 0.0,
        rot: [0.0, 0.0, 0.0, 1.0],
        damping,
        iterations,
        substeps,
        particle_radius,
        friction,
        render_radius,
        color_seed,
        enabled: true,
    }
}

/// Build a squishy solid block: an `nx * ny * nz` particle lattice centered at
/// `center` with full extents `size`. Axis-aligned edges (structural) plus every
/// face diagonal (shear) give it volume and jiggle. `pin_top` fixes the top face
/// so it can dangle; otherwise it is free to tumble.
#[allow(clippy::too_many_arguments)]
pub fn build_soft_box(
    center: Vector,
    size: Vector,
    nx: u32,
    ny: u32,
    nz: u32,
    total_mass: Real,
    compliance: Real,
    damping: Real,
    particle_radius: Real,
    render_radius: f32,
    friction: Real,
    iterations: u32,
    substeps: u32,
    pin_top: bool,
    shape_stiffness: Real,
    color_seed: f32,
) -> SoftBody {
    let nx = nx.max(2);
    let ny = ny.max(2);
    let nz = nz.max(2);
    let count = (nx * ny * nz) as usize;
    let inv_mass = if total_mass > 0.0 {
        count as Real / total_mass
    } else {
        1.0
    };
    let half = size * 0.5;
    let step = Vector::new(
        size.x / (nx - 1) as Real,
        size.y / (ny - 1) as Real,
        size.z / (nz - 1) as Real,
    );
    let idx = |i: u32, j: u32, k: u32| ((k * ny + j) * nx + i) as usize;

    let mut particles = Vec::with_capacity(count);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let pos = center - half
                    + Vector::new(step.x * i as Real, step.y * j as Real, step.z * k as Real);
                let pinned = pin_top && j == ny - 1;
                particles.push(Particle {
                    pos,
                    prev: pos,
                    vel: Vector::ZERO,
                    inv_mass: if pinned { 0.0 } else { inv_mass },
                });
            }
        }
    }

    let mut constraints = Vec::new();
    let mut add = |a: usize, b: usize, ps: &[Particle]| {
        let rest = (ps[a].pos - ps[b].pos).length();
        constraints.push(Constraint { a, b, rest, compliance, lambda: 0.0 });
    };
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let a = idx(i, j, k);
                // Structural edges along each axis.
                if i + 1 < nx {
                    add(a, idx(i + 1, j, k), &particles);
                }
                if j + 1 < ny {
                    add(a, idx(i, j + 1, k), &particles);
                }
                if k + 1 < nz {
                    add(a, idx(i, j, k + 1), &particles);
                }
                // Face diagonals (shear) hold the lattice from collapsing.
                if i + 1 < nx && j + 1 < ny {
                    add(a, idx(i + 1, j + 1, k), &particles);
                    add(idx(i + 1, j, k), idx(i, j + 1, k), &particles);
                }
                if j + 1 < ny && k + 1 < nz {
                    add(a, idx(i, j + 1, k + 1), &particles);
                    add(idx(i, j + 1, k), idx(i, j, k + 1), &particles);
                }
                if i + 1 < nx && k + 1 < nz {
                    add(a, idx(i + 1, j, k + 1), &particles);
                    add(idx(i + 1, j, k), idx(i, j, k + 1), &particles);
                }
                // Volume (body) diagonals: the four space diagonals of each cell.
                // These are what keep a *soft* box from crumpling — without them
                // a distance-only lattice buckles and inverts under load; with
                // them each cell resists collapse, so it squashes and springs
                // back instead of folding up like foil.
                if i + 1 < nx && j + 1 < ny && k + 1 < nz {
                    add(a, idx(i + 1, j + 1, k + 1), &particles);
                    add(idx(i + 1, j, k), idx(i, j + 1, k + 1), &particles);
                    add(idx(i, j + 1, k), idx(i + 1, j, k + 1), &particles);
                    add(idx(i, j, k + 1), idx(i + 1, j + 1, k), &particles);
                }
            }
        }
    }

    // Surface triangles over the six outer faces of the lattice, so the box
    // draws as a solid deforming skin (interior particles are simulated but not
    // referenced by the mesh). Two triangles per boundary cell. Every face must
    // wind so its geometric normal points *outward*: the renderer smooth-shades
    // by area-averaging face normals at shared edge/corner vertices, so a face
    // wound inward cancels its neighbours into garbage normals and shatters the
    // shading. `flip` reverses a quad to (a,d,c,b) for the three faces whose
    // base winding would otherwise point inward.
    let mut indices = Vec::new();
    let quad = |a: u32, b: u32, c: u32, d: u32, flip: bool, out: &mut Vec<u32>| {
        if flip {
            out.extend_from_slice(&[a, d, c, a, c, b]);
        } else {
            out.extend_from_slice(&[a, b, c, a, c, d]);
        }
    };
    // ±Z faces: base winding faces +Z, so flip the k == 0 (−Z) face.
    for &k in &[0u32, nz - 1] {
        let flip = k == 0;
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let (a, b, c, d) = (idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k));
                quad(a as u32, b as u32, c as u32, d as u32, flip, &mut indices);
            }
        }
    }
    // ±Y faces: base winding faces −Y, so flip the j == ny-1 (+Y) face.
    for &j in &[0u32, ny - 1] {
        let flip = j == ny - 1;
        for k in 0..nz - 1 {
            for i in 0..nx - 1 {
                let (a, b, c, d) = (idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j, k + 1), idx(i, j, k + 1));
                quad(a as u32, b as u32, c as u32, d as u32, flip, &mut indices);
            }
        }
    }
    // ±X faces: base winding faces +X, so flip the i == 0 (−X) face.
    for &i in &[0u32, nx - 1] {
        let flip = i == 0;
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                let (a, b, c, d) = (idx(i, j, k), idx(i, j + 1, k), idx(i, j + 1, k + 1), idx(i, j, k + 1));
                quad(a as u32, b as u32, c as u32, d as u32, flip, &mut indices);
            }
        }
    }

    // Shape-matching rest pose: each particle's offset from the rest centre of
    // mass (uniform weights, matching `shape_match`). This is what lets the box
    // spring back to shape after the player squashes it instead of staying
    // crumpled — distance constraints alone cannot undo an inverted cell.
    let mut com = Vector::ZERO;
    for p in &particles {
        com += p.pos;
    }
    com /= count as Real;
    let rest: Vec<Vector> = particles.iter().map(|p| p.pos - com).collect();

    SoftBody {
        particles,
        constraints,
        indices,
        rest,
        // Per-substep fraction pulled toward the best-fit rigid rest pose.
        // Caller-tuned: low enough to visibly squash under gravity/impact, but
        // any positive value guarantees the box relaxes back to shape (a rigid
        // pose can't invert) instead of staying crumpled. 0 = pure lattice.
        shape_stiffness: shape_stiffness.max(0.0),
        rot: [0.0, 0.0, 0.0, 1.0],
        damping,
        iterations,
        substeps,
        particle_radius,
        friction,
        render_radius,
        color_seed,
        enabled: true,
    }
}

fn safe_normalize(v: Vector) -> Vector {
    let n = v.length();
    if n < 1.0e-9 {
        Vector::ZERO
    } else {
        v / n
    }
}

// --- Minimal quaternion helpers for shape matching (stored as [x, y, z, w]) ---

fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1.0e-12 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

/// Rotate `v` by unit quaternion `q` (v + 2s·(u×v) + 2·u×(u×v), u = q.xyz).
fn quat_rotate(q: [f32; 4], v: Vector) -> Vector {
    let u = Vector::new(q[0], q[1], q[2]);
    let t = u.cross(v) * 2.0;
    v + t * q[3] + u.cross(t)
}

/// Extract the rotation that best aligns the identity basis with the covariance
/// columns `a0,a1,a2` (Müller 2016, "A robust method to extract the rotational
/// part of deformations"): iteratively nudge a quaternion toward the rotation
/// that zeroes Σ rᵢ × aᵢ. Warm-started from `warm` so it converges in a couple
/// of iterations once the body is moving.
fn extract_rotation(a0: Vector, a1: Vector, a2: Vector, warm: [f32; 4]) -> [f32; 4] {
    let mut q = quat_normalize(warm);
    for _ in 0..24 {
        let r0 = quat_rotate(q, Vector::new(1.0, 0.0, 0.0));
        let r1 = quat_rotate(q, Vector::new(0.0, 1.0, 0.0));
        let r2 = quat_rotate(q, Vector::new(0.0, 0.0, 1.0));
        let numer = r0.cross(a0) + r1.cross(a1) + r2.cross(a2);
        let denom = (r0.dot(a0) + r1.dot(a1) + r2.dot(a2)).abs() + 1.0e-9;
        let omega = numer / denom;
        let w = omega.length();
        if w < 1.0e-9 {
            break;
        }
        let axis = omega / w;
        let half = w * 0.5;
        let s = half.sin();
        let dq = [axis.x * s, axis.y * s, axis.z * s, half.cos()];
        q = quat_normalize(quat_mul(dq, q));
    }
    q
}
