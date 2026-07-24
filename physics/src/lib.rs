//! Squirrel Physics — WASM core.
//!
//! A netcode-first physics engine built **on Rapier** and extended per
//! `physics-engine-report.md`, compiled to `wasm32` exactly like the 3D engine
//! (`wasm-pack build --target web`). The design decisions the report insists on
//! are all here:
//!
//!  * **Rapier is the solver** (report Part 1) — we do not reimplement rigid
//!    dynamics. `enhanced-determinism` is on so a future native server produces
//!    bit-identical results from identical inputs (Parts 1/3).
//!  * **The tier model** (Part 4) — tier is a property; Advanced bodies carry an
//!    index into a dense `AeroProps` table and get a pre-step force pass.
//!  * **Custom forces in the core, never a JS callback** (Part 5) — buoyancy,
//!    quadratic drag and Magnus run in one tight Rust loop in [`forces`].
//!  * **Kinematic character controller** (Part 5) — the player is a
//!    kinematic-position-based body; Rapier's contact solve shoves the balls.
//!  * **Cheap dynamic-only snapshots** (Part 7) — pos+linvel per dynamic body,
//!    the shape a rollback ring buffer wants.
//!  * **Zero-copy render** (Part 2) — a packed `f32` buffer lives in linear
//!    memory; JS reads it through a `Float32Array` view, no per-frame copy.
//!
//! Like the 3D engine, this crate holds no rendering handles: JS owns the
//! canvas and reads a flat transform buffer by pointer each frame.

mod forces;
mod soft;
mod tiers;

use forces::{AeroProps, Buoyancy, Drag};
use rapier3d::prelude::*;
use soft::SoftBody;
use wasm_bindgen::prelude::*;

/// Floats emitted per body into the render buffer:
/// `[x, y, z, radius, tier, color_seed, awake]`.
const FLOATS_PER_BODY: usize = 7;
/// Upper bound on bodies, so the transform + snapshot buffers are reserved once
/// and never reallocate (their pointers must stay stable for JS views).
const MAX_BODIES: usize = 4096;
/// Floats emitted per soft-body particle into its render buffer:
/// `[x, y, z, radius]`.
const SOFT_FLOATS_PER_PARTICLE: usize = 4;
/// Upper bound on soft-body particles, reserved once so the packed render buffer
/// keeps a stable pointer for the JS view (same rule as the rigid buffer).
const MAX_SOFT_PARTICLES: usize = 8192;

/// One simulated ball. `Simple` and `Advanced` bodies are identical to Rapier;
/// `aero_index` is the only thing that distinguishes them.
struct Ball {
    handle: RigidBodyHandle,
    radius: Real,
    mass: Real,
    tier: u32,
    aero_index: Option<usize>,
    color_seed: f32,
    /// Per-object physics toggle (unified-scene "physics enabled" flag). When
    /// off, the body is switched to `Fixed`: it stops integrating — no gravity,
    /// no forces, no getting shoved — so it freezes in place while still acting
    /// as a static collider. The custom force pass skips it too. Re-enabling
    /// restores it to a `Dynamic` body from its current position.
    enabled: bool,
}

/// Kinematic player state. A pure `(state, input, dt)` step, no hidden
/// accumulators, so it stays rollback-safe (report Part 5).
struct Player {
    handle: RigidBodyHandle,
    radius: Real,
    speed: Real,
    jump_speed: Real,
    gravity: Real,
    vy: Real,
    grounded: bool,
    in_x: Real,
    in_z: Real,
    in_jump: bool,
    // Pit bounds: min_x, max_x, min_z, max_z, floor_y.
    bounds: [Real; 5],
}

#[wasm_bindgen]
pub struct PhysicsWorld {
    // --- Rapier state ---
    gravity: Vector,
    params: IntegrationParameters,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,

    // --- Our extensions ---
    balls: Vec<Ball>,
    /// Dense aero side-table (report Part 4/5). Iterated contiguously each tick.
    aero: Vec<AeroProps>,
    /// Parallel to `aero`: index into `balls` that owns each entry.
    aero_owner: Vec<usize>,
    player: Option<Player>,

    // --- Soft bodies + cloth (our own XPBD solver, see `soft`) ---
    soft_bodies: Vec<SoftBody>,
    /// Packed `[x, y, z, radius]` per particle, all soft bodies concatenated —
    /// the render buffer JS reads by pointer, just like `transforms`.
    soft_transforms: Vec<f32>,
    /// Pit extents `[min_x, max_x, min_z, max_z, floor_y, wall_top]`, captured
    /// by `add_pit`, so soft bodies can collide with the floor + walls.
    pit_bounds: Option<[Real; 6]>,
    /// Scratch list of rigid `(center, radius)` spheres rebuilt each tick and
    /// fed to the soft solver for cloth-vs-body collisions.
    collision_spheres: Vec<(Vector, Real)>,

    // --- Buffers in linear memory (stable pointers for JS views) ---
    transforms: Vec<f32>,
    snapshot_buf: Vec<f32>,

    tick: u32,
}

#[wasm_bindgen]
impl PhysicsWorld {
    /// Create an empty world with downward gravity `gy` (e.g. -12.0).
    #[wasm_bindgen(constructor)]
    pub fn new(gy: f32) -> PhysicsWorld {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let mut params = IntegrationParameters::default();
        params.dt = 1.0 / 60.0;

        PhysicsWorld {
            gravity: Vector::new(0.0, gy, 0.0),
            params,
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            balls: Vec::new(),
            aero: Vec::new(),
            aero_owner: Vec::new(),
            player: None,
            soft_bodies: Vec::new(),
            soft_transforms: Vec::with_capacity(MAX_SOFT_PARTICLES * SOFT_FLOATS_PER_PARTICLE),
            pit_bounds: None,
            collision_spheres: Vec::new(),
            transforms: Vec::with_capacity(MAX_BODIES * FLOATS_PER_BODY),
            snapshot_buf: Vec::with_capacity(MAX_BODIES * 7),
            tick: 0,
        }
    }

    // --- Authoring ---------------------------------------------------------

    /// Build the static pit: a floor and four walls, as parentless (static)
    /// cuboid colliders. `thickness` is how deep the walls/floor extend.
    pub fn add_pit(
        &mut self,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        floor_y: f32,
        wall_top: f32,
        thickness: f32,
    ) {
        let hx = (max_x - min_x) * 0.5;
        let hz = (max_z - min_z) * 0.5;
        let cx = (max_x + min_x) * 0.5;
        let cz = (max_z + min_z) * 0.5;
        let wall_h = (wall_top - floor_y) * 0.5 + thickness;
        let wall_cy = floor_y + (wall_top - floor_y) * 0.5;
        let t = thickness;

        // Remember the interior extents so soft bodies collide with the pit.
        self.pit_bounds = Some([min_x, max_x, min_z, max_z, floor_y, wall_top]);

        // Floor.
        self.insert_static(
            ColliderBuilder::cuboid(hx + t, t, hz + t)
                .translation(Vector::new(cx, floor_y - t, cz))
                .friction(0.7)
                .restitution(0.1)
                .build(),
        );
        // Walls at +X / -X.
        self.insert_static(
            ColliderBuilder::cuboid(t, wall_h, hz + t)
                .translation(Vector::new(max_x + t, wall_cy, cz))
                .build(),
        );
        self.insert_static(
            ColliderBuilder::cuboid(t, wall_h, hz + t)
                .translation(Vector::new(min_x - t, wall_cy, cz))
                .build(),
        );
        // Walls at +Z / -Z.
        self.insert_static(
            ColliderBuilder::cuboid(hx + t, wall_h, t)
                .translation(Vector::new(cx, wall_cy, max_z + t))
                .build(),
        );
        self.insert_static(
            ColliderBuilder::cuboid(hx + t, wall_h, t)
                .translation(Vector::new(cx, wall_cy, min_z - t))
                .build(),
        );
    }

    fn insert_static(&mut self, collider: Collider) {
        self.colliders.insert(collider);
    }

    /// Add a dynamic ball. Returns its index (a stable handle for later
    /// `set_buoyancy` / `set_drag` calls). Starts at `Simple` tier.
    pub fn add_ball(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        mass: f32,
        restitution: f32,
        friction: f32,
        color_seed: f32,
    ) -> u32 {
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector::new(x, y, z))
            .linear_damping(0.15)
            .can_sleep(true)
            .build();
        let handle = self.bodies.insert(rb);
        let collider = ColliderBuilder::ball(radius)
            .restitution(restitution)
            .friction(friction)
            .mass(mass)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);

        let idx = self.balls.len() as u32;
        self.balls.push(Ball {
            handle,
            radius,
            mass,
            tier: tiers::SIMPLE,
            aero_index: None,
            color_seed,
            enabled: true,
        });
        idx
    }

    /// Promote a ball to Advanced and give it buoyancy against a flat surface.
    pub fn set_buoyancy(
        &mut self,
        index: u32,
        fluid_density: f32,
        volume: f32,
        fluid_level_y: f32,
        linear_drag: f32,
    ) {
        let ai = self.ensure_aero(index);
        self.aero[ai].buoyancy = Some(Buoyancy {
            fluid_density,
            volume,
            fluid_level_y,
            linear_drag,
        });
    }

    /// Promote a ball to Advanced and give it quadratic air drag (+ Magnus).
    pub fn set_drag(&mut self, index: u32, cd: f32, ref_area: f32, air_density: f32, magnus: f32) {
        let ai = self.ensure_aero(index);
        self.aero[ai].drag = Some(Drag {
            cd,
            ref_area,
            air_density,
            magnus,
        });
    }

    /// Idempotently promote a ball to Advanced and return its dense aero index.
    fn ensure_aero(&mut self, index: u32) -> usize {
        let b = &mut self.balls[index as usize];
        if let Some(ai) = b.aero_index {
            return ai;
        }
        let ai = self.aero.len();
        b.aero_index = Some(ai);
        b.tier = tiers::ADVANCED;
        self.aero.push(AeroProps::default());
        self.aero_owner.push(index as usize);
        ai
    }

    /// Create the kinematic player at `(x,y,z)`. Must be called after `add_pit`
    /// with the same bounds so it can't leave the pit.
    #[allow(clippy::too_many_arguments)]
    pub fn add_player(
        &mut self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        floor_y: f32,
    ) {
        let rb = RigidBodyBuilder::kinematic_position_based()
            .translation(Vector::new(x, y, z))
            .build();
        let handle = self.bodies.insert(rb);
        let collider = ColliderBuilder::ball(radius).friction(0.4).build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);

        self.player = Some(Player {
            handle,
            radius,
            speed: 7.0,
            jump_speed: 6.5,
            gravity: -20.0,
            vy: 0.0,
            grounded: true,
            in_x: 0.0,
            in_z: 0.0,
            in_jump: false,
            bounds: [min_x, max_x, min_z, max_z, floor_y],
        });
    }

    /// Set the player's movement command for the next tick. `(ix, iz)` is a
    /// world-space move vector (magnitude ≤ 1), `jump` a rising-edge intent.
    pub fn set_player_input(&mut self, ix: f32, iz: f32, jump: bool) {
        if let Some(p) = &mut self.player {
            p.in_x = ix;
            p.in_z = iz;
            p.in_jump = jump;
        }
    }

    /// Per-object physics toggle used by the unified scene layer. `enabled =
    /// false` freezes ball `index` in place (switches it to a `Fixed` body): it
    /// no longer falls, is not pushed by contacts, and skips the custom force
    /// pass, but still collides as a static obstacle. `enabled = true` makes it
    /// `Dynamic` again from wherever it currently sits. Idempotent.
    pub fn set_ball_enabled(&mut self, index: u32, enabled: bool) {
        let Some(ball) = self.balls.get_mut(index as usize) else {
            return;
        };
        if ball.enabled == enabled {
            return;
        }
        ball.enabled = enabled;
        let handle = ball.handle;
        if let Some(body) = self.bodies.get_mut(handle) {
            let ty = if enabled {
                RigidBodyType::Dynamic
            } else {
                // Zero the velocity so it doesn't resume its old momentum when
                // re-enabled, then fix it in place.
                body.set_linvel(Vector::ZERO, false);
                body.set_angvel(Vector::ZERO, false);
                RigidBodyType::Fixed
            };
            body.set_body_type(ty, true);
        }
    }

    /// Whether ball `index` currently participates in the simulation.
    pub fn is_ball_enabled(&self, index: u32) -> bool {
        self.balls
            .get(index as usize)
            .map(|b| b.enabled)
            .unwrap_or(false)
    }

    // --- Soft bodies + cloth -----------------------------------------------

    /// Add a cloth sheet: a scalable rectangle of `nu * nv` particles spanning
    /// `width` along the `(ux,uy,uz)` axis and `height` along `(vx,vy,vz)` from
    /// the corner `(ox,oy,oz)`. `pin_mode`: 0 free, 1 pin the far `v==0` edge,
    /// 2 pin its two corners, 3 pin all four corners. Returns the soft-body id.
    #[allow(clippy::too_many_arguments)]
    pub fn add_cloth(
        &mut self,
        ox: f32, oy: f32, oz: f32,
        ux: f32, uy: f32, uz: f32,
        vx: f32, vy: f32, vz: f32,
        width: f32, height: f32,
        nu: u32, nv: u32,
        total_mass: f32,
        compliance: f32, bend_compliance: f32,
        damping: f32,
        particle_radius: f32, render_radius: f32,
        friction: f32,
        iterations: u32, substeps: u32,
        pin_mode: u32,
        color_seed: f32,
    ) -> u32 {
        let body = soft::build_cloth(
            Vector::new(ox, oy, oz),
            Vector::new(ux, uy, uz),
            Vector::new(vx, vy, vz),
            width, height, nu, nv,
            total_mass, compliance, bend_compliance, damping,
            particle_radius, render_radius, friction,
            iterations, substeps, pin_mode, color_seed,
        );
        let id = self.soft_bodies.len() as u32;
        self.soft_bodies.push(body);
        id
    }

    /// Add a squishy solid block: an `nx * ny * nz` particle lattice centered at
    /// `(cx,cy,cz)` with full extents `(sx,sy,sz)`. `pin_top` dangles it from its
    /// top face; otherwise it falls freely. Returns the soft-body id.
    #[allow(clippy::too_many_arguments)]
    pub fn add_soft_box(
        &mut self,
        cx: f32, cy: f32, cz: f32,
        sx: f32, sy: f32, sz: f32,
        nx: u32, ny: u32, nz: u32,
        total_mass: f32,
        compliance: f32,
        damping: f32,
        particle_radius: f32, render_radius: f32,
        friction: f32,
        iterations: u32, substeps: u32,
        pin_top: bool,
        shape_stiffness: f32,
        color_seed: f32,
    ) -> u32 {
        let body = soft::build_soft_box(
            Vector::new(cx, cy, cz),
            Vector::new(sx, sy, sz),
            nx, ny, nz,
            total_mass, compliance, damping,
            particle_radius, render_radius, friction,
            iterations, substeps, pin_top, shape_stiffness, color_seed,
        );
        let id = self.soft_bodies.len() as u32;
        self.soft_bodies.push(body);
        id
    }

    /// Per-object physics toggle for soft body `id`: `false` freezes it in place
    /// (its solve becomes a no-op), `true` lets it move again. Matches
    /// [`set_ball_enabled`] for rigid bodies.
    pub fn set_soft_enabled(&mut self, id: u32, enabled: bool) {
        if let Some(sb) = self.soft_bodies.get_mut(id as usize) {
            sb.enabled = enabled;
        }
    }

    /// Number of soft bodies (cloth + boxes) in the world.
    pub fn soft_body_count(&self) -> u32 {
        self.soft_bodies.len() as u32
    }

    /// Particle count of soft body `id` (0 if it doesn't exist). The scene uses
    /// this to slice the packed particle buffer per body.
    pub fn soft_body_particle_count(&self, id: u32) -> u32 {
        self.soft_bodies
            .get(id as usize)
            .map(|b| b.particle_count() as u32)
            .unwrap_or(0)
    }

    /// Surface-triangle index count for soft body `id` (3 per triangle). Read
    /// once by JS to build the body's GPU index buffer for mesh rendering.
    pub fn soft_body_index_count(&self, id: u32) -> u32 {
        self.soft_bodies
            .get(id as usize)
            .map(|b| b.indices().len() as u32)
            .unwrap_or(0)
    }

    /// Pointer to soft body `id`'s surface index list (local particle indices).
    /// Stable for the body's lifetime; JS reads it once at mesh creation.
    pub fn soft_body_indices_ptr(&self, id: u32) -> *const u32 {
        self.soft_bodies
            .get(id as usize)
            .map(|b| b.indices().as_ptr())
            .unwrap_or(std::ptr::null())
    }

    // --- Simulation --------------------------------------------------------

    /// Advance one fixed tick of `dt` seconds. Runs the custom force pass, the
    /// player controller, the Rapier step, then repacks the render buffer.
    pub fn step(&mut self, dt: f32) {
        self.params.dt = dt;

        // 1. Pre-step force pass over the dense aero table (Advanced only).
        let gy = self.gravity.y;
        for i in 0..self.aero.len() {
            let owner = self.aero_owner[i];
            let ball = &self.balls[owner];
            if !ball.enabled {
                continue; // physics off for this object — no custom forces
            }
            let (radius, mass) = (ball.radius, ball.mass);
            let handle = ball.handle;
            if let Some(body) = self.bodies.get_mut(handle) {
                if body.is_sleeping() {
                    continue;
                }
                let pos_y = body.translation().y;
                let v = body.linvel();
                let nv = forces::apply_aero(v, pos_y, mass, radius, &self.aero[i], gy, dt);
                body.set_linvel(nv, false);
            }
        }

        // 2. Player controller: pure (state, input, dt), then hand Rapier a new
        //    kinematic target so its contact solve shoves the balls.
        self.step_player(dt);

        // 3. Rapier step. `()` is the no-op hooks + event handler.
        self.pipeline.step(
            self.gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );

        // 4. Soft bodies + cloth: our own XPBD pass, colliding against the pit
        //    and the rigid spheres in their just-updated positions. The spheres
        //    are packed balls-first then the player, so `collision_spheres[i]`
        //    is `balls[i]` and the last entry (if any) is the player — the same
        //    order the trampoline bounce below relies on.
        if !self.soft_bodies.is_empty() {
            self.collision_spheres.clear();
            for ball in &self.balls {
                if let Some(b) = self.bodies.get(ball.handle) {
                    self.collision_spheres.push((b.translation(), ball.radius));
                }
            }
            if let Some(pl) = &self.player {
                if let Some(b) = self.bodies.get(pl.handle) {
                    self.collision_spheres.push((b.translation(), pl.radius));
                }
            }
            // One contact accumulator per sphere, summed across every soft body.
            let mut contacts = vec![soft::RigidContact::zero(); self.collision_spheres.len()];
            let gravity = self.gravity;
            let bounds = self.pit_bounds;
            for sb in &mut self.soft_bodies {
                sb.solve(gravity, dt, &self.collision_spheres, &mut contacts, bounds);
            }
            self.apply_soft_reactions(&contacts);
        }

        self.tick += 1;
        self.pack();
        self.pack_soft();
    }

    /// Bounce every rigid sphere off the soft surfaces it pressed into this tick.
    /// `contacts` is parallel to `collision_spheres`: the balls in order, then
    /// the player. A pinned soft bed thus behaves like a trampoline — the push
    /// momentum is absorbed by the bed's pins (the pit), so this injects the
    /// bounce energy exactly the way a real anchored trampoline does.
    fn apply_soft_reactions(&mut self, contacts: &[soft::RigidContact]) {
        // Restitution off a soft surface, and the impact speed above which it
        // engages. Below the threshold the surface only *supports* the body
        // (cancels the downward approach) so a resting ball doesn't jitter or
        // creep; above it the surface flings the body back.
        const RESTITUTION: Real = 0.85;
        const BOUNCE_THRESHOLD: Real = 1.5;

        for (i, ball) in self.balls.iter().enumerate() {
            let c = contacts[i];
            if c.count == 0 || !ball.enabled {
                continue;
            }
            let n = safe_dir(c.normal);
            // Only an *upward*-springing bed adds energy; its downward deflection
            // (it sinks under the body) must not cancel the bounce, or a soft bed
            // that moves with the body would swallow the impact. Restitution is
            // therefore measured against the ground (the bed is pinned to it).
            let bed_up = (c.bed_vel / c.count as Real).dot(n).max(0.0);
            if let Some(body) = self.bodies.get_mut(ball.handle) {
                if !body.is_dynamic() {
                    continue;
                }
                let v = body.linvel();
                let vn = v.dot(n); // ground-frame approach; negative == into surface
                if vn < 0.0 {
                    let e = if -vn > BOUNCE_THRESHOLD { RESTITUTION } else { 0.0 };
                    body.set_linvel(v + n * (-(1.0 + e) * vn + bed_up), true);
                }
            }
        }

        // The player sphere is packed right after the balls. The kinematic
        // controller only tracks a vertical velocity, so bounce that.
        if let Some(pl) = &mut self.player {
            if let Some(c) = contacts.get(self.balls.len()).copied() {
                if c.count > 0 {
                    let n = safe_dir(c.normal);
                    let bed_up = (c.bed_vel / c.count as Real).dot(n).max(0.0);
                    let v = Vector::new(0.0, pl.vy, 0.0);
                    let vn = v.dot(n);
                    if vn < 0.0 {
                        let e = if -vn > BOUNCE_THRESHOLD { RESTITUTION } else { 0.0 };
                        pl.vy += (-(1.0 + e) * vn + bed_up) * n.y;
                        pl.grounded = true;
                    }
                }
            }
        }
    }

    fn step_player(&mut self, dt: f32) {
        let Some(p) = &mut self.player else { return };
        let [min_x, max_x, min_z, max_z, floor_y] = p.bounds;
        let r = p.radius;

        let vx = p.in_x * p.speed;
        let vz = p.in_z * p.speed;

        if p.grounded {
            p.vy = 0.0;
            if p.in_jump {
                p.vy = p.jump_speed;
                p.grounded = false;
            }
        } else {
            p.vy += p.gravity * dt;
        }

        let cur = self
            .bodies
            .get(p.handle)
            .map(|b| b.translation())
            .unwrap_or(Vector::ZERO);
        let mut nx = cur.x + vx * dt;
        let mut ny = cur.y + p.vy * dt;
        let mut nz = cur.z + vz * dt;

        nx = nx.clamp(min_x + r, max_x - r);
        nz = nz.clamp(min_z + r, max_z - r);
        if ny - r <= floor_y {
            ny = floor_y + r;
            p.vy = 0.0;
            p.grounded = true;
        }

        if let Some(body) = self.bodies.get_mut(p.handle) {
            body.set_next_kinematic_translation(Vector::new(nx, ny, nz));
        }
    }

    // --- Render buffer -----------------------------------------------------

    /// Repack the flat transform buffer JS reads each frame. Layout per body:
    /// `[x, y, z, radius, tier, color_seed, awake]`, balls first, player last.
    fn pack(&mut self) {
        self.transforms.clear();
        for ball in &self.balls {
            if let Some(b) = self.bodies.get(ball.handle) {
                let p = b.translation();
                let awake = if b.is_sleeping() { 0.0 } else { 1.0 };
                self.transforms.extend_from_slice(&[
                    p.x,
                    p.y,
                    p.z,
                    ball.radius,
                    ball.tier as f32,
                    ball.color_seed,
                    awake,
                ]);
            }
        }
        if let Some(pl) = &self.player {
            if let Some(b) = self.bodies.get(pl.handle) {
                let p = b.translation();
                self.transforms.extend_from_slice(&[
                    p.x,
                    p.y,
                    p.z,
                    pl.radius,
                    tiers::PLAYER_MARKER,
                    0.0,
                    1.0,
                ]);
            }
        }
    }

    /// Repack every soft-body particle into the flat `[x, y, z, radius]` render
    /// buffer, in soft-body creation order (so a body's slice is contiguous).
    fn pack_soft(&mut self) {
        self.soft_transforms.clear();
        for sb in &self.soft_bodies {
            sb.pack_into(&mut self.soft_transforms);
        }
    }

    /// Pointer to the packed transform buffer (JS builds a `Float32Array` view).
    pub fn bodies_ptr(&self) -> *const f32 {
        self.transforms.as_ptr()
    }

    /// Number of bodies currently packed (balls + player).
    pub fn bodies_count(&self) -> usize {
        self.transforms.len() / FLOATS_PER_BODY
    }

    /// Floats per body in the transform buffer.
    pub fn floats_per_body(&self) -> usize {
        FLOATS_PER_BODY
    }

    // --- Soft-body render buffer (parallel to the rigid one above) ---------

    /// Pointer to the packed soft-body particle buffer (`[x, y, z, radius]` per
    /// particle). JS builds a `Float32Array` view over it each frame.
    pub fn soft_particles_ptr(&self) -> *const f32 {
        self.soft_transforms.as_ptr()
    }

    /// Total soft-body particles currently packed (all bodies).
    pub fn soft_particle_count(&self) -> usize {
        self.soft_transforms.len() / SOFT_FLOATS_PER_PARTICLE
    }

    /// Floats per particle in the soft-body buffer.
    pub fn soft_floats_per_particle(&self) -> usize {
        SOFT_FLOATS_PER_PARTICLE
    }

    pub fn tick(&self) -> u32 {
        self.tick
    }

    // --- Tier counts (report Part 4: "first thing to check") ---------------

    pub fn count_simple(&self) -> u32 {
        self.balls.iter().filter(|b| b.tier == tiers::SIMPLE).count() as u32
    }

    pub fn count_advanced(&self) -> u32 {
        self.balls
            .iter()
            .filter(|b| b.tier == tiers::ADVANCED)
            .count() as u32
    }

    pub fn count_asleep(&self) -> u32 {
        self.balls
            .iter()
            .filter(|b| {
                self.bodies
                    .get(b.handle)
                    .map(|x| x.is_sleeping())
                    .unwrap_or(false)
            })
            .count() as u32
    }

    // --- Netcode surface: cheap dynamic-only snapshot (report Part 7) ------

    /// Capture pos(3) + linvel(3) for every ball into the snapshot buffer, and
    /// return the float length. The rollback ring buffer stores exactly this.
    pub fn snapshot(&mut self) -> usize {
        self.snapshot_buf.clear();
        for ball in &self.balls {
            if let Some(b) = self.bodies.get(ball.handle) {
                let p = b.translation();
                let v = b.linvel();
                self.snapshot_buf
                    .extend_from_slice(&[p.x, p.y, p.z, v.x, v.y, v.z]);
            }
        }
        self.snapshot_buf.len()
    }

    pub fn snapshot_ptr(&self) -> *const f32 {
        self.snapshot_buf.as_ptr()
    }

    /// Restore ball state from a previously captured snapshot (same body order).
    pub fn restore(&mut self, data: &[f32]) {
        let mut o = 0;
        for ball in &self.balls {
            if o + 6 > data.len() {
                break;
            }
            if let Some(b) = self.bodies.get_mut(ball.handle) {
                b.set_translation(Vector::new(data[o], data[o + 1], data[o + 2]), true);
                b.set_linvel(Vector::new(data[o + 3], data[o + 4], data[o + 5]), true);
            }
            o += 6;
        }
        self.pack();
    }
}

/// Normalize a direction, falling back to straight up for a degenerate (near
/// zero) vector — used for the soft-contact normal, which is only zero when
/// there was no real contact to react to.
fn safe_dir(v: Vector) -> Vector {
    let n = v.length();
    if n < 1.0e-9 {
        Vector::new(0.0, 1.0, 0.0)
    } else {
        v / n
    }
}
