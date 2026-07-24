//! Headless smoke + determinism harness for the physics core (report Part 3:
//! "determinism verification is mandatory"). Runs natively — no wasm needed.
//!
//!   cargo test --release
//!
//! Covers: balls settle without exploding, sleeping engages, a buoyant Advanced
//! ball floats, the player pushes forward, snapshot/restore round-trips, and —
//! the key check — identical inputs produce identical per-tick state hashes.

use squirrel_physics::PhysicsWorld;

/// Deterministic LCG so both runs feed identical spawn positions + inputs.
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn build(seed: u32) -> PhysicsWorld {
    let mut rng = Rng(seed);
    let mut w = PhysicsWorld::new(-12.0);
    w.add_pit(-6.0, 6.0, -6.0, 6.0, 0.0, 6.0, 0.5);
    for _ in 0..120 {
        let r = 0.35 + rng.next() * 0.15;
        w.add_ball(
            -5.0 + rng.next() * 10.0,
            1.0 + rng.next() * 6.0,
            -5.0 + rng.next() * 10.0,
            r,
            r * r * r * 8.0,
            0.4,
            0.55,
            rng.next(),
        );
    }
    let b = w.add_ball(0.0, 5.0, 0.0, 0.7, 0.5, 0.55, 0.4, 0.5);
    w.set_buoyancy(b, 0.6, 1.4, 1.5, 6.0);
    w.add_player(0.0, 0.7, -4.0, 0.7, -6.0, 6.0, -6.0, 6.0, 0.0);
    w
}

/// Read the packed transform buffer (7 floats/body) back into a Vec.
fn read_transforms(w: &PhysicsWorld) -> Vec<f32> {
    let ptr = w.bodies_ptr();
    let n = w.bodies_count() * w.floats_per_body();
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

/// Read the packed soft-body particle buffer (4 floats/particle) into a Vec.
fn read_soft(w: &PhysicsWorld) -> Vec<f32> {
    let ptr = w.soft_particles_ptr();
    let n = w.soft_particle_count() * w.soft_floats_per_particle();
    unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
}

/// FNV-1a over every body's x/y/z — the per-tick state hash.
fn hash(w: &PhysicsWorld) -> u32 {
    let t = read_transforms(w);
    let mut h: u32 = 2166136261;
    for (i, f) in t.iter().enumerate() {
        if i % 7 >= 3 {
            continue; // only x,y,z
        }
        for b in f.to_bits().to_le_bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
    }
    h
}

#[test]
fn settles_without_exploding() {
    let mut w = build(1);
    w.set_player_input(0.0, 1.0, false); // walk forward the whole time
    for _ in 0..600 {
        w.step(1.0 / 60.0);
    }
    let t = read_transforms(&w);
    assert!(
        t.iter().all(|f| f.is_finite() && f.abs() < 1.0e4),
        "state stayed finite and sane (no explosion)"
    );
    // Ball y (index 1 of each 7) never far below the floor.
    let min_y = t.chunks(7).map(|c| c[1]).fold(f32::INFINITY, f32::min);
    assert!(min_y > -0.6, "no ball sank through the floor (min y = {min_y})");

    assert!(w.count_asleep() > 0, "sleeping engaged for settled balls");
    assert!(w.count_advanced() == 1, "one advanced/buoyant ball");
}

#[test]
fn buoyant_ball_floats() {
    let mut w = build(2);
    for _ in 0..600 {
        w.step(1.0 / 60.0);
    }
    // The buoyant ball is body index 120 (after the 120 simple balls).
    let t = read_transforms(&w);
    let y = t[120 * 7 + 1];
    assert!(y > 0.6, "buoyant ball floats near the surface (y = {y})");
}

#[test]
fn player_pushes_forward() {
    let mut w = build(3);
    w.set_player_input(0.0, 1.0, false);
    for _ in 0..180 {
        w.step(1.0 / 60.0);
    }
    // Player is the last packed body; its z should have advanced past start.
    let t = read_transforms(&w);
    let n = w.bodies_count();
    let pz = t[(n - 1) * 7 + 2];
    assert!(pz > -3.9, "player moved forward under input (z = {pz})");
}

#[test]
fn snapshot_restore_roundtrips() {
    let mut w = build(4);
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let len = w.snapshot();
    let snap = unsafe { std::slice::from_raw_parts(w.snapshot_ptr(), len).to_vec() };
    let before = hash(&w);
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    w.restore(&snap);
    assert_eq!(before, hash(&w), "snapshot/restore round-trips to same state");
}

#[test]
fn disabled_ball_freezes_in_place() {
    let mut w = PhysicsWorld::new(-12.0);
    w.add_pit(-6.0, 6.0, -6.0, 6.0, 0.0, 6.0, 0.5);
    let idx = w.add_ball(0.0, 5.0, 0.0, 0.4, 0.5, 0.4, 0.55, 0.5);

    // With physics on, the ball falls under gravity.
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    let y_fell = read_transforms(&w)[idx as usize * 7 + 1];
    assert!(y_fell < 5.0, "ball fell while physics was on (y = {y_fell})");

    // Turn physics off: it must not move for the rest of the sim.
    w.set_ball_enabled(idx, false);
    assert!(!w.is_ball_enabled(idx));
    let y_frozen = read_transforms(&w)[idx as usize * 7 + 1];
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let y_after = read_transforms(&w)[idx as usize * 7 + 1];
    assert!(
        (y_after - y_frozen).abs() < 1.0e-4,
        "disabled ball stayed put (was {y_frozen}, now {y_after})"
    );

    // Re-enable: it falls again.
    w.set_ball_enabled(idx, true);
    assert!(w.is_ball_enabled(idx));
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    let y_resumed = read_transforms(&w)[idx as usize * 7 + 1];
    assert!(y_resumed < y_after - 1.0e-3, "re-enabled ball falls again (y = {y_resumed})");
}

#[test]
fn cloth_pinned_edge_hangs() {
    let mut w = PhysicsWorld::new(-10.0);
    let (nu, nv) = (8u32, 8u32);
    // Origin top-left; u spans +x, v spans -y so the sheet hangs downward.
    let id = w.add_cloth(
        -1.0, 3.0, 0.0, // origin
        1.0, 0.0, 0.0, // u axis
        0.0, -1.0, 0.0, // v axis (down)
        2.0, 2.0, // width, height
        nu, nv, 1.0, // mass
        0.0, 0.0001, // compliance, bend compliance
        0.5, // damping
        0.02, 0.05, // particle / render radius
        0.3, // friction
        12, 4, // iterations, substeps
        1, // pin_mode = pin the v==0 edge
        0.5,
    );
    assert_eq!(w.soft_body_count(), 1);
    assert_eq!(w.soft_body_particle_count(id), nu * nv);

    for _ in 0..240 {
        w.step(1.0 / 60.0);
    }
    let s = read_soft(&w);
    assert!(s.iter().all(|f| f.is_finite()), "cloth stayed finite");
    // The pinned top row (j==0 → first nu particles) holds at the origin height.
    for i in 0..nu as usize {
        let y = s[i * 4 + 1];
        assert!((y - 3.0).abs() < 1.0e-3, "pinned row held (y = {y})");
    }
    // The free bottom row hangs well below the pinned top.
    let top_y = s[1];
    let bottom_y = s[((nv - 1) * nu) as usize * 4 + 1];
    assert!(bottom_y < top_y - 0.5, "cloth hangs (top {top_y}, bottom {bottom_y})");
}

#[test]
fn soft_bodies_settle_in_pit() {
    let mut w = PhysicsWorld::new(-12.0);
    w.add_pit(-4.0, 4.0, -4.0, 4.0, 0.0, 6.0, 0.5);
    // A cloth dropped flat, and a squishy box dropped above it.
    w.add_cloth(
        -1.5, 4.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0, 10, 10, 1.0, 0.0, 0.001, 0.6,
        0.03, 0.06, 0.4, 10, 4, 0, 0.2,
    );
    w.add_soft_box(
        0.0, 5.0, 0.0, 1.0, 1.0, 1.0, 4, 4, 4, 2.0, 0.001, 0.5, 0.03, 0.06, 0.4, 10, 4, false, 0.03, 0.7,
    );
    for _ in 0..400 {
        w.step(1.0 / 60.0);
    }
    let s = read_soft(&w);
    assert!(
        s.iter().all(|f| f.is_finite() && f.abs() < 1.0e4),
        "soft state stayed finite and sane"
    );
    let min_y = s.chunks(4).map(|c| c[1]).fold(f32::INFINITY, f32::min);
    assert!(min_y > -0.2, "soft particles stayed above the floor (min y = {min_y})");
    let max_absx = s.chunks(4).map(|c| c[0].abs()).fold(0.0f32, f32::max);
    assert!(max_absx < 4.1, "soft bodies stayed inside the pit (max |x| = {max_absx})");
}

#[test]
fn soft_box_is_squishy() {
    let mut w = PhysicsWorld::new(-12.0);
    w.add_pit(-4.0, 4.0, -4.0, 4.0, 0.0, 6.0, 0.5);
    let size = 1.6f32;
    // The soft (jelly) parameters the demo/scene now use by default.
    w.add_soft_box(
        0.0, 3.0, 0.0, size, size, size, 5, 5, 5, 1.5, 0.03, 0.9, 0.03, 0.06, 0.4, 8, 5, false, 0.03, 0.5,
    );

    // Vertical extent (max y - min y over the particles) tracks how squashed the
    // box is: constant for a rigid body, varying for a soft one.
    let extent = |w: &PhysicsWorld| {
        let s = read_soft(w);
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for c in s.chunks(4) {
            lo = lo.min(c[1]);
            hi = hi.max(c[1]);
        }
        hi - lo
    };

    let mut min_ext = f32::INFINITY;
    let mut max_ext = f32::NEG_INFINITY;
    for _ in 0..300 {
        w.step(1.0 / 60.0);
        let e = extent(&w);
        min_ext = min_ext.min(e);
        max_ext = max_ext.max(e);
    }

    // Let it settle so we can check the resting shape.
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }

    let s = read_soft(&w);
    assert!(
        s.iter().all(|f| f.is_finite() && f.abs() < 1.0e4),
        "soft box stayed finite and sane"
    );
    // It visibly deforms (extent changes as it squashes + recovers) — not rigid.
    assert!(
        max_ext - min_ext > 0.03,
        "soft box deformed (vertical extent ranged {min_ext}..{max_ext})"
    );
    // ...but keeps its shape on ALL THREE axes — it must not crumple/collapse.
    // The body diagonals are what make this hold; without them a soft box buckles.
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for c in s.chunks(4) {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    for a in 0..3 {
        let ext = hi[a] - lo[a];
        assert!(
            ext > 0.5 * size,
            "soft box kept its shape on axis {a} (extent {ext}, size {size})"
        );
    }
}

#[test]
fn soft_freeze_toggle_holds() {
    let mut w = PhysicsWorld::new(-12.0);
    let id = w.add_cloth(
        -1.0, 5.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 6, 6, 1.0, 0.0, 0.001, 0.5, 0.02,
        0.05, 0.3, 10, 4, 0, 0.5,
    );
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    let fell = read_soft(&w)[1];
    assert!(fell < 5.0, "free cloth fell (y = {fell})");

    w.set_soft_enabled(id, false);
    let frozen = read_soft(&w)[1];
    for _ in 0..120 {
        w.step(1.0 / 60.0);
    }
    let after = read_soft(&w)[1];
    assert!((after - frozen).abs() < 1.0e-4, "disabled cloth stayed put ({frozen} -> {after})");

    w.set_soft_enabled(id, true);
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    assert!(read_soft(&w)[1] < after - 1.0e-3, "re-enabled cloth falls again");
}

#[test]
fn soft_deterministic() {
    let run = || {
        let mut w = PhysicsWorld::new(-12.0);
        w.add_pit(-4.0, 4.0, -4.0, 4.0, 0.0, 6.0, 0.5);
        w.add_cloth(
            -1.5, 4.0, -1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 3.0, 12, 12, 1.0, 0.0, 0.001, 0.6,
            0.03, 0.06, 0.4, 10, 4, 1, 0.2,
        );
        let mut hashes = Vec::new();
        for _ in 0..200 {
            w.step(1.0 / 60.0);
            let s = read_soft(&w);
            let mut h: u32 = 2166136261;
            for f in &s {
                for b in f.to_bits().to_le_bytes() {
                    h ^= b as u32;
                    h = h.wrapping_mul(16777619);
                }
            }
            hashes.push(h);
        }
        hashes
    };
    assert_eq!(run(), run(), "identical soft sims => identical per-tick hashes");
}

#[test]
fn deterministic_same_inputs_same_hashes() {
    let run = |seed: u32| {
        let mut w = build(seed);
        let mut hashes = Vec::new();
        for i in 0..300 {
            let a = (i as f32 * 0.05).sin();
            let b = (i as f32 * 0.03).cos();
            w.set_player_input(a, b, i % 90 == 0);
            w.step(1.0 / 60.0);
            hashes.push(hash(&w));
        }
        hashes
    };
    assert_eq!(run(42), run(42), "identical inputs => identical per-tick hashes");
}

#[test]
fn trampoline_bounces_a_ball() {
    let mut w = PhysicsWorld::new(-12.0);
    w.add_pit(-6.0, 6.0, -6.0, 6.0, 0.0, 6.0, 0.5);
    // A horizontal bed pinned all around its border (pin_mode 4), 2 m above the
    // pit floor: u -> +x, v -> +z so the sheet lies flat.
    w.add_cloth(
        -6.0, 2.0, -6.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 12.0, 12.0, 16, 16, 5.0, 0.002, 0.001, 0.3,
        0.15, 0.2, 0.3, 15, 5, 4, 0.5,
    );
    // Drop a ball straight down onto the middle of the bed.
    let ball = w.add_ball(0.0, 6.0, 0.0, 0.5, 2.0, 0.3, 0.4, 0.5) as usize;
    let y_of = |w: &PhysicsWorld| read_transforms(w)[ball * 7 + 1];

    let mut ys = Vec::new();
    for _ in 0..240 {
        w.step(1.0 / 60.0);
        ys.push(y_of(&w));
    }

    let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
    // It must not punch through the bed to the pit floor (a supported ball sits
    // near y = 2.65; the floor would put it near y = 0.5).
    assert!(min_y > 1.2, "ball stayed on the trampoline (min y = {min_y})");

    // After its deepest dip it must spring clearly back up — a trampoline, not a
    // cushion that merely absorbs the fall.
    let dip = ys.iter().position(|&y| y == min_y).unwrap();
    let rebound = ys[dip..].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        rebound > min_y + 1.0,
        "ball bounced back up (dip {min_y} -> rebound {rebound})"
    );
}
