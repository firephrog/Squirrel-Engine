//! Custom force accumulation, applied pre-step to Advanced-tier bodies only
//! (report Part 5).
//!
//! This is the "modification" the report calls for: Rapier's solver is left
//! alone, and we run our own force pass *in the core* over a dense contiguous
//! table before handing velocities to the pipeline. No per-body callback ever
//! crosses the WASM boundary — the whole point of doing it in Rust.
//!
//! Two numerical traps from the report, both handled:
//!   1. Quadratic drag is stiff at speed — explicit Euler explodes. The linear
//!      part of drag is integrated *semi-implicitly*.
//!   2. Buoyancy is a stiff restoring force near the surface; we clamp the net
//!      buoyant acceleration so no config can turn a bob into a cannon.

use rapier3d::prelude::*;

/// Buoyancy against a flat fluid surface at `fluid_level_y`.
#[derive(Clone)]
pub struct Buoyancy {
    pub fluid_density: Real,
    pub volume: Real,
    pub fluid_level_y: Real,
    pub linear_drag: Real,
}

/// Air resistance: quadratic drag `F = -½ρv²·Cd·A` plus an optional Magnus term.
#[derive(Clone)]
pub struct Drag {
    pub cd: Real,
    pub ref_area: Real,
    pub air_density: Real,
    pub magnus: Real,
}

/// One entry in the dense aero side-table.
#[derive(Clone, Default)]
pub struct AeroProps {
    pub buoyancy: Option<Buoyancy>,
    pub drag: Option<Drag>,
}

/// Pure transform of a body's velocity for one tick. Returns the new linear
/// velocity; the caller writes it back onto the Rapier body before stepping.
/// Gravity is applied by Rapier itself, so buoyancy here is the *upward* term
/// that counteracts it — not net weight.
pub fn apply_aero(
    mut v: Vector,
    pos_y: Real,
    mass: Real,
    radius: Real,
    p: &AeroProps,
    gravity_y: Real,
    dt: Real,
) -> Vector {
    if mass <= 0.0 {
        return v;
    }
    let inv_mass = 1.0 / mass;

    // --- Buoyancy + fluid linear drag (only while submerged) ---------------
    if let Some(b) = &p.buoyancy {
        let bottom = pos_y - radius;
        let submersion = (((b.fluid_level_y - bottom) / (2.0 * radius)).clamp(0.0, 1.0)) as Real;
        if submersion > 0.0 {
            // Archimedes: F = ρ_fluid · V_displaced · -g (g is negative → up).
            let displaced = b.volume * submersion;
            let buoy_force = -gravity_y * b.fluid_density * displaced;
            // Safety clamp: cap net buoyant acceleration to a sane multiple of
            // gravity so a wildly buoyant config can't blow up explicit Euler.
            let max_accel = gravity_y.abs() * 4.0;
            let buoy_accel = (buoy_force * inv_mass).min(max_accel);
            v.y += buoy_accel * dt;

            // Fluid resists motion far more than air — semi-implicit so a high
            // drag coefficient stays unconditionally stable.
            if b.linear_drag > 0.0 {
                let decay = 1.0 / (1.0 + b.linear_drag * submersion * dt * inv_mass);
                v *= decay;
            }
        }
    }

    // --- Dynamic (quadratic) air resistance --------------------------------
    if let Some(d) = &p.drag {
        let speed = v.length();
        if speed > 1e-6 {
            // k grouping the ½ρCdA constant; F = -k·|v|·v.
            let k = 0.5 * d.air_density * d.cd * d.ref_area;
            // Semi-implicit: v_new = v / (1 + k·|v|·dt/m).
            let scale = 1.0 / (1.0 + k * speed * dt * inv_mass);
            v *= scale;

            // Magnus: curve fast movers. We don't track spin for balls, so
            // approximate a spin axis of world-up crossed with velocity.
            if d.magnus != 0.0 {
                let ax = -v.z;
                let az = v.x;
                v.x += ax * d.magnus * inv_mass * dt;
                v.z += az * d.magnus * inv_mass * dt;
            }
        }
    }

    v
}
