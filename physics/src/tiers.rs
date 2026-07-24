//! Body tiers, defined by cost-per-tick (report Part 4).
//!
//! Tier is a *property*, not a subclass: `Simple` and `Advanced` bodies are the
//! same Rapier rigid body with the same collider and the same snapshot layout.
//! The only difference is that an `Advanced` body carries an index into a dense
//! aero side-table and gets a pre-step custom-force pass; a `Simple` body isn't
//! in that table and so costs nothing extra.

/// Cosmetic, client-only. (Reserved — the demo doesn't spawn these yet, but the
/// renderer and packing already carry the tag so it's an additive change.)
pub const KINEMATIC: u32 = 0;
/// One Rapier rigid body: gravity, damping, contacts. No pre-step work.
pub const SIMPLE: u32 = 1;
/// Simple + an entry in the aero/buoyancy table (drag, magnus, buoyancy).
pub const ADVANCED: u32 = 2;
/// Not a rigid body — reserved for a future particle/heightfield solver.
pub const FLUID: u32 = 3;

/// Render-only marker so the renderer can tint the player distinctly. Never a
/// simulation tier.
pub const PLAYER_MARKER: f32 = 100.0;
