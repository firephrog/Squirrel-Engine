// Squirrel Engine — the unified scene layer.
//
// This is where the two cores are *linked*. Each core still does one job:
//   * squirrel_physics (Rapier/WASM) owns rigid-body state — it packs a flat
//     buffer of [x, y, z, radius, tier, color_seed, awake] per body each tick.
//   * squirrel_engine  (WebGPU/WASM) draws in "driven" mode — each frame we hand
//     it a flat buffer of [x, y, z, radius, r, g, b] per instance.
//
// A `GameObject` is one logical thing in the world that ties those two
// representations together: a physics body (by index into the physics buffer)
// and a render instance (a color + radius the engine draws). The `Scene` owns
// the objects and does the per-frame bridge — read physics → interpolate →
// write driven data.
//
// The two toggles the whole merge is about live on the object:
//   * obj.render  = false  → the object is skipped when building driven data,
//                            so the engine never draws it (physics still runs).
//   * obj.physics = false  → the physics body is frozen in place (Rust switches
//                            it to a Fixed body), so it stops falling / being
//                            shoved, but is still drawn. Set it true to thaw.
// The two flags are independent: an object can be simulated-but-invisible,
// visible-but-frozen, both, or neither.

const PHYS_FLOATS = 7; // physics buffer stride: x,y,z,radius,tier,seed,awake
const DRIVE_FLOATS = 7; // engine driven-buffer stride: x,y,z,radius,r,g,b
const SOFT_FLOATS = 4; // soft-body particle stride: x,y,z,radius

const PIN_MODES = { none: 0, edge: 1, corners: 2, four: 3, border: 4 };

const PLAYER_COLOR = [1.0, 0.82, 0.15];
const ADVANCED_COLOR = [0.35, 0.75, 1.0];

/** One thing in the world: a linked (physics body, render instance) pair. */
export class GameObject {
  constructor(scene, { physicsIndex = -1, isPlayer = false, radius, color, pos }) {
    this.scene = scene;
    // Slot into the packed physics buffer, or -1 for a render-only object with
    // no rigid body at all (its transform is whatever `pos` holds).
    this.physicsIndex = physicsIndex;
    this.isPlayer = isPlayer;
    this.radius = radius;
    this.color = color; // [r, g, b]
    this.pos = pos ? pos.slice() : [0, 0, 0]; // used when physicsIndex < 0
    this._render = true;
    this._physics = true;
  }

  /** Whether the engine draws this object. */
  get render() {
    return this._render;
  }
  set render(on) {
    this._render = !!on;
  }

  /** Whether physics is applied to this object (freeze in place when false). */
  get physics() {
    return this._physics;
  }
  set physics(on) {
    on = !!on;
    if (on === this._physics) return;
    this._physics = on;
    // Only bodies that live in the physics world can be toggled; a render-only
    // prop is already "physics off" permanently, and the player is kinematic
    // (its freeze is handled by suppressing input in Scene.setPlayerInput).
    if (this.physicsIndex >= 0 && !this.isPlayer) {
      this.scene.world.set_ball_enabled(this.physicsIndex, on);
    }
  }

  /** Recolor the render instance. */
  setColor(r, g, b) {
    this.color = [r, g, b];
  }
}

/**
 * A soft body or cloth: one physics soft-body (a bag of particles) linked to a
 * run of render instances (one small sphere per particle). Carries the same two
 * toggles as a rigid GameObject.
 */
export class SoftObject {
  constructor(scene, { softId, count, offset, color }) {
    this.scene = scene;
    this.softId = softId; // index into the physics soft-body list
    this.count = count; // particle count
    this.offset = offset; // first particle's index in the packed soft buffer
    this.color = color; // [r, g, b] surface tint
    this.mesh = null; // renderer soft-mesh handle (set by Scene.createSoftMeshes)
    this._pos = null; // scratch xyz buffer for per-frame re-skinning
    this._render = true;
    this._physics = true;
  }

  get render() {
    return this._render;
  }
  set render(on) {
    this._render = !!on;
  }

  /** Freeze / thaw the whole soft body (Rust makes solve a no-op when off). */
  get physics() {
    return this._physics;
  }
  set physics(on) {
    on = !!on;
    if (on === this._physics) return;
    this._physics = on;
    this.scene.world.set_soft_enabled(this.softId, on);
  }

  setColor(r, g, b) {
    this.color = [r, g, b];
    if (this.mesh && this.scene.renderer) {
      this.scene.renderer.setSoftMeshColor(this.mesh, this.color);
    }
  }
}

export class Scene {
  /**
   * @param world     PhysicsWorld  (from squirrel_physics)
   * @param engine    Engine        (from squirrel_engine, driven mode)
   * @param physWasm  the physics wasm module (for `.memory` — the buffer view)
   */
  constructor(world, engine, physWasm) {
    this.world = world;
    this.engine = engine;
    this.physWasm = physWasm;
    this.renderer = null; // set via attachRenderer() once WebGPU is up

    this.objects = []; // rigid GameObjects (balls, player, props), in order
    this.player = null;
    this.ballCount = 0; // rigid bodies added via add*Ball / addPlayer

    this.softBodies = []; // SoftObjects (cloth + soft boxes), in creation order
    this.softParticleTotal = 0; // total particles across all soft bodies

    // Per-frame physics snapshots for interpolation, sized in finalize().
    this.bodyCount = 0;
    this.prev = null;
    this.curr = null;
    this.drive = null;
    this.softPrev = null;
    this.softCurr = null;

    this._physBuffer = null;
    this._physView = null;
    this._softBuffer = null;
    this._softView = null;
  }

  // --- Authoring ----------------------------------------------------------

  /** A plain dynamic ball. Returns its GameObject. */
  addBall(opts) {
    const {
      x, y, z, radius,
      mass = radius * radius * radius * 8,
      restitution = 0.45, friction = 0.55,
      color = colorForSeed(Math.random()),
      seed = Math.random(),
      render = true, physics = true,
    } = opts;
    const idx = this.world.add_ball(x, y, z, radius, mass, restitution, friction, seed);
    const obj = new GameObject(this, { physicsIndex: idx, radius, color });
    this._register(obj, render, physics);
    this.ballCount++;
    return obj;
  }

  /** A buoyant + drag-affected ball (Advanced tier). Returns its GameObject. */
  addBuoyantBall(opts) {
    const {
      x, y, z, radius,
      mass = radius * radius * radius * 1.2,
      restitution = 0.55, friction = 0.4,
      fluidDensity = 0.7, volume = (4 / 3) * Math.PI * radius ** 3,
      fluidLevelY, linearDrag = 6.0,
      cd = 0.47, refArea = Math.PI * radius * radius, airDensity = 1.225, magnus = 0.0,
      color = ADVANCED_COLOR.slice(),
      render = true, physics = true,
    } = opts;
    const idx = this.world.add_ball(x, y, z, radius, mass, restitution, friction, 0.5);
    this.world.set_buoyancy(idx, fluidDensity, volume, fluidLevelY, linearDrag);
    this.world.set_drag(idx, cd, refArea, airDensity, magnus);
    const obj = new GameObject(this, { physicsIndex: idx, radius, color });
    this._register(obj, render, physics);
    this.ballCount++;
    return obj;
  }

  /** The kinematic player. Must be added AFTER every ball (it packs last). */
  addPlayer(opts) {
    const { x, y, z, radius, bounds } = opts;
    this.world.add_player(x, y, z, radius, bounds.minX, bounds.maxX, bounds.minZ, bounds.maxZ, bounds.floorY);
    // The player is the last body in the packed physics buffer.
    const obj = new GameObject(this, {
      physicsIndex: this.ballCount, isPlayer: true, radius, color: PLAYER_COLOR.slice(),
    });
    this._register(obj, true, true);
    this.player = obj;
    return obj;
  }

  /** A render-only prop: drawn at a fixed spot, never in the physics world. */
  addProp(opts) {
    const { x, y, z, radius, color = [0.8, 0.8, 0.85], render = true } = opts;
    const obj = new GameObject(this, { physicsIndex: -1, radius, color, pos: [x, y, z] });
    this._register(obj, render, false);
    return obj;
  }

  /**
   * A cloth sheet: a scalable rectangle of `nu * nv` particles spanning `width`
   * along `u` and `height` along `v` from the corner `origin`. `pin` is one of
   * "none" | "edge" | "corners" | "four". Returns a SoftObject.
   */
  addCloth(opts) {
    const {
      origin, u, v, width, height, nu, nv,
      mass = 1.0,
      compliance = 0.0, bendCompliance = 0.0005,
      damping = 0.5, friction = 0.3,
      iterations = 12, substeps = 4,
      pin = "none",
      color = [0.92, 0.5, 0.35],
      render = true, physics = true,
    } = opts;
    const spacing = Math.min(width / Math.max(nu - 1, 1), height / Math.max(nv - 1, 1));
    const particleRadius = opts.particleRadius ?? spacing * 0.35;
    const renderRadius = opts.renderRadius ?? spacing * 0.6;
    const pinMode = PIN_MODES[pin] ?? 0;
    const id = this.world.add_cloth(
      origin[0], origin[1], origin[2],
      u[0], u[1], u[2],
      v[0], v[1], v[2],
      width, height, nu, nv, mass,
      compliance, bendCompliance, damping,
      particleRadius, renderRadius, friction,
      iterations, substeps, pinMode, Math.random(),
    );
    return this._registerSoft(id, color, render, physics);
  }

  /**
   * A squishy solid block: an `nx * ny * nz` particle lattice centered at
   * `center` with full extents `size`. `pinTop` dangles it. Returns a SoftObject.
   */
  addSoftBox(opts) {
    const {
      center, size, nx, ny, nz,
      mass = 1.5,
      // Soft by default: ~30x more compliant than a near-rigid body so it
      // visibly squashes and jiggles instead of holding a solid cube shape.
      compliance = 0.03,
      damping = 0.9, friction = 0.4,
      iterations = 8, substeps = 5,
      pinTop = false,
      // Per-substep pull toward the rigid rest pose. Low so the box visibly
      // squashes under gravity/impact; any value > 0 guarantees it springs back
      // to shape instead of staying crumpled. Set 0 for a pure floppy lattice.
      // Accept either spelling (camelCase or the Rust-style snake_case).
      shapeStiffness = opts.shape_stiffness ?? 0.03,
      color = [0.55, 0.85, 0.55],
      render = true, physics = true,
    } = opts;
    const minSpan = Math.min(
      size[0] / Math.max(nx - 1, 1),
      size[1] / Math.max(ny - 1, 1),
      size[2] / Math.max(nz - 1, 1),
    );
    const particleRadius = opts.particleRadius ?? minSpan * 0.35;
    const renderRadius = opts.renderRadius ?? minSpan * 0.55;
    const id = this.world.add_soft_box(
      center[0], center[1], center[2],
      size[0], size[1], size[2],
      nx, ny, nz, mass, compliance, damping,
      particleRadius, renderRadius, friction,
      iterations, substeps, pinTop, shapeStiffness, Math.random(),
    );
    return this._registerSoft(id, color, render, physics);
  }

  _register(obj, render, physics) {
    this.objects.push(obj);
    // Apply the requested initial flags (physics setter drives the Rust toggle).
    obj.render = render;
    if (!physics) obj.physics = false;
  }

  _registerSoft(id, color, render, physics) {
    const count = this.world.soft_body_particle_count(id);
    const offset = this.softParticleTotal;
    this.softParticleTotal += count;
    const obj = new SoftObject(this, { softId: id, count, offset, color });
    this.softBodies.push(obj);
    obj.render = render;
    if (!physics) obj.physics = false;
    return obj;
  }

  /**
   * Prime one physics tick, size the interpolation + driven buffers, and put the
   * engine into driven mode. Call once after all objects are authored.
   */
  finalize(fixedDt) {
    this.world.step(fixedDt);
    this.bodyCount = this.world.bodies_count();
    this.prev = this._readPhysics().slice();
    this.curr = this.prev.slice();

    if (this.softParticleTotal > 0) {
      this.softPrev = this._readSoft().slice();
      this.softCurr = this.softPrev.slice();
    } else {
      this.softPrev = new Float32Array(0);
      this.softCurr = new Float32Array(0);
    }

    // Rigid objects (balls + player + props) render as driven sphere instances;
    // soft bodies render as their own deforming meshes (see createSoftMeshes),
    // not through driven data — so the driven buffer only covers the rigid ones.
    this.drive = new Float32Array(this.objects.length * DRIVE_FLOATS);
    this.engine.enable_driven(this.objects.length);
  }

  /** Attach the renderer so soft bodies can be drawn as deforming meshes. */
  attachRenderer(renderer) {
    this.renderer = renderer;
  }

  /**
   * Build a deforming render mesh for every soft body from its surface topology.
   * Call once after the renderer is initialized and attached.
   */
  createSoftMeshes() {
    if (!this.renderer) return;
    for (const sb of this.softBodies) {
      const idxCount = this.world.soft_body_index_count(sb.softId);
      const idxPtr = this.world.soft_body_indices_ptr(sb.softId);
      // Copy the index list out of wasm memory (it can move if memory grows).
      const indices = new Uint32Array(this.physWasm.memory.buffer, idxPtr, idxCount).slice();
      sb.mesh = this.renderer.createSoftMesh(sb.count, indices, sb.color);
      sb._pos = new Float32Array(sb.count * 3);
    }
  }

  // --- Per-frame bridge ---------------------------------------------------

  /** Call right before world.step(): roll the current snapshot back to `prev`. */
  beginStep() {
    this.prev.set(this.curr);
    if (this.softParticleTotal > 0) this.softPrev.set(this.softCurr);
  }

  /** Call right after world.step(): capture the new state into `curr`. */
  endStep() {
    this.curr.set(this._readPhysics());
    if (this.softParticleTotal > 0) this.softCurr.set(this._readSoft());
  }

  /** Feed the player its movement command (suppressed when physics is off). */
  setPlayerInput(ix, iz, jump) {
    if (this.player && this.player.physics) {
      this.world.set_player_input(ix, iz, jump);
    }
  }

  /**
   * Build this frame's driven-instance buffer from the render-enabled objects
   * (interpolating rigid bodies by `alpha`), hand it to the engine, and return
   * the interpolated player position for the follow camera.
   */
  sync(alpha) {
    const { prev, curr, drive } = this;
    let n = 0;
    for (const obj of this.objects) {
      if (!obj.render) continue; // render toggle: skip → engine never draws it
      const d = n * DRIVE_FLOATS;
      const [px, py, pz] = this._objectPos(obj, alpha);
      drive[d] = px;
      drive[d + 1] = py;
      drive[d + 2] = pz;
      drive[d + 3] = obj.radius;
      drive[d + 4] = obj.color[0];
      drive[d + 5] = obj.color[1];
      drive[d + 6] = obj.color[2];
      n++;
    }

    this.engine.set_driven_data(drive.subarray(0, n * DRIVE_FLOATS));

    // Soft bodies render as deforming surfaces: re-skin each mesh from this
    // frame's interpolated particle positions, or hide it when render is off.
    if (this.softParticleTotal > 0 && this.renderer) {
      const { softPrev, softCurr } = this;
      for (const sb of this.softBodies) {
        if (!sb.mesh) continue;
        if (!sb.render) {
          sb.mesh.hidden = true;
          continue;
        }
        sb.mesh.hidden = false;
        const scratch = sb._pos;
        for (let k = 0; k < sb.count; k++) {
          const s = (sb.offset + k) * SOFT_FLOATS;
          const d = k * 3;
          scratch[d] = lerp(softPrev[s], softCurr[s], alpha);
          scratch[d + 1] = lerp(softPrev[s + 1], softCurr[s + 1], alpha);
          scratch[d + 2] = lerp(softPrev[s + 2], softCurr[s + 2], alpha);
        }
        this.renderer.updateSoftMesh(sb.mesh, scratch);
      }
    }

    // Player position for the camera — independent of the player's render flag.
    return this.player ? this._objectPos(this.player, alpha) : [0, 0, 0];
  }

  /** Interpolated world position of an object at blend factor `alpha`. */
  _objectPos(obj, alpha) {
    const i = obj.physicsIndex;
    if (i < 0) return obj.pos; // render-only prop: fixed transform
    const o = i * PHYS_FLOATS;
    const x = lerp(this.prev[o], this.curr[o], alpha);
    const y = lerp(this.prev[o + 1], this.curr[o + 1], alpha);
    const z = lerp(this.prev[o + 2], this.curr[o + 2], alpha);
    obj.pos[0] = x; obj.pos[1] = y; obj.pos[2] = z; // cache last-known
    return obj.pos;
  }

  // A fresh Float32Array over the physics transform buffer. Re-created if the
  // wasm memory grew and detached the old ArrayBuffer.
  _readPhysics() {
    const buf = this.physWasm.memory.buffer;
    if (buf !== this._physBuffer) {
      this._physBuffer = buf;
      this._physView = null;
    }
    if (!this._physView || this._physView.length !== this.bodyCount * PHYS_FLOATS) {
      this._physView = new Float32Array(buf, this.world.bodies_ptr(), Math.max(1, this.bodyCount) * PHYS_FLOATS);
    } else {
      // Same buffer, right length — but the pointer could shift; rebind to be safe.
      this._physView = new Float32Array(buf, this.world.bodies_ptr(), this.bodyCount * PHYS_FLOATS);
    }
    return this._physView;
  }

  // A fresh Float32Array over the packed soft-body particle buffer, rebound each
  // call (pointer/length are stable after finalize, but the wasm ArrayBuffer can
  // be swapped out from under us if memory grows).
  _readSoft() {
    const buf = this.physWasm.memory.buffer;
    const len = this.world.soft_particle_count() * SOFT_FLOATS;
    this._softBuffer = buf;
    this._softView = new Float32Array(buf, this.world.soft_particles_ptr(), len);
    return this._softView;
  }
}

// --- Small color helpers (ported from the original ballpit demo) -----------

function colorForSeed(seed) {
  return hsl(seed, 0.6, 0.55);
}
function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
