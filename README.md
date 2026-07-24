# Squirrel Engine

The merged Squirrel Engine: the **Rapier-based physics core** and the
**WebGPU 3D renderer**, brought into one project and linked through a unified
**scene** layer. Both were previously separate projects
(`clientsidePhysDevelopment` and `3dEngineDevelopment`) that already cooperated
— the physics core packed body transforms and the renderer consumed them in
"driven" mode. This project makes that relationship first-class.

## Layout

```
actualSquirrelEngine/
  physics/            Rust crate → squirrel_physics.wasm  (Rapier sim + soft bodies)
    src/{lib,forces,tiers}.rs
    src/soft.rs       our own XPBD cloth / soft-body solver
    tests/smoke.rs    headless determinism + smoke tests
  render/             Rust crate → squirrel_engine.wasm   (WebGPU renderer)
    src/{lib,bvh,math,mesh}.rs   (mesh::plane = scalable rectangle grid)
  web/
    pkg/              generated: squirrel_physics*         (built by build.ps1)
    engine/           squirrel_engine*  +  renderer.js     (JS/WebGPU layer)
    shaders/          *.wgsl
    js/
      scene.js        the unified scene: GameObject linking + the two toggles
      main.js         the merged demo (ballpit) built on the scene
    index.html
  serve.mjs           static dev server (http://localhost:8083)
  build.ps1           builds both cores and stages them into web/
```

Two WASM modules, not one: each core still holds only its own state and never
sees the other's handles. They are joined at the **scene** (`web/js/scene.js`),
which is exactly how the codebase is architected — JS owns integration.

## The unified object model

A `GameObject` is one logical thing in the world. It links:

* a **physics body** — an index into the physics core's packed transform buffer
  (or none, for a render-only prop), and
* a **render instance** — the color + radius the renderer draws in driven mode.

Every object carries two independent toggles:

| Toggle          | `false` means                                                        |
| --------------- | -------------------------------------------------------------------- |
| `obj.render`    | skipped when building driven data — the renderer never draws it.     |
| `obj.physics`   | frozen in place: the Rust core switches the body to a `Fixed` body, so it stops falling / being shoved (still a static collider). Set `true` to thaw. |

So an object can be simulated-but-invisible, visible-but-frozen, both, or
neither. The physics toggle is backed by `PhysicsWorld::set_ball_enabled` in the
Rust core (`physics/src/lib.rs`); the render toggle is pure scene bookkeeping.

```js
const scene = new Scene(world, engine, physWasm);
const ball = scene.addBall({ x: 0, y: 5, z: 0, radius: 0.4 });
ball.physics = false;  // freeze it where it is, keep drawing it
ball.render  = false;  // stop drawing it, keep simulating it
```

## Soft bodies & cloth

Rapier is a rigid-body solver, so cloth and deformables are a separate, small
**XPBD** (position-based) solver in [`physics/src/soft.rs`](physics/src/soft.rs):
a soft body is a bag of point masses wired by distance constraints. XPBD is used
because it is unconditionally stable (no stiff-spring blow-up) and its
`compliance` parameter means "how stretchy" independent of the timestep. Soft
bodies collide (one-way) against the pit floor/walls and the rigid spheres
(balls + player) each tick, so cloth drapes over the pile and the player.

Soft bodies render as **real deforming surfaces**, not points: each body carries
a surface triangle list (the full grid for cloth, the six outer faces for a
box), and a dedicated two-sided lit pipeline
([softmesh.wgsl](web/shaders/softmesh.wgsl) + the soft-mesh methods in
[renderer.js](web/engine/renderer.js)) re-skins the mesh from the particle
positions every frame — recomputing normals — lit by the same sun/sky as the
rest of the scene. One **new object type**, `SoftObject` in
[scene.js](web/js/scene.js), links a soft body to its mesh and carries the same
`render` / `physics` toggles as a rigid `GameObject` (`physics = false` freezes
it via `PhysicsWorld::set_soft_enabled`).

Two builders, both fully parameterized (mass, compliance, bend, damping,
friction, particle/render radius, solver iterations & substeps, pinning):

```js
// A scalable rectangle of nu×nv particles — the cloth mesh.
const cloth = scene.addCloth({
  origin: [-7, 5, -3], u: [1, 0, 0], v: [0, -1, 0],
  width: 14, height: 4, nu: 24, nv: 12,
  pin: "edge",            // "none" | "edge" | "corners" | "four"
  compliance: 0, bendCompliance: 0.0005, damping: 0.5,
});
cloth.physics = false;    // freeze it in place
cloth.render  = false;    // stop drawing it (still simulated)

// A squishy solid lattice.
const box = scene.addSoftBox({
  center: [3, 7, 3], size: [1.6, 1.6, 1.6], nx: 5, ny: 5, nz: 5,
  compliance: 0.001, pinTop: false,
});
```

The render core also gained [`mesh::plane(width, height, nx, ny)`](render/src/mesh.rs)
— a scalable rectangle grid with the requested vertex counts (a ground plane, a
wall, or the render mesh matching a cloth of the same counts).

## Demo controls

`node serve.mjs`, then open <http://localhost:8083> (WebGPU browser — Chrome/Edge
113+). A ballpit of 240 plain balls + 16 buoyant balls + a kinematic player,
plus a ring of render-only props.

* **WASD** move · **Space** jump · **mouse** look (click to capture) · **wheel** zoom
* **F** — freeze / thaw physics on the plain balls (they stay drawn)
* **V** — show / hide the buoyant balls (they stay simulated)
* **G** — show / hide the render-only props (never in the sim)
* **C** — freeze / thaw the cloth curtain (physics toggle on a soft body)
* **B** — show / hide the soft box (render toggle on a soft body)

## Build

```
./build.ps1          # builds both WASM cores into web/
node serve.mjs
```

Or build a core by hand (run wasm-pack from a shell that doesn't mangle its
`[INFO]` stderr — e.g. bash, not PowerShell 5.1):

```
cd physics && wasm-pack build --target web --out-dir ../web/pkg --release
cd render  && wasm-pack build --target web --release   # then copy pkg/squirrel_engine* → web/engine
```

## Test

```
cd physics && cargo test --release
cd render  && cargo test --release
```
