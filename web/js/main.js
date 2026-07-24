// Squirrel Engine — merged demo.
//
// One app, two WASM cores, linked through the unified Scene layer (scene.js):
//   * squirrel_physics — the Rapier-based sim (bodies, forces, player).
//   * squirrel_engine  — the WebGPU renderer, run in "driven" mode.
//
// Every ball, the player, and each decorative prop is a single GameObject that
// owns both a physics body and a render instance. The two toggles the merge is
// about are demonstrated live:
//   F — freeze / thaw physics on the plain balls (they stop falling, stay drawn)
//   V — show / hide the buoyant balls              (still simulated, not drawn)
//   G — show / hide the render-only props          (drawn, never in the sim)
//   C — freeze / thaw the cloth (soft body)        (physics toggle)
//   B — show / hide the soft box                   (render toggle)
//
// Loop: fixed-timestep physics (accumulator), interpolate, sync driven data,
// render.

import initPhysics, { PhysicsWorld } from "../pkg/squirrel_physics.js";
import initEngine, { Engine } from "../engine/squirrel_engine.js";
import { Renderer } from "../engine/renderer.js";
import { Scene } from "./scene.js";

// --- Diagnostics -----------------------------------------------------------
function showError(title, detail) {
  const el = document.getElementById("hud");
  if (el) {
    el.innerHTML =
      `<b style="color:#ff6b6b">${title}</b><br>` +
      `<span class="dim">${String(detail).replace(/</g, "&lt;")}</span>`;
  }
  console.error(title, detail);
}
addEventListener("error", (e) => showError("Script error", e.message || e.error || e));
addEventListener("unhandledrejection", (e) => showError("Unhandled rejection", e.reason));
if (location.protocol === "file:") {
  showError("Open this over http, not file://", "Run  node serve.mjs  then visit http://localhost:8083");
  throw new Error("file:// — start the local server");
}

const FIXED_DT = 1 / 60;
const MAX_INSTANCES = 4096;

const PIT = {
  minX: -8, maxX: 8, minZ: -8, maxZ: 8,
  floorY: 0, wallTop: 6, fluidLevelY: 1.6,
};
const SIMPLE_BALLS = 200;
const BUOYANT_BALLS = 0;
const rand = (a, b) => a + Math.random() * (b - a);

async function main() {
  // 1. Boot the physics module (the engine module was booted in boot()).
  const physWasm = await initPhysics();

  // 2. Build the physics world + the WebGPU engine (driven mode, sphere mesh).
  const world = new PhysicsWorld(-12.0);
  world.add_pit(PIT.minX, PIT.maxX, PIT.minZ, PIT.maxZ, PIT.floorY, PIT.wallTop, 0.5);

  const engine = new Engine(MAX_INSTANCES);
  engine.init_scene(1);           // one instance → valid (tiny) RT scene buffers
  engine.use_sphere_mesh(20, 28); // swap cube → sphere BEFORE renderer.init

  // 3. Author the world through the unified Scene. Each call makes a GameObject
  //    that links a physics body to a render instance.
  const scene = new Scene(world, engine, physWasm);

  const simpleBalls = [];
  for (let i = 0; i < SIMPLE_BALLS; i++) {
    const r = rand(0.32, 0.5);
    simpleBalls.push(scene.addBall({
      // Spawn above the trampoline bed (y ≈ 1.6) so every ball rains down onto it.
      x: rand(PIT.minX + 1, PIT.maxX - 1), y: rand(3.5, 10), z: rand(PIT.minZ + 1, PIT.maxZ - 1),
      radius: r,
    }));
  }

  const buoyantBalls = [];
  for (let i = 0; i < BUOYANT_BALLS; i++) {
    const r = rand(0.6, 0.85);
    buoyantBalls.push(scene.addBuoyantBall({
      x: rand(PIT.minX + 1, PIT.maxX - 1), y: rand(3, 8), z: rand(PIT.minZ + 1, PIT.maxZ - 1),
      radius: r, fluidLevelY: PIT.fluidLevelY,
    }));
  }

  const player = scene.addPlayer({
    x: 0, y: 0.7, z: 0, radius: 0.7,
    bounds: { minX: PIT.minX, maxX: PIT.maxX, minZ: PIT.minZ, maxZ: PIT.maxZ, floorY: PIT.floorY },
  });

  // Render-only props: a slow ring of markers floating above the pit. They are
  // drawn by the engine but never touch the physics world (physics permanently
  // off), so balls pass straight through them.
  const props = [];
  const PROP_COUNT = 8;
  for (let i = 0; i < PROP_COUNT; i++) {
    const a = (i / PROP_COUNT) * Math.PI * 2;
    props.push(scene.addProp({
      x: Math.cos(a) * 6, y: 5.5, z: Math.sin(a) * 6, radius: 0.4,
      color: [1.0, 0.4, 0.55],
    }));
  }

  // A trampoline for the pit floor: a horizontal soft-body bed pinned all around
  // its border so it stays taut. Balls, the soft box, and the player that land on
  // it get bounced back up — the soft solver now feeds a restitution reaction
  // back into the rigid bodies (the push momentum sinks into the bed's pins, the
  // same way a real anchored trampoline works). It sits ~2.2 above the real floor
  // so there's slack to deflect into and clearance over the player beneath it.
  const trampoline = scene.addCloth({
    origin: [PIT.minX + 0.5, 2.2, PIT.minZ + 0.5],
    u: [1, 0, 0],
    v: [0, 0, 1], // lies flat in the XZ plane
    width: PIT.maxX - PIT.minX - 1, height: PIT.maxZ - PIT.minZ - 1,
    nu: 20, nv: 20,
    mass: 8.0,
    compliance: 0.003,       // taut + springy so it flings, not just cushions
    bendCompliance: 0.002,
    damping: 0.4, friction: 0.3,
    iterations: 15, substeps: 5,
    pin: "border",
    color: [0.3, 0.62, 0.95],
  });

  // A hanging cloth curtain, pinned along its top edge (v runs downward), and a
  // squishy soft box dropped above the pit. Both are soft bodies: bags of
  // particles solved by our XPBD pass, drawn as one small sphere per particle,
  // colliding with the pit, the balls, and the player.
  const cloth = scene.addCloth({
    origin: [PIT.minX + 1, 5.0, -3.0],
    u: [1, 0, 0],
    v: [0, -1, 0],
    width: PIT.maxX - PIT.minX - 2, height: 4.0,
    nu: 24, nv: 12,
    pin: "edge",
    color: [0.95, 0.45, 0.55],
  });
  const softBox = scene.addSoftBox({
    center: [3.0, 6.0, 3.0], size: [4, 4, 4],
    nx: 5, ny: 5, nz: 5,
    mass: 10,
    compliance: 0.1, // soft jelly — squashes and wobbles on impact
    damping: 0.95,
    shape_stiffness: 0.5,
    color: [0.55, 0.9, 0.6],
  });

  // 4. Prime physics + enable driven mode, then bring up WebGPU.
  scene.finalize(FIXED_DT);

  const canvas = document.getElementById("gpu-canvas");
  const renderer = new Renderer(engine, engineModule);
  const shaderFiles = { raster: "main", model: "model", raytrace: "raytrace", blit: "blit", sky: "sky", skybox: "skybox", softmesh: "softmesh" };
  const shaders = {};
  await Promise.all(
    Object.entries(shaderFiles).map(async ([key, file]) => {
      shaders[key] = await fetch(`./shaders/${file}.wgsl`).then((r) => r.text());
    }),
  );

  resizeCanvas();
  await renderer.init(canvas, shaders, MAX_INSTANCES);
  renderer.resize(canvas.width, canvas.height);

  // Hook the renderer up to the scene and build the soft-body meshes (cloth +
  // soft box) now that the pipeline exists.
  scene.attachRenderer(renderer);
  scene.createSoftMeshes();

  // Lighting + sky: a warm sun, gentle sky ambient.
  renderer.setSunDirection(0.45, 0.85, 0.35, 1.25);
  engine.set_sun_color(1.0, 0.96, 0.88, 0.14);
  engine.set_env_params(0.75, 0.22, 1.0);
  engine.set_point_light(0, 0, 12, 0, 60, 1.0, 0.85, 0.6, 2.2);

  // 5. Input + follow camera.
  const cam = { yaw: 0.6, pitch: 0.5, dist: 20 };
  const keys = new Set();
  const toggles = {
    physFrozen: false, buoyantHidden: false, propsHidden: false,
    clothFrozen: false, boxHidden: false,
  };

  addEventListener("keydown", (e) => {
    if (e.repeat) { if (e.code === "Space") e.preventDefault(); return; }
    keys.add(e.code);
    if (e.code === "Space") e.preventDefault();

    // --- The unified-object toggles ---
    if (e.code === "KeyF") {          // physics toggle on the plain balls
      toggles.physFrozen = !toggles.physFrozen;
      for (const b of simpleBalls) b.physics = !toggles.physFrozen;
    } else if (e.code === "KeyV") {   // render toggle on the buoyant balls
      toggles.buoyantHidden = !toggles.buoyantHidden;
      for (const b of buoyantBalls) b.render = !toggles.buoyantHidden;
    } else if (e.code === "KeyG") {   // render toggle on the render-only props
      toggles.propsHidden = !toggles.propsHidden;
      for (const p of props) p.render = !toggles.propsHidden;
    } else if (e.code === "KeyC") {   // physics toggle on the cloth (freeze it)
      toggles.clothFrozen = !toggles.clothFrozen;
      cloth.physics = !toggles.clothFrozen;
    } else if (e.code === "KeyB") {   // render toggle on the soft box
      toggles.boxHidden = !toggles.boxHidden;
      softBox.render = !toggles.boxHidden;
    }
  });
  addEventListener("keyup", (e) => keys.delete(e.code));
  canvas.addEventListener("click", () => canvas.requestPointerLock());
  addEventListener("mousemove", (e) => {
    if (document.pointerLockElement !== canvas) return;
    cam.yaw -= e.movementX * 0.0028;
    cam.pitch = clamp(cam.pitch - e.movementY * 0.0028, 0.12, 1.3);
  });
  addEventListener("wheel", (e) => { cam.dist = clamp(cam.dist + Math.sign(e.deltaY) * 1.5, 7, 40); }, { passive: true });

  function moveBasis() {
    const f = norm3(-Math.sin(cam.yaw), 0, -Math.cos(cam.yaw));
    const r = [-f[2], 0, f[0]];
    return { f, r };
  }
  function pushInput() {
    const { f, r } = moveBasis();
    let x = 0, z = 0;
    if (keys.has("KeyW")) { x += f[0]; z += f[2]; }
    if (keys.has("KeyS")) { x -= f[0]; z -= f[2]; }
    if (keys.has("KeyD")) { x += r[0]; z += r[2]; }
    if (keys.has("KeyA")) { x -= r[0]; z -= r[2]; }
    const len = Math.hypot(x, z);
    if (len > 1e-4) { x /= len; z /= len; }
    scene.setPlayerInput(x, z, keys.has("Space"));
  }

  // 6. Main loop.
  const hud = document.getElementById("hud");
  let accumulator = 0, last = performance.now(), fps = 60, physMs = 0;

  function frame(now) {
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    fps = fps * 0.9 + (1 / Math.max(dt, 1e-3)) * 0.1;

    pushInput();
    accumulator += dt;
    let steps = 0;
    const t0 = performance.now();
    while (accumulator >= FIXED_DT && steps < 5) {
      scene.beginStep();
      world.step(FIXED_DT);
      scene.endStep();
      accumulator -= FIXED_DT;
      steps++;
    }
    if (steps > 0) physMs = physMs * 0.9 + ((performance.now() - t0) / steps) * 0.1;

    const alpha = accumulator / FIXED_DT;
    const [px, py, pz] = scene.sync(alpha); // physics → driven data + player pos

    // Follow camera on the player.
    const tx = px, ty = py + 1.0, tz = pz;
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const ex = tx + cam.dist * cp * Math.sin(cam.yaw);
    const ey = ty + cam.dist * sp;
    const ez = tz + cam.dist * cp * Math.cos(cam.yaw);
    engine.set_camera(ex, ey, ez, tx, ty, tz);

    renderer.frame(dt); // engine.update() consumes driven data, then draws

    const onoff = (b) => (b ? '<span style="color:#7fe08f">on</span>' : '<span style="color:#ff8f8f">off</span>');
    hud.innerHTML =
      `<b>Squirrel Engine Ballpit Test</b><br>` +
      `${fps.toFixed(0)} fps &nbsp;·&nbsp; physics ${physMs.toFixed(2)} ms/tick &nbsp;·&nbsp; tick ${world.tick()}<br>` +
      `bodies ${scene.bodyCount} &nbsp;·&nbsp; simple ${world.count_simple()} · advanced ${world.count_advanced()} · asleep ${world.count_asleep()}<br>` +
      `soft particles ${scene.softParticleTotal} &nbsp;·&nbsp; cloth 24×12 · soft box 5³<br>` +
      `<b>F</b> ball physics ${onoff(!toggles.physFrozen)} &nbsp;·&nbsp; ` +
      `<b>V</b> buoyant render ${onoff(!toggles.buoyantHidden)} &nbsp;·&nbsp; ` +
      `<b>G</b> prop render ${onoff(!toggles.propsHidden)} &nbsp;·&nbsp; ` +
      `<b>C</b> cloth physics ${onoff(!toggles.clothFrozen)} &nbsp;·&nbsp; ` +
      `<b>B</b> box render ${onoff(!toggles.boxHidden)}<br>`;

    requestAnimationFrame(frame);
  }

  function resizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.floor((canvas.clientWidth || window.innerWidth) * dpr);
    const h = Math.floor((canvas.clientHeight || window.innerHeight) * dpr);
    if (renderer.device) renderer.resize(w, h);
    else { canvas.width = w; canvas.height = h; }
  }
  addEventListener("resize", resizeCanvas);
  requestAnimationFrame(frame);
}

// The engine's init() returns the module object (with `.memory`) the Renderer
// needs; capture it once here.
let engineModule = null;
async function boot() {
  engineModule = await initEngine();
  await main();
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
function norm3(x, y, z) { const l = Math.hypot(x, y, z) || 1; return [x / l, y / l, z / l]; }

boot();
