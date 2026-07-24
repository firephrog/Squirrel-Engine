//! Squirrel Engine — WASM core.
//!
//! This crate owns everything the briefing assigns to the WASM side: math,
//! the scene / entity data, per-frame transform computation, the camera
//! matrices, lighting parameters, and the packed **command stream** that JS
//! replays onto a `GPURenderPassEncoder`.
//!
//! It holds **no** WebGPU handles. It reasons purely in terms of integer
//! resource IDs (pipelines, bind groups, buffers) that the JS side resolves
//! to real GPU objects at replay time. The only thing that crosses the
//! boundary each frame is a single pointer to a buffer of `u32`s.
//!
//! Beyond the procedural cube field, the engine tracks a small table of
//! **imported models** (glTF primitives loaded and uploaded by JS). WASM does
//! not parse glTF — JS does that and hands back the buffer IDs — but WASM owns
//! each model's transform and material, appends them to the shared transform /
//! material storage buffers after the visible cubes, and emits the extra
//! `SET_MESH` / `DRAW` commands so the whole scene is still one command stream.

mod bvh;
mod math;
mod mesh;

use math::{vec3, Mat4, Vec3};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Command stream opcodes. JS mirrors these constants in its replay loop.
// ---------------------------------------------------------------------------

/// Terminates the command stream.
const CMD_END: u32 = 0;
/// Draw indexed from the currently-bound mesh:
/// [CMD_DRAW_INDEXED, pipeline_id, bindgroup_id, first_index, index_count,
///  instance_count, material_bindgroup_id, first_instance]
/// `material_bindgroup_id` = the per-draw material bind group (group 2), used by
/// textured glTF models; `NO_MATERIAL` means "no group 2" (the cube field). It
/// occupies the record's 7th slot — base_vertex is always 0 in this engine
/// (each mesh has its own 0-based index buffer), so the slot carries the
/// material id instead.
const CMD_DRAW_INDEXED: u32 = 1;
/// Bind a mesh's vertex + index buffers for the draws that follow:
/// [CMD_SET_MESH, vertex_buffer_id, index_buffer_id, index_format, 0, 0, 0, 0]
/// `index_format`: 0 = uint32, 1 = uint16.
const CMD_SET_MESH: u32 = 2;
/// Each command is a fixed 8 x u32 record.
const CMD_STRIDE: usize = 8;
/// Sentinel material id meaning "this draw binds no group-2 material".
const NO_MATERIAL: u32 = u32::MAX;

/// Floats per model matrix.
const MAT4_FLOATS: usize = 16;
/// Floats per per-instance material record (one `vec4` base color).
const MATERIAL_FLOATS: usize = 4;
/// The cube field's material: alpha < 0 means "tint by surface normal".
const CUBE_MATERIAL: [f32; MATERIAL_FLOATS] = [0.0, 0.0, 0.0, -1.0];
/// Ratio of a cube's bounding-sphere radius (as traced) to its scale.
const SPHERE_RADIUS_SCALE: f32 = 0.75;

/// Camera uniform, laid out to match the WGSL `Camera` struct exactly.
/// `view_proj` (16 f32) + `params` (4 f32) = 20 f32 = 80 bytes.
const CAMERA_FLOATS: usize = 20;

/// Lighting uniform, laid out to match the WGSL `Lighting` struct exactly:
///   sun_dir(4) + sun_color(4) + camera_pos(4) + env_params(4)
///   + 4 point lights * (position(4) + color(4)) = 16 + 32 = 48 f32 = 192 bytes.
const LIGHTING_FLOATS: usize = 48;
/// Point lights carried in the lighting uniform.
const MAX_POINT_LIGHTS: usize = 4;

/// Ray-tracing camera: eye + three basis vectors, one `vec4` each.
/// eye.w = sphere BVH node count, forward.w = time. 16 f32 = 64 bytes.
const RT_CAMERA_FLOATS: usize = 16;

/// Upper bound on imported glTF primitives that can be resident at once. The
/// shared transform / material buffers reserve this many slots past the cube
/// field so appending a model never reallocates in the frame loop.
const MAX_MODELS: usize = 64;

/// Vertical field of view used by both the raster projection and RT rays.
const FOV_Y_DEG: f32 = 60.0;

/// One rotating cube instance.
#[derive(Clone, Copy)]
struct Instance {
    position: Vec3,
    axis: Vec3,
    scale: f32,
    spin: f32,  // radians / second
    phase: f32, // starting angle
}

/// An imported mesh (one glTF primitive). JS uploads the geometry and passes
/// back the resource IDs; WASM owns its placement and material.
#[derive(Clone)]
struct Model {
    vertex_buffer_id: u32,
    index_buffer_id: u32,
    index_count: u32,
    index_format: u32, // 0 = uint32, 1 = uint16
    pipeline_id: u32,  // which render pipeline draws it (textured model pipeline)
    material_bg: u32,  // group-2 material bind group (albedo texture + sampler)
    transform: [f32; MAT4_FLOATS],
    base_color: [f32; 4],
}

#[wasm_bindgen]
pub struct Engine {
    max_instances: usize,
    instances: Vec<Instance>,

    // Imported glTF primitives, drawn after the cube field.
    models: Vec<Model>,

    // --- Buffers that live in WASM linear memory for the lifetime of the
    // engine. Allocated once here so their pointers stay stable and JS can
    // hold views over them (see briefing §3 memory-growth hazard). ---
    transforms: Vec<f32>, // (max_instances + MAX_MODELS) * 16 model matrices
    materials: Vec<f32>,  // (max_instances + MAX_MODELS) * 4, one base color/inst
    camera: Vec<f32>,     // CAMERA_FLOATS (rasterization camera)
    lighting: Vec<f32>,   // LIGHTING_FLOATS (sun + point lights + env)
    rt_camera: Vec<f32>,  // RT_CAMERA_FLOATS (ray-tracing camera: eye + basis)
    commands: Vec<u32>,   // packed command stream
    vertices: Vec<f32>,   // static cube geometry
    indices: Vec<u32>,

    // Camera state.
    eye: Vec3,
    aspect: f32,
    time: f32,

    // Culling: 6 normalized frustum planes (left, right, bottom, top, near,
    // far), the number of instances that survived culling this frame, and a
    // toggle so the effect can be turned off.
    frustum: [[f32; 4]; 6],
    visible: usize,
    culling_enabled: bool,

    // Static BVH over the cube field for the ray tracer, built once in
    // `init_scene` (instances never translate after that — only their
    // rotation animates — so this never needs rebuilding per frame). Node
    // layout matches WGSL's shared `BvhNode` struct, so the same stackless
    // traversal used for imported-model triangles walks this tree too.
    sphere_bvh_nodes: Vec<f32>,
    sphere_bvh_spheres: Vec<f32>, // reordered to match the BVH's leaf layout
    sphere_bvh_node_count: usize,

    // Material upload tracking. The cube field's material never changes, so the
    // buffer only needs uploading when models occupy (or vacate) slots:
    //   materials_refresh  — a full-buffer upload is pending (scene rebuilt or
    //                        models cleared, so previously-dirtied GPU slots
    //                        must be restored).
    //   materials_upload   — bytes JS should upload after this frame's update()
    //                        (0 = skip the writeBuffer entirely).
    //   last_model_base/count — the slots write_models dirtied last frame, so
    //                        they can be restored to the cube palette.
    materials_refresh: bool,
    materials_upload: usize,
    last_model_base: usize,
    last_model_count: usize,

    // --- Driven mode (additive; the cube demo leaves all of this untouched) ---
    // When `driven` is on, instance transforms + colors come from an external
    // source (e.g. the physics ballpit) via `set_driven_data` each frame instead
    // of the built-in cube animation, and the camera follows `set_camera`.
    driven: bool,
    driven_data: Vec<f32>, // [x, y, z, radius, r, g, b] per instance
    manual_cam: bool,
    manual_eye: Vec3,
    manual_target: Vec3,
}

/// Sphere/frustum test. `planes` are normalized (xyz unit length) so `d` and
/// `radius` are both in world units. Inside iff on the inner side of all six.
#[inline]
fn sphere_in_frustum(planes: &[[f32; 4]; 6], c: Vec3, radius: f32) -> bool {
    for p in planes.iter() {
        if p[0] * c.x + p[1] * c.y + p[2] * c.z + p[3] < -radius {
            return false;
        }
    }
    true
}

/// Default lighting: a warm key "sun" from the upper-front, gentle sky ambient,
/// environment ambient + reflection pulled from the skybox cubemap, no point
/// lights. Matches the raytracer's default light direction so both modes agree.
fn default_lighting() -> Vec<f32> {
    let mut l = vec![0.0f32; LIGHTING_FLOATS];
    let dir = vec3(0.4, 0.9, 0.35).normalize(); // direction *toward* the light
    l[0] = dir.x;
    l[1] = dir.y;
    l[2] = dir.z;
    l[3] = 1.15; // sun intensity
    l[4] = 1.0;
    l[5] = 0.96;
    l[6] = 0.88; // sun color (slightly warm)
    l[7] = 0.12; // flat ambient floor
    // camera_pos (8..11) filled each frame; [11] = point light count.
    l[12] = 0.65; // env ambient amount (skybox irradiance)
    l[13] = 0.30; // env reflection amount
    l[14] = 1.0; // exposure
    l[15] = 0.0;
    l
}

#[wasm_bindgen]
impl Engine {
    /// Create an engine sized for at most `max_instances` objects.
    #[wasm_bindgen(constructor)]
    pub fn new(max_instances: u32) -> Engine {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let max = max_instances.max(1) as usize;
        let slots = max + MAX_MODELS;
        Engine {
            max_instances: max,
            instances: Vec::with_capacity(max),
            models: Vec::new(),
            // Pre-reserve so we never reallocate (and never grow memory) in
            // the frame loop. JS builds its views after construction. Room for
            // every cube plus MAX_MODELS imported primitives.
            transforms: vec![0.0; slots * MAT4_FLOATS],
            materials: vec![0.0; slots * MATERIAL_FLOATS],
            camera: vec![0.0; CAMERA_FLOATS],
            lighting: default_lighting(),
            rt_camera: vec![0.0; RT_CAMERA_FLOATS],
            // One instanced cube draw, plus a SET_MESH + DRAW pair per model,
            // plus the terminator.
            commands: vec![0u32; (2 + MAX_MODELS * 2) * CMD_STRIDE],
            vertices: mesh::cube_vertices(),
            indices: mesh::cube_indices(),
            eye: vec3(0.0, 0.0, 0.0),
            aspect: 1.0,
            time: 0.0,
            frustum: [[0.0; 4]; 6],
            visible: 0,
            culling_enabled: true,
            sphere_bvh_nodes: Vec::new(),
            sphere_bvh_spheres: Vec::new(),
            sphere_bvh_node_count: 0,
            materials_refresh: true,
            materials_upload: 0,
            last_model_base: 0,
            last_model_count: 0,
            driven: false,
            driven_data: Vec::new(),
            manual_cam: false,
            manual_eye: vec3(0.0, 0.0, 0.0),
            manual_target: vec3(0.0, 0.0, 0.0),
        }
    }

    // --- Driven mode: external per-instance transforms + a manual camera. All
    // additive and opt-in — used by the physics ballpit. ---------------------

    /// Swap the instanced mesh from the default cube to a unit UV sphere. Call
    /// this *before* `Renderer.init` so JS uploads the sphere geometry.
    pub fn use_sphere_mesh(&mut self, rings: u32, sectors: u32) {
        let (v, i) = mesh::uv_sphere(rings, sectors);
        self.vertices = v;
        self.indices = i;
    }

    /// Turn on driven mode: instance transforms + colors come from
    /// `set_driven_data` each frame rather than the built-in animation. Culling
    /// is disabled so instance order matches the supplied data 1:1.
    pub fn enable_driven(&mut self, count: u32) {
        self.driven = true;
        self.culling_enabled = false;
        let n = (count as usize).min(self.max_instances);
        self.driven_data.resize(n * 7, 0.0);
        self.materials_refresh = true;
    }

    /// Supply this frame's instance data: 7 floats each — position `(x,y,z)`,
    /// `radius`, color `(r,g,b)`. `data.len() / 7` sets the instance count.
    pub fn set_driven_data(&mut self, data: &[f32]) {
        let n = (data.len() / 7).min(self.max_instances);
        self.driven_data.clear();
        self.driven_data.extend_from_slice(&data[..n * 7]);
    }

    /// Set an explicit camera (eye + look-at target), overriding the orbit.
    pub fn set_camera(&mut self, ex: f32, ey: f32, ez: f32, tx: f32, ty: f32, tz: f32) {
        self.manual_cam = true;
        self.manual_eye = vec3(ex, ey, ez);
        self.manual_target = vec3(tx, ty, tz);
    }

    /// Compose transforms + colors for driven mode from `driven_data`.
    fn build_driven(&mut self) {
        let n = {
            let Self { driven_data, transforms, materials, .. } = self;
            let n = driven_data.len() / 7;
            for i in 0..n {
                let o = i * 7;
                let pos = vec3(driven_data[o], driven_data[o + 1], driven_data[o + 2]);
                let scale = driven_data[o + 3];
                // Identity rotation (axis is arbitrary but must be non-zero so
                // the internal normalize is well-defined); uniform scale = radius.
                let model = Mat4::trs(pos, vec3(0.0, 1.0, 0.0), 0.0, scale);
                let tb = i * MAT4_FLOATS;
                transforms[tb..tb + MAT4_FLOATS].copy_from_slice(&model.0);
                let mb = i * MATERIAL_FLOATS;
                materials[mb] = driven_data[o + 4];
                materials[mb + 1] = driven_data[o + 5];
                materials[mb + 2] = driven_data[o + 6];
                materials[mb + 3] = 1.0; // alpha >= 0 => real color, not normal-tint
            }
            n
        };
        self.visible = n;
    }

    /// Populate the scene with `count` procedurally-placed rotating cubes.
    pub fn init_scene(&mut self, count: u32) {
        let count = (count as usize).min(self.max_instances);
        self.instances.clear();

        // Deterministic pseudo-random layout — no host RNG needed.
        let mut seed: u32 = 0x9E37_79B9;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32) / (u32::MAX as f32)
        };

        for _ in 0..count {
            // A wide field so the orbiting camera sits within it and a real
            // fraction falls outside the frustum each frame (culling matters).
            let position = vec3(
                (rng() - 0.5) * 180.0,
                (rng() - 0.5) * 32.0,
                (rng() - 0.5) * 180.0,
            );
            let axis = vec3(rng() - 0.5, rng() - 0.5, rng() - 0.5).normalize();
            self.instances.push(Instance {
                position,
                axis,
                scale: 0.5 + rng() * 1.2,
                spin: (rng() - 0.5) * 3.0,
                phase: rng() * std::f32::consts::TAU,
            });
        }

        // Every slot starts as a cube material; that never changes at runtime,
        // so the per-frame loop no longer touches materials at all. Models
        // overwrite their slots in write_models and restore them on eviction.
        for slot in self.materials.chunks_exact_mut(MATERIAL_FLOATS) {
            slot.copy_from_slice(&CUBE_MATERIAL);
        }
        self.materials_refresh = true;

        // Build the ray tracer's spatial structure once here — instance
        // positions never change after this point, so the tree is valid for
        // the scene's whole lifetime and JS uploads it exactly once.
        let centers: Vec<Vec3> = self.instances.iter().map(|i| i.position).collect();
        let radii: Vec<f32> = self
            .instances
            .iter()
            .map(|i| i.scale * SPHERE_RADIUS_SCALE)
            .collect();
        let built = bvh::build(&centers, &radii);
        self.sphere_bvh_node_count = built.node_count;
        self.sphere_bvh_nodes = built.nodes;
        self.sphere_bvh_spheres = built.spheres;
    }

    /// JS passes the canvas aspect ratio (width / height) so WASM can build
    /// the projection matrix. Called at init and on every resize.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = if aspect > 0.0 { aspect } else { 1.0 };
    }

    /// Advance the simulation by `dt` seconds and rebuild all per-frame data:
    /// transforms, materials, the camera + lighting uniforms, and the command
    /// stream. After this returns, JS uploads the buffers and replays.
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        self.update_camera(); // also extracts the frustum planes, stores eye
        if self.driven {
            self.build_driven(); // external per-instance transforms + colors
        } else {
            self.cull_and_build(); // frustum-culls, compacts visible transforms
            self.write_models(); // append imported primitives after the cubes
        }
        self.update_lighting(); // camera position + live point-light count

        // Decide how many material bytes JS must upload this frame. While
        // models are resident their slots shift with the visible count, so the
        // active region re-uploads; otherwise the buffer is static on the GPU.
        // Driven mode re-uploads its live colors each frame.
        self.materials_upload = if self.materials_refresh {
            self.materials_refresh = false;
            self.materials.len() * 4
        } else if self.driven {
            self.visible * MATERIAL_FLOATS * 4
        } else if !self.models.is_empty() {
            (self.visible + self.models.len()) * MATERIAL_FLOATS * 4
        } else {
            0
        };

        self.build_commands();
    }

    /// Enable or disable frustum culling (on by default).
    pub fn set_culling(&mut self, enabled: bool) {
        self.culling_enabled = enabled;
    }

    // --- Lighting controls (all callable from JS) ---------------------------

    /// Set the directional "sun": `(dx, dy, dz)` is the direction *toward* the
    /// light (normalized internally); `intensity` scales its contribution.
    pub fn set_sun(&mut self, dx: f32, dy: f32, dz: f32, intensity: f32) {
        let d = vec3(dx, dy, dz).normalize();
        self.lighting[0] = d.x;
        self.lighting[1] = d.y;
        self.lighting[2] = d.z;
        self.lighting[3] = intensity;
    }

    /// Set the sun color (`r,g,b`) and the flat ambient floor.
    pub fn set_sun_color(&mut self, r: f32, g: f32, b: f32, ambient: f32) {
        self.lighting[4] = r;
        self.lighting[5] = g;
        self.lighting[6] = b;
        self.lighting[7] = ambient;
    }

    /// Environment (skybox) contribution: `ambient` = how much the cubemap
    /// lights shadowed areas, `reflect` = how mirror-like surfaces are,
    /// `exposure` = overall tonemap exposure.
    pub fn set_env_params(&mut self, ambient: f32, reflect: f32, exposure: f32) {
        self.lighting[12] = ambient;
        self.lighting[13] = reflect;
        self.lighting[14] = exposure;
    }

    /// Configure point light `idx` (`0..4`). `range` is the falloff distance;
    /// `intensity` scales the color. Also raises the active light count so the
    /// shader loops over it.
    pub fn set_point_light(
        &mut self,
        idx: u32,
        x: f32,
        y: f32,
        z: f32,
        range: f32,
        r: f32,
        g: f32,
        b: f32,
        intensity: f32,
    ) {
        let i = idx as usize;
        if i >= MAX_POINT_LIGHTS {
            return;
        }
        let base = 16 + i * 8;
        self.lighting[base] = x;
        self.lighting[base + 1] = y;
        self.lighting[base + 2] = z;
        self.lighting[base + 3] = range;
        self.lighting[base + 4] = r;
        self.lighting[base + 5] = g;
        self.lighting[base + 6] = b;
        self.lighting[base + 7] = intensity;
        // Keep the count at least large enough to include this light.
        let count = self.lighting[11] as usize;
        if i + 1 > count {
            self.lighting[11] = (i + 1) as f32;
        }
    }

    /// Set how many point lights (`0..4`) the shader should evaluate.
    pub fn set_point_light_count(&mut self, count: u32) {
        self.lighting[11] = (count as usize).min(MAX_POINT_LIGHTS) as f32;
    }

    // --- Imported model table (glTF primitives, uploaded by JS) -------------

    /// Register a glTF primitive whose vertex + index buffers (and per-draw
    /// material bind group) JS has already created. `pipeline_id` selects the
    /// render pipeline, `material_bg` the group-2 material (albedo texture).
    /// Returns the model index used by the transform / material setters, or
    /// `u32::MAX` if the table is full.
    pub fn register_model(
        &mut self,
        vertex_buffer_id: u32,
        index_buffer_id: u32,
        index_count: u32,
        index_format: u32,
        pipeline_id: u32,
        material_bg: u32,
    ) -> u32 {
        if self.models.len() >= MAX_MODELS {
            return u32::MAX;
        }
        let idx = self.models.len() as u32;
        self.models.push(Model {
            vertex_buffer_id,
            index_buffer_id,
            index_count,
            index_format,
            pipeline_id,
            material_bg,
            transform: Mat4::identity().0,
            base_color: [0.82, 0.82, 0.85, 1.0],
        });
        idx
    }

    /// Set model `idx`'s full column-major 4x4 model matrix (16 floats).
    pub fn set_model_transform(&mut self, idx: u32, m: &[f32]) {
        let i = idx as usize;
        if i < self.models.len() && m.len() >= MAT4_FLOATS {
            self.models[i].transform.copy_from_slice(&m[..MAT4_FLOATS]);
        }
    }

    /// Set model `idx`'s base color (glTF `baseColorFactor`). Alpha ≥ 0 marks a
    /// real material; the cube field uses alpha < 0 to mean "tint by normal".
    pub fn set_model_base_color(&mut self, idx: u32, r: f32, g: f32, b: f32, a: f32) {
        let i = idx as usize;
        if i < self.models.len() {
            self.models[i].base_color = [r, g, b, a.max(0.0)];
        }
    }

    /// Remove all imported models (e.g. before loading a new file). Forces one
    /// full material upload so any GPU slots the models dirtied are restored.
    pub fn clear_models(&mut self) {
        self.models.clear();
        self.materials_refresh = true;
    }

    /// Number of imported models currently resident.
    pub fn model_count(&self) -> u32 {
        self.models.len() as u32
    }

    /// Maximum imported models (so JS can size its GPU buffers to match).
    pub fn models_capacity(&self) -> u32 {
        MAX_MODELS as u32
    }

    fn update_camera(&mut self) {
        // Manual camera (driven mode) follows an external target; otherwise a
        // slow orbit around the scene. Eye + target + up drive both the
        // rasterization view matrix and the ray-tracing basis.
        let (eye, target) = if self.manual_cam {
            (self.manual_eye, self.manual_target)
        } else {
            let r = 55.0;
            let a = self.time * 0.15;
            (vec3(a.cos() * r, 20.0, a.sin() * r), vec3(0.0, 0.0, 0.0))
        };
        let world_up = vec3(0.0, 1.0, 0.0);
        self.eye = eye;

        // --- Rasterization camera (view_proj matrix) ---
        let view = Mat4::look_at(eye, target, world_up);
        let proj = Mat4::perspective(FOV_Y_DEG.to_radians(), self.aspect, 0.1, 500.0);
        let view_proj = proj.mul(&view);

        self.camera[..MAT4_FLOATS].copy_from_slice(&view_proj.0);
        self.camera[16] = self.time; // params.x
        self.camera[17] = 0.0;
        self.camera[18] = 0.0;
        self.camera[19] = 0.0;

        // Frustum planes for culling, derived from view_proj (Gribb-Hartmann).
        self.extract_frustum(&view_proj.0);

        // --- Ray-tracing camera (eye + basis) ---
        // Same right-handed basis look_at builds, so RT and raster agree.
        let forward = target.sub(eye).normalize();
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward);

        // Pre-scale the basis so a ray is just:
        //   dir = forward + ndc.x * right + ndc.y * up   (ndc in [-1, 1])
        let tan_half = (FOV_Y_DEG.to_radians() * 0.5).tan();
        let rx = tan_half * self.aspect;
        let ry = tan_half;

        // The BVH is static (built once in init_scene), so this is just a
        // constant re-assigned each frame along with the rest of the uniform.
        let sphere_bvh_node_count = self.sphere_bvh_node_count as f32;
        let rt = &mut self.rt_camera;
        rt[0] = eye.x;
        rt[1] = eye.y;
        rt[2] = eye.z;
        rt[3] = sphere_bvh_node_count;
        rt[4] = right.x * rx;
        rt[5] = right.y * rx;
        rt[6] = right.z * rx;
        rt[7] = 0.0;
        rt[8] = up.x * ry;
        rt[9] = up.y * ry;
        rt[10] = up.z * ry;
        rt[11] = 0.0;
        rt[12] = forward.x;
        rt[13] = forward.y;
        rt[14] = forward.z;
        rt[15] = self.time;
    }

    /// Push the live camera position into the lighting uniform so the fragment
    /// shader can compute view-dependent specular / reflections.
    fn update_lighting(&mut self) {
        self.lighting[8] = self.eye.x;
        self.lighting[9] = self.eye.y;
        self.lighting[10] = self.eye.z;
        // [11] (point light count) is owned by the setters.
    }

    /// Extract the 6 normalized frustum planes from a column-major view_proj.
    /// WebGPU clip space uses z in [0, 1], so the near plane is +z (row2).
    fn extract_frustum(&mut self, m: &[f32; 16]) {
        // Rows of the column-major matrix: rowK = (m[0*4+K], m[1*4+K], ...).
        let row = |k: usize| [m[k], m[4 + k], m[8 + k], m[12 + k]];
        let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
        let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
        let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];

        let raw = [
            add(r3, r0), // left
            sub(r3, r0), // right
            add(r3, r1), // bottom
            sub(r3, r1), // top
            r2,          // near (z in [0, 1])
            sub(r3, r2), // far
        ];
        for (i, p) in raw.iter().enumerate() {
            let inv = 1.0 / (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-8);
            self.frustum[i] = [p[0] * inv, p[1] * inv, p[2] * inv, p[3] * inv];
        }
    }

    /// Frustum-cull the scene and compose the surviving instances' model
    /// matrices, compacted into the front of the transform buffer. Materials
    /// are pre-filled at scene build and untouched here. The ray tracer does
    /// not consume this output — it walks the static sphere BVH instead — so
    /// this only serves the raster draw.
    fn cull_and_build(&mut self) {
        // Disjoint field borrows so we can read instances/frustum while
        // writing transforms.
        let Self {
            instances,
            transforms,
            frustum,
            culling_enabled,
            time,
            ..
        } = self;
        let time = *time;
        let cull = *culling_enabled;

        let mut visible = 0usize;
        for inst in instances.iter() {
            // Bounding sphere of the cube (half-diagonal ~0.87 * scale, padded).
            if cull && !sphere_in_frustum(frustum, inst.position, inst.scale * 0.9) {
                continue;
            }
            let angle = inst.phase + inst.spin * time;
            let model = Mat4::trs(inst.position, inst.axis, angle, inst.scale);
            let base = visible * MAT4_FLOATS;
            transforms[base..base + MAT4_FLOATS].copy_from_slice(&model.0);
            visible += 1;
        }
        self.visible = visible;
    }

    /// Append each imported model's transform + material immediately after the
    /// visible cubes, so a single shared storage buffer describes the whole
    /// scene and `@builtin(instance_index)` selects the right slot.
    fn write_models(&mut self) {
        let Self {
            visible,
            models,
            transforms,
            materials,
            last_model_base,
            last_model_count,
            ..
        } = self;
        let base = *visible;

        // Restore the cube palette over the slots last frame's models occupied
        // (the visible count shifts them every frame). At most MAX_MODELS slots.
        for k in 0..*last_model_count {
            let mi = (*last_model_base + k) * MATERIAL_FLOATS;
            materials[mi..mi + MATERIAL_FLOATS].copy_from_slice(&CUBE_MATERIAL);
        }

        for (k, m) in models.iter().enumerate() {
            let ti = (base + k) * MAT4_FLOATS;
            transforms[ti..ti + MAT4_FLOATS].copy_from_slice(&m.transform);
            let mi = (base + k) * MATERIAL_FLOATS;
            materials[mi..mi + MATERIAL_FLOATS].copy_from_slice(&m.base_color);
        }
        *last_model_base = base;
        *last_model_count = models.len();
    }

    /// Build the command stream: one instanced indexed draw for the cube field,
    /// then a `SET_MESH` + single-instance `DRAW` pair per imported model. Each
    /// model draw uses `first_instance` so its vertex shader reads the matrix
    /// slot `write_models` placed after the visible cubes.
    fn build_commands(&mut self) {
        let index_count = self.indices.len() as u32;
        let visible = self.visible as u32;

        let c = &mut self.commands;
        let mut w = 0usize;

        // Cube field: instanced indexed draw (mesh 0, bound by JS before replay).
        c[w] = CMD_DRAW_INDEXED;
        c[w + 1] = 0; // pipeline_id (lit cube pipeline)
        c[w + 2] = 0; // bindgroup_id (per-frame group 0)
        c[w + 3] = 0; // first_index
        c[w + 4] = index_count;
        c[w + 5] = visible; // instance_count
        c[w + 6] = NO_MATERIAL; // no group-2 material
        c[w + 7] = 0; // first_instance
        w += CMD_STRIDE;

        // Imported models: bind each mesh, draw one instance reading slot
        // (visible + k) from the shared transform / material buffers, on its own
        // textured pipeline with its own group-2 material (albedo texture).
        for (k, m) in self.models.iter().enumerate() {
            c[w] = CMD_SET_MESH;
            c[w + 1] = m.vertex_buffer_id;
            c[w + 2] = m.index_buffer_id;
            c[w + 3] = m.index_format;
            c[w + 4] = 0;
            c[w + 5] = 0;
            c[w + 6] = 0;
            c[w + 7] = 0;
            w += CMD_STRIDE;

            c[w] = CMD_DRAW_INDEXED;
            c[w + 1] = m.pipeline_id; // textured model pipeline
            c[w + 2] = 0; // bindgroup_id (shared per-frame group 0)
            c[w + 3] = 0; // first_index
            c[w + 4] = m.index_count;
            c[w + 5] = 1; // instance_count
            c[w + 6] = m.material_bg; // group-2 material (albedo texture)
            c[w + 7] = self.visible as u32 + k as u32; // first_instance -> slot
            w += CMD_STRIDE;
        }

        c[w] = CMD_END;
    }

    // --- Pointer / length accessors. All offsets are byte addresses into
    // linear memory; JS reads them through views over `wasm.memory`. ---

    pub fn commands_ptr(&self) -> *const u32 {
        self.commands.as_ptr()
    }

    pub fn transforms_ptr(&self) -> *const f32 {
        self.transforms.as_ptr()
    }

    pub fn transforms_bytes(&self) -> usize {
        // Visible cubes plus the appended imported models.
        (self.visible + self.models.len()) * MAT4_FLOATS * 4
    }

    pub fn materials_ptr(&self) -> *const f32 {
        self.materials.as_ptr()
    }

    pub fn materials_bytes(&self) -> usize {
        (self.visible + self.models.len()) * MATERIAL_FLOATS * 4
    }

    /// Material bytes JS must upload after this frame's `update()` — 0 when the
    /// GPU copy is already correct (the common case: no models resident).
    pub fn materials_upload_bytes(&self) -> usize {
        self.materials_upload
    }

    /// Static BVH over the cube field (see `bvh::build`), built once in
    /// `init_scene`. Node layout matches WGSL's shared `BvhNode` struct — the
    /// same one used for imported-model triangles — so both are walked by the
    /// same kind of stackless traversal in the ray tracer.
    pub fn sphere_bvh_nodes_ptr(&self) -> *const f32 {
        self.sphere_bvh_nodes.as_ptr()
    }

    pub fn sphere_bvh_nodes_bytes(&self) -> usize {
        self.sphere_bvh_nodes.len() * 4
    }

    /// Sphere records (center.xyz + radius), reordered to match the BVH's
    /// leaf layout. Static — uploaded once, never touched per frame.
    pub fn sphere_bvh_spheres_ptr(&self) -> *const f32 {
        self.sphere_bvh_spheres.as_ptr()
    }

    pub fn sphere_bvh_spheres_bytes(&self) -> usize {
        self.sphere_bvh_spheres.len() * 4
    }

    pub fn sphere_bvh_node_count(&self) -> u32 {
        self.sphere_bvh_node_count as u32
    }

    pub fn camera_ptr(&self) -> *const f32 {
        self.camera.as_ptr()
    }

    pub fn camera_bytes(&self) -> usize {
        CAMERA_FLOATS * 4
    }

    pub fn lighting_ptr(&self) -> *const f32 {
        self.lighting.as_ptr()
    }

    pub fn lighting_bytes(&self) -> usize {
        LIGHTING_FLOATS * 4
    }

    pub fn rt_camera_ptr(&self) -> *const f32 {
        self.rt_camera.as_ptr()
    }

    pub fn rt_camera_bytes(&self) -> usize {
        RT_CAMERA_FLOATS * 4
    }

    pub fn vertices_ptr(&self) -> *const f32 {
        self.vertices.as_ptr()
    }

    pub fn vertices_bytes(&self) -> usize {
        self.vertices.len() * 4
    }

    pub fn indices_ptr(&self) -> *const u32 {
        self.indices.as_ptr()
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    /// Total instances in the scene (before culling).
    pub fn instance_count(&self) -> u32 {
        self.instances.len() as u32
    }

    /// Instances that survived frustum culling this frame (drawn / traced).
    pub fn visible_count(&self) -> u32 {
        self.visible as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Silent-failure guard (briefing §7): the WGSL Camera struct is 80 bytes.
    #[test]
    fn camera_uniform_size() {
        assert_eq!(CAMERA_FLOATS * 4, 80);
    }

    // The WGSL Lighting struct must be 192 bytes (sun + camera + env + 4 points).
    #[test]
    fn lighting_uniform_size() {
        assert_eq!(LIGHTING_FLOATS * 4, 192);
        assert_eq!(default_lighting().len(), LIGHTING_FLOATS);
    }

    // Command records are a fixed 8 u32s on both sides of the boundary.
    #[test]
    fn command_stride() {
        assert_eq!(CMD_STRIDE, 8);
    }

    #[test]
    fn cube_geometry_shape() {
        let e = Engine::new(4);
        assert_eq!(e.vertices.len(), 24 * 6); // 24 verts * (pos+normal)
        assert_eq!(e.indices.len(), 36);
    }

    #[test]
    fn plane_grid_shape() {
        // 3x5 vertex grid, scaled 2 wide x 4 tall.
        let (v, i) = mesh::plane(2.0, 4.0, 3, 5);
        assert_eq!(v.len(), 3 * 5 * 6); // nx*ny verts, 6 floats (pos+normal) each
        assert_eq!(i.len(), (3 - 1) * (5 - 1) * 6); // (nx-1)*(ny-1) quads * 6
        // First vertex sits at the (-w/2, -h/2) corner, facing +Z.
        assert_eq!(&v[0..3], &[-1.0, -2.0, 0.0]);
        assert_eq!(&v[3..6], &[0.0, 0.0, 1.0]);
        // Degenerate counts clamp to a single quad, never panic.
        let (v2, i2) = mesh::plane(1.0, 1.0, 0, 1);
        assert_eq!(v2.len(), 2 * 2 * 6);
        assert_eq!(i2.len(), 6);
    }

    #[test]
    fn models_append_after_visible_cubes() {
        let mut e = Engine::new(8);
        e.init_scene(4);
        e.set_culling(false); // keep all 4 cubes visible
        let id = e.register_model(7, 8, 99, 0, 1, 3);
        assert_eq!(id, 0);
        e.set_model_base_color(0, 0.2, 0.4, 0.9, 1.0);
        e.update(0.016);
        assert_eq!(e.visible_count(), 4);
        assert_eq!(e.model_count(), 1);
        // Transforms/materials cover the 4 cubes + 1 model.
        assert_eq!(e.transforms_bytes(), 5 * MAT4_FLOATS * 4);
        assert_eq!(e.materials_bytes(), 5 * MATERIAL_FLOATS * 4);
        // The model's material sits in slot 4 (after the 4 cubes).
        let mslot = 4 * MATERIAL_FLOATS;
        assert_eq!(e.materials[mslot], 0.2);
        assert_eq!(e.materials[mslot + 3], 1.0);

        // Command stream: cube draw (no material), then SET_MESH + model draw
        // carrying the model's pipeline id + material bind group in the record.
        let c = &e.commands;
        assert_eq!(c[0], CMD_DRAW_INDEXED);
        assert_eq!(c[6], NO_MATERIAL); // cube draw: no group-2 material
        assert_eq!(c[CMD_STRIDE], CMD_SET_MESH);
        let d = 2 * CMD_STRIDE; // model draw record
        assert_eq!(c[d], CMD_DRAW_INDEXED);
        assert_eq!(c[d + 1], 1); // pipeline_id passed to register_model
        assert_eq!(c[d + 6], 3); // material bind group passed to register_model
        assert_eq!(c[d + 7], 4); // first_instance = slot after the 4 cubes
    }

    // The sphere BVH is built once in init_scene and covers every instance,
    // independent of frustum culling or update() ever being called.
    #[test]
    fn sphere_bvh_built_at_scene_init() {
        let mut e = Engine::new(8);
        e.init_scene(4);
        assert!(e.sphere_bvh_node_count() > 0);
        assert_eq!(e.sphere_bvh_spheres_bytes(), 4 * bvh::SPHERE_FLOATS * 4);
        assert_eq!(
            e.sphere_bvh_nodes_bytes(),
            e.sphere_bvh_node_count() as usize * bvh::NODE_FLOATS * 4
        );

        // The RT camera's sphere-BVH node count field is static across frames
        // (culling must not shrink the tree the ray tracer walks).
        e.set_culling(true);
        e.update(0.016);
        let count_a = e.rt_camera[3];
        e.update(1.0);
        assert_eq!(e.rt_camera[3], count_a);
        assert_eq!(count_a, e.sphere_bvh_node_count() as f32);
    }

    // Materials upload only when they can actually differ from the GPU copy:
    // once after scene build, every frame while models are resident, and once
    // more after models are cleared (to restore the dirtied slots).
    #[test]
    fn materials_upload_only_when_dirty() {
        let mut e = Engine::new(8);
        e.init_scene(4);
        e.set_culling(false);

        e.update(0.016);
        assert_eq!(e.materials_upload_bytes(), e.materials.len() * 4); // full refresh
        e.update(0.016);
        assert_eq!(e.materials_upload_bytes(), 0); // static from here on

        e.register_model(7, 8, 99, 0, 1, 3);
        e.update(0.016);
        assert_eq!(e.materials_upload_bytes(), 5 * MATERIAL_FLOATS * 4); // active region

        e.clear_models();
        e.update(0.016);
        assert_eq!(e.materials_upload_bytes(), e.materials.len() * 4); // restore slots
        // The vacated slot is a cube material again.
        assert_eq!(e.materials[4 * MATERIAL_FLOATS + 3], -1.0);
        e.update(0.016);
        assert_eq!(e.materials_upload_bytes(), 0);
    }
}
