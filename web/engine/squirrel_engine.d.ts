/* tslint:disable */
/* eslint-disable */

export class Engine {
    free(): void;
    [Symbol.dispose](): void;
    camera_bytes(): number;
    camera_ptr(): number;
    /**
     * Remove all imported models (e.g. before loading a new file). Forces one
     * full material upload so any GPU slots the models dirtied are restored.
     */
    clear_models(): void;
    commands_ptr(): number;
    /**
     * Turn on driven mode: instance transforms + colors come from
     * `set_driven_data` each frame rather than the built-in animation. Culling
     * is disabled so instance order matches the supplied data 1:1.
     */
    enable_driven(count: number): void;
    index_count(): number;
    indices_ptr(): number;
    /**
     * Populate the scene with `count` procedurally-placed rotating cubes.
     */
    init_scene(count: number): void;
    /**
     * Total instances in the scene (before culling).
     */
    instance_count(): number;
    lighting_bytes(): number;
    lighting_ptr(): number;
    materials_bytes(): number;
    materials_ptr(): number;
    /**
     * Material bytes JS must upload after this frame's `update()` — 0 when the
     * GPU copy is already correct (the common case: no models resident).
     */
    materials_upload_bytes(): number;
    /**
     * Number of imported models currently resident.
     */
    model_count(): number;
    /**
     * Maximum imported models (so JS can size its GPU buffers to match).
     */
    models_capacity(): number;
    /**
     * Create an engine sized for at most `max_instances` objects.
     */
    constructor(max_instances: number);
    /**
     * Register a glTF primitive whose vertex + index buffers (and per-draw
     * material bind group) JS has already created. `pipeline_id` selects the
     * render pipeline, `material_bg` the group-2 material (albedo texture).
     * Returns the model index used by the transform / material setters, or
     * `u32::MAX` if the table is full.
     */
    register_model(vertex_buffer_id: number, index_buffer_id: number, index_count: number, index_format: number, pipeline_id: number, material_bg: number): number;
    rt_camera_bytes(): number;
    rt_camera_ptr(): number;
    /**
     * JS passes the canvas aspect ratio (width / height) so WASM can build
     * the projection matrix. Called at init and on every resize.
     */
    set_aspect(aspect: number): void;
    /**
     * Set an explicit camera (eye + look-at target), overriding the orbit.
     */
    set_camera(ex: number, ey: number, ez: number, tx: number, ty: number, tz: number): void;
    /**
     * Enable or disable frustum culling (on by default).
     */
    set_culling(enabled: boolean): void;
    /**
     * Supply this frame's instance data: 7 floats each — position `(x,y,z)`,
     * `radius`, color `(r,g,b)`. `data.len() / 7` sets the instance count.
     */
    set_driven_data(data: Float32Array): void;
    /**
     * Environment (skybox) contribution: `ambient` = how much the cubemap
     * lights shadowed areas, `reflect` = how mirror-like surfaces are,
     * `exposure` = overall tonemap exposure.
     */
    set_env_params(ambient: number, reflect: number, exposure: number): void;
    /**
     * Set model `idx`'s base color (glTF `baseColorFactor`). Alpha ≥ 0 marks a
     * real material; the cube field uses alpha < 0 to mean "tint by normal".
     */
    set_model_base_color(idx: number, r: number, g: number, b: number, a: number): void;
    /**
     * Set model `idx`'s full column-major 4x4 model matrix (16 floats).
     */
    set_model_transform(idx: number, m: Float32Array): void;
    /**
     * Configure point light `idx` (`0..4`). `range` is the falloff distance;
     * `intensity` scales the color. Also raises the active light count so the
     * shader loops over it.
     */
    set_point_light(idx: number, x: number, y: number, z: number, range: number, r: number, g: number, b: number, intensity: number): void;
    /**
     * Set how many point lights (`0..4`) the shader should evaluate.
     */
    set_point_light_count(count: number): void;
    /**
     * Set the directional "sun": `(dx, dy, dz)` is the direction *toward* the
     * light (normalized internally); `intensity` scales its contribution.
     */
    set_sun(dx: number, dy: number, dz: number, intensity: number): void;
    /**
     * Set the sun color (`r,g,b`) and the flat ambient floor.
     */
    set_sun_color(r: number, g: number, b: number, ambient: number): void;
    sphere_bvh_node_count(): number;
    sphere_bvh_nodes_bytes(): number;
    /**
     * Static BVH over the cube field (see `bvh::build`), built once in
     * `init_scene`. Node layout matches WGSL's shared `BvhNode` struct — the
     * same one used for imported-model triangles — so both are walked by the
     * same kind of stackless traversal in the ray tracer.
     */
    sphere_bvh_nodes_ptr(): number;
    sphere_bvh_spheres_bytes(): number;
    /**
     * Sphere records (center.xyz + radius), reordered to match the BVH's
     * leaf layout. Static — uploaded once, never touched per frame.
     */
    sphere_bvh_spheres_ptr(): number;
    transforms_bytes(): number;
    transforms_ptr(): number;
    /**
     * Advance the simulation by `dt` seconds and rebuild all per-frame data:
     * transforms, materials, the camera + lighting uniforms, and the command
     * stream. After this returns, JS uploads the buffers and replays.
     */
    update(dt: number): void;
    /**
     * Swap the instanced mesh from the default cube to a unit UV sphere. Call
     * this *before* `Renderer.init` so JS uploads the sphere geometry.
     */
    use_sphere_mesh(rings: number, sectors: number): void;
    vertices_bytes(): number;
    vertices_ptr(): number;
    /**
     * Instances that survived frustum culling this frame (drawn / traced).
     */
    visible_count(): number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly engine_camera_bytes: (a: number) => number;
    readonly engine_camera_ptr: (a: number) => number;
    readonly engine_clear_models: (a: number) => void;
    readonly engine_commands_ptr: (a: number) => number;
    readonly engine_enable_driven: (a: number, b: number) => void;
    readonly engine_index_count: (a: number) => number;
    readonly engine_indices_ptr: (a: number) => number;
    readonly engine_init_scene: (a: number, b: number) => void;
    readonly engine_instance_count: (a: number) => number;
    readonly engine_lighting_bytes: (a: number) => number;
    readonly engine_lighting_ptr: (a: number) => number;
    readonly engine_materials_bytes: (a: number) => number;
    readonly engine_materials_ptr: (a: number) => number;
    readonly engine_materials_upload_bytes: (a: number) => number;
    readonly engine_model_count: (a: number) => number;
    readonly engine_models_capacity: (a: number) => number;
    readonly engine_new: (a: number) => number;
    readonly engine_register_model: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
    readonly engine_rt_camera_ptr: (a: number) => number;
    readonly engine_set_aspect: (a: number, b: number) => void;
    readonly engine_set_camera: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
    readonly engine_set_culling: (a: number, b: number) => void;
    readonly engine_set_driven_data: (a: number, b: number, c: number) => void;
    readonly engine_set_env_params: (a: number, b: number, c: number, d: number) => void;
    readonly engine_set_model_base_color: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly engine_set_model_transform: (a: number, b: number, c: number, d: number) => void;
    readonly engine_set_point_light: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly engine_set_point_light_count: (a: number, b: number) => void;
    readonly engine_set_sun: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly engine_set_sun_color: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly engine_sphere_bvh_node_count: (a: number) => number;
    readonly engine_sphere_bvh_nodes_bytes: (a: number) => number;
    readonly engine_sphere_bvh_nodes_ptr: (a: number) => number;
    readonly engine_sphere_bvh_spheres_bytes: (a: number) => number;
    readonly engine_sphere_bvh_spheres_ptr: (a: number) => number;
    readonly engine_transforms_bytes: (a: number) => number;
    readonly engine_transforms_ptr: (a: number) => number;
    readonly engine_update: (a: number, b: number) => void;
    readonly engine_use_sphere_mesh: (a: number, b: number, c: number) => void;
    readonly engine_vertices_bytes: (a: number) => number;
    readonly engine_vertices_ptr: (a: number) => number;
    readonly engine_visible_count: (a: number) => number;
    readonly engine_rt_camera_bytes: (a: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
