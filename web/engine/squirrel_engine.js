/* @ts-self-types="./squirrel_engine.d.ts" */

export class Engine {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        EngineFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_engine_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    camera_bytes() {
        const ret = wasm.engine_camera_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    camera_ptr() {
        const ret = wasm.engine_camera_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Remove all imported models (e.g. before loading a new file). Forces one
     * full material upload so any GPU slots the models dirtied are restored.
     */
    clear_models() {
        wasm.engine_clear_models(this.__wbg_ptr);
    }
    /**
     * @returns {number}
     */
    commands_ptr() {
        const ret = wasm.engine_commands_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Turn on driven mode: instance transforms + colors come from
     * `set_driven_data` each frame rather than the built-in animation. Culling
     * is disabled so instance order matches the supplied data 1:1.
     * @param {number} count
     */
    enable_driven(count) {
        wasm.engine_enable_driven(this.__wbg_ptr, count);
    }
    /**
     * @returns {number}
     */
    index_count() {
        const ret = wasm.engine_index_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    indices_ptr() {
        const ret = wasm.engine_indices_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Populate the scene with `count` procedurally-placed rotating cubes.
     * @param {number} count
     */
    init_scene(count) {
        wasm.engine_init_scene(this.__wbg_ptr, count);
    }
    /**
     * Total instances in the scene (before culling).
     * @returns {number}
     */
    instance_count() {
        const ret = wasm.engine_instance_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    lighting_bytes() {
        const ret = wasm.engine_lighting_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    lighting_ptr() {
        const ret = wasm.engine_lighting_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    materials_bytes() {
        const ret = wasm.engine_materials_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    materials_ptr() {
        const ret = wasm.engine_materials_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Material bytes JS must upload after this frame's `update()` — 0 when the
     * GPU copy is already correct (the common case: no models resident).
     * @returns {number}
     */
    materials_upload_bytes() {
        const ret = wasm.engine_materials_upload_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Number of imported models currently resident.
     * @returns {number}
     */
    model_count() {
        const ret = wasm.engine_model_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Maximum imported models (so JS can size its GPU buffers to match).
     * @returns {number}
     */
    models_capacity() {
        const ret = wasm.engine_models_capacity(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create an engine sized for at most `max_instances` objects.
     * @param {number} max_instances
     */
    constructor(max_instances) {
        const ret = wasm.engine_new(max_instances);
        this.__wbg_ptr = ret;
        EngineFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Register a glTF primitive whose vertex + index buffers (and per-draw
     * material bind group) JS has already created. `pipeline_id` selects the
     * render pipeline, `material_bg` the group-2 material (albedo texture).
     * Returns the model index used by the transform / material setters, or
     * `u32::MAX` if the table is full.
     * @param {number} vertex_buffer_id
     * @param {number} index_buffer_id
     * @param {number} index_count
     * @param {number} index_format
     * @param {number} pipeline_id
     * @param {number} material_bg
     * @returns {number}
     */
    register_model(vertex_buffer_id, index_buffer_id, index_count, index_format, pipeline_id, material_bg) {
        const ret = wasm.engine_register_model(this.__wbg_ptr, vertex_buffer_id, index_buffer_id, index_count, index_format, pipeline_id, material_bg);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    rt_camera_bytes() {
        const ret = wasm.engine_rt_camera_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    rt_camera_ptr() {
        const ret = wasm.engine_rt_camera_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * JS passes the canvas aspect ratio (width / height) so WASM can build
     * the projection matrix. Called at init and on every resize.
     * @param {number} aspect
     */
    set_aspect(aspect) {
        wasm.engine_set_aspect(this.__wbg_ptr, aspect);
    }
    /**
     * Set an explicit camera (eye + look-at target), overriding the orbit.
     * @param {number} ex
     * @param {number} ey
     * @param {number} ez
     * @param {number} tx
     * @param {number} ty
     * @param {number} tz
     */
    set_camera(ex, ey, ez, tx, ty, tz) {
        wasm.engine_set_camera(this.__wbg_ptr, ex, ey, ez, tx, ty, tz);
    }
    /**
     * Enable or disable frustum culling (on by default).
     * @param {boolean} enabled
     */
    set_culling(enabled) {
        wasm.engine_set_culling(this.__wbg_ptr, enabled);
    }
    /**
     * Supply this frame's instance data: 7 floats each — position `(x,y,z)`,
     * `radius`, color `(r,g,b)`. `data.len() / 7` sets the instance count.
     * @param {Float32Array} data
     */
    set_driven_data(data) {
        const ptr0 = passArrayF32ToWasm0(data, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.engine_set_driven_data(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Environment (skybox) contribution: `ambient` = how much the cubemap
     * lights shadowed areas, `reflect` = how mirror-like surfaces are,
     * `exposure` = overall tonemap exposure.
     * @param {number} ambient
     * @param {number} reflect
     * @param {number} exposure
     */
    set_env_params(ambient, reflect, exposure) {
        wasm.engine_set_env_params(this.__wbg_ptr, ambient, reflect, exposure);
    }
    /**
     * Set model `idx`'s base color (glTF `baseColorFactor`). Alpha ≥ 0 marks a
     * real material; the cube field uses alpha < 0 to mean "tint by normal".
     * @param {number} idx
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} a
     */
    set_model_base_color(idx, r, g, b, a) {
        wasm.engine_set_model_base_color(this.__wbg_ptr, idx, r, g, b, a);
    }
    /**
     * Set model `idx`'s full column-major 4x4 model matrix (16 floats).
     * @param {number} idx
     * @param {Float32Array} m
     */
    set_model_transform(idx, m) {
        const ptr0 = passArrayF32ToWasm0(m, wasm.__wbindgen_malloc);
        const len0 = WASM_VECTOR_LEN;
        wasm.engine_set_model_transform(this.__wbg_ptr, idx, ptr0, len0);
    }
    /**
     * Configure point light `idx` (`0..4`). `range` is the falloff distance;
     * `intensity` scales the color. Also raises the active light count so the
     * shader loops over it.
     * @param {number} idx
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} range
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} intensity
     */
    set_point_light(idx, x, y, z, range, r, g, b, intensity) {
        wasm.engine_set_point_light(this.__wbg_ptr, idx, x, y, z, range, r, g, b, intensity);
    }
    /**
     * Set how many point lights (`0..4`) the shader should evaluate.
     * @param {number} count
     */
    set_point_light_count(count) {
        wasm.engine_set_point_light_count(this.__wbg_ptr, count);
    }
    /**
     * Set the directional "sun": `(dx, dy, dz)` is the direction *toward* the
     * light (normalized internally); `intensity` scales its contribution.
     * @param {number} dx
     * @param {number} dy
     * @param {number} dz
     * @param {number} intensity
     */
    set_sun(dx, dy, dz, intensity) {
        wasm.engine_set_sun(this.__wbg_ptr, dx, dy, dz, intensity);
    }
    /**
     * Set the sun color (`r,g,b`) and the flat ambient floor.
     * @param {number} r
     * @param {number} g
     * @param {number} b
     * @param {number} ambient
     */
    set_sun_color(r, g, b, ambient) {
        wasm.engine_set_sun_color(this.__wbg_ptr, r, g, b, ambient);
    }
    /**
     * @returns {number}
     */
    sphere_bvh_node_count() {
        const ret = wasm.engine_sphere_bvh_node_count(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    sphere_bvh_nodes_bytes() {
        const ret = wasm.engine_sphere_bvh_nodes_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Static BVH over the cube field (see `bvh::build`), built once in
     * `init_scene`. Node layout matches WGSL's shared `BvhNode` struct — the
     * same one used for imported-model triangles — so both are walked by the
     * same kind of stackless traversal in the ray tracer.
     * @returns {number}
     */
    sphere_bvh_nodes_ptr() {
        const ret = wasm.engine_sphere_bvh_nodes_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    sphere_bvh_spheres_bytes() {
        const ret = wasm.engine_sphere_bvh_spheres_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Sphere records (center.xyz + radius), reordered to match the BVH's
     * leaf layout. Static — uploaded once, never touched per frame.
     * @returns {number}
     */
    sphere_bvh_spheres_ptr() {
        const ret = wasm.engine_sphere_bvh_spheres_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    transforms_bytes() {
        const ret = wasm.engine_transforms_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    transforms_ptr() {
        const ret = wasm.engine_transforms_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Advance the simulation by `dt` seconds and rebuild all per-frame data:
     * transforms, materials, the camera + lighting uniforms, and the command
     * stream. After this returns, JS uploads the buffers and replays.
     * @param {number} dt
     */
    update(dt) {
        wasm.engine_update(this.__wbg_ptr, dt);
    }
    /**
     * Swap the instanced mesh from the default cube to a unit UV sphere. Call
     * this *before* `Renderer.init` so JS uploads the sphere geometry.
     * @param {number} rings
     * @param {number} sectors
     */
    use_sphere_mesh(rings, sectors) {
        wasm.engine_use_sphere_mesh(this.__wbg_ptr, rings, sectors);
    }
    /**
     * @returns {number}
     */
    vertices_bytes() {
        const ret = wasm.engine_vertices_bytes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    vertices_ptr() {
        const ret = wasm.engine_vertices_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Instances that survived frustum culling this frame (drawn / traced).
     * @returns {number}
     */
    visible_count() {
        const ret = wasm.engine_visible_count(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) Engine.prototype[Symbol.dispose] = Engine.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./squirrel_engine_bg.js": import0,
    };
}

const EngineFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_engine_free(ptr, 1));

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('squirrel_engine_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
