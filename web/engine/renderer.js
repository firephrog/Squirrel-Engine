// Squirrel Engine — JS / WebGPU layer.
//
// This side owns the GPUDevice and its entire object graph. WASM never sees a
// GPU handle; it refers to resources by integer ID into the tables below, and
// this module resolves those IDs to real GPU objects only at replay time.
//
// Beyond the original instanced-cube path this now drives:
//   * lighting     — a WASM-owned Lighting uniform (sun + point lights + env)
//                    consumed by the lit forward shader.
//   * a sky cubemap — rendered procedurally into the six faces of a cube
//                    texture ("cubemap render"), used as the skybox background,
//                    the raster ambient / reflection source, and the RT sky.
//   * glTF models  — vertex/index buffers uploaded here, registered with the
//                    WASM model table, and folded into the command stream.

// Command opcodes — mirror of the Rust constants in lib.rs.
const CMD_END = 0;
const CMD_DRAW_INDEXED = 1;
const CMD_SET_MESH = 2;
const CMD_STRIDE = 8; // u32s per command record

// Interleaved vertex layout: position vec3 + normal vec3.
const VERTEX_STRIDE = 6 * 4; // 24 bytes (cube field)
// Model vertex layout adds UVs: position vec3 + normal vec3 + uv vec2.
const MODEL_VERTEX_STRIDE = 8 * 4; // 32 bytes

// Ray tracer traces imported models as world-space triangles. Rather than a
// brute-force per-pixel loop over every triangle (which made a loaded model
// crawl), the triangles are indexed by a BVH built here on load and traversed
// stacklessly in the compute shader, so each ray tests only a handful. The
// triangle + BVH GPU buffers are sized to the actual model at load time (not
// reserved up front), so a big mesh renders in full instead of being clipped.
// Per triangle: v0(vec4, w=u0) + v1(vec4, w=u1) + v2(vec4, w=u2)
//             + color(vec4, w=textured flag) + uv(vec4: v0v,v1v,v2v).
const RT_TRI_FLOATS = 20;
// BVH node: bmin(vec4, w=escape index) + bmax(vec4, w=leaf tri count)
//         + tri(vec4, x=first tri index).
const RT_NODE_FLOATS = 12;
const RT_LEAF_TRIS = 4; // triangles per BVH leaf (split target)
// Safety cap. A binary BVH over T triangles has < 2T nodes; each node is 48
// bytes, so 2T * 48 must stay under WebGPU's default 128 MiB storage-binding
// limit -> T < ~1.4M. 1.2M leaves headroom and still covers, e.g., the full
// ~871k-triangle Stanford dragon. Larger meshes are truncated with a warning.
const MAX_RT_TRIS = 1200000;

const DEPTH_FORMAT = "depth24plus";
// Linear-radiance cubemap so the sun can exceed 1.0; filterable + renderable.
const ENV_FORMAT = "rgba16float";
const SKY_SIZE = 256; // per-face resolution of the sky cubemap

// Per-face basis: a fragment at screen position (uv in [-1,1]) samples the
// direction  forward + uv.x*right + uv.y*up. These match the WebGPU cube-face
// convention so the rendered image is consistent with textureSample(cube, dir).
const CUBE_FACES = [
  { forward: [1, 0, 0], right: [0, 0, -1], up: [0, 1, 0] }, // +X
  { forward: [-1, 0, 0], right: [0, 0, 1], up: [0, 1, 0] }, // -X
  { forward: [0, 1, 0], right: [1, 0, 0], up: [0, 0, -1] }, // +Y
  { forward: [0, -1, 0], right: [1, 0, 0], up: [0, 0, 1] }, // -Y
  { forward: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0] }, // +Z
  { forward: [0, 0, -1], right: [-1, 0, 0], up: [0, 1, 0] }, // -Z
];

function normalize3(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

/**
 * Build a stackless (escape-linked) BVH over `count` triangles.
 *
 * Input: parallel arrays of per-triangle world-space bounds + centroids
 * (`triMin`, `triMax`, `triCentroid`, each `count * 3` floats). The build
 * reorders triangles by a median split on the longest centroid axis and emits
 * nodes in depth-first order, so a node's first child is always the next node.
 * Each node stores its *escape index* (the node to jump to when its box is
 * missed, or after a leaf's triangles are tested) — the whole traversal is then
 * a single `while` loop with no per-ray stack. See the matching walk in
 * raytrace.wgsl.
 *
 * @returns {{ nodes: Float32Array, order: Int32Array, nodeCount: number }}
 *   `order[i]` is the original triangle index now at BVH position `i`; callers
 *   write triangle records in that order so a leaf's `[first, first+count)`
 *   range is contiguous.
 */
function buildBVH(triMin, triMax, triCentroid, count) {
  const order = new Int32Array(count);
  for (let i = 0; i < count; i++) order[i] = i;

  // Nodes are collected as plain objects first (escape links need a second
  // pass once every node has an index), then flattened.
  const nodes = [];
  // Explicit work stack of [start, end, parentIndexToPatch, isSecondChild].
  // We record each node's index and, for internal nodes, remember to fill the
  // escape link after the whole subtree is emitted.
  const AABB = (start, end) => {
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const t = order[i] * 3;
      for (let k = 0; k < 3; k++) {
        if (triMin[t + k] < mn[k]) mn[k] = triMin[t + k];
        if (triMax[t + k] > mx[k]) mx[k] = triMax[t + k];
      }
    }
    return { mn, mx };
  };

  // Iterative build. Each frame: build a node for [start,end); if it should
  // split, partition and push the two children (right first so left is emitted
  // next and becomes node+1). We fix escape links in a final pass using each
  // node's recorded subtree end.
  const stack = [[0, count]];
  // Parallel record of the subtree extent so escape = subtreeEnd afterwards.
  while (stack.length) {
    const [start, end] = stack.pop();
    const { mn, mx } = AABB(start, end);
    const n = end - start;
    const self = nodes.length;
    const node = { mn, mx, first: start, tris: 0, escape: -1 };
    nodes.push(node);

    if (n <= RT_LEAF_TRIS) {
      node.tris = n; // leaf
      continue;
    }

    // Split on the longest centroid-extent axis at its midpoint; fall back to a
    // median split if everything lands on one side (degenerate spread).
    let cmn = [Infinity, Infinity, Infinity];
    let cmx = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; i++) {
      const c = order[i] * 3;
      for (let k = 0; k < 3; k++) {
        if (triCentroid[c + k] < cmn[k]) cmn[k] = triCentroid[c + k];
        if (triCentroid[c + k] > cmx[k]) cmx[k] = triCentroid[c + k];
      }
    }
    let axis = 0;
    let ext = cmx[0] - cmn[0];
    if (cmx[1] - cmn[1] > ext) { axis = 1; ext = cmx[1] - cmn[1]; }
    if (cmx[2] - cmn[2] > ext) { axis = 2; ext = cmx[2] - cmn[2]; }

    let mid;
    if (ext <= 1e-9) {
      mid = (start + end) >> 1; // all centroids coincide — just halve
    } else {
      const splitPos = 0.5 * (cmn[axis] + cmx[axis]);
      // In-place partition of `order[start..end)` by centroid on `axis`.
      let i = start;
      let j = end - 1;
      while (i <= j) {
        while (i <= j && triCentroid[order[i] * 3 + axis] < splitPos) i++;
        while (i <= j && triCentroid[order[j] * 3 + axis] >= splitPos) j--;
        if (i < j) { const tmp = order[i]; order[i] = order[j]; order[j] = tmp; }
      }
      mid = i;
      if (mid === start || mid === end) mid = (start + end) >> 1; // guard
    }

    node.tris = 0; // internal
    // Push right first so the left child is popped/emitted next (node+1).
    stack.push([mid, end]);
    stack.push([start, mid]);
  }

  // Escape links. In this preorder layout a node's whole subtree is contiguous,
  // so escape[i] = i + subtreeSize[i] (the index just past the subtree), or -1
  // if that runs off the end. Subtree sizes are computed bottom-up: 1 for a
  // leaf, else 1 + left.size + right.size, with the right child sitting right
  // after the left subtree.
  const N = nodes.length;
  const size = new Int32Array(N);
  for (let i = N - 1; i >= 0; i--) {
    if (nodes[i].tris > 0) {
      size[i] = 1; // leaf
    } else {
      const left = i + 1;
      const leftEnd = left + size[left];
      size[i] = 1 + size[left] + size[leftEnd];
    }
  }
  for (let i = 0; i < N; i++) {
    const esc = i + size[i];
    nodes[i].escape = esc >= N ? -1 : esc;
  }

  const flat = new Float32Array(N * RT_NODE_FLOATS);
  for (let i = 0; i < N; i++) {
    const nd = nodes[i];
    const o = i * RT_NODE_FLOATS;
    flat[o + 0] = nd.mn[0]; flat[o + 1] = nd.mn[1]; flat[o + 2] = nd.mn[2];
    flat[o + 3] = nd.escape;
    flat[o + 4] = nd.mx[0]; flat[o + 5] = nd.mx[1]; flat[o + 6] = nd.mx[2];
    flat[o + 7] = nd.tris; // >0 => leaf
    flat[o + 8] = nd.first;
  }
  return { nodes: flat, order, nodeCount: N };
}

export class Renderer {
  constructor(engine, wasm) {
    this.engine = engine;
    this.wasm = wasm; // the wasm exports object; wasm.memory is the linear memory

    // Resource tables. WASM indexes into these by u32 ID.
    this.pipelines = [];
    this.bindGroups = [];
    this.buffers = [];

    // Deforming soft-body / cloth meshes, drawn with the two-sided lit pipeline.
    this.softMeshes = [];

    // GPU resources backing imported glTF meshes, so we can free them on reload.
    this.gltfBufferIds = [];
    this.gltfTextures = [];
    this.gltfMaterialBgIds = [];

    // The single albedo texture view the ray tracer samples for textured model
    // triangles (RT traces one merged triangle soup, so it samples one image;
    // primitives with a different texture fall back to their flat base color).
    // Defaults to the shared 1x1 white texture until a textured model loads.
    this.rtAlbedoView = null;

    // Persistent view over the command stream. Re-created if WASM memory grows.
    this._cmdView = null;
    this._cachedBuffer = null;

    this.device = null;
    this.context = null;
    this.format = null;
    this.depthTexture = null;
    this.canvas = null;

    // Render mode setting: "raster" (default) or "raytrace".
    this.mode = "raster";
    // Ray tracer internal resolution as a fraction of the canvas.
    this.rtScale = 1.0;
    this.rtTexture = null;

    // Sky / sun state driving the cubemap render. The sun direction is kept in
    // sync with the WASM lighting sun so both agree.
    this.sky = {
      sunDir: normalize3([0.4, 0.9, 0.35]),
      sunSize: 0.008,
      sunColor: [1.0, 0.9, 0.7],
      sunIntensity: 6.0,
      top: [0.08, 0.2, 0.55],
      horizon: [0.55, 0.62, 0.72],
      ground: [0.03, 0.035, 0.045],
    };
  }

  // --- Resource table registration. Returns the integer ID WASM will use. ---
  _addPipeline(p) {
    this.pipelines.push(p);
    return this.pipelines.length - 1;
  }
  _addBindGroup(b) {
    this.bindGroups.push(b);
    return this.bindGroups.length - 1;
  }
  _addBuffer(b) {
    this.buffers.push(b);
    return this.buffers.length - 1;
  }

  async init(canvas, shaders, maxInstances) {
    if (!navigator.gpu) {
      throw new Error("WebGPU not available. Use Chrome/Edge 113+ or enable the flag.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No suitable GPU adapter found.");

    const device = await adapter.requestDevice();
    this.device = device;
    this.canvas = canvas;
    this.maxInstances = maxInstances;
    this.modelsCap = this.engine.models_capacity();
    const totalSlots = maxInstances + this.modelsCap;

    device.onuncapturederror = (e) => console.error("[WebGPU uncaptured]", e.error);
    device.lost.then((info) => {
      console.error("[WebGPU device lost]", info.message, info.reason);
    });

    const context = canvas.getContext("webgpu");
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format: this.format, alphaMode: "opaque" });

    device.pushErrorScope("validation");

    const memBuf = () => this.wasm.memory.buffer;

    // --- Cube geometry (uploaded once from WASM memory) ---
    const vBytes = this.engine.vertices_bytes();
    const vertexBuffer = device.createBuffer({
      size: vBytes,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Uint8Array(vertexBuffer.getMappedRange()).set(
      new Uint8Array(memBuf(), this.engine.vertices_ptr(), vBytes)
    );
    vertexBuffer.unmap();
    this.vertexBufferId = this._addBuffer(vertexBuffer);

    const idxCount = this.engine.index_count();
    const iBytes = idxCount * 4;
    const indexBuffer = device.createBuffer({
      size: iBytes,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint8Array(indexBuffer.getMappedRange()).set(
      new Uint8Array(memBuf(), this.engine.indices_ptr(), iBytes)
    );
    indexBuffer.unmap();
    this.indexBufferId = this._addBuffer(indexBuffer);
    this.indexCount = idxCount;

    // --- Per-frame uniform + storage buffers (written every frame) ---
    const cameraBuffer = device.createBuffer({
      size: this.engine.camera_bytes(),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.cameraBufferId = this._addBuffer(cameraBuffer);

    const lightingBuffer = device.createBuffer({
      size: this.engine.lighting_bytes(),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.lightingBufferId = this._addBuffer(lightingBuffer);

    // One matrix (64 bytes) per instance, sized for every cube + imported model.
    const transformBuffer = device.createBuffer({
      size: totalSlots * 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.transformBufferId = this._addBuffer(transformBuffer);

    // One base color (vec4, 16 bytes) per instance, parallel to transforms.
    const materialBuffer = device.createBuffer({
      size: totalSlots * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.materialBufferId = this._addBuffer(materialBuffer);

    // Cube-field ray-tracing scene: a static BVH (see bvh.rs) plus its
    // reordered sphere records. Instance positions never change after scene
    // init, so — like the vertex/index buffers above — this uploads once
    // here and is never touched again per frame.
    const sphereBytes = this.engine.sphere_bvh_spheres_bytes();
    const sphereBuffer = device.createBuffer({
      size: sphereBytes,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint8Array(sphereBuffer.getMappedRange()).set(
      new Uint8Array(memBuf(), this.engine.sphere_bvh_spheres_ptr(), sphereBytes)
    );
    sphereBuffer.unmap();
    this.sphereBufferId = this._addBuffer(sphereBuffer);

    const sphereBvhBytes = this.engine.sphere_bvh_nodes_bytes();
    const sphereBvhBuffer = device.createBuffer({
      size: sphereBvhBytes,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint8Array(sphereBvhBuffer.getMappedRange()).set(
      new Uint8Array(memBuf(), this.engine.sphere_bvh_nodes_ptr(), sphereBvhBytes)
    );
    sphereBvhBuffer.unmap();
    this.sphereBvhBufferId = this._addBuffer(sphereBvhBuffer);

    // --- Environment (sky cubemap) resources ---
    this._initEnvironment();

    // --- Lit forward pipeline (group 0 per-frame, group 1 environment) ---
    const module = device.createShaderModule({ code: shaders.raster });
    const perFrameLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });

    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [perFrameLayout, this.envLayout],
      }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: VERTEX_STRIDE,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
              { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });
    this._addPipeline(pipeline); // id 0

    // @group(0): per-frame camera + transforms + lighting + materials.
    const bindGroup = device.createBindGroup({
      layout: perFrameLayout,
      entries: [
        { binding: 0, resource: { buffer: cameraBuffer } },
        { binding: 1, resource: { buffer: transformBuffer } },
        { binding: 2, resource: { buffer: lightingBuffer } },
        { binding: 3, resource: { buffer: materialBuffer } },
      ],
    });
    this._addBindGroup(bindGroup); // id 0
    this.perFrameLayout = perFrameLayout;

    // --- Textured model pipeline (group 2 = albedo texture + sampler) ---
    this._initModelPipeline(shaders.model);

    // --- Sky cubemap render + skybox background pipelines ---
    this._initSky(shaders.sky, shaders.skybox);

    // --- Ray-tracing resources (compute tracer + fullscreen blit) ---
    this._initRaytrace(shaders.raytrace, shaders.blit);

    // --- Soft-body / cloth mesh pipeline (two-sided, deforming) ---
    if (shaders.softmesh) this._initSoftMesh(shaders.softmesh);

    const err = await device.popErrorScope();
    if (err) throw new Error("Pipeline setup validation error: " + err.message);

    this._createDepthTexture();
    this._createRTTargets();

    // Render the initial sky cubemap.
    this.rebuildSky();
  }

  // Create the cube texture, its sampler, and the shared env bind-group layout
  // + bind group used by the lit and skybox passes.
  _initEnvironment() {
    const device = this.device;
    this.envTexture = device.createTexture({
      size: [SKY_SIZE, SKY_SIZE, 6],
      format: ENV_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      dimension: "2d",
    });
    this.envCubeView = this.envTexture.createView({ dimension: "cube" });
    this.envFaceViews = [];
    for (let i = 0; i < 6; i++) {
      this.envFaceViews.push(
        this.envTexture.createView({ dimension: "2d", baseArrayLayer: i, arrayLayerCount: 1 })
      );
    }
    this.envSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.envLayout = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "float", viewDimension: "cube" },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });
    this.envBindGroup = device.createBindGroup({
      layout: this.envLayout,
      entries: [
        { binding: 0, resource: this.envCubeView },
        { binding: 1, resource: this.envSampler },
      ],
    });
  }

  // Textured model pipeline: pos+normal+uv geometry, group 2 = albedo texture.
  // Shares the per-frame (group 0) and environment (group 1) layouts with the
  // cube pipeline so a pipeline switch keeps those bind groups valid.
  _initModelPipeline(modelCode) {
    const device = this.device;

    this.materialLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    const module = device.createShaderModule({ code: modelCode });
    const pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.perFrameLayout, this.envLayout, this.materialLayout],
      }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: MODEL_VERTEX_STRIDE,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" }, // position
              { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
              { shaderLocation: 2, offset: 24, format: "float32x2" }, // uv
            ],
          },
        ],
      },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      // glTF front faces are CCW; many exported models rely on double-sided, so
      // don't cull — imported meshes shouldn't vanish from the back.
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });
    this.modelPipelineId = this._addPipeline(pipeline); // id 1

    // Sampler + a 1x1 white texture used by models that carry no albedo image
    // (their baseColorFactor then shows through unchanged).
    this.albedoSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    });
    this.whiteTexture = device.createTexture({
      size: [1, 1],
      format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    device.queue.writeTexture(
      { texture: this.whiteTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      { width: 1, height: 1 }
    );
    this.defaultMaterialBgId = this._addBindGroup(
      device.createBindGroup({
        layout: this.materialLayout,
        entries: [
          { binding: 0, resource: this.whiteTexture.createView() },
          { binding: 1, resource: this.albedoSampler },
        ],
      })
    );
  }

  // Build the procedural-sky render pipeline (one uniform + bind group per face)
  // and the fullscreen skybox pipeline that presents the cubemap as background.
  _initSky(skyCode, skyboxCode) {
    const device = this.device;

    // Sky face render: SkyFace uniform (128 bytes) -> one cube face.
    const skyModule = device.createShaderModule({ code: skyCode });
    this.skyFaceLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    });
    this.skyPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.skyFaceLayout] }),
      vertex: { module: skyModule, entryPoint: "vs_main" },
      fragment: {
        module: skyModule,
        entryPoint: "fs_main",
        targets: [{ format: ENV_FORMAT }],
      },
      primitive: { topology: "triangle-list" },
    });

    // Six per-face uniform buffers + bind groups.
    this.skyFaceBuffers = [];
    this.skyFaceBindGroups = [];
    for (let i = 0; i < 6; i++) {
      const buf = device.createBuffer({
        size: 128,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      this.skyFaceBuffers.push(buf);
      this.skyFaceBindGroups.push(
        device.createBindGroup({
          layout: this.skyFaceLayout,
          entries: [{ binding: 0, resource: { buffer: buf } }],
        })
      );
    }

    // Skybox background: RT camera (group 0) + env cubemap (group 1).
    const skyboxModule = device.createShaderModule({ code: skyboxCode });
    this.skyboxCamLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    });
    this.skyboxPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.skyboxCamLayout, this.envLayout],
      }),
      vertex: { module: skyboxModule, entryPoint: "vs_main" },
      fragment: {
        module: skyboxModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
      // Drawn AFTER the opaque scene at z = 1 with less-equal: only pixels the
      // geometry left untouched (depth still at the clear value 1.0) run the
      // cubemap fragment shader — zero sky overdraw instead of fullscreen.
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "less-equal" },
    });
  }

  // Fill the six SkyFace uniforms from the current sky/sun state and render the
  // cubemap. Cheap enough to call whenever the sun or palette changes.
  rebuildSky() {
    const device = this.device;
    const s = this.sky;
    for (let i = 0; i < 6; i++) {
      const f = CUBE_FACES[i];
      const u = new Float32Array(32);
      u.set(f.forward, 0);
      u.set(f.right, 4);
      u.set(f.up, 8);
      u.set(s.sunDir, 12);
      u[15] = s.sunSize;
      u.set(s.sunColor, 16);
      u[19] = s.sunIntensity;
      u.set(s.top, 20);
      u.set(s.horizon, 24);
      u.set(s.ground, 28);
      device.queue.writeBuffer(this.skyFaceBuffers[i], 0, u);
    }

    const encoder = device.createCommandEncoder();
    for (let i = 0; i < 6; i++) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: this.envFaceViews[i], loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } },
        ],
      });
      pass.setPipeline(this.skyPipeline);
      pass.setBindGroup(0, this.skyFaceBindGroups[i]);
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
  }

  /** Point the sun in a new direction (xyz toward the light) and re-render the
   *  sky. Also updates the WASM lighting sun so shading matches the skybox. */
  setSunDirection(x, y, z, intensity = 1.15) {
    this.sky.sunDir = normalize3([x, y, z]);
    this.engine.set_sun(x, y, z, intensity);
    this.rebuildSky();
  }

  _initRaytrace(computeCode, blitCode) {
    const device = this.device;

    // RT camera uniform (eye + basis), written from WASM each frame.
    const rtCameraBuffer = device.createBuffer({
      size: this.engine.rt_camera_bytes(),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.rtCameraBufferId = this._addBuffer(rtCameraBuffer);

    // Imported-model triangles for the ray tracer (world space), the BVH that
    // indexes them, and a small uniform holding the model's bounding sphere +
    // triangle/node counts. The tri + BVH buffers start as 1-element
    // placeholders and are reallocated to fit each model in loadModel; tri_count
    // = 0 means "no model" and the tracer skips the mesh entirely.
    this.rtTriBuffer = device.createBuffer({
      size: RT_TRI_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.rtBvhBuffer = device.createBuffer({
      size: RT_NODE_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.rtInfoBuffer = device.createBuffer({
      size: 32, // vec4 bounding sphere + (tri_count, node_count) + padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.rtInfoBuffer, 0, new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]));

    // Until a textured model is loaded the tracer samples the shared 1x1 white
    // texture, so textured triangles just show their base color.
    this.rtAlbedoView = this.whiteTexture.createView();

    // Skybox camera bind group reuses the RT camera uniform.
    this.skyboxCamBindGroup = device.createBindGroup({
      layout: this.skyboxCamLayout,
      entries: [{ binding: 0, resource: { buffer: rtCameraBuffer } }],
    });

    // Compute pipeline: camera + transforms + sky cubemap -> storage texture.
    const computeModule = device.createShaderModule({ code: computeCode });
    this.rtComputeLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: "rgba8unorm" },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: "float", viewDimension: "cube" },
        },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        // Model albedo texture + sampler (textured RT triangles) and the BVH.
        { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        // Static BVH over the cube field's spheres (see bvh.rs).
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      ],
    });
    this.rtComputePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.rtComputeLayout] }),
      compute: { module: computeModule, entryPoint: "cs_main" },
    });

    // Blit pipeline: sample the RT output texture to the swapchain.
    const blitModule = device.createShaderModule({ code: blitCode });
    this.blitLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    });
    this.blitPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.blitLayout] }),
      vertex: { module: blitModule, entryPoint: "vs_main" },
      fragment: {
        module: blitModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.blitSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  }

  // (Re)create the reduced-resolution RT target and the bind groups that
  // reference it. Called at init and on resize.
  _createRTTargets() {
    const device = this.device;
    const w = Math.max(1, Math.floor(this.canvas.width * this.rtScale));
    const h = Math.max(1, Math.floor(this.canvas.height * this.rtScale));
    this.rtW = w;
    this.rtH = h;

    if (this.rtTexture) this.rtTexture.destroy();
    this.rtTexture = device.createTexture({
      size: [w, h],
      format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.rtOutView = this.rtTexture.createView();

    this._rebuildRTComputeBindGroup();
    this.blitBindGroup = device.createBindGroup({
      layout: this.blitLayout,
      entries: [
        { binding: 0, resource: this.blitSampler },
        { binding: 1, resource: this.rtOutView },
      ],
    });
  }

  // (Re)create the compute bind group. Separate from _createRTTargets because it
  // must also run when a model load swaps the RT albedo texture (binding 7),
  // while the RT output texture (binding 2) only changes on resize.
  _rebuildRTComputeBindGroup() {
    this.rtComputeBindGroup = this.device.createBindGroup({
      layout: this.rtComputeLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers[this.rtCameraBufferId] } },
        { binding: 1, resource: { buffer: this.buffers[this.sphereBufferId] } },
        { binding: 2, resource: this.rtOutView },
        { binding: 3, resource: this.envCubeView },
        { binding: 4, resource: this.envSampler },
        { binding: 5, resource: { buffer: this.rtTriBuffer } },
        { binding: 6, resource: { buffer: this.rtInfoBuffer } },
        { binding: 7, resource: this.rtAlbedoView },
        { binding: 8, resource: this.albedoSampler },
        { binding: 9, resource: { buffer: this.rtBvhBuffer } },
        { binding: 10, resource: { buffer: this.buffers[this.sphereBvhBufferId] } },
      ],
    });
  }

  /** Set the render mode: "raster" or "raytrace". */
  setMode(mode) {
    this.mode = mode === "raytrace" ? "raytrace" : "raster";
    return this.mode;
  }

  /** Toggle between rasterization and ray tracing. Returns the new mode. */
  toggleMode() {
    return this.setMode(this.mode === "raster" ? "raytrace" : "raster");
  }

  /**
   * Upload glTF primitives (from gltf.js) as meshes and register them with the
   * WASM model table. Replaces any previously-loaded model. Returns the number
   * of primitives that were accepted.
   */
  loadModel(primitives) {
    const device = this.device;

    // Free the previous model's GPU buffers/textures and drop it from the table.
    for (const id of this.gltfBufferIds) {
      if (this.buffers[id]) {
        this.buffers[id].destroy();
        this.buffers[id] = null;
      }
    }
    for (const tex of this.gltfTextures) tex.destroy();
    for (const id of this.gltfMaterialBgIds) this.bindGroups[id] = null;
    this.gltfBufferIds = [];
    this.gltfTextures = [];
    this.gltfMaterialBgIds = [];
    this.engine.clear_models();

    // The ray tracer traces one merged triangle soup, so it samples a single
    // albedo texture: the first primitive that carries one. Triangles from other
    // textured primitives fall back to their flat base color in RT (they still
    // texture correctly in raster, which binds each primitive's own texture).
    let rtAlbedoImage = null;
    for (const prim of primitives) {
      if (prim.image) { rtAlbedoImage = prim.image; break; }
    }
    let rtAlbedoTex = null;

    // Accumulate ray-tracer triangles (world space) + per-triangle bounds and
    // centroids for the BVH, plus the overall bounding box.
    const totalTris = primitives.reduce((s, p) => s + Math.floor(p.indexCount / 3), 0);
    const triCap = Math.min(MAX_RT_TRIS, totalTris);
    if (totalTris > MAX_RT_TRIS) {
      console.warn(
        `Squirrel RT: model has ${totalTris} triangles; ray tracer caps at ` +
        `${MAX_RT_TRIS} (raster draws all). Raise MAX_RT_TRIS for the full mesh.`
      );
    }
    const tris = new Float32Array(triCap * RT_TRI_FLOATS);
    const triMin = new Float32Array(triCap * 3);
    const triMax = new Float32Array(triCap * 3);
    const triCen = new Float32Array(triCap * 3);
    let triCount = 0;
    const bmin = [Infinity, Infinity, Infinity];
    const bmax = [-Infinity, -Infinity, -Infinity];

    let accepted = 0;
    for (const prim of primitives) {
      const vBytes = prim.interleaved.byteLength;
      const vbuf = device.createBuffer({
        size: Math.ceil(vBytes / 4) * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
      });
      new Float32Array(vbuf.getMappedRange()).set(prim.interleaved);
      vbuf.unmap();

      const iBytes = prim.indices.byteLength;
      const ibuf = device.createBuffer({
        size: Math.ceil(iBytes / 4) * 4,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      if (prim.indexFormat === 1) {
        new Uint16Array(ibuf.getMappedRange(), 0, prim.indices.length).set(prim.indices);
      } else {
        new Uint32Array(ibuf.getMappedRange(), 0, prim.indices.length).set(prim.indices);
      }
      ibuf.unmap();

      const vId = this._addBuffer(vbuf);
      const iId = this._addBuffer(ibuf);
      this.gltfBufferIds.push(vId, iId);

      // Per-model albedo texture (or the shared white texture) -> material group.
      const { id: materialBg, tex } = this._createMaterialBindGroup(prim.image);
      if (prim.image === rtAlbedoImage && tex) rtAlbedoTex = tex;

      const modelIdx = this.engine.register_model(
        vId, iId, prim.indexCount, prim.indexFormat, this.modelPipelineId, materialBg
      );
      if (modelIdx === 0xffffffff) break; // table full
      this.engine.set_model_transform(modelIdx, prim.transform);
      const c = prim.baseColor;
      this.engine.set_model_base_color(modelIdx, c[0], c[1], c[2], c[3]);
      accepted++;

      // In RT, texture only the triangles whose primitive owns the single
      // albedo the tracer will sample; the rest keep their flat base color.
      const textured = prim.image === rtAlbedoImage && !!rtAlbedoImage;
      triCount = this._appendRTTriangles(
        prim, textured, tris, triMin, triMax, triCen, triCount, bmin, bmax
      );
    }

    // Point the tracer at the chosen albedo (or the white fallback), then build
    // the BVH over the collected triangles and upload everything for the RT.
    this.rtAlbedoView = (rtAlbedoTex || this.whiteTexture).createView();
    this._uploadRTModel(tris, triMin, triMax, triCen, triCount, bmin, bmax);
    this._rebuildRTComputeBindGroup();
    return accepted;
  }

  // Create a group-2 material bind group for a primitive's albedo image (or the
  // shared white texture when it has none). Returns { id, tex } — `tex` is the
  // created GPUTexture (null for the white fallback) so the ray tracer can reuse
  // it as its albedo source.
  _createMaterialBindGroup(image) {
    if (!image) return { id: this.defaultMaterialBgId, tex: null };
    const device = this.device;
    const tex = device.createTexture({
      size: [image.width, image.height],
      format: "rgba8unorm-srgb",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: image }, { texture: tex }, [
      image.width,
      image.height,
    ]);
    this.gltfTextures.push(tex);
    const id = this._addBindGroup(
      device.createBindGroup({
        layout: this.materialLayout,
        entries: [
          { binding: 0, resource: tex.createView() },
          { binding: 1, resource: this.albedoSampler },
        ],
      })
    );
    this.gltfMaterialBgIds.push(id);
    return { id, tex };
  }

  // Append a primitive's world-space triangles (position transformed by its
  // model matrix) into the RT triangle array, recording per-triangle bounds and
  // centroids for the BVH and expanding the overall bounding box. Each record is
  // pos0/1/2 (with u0/u1/u2 in the w lanes), color (w = textured flag), and the
  // three v-coords. Returns the new triangle count (capped at the array size).
  _appendRTTriangles(prim, textured, tris, triMin, triMax, triCen, triCount, bmin, bmax) {
    const m = prim.transform;
    const v = prim.interleaved; // stride 8: pos(3) normal(3) uv(2)
    const idx = prim.indices;
    const col = prim.baseColor;
    const cap = tris.length / RT_TRI_FLOATS;

    const worldPos = (i) => {
      const o = i * 8;
      const x = v[o], y = v[o + 1], z = v[o + 2];
      return [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
      ];
    };
    const uvOf = (i) => [v[i * 8 + 6], v[i * 8 + 7]];

    for (let t = 0; t + 2 < idx.length && triCount < cap; t += 3) {
      const i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
      const p0 = worldPos(i0), p1 = worldPos(i1), p2 = worldPos(i2);
      const uv0 = uvOf(i0), uv1 = uvOf(i1), uv2 = uvOf(i2);
      const base = triCount * RT_TRI_FLOATS;
      // v0.xyz + u0, v1.xyz + u1, v2.xyz + u2
      tris[base + 0] = p0[0]; tris[base + 1] = p0[1]; tris[base + 2] = p0[2]; tris[base + 3] = uv0[0];
      tris[base + 4] = p1[0]; tris[base + 5] = p1[1]; tris[base + 6] = p1[2]; tris[base + 7] = uv1[0];
      tris[base + 8] = p2[0]; tris[base + 9] = p2[1]; tris[base + 10] = p2[2]; tris[base + 11] = uv2[0];
      // color.rgb + textured flag
      tris[base + 12] = col[0]; tris[base + 13] = col[1]; tris[base + 14] = col[2];
      tris[base + 15] = textured ? 1 : 0;
      // uv v-coords for the three vertices
      tris[base + 16] = uv0[1]; tris[base + 17] = uv1[1]; tris[base + 18] = uv2[1];

      const tb = triCount * 3;
      for (let k = 0; k < 3; k++) {
        const lo = Math.min(p0[k], p1[k], p2[k]);
        const hi = Math.max(p0[k], p1[k], p2[k]);
        triMin[tb + k] = lo;
        triMax[tb + k] = hi;
        triCen[tb + k] = (p0[k] + p1[k] + p2[k]) / 3;
        if (lo < bmin[k]) bmin[k] = lo;
        if (hi > bmax[k]) bmax[k] = hi;
      }
      triCount++;
    }
    return triCount;
  }

  // Build the BVH over the collected triangles, reorder the triangle records to
  // match its leaves, (re)allocate the GPU buffers to fit, and push triangles +
  // nodes + info. tri_count = 0 (no model) leaves the tracer skipping the mesh.
  _uploadRTModel(tris, triMin, triMax, triCen, triCount, bmin, bmax) {
    const device = this.device;
    let nodeCount = 0;
    if (triCount > 0) {
      const { nodes, order, nodeCount: nc } = buildBVH(triMin, triMax, triCen, triCount);
      nodeCount = nc;

      // Reorder triangle records into BVH-leaf order so each leaf's triangles
      // are contiguous ([first, first+count)).
      const ordered = new Float32Array(triCount * RT_TRI_FLOATS);
      for (let i = 0; i < triCount; i++) {
        const src = order[i] * RT_TRI_FLOATS;
        ordered.set(tris.subarray(src, src + RT_TRI_FLOATS), i * RT_TRI_FLOATS);
      }

      // Size the GPU buffers to this model (they were placeholders / a previous
      // model's size). writeBuffer sizes must be a multiple of 4 bytes, which
      // whole f32 records already satisfy.
      this.rtTriBuffer.destroy();
      this.rtTriBuffer = device.createBuffer({
        size: triCount * RT_TRI_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.rtBvhBuffer.destroy();
      this.rtBvhBuffer = device.createBuffer({
        size: nodeCount * RT_NODE_FLOATS * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.rtTriBuffer, 0, ordered, 0, triCount * RT_TRI_FLOATS);
      device.queue.writeBuffer(this.rtBvhBuffer, 0, nodes, 0, nodeCount * RT_NODE_FLOATS);
    }

    let cx = 0, cy = 0, cz = 0, radius = 0;
    if (triCount > 0) {
      cx = (bmin[0] + bmax[0]) * 0.5;
      cy = (bmin[1] + bmax[1]) * 0.5;
      cz = (bmin[2] + bmax[2]) * 0.5;
      radius = 0.5 * Math.hypot(bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]);
    }
    device.queue.writeBuffer(
      this.rtInfoBuffer,
      0,
      new Float32Array([cx, cy, cz, radius, triCount, nodeCount, 0, 0])
    );
  }

  // --- Soft-body / cloth meshes -------------------------------------------
  // A deforming triangle surface, lit two-sided by softmesh.wgsl. Reuses the
  // engine's camera + lighting uniform buffers and the sky cubemap so cloth is
  // lit by the same sun/sky as everything else; geometry is world-space and
  // re-uploaded each frame (no per-instance model matrix).
  _initSoftMesh(code) {
    const device = this.device;
    this.softMeshLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    });
    const module = device.createShaderModule({ code });
    this.softMeshPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [this.softMeshLayout, this.envLayout],
      }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: VERTEX_STRIDE, // pos vec3 + normal vec3 (24 bytes)
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x3" },
            ],
          },
        ],
      },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      // Cloth is thin and viewed from both sides → don't cull; the shader flips
      // the normal toward the camera so both faces light correctly.
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  /**
   * Create a deforming mesh from a fixed surface topology. `vertexCount` is the
   * body's particle count; `indices` (Uint32Array) its surface triangles;
   * `color` an [r,g,b] tint. Returns a handle passed back to `updateSoftMesh`.
   */
  createSoftMesh(vertexCount, indices, color) {
    const device = this.device;
    const vbuf = device.createBuffer({
      size: vertexCount * VERTEX_STRIDE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const ibuf = device.createBuffer({
      size: Math.max(4, indices.length * 4),
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(ibuf.getMappedRange(), 0, indices.length).set(indices);
    ibuf.unmap();

    const colorBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(colorBuf, 0, new Float32Array([color[0], color[1], color[2], 1.0]));

    const bindGroup = device.createBindGroup({
      layout: this.softMeshLayout,
      entries: [
        { binding: 0, resource: { buffer: this.buffers[this.cameraBufferId] } },
        { binding: 1, resource: { buffer: this.buffers[this.lightingBufferId] } },
        { binding: 2, resource: { buffer: colorBuf } },
      ],
    });

    const mesh = {
      vbuf,
      ibuf,
      colorBuf,
      bindGroup,
      indexCount: indices.length,
      vertexCount,
      indices, // kept on the CPU for per-frame normal recomputation
      normals: new Float32Array(vertexCount * 3),
      interleaved: new Float32Array(vertexCount * 6),
      hidden: false,
    };
    this.softMeshes.push(mesh);
    return mesh;
  }

  /**
   * Re-skin a soft mesh from this frame's particle positions. `pos` is a
   * Float32Array of `vertexCount * 3` world-space xyz. Normals are recomputed
   * from the surface topology (area-weighted) and the interleaved vertex buffer
   * is re-uploaded.
   */
  updateSoftMesh(mesh, pos) {
    const { indices, normals, interleaved, vertexCount } = mesh;
    normals.fill(0);
    for (let t = 0; t + 2 < indices.length; t += 3) {
      const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
      const ax = pos[a], ay = pos[a + 1], az = pos[a + 2];
      const e1x = pos[b] - ax, e1y = pos[b + 1] - ay, e1z = pos[b + 2] - az;
      const e2x = pos[c] - ax, e2y = pos[c + 1] - ay, e2z = pos[c + 2] - az;
      // Cross(e1, e2): magnitude ∝ triangle area, so this area-weights normals.
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;
      normals[a] += nx; normals[a + 1] += ny; normals[a + 2] += nz;
      normals[b] += nx; normals[b + 1] += ny; normals[b + 2] += nz;
      normals[c] += nx; normals[c + 1] += ny; normals[c + 2] += nz;
    }
    for (let i = 0; i < vertexCount; i++) {
      const o = i * 3, io = i * 6;
      interleaved[io] = pos[o];
      interleaved[io + 1] = pos[o + 1];
      interleaved[io + 2] = pos[o + 2];
      const nx = normals[o], ny = normals[o + 1], nz = normals[o + 2];
      const inv = 1 / (Math.hypot(nx, ny, nz) || 1);
      interleaved[io + 3] = nx * inv;
      interleaved[io + 4] = ny * inv;
      interleaved[io + 5] = nz * inv;
    }
    this.device.queue.writeBuffer(mesh.vbuf, 0, interleaved);
  }

  /** Retint a soft mesh. */
  setSoftMeshColor(mesh, color) {
    this.device.queue.writeBuffer(
      mesh.colorBuf, 0, new Float32Array([color[0], color[1], color[2], 1.0])
    );
  }

  _createDepthTexture() {
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      size: [this.canvas.width, this.canvas.height],
      format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    // Depth view is stable between resizes — cache it instead of re-creating
    // a GPUTextureView object every frame.
    this.depthView = this.depthTexture.createView();
  }

  resize(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this._createDepthTexture();
    this._createRTTargets();
    this.engine.set_aspect(width / height);
  }

  // Re-create the command-stream view if WASM linear memory grew and detached
  // the old ArrayBuffer (briefing §3 hazard).
  _syncMemory() {
    if (this._cachedBuffer !== this.wasm.memory.buffer) {
      this._cachedBuffer = this.wasm.memory.buffer;
      this._cmdView = new Uint32Array(this._cachedBuffer);
    }
  }

  // The one boundary crossing per frame: WASM rebuilds all per-frame data,
  // then we upload it and either replay the command stream (raster) or run
  // the compute ray tracer, depending on the active mode setting.
  frame(dt) {
    const engine = this.engine;
    const device = this.device;

    engine.update(dt);
    this._syncMemory();

    const mem = this.wasm.memory.buffer;

    // Raster path only: the ray tracer's scene (the sphere BVH) is static and
    // was uploaded once in init(), so raytrace mode has no per-frame scene
    // upload at all — just the camera below.
    if (this.mode !== "raytrace") {
      const tBytes = engine.transforms_bytes();
      if (tBytes > 0) {
        device.queue.writeBuffer(this.buffers[this.transformBufferId], 0, mem, engine.transforms_ptr(), tBytes);
      }
    }
    // Materials upload only when WASM says the GPU copy could be stale (scene
    // rebuilt or models resident) — 0 bytes on the common static frame.
    const mBytes = engine.materials_upload_bytes();
    if (mBytes > 0) {
      device.queue.writeBuffer(this.buffers[this.materialBufferId], 0, mem, engine.materials_ptr(), mBytes);
    }

    // The RT camera drives both the ray tracer and the skybox background, so
    // upload it every frame regardless of mode.
    device.queue.writeBuffer(
      this.buffers[this.rtCameraBufferId],
      0,
      mem,
      engine.rt_camera_ptr(),
      engine.rt_camera_bytes()
    );

    if (this.mode === "raytrace") {
      this._frameRaytrace(mem);
    } else {
      this._frameRaster(mem);
    }
  }

  _frameRaster(mem) {
    const device = this.device;
    const engine = this.engine;

    // Fast-path uploads: view straight into WASM memory, no intermediate copy.
    device.queue.writeBuffer(this.buffers[this.cameraBufferId], 0, mem, engine.camera_ptr(), engine.camera_bytes());
    device.queue.writeBuffer(this.buffers[this.lightingBufferId], 0, mem, engine.lighting_ptr(), engine.lighting_bytes());

    // Canvas texture must be re-acquired every frame (briefing §10).
    const view = this.context.getCurrentTexture().createView();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1.0 }, loadOp: "clear", storeOp: "store" },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    });

    // 1. Lit scene. Bind the cube mesh, then replay the stream.
    pass.setVertexBuffer(0, this.buffers[this.vertexBufferId]);
    pass.setIndexBuffer(this.buffers[this.indexBufferId], "uint32");
    this._replay(pass, engine.commands_ptr());

    // 1b. Soft-body / cloth surfaces (deforming meshes), drawn as opaque
    // geometry so the skybox pass below only fills the pixels they leave empty.
    if (this.softMeshes.length && this.softMeshPipeline) {
      pass.setPipeline(this.softMeshPipeline);
      pass.setBindGroup(1, this.envBindGroup);
      for (const m of this.softMeshes) {
        if (m.hidden) continue;
        pass.setBindGroup(0, m.bindGroup);
        pass.setVertexBuffer(0, m.vbuf);
        pass.setIndexBuffer(m.ibuf, "uint32");
        pass.drawIndexed(m.indexCount);
      }
    }

    // 2. Skybox last: its z = 1 fragments pass the less-equal depth test only
    // where no geometry wrote depth, so sky shading runs once per empty pixel
    // instead of fullscreen-then-overdrawn.
    pass.setPipeline(this.skyboxPipeline);
    pass.setBindGroup(0, this.skyboxCamBindGroup);
    pass.setBindGroup(1, this.envBindGroup);
    pass.draw(3);

    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // Ray-tracing path: a compute pass writes the traced image into a
  // reduced-resolution storage texture, then a fullscreen blit upscales it.
  _frameRaytrace(mem) {
    const device = this.device;

    const encoder = device.createCommandEncoder();

    // Compute: trace the scene (RT camera already uploaded in frame()).
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(this.rtComputePipeline);
    cpass.setBindGroup(0, this.rtComputeBindGroup);
    cpass.dispatchWorkgroups(Math.ceil(this.rtW / 8), Math.ceil(this.rtH / 8));
    cpass.end();

    // Blit: upscale to the swapchain (re-acquired each frame).
    const view = this.context.getCurrentTexture().createView();
    const rpass = encoder.beginRenderPass({
      colorAttachments: [
        { view, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    rpass.setPipeline(this.blitPipeline);
    rpass.setBindGroup(0, this.blitBindGroup);
    rpass.draw(3);
    rpass.end();

    device.queue.submit([encoder.finish()]);
  }

  // Tight, monomorphic replay loop: one switch on cmd_type, no allocation,
  // reads through the persistent Uint32Array (briefing §6). Handles mesh
  // switching (glTF models) and the general indexed-draw parameters.
  _replay(pass, commandsBytePtr) {
    const view = this._cmdView;
    let p = commandsBytePtr >>> 2; // byte offset -> u32 index

    let lastPipeline = -1;
    let lastBindGroup = -1;

    for (;;) {
      const type = view[p];
      if (type === CMD_END) break;

      if (type === CMD_DRAW_INDEXED) {
        const pipeId = view[p + 1];
        const bgId = view[p + 2];
        const firstIndex = view[p + 3];
        const indexCount = view[p + 4];
        const instanceCount = view[p + 5];
        const materialBg = view[p + 6]; // group-2 material, or 0xffffffff = none
        const firstInstance = view[p + 7];

        if (pipeId !== lastPipeline) {
          pass.setPipeline(this.pipelines[pipeId]);
          // The lit pipelines all sample the sky cubemap at group(1); rebind it
          // here since a pipeline switch can invalidate higher bind slots.
          pass.setBindGroup(1, this.envBindGroup);
          lastPipeline = pipeId;
          lastBindGroup = -1; // group(0) may have been invalidated too; re-set it
        }
        if (bgId !== lastBindGroup) {
          pass.setBindGroup(0, this.bindGroups[bgId]);
          lastBindGroup = bgId;
        }
        // Per-draw material (group 2) — set AFTER the pipeline so it isn't
        // invalidated by the switch. Textured glTF models only.
        if (materialBg !== 0xffffffff) {
          pass.setBindGroup(2, this.bindGroups[materialBg]);
        }
        pass.drawIndexed(indexCount, instanceCount, firstIndex, 0, firstInstance);
      } else if (type === CMD_SET_MESH) {
        const vId = view[p + 1];
        const iId = view[p + 2];
        const fmt = view[p + 3]; // 0 = uint32, 1 = uint16
        pass.setVertexBuffer(0, this.buffers[vId]);
        pass.setIndexBuffer(this.buffers[iId], fmt === 1 ? "uint16" : "uint32");
      }

      p += CMD_STRIDE;
    }
  }
}
