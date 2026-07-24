// Squirrel Engine — compute ray tracer (cone-traced, filtered, anti-aliased).
//
// WebGPU has no hardware ray-tracing acceleration structures, so this is a
// software tracer that runs entirely on the GPU in a compute pass (the
// briefing's "compute shaders from day one" advantage, §5).
//
// The cube field's spheres never move (only their rotation animates), so
// WASM builds a static BVH over them once at scene init — the same
// stackless, escape-linked layout as the imported-model triangle BVH below —
// and every ray walks that tree instead of brute-forcing all instances.
// Ground plane, hard shadows (any-hit early exit), and one reflection bounce
// round out the scene.
//
// Anti-aliasing strategy (fixes the distant checkerboard moiré):
//   1. CONE TRACING     — each ray carries a footprint radius that grows with
//                         distance (a cone, not an infinitely thin line).
//   2. TEXTURE MIP/FILTER — the procedural checkerboard is *analytically*
//                         box-filtered over that footprint (the procedural
//                         equivalent of mipmapping), so it fades to flat gray
//                         in the distance instead of shimmering.
//   3. SUPERSAMPLING    — AA_SAMPLES jittered rays per pixel average out the
//                         silhouette edges of the spheres.

struct RTCamera {
    eye     : vec4<f32>,   // xyz eye, w = sphere BVH node count
    right   : vec4<f32>,   // xyz basis (pre-scaled by aspect * tan(fov/2))
    up      : vec4<f32>,   // xyz basis (pre-scaled by tan(fov/2))
    forward : vec4<f32>,   // xyz basis, w = time
};

// One imported-model triangle in world space. Positions carry the u texcoord in
// their w lane; `uv` carries the three v texcoords; color.w flags whether this
// triangle samples the albedo texture (1) or shows its flat base color (0).
struct Tri {
    v0    : vec4<f32>,   // xyz position, w = u0
    v1    : vec4<f32>,   // xyz position, w = u1
    v2    : vec4<f32>,   // xyz position, w = u2
    color : vec4<f32>,   // rgb base color, w = textured flag
    uv    : vec4<f32>,   // x = v0.v, y = v1.v, z = v2.v
};
// One BVH node (matches buildBVH in renderer.js). Depth-first layout: a node's
// first child is always the next node, and `escape` is where to jump when this
// node's box is missed (or after a leaf's triangles are tested) — so traversal
// is a single stackless loop.
struct BvhNode {
    bmin : vec4<f32>,   // xyz aabb min, w = escape index (-1 = stop)
    bmax : vec4<f32>,   // xyz aabb max, w = leaf triangle count (0 = internal)
    tri  : vec4<f32>,   // x = first triangle index (leaf only)
};
// Bounding sphere + triangle/node counts of the imported model, so rays that
// miss the model skip the whole BVH walk.
struct RTInfo {
    bounds : vec4<f32>,   // xyz center, w radius
    tri_info : vec4<f32>,   // x = triangle count, y = BVH node count
};

@group(0) @binding(0) var<uniform> cam : RTCamera;
// Static cube-field scene: xyz = sphere center, w = radius, reordered by
// WASM to match the sphere BVH's leaf layout. Uploaded once, not per frame.
@group(0) @binding(1) var<storage, read> spheres : array<vec4<f32>>;
@group(0) @binding(2) var out_tex : texture_storage_2d<rgba8unorm, write>;
// Shared sky cubemap (same one the skybox pass and raster ambient sample), so
// both render modes see identical sky, ambient, and reflections.
@group(0) @binding(3) var env_tex  : texture_cube<f32>;
@group(0) @binding(4) var env_samp : sampler;
// Imported glTF model, traced as world-space triangles indexed by a BVH.
@group(0) @binding(5) var<storage, read> tris : array<Tri>;
@group(0) @binding(6) var<uniform> info : RTInfo;
// The model's single albedo texture (see loadModel) + its sampler.
@group(0) @binding(7) var albedo_tex  : texture_2d<f32>;
@group(0) @binding(8) var albedo_samp : sampler;
@group(0) @binding(9) var<storage, read> bvh : array<BvhNode>;
// BVH over the cube field's spheres (see bvh.rs), same node layout as `bvh`.
@group(0) @binding(10) var<storage, read> sphere_bvh : array<BvhNode>;

// Guard against a malformed BVH looping forever (worst-case visited nodes).
const MAX_BVH_STEPS : u32 = 8192u;
// Rays per pixel (must match the jitter table in cs_main). 4 = 2x2 grid.
const AA_SAMPLES  : u32 = 4u;
const GROUND_Y    : f32 = -15.0;
const EPS         : f32 = 0.02;
const CHECKER_SCALE : f32 = 0.2; // checker cells per world unit

// Pre-normalized light direction (normalize is not const-evaluable in WGSL, so
// this is baked to avoid a per-shade normalize). Source dir: (0.5, 0.9, 0.4).
const LIGHT_DIR = vec3<f32>(0.45267, 0.81481, 0.36214);

struct Hit {
    t     : f32,
    n     : vec3<f32>,
    color : vec3<f32>,
    refl  : f32,
    kind  : i32,   // 0 = miss, 1 = sphere, 2 = ground, 3 = model triangle
};

// Möller–Trumbore ray/triangle intersection. Returns vec3(t, u, v) where u, v
// are the barycentric weights of vertices b and c (so vertex a's weight is
// 1 - u - v); t < 0 signals a miss.
fn tri_hit(ro : vec3<f32>, rd : vec3<f32>, a : vec3<f32>, b : vec3<f32>, c : vec3<f32>) -> vec3<f32> {
    let miss = vec3<f32>(-1.0, 0.0, 0.0);
    let e1 = b - a;
    let e2 = c - a;
    let pv = cross(rd, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-7) { return miss; }
    let inv = 1.0 / det;
    let tv = ro - a;
    let u = dot(tv, pv) * inv;
    if (u < 0.0 || u > 1.0) { return miss; }
    let qv = cross(tv, e1);
    let v = dot(rd, qv) * inv;
    if (v < 0.0 || u + v > 1.0) { return miss; }
    return vec3<f32>(dot(e2, qv) * inv, u, v);
}

// Ray/AABB slab test with a precomputed 1/rd. Returns true if the box is hit
// within (0, tmax]. inv_rd may contain +/-inf for axis-aligned rays — the
// min/max ordering handles that correctly.
fn aabb_hit(ro : vec3<f32>, inv_rd : vec3<f32>, bmin : vec3<f32>, bmax : vec3<f32>, tmax : f32) -> bool {
    let t0 = (bmin - ro) * inv_rd;
    let t1 = (bmax - ro) * inv_rd;
    let tsmall = min(t0, t1);
    let tbig = max(t0, t1);
    let tnear = max(max(tsmall.x, tsmall.y), tsmall.z);
    let tfar = min(min(tbig.x, tbig.y), tbig.z);
    return tfar >= max(tnear, 0.0) && tnear < tmax;
}

fn hash_color(i : u32) -> vec3<f32> {
    let h = f32(i) * 0.61803398875;
    let r = fract(h);
    let g = fract(h * 1.7 + 0.33);
    let b = fract(h * 2.3 + 0.66);
    return vec3<f32>(0.35 + 0.6 * r, 0.35 + 0.6 * g, 0.35 + 0.6 * b);
}

// Sample the shared sky cubemap. `textureSampleLevel` (no implicit derivatives)
// is required in a compute shader.
fn sky(rd : vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(env_tex, env_samp, rd, 0.0).rgb;
}

// Analytically box-filtered checkerboard (Inigo Quilez's filtered checker).
// `p` is the point in checker space; `w` is the filter footprint (full width)
// in the same space. As `w` grows with distance the pattern integrates to 0.5
// (flat gray) — exactly what a mip chain would give, with no moiré.
fn filtered_checker(p : vec2<f32>, w : f32) -> f32 {
    let ww = vec2<f32>(w, w) + vec2<f32>(1e-3, 1e-3);
    let a = abs(fract((p - 0.5 * ww) * 0.5) - vec2<f32>(0.5, 0.5));
    let b = abs(fract((p + 0.5 * ww) * 0.5) - vec2<f32>(0.5, 0.5));
    let i = 2.0 * (a - b) / ww;
    return 0.5 - 0.5 * i.x * i.y;
}

// Ground albedo, filtered over the ray-cone footprint at the hit point.
fn ground_color(p : vec3<f32>, footprint : f32) -> vec3<f32> {
    let c = filtered_checker(p.xz * CHECKER_SCALE, footprint * CHECKER_SCALE);
    return mix(vec3<f32>(0.10, 0.10, 0.12), vec3<f32>(0.85, 0.85, 0.90), c);
}

// Nearest intersection of ray (ro, rd) with the scene. rd must be normalized.
// Ground color is left unset here — it depends on the ray cone and is filtered
// by the caller via ground_color().
fn intersect(ro : vec3<f32>, rd : vec3<f32>) -> Hit {
    var best : Hit;
    best.t = 1e30;
    best.kind = 0;
    best.refl = 0.0;
    best.n = vec3<f32>(0.0, 1.0, 0.0);
    best.color = vec3<f32>(0.0);

    // Cube-field spheres, indexed by their own stackless BVH (built once in
    // WASM — see bvh.rs). Each ray tests only the boxes it pierces instead of
    // every sphere in the scene.
    let sphere_node_count = u32(cam.eye.w);
    if (sphere_node_count > 0u) {
        let inv_rd = 1.0 / rd;
        var i : i32 = 0;
        var steps : u32 = 0u;
        loop {
            if (i < 0 || u32(i) >= sphere_node_count || steps >= MAX_BVH_STEPS) { break; }
            steps = steps + 1u;
            let node = sphere_bvh[i];
            if (aabb_hit(ro, inv_rd, node.bmin.xyz, node.bmax.xyz, best.t)) {
                let leaf_count = u32(node.bmax.w);
                if (leaf_count > 0u) {
                    let first = u32(node.tri.x);
                    for (var k = 0u; k < leaf_count; k = k + 1u) {
                        let s = spheres[first + k];
                        let oc = ro - s.xyz;
                        let b = dot(oc, rd);
                        let c = dot(oc, oc) - s.w * s.w;
                        let disc = b * b - c;
                        if (disc > 0.0) {
                            let sq = sqrt(disc);
                            var t = -b - sq;
                            if (t <= 0.001) { t = -b + sq; }
                            if (t > 0.001 && t < best.t) {
                                best.t = t;
                                let p = ro + rd * t;
                                best.n = normalize(p - s.xyz);
                                best.color = hash_color(first + k);
                                best.refl = 0.35;
                                best.kind = 1;
                            }
                        }
                    }
                }
                i = i + 1; // descend to first child (leaf: escape == i+1)
            } else {
                i = i32(node.bmin.w); // escape past this subtree
            }
        }
    }

    // Imported model triangles, indexed by a stackless BVH. Each ray tests only
    // the boxes it pierces, so a heavy mesh costs ~log(triangles) instead of a
    // full per-pixel triangle loop — the fix for the glTF-in-RT slowdown.
    let node_count = u32(info.tri_info.y);
    if (node_count > 0u) {
        let inv_rd = 1.0 / rd;
        var i : i32 = 0;
        var steps : u32 = 0u;
        loop {
            if (i < 0 || u32(i) >= node_count || steps >= MAX_BVH_STEPS) { break; }
            steps = steps + 1u;
            let node = bvh[i];
            if (aabb_hit(ro, inv_rd, node.bmin.xyz, node.bmax.xyz, best.t)) {
                let leaf_count = u32(node.bmax.w);
                if (leaf_count > 0u) {
                    let first = u32(node.tri.x);
                    for (var k = 0u; k < leaf_count; k = k + 1u) {
                        let tr = tris[first + k];
                        let h = tri_hit(ro, rd, tr.v0.xyz, tr.v1.xyz, tr.v2.xyz);
                        if (h.x > 0.001 && h.x < best.t) {
                            best.t = h.x;
                            var nn = normalize(cross(tr.v1.xyz - tr.v0.xyz, tr.v2.xyz - tr.v0.xyz));
                            if (dot(nn, rd) > 0.0) { nn = -nn; } // two-sided
                            best.n = nn;
                            if (tr.color.w > 0.5) {
                                // Barycentric-interpolate the texcoord, then
                                // sample the model albedo (no derivatives in a
                                // compute pass, so use textureSampleLevel).
                                let bw = 1.0 - h.y - h.z;
                                let uu = bw * tr.v0.w + h.y * tr.v1.w + h.z * tr.v2.w;
                                let vv = bw * tr.uv.x + h.y * tr.uv.y + h.z * tr.uv.z;
                                let tex = textureSampleLevel(albedo_tex, albedo_samp, vec2<f32>(uu, vv), 0.0).rgb;
                                best.color = tex * tr.color.xyz;
                            } else {
                                best.color = tr.color.xyz;
                            }
                            best.refl = 0.05;
                            best.kind = 3;
                        }
                    }
                }
                i = i + 1; // descend to first child (leaf: escape == i+1)
            } else {
                i = i32(node.bmin.w); // escape past this subtree
            }
        }
    }

    // Infinite ground plane.
    if (rd.y < -1e-4) {
        let tg = (GROUND_Y - ro.y) / rd.y;
        if (tg > 0.001 && tg < best.t) {
            best.t = tg;
            best.n = vec3<f32>(0.0, 1.0, 0.0);
            best.refl = 0.1;
            best.kind = 2;
        }
    }
    return best;
}

// Any occluder between p and the (infinitely distant) light. Unlike
// intersect() this exits on the FIRST hit found anywhere in the BVH walk — a
// shadow ray doesn't care which occluder is nearest. The ground can never
// occlude an upward light ray, so only the sphere BVH is walked.
fn in_shadow(p : vec3<f32>, ldir : vec3<f32>) -> bool {
    let sphere_node_count = u32(cam.eye.w);
    if (sphere_node_count == 0u) { return false; }
    let inv_rd = 1.0 / ldir;
    var i : i32 = 0;
    var steps : u32 = 0u;
    loop {
        if (i < 0 || u32(i) >= sphere_node_count || steps >= MAX_BVH_STEPS) { break; }
        steps = steps + 1u;
        let node = sphere_bvh[i];
        if (aabb_hit(p, inv_rd, node.bmin.xyz, node.bmax.xyz, 1e30)) {
            let leaf_count = u32(node.bmax.w);
            if (leaf_count > 0u) {
                let first = u32(node.tri.x);
                for (var k = 0u; k < leaf_count; k = k + 1u) {
                    let s = spheres[first + k];
                    let oc = p - s.xyz;
                    let b = dot(oc, ldir);
                    let c = dot(oc, oc) - s.w * s.w;
                    let disc = b * b - c;
                    if (disc > 0.0) {
                        let sq = sqrt(disc);
                        if (-b - sq > 0.001 || -b + sq > 0.001) { return true; }
                    }
                }
            }
            i = i + 1;
        } else {
            i = i32(node.bmin.w);
        }
    }
    return false;
}

// Local Lambert shading with a hard shadow ray.
fn shade_local(p : vec3<f32>, n : vec3<f32>, base : vec3<f32>) -> vec3<f32> {
    var s = 1.0;
    if (in_shadow(p + n * EPS, LIGHT_DIR)) { s = 0.25; }
    let diff = max(dot(n, LIGHT_DIR), 0.0) * s;
    return base * (0.2 + 0.9 * diff);
}

// Cone footprint (world-space width) where a ray of the given angular `spread`
// meets a plane with vertical component `rdy`. Grazing angles (small |rdy|)
// stretch the footprint, which is precisely what filters the distance to gray.
fn plane_footprint(spread : f32, t : f32, rdy : f32) -> f32 {
    return (spread * t) / max(abs(rdy), 0.05);
}

// Full shade of a hit, given the incoming ray and its cone spread. Handles the
// ground's filtered albedo; the caller adds any reflection bounce.
fn shade_hit(h : Hit, ro : vec3<f32>, rd : vec3<f32>, spread : f32) -> vec3<f32> {
    let p = ro + rd * h.t;
    var base : vec3<f32>;
    if (h.kind == 2) {
        base = ground_color(p, plane_footprint(spread, h.t, rd.y));
    } else {
        base = h.color;
    }
    return shade_local(p, h.n, base);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }

    let res = vec2<f32>(f32(dims.x), f32(dims.y));
    let ro = cam.eye.xyz;

    // Cone half-angle spanning ~one pixel vertically (tan(fov/2) = |cam.up|).
    let spread = 2.0 * length(cam.up.xyz) / res.y;

    // 2x2 rotated-grid jitter for supersampling.
    var jitter = array<vec2<f32>, 4>(
        vec2<f32>(-0.25, -0.25),
        vec2<f32>( 0.25, -0.25),
        vec2<f32>(-0.25,  0.25),
        vec2<f32>( 0.25,  0.25)
    );

    var acc = vec3<f32>(0.0);
    for (var s = 0u; s < AA_SAMPLES; s = s + 1u) {
        let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5 + jitter[s]) / res;
        let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let rd = normalize(cam.forward.xyz + ndc.x * cam.right.xyz + ndc.y * cam.up.xyz);

        var color : vec3<f32>;
        let h = intersect(ro, rd);
        if (h.kind == 0) {
            color = sky(rd);
        } else {
            color = shade_hit(h, ro, rd, spread);

            // One reflection bounce — the cone continues, so its footprint at
            // the reflected hit is the accumulated spread over both segments.
            if (h.refl > 0.0) {
                let p = ro + rd * h.t;
                let r = reflect(rd, h.n);
                let origin = p + h.n * EPS;
                let h2 = intersect(origin, r);
                var rc : vec3<f32>;
                if (h2.kind == 0) {
                    rc = sky(r);
                } else if (h2.kind == 2) {
                    let p2 = origin + r * h2.t;
                    let fp = plane_footprint(spread, h.t + h2.t, r.y);
                    rc = shade_local(p2, h2.n, ground_color(p2, fp));
                } else {
                    rc = shade_local(origin + r * h2.t, h2.n, h2.color);
                }
                color = mix(color, rc, h.refl);
            }
        }
        acc = acc + color;
    }

    // Reinhard tonemap + gamma, matching the raster path so the shared sky
    // cubemap (linear radiance) reads the same in both modes.
    var result = acc / f32(AA_SAMPLES);
    result = result / (result + vec3<f32>(1.0));
    result = pow(result, vec3<f32>(1.0 / 2.2));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(result, 1.0));
}
