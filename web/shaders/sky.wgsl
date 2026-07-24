// Squirrel Engine — procedural sky, rendered *into* a cubemap.
//
// The skybox is built with a real cubemap render: the six faces of a cube
// texture are each drawn once by this shader (one fullscreen triangle per
// face). JS supplies, per face, the world-space basis that maps a fragment's
// screen position to the exact direction that texel is sampled at, so the
// stored image is consistent with later `textureSample(cube, dir)` lookups —
// the same cubemap then serves as the skybox background, the raster ambient /
// reflection source, and the ray tracer's sky.

struct SkyFace {
    forward : vec4<f32>,   // face-center direction
    right   : vec4<f32>,   // maps screen +x
    up      : vec4<f32>,   // maps screen +y
    sun_dir : vec4<f32>,   // xyz dir toward sun, w = sun angular size
    sun_col : vec4<f32>,   // rgb, w = intensity
    top     : vec4<f32>,   // zenith color
    horizon : vec4<f32>,   // horizon color
    ground  : vec4<f32>,   // below-horizon color
};

@group(0) @binding(0) var<uniform> face : SkyFace;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,   // [-1, 1] across the face
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    // Oversized triangle covering the [-1, 1] face square.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let xy = corners[vi];
    var out : VSOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = xy;
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    // The sampling direction for this texel (matches the cube-face convention
    // JS encoded into the basis).
    let dir = normalize(face.forward.xyz + in.uv.x * face.right.xyz + in.uv.y * face.up.xyz);

    // Vertical gradient: ground below the horizon, sky above.
    let t = dir.y;
    var col : vec3<f32>;
    if (t < 0.0) {
        col = mix(face.horizon.rgb, face.ground.rgb, clamp(-t * 2.0, 0.0, 1.0));
    } else {
        col = mix(face.horizon.rgb, face.top.rgb, pow(clamp(t, 0.0, 1.0), 0.5));
    }

    // Sun: a bright disk plus a broad warm glow around it.
    let sd = max(dot(dir, normalize(face.sun_dir.xyz)), 0.0);
    let disk = smoothstep(1.0 - face.sun_dir.w, 1.0 - face.sun_dir.w * 0.25, sd);
    let glow = pow(sd, 8.0) * 0.35;
    col += face.sun_col.rgb * face.sun_col.w * (disk + glow);

    return vec4<f32>(col, 1.0);
}
