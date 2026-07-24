// Squirrel Engine — skybox background pass.
//
// Draws the sky cubemap behind the rasterized scene. A single fullscreen
// triangle reconstructs the world-space view ray for each pixel from the same
// camera basis the ray tracer uses (eye + pre-scaled right/up/forward), then
// samples the cubemap. Rendered LAST in the raster pass at z = 1 with depth
// writes off and a less-equal test, so only pixels the geometry left empty
// (depth still at the 1.0 clear) run this shader — no sky overdraw.

struct RTCamera {
    eye     : vec4<f32>,   // xyz eye (w unused here)
    right   : vec4<f32>,   // xyz basis, pre-scaled by aspect * tan(fov/2)
    up      : vec4<f32>,   // xyz basis, pre-scaled by tan(fov/2)
    forward : vec4<f32>,   // xyz basis (w = time)
};

@group(0) @binding(0) var<uniform> cam : RTCamera;
@group(1) @binding(0) var env_tex  : texture_cube<f32>;
@group(1) @binding(1) var env_samp : sampler;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       ndc : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let xy = corners[vi];
    var out : VSOut;
    // z = 1 keeps the skybox at the far plane if depth testing is enabled.
    out.pos = vec4<f32>(xy, 1.0, 1.0);
    out.ndc = xy;
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    let rd = normalize(cam.forward.xyz + in.ndc.x * cam.right.xyz + in.ndc.y * cam.up.xyz);
    let col = textureSample(env_tex, env_samp, rd).rgb;
    // Match the lit pass's gamma so sky and geometry sit in the same space.
    return vec4<f32>(pow(col, vec3<f32>(1.0 / 2.2)), 1.0);
}
