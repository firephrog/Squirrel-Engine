// Squirrel Engine — fullscreen blit.
//
// Samples the ray tracer's storage-texture output (rendered at a reduced
// internal resolution) and upscales it to the swapchain with linear filtering.

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var tex  : texture_2d<f32>;

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VSOut {
    // Single oversized triangle covering the screen.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    let xy = corners[vi];
    var out : VSOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    return textureSample(tex, samp, in.uv);
}
