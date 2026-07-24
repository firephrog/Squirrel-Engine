// Squirrel Engine — soft-mesh shader (lit, two-sided).
//
// Draws a deforming soft body / cloth as a real triangle surface. Unlike the
// model shader, vertices arrive already in world space (no per-instance model
// matrix — the geometry itself moves each frame), the albedo is a single flat
// color uniform (no texture), and shading is two-sided: the surface normal is
// flipped to face the viewer so a thin sheet lights correctly from both sides.
//
// It reuses the engine's own camera + lighting uniform buffers and the sky
// cubemap, so cloth is lit by exactly the same sun / sky as the rest of the
// scene.
//
//   @group(0) -> camera + lighting + base color
//   @group(1) -> environment (skybox cubemap + sampler)

struct Camera {
    view_proj : mat4x4<f32>,
    params    : vec4<f32>,
};
struct PointLight {
    position : vec4<f32>,   // xyz, w = range
    color    : vec4<f32>,   // rgb, w = intensity
};
struct Lighting {
    sun_dir    : vec4<f32>,
    sun_color  : vec4<f32>,
    camera_pos : vec4<f32>,
    env_params : vec4<f32>,
    points     : array<PointLight, 4>,
};

@group(0) @binding(0) var<uniform> camera : Camera;
@group(0) @binding(1) var<uniform> light  : Lighting;
@group(0) @binding(2) var<uniform> base_color : vec4<f32>;

@group(1) @binding(0) var env_tex  : texture_cube<f32>;
@group(1) @binding(1) var env_samp : sampler;

struct VSOut {
    @builtin(position) clip   : vec4<f32>,
    @location(0)       normal : vec3<f32>,
    @location(1)       world  : vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
) -> VSOut {
    var out : VSOut;
    out.clip = camera.view_proj * vec4<f32>(position, 1.0);
    out.normal = normal;
    out.world = position;
    return out;
}

fn tonemap(c : vec3<f32>, exposure : f32) -> vec3<f32> {
    let x = c * exposure;
    let mapped = x / (x + vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    let view_dir = normalize(light.camera_pos.xyz - in.world);
    // Two-sided: face the normal toward the camera so both sides shade properly.
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    let albedo = base_color.rgb;

    // Ambient: flat floor + skybox irradiance along the normal.
    let env_ambient = textureSample(env_tex, env_samp, n).rgb;
    var color = albedo * (light.sun_color.w + light.env_params.x * env_ambient);

    // Directional sun: Lambert diffuse + Blinn-Phong specular.
    let l = light.sun_dir.xyz;
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + view_dir);
    let spec = pow(max(dot(n, h), 0.0), 32.0);
    color += albedo * light.sun_color.rgb * (light.sun_dir.w * ndl);
    color += light.sun_color.rgb * (light.sun_dir.w * spec * ndl * 0.4);

    // Point lights with smooth range-based attenuation.
    let count = u32(light.camera_pos.w);
    for (var i = 0u; i < count; i = i + 1u) {
        let p = light.points[i];
        let to_light = p.position.xyz - in.world;
        let dist = length(to_light);
        let ld = to_light / max(dist, 1e-4);
        let atten = clamp(1.0 - dist / max(p.position.w, 1e-4), 0.0, 1.0);
        let falloff = atten * atten;
        let pndl = max(dot(n, ld), 0.0);
        color += albedo * p.color.rgb * (p.color.w * pndl * falloff);
    }

    return vec4<f32>(tonemap(color, light.env_params.z), 1.0);
}
