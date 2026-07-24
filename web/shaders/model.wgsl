// Squirrel Engine — textured model shader (lit).
//
// The forward shader for imported glTF primitives. Same lighting model as the
// cube shader (main.wgsl), but the vertex layout carries UVs and the fragment
// samples a per-model albedo texture (group 2), modulated by the per-instance
// baseColorFactor from the shared material buffer.
//
//   @group(0) -> per-frame   (camera + transforms + lighting + materials)
//   @group(1) -> environment (skybox cubemap + sampler)
//   @group(2) -> material    (albedo texture + sampler)

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

@group(0) @binding(0) var<uniform> camera   : Camera;
@group(0) @binding(1) var<storage, read> models    : array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> light    : Lighting;
@group(0) @binding(3) var<storage, read> materials : array<vec4<f32>>;

@group(1) @binding(0) var env_tex  : texture_cube<f32>;
@group(1) @binding(1) var env_samp : sampler;

@group(2) @binding(0) var albedo_tex  : texture_2d<f32>;
@group(2) @binding(1) var albedo_samp : sampler;

struct VSOut {
    @builtin(position)      clip   : vec4<f32>,
    @location(0)            normal : vec3<f32>,
    @location(1)            world  : vec3<f32>,
    @location(2)            uv     : vec2<f32>,
    @location(3) @interpolate(flat) mat_id : u32,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) uv       : vec2<f32>,
    @builtin(instance_index) ii : u32,
) -> VSOut {
    let model = models[ii];
    let world = model * vec4<f32>(position, 1.0);

    var out : VSOut;
    out.clip   = camera.view_proj * world;
    out.normal = (model * vec4<f32>(normal, 0.0)).xyz;
    out.world  = world.xyz;
    out.uv     = uv;
    out.mat_id = ii;
    return out;
}

fn tonemap(c : vec3<f32>, exposure : f32) -> vec3<f32> {
    let x = c * exposure;
    let mapped = x / (x + vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let view_dir = normalize(light.camera_pos.xyz - in.world);

    // Albedo = sampled texture * per-instance baseColorFactor. A model with no
    // texture binds a 1x1 white texture, so the factor shows through unchanged.
    let tex = textureSample(albedo_tex, albedo_samp, in.uv).rgb;
    let albedo = tex * materials[in.mat_id].rgb;

    // Ambient: flat floor + skybox irradiance along the normal.
    let env_ambient = textureSample(env_tex, env_samp, n).rgb;
    var color = albedo * (light.sun_color.w + light.env_params.x * env_ambient);

    // Directional sun: Lambert diffuse + Blinn-Phong specular.
    let l = light.sun_dir.xyz;
    let ndl = max(dot(n, l), 0.0);
    let h = normalize(l + view_dir);
    let spec = pow(max(dot(n, h), 0.0), 48.0);
    color += albedo * light.sun_color.rgb * (light.sun_dir.w * ndl);
    color += light.sun_color.rgb * (light.sun_dir.w * spec * ndl * 0.5);

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
        let ph = normalize(ld + view_dir);
        let pspec = pow(max(dot(n, ph), 0.0), 48.0);
        color += albedo * p.color.rgb * (p.color.w * pndl * falloff);
        color += p.color.rgb * (p.color.w * pspec * pndl * falloff * 0.5);
    }

    // Environment reflection along the view-reflection ray.
    let r = reflect(-view_dir, n);
    let env_reflect = textureSample(env_tex, env_samp, r).rgb;
    color = mix(color, env_reflect, light.env_params.y * 0.5);

    return vec4<f32>(tonemap(color, light.env_params.z), 1.0);
}
