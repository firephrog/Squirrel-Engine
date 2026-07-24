// Squirrel Engine — main forward shader (lit).
//
// Bind group convention (briefing §4):
//   @group(0) -> per-frame   (camera + instance transforms + lighting + materials)
//   @group(1) -> environment (skybox cubemap + sampler)
//   @group(2) -> per-draw    (reserved)
//
// Transforms use the "storage buffer + instance indexing" model: every
// object's model matrix lives in one storage buffer, selected by
// @builtin(instance_index). A parallel material storage buffer, indexed the
// same way, carries a per-instance base color (alpha < 0 => tint by normal,
// used by the procedural cube field; alpha >= 0 => a real material color, used
// by imported glTF primitives).

// Must match CameraUniform on the Rust side: 80 bytes.
struct Camera {
    view_proj : mat4x4<f32>,
    params    : vec4<f32>,   // x = time
};

// Must match the Lighting struct on the Rust side: 192 bytes.
struct PointLight {
    // xyz position, w = range (falloff distance)
    position : vec4<f32>,
    // rgb color, w = intensity
    color    : vec4<f32>,
};
struct Lighting {
    sun_dir    : vec4<f32>,   // xyz dir *toward* the sun, w = intensity
    sun_color  : vec4<f32>,   // rgb, w = flat ambient floor
    camera_pos : vec4<f32>,   // xyz eye, w = active point-light count
    env_params : vec4<f32>,   // x = env ambient, y = env reflect, z = exposure
    points     : array<PointLight, 4>,
};

@group(0) @binding(0) var<uniform> camera   : Camera;
@group(0) @binding(1) var<storage, read> models    : array<mat4x4<f32>>;
@group(0) @binding(2) var<uniform> light    : Lighting;
@group(0) @binding(3) var<storage, read> materials : array<vec4<f32>>;

@group(1) @binding(0) var env_tex  : texture_cube<f32>;
@group(1) @binding(1) var env_samp : sampler;

struct VSOut {
    @builtin(position)      clip     : vec4<f32>,
    @location(0)            normal   : vec3<f32>,
    @location(1)            world    : vec3<f32>,
    @location(2) @interpolate(flat) mat_id : u32,
};

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @builtin(instance_index) ii : u32,
) -> VSOut {
    let model = models[ii];
    let world = model * vec4<f32>(position, 1.0);

    var out : VSOut;
    out.clip   = camera.view_proj * world;
    // Model matrices here are TRS with uniform scale, so the upper-left block
    // is a rotation * scale — safe to transform the normal with directly.
    out.normal = (model * vec4<f32>(normal, 0.0)).xyz;
    out.world  = world.xyz;
    out.mat_id = ii;
    return out;
}

// Reinhard-ish tonemap + gamma so bright specular / multi-light sums don't clip.
fn tonemap(c : vec3<f32>, exposure : f32) -> vec3<f32> {
    let x = c * exposure;
    let mapped = x / (x + vec3<f32>(1.0));
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let view_dir = normalize(light.camera_pos.xyz - in.world);

    // Albedo: real material color, or the cube field's normal-derived palette
    // when the material alpha is negative.
    let mat = materials[in.mat_id];
    var albedo : vec3<f32>;
    if (mat.a < 0.0) {
        albedo = 0.5 + 0.5 * n;
    } else {
        albedo = mat.rgb;
    }

    // Ambient: a flat floor plus environment irradiance sampled from the
    // skybox in the surface-normal direction (a cheap image-based ambient).
    let env_ambient = textureSample(env_tex, env_samp, n).rgb;
    let ambient = albedo * (light.sun_color.w + light.env_params.x * env_ambient);

    var color = ambient;

    // Directional sun: Lambert diffuse + Blinn-Phong specular.
    // sun_dir arrives normalized from the WASM side (set_sun normalizes).
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

    // Environment reflection: sample the skybox along the reflected view ray.
    let r = reflect(-view_dir, n);
    let env_reflect = textureSample(env_tex, env_samp, r).rgb;
    color = mix(color, env_reflect, light.env_params.y);

    return vec4<f32>(tonemap(color, light.env_params.z), 1.0);
}
