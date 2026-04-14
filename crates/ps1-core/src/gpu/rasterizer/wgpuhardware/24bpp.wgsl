const VERTICES: array<vec4f, 4> = array<vec4f, 4>(
    vec4f(-1.0, -1.0, 0.0, 1.0),
    vec4f(1.0, -1.0, 0.0, 1.0),
    vec4f(-1.0, 1.0, 0.0, 1.0),
    vec4f(1.0, 1.0, 0.0, 1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    return VERTICES[vertex_index];
}

@group(0) @binding(0)
var native_vram: texture_storage_2d<r32uint, read>;

struct Render24bppArgs {
    frame_position: vec2u,
    display_start: vec2u,
    display_offset: vec2u,
    display_end: vec2u,
}

var<immediate> args: Render24bppArgs;

@fragment
fn fs_main(@builtin(position) in_position: vec4f) -> @location(0) vec4f {
    let position = vec2u(in_position.xy);
    if position.x < args.display_start.x
        || position.x >= args.display_end.x
        || position.y < args.display_start.y
        || position.y >= args.display_end.y
    {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let tex_y = (args.frame_position.y + position.y + args.display_offset.y - args.display_start.y) & 511;
    let tex_x_offset = position.x + args.display_offset.x - args.display_start.x;
    let tex_x = (args.frame_position.x + 3 * tex_x_offset / 2) & 1023;

    let first_texel = textureLoad(native_vram, vec2u(tex_x, tex_y)).r;
    let second_texel = textureLoad(native_vram, vec2u((tex_x + 1) & 1023, tex_y)).r;

    let color = select(
        vec3u((first_texel >> 8) & 255, second_texel & 255, (second_texel >> 8) & 255),
        vec3u(first_texel & 255, (first_texel >> 8) & 255, second_texel & 255),
        (tex_x_offset & 1) == 0,
    );

    return vec4f(vec3f(color) / 255.0, 1.0);
}