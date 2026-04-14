// Texture sampling functions require the following bindings:
//   native_vram: texture_storage_2d<r32uint, read>
//
//   scaled_vram_copy: texture_storage_2d<rgba8unorm, read>
//
//   draw_settings: DrawSettings

enable dual_source_blending;

struct DrawSettings {
    force_mask_bit: u32,
    resolution_scale: u32,
    high_color: u32,
    dithering: u32,
    high_res_dithering: u32,
    perspective_texture_mapping: u32,
}

struct UntexturedVertex {
    @location(0) position: vec3f,
    @location(1) color: vec3u,
    @location(2) ditherable: u32,
}

struct UntexturedVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) ditherable: u32,
}

fn vram_position_to_vertex(position: vec3f) -> vec4f {
    var x = (position.x - 512.0) / 512.0;
    var y = -(position.y - 256.0) / 256.0;

    if draw_settings.resolution_scale == 1 {
        x += 0.5 / 512.0;
        y -= 0.5 / 256.0;
    }

    return vec4f(x, y, 0.0, 1.0);
}

const TEXTURE_4BPP: u32 = 0;
const TEXTURE_8BPP: u32 = 1;
const TEXTURE_15BPP: u32 = 2;

struct TexturedVertex {
    @location(0) position: vec3f,
    @location(1) color: vec3u,
    @location(2) uv: vec2u,
    @location(3) texpage: vec2u,
    @location(4) tex_window_mask: vec2u,
    @location(5) tex_window_offset: vec2u,
    @location(6) clut: vec2u,
    @location(7) flags: u32,
    @location(8) integer_position: vec2i,
    @location(9) other_positions: vec4i,
    @location(10) other_uv: vec4u,
}

struct TexturedVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) uv: vec2f,
    @location(2) texpage: vec2u,
    @location(3) tex_window_mask: vec2u,
    @location(4) tex_window_offset: vec2u,
    @location(5) clut: vec2u,
    @location(6) flags: u32,
    // vec2(dU/dX + dU/dY, dV/dX + dV/dY)
    @location(7) @interpolate(flat) duv: vec2f,
}

const COLOR_DEPTH_FLAGS: u32 = 3;
const MODULATED_FLAG: u32 = 4;
const DITHERABLE_FLAG: u32 = 8;

struct Flags {
    color_depth: u32,
    modulated: bool,
    ditherable: bool,
}

fn parse_flags(flags: u32) -> Flags {
    return Flags(
        flags & COLOR_DEPTH_FLAGS,
        (flags & MODULATED_FLAG) != 0,
        (flags & DITHERABLE_FLAG) != 0,
    );
}

fn flags_ditherable(flags: u32) -> bool {
    return (flags & DITHERABLE_FLAG) != 0;
}

struct TexturedRectVertex {
    @location(0) position: vec2i,
    @location(1) color: vec3u,
    @location(2) texpage: vec2u,
    @location(3) tex_window_mask: vec2u,
    @location(4) tex_window_offset: vec2u,
    @location(5) clut: vec2u,
    @location(6) flags: u32,
    @location(7) base_position: vec2i,
    @location(8) base_uv: vec2u,
}

struct TexturedRectVertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
    @location(1) texpage: vec2u,
    @location(2) tex_window_mask: vec2u,
    @location(3) tex_window_offset: vec2u,
    @location(4) clut: vec2u,
    @location(5) flags: u32,
    @location(6) base_position: vec2i,
    @location(7) base_uv: vec2u,
}

fn compute_dx(component: vec3i, v0: vec2i, v1: vec2i, v2: vec2i) -> i32 {
    return component[0] * (v1.y - v2.y)
        + component[1] * (v2.y - v0.y)
        + component[2] * (v0.y - v1.y);
}

fn compute_dy(component: vec3i, v0: vec2i, v1: vec2i, v2: vec2i) -> i32 {
    return component[0] * (v2.x - v1.x)
        + component[1] * (v0.x - v2.x)
        + component[2] * (v1.x - v0.x);
}

fn compute_duv(vert0: vec2i, uv0: vec2u, other_positions: vec4i, other_uv: vec4u) -> vec2f {
    let vert1 = other_positions.xy;
    let vert2 = other_positions.zw;
    let uv1 = other_uv.xy;
    let uv2 = other_uv.zw;

    let denominator = f32((vert1.x - vert0.x) * (vert2.y - vert0.y) - (vert1.y - vert0.y) * (vert2.x - vert0.x));

    let u = vec3i(vec3u(uv0.x, uv1.x, uv2.x));
    let v = vec3i(vec3u(uv0.y, uv1.y, uv2.y));

    let du_dx = compute_dx(u, vert0, vert1, vert2);
    let du_dy = compute_dy(u, vert0, vert1, vert2);
    let dv_dx = compute_dx(v, vert0, vert1, vert2);
    let dv_dy = compute_dy(v, vert0, vert1, vert2);

    let du = f32(du_dx + du_dy) / denominator;
    let dv = f32(dv_dx + dv_dy) / denominator;
    return vec2f(du, dv);
}

struct SemiTransparentOutput {
    @location(0) @blend_src(0) color: vec4f,
    @location(0) @blend_src(1) blend: vec4f,
}

fn read_4bpp_texture(uv: vec2u, texpage: vec2u, clut: vec2u) -> u32 {
    let x = texpage.x + (uv.x >> 2);
    let y = texpage.y + uv.y;
    let shift = (uv.x & 3) << 2;

    let texel = textureLoad(native_vram, vec2u(x, y)).r;
    let clut_index = (texel >> shift) & 15;
    let clut_position = clut + vec2u(clut_index, 0);
    return textureLoad(native_vram, clut_position).r;
}

fn read_8bpp_texture(uv: vec2u, texpage: vec2u, clut: vec2u) -> u32 {
    let x = (texpage.x + (uv.x >> 1)) & 1023;
    let y = texpage.y + uv.y;
    let shift = (uv.x & 1) << 3;

    let texel = textureLoad(native_vram, vec2u(x, y)).r;
    let clut_index = (texel >> shift) & 255;
    let clut_x = (clut.x + clut_index) & 1023;
    return textureLoad(native_vram, vec2u(clut_x, clut.y)).r;
}

fn apply_texture_window(uv: vec2u, mask: vec2u, offset: vec2u) -> vec2u {
    return (uv & ~mask) | (offset & mask);
}

fn apply_modulation(texel: vec4f, input_color: vec3f) -> vec4f {
    let rgb = saturate(floor(texel.rgb * 1.9921875 * input_color * 255.0) / 255.0);
    return vec4f(rgb, texel.a);
}

fn convert_texel_low_color(texel: vec3u) -> vec3f {
    return vec3f(texel << vec3u(3)) / 255.0;
}

fn convert_texel_high_color(texel: vec3u) -> vec3f {
    return vec3f(texel) / 31.0;
}

fn sample_texture(
    input_color: vec3f,
    input_uv: vec2u,
    texpage: vec2u,
    tex_window_mask: vec2u,
    tex_window_offset: vec2u,
    clut: vec2u,
    flags: Flags,
) -> vec4f {
    let uv = apply_texture_window(input_uv, tex_window_mask, tex_window_offset);

    var color: u32;
    if flags.color_depth == TEXTURE_4BPP {
        color = read_4bpp_texture(uv, texpage, clut);
    } else if flags.color_depth == TEXTURE_8BPP {
        color = read_8bpp_texture(uv, texpage, clut);
    } else {
        discard;
    }

    if color == 0 {
        discard;
    }

    let texel_parsed = (vec3u(color) >> vec3u(0, 5, 10)) & vec3u(0x1F);
    var texel_rgb: vec3f;
    if draw_settings.high_color != 0 {
        texel_rgb = convert_texel_high_color(texel_parsed);
    } else {
        texel_rgb = convert_texel_low_color(texel_parsed);
    }

    let a = f32((color >> 15) & 1);
    var texel = vec4f(texel_rgb, a);

    if flags.modulated {
        texel = apply_modulation(texel, input_color);
    }

    return texel;
}

fn sample_15bpp_texture(
    input_color: vec3f,
    scaled_uv: vec2u,
    texpage: vec2u,
    modulated: bool,
) -> vec4f {
    let scale = draw_settings.resolution_scale;
    let x = (scale * texpage.x + scaled_uv.x) % (scale * 1024);
    let y = scale * texpage.y + scaled_uv.y;
    var texel = textureLoad(scaled_vram_copy, vec2u(x, y));

    if texel.r == 0.0 && texel.g == 0.0 && texel.b == 0.0 && texel.a == 0.0 {
        discard;
    }

    if draw_settings.high_color == 0 {
        // Mask out the lowest 3 bits of each component
        let texel_rgb = round(8.0 * floor(texel.rgb * 255.0 / 8.0)) / 255.0;
        texel.r = texel_rgb.r;
        texel.g = texel_rgb.g;
        texel.b = texel_rgb.b;
    }

    if modulated {
        texel = apply_modulation(texel, input_color);
    }

    return texel;
}

fn round_uv(uv: vec2f, duv: vec2f) -> vec2u {
    if draw_settings.resolution_scale == 1 {
        return vec2u(round(uv));
    }

    // This doesn't make a whole lot of sense but it seems to work decently with both 2D and 3D graphics.
    // The basic idea is to round in the direction that U and V would change when moving up and left, but not to round
    // too far if dU or dV is very small
    let clamped_duv = clamp(duv, vec2f(-0.5), vec2f(0.5));
    return vec2u(round(uv - clamped_duv));
}

fn sample_texture_triangle(input: TexturedVertexOutput) -> vec4f {
    let flags = parse_flags(input.flags);

    if flags.color_depth == TEXTURE_15BPP {
        let fractional_uv = fract(input.uv);
        let integral_uv = vec2u(input.uv);
        let masked_uv = apply_texture_window(integral_uv, input.tex_window_mask, input.tex_window_offset);

        let scale = draw_settings.resolution_scale;
        let fractional_uv_scaled = round_uv(f32(scale) * fractional_uv, input.duv);
        let scaled_uv = scale * masked_uv + fractional_uv_scaled;

        return sample_15bpp_texture(input.color, scaled_uv, input.texpage, flags.modulated);
    }

    let uv = round_uv(input.uv, input.duv);

    return sample_texture(
        input.color,
        uv,
        input.texpage,
        input.tex_window_mask,
        input.tex_window_offset,
        input.clut,
        flags,
    );
}

fn sample_texture_rect(input: TexturedRectVertexOutput) -> vec4f {
    let uv_offset = (vec2i(input.position.xy) - i32(draw_settings.resolution_scale) * input.base_position)
        / i32(draw_settings.resolution_scale);
    let uv = (input.base_uv + vec2u(uv_offset)) & vec2u(255, 255);

    let flags = parse_flags(input.flags);

    if flags.color_depth == TEXTURE_15BPP {
        let scale = draw_settings.resolution_scale;
        let scaled_uv = scale * uv + (vec2u(input.position.xy) % scale);
        return sample_15bpp_texture(input.color, scaled_uv, input.texpage, flags.modulated);
    }

    return sample_texture(
        input.color,
        uv,
        input.texpage,
        input.tex_window_mask,
        input.tex_window_offset,
        input.clut,
        flags,
    );
}

fn truncate_color(color: vec3f) -> vec3f {
    // Truncate from 24bpp to 15bpp
    let truncated = vec3u(round(color * 255.0)) >> vec3u(3);
    return vec3f(truncated) / 31.0;
}

var<private> DITHER_TABLE: array<vec4f, 4> = array<vec4f, 4>(
    vec4f(-0.01568627450980392, 0.0, -0.011764705882352941, 0.00392156862745098),
    vec4f(0.00784313725490196, -0.00784313725490196, 0.011764705882352941, -0.00392156862745098),
    vec4f(-0.011764705882352941, 0.00392156862745098, -0.01568627450980392, 0.0),
    vec4f(0.011764705882352941, -0.00392156862745098, 0.00784313725490196, -0.00784313725490196),
);

fn apply_dither(color: vec3f, position: vec2u) -> vec3f {
    let dither = DITHER_TABLE[position.y & 3][position.x & 3];
    return saturate(color + vec3f(dither));
}

fn finalize_color_tri(pixel: vec4f, position: vec2f, ditherable: bool) -> vec4f {
    var color = pixel.rgb;

    if draw_settings.dithering != 0 && ditherable {
        let scaled_position = vec2u(position);
        let dither_position = select(
            scaled_position / draw_settings.resolution_scale,
            scaled_position,
            draw_settings.high_res_dithering != 0,
        );
        color = apply_dither(color, dither_position);
    }

    if draw_settings.high_color == 0 {
        color = truncate_color(color);
    }

    return vec4f(color, pixel.a);
}

fn finalize_color_rect(pixel: vec4f) -> vec4f {
    if draw_settings.high_color == 0 {
        return vec4f(truncate_color(pixel.rgb), pixel.a);
    }

    return pixel;
}