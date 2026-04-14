@group(0) @binding(0)
var scaled_vram: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1)
var native_vram: texture_storage_2d<r32uint, read>;
@group(0) @binding(2)
var scaled_vram_copy: texture_storage_2d<rgba8unorm, read>;

var<immediate> draw_settings: DrawSettings;

const DUMMY_RETURN: vec4f = vec4f(0.0, 0.0, 0.0, 0.0);

@fragment
fn fs_untextured_average(input: UntexturedVertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let input_color = finalize_color_tri(vec4f(input.color, 0.0), input.position.xy, input.ditherable != 0);
    let blended_color = saturate(0.5 * input_color.rgb + 0.5 * existing.rgb);
    let pixel = vec4f(blended_color, f32(draw_settings.force_mask_bit));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }

    return DUMMY_RETURN;
}

fn blend_average(texel: vec4f, existing: vec4f) -> vec3f {
    return select(
        texel.rgb,
        saturate(0.5 * texel.rgb + 0.5 * existing.rgb),
        texel.a != 0.0,
    );
}

@fragment
fn fs_textured_average(input: TexturedVertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_tri(sample_texture_triangle(input), input.position.xy, flags_ditherable(input.flags));
    let blended_color = blend_average(texel, existing);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }

    return DUMMY_RETURN;
}

@fragment
fn fs_textured_rect_average(input: TexturedRectVertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_rect(sample_texture_rect(input));
    let blended_color = blend_average(texel, existing);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }

    return DUMMY_RETURN;
}

fn blend_add(texel: vec4f, existing: vec4f, factor: f32) -> vec3f {
    return select(
       texel.rgb,
       saturate(factor * texel.rgb + existing.rgb),
       texel.a != 0.0,
   );
}

fn textured_add(input: TexturedVertexOutput, factor: f32) -> vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_tri(sample_texture_triangle(input), input.position.xy, flags_ditherable(input.flags));
    let blended_color = blend_add(texel, existing, factor);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }

    return DUMMY_RETURN;
}

fn textured_rect_add(input: TexturedRectVertexOutput, factor: f32) -> vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_rect(sample_texture_rect(input));
    let blended_color = blend_add(texel, existing, factor);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }
    
    return DUMMY_RETURN;
}

@fragment
fn fs_textured_add(input: TexturedVertexOutput) -> @location(0) vec4f {
    return textured_add(input, 1.0);
}

@fragment
fn fs_textured_rect_add(input: TexturedRectVertexOutput) -> @location(0) vec4f {
    return textured_rect_add(input, 1.0);
}

fn blend_subtract(texel: vec4f, existing: vec4f) -> vec3f {
    return select(
        texel.rgb,
        saturate(existing.rgb - texel.rgb),
        texel.a != 0.0,
    );
}

@fragment
fn fs_textured_subtract(input: TexturedVertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_tri(sample_texture_triangle(input), input.position.xy, flags_ditherable(input.flags));
    let blended_color = blend_subtract(texel, existing);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }
    
    return DUMMY_RETURN;
}

@fragment
fn fs_textured_rect_subtract(input: TexturedRectVertexOutput) -> @location(0) vec4f {
    let position = vec2u(input.position.xy);
    let existing = textureLoad(scaled_vram, position);
    if existing.a != 0.0 {
        discard;
    }

    let texel = finalize_color_rect(sample_texture_rect(input));
    let blended_color = blend_subtract(texel, existing);

    let pixel = vec4f(blended_color, max(texel.a, f32(draw_settings.force_mask_bit)));
    textureStore(scaled_vram, position, pixel);

    // DX12 won't compile shaders that never return anything
    if true {
        discard;
    }
    
    return DUMMY_RETURN;
}

@fragment
fn fs_textured_add_quarter(input: TexturedVertexOutput) -> @location(0) vec4f {
    return textured_add(input, 0.25);
}

@fragment
fn fs_textured_rect_add_quarter(input: TexturedRectVertexOutput) -> @location(0) vec4f {
    return textured_rect_add(input, 0.25);
}