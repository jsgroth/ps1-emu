#![allow(clippy::many_single_char_names)]
#![allow(unsafe_op_in_unsafe_fn)]

use crate::gpu::gp0::{
    DrawSettings, SemiTransparencyMode, TextureColorDepthBits, TexturePage, TextureWindow,
};
use crate::gpu::rasterizer::simd::AlignedVram;
use crate::gpu::rasterizer::{
    LineShading, RectangleTextureMapping, Shading, TextureMappingMode, TriangleShading,
    TriangleTextureMapping,
};
use crate::gpu::{Color, Vertex, rasterizer};
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;
use std::{cmp, mem};

const DITHER_TABLE: &[[i16; 16]; 4] = &[
    [-4, 0, -3, 1, -4, 0, -3, 1, -4, 0, -3, 1, -4, 0, -3, 1],
    [2, -2, 3, -1, 2, -2, 3, -1, 2, -2, 3, -1, 2, -2, 3, -1],
    [-3, 1, -4, 0, -3, 1, -4, 0, -3, 1, -4, 0, -3, 1, -4, 0],
    [3, -1, 2, -2, 3, -1, 2, -2, 3, -1, 2, -2, 3, -1, 2, -2],
];

#[derive(Debug, Clone, Copy)]
struct Step {
    x: i32,
    y: i32,
}

impl Step {
    const ZERO: Self = Self { x: 0, y: 0 };

    fn new(v: [Vertex; 3], component: [u8; 3], denominator: i32) -> Self {
        Self {
            x: compute_x_step(v, component, denominator),
            y: compute_y_step(v, component, denominator),
        }
    }
}

#[derive(Debug, Clone)]
struct Interpolator {
    base_x: i32,
    base_y: i32,
    base_r: i32,
    base_g: i32,
    base_b: i32,
    base_u: i32,
    base_v: i32,
    r_step: Step,
    g_step: Step,
    b_step: Step,
    u_step: Step,
    v_step: Step,
}

impl Interpolator {
    // PS1 GPU appears to use fixed-point decimal with 12 fractional bits
    // U/V interpolation does not look correct otherwise
    const SHIFT: u8 = 12;

    #[target_feature(enable = "avx2")]
    unsafe fn new(
        v: [Vertex; 3],
        denominator: i32,
        shading: TriangleShading,
        texture_mapping: Option<&TriangleTextureMapping>,
    ) -> Self {
        let (base_vertex_idx, base_vertex) = v
            .into_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.x.cmp(&b.x).then(a.y.cmp(&b.y)))
            .unwrap();

        let base_color = match shading {
            Shading::Flat(color) => color,
            Shading::Gouraud(colors) => colors[base_vertex_idx],
        };

        let (r_step, g_step, b_step) = match shading {
            Shading::Flat(_) => (Step::ZERO, Step::ZERO, Step::ZERO),
            Shading::Gouraud(colors) => {
                let r = colors.map(|color| color.r);
                let g = colors.map(|color| color.g);
                let b = colors.map(|color| color.b);

                (
                    Step::new(v, r, denominator),
                    Step::new(v, g, denominator),
                    Step::new(v, b, denominator),
                )
            }
        };

        let (base_tex_coords, u_step, v_step) = match texture_mapping {
            Some(mapping) => {
                let base_tex_coords = (mapping.u[base_vertex_idx], mapping.v[base_vertex_idx]);

                (
                    base_tex_coords,
                    Step::new(v, mapping.u, denominator),
                    Step::new(v, mapping.v, denominator),
                )
            }
            None => ((0, 0), Step::ZERO, Step::ZERO),
        };

        Self {
            base_x: base_vertex.x,
            base_y: base_vertex.y,
            base_r: shift_base_component(base_color.r),
            base_g: shift_base_component(base_color.g),
            base_b: shift_base_component(base_color.b),
            base_u: shift_base_component(base_tex_coords.0),
            base_v: shift_base_component(base_tex_coords.1),
            r_step,
            g_step,
            b_step,
            u_step,
            v_step,
        }
    }

    // Compute the interpolated colors for the given points.
    // Input vectors should be i32x8.
    // Return values are R/G/B color components as i32x8 vectors.
    #[target_feature(enable = "avx2")]
    unsafe fn interpolate_color(&self, px: __m256i, py: __m256i) -> (__m256i, __m256i, __m256i) {
        let r = interpolate_component(px, py, self.base_x, self.base_y, self.base_r, self.r_step);
        let g = interpolate_component(px, py, self.base_x, self.base_y, self.base_g, self.g_step);
        let b = interpolate_component(px, py, self.base_x, self.base_y, self.base_b, self.b_step);

        (r, g, b)
    }

    // Compute the interpolated U/V coordinates for the given points.
    // Input vectors should be i32x8.
    // Return values are U/V coordinates as i32x8 vectors.
    #[target_feature(enable = "avx2")]
    unsafe fn interpolate_uv(&self, px: __m256i, py: __m256i) -> (__m256i, __m256i) {
        let u = interpolate_component(px, py, self.base_x, self.base_y, self.base_u, self.u_step);
        let v = interpolate_component(px, py, self.base_x, self.base_y, self.base_v, self.v_step);

        (u, v)
    }
}

fn shift_base_component(component: u8) -> i32 {
    (i32::from(component) << Interpolator::SHIFT) | (1 << (Interpolator::SHIFT - 1))
}

fn compute_x_step(v: [Vertex; 3], component: [u8; 3], denominator: i32) -> i32 {
    let component = component.map(i32::from);
    let raw = component[0] * (v[1].y - v[2].y)
        + component[1] * (v[2].y - v[0].y)
        + component[2] * (v[0].y - v[1].y);
    (raw << Interpolator::SHIFT) / denominator
}

fn compute_y_step(v: [Vertex; 3], component: [u8; 3], denominator: i32) -> i32 {
    let component = component.map(i32::from);
    let raw = component[0] * (v[2].x - v[1].x)
        + component[1] * (v[0].x - v[2].x)
        + component[2] * (v[1].x - v[0].x);
    (raw << Interpolator::SHIFT) / denominator
}

#[target_feature(enable = "avx2")]
unsafe fn interpolate_component(
    px: __m256i,
    py: __m256i,
    base_x: i32,
    base_y: i32,
    base_component: i32,
    step: Step,
) -> __m256i {
    let dx = _mm256_sub_epi32(px, _mm256_set1_epi32(base_x));
    let dy = _mm256_sub_epi32(py, _mm256_set1_epi32(base_y));

    let value_fixedpoint = _mm256_add_epi32(
        _mm256_set1_epi32(base_component),
        _mm256_add_epi32(
            _mm256_mullo_epi32(dx, _mm256_set1_epi32(step.x)),
            _mm256_mullo_epi32(dy, _mm256_set1_epi32(step.y)),
        ),
    );

    mask_epi32_to_u8(_mm256_srli_epi32::<{ Interpolator::SHIFT as i32 }>(value_fixedpoint))
}

#[target_feature(enable = "avx2")]
unsafe fn mask_epi32_to_u8(v: __m256i) -> __m256i {
    _mm256_and_si256(v, _mm256_set1_epi32(0xFF))
}

fn i11(value: i32) -> i32 {
    (value << 21) >> 21
}

#[target_feature(enable = "avx2")]
unsafe fn i11_epi16(value: __m256i) -> __m256i {
    _mm256_srai_epi16::<5>(_mm256_slli_epi16::<5>(value))
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// This function must only be called when running on a CPU that supports AVX2 instructions.
pub unsafe fn rasterize_triangle(
    vram: &mut AlignedVram,
    &DrawSettings {
        dithering_enabled,
        draw_offset,
        draw_area_top_left,
        draw_area_bottom_right,
        force_mask_bit,
        check_mask_bit,
        ..
    }: &DrawSettings,
    x_bounds: (i32, i32),
    y_bounds: (i32, i32),
    vertices: [Vertex; 3],
    shading: TriangleShading,
    texture_mapping: Option<TriangleTextureMapping>,
    semi_transparency_mode: Option<SemiTransparencyMode>,
) {
    let vram_ptr = vram.as_mut_ptr();

    let forced_mask_bit = i16::from(force_mask_bit) << 15;

    let v01_is_not_bottom_right = is_not_bottom_right_edge(vertices[0], vertices[1]);
    let v12_is_not_bottom_right = is_not_bottom_right_edge(vertices[1], vertices[2]);
    let v20_is_not_bottom_right = is_not_bottom_right_edge(vertices[2], vertices[0]);

    // AVX2 loads/stores must be aligned to a 16-halfword/32-byte boundary
    let min_x_aligned = ((x_bounds.0 + draw_offset.x) & !0xF) - draw_offset.x;
    let max_x_aligned = ((x_bounds.1 + draw_offset.x) & !0xF) - draw_offset.x;

    let interpolation_denominator =
        rasterizer::cross_product_z(vertices[0], vertices[1], vertices[2]);
    if interpolation_denominator == 0 {
        // Points are collinear, do nothing
        return;
    }

    let interpolator =
        Interpolator::new(vertices, interpolation_denominator, shading, texture_mapping.as_ref());

    for y in y_bounds.0..=y_bounds.1 {
        let y_offset = i11(y + draw_offset.y);
        if y_offset < draw_area_top_left.y || y_offset > draw_area_bottom_right.y {
            continue;
        }

        let py = _mm256_set1_epi32(y);

        for x in (min_x_aligned..=max_x_aligned).step_by(16) {
            let x_offset = i11(x + draw_offset.x);
            if !(0..1024).contains(&x_offset) {
                continue;
            }

            // Determine which X coordinates are inside the triangle.
            // The 16 X coordinates are split up such that vectors can later be converted from
            // two i32x8 vectors to a single i16x16 vector using _mm256_packs_epi32
            let px1 = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 8, x + 9, x + 10, x + 11);
            let inside_mask_1 = compute_write_mask(
                vertices,
                px1,
                py,
                x_bounds,
                v01_is_not_bottom_right,
                v12_is_not_bottom_right,
                v20_is_not_bottom_right,
            );

            let px2 = _mm256_setr_epi32(x + 4, x + 5, x + 6, x + 7, x + 12, x + 13, x + 14, x + 15);
            let inside_mask_2 = compute_write_mask(
                vertices,
                px2,
                py,
                x_bounds,
                v01_is_not_bottom_right,
                v12_is_not_bottom_right,
                v20_is_not_bottom_right,
            );

            let mut inside_mask = _mm256_packs_epi32(inside_mask_1, inside_mask_2);

            // If no points are inside the triangle, bail out early
            if _mm256_testz_si256(inside_mask, _mm256_set1_epi16(!0)) != 0 {
                continue;
            }

            // Check if inside draw area X coordinate range
            let px = _mm256_packs_epi32(px1, px2);
            let px_offset =
                i11_epi16(_mm256_add_epi16(px, _mm256_set1_epi16(draw_offset.x as i16)));
            inside_mask = _mm256_and_si256(
                inside_mask,
                _mm256_andnot_si256(
                    _mm256_cmpgt_epi16(_mm256_set1_epi16(draw_area_top_left.x as i16), px_offset),
                    _mm256_cmpgt_epi16(
                        _mm256_set1_epi16(draw_area_bottom_right.x as i16 + 1),
                        px_offset,
                    ),
                ),
            );

            if _mm256_testz_si256(inside_mask, _mm256_set1_epi16(!0)) != 0 {
                continue;
            }

            // Apply shading to determine initial color
            let (mut r, mut g, mut b) = match shading {
                TriangleShading::Flat(color) => (
                    _mm256_set1_epi16(color.r.into()),
                    _mm256_set1_epi16(color.g.into()),
                    _mm256_set1_epi16(color.b.into()),
                ),
                TriangleShading::Gouraud(..) => {
                    let (r1, g1, b1) = interpolator.interpolate_color(px1, py);
                    let (r2, g2, b2) = interpolator.interpolate_color(px2, py);
                    (
                        _mm256_packs_epi32(r1, r2),
                        _mm256_packs_epi32(g1, g2),
                        _mm256_packs_epi32(b1, b2),
                    )
                }
            };

            // Default to values for an untextured triangle: bit 15 is set only if the force
            // mask bit setting is on, and all pixels are semi-transparent
            let mut mask_bits = _mm256_set1_epi16(forced_mask_bit);
            let mut semi_transparency_bits = _mm256_set1_epi16(1 << 15);

            // Apply texture mapping if present
            if let Some(texture_mapping) = &texture_mapping {
                // Interpolate U/V coordinates
                let (u1, v1) = interpolator.interpolate_uv(px1, py);
                let (u2, v2) = interpolator.interpolate_uv(px2, py);
                let (u, v) = (_mm256_packus_epi32(u1, u2), _mm256_packus_epi32(v1, v2));

                // Read 16 texels from the texture
                let texels = read_texture(
                    vram_ptr,
                    &texture_mapping.texpage,
                    &texture_mapping.window,
                    texture_mapping.clut_x.into(),
                    texture_mapping.clut_y.into(),
                    u,
                    v,
                );

                // Mask out any pixels where the texel value is $0000
                inside_mask = _mm256_andnot_si256(
                    _mm256_cmpeq_epi16(texels, _mm256_setzero_si256()),
                    inside_mask,
                );

                // Texels are semi-transparent only if bit 15 is set
                semi_transparency_bits = _mm256_and_si256(texels, _mm256_set1_epi16(1 << 15));
                mask_bits = _mm256_or_si256(mask_bits, semi_transparency_bits);

                let (tr, tg, tb) = convert_15bit_to_24bit(texels);

                // Optionally apply texture color modulation
                match texture_mapping.mode {
                    TextureMappingMode::Raw => {
                        r = tr;
                        g = tg;
                        b = tb;
                    }
                    TextureMappingMode::Modulated => {
                        r = modulate_texture_color(tr, r);
                        g = modulate_texture_color(tg, g);
                        b = modulate_texture_color(tb, b);
                    }
                }
            }

            // Load the existing row of 16 pixels
            let vram_addr =
                vram_ptr.add(1024 * y_offset as usize + x_offset as usize).cast::<__m256i>();
            let existing = _mm256_load_si256(vram_addr);

            if check_mask_bit {
                // Mask out any pixels where the existing pixel has bit 15 set
                inside_mask = _mm256_and_si256(
                    inside_mask,
                    _mm256_cmpeq_epi16(
                        _mm256_and_si256(existing, _mm256_set1_epi16(1 << 15)),
                        _mm256_setzero_si256(),
                    ),
                );
            }

            // If dithering is enabled, apply dithering before truncating to RGB555.
            // Dithering is applied only if Gouraud shading or texture color modulation is enabled
            if dithering_enabled
                && (matches!(shading, TriangleShading::Gouraud { .. })
                    || texture_mapping
                        .as_ref()
                        .is_some_and(|mapping| mapping.mode == TextureMappingMode::Modulated))
            {
                let dither_vector: __m256i = mem::transmute(DITHER_TABLE[(y & 3) as usize]);

                let u8_max = _mm256_set1_epi16(255);
                r = _mm256_min_epi16(
                    u8_max,
                    _mm256_max_epi16(_mm256_setzero_si256(), _mm256_add_epi16(r, dither_vector)),
                );
                g = _mm256_min_epi16(
                    u8_max,
                    _mm256_max_epi16(_mm256_setzero_si256(), _mm256_add_epi16(g, dither_vector)),
                );
                b = _mm256_min_epi16(
                    u8_max,
                    _mm256_max_epi16(_mm256_setzero_si256(), _mm256_add_epi16(b, dither_vector)),
                );
            }

            // Truncate color from from 8-bit RGB components to 5-bit
            r = _mm256_srli_epi16::<3>(r);
            g = _mm256_srli_epi16::<3>(g);
            b = _mm256_srli_epi16::<3>(b);

            // If semi-transparency is enabled, blend existing colors with new colors
            if let Some(semi_transparency_mode) = semi_transparency_mode
                && _mm256_testz_si256(semi_transparency_bits, _mm256_set1_epi16(!0)) == 0
            {
                let (existing_r, existing_g, existing_b) = split_15bit_color(existing);
                let semi_transparency_mask =
                    _mm256_cmpeq_epi16(semi_transparency_bits, _mm256_setzero_si256());

                (r, g, b) = apply_semi_transparency(
                    (existing_r, existing_g, existing_b),
                    (r, g, b),
                    semi_transparency_mask,
                    semi_transparency_mode,
                );
            }

            // Combine color components and OR in bit 15 (either force mask bit or texel bit 15)
            let color = _mm256_or_si256(combine_rgb_components(r, g, b), mask_bits);

            // Store the row of pixels, using the write mask to control which are written
            _mm256_store_si256(
                vram_addr,
                _mm256_or_si256(
                    _mm256_and_si256(inside_mask, color),
                    _mm256_andnot_si256(inside_mask, existing),
                ),
            );
        }
    }
}

fn is_not_bottom_right_edge(v0: Vertex, v1: Vertex) -> i32 {
    let is_bottom_right = v1.y > v0.y || (v1.y == v0.y && v1.x < v0.x);
    if is_bottom_right { 0 } else { !0 }
}

// Determine which of the 8 points are inside the triangle.
// Input vectors should be i32x8.
// Return value is i32x8 where each lane is all 1s if inside the triangle and all 0s if outside.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn compute_write_mask(
    vertices: [Vertex; 3],
    px: __m256i,
    py: __m256i,
    x_bounds: (i32, i32),
    v01_is_not_bottom_right: i32,
    v12_is_not_bottom_right: i32,
    v20_is_not_bottom_right: i32,
) -> __m256i {
    _mm256_and_si256(
        _mm256_and_si256(
            check_edge(vertices[0], vertices[1], px, py, v01_is_not_bottom_right),
            _mm256_and_si256(
                check_edge(vertices[1], vertices[2], px, py, v12_is_not_bottom_right),
                check_edge(vertices[2], vertices[0], px, py, v20_is_not_bottom_right),
            ),
        ),
        _mm256_andnot_si256(
            _mm256_cmpgt_epi32(_mm256_set1_epi32(x_bounds.0), px),
            _mm256_cmpgt_epi32(_mm256_set1_epi32(x_bounds.1 + 1), px),
        ),
    )
}

// Determine which of the 8 points are inside a single triangle edge.
// Input vectors should be i32x8.
// Return value is i32x8 where each lane is all 1s if inside the edge and all 0s if outside.
#[target_feature(enable = "avx2")]
unsafe fn check_edge(
    v0: Vertex,
    v1: Vertex,
    px: __m256i,
    py: __m256i,
    is_not_bottom_right: i32,
) -> __m256i {
    let cpz = cross_product_z(v0, v1, px, py);
    let zero = _mm256_setzero_si256();

    _mm256_or_si256(
        _mm256_cmpgt_epi32(cpz, zero),
        _mm256_and_si256(_mm256_cmpeq_epi32(cpz, zero), _mm256_set1_epi32(is_not_bottom_right)),
    )
}

// Compute the Z component of the cross product (v1 - v0) x (P - v0) for each point P.
// Input vectors should be i32x8 and return value is i32x8.
#[target_feature(enable = "avx2")]
unsafe fn cross_product_z(v0: Vertex, v1: Vertex, px: __m256i, py: __m256i) -> __m256i {
    _mm256_sub_epi32(
        _mm256_mullo_epi32(
            _mm256_set1_epi32(v1.x - v0.x),
            _mm256_sub_epi32(py, _mm256_set1_epi32(v0.y)),
        ),
        _mm256_mullo_epi32(
            _mm256_set1_epi32(v1.y - v0.y),
            _mm256_sub_epi32(px, _mm256_set1_epi32(v0.x)),
        ),
    )
}

// Apply semi-transparency blending.
// Input color vectors should be i16x16.
// Return values are i16x16, with all color components clamped to [0, 255].
#[target_feature(enable = "avx2")]
unsafe fn apply_semi_transparency(
    (existing_r, existing_g, existing_b): (__m256i, __m256i, __m256i),
    (r, g, b): (__m256i, __m256i, __m256i),
    semi_transparency_mask: __m256i,
    semi_transparency_mode: SemiTransparencyMode,
) -> (__m256i, __m256i, __m256i) {
    let (blended_r, blended_g, blended_b) = match semi_transparency_mode {
        SemiTransparencyMode::Average => (
            blend_average(existing_r, r),
            blend_average(existing_g, g),
            blend_average(existing_b, b),
        ),
        SemiTransparencyMode::Add => {
            (blend_add(existing_r, r), blend_add(existing_g, g), blend_add(existing_b, b))
        }
        SemiTransparencyMode::Subtract => (
            blend_subtract(existing_r, r),
            blend_subtract(existing_g, g),
            blend_subtract(existing_b, b),
        ),
        SemiTransparencyMode::AddQuarter => (
            blend_add_quarter(existing_r, r),
            blend_add_quarter(existing_g, g),
            blend_add_quarter(existing_b, b),
        ),
    };

    let r = _mm256_or_si256(
        _mm256_andnot_si256(semi_transparency_mask, blended_r),
        _mm256_and_si256(semi_transparency_mask, r),
    );
    let g = _mm256_or_si256(
        _mm256_andnot_si256(semi_transparency_mask, blended_g),
        _mm256_and_si256(semi_transparency_mask, g),
    );
    let b = _mm256_or_si256(
        _mm256_andnot_si256(semi_transparency_mask, blended_b),
        _mm256_and_si256(semi_transparency_mask, b),
    );

    (r, g, b)
}

// Read a row of 16 texels from a texture in VRAM.
// U and V vectors should be i16x16.
// Return value is an i16x16 vector containing raw 16-bit texel values (RGB555 + semi-transparency bit).
#[target_feature(enable = "avx2")]
unsafe fn read_texture(
    vram: *mut u16,
    texpage: &TexturePage,
    texture_window: &TextureWindow,
    clut_x: u32,
    clut_y: u32,
    u: __m256i,
    v: __m256i,
) -> __m256i {
    let x_mask = _mm256_set1_epi16((texture_window.x_mask << 3) as i16);
    let y_mask = _mm256_set1_epi16((texture_window.y_mask << 3) as i16);

    let masked_u = _mm256_or_si256(
        _mm256_andnot_si256(x_mask, u),
        _mm256_and_si256(x_mask, _mm256_set1_epi16((texture_window.x_offset << 3) as i16)),
    );
    let masked_v = _mm256_or_si256(
        _mm256_andnot_si256(y_mask, v),
        _mm256_and_si256(y_mask, _mm256_set1_epi16((texture_window.y_offset << 3) as i16)),
    );

    let (masked_u0, masked_u1) = unpack_epi16_vector(masked_u);
    let (masked_v0, masked_v1) = unpack_epi16_vector(masked_v);

    let (texels0, texels1) = match texpage.color_depth {
        TextureColorDepthBits::Four => (
            read_4bpp_texture(vram, texpage, clut_x, clut_y, masked_u0, masked_v0),
            read_4bpp_texture(vram, texpage, clut_x, clut_y, masked_u1, masked_v1),
        ),
        TextureColorDepthBits::Eight => (
            read_8bpp_texture(vram, texpage, clut_x, clut_y, masked_u0, masked_v0),
            read_8bpp_texture(vram, texpage, clut_x, clut_y, masked_u1, masked_v1),
        ),
        TextureColorDepthBits::Fifteen => (
            read_15bpp_texture(vram, texpage, masked_u0, masked_v0),
            read_15bpp_texture(vram, texpage, masked_u1, masked_v1),
        ),
    };

    _mm256_packus_epi32(texels0, texels1)
}

// Read a row of 8 texels from a 4bpp texture.
// U and V vectors should be i32x8.
// Return value is u16s stored in an i32x8 vector.
#[target_feature(enable = "avx2")]
unsafe fn read_4bpp_texture(
    vram: *mut u16,
    texpage: &TexturePage,
    clut_x: u32,
    clut_y: u32,
    u: __m256i,
    v: __m256i,
) -> __m256i {
    let vram_y = _mm256_add_epi32(v, _mm256_set1_epi32(texpage.y_base as i32));
    let vram_x_words = _mm256_add_epi32(
        _mm256_srli_epi32::<3>(u),
        _mm256_set1_epi32((32 * texpage.x_base) as i32),
    );

    let vram_addr = _mm256_or_si256(_mm256_slli_epi32::<9>(vram_y), vram_x_words);
    let vram_shift = _mm256_slli_epi32::<2>(_mm256_and_si256(u, _mm256_set1_epi32(0x07)));

    let vram_words = _mm256_i32gather_epi32::<4>(vram.cast::<i32>(), vram_addr);
    let clut_indices =
        _mm256_and_si256(_mm256_srlv_epi32(vram_words, vram_shift), _mm256_set1_epi32(0xF));

    let clut_offset = ((1024 * clut_y) | (16 * clut_x)) as usize;
    let clut = vram.add(clut_offset);

    read_clut(clut, clut_indices)
}

#[target_feature(enable = "avx2")]
unsafe fn read_clut(clut: *mut u16, clut_indices: __m256i) -> __m256i {
    let addrs = _mm256_srli_epi32::<1>(clut_indices);
    let shifts = _mm256_slli_epi32::<4>(_mm256_and_si256(clut_indices, _mm256_set1_epi32(1)));

    let colors = _mm256_i32gather_epi32::<4>(clut.cast::<i32>(), addrs);
    _mm256_and_si256(_mm256_srlv_epi32(colors, shifts), _mm256_set1_epi32(0xFFFF))
}

// Read a row of 8 texels from an 8bpp texture.
// U and V coordinates should be i32x8.
// Return value is u16s stored in an i32x8 vector.
#[target_feature(enable = "avx2")]
unsafe fn read_8bpp_texture(
    vram: *mut u16,
    texpage: &TexturePage,
    clut_x: u32,
    clut_y: u32,
    u: __m256i,
    v: __m256i,
) -> __m256i {
    let vram_y = _mm256_add_epi32(v, _mm256_set1_epi32(texpage.y_base as i32));
    let vram_x_words = _mm256_and_si256(
        _mm256_add_epi32(
            _mm256_srli_epi32::<2>(u),
            _mm256_set1_epi32((32 * texpage.x_base) as i32),
        ),
        _mm256_set1_epi32(0x1FF),
    );

    let vram_addr = _mm256_or_si256(_mm256_slli_epi32::<9>(vram_y), vram_x_words);
    let vram_shift = _mm256_slli_epi32::<3>(_mm256_and_si256(u, _mm256_set1_epi32(3)));

    let vram_words = _mm256_i32gather_epi32::<4>(vram.cast::<i32>(), vram_addr);
    let clut_indices =
        _mm256_and_si256(_mm256_srlv_epi32(vram_words, vram_shift), _mm256_set1_epi32(0xFF));

    let clut_offset = ((1024 * clut_y) | (16 * clut_x)) as usize;
    let clut = vram.add(clut_offset);

    read_clut(clut, clut_indices)
}

// Read a row of 8 texels from a 15bpp texture.
// U and V vectors should be i32x8.
// Return value is u16s stored in an i32x8 vector.
#[target_feature(enable = "avx2")]
unsafe fn read_15bpp_texture(
    vram: *mut u16,
    texpage: &TexturePage,
    u: __m256i,
    v: __m256i,
) -> __m256i {
    let vram_y = _mm256_add_epi32(v, _mm256_set1_epi32(texpage.y_base as i32));
    let vram_x_words = _mm256_and_si256(
        _mm256_add_epi32(
            _mm256_srli_epi32::<1>(u),
            _mm256_set1_epi32((32 * texpage.x_base) as i32),
        ),
        _mm256_set1_epi32(0x1FF),
    );

    let vram_addr = _mm256_or_si256(_mm256_slli_epi32::<9>(vram_y), vram_x_words);
    let vram_shift = _mm256_slli_epi32::<4>(_mm256_and_si256(u, _mm256_set1_epi32(1)));

    let vram_words = _mm256_i32gather_epi32::<4>(vram.cast::<i32>(), vram_addr);

    _mm256_and_si256(_mm256_srlv_epi32(vram_words, vram_shift), _mm256_set1_epi32(0xFFFF))
}

// Apply texture color modulation to a single color component.
// Input vectors should be i16x16 and the return value is i16x16.
#[target_feature(enable = "avx2")]
unsafe fn modulate_texture_color(tex_color: __m256i, shading_color: __m256i) -> __m256i {
    _mm256_min_epi16(
        _mm256_set1_epi16(255),
        _mm256_srli_epi16::<7>(_mm256_mullo_epi16(tex_color, shading_color)),
    )
}

// Apply average blending: (B + F) / 2
// Input vectors should be i16x16 and the return value is i16x16
#[target_feature(enable = "avx2")]
unsafe fn blend_average(back: __m256i, front: __m256i) -> __m256i {
    _mm256_srli_epi16::<1>(_mm256_add_epi16(back, front))
}

// Apply additive blending: B + F
// Input vectors should be i16x16 and the return value is i16x16, with each lane clamped to [0, 31]
#[target_feature(enable = "avx2")]
unsafe fn blend_add(back: __m256i, front: __m256i) -> __m256i {
    _mm256_min_epi16(_mm256_add_epi16(back, front), _mm256_set1_epi16(0x1F))
}

// Apply subtractive blending: B - F
// Input vectors should be i16x16 and the return value is i16x16, with each lane clamped to [0, 31]
#[target_feature(enable = "avx2")]
unsafe fn blend_subtract(back: __m256i, front: __m256i) -> __m256i {
    _mm256_max_epi16(_mm256_sub_epi16(back, front), _mm256_setzero_si256())
}

// Apply partial additive blending: B + F/4
// Input vectors should be i16x16 and the return value is i16x16, with each lane clamped to [0, 31]
#[target_feature(enable = "avx2")]
unsafe fn blend_add_quarter(back: __m256i, front: __m256i) -> __m256i {
    _mm256_min_epi16(_mm256_adds_epu8(back, _mm256_srli_epi16::<2>(front)), _mm256_set1_epi16(0x1F))
}

const LOW_SHUFFLE_MASK: &[u8; 32] = &[
    0, 1, 0x80, 0x80, 2, 3, 0x80, 0x80, 4, 5, 0x80, 0x80, 6, 7, 0x80, 0x80, 16, 17, 0x80, 0x80, 18,
    19, 0x80, 0x80, 20, 21, 0x80, 0x80, 22, 23, 0x80, 0x80,
];

const HIGH_SHUFFLE_MASK: &[u8; 32] = &[
    8, 9, 0x80, 0x80, 10, 11, 0x80, 0x80, 12, 13, 0x80, 0x80, 14, 15, 0x80, 0x80, 24, 25, 0x80,
    0x80, 26, 27, 0x80, 0x80, 28, 29, 0x80, 0x80, 30, 31, 0x80, 0x80,
];

// Unpack an i16x16 vector into two i32x8 vectors such that the two vectors can later be repacked
// using _mm256_packus_epi32 or _mm256_packs_epi32
#[target_feature(enable = "avx2")]
unsafe fn unpack_epi16_vector(v: __m256i) -> (__m256i, __m256i) {
    let low_shuffle_mask: __m256i = mem::transmute(*LOW_SHUFFLE_MASK);
    let high_shuffle_mask: __m256i = mem::transmute(*HIGH_SHUFFLE_MASK);

    let low = _mm256_shuffle_epi8(v, low_shuffle_mask);
    let high = _mm256_shuffle_epi8(v, high_shuffle_mask);

    (low, high)
}

// Combine 5-bit RGB color components into 15bpp color values.
// Input vectors should be i16x16 and the return value is i16x16
#[target_feature(enable = "avx2")]
unsafe fn combine_rgb_components(r: __m256i, g: __m256i, b: __m256i) -> __m256i {
    _mm256_or_si256(r, _mm256_or_si256(_mm256_slli_epi16::<5>(g), _mm256_slli_epi16::<10>(b)))
}

// Convert a raw 15-bit color value from VRAM to individual 8-bit RGB color components
// Input vector should be i16x16 and the return values are i16x16
#[target_feature(enable = "avx2")]
unsafe fn convert_15bit_to_24bit(texels: __m256i) -> (__m256i, __m256i, __m256i) {
    let mask = _mm256_set1_epi16(0x00F8);
    let r = _mm256_and_si256(_mm256_slli_epi16::<3>(texels), mask);
    let g = _mm256_and_si256(_mm256_srli_epi16::<2>(texels), mask);
    let b = _mm256_and_si256(_mm256_srli_epi16::<7>(texels), mask);

    (r, g, b)
}

// Split a raw 15-bit color value from VRAM into the separate 5-bit RGB color components.
// Input vector should be i16x16 and the return values are i16x16
#[target_feature(enable = "avx2")]
unsafe fn split_15bit_color(texels: __m256i) -> (__m256i, __m256i, __m256i) {
    let mask = _mm256_set1_epi16(0x001F);
    let r = _mm256_and_si256(texels, mask);
    let g = _mm256_and_si256(_mm256_srli_epi16::<5>(texels), mask);
    let b = _mm256_and_si256(_mm256_srli_epi16::<10>(texels), mask);

    (r, g, b)
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// This function must only be called when running on a CPU that supports AVX2 instructions.
pub unsafe fn rasterize_rectangle(
    vram: &mut AlignedVram,
    &DrawSettings {
        draw_offset,
        draw_area_top_left,
        draw_area_bottom_right,
        force_mask_bit,
        check_mask_bit,
        ..
    }: &DrawSettings,
    top_left: Vertex,
    width: i32,
    height: i32,
    color: Color,
    texture_mapping: Option<RectangleTextureMapping>,
    semi_transparency_mode: Option<SemiTransparencyMode>,
) {
    let vram_ptr = vram.as_mut_ptr();

    let forced_mask_bit = i16::from(force_mask_bit) << 15;

    // AVX2 loads/stores must be aligned to a 16-halfword/32-byte boundary
    let min_x_aligned = ((top_left.x + draw_offset.x) & !0xF) - draw_offset.x;
    let max_x_aligned = ((top_left.x + width + draw_offset.x) & !0xF) - draw_offset.x;

    let color_r = _mm256_set1_epi16(color.r.into());
    let color_g = _mm256_set1_epi16(color.g.into());
    let color_b = _mm256_set1_epi16(color.b.into());

    for dy in 0..height {
        let y_offset = i11(top_left.y + dy + draw_offset.y);
        if y_offset < draw_area_top_left.y || y_offset > draw_area_bottom_right.y {
            continue;
        }

        let vram_row_addr = 1024 * y_offset as usize;
        for x in (min_x_aligned..=max_x_aligned).step_by(16) {
            let x_offset = i11(x + draw_offset.x) as i16;
            if !(0..1024).contains(&x_offset) {
                continue;
            }

            let x = x as i16;

            let px = _mm256_setr_epi16(
                x,
                x + 1,
                x + 2,
                x + 3,
                x + 4,
                x + 5,
                x + 6,
                x + 7,
                x + 8,
                x + 9,
                x + 10,
                x + 11,
                x + 12,
                x + 13,
                x + 14,
                x + 15,
            );

            // Mask out pixels that are outside of the rectangle
            let mut write_mask = _mm256_andnot_si256(
                _mm256_cmpgt_epi16(_mm256_set1_epi16(top_left.x as i16), px),
                _mm256_cmpgt_epi16(_mm256_set1_epi16((top_left.x + width) as i16), px),
            );

            // Mask out pixels that are outside of the drawing area
            let px_offset =
                i11_epi16(_mm256_add_epi16(px, _mm256_set1_epi16(draw_offset.x as i16)));

            write_mask = _mm256_and_si256(
                write_mask,
                _mm256_andnot_si256(
                    _mm256_cmpgt_epi16(_mm256_set1_epi16(draw_area_top_left.x as i16), px_offset),
                    _mm256_cmpgt_epi16(
                        _mm256_set1_epi16((draw_area_bottom_right.x + 1) as i16),
                        px_offset,
                    ),
                ),
            );

            // Read existing pixel values from VRAM
            let vram_addr = vram_ptr.add(vram_row_addr + x_offset as usize).cast::<__m256i>();
            let existing = _mm256_load_si256(vram_addr);

            if check_mask_bit {
                // Mask out any pixels where the existing value has bit 15 set
                write_mask = _mm256_and_si256(
                    write_mask,
                    _mm256_cmpeq_epi16(
                        _mm256_and_si256(existing, _mm256_set1_epi16(1 << 15)),
                        _mm256_setzero_si256(),
                    ),
                );
            }

            // Initialize color to the color from the command word
            let mut r = color_r;
            let mut g = color_g;
            let mut b = color_b;

            // Default to values for an untextured rectangle: bit 15 is set only if the force
            // mask bit setting is on, and all pixels are semi-transparent
            let mut mask_bits = _mm256_set1_epi16(forced_mask_bit);
            let mut semi_transparency_bits = _mm256_set1_epi16(1 << 15);

            // Apply texture mapping if present
            if let Some(texture_mapping) = texture_mapping {
                // Compute U and V coordinates based on X and Y values, wrapping within [0, 255]
                let u = _mm256_and_si256(
                    _mm256_set1_epi16(0x00FF),
                    _mm256_add_epi16(
                        px,
                        _mm256_set1_epi16(
                            texture_mapping.u[0].wrapping_sub(top_left.x as u8).into(),
                        ),
                    ),
                );
                let v = _mm256_set1_epi16(texture_mapping.v[0].wrapping_add(dy as u8).into());

                // Read a row of 16 texels from the texture
                let texels = read_texture(
                    vram_ptr,
                    &texture_mapping.texpage,
                    &texture_mapping.window,
                    texture_mapping.clut_x.into(),
                    texture_mapping.clut_y.into(),
                    u,
                    v,
                );

                // Mask out any pixels where the texel value is $0000
                write_mask = _mm256_andnot_si256(
                    _mm256_cmpeq_epi16(texels, _mm256_setzero_si256()),
                    write_mask,
                );

                // Texture pixels are semi-transparent only if texel bit 15 is set
                semi_transparency_bits = _mm256_and_si256(texels, _mm256_set1_epi16(1 << 15));
                mask_bits = _mm256_or_si256(mask_bits, semi_transparency_bits);

                let (tr, tg, tb) = convert_15bit_to_24bit(texels);

                // Optionally apply texture color modulation
                match texture_mapping.mode {
                    TextureMappingMode::Raw => {
                        r = tr;
                        g = tg;
                        b = tb;
                    }
                    TextureMappingMode::Modulated => {
                        r = modulate_texture_color(tr, r);
                        g = modulate_texture_color(tg, g);
                        b = modulate_texture_color(tb, b);
                    }
                }
            }

            // Truncate color from from 8-bit RGB components to 5-bit
            r = _mm256_srli_epi16::<3>(r);
            g = _mm256_srli_epi16::<3>(g);
            b = _mm256_srli_epi16::<3>(b);

            // If semi-transparency is enabled, blend existing colors with new colors
            if let Some(semi_transparency_mode) = semi_transparency_mode {
                let semi_transparency_mask =
                    _mm256_cmpeq_epi16(semi_transparency_bits, _mm256_setzero_si256());

                let (existing_r, existing_g, existing_b) = split_15bit_color(existing);
                (r, g, b) = apply_semi_transparency(
                    (existing_r, existing_g, existing_b),
                    (r, g, b),
                    semi_transparency_mask,
                    semi_transparency_mode,
                );
            }

            // Combine color components and OR in the mask bit (force mask bit or texel bit 15)
            let color = _mm256_or_si256(combine_rgb_components(r, g, b), mask_bits);

            // Store the row of 16 pixels, using the write mask to control which are written
            _mm256_store_si256(
                vram_addr,
                _mm256_or_si256(
                    _mm256_and_si256(write_mask, color),
                    _mm256_andnot_si256(write_mask, existing),
                ),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
/// # Safety
///
/// This function must only be called when running on a CPU that supports AVX2 instructions.
pub unsafe fn rasterize_line(
    vram: &mut AlignedVram,
    vertices: [Vertex; 2],
    draw_area_top_left: Vertex,
    draw_area_bottom_right: Vertex,
    shading: LineShading,
    semi_transparency_mode: Option<SemiTransparencyMode>,
    dithering_enabled: bool,
    force_mask_bit: bool,
    check_mask_bit: bool,
) {
    let x_diff = vertices[1].x - vertices[0].x;
    let y_diff = vertices[1].y - vertices[0].y;

    if x_diff == 0 && y_diff == 0 {
        // Rasterize a single pixel
        let color = match shading {
            LineShading::Flat(color) | LineShading::Gouraud([color, _]) => color,
        };

        if (draw_area_top_left.x..=draw_area_bottom_right.x).contains(&vertices[0].x)
            && (draw_area_top_left.y..=draw_area_bottom_right.y).contains(&vertices[0].y)
        {
            rasterize_line_pixels(
                vram,
                _mm_set1_epi32(vertices[0].x),
                _mm_set1_epi32(vertices[0].y),
                _mm_set1_epi32(color.r.into()),
                _mm_set1_epi32(color.g.into()),
                _mm_set1_epi32(color.b.into()),
                _mm_setr_epi32(!0, 0, 0, 0),
                semi_transparency_mode,
                dithering_enabled,
                force_mask_bit,
                check_mask_bit,
            );
        }

        return;
    }

    if x_diff.abs() > y_diff.abs() {
        rasterize_line_h_oriented(
            vram,
            vertices,
            draw_area_top_left,
            draw_area_bottom_right,
            shading,
            semi_transparency_mode,
            dithering_enabled,
            force_mask_bit,
            check_mask_bit,
        );
    } else {
        rasterize_line_v_oriented(
            vram,
            vertices,
            draw_area_top_left,
            draw_area_bottom_right,
            shading,
            semi_transparency_mode,
            dithering_enabled,
            force_mask_bit,
            check_mask_bit,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn rasterize_line_h_oriented(
    vram: &mut AlignedVram,
    mut v: [Vertex; 2],
    draw_area_top_left: Vertex,
    draw_area_bottom_right: Vertex,
    mut shading: LineShading,
    semi_transparency_mode: Option<SemiTransparencyMode>,
    dithering_enabled: bool,
    force_mask_bit: bool,
    check_mask_bit: bool,
) {
    if v[0].x > v[1].x {
        v.swap(0, 1);

        if let LineShading::Gouraud(colors) = &mut shading {
            colors.swap(0, 1);
        }
    }

    let min_x = cmp::max(v[0].x, draw_area_top_left.x);
    let max_x = cmp::min(v[1].x, draw_area_bottom_right.x);
    let min_y = cmp::max(draw_area_top_left.y, cmp::min(v[0].y, v[1].y));
    let max_y = cmp::min(draw_area_bottom_right.y, cmp::max(v[0].y, v[1].y));

    let x_interval = v[1].x - v[0].x;

    let y_step = ((v[1].y - v[0].y) << 16) / x_interval;

    let (r_step, g_step, b_step) = match shading {
        LineShading::Flat(_) => (0.0, 0.0, 0.0),
        LineShading::Gouraud(colors) => gouraud_color_steps(colors, x_interval.into()),
    };

    let y_step_v = _mm_set1_epi32(4 * y_step);
    let r_step_v = _mm256_set1_pd(4.0 * r_step);
    let g_step_v = _mm256_set1_pd(4.0 * g_step);
    let b_step_v = _mm256_set1_pd(4.0 * b_step);

    let first_color = match shading {
        LineShading::Flat(color) | LineShading::Gouraud([color, _]) => color,
    };

    let first_y = (v[0].y << 16) | (1 << 15);
    let mut y =
        _mm_setr_epi32(first_y, first_y + y_step, first_y + 2 * y_step, first_y + 3 * y_step);
    let mut r = first_step_vector(first_color.r.into(), r_step);
    let mut g = first_step_vector(first_color.g.into(), g_step);
    let mut b = first_step_vector(first_color.b.into(), b_step);

    for x in (v[0].x..=v[1].x).step_by(4) {
        let xr = _mm_setr_epi32(x, x + 1, x + 2, x + 3);
        let yr = _mm_srai_epi32::<16>(y);

        let write_mask = _mm_and_si128(
            _mm_and_si128(
                _mm_cmpgt_epi32(xr, _mm_set1_epi32(min_x - 1)),
                _mm_cmplt_epi32(xr, _mm_set1_epi32(max_x + 1)),
            ),
            _mm_and_si128(
                _mm_cmpgt_epi32(yr, _mm_set1_epi32(min_y - 1)),
                _mm_cmplt_epi32(yr, _mm_set1_epi32(max_y + 1)),
            ),
        );

        if _mm_testz_si128(write_mask, _mm_set1_epi32(-1)) == 0 {
            let rr = round_pd_to_epi32(r);
            let gr = round_pd_to_epi32(g);
            let br = round_pd_to_epi32(b);

            rasterize_line_pixels(
                vram,
                xr,
                yr,
                rr,
                gr,
                br,
                write_mask,
                semi_transparency_mode,
                dithering_enabled,
                force_mask_bit,
                check_mask_bit,
            );
        }

        y = _mm_add_epi32(y, y_step_v);
        r = _mm256_add_pd(r, r_step_v);
        g = _mm256_add_pd(g, g_step_v);
        b = _mm256_add_pd(b, b_step_v);
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn rasterize_line_v_oriented(
    vram: &mut AlignedVram,
    mut v: [Vertex; 2],
    draw_area_top_left: Vertex,
    draw_area_bottom_right: Vertex,
    mut shading: LineShading,
    semi_transparency_mode: Option<SemiTransparencyMode>,
    dithering_enabled: bool,
    force_mask_bit: bool,
    check_mask_bit: bool,
) {
    if v[0].y > v[1].y {
        v.swap(0, 1);

        if let LineShading::Gouraud(colors) = &mut shading {
            colors.swap(0, 1);
        }
    }

    let min_x = cmp::max(draw_area_top_left.x, cmp::min(v[0].x, v[1].x));
    let max_x = cmp::min(draw_area_bottom_right.x, cmp::max(v[0].x, v[1].x));
    let min_y = cmp::max(v[0].y, draw_area_top_left.y);
    let max_y = cmp::min(v[1].y, draw_area_bottom_right.y);
    if min_x > max_x || min_y > max_y {
        return;
    }

    let y_interval = v[1].y - v[0].y;

    let x_step = ((v[1].x - v[0].x) << 16) / y_interval;

    let (r_step, g_step, b_step) = match shading {
        LineShading::Flat(_) => (0.0, 0.0, 0.0),
        LineShading::Gouraud(colors) => gouraud_color_steps(colors, y_interval.into()),
    };

    let x_step_v = _mm_set1_epi32(4 * x_step);
    let r_step_v = _mm256_set1_pd(4.0 * r_step);
    let g_step_v = _mm256_set1_pd(4.0 * g_step);
    let b_step_v = _mm256_set1_pd(4.0 * b_step);

    let first_color = match shading {
        LineShading::Flat(color) | LineShading::Gouraud([color, _]) => color,
    };

    let first_x = (v[0].x << 16) | (1 << 15);
    let mut x =
        _mm_setr_epi32(first_x, first_x + x_step, first_x + 2 * x_step, first_x + 3 * x_step);
    let mut r = first_step_vector(first_color.r.into(), r_step);
    let mut g = first_step_vector(first_color.g.into(), g_step);
    let mut b = first_step_vector(first_color.b.into(), b_step);

    for y in (v[0].y..=v[1].y).step_by(4) {
        let xr = _mm_srai_epi32::<16>(x);
        let yr = _mm_setr_epi32(y, y + 1, y + 2, y + 3);

        let write_mask = _mm_and_si128(
            _mm_and_si128(
                _mm_cmpgt_epi32(xr, _mm_set1_epi32(min_x - 1)),
                _mm_cmplt_epi32(xr, _mm_set1_epi32(max_x + 1)),
            ),
            _mm_and_si128(
                _mm_cmpgt_epi32(yr, _mm_set1_epi32(min_y - 1)),
                _mm_cmplt_epi32(yr, _mm_set1_epi32(max_y + 1)),
            ),
        );

        if _mm_testz_si128(write_mask, _mm_set1_epi32(-1)) == 0 {
            let rr = round_pd_to_epi32(r);
            let gr = round_pd_to_epi32(g);
            let br = round_pd_to_epi32(b);

            rasterize_line_pixels(
                vram,
                xr,
                yr,
                rr,
                gr,
                br,
                write_mask,
                semi_transparency_mode,
                dithering_enabled,
                force_mask_bit,
                check_mask_bit,
            );
        }

        x = _mm_add_epi32(x, x_step_v);
        r = _mm256_add_pd(r, r_step_v);
        g = _mm256_add_pd(g, g_step_v);
        b = _mm256_add_pd(b, b_step_v);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn round_pd_to_epi32(pd: __m256d) -> __m128i {
    _mm256_cvtpd_epi32(_mm256_round_pd::<_MM_FROUND_TO_NEAREST_INT>(pd))
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn rasterize_line_pixels(
    vram: &mut AlignedVram,
    x: __m128i,
    y: __m128i,
    r: __m128i,
    g: __m128i,
    b: __m128i,
    write_mask: __m128i,
    semi_transparency_mode: Option<SemiTransparencyMode>,
    dithering_enabled: bool,
    force_mask_bit: bool,
    check_mask_bit: bool,
) {
    let forced_mask_bit = u16::from(force_mask_bit) << 15;

    let x_arr: [i32; 4] = mem::transmute(x);
    let y_arr: [i32; 4] = mem::transmute(y);
    let write_mask_arr: [i32; 4] = mem::transmute(write_mask);

    let r_arr: [i32; 4] = mem::transmute(r);
    let g_arr: [i32; 4] = mem::transmute(g);
    let b_arr: [i32; 4] = mem::transmute(b);

    for i in 0..4 {
        if write_mask_arr[i] == 0 {
            continue;
        }

        let vram_addr = (1024 * y_arr[i] + x_arr[i]) as usize;
        if check_mask_bit && vram[vram_addr] & 0x8000 != 0 {
            continue;
        }

        let (r, g, b) = match semi_transparency_mode {
            Some(mode) => {
                let existing = vram[vram_addr];
                let existing_r: i32 = ((existing & 0x1F) << 3).into();
                let existing_g: i32 = (((existing >> 5) & 0x1F) << 3).into();
                let existing_b: i32 = (((existing >> 10) & 0x1F) << 3).into();

                match mode {
                    SemiTransparencyMode::Average => (
                        (existing_r + r_arr[i]) >> 1,
                        (existing_g + g_arr[i]) >> 1,
                        (existing_b + b_arr[i]) >> 1,
                    ),
                    SemiTransparencyMode::Add => (
                        cmp::min(255, existing_r + r_arr[i]),
                        cmp::min(255, existing_g + g_arr[i]),
                        cmp::min(255, existing_b + b_arr[i]),
                    ),
                    SemiTransparencyMode::Subtract => (
                        cmp::max(0, existing_r - r_arr[i]),
                        cmp::max(0, existing_g - g_arr[i]),
                        cmp::max(0, existing_b - b_arr[i]),
                    ),
                    SemiTransparencyMode::AddQuarter => (
                        cmp::min(255, existing_r + (r_arr[i] >> 2)),
                        cmp::min(255, existing_g + (g_arr[i] >> 2)),
                        cmp::min(255, existing_b + (b_arr[i] >> 2)),
                    ),
                }
            }
            None => (r_arr[i], g_arr[i], b_arr[i]),
        };

        let (r, g, b) = if dithering_enabled {
            let dither_value =
                DITHER_TABLE[(y_arr[i] & 3) as usize][(3 - (x_arr[i] & 3)) as usize] as i8;

            (
                (r as u8).saturating_add_signed(dither_value),
                (g as u8).saturating_add_signed(dither_value),
                (b as u8).saturating_add_signed(dither_value),
            )
        } else {
            (r as u8, g as u8, b as u8)
        };

        vram[vram_addr] = u16::from(r >> 3)
            | (u16::from(g & 0xF8) << 2)
            | (u16::from(b & 0xF8) << 7)
            | forced_mask_bit;
    }
}

fn gouraud_color_steps([c0, c1]: [Color; 2], interval: f64) -> (f64, f64, f64) {
    (
        (f64::from(c1.r) - f64::from(c0.r)) / interval,
        (f64::from(c1.g) - f64::from(c0.g)) / interval,
        (f64::from(c1.b) - f64::from(c0.b)) / interval,
    )
}

#[target_feature(enable = "avx2")]
unsafe fn first_step_vector(first: f64, step: f64) -> __m256d {
    _mm256_setr_pd(first, first + step, first + 2.0 * step, first + 3.0 * step)
}

// Tests don't assert - they're meant to be run with miri to check for undefined behavior:
//   RUSTFLAGS="-C target-feature=+avx2" cargo +nightly miri test gpu
#[cfg(test)]
#[cfg(target_feature = "avx2")]
mod tests {
    use super::*;

    #[test]
    fn untextured_triangle() {
        let mut vram = AlignedVram::new_on_heap();

        unsafe {
            rasterize_triangle(
                &mut vram,
                &DrawSettings {
                    drawing_in_display_allowed: true,
                    dithering_enabled: true,
                    draw_area_top_left: Vertex { x: 0, y: 0 },
                    draw_area_bottom_right: Vertex { x: 1023, y: 511 },
                    draw_offset: Vertex { x: 0, y: 0 },
                    force_mask_bit: false,
                    check_mask_bit: true,
                },
                (450, 550),
                (0, 60),
                [Vertex { x: 500, y: 0 }, Vertex { x: 520, y: 20 }, Vertex { x: 480, y: 20 }],
                TriangleShading::Gouraud([
                    Color::rgb(0, 0, 0),
                    Color::rgb(255, 0, 0),
                    Color::rgb(0, 255, 0),
                ]),
                None,
                Some(SemiTransparencyMode::Add),
            );
        }
    }

    fn textured_triangle(color_depth: TextureColorDepthBits) {
        let mut vram = AlignedVram::new_on_heap();

        unsafe {
            rasterize_triangle(
                &mut vram,
                &DrawSettings {
                    drawing_in_display_allowed: true,
                    dithering_enabled: true,
                    draw_area_top_left: Vertex { x: 0, y: 0 },
                    draw_area_bottom_right: Vertex { x: 1023, y: 511 },
                    draw_offset: Vertex { x: 0, y: 0 },
                    force_mask_bit: false,
                    check_mask_bit: true,
                },
                (450, 550),
                (0, 60),
                [Vertex { x: 500, y: 0 }, Vertex { x: 520, y: 20 }, Vertex { x: 480, y: 20 }],
                TriangleShading::Gouraud([
                    Color::rgb(0, 0, 0),
                    Color::rgb(255, 0, 0),
                    Color::rgb(0, 255, 0),
                ]),
                Some(TriangleTextureMapping {
                    mode: TextureMappingMode::Modulated,
                    texpage: TexturePage {
                        x_base: 0,
                        y_base: 256,
                        semi_transparency_mode: SemiTransparencyMode::Add,
                        color_depth,
                        rectangle_x_flip: false,
                        rectangle_y_flip: false,
                    },
                    window: TextureWindow::default(),
                    clut_x: 1,
                    clut_y: 1,
                    u: [0, 5, 10],
                    v: [20, 10, 0],
                }),
                Some(SemiTransparencyMode::Add),
            );
        }
    }

    #[test]
    fn textured_triangle_15bpp() {
        textured_triangle(TextureColorDepthBits::Fifteen);
    }

    #[test]
    fn textured_triangle_8bpp() {
        textured_triangle(TextureColorDepthBits::Eight);
    }

    #[test]
    fn textured_triangle_4bpp() {
        textured_triangle(TextureColorDepthBits::Four);
    }

    #[test]
    fn untextured_rectangle() {
        let mut vram = AlignedVram::new_on_heap();

        unsafe {
            rasterize_rectangle(
                &mut vram,
                &DrawSettings {
                    drawing_in_display_allowed: true,
                    dithering_enabled: true,
                    draw_area_top_left: Vertex { x: 0, y: 0 },
                    draw_area_bottom_right: Vertex { x: 1023, y: 511 },
                    draw_offset: Vertex { x: 0, y: 0 },
                    force_mask_bit: false,
                    check_mask_bit: true,
                },
                Vertex { x: 450, y: 550 },
                60,
                60,
                Color::rgb(0, 0, 255),
                None,
                Some(SemiTransparencyMode::Add),
            );
        }
    }

    fn textured_rectangle(color_depth: TextureColorDepthBits) {
        let mut vram = AlignedVram::new_on_heap();

        unsafe {
            rasterize_rectangle(
                &mut vram,
                &DrawSettings {
                    drawing_in_display_allowed: true,
                    dithering_enabled: true,
                    draw_area_top_left: Vertex { x: 0, y: 0 },
                    draw_area_bottom_right: Vertex { x: 1023, y: 511 },
                    draw_offset: Vertex { x: 0, y: 0 },
                    force_mask_bit: false,
                    check_mask_bit: true,
                },
                Vertex { x: 450, y: 550 },
                60,
                60,
                Color::rgb(0, 0, 255),
                Some(RectangleTextureMapping {
                    mode: TextureMappingMode::Modulated,
                    texpage: TexturePage {
                        x_base: 0,
                        y_base: 256,
                        semi_transparency_mode: SemiTransparencyMode::Add,
                        color_depth,
                        rectangle_x_flip: false,
                        rectangle_y_flip: false,
                    },
                    window: TextureWindow::default(),
                    clut_x: 1,
                    clut_y: 1,
                    u: [0],
                    v: [20],
                }),
                Some(SemiTransparencyMode::Add),
            );
        }
    }

    #[test]
    fn textured_rectangle_15bpp() {
        textured_rectangle(TextureColorDepthBits::Fifteen);
    }

    #[test]
    fn textured_rectangle_8bpp() {
        textured_rectangle(TextureColorDepthBits::Eight);
    }

    #[test]
    fn textured_rectangle_4bpp() {
        textured_rectangle(TextureColorDepthBits::Four);
    }

    #[test]
    fn line() {
        let mut vram = AlignedVram::new_on_heap();

        unsafe {
            rasterize_line(
                &mut vram,
                [Vertex { x: 10, y: 20 }, Vertex { x: 30, y: 10 }],
                Vertex { x: 0, y: 0 },
                Vertex { x: 1023, y: 511 },
                LineShading::Gouraud([Color::rgb(255, 0, 0), Color::rgb(0, 255, 0)]),
                Some(SemiTransparencyMode::Add),
                true,
                false,
                false,
            );
        }
    }
}
