//! A naive software rasterizer implementation. Very very slow

#![allow(clippy::many_single_char_names)]

use crate::gpu::gp0::{
    DrawSettings, SemiTransparencyMode, TextureColorDepthBits, TexturePage, TextureWindow,
};
use crate::gpu::rasterizer::software::SoftwareRenderer;
use crate::gpu::rasterizer::{
    CpuVramBlitArgs, DrawLineArgs, DrawRectangleArgs, DrawTriangleArgs, LineShading,
    RasterizerInterface, RectangleTextureMapping, TextureMappingMode, TriangleShading,
    TriangleTextureMapping, VramVramBlitArgs, cross_product_z, software, swap_vertices,
    vertices_valid,
};
use crate::gpu::registers::Registers;
use crate::gpu::{Color, Vertex, Vram, VramArray, WgpuResources};
use std::cmp;
use wgpu::Texture;

const DITHER_TABLE: &[[i8; 4]; 4] =
    &[[-4, 0, -3, 1], [2, -2, 3, -1], [-3, 1, -4, 0], [3, -1, 2, -2]];

impl Color {
    pub(super) fn from_15_bit(color: u16) -> Self {
        let r = (color & 0x1F) << 3;
        let g = ((color >> 5) & 0x1F) << 3;
        let b = ((color >> 10) & 0x1F) << 3;
        Self::rgb(r as u8, g as u8, b as u8)
    }

    pub(super) fn truncate_to_15_bit(self) -> u16 {
        let r: u16 = (self.r >> 3).into();
        let g: u16 = (self.g >> 3).into();
        let b: u16 = (self.b >> 3).into();

        r | (g << 5) | (b << 10)
    }

    pub(super) fn dither(self, dither_value: i8) -> Self {
        Self {
            r: self.r.saturating_add_signed(dither_value),
            g: self.g.saturating_add_signed(dither_value),
            b: self.b.saturating_add_signed(dither_value),
        }
    }
}

impl SemiTransparencyMode {
    fn apply(self, back: u16, front: u16) -> u16 {
        match self {
            Self::Average => apply_semi_transparency(back, front, u16::midpoint),
            Self::Add => apply_semi_transparency(back, front, |b, f| cmp::min(31, b + f)),
            Self::Subtract => apply_semi_transparency(back, front, |b, f| {
                cmp::max(0, (b as i16) - (f as i16)) as u16
            }),
            Self::AddQuarter => {
                apply_semi_transparency(back, front, |b, f| cmp::min(31, b + (f / 4)))
            }
        }
    }
}

fn apply_semi_transparency<F>(back: u16, front: u16, op: F) -> u16
where
    F: Fn(u16, u16) -> u16,
{
    let r = op(back & 0x1F, front & 0x1F);
    let g = op((back >> 5) & 0x1F, (front >> 5) & 0x1F);
    let b = op((back >> 10) & 0x1F, (front >> 10) & 0x1F);
    r | (g << 5) | (b << 10)
}

#[derive(Debug)]
pub struct NaiveSoftwareRasterizer {
    vram: Vram,
    renderer: SoftwareRenderer,
}

impl NaiveSoftwareRasterizer {
    pub fn new(device: &wgpu::Device) -> Self {
        Self { vram: Vram::new(), renderer: SoftwareRenderer::new(device) }
    }

    pub fn from_vram(device: &wgpu::Device, vram: &Vram) -> Self {
        let vram_array: Box<VramArray> = vram.to_vec().into_boxed_slice().try_into().unwrap();
        Self { vram: vram_array.into(), renderer: SoftwareRenderer::new(device) }
    }
}

#[derive(Debug, Clone)]
struct Interpolator {
    base_vertex: Vertex,
    base_color: Color,
    base_tex_coords: (u8, u8),
    color_x_steps: (i32, i32, i32),
    color_y_steps: (i32, i32, i32),
    tex_x_steps: (i32, i32),
    tex_y_steps: (i32, i32),
}

impl Interpolator {
    // PS1 GPU appears to use fixed-point decimal with 12 fractional bits
    // U/V interpolation does not look correct otherwise
    const SHIFT: u8 = 12;

    fn new(
        v: [Vertex; 3],
        shading: TriangleShading,
        texture_mapping: Option<&TriangleTextureMapping>,
    ) -> Self {
        // Interpolate from the "first" vertex, sorted by X then Y
        let (base_vertex_idx, base_vertex) = v
            .into_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.x.cmp(&b.x).then(a.y.cmp(&b.y)))
            .unwrap();

        let denominator = cross_product_z(v[0], v[1], v[2]);
        assert_ne!(denominator, 0);

        let (base_color, color_x_steps, color_y_steps) = match shading {
            TriangleShading::Flat(color) => (color, (0, 0, 0), (0, 0, 0)),
            TriangleShading::Gouraud(colors) => {
                let base_color = colors[base_vertex_idx];

                let r = colors.map(|color| color.r);
                let g = colors.map(|color| color.g);
                let b = colors.map(|color| color.b);

                let x_steps = (
                    compute_x_step(v, r, denominator),
                    compute_x_step(v, g, denominator),
                    compute_x_step(v, b, denominator),
                );
                let y_steps = (
                    compute_y_step(v, r, denominator),
                    compute_y_step(v, g, denominator),
                    compute_y_step(v, b, denominator),
                );

                (base_color, x_steps, y_steps)
            }
        };

        let (base_tex_coords, tex_x_steps, tex_y_steps) = match texture_mapping {
            Some(mapping) => {
                let base_coords = (mapping.u[base_vertex_idx], mapping.v[base_vertex_idx]);

                let x_steps = (
                    compute_x_step(v, mapping.u, denominator),
                    compute_x_step(v, mapping.v, denominator),
                );
                let y_steps = (
                    compute_y_step(v, mapping.u, denominator),
                    compute_y_step(v, mapping.v, denominator),
                );

                (base_coords, x_steps, y_steps)
            }
            None => ((0, 0), (0, 0), (0, 0)),
        };

        Self {
            base_vertex,
            base_color,
            base_tex_coords,
            color_x_steps,
            color_y_steps,
            tex_x_steps,
            tex_y_steps,
        }
    }

    fn interpolate_color(&self, p: Vertex) -> Color {
        let r = interpolate_component(
            p,
            self.base_vertex,
            self.base_color.r,
            self.color_x_steps.0,
            self.color_y_steps.0,
        );
        let g = interpolate_component(
            p,
            self.base_vertex,
            self.base_color.g,
            self.color_x_steps.1,
            self.color_y_steps.1,
        );
        let b = interpolate_component(
            p,
            self.base_vertex,
            self.base_color.b,
            self.color_x_steps.2,
            self.color_y_steps.2,
        );

        Color { r, g, b }
    }

    fn interpolate_uv(&self, p: Vertex) -> (u8, u8) {
        let u = interpolate_component(
            p,
            self.base_vertex,
            self.base_tex_coords.0,
            self.tex_x_steps.0,
            self.tex_y_steps.0,
        );
        let v = interpolate_component(
            p,
            self.base_vertex,
            self.base_tex_coords.1,
            self.tex_x_steps.1,
            self.tex_y_steps.1,
        );

        (u, v)
    }
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

fn interpolate_component(
    p: Vertex,
    base_vertex: Vertex,
    base_component: u8,
    x_step: i32,
    y_step: i32,
) -> u8 {
    let dx = p.x - base_vertex.x;
    let dy = p.y - base_vertex.y;

    let base_component = i32::from(base_component) << Interpolator::SHIFT;
    let shifted_value = base_component + x_step * dx + y_step * dy;
    let value = (shifted_value + (1 << (Interpolator::SHIFT - 1))) >> Interpolator::SHIFT;

    value as u8
}

impl RasterizerInterface for NaiveSoftwareRasterizer {
    fn draw_triangle(
        &mut self,
        DrawTriangleArgs {
            vertices: mut v,
            mut shading,
            semi_transparent,
            semi_transparency_mode,
            mut texture_mapping,
            ..
        }: DrawTriangleArgs,
        draw_settings: &DrawSettings,
    ) {
        if !draw_settings.is_drawing_area_valid() {
            return;
        }

        if !vertices_valid(v[0], v[1]) || !vertices_valid(v[1], v[2]) || !vertices_valid(v[0], v[2])
        {
            return;
        }

        // Determine if the vertices are in clockwise order; if not, swap the first 2
        let double_area = cross_product_z(v[0], v[1], v[2]);
        if double_area < 0 {
            swap_vertices(&mut v, &mut shading, texture_mapping.as_mut());
        }

        // If vertices are collinear, draw nothing
        if double_area == 0 {
            return;
        }

        log::trace!("Triangle vertices: {v:?}");

        // Compute bounding box
        let min_x = cmp::min(v[0].x, cmp::min(v[1].x, v[2].x));
        let max_x = cmp::max(v[0].x, cmp::max(v[1].x, v[2].x));
        let min_y = cmp::min(v[0].y, cmp::min(v[1].y, v[2].y));
        let max_y = cmp::max(v[0].y, cmp::max(v[1].y, v[2].y));

        log::trace!("Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})");

        let interpolator = Interpolator::new(v, shading, texture_mapping.as_ref());

        // Iterate over every pixel in the bounding box to determine which ones to rasterize
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                draw_triangle_pixel(
                    px,
                    py,
                    draw_settings,
                    &interpolator,
                    &mut self.vram,
                    DrawTrianglePixelArgs {
                        v,
                        shading,
                        semi_transparent,
                        semi_transparency_mode,
                        texture_mapping,
                    },
                );
            }
        }
    }

    fn draw_line(
        &mut self,
        DrawLineArgs { vertices, shading, semi_transparent, semi_transparency_mode }: DrawLineArgs,
        draw_settings: &DrawSettings,
    ) {
        if !draw_settings.is_drawing_area_valid() {
            return;
        }

        if !vertices_valid(vertices[0], vertices[1]) {
            return;
        }

        // Apply drawing offset
        let vertices = vertices.map(|vertex| Vertex {
            x: vertex.x + draw_settings.draw_offset.x,
            y: vertex.y + draw_settings.draw_offset.y,
        });

        let x_diff = vertices[1].x - vertices[0].x;
        let y_diff = vertices[1].y - vertices[0].y;

        if x_diff == 0 && y_diff == 0 {
            // Draw a single pixel with the color of the first vertex (if it's inside the drawing area)
            let color = match shading {
                LineShading::Flat(color) | LineShading::Gouraud([color, _]) => color,
            };
            draw_line_pixel(
                vertices[0],
                color,
                semi_transparent,
                semi_transparency_mode,
                draw_settings,
                &mut self.vram,
            );
            return;
        }

        let (r_diff, g_diff, b_diff) = match shading {
            LineShading::Flat(_) => (0, 0, 0),
            LineShading::Gouraud([color0, color1]) => (
                i32::from(color1.r) - i32::from(color0.r),
                i32::from(color1.g) - i32::from(color0.g),
                i32::from(color1.b) - i32::from(color0.b),
            ),
        };

        let (x_step, y_step, r_step, g_step, b_step) = if x_diff.abs() >= y_diff.abs() {
            let y_step = f64::from(y_diff) / f64::from(x_diff.abs());
            let r_step = f64::from(r_diff) / f64::from(x_diff.abs());
            let g_step = f64::from(g_diff) / f64::from(x_diff.abs());
            let b_step = f64::from(b_diff) / f64::from(x_diff.abs());
            (f64::from(x_diff.signum()), y_step, r_step, g_step, b_step)
        } else {
            let x_step = f64::from(x_diff) / f64::from(y_diff.abs());
            let r_step = f64::from(r_diff) / f64::from(y_diff.abs());
            let g_step = f64::from(g_diff) / f64::from(y_diff.abs());
            let b_step = f64::from(b_diff) / f64::from(y_diff.abs());
            (x_step, f64::from(y_diff.signum()), r_step, g_step, b_step)
        };

        let first_color = match shading {
            LineShading::Flat(color) | LineShading::Gouraud([color, _]) => color,
        };
        let mut r = f64::from(first_color.r);
        let mut g = f64::from(first_color.g);
        let mut b = f64::from(first_color.b);

        let mut x = f64::from(vertices[0].x);
        let mut y = f64::from(vertices[0].y);
        while x.round() as i32 != vertices[1].x || y.round() as i32 != vertices[1].y {
            let vertex = Vertex { x: x.round() as i32, y: y.round() as i32 };
            let color = Color::rgb(r.round() as u8, g.round() as u8, b.round() as u8);
            draw_line_pixel(
                vertex,
                color,
                semi_transparent,
                semi_transparency_mode,
                draw_settings,
                &mut self.vram,
            );

            x += x_step;
            y += y_step;
            r += r_step;
            g += g_step;
            b += b_step;
        }

        // Draw the last pixel
        let color = Color::rgb(r.round() as u8, g.round() as u8, b.round() as u8);
        draw_line_pixel(
            vertices[1],
            color,
            semi_transparent,
            semi_transparency_mode,
            draw_settings,
            &mut self.vram,
        );
    }

    fn draw_rectangle(
        &mut self,
        DrawRectangleArgs {
            top_left,
            width,
            height,
            color,
            semi_transparent,
            semi_transparency_mode,
            texture_mapping,
        }: DrawRectangleArgs,
        draw_settings: &DrawSettings,
    ) {
        if !draw_settings.is_drawing_area_valid() || width == 0 || height == 0 {
            return;
        }

        let args = SoftwareRectangleArgs {
            top_left,
            width: width as i32,
            height: height as i32,
            draw_area_top_left: draw_settings.draw_area_top_left,
            draw_area_bottom_right: draw_settings.draw_area_bottom_right,
            draw_offset: draw_settings.draw_offset,
            color,
            semi_transparent,
            semi_transparency_mode,
            force_mask_bit: draw_settings.force_mask_bit,
            check_mask_bit: draw_settings.check_mask_bit,
        };
        match texture_mapping {
            None => draw_solid_rectangle(args, &mut self.vram),
            Some(texture_mapping) => draw_textured_rectangle(args, texture_mapping, &mut self.vram),
        }
    }

    fn vram_fill(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color) {
        software::vram_fill(&mut self.vram, x, y, width, height, color);
    }

    fn cpu_to_vram_blit(&mut self, args: CpuVramBlitArgs, data: &[u16]) {
        software::cpu_to_vram_blit(&mut self.vram, args, data);
    }

    fn vram_to_cpu_blit(&mut self, x: u32, y: u32, width: u32, height: u32, out: &mut Vec<u16>) {
        software::vram_to_cpu_blit(&self.vram, x, y, width, height, out);
    }

    fn vram_to_vram_blit(&mut self, args: VramVramBlitArgs) {
        software::vram_to_vram_blit(&mut self.vram, args);
    }

    fn generate_frame_texture(
        &mut self,
        registers: &Registers,
        wgpu_resources: &mut WgpuResources,
    ) -> &Texture {
        self.renderer.generate_frame_texture(registers, wgpu_resources, &self.vram)
    }

    fn clone_vram(&mut self) -> Vram {
        self.vram.clone()
    }
}

struct DrawTrianglePixelArgs {
    v: [Vertex; 3],
    shading: TriangleShading,
    semi_transparent: bool,
    semi_transparency_mode: SemiTransparencyMode,
    texture_mapping: Option<TriangleTextureMapping>,
}

fn draw_triangle_pixel(
    px: i32,
    py: i32,
    draw_settings: &DrawSettings,
    interpolator: &Interpolator,
    vram: &mut Vram,
    DrawTrianglePixelArgs {
        v,
        shading,
        semi_transparent,
        semi_transparency_mode,
        texture_mapping,
    }: DrawTrianglePixelArgs,
) {
    let px_offset = i11(px + draw_settings.draw_offset.x);
    let py_offset = i11(py + draw_settings.draw_offset.y);
    let p_offset = Vertex { x: px_offset, y: py_offset };
    if !draw_settings.drawing_area_contains_vertex(p_offset) {
        return;
    }

    let vram_addr = (1024 * py_offset + px_offset) as usize;
    if draw_settings.check_mask_bit && vram[vram_addr] & 0x8000 != 0 {
        return;
    }

    let p = Vertex { x: px, y: py };

    // A given point is contained within the triangle if the Z component of the cross-product of
    // v0->p and v0->v1 is non-negative for each edge v0->v1 (assuming the vertices are ordered
    // clockwise)
    for edge in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
        let cpz = cross_product_z(edge.0, edge.1, p);
        if cpz < 0 {
            return;
        }

        // If the cross product is 0, the point is collinear with these two vertices.
        // The PS1 GPU does not draw edges on the bottom of the triangle when this happens,
        // nor does it draw a vertical right edge
        if cpz == 0 {
            // Since the vertices are clockwise, increasing Y means this edge is on the
            // bottom or right of the triangle.
            if edge.1.y > edge.0.y {
                return;
            }

            // If the Y values are equal and X is decreasing, this is a horizontal bottom edge
            if edge.1.y == edge.0.y && edge.1.x < edge.0.x {
                return;
            }
        }
    }

    let shading_color = interpolator.interpolate_color(p);

    let (textured_color, mask_bit) = match &texture_mapping {
        None => (shading_color, false),
        Some(texture_mapping) => {
            let (tex_u, tex_v) = interpolator.interpolate_uv(p);

            let texture_pixel = sample_texture(
                vram,
                &texture_mapping.texpage,
                &texture_mapping.window,
                texture_mapping.clut_x.into(),
                texture_mapping.clut_y.into(),
                tex_u.into(),
                tex_v.into(),
            );
            if texture_pixel == 0x0000 {
                // Pixel values of $0000 are fully transparent and are not written to VRAM
                return;
            }

            let raw_texture_color = Color::from_15_bit(texture_pixel);

            let texture_color = match texture_mapping.mode {
                TextureMappingMode::Raw => raw_texture_color,
                TextureMappingMode::Modulated => modulate_color(raw_texture_color, shading_color),
            };

            (texture_color, texture_pixel & 0x8000 != 0)
        }
    };

    // Dithering is applied if the dither flag is set and either Gouraud shading or texture
    // modulation is used
    let dithered_color = if draw_settings.dithering_enabled
        && (matches!(shading, TriangleShading::Gouraud(..))
            || texture_mapping.as_ref().is_some_and(|texture_mapping| {
                texture_mapping.mode == TextureMappingMode::Modulated
            })) {
        let dither_value = DITHER_TABLE[(py & 3) as usize][(px & 3) as usize];
        textured_color.dither(dither_value)
    } else {
        textured_color
    };

    let truncated_color = dithered_color.truncate_to_15_bit();

    let blended_color = if semi_transparent && (texture_mapping.is_none() || mask_bit) {
        let existing_pixel = vram[vram_addr];

        let semi_transparency_mode = match &texture_mapping {
            None => semi_transparency_mode,
            Some(texture_mapping) => texture_mapping.texpage.semi_transparency_mode,
        };

        semi_transparency_mode.apply(existing_pixel, truncated_color)
    } else {
        truncated_color
    };

    vram[vram_addr] = blended_color | (u16::from(mask_bit || draw_settings.force_mask_bit) << 15);
}

pub(super) fn draw_line_pixel(
    v: Vertex,
    raw_color: Color,
    semi_transparency: bool,
    semi_transparency_mode: SemiTransparencyMode,
    draw_settings: &DrawSettings,
    vram: &mut Vram,
) {
    if !draw_settings.drawing_area_contains_vertex(v) {
        return;
    }

    let vram_addr = (1024 * v.y + v.x) as usize;
    if draw_settings.check_mask_bit && vram[vram_addr] & 0x8000 != 0 {
        return;
    }

    let dithered_color = if draw_settings.dithering_enabled {
        let dither_value = DITHER_TABLE[(v.y & 3) as usize][(v.x & 3) as usize];
        raw_color.dither(dither_value)
    } else {
        raw_color
    };

    let color = if semi_transparency {
        let existing_color = vram[vram_addr];
        semi_transparency_mode.apply(existing_color, dithered_color.truncate_to_15_bit())
    } else {
        dithered_color.truncate_to_15_bit()
    };

    vram[vram_addr] = color | (u16::from(draw_settings.force_mask_bit) << 15);
}

struct SoftwareRectangleArgs {
    top_left: Vertex,
    width: i32,
    height: i32,
    draw_area_top_left: Vertex,
    draw_area_bottom_right: Vertex,
    draw_offset: Vertex,
    color: Color,
    semi_transparent: bool,
    semi_transparency_mode: SemiTransparencyMode,
    force_mask_bit: bool,
    check_mask_bit: bool,
}

fn draw_solid_rectangle(
    SoftwareRectangleArgs {
        top_left,
        width,
        height,
        draw_area_top_left,
        draw_area_bottom_right,
        draw_offset,
        color,
        semi_transparent,
        semi_transparency_mode,
        force_mask_bit,
        check_mask_bit,
    }: SoftwareRectangleArgs,
    vram: &mut Vram,
) {
    let forced_mask_bit = u16::from(force_mask_bit) << 15;

    for dy in 0..height {
        let y = i11(top_left.y + dy + draw_offset.y);
        if y < draw_area_top_left.y || y > draw_area_bottom_right.y {
            continue;
        }

        for dx in 0..width {
            let x = i11(top_left.x + dx + draw_offset.x);
            if x < draw_area_top_left.x || x > draw_area_bottom_right.x {
                continue;
            }

            let vram_addr = (1024 * y + x) as usize;
            if check_mask_bit && vram[vram_addr] & 0x8000 != 0 {
                continue;
            }

            let color = if semi_transparent {
                let existing_color = vram[vram_addr];
                semi_transparency_mode.apply(existing_color, color.truncate_to_15_bit())
            } else {
                color.truncate_to_15_bit()
            };

            vram[vram_addr] = color | forced_mask_bit;
        }
    }
}

fn draw_textured_rectangle(
    SoftwareRectangleArgs {
        top_left,
        width,
        height,
        draw_area_top_left,
        draw_area_bottom_right,
        draw_offset,
        color: rectangle_color,
        semi_transparent,
        semi_transparency_mode,
        force_mask_bit,
        check_mask_bit,
    }: SoftwareRectangleArgs,
    texture_mapping: RectangleTextureMapping,
    vram: &mut Vram,
) {
    let base_u = texture_mapping.u[0];
    let base_v = texture_mapping.v[0];

    for dy in 0..height {
        let y = i11(top_left.y + dy + draw_offset.y);
        if y < draw_area_top_left.y || y > draw_area_bottom_right.y {
            continue;
        }

        let v = base_v.wrapping_add(dy as u8);
        for dx in 0..width {
            let x = i11(top_left.x + dx + draw_offset.x);
            if x < draw_area_top_left.x || x > draw_area_bottom_right.x {
                continue;
            }

            let vram_addr = (1024 * y + x) as usize;
            if check_mask_bit && vram[vram_addr] & 0x8000 != 0 {
                continue;
            }

            let u = base_u.wrapping_add(dx as u8);
            let texture_pixel = sample_texture(
                vram,
                &texture_mapping.texpage,
                &texture_mapping.window,
                texture_mapping.clut_x.into(),
                texture_mapping.clut_y.into(),
                u.into(),
                v.into(),
            );
            if texture_pixel == 0x0000 {
                continue;
            }

            let raw_texture_color = Color::from_15_bit(texture_pixel);

            let texture_color = match texture_mapping.mode {
                TextureMappingMode::Raw => raw_texture_color,
                TextureMappingMode::Modulated => modulate_color(raw_texture_color, rectangle_color),
            };

            let texture_mask_bit = texture_pixel & 0x8000 != 0;
            let masked_color = if semi_transparent && texture_mask_bit {
                let existing_color = vram[vram_addr];
                semi_transparency_mode.apply(existing_color, texture_color.truncate_to_15_bit())
            } else {
                texture_color.truncate_to_15_bit()
            };

            vram[vram_addr] = masked_color | (u16::from(texture_mask_bit | force_mask_bit) << 15);
        }
    }
}

fn modulate(texture_color: u8, shading_color: u8) -> u8 {
    cmp::min(255, u16::from(texture_color) * u16::from(shading_color) / 128) as u8
}

fn modulate_color(texture_color: Color, shading_color: Color) -> Color {
    Color {
        r: modulate(texture_color.r, shading_color.r),
        g: modulate(texture_color.g, shading_color.g),
        b: modulate(texture_color.b, shading_color.b),
    }
}

fn sample_texture(
    vram: &Vram,
    texpage: &TexturePage,
    texture_window: &TextureWindow,
    clut_x: u32,
    clut_y: u32,
    u: u32,
    v: u32,
) -> u16 {
    let u = (u & !(texture_window.x_mask << 3))
        | ((texture_window.x_offset & texture_window.x_mask) << 3);
    let v = (v & !(texture_window.y_mask << 3))
        | ((texture_window.y_offset & texture_window.y_mask) << 3);

    let y = texpage.y_base + v;

    match texpage.color_depth {
        TextureColorDepthBits::Four => {
            let vram_addr = 1024 * y + 64 * texpage.x_base + u / 4;
            let shift = 4 * (u % 4);
            let clut_index: u32 = ((vram[vram_addr as usize] >> shift) & 0xF).into();

            let clut_base_addr = 1024 * clut_y + 16 * clut_x;
            let clut_addr = clut_base_addr + clut_index;

            vram[clut_addr as usize]
        }
        TextureColorDepthBits::Eight => {
            let vram_x_bytes = (64 * texpage.x_base + u / 2) & 0x3FF;
            let vram_addr = 1024 * y + vram_x_bytes;
            let shift = 8 * (u % 2);
            let clut_index: u32 = ((vram[vram_addr as usize] >> shift) & 0xFF).into();

            let clut_base_addr = 1024 * clut_y + 16 * clut_x;
            let clut_addr = clut_base_addr + clut_index;

            vram[clut_addr as usize]
        }
        TextureColorDepthBits::Fifteen => {
            let vram_x_pixels = (64 * texpage.x_base + u) & 0x3FF;
            let vram_addr = (1024 * y + vram_x_pixels) as usize;
            vram[vram_addr]
        }
    }
}

fn i11(value: i32) -> i32 {
    (value << 21) >> 21
}
