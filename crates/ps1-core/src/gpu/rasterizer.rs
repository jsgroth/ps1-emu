//! Rasterizer interface and dispatch code

use crate::api::{DisplayConfig, PgxpConfig};
use crate::gpu::gp0::{DrawSettings, SemiTransparencyMode, TexturePage, TextureWindow};
use crate::gpu::rasterizer::naive::NaiveSoftwareRasterizer;
use crate::gpu::rasterizer::simd::SimdSoftwareRasterizer;
use crate::gpu::rasterizer::wgpuhardware::WgpuRasterizer;
use crate::gpu::registers::{Registers, VerticalResolution};
use crate::gpu::{Color, Vertex, VideoMode, Vram, WgpuResources};
use crate::pgxp::PreciseVertex;
use bincode::{Decode, Encode};
use std::cmp;
use std::fmt::{Display, Formatter};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use wgpu::PipelineCompilationOptions;

pub mod naive;
#[cfg(target_arch = "x86_64")]
pub mod simd;
mod software;
pub mod wgpuhardware;

#[cfg(not(target_arch = "x86_64"))]
pub mod simd {
    pub type SimdSoftwareRasterizer = crate::gpu::rasterizer::naive::NaiveSoftwareRasterizer;
}

#[derive(Debug, Clone, Copy)]
pub enum Shading<const N: usize> {
    Flat(Color),
    Gouraud([Color; N]),
}

pub type LineShading = Shading<2>;
pub type TriangleShading = Shading<3>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureMappingMode {
    Raw,
    Modulated,
}

#[derive(Debug, Clone, Copy)]
pub struct TextureMapping<const N: usize> {
    pub mode: TextureMappingMode,
    pub texpage: TexturePage,
    pub window: TextureWindow,
    pub clut_x: u16,
    pub clut_y: u16,
    pub u: [u8; N],
    pub v: [u8; N],
}

pub type TriangleTextureMapping = TextureMapping<3>;
pub type RectangleTextureMapping = TextureMapping<1>;

#[derive(Debug)]
pub struct DrawTriangleArgs {
    pub vertices: [Vertex; 3],
    pub pgxp_vertices: Option<[PreciseVertex; 3]>,
    pub shading: TriangleShading,
    pub semi_transparent: bool,
    pub semi_transparency_mode: SemiTransparencyMode,
    pub texture_mapping: Option<TriangleTextureMapping>,
}

#[derive(Debug)]
pub struct DrawLineArgs {
    pub vertices: [Vertex; 2],
    pub shading: LineShading,
    pub semi_transparent: bool,
    pub semi_transparency_mode: SemiTransparencyMode,
}

#[derive(Debug)]
pub struct DrawRectangleArgs {
    pub top_left: Vertex,
    pub width: u32,
    pub height: u32,
    pub color: Color,
    pub semi_transparent: bool,
    pub semi_transparency_mode: SemiTransparencyMode,
    pub texture_mapping: Option<RectangleTextureMapping>,
}

#[derive(Debug)]
pub struct CpuVramBlitArgs {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub force_mask_bit: bool,
    pub check_mask_bit: bool,
}

#[derive(Debug)]
pub struct VramVramBlitArgs {
    pub source_x: u32,
    pub source_y: u32,
    pub dest_x: u32,
    pub dest_y: u32,
    pub width: u32,
    pub height: u32,
    pub force_mask_bit: bool,
    pub check_mask_bit: bool,
}

pub trait RasterizerInterface {
    fn draw_triangle(&mut self, args: DrawTriangleArgs, draw_settings: &DrawSettings);

    fn draw_line(&mut self, args: DrawLineArgs, draw_settings: &DrawSettings);

    fn draw_rectangle(&mut self, args: DrawRectangleArgs, draw_settings: &DrawSettings);

    fn vram_fill(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color);

    fn cpu_to_vram_blit(&mut self, args: CpuVramBlitArgs, data: &[u16]);

    fn vram_to_cpu_blit(&mut self, x: u32, y: u32, width: u32, height: u32, out: &mut Vec<u16>);

    fn vram_to_vram_blit(&mut self, args: VramVramBlitArgs);

    fn generate_frame_texture(
        &mut self,
        registers: &Registers,
        wgpu_resources: &mut WgpuResources,
    ) -> &wgpu::Texture;

    fn clone_vram(&mut self) -> Vram;

    fn clear_texture_cache(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum RasterizerType {
    NaiveSoftware,
    SimdSoftware,
    WgpuHardware,
}

impl Default for RasterizerType {
    fn default() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Self::SimdSoftware;
            }
        }

        Self::NaiveSoftware
    }
}

pub struct Rasterizer(pub Box<dyn RasterizerInterface + Send + Sync>);

impl Deref for Rasterizer {
    type Target = Box<dyn RasterizerInterface + Send + Sync>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Rasterizer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Rasterizer {
    pub fn new(
        wgpu_device: &Arc<wgpu::Device>,
        wgpu_queue: &Arc<wgpu::Queue>,
        display_config: DisplayConfig,
        pgxp_config: PgxpConfig,
    ) -> Self {
        match display_config.rasterizer_type {
            RasterizerType::NaiveSoftware => {
                Self(Box::new(NaiveSoftwareRasterizer::new(wgpu_device)))
            }
            RasterizerType::SimdSoftware => {
                Self(Box::new(SimdSoftwareRasterizer::new(wgpu_device)))
            }
            RasterizerType::WgpuHardware => Self(Box::new(WgpuRasterizer::new(
                Arc::clone(wgpu_device),
                Arc::clone(wgpu_queue),
                display_config.to_wgpu_rasterizer_config(),
                pgxp_config,
            ))),
        }
    }

    pub fn save_state(&mut self) -> RasterizerState {
        let vram = self.clone_vram();
        RasterizerState { vram }
    }

    pub fn from_state(
        state: RasterizerState,
        wgpu_device: &Arc<wgpu::Device>,
        wgpu_queue: &Arc<wgpu::Queue>,
        display_config: DisplayConfig,
        pgxp_config: PgxpConfig,
    ) -> Self {
        match display_config.rasterizer_type {
            RasterizerType::NaiveSoftware => {
                Self(Box::new(NaiveSoftwareRasterizer::from_vram(wgpu_device, &state.vram)))
            }
            RasterizerType::SimdSoftware => {
                Self(Box::new(SimdSoftwareRasterizer::from_vram(wgpu_device, &state.vram)))
            }
            RasterizerType::WgpuHardware => {
                let rasterizer = WgpuRasterizer::new(
                    Arc::clone(wgpu_device),
                    Arc::clone(wgpu_queue),
                    display_config.to_wgpu_rasterizer_config(),
                    pgxp_config,
                );
                rasterizer.copy_vram_from(&state.vram);
                Self(Box::new(rasterizer))
            }
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct RasterizerState {
    pub vram: Vram,
}

impl DrawSettings {
    fn is_drawing_area_valid(&self) -> bool {
        self.draw_area_top_left.x <= self.draw_area_bottom_right.x
            && self.draw_area_top_left.y <= self.draw_area_bottom_right.y
    }

    fn drawing_area_contains_vertex(&self, vertex: Vertex) -> bool {
        (self.draw_area_top_left.x..=self.draw_area_bottom_right.x).contains(&vertex.x)
            && (self.draw_area_top_left.y..=self.draw_area_bottom_right.y).contains(&vertex.y)
    }
}

fn vertices_valid(v0: Vertex, v1: Vertex) -> bool {
    // The GPU will not render any lines or polygons where the distance between any two vertices is
    // larger than 1023 horizontally or 511 vertically
    (v0.x - v1.x).abs() < 1024 && (v0.y - v1.y).abs() < 512
}

fn swap_vertices(
    vertices: &mut [Vertex; 3],
    shading: &mut TriangleShading,
    texture_mapping: Option<&mut TriangleTextureMapping>,
) {
    vertices.swap(0, 1);

    if let Some(texture_mapping) = texture_mapping {
        texture_mapping.u.swap(0, 1);
        texture_mapping.v.swap(0, 1);
    }

    if let TriangleShading::Gouraud(colors) = shading {
        colors.swap(0, 1);
    }
}

// Z component of the cross product between v0->v1 and v0->v2
fn cross_product_z(v0: Vertex, v1: Vertex, v2: Vertex) -> i32 {
    (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x)
}

struct ScreenSize {
    left: i32,
    right: i32,
    top: i32,
    bottom: i32,
    v_overscan_rows: i32,
}

impl ScreenSize {
    const NTSC: Self = Self { left: 0x260, right: 0xC60, top: 16, bottom: 256, v_overscan_rows: 8 };

    const PAL: Self = Self { left: 0x274, right: 0xC74, top: 20, bottom: 308, v_overscan_rows: 10 };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameCoords {
    // Frame buffer coordinates
    frame_x: u32,
    frame_y: u32,
    // Pixel offsets to apply to the frame buffer coordinates (if X1/Y1 are less than standard values)
    display_x_offset: u32,
    display_y_offset: u32,
    // First X/Y pixel values to display (if X1/Y1 are greater than standard values)
    display_x_start: u32,
    display_y_start: u32,
    // Number of pixels to render from the frame buffer in each dimension
    display_width: u32,
    display_height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct FrameSize {
    width: u32,
    height: u32,
}

impl Display for FrameSize {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ width: {}, height: {} }}", self.width, self.height)
    }
}

fn compute_frame_location(
    registers: &Registers,
    display_config: DisplayConfig,
) -> (Option<FrameCoords>, FrameSize) {
    let crop_v_overscan = display_config.crop_vertical_overscan;
    let screen_size = match registers.video_mode {
        VideoMode::Ntsc => ScreenSize::NTSC,
        VideoMode::Pal => ScreenSize::PAL,
    };

    let screen_top = if crop_v_overscan {
        screen_size.top + screen_size.v_overscan_rows
    } else {
        screen_size.top
    };
    let screen_bottom = if crop_v_overscan {
        screen_size.bottom - screen_size.v_overscan_rows
    } else {
        screen_size.bottom
    };

    let dot_clock_divider: i32 = registers.dot_clock_divider().into();
    let frame_width = (screen_size.right - screen_size.left) / dot_clock_divider;

    let height_multipler =
        if registers.interlaced && registers.v_resolution == VerticalResolution::Double {
            2
        } else {
            1
        };
    let frame_height = (if crop_v_overscan {
        screen_size.bottom - screen_size.top - 2 * screen_size.v_overscan_rows
    } else {
        screen_size.bottom - screen_size.top
    }) * height_multipler;

    let frame_size = FrameSize { width: frame_width as u32, height: frame_height as u32 };

    let x1 = registers.x_display_range.0 as i32;
    let x2 = cmp::min(screen_size.right, registers.x_display_range.1 as i32);
    let y1 = registers.y_display_range.0 as i32;
    let y2 = cmp::min(screen_bottom, registers.y_display_range.1 as i32);

    let mut display_width_clocks = x2 - x1;
    let mut display_height = (y2 - y1) * height_multipler;

    let mut display_x_offset = 0;
    if x1 < screen_size.left {
        display_x_offset = (screen_size.left - x1) / dot_clock_divider;
        display_width_clocks -= screen_size.left - x1;
    }

    let mut display_y_offset = 0;
    if y1 < screen_top {
        display_y_offset = (screen_top - y1) * height_multipler;
        display_height -= (screen_top - y1) * height_multipler;
    }

    let mut display_width = display_width_clocks / dot_clock_divider;
    if display_width <= 0 || display_height <= 0 {
        return (None, frame_size);
    }

    let display_x_start = cmp::max(0, (x1 - screen_size.left) / dot_clock_divider);
    let display_y_start = cmp::max(0, (y1 - screen_top) * height_multipler);

    // Clamp display width in case of errors caused by dot clock division
    display_width = cmp::min(display_width, frame_width - display_x_start);

    assert!(
        display_y_start + display_height <= frame_height,
        "Vertical display range: {display_y_start} + {display_height} <= {frame_height}"
    );

    (
        Some(FrameCoords {
            frame_x: registers.display_area_x,
            frame_y: registers.display_area_y,
            display_x_offset: display_x_offset as u32,
            display_y_offset: display_y_offset as u32,
            display_x_start: display_x_start as u32,
            display_y_start: display_y_start as u32,
            display_width: display_width as u32,
            display_height: display_height as u32,
        }),
        frame_size,
    )
}

#[derive(Debug)]
struct ClearPipeline {
    pipeline: wgpu::RenderPipeline,
}

impl ClearPipeline {
    fn new(device: &wgpu::Device, frame_format: wgpu::TextureFormat) -> Self {
        let clear_module =
            device.create_shader_module(wgpu::include_wgsl!("rasterizer/clear.wgsl"));

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: "clear_pipeline".into(),
            layout: None,
            vertex: wgpu::VertexState {
                module: &clear_module,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &clear_module,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: frame_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }

    fn draw(&self, frame: &wgpu::Texture, encoder: &mut wgpu::CommandEncoder) {
        let frame_view = frame.create_view(&wgpu::TextureViewDescriptor::default());

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: "clear_render_pass".into(),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..wgpu::RenderPassDescriptor::default()
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..4, 0..1);
    }
}
