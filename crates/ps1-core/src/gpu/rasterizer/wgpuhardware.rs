mod blit;
mod draw;
mod hazards;
mod sync;
mod twentyfour;

use crate::api::ColorDepthBits;
use crate::gpu::gp0::{DrawSettings, SemiTransparencyMode, TextureColorDepthBits, TexturePage};
use crate::gpu::rasterizer::wgpuhardware::blit::{
    CpuVramBlitPipeline, VramCopyPipeline, VramCpuBlitter, VramFillPipeline,
};
use crate::gpu::rasterizer::wgpuhardware::draw::{DrawPipelines, MaskBitPipelines};
use crate::gpu::rasterizer::wgpuhardware::hazards::HazardTracker;
use crate::gpu::rasterizer::wgpuhardware::sync::{
    NativeScaledSyncPipeline, ScaledNativeSyncBuffers, ScaledNativeSyncPipeline,
};
use crate::gpu::rasterizer::wgpuhardware::twentyfour::TwentyFourBppPipeline;
use crate::gpu::rasterizer::{
    ClearPipeline, CpuVramBlitArgs, DrawLineArgs, DrawRectangleArgs, DrawTriangleArgs, FrameCoords,
    FrameSize, RasterizerInterface, TriangleShading, TriangleTextureMapping, VramVramBlitArgs,
    vertices_valid,
};
use crate::gpu::registers::Registers;
use crate::gpu::{Color, Vertex, Vram, WgpuResources, rasterizer};
use std::collections::HashMap;
use std::ops::{BitOr, BitOrAssign, Range};
use std::sync::Arc;
use std::{array, cmp, iter};
use wgpu::{
    BindGroup, Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder,
    CommandEncoderDescriptor, ComputePassDescriptor, Device, Extent3d, ImageCopyBuffer,
    ImageCopyTexture, ImageDataLayout, LoadOp, Maintain, MapMode, Operations, Origin3d, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, Texture, TextureAspect,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};

macro_rules! include_wgsl_concat {
    ($($filename:literal),* $(,)?) => {
        wgpu::ShaderModuleDescriptor {
            label: None,
            source: {
                let contents = concat!(
                    $(include_str!($filename),)*
                );

                wgpu::ShaderSource::Wgsl(contents.into())
            }
        }
    }
}

use crate::pgxp::PgxpConfig;
use include_wgsl_concat;

const VRAM_WIDTH: u32 = 1024;
const VRAM_HEIGHT: u32 = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawCommandType {
    Draw,
    DrawCheckMaskBit,
    Blit,
    Copy,
    ScaledNativeSync,
}

#[derive(Debug)]
enum DrawCommand {
    DrawTriangle { args: DrawTriangleArgs, draw_settings: DrawSettings },
    DrawRectangle { args: DrawRectangleArgs, draw_settings: DrawSettings },
    DrawLine { args: DrawLineArgs, draw_settings: DrawSettings },
    CpuVramBlit { args: CpuVramBlitArgs, buffer_bind_group: BindGroup, sync_vertex_buffer: Buffer },
    VramCopy { args: VramVramBlitArgs },
    VramFill { x: u32, y: u32, width: u32, height: u32, color: Color, sync_vertex_buffer: Buffer },
    ScaledNativeSync { bounding_box: (Vertex, Vertex), buffers: ScaledNativeSyncBuffers },
}

impl DrawCommand {
    fn to_type(&self) -> DrawCommandType {
        match self {
            Self::DrawTriangle { args, draw_settings } => {
                if must_use_mask_bit_pipeline(
                    args.semi_transparent,
                    args.semi_transparency_mode,
                    args.texture_mapping.is_some(),
                    draw_settings.check_mask_bit,
                ) {
                    DrawCommandType::DrawCheckMaskBit
                } else {
                    DrawCommandType::Draw
                }
            }
            Self::DrawRectangle { args, draw_settings } => {
                if must_use_mask_bit_pipeline(
                    args.semi_transparent,
                    args.semi_transparency_mode,
                    args.texture_mapping.is_some(),
                    draw_settings.check_mask_bit,
                ) {
                    DrawCommandType::DrawCheckMaskBit
                } else {
                    DrawCommandType::Draw
                }
            }
            Self::DrawLine { args, draw_settings } => {
                if must_use_mask_bit_pipeline(
                    args.semi_transparent,
                    args.semi_transparency_mode,
                    false,
                    draw_settings.check_mask_bit,
                ) {
                    DrawCommandType::DrawCheckMaskBit
                } else {
                    DrawCommandType::Draw
                }
            }
            Self::CpuVramBlit { .. } | Self::VramFill { .. } => DrawCommandType::Blit,
            Self::VramCopy { .. } => DrawCommandType::Copy,
            Self::ScaledNativeSync { .. } => DrawCommandType::ScaledNativeSync,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HazardCheck {
    NotFound,
    Found,
}

impl BitOr for HazardCheck {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Self::Found, _) | (_, Self::Found) => Self::Found,
            (Self::NotFound, Self::NotFound) => Self::NotFound,
        }
    }
}

impl BitOrAssign for HazardCheck {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuRasterizerConfig {
    pub resolution_scale: u32,
    // Render draw command output in 24bpp color instead of native 15bpp color
    pub high_color: bool,
    // Whether the dithering flag is respected; only functional in 15bpp color mode
    pub dithering_allowed: bool,
    // Whether to apply dithering at native resolution or scaled resolution
    pub high_res_dithering: bool,
}

#[derive(Debug, Clone, Copy)]
struct InternalConfig {
    resolution_scale: u32,
    high_color: bool,
    dithering_allowed: bool,
    high_res_dithering: bool,
    pgxp_perspective_texture_mapping: bool,
}

impl InternalConfig {
    fn new(rasterizer_config: WgpuRasterizerConfig, pgxp_config: PgxpConfig) -> Self {
        Self {
            resolution_scale: rasterizer_config.resolution_scale,
            high_color: rasterizer_config.high_color,
            dithering_allowed: rasterizer_config.dithering_allowed,
            high_res_dithering: rasterizer_config.high_res_dithering,
            pgxp_perspective_texture_mapping: pgxp_config.perspective_texture_mapping(),
        }
    }
}

// Hack: Limit how rapidly the rasterizer will sync the VRAMs; this fixes terrible performance in
// Valkyrie Profile whenever Freya is onscreen. Her ripple effects are drawn using a bunch of
// textured quads that sample from nearby in the current frame buffer.
const SCALED_NATIVE_SYNC_DELAY: u8 = 5;

#[derive(Debug)]
pub struct WgpuRasterizer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: InternalConfig,
    // rgba8unorm at scaled resolution; used for rendering
    scaled_vram: Texture,
    // rgba8unorm at scaled resolution; used for 15bpp texture sampling
    scaled_vram_copy: Texture,
    // r32uint at native resolution; used for 4bpp/8bpp texture sampling and blitting
    native_vram: Texture,
    // rgba8unorm at scaled resolution; used as a no-op render attachment with some mask bit shaders
    dummy_vram: Texture,
    frame_textures: HashMap<(FrameSize, u32), Texture>,
    hazard_tracker: HazardTracker,
    clear_pipeline: ClearPipeline,
    render_24bpp_pipeline: TwentyFourBppPipeline,
    draw_pipelines: DrawPipelines,
    mask_bit_pipelines: MaskBitPipelines,
    cpu_vram_blit_pipeline: CpuVramBlitPipeline,
    vram_cpu_blitter: VramCpuBlitter,
    vram_copy_pipeline: VramCopyPipeline,
    vram_fill_pipeline: VramFillPipeline,
    native_scaled_sync_pipeline: NativeScaledSyncPipeline,
    scaled_native_sync_pipeline: ScaledNativeSyncPipeline,
    scaled_native_sync_delay: u8,
    draw_commands: Vec<DrawCommand>,
}

impl WgpuRasterizer {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        rasterizer_config: WgpuRasterizerConfig,
        pgxp_config: PgxpConfig,
    ) -> Self {
        log::info!(
            "Creating wgpu hardware rasterizer with resolution_scale={}, high_color={}, 15bpp_dithering={}",
            rasterizer_config.resolution_scale,
            rasterizer_config.high_color,
            rasterizer_config.dithering_allowed
        );

        let resolution_scale = rasterizer_config.resolution_scale;
        let scaled_vram = device.create_texture(&TextureDescriptor {
            label: "scaled_vram_texture".into(),
            size: Extent3d {
                width: resolution_scale * VRAM_WIDTH,
                height: resolution_scale * VRAM_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        });

        let scaled_vram_copy = device.create_texture(&TextureDescriptor {
            label: "scaled_vram_copy_texture".into(),
            size: scaled_vram.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: scaled_vram.dimension(),
            format: scaled_vram.format(),
            usage: TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let native_vram = device.create_texture(&TextureDescriptor {
            label: "native_vram_texture".into(),
            size: Extent3d { width: VRAM_WIDTH, height: VRAM_HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // R32 because storage textures don't support R16
            format: TextureFormat::R32Uint,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::STORAGE_BINDING
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let dummy_vram = device.create_texture(&TextureDescriptor {
            label: "dummy_vram_texture".into(),
            size: scaled_vram.size(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: scaled_vram.dimension(),
            format: scaled_vram.format(),
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let clear_pipeline = ClearPipeline::new(&device, TextureFormat::Rgba8Unorm);

        let render_24bpp_pipeline = TwentyFourBppPipeline::new(&device, &native_vram);

        let draw_shader = device.create_shader_module(include_wgsl_concat!(
            "wgpuhardware/draw_common.wgsl",
            "wgpuhardware/draw.wgsl"
        ));
        let draw_pipelines =
            DrawPipelines::new(&device, &draw_shader, &native_vram, &scaled_vram_copy);
        let mask_bit_pipelines = MaskBitPipelines::new(
            &device,
            &draw_shader,
            &native_vram,
            &scaled_vram,
            &scaled_vram_copy,
        );

        let cpu_vram_blit_pipeline = CpuVramBlitPipeline::new(&device, &native_vram);
        let vram_cpu_blitter = VramCpuBlitter::new(&device);
        let vram_copy_pipeline = VramCopyPipeline::new(&device, &scaled_vram);
        let vram_fill_pipeline = VramFillPipeline::new(&device, &native_vram);

        let native_scaled_sync_pipeline =
            NativeScaledSyncPipeline::new(&device, &native_vram, resolution_scale);
        let scaled_native_sync_pipeline =
            ScaledNativeSyncPipeline::new(&device, &scaled_vram, resolution_scale);

        Self {
            device,
            queue,
            config: InternalConfig::new(rasterizer_config, pgxp_config),
            scaled_vram,
            scaled_vram_copy,
            native_vram,
            dummy_vram,
            frame_textures: HashMap::with_capacity(20),
            hazard_tracker: HazardTracker::new(),
            clear_pipeline,
            render_24bpp_pipeline,
            draw_pipelines,
            mask_bit_pipelines,
            cpu_vram_blit_pipeline,
            vram_cpu_blitter,
            vram_copy_pipeline,
            vram_fill_pipeline,
            native_scaled_sync_pipeline,
            scaled_native_sync_pipeline,
            draw_commands: Vec::with_capacity(2000),
            scaled_native_sync_delay: 0,
        }
    }

    pub fn copy_vram_from(&self, vram: &Vram) {
        let vram_u32: Vec<_> = vram.iter().copied().map(u32::from).collect();

        self.queue.write_texture(
            self.native_vram.as_image_copy(),
            bytemuck::cast_slice(&vram_u32),
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * VRAM_WIDTH),
                rows_per_image: None,
            },
            self.native_vram.size(),
        );

        let sync_vertex_buffer =
            self.native_scaled_sync_pipeline.prepare(&self.device, [0, 0], [1024, 512]);

        let scaled_vram_view = self.scaled_vram.create_view(&TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "load_state_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &scaled_vram_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                ..RenderPassDescriptor::default()
            });

            self.native_scaled_sync_pipeline.draw(&sync_vertex_buffer, &mut render_pass);
        }

        encoder.copy_texture_to_texture(
            self.scaled_vram.as_image_copy(),
            self.scaled_vram_copy.as_image_copy(),
            self.scaled_vram.size(),
        );

        self.queue.submit(iter::once(encoder.finish()));
    }

    fn get_and_clear_frame(
        &mut self,
        frame_size: FrameSize,
        command_buffers: &mut Vec<CommandBuffer>,
    ) -> &Texture {
        let frame = get_or_create_frame_texture(
            &self.device,
            frame_size,
            self.config.resolution_scale,
            &mut self.frame_textures,
        );

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        self.clear_pipeline.draw(frame, &mut encoder);
        command_buffers.push(encoder.finish());

        frame
    }

    fn render_24bpp(
        &mut self,
        frame_coords: FrameCoords,
        frame_size: FrameSize,
        command_buffers: &mut Vec<CommandBuffer>,
    ) -> &Texture {
        let frame =
            get_or_create_frame_texture(&self.device, frame_size, 1, &mut self.frame_textures);
        let frame_view = frame.create_view(&TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "render_24bpp_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                ..RenderPassDescriptor::default()
            });

            self.render_24bpp_pipeline.draw(frame_coords, &mut render_pass);
        }

        command_buffers.push(encoder.finish());

        frame
    }

    fn flush_draw_commands(&mut self) -> Option<CommandBuffer> {
        if self.draw_commands.is_empty() {
            return None;
        }

        self.vram_cpu_blitter.out_of_sync = true;
        self.scaled_native_sync_delay = 0;

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());

        let mut i = 0;
        while let Some(command) = self.draw_commands.get(i) {
            let command_type = command.to_type();

            let mut j = i + 1;
            while j < self.draw_commands.len() && self.draw_commands[j].to_type() == command_type {
                j += 1;
            }

            match command_type {
                DrawCommandType::Draw => {
                    self.execute_draw(i..j, &mut encoder);
                }
                DrawCommandType::DrawCheckMaskBit => {
                    self.execute_mask_bit_draw(i..j, &mut encoder);
                }
                DrawCommandType::Blit => {
                    self.execute_blits(i..j, &mut encoder);
                }
                DrawCommandType::Copy => {
                    self.execute_copy(i..j, &mut encoder);
                }
                DrawCommandType::ScaledNativeSync => {
                    // This is an internal command and there shouldn't be more than one of these
                    // consecutively anyway, no need to batch
                    for draw_command in &self.draw_commands[i..j] {
                        let DrawCommand::ScaledNativeSync { bounding_box, buffers } = draw_command
                        else {
                            continue;
                        };

                        self.execute_scaled_native_sync(*bounding_box, buffers, &mut encoder);
                    }
                }
            }

            i = j;
        }

        self.draw_commands.clear();

        self.flush_rendered_to_native(&mut encoder);

        Some(encoder.finish())
    }

    fn execute_draw(&mut self, draw_command_range: Range<usize>, encoder: &mut CommandEncoder) {
        log::debug!("Executing {} draw commands", draw_command_range.len());

        for draw_command in &self.draw_commands[draw_command_range.clone()] {
            match draw_command {
                DrawCommand::DrawTriangle { args, draw_settings } => {
                    self.draw_pipelines.add_triangle(args, draw_settings);
                }
                DrawCommand::DrawRectangle { args, draw_settings } => {
                    self.draw_pipelines.add_rectangle(args, draw_settings);
                }
                DrawCommand::DrawLine { args, draw_settings } => {
                    self.draw_pipelines.add_line(args, draw_settings);
                }
                DrawCommand::CpuVramBlit { .. }
                | DrawCommand::VramFill { .. }
                | DrawCommand::VramCopy { .. }
                | DrawCommand::ScaledNativeSync { .. } => {}
            }
        }

        let draw_buffers = self.draw_pipelines.prepare(&self.device);

        let scaled_vram_view = self.scaled_vram.create_view(&TextureViewDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "draw_triangles_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &scaled_vram_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                ..RenderPassDescriptor::default()
            });

            self.draw_pipelines.draw(&draw_buffers, self.config, &mut render_pass);
        }
    }

    fn execute_mask_bit_draw(
        &mut self,
        draw_command_range: Range<usize>,
        encoder: &mut CommandEncoder,
    ) {
        log::debug!("Executing {} draw commands with check mask bit", draw_command_range.len());

        for draw_command in &self.draw_commands[draw_command_range.clone()] {
            match draw_command {
                DrawCommand::DrawTriangle { args, draw_settings } => {
                    self.mask_bit_pipelines.add_triangle(args, draw_settings);
                }
                DrawCommand::DrawRectangle { args, draw_settings } => {
                    self.mask_bit_pipelines.add_rectangle(args, draw_settings);
                }
                DrawCommand::DrawLine { args, draw_settings } => {
                    self.mask_bit_pipelines.add_line(args, draw_settings);
                }
                DrawCommand::CpuVramBlit { .. }
                | DrawCommand::VramCopy { .. }
                | DrawCommand::VramFill { .. }
                | DrawCommand::ScaledNativeSync { .. } => {}
            }
        }

        let draw_buffers = self.mask_bit_pipelines.prepare(&self.device);

        let dummy_vram_view = self.dummy_vram.create_view(&TextureViewDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "draw_triangles_mask_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &dummy_vram_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Discard,
                    },
                })],
                ..RenderPassDescriptor::default()
            });

            self.mask_bit_pipelines.draw(&draw_buffers, self.config, &mut render_pass);
        }
    }

    fn execute_blits(&self, draw_command_range: Range<usize>, encoder: &mut CommandEncoder) {
        log::debug!("Executing {} blit commands", draw_command_range.len());

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: "cpu_vram_blit_compute_pass".into(),
                timestamp_writes: None,
            });

            for draw_command in &self.draw_commands[draw_command_range.clone()] {
                match draw_command {
                    DrawCommand::CpuVramBlit { args, buffer_bind_group, .. } => {
                        self.cpu_vram_blit_pipeline.dispatch(
                            args,
                            buffer_bind_group,
                            &mut compute_pass,
                        );
                    }
                    &DrawCommand::VramFill { x, y, width, height, color, .. } => {
                        self.vram_fill_pipeline.dispatch(
                            x,
                            y,
                            width,
                            height,
                            color,
                            &mut compute_pass,
                        );
                    }
                    DrawCommand::DrawTriangle { .. }
                    | DrawCommand::DrawRectangle { .. }
                    | DrawCommand::DrawLine { .. }
                    | DrawCommand::ScaledNativeSync { .. }
                    | DrawCommand::VramCopy { .. } => {}
                }
            }
        }

        let scaled_vram_view = self.scaled_vram.create_view(&TextureViewDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "cpu_vram_blit_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &scaled_vram_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                ..RenderPassDescriptor::default()
            });

            for draw_command in &self.draw_commands[draw_command_range.clone()] {
                match draw_command {
                    DrawCommand::CpuVramBlit { sync_vertex_buffer, .. }
                    | DrawCommand::VramFill { sync_vertex_buffer, .. } => {
                        self.native_scaled_sync_pipeline.draw(sync_vertex_buffer, &mut render_pass);
                    }
                    DrawCommand::DrawTriangle { .. }
                    | DrawCommand::DrawRectangle { .. }
                    | DrawCommand::DrawLine { .. }
                    | DrawCommand::ScaledNativeSync { .. }
                    | DrawCommand::VramCopy { .. } => {}
                }
            }
        }

        for draw_command in &self.draw_commands[draw_command_range] {
            match draw_command {
                DrawCommand::CpuVramBlit { args, .. } => {
                    self.copy_scaled_vram([args.x, args.y], [args.width, args.height], encoder);
                }
                &DrawCommand::VramFill { x, y, width, height, .. } => {
                    self.copy_scaled_vram([x, y], [width, height], encoder);
                }
                DrawCommand::DrawTriangle { .. }
                | DrawCommand::DrawRectangle { .. }
                | DrawCommand::DrawLine { .. }
                | DrawCommand::ScaledNativeSync { .. }
                | DrawCommand::VramCopy { .. } => {}
            }
        }
    }

    fn execute_copy(&self, draw_command_range: Range<usize>, encoder: &mut CommandEncoder) {
        log::debug!("Executing {} VRAM copy commands", draw_command_range.len());

        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: "vram_copy_compute_pass".into(),
                timestamp_writes: None,
            });

            for draw_command in &self.draw_commands[draw_command_range] {
                let DrawCommand::VramCopy { args } = draw_command else { continue };

                self.vram_copy_pipeline.dispatch(
                    args,
                    self.config.resolution_scale,
                    &mut compute_pass,
                );
            }
        }
    }

    fn execute_scaled_native_sync(
        &self,
        bounding_box: (Vertex, Vertex),
        buffers: &ScaledNativeSyncBuffers,
        encoder: &mut CommandEncoder,
    ) {
        log::debug!("Syncing scaled VRAM to native");

        let native_vram_view = self.native_vram.create_view(&TextureViewDescriptor::default());
        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: "scaled_native_sync_render_pass".into(),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &native_vram_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                ..RenderPassDescriptor::default()
            });

            self.scaled_native_sync_pipeline.draw(buffers, &mut render_pass);
        }

        self.copy_scaled_vram(
            [bounding_box.0.x as u32, bounding_box.0.y as u32],
            [
                (bounding_box.1.x - bounding_box.0.x) as u32,
                (bounding_box.1.y - bounding_box.0.y) as u32,
            ],
            encoder,
        );
    }

    fn copy_scaled_vram(&self, position: [u32; 2], size: [u32; 2], encoder: &mut CommandEncoder) {
        if position[0] + size[0] > VRAM_WIDTH {
            self.copy_scaled_vram(position, [VRAM_WIDTH - position[0], size[1]], encoder);
            self.copy_scaled_vram(
                [0, position[1]],
                [size[0] - (VRAM_WIDTH - position[0]), size[1]],
                encoder,
            );
            return;
        }

        if position[1] + size[1] > VRAM_HEIGHT {
            self.copy_scaled_vram(position, [size[0], VRAM_HEIGHT - size[1]], encoder);
            self.copy_scaled_vram(
                [position[0], 0],
                [size[0], size[1] - (VRAM_HEIGHT - size[1])],
                encoder,
            );
            return;
        }

        let resolution_scale = self.config.resolution_scale;
        let scaled_position = position.map(|value| value * resolution_scale);
        let scaled_size = size.map(|value| value * resolution_scale);

        let copy_origin = Origin3d { x: scaled_position[0], y: scaled_position[1], z: 0 };

        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &self.scaled_vram,
                mip_level: 0,
                origin: copy_origin,
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: &self.scaled_vram_copy,
                mip_level: 0,
                origin: copy_origin,
                aspect: TextureAspect::All,
            },
            Extent3d { width: scaled_size[0], height: scaled_size[1], depth_or_array_layers: 1 },
        );
    }

    fn push_scaled_native_sync_command(&mut self) {
        let Some(bounding_box) = self.hazard_tracker.bounding_box() else { return };

        let buffers = self.scaled_native_sync_pipeline.prepare(
            &self.device,
            bounding_box,
            self.hazard_tracker.atlas.as_ref(),
        );
        self.draw_commands.push(DrawCommand::ScaledNativeSync { bounding_box, buffers });

        self.hazard_tracker.clear();
    }

    fn flush_rendered_to_native(&mut self, encoder: &mut CommandEncoder) {
        let Some(bounding_box) = self.hazard_tracker.bounding_box() else { return };

        let buffers = self.scaled_native_sync_pipeline.prepare(
            &self.device,
            bounding_box,
            self.hazard_tracker.atlas.as_ref(),
        );
        self.execute_scaled_native_sync(bounding_box, &buffers, encoder);

        self.hazard_tracker.clear();
    }

    #[must_use]
    fn check_textured_triangle_bounding_box(&self, args: &DrawTriangleArgs) -> HazardCheck {
        let Some(texture_mapping) = &args.texture_mapping else { return HazardCheck::NotFound };

        let min_u: u32 = texture_mapping.u.into_iter().min().unwrap().into();
        let max_u: u32 = texture_mapping.u.into_iter().max().unwrap().into();
        let min_v: u32 = texture_mapping.v.into_iter().min().unwrap().into();
        let max_v: u32 = texture_mapping.v.into_iter().max().unwrap().into();

        let mut hazard_found = self.check_texture_bounding_box(
            &texture_mapping.texpage,
            (min_u, min_v),
            (max_u, max_v),
        );

        hazard_found |= self.check_clut_bounding_box(
            texture_mapping.texpage.color_depth,
            texture_mapping.clut_x,
            texture_mapping.clut_y,
        );

        hazard_found
    }

    #[must_use]
    fn check_textured_rect_bounding_box(&mut self, args: &DrawRectangleArgs) -> HazardCheck {
        let Some(texture_mapping) = &args.texture_mapping else { return HazardCheck::NotFound };

        let u: u32 = texture_mapping.u[0].into();
        let v: u32 = texture_mapping.v[0].into();

        let u_overflow = u + args.width > 256;
        let v_overflow = v + args.height > 256;

        let mut hazard_found = HazardCheck::NotFound;

        if u_overflow && v_overflow {
            let overflowed_u = cmp::min(255, args.width - (256 - u) - 1);
            let overflowed_v = cmp::min(255, args.height - (256 - v) - 1);

            hazard_found |=
                self.check_texture_bounding_box(&texture_mapping.texpage, (u, v), (255, 255));
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (u, 0),
                (255, overflowed_v),
            );
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (0, v),
                (overflowed_u, 255),
            );
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (0, 0),
                (overflowed_u, overflowed_v),
            );
        } else if u_overflow {
            let overflowed_u = cmp::min(255, args.width - (256 - u) - 1);

            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (u, v),
                (255, v + args.height - 1),
            );
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (0, v),
                (overflowed_u, v + args.height - 1),
            );
        } else if v_overflow {
            let overflowed_v = cmp::min(255, args.height - (256 - v) - 1);

            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (u, v),
                (u + args.width - 1, 255),
            );
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (u, 0),
                (u + args.width - 1, overflowed_v),
            );
        } else {
            hazard_found |= self.check_texture_bounding_box(
                &texture_mapping.texpage,
                (u, v),
                (u + args.width - 1, v + args.height - 1),
            );
        }

        hazard_found |= self.check_clut_bounding_box(
            texture_mapping.texpage.color_depth,
            texture_mapping.clut_x,
            texture_mapping.clut_y,
        );

        hazard_found
    }

    #[must_use]
    fn check_texture_bounding_box(
        &self,
        texpage: &TexturePage,
        top_left: (u32, u32),
        bottom_right: (u32, u32),
    ) -> HazardCheck {
        let x_base = 64 * texpage.x_base;
        let y_base = texpage.y_base;

        let u_shift = match texpage.color_depth {
            TextureColorDepthBits::Four => 2,
            TextureColorDepthBits::Eight => 1,
            TextureColorDepthBits::Fifteen => 0,
        };

        let x = (x_base + (top_left.0 >> u_shift)) & (VRAM_WIDTH - 1);
        let y = (y_base + top_left.1) & (VRAM_HEIGHT - 1);
        let width = (bottom_right.0 - top_left.0 + 1) >> u_shift;
        let height = bottom_right.1 - top_left.1 + 1;

        let hazard = if x + width > VRAM_WIDTH {
            self.hazard_tracker.any_marked_rendered(
                Vertex::new(x as i32, y as i32),
                Vertex::new(VRAM_WIDTH as i32, (y + height) as i32),
            ) || self.hazard_tracker.any_marked_rendered(
                Vertex::new(0, y as i32),
                Vertex::new((width - (VRAM_WIDTH - x)) as i32, (y + height) as i32),
            )
        } else {
            self.hazard_tracker.any_marked_rendered(
                Vertex::new(x as i32, y as i32),
                Vertex::new((x + width) as i32, (y + height) as i32),
            )
        };

        if hazard { HazardCheck::Found } else { HazardCheck::NotFound }
    }

    #[must_use]
    fn check_clut_bounding_box(
        &self,
        depth: TextureColorDepthBits,
        clut_x: u16,
        clut_y: u16,
    ) -> HazardCheck {
        let clut_x = 16 * clut_x;

        let hazard = match depth {
            TextureColorDepthBits::Four => self.hazard_tracker.any_marked_rendered(
                Vertex::new(clut_x.into(), clut_y.into()),
                Vertex::new((clut_x + 16).into(), (clut_y + 1).into()),
            ),
            TextureColorDepthBits::Eight => {
                if clut_x + 256 > VRAM_WIDTH as u16 {
                    self.hazard_tracker.any_marked_rendered(
                        Vertex::new(clut_x.into(), clut_y.into()),
                        Vertex::new(VRAM_WIDTH as i32, (clut_y + 1).into()),
                    ) || self.hazard_tracker.any_marked_rendered(
                        Vertex::new(0, clut_y.into()),
                        Vertex::new(
                            (256 - (VRAM_WIDTH as u16 - clut_x)).into(),
                            (clut_y + 1).into(),
                        ),
                    )
                } else {
                    self.hazard_tracker.any_marked_rendered(
                        Vertex::new(clut_x.into(), clut_y.into()),
                        Vertex::new((clut_x + 256).into(), (clut_y + 1).into()),
                    )
                }
            }
            TextureColorDepthBits::Fifteen => return HazardCheck::NotFound,
        };

        if hazard { HazardCheck::Found } else { HazardCheck::NotFound }
    }

    fn mark_vram_copy_rendered(&mut self, args: &VramVramBlitArgs) {
        let x_overflow = args.dest_x + args.width > VRAM_WIDTH;
        let y_overflow = args.dest_y + args.height > VRAM_HEIGHT;

        if x_overflow && y_overflow {
            let x_end = (args.width - (VRAM_WIDTH - args.dest_x)) as i32;
            let y_end = (args.height - (VRAM_HEIGHT - args.dest_y)) as i32;

            self.hazard_tracker.mark_rendered(
                Vertex::new(args.dest_x as i32, args.dest_y as i32),
                Vertex::new(VRAM_WIDTH as i32, VRAM_HEIGHT as i32),
            );
            self.hazard_tracker.mark_rendered(
                Vertex::new(args.dest_x as i32, 0),
                Vertex::new(VRAM_WIDTH as i32, y_end),
            );
            self.hazard_tracker.mark_rendered(
                Vertex::new(0, args.dest_y as i32),
                Vertex::new(x_end, VRAM_HEIGHT as i32),
            );
            self.hazard_tracker.mark_rendered(Vertex::new(0, 0), Vertex::new(x_end, y_end));
        } else if x_overflow {
            let x_end = (args.width - (VRAM_WIDTH - args.dest_x)) as i32;
            let y_end = (args.dest_y + args.width) as i32;

            self.hazard_tracker.mark_rendered(
                Vertex::new(args.dest_x as i32, args.dest_y as i32),
                Vertex::new(VRAM_WIDTH as i32, y_end),
            );
            self.hazard_tracker
                .mark_rendered(Vertex::new(0, args.dest_y as i32), Vertex::new(x_end, y_end));
        } else if y_overflow {
            let x_end = (args.dest_x + args.width) as i32;
            let y_end = (args.height - (VRAM_HEIGHT - args.dest_y)) as i32;

            self.hazard_tracker.mark_rendered(
                Vertex::new(args.dest_x as i32, args.dest_y as i32),
                Vertex::new(x_end, VRAM_HEIGHT as i32),
            );
            self.hazard_tracker
                .mark_rendered(Vertex::new(args.dest_x as i32, 0), Vertex::new(x_end, y_end));
        } else {
            self.hazard_tracker.mark_rendered(
                Vertex::new(args.dest_x as i32, args.dest_y as i32),
                Vertex::new((args.dest_x + args.width) as i32, (args.dest_y + args.height) as i32),
            );
        }
    }
}

fn must_use_mask_bit_pipeline(
    semi_transparent: bool,
    semi_transparency_mode: SemiTransparencyMode,
    textured: bool,
    check_mask_bit: bool,
) -> bool {
    if !check_mask_bit {
        return false;
    }

    semi_transparent && (textured || semi_transparency_mode == SemiTransparencyMode::Average)
}

impl RasterizerInterface for WgpuRasterizer {
    fn draw_triangle(&mut self, args: DrawTriangleArgs, draw_settings: &DrawSettings) {
        if !draw_settings.is_drawing_area_valid() {
            return;
        }

        if !vertices_valid(args.vertices[0], args.vertices[1])
            || !vertices_valid(args.vertices[1], args.vertices[2])
            || !vertices_valid(args.vertices[2], args.vertices[0])
        {
            return;
        }

        self.scaled_native_sync_delay = self.scaled_native_sync_delay.saturating_sub(1);

        if self.check_textured_triangle_bounding_box(&args) == HazardCheck::Found {
            if self.scaled_native_sync_delay == 0 {
                self.push_scaled_native_sync_command();
            }
            self.scaled_native_sync_delay = SCALED_NATIVE_SYNC_DELAY;
        }

        if let Some((bounding_box_top_left, bounding_box_top_right)) =
            triangle_bounding_box(&args, draw_settings)
        {
            self.hazard_tracker.mark_rendered(bounding_box_top_left, bounding_box_top_right);
        }

        if self.config.resolution_scale != 1
            && args.pgxp_vertices.is_none()
            && let Some(command) = check_for_tiny_triangle(&args, draw_settings)
        {
            self.draw_commands.push(command);
        }

        self.draw_commands
            .push(DrawCommand::DrawTriangle { args, draw_settings: draw_settings.clone() });
    }

    fn draw_line(&mut self, args: DrawLineArgs, draw_settings: &DrawSettings) {
        if !draw_settings.is_drawing_area_valid() {
            return;
        }

        // TODO mark points rendered, possibly just the bounding box

        self.draw_commands
            .push(DrawCommand::DrawLine { args, draw_settings: draw_settings.clone() });
    }

    fn draw_rectangle(&mut self, mut args: DrawRectangleArgs, draw_settings: &DrawSettings) {
        if !draw_settings.is_drawing_area_valid() || args.width == 0 || args.height == 0 {
            return;
        }

        // Pre-apply draw offset to handle the draw offset possibly causing the rectangle's
        // coordinates to wrap.
        // Skullmonkeys depends on this or many graphics will be missing
        let top_left = args.top_left + draw_settings.draw_offset;
        args.top_left = Vertex { x: i11(top_left.x), y: i11(top_left.y) };
        let draw_settings =
            DrawSettings { draw_offset: Vertex::new(0, 0), ..draw_settings.clone() };

        let Some((bounding_box_top_left, bounding_box_top_right)) =
            rectangle_bounding_box(&args, &draw_settings)
        else {
            return;
        };

        self.scaled_native_sync_delay = self.scaled_native_sync_delay.saturating_sub(1);

        if self.check_textured_rect_bounding_box(&args) == HazardCheck::Found {
            if self.scaled_native_sync_delay == 0 {
                self.push_scaled_native_sync_command();
            }
            self.scaled_native_sync_delay = SCALED_NATIVE_SYNC_DELAY;
        }

        self.hazard_tracker.mark_rendered(bounding_box_top_left, bounding_box_top_right);

        // TODO proper scaled/native sync

        self.draw_commands.push(DrawCommand::DrawRectangle { args, draw_settings });
    }

    fn vram_fill(&mut self, x: u32, y: u32, width: u32, height: u32, color: Color) {
        let sync_vertex_buffer =
            self.native_scaled_sync_pipeline.prepare(&self.device, [x, y], [width, height]);

        self.draw_commands.push(DrawCommand::VramFill {
            x,
            y,
            width,
            height,
            color,
            sync_vertex_buffer,
        });
    }

    fn cpu_to_vram_blit(&mut self, args: CpuVramBlitArgs, data: &[u16]) {
        let buffer_bind_group = self.cpu_vram_blit_pipeline.prepare(&self.device, &args, data);
        let sync_vertex_buffer = self.native_scaled_sync_pipeline.prepare(
            &self.device,
            [args.x, args.y],
            [args.width, args.height],
        );

        self.draw_commands.push(DrawCommand::CpuVramBlit {
            args,
            buffer_bind_group,
            sync_vertex_buffer,
        });
    }

    fn vram_to_cpu_blit(&mut self, x: u32, y: u32, width: u32, height: u32, out: &mut Vec<u16>) {
        log::debug!("VRAM-to-CPU blit: position ({x}, {y}) size ({width}, {height})");

        let draw_command_buffer = self.flush_draw_commands();
        assert!(draw_command_buffer.is_none() || self.vram_cpu_blitter.out_of_sync);

        if self.vram_cpu_blitter.out_of_sync {
            log::debug!("Syncing VRAM to host RAM");
            self.vram_cpu_blitter.blit_from_gpu(
                &self.device,
                &self.queue,
                &self.native_vram,
                draw_command_buffer,
            );
            self.vram_cpu_blitter.out_of_sync = false;
        }

        self.vram_cpu_blitter.copy_blit_output(x, y, width, height, out);
    }

    fn vram_to_vram_blit(&mut self, args: VramVramBlitArgs) {
        log::debug!("VRAM-to-VRAM blit: {args:?}");

        self.mark_vram_copy_rendered(&args);

        self.draw_commands.push(DrawCommand::VramCopy { args });
    }

    fn generate_frame_texture(
        &mut self,
        registers: &Registers,
        wgpu_resources: &mut WgpuResources,
    ) -> &Texture {
        log::debug!("Rendering frame to display");

        if let Some(command_buffer) = self.flush_draw_commands() {
            wgpu_resources.queued_command_buffers.push(command_buffer);
        }

        if wgpu_resources.display_config.dump_vram {
            return &self.scaled_vram;
        }

        let (frame_coords, frame_size) =
            rasterizer::compute_frame_location(registers, wgpu_resources.display_config);
        let Some(frame_coords) = frame_coords else {
            return self
                .get_and_clear_frame(frame_size, &mut wgpu_resources.queued_command_buffers);
        };

        if !registers.display_enabled {
            return self
                .get_and_clear_frame(frame_size, &mut wgpu_resources.queued_command_buffers);
        }

        log::debug!("  Frame size {frame_size:?}, frame coords {frame_coords:?}");

        if registers.display_area_color_depth == ColorDepthBits::TwentyFour {
            return self.render_24bpp(
                frame_coords,
                frame_size,
                &mut wgpu_resources.queued_command_buffers,
            );
        }

        let resolution_scale = self.config.resolution_scale;
        let frame = get_or_create_frame_texture(
            &self.device,
            frame_size,
            resolution_scale,
            &mut self.frame_textures,
        );

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        self.clear_pipeline.draw(frame, &mut encoder);

        // TODO bounds check
        let source_x = frame_coords.frame_x + frame_coords.display_x_offset;
        let source_y = frame_coords.frame_y + frame_coords.display_y_offset;
        encoder.copy_texture_to_texture(
            ImageCopyTexture {
                texture: &self.scaled_vram,
                mip_level: 0,
                origin: Origin3d {
                    x: resolution_scale * source_x,
                    y: resolution_scale * source_y,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            ImageCopyTexture {
                texture: frame,
                mip_level: 0,
                origin: Origin3d {
                    x: resolution_scale * frame_coords.display_x_start,
                    y: resolution_scale * frame_coords.display_y_start,
                    z: 0,
                },
                aspect: TextureAspect::All,
            },
            Extent3d {
                width: resolution_scale * frame_coords.display_width,
                height: resolution_scale * frame_coords.display_height,
                depth_or_array_layers: 1,
            },
        );

        wgpu_resources.queued_command_buffers.push(encoder.finish());

        frame
    }

    fn clone_vram(&mut self) -> Vram {
        let flush_command_buffer = self.flush_draw_commands();

        let vram_buffer = self.device.create_buffer(&BufferDescriptor {
            label: "vram_buffer".into(),
            size: (4 * VRAM_WIDTH * VRAM_HEIGHT).into(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            self.native_vram.as_image_copy(),
            ImageCopyBuffer {
                buffer: &vram_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * VRAM_WIDTH),
                    rows_per_image: None,
                },
            },
            self.native_vram.size(),
        );

        self.queue.submit(flush_command_buffer.into_iter().chain(iter::once(encoder.finish())));

        let vram_buffer_slice = vram_buffer.slice(..);
        vram_buffer_slice.map_async(MapMode::Read, Result::unwrap);
        self.device.poll(Maintain::Wait);

        let vram_buffer_view = vram_buffer_slice.get_mapped_range();

        let mut vram = Vram::new();
        for (chunk, vram_value) in vram_buffer_view.chunks_exact(4).zip(vram.iter_mut()) {
            *vram_value = u16::from_le_bytes([chunk[0], chunk[1]]);
        }

        vram
    }

    fn clear_texture_cache(&mut self) {
        self.scaled_native_sync_delay = 0;
    }
}

fn triangle_bounding_box(
    args: &DrawTriangleArgs,
    draw_settings: &DrawSettings,
) -> Option<(Vertex, Vertex)> {
    let v_x = args.vertices.map(|v| v.x + draw_settings.draw_offset.x);
    let v_y = args.vertices.map(|v| v.y + draw_settings.draw_offset.y);

    let min_x = cmp::max(draw_settings.draw_area_top_left.x, v_x.into_iter().min().unwrap());
    let max_x =
        cmp::min(draw_settings.draw_area_bottom_right.x + 1, v_x.into_iter().max().unwrap());
    let min_y = cmp::max(draw_settings.draw_area_top_left.y, v_y.into_iter().min().unwrap());
    let max_y =
        cmp::min(draw_settings.draw_area_bottom_right.y + 1, v_y.into_iter().max().unwrap());

    if min_x >= max_x || min_y >= max_y {
        return None;
    }

    Some((Vertex::new(min_x, min_y), Vertex::new(max_x, max_y)))
}

fn rectangle_bounding_box(
    args: &DrawRectangleArgs,
    draw_settings: &DrawSettings,
) -> Option<(Vertex, Vertex)> {
    let top_left = args.top_left + draw_settings.draw_offset;

    let min_x = cmp::max(draw_settings.draw_area_top_left.x, top_left.x);
    let max_x =
        cmp::min(draw_settings.draw_area_bottom_right.x + 1, top_left.x + args.width as i32);
    let min_y = cmp::max(draw_settings.draw_area_top_left.y, top_left.y);
    let max_y =
        cmp::min(draw_settings.draw_area_bottom_right.y + 1, top_left.y + args.height as i32);

    if min_x >= max_x || min_y >= max_y {
        return None;
    }

    Some((Vertex::new(min_x, min_y), Vertex::new(max_x, max_y)))
}

#[derive(Debug, Clone, Copy)]
struct IndexedVertex {
    idx: usize,
    v: Vertex,
}

// Check if the three triangle vertices form a right triangle that is either 1 pixel wide or 1 pixel
// tall. If so, add an extra draw command that will draw an opposing right triangle and form a
// 1xN or Nx1 rectangle. Doom depends on this for correct rendering at higher resolutions
fn check_for_tiny_triangle(
    args: &DrawTriangleArgs,
    draw_settings: &DrawSettings,
) -> Option<DrawCommand> {
    // Skip the later sorts/checks if all vertices have different X coordinates or Y coordinates
    if (args.vertices[0].x != args.vertices[1].x
        && args.vertices[0].x != args.vertices[2].x
        && args.vertices[1].x != args.vertices[2].x)
        || (args.vertices[0].y != args.vertices[1].y
            && args.vertices[0].y != args.vertices[2].y
            && args.vertices[1].y != args.vertices[2].y)
    {
        return None;
    }

    let mut vertices: [_; 3] = array::from_fn(|i| IndexedVertex { idx: i, v: args.vertices[i] });

    // Check if the triangle is one pixel wide
    vertices.sort_by(|a, b| a.v.x.cmp(&b.v.x));
    if vertices[0].v.x == vertices[1].v.x && vertices[0].v.x + 1 == vertices[2].v.x {
        if vertices[0].v.y == vertices[2].v.y {
            let vertex = Vertex::new(vertices[2].v.x, vertices[1].v.y);
            return Some(expand_tiny_triangle(args, draw_settings, vertices, 1, vertex));
        } else if vertices[1].v.y == vertices[2].v.y {
            let vertex = Vertex::new(vertices[2].v.x, vertices[0].v.y);
            return Some(expand_tiny_triangle(args, draw_settings, vertices, 0, vertex));
        }
    }

    // Check if the triangle is one pixel tall
    vertices.sort_by(|a, b| a.v.y.cmp(&b.v.y));
    if vertices[0].v.y == vertices[1].v.y && vertices[0].v.y + 1 == vertices[2].v.y {
        if vertices[0].v.x == vertices[2].v.x {
            let vertex = Vertex::new(vertices[1].v.x, vertices[2].v.y);
            return Some(expand_tiny_triangle(args, draw_settings, vertices, 1, vertex));
        } else if vertices[1].v.x == vertices[2].v.x {
            let vertex = Vertex::new(vertices[0].v.x, vertices[2].v.y);
            return Some(expand_tiny_triangle(args, draw_settings, vertices, 0, vertex));
        }
    }

    None
}

fn expand_tiny_triangle(
    args: &DrawTriangleArgs,
    draw_settings: &DrawSettings,
    vertices: [IndexedVertex; 3],
    expand_idx: usize,
    fourth_vertex: Vertex,
) -> DrawCommand {
    DrawCommand::DrawTriangle {
        args: DrawTriangleArgs {
            vertices: [
                args.vertices[vertices[2].idx],
                args.vertices[vertices[expand_idx].idx],
                fourth_vertex,
            ],
            pgxp_vertices: None,
            shading: match args.shading {
                TriangleShading::Flat(color) => TriangleShading::Flat(color),
                TriangleShading::Gouraud(colors) => {
                    let expand_color = colors[vertices[expand_idx].idx];
                    TriangleShading::Gouraud([colors[vertices[2].idx], expand_color, expand_color])
                }
            },
            semi_transparent: args.semi_transparent,
            semi_transparency_mode: args.semi_transparency_mode,
            texture_mapping: args.texture_mapping.map(|mapping| TriangleTextureMapping {
                u: {
                    let expand_u = mapping.u[vertices[expand_idx].idx];
                    [mapping.u[vertices[2].idx], expand_u, expand_u]
                },
                v: {
                    let expand_v = mapping.v[vertices[expand_idx].idx];
                    [mapping.v[vertices[2].idx], expand_v, expand_v]
                },
                ..mapping
            }),
        },
        draw_settings: draw_settings.clone(),
    }
}

fn get_or_create_frame_texture<'a>(
    device: &Device,
    frame_size: FrameSize,
    resolution_scale: u32,
    frame_textures: &'a mut HashMap<(FrameSize, u32), Texture>,
) -> &'a Texture {
    frame_textures.entry((frame_size, resolution_scale)).or_insert_with(|| {
        device.create_texture(&TextureDescriptor {
            label: "frame_texture".into(),
            size: Extent3d {
                width: resolution_scale * frame_size.width,
                height: resolution_scale * frame_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::COPY_SRC
                | TextureUsages::COPY_DST
                | TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[TextureFormat::Rgba8UnormSrgb],
        })
    })
}

fn i11(value: i32) -> i32 {
    (value << 21) >> 21
}
