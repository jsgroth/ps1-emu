use crate::gpu::rasterizer::FrameCoords;
use bytemuck::{Pod, Zeroable};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, ColorTargetState, ColorWrites, Device,
    FragmentState, FrontFace, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, ShaderStages, StorageTextureAccess, Texture,
    TextureFormat, TextureViewDescriptor, TextureViewDimension, VertexState,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct Render24BppArgs {
    frame_position: [u32; 2],
    display_start: [u32; 2],
    display_offset: [u32; 2],
    display_end: [u32; 2],
}

impl Render24BppArgs {
    fn new(frame_coords: FrameCoords) -> Self {
        Self {
            frame_position: [frame_coords.frame_x, frame_coords.frame_y],
            display_start: [frame_coords.display_x_start, frame_coords.display_y_start],
            display_offset: [frame_coords.display_x_offset, frame_coords.display_y_offset],
            display_end: [
                frame_coords.display_x_start + frame_coords.display_width,
                frame_coords.display_y_offset + frame_coords.display_height,
            ],
        }
    }
}

#[derive(Debug)]
pub struct TwentyFourBppPipeline {
    bind_group: BindGroup,
    pipeline: RenderPipeline,
}

impl TwentyFourBppPipeline {
    pub fn new(device: &Device, native_vram: &Texture) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: "render_24bpp_bind_group_layout".into(),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadOnly,
                    format: native_vram.format(),
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let native_vram_view = native_vram.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "render_24bpp_bind_group".into(),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&native_vram_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "render_24bpp_pipeline_layout".into(),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: size_of::<Render24BppArgs>() as u32,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("24bpp.wgsl"));
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: "render_24bpp_pipeline".into(),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self { bind_group, pipeline }
    }

    pub fn draw<'rpass>(
        &'rpass self,
        frame_coords: FrameCoords,
        render_pass: &mut RenderPass<'rpass>,
    ) {
        let args = Render24BppArgs::new(frame_coords);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_immediates(0, bytemuck::cast_slice(&[args]));
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1);
    }
}
