use crate::gpu::gp0::{
    DrawSettings, SemiTransparencyMode, TextureColorDepthBits, TexturePage, TextureWindow,
};
use crate::gpu::rasterizer::wgpuhardware::{InternalConfig, include_wgsl_concat};
use crate::gpu::rasterizer::{
    DrawLineArgs, DrawRectangleArgs, DrawTriangleArgs, LineShading, RectangleTextureMapping,
    TextureMapping, TextureMappingMode, TriangleShading, TriangleTextureMapping,
};
use crate::gpu::{Color, Vertex};
use bytemuck::{Pod, Zeroable};
use std::array;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendFactor,
    BlendOperation, BlendState, Buffer, BufferUsages, ColorTargetState, ColorWrites, Device,
    FragmentState, FrontFace, IndexFormat, MultisampleState, PipelineCompilationOptions,
    PipelineLayoutDescriptor, PolygonMode, PrimitiveState, PrimitiveTopology, RenderPass,
    RenderPipeline, RenderPipelineDescriptor, ShaderModule, ShaderStages, StorageTextureAccess,
    Texture, TextureFormat, TextureViewDescriptor, TextureViewDimension, VertexAttribute,
    VertexBufferLayout, VertexState, VertexStepMode,
};

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ShaderDrawSettings {
    force_mask_bit: u32,
    resolution_scale: u32,
    high_color: u32,
    dithering: u32,
    high_res_dithering: u32,
    perspective_texture_mapping: u32,
}

impl ShaderDrawSettings {
    fn new(draw_settings: &DrawSettings, config: InternalConfig) -> Self {
        let dithering =
            config.dithering_allowed && !config.high_color && draw_settings.dithering_enabled;

        Self {
            force_mask_bit: draw_settings.force_mask_bit.into(),
            resolution_scale: config.resolution_scale,
            high_color: config.high_color.into(),
            dithering: dithering.into(),
            high_res_dithering: config.high_res_dithering.into(),
            perspective_texture_mapping: config.pgxp_perspective_texture_mapping.into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct UntexturedVertex {
    position: [f32; 3],
    color: [u32; 3],
    ditherable: u32,
}

impl UntexturedVertex {
    const ATTRIBUTES: [VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32x3, 2 => Uint32];

    const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct TexturedVertex {
    position: [f32; 3],
    color: [u32; 3],
    uv: [u32; 2],
    texpage: [u32; 2],
    tex_window_mask: [u32; 2],
    tex_window_offset: [u32; 2],
    clut: [u32; 2],
    flags: u32,
    integer_position: [i32; 2],
    other_positions: [i32; 4],
    other_uv: [u32; 4],
}

fn vertex_texpage(texpage: &TexturePage) -> [u32; 2] {
    [64 * texpage.x_base, texpage.y_base]
}

fn vertex_tex_window_mask(window: TextureWindow) -> [u32; 2] {
    [window.x_mask << 3, window.y_mask << 3]
}

fn vertex_tex_window_offset(window: TextureWindow) -> [u32; 2] {
    [window.x_offset << 3, window.y_offset << 3]
}

fn vertex_clut<const N: usize>(mapping: &TextureMapping<N>) -> [u32; 2] {
    [(16 * mapping.clut_x).into(), mapping.clut_y.into()]
}

fn vertex_color_depth(color_depth: TextureColorDepthBits) -> u32 {
    match color_depth {
        TextureColorDepthBits::Four => 0,
        TextureColorDepthBits::Eight => 1,
        TextureColorDepthBits::Fifteen => 2,
    }
}

impl TexturedVertex {
    const ATTRIBUTES: [VertexAttribute; 11] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Uint32x3,
        2 => Uint32x2,
        3 => Uint32x2,
        4 => Uint32x2,
        5 => Uint32x2,
        6 => Uint32x2,
        7 => Uint32,
        8 => Sint32x2,
        9 => Sint32x4,
        10 => Uint32x4,
    ];

    const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };

    fn new_vertices(
        positions: &[[i32; 2]; 3],
        precise_positions: &[[f32; 3]; 3],
        colors: [Color; 3],
        texture_mapping: &TriangleTextureMapping,
        ditherable: bool,
    ) -> [Self; 3] {
        let flags =
            generate_flags(texture_mapping.texpage.color_depth, texture_mapping.mode, ditherable);

        array::from_fn(|i| {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;

            Self {
                position: precise_positions[i],
                color: [colors[i].r.into(), colors[i].g.into(), colors[i].b.into()],
                uv: [texture_mapping.u[i].into(), texture_mapping.v[i].into()],
                texpage: vertex_texpage(&texture_mapping.texpage),
                tex_window_mask: vertex_tex_window_mask(texture_mapping.window),
                tex_window_offset: vertex_tex_window_offset(texture_mapping.window),
                clut: vertex_clut(texture_mapping),
                flags,
                integer_position: positions[i],
                other_positions: [
                    positions[j][0],
                    positions[j][1],
                    positions[k][0],
                    positions[k][1],
                ],
                other_uv: [
                    texture_mapping.u[j].into(),
                    texture_mapping.v[j].into(),
                    texture_mapping.u[k].into(),
                    texture_mapping.v[k].into(),
                ],
            }
        })
    }
}

fn generate_flags(
    color_depth: TextureColorDepthBits,
    mode: TextureMappingMode,
    ditherable: bool,
) -> u32 {
    // Must match Flags in draw_common.wgsl
    let color_depth_bits = vertex_color_depth(color_depth);
    let modulated_bit = u32::from(mode == TextureMappingMode::Modulated) << 2;
    let ditherable_bit = u32::from(ditherable) << 3;
    color_depth_bits | modulated_bit | ditherable_bit
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct TexturedRectVertex {
    position: [i32; 2],
    color: [u32; 3],
    texpage: [u32; 2],
    tex_window_mask: [u32; 2],
    tex_window_offset: [u32; 2],
    clut: [u32; 2],
    flags: u32,
    base_position: [i32; 2],
    base_uv: [u32; 2],
}

impl TexturedRectVertex {
    const ATTRIBUTES: [VertexAttribute; 9] = wgpu::vertex_attr_array![
        0 => Sint32x2,
        1 => Uint32x3,
        2 => Uint32x2,
        3 => Uint32x2,
        4 => Uint32x2,
        5 => Uint32x2,
        6 => Uint32,
        7 => Sint32x2,
        8 => Uint32x2,
    ];

    const LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
        array_stride: size_of::<Self>() as u64,
        step_mode: VertexStepMode::Vertex,
        attributes: &Self::ATTRIBUTES,
    };

    fn new_vertices(
        args: &DrawRectangleArgs,
        texture_mapping: &RectangleTextureMapping,
        draw_settings: &DrawSettings,
    ) -> [Self; 4] {
        let top_left = args.top_left + draw_settings.draw_offset;
        let vertices = rect_vertices(args, draw_settings.draw_offset);

        let flags =
            generate_flags(texture_mapping.texpage.color_depth, texture_mapping.mode, false);

        array::from_fn(|i| Self {
            position: [vertices[i].x, vertices[i].y],
            color: [args.color.r.into(), args.color.g.into(), args.color.b.into()],
            texpage: vertex_texpage(&texture_mapping.texpage),
            tex_window_mask: vertex_tex_window_mask(texture_mapping.window),
            tex_window_offset: vertex_tex_window_offset(texture_mapping.window),
            clut: vertex_clut(texture_mapping),
            flags,
            base_position: [top_left.x, top_left.y],
            base_uv: [texture_mapping.u[0].into(), texture_mapping.v[0].into()],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawPipeline {
    UntexturedTriangle(Option<SemiTransparencyMode>),
    TexturedTriangle(Option<SemiTransparencyMode>),
    TexturedRectangle(Option<SemiTransparencyMode>),
}

#[derive(Debug)]
struct DrawBatch {
    draw_settings: DrawSettings,
    pipeline: DrawPipeline,
    start: u32,
    end: u32,
}

impl DrawBatch {
    fn matches(&self, draw_settings: &DrawSettings, pipeline: DrawPipeline) -> bool {
        draw_settings == &self.draw_settings && pipeline == self.pipeline
    }
}

#[derive(Debug)]
pub struct DrawBuffers {
    untextured_triangle: Buffer,
    textured_triangle: Buffer,
    textured_rectangle_vertex: Buffer,
    textured_rectangle_index: Buffer,
}

#[derive(Debug)]
pub struct DrawPipelines {
    untextured_buffer: Vec<UntexturedVertex>,
    untextured_opaque_pipeline: RenderPipeline,
    untextured_opaque_mask_pipeline: RenderPipeline,
    untextured_average_pipeline: RenderPipeline,
    untextured_add_pipeline: RenderPipeline,
    untextured_add_mask_pipeline: RenderPipeline,
    untextured_subtract_pipeline: RenderPipeline,
    untextured_subtract_mask_pipeline: RenderPipeline,
    untextured_add_quarter_pipeline: RenderPipeline,
    untextured_add_quarter_mask_pipeline: RenderPipeline,
    textured_buffer: Vec<TexturedVertex>,
    textured_bind_group: BindGroup,
    textured_opaque_pipeline: RenderPipeline,
    textured_opaque_mask_pipeline: RenderPipeline,
    textured_average_pipeline: RenderPipeline,
    textured_add_pipeline: RenderPipeline,
    textured_subtract_pipeline_opaque: RenderPipeline,
    textured_subtract_pipeline_transparent: RenderPipeline,
    textured_add_quarter_pipeline: RenderPipeline,
    textured_rect_buffer: Vec<TexturedRectVertex>,
    textured_rect_indices: Vec<u32>,
    textured_opaque_rect_pipeline: RenderPipeline,
    textured_opaque_rect_mask_pipeline: RenderPipeline,
    textured_average_rect_pipeline: RenderPipeline,
    textured_add_rect_pipeline: RenderPipeline,
    textured_subtract_rect_pipeline_opaque: RenderPipeline,
    textured_subtract_rect_pipeline_transparent: RenderPipeline,
    textured_add_quarter_rect_pipeline: RenderPipeline,
    batches: Vec<DrawBatch>,
}

fn rect_vertices(args: &DrawRectangleArgs, draw_offset: Vertex) -> [Vertex; 4] {
    let top_left = args.top_left + draw_offset;

    [
        top_left,
        top_left + Vertex::new(args.width as i32, 0),
        top_left + Vertex::new(0, args.height as i32),
        top_left + Vertex::new(args.width as i32, args.height as i32),
    ]
}

impl DrawPipelines {
    const INITIAL_BUFFER_CAPACITY: u64 = 15000;

    const CHECK_MASK_COMPONENT: BlendComponent = BlendComponent {
        src_factor: BlendFactor::OneMinusDstAlpha,
        dst_factor: BlendFactor::DstAlpha,
        operation: BlendOperation::Add,
    };

    const REPLACE_CHECK_MASK: BlendState =
        BlendState { color: Self::CHECK_MASK_COMPONENT, alpha: Self::CHECK_MASK_COMPONENT };

    const AVERAGE_BLEND: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::Src1Alpha,
            dst_factor: BlendFactor::OneMinusSrc1Alpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::REPLACE,
    };

    const ADDITIVE_BLEND_SINGLE_SOURCE: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::REPLACE,
    };

    const ADDITIVE_BLEND_DUAL_SOURCE: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Src1Alpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::REPLACE,
    };

    const ADDITIVE_BLEND_CHECK_MASK: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::OneMinusDstAlpha,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        },
        alpha: Self::CHECK_MASK_COMPONENT,
    };

    const SUBTRACTIVE_BLEND: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::ReverseSubtract,
        },
        alpha: BlendComponent::REPLACE,
    };

    const SUBTRACTIVE_BLEND_CHECK_MASK: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::OneMinusDstAlpha,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::ReverseSubtract,
        },
        alpha: Self::CHECK_MASK_COMPONENT,
    };

    const ADD_QUARTER_BLEND: BlendState = BlendState {
        color: BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::Src1Alpha,
            operation: BlendOperation::Add,
        },
        alpha: BlendComponent::REPLACE,
    };

    pub fn new(
        device: &Device,
        draw_shader: &ShaderModule,
        native_vram: &Texture,
        scaled_vram_copy: &Texture,
    ) -> Self {
        let untextured_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "untextured_opaque_triangle_pipeline_layout".into(),
            bind_group_layouts: &[],
            immediate_size: size_of::<ShaderDrawSettings>() as u32,
        });

        let new_untextured_triangle_pipeline = |fs_entry_point: &str, blend: Option<BlendState>| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: format!("untextured_triangle_pipeline_{fs_entry_point}").as_str().into(),
                layout: Some(&untextured_layout),
                vertex: VertexState {
                    module: draw_shader,
                    entry_point: Some("vs_untextured"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[UntexturedVertex::LAYOUT],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
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
                    module: draw_shader,
                    entry_point: Some(fs_entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend,
                        write_mask: ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let untextured_opaque_pipeline =
            new_untextured_triangle_pipeline("fs_untextured_opaque", None);

        let untextured_opaque_mask_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_opaque",
            Some(Self::REPLACE_CHECK_MASK),
        );

        let untextured_average_pipeline =
            new_untextured_triangle_pipeline("fs_untextured_average", Some(Self::AVERAGE_BLEND));

        let untextured_add_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_opaque",
            Some(Self::ADDITIVE_BLEND_SINGLE_SOURCE),
        );

        let untextured_add_mask_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_opaque",
            Some(Self::ADDITIVE_BLEND_CHECK_MASK),
        );

        let untextured_subtract_pipeline =
            new_untextured_triangle_pipeline("fs_untextured_opaque", Some(Self::SUBTRACTIVE_BLEND));

        let untextured_subtract_mask_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_opaque",
            Some(Self::SUBTRACTIVE_BLEND_CHECK_MASK),
        );

        let untextured_add_quarter_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_add_quarter",
            Some(Self::ADDITIVE_BLEND_SINGLE_SOURCE),
        );

        let untextured_add_quarter_mask_pipeline = new_untextured_triangle_pipeline(
            "fs_untextured_add_quarter",
            Some(Self::ADDITIVE_BLEND_CHECK_MASK),
        );

        let textured_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: "textured_opaque_triangle_bind_group_layout".into(),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadOnly,
                            format: native_vram.format(),
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadOnly,
                            format: scaled_vram_copy.format(),
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let native_vram_view = native_vram.create_view(&TextureViewDescriptor::default());
        let scaled_vram_copy_view = scaled_vram_copy.create_view(&TextureViewDescriptor::default());
        let textured_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "textured_opaque_triangle_bind_group".into(),
            layout: &textured_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&native_vram_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&scaled_vram_copy_view),
                },
            ],
        });

        let textured_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "textured_opaque_triangle_pipeline_layout".into(),
            bind_group_layouts: &[Some(&textured_bind_group_layout)],
            immediate_size: size_of::<ShaderDrawSettings>() as u32,
        });

        let new_textured_pipeline =
            |vertex_buffer_layout: VertexBufferLayout<'_>,
             vs_entry_point: &str,
             fs_entry_point: &str,
             blend: Option<BlendState>| {
                device.create_render_pipeline(&RenderPipelineDescriptor {
                    label: format!("textured_draw_pipeline_{fs_entry_point}").as_str().into(),
                    layout: Some(&textured_pipeline_layout),
                    vertex: VertexState {
                        module: draw_shader,
                        entry_point: Some(vs_entry_point),
                        compilation_options: PipelineCompilationOptions::default(),
                        buffers: &[vertex_buffer_layout],
                    },
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
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
                        module: draw_shader,
                        entry_point: Some(fs_entry_point),
                        compilation_options: PipelineCompilationOptions::default(),
                        targets: &[Some(ColorTargetState {
                            format: TextureFormat::Rgba8Unorm,
                            blend,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            };

        let new_textured_triangle_pipeline = |fs_entry_point: &str, blend: Option<BlendState>| {
            new_textured_pipeline(TexturedVertex::LAYOUT, "vs_textured", fs_entry_point, blend)
        };

        let textured_opaque_pipeline = new_textured_triangle_pipeline("fs_textured_opaque", None);

        let textured_opaque_mask_pipeline =
            new_textured_triangle_pipeline("fs_textured_opaque", Some(Self::REPLACE_CHECK_MASK));

        let textured_average_pipeline =
            new_textured_triangle_pipeline("fs_textured_average", Some(Self::AVERAGE_BLEND));

        let textured_add_pipeline = new_textured_triangle_pipeline(
            "fs_textured_add",
            Some(Self::ADDITIVE_BLEND_DUAL_SOURCE),
        );

        let textured_subtract_pipeline_opaque =
            new_textured_triangle_pipeline("fs_textured_subtract_opaque_texels", None);

        let textured_subtract_pipeline_transparent = new_textured_triangle_pipeline(
            "fs_textured_subtract_transparent_texels",
            Some(Self::SUBTRACTIVE_BLEND),
        );

        let textured_add_quarter_pipeline = new_textured_triangle_pipeline(
            "fs_textured_add_quarter",
            Some(Self::ADD_QUARTER_BLEND),
        );

        let new_textured_rect_pipeline = |fs_entry_point: &str, blend: Option<BlendState>| {
            new_textured_pipeline(
                TexturedRectVertex::LAYOUT,
                "vs_textured_rect",
                fs_entry_point,
                blend,
            )
        };

        let textured_opaque_rect_pipeline =
            new_textured_rect_pipeline("fs_textured_rect_opaque", None);

        let textured_opaque_rect_mask_pipeline =
            new_textured_rect_pipeline("fs_textured_rect_opaque", Some(Self::REPLACE_CHECK_MASK));

        let textured_average_rect_pipeline =
            new_textured_rect_pipeline("fs_textured_rect_average", Some(Self::AVERAGE_BLEND));

        let textured_add_rect_pipeline = new_textured_rect_pipeline(
            "fs_textured_rect_add",
            Some(Self::ADDITIVE_BLEND_DUAL_SOURCE),
        );

        let textured_subtract_rect_pipeline_opaque =
            new_textured_rect_pipeline("fs_textured_rect_subtract_opaque_texels", None);

        let textured_subtract_rect_pipeline_transparent = new_textured_rect_pipeline(
            "fs_textured_rect_subtract_transparent_texels",
            Some(Self::SUBTRACTIVE_BLEND),
        );

        let textured_add_quarter_rect_pipeline = new_textured_rect_pipeline(
            "fs_textured_rect_add_quarter",
            Some(Self::ADD_QUARTER_BLEND),
        );

        Self {
            untextured_buffer: Vec::with_capacity(Self::INITIAL_BUFFER_CAPACITY as usize),
            untextured_opaque_pipeline,
            untextured_opaque_mask_pipeline,
            untextured_average_pipeline,
            untextured_add_pipeline,
            untextured_add_mask_pipeline,
            untextured_subtract_pipeline,
            untextured_subtract_mask_pipeline,
            untextured_add_quarter_pipeline,
            untextured_add_quarter_mask_pipeline,
            textured_buffer: Vec::with_capacity(Self::INITIAL_BUFFER_CAPACITY as usize),
            textured_bind_group,
            textured_opaque_pipeline,
            textured_opaque_mask_pipeline,
            textured_average_pipeline,
            textured_add_pipeline,
            textured_subtract_pipeline_opaque,
            textured_subtract_pipeline_transparent,
            textured_add_quarter_pipeline,
            textured_rect_buffer: Vec::with_capacity(Self::INITIAL_BUFFER_CAPACITY as usize),
            textured_rect_indices: Vec::with_capacity(Self::INITIAL_BUFFER_CAPACITY as usize),
            textured_opaque_rect_pipeline,
            textured_opaque_rect_mask_pipeline,
            textured_average_rect_pipeline,
            textured_add_rect_pipeline,
            textured_subtract_rect_pipeline_opaque,
            textured_subtract_rect_pipeline_transparent,
            textured_add_quarter_rect_pipeline,
            batches: Vec::with_capacity(Self::INITIAL_BUFFER_CAPACITY as usize),
        }
    }

    pub fn add_triangle(&mut self, args: &DrawTriangleArgs, draw_settings: &DrawSettings) {
        add_triangle_to_batch(
            args,
            draw_settings,
            &mut self.untextured_buffer,
            &mut self.textured_buffer,
            &mut self.batches,
            |pipeline| {
                pipeline != DrawPipeline::TexturedTriangle(Some(SemiTransparencyMode::Subtract))
            },
        );
    }

    pub fn add_rectangle(&mut self, args: &DrawRectangleArgs, draw_settings: &DrawSettings) {
        add_rectangle_to_batch(
            args,
            draw_settings,
            &mut self.untextured_buffer,
            &mut self.textured_buffer,
            &mut self.textured_rect_buffer,
            &mut self.batches,
            |pipeline| {
                pipeline != DrawPipeline::TexturedTriangle(Some(SemiTransparencyMode::Subtract))
            },
        );
    }

    pub fn add_line(&mut self, args: &DrawLineArgs, draw_settings: &DrawSettings) {
        add_line_to_batch(args, draw_settings, &mut self.untextured_buffer, &mut self.batches);
    }

    pub fn prepare(&mut self, device: &Device) -> DrawBuffers {
        let untextured_triangle = device.create_buffer_init(&BufferInitDescriptor {
            label: "untextured_triangle_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.untextured_buffer),
            usage: BufferUsages::VERTEX,
        });

        let textured_triangle = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_triangle_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_buffer),
            usage: BufferUsages::VERTEX,
        });

        let textured_rectangle_vertex = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_rectangle_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_rect_buffer),
            usage: BufferUsages::VERTEX,
        });

        populate_rect_index_buffer(&self.textured_rect_buffer, &mut self.textured_rect_indices);
        let textured_rectangle_index = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_rectangle_index_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_rect_indices),
            usage: BufferUsages::INDEX,
        });

        self.untextured_buffer.clear();
        self.textured_buffer.clear();
        self.textured_rect_buffer.clear();
        self.textured_rect_indices.clear();

        DrawBuffers {
            untextured_triangle,
            textured_triangle,
            textured_rectangle_vertex,
            textured_rectangle_index,
        }
    }

    pub fn draw<'rpass>(
        &'rpass mut self,
        buffers: &'rpass DrawBuffers,
        config: InternalConfig,
        render_pass: &mut RenderPass<'rpass>,
    ) {
        for batch in self.batches.drain(..) {
            let draw_settings = ShaderDrawSettings::new(&batch.draw_settings, config);
            set_scissor_rect(render_pass, &batch.draw_settings, config.resolution_scale);

            let check_mask_bit = batch.draw_settings.check_mask_bit;

            match batch.pipeline {
                DrawPipeline::UntexturedTriangle(semi_transparency_mode) => {
                    let pipeline = match semi_transparency_mode {
                        Some(SemiTransparencyMode::Average) => &self.untextured_average_pipeline,
                        Some(SemiTransparencyMode::Add) => {
                            if check_mask_bit {
                                &self.untextured_add_mask_pipeline
                            } else {
                                &self.untextured_add_pipeline
                            }
                        }
                        Some(SemiTransparencyMode::Subtract) => {
                            if check_mask_bit {
                                &self.untextured_subtract_mask_pipeline
                            } else {
                                &self.untextured_subtract_pipeline
                            }
                        }
                        Some(SemiTransparencyMode::AddQuarter) => {
                            if check_mask_bit {
                                &self.untextured_add_quarter_mask_pipeline
                            } else {
                                &self.untextured_add_quarter_pipeline
                            }
                        }
                        None => {
                            if check_mask_bit {
                                &self.untextured_opaque_mask_pipeline
                            } else {
                                &self.untextured_opaque_pipeline
                            }
                        }
                    };

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_vertex_buffer(0, buffers.untextured_triangle.slice(..));

                    render_pass.draw(batch.start..batch.end, 0..1);
                }
                DrawPipeline::TexturedTriangle(Some(SemiTransparencyMode::Subtract)) => {
                    render_pass.set_pipeline(&self.textured_subtract_pipeline_opaque);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffers.textured_triangle.slice(..));

                    render_pass.draw(batch.start..batch.end, 0..1);

                    render_pass.set_pipeline(&self.textured_subtract_pipeline_transparent);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));

                    render_pass.draw(batch.start..batch.end, 0..1);
                }
                DrawPipeline::TexturedTriangle(semi_transparency_mode) => {
                    let pipeline = match semi_transparency_mode {
                        Some(SemiTransparencyMode::Average) => &self.textured_average_pipeline,
                        Some(SemiTransparencyMode::Add) => &self.textured_add_pipeline,
                        Some(SemiTransparencyMode::Subtract) => unreachable!(),
                        Some(SemiTransparencyMode::AddQuarter) => {
                            &self.textured_add_quarter_pipeline
                        }
                        None => {
                            if check_mask_bit {
                                &self.textured_opaque_mask_pipeline
                            } else {
                                &self.textured_opaque_pipeline
                            }
                        }
                    };

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffers.textured_triangle.slice(..));

                    render_pass.draw(batch.start..batch.end, 0..1);
                }
                DrawPipeline::TexturedRectangle(Some(SemiTransparencyMode::Subtract)) => {
                    render_pass.set_pipeline(&self.textured_subtract_rect_pipeline_opaque);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffers.textured_rectangle_vertex.slice(..));
                    render_pass.set_index_buffer(
                        buffers.textured_rectangle_index.slice(..),
                        IndexFormat::Uint32,
                    );

                    let start_indexed = batch.start * 3 / 2;
                    let end_indexed = batch.end * 3 / 2;
                    render_pass.draw_indexed(start_indexed..end_indexed, 0, 0..1);

                    render_pass.set_pipeline(&self.textured_subtract_rect_pipeline_transparent);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));

                    render_pass.draw_indexed(start_indexed..end_indexed, 0, 0..1);
                }
                DrawPipeline::TexturedRectangle(semi_transparency_mode) => {
                    let pipeline = match semi_transparency_mode {
                        Some(SemiTransparencyMode::Average) => &self.textured_average_rect_pipeline,
                        Some(SemiTransparencyMode::Add) => &self.textured_add_rect_pipeline,
                        Some(SemiTransparencyMode::AddQuarter) => {
                            &self.textured_add_quarter_rect_pipeline
                        }
                        None => {
                            if check_mask_bit {
                                &self.textured_opaque_rect_mask_pipeline
                            } else {
                                &self.textured_opaque_rect_pipeline
                            }
                        }
                        Some(SemiTransparencyMode::Subtract) => unreachable!(),
                    };

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, buffers.textured_rectangle_vertex.slice(..));
                    render_pass.set_index_buffer(
                        buffers.textured_rectangle_index.slice(..),
                        IndexFormat::Uint32,
                    );

                    let start_indexed = batch.start * 3 / 2;
                    let end_indexed = batch.end * 3 / 2;
                    render_pass.draw_indexed(start_indexed..end_indexed, 0, 0..1);
                }
            }
        }
    }
}

fn set_scissor_rect(
    render_pass: &mut RenderPass<'_>,
    draw_settings: &DrawSettings,
    resolution_scale: u32,
) {
    let width =
        (draw_settings.draw_area_bottom_right.x - draw_settings.draw_area_top_left.x + 1) as u32;
    let height =
        (draw_settings.draw_area_bottom_right.y - draw_settings.draw_area_top_left.y + 1) as u32;

    render_pass.set_scissor_rect(
        resolution_scale * draw_settings.draw_area_top_left.x as u32,
        resolution_scale * draw_settings.draw_area_top_left.y as u32,
        resolution_scale * width,
        resolution_scale * height,
    );
}

fn add_triangle_to_batch(
    args: &DrawTriangleArgs,
    draw_settings: &DrawSettings,
    untextured_buffer: &mut Vec<UntexturedVertex>,
    textured_buffer: &mut Vec<TexturedVertex>,
    batches: &mut Vec<DrawBatch>,
    can_share_batch_fn: impl Fn(DrawPipeline) -> bool,
) {
    let semi_transparency_mode = args.semi_transparent.then_some(args.semi_transparency_mode);
    let pipeline = match &args.texture_mapping {
        Some(_) => DrawPipeline::TexturedTriangle(semi_transparency_mode),
        None => DrawPipeline::UntexturedTriangle(semi_transparency_mode),
    };

    if !can_share_batch_fn(pipeline)
        || !batches.last().is_some_and(|batch| batch.matches(draw_settings, pipeline))
    {
        let start = match &args.texture_mapping {
            Some(_) => textured_buffer.len() as u32,
            None => untextured_buffer.len() as u32,
        };
        batches.push(DrawBatch {
            draw_settings: draw_settings.clone(),
            pipeline,
            start,
            end: start,
        });
    }

    let positions = args
        .vertices
        .map(|v| [v.x + draw_settings.draw_offset.x, v.y + draw_settings.draw_offset.y]);
    let colors = match args.shading {
        TriangleShading::Flat(color) => [color; 3],
        TriangleShading::Gouraud(colors) => colors,
    };

    let precise_positions = args.pgxp_vertices.map_or_else(
        || array::from_fn(|i| [positions[i][0] as f32, positions[i][1] as f32, 0.0]),
        |pgxp_vertices| {
            let all_z_valid = pgxp_vertices.iter().all(|v| v.z.is_some());

            pgxp_vertices.map(|v| {
                let z = if all_z_valid { v.z.unwrap_or(0) } else { 0 };
                [
                    (v.x + f64::from(draw_settings.draw_offset.x)) as f32,
                    (v.y + f64::from(draw_settings.draw_offset.y)) as f32,
                    z.into(),
                ]
            })
        },
    );

    let gouraud_shaded = matches!(args.shading, TriangleShading::Gouraud(..));
    match &args.texture_mapping {
        Some(mapping) => {
            let ditherable = gouraud_shaded || mapping.mode == TextureMappingMode::Modulated;
            textured_buffer.extend(TexturedVertex::new_vertices(
                &positions,
                &precise_positions,
                colors,
                mapping,
                ditherable,
            ));
        }
        None => {
            for (i, position) in precise_positions.into_iter().enumerate() {
                untextured_buffer.push(UntexturedVertex {
                    position,
                    color: [colors[i].r.into(), colors[i].g.into(), colors[i].b.into()],
                    ditherable: gouraud_shaded.into(),
                });
            }
        }
    }

    batches.last_mut().unwrap().end += 3;
}

fn add_rectangle_to_batch(
    args: &DrawRectangleArgs,
    draw_settings: &DrawSettings,
    untextured_buffer: &mut Vec<UntexturedVertex>,
    textured_buffer: &mut Vec<TexturedVertex>,
    textured_rect_buffer: &mut Vec<TexturedRectVertex>,
    batches: &mut Vec<DrawBatch>,
    can_share_batch_fn: impl Copy + Fn(DrawPipeline) -> bool,
) {
    match &args.texture_mapping {
        Some(texture_mapping) => {
            let semi_transparency_mode =
                args.semi_transparent.then_some(args.semi_transparency_mode);
            let pipeline = DrawPipeline::TexturedRectangle(semi_transparency_mode);

            if !can_share_batch_fn(pipeline)
                || !batches.last().is_some_and(|batch| batch.matches(draw_settings, pipeline))
            {
                let start = textured_rect_buffer.len() as u32;
                batches.push(DrawBatch {
                    draw_settings: draw_settings.clone(),
                    pipeline,
                    start,
                    end: start,
                });
            }

            let vertices = TexturedRectVertex::new_vertices(args, texture_mapping, draw_settings);
            textured_rect_buffer.extend(vertices);

            batches.last_mut().unwrap().end += 4;
        }
        None => {
            let v = rect_vertices(args, Vertex::new(0, 0));
            for vertices in [[v[0], v[1], v[2]], [v[1], v[2], v[3]]] {
                add_triangle_to_batch(
                    &DrawTriangleArgs {
                        vertices,
                        pgxp_vertices: None,
                        shading: TriangleShading::Flat(args.color),
                        semi_transparent: args.semi_transparent,
                        semi_transparency_mode: args.semi_transparency_mode,
                        texture_mapping: None,
                    },
                    draw_settings,
                    untextured_buffer,
                    textured_buffer,
                    batches,
                    can_share_batch_fn,
                );
            }
        }
    }
}

fn add_line_to_batch(
    args: &DrawLineArgs,
    draw_settings: &DrawSettings,
    untextured_buffer: &mut Vec<UntexturedVertex>,
    batches: &mut Vec<DrawBatch>,
) {
    let dy = args.vertices[1].y - args.vertices[0].y;
    let dx = args.vertices[1].x - args.vertices[0].x;

    let v = args.vertices.map(|v| v + draw_settings.draw_offset);
    let positions = if dx == 0 || dx.abs() <= dy.abs() {
        // Vertically oriented line
        if v[0].y <= v[1].y {
            // First vertex is higher
            [[v[0].x, v[0].y], [v[0].x + 1, v[0].y], [v[1].x, v[1].y + 1], [v[1].x + 1, v[1].y + 1]]
        } else {
            // First vertex is lower
            [[v[0].x, v[0].y + 1], [v[0].x + 1, v[0].y + 1], [v[1].x, v[1].y], [v[1].x + 1, v[1].y]]
        }
    } else {
        // Horizontally oriented line
        if v[0].x <= v[1].x {
            // First vertex is farther left
            [[v[0].x, v[0].y], [v[0].x, v[0].y + 1], [v[1].x + 1, v[1].y], [v[1].x + 1, v[1].y + 1]]
        } else {
            // First vertex is farther right
            [[v[0].x + 1, v[0].y], [v[0].x + 1, v[0].y + 1], [v[1].x, v[1].y], [v[1].x, v[1].y + 1]]
        }
    };

    let colors = match args.shading {
        LineShading::Flat(color) => [color; 4],
        LineShading::Gouraud(colors) => [colors[0], colors[0], colors[1], colors[1]],
    };
    let colors: [[u32; 3]; 4] =
        colors.map(|color| [color.r.into(), color.g.into(), color.b.into()]);

    let pipeline = DrawPipeline::UntexturedTriangle(
        args.semi_transparent.then_some(args.semi_transparency_mode),
    );
    if !batches.last().is_some_and(|batch| batch.matches(draw_settings, pipeline)) {
        let start = untextured_buffer.len() as u32;
        batches.push(DrawBatch {
            draw_settings: draw_settings.clone(),
            pipeline,
            start,
            end: start,
        });
    }

    for range in [0..3, 1..4] {
        for i in range {
            untextured_buffer.push(UntexturedVertex {
                position: [positions[i][0] as f32, positions[i][1] as f32, 0.0],
                color: colors[i],
                ditherable: true.into(),
            });
        }
    }

    batches.last_mut().unwrap().end += 6;
}

fn populate_rect_index_buffer(vertex_buffer: &[TexturedRectVertex], index_buffer: &mut Vec<u32>) {
    for i in (0..vertex_buffer.len()).step_by(4) {
        let i = i as u32;
        index_buffer.extend([i, i + 1, i + 2, i + 1, i + 2, i + 3]);
    }
}

#[derive(Debug)]
pub struct MaskBitPipelines {
    untextured_buffer: Vec<UntexturedVertex>,
    textured_buffer: Vec<TexturedVertex>,
    textured_rect_buffer: Vec<TexturedRectVertex>,
    textured_rect_indices: Vec<u32>,
    untextured_bind_group: BindGroup,
    untextured_average_pipeline: RenderPipeline,
    textured_bind_group: BindGroup,
    textured_average_pipeline: RenderPipeline,
    textured_add_pipeline: RenderPipeline,
    textured_subtract_pipeline: RenderPipeline,
    textured_add_quarter_pipeline: RenderPipeline,
    textured_average_rect_pipeline: RenderPipeline,
    textured_add_rect_pipeline: RenderPipeline,
    textured_subtract_rect_pipeline: RenderPipeline,
    textured_add_quarter_rect_pipeline: RenderPipeline,
    batches: Vec<DrawBatch>,
}

impl MaskBitPipelines {
    pub fn new(
        device: &Device,
        draw_shader: &ShaderModule,
        native_vram: &Texture,
        scaled_vram: &Texture,
        scaled_vram_copy: &Texture,
    ) -> Self {
        let native_vram_view = native_vram.create_view(&TextureViewDescriptor::default());
        let scaled_vram_view = scaled_vram.create_view(&TextureViewDescriptor::default());
        let scaled_vram_copy_view = scaled_vram_copy.create_view(&TextureViewDescriptor::default());

        let untextured_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: "untextured_mask_bind_group_layout".into(),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::ReadWrite,
                        format: scaled_vram.format(),
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let untextured_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "untextured_mask_bind_group".into(),
            layout: &untextured_bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&scaled_vram_view),
            }],
        });

        let untextured_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "untextured_mask_pipeline_layout".into(),
            bind_group_layouts: &[Some(&untextured_bind_group_layout)],
            immediate_size: size_of::<ShaderDrawSettings>() as u32,
        });

        let mask_shader =
            device.create_shader_module(include_wgsl_concat!("draw_common.wgsl", "maskbit.wgsl"));

        let untextured_average_pipeline =
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: "untextured_mask_average_pipeline".into(),
                layout: Some(&untextured_pipeline_layout),
                vertex: VertexState {
                    module: draw_shader,
                    entry_point: Some("vs_untextured"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[UntexturedVertex::LAYOUT],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
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
                    module: &mask_shader,
                    entry_point: Some("fs_untextured_average"),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::empty(),
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let textured_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: "textured_mask_bind_group_layout".into(),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadWrite,
                            format: scaled_vram.format(),
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadOnly,
                            format: native_vram.format(),
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::FRAGMENT,
                        ty: BindingType::StorageTexture {
                            access: StorageTextureAccess::ReadOnly,
                            format: scaled_vram_copy.format(),
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let textured_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "textured_mask_bind_group".into(),
            layout: &textured_bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&scaled_vram_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&native_vram_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&scaled_vram_copy_view),
                },
            ],
        });

        let textured_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "textured_mask_pipeline_layout".into(),
            bind_group_layouts: &[Some(&textured_bind_group_layout)],
            immediate_size: size_of::<ShaderDrawSettings>() as u32,
        });

        let new_textured_pipeline = |vertex_buffer_layout: VertexBufferLayout<'_>,
                                     vs_entry_point: &str,
                                     fs_entry_point: &str| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: format!("textured_mask_{fs_entry_point}_pipeline").as_str().into(),
                layout: Some(&textured_pipeline_layout),
                vertex: VertexState {
                    module: draw_shader,
                    entry_point: Some(vs_entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[vertex_buffer_layout],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
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
                    module: &mask_shader,
                    entry_point: Some(fs_entry_point),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::empty(),
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };

        let new_textured_triangle_pipeline = |fs_entry_point: &str| {
            new_textured_pipeline(TexturedVertex::LAYOUT, "vs_textured", fs_entry_point)
        };

        let textured_average_pipeline = new_textured_triangle_pipeline("fs_textured_average");
        let textured_add_pipeline = new_textured_triangle_pipeline("fs_textured_add");
        let textured_subtract_pipeline = new_textured_triangle_pipeline("fs_textured_subtract");
        let textured_add_quarter_pipeline =
            new_textured_triangle_pipeline("fs_textured_add_quarter");

        let new_textured_rect_pipeline = |fs_entry_point: &str| {
            new_textured_pipeline(TexturedRectVertex::LAYOUT, "vs_textured_rect", fs_entry_point)
        };

        let textured_average_rect_pipeline = new_textured_rect_pipeline("fs_textured_rect_average");
        let textured_add_rect_pipeline = new_textured_rect_pipeline("fs_textured_rect_add");
        let textured_subtract_rect_pipeline =
            new_textured_rect_pipeline("fs_textured_rect_subtract");
        let textured_add_quarter_rect_pipeline =
            new_textured_rect_pipeline("fs_textured_rect_add_quarter");

        Self {
            untextured_buffer: Vec::with_capacity(DrawPipelines::INITIAL_BUFFER_CAPACITY as usize),
            textured_buffer: Vec::with_capacity(DrawPipelines::INITIAL_BUFFER_CAPACITY as usize),
            textured_rect_buffer: Vec::with_capacity(
                DrawPipelines::INITIAL_BUFFER_CAPACITY as usize,
            ),
            textured_rect_indices: Vec::with_capacity(
                DrawPipelines::INITIAL_BUFFER_CAPACITY as usize,
            ),
            untextured_bind_group,
            untextured_average_pipeline,
            textured_bind_group,
            textured_average_pipeline,
            textured_add_pipeline,
            textured_subtract_pipeline,
            textured_add_quarter_pipeline,
            textured_average_rect_pipeline,
            textured_add_rect_pipeline,
            textured_subtract_rect_pipeline,
            textured_add_quarter_rect_pipeline,
            batches: Vec::with_capacity(DrawPipelines::INITIAL_BUFFER_CAPACITY as usize),
        }
    }

    pub fn add_triangle(&mut self, args: &DrawTriangleArgs, draw_settings: &DrawSettings) {
        add_triangle_to_batch(
            args,
            draw_settings,
            &mut self.untextured_buffer,
            &mut self.textured_buffer,
            &mut self.batches,
            |_| true,
        );
    }

    pub fn add_rectangle(&mut self, args: &DrawRectangleArgs, draw_settings: &DrawSettings) {
        add_rectangle_to_batch(
            args,
            draw_settings,
            &mut self.untextured_buffer,
            &mut self.textured_buffer,
            &mut self.textured_rect_buffer,
            &mut self.batches,
            |_| true,
        );
    }

    pub fn add_line(&mut self, args: &DrawLineArgs, draw_settings: &DrawSettings) {
        add_line_to_batch(args, draw_settings, &mut self.untextured_buffer, &mut self.batches);
    }

    pub fn prepare(&mut self, device: &Device) -> DrawBuffers {
        let untextured_triangle = device.create_buffer_init(&BufferInitDescriptor {
            label: "untextured_triangle_mask_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.untextured_buffer),
            usage: BufferUsages::VERTEX,
        });

        let textured_triangle = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_triangle_mask_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_buffer),
            usage: BufferUsages::VERTEX,
        });

        let textured_rectangle_vertex = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_rect_mask_vertex_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_rect_buffer),
            usage: BufferUsages::VERTEX,
        });

        populate_rect_index_buffer(&self.textured_rect_buffer, &mut self.textured_rect_indices);
        let textured_rectangle_index = device.create_buffer_init(&BufferInitDescriptor {
            label: "textured_rect_mask_index_buffer".into(),
            contents: bytemuck::cast_slice(&self.textured_rect_indices),
            usage: BufferUsages::INDEX,
        });

        self.untextured_buffer.clear();
        self.textured_buffer.clear();
        self.textured_rect_buffer.clear();
        self.textured_rect_indices.clear();

        DrawBuffers {
            untextured_triangle,
            textured_triangle,
            textured_rectangle_vertex,
            textured_rectangle_index,
        }
    }

    pub fn draw<'rpass>(
        &'rpass mut self,
        buffers: &'rpass DrawBuffers,
        config: InternalConfig,
        render_pass: &mut RenderPass<'rpass>,
    ) {
        for batch in self.batches.drain(..) {
            let draw_settings = ShaderDrawSettings::new(&batch.draw_settings, config);
            set_scissor_rect(render_pass, &batch.draw_settings, config.resolution_scale);

            match batch.pipeline {
                DrawPipeline::UntexturedTriangle(Some(SemiTransparencyMode::Average)) => {
                    render_pass.set_pipeline(&self.untextured_average_pipeline);
                    render_pass.set_bind_group(0, &self.untextured_bind_group, &[]);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_vertex_buffer(0, buffers.untextured_triangle.slice(..));

                    for start in (batch.start..batch.end).step_by(3) {
                        render_pass.draw(start..start + 3, 0..1);
                    }
                }
                DrawPipeline::UntexturedTriangle(semi_transparency_mode) => panic!(
                    "Unexpected untextured semi-transparency mode for mask bit pipeline: {semi_transparency_mode:?}"
                ),
                DrawPipeline::TexturedTriangle(semi_transparency_mode) => {
                    let pipeline = match semi_transparency_mode {
                        Some(SemiTransparencyMode::Average) => &self.textured_average_pipeline,
                        Some(SemiTransparencyMode::Add) => &self.textured_add_pipeline,
                        Some(SemiTransparencyMode::Subtract) => &self.textured_subtract_pipeline,
                        Some(SemiTransparencyMode::AddQuarter) => {
                            &self.textured_add_quarter_pipeline
                        }
                        None => panic!(
                            "mask bit pipeline invoked for a non-semi-transparent textured triangle"
                        ),
                    };

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_vertex_buffer(0, buffers.textured_triangle.slice(..));

                    for start in (batch.start..batch.end).step_by(3) {
                        render_pass.draw(start..start + 3, 0..1);
                    }
                }
                DrawPipeline::TexturedRectangle(semi_transparency_mode) => {
                    let pipeline = match semi_transparency_mode {
                        Some(SemiTransparencyMode::Average) => &self.textured_average_rect_pipeline,
                        Some(SemiTransparencyMode::Add) => &self.textured_add_rect_pipeline,
                        Some(SemiTransparencyMode::Subtract) => {
                            &self.textured_subtract_rect_pipeline
                        }
                        Some(SemiTransparencyMode::AddQuarter) => {
                            &self.textured_add_quarter_rect_pipeline
                        }
                        None => panic!(
                            "mask bit pipeline invoked for a non-semi-transparent textured rectangle"
                        ),
                    };

                    render_pass.set_pipeline(pipeline);
                    render_pass.set_bind_group(0, &self.textured_bind_group, &[]);
                    render_pass.set_immediates(0, bytemuck::cast_slice(&[draw_settings]));
                    render_pass.set_vertex_buffer(0, buffers.textured_rectangle_vertex.slice(..));
                    render_pass.set_index_buffer(
                        buffers.textured_rectangle_index.slice(..),
                        IndexFormat::Uint32,
                    );

                    let indexed_start = batch.start * 3 / 2;
                    let indexed_end = batch.end * 3 / 2;
                    for start in (indexed_start..indexed_end).step_by(6) {
                        render_pass.draw_indexed(start..start + 6, 0, 0..1);
                    }
                }
            }
        }
    }
}
