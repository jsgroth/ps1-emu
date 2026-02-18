use crate::gpu::Color;
use crate::gpu::rasterizer::wgpuhardware::{VRAM_HEIGHT, VRAM_WIDTH};
use crate::gpu::rasterizer::{CpuVramBlitArgs, VramVramBlitArgs};
use bytemuck::{Pod, Zeroable};
use std::{iter, mem};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBinding, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoderDescriptor, ComputePass,
    ComputePipeline, ComputePipelineDescriptor, Device, ImageCopyBuffer, ImageDataLayout, Maintain,
    MapMode, PipelineCompilationOptions, PipelineLayoutDescriptor, PushConstantRange, Queue,
    ShaderStages, StorageTextureAccess, Texture, TextureViewDescriptor, TextureViewDimension,
};

// Must match CpuVramBlitArgs in cpuvramblit.wgsl
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ShaderCpuVramBlitArgs {
    position: [u32; 2],
    size: [u32; 2],
    force_mask_bit: u32,
    check_mask_bit: u32,
}

#[derive(Debug)]
pub struct CpuVramBlitPipeline {
    ram_buffer: Vec<u32>,
    bind_group_0: BindGroup,
    bind_group_layout_1: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl CpuVramBlitPipeline {
    // Must match X/Y workgroup size in shader
    const WORKGROUP_SIZE: u32 = 16;

    pub fn new(device: &Device, native_vram: &Texture) -> Self {
        let bind_group_layout_0 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: "cpu_vram_blit_bind_group_layout_0".into(),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: native_vram.format(),
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let bind_group_layout_1 = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: "cpu_vram_blit_bind_group_layout_1".into(),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let native_vram_view = native_vram.create_view(&TextureViewDescriptor::default());
        let bind_group_0 = device.create_bind_group(&BindGroupDescriptor {
            label: "cpu_vram_blit_bind_group".into(),
            layout: &bind_group_layout_0,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&native_vram_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "cpu_vram_blit_pipeline_layout".into(),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..mem::size_of::<ShaderCpuVramBlitArgs>() as u32,
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("cpuvramblit.wgsl"));
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "cpu_vram_blit_pipeline".into(),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "cpu_vram_blit",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            ram_buffer: Vec::with_capacity((VRAM_WIDTH * VRAM_HEIGHT) as usize),
            bind_group_0,
            bind_group_layout_1,
            pipeline,
        }
    }

    pub fn prepare(
        &mut self,
        device: &Device,
        args: &CpuVramBlitArgs,
        buffer: &[u16],
    ) -> BindGroup {
        let copy_len = (args.width * args.height) as usize;

        self.ram_buffer.clear();
        self.ram_buffer.extend(buffer.iter().copied().map(u32::from).take(copy_len));

        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: "cpu_vram_blit_buffer".into(),
            contents: bytemuck::cast_slice(&self.ram_buffer),
            usage: BufferUsages::STORAGE,
        });

        device.create_bind_group(&BindGroupDescriptor {
            label: "cpu_vram_blit_bind_group_1".into(),
            layout: &self.bind_group_layout_1,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        })
    }

    pub fn dispatch<'cpass>(
        &'cpass self,
        args: &CpuVramBlitArgs,
        bind_group_1: &'cpass BindGroup,
        compute_pass: &mut ComputePass<'cpass>,
    ) {
        let shader_args = ShaderCpuVramBlitArgs {
            position: [args.x, args.y],
            size: [args.width, args.height],
            force_mask_bit: args.force_mask_bit.into(),
            check_mask_bit: args.check_mask_bit.into(),
        };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group_0, &[]);
        compute_pass.set_bind_group(1, bind_group_1, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[shader_args]));

        let x_groups = args.width / Self::WORKGROUP_SIZE
            + u32::from(!args.width.is_multiple_of(Self::WORKGROUP_SIZE));
        let y_groups = args.height / Self::WORKGROUP_SIZE
            + u32::from(!args.height.is_multiple_of(Self::WORKGROUP_SIZE));
        compute_pass.dispatch_workgroups(x_groups, y_groups, 1);
    }
}

#[derive(Debug)]
pub struct VramCpuBlitter {
    ram_buffer: Vec<u16>,
    blit_buffer: Buffer,
    pub out_of_sync: bool,
}

impl VramCpuBlitter {
    pub fn new(device: &Device) -> Self {
        let blit_buffer = device.create_buffer(&BufferDescriptor {
            label: "vram_cpu_blit_buffer".into(),
            size: (4 * VRAM_WIDTH * VRAM_HEIGHT).into(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            ram_buffer: Vec::with_capacity((VRAM_WIDTH * VRAM_HEIGHT) as usize),
            blit_buffer,
            out_of_sync: true,
        }
    }

    pub fn blit_from_gpu(
        &mut self,
        device: &Device,
        queue: &Queue,
        native_vram: &Texture,
        draw_command_buffer: Option<CommandBuffer>,
    ) {
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

        encoder.copy_texture_to_buffer(
            native_vram.as_image_copy(),
            ImageCopyBuffer {
                buffer: &self.blit_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * VRAM_WIDTH),
                    rows_per_image: None,
                },
            },
            native_vram.size(),
        );

        queue.submit(draw_command_buffer.into_iter().chain(iter::once(encoder.finish())));

        let blit_buffer_slice = self.blit_buffer.slice(..);
        blit_buffer_slice.map_async(MapMode::Read, Result::unwrap);
        device.poll(Maintain::Wait);

        self.ram_buffer.clear();
        {
            let map_buffer_view = blit_buffer_slice.get_mapped_range();
            for chunk in map_buffer_view.chunks_exact(4) {
                self.ram_buffer.push(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }

        self.blit_buffer.unmap();
    }

    pub fn copy_blit_output(&self, x: u32, y: u32, width: u32, height: u32, out: &mut Vec<u16>) {
        for dy in 0..height {
            let y = (y + dy) & (VRAM_HEIGHT - 1);
            for dx in 0..width {
                let x = (x + dx) & (VRAM_WIDTH - 1);
                let vram_addr = y * VRAM_WIDTH + x;
                out.push(self.ram_buffer[vram_addr as usize]);
            }
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ShaderVramCopyArgs {
    source: [u32; 2],
    destination: [u32; 2],
    size: [u32; 2],
    force_mask_bit: u32,
    check_mask_bit: u32,
    resolution_scale: u32,
}

impl ShaderVramCopyArgs {
    fn new(args: &VramVramBlitArgs, resolution_scale: u32) -> Self {
        Self {
            source: [args.source_x, args.source_y].map(|v| v * resolution_scale),
            destination: [args.dest_x, args.dest_y].map(|v| v * resolution_scale),
            size: [args.width, args.height].map(|v| v * resolution_scale),
            force_mask_bit: args.force_mask_bit.into(),
            check_mask_bit: args.check_mask_bit.into(),
            resolution_scale,
        }
    }
}

#[derive(Debug)]
pub struct VramCopyPipeline {
    bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl VramCopyPipeline {
    const WORKGROUP_SIZE: u32 = 16;

    pub fn new(device: &Device, scaled_vram: &Texture) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: "vram_copy_bind_group_layout".into(),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::ReadWrite,
                    format: scaled_vram.format(),
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let scaled_vram_view = scaled_vram.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "vram_copy_bind_group".into(),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&scaled_vram_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "vram_copy_pipeline_layout".into(),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..mem::size_of::<ShaderVramCopyArgs>() as u32,
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("vramcopy.wgsl"));
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "vram_copy_pipeline".into(),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "vram_copy",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { bind_group, pipeline }
    }

    pub fn dispatch<'cpass>(
        &'cpass self,
        args: &VramVramBlitArgs,
        resolution_scale: u32,
        compute_pass: &mut ComputePass<'cpass>,
    ) {
        let vram_copy_args = ShaderVramCopyArgs::new(args, resolution_scale);

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[vram_copy_args]));
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let x_workgroups = (resolution_scale * args.width).div_ceil(Self::WORKGROUP_SIZE);
        let y_workgroups = (resolution_scale * args.height).div_ceil(Self::WORKGROUP_SIZE);
        compute_pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
struct ShaderVramFillArgs {
    position: [u32; 2],
    size: [u32; 2],
    color: u32,
}

#[derive(Debug)]
pub struct VramFillPipeline {
    bind_group: BindGroup,
    pipeline: ComputePipeline,
}

impl VramFillPipeline {
    const WORKGROUP_SIZE: u32 = 16;

    pub fn new(device: &Device, native_vram: &Texture) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: "vram_fill_bind_group_layout".into(),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: native_vram.format(),
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            }],
        });

        let native_vram_view = native_vram.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: "vram_fill_bind_group".into(),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&native_vram_view),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: "vram_fill_pipeline_layout".into(),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..mem::size_of::<ShaderVramFillArgs>() as u32,
            }],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("vramfill.wgsl"));
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: "vram_fill_pipeline".into(),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "vram_fill",
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        Self { bind_group, pipeline }
    }

    pub fn dispatch<'cpass>(
        &'cpass self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        color: Color,
        compute_pass: &mut ComputePass<'cpass>,
    ) {
        let args = ShaderVramFillArgs {
            position: [x, y],
            size: [width, height],
            color: u32::from(color.r >> 3)
                | (u32::from(color.g >> 3) << 5)
                | (u32::from(color.b >> 3) << 10),
        };

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::cast_slice(&[args]));

        let x_workgroups =
            width / Self::WORKGROUP_SIZE + u32::from(!width.is_multiple_of(Self::WORKGROUP_SIZE));
        let y_workgroups =
            height / Self::WORKGROUP_SIZE + u32::from(!height.is_multiple_of(Self::WORKGROUP_SIZE));
        compute_pass.dispatch_workgroups(x_workgroups, y_workgroups, 1);
    }
}
