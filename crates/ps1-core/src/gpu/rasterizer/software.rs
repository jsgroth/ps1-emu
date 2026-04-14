use crate::api::ColorDepthBits;
use crate::gpu::rasterizer::{
    ClearPipeline, CpuVramBlitArgs, FrameCoords, FrameSize, ScreenSize, VramVramBlitArgs,
};
use crate::gpu::registers::Registers;
use crate::gpu::{Color, VramArray, WgpuResources, rasterizer};
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;
use wgpu::CommandBuffer;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RgbaColor {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl RgbaColor {
    const BLACK: Self = Self::rgb(0, 0, 0);

    const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }
}

const FRAME_BUFFER_LEN: usize =
    (1024 * 2 * (ScreenSize::PAL.bottom - ScreenSize::PAL.top)) as usize;

type FrameBuffer = [RgbaColor; FRAME_BUFFER_LEN];

#[derive(Debug)]
pub struct SoftwareRenderer {
    frame_buffer: Box<FrameBuffer>,
    frame_textures: HashMap<FrameSize, wgpu::Texture>,
    clear_pipeline: ClearPipeline,
}

impl SoftwareRenderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let clear_pipeline = ClearPipeline::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

        Self {
            frame_buffer: vec![RgbaColor::BLACK; FRAME_BUFFER_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            frame_textures: HashMap::new(),
            clear_pipeline,
        }
    }

    pub fn generate_frame_texture(
        &mut self,
        registers: &Registers,
        wgpu_resources: &mut WgpuResources,
        vram: &VramArray,
    ) -> &wgpu::Texture {
        if wgpu_resources.display_config.dump_vram {
            return self.write_frame(
                &wgpu_resources.device,
                &wgpu_resources.queue,
                FrameSize { width: 1024, height: 512 },
                FrameCoords {
                    frame_x: 0,
                    frame_y: 0,
                    display_x_offset: 0,
                    display_y_offset: 0,
                    display_x_start: 0,
                    display_y_start: 0,
                    display_width: 1024,
                    display_height: 512,
                },
                ColorDepthBits::Fifteen,
                vram,
            );
        }

        let (frame_coords, frame_size) =
            rasterizer::compute_frame_location(registers, wgpu_resources.display_config);
        let Some(frame_coords) = frame_coords else {
            return self.clear_frame(
                &wgpu_resources.device,
                &mut wgpu_resources.queued_command_buffers,
                frame_size,
            );
        };

        log::debug!(
            "Computed frame coords {frame_coords:?} and frame_size {frame_size:?} from video_mode={}, X1={}, X2={}, Y1={}, Y2={}, dot_clock_divider={}, v_resolution={:?}",
            registers.video_mode,
            registers.x_display_range.0,
            registers.x_display_range.1,
            registers.y_display_range.0,
            registers.y_display_range.1,
            registers.dot_clock_divider(),
            registers.v_resolution
        );

        if !registers.display_enabled {
            return self.clear_frame(
                &wgpu_resources.device,
                &mut wgpu_resources.queued_command_buffers,
                frame_size,
            );
        }

        self.write_frame(
            &wgpu_resources.device,
            &wgpu_resources.queue,
            frame_size,
            frame_coords,
            registers.display_area_color_depth,
            vram,
        )
    }

    fn clear_frame(
        &mut self,
        device: &wgpu::Device,
        command_buffers: &mut Vec<CommandBuffer>,
        frame_size: FrameSize,
    ) -> &wgpu::Texture {
        let texture = get_or_create_frame_texture(device, frame_size, &mut self.frame_textures);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: "clear_encoder".into(),
        });

        self.clear_pipeline.draw(texture, &mut encoder);

        command_buffers.push(encoder.finish());

        texture
    }

    fn write_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame_size: FrameSize,
        frame_coords: FrameCoords,
        color_depth: ColorDepthBits,
        vram: &VramArray,
    ) -> &wgpu::Texture {
        populate_frame_buffer(frame_size, frame_coords, color_depth, vram, &mut self.frame_buffer);

        let frame_texture =
            get_or_create_frame_texture(device, frame_size, &mut self.frame_textures);

        queue.write_texture(
            frame_texture.as_image_copy(),
            bytemuck::cast_slice(self.frame_buffer.as_ref()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(1024 * 4),
                rows_per_image: None,
            },
            frame_texture.size(),
        );

        frame_texture
    }
}

fn get_or_create_frame_texture<'a>(
    device: &wgpu::Device,
    frame_size: FrameSize,
    map: &'a mut HashMap<FrameSize, wgpu::Texture>,
) -> &'a wgpu::Texture {
    map.entry(frame_size).or_insert_with(|| {
        log::info!("Creating PS1 GPU frame texture of size {frame_size}");

        device.create_texture(&wgpu::TextureDescriptor {
            label: "frame_texture".into(),
            size: wgpu::Extent3d {
                width: frame_size.width,
                height: frame_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
    })
}

const RGB_5_TO_8: &[u8; 32] = &[
    0, 8, 16, 25, 33, 41, 49, 58, 66, 74, 82, 90, 99, 107, 115, 123, 132, 140, 148, 156, 165, 173,
    181, 189, 197, 206, 214, 222, 230, 239, 247, 255,
];

fn populate_frame_buffer(
    frame_size: FrameSize,
    frame_coords: FrameCoords,
    color_depth: ColorDepthBits,
    vram: &VramArray,
    frame_buffer: &mut FrameBuffer,
) {
    let x_range =
        frame_coords.display_x_start..frame_coords.display_x_start + frame_coords.display_width;
    let y_range =
        frame_coords.display_y_start..frame_coords.display_y_start + frame_coords.display_height;

    for y in 0..frame_size.height {
        if !y_range.contains(&y) {
            frame_buffer[1024 * y as usize..1024 * (y + 1) as usize].fill(RgbaColor::BLACK);
            continue;
        }

        let vram_y = ((frame_coords.frame_y + y + frame_coords.display_y_offset)
            .wrapping_sub(frame_coords.display_y_start))
            & 0x1FF;
        let vram_row_addr = (1024 * vram_y) as usize;

        // Fill pixels outside of the horizontal display range with solid black
        let fb_row_addr = 1024 * y as usize;
        frame_buffer[fb_row_addr..fb_row_addr + x_range.start as usize].fill(RgbaColor::BLACK);
        frame_buffer[fb_row_addr + x_range.end as usize..fb_row_addr + frame_size.width as usize]
            .fill(RgbaColor::BLACK);

        for x in x_range.clone() {
            let frame_buffer_addr = fb_row_addr + x as usize;

            match color_depth {
                ColorDepthBits::Fifteen => {
                    let vram_x = ((frame_coords.frame_x + x + frame_coords.display_x_offset)
                        .wrapping_sub(frame_coords.display_x_start))
                        & 0x3FF;
                    let vram_addr = vram_row_addr | (vram_x as usize);
                    let vram_color = vram[vram_addr];

                    let r = RGB_5_TO_8[(vram_color & 0x1F) as usize];
                    let g = RGB_5_TO_8[((vram_color >> 5) & 0x1F) as usize];
                    let b = RGB_5_TO_8[((vram_color >> 10) & 0x1F) as usize];
                    frame_buffer[frame_buffer_addr] = RgbaColor::rgb(r, g, b);
                }
                ColorDepthBits::TwentyFour => {
                    let effective_x = (x + frame_coords.display_x_offset)
                        .wrapping_sub(frame_coords.display_x_start)
                        & 0x3FF;
                    let vram_x = frame_coords.frame_x + 3 * effective_x / 2;
                    let first_halfword = vram[vram_row_addr | (vram_x & 0x3FF) as usize];
                    let second_halfword = vram[vram_row_addr | ((vram_x + 1) & 0x3FF) as usize];

                    let color = if effective_x.is_multiple_of(2) {
                        RgbaColor::rgb(
                            first_halfword as u8,
                            (first_halfword >> 8) as u8,
                            second_halfword as u8,
                        )
                    } else {
                        RgbaColor::rgb(
                            (first_halfword >> 8) as u8,
                            second_halfword as u8,
                            (second_halfword >> 8) as u8,
                        )
                    };
                    frame_buffer[frame_buffer_addr] = color;
                }
            }
        }
    }
}

pub fn vram_fill(vram: &mut VramArray, x: u32, y: u32, width: u32, height: u32, color: Color) {
    let fill_x = x & 0x3F0;
    let fill_y = y & 0x1FF;
    let width = ((width & 0x3FF) + 0xF) & !0xF;
    let height = height & 0x1FF;

    let color = color.truncate_to_15_bit();

    for y_offset in 0..height {
        for x_offset in 0..width {
            let x = (fill_x + x_offset) & 0x3FF;
            let y = (fill_y + y_offset) & 0x1FF;

            let vram_addr = (1024 * y + x) as usize;
            vram[vram_addr] = color;
        }
    }
}

pub fn cpu_to_vram_blit(
    vram: &mut VramArray,
    CpuVramBlitArgs { x, y, width, height, force_mask_bit, check_mask_bit }: CpuVramBlitArgs,
    data: &[u16],
) {
    let forced_mask_bit = u16::from(force_mask_bit) << 15;

    let mut row = 0;
    let mut col = 0;

    for &halfword in data {
        let vram_x = (x + col) & 0x3FF;
        let vram_y = (y + row) & 0x1FF;
        let vram_addr = (1024 * vram_y + vram_x) as usize;

        if !check_mask_bit || vram[vram_addr] & 0x8000 == 0 {
            vram[vram_addr] = halfword | forced_mask_bit;
        }

        col += 1;
        if col == width {
            col = 0;
            row += 1;

            if row == height {
                return;
            }
        }
    }
}

pub fn vram_to_cpu_blit(
    vram: &VramArray,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    out: &mut Vec<u16>,
) {
    for row in 0..height {
        let vram_y = (y + row) & 0x1FF;
        for col in 0..width {
            let vram_x = (x + col) & 0x3FF;
            let vram_addr = (1024 * vram_y + vram_x) as usize;
            out.push(vram[vram_addr]);
        }
    }
}

pub fn vram_to_vram_blit(vram: &mut VramArray, args: VramVramBlitArgs) {
    let forced_mask_bit = u16::from(args.force_mask_bit) << 15;

    let mut source_y = args.source_y;
    let mut dest_y = args.dest_y;

    for _ in 0..args.height {
        let mut source_x = args.source_x;
        let mut dest_x = args.dest_x;

        for _ in 0..args.width {
            let source_addr = (1024 * source_y + source_x) as usize;
            let dest_addr = (1024 * dest_y + dest_x) as usize;

            if !args.check_mask_bit || vram[dest_addr] & 0x8000 == 0 {
                vram[dest_addr] = vram[source_addr] | forced_mask_bit;
            }

            source_x = source_x.wrapping_add(1) & 0x3FF;
            dest_x = dest_x.wrapping_add(1) & 0x3FF;
        }

        source_y = source_y.wrapping_add(1) & 0x1FF;
        dest_y = dest_y.wrapping_add(1) & 0x1FF;
    }
}
