use crate::config::{AspectRatio, VideoConfig};
use crate::emuthread::{EmulatorSwapChain, QueuedFrame};
use crate::{Never, emuthread};
use ps1_core::api::Renderer;
use std::iter;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use wgpu::PipelineCompilationOptions;
use winit::dpi::PhysicalSize;

pub struct SwapChainRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    swap_chain: EmulatorSwapChain,
    in_progress_renders: Arc<AtomicU32>,
}

impl SwapChainRenderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        swap_chain: EmulatorSwapChain,
    ) -> Self {
        Self { device, queue, swap_chain, in_progress_renders: Arc::new(AtomicU32::new(0)) }
    }

    pub fn clear_swap_chain(&self) {
        self.swap_chain.rendered_frames.lock().unwrap().clear();
    }
}

impl Renderer for SwapChainRenderer {
    type Err = Never;

    fn render_frame(
        &mut self,
        command_buffers: impl Iterator<Item = wgpu::CommandBuffer>,
        frame: &wgpu::Texture,
        pixel_aspect_ratio: f64,
    ) -> Result<(), Self::Err> {
        // Load/compare followed by increment is fine because this value is only incremented from one
        // thread; other thread modifications are decrements
        if self.in_progress_renders.load(Ordering::Relaxed) >= emuthread::SWAP_CHAIN_LEN as u32 {
            log::warn!(
                "Skipping frame because {} renders are already in progress",
                emuthread::SWAP_CHAIN_LEN
            );

            // Wait until at least one of the in-progress frames is complete
            while self.in_progress_renders.load(Ordering::Relaxed)
                >= emuthread::SWAP_CHAIN_LEN as u32
            {
                self.device.poll(wgpu::Maintain::Poll);
                emuthread::sleep(Duration::from_micros(250));
            }

            self.queue.submit(command_buffers);
            return Ok(());
        }

        self.in_progress_renders.fetch_add(1, Ordering::Relaxed);

        let submission = self.queue.submit(command_buffers);

        let queued_frame = QueuedFrame {
            view: frame.create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                ..wgpu::TextureViewDescriptor::default()
            }),
            size: frame.size(),
            pixel_aspect_ratio,
        };

        if self.swap_chain.async_rendering.load(Ordering::Relaxed) {
            let in_progress_renders = Arc::clone(&self.in_progress_renders);
            self.swap_chain.rendered_frames.lock().unwrap().push_back(queued_frame);
            self.queue.on_submitted_work_done(move || {
                in_progress_renders.fetch_sub(1, Ordering::Relaxed);
            });
        } else {
            self.swap_chain.rendered_frames.lock().unwrap().push_back(queued_frame);

            self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
            self.in_progress_renders.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }
}

pub struct SurfaceRenderer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    swap_chain: EmulatorSwapChain,
    surface_size: wgpu::Extent3d,
    sampler_bind_group_layout: wgpu::BindGroupLayout,
    sampler_bind_group: wgpu::BindGroup,
    frame_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    aspect_ratio: AspectRatio,
}

impl SurfaceRenderer {
    pub fn new(
        config: &VideoConfig,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        swap_chain: EmulatorSwapChain,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let sampler_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: "sampler_bind_group_layout".into(),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                }],
            });

        let filter_mode = config.filter_mode.to_wgpu();
        let sampler_bind_group =
            create_sampler_bind_group(&device, &sampler_bind_group_layout, filter_mode);

        let frame_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: "frame_bind_group_layout".into(),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: "render_pipeline_layout".into(),
            bind_group_layouts: &[&sampler_bind_group_layout, &frame_bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("render.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: "render_pipeline".into(),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
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
                module: &shader,
                entry_point: "fs_main",
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            swap_chain,
            surface_size: wgpu::Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            sampler_bind_group_layout,
            sampler_bind_group,
            frame_bind_group_layout,
            pipeline,
            aspect_ratio: config.aspect_ratio,
        }
    }

    pub fn update_config(&mut self, config: &VideoConfig) {
        self.sampler_bind_group = create_sampler_bind_group(
            &self.device,
            &self.sampler_bind_group_layout,
            config.filter_mode.to_wgpu(),
        );
        self.aspect_ratio = config.aspect_ratio;
    }

    pub fn handle_resize(&mut self, size: PhysicalSize<u32>) {
        self.surface_size.width = size.width;
        self.surface_size.height = size.height;
    }

    pub fn render_frame_if_available(&mut self, surface: &wgpu::Surface<'_>) -> anyhow::Result<()> {
        let Some(frame) = self.swap_chain.rendered_frames.lock().unwrap().pop_front() else {
            return Ok(());
        };

        let output = surface.get_current_texture()?;
        let output_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let frame_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: "frame_bind_group".into(),
            layout: &self.frame_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&frame.view),
            }],
        });

        let viewport = determine_viewport(frame.size, self.surface_size, frame.pixel_aspect_ratio);
        log::trace!("Rendering to viewport {viewport:?}");

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: "surface_render_pass".into(),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..wgpu::RenderPassDescriptor::default()
            });

            if self.aspect_ratio == AspectRatio::Native {
                render_pass.set_viewport(
                    viewport.x,
                    viewport.y,
                    viewport.width,
                    viewport.height,
                    0.0,
                    1.0,
                );
            }

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.sampler_bind_group, &[]);
            render_pass.set_bind_group(1, &frame_bind_group, &[]);

            render_pass.draw(0..4, 0..1);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_sampler_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    filter_mode: wgpu::FilterMode,
) -> wgpu::BindGroup {
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: "frame_sampler".into(),
        mag_filter: filter_mode,
        min_filter: filter_mode,
        mipmap_filter: filter_mode,
        ..wgpu::SamplerDescriptor::default()
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: "sampler_bind_group".into(),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(&sampler),
        }],
    })
}

#[derive(Debug)]
struct Viewport {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

fn determine_viewport(
    frame_size: wgpu::Extent3d,
    surface_size: wgpu::Extent3d,
    pixel_aspect_ratio: f64,
) -> Viewport {
    let aspect_correct_width =
        (f64::from(surface_size.height) * pixel_aspect_ratio * f64::from(frame_size.width)
            / f64::from(frame_size.height)) as f32;
    if aspect_correct_width <= surface_size.width as f32 {
        return Viewport {
            x: (surface_size.width as f32 - aspect_correct_width) / 2.0,
            y: 0.0,
            width: aspect_correct_width,
            height: surface_size.height as f32,
        };
    }

    let aspect_correct_height = (f64::from(surface_size.width) / pixel_aspect_ratio
        * f64::from(frame_size.height)
        / f64::from(frame_size.width)) as f32;
    Viewport {
        x: 0.0,
        y: (surface_size.height as f32 - aspect_correct_height) / 2.0,
        width: surface_size.width as f32,
        height: aspect_correct_height,
    }
}
