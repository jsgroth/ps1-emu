use crate::app::App;
use crate::config::AppConfig;
use crate::emustate::EmulatorState;
use crate::{OpenFileType, UserEvent};
use egui::ViewportId;
use egui_wgpu::{RendererOptions, ScreenDescriptor};
use rfd::FileDialog;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::{iter, thread};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowAttributes};

pub struct GuiState {
    app: App,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    egui_renderer: egui_wgpu::Renderer,
    egui_state: egui_winit::State,
    egui_event_repaint: bool,
    egui_callback_repaint_count: Arc<AtomicU32>,
    egui_callback_next_repaint: Arc<Mutex<Instant>>,
    file_dialog_open: bool,
    // SAFETY: The window must be dropped after the surface
    window: Window,
    initial_render_complete: bool,
    window_focused: bool,
}

impl GuiState {
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn new(app: App, event_loop: &EventLoop<UserEvent>) -> anyhow::Result<Self> {
        #[allow(deprecated)]
        let window = event_loop.create_window(
            WindowAttributes::default()
                .with_title("CoffeePSX")
                .with_inner_size(LogicalSize::new(800, 600))
                .with_visible(false),
        )?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        // SAFETY: The surface must not outlive the window
        let surface = unsafe {
            instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_display_and_window(
                &window, &window,
            )?)
        }?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;

        let surface_capabilities = surface.get_capabilities(&adapter);

        let supports_mailbox =
            surface_capabilities.present_modes.contains(&wgpu::PresentMode::Mailbox);
        let present_mode = if supports_mailbox {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::AutoNoVsync
        };

        let texture_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .unwrap_or_else(|| {
                let format = surface_capabilities
                    .formats
                    .first()
                    .copied()
                    .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
                log::error!(
                    "Surface does not support any non-sRGB formats; defaulting to {format:?}"
                );
                format
            });

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: texture_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::default(),
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_config.format,
            RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                predictable_texture_filtering: false,
            },
        );

        let egui_state = egui_winit::State::new(
            egui::Context::default(),
            ViewportId::default(),
            &window,
            None,
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );

        let egui_callback_repaint_count = Arc::new(AtomicU32::new(0));
        let egui_callback_next_repaint = Arc::new(Mutex::new(Instant::now()));
        {
            let egui_callback_repaint_count = Arc::clone(&egui_callback_repaint_count);
            let egui_callback_next_repaint = Arc::clone(&egui_callback_next_repaint);

            egui_state.egui_ctx().set_request_repaint_callback(move |info| {
                if info.delay != Duration::ZERO {
                    *egui_callback_next_repaint.lock().unwrap() = Instant::now() + info.delay;
                }

                egui_callback_repaint_count.fetch_add(1, Ordering::Release);
            });
        }

        Ok(Self {
            app,
            surface,
            surface_config,
            device,
            queue,
            egui_renderer,
            egui_state,
            egui_event_repaint: true,
            egui_callback_repaint_count,
            egui_callback_next_repaint,
            file_dialog_open: false,
            window,
            initial_render_complete: false,
            window_focused: false,
        })
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn handle_event(
        &mut self,
        event: &Event<UserEvent>,
        elwt: &ActiveEventLoop,
        emu_state: &EmulatorState,
        proxy: &EventLoopProxy<UserEvent>,
    ) {
        if let Event::UserEvent(user_event) = event {
            let response = self.app.handle_event(user_event);
            if response.repaint {
                self.repaint(emu_state, proxy);
            }
        }

        match event {
            Event::WindowEvent { event: win_event, window_id }
                if *window_id == self.window.id() =>
            {
                let egui_response = self.egui_state.on_window_event(&self.window, win_event);
                self.egui_event_repaint |= egui_response.repaint;

                if egui_response.consumed {
                    return;
                }

                match win_event {
                    WindowEvent::CloseRequested => {
                        elwt.exit();
                    }
                    WindowEvent::Resized(size) => {
                        self.surface_config.width = size.width;
                        self.surface_config.height = size.height;
                        self.surface.configure(&self.device, &self.surface_config);
                    }
                    &WindowEvent::KeyboardInput {
                        event:
                            KeyEvent {
                                physical_key: PhysicalKey::Code(keycode),
                                state: ElementState::Pressed,
                                ..
                            },
                        ..
                    } => {
                        self.app.handle_key_press(keycode);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                let egui_callback_repaint_count =
                    self.egui_callback_repaint_count.load(Ordering::Relaxed);
                let egui_callback_next_repaint = *self.egui_callback_next_repaint.lock().unwrap();
                let now = Instant::now();
                if !self.egui_event_repaint
                    && (egui_callback_repaint_count == 0 || egui_callback_next_repaint > now)
                {
                    return;
                }

                if self.file_dialog_open {
                    // Don't repaint GUI while an open file dialog is open
                    return;
                }

                self.repaint(emu_state, proxy);

                self.egui_event_repaint = false;

                // Conditionally decrement callback repaint because egui may request a repaint
                // during rendering, in which case count should not be decremented until after the
                // next repaint
                if egui_callback_repaint_count != 0 && egui_callback_next_repaint <= now {
                    self.egui_callback_repaint_count.fetch_sub(1, Ordering::Relaxed);
                }

                if !self.initial_render_complete {
                    self.initial_render_complete = true;
                    self.window.set_visible(true);
                }

                if !self.window_focused {
                    self.window.focus_window();
                    self.window_focused = self.window.has_focus();
                }
            }
            Event::UserEvent(UserEvent::OpenFileDialog { file_type, initial_dir }) => {
                self.file_dialog_open = true;

                async_open_file_dialog(*file_type, initial_dir.as_ref(), proxy);
            }
            Event::UserEvent(UserEvent::FileOpened(..)) => {
                self.file_dialog_open = false;
            }
            _ => {}
        }
    }

    fn repaint(&mut self, emu_state: &EmulatorState, proxy: &EventLoopProxy<UserEvent>) {
        let viewport_id = self.egui_state.egui_input().viewport_id;
        let egui_ctx = self.egui_state.egui_ctx().clone();
        if let Some(viewport_info) =
            self.egui_state.egui_input_mut().viewports.get_mut(&viewport_id)
        {
            egui_winit::update_viewport_info(viewport_info, &egui_ctx, &self.window, false);
        }

        let egui_input = self.egui_state.take_egui_input(&self.window);

        let full_output = egui_ctx.run_ui(egui_input, |ui| {
            self.app.render(ui, emu_state, proxy);
        });

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let mut suboptimal = false;
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
                suboptimal = true;
                texture
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout => {
                log::error!("wgpu surface timeout error");
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                log::error!("wgpu surface lost error");
                return;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("wgpu surface validation error");
                return;
            }
        };

        let output_view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: egui_ctx.pixels_per_point(),
        };

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: "egui_rpass".into(),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..wgpu::RenderPassDescriptor::default()
            });

            // egui-wgpu requires a RenderPass with static lifetime
            let mut render_pass = render_pass.forget_lifetime();

            self.egui_renderer.render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        self.queue.submit(iter::once(encoder.finish()));
        output.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        if suboptimal {
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    pub fn app_config_mut(&mut self) -> &mut AppConfig {
        self.app.config_mut()
    }
}

fn async_open_file_dialog(
    file_type: OpenFileType,
    initial_dir: Option<&PathBuf>,
    proxy: &EventLoopProxy<UserEvent>,
) {
    let (name, extensions): (_, &[_]) = match file_type {
        OpenFileType::Open => ("PS1", &["cue", "chd", "exe"]),
        OpenFileType::DiscChange => ("PS1", &["cue", "chd"]),
        OpenFileType::BiosPath => ("BIOS", &["bin", "BIN"]),
        OpenFileType::SearchDir => {
            let proxy = proxy.clone();
            thread::spawn(move || {
                let search_dir = FileDialog::new().pick_folder();
                proxy
                    .send_event(UserEvent::FileOpened(OpenFileType::SearchDir, search_dir))
                    .unwrap();
            });
            return;
        }
    };

    let mut file_dialog =
        FileDialog::new().add_filter(name, extensions).add_filter("All Files", &["*"]);
    if let Some(initial_dir) = initial_dir {
        file_dialog = file_dialog.set_directory(initial_dir);
    }

    let proxy = proxy.clone();
    thread::spawn(move || {
        let path = file_dialog.pick_file();
        proxy.send_event(UserEvent::FileOpened(file_type, path)).unwrap();
    });
}
