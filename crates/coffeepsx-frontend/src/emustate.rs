use crate::config::{AppConfig, InputConfig, Rasterizer, VSyncMode, VideoConfig};
use crate::emuthread::{EmulationThreadHandle, EmulatorThreadCommand, SurfaceRenderEffect};
use crate::input::InputMapper;
use crate::{OpenFileType, UserEvent};
use anyhow::anyhow;
use sdl2::controller::GameController;
use sdl2::event::Event as SdlEvent;
use sdl2::{EventPump, GameControllerSubsystem, Sdl};
use std::cmp;
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoopProxy};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowAttributes};

#[derive(Debug)]
struct EmulatorWindow {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    vsync_mode: VSyncMode,
    fast_forwarding: bool,
    supported_present_modes: Vec<wgpu::PresentMode>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    // SAFETY: The window must be dropped after the surface
    window: Window,
    initial_render_complete: bool,
    window_focused: bool,
}

impl EmulatorWindow {
    fn new(
        file_path: Option<&Path>,
        event_loop: &ActiveEventLoop,
        config: &AppConfig,
    ) -> anyhow::Result<Self> {
        let window_title = match file_path {
            Some(file_path) => determine_window_title(file_path),
            None => "(BIOS)".into(),
        };
        let window_size = LogicalSize::new(config.video.window_width, config.video.window_height);

        let mut window_attrs = WindowAttributes::default()
            .with_title(window_title)
            .with_inner_size(window_size)
            .with_visible(false);
        if config.video.launch_in_fullscreen {
            window_attrs = window_attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }

        #[allow(deprecated)]
        let window = event_loop.create_window(window_attrs)?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: config.graphics.wgpu_backend.to_wgpu(),
            backend_options: wgpu::BackendOptions {
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler: wgpu::Dx12Compiler::DynamicDxc {
                        dxc_path: "dxcompiler.dll".into(),
                    },
                    ..wgpu::Dx12BackendOptions::default()
                },
                ..wgpu::BackendOptions::default()
            },
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

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
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: "emulator_device".into(),
                required_features: ps1_core::required_wgpu_features(),
                required_limits: ps1_core::required_wgpu_limits(),
                memory_hints: wgpu::MemoryHints::default(),
                ..wgpu::DeviceDescriptor::default()
            }))?;

        let surface_capabilities = surface.get_capabilities(&adapter);

        let vsync_mode = config.video.vsync_mode;
        let present_mode = vsync_mode.to_present_mode();
        if !surface_capabilities.present_modes.contains(&present_mode) {
            return Err(anyhow!(
                "wgpu surface does not support requested VSync mode {:?}",
                config.video.vsync_mode
            ));
        }

        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or_else(|| {
                let format = surface_capabilities
                    .formats
                    .first()
                    .copied()
                    .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
                log::error!("Surface does not support any sRGB texture formats; using {format:?}");
                format
            });

        log::info!("Using texture format {surface_format:?} in emulator window");

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: vsync_mode.to_present_mode(),
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::default(),
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        Ok(Self {
            surface,
            surface_config,
            vsync_mode,
            fast_forwarding: false,
            supported_present_modes: surface_capabilities.present_modes,
            device: Arc::new(device),
            queue: Arc::new(queue),
            window,
            initial_render_complete: false,
            window_focused: false,
        })
    }

    pub fn update_config(&mut self, video_config: &VideoConfig) {
        let vsync_mode = video_config.vsync_mode;
        let present_mode = vsync_mode.to_present_mode();
        if !self.supported_present_modes.contains(&present_mode) {
            log::error!(
                "wgpu present mode {present_mode:?} is not supported; not changing VSync mode"
            );
            return;
        }

        self.vsync_mode = vsync_mode;
        self.reconfigure_surface();
    }

    pub fn set_fast_forwarding(&mut self, fast_forwarding: bool) {
        self.fast_forwarding = fast_forwarding;
        self.reconfigure_surface();
    }

    fn reconfigure_surface(&mut self) {
        self.surface_config.present_mode = if self.fast_forwarding {
            wgpu::PresentMode::AutoNoVsync
        } else {
            self.vsync_mode.to_present_mode()
        };
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn toggle_fullscreen(&self) {
        let new_fullscreen = match self.window.fullscreen() {
            Some(_) => None,
            None => Some(Fullscreen::Borderless(None)),
        };
        self.window.set_fullscreen(new_fullscreen);
    }
}

fn determine_window_title(path: &Path) -> String {
    path.with_extension("")
        .file_name()
        .and_then(OsStr::to_str)
        .map_or_else(|| "PS1".into(), String::from)
}

struct RunningState {
    window: EmulatorWindow,
    emu_thread: EmulationThreadHandle,
}

struct Controllers {
    subsystem: GameControllerSubsystem,
    controllers: HashMap<u32, GameController>,
    instance_id_to_device_id: HashMap<u32, u32>,
}

impl Controllers {
    fn new(subsystem: GameControllerSubsystem) -> Self {
        Self { subsystem, controllers: HashMap::new(), instance_id_to_device_id: HashMap::new() }
    }

    fn handle_device_added(&mut self, which: u32) -> anyhow::Result<()> {
        let controller = self.subsystem.open(which)?;

        log::info!("Controller added (idx {which}): '{}'", controller.name());

        self.instance_id_to_device_id.insert(controller.instance_id(), which);
        self.controllers.insert(which, controller);

        Ok(())
    }

    fn handle_device_removed(&mut self, which: u32) {
        let Some(device_id) = self.instance_id_to_device_id.remove(&which) else { return };
        let Some(controller) = self.controllers.remove(&device_id) else { return };

        log::info!("Controller removed: '{}'", controller.name());
    }

    fn get_device_id(&self, instance_id: u32) -> Option<u32> {
        self.instance_id_to_device_id.get(&instance_id).copied()
    }
}

pub struct EmulatorState {
    running: Option<RunningState>,
    sdl_ctx: Sdl,
    sdl_event_pump: EventPump,
    controllers: Controllers,
    input_mapper: InputMapper,
}

impl EmulatorState {
    #[allow(clippy::new_without_default)]
    #[allow(clippy::missing_errors_doc)]
    pub fn new(input_config: &InputConfig) -> anyhow::Result<Self> {
        let sdl_ctx = sdl2::init().map_err(|err| anyhow!("Error initializing SDL2: {err}"))?;
        let sdl_event_pump = sdl_ctx
            .event_pump()
            .map_err(|err| anyhow!("Error initializing SDL2 event pump: {err}"))?;
        let controller_subsystem = sdl_ctx
            .game_controller()
            .map_err(|err| anyhow!("Error initializing SDL2 game controller subsystem: {err}"))?;
        let controllers = Controllers::new(controller_subsystem);
        let input_mapper = InputMapper::new(input_config);

        Ok(Self { running: None, sdl_ctx, sdl_event_pump, controllers, input_mapper })
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn handle_event(
        &mut self,
        event: &Event<UserEvent>,
        elwt: &ActiveEventLoop,
        proxy: &EventLoopProxy<UserEvent>,
        app_config: &mut AppConfig,
    ) -> anyhow::Result<()> {
        match event {
            Event::UserEvent(UserEvent::FileOpened(OpenFileType::Open, Some(file_path))) => {
                return self.start_emulator(Some(file_path), elwt, app_config);
            }
            Event::UserEvent(UserEvent::RunBios) => {
                return self.start_emulator(None, elwt, app_config);
            }
            Event::UserEvent(UserEvent::AppConfigChanged) => {
                self.input_mapper = InputMapper::new(&app_config.input);
            }
            Event::AboutToWait => {
                self.process_sdl_events(proxy)?;
            }
            _ => {}
        }

        let Some(RunningState { window, emu_thread }) = &mut self.running else { return Ok(()) };

        match event {
            Event::UserEvent(UserEvent::AppConfigChanged) => {
                window.update_config(&app_config.video);
                emu_thread.handle_config_change(app_config)?;
            }
            &Event::UserEvent(UserEvent::ControllerButton { player, button, pressed }) => {
                log::debug!("Player {player:?} digital input: {button:?} pressed={pressed}");
                emu_thread.send_command(EmulatorThreadCommand::DigitalInput {
                    player,
                    button,
                    pressed,
                });
            }
            &Event::UserEvent(UserEvent::ControllerAnalog { player, input, value }) => {
                log::debug!("Player {player:?} analog input: {input:?} value={value}");
                emu_thread.send_command(EmulatorThreadCommand::AnalogInput {
                    player,
                    input,
                    value,
                });
            }
            Event::UserEvent(UserEvent::FileOpened(OpenFileType::DiscChange, Some(disc_path))) => {
                log::info!("Changing disc to '{}'", disc_path.display());
                emu_thread.send_command(EmulatorThreadCommand::ChangeDisc {
                    disc_path: disc_path.clone(),
                });
                window.window.set_title(&determine_window_title(disc_path));
            }
            Event::UserEvent(UserEvent::RemoveDisc) => {
                log::info!("Removing disc");
                emu_thread.send_command(EmulatorThreadCommand::RemoveDisc);
            }
            Event::UserEvent(UserEvent::Reset) => {
                emu_thread.send_command(EmulatorThreadCommand::Reset);
            }
            Event::UserEvent(UserEvent::PowerOff) => {
                emu_thread.send_command(EmulatorThreadCommand::Stop);
                self.running = None;
            }
            Event::WindowEvent { event: win_event, window_id }
                if *window_id == window.window.id() =>
            {
                match win_event {
                    WindowEvent::CloseRequested => {
                        emu_thread.send_command(EmulatorThreadCommand::Stop);
                        self.running = None;
                    }
                    WindowEvent::Resized(size) => {
                        window.surface_config.width = size.width;
                        window.surface_config.height = size.height;
                        window.surface.configure(&window.device, &window.surface_config);

                        match window.window.fullscreen() {
                            Some(_) => {
                                window.window.set_cursor_visible(false);
                            }
                            None => {
                                let logical_size = size.to_logical(window.window.scale_factor());
                                app_config.video.window_width = logical_size.width;
                                app_config.video.window_height = logical_size.height;

                                window.window.set_cursor_visible(true);
                            }
                        }

                        emu_thread.handle_resize(*size);
                    }
                    &WindowEvent::KeyboardInput {
                        event: KeyEvent { physical_key, state, .. },
                        ..
                    } => {
                        if let PhysicalKey::Code(keycode) = physical_key {
                            let pressed = state == ElementState::Pressed;
                            self.input_mapper.map_keyboard(keycode, pressed, proxy);
                        }

                        let hotkey = check_hotkey(physical_key, state);
                        match hotkey {
                            Some(Hotkey::Quit) => {
                                emu_thread.send_command(EmulatorThreadCommand::Stop);
                                self.running = None;
                            }
                            Some(Hotkey::ToggleFullscreen) => {
                                window.toggle_fullscreen();
                            }
                            Some(Hotkey::ToggleVramDisplay) => {
                                app_config.debug.vram_display = !app_config.debug.vram_display;
                                emu_thread.send_command(EmulatorThreadCommand::UpdateConfig(
                                    app_config.clone().into(),
                                ));
                            }
                            Some(Hotkey::EnableHardwareRasterizer) => {
                                app_config.graphics.rasterizer = Rasterizer::Hardware;
                                emu_thread.send_command(EmulatorThreadCommand::UpdateConfig(
                                    app_config.clone().into(),
                                ));
                                log::info!(
                                    "Using hardware rasterizer with resolution scale {}",
                                    app_config.graphics.hardware_resolution_scale
                                );
                            }
                            Some(Hotkey::EnableSoftwareRasterizer) => {
                                app_config.graphics.rasterizer = Rasterizer::Software;
                                emu_thread.send_command(EmulatorThreadCommand::UpdateConfig(
                                    app_config.clone().into(),
                                ));
                                log::info!("Using software rasterizer");
                            }
                            Some(Hotkey::DecreaseResolutionScale) => {
                                let scale =
                                    cmp::max(1, app_config.graphics.hardware_resolution_scale - 1);
                                app_config.graphics.hardware_resolution_scale = scale;
                                emu_thread.send_command(EmulatorThreadCommand::UpdateConfig(
                                    app_config.clone().into(),
                                ));
                                log::info!("Set resolution scale to {scale}");
                            }
                            Some(Hotkey::IncreaseResolutionScale) => {
                                let scale =
                                    cmp::min(16, app_config.graphics.hardware_resolution_scale + 1);
                                app_config.graphics.hardware_resolution_scale = scale;
                                emu_thread.send_command(EmulatorThreadCommand::UpdateConfig(
                                    app_config.clone().into(),
                                ));
                                log::info!("Set resolution scale to {scale}");
                            }
                            Some(Hotkey::SaveState) => {
                                emu_thread.send_command(EmulatorThreadCommand::SaveState);
                            }
                            Some(Hotkey::LoadState) => {
                                emu_thread.send_command(EmulatorThreadCommand::LoadState);
                            }
                            Some(Hotkey::Pause) => {
                                emu_thread.send_command(EmulatorThreadCommand::TogglePause);
                            }
                            Some(Hotkey::StepFrame) => {
                                emu_thread.send_command(EmulatorThreadCommand::StepFrame);
                            }
                            Some(Hotkey::FastForward) => {
                                let enabled = state == ElementState::Pressed;
                                emu_thread
                                    .send_command(EmulatorThreadCommand::FastForward { enabled });
                                window.set_fast_forwarding(enabled);
                            }
                            None => {}
                        }
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => match emu_thread.render_frame_if_available(&window.surface)? {
                SurfaceRenderEffect::FrameRendered { suboptimal } => {
                    if suboptimal {
                        window.surface.configure(&window.device, &window.surface_config);
                    }

                    if !window.initial_render_complete {
                        window.initial_render_complete = true;
                        window.window.set_visible(true);
                    }

                    if !window.window_focused {
                        window.window.focus_window();
                        window.window_focused = window.window.has_focus();
                    }
                }
                SurfaceRenderEffect::SurfaceOutdated => {
                    window.surface.configure(&window.device, &window.surface_config);
                }
                SurfaceRenderEffect::None => {}
            },
            _ => {}
        }

        Ok(())
    }

    fn process_sdl_events(&mut self, proxy: &EventLoopProxy<UserEvent>) -> anyhow::Result<()> {
        for event in self.sdl_event_pump.poll_iter() {
            match event {
                SdlEvent::ControllerDeviceAdded { which, .. } => {
                    self.controllers.handle_device_added(which)?;
                }
                SdlEvent::ControllerDeviceRemoved { which, .. } => {
                    self.controllers.handle_device_removed(which);
                }
                SdlEvent::ControllerButtonDown { which, button, .. } => {
                    let Some(device_id) = self.controllers.get_device_id(which) else { continue };
                    self.input_mapper.map_sdl_button(device_id, button, true, proxy);

                    proxy
                        .send_event(UserEvent::SdlButtonPress { which: device_id, button })
                        .unwrap();
                }
                SdlEvent::ControllerButtonUp { which, button, .. } => {
                    let Some(device_id) = self.controllers.get_device_id(which) else { continue };
                    self.input_mapper.map_sdl_button(device_id, button, false, proxy);
                }
                SdlEvent::ControllerAxisMotion { which, axis, value, .. } => {
                    let Some(device_id) = self.controllers.get_device_id(which) else { continue };
                    self.input_mapper.map_sdl_axis(device_id, axis, value, proxy);

                    proxy
                        .send_event(UserEvent::SdlAxisMotion { which: device_id, axis, value })
                        .unwrap();
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn start_emulator(
        &mut self,
        file_path: Option<&Path>,
        elwt: &ActiveEventLoop,
        app_config: &AppConfig,
    ) -> anyhow::Result<()> {
        if let Some(RunningState { emu_thread, .. }) = &self.running {
            emu_thread.send_command(EmulatorThreadCommand::Stop);
        }

        let window = EmulatorWindow::new(file_path, elwt, app_config)?;

        let emu_thread = EmulationThreadHandle::spawn(
            &self.sdl_ctx,
            file_path,
            app_config,
            &window.surface_config,
            Arc::clone(&window.device),
            Arc::clone(&window.queue),
        )?;

        self.running = Some(RunningState { window, emu_thread });

        Ok(())
    }

    pub fn is_emulator_running(&self) -> bool {
        self.running.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Hotkey {
    Quit,
    ToggleFullscreen,
    ToggleVramDisplay,
    EnableHardwareRasterizer,
    EnableSoftwareRasterizer,
    DecreaseResolutionScale,
    IncreaseResolutionScale,
    SaveState,
    LoadState,
    Pause,
    StepFrame,
    FastForward,
}

fn check_hotkey(key: PhysicalKey, state: ElementState) -> Option<Hotkey> {
    let PhysicalKey::Code(keycode) = key else { return None };
    let pressed = state == ElementState::Pressed;

    // TODO configurable
    match keycode {
        KeyCode::Escape if pressed => Some(Hotkey::Quit),
        KeyCode::F9 if pressed => Some(Hotkey::ToggleFullscreen),
        KeyCode::Quote if pressed => Some(Hotkey::ToggleVramDisplay),
        KeyCode::Digit0 if pressed => Some(Hotkey::EnableHardwareRasterizer),
        KeyCode::Minus if pressed => Some(Hotkey::EnableSoftwareRasterizer),
        KeyCode::BracketLeft if pressed => Some(Hotkey::DecreaseResolutionScale),
        KeyCode::BracketRight if pressed => Some(Hotkey::IncreaseResolutionScale),
        KeyCode::F5 if pressed => Some(Hotkey::SaveState),
        KeyCode::F6 if pressed => Some(Hotkey::LoadState),
        KeyCode::KeyP if pressed => Some(Hotkey::Pause),
        KeyCode::KeyN if pressed => Some(Hotkey::StepFrame),
        KeyCode::Tab => Some(Hotkey::FastForward),
        _ => None,
    }
}
