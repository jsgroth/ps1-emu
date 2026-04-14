mod input;

use crate::app::input::{ConfigurableInput, ControllerNumber, InputSet};
use crate::config::input::SingleInput;
use crate::config::{
    AppConfig, AspectRatio, FilterMode, FiltersConfig, MemoryCardMode, Rasterizer, VSyncMode,
    WgpuBackend,
};
use crate::emustate::EmulatorState;
use crate::{OpenFileType, UserEvent, config};
use egui::{
    Align, Button, CentralPanel, Color32, ComboBox, Context, Grid, Key, KeyboardShortcut, Layout,
    Modifiers, Panel, Response, TextEdit, Ui, UiKind, Vec2, Widget, Window,
};
use egui_extras::{Column, TableBuilder};
use ps1_core::api::AdpcmInterpolation;
use ps1_core::input::ControllerType;
use regex::Regex;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::str::FromStr;
use std::sync::LazyLock;
use winit::event_loop::EventLoopProxy;
use winit::keyboard::KeyCode;

struct NumericText {
    value: String,
    invalid: bool,
}

impl NumericText {
    fn new(value: impl ToString) -> Self {
        Self { value: value.to_string(), invalid: false }
    }

    fn add_ui<T: Copy + FromStr>(
        &mut self,
        ui: &mut Ui,
        config_value: &mut T,
        validator: impl FnOnce(T) -> bool,
    ) {
        let text_edit = TextEdit::singleline(&mut self.value).desired_width(60.0);
        if ui.add(text_edit).changed() {
            match self.value.parse::<T>() {
                Ok(value) if validator(value) => {
                    *config_value = value;
                    self.invalid = false;
                }
                _ => {
                    self.invalid = true;
                }
            }
        }
    }
}

struct AppState {
    video_window_open: bool,
    graphics_window_open: bool,
    audio_window_open: bool,
    input_window_open: bool,
    paths_window_open: bool,
    memcards_window_open: bool,
    debug_window_open: bool,
    audio_sync_threshold: NumericText,
    audio_device_queue_size: NumericText,
    internal_audio_buffer_size: NumericText,
    selected_controller: ControllerNumber,
    selected_input_set: InputSet,
    waiting_for_input: Option<(ControllerNumber, InputSet, ConfigurableInput)>,
    unfiltered_file_list: Rc<[FileMetadata]>,
    filtered_file_list: Rc<[FileMetadata]>,
    change_disc_list: Rc<[ChangeDiscEntry]>,
    last_opened_disc_path: Option<PathBuf>,
    last_serialized_config: AppConfig,
    filter_by_title: String,
    filter_by_title_lower: String,
    last_filter_by_title: String,
}

impl AppState {
    fn new(config: &AppConfig) -> Self {
        let unfiltered_file_list =
            do_file_search(&config.paths.search, config.paths.search_recursively);
        let filtered_file_list = filter_file_list(&unfiltered_file_list, "", &config.filters);

        Self {
            video_window_open: false,
            graphics_window_open: false,
            audio_window_open: false,
            input_window_open: false,
            paths_window_open: false,
            memcards_window_open: false,
            debug_window_open: false,
            audio_sync_threshold: NumericText::new(config.audio.sync_threshold),
            audio_device_queue_size: NumericText::new(config.audio.device_queue_size),
            internal_audio_buffer_size: NumericText::new(config.audio.internal_buffer_size),
            selected_controller: ControllerNumber::One,
            selected_input_set: InputSet::One,
            waiting_for_input: None,
            unfiltered_file_list: unfiltered_file_list.into(),
            filtered_file_list: filtered_file_list.into(),
            change_disc_list: Rc::default(),
            last_opened_disc_path: None,
            last_serialized_config: config.clone(),
            filter_by_title: String::new(),
            filter_by_title_lower: String::new(),
            last_filter_by_title: String::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AppEventResponse {
    pub repaint: bool,
}

pub struct App {
    config_path: PathBuf,
    config: AppConfig,
    state: AppState,
}

impl App {
    #[must_use]
    pub fn new(config_path: PathBuf) -> Self {
        let config = read_config(&config_path).unwrap_or_else(|err| {
            log::warn!(
                "Unable to read config from '{}', using default: {err}",
                config_path.display()
            );
            AppConfig::default()
        });

        let state = AppState::new(&config);

        Self { config_path, config, state }
    }

    #[must_use]
    pub fn handle_event(&mut self, event: &UserEvent) -> AppEventResponse {
        match event {
            UserEvent::FileOpened(OpenFileType::BiosPath, Some(path)) => {
                self.config.paths.bios = Some(path.clone());
            }
            UserEvent::FileOpened(OpenFileType::Open | OpenFileType::DiscChange, Some(path)) => {
                self.state.last_opened_disc_path = Some(path.clone());
                self.refresh_change_disc_list(path);
            }
            UserEvent::FileOpened(OpenFileType::SearchDir, Some(path)) => {
                self.config.paths.search.push(path.clone());
            }
            &UserEvent::SdlButtonPress { which, button } => {
                return self.handle_sdl_button_press(which, button);
            }
            &UserEvent::SdlAxisMotion { which, axis, value } => {
                return self.handle_sdl_axis_motion(which, axis, value);
            }
            _ => {}
        }

        AppEventResponse { repaint: false }
    }

    pub fn handle_key_press(&mut self, keycode: KeyCode) {
        let Some((controller_number, input_set, configurable_input)) =
            self.state.waiting_for_input.take()
        else {
            return;
        };

        self.update_input(
            controller_number,
            input_set,
            configurable_input,
            Some(SingleInput::Keyboard { keycode }),
        );
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn render(
        &mut self,
        ui: &mut Ui,
        emu_state: &EmulatorState,
        proxy: &EventLoopProxy<UserEvent>,
    ) {
        self.render_menu(ui, emu_state, proxy);
        self.render_central_panel(ui, proxy);

        if self.state.video_window_open {
            self.render_video_window(ui);
        }

        if self.state.graphics_window_open {
            self.render_graphics_window(ui);
        }

        if self.state.audio_window_open {
            self.render_audio_window(ui);
        }

        if self.state.input_window_open {
            self.render_input_window(ui);
        }

        if self.state.paths_window_open {
            self.render_paths_window(ui, proxy);
        }

        if self.state.memcards_window_open {
            self.render_memcards_window(ui);
        }

        if self.state.debug_window_open {
            self.render_debug_window(ui);
        }

        if self.config != self.state.last_serialized_config {
            if let Err(err) = self.serialize_config() {
                log::error!(
                    "Error serializing config file to '{}': {err}",
                    self.config_path.display()
                );
            }
            self.state.last_serialized_config.clone_from(&self.config);

            self.refresh_file_list();

            proxy.send_event(UserEvent::AppConfigChanged).unwrap();
        } else if self.state.filter_by_title != self.state.last_filter_by_title {
            self.refresh_file_list();
            self.state.last_filter_by_title.clone_from(&self.state.filter_by_title);
        }
    }

    fn refresh_file_list(&mut self) {
        self.state.unfiltered_file_list =
            do_file_search(&self.config.paths.search, self.config.paths.search_recursively).into();
        self.state.filtered_file_list = filter_file_list(
            &self.state.unfiltered_file_list,
            &self.state.filter_by_title_lower,
            &self.config.filters,
        )
        .into();

        if let Some(path) = &self.state.last_opened_disc_path {
            self.refresh_change_disc_list(&path.clone());
        }
    }

    fn refresh_change_disc_list(&mut self, path: &Path) {
        static DISC_REGEX: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r" \(Disc ([1-9])\)").unwrap());

        let Some(path_str) = path.to_str() else {
            self.state.change_disc_list = Rc::default();
            return;
        };

        if DISC_REGEX.find(path_str).is_none() {
            self.state.change_disc_list = Rc::default();
            return;
        }

        // This could be more efficient, but it doesn't really matter since this executes very infrequently
        let mut change_disc_list = Vec::new();
        for disc_number in 1..=9 {
            let new_path = DISC_REGEX.replace_all(path_str, format!(" (Disc {disc_number})"));
            for metadata in &*self.state.unfiltered_file_list {
                let Some(metadata_path_str) = metadata.full_path.to_str() else { continue };
                if metadata_path_str == new_path {
                    change_disc_list.push(ChangeDiscEntry {
                        label: format!("Disc {disc_number}"),
                        file: metadata.clone(),
                    });
                }
            }
        }

        self.state.change_disc_list = change_disc_list.into();
    }

    fn render_menu(
        &mut self,
        ui: &mut Ui,
        emu_state: &EmulatorState,
        proxy: &EventLoopProxy<UserEvent>,
    ) {
        let open_shortcut = KeyboardShortcut::new(Modifiers::CTRL, Key::O);
        if ui.input_mut(|input| input.consume_shortcut(&open_shortcut)) {
            proxy
                .send_event(UserEvent::OpenFileDialog {
                    file_type: OpenFileType::Open,
                    initial_dir: None,
                })
                .unwrap();
        }

        let quit_shortcut = KeyboardShortcut::new(Modifiers::CTRL, Key::Q);
        if ui.input_mut(|input| input.consume_shortcut(&quit_shortcut)) {
            proxy.send_event(UserEvent::Close).unwrap();
        }

        Panel::top("menu_panel").show_inside(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    let open_button =
                        Button::new("Open").shortcut_text(ui.format_shortcut(&open_shortcut));
                    if ui.add(open_button).clicked() {
                        proxy
                            .send_event(UserEvent::OpenFileDialog {
                                file_type: OpenFileType::Open,
                                initial_dir: None,
                            })
                            .unwrap();
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Run BIOS").clicked() {
                        proxy.send_event(UserEvent::RunBios).unwrap();
                        ui.close_kind(UiKind::Menu);
                    }

                    let quit_button =
                        Button::new("Quit").shortcut_text(ui.format_shortcut(&quit_shortcut));
                    if ui.add(quit_button).clicked() {
                        proxy.send_event(UserEvent::Close).unwrap();
                    }
                });

                ui.menu_button("Settings", |ui| {
                    if ui.button("Video").clicked() {
                        self.state.video_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Graphics").clicked() {
                        self.state.graphics_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Audio").clicked() {
                        self.state.audio_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Input").clicked() {
                        self.state.input_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Paths").clicked() {
                        self.state.paths_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Memory Cards").clicked() {
                        self.state.memcards_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }

                    if ui.button("Debug").clicked() {
                        self.state.debug_window_open = true;
                        ui.close_kind(UiKind::Menu);
                    }
                });

                ui.menu_button("Emulation", |ui| {
                    ui.add_enabled_ui(emu_state.is_emulator_running(), |ui| {
                        if ui.button("Reset").clicked() {
                            proxy.send_event(UserEvent::Reset).unwrap();
                            ui.close_kind(UiKind::Menu);
                        }

                        if ui.button("Power Off").clicked() {
                            proxy.send_event(UserEvent::PowerOff).unwrap();
                            ui.close_kind(UiKind::Menu);
                        }

                        ui.add_space(10.0);

                        ui.menu_button("Change Disc", |ui| {
                            self.render_change_disc_submenu(proxy, ui);
                        });

                        if ui.button("Remove Disc").clicked() {
                            proxy.send_event(UserEvent::RemoveDisc).unwrap();
                            ui.close_kind(UiKind::Menu);
                        }
                    });
                });
            });
        });
    }

    fn render_change_disc_submenu(&mut self, proxy: &EventLoopProxy<UserEvent>, ui: &mut Ui) {
        for change_disc_entry in &*self.state.change_disc_list {
            if ui.button(&change_disc_entry.label).clicked() {
                proxy
                    .send_event(UserEvent::FileOpened(
                        OpenFileType::DiscChange,
                        Some(change_disc_entry.file.full_path.clone()),
                    ))
                    .unwrap();
                ui.close_kind(UiKind::Menu);
            }
        }

        if ui.button("Select file...").clicked() {
            let initial_dir = self
                .state
                .last_opened_disc_path
                .as_ref()
                .and_then(|disc_path| disc_path.parent())
                .map(PathBuf::from);

            proxy
                .send_event(UserEvent::OpenFileDialog {
                    file_type: OpenFileType::DiscChange,
                    initial_dir,
                })
                .unwrap();
            ui.close_kind(UiKind::Menu);
        }
    }

    fn render_video_window(&mut self, ctx: &Context) {
        Window::new("Video Settings")
            .open(&mut self.state.video_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.checkbox(&mut self.config.video.launch_in_fullscreen, "Launch in fullscreen");

                ui.group(|ui| {
                    ui.label("VSync mode");

                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut self.config.video.vsync_mode,
                            VSyncMode::Enabled,
                            "Enabled",
                        )
                        .on_hover_text("wgpu Fifo present mode");
                        ui.radio_value(
                            &mut self.config.video.vsync_mode,
                            VSyncMode::Disabled,
                            "Disabled",
                        )
                        .on_hover_text("wgpu Immediate present mode");
                        ui.radio_value(&mut self.config.video.vsync_mode, VSyncMode::Fast, "Fast")
                            .on_hover_text("wgpu Mailbox present mode");
                    });
                });

                ui.group(|ui| {
                    ui.label("Aspect ratio");

                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut self.config.video.aspect_ratio,
                            AspectRatio::Native,
                            "Native",
                        )
                        .on_hover_text("NTSC or PAL based on video mode");
                        ui.radio_value(
                            &mut self.config.video.aspect_ratio,
                            AspectRatio::Stretched,
                            "Stretched",
                        )
                        .on_hover_text("Stretched to fill the window");
                    });
                });

                ui.group(|ui| {
                    ui.label("Image filtering");

                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut self.config.video.filter_mode,
                            FilterMode::Linear,
                            "Bilinear interpolation",
                        );
                        ui.radio_value(
                            &mut self.config.video.filter_mode,
                            FilterMode::Nearest,
                            "Nearest neighbor",
                        );
                    });
                });

                ui.checkbox(
                    &mut self.config.video.crop_vertical_overscan,
                    "Crop vertical overscan",
                )
                .on_hover_text("Crop vertical display to 224px NTSC / 268px PAL");
            });
    }

    fn render_graphics_window(&mut self, ctx: &Context) {
        Window::new("Graphics Settings")
            .open(&mut self.state.graphics_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.group(|ui| {
                    ui.label("Rasterizer");

                    ui.horizontal(|ui| {
                        ui.radio_value(
                            &mut self.config.graphics.rasterizer,
                            Rasterizer::Software,
                            "Software",
                        )
                        .on_hover_text("CPU-based; more accurate but no enhancements");
                        ui.radio_value(
                            &mut self.config.graphics.rasterizer,
                            Rasterizer::Hardware,
                            "Hardware (wgpu)",
                        )
                        .on_hover_text("GPU-based; supports enhancements but less accurate");
                    });
                });

                let is_hw_rasterizer = self.config.graphics.rasterizer == Rasterizer::Hardware;
                let disabled_hover_text = "Hardware rasterizer only";

                ui.add_enabled_ui(is_hw_rasterizer, |ui| {
                    ui.group(|ui| {
                        ui.label("wgpu backend (requires game restart)")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.config.graphics.wgpu_backend,
                                WgpuBackend::Auto,
                                "Auto",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.graphics.wgpu_backend,
                                WgpuBackend::Vulkan,
                                "Vulkan",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.graphics.wgpu_backend,
                                WgpuBackend::DirectX12,
                                "DirectX 12",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.graphics.wgpu_backend,
                                WgpuBackend::Metal,
                                "Metal",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                        });
                    });

                    ui.group(|ui| {
                        ui.label("Draw command color depth")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.horizontal(|ui| {
                            ui.radio_value(&mut self.config.graphics.hardware_high_color, false, "15bpp (Native)")
                                .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(&mut self.config.graphics.hardware_high_color, true, "24bpp (High color)")
                                .on_hover_text("Works very well with most games but sometimes changes a game's look (e.g. Silent Hill)")
                                .on_disabled_hover_text(disabled_hover_text);
                        });
                    });

                    ui.horizontal(|ui| {
                        let format_scale = |scale| match scale {
                            1 => "1x (Native)".into(),
                            _ => format!("{scale}x")
                        };

                        ComboBox::from_label("Resolution scale")
                            .selected_text(format_scale(self.config.graphics.hardware_resolution_scale))
                            .show_ui(ui, |ui| {
                                for scale in 1..=16 {
                                    ui.selectable_value(&mut self.config.graphics.hardware_resolution_scale, scale, format_scale(scale));
                                }
                            });
                    });

                    ui.add_enabled_ui(!self.config.graphics.hardware_high_color, |ui| {
                        let disabled_hover_text = "Hardware rasterizer 15bpp mode only";

                        ui.checkbox(&mut self.config.graphics.hardware_15bpp_dithering, "Dithering enabled")
                            .on_hover_text("Whether to respect the PS1 GPU's dithering flag")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.checkbox(&mut self.config.graphics.high_res_dithering, "High-resolution dithering")
                            .on_hover_text("Apply dithering at scaled resolution instead of native")
                            .on_disabled_hover_text(disabled_hover_text);
                    });
                });

                ui.checkbox(
                    &mut self.config.graphics.async_swap_chain_rendering,
                    "Asynchronous GPU rendering",
                )
                .on_hover_text("Should improve performance, but can cause skipped frames and input latency")
                .on_disabled_hover_text(disabled_hover_text);

                ui.add_enabled_ui(!is_hw_rasterizer && config::supports_avx2(), |ui| {
                    ui.checkbox(&mut self.config.graphics.avx2_software_rasterizer, "Use AVX2 software rasterizer")
                        .on_hover_text("Significantly improves software rasterizer performance if AVX2 is supported");
                });

                ui.add_enabled_ui(is_hw_rasterizer, |ui| {
                    ui.group(|ui| {
                        ui.label("PGXP (Enhanced vertex coordinate precision)")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.checkbox(&mut self.config.graphics.pgxp_enabled, "Enabled")
                            .on_hover_text("Reduces model wobble in most 3D games")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.add_enabled_ui(self.config.graphics.pgxp_enabled, |ui| {
                            ui.checkbox(&mut self.config.graphics.pgxp_precise_culling, "High-precision culling")
                                .on_hover_text("Perform culling calculations using high-precision vertex coordinates")
                                .on_disabled_hover_text("Requires PGXP");

                            ui.checkbox(&mut self.config.graphics.pgxp_perspective_texture_mapping, "Perspective-correct texture mapping")
                                .on_hover_text("Reduces affine texture warping in most 3D games")
                                .on_disabled_hover_text("Requires PGXP");
                        });
                    });
                });
            });
    }

    fn render_audio_window(&mut self, ctx: &Context) {
        Window::new("Audio Settings")
            .open(&mut self.state.audio_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.group(|ui| {
                    ui.label("SPU ADPCM sample interpolation");

                    ui.radio_value(
                        &mut self.config.audio.adpcm_interpolation,
                        AdpcmInterpolation::Gaussian,
                        "Gaussian (Native)",
                    );
                    ui.radio_value(
                        &mut self.config.audio.adpcm_interpolation,
                        AdpcmInterpolation::Hermite,
                        "Cubic Hermite (Sharp)",
                    );
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    let hover_text =
                        "Higher values reduce audio stutters but increase audio latency";

                    self.state.audio_sync_threshold.add_ui(
                        ui,
                        &mut self.config.audio.sync_threshold,
                        |value| value != 0,
                    );

                    ui.label("Audio sync threshold (samples)").on_hover_text(hover_text);
                });

                if self.state.audio_sync_threshold.invalid {
                    ui.colored_label(
                        Color32::RED,
                        "Audio sync threshold must be a non-negative integer",
                    );
                }

                ui.horizontal(|ui| {
                    self.state.audio_device_queue_size.add_ui(
                        ui,
                        &mut self.config.audio.device_queue_size,
                        |value| value >= 8 && value.is_power_of_two(),
                    );

                    ui.label("Audio device queue size (samples)");
                });

                if self.state.audio_device_queue_size.invalid {
                    ui.colored_label(
                        Color32::RED,
                        "Audio device queue size must be a power of two",
                    );
                }

                ui.horizontal(|ui| {
                    self.state.internal_audio_buffer_size.add_ui(
                        ui,
                        &mut self.config.audio.internal_buffer_size,
                        |_value| true,
                    );

                    ui.label("Internal audio buffer size (samples)");
                });

                if self.state.internal_audio_buffer_size.invalid {
                    ui.colored_label(
                        Color32::RED,
                        "Internal audio buffer size must be a non-negative integer",
                    );
                }
            });
    }

    fn render_input_window(&mut self, ctx: &Context) {
        let mut open = self.state.input_window_open;
        Window::new("Input Settings").open(&mut open).show(ctx, |ui| {
            ui.add_enabled_ui(self.state.waiting_for_input.is_none(), |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.state.selected_controller,
                        ControllerNumber::One,
                        "Controller 1",
                    );
                    ui.selectable_value(
                        &mut self.state.selected_controller,
                        ControllerNumber::Two,
                        "Controller 2",
                    );
                });

                ui.add_space(10.0);

                ui.group(|ui| {
                    ui.label("Device");

                    let device_field = match self.state.selected_controller {
                        ControllerNumber::One => &mut self.config.input.p1_device,
                        ControllerNumber::Two => &mut self.config.input.p2_device,
                    };

                    ui.horizontal(|ui| {
                        ui.radio_value(device_field, ControllerType::None, "None");
                        ui.radio_value(device_field, ControllerType::Digital, "Digital controller");
                        ui.radio_value(device_field, ControllerType::DualShock, "DualShock");
                    });
                });

                ui.add_space(10.0);

                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.state.selected_input_set,
                        InputSet::One,
                        "Input Set 1",
                    );
                    ui.selectable_value(
                        &mut self.state.selected_input_set,
                        InputSet::Two,
                        "Input Set 2",
                    );
                });

                ui.add_space(10.0);

                self.render_input_set_settings(ui);
            });
        });
        self.state.input_window_open = open;
    }

    fn render_paths_window(&mut self, ctx: &Context, proxy: &EventLoopProxy<UserEvent>) {
        Window::new("Paths Settings")
            .open(&mut self.state.paths_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let button_text = self
                        .config
                        .paths
                        .bios
                        .as_ref()
                        .and_then(|path| path.to_str())
                        .unwrap_or("<None>");
                    if ui.button(button_text).clicked() {
                        let initial_dir = self
                            .config
                            .paths
                            .bios
                            .as_ref()
                            .and_then(|path| path.parent())
                            .map(PathBuf::from);

                        proxy
                            .send_event(UserEvent::OpenFileDialog {
                                file_type: OpenFileType::BiosPath,
                                initial_dir,
                            })
                            .unwrap();
                    }

                    ui.label("BIOS path");
                });

                ui.group(|ui| {
                    ui.heading("Search paths");

                    Grid::new("search_paths_grid").show(ui, |ui| {
                        for path in self.config.paths.search.clone() {
                            ui.label(path.display().to_string());

                            if ui.button("Remove").clicked() {
                                self.config.paths.search.retain(|p| p != &path);
                            }

                            ui.end_row();
                        }
                    });

                    if ui.button("Add").clicked() {
                        proxy
                            .send_event(UserEvent::OpenFileDialog {
                                file_type: OpenFileType::SearchDir,
                                initial_dir: None,
                            })
                            .unwrap();
                    }

                    ui.checkbox(&mut self.config.paths.search_recursively, "Search recursively");
                });
            });
    }

    fn render_memcards_window(&mut self, ctx: &Context) {
        Window::new("Memory Card Settings")
            .open(&mut self.state.memcards_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.checkbox(
                    &mut self.config.memory_cards.slot_1_enabled,
                    "Memory card enabled in slot 1",
                );
                ui.checkbox(
                    &mut self.config.memory_cards.slot_2_enabled,
                    "Memory card enabled in slot 2",
                );

                ui.add(MemoryCardModeWidget::new(
                    "Memory card slot 1 mode",
                    &mut self.config.memory_cards.slot_1_mode,
                ));
                ui.add(MemoryCardModeWidget::new(
                    "Memory card slot 2 mode",
                    &mut self.config.memory_cards.slot_2_mode,
                ));
            });
    }

    fn render_debug_window(&mut self, ctx: &Context) {
        Window::new("Debug Settings")
            .open(&mut self.state.debug_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.checkbox(&mut self.config.debug.tty_enabled, "TTY output enabled")
                    .on_hover_text("Print TTY output to stdout");

                ui.checkbox(&mut self.config.debug.vram_display, "VRAM display").on_hover_text(
                    "Display the entire contents of VRAM instead of only the current frame buffer",
                );
            });
    }

    fn render_central_panel(&mut self, ui: &mut Ui, proxy: &EventLoopProxy<UserEvent>) {
        CentralPanel::default().show_inside(ui, |ui| {
            let bios_path_configured = self.config.paths.bios.is_some();
            let search_paths_configured = !self.config.paths.search.is_empty();

            if !bios_path_configured || !search_paths_configured {
                ui.centered_and_justified(|ui| {
                    let label = if !bios_path_configured && !search_paths_configured {
                        "Configure BIOS path and search path(s)"
                    } else if !bios_path_configured {
                        "Configure BIOS path"
                    } else {
                        "Configure search path(s)"
                    };
                    if ui.button(label).clicked() {
                        self.state.paths_window_open = true;
                    }
                });

                return;
            }

            ui.horizontal(|ui| {
                if ui
                    .add(
                        TextEdit::singleline(&mut self.state.filter_by_title)
                            .desired_width(500.0)
                            .hint_text("Filter by name"),
                    )
                    .changed()
                {
                    self.state.filter_by_title_lower = self.state.filter_by_title.to_lowercase();
                }

                if ui.button("Clear").clicked() {
                    self.state.filter_by_title.clear();
                    self.state.filter_by_title_lower.clear();
                }

                ui.add_space(40.0);

                ui.checkbox(&mut self.config.filters.exe, "EXE");
                ui.checkbox(&mut self.config.filters.cue, "CUE");
                ui.checkbox(&mut self.config.filters.chd, "CHD");
            });

            ui.add_space(15.0);

            TableBuilder::new(ui)
                .auto_shrink([false; 2])
                .striped(true)
                .max_scroll_height(3000.0)
                .cell_layout(Layout::left_to_right(Align::Center))
                .column(Column::auto().at_most(500.0))
                .column(Column::auto())
                .column(Column::remainder())
                .header(25.0, |mut row| {
                    row.col(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("Name");
                        });
                    });

                    row.col(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("File Type");
                        });
                    });

                    // Blank column to make stripes extend to the right
                    row.col(|_ui| {});
                })
                .body(|body| {
                    let file_list = Rc::clone(&self.state.filtered_file_list);
                    body.rows(30.0, file_list.len(), |mut row| {
                        let metadata = &file_list[row.index()];

                        row.col(|ui| {
                            if ui
                                .add(
                                    Button::new(&metadata.file_name_no_ext)
                                        .min_size(Vec2::new(500.0, 25.0))
                                        .wrap(),
                                )
                                .clicked()
                            {
                                proxy
                                    .send_event(UserEvent::FileOpened(
                                        OpenFileType::Open,
                                        Some(metadata.full_path.clone()),
                                    ))
                                    .unwrap();
                            }
                        });

                        row.col(|ui| {
                            ui.centered_and_justified(|ui| {
                                ui.label(metadata.extension.as_str());
                            });
                        });

                        // Blank column to make stripes extend to the right
                        row.col(|_ui| {});
                    });
                });
        });
    }

    fn serialize_config(&mut self) -> anyhow::Result<()> {
        let config_str = toml::to_string_pretty(&self.config)?;
        fs::write(&self.config_path, config_str)?;

        log::debug!("Serialized config file to '{}'", self.config_path.display());

        Ok(())
    }

    #[must_use]
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    #[must_use]
    pub fn config_mut(&mut self) -> &mut AppConfig {
        &mut self.config
    }
}

struct MemoryCardModeWidget<'a> {
    label: &'a str,
    current_value: &'a mut MemoryCardMode,
}

impl<'a> MemoryCardModeWidget<'a> {
    fn new(label: &'a str, current_value: &'a mut MemoryCardMode) -> Self {
        Self { label, current_value }
    }
}

impl Widget for MemoryCardModeWidget<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        ui.group(|ui| {
            ui.label(self.label);

            ui.radio_value(
                self.current_value,
                MemoryCardMode::PerGame,
                "Per-game memory card files",
            );
            ui.radio_value(self.current_value, MemoryCardMode::Shared, "Shared memory card file");
        })
        .response
    }
}

fn read_config<P: AsRef<Path>>(path: P) -> anyhow::Result<AppConfig> {
    let path = path.as_ref();

    let config_str = fs::read_to_string(path)?;
    let config: AppConfig = toml::from_str(&config_str)?;

    Ok(config)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileExtension {
    Exe,
    Cue,
    Chd,
}

impl FileExtension {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exe => "EXE",
            Self::Cue => "CUE",
            Self::Chd => "CHD",
        }
    }
}

#[derive(Debug, Clone)]
struct FileMetadata {
    file_name_no_ext: String,
    extension: FileExtension,
    full_path: PathBuf,
}

fn do_file_search(search_dirs: &[PathBuf], recursive: bool) -> Vec<FileMetadata> {
    let mut visited_dirs = HashSet::new();
    let mut files = Vec::new();
    for search_dir in search_dirs {
        do_file_search_inner(search_dir, recursive, &mut visited_dirs, &mut files);
    }

    files
}

fn filter_file_list(
    files: &[FileMetadata],
    filter_by_title_lower: &str,
    file_filters: &FiltersConfig,
) -> Vec<FileMetadata> {
    let mut files = files.to_vec();

    files.retain(|metadata| {
        let name_match = metadata.file_name_no_ext.to_lowercase().contains(filter_by_title_lower);
        let extension_match = (metadata.extension == FileExtension::Exe && file_filters.exe)
            || (metadata.extension == FileExtension::Cue && file_filters.cue)
            || (metadata.extension == FileExtension::Chd && file_filters.chd);

        name_match && extension_match
    });

    files.sort_by(|a, b| a.file_name_no_ext.cmp(&b.file_name_no_ext));

    files
}

fn do_file_search_inner(
    dir: &Path,
    recursive: bool,
    visited_dirs: &mut HashSet<PathBuf>,
    out: &mut Vec<FileMetadata>,
) {
    if !visited_dirs.insert(dir.into()) {
        return;
    }

    let Ok(read_dir) = fs::read_dir(dir) else { return };
    for dir_entry in read_dir {
        let Ok(dir_entry) = dir_entry else { continue };
        let Ok(file_type) = dir_entry.file_type() else { continue };

        let entry_path = dir_entry.path();
        let path_no_ext = entry_path.with_extension("");
        let Some(file_name_no_ext) = path_no_ext.file_name().and_then(OsStr::to_str) else {
            continue;
        };

        if file_type.is_dir() && recursive {
            do_file_search_inner(&entry_path, true, visited_dirs, out);
        } else if file_type.is_file() {
            let Some(extension) = entry_path.extension().and_then(OsStr::to_str) else { continue };
            let ext_lower = extension.to_lowercase();
            if matches!(ext_lower.as_str(), "exe" | "cue" | "chd") {
                // TODO check that EXE is a PS1 executable
                out.push(FileMetadata {
                    file_name_no_ext: file_name_no_ext.into(),
                    extension: match ext_lower.as_str() {
                        "exe" => FileExtension::Exe,
                        "cue" => FileExtension::Cue,
                        "chd" => FileExtension::Chd,
                        _ => unreachable!("nested match expressions"),
                    },
                    full_path: entry_path,
                });
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ChangeDiscEntry {
    label: String,
    file: FileMetadata,
}
