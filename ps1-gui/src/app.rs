use crate::config::{AppConfig, FilterMode, Rasterizer, VSyncMode, WgpuBackend};
use crate::{config, OpenFileType, UserEvent};
use egui::{
    Align, Button, CentralPanel, Color32, Context, Key, KeyboardShortcut, Layout, Modifiers,
    Slider, TextEdit, TopBottomPanel, Vec2, Window,
};
use egui_extras::{Column, TableBuilder};
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use winit::event_loop::EventLoopProxy;

struct AppState {
    video_window_open: bool,
    graphics_window_open: bool,
    audio_window_open: bool,
    paths_window_open: bool,
    audio_sync_threshold_text: String,
    audio_sync_threshold_invalid: bool,
    audio_device_queue_size_text: String,
    audio_device_queue_size_invalid: bool,
    file_list: Rc<[FileMetadata]>,
    last_serialized_config: AppConfig,
    filter_by_title: String,
    filter_by_title_lower: String,
    last_filter_by_title: String,
}

impl AppState {
    fn new(config: &AppConfig) -> Self {
        let file_list = do_file_search(&config.paths.search, config.paths.search_recursively, "");

        Self {
            video_window_open: false,
            graphics_window_open: false,
            audio_window_open: false,
            paths_window_open: false,
            audio_sync_threshold_text: config.audio.sync_threshold.to_string(),
            audio_sync_threshold_invalid: false,
            audio_device_queue_size_text: config.audio.device_queue_size.to_string(),
            audio_device_queue_size_invalid: false,
            file_list: file_list.into(),
            last_serialized_config: config.clone(),
            filter_by_title: String::new(),
            filter_by_title_lower: String::new(),
            last_filter_by_title: String::new(),
        }
    }
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

    #[allow(clippy::single_match)]
    pub fn handle_event(&mut self, event: &UserEvent) {
        match event {
            UserEvent::FileOpened(OpenFileType::BiosPath, Some(path)) => {
                self.config.paths.bios = Some(path.clone());
            }
            UserEvent::FileOpened(OpenFileType::SearchDir, Some(path)) => {
                self.config.paths.search.push(path.clone());
            }
            _ => {}
        }
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn render(&mut self, ctx: &Context, proxy: &EventLoopProxy<UserEvent>) {
        self.render_menu(ctx, proxy);
        self.render_central_panel(ctx, proxy);

        if self.state.video_window_open {
            self.render_video_window(ctx);
        }

        if self.state.graphics_window_open {
            self.render_graphics_window(ctx);
        }

        if self.state.audio_window_open {
            self.render_audio_window(ctx);
        }

        if self.state.paths_window_open {
            self.render_paths_window(ctx, proxy);
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
        self.state.file_list = do_file_search(
            &self.config.paths.search,
            self.config.paths.search_recursively,
            &self.state.filter_by_title_lower,
        )
        .into();
    }

    fn render_menu(&mut self, ctx: &Context, proxy: &EventLoopProxy<UserEvent>) {
        let open_shortcut = KeyboardShortcut::new(Modifiers::CTRL, Key::O);
        if ctx.input_mut(|input| input.consume_shortcut(&open_shortcut)) {
            proxy
                .send_event(UserEvent::OpenFile {
                    file_type: OpenFileType::Open,
                    initial_dir: None,
                })
                .unwrap();
        }

        let quit_shortcut = KeyboardShortcut::new(Modifiers::CTRL, Key::Q);
        if ctx.input_mut(|input| input.consume_shortcut(&quit_shortcut)) {
            proxy.send_event(UserEvent::Close).unwrap();
        }

        TopBottomPanel::top("menu_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    let open_button =
                        Button::new("Open").shortcut_text(ctx.format_shortcut(&open_shortcut));
                    if ui.add(open_button).clicked() {
                        proxy
                            .send_event(UserEvent::OpenFile {
                                file_type: OpenFileType::Open,
                                initial_dir: None,
                            })
                            .unwrap();
                        ui.close_menu();
                    }

                    if ui.button("Run BIOS").clicked() {
                        proxy.send_event(UserEvent::RunBios).unwrap();
                        ui.close_menu();
                    }

                    let quit_button =
                        Button::new("Quit").shortcut_text(ctx.format_shortcut(&quit_shortcut));
                    if ui.add(quit_button).clicked() {
                        proxy.send_event(UserEvent::Close).unwrap();
                    }
                });

                ui.menu_button("Settings", |ui| {
                    if ui.button("Video").clicked() {
                        self.state.video_window_open = true;
                        ui.close_menu();
                    }

                    if ui.button("Graphics").clicked() {
                        self.state.graphics_window_open = true;
                        ui.close_menu();
                    }

                    if ui.button("Audio").clicked() {
                        self.state.audio_window_open = true;
                        ui.close_menu();
                    }

                    if ui.button("Paths").clicked() {
                        self.state.paths_window_open = true;
                        ui.close_menu();
                    }
                });
            });
        });
    }

    fn render_video_window(&mut self, ctx: &Context) {
        Window::new("Video Settings")
            .open(&mut self.state.video_window_open)
            .resizable(false)
            .show(ctx, |ui| {
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

                ui.checkbox(&mut self.config.video.vram_display, "VRAM display").on_hover_text(
                    "Display the entire contents of VRAM instead of only the current frame buffer",
                );
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
                            &mut self.config.video.rasterizer,
                            Rasterizer::Software,
                            "Software",
                        )
                        .on_hover_text("CPU-based; more accurate but no enhancements");
                        ui.radio_value(
                            &mut self.config.video.rasterizer,
                            Rasterizer::Hardware,
                            "Hardware (wgpu)",
                        )
                        .on_hover_text("GPU-based; supports enhancements but less accurate");
                    });
                });

                let is_hw_rasterizer = self.config.video.rasterizer == Rasterizer::Hardware;
                let disabled_hover_text = "Hardware rasterizer only";

                ui.add_enabled_ui(is_hw_rasterizer, |ui| {
                    ui.group(|ui| {
                        ui.label("wgpu backend (requires game restart)")
                            .on_disabled_hover_text(disabled_hover_text);

                        ui.horizontal(|ui| {
                            ui.radio_value(
                                &mut self.config.video.wgpu_backend,
                                WgpuBackend::Auto,
                                "Auto",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.video.wgpu_backend,
                                WgpuBackend::Vulkan,
                                "Vulkan",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.video.wgpu_backend,
                                WgpuBackend::DirectX12,
                                "DirectX 12",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                            ui.radio_value(
                                &mut self.config.video.wgpu_backend,
                                WgpuBackend::Metal,
                                "Metal",
                            )
                            .on_disabled_hover_text(disabled_hover_text);
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.label("Resolution scale:").on_disabled_hover_text(disabled_hover_text);

                        ui.add(Slider::new(
                            &mut self.config.video.hardware_resolution_scale,
                            1..=16,
                        ))
                        .on_disabled_hover_text(disabled_hover_text);
                    });
                });

                ui.checkbox(
                    &mut self.config.video.async_swap_chain_rendering,
                    "Asynchronous rendering",
                )
                .on_hover_text("Should improve performance, but can cause skipped frames and input latency if GPU cannot keep up")
                .on_disabled_hover_text(disabled_hover_text);

                ui.add_enabled_ui(!is_hw_rasterizer && config::supports_avx2(), |ui| {
                    ui.checkbox(&mut self.config.video.avx2_software_rasterizer, "Use AVX2 software rasterizer")
                        .on_hover_text("Significantly improves software rasterizer performance if AVX2 is supported");
                });
            });
    }

    fn render_audio_window(&mut self, ctx: &Context) {
        Window::new("Audio Settings")
            .open(&mut self.state.audio_window_open)
            .resizable(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let hover_text =
                        "Higher values reduce audio stutters but increase audio latency";

                    if ui
                        .add(
                            TextEdit::singleline(&mut self.state.audio_sync_threshold_text)
                                .desired_width(30.0),
                        )
                        .on_hover_text(hover_text)
                        .changed()
                    {
                        match self.state.audio_sync_threshold_text.parse::<u32>() {
                            Ok(value) if value != 0 => {
                                self.config.audio.sync_threshold = value;
                                self.state.audio_sync_threshold_invalid = false;
                            }
                            _ => {
                                self.state.audio_sync_threshold_invalid = true;
                            }
                        }
                    }

                    ui.label("Audio sync threshold (samples)").on_hover_text(hover_text);
                });

                if self.state.audio_sync_threshold_invalid {
                    ui.colored_label(
                        Color32::RED,
                        "Audio sync threshold must be a non-negative integer",
                    );
                }

                ui.horizontal(|ui| {
                    if ui
                        .add(
                            TextEdit::singleline(&mut self.state.audio_device_queue_size_text)
                                .desired_width(30.0),
                        )
                        .changed()
                    {
                        match self.state.audio_device_queue_size_text.parse::<u16>() {
                            Ok(value) if value >= 8 && value.count_ones() == 1 => {
                                self.config.audio.device_queue_size = value;
                                self.state.audio_device_queue_size_invalid = false;
                            }
                            _ => {
                                self.state.audio_device_queue_size_invalid = true;
                            }
                        }
                    }

                    ui.label("Audio device queue size (samples)");
                });

                if self.state.audio_device_queue_size_invalid {
                    ui.colored_label(
                        Color32::RED,
                        "Audio device queue size must be a power of two",
                    );
                }
            });
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
                            .send_event(UserEvent::OpenFile {
                                file_type: OpenFileType::BiosPath,
                                initial_dir,
                            })
                            .unwrap();
                    }

                    ui.label("BIOS path");
                });

                ui.group(|ui| {
                    ui.heading("Search paths");

                    for path in self.config.paths.search.clone() {
                        ui.horizontal(|ui| {
                            ui.label(path.display().to_string());

                            if ui.button("Remove").clicked() {
                                self.config.paths.search.retain(|p| p != &path);
                            }
                        });
                    }

                    if ui.button("Add").clicked() {
                        proxy
                            .send_event(UserEvent::OpenFile {
                                file_type: OpenFileType::SearchDir,
                                initial_dir: None,
                            })
                            .unwrap();
                    }
                });

                ui.checkbox(&mut self.config.paths.search_recursively, "Search recursively");
            });
    }

    fn render_central_panel(&mut self, ctx: &Context, proxy: &EventLoopProxy<UserEvent>) {
        CentralPanel::default().show(ctx, |ui| {
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
            });

            ui.add_space(15.0);

            TableBuilder::new(ui)
                .auto_shrink([false; 2])
                .striped(true)
                .max_scroll_height(2000.0)
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
                .body(|mut body| {
                    let file_list = Rc::clone(&self.state.file_list);
                    for metadata in file_list.as_ref() {
                        body.row(30.0, |mut row| {
                            row.col(|ui| {
                                if ui
                                    .add(
                                        Button::new(&metadata.file_name_no_ext)
                                            .min_size(Vec2::new(500.0, 25.0))
                                            .wrap(true),
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
                                    ui.label(metadata.extension.to_uppercase());
                                });
                            });

                            // Blank column to make stripes extend to the right
                            row.col(|_ui| {});
                        });
                    }
                });
        });
    }

    fn serialize_config(&mut self) -> anyhow::Result<()> {
        let config_str = toml::to_string_pretty(&self.config)?;
        fs::write(&self.config_path, config_str)?;

        log::debug!("Serialized config file to '{}'", self.config_path.display());

        Ok(())
    }

    pub fn config_mut(&mut self) -> &mut AppConfig {
        &mut self.config
    }
}

fn read_config<P: AsRef<Path>>(path: P) -> anyhow::Result<AppConfig> {
    let path = path.as_ref();

    let config_str = fs::read_to_string(path)?;
    let config: AppConfig = toml::from_str(&config_str)?;

    Ok(config)
}

#[derive(Debug, Clone)]
struct FileMetadata {
    file_name_no_ext: String,
    extension: String,
    full_path: PathBuf,
}

fn do_file_search(
    search_dirs: &[PathBuf],
    recursive: bool,
    filter_by_title: &str,
) -> Vec<FileMetadata> {
    let mut visited_dirs = HashSet::new();
    let mut files = Vec::new();
    for search_dir in search_dirs {
        do_file_search_inner(search_dir, recursive, filter_by_title, &mut visited_dirs, &mut files);
    }

    files.sort_by(|a, b| a.file_name_no_ext.cmp(&b.file_name_no_ext));

    files
}

fn do_file_search_inner(
    dir: &Path,
    recursive: bool,
    filter_by_title: &str,
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

        if !filter_by_title.is_empty() && !file_name_no_ext.to_lowercase().contains(filter_by_title)
        {
            continue;
        }

        if file_type.is_dir() {
            if recursive {
                do_file_search_inner(&entry_path, true, filter_by_title, visited_dirs, out);
            }
            continue;
        }

        let Some(extension) = entry_path.extension().and_then(OsStr::to_str) else { continue };
        if matches!(extension, "exe" | "cue" | "chd") {
            // TODO check that EXE is a PS1 executable
            out.push(FileMetadata {
                file_name_no_ext: file_name_no_ext.into(),
                extension: extension.into(),
                full_path: entry_path,
            });
        }
    }
}
