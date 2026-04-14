mod audio;
mod renderer;

use crate::Never;
use crate::config::{AppConfig, GraphicsConfig, MemoryCardConfig};
use crate::emuthread::audio::{AudioQueue, QueueAudioCallback, QueueAudioOutput};
use crate::emuthread::renderer::{SurfaceRenderer, SwapChainRenderer};
use anyhow::{Context, anyhow};
use cdrom::reader::{CdRom, CdRomFileFormat};
use ps1_core::api::{
    LoadedMemoryCards, MemoryCardSlot, Ps1Emulator, Ps1EmulatorBuilder, Ps1EmulatorState,
    SaveWriter, TickEffect, TickError,
};
use ps1_core::input::{AnalogJoypadState, DigitalJoypadState, Ps1Inputs};
pub use renderer::SurfaceRenderEffect;
use sdl2::audio::AudioDevice;
use sdl2::{AudioSubsystem, Sdl};
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;
use std::{fs, io, thread};
use winit::dpi::PhysicalSize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    One,
    Two,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ps1Button {
    Up,
    Down,
    Left,
    Right,
    Cross,
    Circle,
    Square,
    Triangle,
    L1,
    L2,
    R1,
    R2,
    Start,
    Select,
    Analog,
    L3,
    R3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ps1AnalogInput {
    LeftStickX,
    LeftStickY,
    RightStickX,
    RightStickY,
}

#[derive(Debug)]
pub enum EmulatorThreadCommand {
    Reset,
    Stop,
    DigitalInput { player: Player, button: Ps1Button, pressed: bool },
    AnalogInput { player: Player, input: Ps1AnalogInput, value: i16 },
    ChangeDisc { disc_path: PathBuf },
    RemoveDisc,
    UpdateConfig(Box<AppConfig>),
    SaveState,
    LoadState,
    TogglePause,
    StepFrame,
    FastForward { enabled: bool },
}

#[derive(Debug)]
struct QueuedFrame {
    view: wgpu::TextureView,
    size: wgpu::Extent3d,
    pixel_aspect_ratio: f64,
}

#[derive(Debug)]
struct FixedSizeDeque<const N: usize>(VecDeque<QueuedFrame>);

impl<const N: usize> FixedSizeDeque<N> {
    fn new() -> Self {
        assert_ne!(N, 0);

        Self(VecDeque::with_capacity(N))
    }

    fn push_back(&mut self, value: QueuedFrame) {
        if self.0.len() == N {
            self.0.pop_front();
        }
        self.0.push_back(value);
    }

    #[must_use]
    fn pop_front(&mut self) -> Option<QueuedFrame> {
        self.0.pop_front()
    }

    fn clear(&mut self) {
        self.0.clear();
    }
}

pub const SWAP_CHAIN_LEN: usize = 3;

type SwapChainTextureBuffer = FixedSizeDeque<{ SWAP_CHAIN_LEN }>;

#[derive(Debug, Clone)]
pub struct EmulatorSwapChain {
    rendered_frames: Arc<Mutex<SwapChainTextureBuffer>>,
    async_rendering: Arc<AtomicBool>,
}

impl EmulatorSwapChain {
    fn new(video_config: &GraphicsConfig) -> Self {
        Self {
            rendered_frames: Arc::new(Mutex::new(SwapChainTextureBuffer::new())),
            async_rendering: Arc::new(AtomicBool::new(video_config.async_swap_chain_rendering)),
        }
    }

    fn update_config(&self, config: &GraphicsConfig) {
        self.async_rendering.store(config.async_swap_chain_rendering, Ordering::Relaxed);
    }
}

pub struct EmulationThreadHandle {
    swap_chain: EmulatorSwapChain,
    surface_renderer: SurfaceRenderer,
    audio_subsystem: AudioSubsystem,
    audio_queue: AudioQueue,
    audio_device: AudioDevice<QueueAudioCallback>,
    command_sender: Sender<EmulatorThreadCommand>,
}

impl EmulationThreadHandle {
    #[allow(clippy::missing_errors_doc)]
    pub fn spawn(
        sdl_ctx: &Sdl,
        file_path: Option<&Path>,
        config: &AppConfig,
        surface_config: &wgpu::SurfaceConfiguration,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
    ) -> anyhow::Result<Self> {
        let Some(bios_path) = &config.paths.bios else {
            return Err(anyhow!("BIOS path is required to run emulator"));
        };

        let bios = fs::read(bios_path)
            .with_context(|| format!("Failed to read BIOS from '{}'", bios_path.display()))?;

        let emulator_config = config.to_emulator_config();

        let save_writer = FsSaveWriter::new(file_path, &config.memory_cards)?;
        let memory_cards = load_memory_cards(&save_writer);

        let builder = Ps1EmulatorBuilder::new(bios, Arc::clone(&device), Arc::clone(&queue))
            .with_config(emulator_config)
            .with_memory_cards_enabled(config.memory_cards.cards_enabled())
            .with_memory_cards(memory_cards);

        let emulator = match file_path {
            Some(file_path) => match file_path.extension().and_then(OsStr::to_str) {
                Some(extension @ ("cue" | "chd")) => {
                    let format = match extension {
                        "cue" => CdRomFileFormat::CueBin,
                        "chd" => CdRomFileFormat::Chd,
                        _ => unreachable!("nested match expressions"),
                    };

                    let disc = CdRom::open(file_path, format)?;
                    builder.with_disc(disc).build()?
                }
                Some("exe") => {
                    let exe = fs::read(file_path).with_context(|| {
                        format!("Failed to read EXE from path {}", file_path.display())
                    })?;

                    let mut emulator = builder.build()?;
                    emulator.run_until_exe_sideloaded(&exe)?;

                    emulator
                }
                Some(extension) => {
                    return Err(anyhow!("Unsupported file extension {extension}"));
                }
                None => {
                    return Err(anyhow!(
                        "Unable to determine file extension of '{}'",
                        file_path.display()
                    ));
                }
            },
            None => builder.build()?,
        };

        let swap_chain = EmulatorSwapChain::new(&config.graphics);
        let swap_chain_renderer =
            SwapChainRenderer::new(Arc::clone(&device), Arc::clone(&queue), swap_chain.clone());

        let audio_subsystem = sdl_ctx
            .audio()
            .map_err(|err| anyhow!("Failed to initialize SDL2 audio subsystem: {err}"))?;
        let audio_queue = Arc::new(Mutex::new(VecDeque::with_capacity(44100 / 30)));
        let audio_callback = QueueAudioCallback::new(Arc::clone(&audio_queue));

        let audio_spec = audio::new_spec(&config.audio);
        let audio_device = audio_subsystem
            .open_playback(None, &audio_spec, move |_| audio_callback)
            .map_err(|err| anyhow!("Failed to initialize SDL2 audio callback: {err}"))?;
        audio_device.resume();

        let audio_output = QueueAudioOutput::new(Arc::clone(&audio_queue));

        let (command_sender, command_receiver) = mpsc::channel();

        let save_state_path = determine_save_state_path(file_path)?;

        let mut inputs = Ps1Inputs::default();
        update_input_config(config, &mut inputs);

        log::info!("Launching emulator with config:\n{config:#?}");

        spawn_emu_thread(
            &config.memory_cards,
            EmulatorRunner {
                emulator,
                renderer: swap_chain_renderer,
                audio_output,
                audio_sync_threshold: config.audio.sync_threshold,
                save_writer,
                inputs,
                disc_path: file_path.map(PathBuf::from),
                save_state_path,
                command_receiver,
            },
        );

        let surface_renderer = SurfaceRenderer::new(
            &config.video,
            Arc::clone(&device),
            Arc::clone(&queue),
            swap_chain.clone(),
            surface_config,
        );

        Ok(Self {
            swap_chain,
            surface_renderer,
            audio_subsystem,
            audio_queue,
            audio_device,
            command_sender,
        })
    }

    pub fn handle_resize(&mut self, size: PhysicalSize<u32>) {
        self.surface_renderer.handle_resize(size);
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn handle_config_change(&mut self, config: &AppConfig) -> anyhow::Result<()> {
        self.swap_chain.update_config(&config.graphics);
        self.surface_renderer.update_config(&config.video);

        if config.audio.device_queue_size != self.audio_device.spec().samples {
            self.audio_device.pause();

            let audio_spec = audio::new_spec(&config.audio);
            let audio_callback = QueueAudioCallback::new(Arc::clone(&self.audio_queue));
            self.audio_device = self
                .audio_subsystem
                .open_playback(None, &audio_spec, move |_| audio_callback)
                .map_err(|err| anyhow!("Error recreating audio device: {err}"))?;
            self.audio_device.resume();
        }

        self.send_command(EmulatorThreadCommand::UpdateConfig(Box::new(config.clone())));

        Ok(())
    }

    pub fn send_command(&self, command: EmulatorThreadCommand) {
        if matches!(command, EmulatorThreadCommand::Stop) {
            self.audio_device.pause();
        }

        if let Err(err) = self.command_sender.send(command) {
            log::error!("Failed to send command to emulator thread: {err}");
        }
    }

    pub fn swap_chain(&mut self) -> &mut EmulatorSwapChain {
        &mut self.swap_chain
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn render_frame_if_available(
        &mut self,
        surface: &wgpu::Surface<'_>,
    ) -> anyhow::Result<SurfaceRenderEffect> {
        self.surface_renderer.render_frame_if_available(surface)
    }
}

fn load_memory_cards(save_writer: &FsSaveWriter) -> LoadedMemoryCards {
    let slot_1 = fs::read(&save_writer.card_1_path).ok();
    let slot_2 = fs::read(&save_writer.card_2_path).ok();

    LoadedMemoryCards { slot_1, slot_2 }
}

struct EmulatorRunner {
    emulator: Ps1Emulator,
    renderer: SwapChainRenderer,
    audio_output: QueueAudioOutput,
    audio_sync_threshold: u32,
    save_writer: FsSaveWriter,
    inputs: Ps1Inputs,
    disc_path: Option<PathBuf>,
    save_state_path: PathBuf,
    command_receiver: Receiver<EmulatorThreadCommand>,
}

impl EmulatorRunner {
    fn process_next_frame(&mut self) -> Result<(), TickError<Never, Never, io::Error>> {
        while self.emulator.tick(
            self.inputs,
            &mut self.renderer,
            &mut self.audio_output,
            &mut self.save_writer,
        )? != TickEffect::FrameRendered
        {}

        Ok(())
    }
}

fn spawn_emu_thread(memory_card_config: &MemoryCardConfig, mut runner: EmulatorRunner) {
    let memory_card_config = memory_card_config.clone();

    thread::spawn(move || {
        let mut paused = false;
        let mut step_frame = false;
        let mut fast_forward = false;

        let mut memory_card_config = memory_card_config;

        loop {
            if (!paused || step_frame)
                && (fast_forward
                    || (runner.audio_output.samples_len() as u32) < runner.audio_sync_threshold)
            {
                if let Err(err) = runner.process_next_frame() {
                    log::error!("Video/audio/save write error: {err:?}");
                }

                step_frame = false;
            }

            if fast_forward
                && (runner.audio_output.samples_len() as u32) >= 2 * runner.audio_sync_threshold
            {
                runner.audio_output.truncate_front(runner.audio_sync_threshold as usize);
            }

            while let Ok(command) = runner.command_receiver.try_recv() {
                match command {
                    EmulatorThreadCommand::Reset => {
                        runner.emulator.reset();
                    }
                    EmulatorThreadCommand::Stop => {
                        log::info!("Stopping emulator thread");
                        return;
                    }
                    EmulatorThreadCommand::DigitalInput { player, button, pressed } => {
                        update_digital_inputs(&mut runner.inputs, player, button, pressed);
                    }
                    EmulatorThreadCommand::AnalogInput { player, input, value } => {
                        update_analog_inputs(&mut runner.inputs, player, input, value);
                    }
                    EmulatorThreadCommand::ChangeDisc { disc_path } => {
                        try_change_disc(&mut runner.emulator, &disc_path);
                        runner.disc_path = Some(disc_path);

                        update_memcard_config(&memory_card_config, &mut runner);
                    }
                    EmulatorThreadCommand::RemoveDisc => {
                        runner.emulator.change_disc(None);
                        runner.disc_path = None;

                        update_memcard_config(&memory_card_config, &mut runner);
                    }
                    EmulatorThreadCommand::UpdateConfig(config) => {
                        runner.emulator.update_config(config.to_emulator_config());
                        runner.audio_sync_threshold = config.audio.sync_threshold;
                        update_input_config(&config, &mut runner.inputs);

                        if memory_card_config != config.memory_cards {
                            update_memcard_config(&config.memory_cards, &mut runner);
                            memory_card_config = config.memory_cards;
                        }
                    }
                    EmulatorThreadCommand::SaveState => {
                        match save_state(&mut runner.emulator, &runner.save_state_path) {
                            Ok(()) => {
                                log::info!("Saved state to '{}'", runner.save_state_path.display());
                            }
                            Err(err) => {
                                log::error!(
                                    "Error saving state to '{}': {err}",
                                    runner.save_state_path.display()
                                );
                            }
                        }
                    }
                    EmulatorThreadCommand::LoadState => {
                        match load_state(&mut runner.emulator, &runner.save_state_path) {
                            Ok(()) => {
                                log::info!(
                                    "Loaded state from '{}'",
                                    runner.save_state_path.display()
                                );
                            }
                            Err(err) => {
                                log::error!(
                                    "Error loading state from '{}': {err}",
                                    runner.save_state_path.display()
                                );
                            }
                        }
                    }
                    EmulatorThreadCommand::TogglePause => {
                        paused = !paused;
                    }
                    EmulatorThreadCommand::StepFrame => {
                        step_frame = true;
                    }
                    EmulatorThreadCommand::FastForward { enabled } => {
                        fast_forward = enabled;

                        // Clear swap chain when fast forward ends to prevent a temporary input
                        // latency increase due to buffered frames
                        if !fast_forward {
                            runner.renderer.clear_swap_chain();
                        }
                    }
                }
            }

            if !fast_forward {
                thread::sleep(Duration::from_millis(1));
            }
        }
    });
}

fn update_input_config(config: &AppConfig, inputs: &mut Ps1Inputs) {
    inputs.p1.controller_type = config.input.p1_device;
    inputs.p2.controller_type = config.input.p2_device;

    inputs.p1.digital = DigitalJoypadState::default();
    inputs.p2.digital = DigitalJoypadState::default();
    inputs.p1.analog = AnalogJoypadState::default();
    inputs.p2.analog = AnalogJoypadState::default();
}

fn update_memcard_config(config: &MemoryCardConfig, runner: &mut EmulatorRunner) {
    if let Err(err) = runner.save_writer.update_config(runner.disc_path.as_ref(), config) {
        log::error!("Error updating memory card config: {err}");
        return;
    }

    log::info!("Memcard 1 enabled: {}", config.slot_1_enabled);
    log::info!("Memcard 2 enabled: {}", config.slot_2_enabled);

    let memory_cards = load_memory_cards(&runner.save_writer);
    runner.emulator.update_memory_cards(config.cards_enabled(), memory_cards);
}

macro_rules! impl_update_digital_inputs {
    ($inputs:expr, $input_button:expr, $pressed:expr, [$($button:ident => $setter:ident),* $(,)?]) => {
        match $input_button {
            $(
                Ps1Button::$button => $inputs.digital.$setter($pressed),
            )*
            Ps1Button::Analog => $inputs.analog.analog_button = $pressed,
            Ps1Button::L3 => $inputs.analog.l3 = $pressed,
            Ps1Button::R3 => $inputs.analog.r3 = $pressed,
        }
    }
}

fn update_digital_inputs(inputs: &mut Ps1Inputs, player: Player, button: Ps1Button, pressed: bool) {
    let player_inputs = match player {
        Player::One => &mut inputs.p1,
        Player::Two => &mut inputs.p2,
    };

    impl_update_digital_inputs!(player_inputs, button, pressed, [
        Up => set_up,
        Down => set_down,
        Left => set_left,
        Right => set_right,
        Cross => set_cross,
        Circle => set_circle,
        Square => set_square,
        Triangle => set_triangle,
        L1 => set_l1,
        L2 => set_l2,
        R1 => set_r1,
        R2 => set_r2,
        Start => set_start,
        Select => set_select,
    ]);
}

fn update_analog_inputs(inputs: &mut Ps1Inputs, player: Player, input: Ps1AnalogInput, value: i16) {
    let player_inputs = match player {
        Player::One => &mut inputs.p1,
        Player::Two => &mut inputs.p2,
    };

    // Map from [-32768, 32767] to [0, 255]
    let converted_value = ((i32::from(value) + 0x8000) >> 8) as u8;
    match input {
        Ps1AnalogInput::LeftStickX => player_inputs.analog.left_x = converted_value,
        Ps1AnalogInput::LeftStickY => player_inputs.analog.left_y = converted_value,
        Ps1AnalogInput::RightStickX => player_inputs.analog.right_x = converted_value,
        Ps1AnalogInput::RightStickY => player_inputs.analog.right_y = converted_value,
    }
}

fn try_change_disc(emulator: &mut Ps1Emulator, disc_path: &Path) {
    let Some(extension) = disc_path.extension().and_then(OsStr::to_str) else {
        log::error!("Unable to determine file extension of disc path '{}'", disc_path.display());
        return;
    };

    let format = match extension.to_ascii_lowercase().as_str() {
        "chd" => CdRomFileFormat::Chd,
        "cue" => CdRomFileFormat::CueBin,
        _ => {
            log::error!("Unsupported disc file extension '{extension}'");
            return;
        }
    };

    let disc = match CdRom::open(disc_path, format) {
        Ok(disc) => disc,
        Err(err) => {
            log::error!("Error opening disc at '{}': {err}", disc_path.display());
            return;
        }
    };

    emulator.change_disc(Some(disc));
}

macro_rules! bincode_config {
    () => {
        bincode::config::standard()
            .with_little_endian()
            .with_fixed_int_encoding()
            .with_limit::<1_000_000_000>()
    };
}

fn save_state(emulator: &mut Ps1Emulator, path: &Path) -> anyhow::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    bincode::encode_into_std_write(emulator.save_state(), &mut writer, bincode_config!())?;

    Ok(())
}

fn load_state(emulator: &mut Ps1Emulator, path: &Path) -> anyhow::Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let state: Ps1EmulatorState = bincode::decode_from_std_read(&mut reader, bincode_config!())?;

    *emulator = Ps1Emulator::from_state(state, emulator.take_unserialized_fields());

    Ok(())
}

const SAVE_STATES_DIRECTORY: &str = "states";

struct FsSaveWriter {
    card_1_path: PathBuf,
    card_2_path: PathBuf,
}

impl FsSaveWriter {
    fn new(disc_path: Option<&Path>, config: &MemoryCardConfig) -> anyhow::Result<Self> {
        let card_1_path = config.slot_1_path(disc_path);
        let card_2_path = config.slot_2_path(disc_path);

        ensure_parent_dir_exists(&card_1_path)?;
        ensure_parent_dir_exists(&card_2_path)?;

        log::info!("Memcard 1 path set to '{}'", card_1_path.display());
        log::info!("Memcard 2 path set to '{}'", card_2_path.display());

        Ok(Self { card_1_path, card_2_path })
    }

    fn update_config<P: AsRef<Path> + Copy>(
        &mut self,
        disc_path: Option<P>,
        config: &MemoryCardConfig,
    ) -> anyhow::Result<()> {
        self.card_1_path = config.slot_1_path(disc_path);
        self.card_2_path = config.slot_2_path(disc_path);

        ensure_parent_dir_exists(&self.card_1_path)?;
        ensure_parent_dir_exists(&self.card_2_path)?;

        log::info!("Changed memcard 1 path to '{}'", self.card_1_path.display());
        log::info!("Changed memcard 2 path to '{}'", self.card_2_path.display());

        Ok(())
    }
}

fn ensure_parent_dir_exists(path: &Path) -> anyhow::Result<()> {
    let Some(parent) = path.parent() else { return Ok(()) };

    if !parent.exists() {
        fs::create_dir_all(parent)?;
    }

    Ok(())
}

impl SaveWriter for FsSaveWriter {
    type Err = io::Error;

    fn save_memory_card(
        &mut self,
        slot: MemoryCardSlot,
        card_data: &[u8],
    ) -> Result<(), Self::Err> {
        let path = match slot {
            MemoryCardSlot::One => &self.card_1_path,
            MemoryCardSlot::Two => &self.card_2_path,
        };

        let temp_path = path.with_extension("mcdtmp");
        fs::write(&temp_path, card_data)?;
        fs::rename(temp_path, path)?;

        log::debug!("Saved memory card 1 to {}", path.display());

        Ok(())
    }
}

fn determine_save_state_path(file_path: Option<&Path>) -> anyhow::Result<PathBuf> {
    let path_no_ext = file_path.unwrap_or(&PathBuf::from("bios")).with_extension("");
    let file_name_no_ext = path_no_ext.file_name().and_then(OsStr::to_str).ok_or_else(|| {
        anyhow!("Unable to determine file extension for path: {}", path_no_ext.display())
    })?;

    let state_file_name = format!("{file_name_no_ext}.sst");
    let state_path = PathBuf::from(SAVE_STATES_DIRECTORY).join(state_file_name);

    ensure_parent_dir_exists(&state_path)?;

    Ok(state_path)
}
