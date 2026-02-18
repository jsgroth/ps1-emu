//! PS1 public interface and main loop

use crate::bus::Bus;
use crate::cd::{CdController, CdControllerState};
use crate::cpu::R3000;
use crate::dma::{DmaContext, DmaController};
use crate::gpu::Gpu;
use crate::gpu::GpuState;
use crate::input::Ps1Inputs;
use crate::interrupts::{InterruptRegisters, InterruptType};
use crate::mdec::MacroblockDecoder;
use crate::memory::{Memory, MemoryControl};
use crate::scheduler::{Scheduler, SchedulerEvent, SchedulerEventType};
use crate::sio::{SerialPort0, SerialPort1};
use crate::spu::Spu;
use crate::timers::Timers;
use bincode::{Decode, Encode};
use cdrom::CdRomError;
use cdrom::reader::CdRom;
use proc_macros::SaveState;
use std::fmt::{Display, Formatter};
use std::num::NonZeroU32;
use std::sync::Arc;
use thiserror::Error;

pub use crate::gpu::DisplayConfig;
pub use crate::pgxp::PgxpConfig;
use crate::sio::memcard::MemoryCard;

pub const DEFAULT_AUDIO_BUFFER_SIZE: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum ColorDepthBits {
    #[default]
    Fifteen = 0,
    TwentyFour = 1,
}

impl Display for ColorDepthBits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fifteen => write!(f, "15-bit"),
            Self::TwentyFour => write!(f, "24-bit"),
        }
    }
}

impl ColorDepthBits {
    pub(crate) fn from_bit(bit: bool) -> Self {
        if bit { Self::TwentyFour } else { Self::Fifteen }
    }
}

pub trait Renderer {
    type Err;

    /// # Errors
    ///
    /// Should propagate any error encountered while rendering the frame.
    fn render_frame(
        &mut self,
        command_buffers: impl Iterator<Item = wgpu::CommandBuffer>,
        frame: &wgpu::Texture,
        pixel_aspect_ratio: f64,
    ) -> Result<(), Self::Err>;
}

pub trait AudioOutput {
    type Err;

    /// # Errors
    ///
    /// Should propagate any error encountered while queueing the samples.
    fn queue_samples(&mut self, samples: &[(i16, i16)]) -> Result<(), Self::Err>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCardSlot {
    One = 1,
    Two = 2,
}

pub trait SaveWriter {
    type Err;

    /// # Errors
    ///
    /// Should propagate any error encountered while persisting the memory card.
    fn save_memory_card(&mut self, slot: MemoryCardSlot, card_data: &[u8])
    -> Result<(), Self::Err>;
}

#[derive(Debug, Error)]
pub enum Ps1Error {
    #[error("Incorrect BIOS ROM size; expected 512KB, was {bios_len}")]
    IncorrectBiosSize { bios_len: usize },
    #[error("EXE format is invalid")]
    InvalidExeFormat,
}

pub type Ps1Result<T> = Result<T, Ps1Error>;

#[derive(Debug, Error)]
pub enum TickError<RErr, AErr, SErr> {
    #[error("Error rendering frame: {0}")]
    Render(RErr),
    #[error("Error queueing audio samples: {0}")]
    Audio(AErr),
    #[error("Error saving memory card: {0}")]
    SaveWrite(SErr),
    #[error("CD-ROM error: {0}")]
    CdRom(#[from] CdRomError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AdpcmInterpolation {
    #[default]
    Gaussian,
    Hermite,
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct Ps1EmulatorConfig {
    pub display: DisplayConfig,
    pub pgxp: PgxpConfig,
    pub adpcm_interpolation: AdpcmInterpolation,
    pub internal_audio_buffer_size: NonZeroU32,
    pub tty_enabled: bool,
}

impl Default for Ps1EmulatorConfig {
    fn default() -> Self {
        Self {
            display: DisplayConfig::default(),
            pgxp: PgxpConfig::default(),
            adpcm_interpolation: AdpcmInterpolation::default(),
            internal_audio_buffer_size: NonZeroU32::new(DEFAULT_AUDIO_BUFFER_SIZE).unwrap(),
            tty_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryCardsEnabled {
    pub slot_1: bool,
    pub slot_2: bool,
}

impl Default for MemoryCardsEnabled {
    fn default() -> Self {
        Self { slot_1: true, slot_2: false }
    }
}

#[derive(Debug, Clone)]
pub struct LoadedMemoryCards {
    pub slot_1: Option<Vec<u8>>,
    pub slot_2: Option<Vec<u8>>,
}

pub struct UnserializedFields {
    disc: Option<CdRom>,
    memory_cards: LoadedMemoryCards,
    wgpu_device: Arc<wgpu::Device>,
    wgpu_queue: Arc<wgpu::Queue>,
    config: Ps1EmulatorConfig,
    memory_cards_enabled: MemoryCardsEnabled,
}

#[derive(SaveState)]
pub struct Ps1Emulator {
    cpu: R3000,
    #[save_state(to = GpuState)]
    gpu: Gpu,
    spu: Spu,
    audio_buffer: Vec<(i16, i16)>,
    #[save_state(to = CdControllerState)]
    cd_controller: CdController,
    mdec: MacroblockDecoder,
    memory: Memory,
    memory_control: MemoryControl,
    dma_controller: DmaController,
    interrupt_registers: InterruptRegisters,
    sio0: SerialPort0,
    sio1: SerialPort1,
    timers: Timers,
    scheduler: Scheduler,
    last_render_cycles: u64,
    #[save_state(skip)]
    config: Ps1EmulatorConfig,
    tty_buffer: String,
}

#[derive(Debug)]
pub struct Ps1EmulatorBuilder {
    bios_rom: Vec<u8>,
    wgpu_device: Arc<wgpu::Device>,
    wgpu_queue: Arc<wgpu::Queue>,
    config: Ps1EmulatorConfig,
    memory_cards_enabled: MemoryCardsEnabled,
    loaded_memory_cards: Option<LoadedMemoryCards>,
    disc: Option<CdRom>,
}

impl Ps1EmulatorBuilder {
    #[must_use]
    pub fn new(
        bios_rom: Vec<u8>,
        wgpu_device: Arc<wgpu::Device>,
        wgpu_queue: Arc<wgpu::Queue>,
    ) -> Self {
        Self {
            bios_rom,
            wgpu_device,
            wgpu_queue,
            config: Ps1EmulatorConfig::default(),
            memory_cards_enabled: MemoryCardsEnabled::default(),
            loaded_memory_cards: None,
            disc: None,
        }
    }

    #[must_use]
    pub fn with_disc(mut self, disc: CdRom) -> Self {
        self.disc = Some(disc);
        self
    }

    #[must_use]
    pub fn with_memory_cards_enabled(mut self, memory_cards_enabled: MemoryCardsEnabled) -> Self {
        self.memory_cards_enabled = memory_cards_enabled;
        self
    }

    #[must_use]
    pub fn with_memory_cards(mut self, loaded_memory_cards: LoadedMemoryCards) -> Self {
        self.loaded_memory_cards = Some(loaded_memory_cards);
        self
    }

    #[must_use]
    pub fn with_config(mut self, config: Ps1EmulatorConfig) -> Self {
        self.config = config;
        self
    }

    /// # Errors
    ///
    /// Will return an error if the BIOS ROM is invalid.
    pub fn build(self) -> Ps1Result<Ps1Emulator> {
        Ps1Emulator::new(
            self.bios_rom,
            self.wgpu_device,
            self.wgpu_queue,
            self.config,
            self.memory_cards_enabled,
            self.loaded_memory_cards.unwrap_or(LoadedMemoryCards { slot_1: None, slot_2: None }),
            self.disc,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickEffect {
    None,
    FrameRendered,
}

// The SPU/CD-ROM clock rate is exactly 1/768 the CPU clock rate
// This _should_ be 44100 Hz, but it may not be exactly depending on the exact oscillator speed
const SPU_CLOCK_DIVIDER: u64 = 768;

macro_rules! new_bus {
    ($self:expr) => {
        Bus {
            gpu: &mut $self.gpu,
            spu: &mut $self.spu,
            cd_controller: &mut $self.cd_controller,
            mdec: &mut $self.mdec,
            memory: &mut $self.memory,
            memory_control: &mut $self.memory_control,
            dma_controller: &mut $self.dma_controller,
            interrupt_registers: &mut $self.interrupt_registers,
            sio0: &mut $self.sio0,
            sio1: &mut $self.sio1,
            timers: &mut $self.timers,
            scheduler: &mut $self.scheduler,
        }
    };
}

macro_rules! new_dma_ctx {
    ($self:expr) => {
        DmaContext {
            memory: &mut $self.memory,
            gpu: &mut $self.gpu,
            spu: &mut $self.spu,
            mdec: &mut $self.mdec,
            cd_controller: &mut $self.cd_controller,
            scheduler: &mut $self.scheduler,
            interrupt_registers: &mut $self.interrupt_registers,
        }
    };
}

impl Ps1Emulator {
    /// # Errors
    ///
    /// Will return an error if the BIOS ROM is invalid.
    pub fn new(
        bios_rom: Vec<u8>,
        wgpu_device: Arc<wgpu::Device>,
        wgpu_queue: Arc<wgpu::Queue>,
        config: Ps1EmulatorConfig,
        memory_cards_enabled: MemoryCardsEnabled,
        loaded_memory_cards: LoadedMemoryCards,
        disc: Option<CdRom>,
    ) -> Ps1Result<Self> {
        let memory = Memory::new(bios_rom)?;

        let mut emulator = Self {
            cpu: R3000::new(config.pgxp),
            gpu: Gpu::new(wgpu_device, wgpu_queue, config.display, config.pgxp),
            spu: Spu::new(config.adpcm_interpolation),
            audio_buffer: Vec::with_capacity(1600),
            cd_controller: CdController::new(disc),
            mdec: MacroblockDecoder::new(),
            memory,
            memory_control: MemoryControl::new(),
            dma_controller: DmaController::new(config.pgxp),
            interrupt_registers: InterruptRegisters::new(),
            sio0: SerialPort0::new_sio0(memory_cards_enabled, loaded_memory_cards),
            sio1: SerialPort1::new_sio1(),
            timers: Timers::new(),
            scheduler: Scheduler::new(),
            last_render_cycles: 0,
            config,
            tty_buffer: String::new(),
        };
        emulator.schedule_initial_events();

        Ok(emulator)
    }

    #[allow(clippy::missing_panics_doc)]
    pub fn reset(&mut self) {
        let bios_rom = self.memory.clone_bios_rom();
        let unserialized = self.take_unserialized_fields();

        *self = Ps1Emulator::new(
            bios_rom,
            unserialized.wgpu_device,
            unserialized.wgpu_queue,
            self.config,
            unserialized.memory_cards_enabled,
            unserialized.memory_cards,
            unserialized.disc,
        )
        .expect("Emulator creation during reset should never fail");
    }

    fn schedule_initial_events(&mut self) {
        self.timers.schedule_next_vblank(&mut self.scheduler, &mut self.interrupt_registers);
        self.scheduler.update_or_push_event(SchedulerEvent::spu_and_cd_clock(SPU_CLOCK_DIVIDER));
    }

    #[inline]
    #[must_use]
    pub fn cpu_pc(&self) -> u32 {
        self.cpu.pc()
    }

    /// # Errors
    ///
    /// Will return an error if the EXE does not appear to be a PS1 executable based on the header.
    pub fn run_until_exe_sideloaded(&mut self, exe: &[u8]) -> Ps1Result<()> {
        let mut bus = new_bus!(self);
        while self.cpu.pc() != 0x80030000 {
            let _ = self.cpu.execute_instruction(&mut bus);
        }

        self.sideload_exe(exe)
    }

    /// # Errors
    ///
    /// Will return an error if the EXE does not appear to be a PS1 executable based on the header.
    #[allow(clippy::missing_panics_doc)]
    pub fn sideload_exe(&mut self, exe: &[u8]) -> Ps1Result<()> {
        if exe.len() < 0x800 || &exe[..0x008] != "PS-X EXE".as_bytes() {
            return Err(Ps1Error::InvalidExeFormat);
        }

        let pc = u32::from_le_bytes(exe[0x010..0x014].try_into().unwrap());
        let initial_gp = u32::from_le_bytes(exe[0x014..0x018].try_into().unwrap());
        let ram_dest_addr = u32::from_le_bytes(exe[0x018..0x01C].try_into().unwrap());
        let exe_size = u32::from_le_bytes(exe[0x01C..0x020].try_into().unwrap());
        let initial_sp = u32::from_le_bytes(exe[0x030..0x034].try_into().unwrap());
        let initial_sp_offset = u32::from_le_bytes(exe[0x034..0x038].try_into().unwrap());

        self.cpu.set_pc(pc);
        self.cpu.set_gpr(28, initial_gp);

        if initial_sp != 0 {
            self.cpu.set_gpr(29, initial_sp);
            self.cpu.set_gpr(30, initial_sp);
        }

        if initial_sp_offset != 0 {
            for r in [29, 30] {
                let r_value = self.cpu.get_gpr(r);
                self.cpu.set_gpr(r, r_value.wrapping_add(initial_sp_offset));
            }
        }

        let exe_data = &exe[0x800..0x800 + exe_size as usize];
        self.memory.copy_to_main_ram(exe_data, ram_dest_addr & 0x1FFFFFFF);

        Ok(())
    }

    /// # Errors
    ///
    /// Will propagate any error encountered while rendering a frame.
    #[inline]
    #[allow(clippy::type_complexity)]
    pub fn tick<R: Renderer, A: AudioOutput, S: SaveWriter>(
        &mut self,
        inputs: Ps1Inputs,
        renderer: &mut R,
        audio_output: &mut A,
        save_writer: &mut S,
    ) -> Result<TickEffect, TickError<R::Err, A::Err, S::Err>> {
        self.sio0.set_inputs(inputs);

        if self.dma_controller.cpu_wait_cycles() != 0 {
            // TODO the CPU can run in parallel to a DMA as long as it doesn't access main RAM
            // or an I/O register
            let cycles = self.dma_controller.take_cpu_wait_cycles();
            self.scheduler.increment_cpu_cycles(cycles.into());
        }

        let mut bus = new_bus!(self);
        while !bus.scheduler.is_event_ready() {
            let cycles = self.cpu.execute_instruction(&mut bus);
            bus.scheduler.increment_cpu_cycles(cycles.into());

            if self.config.tty_enabled {
                check_for_putchar_call(&self.cpu, &mut self.tty_buffer);
            }
        }

        let tick_effect = if self.scheduler.is_event_ready() {
            self.process_scheduler_events(renderer, audio_output, save_writer)?
        } else {
            TickEffect::None
        };

        if self.scheduler.cpu_cycle_counter() - self.last_render_cycles >= 33_868_800 / 30 {
            // Force a frame render
            // TODO handle this with the scheduler if the GPU stops generating VBlank IRQs due to
            // invalid Y1/Y2
            self.render_frame(renderer, audio_output, save_writer)?;
            return Ok(TickEffect::FrameRendered);
        }

        Ok(tick_effect)
    }

    #[allow(clippy::type_complexity)]
    fn render_frame<R: Renderer, A: AudioOutput, S: SaveWriter>(
        &mut self,
        renderer: &mut R,
        audio_output: &mut A,
        save_writer: &mut S,
    ) -> Result<(), TickError<R::Err, A::Err, S::Err>> {
        self.last_render_cycles = self.scheduler.cpu_cycle_counter();

        let pixel_aspect_ratio = self.gpu.pixel_aspect_ratio();
        let (frame, command_buffers) = self.gpu.generate_frame_texture();
        renderer
            .render_frame(command_buffers, frame, pixel_aspect_ratio)
            .map_err(TickError::Render)?;

        self.drain_audio_samples(audio_output).map_err(TickError::Audio)?;

        let (memory_card_1, memory_card_2) = self.sio0.memory_cards();
        save_memory_card(MemoryCardSlot::One, memory_card_1, save_writer)
            .map_err(TickError::SaveWrite)?;
        save_memory_card(MemoryCardSlot::Two, memory_card_2, save_writer)
            .map_err(TickError::SaveWrite)?;

        Ok(())
    }

    fn drain_audio_samples<A: AudioOutput>(&mut self, audio_output: &mut A) -> Result<(), A::Err> {
        audio_output.queue_samples(&self.audio_buffer)?;
        self.audio_buffer.clear();

        Ok(())
    }

    #[inline]
    #[allow(clippy::type_complexity)]
    fn process_scheduler_events<R: Renderer, A: AudioOutput, S: SaveWriter>(
        &mut self,
        renderer: &mut R,
        audio_output: &mut A,
        save_writer: &mut S,
    ) -> Result<TickEffect, TickError<R::Err, A::Err, S::Err>> {
        let mut tick_effect = TickEffect::None;

        while let Some(event) = self.scheduler.pop_ready_event() {
            match event.event_type {
                SchedulerEventType::VBlank => {
                    // VBlank event: Generate VBlank IRQ and render the current display frame buffer
                    // to video output.
                    // Triggers once per frame (when scanline == Y2) unless the GPU's vertical
                    // display range is invalid
                    self.interrupt_registers.set_interrupt_flag(InterruptType::VBlank);
                    self.timers
                        .schedule_next_vblank(&mut self.scheduler, &mut self.interrupt_registers);

                    self.sio0.catch_up(&mut self.scheduler, &mut self.interrupt_registers);
                    self.sio1.catch_up(&mut self.scheduler, &mut self.interrupt_registers);

                    self.render_frame(renderer, audio_output, save_writer)?;

                    tick_effect = TickEffect::FrameRendered;
                }
                SchedulerEventType::SpuAndCdClock => {
                    // SPU/CD-ROM clock event: Clock the CD-ROM controller and the SPU, then push
                    // the current stereo audio sample to audio output.
                    // Triggers every 768 CPU clocks which is 44100 Hz
                    self.cd_controller.clock(&mut self.interrupt_registers)?;
                    self.audio_buffer
                        .push(self.spu.clock(&self.cd_controller, &mut self.interrupt_registers));

                    if (self.audio_buffer.len() as u32)
                        >= self.config.internal_audio_buffer_size.get()
                    {
                        self.drain_audio_samples(audio_output).map_err(TickError::Audio)?;
                    }

                    self.scheduler.update_or_push_event(SchedulerEvent::spu_and_cd_clock(
                        event.cpu_cycles + SPU_CLOCK_DIVIDER,
                    ));
                }
                SchedulerEventType::ProcessDma => {
                    // Process the highest-priority active DMA that is ready to transfer
                    self.dma_controller.process(new_dma_ctx!(self));
                }
                SchedulerEventType::Timer0Irq
                | SchedulerEventType::Timer1Irq
                | SchedulerEventType::Timer2Irq => {
                    self.timers.catch_up(&mut self.scheduler, &mut self.interrupt_registers);
                    self.timers
                        .schedule_timer_events(&mut self.scheduler, &mut self.interrupt_registers);
                }
                SchedulerEventType::Sio0Irq | SchedulerEventType::Sio0Tx => {
                    self.sio0.catch_up(&mut self.scheduler, &mut self.interrupt_registers);
                }
                SchedulerEventType::Sio1Irq | SchedulerEventType::Sio1Tx => {
                    self.sio1.catch_up(&mut self.scheduler, &mut self.interrupt_registers);
                }
            }
        }

        Ok(tick_effect)
    }

    pub fn change_disc(&mut self, disc: Option<CdRom>) {
        self.cd_controller.change_disc(disc);
    }

    pub fn update_config(&mut self, config: Ps1EmulatorConfig) {
        self.cpu.update_pgxp_config(config.pgxp);
        self.dma_controller.update_pgxp_config(config.pgxp);
        self.gpu.update_config(config.display, config.pgxp);
        self.spu.update_adpcm_interpolation(config.adpcm_interpolation);
        self.config = config;
    }

    pub fn update_memory_cards(&mut self, enabled: MemoryCardsEnabled, loaded: LoadedMemoryCards) {
        self.sio0.update_memory_cards(enabled, loaded);
    }

    #[must_use]
    pub fn take_unserialized_fields(&mut self) -> UnserializedFields {
        let (wgpu_device, wgpu_queue) = self.gpu.get_wgpu_resources();
        let (memory_cards_enabled, memory_cards) = self.sio0.clone_unserialized_fields();

        UnserializedFields {
            disc: self.cd_controller.take_disc(),
            memory_cards,
            wgpu_device,
            wgpu_queue,
            config: self.config,
            memory_cards_enabled,
        }
    }

    pub fn from_state(mut state: Ps1EmulatorState, unserialized: UnserializedFields) -> Self {
        // Don't load memory cards from save states
        state
            .sio0
            .update_memory_cards(unserialized.memory_cards_enabled, unserialized.memory_cards);

        let mut emulator = Self {
            cpu: state.cpu,
            gpu: Gpu::from_state(
                state.gpu,
                unserialized.wgpu_device,
                unserialized.wgpu_queue,
                unserialized.config.display,
            ),
            spu: state.spu,
            audio_buffer: state.audio_buffer,
            cd_controller: CdController::from_state(state.cd_controller, unserialized.disc),
            mdec: state.mdec,
            memory: state.memory,
            memory_control: state.memory_control,
            dma_controller: state.dma_controller,
            interrupt_registers: state.interrupt_registers,
            sio0: state.sio0,
            sio1: state.sio1,
            timers: state.timers,
            scheduler: state.scheduler,
            last_render_cycles: state.last_render_cycles,
            config: unserialized.config,
            tty_buffer: state.tty_buffer,
        };

        emulator.update_config(unserialized.config);

        emulator
    }
}

fn save_memory_card<S: SaveWriter>(
    slot: MemoryCardSlot,
    card: Option<&mut MemoryCard>,
    save_writer: &mut S,
) -> Result<(), S::Err> {
    let Some(card) = card else { return Ok(()) };

    if !card.get_and_clear_dirty() {
        // Data has not changed since last save write
        return Ok(());
    }

    save_writer.save_memory_card(slot, card.data())
}

fn check_for_putchar_call(cpu: &R3000, tty_buffer: &mut String) {
    // BIOS function calls work by jumping to $A0 (A functions), $B0 (B functions), or
    // $C0 (C functions) with the function number specified in R9.
    //
    // A($3C) and B($3D) are both the putchar() function, which prints the ASCII character
    // in R4 to the TTY.
    let pc = cpu.pc() & 0x1FFFFFFF;
    let r9 = cpu.get_gpr(9);
    if (pc == 0xA0 && r9 == 0x3C) || (pc == 0xB0 && r9 == 0x3D) {
        let r4 = cpu.get_gpr(4);
        let c = r4 as u8 as char;
        if c == '\n' {
            println!("TTY: {tty_buffer}");
            tty_buffer.clear();
        } else {
            tty_buffer.push(c);
        }
    }
}
