//! PS1 CD-ROM controller
//!
//! The controller is emulated as a whole instead of separately emulating the drive, the 68HC05, and
//! the DSP at a low level. This works because of the restricted interface exposed to the CPU and DMA.

mod audio;
mod control;
mod fifo;
mod macros;
mod read;
mod response;
mod seek;
mod status;
mod xaadpcm;

use crate::cd::audio::PlayState;
use crate::cd::control::DriveMode;
use crate::cd::fifo::{DataFifo, ParameterFifo};
use crate::cd::read::ReadState;
use crate::cd::xaadpcm::XaAdpcmState;
use crate::interrupts::{InterruptRegisters, InterruptType};
use crate::num::U8Ext;
use bincode::{Decode, Encode};
use cdrom::CdRomResult;
use cdrom::cdtime::CdTime;
use cdrom::reader::CdRom;
#[allow(clippy::wildcard_imports)]
use macros::*;
use proc_macros::SaveState;
use std::{array, cmp};

// Roughly 23,796 CPU cycles
const RECEIVE_COMMAND_CYCLES_STOPPED: u32 = 31;

// Roughly 50,401 CPU cycles
const RECEIVE_COMMAND_CYCLES_RUNNING: u32 = 65;

// Roughly 81,102 CPU cycles
const INIT_COMMAND_CYCLES: u32 = 105;

// Roughly half a second
// TODO is this too fast?
const SPIN_UP_CYCLES: u32 = 22_050;

#[derive(Debug, Clone, Copy, Encode, Decode)]
struct CdInterruptRegisters {
    enabled: u8,
    flags: u8,
    prev_pending: bool,
}

impl CdInterruptRegisters {
    fn new() -> Self {
        Self { enabled: 0, flags: 0, prev_pending: false }
    }

    // Used for the CD-ROM IRQ bit in I_STAT
    fn pending(self) -> bool {
        self.enabled & self.flags != 0
    }

    // Used to determine whether an interrupt is currently "queued"
    // Spyro 3 depends on interrupts "queueing" when flags are set even if the interrupts are not enabled
    fn int_queued(self) -> bool {
        self.flags & 7 != 0
    }

    fn read_enabled(self) -> u8 {
        // Bits 5-7 apparently always read as 1?
        let enabled = 0xE0 | self.enabled;
        log::debug!("Interrupts enabled read: {enabled:02X}");
        enabled
    }

    fn read_flags(self) -> u8 {
        // Bits 5-7 apparently always read as 1?
        let flags = 0xE0 | self.flags;
        log::trace!("Interrupt flags read: {flags:02X}");
        flags
    }

    fn write_enabled(&mut self, value: u8) {
        self.enabled = value & 0x1F;
        if !self.pending() {
            // Necessary to prevent a freeze if another interrupt is generated on the next clock
            // cycle
            self.prev_pending = false;
        }

        log::debug!("Interrupts enabled write: {:02X}", self.enabled);
    }

    fn write_flags(&mut self, value: u8, parameter_fifo: &mut ParameterFifo) {
        // Bits 0-4 acknowledge interrupts if set
        self.flags &= !(value & 0x1F);
        if !self.pending() {
            // Necessary to prevent a freeze if another interrupt is generated on the next clock
            // cycle
            self.prev_pending = false;
        }

        log::debug!("Acknowledged CD-ROM interrupts: {:02X}", value & 0x1F);

        // Bit 6 resets the parameter FIFO if set
        if value.bit(6) {
            parameter_fifo.reset();
            log::debug!("  Reset parameter FIFO");
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
enum Command {
    Demute,
    GetId,
    GetLocL,
    GetLocP,
    GetStat,
    GetTD,
    GetTN,
    Init,
    MotorOn,
    Mute,
    Pause,
    Play,
    ReadN,
    ReadS,
    ReadToc,
    SeekL,
    SeekP,
    SetFilter,
    SetLoc,
    SetMode,
    Stop,
    Test,
}

#[derive(Debug, Clone, Copy, Encode, Decode, Default)]
enum CommandState {
    #[default]
    Idle,
    CommandQueued {
        command: Command,
        cycles: u32,
    },
    ReceivingCommand {
        command: Command,
        cycles_remaining: u32,
    },
    GeneratingSecondResponse {
        command: Command,
        cycles_remaining: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
enum SeekNextState {
    Pause,
    Read,
    Play,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
enum SpinUpNextState {
    Pause,
    Seek(CdTime, SeekNextState),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode, Default)]
enum DriveState {
    #[default]
    Stopped,
    SpinningUp {
        cycles_remaining: u32,
        next: SpinUpNextState,
    },
    Seeking {
        destination: CdTime,
        cycles_remaining: u32,
        next: SeekNextState,
    },
    PreparingToRead {
        time: CdTime,
        cycles_remaining: u32,
    },
    Reading(ReadState),
    PreparingToPlay {
        time: CdTime,
        cycles_remaining: u32,
    },
    Playing(PlayState),
    Paused {
        time: CdTime,
        int2_queued: bool,
    },
}

impl DriveState {
    fn current_time(self) -> CdTime {
        match self {
            Self::Stopped | Self::SpinningUp { .. } => CdTime::ZERO,
            Self::Paused { time, .. }
            | Self::PreparingToRead { time, .. }
            | Self::Reading(ReadState { time, .. })
            | Self::PreparingToPlay { time, .. }
            | Self::Playing(PlayState { time, .. })
            | Self::Seeking { destination: time, .. } => time,
        }
    }

    fn is_stopped_or_spinning_up(self) -> bool {
        matches!(self, Self::Stopped | Self::SpinningUp { .. })
    }
}

const BYTES_PER_SECTOR: usize = 2352;

type SectorBuffer = [u8; BYTES_PER_SECTOR];

#[derive(Debug, SaveState)]
pub struct CdController {
    index: u8,
    #[save_state(skip)]
    disc: Option<CdRom>,
    interrupts: CdInterruptRegisters,
    parameter_fifo: ParameterFifo,
    response_fifo: ParameterFifo,
    data_fifo: DataFifo,
    sector_buffer: Box<SectorBuffer>,
    command_state: CommandState,
    drive_state: DriveState,
    drive_mode: DriveMode,
    seek_location: Option<CdTime>,
    scex_read: bool,
    audio_muted: bool,
    shell_opened: bool,
    current_audio_sample: (i16, i16),
    cd_to_spu_volume: [[u8; 2]; 2],
    next_cd_to_spu_volume: [[u8; 2]; 2],
    xa_adpcm: XaAdpcmState,
}

impl CdController {
    pub fn new(disc: Option<CdRom>) -> Self {
        // Pretend the SCEx region code was always read if there's a disc in the drive
        let scex_read = disc.is_some();

        Self {
            index: 0,
            disc,
            interrupts: CdInterruptRegisters::new(),
            parameter_fifo: ParameterFifo::new(),
            response_fifo: ParameterFifo::new(),
            data_fifo: DataFifo::new(),
            sector_buffer: Box::new(array::from_fn(|_| 0)),
            command_state: CommandState::default(),
            drive_state: DriveState::default(),
            drive_mode: DriveMode::new(),
            seek_location: None,
            scex_read,
            audio_muted: false,
            shell_opened: false,
            current_audio_sample: (0, 0),
            cd_to_spu_volume: [[0; 2]; 2],
            next_cd_to_spu_volume: [[0; 2]; 2],
            xa_adpcm: XaAdpcmState::new(),
        }
    }

    pub fn from_state(state: CdControllerState, disc: Option<CdRom>) -> Self {
        Self {
            index: state.index,
            disc,
            interrupts: state.interrupts,
            parameter_fifo: state.parameter_fifo,
            response_fifo: state.response_fifo,
            data_fifo: state.data_fifo,
            sector_buffer: state.sector_buffer,
            command_state: state.command_state,
            drive_state: state.drive_state,
            drive_mode: state.drive_mode,
            seek_location: state.seek_location,
            scex_read: state.scex_read,
            audio_muted: state.audio_muted,
            shell_opened: state.shell_opened,
            current_audio_sample: state.current_audio_sample,
            cd_to_spu_volume: state.cd_to_spu_volume,
            next_cd_to_spu_volume: state.next_cd_to_spu_volume,
            xa_adpcm: state.xa_adpcm,
        }
    }

    // 44100 Hz clock
    pub fn clock(&mut self, interrupt_registers: &mut InterruptRegisters) -> CdRomResult<()> {
        self.current_audio_sample = (0, 0);

        self.advance_drive_state()?;
        self.advance_command_state();

        let interrupt_pending = self.interrupts.pending();
        if !self.interrupts.prev_pending && interrupt_pending {
            // Flag a CD-ROM interrupt on any 0->1 transition
            // TODO apparently there should be a small delay before the interrupt flag is set in I_STAT?
            interrupt_registers.set_interrupt_flag(InterruptType::CdRom);
            log::debug!("CD-ROM INT{} generated", self.interrupts.enabled & self.interrupts.flags);
        }
        self.interrupts.prev_pending = interrupt_pending;

        Ok(())
    }

    fn advance_drive_state(&mut self) -> CdRomResult<()> {
        self.drive_state = match self.drive_state {
            DriveState::Stopped => DriveState::Stopped,
            DriveState::SpinningUp { cycles_remaining: 1, next: SpinUpNextState::Pause } => {
                log::debug!(
                    "Drive finished spinning up; generating INT2 and pausing at start of disc"
                );
                DriveState::Paused { time: CdTime::ZERO, int2_queued: true }
            }
            DriveState::SpinningUp {
                cycles_remaining: 1,
                next: SpinUpNextState::Seek(time, seek_next),
            } => {
                log::debug!("Drive finished spinning up, now seeking to {time}");
                let seek_cycles =
                    cmp::max(seek::MIN_SEEK_CYCLES, seek::estimate_seek_cycles(CdTime::ZERO, time));
                DriveState::Seeking {
                    destination: time,
                    cycles_remaining: seek_cycles,
                    next: seek_next,
                }
            }
            DriveState::SpinningUp { cycles_remaining, next } => {
                DriveState::SpinningUp { cycles_remaining: cycles_remaining - 1, next }
            }
            DriveState::Seeking {
                destination,
                cycles_remaining: 1,
                next: SeekNextState::Pause,
            } => {
                log::debug!("Drive finished seeking to {destination}; generating INT2 and pausing");
                DriveState::Paused { time: destination, int2_queued: true }
            }
            DriveState::Seeking { destination, cycles_remaining: 1, next: SeekNextState::Read } => {
                log::debug!("Drive finished seeking to {destination}; preparing to read");
                DriveState::PreparingToRead {
                    time: destination,
                    cycles_remaining: 5 * self.drive_mode.speed.cycles_between_sectors(),
                }
            }
            DriveState::Seeking { destination, cycles_remaining: 1, next: SeekNextState::Play } => {
                log::debug!("Drive finished seeking to {destination}; preparing to read");
                DriveState::PreparingToPlay {
                    time: destination,
                    cycles_remaining: 5 * self.drive_mode.speed.cycles_between_sectors(),
                }
            }
            DriveState::Seeking { destination, cycles_remaining, next } => {
                DriveState::Seeking { destination, cycles_remaining: cycles_remaining - 1, next }
            }
            DriveState::PreparingToRead { time, cycles_remaining: 1 } => {
                self.xa_adpcm.clear_buffers();
                self.read_data_sector(time)?
            }
            DriveState::PreparingToRead { time, cycles_remaining } => {
                DriveState::PreparingToRead { time, cycles_remaining: cycles_remaining - 1 }
            }
            DriveState::Reading(state) => self.progress_read_state(state)?,
            DriveState::PreparingToPlay { time, cycles_remaining: 1 } => {
                self.read_audio_sector(PlayState::new(time), true)?
            }
            DriveState::PreparingToPlay { time, cycles_remaining } => {
                DriveState::PreparingToPlay { time, cycles_remaining: cycles_remaining - 1 }
            }
            DriveState::Playing(state) => self.progress_play_state(state)?,
            DriveState::Paused { time, mut int2_queued } => {
                if int2_queued && !self.interrupts.int_queued() {
                    self.int2(&[stat!(self)]);
                    int2_queued = false;
                }

                DriveState::Paused { time, int2_queued }
            }
        };

        Ok(())
    }

    fn advance_command_state(&mut self) {
        self.command_state = match self.command_state {
            CommandState::Idle => CommandState::Idle,
            CommandState::CommandQueued { command, cycles } => {
                if !self.interrupts.int_queued() {
                    CommandState::ReceivingCommand { command, cycles_remaining: cycles }
                } else {
                    // The controller will not acccept a command if any interrupts are queued
                    CommandState::CommandQueued { command, cycles }
                }
            }
            CommandState::ReceivingCommand { command, cycles_remaining: 1 } => {
                self.execute_command(command)
            }
            CommandState::ReceivingCommand { command, cycles_remaining } => {
                CommandState::ReceivingCommand { command, cycles_remaining: cycles_remaining - 1 }
            }
            CommandState::GeneratingSecondResponse { command, cycles_remaining: 1 } => {
                if !self.interrupts.int_queued() {
                    self.generate_second_response(command)
                } else {
                    // If an interrupt is pending, the controller waits until it is cleared
                    CommandState::GeneratingSecondResponse { command, cycles_remaining: 1 }
                }
            }
            CommandState::GeneratingSecondResponse { command, cycles_remaining } => {
                CommandState::GeneratingSecondResponse {
                    command,
                    cycles_remaining: cycles_remaining - 1,
                }
            }
        };
    }

    fn execute_command(&mut self, command: Command) -> CommandState {
        log::debug!("Executing command {command:?}");

        let new_state = match command {
            Command::Demute => self.execute_demute(),
            Command::GetId => self.execute_get_id(),
            Command::GetLocL => self.execute_get_loc_l(),
            Command::GetLocP => self.execute_get_loc_p(),
            Command::GetStat => self.execute_get_stat(),
            Command::GetTD => self.execute_get_td(),
            Command::GetTN => self.execute_get_tn(),
            Command::Init => self.execute_init(),
            Command::MotorOn => self.execute_motor_on(),
            Command::Mute => self.execute_mute(),
            Command::Pause => self.execute_pause(),
            Command::Play => self.execute_play(),
            Command::ReadN | Command::ReadS => self.execute_read(),
            Command::ReadToc => self.execute_read_toc(),
            Command::SeekL | Command::SeekP => self.execute_seek(),
            Command::SetFilter => self.execute_set_filter(),
            Command::SetLoc => self.execute_set_loc(),
            Command::SetMode => self.execute_set_mode(),
            Command::Stop => self.execute_stop(),
            Command::Test => self.execute_test(),
        };

        self.parameter_fifo.reset();

        log::debug!("  New state: {new_state:?}");

        new_state
    }

    fn generate_second_response(&mut self, command: Command) -> CommandState {
        log::debug!("Generating second response for command {command:?}");

        match command {
            Command::GetId => self.get_id_second_response(),
            Command::Init => self.init_second_response(),
            Command::Pause => self.pause_second_response(),
            Command::ReadToc => self.read_toc_second_response(),
            Command::Stop => self.stop_second_response(),
            _ => panic!("Invalid state, command {command:?} should not send a second response"),
        }
    }

    // $19: Test(sub_function) -> varies based on sub-function
    // Only sub-function $20 (get CD controller ROM version) is implemented
    fn execute_test(&mut self) -> CommandState {
        if self.parameter_fifo.len() != 1 {
            self.int5(&[stat!(self, ERROR), status::INVALID_COMMAND]);
            return CommandState::Idle;
        }

        match self.parameter_fifo.pop() {
            // $04: Start checking for SCEx string and reset SCEx counters
            // Some games (e.g. Spyro 3) use this and $05 for mod chip detection. They execute $04
            // while the drive is running, then execute $05 after a few seconds and validate that
            // the drive did not successfully read an 'SCEx' string.
            0x04 => {
                self.int3(&[stat!(self)]);

                // The SCEx string is contained in the lead-in area, and I don't think the drive
                // will ever re-read the lead-in after booting? Unless a Reset command is executed
                self.scex_read = false;
            }
            // $05: Get SCEx counters
            0x05 => {
                // First value is number of 'Sxxx' strings read and second value is number of 'SCEx' strings
                // Pretend they are always equal
                if self.scex_read {
                    self.int3(&[0x01, 0x01]);
                } else {
                    self.int3(&[0x00, 0x00]);
                }
            }
            // $20: Get controller BIOS version
            0x20 => {
                // TODO use a different BIOS version?
                self.int3(&[0x95, 0x07, 0x24, 0xC1]);
            }
            other => todo!("Test sub-function {other:02X}"),
        }

        CommandState::Idle
    }

    pub fn read_port(&mut self, address: u32) -> u8 {
        log::trace!("CD-ROM register read: {address:08X}.{}", self.index);

        match (address & 3, self.index) {
            (0, _) => {
                // $1F801800 R: Index/status register
                self.read_status_register()
            }
            (1, _) => {
                // $1F801801 R: Response FIFO
                self.read_response_fifo()
            }
            (2, _) => {
                // $1F801802 R: Data FIFO
                self.read_data_fifo()
            }
            (3, 0 | 2) => {
                // $1F801803.0/2 R: Interrupts enabled register
                self.interrupts.read_enabled()
            }
            (3, 1 | 3) => {
                // $1F801803.1/3 R: Interrupt flags register
                self.interrupts.read_flags()
            }
            _ => panic!("Invalid CD-ROM read: address {address:08X}, index {}", self.index),
        }
    }

    pub fn write_port(&mut self, address: u32, value: u8) {
        log::trace!("CD-ROM register write: {address:08X}.{} {value:02X}", self.index);

        match (address & 3, self.index) {
            (0, _) => {
                // $1F801800 W: Index/status register
                self.write_index_register(value);
            }
            (1, 0) => {
                // $1F801801.0 W: Command register
                self.write_command(value);
            }
            (1, 3) => {
                // $1F801801.3 W: Right CD to Right SPU volume
                log::debug!("R CD to R SPU volume: {value:02X}");
                self.next_cd_to_spu_volume[1][1] = value;
            }
            (2, 0) => {
                // $1F801802.0 W: Parameter FIFO
                self.write_parameter_fifo(value);
            }
            (2, 1) => {
                // $1F801802.1 W: Interrupts enabled register
                self.interrupts.write_enabled(value);
            }
            (2, 2) => {
                // $1F801802.2 W: Left CD to Left SPU volume
                log::debug!("L CD to L SPU volume: {value:02X}");
                self.next_cd_to_spu_volume[0][0] = value;
            }
            (2, 3) => {
                // $1F801802.3 W: Right CD to Left SPU volume
                log::debug!("R CD to L SPU volume: {value:02X}");
                self.next_cd_to_spu_volume[0][1] = value;
            }
            (3, 0) => {
                // $1F801803.0 W: Request register
                self.write_request_register(value);
            }
            (3, 1) => {
                // $1F801803.1 W: Interrupt flags register
                self.interrupts.write_flags(value, &mut self.parameter_fifo);
            }
            (3, 2) => {
                // $1F801803.2 W: Left CD to Right SPU volume
                log::debug!("L CD to R SPU volume: {value:02X}");
                self.next_cd_to_spu_volume[1][0] = value;
            }
            (3, 3) => {
                // $1F801803.3 W: Apply audio volume changes
                self.write_apply_volume_register(value);
            }
            _ => todo!("CD-ROM write {address:08X}.{} {value:02X}", self.index),
        }
    }

    pub fn read_data_fifo(&mut self) -> u8 {
        self.data_fifo.pop()
    }

    fn read_status_register(&self) -> u8 {
        // TODO Bit 2: XA-ADPCM FIFO not empty (hardcoded to 0)
        let receiving_command = matches!(
            self.command_state,
            CommandState::CommandQueued { .. } | CommandState::ReceivingCommand { .. }
        );
        let status = self.index
            | (u8::from(self.parameter_fifo.empty()) << 3)
            | (u8::from(!self.parameter_fifo.full()) << 4)
            | (u8::from(!self.response_fifo.fully_consumed()) << 5)
            | (u8::from(!self.data_fifo.fully_consumed()) << 6)
            | (u8::from(receiving_command) << 7);

        log::debug!("Status read: {status:02X}");

        status
    }

    fn write_index_register(&mut self, value: u8) {
        // Only bits 0-1 (index) are writable
        self.index = value & 3;
        log::trace!("Index changed to {}", self.index);
    }

    fn write_command(&mut self, command_byte: u8) {
        let std_receive_cycles = match self.drive_state {
            DriveState::Stopped => RECEIVE_COMMAND_CYCLES_STOPPED,
            _ => RECEIVE_COMMAND_CYCLES_RUNNING,
        };

        let (command, cycles) = match command_byte {
            0x01 => (Command::GetStat, std_receive_cycles),
            0x02 => (Command::SetLoc, std_receive_cycles),
            0x03 => (Command::Play, std_receive_cycles),
            0x06 => (Command::ReadN, std_receive_cycles),
            0x07 => (Command::MotorOn, std_receive_cycles),
            0x08 => (Command::Stop, std_receive_cycles),
            0x09 => (Command::Pause, std_receive_cycles),
            0x0A => (Command::Init, INIT_COMMAND_CYCLES),
            0x0B => (Command::Mute, std_receive_cycles),
            0x0C => (Command::Demute, std_receive_cycles),
            0x0D => (Command::SetFilter, std_receive_cycles),
            0x0E => (Command::SetMode, std_receive_cycles),
            0x10 => (Command::GetLocL, std_receive_cycles),
            0x11 => (Command::GetLocP, std_receive_cycles),
            0x13 => (Command::GetTN, std_receive_cycles),
            0x14 => (Command::GetTD, std_receive_cycles),
            0x15 => (Command::SeekL, std_receive_cycles),
            0x16 => (Command::SeekP, std_receive_cycles),
            0x19 => (Command::Test, std_receive_cycles),
            0x1A => (Command::GetId, std_receive_cycles),
            0x1B => (Command::ReadS, std_receive_cycles),
            0x1E => (Command::ReadToc, INIT_COMMAND_CYCLES),
            _ => todo!("Command byte {command_byte:02X}"),
        };
        self.command_state = if self.interrupts.int_queued() {
            CommandState::CommandQueued { command, cycles }
        } else {
            CommandState::ReceivingCommand { command, cycles_remaining: cycles }
        };

        log::debug!("Received command, new state: {:?}", self.command_state);
    }

    fn read_response_fifo(&mut self) -> u8 {
        let value = self.response_fifo.pop();
        log::debug!("Response FIFO read: {value:02X}");
        value
    }

    fn write_parameter_fifo(&mut self, value: u8) {
        self.parameter_fifo.push(value);
        log::debug!("  Parameter FIFO write (idx {}): {value:02X}", self.parameter_fifo.len() - 1);
    }

    #[allow(clippy::unused_self)]
    fn write_request_register(&mut self, value: u8) {
        if value.bit(5) {
            todo!("SMEN bit set in request register (command start interrupt)");
        }

        // TODO BFRD bit: Set by the host to accept a sector into the data FIFO (?)

        log::debug!("Request register write: {value:02X}");
        log::debug!("  SMEN: {}", value.bit(5));
        log::debug!("  BFRD: {}", value.bit(7));
    }

    fn write_apply_volume_register(&mut self, value: u8) {
        self.xa_adpcm.muted = value.bit(0);

        if value.bit(5) {
            self.cd_to_spu_volume = self.next_cd_to_spu_volume;
        }

        log::debug!("Apply volume write: {value:02X}");
        log::debug!("  ADPCM muted: {}", self.xa_adpcm.muted);
        log::debug!("  Applied CD-to-SPU volume changes: {}", value.bit(5));
    }

    fn read_sector_atime(&mut self, time: CdTime) -> CdRomResult<()> {
        let disc =
            self.disc.as_mut().expect("read_sector_atime() called with no disc in the drive");

        let Some(track) = disc.cue().find_track_by_time(time) else {
            // TODO INT4+pause at disc end
            todo!("Read to end of disc");
        };

        let track_number = track.number;
        let relative_time = time - track.start_time;

        log::debug!("Reading sector at atime {time}, track {track_number} time {relative_time}");

        disc.read_sector(track_number, relative_time, self.sector_buffer.as_mut())?;

        Ok(())
    }

    pub fn current_audio_sample(&self) -> (i16, i16) {
        if self.audio_muted { (0, 0) } else { self.current_audio_sample }
    }

    pub fn spu_volume_matrix(&self) -> [[u8; 2]; 2] {
        self.cd_to_spu_volume
    }

    pub fn take_disc(&mut self) -> Option<CdRom> {
        self.disc.take()
    }

    pub fn change_disc(&mut self, disc: Option<CdRom>) {
        self.disc = disc;
        self.shell_opened = true;
        self.drive_state = DriveState::Stopped;

        self.int5(&[stat!(self), status::SHELL_OPENED]);
    }
}

fn bcd_to_binary(value: u8) -> u8 {
    10 * (value >> 4) + (value & 0xF)
}

fn binary_to_bcd(value: u8) -> u8 {
    ((value / 10) << 4) | (value % 10)
}
