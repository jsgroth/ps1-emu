mod envelope;
mod reverb;
mod voice;

use crate::cpu::OpSize;
use crate::num::U32Ext;
use crate::spu::envelope::VolumeControl;
use crate::spu::reverb::ReverbSettings;
use crate::spu::voice::Voice;
use std::array;

const AUDIO_RAM_LEN: usize = 512 * 1024;
const AUDIO_RAM_MASK: u32 = (AUDIO_RAM_LEN - 1) as u32;

const NUM_VOICES: usize = 24;

// The SPU clock rate is exactly 1/768 the CPU clock rate
// This _should_ be 44.1 KHz, but it may not be exactly depending on the exact oscillator speed
const SPU_CLOCK_DIVIDER: u32 = 768;

type AudioRam = [u8; AUDIO_RAM_LEN];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum DataPortMode {
    #[default]
    Off = 0,
    ManualWrite = 1,
    DmaWrite = 2,
    DmaRead = 3,
}

impl DataPortMode {
    fn from_bits(bits: u32) -> Self {
        match bits & 3 {
            0 => Self::Off,
            1 => Self::ManualWrite,
            2 => Self::DmaWrite,
            3 => Self::DmaRead,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }

    const fn is_dma(self) -> bool {
        matches!(self, Self::DmaWrite | Self::DmaRead)
    }
}

#[derive(Debug, Clone)]
struct DataPort {
    mode: DataPortMode,
    start_address: u32,
    current_address: u32,
}

impl DataPort {
    fn new() -> Self {
        Self {
            mode: DataPortMode::default(),
            start_address: 0,
            current_address: 0,
        }
    }

    // $1F801DA6: Sound RAM data transfer address
    fn write_transfer_address(&mut self, value: u32) {
        // Address is in 8-byte units
        // Writing start address also sets an internal current address register
        self.start_address = (value & 0xFFFF) << 3;
        self.current_address = self.start_address;

        log::trace!(
            "Sound RAM data transfer address: {:05X}",
            self.start_address
        );
    }
}

#[derive(Debug, Clone)]
struct ControlRegisters {
    spu_enabled: bool,
    amplifier_enabled: bool,
    external_audio_enabled: bool,
    cd_audio_enabled: bool,
    external_audio_reverb_enabled: bool,
    cd_audio_reverb_enabled: bool,
    irq_enabled: bool,
    noise_shift: u8,
    noise_step: u8,
    // Recorded in case software reads the KON or KOFF registers
    last_key_on_write: u32,
    last_key_off_write: u32,
}

impl ControlRegisters {
    fn new() -> Self {
        Self {
            spu_enabled: false,
            amplifier_enabled: false,
            external_audio_enabled: false,
            cd_audio_enabled: false,
            external_audio_reverb_enabled: false,
            cd_audio_reverb_enabled: false,
            irq_enabled: false,
            noise_shift: 0,
            noise_step: 0,
            last_key_on_write: 0,
            last_key_off_write: 0,
        }
    }

    // $1F801DAA: SPU control register (SPUCNT)
    fn read_spucnt(&self, data_port: &DataPort, reverb: &ReverbSettings) -> u32 {
        (u32::from(self.spu_enabled) << 15)
            | (u32::from(self.amplifier_enabled) << 14)
            | (u32::from(self.noise_shift) << 10)
            | (u32::from(self.noise_step) << 8)
            | (u32::from(reverb.writes_enabled) << 7)
            | (u32::from(self.irq_enabled) << 6)
            | ((data_port.mode as u32) << 4)
            | (u32::from(self.external_audio_reverb_enabled) << 3)
            | (u32::from(self.cd_audio_reverb_enabled) << 2)
            | (u32::from(self.external_audio_enabled) << 1)
            | u32::from(self.cd_audio_enabled)
    }

    // $1F801DAA: SPU control register (SPUCNT)
    fn write_spucnt(&mut self, value: u32, data_port: &mut DataPort, reverb: &mut ReverbSettings) {
        self.spu_enabled = value.bit(15);
        self.amplifier_enabled = value.bit(14);
        self.noise_shift = ((value >> 10) & 0xF) as u8;
        self.noise_step = ((value >> 8) & 3) as u8;
        reverb.writes_enabled = value.bit(7);
        self.irq_enabled = value.bit(6);
        data_port.mode = DataPortMode::from_bits(value >> 4);
        self.external_audio_reverb_enabled = value.bit(3);
        self.cd_audio_reverb_enabled = value.bit(2);
        self.external_audio_enabled = value.bit(1);
        self.cd_audio_enabled = value.bit(0);

        log::trace!("SPUCNT write");
        log::trace!("  SPU enabled: {}", self.spu_enabled);
        log::trace!("  Amplifier enabled: {}", self.amplifier_enabled);
        log::trace!("  Noise shift: {}", self.noise_shift);
        log::trace!("  Noise step: {}", self.noise_step + 4);
        log::trace!("  Reverb writes enabled: {}", reverb.writes_enabled);
        log::trace!("  IRQ enabled: {}", self.irq_enabled);
        log::trace!("  Data port mode: {:?}", data_port.mode);
        log::trace!(
            "  External audio reverb enabled: {}",
            self.external_audio_reverb_enabled
        );
        log::trace!(
            "  CD audio reverb enabled: {}",
            self.cd_audio_reverb_enabled
        );
        log::trace!("  External audio enabled: {}", self.external_audio_enabled);
        log::trace!("  CD audio enabled: {}", self.cd_audio_enabled);
    }

    fn record_kon_low_write(&mut self, value: u32) {
        self.last_key_on_write = (self.last_key_on_write & !0xFFFF) | (value & 0xFFFF);
    }

    fn record_kon_high_write(&mut self, value: u32) {
        self.last_key_on_write = (self.last_key_on_write & 0xFFFF) | (value << 16);
    }

    fn record_koff_low_write(&mut self, value: u32) {
        self.last_key_off_write = (self.last_key_off_write & !0xFFFF) | (value & 0xFFFF);
    }

    fn record_koff_high_write(&mut self, value: u32) {
        self.last_key_off_write = (self.last_key_off_write & 0xFFFF) | (value << 16);
    }
}

#[derive(Debug, Clone)]
pub struct Spu {
    audio_ram: Box<AudioRam>,
    voices: [Voice; NUM_VOICES],
    control: ControlRegisters,
    volume: VolumeControl,
    data_port: DataPort,
    reverb: ReverbSettings,
    cpu_cycles: u32,
}

impl Spu {
    pub fn new() -> Self {
        Self {
            audio_ram: vec![0; AUDIO_RAM_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            voices: array::from_fn(|_| Voice::new()),
            control: ControlRegisters::new(),
            volume: VolumeControl::new(),
            data_port: DataPort::new(),
            reverb: ReverbSettings::default(),
            cpu_cycles: 0,
        }
    }

    pub fn tick(&mut self, cpu_cycles: u32, audio_queue: &mut Vec<(i16, i16)>) {
        self.cpu_cycles += cpu_cycles;
        while self.cpu_cycles >= SPU_CLOCK_DIVIDER {
            self.cpu_cycles -= SPU_CLOCK_DIVIDER;
            audio_queue.push(self.clock());
        }
    }

    #[allow(clippy::unused_self)]
    fn clock(&mut self) -> (i16, i16) {
        // TODO actually clock the SPU
        (0, 0)
    }

    pub fn read_register(&mut self, address: u32, size: OpSize) -> u32 {
        log::trace!("SPU register read: {address:08X}");

        if size == OpSize::Word {
            let low_halfword = self.read_register(address, OpSize::HalfWord);
            let high_halfword = self.read_register(address | 2, OpSize::HalfWord);
            return low_halfword | (high_halfword << 16);
        }

        let value = match address & 0xFFFE {
            // KON/KOFF are normally write-only, but reads return the last written value
            0x1D88 => self.control.last_key_on_write & 0xFFFF,
            0x1D8A => self.control.last_key_on_write >> 16,
            0x1D8C => self.control.last_key_off_write & 0xFFFF,
            0x1D8E => self.control.last_key_off_write >> 16,
            0x1DAA => self.control.read_spucnt(&self.data_port, &self.reverb),
            // TODO return an actual value for sound RAM data transfer control?
            0x1DAC => 0x0004,
            0x1DAE => self.read_status_register(),
            _ => todo!("SPU read register {address:08X}"),
        };

        match size {
            OpSize::Byte => {
                if !address.bit(0) {
                    value & 0xFF
                } else {
                    (value >> 8) & 0xFF
                }
            }
            OpSize::HalfWord => value,
            OpSize::Word => unreachable!("size Word should have early returned"),
        }
    }

    pub fn write_register(&mut self, address: u32, value: u32, size: OpSize) {
        log::trace!("SPU register write: {address:08X} {value:08X} {size:?}");

        match size {
            OpSize::Byte => {
                if address.bit(0) {
                    // 8-bit writes to odd addresses do nothing
                    return;
                }
            }
            OpSize::HalfWord => {}
            OpSize::Word => {
                self.write_register(address, value & 0xFFFF, OpSize::HalfWord);
                self.write_register(address | 2, value >> 16, OpSize::HalfWord);
                return;
            }
        }

        match address & 0xFFFF {
            0x1C00..=0x1D7F => self.write_voice_register(address, value),
            0x1D80 => self.volume.write_main_l(value),
            0x1D82 => self.volume.write_main_r(value),
            0x1D84 => self.reverb.write_output_volume_l(value),
            0x1D86 => self.reverb.write_output_volume_r(value),
            0x1D88 => self.key_on_low(value),
            0x1D8A => self.key_on_high(value),
            0x1D8C => self.key_off_low(value),
            0x1D8E => self.key_off_high(value),
            0x1D90 => log::warn!("Unimplemented FM/LFO mode write (low halfword): {value:04X}"),
            0x1D92 => log::warn!("Unimplemented FM/LFO mode write (high halfword): {value:04X}"),
            0x1D94 => log::warn!("Unimplemented noise mode write (low halfword): {value:04X}"),
            0x1D96 => log::warn!("Unimplemented noise mode write (high halfword): {value:04X}"),
            0x1D98 => log::warn!("Unimplemented voice reverb enabled write (0-15): {value:04X}"),
            0x1D9A => log::warn!("Unimplemented voice reverb enabled write (16-23): {value:04X}"),
            0x1DA2 => self.reverb.write_buffer_start_address(value),
            0x1DA6 => self.data_port.write_transfer_address(value),
            0x1DA8 => self.write_data_port(value),
            0x1DAA => self
                .control
                .write_spucnt(value, &mut self.data_port, &mut self.reverb),
            0x1DAC => {
                // Sound RAM data transfer control register; writing any value other than $0004
                // would be highly unexpected
                if value & 0xFFFF != 0x0004 {
                    todo!("Unexpected sound RAM data transfer control write: {value:04X}");
                }
            }
            0x1DB0 => self.volume.write_cd_l(value),
            0x1DB2 => self.volume.write_cd_r(value),
            0x1DB4 => log::warn!("Unimplemented external audio volume L write: {value:04X}"),
            0x1DB6 => log::warn!("Unimplemented external audio volume R write: {value:04X}"),
            0x1DC0..=0x1DFF => self.reverb.write_register(address, value),
            _ => todo!("SPU write {address:08X} {value:08X}"),
        }
    }

    // $1F801C00-$1F801D7F: Individual voice registers
    fn write_voice_register(&mut self, address: u32, value: u32) {
        let voice = get_voice_number(address);
        if voice >= NUM_VOICES {
            log::error!("Invalid voice register write: {address:08X} {value:04X}");
            return;
        }

        match address & 0xF {
            0x0 => {
                // $1F801C00: Voice volume L
                self.voices[voice].write_volume_l(value);
                log::trace!("Voice {voice} volume L: {:?}", self.voices[voice].volume_l);
            }
            0x2 => {
                // $1F801C02: Voice volume R
                self.voices[voice].write_volume_r(value);
                log::trace!("Voice {voice} volume R: {:?}", self.voices[voice].volume_r);
            }
            0x4 => {
                // $1F801C04: Voice sample rate
                self.voices[voice].write_sample_rate(value);
                log::trace!(
                    "Voice {voice} sample rate: {:04X}",
                    self.voices[voice].sample_rate
                );
            }
            0x6 => {
                // $1F801C06: ADPCM start address
                self.voices[voice].write_start_address(value);
                log::trace!(
                    "Voice {voice} start address: {:05X}",
                    self.voices[voice].start_address
                );
            }
            0x8 => {
                // $1F801C08: ADSR settings, low halfword
                self.voices[voice].adsr.write_low(value);
                log::trace!(
                    "Voice {voice} ADSR settings (low): {:?}",
                    self.voices[voice].adsr
                );
            }
            0xA => {
                // $1F801C0A: ADSR settings, high halfword
                self.voices[voice].adsr.write_high(value);
                log::trace!(
                    "Voice {voice} ADSR settings (high): {:?}",
                    self.voices[voice].adsr
                );
            }
            _ => todo!("voice {voice} register write: {address:08X} {value:04X}"),
        }
    }

    // $1F801DAE: SPU status register (SPUSTAT)
    fn read_status_register(&self) -> u32 {
        // TODO: bit 11 (writing to first/second half of capture buffers)
        // TODO: bit 10 (data transfer busy) is hardcoded
        // TODO: bit 6 (IRQ)
        // TODO: timing? switching to DMA read mode should not immediately set bits 7 and 9
        let value = (u32::from(self.data_port.mode == DataPortMode::DmaRead) << 9)
            | (u32::from(self.data_port.mode == DataPortMode::DmaWrite) << 8)
            | (u32::from(self.data_port.mode.is_dma()) << 7)
            | ((self.data_port.mode as u32) << 5)
            | (u32::from(self.control.external_audio_reverb_enabled) << 3)
            | (u32::from(self.control.cd_audio_reverb_enabled) << 2)
            | (u32::from(self.control.external_audio_enabled) << 1)
            | u32::from(self.control.cd_audio_enabled);

        log::trace!("SPUSTAT read: {value:08X}");

        value
    }

    // $1F801DA8: Sound RAM data transfer FIFO port
    fn write_data_port(&mut self, value: u32) {
        // TODO emulate the 32-halfword FIFO?
        // TODO check current state? (requires FIFO emulation, the BIOS writes while mode is off)
        let [lsb, msb] = (value as u16).to_le_bytes();
        self.audio_ram[self.data_port.current_address as usize] = lsb;
        self.audio_ram[(self.data_port.current_address + 1) as usize] = msb;

        log::trace!(
            "Wrote to {:05X} in audio RAM",
            self.data_port.current_address
        );

        self.data_port.current_address = (self.data_port.current_address + 2) & AUDIO_RAM_MASK;
    }

    // $1F801D88: Key on (voices 0-15)
    fn key_on_low(&mut self, value: u32) {
        log::trace!("Key on low write: {value:04X}");

        for voice in 0..16 {
            if value.bit(voice) {
                log::trace!("Keying on voice {voice}");
                self.voices[voice as usize].key_on();
            }
        }

        self.control.record_kon_low_write(value);
    }

    // $1F801D8A: Key on (voices 16-23)
    fn key_on_high(&mut self, value: u32) {
        log::trace!("Key on high write: {value:04X}");

        for voice in 16..24 {
            if value.bit(voice - 16) {
                log::trace!("Keying on voice {voice}");
                self.voices[voice as usize].key_on();
            }
        }

        self.control.record_kon_high_write(value);
    }

    // $1F801D8C: Key off (voices 0-15)
    fn key_off_low(&mut self, value: u32) {
        log::trace!("Key off low write: {value:04X}");

        for voice in 0..16 {
            if value.bit(voice) {
                log::trace!("Keying off voice {voice}");
                self.voices[voice as usize].key_off();
            }
        }

        self.control.record_koff_low_write(value);
    }

    // $1F801D8E: Key off (voices 16-23)
    fn key_off_high(&mut self, value: u32) {
        log::trace!("Key off high write: {value:04X}");

        for voice in 16..24 {
            if value.bit(voice - 16) {
                log::trace!("Keying off voice {voice}");
                self.voices[voice as usize].key_off();
            }
        }

        self.control.record_koff_high_write(value);
    }
}

fn get_voice_number(address: u32) -> usize {
    ((address >> 4) & 0x1F) as usize
}