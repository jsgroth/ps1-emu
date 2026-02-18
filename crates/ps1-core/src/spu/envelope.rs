//! Volume and envelope code

use crate::num::{U16Ext, U32Ext};
use bincode::{Decode, Encode};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum EnvelopeMode {
    #[default]
    Linear = 0,
    Exponential = 1,
}

impl EnvelopeMode {
    fn from_bit(bit: bool) -> Self {
        if bit { Self::Exponential } else { Self::Linear }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum EnvelopeDirection {
    #[default]
    Increasing = 0,
    Decreasing = 1,
}

impl EnvelopeDirection {
    fn from_bit(bit: bool) -> Self {
        if bit { Self::Decreasing } else { Self::Increasing }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum SweepPhase {
    Positive = 0,
    Negative = 1,
}

impl SweepPhase {
    fn from_bit(bit: bool) -> Self {
        if bit { Self::Negative } else { Self::Positive }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct EnvelopeSettings {
    pub step: u8,
    pub shift: u8,
    pub direction: EnvelopeDirection,
    pub mode: EnvelopeMode,
    pub phase: SweepPhase,
}

impl EnvelopeSettings {
    pub fn clock(self, level: &mut i16, counter: &mut u16) {
        // Step is interpreted as (7 - N) for increasing and -(8 - N) for decreasing
        // -(8 - N) is just the 1's complement of (7 - N)
        // If sweep phase is negative, invert step direction
        let mut step = i32::from(7 - self.step);
        if (self.direction == EnvelopeDirection::Decreasing) ^ (self.phase == SweepPhase::Negative)
        {
            step = !step;
        }

        // Exponential increase is faked by increasing volume at a 4x slower rate when volume is
        // greater than $6000 out of $7FFF
        let effective_shift = if self.direction == EnvelopeDirection::Increasing
            && self.mode == EnvelopeMode::Exponential
            && *level > 0x6000
        {
            self.shift + 2
        } else {
            self.shift
        };

        // Step is left shifted if shift is less than 11
        step <<= 11_u8.saturating_sub(effective_shift);

        // Exponential decrease slows down the change rate as level approaches 0
        let prev_level: i32 = (*level).into();
        if self.direction == EnvelopeDirection::Decreasing && self.mode == EnvelopeMode::Exponential
        {
            step = (step * prev_level) >> 15;
        }

        // If shift is greater than 11, envelope updates less frequently than once per clock
        // Hardware tests have apparently shown that shift values above 26 function similarly to shift=26
        let counter_shift = effective_shift.saturating_sub(11);
        let mut counter_increment = if counter_shift < 16 { 0x8000 >> counter_shift } else { 0 };

        // If step and shift are all 1s, the counter does not increment and the envelope never updates.
        // Decay phase has a lower max shift value, but Decay step is fixed to 0 so this can never
        // happen for Decay phase; simply check for step=3 and shift=31 which are the max values for
        // all other phases
        if counter_increment == 0 && (self.step != 3 || self.shift != 31) {
            counter_increment = 1;
        }

        // Update envelope level if counter has crossed 0x8000
        *counter += counter_increment;
        if !counter.bit(15) {
            // Envelope level does not update this cycle
            return;
        }

        // Reset counter when updating level
        *counter = 0;

        // Update level; saturation range depends on envelope settings
        let new_level = prev_level + step;
        *level = match (self.direction, self.phase) {
            (EnvelopeDirection::Increasing, _) => {
                new_level.clamp(i16::MIN.into(), i16::MAX.into()) as i16
            }
            (EnvelopeDirection::Decreasing, SweepPhase::Negative) => {
                new_level.clamp(i16::MIN.into(), 0) as i16
            }
            (EnvelopeDirection::Decreasing, SweepPhase::Positive) => {
                new_level.clamp(0, i16::MAX.into()) as i16
            }
        };
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode, Default)]
pub enum SweepSetting {
    #[default]
    Fixed,
    Sweep(EnvelopeSettings),
}

impl SweepSetting {
    fn parse(value: u32) -> Self {
        if !value.bit(15) {
            return Self::Fixed;
        }

        let envelope_settings = EnvelopeSettings {
            step: (value & 3) as u8,
            shift: ((value >> 2) & 0x1F) as u8,
            direction: EnvelopeDirection::from_bit(value.bit(13)),
            mode: EnvelopeMode::from_bit(value.bit(14)),
            phase: SweepPhase::from_bit(value.bit(12)),
        };

        Self::Sweep(envelope_settings)
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
pub struct SweepEnvelope {
    pub volume: i16,
    pub setting: SweepSetting,
    counter: u16,
}

impl SweepEnvelope {
    pub fn new() -> Self {
        Self { volume: 0, setting: SweepSetting::default(), counter: 0 }
    }

    pub fn write(&mut self, value: u32) {
        self.setting = SweepSetting::parse(value);

        // Writing a fixed volume (bit 15 = 0) also sets current volume
        if !value.bit(15) {
            self.volume = (value << 1) as i16;
        }
    }

    pub fn read(&self) -> u32 {
        match self.setting {
            SweepSetting::Fixed => u32::from(self.volume as u16) >> 1,
            SweepSetting::Sweep(envelope) => {
                (1 << 15)
                    | ((envelope.mode as u32) << 14)
                    | ((envelope.direction as u32) << 13)
                    | ((envelope.phase as u32) << 12)
                    | (u32::from(envelope.shift) << 2)
                    | u32::from(envelope.step)
            }
        }
    }

    pub fn clock(&mut self) {
        let SweepSetting::Sweep(envelope_settings) = self.setting else {
            return;
        };

        envelope_settings.clock(&mut self.volume, &mut self.counter);
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct VolumeControl {
    pub main_l: SweepEnvelope,
    pub main_r: SweepEnvelope,
    pub cd_l: i16,
    pub cd_r: i16,
}

impl VolumeControl {
    pub fn new() -> Self {
        Self { main_l: SweepEnvelope::new(), main_r: SweepEnvelope::new(), cd_l: 0, cd_r: 0 }
    }

    // $1F801D80: Main volume left
    pub fn write_main_l(&mut self, value: u32) {
        self.main_l.write(value);
        log::trace!("Main volume L write: {:?}", self.main_l);
    }

    // $1F801D82: Main volume right
    pub fn write_main_r(&mut self, value: u32) {
        self.main_r.write(value);
        log::trace!("Main volume R write: {:?}", self.main_r);
    }

    // $1F801DB0: CD volume left
    pub fn write_cd_l(&mut self, value: u32) {
        self.cd_l = value as i16;
        log::trace!("CD volume L write: {}", self.cd_l);
    }

    // $1F801DB2: CD volume right
    pub fn write_cd_r(&mut self, value: u32) {
        self.cd_r = value as i16;
        log::trace!("CD volume R write: {}", self.cd_r);
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct AdsrSettings {
    pub attack_step: u8,
    pub attack_shift: u8,
    pub attack_mode: EnvelopeMode,
    pub decay_shift: u8,
    pub sustain_level: u16,
    pub sustain_step: u8,
    pub sustain_shift: u8,
    pub sustain_direction: EnvelopeDirection,
    pub sustain_mode: EnvelopeMode,
    pub release_shift: u8,
    pub release_mode: EnvelopeMode,
}

impl AdsrSettings {
    pub fn new() -> Self {
        Self {
            attack_step: 0,
            attack_shift: 0,
            attack_mode: EnvelopeMode::default(),
            decay_shift: 0,
            sustain_level: parse_sustain_level(0),
            sustain_step: 0,
            sustain_shift: 0,
            sustain_direction: EnvelopeDirection::default(),
            sustain_mode: EnvelopeMode::default(),
            release_shift: 0,
            release_mode: EnvelopeMode::default(),
        }
    }

    // $1F801C08 + N*$10: ADSR settings, low halfword
    pub fn write_low(&mut self, value: u32) {
        self.attack_mode = EnvelopeMode::from_bit(value.bit(15));
        self.attack_shift = ((value >> 10) & 0x1F) as u8;
        self.attack_step = ((value >> 8) & 0x3) as u8;
        self.decay_shift = ((value >> 4) & 0x0F) as u8;
        self.sustain_level = parse_sustain_level(value & 0xF);
    }

    pub fn read_low(&self) -> u32 {
        reverse_sustain_level(self.sustain_level)
            | (u32::from(self.decay_shift) << 4)
            | (u32::from(self.attack_step) << 8)
            | (u32::from(self.attack_shift) << 10)
            | ((self.attack_mode as u32) << 15)
    }

    // $1F801C0A + N*$10: ADSR settings, high halfword
    pub fn write_high(&mut self, value: u32) {
        self.sustain_mode = EnvelopeMode::from_bit(value.bit(15));
        self.sustain_direction = EnvelopeDirection::from_bit(value.bit(14));
        self.sustain_shift = ((value >> 8) & 0x1F) as u8;
        self.sustain_step = ((value >> 6) & 0x3) as u8;
        self.release_mode = EnvelopeMode::from_bit(value.bit(5));
        self.release_shift = (value & 0x1F) as u8;
    }

    pub fn read_high(&self) -> u32 {
        u32::from(self.release_shift)
            | ((self.release_mode as u32) << 5)
            | (u32::from(self.sustain_step) << 6)
            | (u32::from(self.sustain_shift) << 8)
            | ((self.sustain_direction as u32) << 14)
            | ((self.sustain_mode as u32) << 15)
    }
}

fn parse_sustain_level(value: u32) -> u16 {
    (((value & 0xF) + 1) << 11) as u16
}

fn reverse_sustain_level(value: u16) -> u32 {
    (u32::from(value) >> 11) - 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum AdsrPhase {
    Attack,
    Decay,
    Sustain,
    #[default]
    Release,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct AdsrEnvelope {
    pub level: i16,
    pub settings: AdsrSettings,
    pub phase: AdsrPhase,
    counter: u16,
}

impl AdsrEnvelope {
    pub fn new() -> Self {
        Self { level: 0, settings: AdsrSettings::new(), phase: AdsrPhase::default(), counter: 0 }
    }

    pub fn clock(&mut self) {
        if self.phase == AdsrPhase::Attack && self.level == i16::MAX {
            self.phase = AdsrPhase::Decay;
        }

        if self.phase == AdsrPhase::Decay && (self.level as u16) <= self.settings.sustain_level {
            self.phase = AdsrPhase::Sustain;
        }

        let envelope_settings = match self.phase {
            AdsrPhase::Attack => EnvelopeSettings {
                step: self.settings.attack_step,
                shift: self.settings.attack_shift,
                direction: EnvelopeDirection::Increasing,
                mode: self.settings.attack_mode,
                phase: SweepPhase::Positive,
            },
            AdsrPhase::Decay => EnvelopeSettings {
                step: 0,
                shift: self.settings.decay_shift,
                direction: EnvelopeDirection::Decreasing,
                mode: EnvelopeMode::Exponential,
                phase: SweepPhase::Positive,
            },
            AdsrPhase::Sustain => EnvelopeSettings {
                step: self.settings.sustain_step,
                shift: self.settings.sustain_shift,
                direction: self.settings.sustain_direction,
                mode: self.settings.sustain_mode,
                phase: SweepPhase::Positive,
            },
            AdsrPhase::Release => EnvelopeSettings {
                step: 0,
                shift: self.settings.release_shift,
                direction: EnvelopeDirection::Decreasing,
                mode: self.settings.release_mode,
                phase: SweepPhase::Positive,
            },
        };

        envelope_settings.clock(&mut self.level, &mut self.counter);
    }

    pub fn key_on(&mut self) {
        self.phase = AdsrPhase::Attack;
        self.level = 0;
    }

    pub fn key_off(&mut self) {
        self.phase = AdsrPhase::Release;
    }
}
