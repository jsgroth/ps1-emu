//! SPU reverb code

mod fir;

use crate::num::U32Ext;
use crate::spu;
use crate::spu::reverb::fir::FirSampleDeque;
use crate::spu::voice::Voice;
use crate::spu::{I32Ext, SoundRam, multiply_volume, multiply_volume_i32};
use bincode::{Decode, Encode};
use std::cmp;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum ReverbClock {
    #[default]
    Left,
    Right,
}

impl ReverbClock {
    #[must_use]
    fn invert(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Encode, Decode)]
struct StereoValue<T> {
    l: T,
    r: T,
}

impl<T: Copy> StereoValue<T> {
    fn get(self, clock: ReverbClock) -> T {
        match clock {
            ReverbClock::Left => self.l,
            ReverbClock::Right => self.r,
        }
    }
}

type StereoI16 = StereoValue<i16>;
type StereoI32 = StereoValue<i32>;
type StereoU32 = StereoValue<u32>;

#[derive(Debug, Clone, Default, Encode, Decode)]
pub struct ReverbUnit {
    pub writes_enabled: bool,
    pub cd_enabled: bool,
    voices_enabled: [bool; spu::NUM_VOICES],
    buffer_start_addr: u32,
    buffer_current_addr: u32,
    input_volume: StereoI32,
    output_volume: StereoI16,
    comb_volumes: [i32; 4],
    reflection_volume_1: i32,
    reflection_volume_2: i32,
    apf_volume_1: i32,
    apf_volume_2: i32,
    comb_addrs: [StereoU32; 4],
    same_reflect_addr_1: StereoU32,
    same_reflect_addr_2: StereoU32,
    diff_reflect_addr_1: StereoU32,
    diff_reflect_addr_2: StereoU32,
    apf_addr_1: StereoU32,
    apf_offset_1: u32,
    apf_addr_2: StereoU32,
    apf_offset_2: u32,
    clock: ReverbClock,
    input_buffer_l: FirSampleDeque,
    input_buffer_r: FirSampleDeque,
    output_buffer_l: FirSampleDeque,
    output_buffer_r: FirSampleDeque,
    pub current_output: (i16, i16),
}

impl ReverbUnit {
    pub fn clock(
        &mut self,
        voices: &[Voice; spu::NUM_VOICES],
        cd_sample: (i16, i16),
        sound_ram: &mut SoundRam,
    ) {
        // Input samples are pushed into buffers on every 44100 Hz clock
        let (input_sample_l, input_sample_r) = self.compute_input_sample(voices, cd_sample);
        self.input_buffer_l.push(input_sample_l.into());
        self.input_buffer_r.push(input_sample_r.into());

        // The reverb unit as a whole processes at only 22050 Hz.
        // Emulate this by alternating between L processing and R processing on every 44100 Hz clock
        let input_sample = match self.clock {
            ReverbClock::Left => fir::filter(&self.input_buffer_l),
            ReverbClock::Right => fir::filter(&self.input_buffer_r),
        };

        // Apply input volume
        let input_sample = multiply_volume_i32(input_sample, self.input_volume.get(self.clock));

        self.perform_same_side_reflection(input_sample, sound_ram);
        self.perform_different_side_reflection(input_sample, sound_ram);

        let comb_filter_output = self.apply_comb_filter(sound_ram);
        let apf1_output = self.apply_all_pass_filter_1(comb_filter_output, sound_ram);
        let apf2_output = self.apply_all_pass_filter_2(apf1_output, sound_ram);

        let output_sample = apf2_output.clamp(i16::MIN.into(), i16::MAX.into());

        // Push the output sample into the correct buffer and push a padding zero to the other
        match self.clock {
            ReverbClock::Left => {
                self.output_buffer_l.push(output_sample);
                self.output_buffer_r.push(0);
            }
            ReverbClock::Right => {
                self.output_buffer_l.push(0);
                self.output_buffer_r.push(output_sample);
            }
        }

        // Double FIR filter outputs to account for zero padding
        let filtered_l = (fir::filter(&self.output_buffer_l) << 1).clamp_to_i16();
        let filtered_r = (fir::filter(&self.output_buffer_r) << 1).clamp_to_i16();

        // Apply output volume after filtering so that volume changes will take immediate effect
        self.current_output = (
            multiply_volume(filtered_l, self.output_volume.l),
            multiply_volume(filtered_r, self.output_volume.r),
        );

        // Increment buffer address only after processing both L and R samples
        if self.clock == ReverbClock::Right {
            self.buffer_current_addr = cmp::max(
                self.buffer_start_addr,
                self.buffer_current_addr.wrapping_add(2) & spu::SOUND_RAM_MASK,
            );
        }

        self.clock = self.clock.invert();
    }

    fn compute_input_sample(
        &self,
        voices: &[Voice; spu::NUM_VOICES],
        cd_sample: (i16, i16),
    ) -> (i16, i16) {
        let mut input_sample_l = 0_i32;
        let mut input_sample_r = 0_i32;
        for (voice, reverb_enabled) in voices.iter().zip(self.voices_enabled) {
            if !reverb_enabled {
                continue;
            }

            let (voice_sample_l, voice_sample_r) = voice.current_sample;
            input_sample_l += i32::from(voice_sample_l);
            input_sample_r += i32::from(voice_sample_r);
        }

        if self.cd_enabled {
            let (cd_l, cd_r) = cd_sample;
            input_sample_l += i32::from(cd_l);
            input_sample_r += i32::from(cd_r);
        }

        (input_sample_l.clamp_to_i16(), input_sample_r.clamp_to_i16())
    }

    fn perform_same_side_reflection(&mut self, input_sample: i32, sound_ram: &mut SoundRam) {
        if !self.writes_enabled {
            return;
        }

        let m_addr = self.same_reflect_addr_1.get(self.clock);
        let d_addr = self.same_reflect_addr_2.get(self.clock);
        self.perform_reflection(input_sample, m_addr, d_addr, sound_ram);
    }

    fn perform_different_side_reflection(&mut self, input_sample: i32, sound_ram: &mut SoundRam) {
        if !self.writes_enabled {
            return;
        }

        let m_addr = self.diff_reflect_addr_1.get(self.clock);
        let d_addr = self.diff_reflect_addr_2.get(self.clock.invert());
        self.perform_reflection(input_sample, m_addr, d_addr, sound_ram);
    }

    fn perform_reflection(
        &mut self,
        input_sample: i32,
        m_addr: u32,
        d_addr: u32,
        sound_ram: &mut SoundRam,
    ) {
        let v_iir = self.reflection_volume_1;
        let v_wall = self.reflection_volume_2;

        let m_sample: i32 =
            sound_ram.get_i16(self.relative_buffer_address(m_addr.wrapping_sub(2))).into();
        let d_sample: i32 = sound_ram.get_i16(self.relative_buffer_address(d_addr)).into();

        let reflect_sample = m_sample
            + multiply_volume_i32(
                (input_sample + multiply_volume_i32(d_sample, v_wall) - m_sample)
                    .clamp(i16::MIN.into(), i16::MAX.into()),
                v_iir,
            );
        sound_ram.set_i16(self.relative_buffer_address(m_addr), reflect_sample.clamp_to_i16());
    }

    fn apply_comb_filter(&self, sound_ram: &SoundRam) -> i32 {
        (0..4)
            .map(|i| {
                let ram_addr = self.relative_buffer_address(self.comb_addrs[i].get(self.clock));
                let comb_sample: i32 = sound_ram.get_i16(ram_addr).into();
                multiply_volume_i32(comb_sample, self.comb_volumes[i])
            })
            .sum::<i32>()
            .clamp(i16::MIN.into(), i16::MAX.into())
    }

    fn apply_all_pass_filter_1(&self, comb_filter_output: i32, sound_ram: &mut SoundRam) -> i32 {
        let m_apf1 = self.apf_addr_1.get(self.clock);
        let d_apf1 = self.apf_offset_1;
        let v_apf1 = self.apf_volume_1;
        self.apply_all_pass_filter(comb_filter_output, m_apf1, d_apf1, v_apf1, sound_ram)
    }

    fn apply_all_pass_filter_2(&self, apf1_output: i32, sound_ram: &mut SoundRam) -> i32 {
        let m_apf2 = self.apf_addr_2.get(self.clock);
        let d_apf2 = self.apf_offset_2;
        let v_apf2 = self.apf_volume_2;
        self.apply_all_pass_filter(apf1_output, m_apf2, d_apf2, v_apf2, sound_ram)
    }

    fn apply_all_pass_filter(
        &self,
        prev_output: i32,
        m_apf: u32,
        d_apf: u32,
        v_apf: i32,
        sound_ram: &mut SoundRam,
    ) -> i32 {
        let apf_input_sample: i32 =
            sound_ram.get_i16(self.relative_buffer_address(m_apf.wrapping_sub(d_apf))).into();

        let new_apf_sample =
            (prev_output - multiply_volume_i32(apf_input_sample, v_apf)).clamp_to_i16();
        if self.writes_enabled {
            sound_ram.set_i16(self.relative_buffer_address(m_apf), new_apf_sample);
        }

        apf_input_sample + multiply_volume_i32(new_apf_sample.into(), v_apf)
    }

    fn reverb_buffer_len(&self) -> u32 {
        spu::SOUND_RAM_LEN as u32 - self.buffer_start_addr
    }

    fn relative_buffer_address(&self, register_addr: u32) -> u32 {
        let buffer_offset = self.buffer_current_addr - self.buffer_start_addr;
        let register_offset = buffer_offset.wrapping_add(register_addr) % self.reverb_buffer_len();
        self.buffer_start_addr.wrapping_add(register_offset)
    }

    // $1F801D84: Reverb output volume L (vLOUT)
    pub fn write_output_volume_l(&mut self, value: u32) {
        self.output_volume.l = parse_volume(value) as i16;
        log::trace!("Reverb output volume L: {}", self.output_volume.l);
    }

    pub fn read_output_volume_l(&self) -> u32 {
        (self.output_volume.l as u16).into()
    }

    // $1F801D86: Reverb output volume R (vROUT)
    pub fn write_output_volume_r(&mut self, value: u32) {
        self.output_volume.r = parse_volume(value) as i16;
        log::trace!("Reverb output volume R: {}", self.output_volume.r);
    }

    pub fn read_output_volume_r(&self) -> u32 {
        (self.output_volume.r as u16).into()
    }

    // $1F801D98: Reverb enabled, low halfword (EON)
    pub fn write_reverb_on_low(&mut self, value: u32) {
        for i in 0..16 {
            self.voices_enabled[i] = value.bit(i as u8);
        }

        log::trace!("Reverb enabled (voices 0-15): {value:04X}");
    }

    pub fn read_reverb_on_low(&self) -> u32 {
        (0..16).map(|i| u32::from(self.voices_enabled[i]) << i).reduce(|a, b| a | b).unwrap()
    }

    // $1F801D9A: Reverb enabled, high halfword (EON)
    pub fn write_reverb_on_high(&mut self, value: u32) {
        for i in 16..24 {
            self.voices_enabled[i] = value.bit((i - 16) as u8);
        }

        log::trace!("Reverb enabled (voices 16-23): {value:04X}");
    }

    pub fn read_reverb_on_high(&self) -> u32 {
        (0..8).map(|i| u32::from(self.voices_enabled[i + 16]) << i).reduce(|a, b| a | b).unwrap()
    }

    // $1F801DA2: Reverb buffer start address (mBASE)
    pub fn write_buffer_start_address(&mut self, value: u32) {
        // Writing start address also sets current address
        self.buffer_start_addr = parse_address(value);
        self.buffer_current_addr = self.buffer_start_addr;
        log::trace!("Reverb buffer start address: {:05X}", self.buffer_start_addr);
    }

    pub fn read_buffer_start_address(&self) -> u32 {
        reverse_address(self.buffer_start_addr)
    }

    // $1F801DC0-$1F801DFF: The majority of the reverb registers
    pub fn write_register(&mut self, address: u32, value: u32) {
        match address & 0xFFFF {
            0x1DC0 => self.write_dapf1(value),
            0x1DC2 => self.write_dapf2(value),
            0x1DC4 => self.write_viir(value),
            0x1DC6 => self.write_vcomb1(value),
            0x1DC8 => self.write_vcomb2(value),
            0x1DCA => self.write_vcomb3(value),
            0x1DCC => self.write_vcomb4(value),
            0x1DCE => self.write_vwall(value),
            0x1DD0 => self.write_vapf1(value),
            0x1DD2 => self.write_vapf2(value),
            0x1DD4 => self.write_mlsame(value),
            0x1DD6 => self.write_mrsame(value),
            0x1DD8 => self.write_mlcomb1(value),
            0x1DDA => self.write_mrcomb1(value),
            0x1DDC => self.write_mlcomb2(value),
            0x1DDE => self.write_mrcomb2(value),
            0x1DE0 => self.write_dlsame(value),
            0x1DE2 => self.write_drsame(value),
            0x1DE4 => self.write_mldiff(value),
            0x1DE6 => self.write_mrdiff(value),
            0x1DE8 => self.write_mlcomb3(value),
            0x1DEA => self.write_mrcomb3(value),
            0x1DEC => self.write_mlcomb4(value),
            0x1DEE => self.write_mrcomb4(value),
            0x1DF0 => self.write_dldiff(value),
            0x1DF2 => self.write_drdiff(value),
            0x1DF4 => self.write_mlapf1(value),
            0x1DF6 => self.write_mrapf1(value),
            0x1DF8 => self.write_mlapf2(value),
            0x1DFA => self.write_mrapf2(value),
            0x1DFC => self.write_vlin(value),
            0x1DFE => self.write_vrin(value),
            _ => todo!("reverb register write {address:08X} {value:04X}"),
        }
    }

    // $1F801DC0: APF offset 1 (dAPF1)
    fn write_dapf1(&mut self, value: u32) {
        self.apf_offset_1 = parse_address(value);
        log::trace!("dAPF1: {:05X}", self.apf_offset_1);
    }

    // $1F801DC2: APF offset 2 (dAPF2)
    fn write_dapf2(&mut self, value: u32) {
        self.apf_offset_2 = parse_address(value);
        log::trace!("dAPF2: {:05X}", self.apf_offset_2);
    }

    // $1F801DC4: Reflection volume 1 (vIIR)
    fn write_viir(&mut self, value: u32) {
        self.reflection_volume_1 = parse_volume(value);
        log::trace!("vIIR: {}", self.reflection_volume_1);
    }

    // $1F801DC6: Comb volume 1 (vCOMB1)
    fn write_vcomb1(&mut self, value: u32) {
        self.comb_volumes[0] = parse_volume(value);
        log::trace!("vCOMB1: {}", self.comb_volumes[0]);
    }

    // $1F801DC8: Comb volume 2 (vCOMB2)
    fn write_vcomb2(&mut self, value: u32) {
        self.comb_volumes[1] = parse_volume(value);
        log::trace!("vCOMB2: {}", self.comb_volumes[1]);
    }

    // $1F801DCA: Comb volume 3 (vCOMB3)
    fn write_vcomb3(&mut self, value: u32) {
        self.comb_volumes[2] = parse_volume(value);
        log::trace!("vCOMB3: {}", self.comb_volumes[2]);
    }

    // $1F801DCC: Comb volume 4 (vCOMB4)
    fn write_vcomb4(&mut self, value: u32) {
        self.comb_volumes[3] = parse_volume(value);
        log::trace!("vCOMB4: {}", self.comb_volumes[3]);
    }

    // $1F801DCE: Reflection volume 2 (vWALL)
    fn write_vwall(&mut self, value: u32) {
        self.reflection_volume_2 = parse_volume(value);
        log::trace!("vWALL: {}", self.reflection_volume_2);
    }

    // $1F801DD0: APF volume 1 (vAPF1)
    fn write_vapf1(&mut self, value: u32) {
        self.apf_volume_1 = parse_volume(value);
        log::trace!("vAPF1: {}", self.apf_volume_1);
    }

    // $1F801DD2: APF volume 2 (vAPF2)
    fn write_vapf2(&mut self, value: u32) {
        self.apf_volume_2 = parse_volume(value);
        log::trace!("vAPF2: {}", self.apf_volume_2);
    }

    // $1F801DD4: Same-side reflection address 1 left (mLSAME)
    fn write_mlsame(&mut self, value: u32) {
        self.same_reflect_addr_1.l = parse_address(value);
        log::trace!("mLSAME: {:05X}", self.same_reflect_addr_1.l);
    }

    // $1F801DD6: Same-side reflection address 1 right (mRSAME)
    fn write_mrsame(&mut self, value: u32) {
        self.same_reflect_addr_1.r = parse_address(value);
        log::trace!("mRSAME: {:05X}", self.same_reflect_addr_1.r);
    }

    // $1F801DD8: Comb address 1 left (mLCOMB1)
    fn write_mlcomb1(&mut self, value: u32) {
        self.comb_addrs[0].l = parse_address(value);
        log::trace!("mLCOMB1: {:05X}", self.comb_addrs[0].l);
    }

    // $1F801DDA: Comb address 1 right (mRCOMB1)
    fn write_mrcomb1(&mut self, value: u32) {
        self.comb_addrs[0].r = parse_address(value);
        log::trace!("mRCOMB1: {:05X}", self.comb_addrs[0].r);
    }

    // $1F801DDC: Comb address 2 left (mLCOMB2)
    fn write_mlcomb2(&mut self, value: u32) {
        self.comb_addrs[1].l = parse_address(value);
        log::trace!("mLCOMB2: {:05X}", self.comb_addrs[1].l);
    }

    // $1F801DDE: Comb address 2 right (mRCOMB2)
    fn write_mrcomb2(&mut self, value: u32) {
        self.comb_addrs[1].r = parse_address(value);
        log::trace!("mRCOMB2: {:05X}", self.comb_addrs[1].r);
    }

    // $1F801DE0: Same-side reflection address 2 left (dLSAME)
    fn write_dlsame(&mut self, value: u32) {
        self.same_reflect_addr_2.l = parse_address(value);
        log::trace!("dLSAME: {:05X}", self.same_reflect_addr_2.l);
    }

    // $1F801DE2: Same-side reflection address 2 right (dRSAME)
    fn write_drsame(&mut self, value: u32) {
        self.same_reflect_addr_2.r = parse_address(value);
        log::trace!("dRSAME: {:05X}", self.same_reflect_addr_2.r);
    }

    // $1F801DE4: Different-side reflection address 1 left (mLDIFF)
    fn write_mldiff(&mut self, value: u32) {
        self.diff_reflect_addr_1.l = parse_address(value);
        log::trace!("mLDIFF: {:05X}", self.diff_reflect_addr_1.l);
    }

    // $1F801DE6: Different-side reflection address 1 right (mRDIFF)
    fn write_mrdiff(&mut self, value: u32) {
        self.diff_reflect_addr_1.r = parse_address(value);
        log::trace!("mRDIFF: {:05X}", self.diff_reflect_addr_1.r);
    }

    // $1F801DE8: Comb address 3 left (mLCOMB3)
    fn write_mlcomb3(&mut self, value: u32) {
        self.comb_addrs[2].l = parse_address(value);
        log::trace!("mLCOMB3: {:05X}", self.comb_addrs[2].l);
    }

    // $1F801DEA: Comb address 3 right (mRCOMB3)
    fn write_mrcomb3(&mut self, value: u32) {
        self.comb_addrs[2].r = parse_address(value);
        log::trace!("mRCOMB3: {:05X}", self.comb_addrs[2].r);
    }

    // $1F801DEC: Comb address 4 left (mLCOMB4)
    fn write_mlcomb4(&mut self, value: u32) {
        self.comb_addrs[3].l = parse_address(value);
        log::trace!("mLCOMB4: {:05X}", self.comb_addrs[3].l);
    }

    // $1F801DEE: Comb address 4 right (mRCOMB4)
    fn write_mrcomb4(&mut self, value: u32) {
        self.comb_addrs[3].r = parse_address(value);
        log::trace!("mRCOMB4: {:05X}", self.comb_addrs[3].r);
    }

    // $1F801DF0: Different-side reflection address 2 left (dLDIFF)
    fn write_dldiff(&mut self, value: u32) {
        self.diff_reflect_addr_2.l = parse_address(value);
        log::trace!("dLDIFF: {:05X}", self.diff_reflect_addr_2.l);
    }

    // $1F801DF2: Different-side reflection address 2 right (dRDIFF)
    fn write_drdiff(&mut self, value: u32) {
        self.diff_reflect_addr_2.r = parse_address(value);
        log::trace!("dRDIFF: {:05X}", self.diff_reflect_addr_2.r);
    }

    // $1F801DF4: APF address 1 left (mLAPF1)
    fn write_mlapf1(&mut self, value: u32) {
        self.apf_addr_1.l = parse_address(value);
        log::trace!("mLAPF1: {:05X}", self.apf_addr_1.l);
    }

    // $1F801DF6: APF address 1 right (mRAPF1)
    fn write_mrapf1(&mut self, value: u32) {
        self.apf_addr_1.r = parse_address(value);
        log::trace!("mRAPF1: {:05X}", self.apf_addr_1.r);
    }

    // $1F801DF8: APF address 2 left (mLAPF2)
    fn write_mlapf2(&mut self, value: u32) {
        self.apf_addr_2.l = parse_address(value);
        log::trace!("mLAPF2: {:05X}", self.apf_addr_2.l);
    }

    // $1F801DFA: APF address 2 right (mRAPF2)
    fn write_mrapf2(&mut self, value: u32) {
        self.apf_addr_2.r = parse_address(value);
        log::trace!("mRAPF2: {:05X}", self.apf_addr_2.r);
    }

    // $1F801DFC: Input volume left (vLIN)
    fn write_vlin(&mut self, value: u32) {
        self.input_volume.l = parse_volume(value);
        log::trace!("vLIN: {}", self.input_volume.l);
    }

    // $1F801DFE: Input volume right (vRIN)
    fn write_vrin(&mut self, value: u32) {
        self.input_volume.r = parse_volume(value);
        log::trace!("vRIN: {}", self.input_volume.r);
    }
}

fn parse_address(value: u32) -> u32 {
    // All address registers are in 8-byte units
    (value & 0xFFFF) << 3
}

fn reverse_address(address: u32) -> u32 {
    address >> 3
}

fn parse_volume(value: u32) -> i32 {
    // All volume registers are signed 16-bit values
    (value as i16).into()
}
