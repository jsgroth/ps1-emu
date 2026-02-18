//! PS1 Macroblock Decoder (MDEC), a hardware image decompressor
//!
//! Implementation largely based on <https://psx-spx.consoledev.net/macroblockdecodermdec/>

mod tables;

use crate::num::U32Ext;
use bincode::{Decode, Encode};
use std::collections::VecDeque;
use std::mem;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum DepthBits {
    #[default]
    Four = 0,
    Eight = 1,
    TwentyFour = 2,
    Fifteen = 3,
}

impl DepthBits {
    fn from_bits(value: u32) -> Self {
        match value & 3 {
            0 => Self::Four,
            1 => Self::Eight,
            2 => Self::TwentyFour,
            3 => Self::Fifteen,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
struct DecodeConfig {
    depth: DepthBits,
    signed: bool,
    bit_15: bool,
    parameters_remaining: u16,
}

impl Default for DecodeConfig {
    fn default() -> Self {
        Self { depth: DepthBits::default(), signed: false, bit_15: false, parameters_remaining: 1 }
    }
}

impl DecodeConfig {
    fn from_command(command: u32) -> Self {
        Self {
            depth: DepthBits::from_bits(command >> 27),
            signed: command.bit(26),
            bit_15: command.bit(25),
            parameters_remaining: command as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, Encode, Decode)]
enum CommandState {
    Idle,
    ReceivingCompressedData,
    ReceivingLuminanceTable { bytes_remaining: u8, color_table_after: bool },
    ReceivingColorTable { bytes_remaining: u8 },
    ReceivingScaleTable { halfwords_remaining: u8 },
}

#[derive(Debug, Clone, Copy, Default, Encode, Decode)]
struct Color {
    r: i16,
    g: i16,
    b: i16,
}

#[derive(Debug, Clone, Encode, Decode)]
struct Buffers {
    cr_block: [i32; 64],
    cb_block: [i32; 64],
    y_block: [i32; 64],
    idct_buffer: [i32; 64],
    color_out_buffer: [Color; 256],
    mono_out_buffer: [u8; 64],
}

impl Default for Buffers {
    fn default() -> Self {
        Self {
            cr_block: [0; 64],
            cb_block: [0; 64],
            y_block: [0; 64],
            idct_buffer: [0; 64],
            color_out_buffer: [Color::default(); 256],
            mono_out_buffer: [0; 64],
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct MacroblockDecoder {
    command_state: CommandState,
    decode_config: DecodeConfig,
    data_in: VecDeque<u16>,
    data_out: VecDeque<u8>,
    enable_data_in: bool,
    enable_data_out: bool,
    luminance_quant_table: [u8; 64],
    color_quant_table: [u8; 64],
    scale_table: [i16; 64],
    buffers: Box<Buffers>,
}

macro_rules! decode_ctx {
    ($self:expr, color) => {
        DecodeContext {
            idct_buffer: &mut $self.buffers.idct_buffer,
            data_in: &mut $self.data_in,
            quant_table: &$self.color_quant_table,
            scale_table: &$self.scale_table,
        }
    };
    ($self:expr, luminance) => {
        DecodeContext {
            idct_buffer: &mut $self.buffers.idct_buffer,
            data_in: &mut $self.data_in,
            quant_table: &$self.luminance_quant_table,
            scale_table: &$self.scale_table,
        }
    };
}

impl MacroblockDecoder {
    pub fn new() -> Self {
        Self {
            command_state: CommandState::Idle,
            decode_config: DecodeConfig::default(),
            data_in: VecDeque::with_capacity(2 * 65536),
            data_out: VecDeque::with_capacity(2 * 4 * 65536),
            enable_data_in: false,
            enable_data_out: false,
            luminance_quant_table: [0; 64],
            color_quant_table: [0; 64],
            scale_table: [0; 64],
            buffers: Box::default(),
        }
    }

    // $1F801820 R: MDEC data/response register
    pub fn read_data(&mut self) -> u32 {
        let mut bytes = [0; 4];
        for byte in &mut bytes {
            *byte = self.data_out.pop_front().unwrap_or(0);
        }
        u32::from_le_bytes(bytes)
    }

    // $1F801820 W: MDEC command/parameter register
    pub fn write_command(&mut self, value: u32) {
        log::trace!("MDEC command write: {value:08X}");

        self.command_state = match self.command_state {
            CommandState::Idle => match value >> 29 {
                // MDEC(0) is a no-op and MDEC(4-7) are invalid commands
                0 | 4..=7 => {
                    log::debug!("MDEC no-op command: {value:08X}");
                    CommandState::Idle
                }
                // MDEC(1): Decode macroblocks
                1 => {
                    self.decode_config = DecodeConfig::from_command(value);

                    log::debug!("MDEC decode command: {value:08X}");
                    log::debug!("  Decode config: {:?}", self.decode_config);

                    if self.decode_config.parameters_remaining != 0 {
                        CommandState::ReceivingCompressedData
                    } else {
                        CommandState::Idle
                    }
                }
                // MDEC(2): Set quant tables
                2 => {
                    log::debug!("MDEC set quant tables command: {value:08X}");
                    CommandState::ReceivingLuminanceTable {
                        bytes_remaining: 64,
                        color_table_after: value.bit(0),
                    }
                }
                // MDEC(3): Set scale table
                3 => {
                    log::debug!("MDEC set scale table command: {value:08X}");
                    CommandState::ReceivingScaleTable { halfwords_remaining: 64 }
                }
                _ => unreachable!("u32 >> 29 is always 0-7"),
            },
            CommandState::ReceivingCompressedData => {
                self.push_data_word(value);

                self.decode_config.parameters_remaining -= 1;
                if self.decode_config.parameters_remaining == 0 {
                    self.decode_macroblocks();
                    CommandState::Idle
                } else {
                    CommandState::ReceivingCompressedData
                }
            }
            CommandState::ReceivingLuminanceTable { mut bytes_remaining, color_table_after } => {
                self.push_data_word(value);

                bytes_remaining -= 4;
                if bytes_remaining == 0 {
                    self.populate_luminance_table();
                    if color_table_after {
                        CommandState::ReceivingColorTable { bytes_remaining: 64 }
                    } else {
                        CommandState::Idle
                    }
                } else {
                    CommandState::ReceivingLuminanceTable { bytes_remaining, color_table_after }
                }
            }
            CommandState::ReceivingColorTable { mut bytes_remaining } => {
                self.push_data_word(value);

                bytes_remaining -= 4;
                if bytes_remaining == 0 {
                    self.populate_color_table();
                    CommandState::Idle
                } else {
                    CommandState::ReceivingColorTable { bytes_remaining }
                }
            }
            CommandState::ReceivingScaleTable { mut halfwords_remaining } => {
                self.push_data_word(value);

                halfwords_remaining -= 2;
                if halfwords_remaining == 0 {
                    self.populate_scale_table();
                    CommandState::Idle
                } else {
                    CommandState::ReceivingScaleTable { halfwords_remaining }
                }
            }
        };
    }

    fn push_data_word(&mut self, word: u32) {
        self.data_in.push_back(word as u16);
        self.data_in.push_back((word >> 16) as u16);
    }

    fn decode_macroblocks(&mut self) {
        self.data_out.clear();

        match self.decode_config.depth {
            DepthBits::TwentyFour | DepthBits::Fifteen => self.decode_colored_macroblocks(),
            DepthBits::Four | DepthBits::Eight => self.decode_monochrome_macroblocks(),
        }
    }

    fn decode_colored_macroblocks(&mut self) {
        let mut count = 0;
        loop {
            if !decode_block(&mut self.buffers.cr_block, decode_ctx!(self, color)) {
                break;
            }

            decode_block(&mut self.buffers.cb_block, decode_ctx!(self, color));

            for (base_row, base_col) in [(0, 0), (0, 8), (8, 0), (8, 8)] {
                decode_block(&mut self.buffers.y_block, decode_ctx!(self, luminance));
                yuv_to_rgb(base_col, base_row, self.decode_config.signed, &mut self.buffers);
            }

            match self.decode_config.depth {
                DepthBits::TwentyFour => {
                    for row in 0..16 {
                        for col in 0..16 {
                            let Color { r, g, b } = self.buffers.color_out_buffer[16 * row + col];
                            self.data_out.push_back(r as u8);
                            self.data_out.push_back(g as u8);
                            self.data_out.push_back(b as u8);
                        }
                    }
                }
                DepthBits::Fifteen => {
                    for row in 0..16 {
                        for col in 0..16 {
                            let Color { r, g, b } = self.buffers.color_out_buffer[16 * row + col];

                            let r = (r as u8) >> 3;
                            let g = (g as u8) >> 3;
                            let b = (b as u8) >> 3;
                            let rgb555_color = u16::from(r)
                                | (u16::from(g) << 5)
                                | (u16::from(b) << 10)
                                | (u16::from(self.decode_config.bit_15) << 15);

                            let [lsb, msb] = rgb555_color.to_le_bytes();
                            self.data_out.push_back(lsb);
                            self.data_out.push_back(msb);
                        }
                    }
                }
                DepthBits::Four | DepthBits::Eight => {
                    panic!("decode_colored_macroblocks() called in 4bpp/8bpp mode")
                }
            }

            count += 1;
        }

        log::debug!("Decoded {count} 16x16 colored macroblocks");
    }

    fn decode_monochrome_macroblocks(&mut self) {
        let mut count = 0;

        loop {
            if !decode_block(&mut self.buffers.y_block, decode_ctx!(self, luminance)) {
                break;
            }

            y_to_mono(self.decode_config.signed, &mut self.buffers);

            match self.decode_config.depth {
                DepthBits::Four => {
                    for chunk in self.buffers.mono_out_buffer.chunks_exact(2) {
                        let byte = (chunk[0] >> 4) | (chunk[1] & 0xF0);
                        self.data_out.push_back(byte);
                    }
                }
                DepthBits::Eight => {
                    for &byte in &self.buffers.mono_out_buffer {
                        self.data_out.push_back(byte);
                    }
                }
                DepthBits::Fifteen | DepthBits::TwentyFour => {
                    panic!("decode_monochrome_macroblocks called with 15bpp/24bpp color")
                }
            }

            count += 1;
        }

        log::debug!("Decoded {count} 8x8 monochrome macroblocks");
    }

    fn populate_luminance_table(&mut self) {
        log::debug!("Populating luminance quant table");

        for (i, &halfword) in self.data_in.iter().enumerate() {
            self.luminance_quant_table[2 * i..2 * (i + 1)].copy_from_slice(&halfword.to_le_bytes());
        }
        self.data_in.clear();
    }

    fn populate_color_table(&mut self) {
        log::debug!("Populating color quant table");

        for (i, &halfword) in self.data_in.iter().enumerate() {
            self.color_quant_table[2 * i..2 * (i + 1)].copy_from_slice(&halfword.to_le_bytes());
        }
        self.data_in.clear();
    }

    fn populate_scale_table(&mut self) {
        log::debug!("Populating scale table");

        for (i, &halfword) in self.data_in.iter().enumerate() {
            self.scale_table[i] = halfword as i16;
        }
        self.data_in.clear();
    }

    pub fn data_out_request(&self) -> bool {
        self.enable_data_out && !self.data_out.is_empty()
    }

    // $1F801824 R: MDEC status register
    pub fn read_status(&self) -> u32 {
        // TODO bit 30: data in FIFO full
        // TODO bits 18-16: current block (hardcoded to 4)
        let command_busy =
            !matches!(self.command_state, CommandState::Idle) || !self.data_out.is_empty();

        let value = (u32::from(self.data_out.is_empty()) << 31)
            | (u32::from(command_busy) << 29)
            | (u32::from(self.data_out_request()) << 27)
            | (u32::from(self.enable_data_in) << 28)
            | ((self.decode_config.depth as u32) << 25)
            | (u32::from(self.decode_config.signed) << 24)
            | (u32::from(self.decode_config.bit_15) << 23)
            | 0x00040000
            | u32::from(self.decode_config.parameters_remaining.wrapping_sub(1));

        log::debug!("MDEC status read: {value:08X}");

        value
    }

    // $1F801824 W: MDEC control/reset register
    pub fn write_control(&mut self, value: u32) {
        if value.bit(31) {
            // TODO actually do reset
            // aborts all commands and makes status read $80040000
            self.command_state = CommandState::Idle;
            self.decode_config = DecodeConfig::default();
            self.data_in.clear();
            self.data_out.clear();
        }

        self.enable_data_in = value.bit(30);
        self.enable_data_out = value.bit(29);

        log::debug!("MDEC control write: {value:08X}");
        log::debug!("  MDEC reset: {}", value.bit(31));
        log::debug!("  Data in request enabled: {}", self.enable_data_in);
        log::debug!("  Data out request enabled: {}", self.enable_data_out);
    }
}

struct DecodeContext<'a> {
    idct_buffer: &'a mut [i32; 64],
    data_in: &'a mut VecDeque<u16>,
    quant_table: &'a [u8; 64],
    scale_table: &'a [i16; 64],
}

fn decode_block(
    block: &mut [i32; 64],
    DecodeContext { idct_buffer, data_in, quant_table, scale_table }: DecodeContext<'_>,
) -> bool {
    block.fill(0);

    while data_in.front().copied() == Some(0xFE00) {
        // Padding words, skip
        data_in.pop_front();
    }

    let Some(mut hw) = data_in.pop_front() else { return false };

    let q_scale = hw >> 10;

    let mut value = i10(hw) * i32::from(quant_table[0]);

    let mut k = 0;
    loop {
        if q_scale == 0 {
            value = i10(hw) * 2;
        }

        value = value.clamp(-0x400, 0x3FF);
        // value = (f64::from(value) * tables::SCALE_ZAG[k as usize]).round() as i32;

        if q_scale > 0 {
            block[tables::ZAG_ZIG[k as usize] as usize] = value;
        } else {
            block[k as usize] = value;
        }

        if k == 63 {
            break;
        }

        hw = data_in.pop_front().unwrap_or(0xFE00);
        k += (hw >> 10) + 1;

        if k >= 64 {
            break;
        }

        value = (i10(hw) * i32::from(quant_table[k as usize]) * i32::from(q_scale) + 4) / 8;
    }

    idct_core(block, idct_buffer, scale_table);

    true
}

fn idct_core(block: &mut [i32; 64], buffer: &mut [i32; 64], scale_table: &[i16; 64]) {
    let mut src = block;
    let mut dst = buffer;

    for _ in 0..2 {
        for x in 0..8 {
            for y in 0..8 {
                let mut sum = 0;
                for z in 0..8 {
                    sum += src[8 * z + y] * (i32::from(scale_table[8 * z + x]) / 8);
                }
                dst[8 * y + x] = (sum + 0xFFF) / 0x2000;
            }
        }

        mem::swap(&mut src, &mut dst);
    }
}

fn yuv_to_rgb(base_col: usize, base_row: usize, signed: bool, buffers: &mut Buffers) {
    for row in 0..8 {
        for col in 0..8 {
            let mut r = buffers.cr_block
                [8 * usize::midpoint(base_row, row) + usize::midpoint(base_col, col)];
            let mut b = buffers.cb_block
                [8 * usize::midpoint(base_row, row) + usize::midpoint(base_col, col)];

            let g = (-0.3437 * f64::from(b) - 0.7143 * f64::from(r)).round() as i32;
            r = (1.402 * f64::from(r)).round() as i32;
            b = (1.772 * f64::from(b)).round() as i32;

            let y = buffers.y_block[8 * row + col];
            let mut r = (y + r).clamp(-128, 127) as i16;
            let mut g = (y + g).clamp(-128, 127) as i16;
            let mut b = (y + b).clamp(-128, 127) as i16;

            if !signed {
                r += 128;
                g += 128;
                b += 128;
            }

            buffers.color_out_buffer[16 * (base_row + row) + (base_col + col)] = Color { r, g, b };
        }
    }
}

fn y_to_mono(signed: bool, buffers: &mut Buffers) {
    for (i, &y) in buffers.y_block.iter().enumerate() {
        // Clip to signed 9-bit
        let y = (y << (32 - 9)) >> (32 - 9);

        // Clamp to signed 8-bit
        let mut y = y.clamp(-128, 127);

        if !signed {
            y += 128;
        }

        buffers.mono_out_buffer[i] = y as u8;
    }
}

fn i10(value: u16) -> i32 {
    (((value as i16) << 6) >> 6).into()
}
