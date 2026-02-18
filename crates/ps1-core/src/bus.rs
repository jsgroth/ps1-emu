//! PS1 memory map

use crate::cd::CdController;
use crate::cpu::OpSize;
use crate::dma::DmaController;
use crate::gpu::Gpu;
use crate::interrupts::InterruptRegisters;
use crate::mdec::MacroblockDecoder;
use crate::memory::{Memory, MemoryControl};
use crate::pgxp::PreciseVertex;
use crate::scheduler::Scheduler;
use crate::sio::{SerialPort0, SerialPort1};
use crate::spu::Spu;
use crate::timers::Timers;

pub struct Bus<'a> {
    pub gpu: &'a mut Gpu,
    pub spu: &'a mut Spu,
    pub cd_controller: &'a mut CdController,
    pub mdec: &'a mut MacroblockDecoder,
    pub memory: &'a mut Memory,
    pub memory_control: &'a mut MemoryControl,
    pub dma_controller: &'a mut DmaController,
    pub interrupt_registers: &'a mut InterruptRegisters,
    pub sio0: &'a mut SerialPort0,
    pub sio1: &'a mut SerialPort1,
    pub timers: &'a mut Timers,
    pub scheduler: &'a mut Scheduler,
}

macro_rules! memory_map {
    ($address:expr, [
        main_ram => $main_ram:expr,
        expansion_1 => $expansion_1:expr,
        scratchpad => $scratchpad:expr,
        io_registers => $io_registers:expr,
        $(expansion_2 => $expansion_2:expr,)?
        $(bios => $bios:expr,)?
        _ => $default:expr $(,)?
    ]) => {
        match $address {
            0x00000000..=0x007FFFFF => $main_ram,
            0x1F000000..=0x1F7FFFFF => $expansion_1,
            0x1F800000..=0x1F800FFF => $scratchpad,
            0x1F801000..=0x1F801FFF => $io_registers,
            $(0x1F802000..=0x1F803FFF => $expansion_2,)?
            $(0x1FC00000..=0x1FFFFFFF => $bios,)?
            _ => $default
        }
    }
}

impl Bus<'_> {
    // TODO memory control for main RAM and BIOS ROM
    pub fn read_u8(&mut self, address: u32) -> u32 {
        memory_map!(address, [
            main_ram => self.memory.read_main_ram_u8(address).into(),
            expansion_1 => {
                log::debug!("Unhandled 8-bit expansion 1 read {address:08X}");
                0
            },
            scratchpad => self.memory.read_scratchpad_u8(address).into(),
            io_registers => self.read_io_register(address, OpSize::Byte),
            bios => self.memory.read_bios_u8(address).into(),
            _ => todo!("8-bit read {address:08X}")
        ])
    }

    pub fn read_u16(&mut self, address: u32) -> u32 {
        memory_map!(address, [
            main_ram => self.memory.read_main_ram_u16(address).into(),
            expansion_1 => {
                log::debug!("Unhandled 16-bit expansion 1 read {address:08X}");
                0
            },
            scratchpad => self.memory.read_scratchpad_u16(address).into(),
            io_registers => self.read_io_register(address, OpSize::HalfWord),
            bios => self.memory.read_bios_u16(address).into(),
            _ => todo!("16-bit read {address:08X}")
        ])
    }

    pub fn read_u32(&mut self, address: u32) -> u32 {
        memory_map!(address, [
            main_ram => self.memory.read_main_ram_u32(address),
            expansion_1 => {
                log::debug!("Unhandled 32-bit expansion 1 read {address:08X}");
                0
            },
            scratchpad => self.memory.read_scratchpad_u32(address),
            io_registers => self.read_io_register(address, OpSize::Word),
            bios => self.memory.read_bios_u32(address),
            _ => todo!("32-bit read {address:08X}")
        ])
    }

    pub fn read_pgxp(&self, address: u32) -> PreciseVertex {
        match address & 0x1FFFFFFF {
            0x00000000..=0x007FFFFF => self.memory.read_main_ram_pgxp(address),
            0x1F800000..=0x1F8003FF => self.memory.read_scratchpad_pgxp(address),
            _ => PreciseVertex::INVALID,
        }
    }

    pub fn write_u8(&mut self, address: u32, value: u32) {
        memory_map!(address, [
            main_ram => self.memory.write_main_ram_u8(address, value as u8),
            expansion_1 => unimplemented_expansion_write("Expansion Device 1", address, value, OpSize::Byte),
            scratchpad => self.memory.write_scratchpad_u8(address, value as u8),
            io_registers => self.write_io_register(address, value, OpSize::Byte),
            expansion_2 => unimplemented_expansion_write("Expansion Device 2", address, value, OpSize::Byte),
            _ => todo!("8-bit write {address:08X} {value:08X}")
        ]);
    }

    pub fn write_u16(&mut self, address: u32, value: u32) {
        memory_map!(address, [
            main_ram => self.memory.write_main_ram_u16(address, value as u16),
            expansion_1 => unimplemented_expansion_write("Expansion Device 1", address, value, OpSize::HalfWord),
            scratchpad => self.memory.write_scratchpad_u16(address, value as u16),
            io_registers => self.write_io_register(address, value, OpSize::HalfWord),
            expansion_2 => unimplemented_expansion_write("Expansion Device 2", address, value, OpSize::HalfWord),
            _ => todo!("16-bit write {address:08X} {value:08X}")
        ]);
    }

    pub fn write_u32(&mut self, address: u32, value: u32) {
        memory_map!(address, [
            main_ram => self.memory.write_main_ram_u32(address, value),
            expansion_1 => unimplemented_expansion_write("Expansion Device 1", address, value, OpSize::Word),
            scratchpad => self.memory.write_scratchpad_u32(address, value),
            io_registers => self.write_io_register(address, value, OpSize::Word),
            expansion_2 => unimplemented_expansion_write("Expansion Device 2", address, value, OpSize::Word),
            _ => todo!("32-bit write {address:08X} {value:08X}")
        ]);
    }

    pub fn write_pgxp(&mut self, address: u32, vertex: PreciseVertex) {
        match address & 0x1FFFFFFF {
            0x00000000..=0x007FFFFF => self.memory.write_main_ram_pgxp(address, vertex),
            0x1F800000..=0x1F8003FF => self.memory.write_scratchpad_pgxp(address, vertex),
            _ => {}
        }
    }

    pub fn hardware_interrupt_pending(&self) -> bool {
        self.interrupt_registers.interrupt_pending()
    }

    #[allow(clippy::match_same_arms)]
    fn read_io_register(&mut self, address: u32, size: OpSize) -> u32 {
        log::debug!("I/O register read: {address:08X} {size:?}");

        match address & 0xFFFF {
            0x1000..=0x1020 => self.memory_control.read_register(address),
            0x1040 => self.sio0.read_rx_data(),
            0x1044 => self.sio0.read_status(self.scheduler, self.interrupt_registers),
            0x104A => self.sio0.read_control(),
            0x104E => self.sio0.read_baudrate_reload(),
            0x1050 => self.sio1.read_rx_data(),
            0x1054 => self.sio1.read_status(self.scheduler, self.interrupt_registers),
            0x105A => self.sio1.read_control(),
            0x105E => self.sio1.read_baudrate_reload(),
            0x1060 => self.memory_control.read_ram_size(),
            0x1070 => self.interrupt_registers.read_interrupt_status(),
            0x1074 => self.interrupt_registers.read_interrupt_mask(),
            0x10F0 => self.dma_controller.read_control(),
            0x1080..=0x10EF => match (address >> 2) & 3 {
                0 => self.dma_controller.read_channel_address(address),
                1 => self.dma_controller.read_channel_length(address),
                2 => self.dma_controller.read_channel_control(address),
                _ => todo!("DMA register read {address:08X} {size:?}"),
            },
            0x10F4 => self.dma_controller.read_interrupt(),
            0x10F6 => self.dma_controller.read_interrupt() >> 16,
            0x1100..=0x113F => {
                self.timers.read_register(address, self.scheduler, self.interrupt_registers)
            }
            0x1800..=0x1803 => read_cd_controller(self.cd_controller, address, size),
            0x1810 => self.gpu.read_port(),
            0x1814 => {
                self.gpu.read_status_register(self.timers, self.scheduler, self.interrupt_registers)
            }
            0x1820 => self.mdec.read_data(),
            0x1824 => self.mdec.read_status(),
            0x1C00..=0x1FFF => self.spu.read_register(address, size),
            _ => todo!("I/O register read {address:08X} {size:?}"),
        }
    }

    fn write_io_register(&mut self, address: u32, value: u32, size: OpSize) {
        log::debug!("I/O register write: {address:08X} {value:08X} {size:?}");

        // TODO figure out how to properly handle 8-bit/16-bit writes to I/O registers
        match address & 0xFFFF {
            0x1000..=0x1020 => self.memory_control.write_register(address, value),
            0x1040 => self.sio0.write_tx_data(value, self.scheduler, self.interrupt_registers),
            0x1048 => self.sio0.write_mode(value, self.scheduler, self.interrupt_registers),
            0x104A => self.sio0.write_control(value, self.scheduler, self.interrupt_registers),
            0x104E => {
                self.sio0.write_baudrate_reload(value, self.scheduler, self.interrupt_registers);
            }
            0x1050 => self.sio1.write_tx_data(value, self.scheduler, self.interrupt_registers),
            0x1058 => self.sio1.write_mode(value, self.scheduler, self.interrupt_registers),
            0x105A => self.sio1.write_control(value, self.scheduler, self.interrupt_registers),
            0x105E => {
                self.sio1.write_baudrate_reload(value, self.scheduler, self.interrupt_registers);
            }
            0x1060 => self.memory_control.write_ram_size(value),
            0x1070 => self.interrupt_registers.write_interrupt_status(value),
            0x1074 => self.interrupt_registers.write_interrupt_mask(value),
            0x1080..=0x10EF => match (address >> 2) & 3 {
                0 => self.dma_controller.write_channel_address(address, value),
                1 => self.dma_controller.write_channel_length(address, value),
                2 => self.dma_controller.write_channel_control(address, value, self.scheduler),
                3 => todo!("Invalid DMA register write: {address:08X} {value:08X} {size:?}"),
                _ => unreachable!("value & 3 is always <= 3"),
            },
            0x10F0 => self.dma_controller.write_control(value, self.scheduler),
            0x10F4 => self.dma_controller.write_interrupt(value, self.interrupt_registers),
            0x10F6 => self.dma_controller.write_interrupt(
                (size.mask(value) << 16) | (self.dma_controller.read_interrupt() & 0xFFFF),
                self.interrupt_registers,
            ),
            0x1100..=0x112F => {
                self.timers.write_register(
                    address,
                    value,
                    self.scheduler,
                    self.interrupt_registers,
                );
            }
            0x1800..=0x1803 => self.cd_controller.write_port(address, value as u8),
            0x1810 => self.gpu.write_gp0_command(value),
            0x1814 => self.gpu.write_gp1_command(
                value,
                self.timers,
                self.scheduler,
                self.interrupt_registers,
            ),
            0x1820 => self.mdec.write_command(value),
            0x1824 => self.mdec.write_control(value),
            0x1C00..=0x1FFF => self.spu.write_register(address, value, size),
            _ => todo!("I/O register write {address:08X} {value:08X} {size:?}"),
        }
    }
}

fn read_cd_controller(cd_controller: &mut CdController, address: u32, size: OpSize) -> u32 {
    match size {
        OpSize::Byte => cd_controller.read_port(address).into(),
        OpSize::HalfWord => {
            // 16-bit reads simply perform two consecutive 8-bit reads
            let lsb = cd_controller.read_port(address);
            let msb = cd_controller.read_port(address);
            u16::from_le_bytes([lsb, msb]).into()
        }
        OpSize::Word => {
            // 32-bit reads simply perform four consecutive 8-bit reads.
            // Due to alignment, 32-bit reads can only access the status register which should not
            // change between reads. Simply duplicate the byte four times
            let byte = cd_controller.read_port(address);
            u32::from_le_bytes([byte, byte, byte, byte])
        }
    }
}

fn unimplemented_expansion_write(name: &str, address: u32, value: u32, size: OpSize) {
    log::debug!("Unimplemented {name} write: {address:08X} {value:08X} {size:?}");
}
