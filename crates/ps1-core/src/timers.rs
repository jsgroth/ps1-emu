//! PS1 hardware timers, as well as the video clock timer
//!
//! There are 3 hardware timers:
//! - Timer 0: Can track the CPU clock or the GPU dot clock
//! - Timer 1: Can track the CPU clock or GPU horizontal retraces (i.e. a scanline counter)
//! - Timer 2: Tracks the CPU clock, either as-is or divided by 8
//!
//! The GPU has a clock rate of 53.693175 MHz (NTSC) or 53.203425 MHz (PAL), with the following
//! video timings:
//!
//! Lines per frame:
//! - NTSC interlaced: 262.5
//! - NTSC progressive: 263
//! - PAL interlaced: 312.5
//! - PAL progressive: 314
//!
//! GPU clocks per line:
//! - NTSC: 3412.5
//! - PAL: 3405
//!
//! Refresh rate:
//! - NTSC interlaced: 59.940 Hz
//! - NTSC progressive: 59.826 Hz
//! - PAL interlaced: 50.000 Hz
//! - PAL progressive: 49.761 Hz

#[cfg(test)]
mod tests;

use crate::gpu::VideoMode;
use crate::interrupts::{InterruptRegisters, InterruptType};
use crate::num::U32Ext;
use crate::scheduler::{Scheduler, SchedulerEvent, SchedulerEventType};
use bincode::{Decode, Encode};
use std::cmp;
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum IrqRepeatMode {
    #[default]
    Once = 0,
    Repeat = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum IrqPulseMode {
    #[default]
    ShortPulse = 0,
    Toggle = 1,
}

#[derive(Debug, Clone, Encode, Decode)]
struct SystemTimer {
    interrupt_type: InterruptType,
    counter: u16,
    target: u16,
    reset_at_target: bool,
    irq_at_target: bool,
    irq_at_max: bool,
    irq_repeat_mode: IrqRepeatMode,
    irq_pulse_mode: IrqPulseMode,
    irq: bool,
    irq_since_mode_write: bool,
    reached_target: bool,
    reached_max: bool,
}

impl SystemTimer {
    fn new(interrupt_type: InterruptType) -> Self {
        Self {
            interrupt_type,
            counter: 0,
            target: 0,
            reset_at_target: false,
            irq_at_target: false,
            irq_at_max: false,
            irq_repeat_mode: IrqRepeatMode::default(),
            irq_pulse_mode: IrqPulseMode::default(),
            irq: false,
            irq_since_mode_write: false,
            reached_target: false,
            reached_max: false,
        }
    }

    fn clock(&mut self, clocks: u64, interrupt_registers: &mut InterruptRegisters) {
        if clocks == 0 {
            return;
        }

        if self.reset_at_target && self.target != 0xFFFF {
            if self.irq_at_target
                && self.irq_repeat_mode == IrqRepeatMode::Repeat
                && self.irq_pulse_mode == IrqPulseMode::Toggle
            {
                self.clock_reset_at_target_toggle_irq(clocks, interrupt_registers);
            } else {
                self.clock_reset_at_target_pulse_irq(clocks, interrupt_registers);
            }
        } else if (self.irq_at_target || self.irq_at_max)
            && self.irq_repeat_mode == IrqRepeatMode::Repeat
            && self.irq_pulse_mode == IrqPulseMode::Toggle
        {
            self.clock_reset_at_max_toggle_irq(clocks, interrupt_registers);
        } else {
            self.clock_reset_at_max_pulse_irq(clocks, interrupt_registers);
        }
    }

    // Reset-at-target is set and repeating toggle IRQs are not enabled
    fn clock_reset_at_target_pulse_irq(
        &mut self,
        mut clocks: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        // Reset counter to 0 if at or above target
        if self.counter >= self.target {
            self.counter = 0;
            clocks -= 1;
        }

        let clocks_till_target: u64 = (self.target - self.counter).into();
        match clocks.cmp(&clocks_till_target) {
            Ordering::Less => {
                self.counter += clocks as u16;
            }
            Ordering::Equal => {
                self.counter = self.target;
                self.reached_target = true;
                if self.irq_at_target {
                    self.trigger_irq(interrupt_registers);
                }
            }
            Ordering::Greater => {
                self.reached_target = true;
                if self.irq_at_target {
                    self.trigger_irq(interrupt_registers);
                }

                clocks -= clocks_till_target + 1;
                self.counter = (clocks % (u64::from(self.target) + 1)) as u16;
            }
        }
    }

    // Reset-at-target is set and repeating toggle IRQs are enabled
    fn clock_reset_at_target_toggle_irq(
        &mut self,
        mut clocks: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        assert!(self.irq_at_target);

        // Reset counter to 0 if at or above target
        if self.counter >= self.target {
            self.counter = 0;
            clocks -= 1;
        }

        let clocks_till_target: u64 = (self.target - self.counter).into();
        match clocks.cmp(&clocks_till_target) {
            Ordering::Less => {
                self.counter += clocks as u16;
            }
            Ordering::Equal => {
                self.counter = self.target;
                self.reached_target = true;
                self.trigger_irq(interrupt_registers);
            }
            Ordering::Greater => {
                self.reached_target = true;
                self.trigger_irq(interrupt_registers);

                clocks -= clocks_till_target + 1;

                let target: u64 = self.target.into();
                while clocks >= target {
                    self.trigger_irq(interrupt_registers);
                    if clocks == target {
                        self.counter = self.target;
                        return;
                    }

                    clocks -= target + 1;
                }

                self.counter = clocks as u16;
            }
        }
    }

    // Reset-at-target is not set (or target is $FFFF) and repeating toggle IRQs are not enabled
    fn clock_reset_at_max_pulse_irq(
        &mut self,
        mut clocks: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        // Reset counter to 0 if already at max
        if self.counter == 0xFFFF {
            self.counter = 0;
            clocks -= 1;

            if self.target == 0 {
                self.reached_target = true;
                if self.irq_at_target {
                    self.trigger_irq(interrupt_registers);
                }
            }
        }

        let reached_max = clocks >= u64::from(0xFFFF - self.counter);

        let reached_target = (self.target > self.counter
            && clocks >= u64::from(self.target - self.counter))
            || (self.target <= self.counter
                && clocks >= 0x10000 - u64::from(self.counter - self.target));

        self.reached_max |= reached_max;
        self.reached_target |= reached_target;

        if (self.irq_at_target && reached_target) || (self.irq_at_max && reached_max) {
            self.trigger_irq(interrupt_registers);
        }

        self.counter = self.counter.wrapping_add(clocks as u16);
    }

    // Reset-at-target is not set (or target is $FFFF) and repeating toggle IRQs are enabled
    fn clock_reset_at_max_toggle_irq(
        &mut self,
        mut clocks: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        // If target is $FFFF, target IRQs should probably only fire if max IRQs are not enabled?
        let target_irqs_enabled = self.irq_at_target && (self.target != 0xFFFF || !self.irq_at_max);

        // Check if counter will not reach max
        if clocks < u64::from(0xFFFF - self.counter) {
            let reached_target =
                self.target > self.counter && clocks >= u64::from(self.target - self.counter);
            self.reached_target |= reached_target;
            if target_irqs_enabled {
                self.trigger_irq(interrupt_registers);
            }

            self.counter = self.counter.wrapping_add(clocks as u16);
            return;
        }

        // Only trigger IRQs on initial advance if counter was not already at max
        if self.counter != 0xFFFF {
            let reached_target_start = self.target > self.counter;
            self.reached_target |= reached_target_start;
            self.reached_max = true;

            if target_irqs_enabled && reached_target_start {
                self.trigger_irq(interrupt_registers);
            }

            if self.irq_at_max {
                self.trigger_irq(interrupt_registers);
            }
        }

        // Advance counter to $FFFF
        clocks -= u64::from(0xFFFF - self.counter);
        self.counter = 0xFFFF;

        // Each 65536-step loop will hit both the target and the max
        while clocks >= 0x10000 {
            self.reached_target = true;

            if target_irqs_enabled {
                self.trigger_irq(interrupt_registers);
            }

            if self.irq_at_max {
                self.trigger_irq(interrupt_registers);
            }

            clocks -= 0x10000;
        }

        if clocks == 0 {
            return;
        }

        self.counter = (clocks - 1) as u16;

        let reached_target_end = self.target <= self.counter;
        self.reached_target |= reached_target_end;
        if target_irqs_enabled && reached_target_end {
            self.trigger_irq(interrupt_registers);
        }

        let reached_max_end = self.counter == 0xFFFF;
        self.reached_max |= reached_max_end;
        if self.irq_at_max && reached_max_end {
            self.trigger_irq(interrupt_registers);
        }
    }

    fn trigger_irq(&mut self, interrupt_registers: &mut InterruptRegisters) {
        if self.irq_repeat_mode == IrqRepeatMode::Once && self.irq_since_mode_write {
            return;
        }

        self.irq_since_mode_write = true;

        match self.irq_pulse_mode {
            IrqPulseMode::ShortPulse => {
                interrupt_registers.set_interrupt_flag(self.interrupt_type);
            }
            IrqPulseMode::Toggle => {
                self.irq = !self.irq;
                if self.irq {
                    interrupt_registers.set_interrupt_flag(self.interrupt_type);
                }
            }
        }
    }

    fn clocks_until_irq(&self) -> Option<u64> {
        if !self.irq_at_target && !self.irq_at_max {
            // IRQs not enabled
            return None;
        }

        if self.irq_repeat_mode == IrqRepeatMode::Once && self.irq_since_mode_write {
            // IRQ set to one-shot mode and has already fired
            return None;
        }

        let clocks_until_target = self.irq_at_target.then(|| {
            if self.target > self.counter {
                u64::from(self.target - self.counter)
            } else if self.reset_at_target {
                u64::from(self.target) + 1
            } else {
                0x10000 - u64::from(self.counter - self.target)
            }
        });

        let clocks_until_max =
            (self.irq_at_max && (!self.reset_at_target || self.target == 0xFFFF)).then(|| {
                if self.counter == 0xFFFF { 0x10000 } else { u64::from(0xFFFF - self.counter) }
            });

        match (clocks_until_target, clocks_until_max) {
            (Some(a), Some(b)) => Some(cmp::min(a, b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }

    fn write_counter(&mut self, value: u32) {
        self.counter = value as u16;
        log::debug!("  Counter: {:04X}", self.counter);
    }

    fn write_target(&mut self, value: u32) {
        self.target = value as u16;
        log::debug!("  Target value: {:04X}", self.target);
    }

    fn read_mode(&mut self) -> u32 {
        let mode = (u32::from(self.reset_at_target) << 3)
            | (u32::from(self.irq_at_target) << 4)
            | (u32::from(self.irq_at_max) << 5)
            | ((self.irq_repeat_mode as u32) << 6)
            | ((self.irq_pulse_mode as u32) << 7)
            | (u32::from(!self.irq) << 10)
            | (u32::from(self.reached_target) << 11)
            | (u32::from(self.reached_max) << 12);

        self.reached_target = false;
        self.reached_max = false;

        mode
    }

    fn write_mode(&mut self, value: u32) {
        self.reset_at_target = value.bit(3);
        self.irq_at_target = value.bit(4);
        self.irq_at_max = value.bit(5);
        self.irq_repeat_mode =
            if value.bit(6) { IrqRepeatMode::Repeat } else { IrqRepeatMode::Once };
        self.irq_pulse_mode =
            if value.bit(7) { IrqPulseMode::Toggle } else { IrqPulseMode::ShortPulse };

        if value.bit(10) || self.irq_pulse_mode == IrqPulseMode::ShortPulse {
            self.irq = false;
        }

        self.counter = 0;
        self.irq_since_mode_write = false;

        log::debug!("  Reset at target: {}", self.reset_at_target);
        log::debug!("  IRQ at target: {}", self.irq_at_target);
        log::debug!("  IRQ at max: {}", self.irq_at_max);
        log::debug!("  IRQ repeat mode: {:?}", self.irq_repeat_mode);
        log::debug!("  IRQ pulse mode: {:?}", self.irq_pulse_mode);
    }
}

const NTSC_CYCLES_PER_LINE: u64 = 3412;
const NTSC_LINES_PER_FRAME: u16 = 263;
const NTSC_GPU_CLOCK: u64 = 53_693_175;

const PAL_CYCLES_PER_LINE: u64 = 3405;
const PAL_LINES_PER_FRAME_INTERLACED: u16 = 313;
const PAL_LINES_PER_FRAME_PROGRESSIVE: u16 = 314;
const PAL_GPU_CLOCK: u64 = 53_203_425;

const CPU_CLOCK: u64 = 33_868_800;

impl VideoMode {
    const fn gpu_clock(self) -> u64 {
        match self {
            Self::Ntsc => NTSC_GPU_CLOCK,
            Self::Pal => PAL_GPU_CLOCK,
        }
    }

    const fn lines_per_frame(self, interlaced: bool) -> u16 {
        match (self, interlaced) {
            (Self::Ntsc, _) => NTSC_LINES_PER_FRAME,
            (Self::Pal, true) => PAL_LINES_PER_FRAME_INTERLACED,
            (Self::Pal, false) => PAL_LINES_PER_FRAME_PROGRESSIVE,
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
struct GpuTimer {
    cycle_product: u64,
    line: u16,
    line_cycle: u64,
    x1: u64,
    x2: u64,
    y1: u16,
    y2: u16,
    video_mode: VideoMode,
    dot_clock_divider: u64,
    interlaced: bool,
    odd_frame: bool,
}

impl GpuTimer {
    fn new() -> Self {
        Self {
            cycle_product: 0,
            line: 0,
            line_cycle: 0,
            x1: 0x200,
            x2: 0x200 + 2560,
            y1: 16,
            y2: 256,
            video_mode: VideoMode::default(),
            dot_clock_divider: 10,
            interlaced: false,
            odd_frame: false,
        }
    }

    fn cycles_in_line(&self) -> u64 {
        match self.video_mode {
            VideoMode::Ntsc => {
                // Emulate 3412.5 cycles per line by using the lowest bit of the line to conditionally add 1 cycle.
                // This is not _exactly_ accurate since there are an odd number of lines per frame, but it
                // should be close enough
                let base_dots = if self.interlaced && self.line == NTSC_LINES_PER_FRAME - 1 {
                    NTSC_CYCLES_PER_LINE / 2
                } else {
                    NTSC_CYCLES_PER_LINE
                };
                base_dots + u64::from(self.line & 1)
            }
            VideoMode::Pal => {
                if self.interlaced && self.line == PAL_LINES_PER_FRAME_INTERLACED - 1 {
                    PAL_CYCLES_PER_LINE / 2
                } else {
                    PAL_CYCLES_PER_LINE
                }
            }
        }
    }

    fn increment_line(&mut self) {
        self.line += 1;
        if self.line >= self.video_mode.lines_per_frame(self.interlaced) {
            self.line = 0;
            self.odd_frame = !self.odd_frame;
        }
    }

    fn in_vblank(&self) -> bool {
        self.y1 >= self.y2 || !(self.y1..self.y2).contains(&self.line)
    }

    fn at_vblank_start(&self) -> bool {
        self.y1 < self.y2 && self.line == self.y2 && self.line_cycle == 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum Timer0ClockSource {
    #[default]
    System,
    Dot,
}

impl Timer0ClockSource {
    fn clocks(self, dot_clocks: u64, gpu_clocks: u64, video_mode: VideoMode) -> u64 {
        match self {
            Self::System => gpu_clocks * CPU_CLOCK / video_mode.gpu_clock(),
            Self::Dot => dot_clocks,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum Timer1ClockSource {
    #[default]
    System,
    HRetrace,
}

impl Timer1ClockSource {
    fn clocks(self, h_retrace_clocks: u64, gpu_clocks: u64, video_mode: VideoMode) -> u64 {
        match self {
            Self::System => gpu_clocks * CPU_CLOCK / video_mode.gpu_clock(),
            Self::HRetrace => h_retrace_clocks,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum Timer2ClockSource {
    #[default]
    System,
    SystemDiv8,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum Timer01SyncMode {
    #[default]
    PauseDuringBlank,
    ResetAtBlank,
    PauseOutsideBlank,
    PauseTillNextBlank,
}

impl Timer01SyncMode {
    fn from_raw(raw_sync_mode: u8) -> Self {
        match raw_sync_mode & 3 {
            0 => Self::PauseDuringBlank,
            1 => Self::ResetAtBlank,
            2 => Self::PauseOutsideBlank,
            3 => Self::PauseTillNextBlank,
            _ => unreachable!("value & 3 is always <= 3"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
enum Timer2SyncMode {
    #[default]
    FreeRun,
    Stop,
}

#[derive(Debug, Clone, Copy)]
pub struct GpuStatus {
    pub in_vblank: bool,
    pub odd_scanline: bool,
    pub odd_frame: bool,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Timers {
    last_update_cycles: u64,
    timers: [SystemTimer; 3],
    gpu: GpuTimer,
    timer_0_clock_source: Timer0ClockSource,
    timer_1_clock_source: Timer1ClockSource,
    timer_2_clock_source: Timer2ClockSource,
    sync_enabled: [bool; 3],
    timer_0_sync_mode: Timer01SyncMode,
    timer_1_sync_mode: Timer01SyncMode,
    timer_2_sync_mode: Timer2SyncMode,
    raw_clock_sources: [u8; 3],
    raw_sync_modes: [u8; 3],
}

impl Timers {
    pub fn new() -> Self {
        Self {
            last_update_cycles: 0,
            timers: [
                SystemTimer::new(InterruptType::Timer0),
                SystemTimer::new(InterruptType::Timer1),
                SystemTimer::new(InterruptType::Timer2),
            ],
            gpu: GpuTimer::new(),
            timer_0_clock_source: Timer0ClockSource::default(),
            timer_1_clock_source: Timer1ClockSource::default(),
            timer_2_clock_source: Timer2ClockSource::default(),
            sync_enabled: [false; 3],
            timer_0_sync_mode: Timer01SyncMode::default(),
            timer_1_sync_mode: Timer01SyncMode::default(),
            timer_2_sync_mode: Timer2SyncMode::default(),
            raw_clock_sources: [0; 3],
            raw_sync_modes: [0; 3],
        }
    }

    pub fn catch_up(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        let cpu_elapsed = scheduler.cpu_cycle_counter() - self.last_update_cycles;
        if cpu_elapsed == 0 {
            return;
        }

        if !self.sync_enabled[0] && !self.sync_enabled[1] {
            self.catch_up_gpu_no_sync(cpu_elapsed, interrupt_registers);

            if self.timer_0_clock_source == Timer0ClockSource::System {
                self.timers[0].clock(cpu_elapsed, interrupt_registers);
            }

            if self.timer_1_clock_source == Timer1ClockSource::System {
                self.timers[1].clock(cpu_elapsed, interrupt_registers);
            }
        } else {
            self.catch_up_gpu_with_sync(cpu_elapsed, interrupt_registers);
        }

        if !(self.sync_enabled[2] && self.timer_2_sync_mode == Timer2SyncMode::Stop) {
            match self.timer_2_clock_source {
                Timer2ClockSource::System => {
                    self.timers[2].clock(cpu_elapsed, interrupt_registers);
                }
                Timer2ClockSource::SystemDiv8 => {
                    let elapsed_div_8 =
                        scheduler.cpu_cycle_counter() / 8 - self.last_update_cycles / 8;
                    self.timers[2].clock(elapsed_div_8, interrupt_registers);
                }
            }
        }

        self.last_update_cycles = scheduler.cpu_cycle_counter();
    }

    // Catch up the video clock timer with no timer 0/1 synchronization modes enabled.
    // Will clock timers 0 and 1 if they are tracking the GPU clocks but not if they are tracking
    // the CPU clock
    fn catch_up_gpu_no_sync(
        &mut self,
        cpu_elapsed: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.gpu.cycle_product += cpu_elapsed * self.gpu.video_mode.gpu_clock();
        let mut gpu_elapsed = self.gpu.cycle_product / CPU_CLOCK;
        self.gpu.cycle_product %= CPU_CLOCK;

        // Check if there aren't enough GPU cycles to reach the end of the current line
        if gpu_elapsed < self.gpu.cycles_in_line() - self.gpu.line_cycle {
            if self.timer_0_clock_source == Timer0ClockSource::Dot {
                let dot_clocks = (self.gpu.line_cycle + gpu_elapsed) / self.gpu.dot_clock_divider
                    - self.gpu.line_cycle / self.gpu.dot_clock_divider;
                self.timers[0].clock(dot_clocks, interrupt_registers);
            }

            self.gpu.line_cycle += gpu_elapsed;

            return;
        }

        // Advance to start of next line
        let mut dot_clocks = self.gpu.cycles_in_line() / self.gpu.dot_clock_divider
            - self.gpu.line_cycle / self.gpu.dot_clock_divider;
        let mut h_retrace_clocks = 1;
        gpu_elapsed -= self.gpu.cycles_in_line() - self.gpu.line_cycle;
        self.gpu.line_cycle = 0;

        loop {
            self.gpu.increment_line();

            // Check if this is the last line reached
            let cycles_in_line = self.gpu.cycles_in_line();
            if gpu_elapsed < cycles_in_line {
                self.gpu.line_cycle = gpu_elapsed;
                dot_clocks += gpu_elapsed / self.gpu.dot_clock_divider;

                // Clock timers 0 and 1 if they are tracking GPU clocks
                if self.timer_0_clock_source == Timer0ClockSource::Dot {
                    self.timers[0].clock(dot_clocks, interrupt_registers);
                }

                if self.timer_1_clock_source == Timer1ClockSource::HRetrace {
                    self.timers[1].clock(h_retrace_clocks, interrupt_registers);
                }

                return;
            }

            dot_clocks += cycles_in_line / self.gpu.dot_clock_divider;
            h_retrace_clocks += 1;
            gpu_elapsed -= cycles_in_line;
        }
    }

    // Catch up the video timer with timer 0/1 synchronization modes possibly enabled
    fn catch_up_gpu_with_sync(
        &mut self,
        cpu_elapsed: u64,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.gpu.cycle_product += cpu_elapsed * self.gpu.video_mode.gpu_clock();
        let mut gpu_elapsed = self.gpu.cycle_product / CPU_CLOCK;
        self.gpu.cycle_product %= CPU_CLOCK;

        let h_range_valid = self.gpu.x1 < self.gpu.x2;

        let mut dot_clocks = 0;
        let mut h_retrace_clocks = 0;
        let mut timer_0_gpu_clocks = 0;
        let mut timer_1_gpu_clocks = 0;

        let mut in_vblank = self.gpu.in_vblank();

        while gpu_elapsed != 0 {
            if h_range_valid
                && self.gpu.line_cycle < self.gpu.x1
                && self.gpu.x1 < self.gpu.cycles_in_line()
            {
                // In HBlank (left border), skip to end of HBlank
                let gpu_clocks = cmp::min(self.gpu.x1 - self.gpu.line_cycle, gpu_elapsed);
                let new_line_cycle = self.gpu.line_cycle + gpu_clocks;

                if !is_timer_paused(InBlank::Yes, self.sync_enabled[0], self.timer_0_sync_mode) {
                    dot_clocks += new_line_cycle / self.gpu.dot_clock_divider
                        - self.gpu.line_cycle / self.gpu.dot_clock_divider;
                    timer_0_gpu_clocks += gpu_clocks;
                }

                if !is_timer_paused(in_vblank.into(), self.sync_enabled[1], self.timer_1_sync_mode)
                {
                    timer_1_gpu_clocks += gpu_clocks;
                }

                self.gpu.line_cycle = new_line_cycle;
                gpu_elapsed -= gpu_clocks;
            } else if h_range_valid
                && self.gpu.line_cycle < self.gpu.x2
                && self.gpu.x2 < self.gpu.cycles_in_line()
            {
                // In active display, skip to start of HBlank
                let gpu_clocks = cmp::min(self.gpu.x2 - self.gpu.line_cycle, gpu_elapsed);
                let new_line_cycle = self.gpu.line_cycle + gpu_clocks;

                if !is_timer_paused(InBlank::No, self.sync_enabled[0], self.timer_0_sync_mode) {
                    dot_clocks += new_line_cycle / self.gpu.dot_clock_divider
                        - self.gpu.line_cycle / self.gpu.dot_clock_divider;
                    timer_0_gpu_clocks += gpu_clocks;
                }

                if !is_timer_paused(in_vblank.into(), self.sync_enabled[1], self.timer_1_sync_mode)
                {
                    timer_1_gpu_clocks += gpu_clocks;
                }

                self.gpu.line_cycle = new_line_cycle;
                gpu_elapsed -= gpu_clocks;

                // Process start-of-HBlank sync events
                if self.sync_enabled[0] && self.gpu.line_cycle == self.gpu.x2 {
                    match self.timer_0_sync_mode {
                        Timer01SyncMode::ResetAtBlank | Timer01SyncMode::PauseOutsideBlank => {
                            // Clock timer 0 to current position and then reset
                            self.timers[0].clock(
                                self.timer_0_clock_source.clocks(
                                    dot_clocks,
                                    timer_0_gpu_clocks,
                                    self.gpu.video_mode,
                                ),
                                interrupt_registers,
                            );

                            dot_clocks = 0;
                            timer_0_gpu_clocks = 0;
                            self.timers[0].counter = 0;
                        }
                        Timer01SyncMode::PauseTillNextBlank => {
                            // Disable timer 0 synchronization
                            self.sync_enabled[0] = false;
                        }
                        Timer01SyncMode::PauseDuringBlank => {}
                    }
                }
            } else {
                // In HBlank (right border), skip to start of next line
                let gpu_clocks =
                    cmp::min(self.gpu.cycles_in_line() - self.gpu.line_cycle, gpu_elapsed);
                gpu_elapsed -= gpu_clocks;

                let new_line_cycle = self.gpu.line_cycle + gpu_clocks;
                let reached_end_of_line = new_line_cycle == self.gpu.cycles_in_line();

                if !is_timer_paused(InBlank::Yes, self.sync_enabled[0], self.timer_0_sync_mode) {
                    dot_clocks += new_line_cycle / self.gpu.dot_clock_divider
                        - self.gpu.line_cycle / self.gpu.dot_clock_divider;
                    timer_0_gpu_clocks += gpu_clocks;
                }

                if !is_timer_paused(in_vblank.into(), self.sync_enabled[1], self.timer_1_sync_mode)
                {
                    if reached_end_of_line {
                        h_retrace_clocks += 1;
                    }
                    timer_1_gpu_clocks += gpu_clocks;
                }

                if reached_end_of_line {
                    self.gpu.line_cycle = 0;
                    self.gpu.increment_line();
                    in_vblank = self.gpu.in_vblank();
                } else {
                    self.gpu.line_cycle = new_line_cycle;
                }

                // Process start-of-VBlank sync events
                if self.gpu.at_vblank_start() && self.sync_enabled[1] {
                    match self.timer_1_sync_mode {
                        Timer01SyncMode::ResetAtBlank | Timer01SyncMode::PauseOutsideBlank => {
                            // Clock timer 1 to current position and then reset
                            self.timers[1].clock(
                                self.timer_1_clock_source.clocks(
                                    h_retrace_clocks,
                                    timer_1_gpu_clocks,
                                    self.gpu.video_mode,
                                ),
                                interrupt_registers,
                            );

                            h_retrace_clocks = 0;
                            timer_1_gpu_clocks = 0;
                            self.timers[1].counter = 0;
                        }
                        Timer01SyncMode::PauseTillNextBlank => {
                            // Disable timer 1 synchronization
                            self.sync_enabled[1] = false;
                        }
                        Timer01SyncMode::PauseDuringBlank => {}
                    }
                }
            }
        }

        self.timers[0].clock(
            self.timer_0_clock_source.clocks(dot_clocks, timer_0_gpu_clocks, self.gpu.video_mode),
            interrupt_registers,
        );
        self.timers[1].clock(
            self.timer_1_clock_source.clocks(
                h_retrace_clocks,
                timer_1_gpu_clocks,
                self.gpu.video_mode,
            ),
            interrupt_registers,
        );
    }

    pub fn get_gpu_status(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) -> GpuStatus {
        self.catch_up(scheduler, interrupt_registers);

        GpuStatus {
            in_vblank: self.gpu.in_vblank(),
            odd_scanline: !self.gpu.line.is_multiple_of(2),
            odd_frame: self.gpu.odd_frame,
        }
    }

    pub fn update_horizontal_display_range(
        &mut self,
        x1: u16,
        x2: u16,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        self.gpu.x1 = x1.into();
        self.gpu.x2 = x2.into();
    }

    pub fn update_vertical_display_range(
        &mut self,
        y1: u16,
        y2: u16,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        let prev_y2 = self.gpu.y2;
        self.gpu.y1 = y1;
        self.gpu.y2 = y2;

        if prev_y2 != self.gpu.y2 {
            self.schedule_next_vblank(scheduler, interrupt_registers);
        }
    }

    pub fn update_display_mode(
        &mut self,
        dot_clock_divider: u16,
        interlaced: bool,
        video_mode: VideoMode,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        let prev_interlaced = self.gpu.interlaced;
        let prev_video_mode = self.gpu.video_mode;
        self.gpu.dot_clock_divider = dot_clock_divider.into();
        self.gpu.interlaced = interlaced;
        self.gpu.video_mode = video_mode;

        if prev_interlaced != self.gpu.interlaced || prev_video_mode != self.gpu.video_mode {
            if self.gpu.line >= self.gpu.video_mode.lines_per_frame(self.gpu.interlaced)
                || self.gpu.line_cycle >= self.gpu.cycles_in_line()
            {
                self.gpu.line_cycle = 0;
                self.gpu.increment_line();
            }

            self.schedule_next_vblank(scheduler, interrupt_registers);
        }
    }

    pub fn schedule_timer_events(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        match self.timers[0].clocks_until_irq() {
            Some(clocks) => {
                let cpu_cycles = match self.timer_0_clock_source {
                    Timer0ClockSource::System => clocks,
                    Timer0ClockSource::Dot => {
                        // This is not exactly right (will underestimate the number of GPU clocks),
                        // but should be close enough
                        let gpu_cycles = clocks * self.gpu.dot_clock_divider;
                        gpu_cycles * CPU_CLOCK / self.gpu.video_mode.gpu_clock() + 1
                    }
                };
                scheduler.update_or_push_event(SchedulerEvent {
                    event_type: SchedulerEventType::Timer0Irq,
                    cpu_cycles: scheduler.cpu_cycle_counter() + cpu_cycles,
                });
            }
            None => {
                scheduler.remove_event(SchedulerEventType::Timer0Irq);
            }
        }

        match self.timers[1].clocks_until_irq() {
            Some(clocks) => {
                let cpu_cycles = match self.timer_1_clock_source {
                    Timer1ClockSource::System => clocks,
                    Timer1ClockSource::HRetrace => {
                        // This not exactly right but should be close enough
                        // 6825 == 3412.5 * 2
                        let gpu_cycles = clocks * 6825 / 2;
                        gpu_cycles * CPU_CLOCK / self.gpu.video_mode.gpu_clock() + 1
                    }
                };
                scheduler.update_or_push_event(SchedulerEvent {
                    event_type: SchedulerEventType::Timer1Irq,
                    cpu_cycles: scheduler.cpu_cycle_counter() + cpu_cycles,
                });
            }
            None => {
                scheduler.remove_event(SchedulerEventType::Timer1Irq);
            }
        }

        match self.timers[2].clocks_until_irq() {
            Some(clocks) => {
                let cpu_cycles = match self.timer_2_clock_source {
                    Timer2ClockSource::System => clocks,
                    Timer2ClockSource::SystemDiv8 => {
                        clocks * 8 - (scheduler.cpu_cycle_counter() % 8)
                    }
                };
                scheduler.update_or_push_event(SchedulerEvent {
                    event_type: SchedulerEventType::Timer2Irq,
                    cpu_cycles: scheduler.cpu_cycle_counter() + cpu_cycles,
                });
            }
            None => {
                scheduler.remove_event(SchedulerEventType::Timer2Irq);
            }
        }
    }

    pub fn schedule_next_vblank(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        if self.gpu.y1 >= self.gpu.y2 {
            // Invalid vertical display range
            scheduler.remove_event(SchedulerEventType::VBlank);
            return;
        }

        self.catch_up(scheduler, interrupt_registers);

        let mut gpu_cycles = self.gpu.cycles_in_line() - self.gpu.line_cycle;

        let cycles_per_line = match self.gpu.video_mode {
            VideoMode::Ntsc => NTSC_CYCLES_PER_LINE,
            VideoMode::Pal => PAL_CYCLES_PER_LINE,
        };

        let is_ntsc = u16::from(self.gpu.video_mode == VideoMode::Ntsc);

        let start_y = if self.gpu.line < self.gpu.y2 { self.gpu.line + 1 } else { 0 };
        gpu_cycles += (start_y..self.gpu.y2)
            .map(|y| cycles_per_line + u64::from(is_ntsc & y & 1))
            .sum::<u64>();

        let lines_per_frame = self.gpu.video_mode.lines_per_frame(self.gpu.interlaced);
        if self.gpu.line >= self.gpu.y2 && self.gpu.line != lines_per_frame - 1 {
            gpu_cycles += if self.gpu.interlaced { cycles_per_line / 2 } else { cycles_per_line };
            gpu_cycles += (self.gpu.line + 1..lines_per_frame - 1)
                .map(|y| cycles_per_line + u64::from(is_ntsc & y & 1))
                .sum::<u64>();
        }

        let cpu_cycles = gpu_cycles * CPU_CLOCK / self.gpu.video_mode.gpu_clock() + 1;
        scheduler.update_or_push_event(SchedulerEvent::vblank(
            scheduler.cpu_cycle_counter() + cpu_cycles,
        ));
    }

    pub fn read_register(
        &mut self,
        address: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) -> u32 {
        self.catch_up(scheduler, interrupt_registers);

        let timer_idx = ((address >> 4) & 3) as usize;
        if timer_idx == 3 {
            return 0;
        }

        log::trace!("Timer {timer_idx} read: {address:08X}");

        match address & 0xF {
            0x0 => self.timers[timer_idx].counter.into(),
            0x4 => self.handle_mode_read(timer_idx),
            0x8 => self.timers[timer_idx].target.into(),
            _ => todo!("timer read {address:08X}"),
        }
    }

    pub fn write_register(
        &mut self,
        address: u32,
        value: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        let timer_idx = ((address >> 4) & 3) as usize;
        if timer_idx == 3 {
            return;
        }

        log::debug!("Timer {timer_idx} write: {address:08X} {value:04X}");

        match address & 0xF {
            0x0 => self.timers[timer_idx].write_counter(value),
            0x4 => self.handle_mode_write(timer_idx, value),
            0x8 => self.timers[timer_idx].write_target(value),
            _ => todo!("timer write {address:08X} {value:04X}"),
        }

        self.schedule_timer_events(scheduler, interrupt_registers);
    }

    fn handle_mode_read(&mut self, timer_idx: usize) -> u32 {
        let base_mode = u32::from(self.sync_enabled[timer_idx])
            | (u32::from(self.raw_sync_modes[timer_idx]) << 1)
            | (u32::from(self.raw_clock_sources[timer_idx]) << 8);

        base_mode | self.timers[timer_idx].read_mode()
    }

    fn handle_mode_write(&mut self, timer_idx: usize, value: u32) {
        self.timers[timer_idx].write_mode(value);

        self.sync_enabled[timer_idx] = value.bit(0);

        let raw_sync_mode = ((value >> 1) & 3) as u8;
        self.raw_sync_modes[timer_idx] = raw_sync_mode;

        match timer_idx {
            0 => {
                self.timer_0_sync_mode = Timer01SyncMode::from_raw(raw_sync_mode);
            }
            1 => {
                self.timer_1_sync_mode = Timer01SyncMode::from_raw(raw_sync_mode);
            }
            2 => {
                self.timer_2_sync_mode = match raw_sync_mode {
                    0 | 3 => Timer2SyncMode::Stop,
                    1 | 2 => Timer2SyncMode::FreeRun,
                    _ => unreachable!(),
                }
            }
            _ => panic!("Invalid timer index {timer_idx}, should be 0/1/2"),
        }

        let raw_clock_source = ((value >> 8) & 3) as u8;
        self.raw_clock_sources[timer_idx] = raw_clock_source;

        match timer_idx {
            0 => {
                self.timer_0_clock_source = match raw_clock_source {
                    0 | 2 => Timer0ClockSource::System,
                    1 | 3 => Timer0ClockSource::Dot,
                    _ => unreachable!(),
                }
            }
            1 => {
                self.timer_1_clock_source = match raw_clock_source {
                    0 | 2 => Timer1ClockSource::System,
                    1 | 3 => Timer1ClockSource::HRetrace,
                    _ => unreachable!(),
                }
            }
            2 => {
                self.timer_2_clock_source = match raw_clock_source {
                    0 | 1 => Timer2ClockSource::System,
                    2 | 3 => Timer2ClockSource::SystemDiv8,
                    _ => unreachable!(),
                }
            }
            _ => panic!("Invalid timer index {timer_idx}, should be 0/1/2"),
        }

        log::debug!("  Synchronization enabled: {:?}", self.sync_enabled);
        log::debug!(
            "  Sync modes: [{:?} {:?} {:?}]",
            self.timer_0_sync_mode,
            self.timer_1_sync_mode,
            self.timer_2_sync_mode
        );
        log::debug!(
            "  Clock sources: [{:?} {:?} {:?}]",
            self.timer_0_clock_source,
            self.timer_1_clock_source,
            self.timer_2_clock_source
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InBlank {
    No,
    Yes,
}

impl From<bool> for InBlank {
    fn from(value: bool) -> Self {
        if value { Self::Yes } else { Self::No }
    }
}

fn is_timer_paused(in_blank: InBlank, sync_enabled: bool, sync_mode: Timer01SyncMode) -> bool {
    if !sync_enabled {
        return false;
    }

    matches!(
        (in_blank, sync_mode),
        (InBlank::Yes, Timer01SyncMode::PauseDuringBlank | Timer01SyncMode::PauseTillNextBlank)
            | (InBlank::No, Timer01SyncMode::PauseOutsideBlank)
    )
}
