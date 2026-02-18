//! PS1 serial I/O ports (SIO0 / SIO1)
//!
//! SIO0 is used to communicate with controllers and memory cards
//!
//! SIO1 is mostly unused, but some games used it for link cable functionality (not emulated)

mod controllers;
pub mod memcard;
mod rxfifo;

use crate::api::{LoadedMemoryCards, MemoryCardsEnabled};
use crate::input::{ControllerState, ControllerType, Ps1Inputs};
use crate::interrupts::{InterruptRegisters, InterruptType};
use crate::num::U32Ext;
use crate::scheduler::{Scheduler, SchedulerEvent, SchedulerEventType};
use crate::sio::controllers::{DigitalController, DualShock, DualShockControllerState};
use crate::sio::memcard::{ConnectedMemoryCard, MemoryCard};
use crate::sio::rxfifo::RxFifo;
use bincode::{Decode, Encode};
use std::cmp;

#[derive(Debug, Clone, Copy, Encode, Decode)]
struct BaudrateTimer {
    timer: u32,
    raw_reload_value: u32,
    reload_factor: u32,
}

impl BaudrateTimer {
    fn new() -> Self {
        Self { timer: 0x0088, raw_reload_value: 0x0088, reload_factor: 2 }
    }

    fn tick(&mut self, mut cpu_cycles: u32) {
        // TODO this is terribly inefficient; improve this
        while cpu_cycles != 0 {
            let elapsed = cmp::min(self.timer, cpu_cycles);
            self.timer -= elapsed;
            cpu_cycles -= elapsed;

            if self.timer == 0 {
                self.timer = self.reload_value();
            }
        }
    }

    fn reload_value(&mut self) -> u32 {
        cmp::max(1, self.raw_reload_value * self.reload_factor / 2)
    }

    fn update_reload_value(&mut self, value: u32) {
        self.raw_reload_value = value;

        // Updating reload value triggers an immediate reload
        self.timer = self.reload_value();
    }

    fn update_reload_factor(&mut self, value: u32) {
        self.reload_factor = match value & 3 {
            0 => 1,
            1 => 2,
            2 => 16,
            3 => 64,
            _ => unreachable!("value & 3 is always <= 3"),
        };
        log::debug!("SIO0 Baudrate timer reload factor: {}", self.reload_factor);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Encode, Decode)]
pub enum Port {
    #[default]
    One = 0,
    Two = 1,
}

impl Port {
    fn from_bit(bit: bool) -> Self {
        if bit { Self::Two } else { Self::One }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
enum TxFifoState {
    Empty,
    Queued(u8),
    Transferring { value: u8, cycles_remaining: u32, next: Option<u8> },
}

impl TxFifoState {
    fn ready_for_new_byte(self) -> bool {
        matches!(self, Self::Empty | Self::Transferring { next: None, .. })
    }
}

const CONTROLLER_TRANSFER_CYCLES: u32 = 500;
const ACK_LOW_CYCLES: u64 = 100;

#[derive(Debug, Clone, Encode, Decode)]
enum SerialDevice<Device> {
    NoDevice,
    Disconnected,
    Connected(Device),
}

pub trait SerialDevices {
    type Device;

    fn connect(&self, tx: u8, port: Port) -> Option<Self::Device>;

    fn process_tx_write(
        &mut self,
        device: Self::Device,
        tx: u8,
        rx: &mut RxFifo,
    ) -> Option<Self::Device>;
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Sio0Devices {
    p1_joypad_state: ControllerState,
    p2_joypad_state: ControllerState,
    p1_dualshock_state: DualShockControllerState,
    p2_dualshock_state: DualShockControllerState,
    memory_card_1: Option<MemoryCard>,
    memory_card_2: Option<MemoryCard>,
}

impl Sio0Devices {
    fn new(
        memory_cards_enabled: MemoryCardsEnabled,
        loaded_memory_cards: LoadedMemoryCards,
    ) -> Self {
        Self {
            p1_joypad_state: ControllerState::default_p1(),
            p2_joypad_state: ControllerState::default_p2(),
            p1_dualshock_state: DualShockControllerState::default(),
            p2_dualshock_state: DualShockControllerState::default(),
            memory_card_1: memory_cards_enabled
                .slot_1
                .then(|| MemoryCard::new(loaded_memory_cards.slot_1)),
            memory_card_2: memory_cards_enabled
                .slot_2
                .then(|| MemoryCard::new(loaded_memory_cards.slot_2)),
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub enum Sio0Device {
    DigitalController(DigitalController),
    DualShock(DualShock),
    MemoryCard(ConnectedMemoryCard),
}

const CONTROLLER_ADDRESS: u8 = 0x01;
const MEMORY_CARD_ADDRESS: u8 = 0x81;

impl SerialDevices for Sio0Devices {
    type Device = Sio0Device;

    fn connect(&self, tx: u8, port: Port) -> Option<Self::Device> {
        match (tx, port) {
            (CONTROLLER_ADDRESS, Port::One) => {
                initial_controller_state(Port::One, self.p1_joypad_state)
            }
            (CONTROLLER_ADDRESS, Port::Two) => {
                initial_controller_state(Port::Two, self.p2_joypad_state)
            }
            (MEMORY_CARD_ADDRESS, Port::One) if self.memory_card_1.is_some() => {
                Some(Sio0Device::MemoryCard(ConnectedMemoryCard::initial(Port::One)))
            }
            (MEMORY_CARD_ADDRESS, Port::Two) if self.memory_card_2.is_some() => {
                Some(Sio0Device::MemoryCard(ConnectedMemoryCard::initial(Port::Two)))
            }
            _ => None,
        }
    }

    fn process_tx_write(
        &mut self,
        device: Self::Device,
        tx: u8,
        rx: &mut RxFifo,
    ) -> Option<Self::Device> {
        match device {
            Sio0Device::DigitalController(controller) => {
                controller.process(tx, rx).map(Sio0Device::DigitalController)
            }
            Sio0Device::DualShock(dual_shock) => {
                let dualshock_state = match dual_shock.port {
                    Port::One => &mut self.p1_dualshock_state,
                    Port::Two => &mut self.p2_dualshock_state,
                };
                dual_shock.process(tx, rx, dualshock_state).map(Sio0Device::DualShock)
            }
            Sio0Device::MemoryCard(connected_memory_card) => {
                let memory_card = match connected_memory_card.port {
                    Port::One => self.memory_card_1.as_mut(),
                    Port::Two => self.memory_card_2.as_mut(),
                };

                memory_card
                    .and_then(|memory_card| connected_memory_card.process(tx, rx, memory_card))
                    .map(Sio0Device::MemoryCard)
            }
        }
    }
}

fn initial_controller_state(port: Port, state: ControllerState) -> Option<Sio0Device> {
    match state.controller_type {
        ControllerType::None => None,
        ControllerType::Digital => {
            Some(Sio0Device::DigitalController(DigitalController::initial(state.digital)))
        }
        ControllerType::DualShock => {
            Some(Sio0Device::DualShock(DualShock::initial(port, state.digital, state.analog)))
        }
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Sio1Devices;

impl SerialDevices for Sio1Devices {
    type Device = ();

    fn connect(&self, _tx: u8, _port: Port) -> Option<Self::Device> {
        None
    }

    fn process_tx_write(
        &mut self,
        _device: Self::Device,
        _tx: u8,
        _rx: &mut RxFifo,
    ) -> Option<Self::Device> {
        None
    }
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct SerialPort<Devices, Device> {
    devices: Devices,
    active_device: Option<SerialDevice<Device>>,
    irq_event_type: SchedulerEventType,
    tx_event_type: SchedulerEventType,
    last_update_cycles: u64,
    tx_fifo: TxFifoState,
    rx_fifo: RxFifo,
    tx_enabled: bool,
    dtr_on: bool,
    rx_enabled: bool,
    rx_interrupt_bytes: u8,
    tx_interrupt_enabled: bool,
    rx_interrupt_enabled: bool,
    dsr_interrupt_enabled: bool,
    selected_port: Port,
    baudrate_timer: BaudrateTimer,
    ack: bool,
    irq: bool,
    pending_irq7: bool,
    ack_low_cycles: u16,
}

pub type SerialPort0 = SerialPort<Sio0Devices, Sio0Device>;
pub type SerialPort1 = SerialPort<Sio1Devices, ()>;

impl SerialPort0 {
    pub fn new_sio0(
        memory_cards_enabled: MemoryCardsEnabled,
        loaded_memory_cards: LoadedMemoryCards,
    ) -> Self {
        Self::new(
            Sio0Devices::new(memory_cards_enabled, loaded_memory_cards),
            SchedulerEventType::Sio0Irq,
            SchedulerEventType::Sio0Tx,
        )
    }

    pub fn set_inputs(&mut self, inputs: Ps1Inputs) {
        update_dualshock_state(
            self.devices.p1_joypad_state,
            inputs.p1,
            &mut self.devices.p1_dualshock_state,
        );
        update_dualshock_state(
            self.devices.p2_joypad_state,
            inputs.p2,
            &mut self.devices.p2_dualshock_state,
        );

        self.devices.p1_joypad_state = inputs.p1;
        self.devices.p2_joypad_state = inputs.p2;
    }

    pub fn memory_cards(&mut self) -> (Option<&mut MemoryCard>, Option<&mut MemoryCard>) {
        (self.devices.memory_card_1.as_mut(), self.devices.memory_card_2.as_mut())
    }

    pub fn update_memory_cards(&mut self, enabled: MemoryCardsEnabled, loaded: LoadedMemoryCards) {
        self.devices.memory_card_1 = enabled.slot_1.then(|| MemoryCard::new(loaded.slot_1));
        self.devices.memory_card_2 = enabled.slot_2.then(|| MemoryCard::new(loaded.slot_2));
    }

    pub fn clone_unserialized_fields(&self) -> (MemoryCardsEnabled, LoadedMemoryCards) {
        let enabled = MemoryCardsEnabled {
            slot_1: self.devices.memory_card_1.is_some(),
            slot_2: self.devices.memory_card_2.is_some(),
        };

        let loaded = LoadedMemoryCards {
            slot_1: self.devices.memory_card_1.as_ref().map(|card| card.data().to_vec()),
            slot_2: self.devices.memory_card_2.as_ref().map(|card| card.data().to_vec()),
        };

        (enabled, loaded)
    }
}

fn update_dualshock_state(
    previous_inputs: ControllerState,
    inputs: ControllerState,
    dualshock_state: &mut DualShockControllerState,
) {
    if inputs.controller_type != previous_inputs.controller_type {
        *dualshock_state = DualShockControllerState::default();
    }

    if !previous_inputs.analog.analog_button
        && inputs.analog.analog_button
        && inputs.controller_type == ControllerType::DualShock
    {
        dualshock_state.toggle_analog_mode();
    }
}

impl SerialPort1 {
    pub fn new_sio1() -> Self {
        Self::new(Sio1Devices, SchedulerEventType::Sio1Irq, SchedulerEventType::Sio1Tx)
    }
}

impl<Devices: SerialDevices> SerialPort<Devices, Devices::Device> {
    fn new(
        devices: Devices,
        irq_event_type: SchedulerEventType,
        tx_event_type: SchedulerEventType,
    ) -> Self {
        Self {
            devices,
            active_device: None,
            irq_event_type,
            tx_event_type,
            last_update_cycles: 0,
            tx_fifo: TxFifoState::Empty,
            rx_fifo: RxFifo::new(),
            tx_enabled: false,
            dtr_on: false,
            rx_enabled: false,
            rx_interrupt_bytes: 1,
            tx_interrupt_enabled: false,
            rx_interrupt_enabled: false,
            dsr_interrupt_enabled: false,
            selected_port: Port::default(),
            baudrate_timer: BaudrateTimer::new(),
            ack: false,
            irq: false,
            pending_irq7: false,
            ack_low_cycles: 0,
        }
    }

    pub fn catch_up(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        let cycles_elapsed = (scheduler.cpu_cycle_counter() - self.last_update_cycles) as u32;
        if cycles_elapsed == 0 {
            return;
        }
        self.last_update_cycles = scheduler.cpu_cycle_counter();

        self.baudrate_timer.tick(cycles_elapsed);

        if self.ack_low_cycles != 0 {
            self.ack_low_cycles =
                u32::from(self.ack_low_cycles).saturating_sub(cycles_elapsed) as u16;
            if self.ack_low_cycles == 0 {
                if self.pending_irq7 {
                    interrupt_registers.set_interrupt_flag(InterruptType::Sio);
                }

                self.ack = false;
                self.pending_irq7 = false;
            }
        }

        match self.tx_fifo {
            TxFifoState::Empty => {}
            TxFifoState::Queued(value) => {
                if self.tx_enabled {
                    self.tx_fifo = TxFifoState::Transferring {
                        value,
                        cycles_remaining: CONTROLLER_TRANSFER_CYCLES,
                        next: None,
                    };
                    scheduler.update_or_push_event(SchedulerEvent {
                        event_type: self.tx_event_type,
                        cpu_cycles: scheduler.cpu_cycle_counter()
                            + u64::from(CONTROLLER_TRANSFER_CYCLES),
                    });
                }
            }
            TxFifoState::Transferring { value, cycles_remaining, next } => {
                let cycles_remaining = cycles_remaining.saturating_sub(cycles_elapsed);
                if cycles_remaining == 0 {
                    self.process_tx_write(value, scheduler);

                    self.tx_fifo = match (next, self.tx_enabled) {
                        (Some(next_value), true) => {
                            scheduler.update_or_push_event(SchedulerEvent {
                                event_type: self.tx_event_type,
                                cpu_cycles: scheduler.cpu_cycle_counter()
                                    + u64::from(CONTROLLER_TRANSFER_CYCLES),
                            });

                            TxFifoState::Transferring {
                                value: next_value,
                                cycles_remaining: CONTROLLER_TRANSFER_CYCLES,
                                next: None,
                            }
                        }
                        (Some(next_value), false) => TxFifoState::Queued(next_value),
                        (None, _) => TxFifoState::Empty,
                    };
                } else {
                    self.tx_fifo = TxFifoState::Transferring { value, cycles_remaining, next };
                }
            }
        }
    }

    fn process_tx_write(&mut self, value: u8, scheduler: &mut Scheduler) {
        log::debug!("Processing SIO0 TX_DATA write {value:02X}");

        self.active_device = match self.active_device.take() {
            Some(SerialDevice::NoDevice) => {
                // If software is attempting to communicate with a device that is not connected,
                // send 0 until it resets the port
                self.rx_fifo.push(0);
                Some(SerialDevice::NoDevice)
            }
            Some(SerialDevice::Disconnected) => Some(SerialDevice::Disconnected),
            Some(SerialDevice::Connected(device)) => Some(
                self.devices
                    .process_tx_write(device, value, &mut self.rx_fifo)
                    .map_or(SerialDevice::Disconnected, SerialDevice::Connected),
            ),
            None => {
                self.rx_fifo.push(0);

                Some(
                    self.devices
                        .connect(value, self.selected_port)
                        .map_or(SerialDevice::NoDevice, SerialDevice::Connected),
                )
            }
        };

        self.ack = true;
        if self.dsr_interrupt_enabled && self.device_is_connected() {
            if !self.irq {
                self.pending_irq7 = true;
            }
            self.irq = true;
        }

        self.ack_low_cycles = ACK_LOW_CYCLES as u16;
        scheduler.update_or_push_event(SchedulerEvent {
            event_type: self.irq_event_type,
            cpu_cycles: scheduler.cpu_cycle_counter() + ACK_LOW_CYCLES,
        });
    }

    fn device_is_connected(&self) -> bool {
        self.active_device.as_ref().is_some_and(|device| {
            !matches!(device, SerialDevice::NoDevice | SerialDevice::Disconnected)
        })
    }

    pub fn write_tx_data(
        &mut self,
        tx_data: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        let tx_data = tx_data as u8;

        self.tx_fifo = match self.tx_fifo {
            TxFifoState::Empty | TxFifoState::Queued(_) => {
                if self.tx_enabled {
                    scheduler.update_or_push_event(SchedulerEvent {
                        event_type: self.tx_event_type,
                        cpu_cycles: scheduler.cpu_cycle_counter()
                            + u64::from(CONTROLLER_TRANSFER_CYCLES),
                    });

                    TxFifoState::Transferring {
                        value: tx_data,
                        cycles_remaining: CONTROLLER_TRANSFER_CYCLES,
                        next: None,
                    }
                } else {
                    TxFifoState::Queued(tx_data)
                }
            }
            TxFifoState::Transferring { value, cycles_remaining, .. } => {
                scheduler.update_or_push_event(SchedulerEvent {
                    event_type: self.tx_event_type,
                    cpu_cycles: scheduler.cpu_cycle_counter() + u64::from(cycles_remaining),
                });

                TxFifoState::Transferring { value, cycles_remaining, next: Some(tx_data) }
            }
        };

        log::debug!("SIO0_TX_DATA write: {tx_data:02X}");
    }

    pub fn read_rx_data(&mut self) -> u32 {
        let value = self.rx_fifo.pop();
        log::debug!("RX_DATA read: {value:02X}");
        value.into()
    }

    // $1F801044: SIO0_STAT
    pub fn read_status(
        &mut self,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) -> u32 {
        self.catch_up(scheduler, interrupt_registers);

        let ack = self.ack && self.device_is_connected();

        let value = u32::from(self.tx_fifo.ready_for_new_byte())
            | (u32::from(!self.rx_fifo.empty()) << 1)
            | (u32::from(self.tx_fifo == TxFifoState::Empty) << 2)
            | (u32::from(ack) << 7)
            | (u32::from(self.irq) << 9)
            | (self.baudrate_timer.timer << 11);

        log::debug!("SIO0_STAT read: {value:08X}");
        value
    }

    // $1F801048: SIO0_MODE
    pub fn write_mode(
        &mut self,
        value: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        self.baudrate_timer.update_reload_factor(value);

        if value & 0xC != 0xC {
            todo!("Expected character length to be 8-bit (3), was {}", (value >> 2) & 3);
        }

        if value.bit(4) {
            todo!("Expected parity bit to be clear, was set");
        }

        if value.bit(8) {
            todo!("Expected clock polarity to be high-when-idle (0), was low-when-idle (1)");
        }
    }

    // $1F80104A: SIO0_CTRL
    pub fn read_control(&self) -> u32 {
        let rx_mode = match self.rx_interrupt_bytes {
            1 => 0,
            2 => 1,
            4 => 2,
            8 => 3,
            _ => panic!("Unexpected RX IRQ FIFO length: {}", self.rx_interrupt_bytes),
        };

        let value = u32::from(self.tx_enabled)
            | (u32::from(self.dtr_on) << 1)
            | (u32::from(self.rx_enabled) << 2)
            | (rx_mode << 8)
            | (u32::from(self.tx_interrupt_enabled) << 10)
            | (u32::from(self.rx_interrupt_enabled) << 11)
            | (u32::from(self.dsr_interrupt_enabled) << 12)
            | ((self.selected_port as u32) << 13);

        log::debug!("SIO0_CTRL read: {value:04X}");
        value
    }

    // $1F80104A: SIO0_CTRL
    pub fn write_control(
        &mut self,
        value: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        if value.bit(6) {
            // Reset bit
            log::debug!("SIO0 reset");
            self.write_mode(0xC, scheduler, interrupt_registers);
            self.write_control(0, scheduler, interrupt_registers);
            self.rx_fifo.clear();
            return;
        }

        let prev_port = self.selected_port;

        self.tx_enabled = value.bit(0);
        self.dtr_on = value.bit(1);
        self.rx_enabled = value.bit(2);
        self.rx_interrupt_bytes = 1 << ((value >> 8) & 3);
        self.rx_interrupt_enabled = value.bit(10);
        self.tx_interrupt_enabled = value.bit(11);
        self.dsr_interrupt_enabled = value.bit(12);
        self.selected_port = Port::from_bit(value.bit(13));

        if value.bit(4) {
            self.irq = false;
        }

        if !self.dtr_on || self.selected_port != prev_port {
            self.tx_fifo = TxFifoState::Empty;
            self.active_device = None;
        }

        if self.tx_enabled
            && let TxFifoState::Queued(tx) = self.tx_fifo
        {
            self.tx_fifo = TxFifoState::Transferring {
                value: tx,
                cycles_remaining: CONTROLLER_TRANSFER_CYCLES,
                next: None,
            };
            scheduler.update_or_push_event(SchedulerEvent {
                event_type: self.tx_event_type,
                cpu_cycles: scheduler.cpu_cycle_counter() + u64::from(CONTROLLER_TRANSFER_CYCLES),
            });
        }

        log::debug!("SIO0_CTRL write: {value:04X}");
        log::debug!("  TX enabled: {}", self.tx_enabled);
        log::debug!("  DTR output on: {}", self.dtr_on);
        log::debug!("  RX enabled: {}", self.rx_enabled);
        log::debug!("  RX IRQ mode (FIFO length): {}", self.rx_interrupt_bytes);
        log::debug!("  RX IRQ enabled: {}", self.rx_interrupt_enabled);
        log::debug!("  TX IRQ enabled: {}", self.tx_interrupt_enabled);
        log::debug!("  DSR IRQ enabled: {}", self.dsr_interrupt_enabled);
        log::debug!("  Selected port: {:?}", self.selected_port);
    }

    // $1F80104E: SIO0_BAUD (Baudrate timer reload value)
    pub fn write_baudrate_reload(
        &mut self,
        value: u32,
        scheduler: &mut Scheduler,
        interrupt_registers: &mut InterruptRegisters,
    ) {
        self.catch_up(scheduler, interrupt_registers);

        self.baudrate_timer.update_reload_value(value);

        log::debug!("SIO0 Baudrate timer reload value: {value:04X}");
    }

    pub fn read_baudrate_reload(&self) -> u32 {
        self.baudrate_timer.raw_reload_value
    }
}
