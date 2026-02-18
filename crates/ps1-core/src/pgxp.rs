use crate::gpu::Vertex;
use crate::memory;
use bincode::de::{BorrowDecoder, Decoder};
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{BorrowDecode, Decode, Encode};
use std::{array, mem};

macro_rules! impl_fake_encode_decode {
    ($t:ty) => {
        impl Encode for $t {
            fn encode<E: Encoder>(&self, _encoder: &mut E) -> Result<(), EncodeError> {
                Ok(())
            }
        }

        impl<Context> Decode<Context> for $t {
            fn decode<D: Decoder>(_decoder: &mut D) -> Result<Self, DecodeError> {
                Ok(Self::new())
            }
        }

        impl<'de, Context> BorrowDecode<'de, Context> for $t {
            fn borrow_decode<D: BorrowDecoder<'de>>(_decoder: &mut D) -> Result<Self, DecodeError> {
                Ok(Self::new())
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PgxpConfig {
    // Enable basic PGXP: capture fractional vertex coordinates from RTPS/RTPT instructions and
    // have move/load/store instructions pass the coordinates through registers and memory to the GPU
    pub enabled: bool,
    // Perform NCLIP calculations using precise vertex coordinates when available; this can fill in
    // gaps in geometry that are not visible when PGXP is off
    pub precise_nclip: bool,
    // Perform perspective-correct UV interpolation using the Z coordinates from RTPS/RTPT
    pub perspective_texture_mapping: bool,
}

impl_fake_encode_decode!(PgxpConfig);

impl Default for PgxpConfig {
    fn default() -> Self {
        Self { enabled: false, precise_nclip: true, perspective_texture_mapping: true }
    }
}

impl PgxpConfig {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn precise_nclip(self) -> bool {
        self.enabled && self.precise_nclip
    }

    pub(crate) fn perspective_texture_mapping(self) -> bool {
        self.enabled && self.perspective_texture_mapping
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Encode, Decode)]
pub struct PreciseVertex {
    pub x: f64,
    pub y: f64,
    pub z: Option<u16>,
}

impl PreciseVertex {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: None };

    pub const INVALID: Self = Self { x: f64::INFINITY, y: f64::INFINITY, z: None };

    pub fn matches(&self, vertex: Vertex) -> bool {
        let fx = self.x as i32;
        let fy = self.y as i32;

        (fx == vertex.x || fx.wrapping_sub(1) == vertex.x)
            && (fy == vertex.y || fy.wrapping_sub(1) == vertex.y)
    }

    pub fn is_valid(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl Default for PreciseVertex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl From<Vertex> for PreciseVertex {
    fn from(value: Vertex) -> Self {
        Self { x: value.x.into(), y: value.y.into(), z: None }
    }
}

#[derive(Debug, Clone)]
pub struct PgxpCpuRegisters {
    pub gpr: [PreciseVertex; 32],
    pub delayed_load: (u32, PreciseVertex),
    pub delayed_load_next: (u32, PreciseVertex),
}

impl_fake_encode_decode!(PgxpCpuRegisters);

impl PgxpCpuRegisters {
    pub fn new() -> Self {
        let mut gpr = array::from_fn(|_| PreciseVertex::default());
        gpr[0] = PreciseVertex::ZERO;

        Self {
            gpr,
            delayed_load: (0, PreciseVertex::default()),
            delayed_load_next: (0, PreciseVertex::default()),
        }
    }

    pub fn write_gpr(&mut self, register: u32, vertex: PreciseVertex) {
        if register == 0 {
            return;
        }

        if self.delayed_load.0 == register {
            self.delayed_load = (0, PreciseVertex::INVALID);
        }

        self.gpr[register as usize] = vertex;
    }

    pub fn write_gpr_delayed(&mut self, register: u32, vertex: PreciseVertex) {
        if register == 0 {
            return;
        }

        if self.delayed_load.0 == register {
            self.delayed_load = (0, PreciseVertex::INVALID);
        }

        self.delayed_load_next = (register, vertex);
    }

    pub fn process_delayed_loads(&mut self) {
        if self.delayed_load.0 != 0 {
            let (register, vertex) = self.delayed_load;
            self.gpr[register as usize] = vertex;
            self.delayed_load.0 = 0;
        }

        if self.delayed_load_next.0 != 0 {
            self.delayed_load = mem::take(&mut self.delayed_load_next);
        }
    }
}

#[derive(Debug, Clone)]
pub struct PgxpGteRegisters {
    pub sxy: [PreciseVertex; 3],
}

impl PgxpGteRegisters {
    pub fn new() -> Self {
        Self { sxy: array::from_fn(|_| PreciseVertex::default()) }
    }

    pub fn convert_and_push_fifo(&mut self, sx_decimal: i64, sy_decimal: i64, sz: u16) {
        // Screen X/Y coordinates have 16 fractional bits, and the integer part is saturated to
        // signed 11-bit
        let sx = ((sx_decimal as f64) / 65536.0).clamp(-1024.0, 1024.0);
        let sy = ((sy_decimal as f64) / 65536.0).clamp(-1024.0, 1024.0);

        self.push_fifo(PreciseVertex { x: sx, y: sy, z: Some(sz) });
    }

    pub fn push_fifo(&mut self, vertex: PreciseVertex) {
        self.sxy[0] = self.sxy[1];
        self.sxy[1] = self.sxy[2];
        self.sxy[2] = vertex;
    }

    pub fn all_valid(
        &self,
        (sx0, sy0): (i64, i64),
        (sx1, sy1): (i64, i64),
        (sx2, sy2): (i64, i64),
    ) -> bool {
        self.sxy[0].is_valid()
            && self.sxy[0].matches(Vertex { x: sx0 as i32, y: sy0 as i32 })
            && self.sxy[1].is_valid()
            && self.sxy[1].matches(Vertex { x: sx1 as i32, y: sy1 as i32 })
            && self.sxy[2].is_valid()
            && self.sxy[2].matches(Vertex { x: sx2 as i32, y: sy2 as i32 })
    }
}

impl_fake_encode_decode!(PgxpGteRegisters);

const PGXP_MAIN_RAM_LEN: usize = memory::MAIN_RAM_LEN >> 2;
const PGXP_SCRATCHPAD_LEN: usize = memory::SCRATCHPAD_LEN >> 2;

const PGXP_MAIN_RAM_MASK: u32 = memory::MAIN_RAM_MASK >> 2;
const PGXP_SCRATCHPAD_MASK: u32 = memory::SCRATCHPAD_MASK >> 2;

type PgxpMainRam = [PreciseVertex; PGXP_MAIN_RAM_LEN];
type PgxpScratchpad = [PreciseVertex; PGXP_SCRATCHPAD_LEN];

#[derive(Debug, Clone)]
pub struct PgxpMemory {
    main_ram: Box<PgxpMainRam>,
    scratchpad: Box<PgxpScratchpad>,
}

impl_fake_encode_decode!(PgxpMemory);

impl PgxpMemory {
    pub fn new() -> Self {
        Self {
            main_ram: vec![PreciseVertex::default(); PGXP_MAIN_RAM_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            scratchpad: vec![PreciseVertex::default(); PGXP_SCRATCHPAD_LEN]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    pub fn read_main_ram(&self, address: u32) -> PreciseVertex {
        self.main_ram[((address >> 2) & PGXP_MAIN_RAM_MASK) as usize]
    }

    pub fn write_main_ram(&mut self, address: u32, vertex: PreciseVertex) {
        self.main_ram[((address >> 2) & PGXP_MAIN_RAM_MASK) as usize] = vertex;
    }

    pub fn read_scratchpad(&self, address: u32) -> PreciseVertex {
        self.scratchpad[((address >> 2) & PGXP_SCRATCHPAD_MASK) as usize]
    }

    pub fn write_scratchpad(&mut self, address: u32, vertex: PreciseVertex) {
        self.scratchpad[((address >> 2) & PGXP_SCRATCHPAD_MASK) as usize] = vertex;
    }
}
