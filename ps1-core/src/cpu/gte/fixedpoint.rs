use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPointDecimal<const FRACTION_BITS: u8>(i64);

impl<const FRACTION_BITS: u8> FixedPointDecimal<FRACTION_BITS> {
    pub const ZERO: Self = Self(0);

    pub fn new(value: i64) -> Self {
        Self(value)
    }

    pub fn shift_to<const NEW_FRACTION_BITS: u8>(self) -> FixedPointDecimal<NEW_FRACTION_BITS> {
        if NEW_FRACTION_BITS > FRACTION_BITS {
            FixedPointDecimal(self.0 << (NEW_FRACTION_BITS - FRACTION_BITS))
        } else {
            FixedPointDecimal(self.0 >> (FRACTION_BITS - NEW_FRACTION_BITS))
        }
    }

    pub fn reinterpret<const NEW_FRACTION_BITS: u8>(self) -> FixedPointDecimal<NEW_FRACTION_BITS> {
        FixedPointDecimal(self.0)
    }

    pub fn clip_to_i32(self) -> Self {
        Self((self.0 << 32) >> 32)
    }
}

impl<const FRACTION_BITS: u8> From<FixedPointDecimal<FRACTION_BITS>> for i64 {
    fn from(value: FixedPointDecimal<FRACTION_BITS>) -> Self {
        value.0
    }
}

impl<const FRACTION_BITS: u8> Add for FixedPointDecimal<FRACTION_BITS> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<const FRACTION_BITS: u8> Sub for FixedPointDecimal<FRACTION_BITS> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

macro_rules! impl_mul {
    (@single $lhs:literal, $rhs:literal) => {
        impl Mul<FixedPointDecimal<$rhs>> for FixedPointDecimal<$lhs> {
            type Output = FixedPointDecimal<{$lhs + $rhs}>;

            fn mul(self, rhs: FixedPointDecimal<$rhs>) -> Self::Output {
                FixedPointDecimal(self.0 * rhs.0)
            }
        }
    };
    ($lhs:literal, $rhs:literal) => {
        impl_mul!(@single $lhs, $rhs);
        impl_mul!(@single $rhs, $lhs);
    };
}

impl_mul!(@single 0, 0);
impl_mul!(0, 12);
impl_mul!(4, 12);
impl_mul!(0, 16);
impl_mul!(8, 16);

// V0-2 and IR1-3 components are 1/15/0
pub type Vector16Component = FixedPointDecimal<0>;

pub fn vector16_component(value: u32) -> Vector16Component {
    FixedPointDecimal((value as i16).into())
}

// TRX/TRY/TRZ are 1/31/0
pub type TranslationComponent = FixedPointDecimal<0>;

pub fn translation_component(value: u32) -> TranslationComponent {
    FixedPointDecimal((value as i32).into())
}

// RT/LLM/LCM components are 1/3/12
pub type MatrixComponent = FixedPointDecimal<12>;

pub fn matrix_component(value: u32) -> MatrixComponent {
    FixedPointDecimal((value as i16).into())
}

// Division results are unsigned with 16 fractional bits
pub type DivisionResult = FixedPointDecimal<16>;

pub fn division_result(value: u32) -> DivisionResult {
    FixedPointDecimal(value.into())
}

// OFX/OFY are 1/15/16
pub type ScreenOffset = FixedPointDecimal<16>;

pub fn screen_offset(value: u32) -> ScreenOffset {
    FixedPointDecimal((value as i32).into())
}

// SX/SY components are 1/15/0
pub type ScreenCoordinate = FixedPointDecimal<0>;

pub fn screen_xy(value: u32) -> ScreenCoordinate {
    FixedPointDecimal((value as i16).into())
}

// SZ components are 0/16/0
pub fn screen_z(value: u32) -> ScreenCoordinate {
    FixedPointDecimal((value as u16).into())
}

// ZSF3/ZSF4 are 1/3/12
pub type ZScaleFactor = FixedPointDecimal<12>;

pub fn z_scale_factor(value: u32) -> ZScaleFactor {
    FixedPointDecimal((value as i16).into())
}

// DQA is 1/7/8, DQB is 1/7/24
pub type DepthCueingCoefficient = FixedPointDecimal<8>;
pub type DepthCueingOffset = FixedPointDecimal<24>;

pub fn dqa(value: u32) -> DepthCueingCoefficient {
    FixedPointDecimal((value as i16).into())
}

pub fn dqb(value: u32) -> DepthCueingOffset {
    FixedPointDecimal((value as i32).into())
}

// BK components are 1/19/12
pub type BackgroundColor = FixedPointDecimal<12>;

pub fn background_color(value: u32) -> BackgroundColor {
    FixedPointDecimal((value as i32).into())
}

// FC components are 1/27/4
pub type FarColor = FixedPointDecimal<4>;

pub fn far_color(value: u32) -> FarColor {
    FixedPointDecimal((value as i32).into())
}

// RGBC components are 0/8/0
pub type RgbComponent = FixedPointDecimal<0>;

pub fn rgb(value: u32) -> [RgbComponent; 3] {
    let [r, g, b, _] = value.to_le_bytes();
    [FixedPointDecimal(r.into()), FixedPointDecimal(g.into()), FixedPointDecimal(b.into())]
}

// IR0 is 1/3/12
pub type DepthInterpolationFactor = FixedPointDecimal<12>;

pub fn ir0(value: u32) -> DepthInterpolationFactor {
    FixedPointDecimal((value as i16).into())
}
