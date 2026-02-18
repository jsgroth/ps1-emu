//! Boxed array wrapper struct
//!
//! This exists because bincode deserializes boxed arrays onto the stack, which can cause a stack
//! overflow when deserializing main RAM (2MB), VRAM (1MB), or sound RAM (512KB). This wrapper
//! has `Decode`/`BorrowDecode` implementations that allocate on the heap and deserialize directly
//! into heap memory.

use bincode::de::read::Reader;
use bincode::de::{BorrowDecoder, Decoder};
use bincode::error::DecodeError;
use bincode::{BorrowDecode, Decode, Encode};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

#[repr(transparent)]
#[derive(Debug, Clone, Encode)]
pub struct BoxedArray<T, const LEN: usize>(pub Box<[T; LEN]>);

impl<T, const LEN: usize> From<Box<[T; LEN]>> for BoxedArray<T, LEN> {
    fn from(value: Box<[T; LEN]>) -> Self {
        Self(value)
    }
}

impl<T: Debug + Clone + Default, const LEN: usize> BoxedArray<T, LEN> {
    pub fn new() -> Self {
        Self(vec![T::default(); LEN].into_boxed_slice().try_into().unwrap())
    }
}

impl<T: Copy, const LEN: usize> Deref for BoxedArray<T, LEN> {
    type Target = [T; LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Copy, const LEN: usize> DerefMut for BoxedArray<T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const LEN: usize, Context> Decode<Context> for BoxedArray<u8, LEN> {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut array: Box<[u8; LEN]> = vec![0; LEN].into_boxed_slice().try_into().unwrap();
        decoder.reader().read(array.as_mut_slice())?;
        Ok(Self(array))
    }
}

impl<'de, const LEN: usize, Context> BorrowDecode<'de, Context> for BoxedArray<u8, LEN> {
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut array: Box<[u8; LEN]> = vec![0; LEN].into_boxed_slice().try_into().unwrap();
        decoder.reader().read(array.as_mut_slice())?;
        Ok(Self(array))
    }
}

impl<const LEN: usize, Context> Decode<Context> for BoxedArray<u16, LEN> {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut array: Box<[u16; LEN]> = vec![0; LEN].into_boxed_slice().try_into().unwrap();
        for value in array.as_mut() {
            *value = u16::decode(decoder)?;
        }
        Ok(Self(array))
    }
}

impl<'de, const LEN: usize, Context> BorrowDecode<'de, Context> for BoxedArray<u16, LEN> {
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut array: Box<[u16; LEN]> = vec![0; LEN].into_boxed_slice().try_into().unwrap();
        for value in array.as_mut() {
            *value = u16::decode(decoder)?;
        }
        Ok(Self(array))
    }
}
