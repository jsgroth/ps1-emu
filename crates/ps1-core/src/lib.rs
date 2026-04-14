pub mod api;
mod boxedarray;
mod bus;
mod cd;
mod cpu;
mod dma;
mod gpu;
pub mod input;
mod interrupts;
mod mdec;
mod memory;
mod num;
mod pgxp;
mod scheduler;
mod sio;
mod spu;
mod timers;

pub use gpu::RasterizerType;

#[must_use]
pub fn required_wgpu_features() -> wgpu::Features {
    wgpu::Features::IMMEDIATES
        | wgpu::Features::DUAL_SOURCE_BLENDING
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
}

#[must_use]
pub fn required_wgpu_limits() -> wgpu::Limits {
    wgpu::Limits {
        max_texture_dimension_2d: 16 * 1024,
        max_immediate_size: 128,
        ..wgpu::Limits::default()
    }
}
