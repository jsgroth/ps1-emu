use crate::config::AudioConfig;
use crate::Never;
use ps1_core::api::AudioOutput;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub const FREQUENCY: i32 = 44100;
pub const CHANNELS: u8 = 2;

pub fn new_spec(config: &AudioConfig) -> AudioSpecDesired {
    AudioSpecDesired {
        freq: Some(FREQUENCY),
        channels: Some(CHANNELS),
        samples: Some(config.device_queue_size),
    }
}

pub type AudioQueue = Arc<Mutex<VecDeque<(f32, f32)>>>;

pub struct QueueAudioCallback {
    audio_queue: AudioQueue,
}

impl QueueAudioCallback {
    pub fn new(audio_queue: AudioQueue) -> Self {
        Self { audio_queue }
    }
}

impl AudioCallback for QueueAudioCallback {
    type Channel = f32;

    fn callback(&mut self, out: &mut [Self::Channel]) {
        let mut queue = self.audio_queue.lock().unwrap();
        for chunk in out.chunks_exact_mut(2) {
            let (l, r) = queue.pop_front().unwrap_or((0.0, 0.0));
            chunk[0] = l;
            chunk[1] = r;
        }
    }
}

pub struct QueueAudioOutput {
    audio_queue: AudioQueue,
}

impl QueueAudioOutput {
    pub fn new(audio_queue: AudioQueue) -> Self {
        Self { audio_queue }
    }

    pub fn samples_len(&self) -> usize {
        self.audio_queue.lock().unwrap().len()
    }
}

impl AudioOutput for QueueAudioOutput {
    type Err = Never;

    fn queue_samples(&mut self, samples: &[(f64, f64)]) -> Result<(), Self::Err> {
        let mut queue = self.audio_queue.lock().unwrap();
        for &(sample_l, sample_r) in samples {
            queue.push_back((sample_l as f32, sample_r as f32));
        }

        Ok(())
    }
}
