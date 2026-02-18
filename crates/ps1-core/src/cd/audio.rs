//! CD-ROM audio commands

use crate::cd;
#[allow(clippy::wildcard_imports)]
use crate::cd::macros::*;
use crate::cd::{CdController, CommandState, DriveState, SeekNextState, seek, status};
use bincode::{Decode, Encode};
use cdrom::CdRomResult;
use cdrom::cdtime::CdTime;

pub const CD_DA_SAMPLES_PER_SECTOR: u16 = 588;
const SECTORS_BETWEEN_REPORTS: u8 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub enum AudioReportType {
    Absolute,
    Relative,
}

impl AudioReportType {
    fn toggle(self) -> Self {
        match self {
            Self::Absolute => Self::Relative,
            Self::Relative => Self::Absolute,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Encode, Decode)]
pub struct PlayState {
    pub time: CdTime,
    pub sample_idx: u16,
    pub sectors_till_report: u8,
    pub next_report_type: AudioReportType,
}

impl PlayState {
    pub fn new(time: CdTime) -> Self {
        Self {
            time,
            sample_idx: 0,
            sectors_till_report: 1,
            next_report_type: AudioReportType::Absolute,
        }
    }
}

impl CdController {
    // $0B: Mute() -> INT3(stat)
    // Mutes CD audio output, both CD-DA and ADPCM
    pub(super) fn execute_mute(&mut self) -> CommandState {
        self.audio_muted = true;

        self.int3(&[stat!(self)]);

        CommandState::Idle
    }

    // $0C: Demute() -> INT3(stat)
    // Demutes CD audio output, both CD-DA and ADPCM
    pub(super) fn execute_demute(&mut self) -> CommandState {
        self.audio_muted = false;

        self.int3(&[stat!(self)]);

        CommandState::Idle
    }

    // $03: Play(track?) -> INT3(stat), INT1(report)*
    // Begins audio playback from the specified location.
    // If track parameter is present and non-zero, begins playback from the start of that track.
    // If track parameter is zero or not present, begins playback from the last SetLoc location, or
    // the current time if there is no unprocessed SetLoc location.
    pub(super) fn execute_play(&mut self) -> CommandState {
        if self.disc.is_none() {
            self.int5(&[stat!(self), status::CANNOT_RESPOND_YET]);
            return CommandState::Idle;
        }

        self.int3(&[stat!(self)]);

        let Some(disc) = &self.disc else {
            // TODO generate error response
            todo!("Play command issued with no disc");
        };

        let track_number = if self.parameter_fifo.empty() {
            0
        } else {
            cd::bcd_to_binary(self.parameter_fifo.pop())
        };

        let current_time = self.drive_state.current_time();
        let num_tracks = disc.cue().last_track().number;
        let track_start_time = if track_number == 0 {
            // Track number 0 (or not set) means use SetLoc location
            self.seek_location.take().unwrap_or(current_time)
        } else {
            // Track numbers above the last track number should wrap back around to 1
            let wrapped_track_number = ((track_number - 1) % num_tracks) + 1;
            disc.cue().track(wrapped_track_number).effective_start_time()
        };

        log::debug!("Executing Play command: track {track_number}, start time {track_start_time}");

        self.drive_state =
            seek::determine_drive_state(self.drive_state, track_start_time, SeekNextState::Play);

        CommandState::Idle
    }

    pub(super) fn read_audio_sector(
        &mut self,
        PlayState { time, mut sectors_till_report, mut next_report_type, .. }: PlayState,
        first_sector: bool,
    ) -> CdRomResult<DriveState> {
        let Some(disc) = &self.disc else {
            self.int5(&[stat!(self), status::CANNOT_RESPOND_YET]);
            return Ok(DriveState::Stopped);
        };

        let num_tracks = disc.cue().last_track().number;
        let track = disc.cue().find_track_by_time(time);
        let Some(track) = track else {
            // At end of disc
            // If auto-pause is enabled, pause at the end of the last track; otherwise stop the drive
            // In both cases, generate INT4
            if num_tracks > 1 && self.drive_mode.auto_pause_audio {
                self.drive_state =
                    DriveState::Paused { time: time - CdTime::new(0, 0, 1), int2_queued: false };
            } else {
                self.drive_state = DriveState::Stopped;
            }

            self.int4(&[stat!(self)]);
            return Ok(self.drive_state);
        };

        // If auto-pause is enabled and the drive moved to a new audio track, pause at the end of
        // the previous track and generate INT4
        if self.drive_mode.auto_pause_audio
            && !first_sector
            && track.number > 2
            && time == track.start_time
        {
            self.drive_state =
                DriveState::Paused { time: time - CdTime::new(0, 0, 1), int2_queued: false };
            self.int4(&[stat!(self)]);
            return Ok(self.drive_state);
        }

        if sectors_till_report == 1 {
            if self.drive_mode.audio_report_interrupts {
                // Generate audio report
                //   Absolute report: INT1(stat, track, index, amm, ass, asect, peaklo, peakhi)
                //   Relative report: INT1(stat, track, index, mm, ss | 0x80, sect, peaklo, peakhi)
                let index = u8::from(time >= track.effective_start_time());
                let report_time = match next_report_type {
                    AudioReportType::Absolute => time,
                    AudioReportType::Relative => time.saturating_sub(track.effective_start_time()),
                };

                let track_number = cd::binary_to_bcd(track.number);
                let minutes = cd::binary_to_bcd(report_time.minutes);
                let mut seconds = cd::binary_to_bcd(report_time.seconds);
                let frames = cd::binary_to_bcd(report_time.frames);

                if next_report_type == AudioReportType::Relative {
                    seconds |= 0x80;
                }

                // TODO check if an interrupt is pending?
                self.int1(&[
                    stat!(self),
                    track_number,
                    index,
                    minutes,
                    seconds,
                    frames,
                    0x00,
                    0x00,
                ]);
            }

            sectors_till_report = SECTORS_BETWEEN_REPORTS;
            next_report_type = next_report_type.toggle();
        }

        self.read_sector_atime(time)?;

        Ok(DriveState::Playing(PlayState {
            time: time + CdTime::new(0, 0, 1),
            sample_idx: 0,
            sectors_till_report: sectors_till_report - 1,
            next_report_type,
        }))
    }

    pub(super) fn progress_play_state(
        &mut self,
        PlayState { time, mut sample_idx, sectors_till_report, next_report_type }: PlayState,
    ) -> CdRomResult<DriveState> {
        if self.drive_mode.cd_da_enabled {
            let sample_addr = (sample_idx * 4) as usize;

            let sample_l = i16::from_le_bytes([
                self.sector_buffer[sample_addr],
                self.sector_buffer[sample_addr + 1],
            ]);
            let sample_r = i16::from_le_bytes([
                self.sector_buffer[sample_addr + 2],
                self.sector_buffer[sample_addr + 3],
            ]);
            self.current_audio_sample = (sample_l, sample_r);
        }

        sample_idx += 1;
        if sample_idx == CD_DA_SAMPLES_PER_SECTOR {
            self.read_audio_sector(
                PlayState { time, sample_idx: 0, sectors_till_report, next_report_type },
                false,
            )
        } else {
            Ok(DriveState::Playing(PlayState {
                time,
                sample_idx,
                sectors_till_report,
                next_report_type,
            }))
        }
    }
}
