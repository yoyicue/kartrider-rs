//! Frame-clock to physics-substep scheduler.
//!
//! Faithful port of `scripts/evaluate_physics_frame_scheduler.py`. Splits one
//! unsigned 32-bit frame-clock delta into ordered 1/2 ms integer substeps.
//! Never derive elapsed time from `count * width` — accumulate actual widths.

use crate::num::fmul;

pub const MAX_FRAME_DELTA_MS: u32 = 500;
pub const MAX_PHYSICS_SUBSTEP_MS: u32 = 2;

/// Exact binary32 value loaded at KartRider VA 0x1222C90
/// (little-endian bytes `6f 12 83 3a`), widened to f64.
pub const MILLISECONDS_TO_SECONDS_F32: f64 = f32::from_bits(0x3a83_126f) as f64;

/// Result of one scheduler evaluation (port of the Python result dict).
#[derive(Debug, Clone, PartialEq)]
pub struct FrameSchedule {
    pub previous_tick_ms: u32,
    pub current_tick_ms: u32,
    pub raw_frame_delta_ms: u32,
    pub capped_frame_delta_ms: u32,
    pub frame_delta_capped: bool,
    pub substeps_ms: Vec<u32>,
    /// Binary32 seconds per substep, widened to f64 (see crate::num).
    pub substeps_seconds_f32: Vec<f64>,
}

/// `evaluate_physics_frame_scheduler(previous_tick_ms, current_tick_ms)`.
///
/// The unsigned subtraction preserves the native 32-bit clock wrap. A stalled
/// frame is capped before the loop. Each loop iteration consumes
/// `min(remaining, 2)` integer milliseconds and converts that integer to
/// binary32 seconds using the executable's literal at VA 0x1222C90.
pub fn evaluate_physics_frame_scheduler(
    previous_tick_ms: u32,
    current_tick_ms: u32,
) -> FrameSchedule {
    let raw_delta_ms = current_tick_ms.wrapping_sub(previous_tick_ms);
    let capped_delta_ms = raw_delta_ms.min(MAX_FRAME_DELTA_MS);
    let mut remaining = capped_delta_ms;
    let mut substeps_ms = Vec::new();
    while remaining > 0 {
        let substep_ms = remaining.min(MAX_PHYSICS_SUBSTEP_MS);
        substeps_ms.push(substep_ms);
        remaining -= substep_ms;
    }
    let substeps_seconds_f32 = substeps_ms
        .iter()
        .map(|&value| fmul(value as f64, MILLISECONDS_TO_SECONDS_F32))
        .collect();
    FrameSchedule {
        previous_tick_ms,
        current_tick_ms,
        raw_frame_delta_ms: raw_delta_ms,
        capped_frame_delta_ms: capped_delta_ms,
        frame_delta_capped: raw_delta_ms > MAX_FRAME_DELTA_MS,
        substeps_ms,
        substeps_seconds_f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn milliseconds_constant_matches_va_literal() {
        assert_eq!((MILLISECONDS_TO_SECONDS_F32 as f32).to_bits(), 0x3a83_126f);
        assert_eq!(MILLISECONDS_TO_SECONDS_F32, 0.001_f32 as f64);
    }

    #[test]
    fn odd_delta_ends_in_one_ms_tail() {
        let s = evaluate_physics_frame_scheduler(1000, 1005);
        assert_eq!(s.substeps_ms, vec![2, 2, 1]);
        assert!(!s.frame_delta_capped);
    }

    #[test]
    fn stalled_frame_is_capped() {
        let s = evaluate_physics_frame_scheduler(0, 10_000);
        assert_eq!(s.capped_frame_delta_ms, MAX_FRAME_DELTA_MS);
        assert!(s.frame_delta_capped);
        assert_eq!(s.substeps_ms.len(), 250);
    }

    #[test]
    fn clock_wraps_as_uint32() {
        let s = evaluate_physics_frame_scheduler(0xFFFF_FFFF, 2);
        assert_eq!(s.raw_frame_delta_ms, 3);
        assert_eq!(s.substeps_ms, vec![2, 1]);
    }

    #[test]
    fn zero_delta_yields_no_substeps() {
        let s = evaluate_physics_frame_scheduler(42, 42);
        assert!(s.substeps_ms.is_empty());
    }
}
