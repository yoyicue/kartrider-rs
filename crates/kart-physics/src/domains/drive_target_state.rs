//! Port of `scripts/evaluate_drive_target_state.py` — exact
//! GoPlayKart+0xAA4/+0xAA8 drive-target update at B2AB55..B2AD05. Always
//! f32-quantized (the Python module has no numeric_mode switch).

use crate::num::{f32q, fdiv, fmul};

/// `GAME_PI_F32`: dword_11ECAFC at IDA VA 0x11ECAFC (bytes D8 0F 49 40).
/// This is deliberately three float32 ULPs below the usual nearest-float
/// value of pi.
pub const GAME_PI_F32: f64 = f32::from_bits(0x4049_0fd8) as f64;

pub const ORDERING: &str = "signed axis * tune660 -> exp(-abs((ADC/71C)*BB8)) -> AA4; when selected slot is nonzero, same-sign magnitude growth clamps AA4 to prior AA8; otherwise AA8 follows AA4";
pub const TUNE_ACCESSOR: &str =
    "sub_6F3B80: float32(dword_11ECAFC[3.141592025756836] * object[+0x718] / 180)";

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2AA70";
pub const STATIC_RANGE_VA: &str = "0xB2AB55-0xB2AD05";

/// Python `evaluate_drive_target_state(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveTargetStateResult {
    pub raw_target: f64,
    pub exponent_input: f64,
    pub attenuation: f64,
    pub candidate_aa4: f64,
    pub field_aa4_after: f64,
    pub field_aa8_after: f64,
    pub overshoot_clamped_to_aa8: bool,
    pub ordering: &'static str,
}

/// `evaluate_drive_target_state(...)`.
pub fn evaluate_drive_target_state(
    signed_axis: f64,
    scaled_tune_660: f64,
    field_adc: f64,
    field_71c: f64,
    decoded_bb8: f64,
    selected_slot: f64,
    field_aa8_before: f64,
) -> Result<DriveTargetStateResult, String> {
    let values = [
        signed_axis,
        scaled_tune_660,
        field_adc,
        field_71c,
        decoded_bb8,
        selected_slot,
        field_aa8_before,
    ];
    if !values.iter().all(|value| value.is_finite()) {
        return Err("drive-target inputs must be finite".to_string());
    }
    if field_71c == 0.0 {
        return Err("field_71c must be nonzero".to_string());
    }
    let raw_target = fmul(signed_axis, scaled_tune_660);
    let ratio = fdiv(field_adc, field_71c);
    let exponent_input = f32q(ratio * f32q(decoded_bb8));
    // Python `math.exp(-abs(x))` is the platform libm exp; `f64::exp` issues
    // the same call on the same platform.
    let attenuation = f32q((-exponent_input.abs()).exp());
    let candidate = f32q(raw_target * attenuation);
    let clamp = selected_slot != 0.0
        && f32q(field_aa8_before * candidate) > 0.0
        && candidate.abs() > field_aa8_before.abs();
    let aa4_after = if clamp {
        f32q(field_aa8_before)
    } else {
        candidate
    };
    let aa8_after = if clamp {
        f32q(field_aa8_before)
    } else {
        candidate
    };
    Ok(DriveTargetStateResult {
        raw_target,
        exponent_input,
        attenuation,
        candidate_aa4: candidate,
        field_aa4_after: aa4_after,
        field_aa8_after: aa8_after,
        overshoot_clamped_to_aa8: clamp,
        ordering: ORDERING,
    })
}

/// Python `evaluate_drive_target_runtime(...)` result dict minus
/// `static_source`: the state result plus the three extra keys.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveTargetRuntimeResult {
    pub state: DriveTargetStateResult,
    pub field_718: f64,
    pub scaled_tune_660: f64,
    pub tune_accessor: &'static str,
}

/// `evaluate_drive_target_runtime(...)` — evaluate directly from runtime
/// object fields used by the IDA accessors.
pub fn evaluate_drive_target_runtime(
    signed_axis: f64,
    field_718: f64,
    field_adc: f64,
    field_71c: f64,
    decoded_bb8: f64,
    selected_slot: f64,
    field_aa8_before: f64,
) -> Result<DriveTargetRuntimeResult, String> {
    if !field_718.is_finite() {
        return Err("field_718 must be finite".to_string());
    }
    // sub_6F3B80: dword_11ECAFC (pi) * object[+0x718] /
    // dword_11ECB00 (180.0), with an SSE float32 commit at each op.
    let scaled_tune = fdiv(fmul(GAME_PI_F32, field_718), 180.0);
    let state = evaluate_drive_target_state(
        signed_axis,
        scaled_tune,
        field_adc,
        field_71c,
        decoded_bb8,
        selected_slot,
        field_aa8_before,
    )?;
    Ok(DriveTargetRuntimeResult {
        state,
        field_718: f32q(field_718),
        scaled_tune_660: scaled_tune,
        tune_accessor: TUNE_ACCESSOR,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_sign_growth_clamps_to_aa8() {
        // Constants computed with the Python authority:
        //   evaluate_drive_target_state(signed_axis=1.0, scaled_tune_660=0.5,
        //     field_adc=0.25, field_71c=0.75, decoded_bb8=2.0,
        //     selected_slot=1.0, field_aa8_before=0.1)
        let r = evaluate_drive_target_state(1.0, 0.5, 0.25, 0.75, 2.0, 1.0, 0.1).unwrap();
        assert_eq!(r.raw_target, 0.5);
        assert_eq!(r.exponent_input, 0.6666666865348816);
        assert_eq!(r.attenuation, 0.51341712474823);
        assert_eq!(r.candidate_aa4, 0.256708562374115);
        assert!(r.overshoot_clamped_to_aa8);
        assert_eq!(r.field_aa4_after, 0.10000000149011612);
        assert_eq!(r.field_aa8_after, 0.10000000149011612);
    }

    #[test]
    fn runtime_accessor_path() {
        // Constants computed with the Python authority:
        //   evaluate_drive_target_runtime(signed_axis=1.0, field_718=30.0,
        //     field_adc=0.25, field_71c=0.75, decoded_bb8=2.0,
        //     selected_slot=0.0, field_aa8_before=0.1)
        let r = evaluate_drive_target_runtime(1.0, 30.0, 0.25, 0.75, 2.0, 0.0, 0.1).unwrap();
        assert_eq!(r.scaled_tune_660, 0.5235986709594727);
        assert_eq!(r.state.raw_target, 0.5235986709594727);
        assert_eq!(r.state.candidate_aa4, 0.2688245177268982);
        assert!(!r.state.overshoot_clamped_to_aa8);
        assert_eq!(r.state.field_aa8_after, 0.2688245177268982);
        assert_eq!(r.field_718, 30.0);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            evaluate_drive_target_state(1.0, 0.5, 0.25, 0.0, 2.0, 0.0, 0.1).unwrap_err(),
            "field_71c must be nonzero"
        );
        assert_eq!(
            evaluate_drive_target_state(1.0, 0.5, f64::NAN, 0.75, 2.0, 0.0, 0.1).unwrap_err(),
            "drive-target inputs must be finite"
        );
        assert_eq!(
            evaluate_drive_target_runtime(1.0, f64::INFINITY, 0.25, 0.75, 2.0, 0.0, 0.1)
                .unwrap_err(),
            "field_718 must be finite"
        );
    }
}
