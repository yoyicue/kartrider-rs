//! Port of `scripts/evaluate_low_projection_ground_branch.py` — executable
//! low-projection branch of GoPlayKart_ApplyGroundDriveForces
//! (B2C196..B2C51C). Both `numeric_mode` paths.

use crate::num::{f32q, fadd, fdiv, fmul, fsub, max2_f64, min2_f64};

/// `PROJECTION_THRESHOLD`.
pub const PROJECTION_THRESHOLD: f64 = 5.0;
/// `STATE_DECAY_STEP`.
pub const STATE_DECAY_STEP: f64 = 0.004000000189989805;

pub const ORDERING: &str = "select low projection -> clear/decay drift-recovery state and settle B1C -> compute reduced force against fixed denominator 5 -> submit force/torque";

// static_source provenance (Python result key `static_source`).
pub const STATIC_GATE_VA: &str = "0xB2AD88";
pub const STATIC_BODY_VA: &str = "0xB2C196-0xB2C51C";

/// `_decay_toward_zero(value)` with the module-level step.
fn decay_toward_zero(value: f64) -> f64 {
    if value > 0.0 {
        return max2_f64(0.0, value - STATE_DECAY_STEP);
    }
    if value < 0.0 {
        return min2_f64(0.0, value + STATE_DECAY_STEP);
    }
    0.0
}

/// Python `state_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct LowProjectionStateAfter {
    pub state_5c8: f64,
    pub field_5d4: f64,
    pub selector_af4: bool,
    pub field_af8: f64,
    pub selector_b08: bool,
    pub field_b0c: f64,
    pub selector_b5c: bool,
    pub b1c_charge_settlement: bool,
    pub b10: f64,
    pub b20: f64,
    pub b38: f64,
    pub b2c: f64,
}

/// Python result dict for the selected path, minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct LowProjectionSelected {
    pub projection_len: f64,
    pub forward_target: f64,
    pub normalized_velocity: f64,
    pub longitudinal_correction: f64,
    pub lateral_correction: f64,
    pub longitudinal: f64,
    pub lateral: f64,
    pub local_force: [f64; 3],
    pub local_torque: [f64; 3],
    pub numeric_mode: String,
    pub state_after: LowProjectionStateAfter,
    pub ordering: &'static str,
}

/// Python result: either the early `{"selected": False, ...}` gate row or
/// the full selected-path row.
#[derive(Debug, Clone, PartialEq)]
pub enum LowProjectionBranchResult {
    /// `{"selected": False, "reason": "projection_above_threshold", ...}`
    NotSelected { projection_len: f64 },
    /// `{"selected": True, "reason": "projection_at_or_below_5", ...}`
    Selected(Box<LowProjectionSelected>),
}

/// `evaluate_low_projection_branch(...)`.
pub fn evaluate_low_projection_branch(
    projection_len: f64,
    ground_forward_term: f64,
    drive_direction: f64,
    normalized_velocity_raw: f64,
    correction_raw: f64,
    mass_scale: f64,
    base_longitudinal_gain: f64,
    base_lateral_gain: f64,
    axis_long: f64,
    axis_lat: f64,
    state_5c8: f64,
    b1c_charge_settlement: bool,
    b10: f64,
    b20: f64,
    b2c: f64,
    b38: f64,
    cap_760: f64,
    numeric_mode: &str,
) -> Result<LowProjectionBranchResult, String> {
    if projection_len > PROJECTION_THRESHOLD {
        return Ok(LowProjectionBranchResult::NotSelected { projection_len });
    }

    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    let normalized_velocity: f64;
    let longitudinal_correction: f64;
    let lateral_correction: f64;
    let forward_target: f64;
    let longitudinal: f64;
    let lateral: f64;
    let force_x: f64;
    let torque_z: f64;
    if numeric_mode == "float32" {
        let threshold = f32q(PROJECTION_THRESHOLD);
        normalized_velocity = fdiv(normalized_velocity_raw, threshold);
        longitudinal_correction = fdiv(fmul(correction_raw, axis_long), threshold);
        lateral_correction = fdiv(fmul(correction_raw, axis_lat), threshold);
        forward_target = if projection_len >= 0.5 {
            fmul(ground_forward_term, drive_direction)
        } else {
            0.0
        };
        let longitudinal_inner = fsub(
            f32q(forward_target - normalized_velocity),
            longitudinal_correction,
        );
        let lateral_inner = fadd(-f32q(normalized_velocity), lateral_correction);
        longitudinal = fmul(fmul(mass_scale, base_longitudinal_gain), longitudinal_inner);
        lateral = fmul(fmul(mass_scale, base_lateral_gain), lateral_inner);
        force_x = fadd(longitudinal, lateral);
        torque_z = fsub(fmul(axis_long, longitudinal), fmul(axis_lat, lateral));
    } else {
        normalized_velocity = normalized_velocity_raw / PROJECTION_THRESHOLD;
        longitudinal_correction = correction_raw * axis_long / PROJECTION_THRESHOLD;
        lateral_correction = correction_raw * axis_lat / PROJECTION_THRESHOLD;
        forward_target = if projection_len >= 0.5 {
            ground_forward_term * drive_direction
        } else {
            0.0
        };
        longitudinal = mass_scale
            * base_longitudinal_gain
            * (forward_target - normalized_velocity - longitudinal_correction);
        lateral = mass_scale * base_lateral_gain * (-normalized_velocity + lateral_correction);
        force_x = longitudinal + lateral;
        torque_z = axis_long * longitudinal - axis_lat * lateral;
    }

    let state_after = LowProjectionStateAfter {
        state_5c8: decay_toward_zero(state_5c8),
        field_5d4: 0.0,
        selector_af4: false,
        field_af8: 0.0,
        selector_b08: false,
        field_b0c: 0.0,
        selector_b5c: false,
        b1c_charge_settlement: false,
        b10: if b1c_charge_settlement {
            min2_f64(b10 + b2c, cap_760)
        } else {
            b10
        },
        b20: if b1c_charge_settlement { 0.0 } else { b20 },
        b38: if b1c_charge_settlement { b2c } else { b38 },
        b2c: if b1c_charge_settlement { 0.0 } else { b2c },
    };
    Ok(LowProjectionBranchResult::Selected(Box::new(
        LowProjectionSelected {
            projection_len,
            forward_target,
            normalized_velocity,
            longitudinal_correction,
            lateral_correction,
            longitudinal,
            lateral,
            local_force: [force_x, 0.0, 0.0],
            local_torque: [0.0, 0.0, torque_z],
            numeric_mode: numeric_mode.to_string(),
            state_after,
            ordering: ORDERING,
        },
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float32_selected() {
        // Constants computed with the Python authority:
        //   evaluate_low_projection_branch(projection_len=2.5, ...,
        //     numeric_mode="float32")
        let r = evaluate_low_projection_branch(
            2.5, 1.25, 1.0, 2.0, 0.5, 80.0, 2.0, 1.5, 0.5, 0.5, 0.0, false, 0.0, 0.0, 0.0,
            0.0, 100.0, "float32",
        )
        .unwrap();
        let LowProjectionBranchResult::Selected(selected) = r else {
            panic!("projection 2.5 must select the branch")
        };
        assert_eq!(selected.normalized_velocity, 0.4000000059604645);
        assert_eq!(selected.longitudinal_correction, 0.05000000074505806);
        assert_eq!(selected.longitudinal, 128.0);
        assert_eq!(selected.lateral, -42.0);
        assert_eq!(selected.local_force, [86.0, 0.0, 0.0]);
        assert_eq!(selected.local_torque, [0.0, 0.0, 85.0]);
    }

    #[test]
    fn above_threshold_not_selected() {
        let r = evaluate_low_projection_branch(
            5.001, 1.25, 1.0, 2.0, 0.5, 80.0, 2.0, 1.5, 0.5, 0.5, 0.0, false, 0.0, 0.0, 0.0,
            0.0, 100.0, "float64",
        )
        .unwrap();
        assert_eq!(
            r,
            LowProjectionBranchResult::NotSelected {
                projection_len: 5.001
            }
        );
    }

    #[test]
    fn error_guard_matches_python_message() {
        assert_eq!(
            evaluate_low_projection_branch(
                2.5, 1.25, 1.0, 2.0, 0.5, 80.0, 2.0, 1.5, 0.5, 0.5, 0.0, false, 0.0, 0.0,
                0.0, 0.0, 100.0, "float16",
            )
            .unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }
}
