//! Port of `scripts/evaluate_ground_drive_force_body.py` — executable
//! normal/slip ground-drive force body at B2BCA1..B2C186. Both
//! `numeric_mode` paths.
//!
//! surface_mode=2 is generic slip. Water-specific constraints are deliberately
//! absent because they require map/collision context in addition to this mode.

use crate::domains::surface_tune_defaults::{ground_surface_profile, GroundSurfaceProfile};
use crate::num::{f32q, fadd, fmul, fsub, max2_f64, min2_f64};

/// `NORMAL_STATE_DECAY_STEP`.
pub const NORMAL_STATE_DECAY_STEP: f64 = 0.0020000000949949026;

pub const ORDERING: &str = "select normal path -> clear +0x5D4 and decay +0x5C8 -> compute normal/nonzero/slip force/torque scalars -> settle B1C -> submit force/torque";
pub const CONTEXT_POLICY: &str = "surface_mode=2 means generic slip; add water constraints only when map/collision context independently selects them";

// static_source provenance (Python result key `static_source`).
pub const STATIC_BODY_VA: &str = "0xB2BCA1-0xB2C186";
pub const STATIC_SURFACE2_FORWARD_VA: &str = "0xB2BD35-0xB2BD84";
pub const STATIC_FORCE_SINK_VA: &str = "0xB2C111";
pub const STATIC_TORQUE_SINK_VA: &str = "0xB2C177";
pub const STATIC_STATE_TRANSITION_VA: &str = "0xB2BBD9-0xB2C072";

/// `_decay_toward_zero(value)` with the module-level step.
fn decay_toward_zero(value: f64) -> f64 {
    if value > 0.0 {
        return max2_f64(0.0, value - NORMAL_STATE_DECAY_STEP);
    }
    if value < 0.0 {
        return min2_f64(0.0, value + NORMAL_STATE_DECAY_STEP);
    }
    0.0
}

/// Python `state_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundDriveStateAfter {
    pub state_5c8: f64,
    pub field_5d4: f64,
    pub b1c_charge_settlement: bool,
    pub b10: f64,
    pub b20: f64,
    pub b38: f64,
    pub b2c: f64,
}

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundDriveForceBodyResult {
    pub path: &'static str,
    pub surface_mode: i64,
    pub surface_profile_source: &'static str,
    pub slip_factor: Option<f64>,
    pub forward_target: f64,
    pub longitudinal: f64,
    pub lateral: f64,
    pub local_force: [f64; 3],
    pub local_torque: [f64; 3],
    pub torque_y_gain: f64,
    pub numeric_mode: String,
    pub state_after: GroundDriveStateAfter,
    pub ordering: &'static str,
    pub context_policy: &'static str,
}

/// `evaluate_ground_drive_force_body(...)`.
pub fn evaluate_ground_drive_force_body(
    surface_mode: i64,
    speed_magnitude: f64,
    ground_forward_term: f64,
    contact_basis_product: f64,
    drive_direction: f64,
    normalized_velocity: f64,
    longitudinal_correction: f64,
    lateral_correction: f64,
    mass_scale: f64,
    base_longitudinal_gain: f64,
    base_lateral_gain: f64,
    ground_force_multiplier: f64,
    settlement_multiplier: f64,
    tune_75c: f64,
    axis_long: f64,
    axis_lat: f64,
    surface_profile: Option<GroundSurfaceProfile>,
    torque_y_scale: f64,
    torque_y_gain: Option<f64>,
    state_5c8: f64,
    b1c_charge_settlement: bool,
    b10: f64,
    b20: f64,
    b2c: f64,
    b38: f64,
    cap_760: f64,
    numeric_mode: &str,
) -> Result<GroundDriveForceBodyResult, String> {
    let path: &'static str;
    let forward_target: f64;
    let surface_force_scale: f64;
    let mut slip_factor: Option<f64> = None;
    let mut profile_source: &'static str = "not_used";
    if surface_mode == 0 {
        path = "normal";
        forward_target = ground_forward_term;
        surface_force_scale = 1.0;
    } else {
        profile_source = if surface_profile.is_some() {
            "caller_override"
        } else {
            "decoded_playkart_default"
        };
        let profile = match surface_profile {
            Some(profile) => profile,
            None => ground_surface_profile(surface_mode)?.expect("mode != 0 yields a profile"),
        };
        let denominator = profile.speed_denominator;
        if denominator == 0.0 {
            return Err("surface speed_denominator must be nonzero".to_string());
        }
        let ratio = speed_magnitude / denominator;
        let factor = min2_f64(
            1.0,
            profile.speed_quadratic_gain * ratio * ratio + profile.speed_bias,
        );
        slip_factor = Some(factor);
        if surface_mode == 2 {
            path = "generic_slip";
            forward_target = contact_basis_product * factor;
        } else {
            path = "nonzero_surface";
            forward_target = (1.0 + factor) * ground_forward_term;
        }
        surface_force_scale = profile.force_scale;
    }

    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    let longitudinal: f64;
    let lateral: f64;
    let force_x: f64;
    let settlement_y: f64;
    if numeric_mode == "float32" {
        let common_scale = fmul(ground_force_multiplier, surface_force_scale);
        let mut longitudinal_inner = fsub(fmul(forward_target, drive_direction), normalized_velocity);
        longitudinal_inner = fsub(longitudinal_inner, longitudinal_correction);
        let longitudinal_scale = fmul(fmul(common_scale, mass_scale), base_longitudinal_gain);
        longitudinal = fmul(longitudinal_scale, longitudinal_inner);
        let lateral_inner = fadd(-f32q(normalized_velocity), lateral_correction);
        let lateral_scale = fmul(fmul(common_scale, mass_scale), base_lateral_gain);
        lateral = fmul(lateral_scale, lateral_inner);
        force_x = fadd(longitudinal, lateral);
        settlement_y = fmul(-force_x.abs(), settlement_multiplier);
    } else {
        let common_scale = ground_force_multiplier * surface_force_scale;
        longitudinal = common_scale
            * mass_scale
            * base_longitudinal_gain
            * (forward_target * drive_direction - normalized_velocity - longitudinal_correction);
        lateral = common_scale
            * mass_scale
            * base_lateral_gain
            * (-normalized_velocity + lateral_correction);
        force_x = longitudinal + lateral;
        settlement_y = -force_x.abs() * settlement_multiplier;
    }
    let selected_torque_y_gain = torque_y_gain.unwrap_or(tune_75c);
    let torque_y: f64;
    let torque_z: f64;
    if numeric_mode == "float32" {
        torque_y = fmul(fmul(-force_x, selected_torque_y_gain), torque_y_scale);
        torque_z = fsub(fmul(axis_long, longitudinal), fmul(axis_lat, lateral));
    } else {
        torque_y = -force_x * selected_torque_y_gain * torque_y_scale;
        torque_z = axis_long * longitudinal - axis_lat * lateral;
    }
    let state_after = GroundDriveStateAfter {
        state_5c8: decay_toward_zero(state_5c8),
        field_5d4: 0.0,
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
    Ok(GroundDriveForceBodyResult {
        path,
        surface_mode,
        surface_profile_source: if surface_mode == 0 {
            "not_used"
        } else {
            profile_source
        },
        slip_factor,
        forward_target,
        longitudinal,
        lateral,
        local_force: [force_x, settlement_y, 0.0],
        local_torque: [0.0, torque_y, torque_z],
        torque_y_gain: selected_torque_y_gain,
        numeric_mode: numeric_mode.to_string(),
        state_after,
        ordering: ORDERING,
        context_policy: CONTEXT_POLICY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn call(
        surface_mode: i64,
        numeric_mode: &str,
        surface_profile: Option<GroundSurfaceProfile>,
    ) -> Result<GroundDriveForceBodyResult, String> {
        evaluate_ground_drive_force_body(
            surface_mode, 12.5, 1.25, 0.75, 1.0, 0.5, 0.125, -0.25, 80.0, 2.0, 1.5, 1.0, 0.05,
            0.8, 0.5, 0.5, surface_profile, 1.0, None, 0.0, false, 0.0, 0.0, 0.0, 0.0, 100.0,
            numeric_mode,
        )
    }

    #[test]
    fn float32_normal_path() {
        // Constants computed with the Python authority:
        //   evaluate_ground_drive_force_body(surface_mode=0, ..., numeric_mode="float32")
        let r = call(0, "float32", None).unwrap();
        assert_eq!(r.path, "normal");
        assert_eq!(r.surface_profile_source, "not_used");
        assert_eq!(r.slip_factor, None);
        assert_eq!(r.longitudinal, 100.0);
        assert_eq!(r.lateral, -90.0);
        assert_eq!(r.local_force, [10.0, -0.5, 0.0]);
        assert_eq!(r.local_torque, [0.0, -8.0, 95.0]);
    }

    #[test]
    fn float64_generic_slip_path() {
        // Constants computed with the Python authority (surface_mode=2,
        // speed_magnitude=120.0, numeric_mode="float64").
        let r = evaluate_ground_drive_force_body(
            2, 120.0, 1.25, 0.75, 1.0, 0.5, 0.125, -0.25, 80.0, 2.0, 1.5, 1.0, 0.05, 0.8, 0.5,
            0.5, None, 1.0, None, 0.0, false, 0.0, 0.0, 0.0, 0.0, 100.0, "float64",
        )
        .unwrap();
        assert_eq!(r.path, "generic_slip");
        assert_eq!(r.slip_factor, Some(1.0)); // clamped by min(1.0, ...)
        assert_eq!(r.forward_target, 0.75);
        assert_eq!(r.local_force, [-35.0, -1.75, 0.0]);
        assert_eq!(r.local_torque, [0.0, 28.0, 27.5]);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            call(3, "float64", None).unwrap_err(),
            "unsupported initialized surface_mode 3"
        );
        let zero_denominator = GroundSurfaceProfile {
            speed_denominator: 0.0,
            speed_quadratic_gain: 2.0,
            speed_bias: 0.5,
            force_scale: 1.0,
        };
        assert_eq!(
            call(2, "float64", Some(zero_denominator)).unwrap_err(),
            "surface speed_denominator must be nonzero"
        );
        assert_eq!(
            call(0, "float16", None).unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }
}
