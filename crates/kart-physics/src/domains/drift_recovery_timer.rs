//! Port of `scripts/evaluate_drift_recovery_timer.py` — executable
//! selector/timer schedule for the drift recovery branch, plus the
//! surface-mode-1 recovery force body at B2B396..B2C186. Pure f64 (the
//! Python module has no numeric_mode switch).

use crate::domains::surface_tune_defaults::{
    active_non_surface1_profile, surface1_recovery_profiles, NonSurfaceProfile,
    RecoverySurfaceProfile,
};
use crate::num::{max2_f64, min2_f64};

pub const FORCE_ORDERING: &str = "select subpath -> update blends -> calculate force/torque components -> decay 65C/AF8 timers -> submit precomputed force/torque";
pub const SUBSTEP_ORDERING: &str = "entry gates -> compute recovery components -> surface1 max(0,+0x65C-dt) -> common max(0,+0xAF8-dt) -> force/torque sinks";
pub const SUBSTEP_SCOPE: &str =
    "selector/timer schedule; use evaluate_surface1_recovery_force for the selected force body";

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2AA70";
pub const STATIC_RECOVERY_BODY_VA: &str = "0xB2B396-0xB2B7E8";
pub const STATIC_FORCE_SINK_VA: &str = "0xB2C111";
pub const STATIC_TORQUE_SINK_VA: &str = "0xB2C177";

/// `_approach_one(value, rate)`.
fn approach_one(value: f64, rate: f64) -> f64 {
    value + (1.0 - value) * rate
}

/// Python `evaluate_surface1_recovery_force(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct Surface1RecoveryForceResult {
    pub path: &'static str,
    pub surface_mode: i64,
    pub surface_profile_source: &'static str,
    pub surface_force_scale_read: f64,
    pub surface_force_scale_applied: bool,
    pub speed_ratio_sq: f64,
    pub blend_ab0_before: f64,
    pub blend_ab0_after: f64,
    pub blend_ab4_before: f64,
    pub blend_ab4_after: f64,
    pub forward_target: f64,
    pub longitudinal: f64,
    pub lateral: f64,
    pub local_force: [f64; 3],
    pub local_torque: [f64; 3],
    pub af0_torque_gate: f64,
    pub ordering: &'static str,
}

/// `evaluate_surface1_recovery_force(...)`.
///
/// The caller supplies decoded runtime/tuning values. This intentionally does
/// not guess the producer semantics of the contact-basis or correction terms.
pub fn evaluate_surface1_recovery_force(
    selector_af4: bool,
    speed_magnitude: f64,
    speed_profile_denominator: f64,
    blend_ab0: f64,
    blend_ab4: f64,
    active_ab0_rate: f64,
    active_ab4_rate: f64,
    timer_ab0_rate: f64,
    contact_basis_product: f64,
    ground_forward_term: f64,
    drive_direction: f64,
    normalized_velocity: f64,
    longitudinal_correction: f64,
    lateral_correction: f64,
    mass_scale: f64,
    base_longitudinal_gain: f64,
    base_lateral_gain: f64,
    ground_force_multiplier: f64,
    tune_758: f64,
    axis_long: f64,
    axis_lat: f64,
    active_surface: Option<RecoverySurfaceProfile>,
    timer_surface: Option<RecoverySurfaceProfile>,
    field_af0: f64,
    torque_y_scale: f64,
    surface_mode: i64,
    non_surface_profile: Option<NonSurfaceProfile>,
) -> Result<Surface1RecoveryForceResult, String> {
    if speed_profile_denominator == 0.0 {
        return Err("speed_profile_denominator must be nonzero".to_string());
    }
    let defaults = surface1_recovery_profiles();
    let active_surface_source: &'static str = if active_surface.is_some() {
        "caller_override"
    } else {
        "decoded_playkart_default"
    };
    let timer_surface_source: &'static str = if timer_surface.is_some() {
        "caller_override"
    } else {
        "decoded_playkart_default"
    };
    let active_surface = active_surface.unwrap_or(defaults.active_surface);
    let timer_surface = timer_surface.unwrap_or(defaults.timer_surface);
    // Python `(speed / denominator) ** 2` is a libm pow() call; powf(2.0)
    // issues the identical call on the same platform.
    let speed_ratio_sq = min2_f64(
        1.0,
        (speed_magnitude / speed_profile_denominator).powf(2.0),
    );

    let path: &'static str;
    let mut non_surface_source: &'static str = "not_used";
    let blend_ab0_after: f64;
    let blend_ab4_after: f64;
    let forward_target: f64;
    let longitudinal_gain: f64;
    let lateral_gain: f64;
    let surface_force_scale: f64;
    let common_scale: f64;
    if surface_mode != 1 {
        path = if selector_af4 {
            "active_latch_non_surface1"
        } else {
            "recovery_gate_non_surface1"
        };
        blend_ab0_after = blend_ab0;
        blend_ab4_after = blend_ab4;
        if selector_af4 {
            non_surface_source = if non_surface_profile.is_some() {
                "caller_override"
            } else {
                "decoded_playkart_default"
            };
            let profile = match non_surface_profile {
                Some(profile) => profile,
                None => active_non_surface1_profile(surface_mode)?,
            };
            forward_target = contact_basis_product;
            longitudinal_gain = base_longitudinal_gain + profile.longitudinal_gain_offset;
            lateral_gain = base_lateral_gain + profile.lateral_gain_offset;
            surface_force_scale = profile.force_scale;
        } else {
            forward_target = ground_forward_term;
            longitudinal_gain = base_longitudinal_gain;
            lateral_gain = base_lateral_gain;
            surface_force_scale = 1.0;
        }
        common_scale = ground_force_multiplier * surface_force_scale;
    } else if selector_af4 {
        path = "active_latch";
        blend_ab0_after = approach_one(blend_ab0, active_ab0_rate * speed_ratio_sq);
        blend_ab4_after = approach_one(blend_ab4, active_ab4_rate);
        forward_target = contact_basis_product * active_surface.forward_target_gain;
        longitudinal_gain = base_longitudinal_gain + active_surface.longitudinal_gain_offset;
        lateral_gain = base_lateral_gain + active_surface.lateral_gain_offset - blend_ab4_after;
        surface_force_scale = active_surface.force_scale;
        // B2B695..B2B7A5 reads the selected second-record +0x0C into var_60,
        // but the surface-1 force expression does not consume it.
        common_scale = ground_force_multiplier * min2_f64(1.0, blend_ab0_after);
    } else {
        path = "timer_only";
        blend_ab0_after = approach_one(blend_ab0, timer_ab0_rate);
        blend_ab4_after = 0.0;
        forward_target = ground_forward_term * timer_surface.forward_target_gain;
        longitudinal_gain = base_longitudinal_gain + timer_surface.longitudinal_gain_offset;
        lateral_gain = base_lateral_gain + timer_surface.lateral_gain_offset;
        surface_force_scale = timer_surface.force_scale;
        // Likewise, timer-only reads second-record +0x18 but does not multiply
        // it into the surface-1 force body.
        common_scale = ground_force_multiplier * min2_f64(1.0, blend_ab0_after);
    }

    let blend_ab0_after = min2_f64(1.0, blend_ab0_after);
    let longitudinal = common_scale
        * mass_scale
        * longitudinal_gain
        * (forward_target * drive_direction - normalized_velocity - longitudinal_correction);
    let lateral = common_scale
        * mass_scale
        * lateral_gain
        * (-normalized_velocity + lateral_correction);
    let force_x = longitudinal + lateral;
    let af0_torque_gate = if field_af0 <= 10.0 { 0.5 } else { 1.0 };
    let torque_y = -force_x * tune_758 * af0_torque_gate * torque_y_scale;
    let torque_z = axis_long * longitudinal - axis_lat * lateral;
    Ok(Surface1RecoveryForceResult {
        path,
        surface_mode,
        surface_profile_source: if surface_mode != 1 {
            non_surface_source
        } else if selector_af4 {
            active_surface_source
        } else {
            timer_surface_source
        },
        surface_force_scale_read: surface_force_scale,
        surface_force_scale_applied: surface_mode != 1 && selector_af4,
        speed_ratio_sq,
        blend_ab0_before: blend_ab0,
        blend_ab0_after,
        blend_ab4_before: blend_ab4,
        blend_ab4_after,
        forward_target,
        longitudinal,
        lateral,
        local_force: [force_x, 0.0, 0.0],
        local_torque: [0.0, torque_y, torque_z],
        af0_torque_gate,
        ordering: FORCE_ORDERING,
    })
}

/// `seed_recovery_timer(field_af0)`.
pub fn seed_recovery_timer(field_af0: f64) -> f64 {
    field_af0 * field_af0 / 1440.0
}

/// Python selector snapshot dict in `apply_selector_transition`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectorState {
    pub selector_af4: bool,
    pub selector_b08: bool,
    pub selector_b5c: bool,
    pub selector_f08: bool,
    pub selector_fec: i64,
    pub selector_ff8: i64,
    pub recovery_timer: f64,
}

/// Python `apply_selector_transition(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectorTransitionResult {
    pub applied: bool,
    pub reason: &'static str,
    pub before: SelectorState,
    pub after: SelectorState,
}

/// `apply_selector_transition(...)`.
pub fn apply_selector_transition(
    enable: bool,
    decoded_e8c: i64,
    surface_mode: i64,
    selector_af4: bool,
    selector_b08: bool,
    selector_b5c: bool,
    selector_f08: bool,
    selector_fec: i64,
    selector_ff8: i64,
    field_af8: f64,
    field_adc: f64,
    contact_188: bool,
    drive_raw_aa0: f64,
    field_af0: f64,
    recovery_timer: f64,
) -> SelectorTransitionResult {
    let before = SelectorState {
        selector_af4,
        selector_b08,
        selector_b5c,
        selector_f08,
        selector_fec,
        selector_ff8,
        recovery_timer,
    };
    if decoded_e8c != 0 {
        return SelectorTransitionResult {
            applied: false,
            reason: "decoded_e8c_nonzero",
            before,
            after: before,
        };
    }
    let mut selector_af4 = selector_af4;
    let mut selector_b08 = selector_b08;
    let mut selector_b5c = selector_b5c;
    let mut selector_f08 = selector_f08;
    let mut selector_fec = selector_fec;
    let mut selector_ff8 = selector_ff8;
    let mut recovery_timer = recovery_timer;
    if enable {
        if field_af8 <= 0.0 {
            selector_af4 = true;
            selector_b08 = true;
            selector_b5c = field_adc > 0.0;
            if !contact_188 {
                selector_f08 = true;
            }
        }
        if selector_ff8 != 0 {
            selector_fec = selector_ff8;
        }
        if drive_raw_aa0 > 0.0 {
            selector_ff8 = 1;
        } else if drive_raw_aa0 < 0.0 {
            selector_ff8 = 2;
        }
    } else {
        if surface_mode == 1 && selector_af4 && recovery_timer <= 0.0 {
            recovery_timer = seed_recovery_timer(field_af0);
        }
        selector_af4 = false;
        selector_f08 = false;
    }
    let after = SelectorState {
        selector_af4,
        selector_b08,
        selector_b5c,
        selector_f08,
        selector_fec,
        selector_ff8,
        recovery_timer,
    };
    SelectorTransitionResult {
        applied: true,
        reason: if enable { "enable" } else { "disable" },
        before,
        after,
    }
}

/// Python `evaluate_recovery_substep(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoverySubstepResult {
    pub branch_for_force: bool,
    pub recovery_timer_before: f64,
    pub recovery_timer_after: f64,
    pub timer_expired_after_force: bool,
    pub timer_expired_before_sink: bool,
    pub field_af8_before: f64,
    pub field_af8_after: f64,
    pub field_af8_expired_after_force: bool,
    pub field_af8_expired_before_sink: bool,
    pub ordering: &'static str,
    pub scope: &'static str,
}

/// `evaluate_recovery_substep(...)`.
pub fn evaluate_recovery_substep(
    surface_mode: i64,
    selector_af4: bool,
    field_af8: f64,
    recovery_timer: f64,
    other_recovery_gate: bool,
    dt: f64,
) -> RecoverySubstepResult {
    // Selection uses entry state. The decrement at 0xB2B7B8..0xB2B7E8 is
    // after component calculation but before the sinks at B2C111/B2C177.
    let branch_for_force =
        selector_af4 || field_af8 > 0.0 || recovery_timer > 0.0 || other_recovery_gate;
    let mut timer_after = recovery_timer;
    let mut field_af8_after = field_af8;
    if branch_for_force && surface_mode == 1 {
        timer_after = max2_f64(0.0, recovery_timer - dt);
    }
    if branch_for_force {
        field_af8_after = max2_f64(0.0, field_af8 - dt);
    }
    RecoverySubstepResult {
        branch_for_force,
        recovery_timer_before: recovery_timer,
        recovery_timer_after: timer_after,
        timer_expired_after_force: branch_for_force && recovery_timer > 0.0 && timer_after == 0.0,
        timer_expired_before_sink: branch_for_force && recovery_timer > 0.0 && timer_after == 0.0,
        field_af8_before: field_af8,
        field_af8_after,
        field_af8_expired_after_force: branch_for_force
            && field_af8 > 0.0
            && field_af8_after == 0.0,
        field_af8_expired_before_sink: branch_for_force
            && field_af8 > 0.0
            && field_af8_after == 0.0,
        ordering: SUBSTEP_ORDERING,
        scope: SUBSTEP_SCOPE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call_force(selector_af4: bool) -> Surface1RecoveryForceResult {
        evaluate_surface1_recovery_force(
            selector_af4, 50.0, 340.0, 0.25, 0.25, 0.1, 0.2, 0.05, 0.75, 1.25, 1.0, 0.5, 0.125,
            -0.25, 80.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.5, None, None, 5.0, 1.0, 1, None,
        )
        .unwrap()
    }

    #[test]
    fn surface1_active_latch() {
        // Constants computed with the Python authority:
        //   evaluate_surface1_recovery_force(selector_af4=True, surface_mode=1, ...)
        let r = call_force(true);
        assert_eq!(r.path, "active_latch");
        assert_eq!(r.speed_ratio_sq, 0.021626297577854673);
        assert_eq!(r.blend_ab0_after, 0.2516219723183391);
        assert_eq!(r.blend_ab4_after, 0.4);
        assert_eq!(r.longitudinal, 5.032439446366782);
        assert_eq!(r.lateral, -16.607050173010382);
        assert_eq!(r.local_force, [-11.5746107266436, 0.0, 0.0]);
        assert_eq!(r.local_torque, [0.0, 4.629844290657441, 10.819744809688583]);
    }

    #[test]
    fn surface1_timer_only() {
        // Constants computed with the Python authority (selector_af4=False).
        let r = call_force(false);
        assert_eq!(r.path, "timer_only");
        assert_eq!(r.blend_ab0_after, 0.2875);
        assert_eq!(r.blend_ab4_after, 0.0);
        assert_eq!(r.longitudinal, 28.75);
        assert_eq!(r.local_torque, [0.0, -1.1500000000000001, 27.3125]);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            evaluate_surface1_recovery_force(
                true, 50.0, 0.0, 0.25, 0.25, 0.1, 0.2, 0.05, 0.75, 1.25, 1.0, 0.5, 0.125,
                -0.25, 80.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.5, None, None, 5.0, 1.0, 1, None,
            )
            .unwrap_err(),
            "speed_profile_denominator must be nonzero"
        );
        // surface_mode=3 with the active latch and no override profile hits
        // the decoded-defaults ValueError.
        assert_eq!(
            evaluate_surface1_recovery_force(
                true, 50.0, 340.0, 0.25, 0.25, 0.1, 0.2, 0.05, 0.75, 1.25, 1.0, 0.5, 0.125,
                -0.25, 80.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.5, None, None, 5.0, 1.0, 3, None,
            )
            .unwrap_err(),
            "unsupported initialized surface_mode 3"
        );
    }

    #[test]
    fn seed_matches_python() {
        // seed_recovery_timer(12.0) == 0.1 per the Python authority.
        assert_eq!(seed_recovery_timer(12.0), 0.1);
    }

    #[test]
    fn recovery_substep_expiry() {
        // Constants computed with the Python authority:
        //   evaluate_recovery_substep(surface_mode=1, field_af8=0.1,
        //     recovery_timer=0.001, dt=0.002)
        let r = evaluate_recovery_substep(1, false, 0.1, 0.001, false, 0.002);
        assert!(r.branch_for_force);
        assert_eq!(r.recovery_timer_after, 0.0);
        assert!(r.timer_expired_after_force);
        assert_eq!(r.field_af8_after, 0.098);
        assert!(!r.field_af8_expired_after_force);
    }

    #[test]
    fn selector_transition_disable_seeds_timer() {
        // Constants computed with the Python authority (enable=False,
        // selector_af4=True, field_af0=5.0, recovery_timer=0.0).
        let r = apply_selector_transition(
            false, 0, 1, true, true, true, true, 0, 0, 0.0, 0.5, false, 1.0, 5.0, 0.0,
        );
        assert!(r.applied);
        assert_eq!(r.reason, "disable");
        assert!(!r.after.selector_af4);
        assert!(!r.after.selector_f08);
        assert!(r.after.selector_b08); // unchanged by disable
        assert_eq!(r.after.recovery_timer, 0.017361111111111112);
    }
}
