//! Port of `scripts/evaluate_soft_contact_source.py` — static base source
//! producer for the soft-contact kind8 velocity add, plus the optional
//! low-speed wall-turn source (the only observed producer of kind8 source.z).
//!
//! Numeric discipline: this module is plain f64 (the Python module has no
//! `_f32` helpers); every `sum(...)` routes through
//! [`crate::num::sum_f64`]/[`crate::num::dot3_sum`] (CPython Neumaier), and
//! `min`/`max` route through [`crate::num::min2_f64`]/[`crate::num::max2_f64`]
//! (CPython two-operand semantics).

use crate::num::{dot3_sum, max2_f64, min2_f64, sum_f64};

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2DC80";
pub const STATIC_RANGE_VA: &str = "0xB2E210-0xB2E315";

// static_optional_source provenance (Python result key
// `static_optional_source`, range + constants).
pub const STATIC_OPTIONAL_RANGE_VA: &str = "0xB2E390-0xB2E6B7";
pub const OPTIONAL_SPEED_THRESHOLD: f64 = 100.0;
pub const OPTIONAL_PHASE_GAIN: f64 = 3.0;
pub const OPTIONAL_ANGULAR_CUBIC_GAIN: f64 = 6.0;
pub const OPTIONAL_ANGULAR_BASE: f64 = 2.0;
pub const OPTIONAL_NEGATIVE_SIDE_PHASE_FLOOR: f64 = 1.5;

/// Python result key `formula`.
pub const FORMULA: &str =
    "xy(-1.5*normal - normalize(tangent)*min(1.5*|normal|,0.6*|tangent|)); z=0";

/// `math.sqrt(sum(value * value for value in vector))` — CPython Neumaier
/// square-sum, not a naive fold.
fn length3_sum(values: [f64; 3]) -> f64 {
    sum_f64(&[
        values[0] * values[0],
        values[1] * values[1],
        values[2] * values[2],
    ])
    .sqrt()
}

/// Python `evaluate_soft_contact_base_source(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftContactBaseSource {
    pub velocity_source: [f64; 3],
    pub normal_component: [f64; 3],
    pub tangent_component: [f64; 3],
    pub tangent_remove: f64,
    pub formula: &'static str,
}

/// `evaluate_soft_contact_base_source(...)`: the planar wall response.
/// Component-count validation in Python is unreachable here (fixed
/// `[f64; 3]` inputs).
pub fn evaluate_soft_contact_base_source(
    velocity_c0_before: [f64; 3],
    hit_normal: [f64; 3],
    tangent_gain: f64,
    normal_cap_gain: f64,
    normal_push_gain: f64,
) -> SoftContactBaseSource {
    let dot = dot3_sum(velocity_c0_before, hit_normal);
    let normal = [
        hit_normal[0] * dot,
        hit_normal[1] * dot,
        hit_normal[2] * dot,
    ];
    let tangent = [
        velocity_c0_before[0] - normal[0],
        velocity_c0_before[1] - normal[1],
        velocity_c0_before[2] - normal[2],
    ];
    let normal_length = length3_sum(normal);
    let tangent_length = length3_sum(tangent);
    // `if tangent_length` is falsy on +0.0 exactly like Python.
    let tangent_direction = if tangent_length != 0.0 {
        [
            tangent[0] / tangent_length,
            tangent[1] / tangent_length,
            tangent[2] / tangent_length,
        ]
    } else {
        [0.0, 0.0, 0.0]
    };
    let tangent_remove = min2_f64(
        normal_length * normal_cap_gain,
        tangent_length * tangent_gain,
    );
    // The base source is a planar wall response; the optional low-speed
    // state logic in `evaluate_soft_contact_source` is the only observed
    // producer of kind8 source.z (z stays 0.0 here).
    let velocity_source = [
        -normal_push_gain * normal[0] - tangent_direction[0] * tangent_remove,
        -normal_push_gain * normal[1] - tangent_direction[1] * tangent_remove,
        0.0,
    ];
    SoftContactBaseSource {
        velocity_source,
        normal_component: normal,
        tangent_component: tangent,
        tangent_remove,
        formula: FORMULA,
    }
}

/// Python `optional` sub-dict of `evaluate_soft_contact_source(...)`.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftContactOptional {
    pub phase: f64,
    pub side_test: f64,
    pub scale: f64,
    pub turn_sign: f64,
    pub velocity_extra: [f64; 3],
    pub angular_z_delta: f64,
}

/// Python `evaluate_soft_contact_source(...)` result dict minus
/// `static_source` / `static_optional_source`. Flat exactly like the Python
/// `{**base_result, ...}` merge: `velocity_source` is the (possibly
/// optional-augmented) source, while `normal_component` /
/// `tangent_component` / `tangent_remove` / `formula` come from the base.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftContactSource {
    pub velocity_source: [f64; 3],
    pub normal_component: [f64; 3],
    pub tangent_component: [f64; 3],
    pub tangent_remove: f64,
    pub formula: &'static str,
    pub optional_active: bool,
    pub optional: Option<SoftContactOptional>,
    pub angular_z_delta: f64,
}

/// `evaluate_soft_contact_source(...)`: base source plus the optional
/// low-speed wall-turn source. Returns `Err("float division by zero")` where
/// Python raises `ZeroDivisionError` (the unguarded `-basis_ad0/|basis_ad0|`
/// normalize with a zero-length basis vector).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_soft_contact_source(
    velocity_c0_before: [f64; 3],
    hit_normal: [f64; 3],
    basis_ad0: [f64; 3],
    basis_ab8: [f64; 3],
    drive_selected_slot: f64,
    drive_raw_aa0: f64,
    drive_sign_aac: bool,
    selector_af4: bool,
    flag_188: bool,
) -> Result<SoftContactSource, String> {
    let base = evaluate_soft_contact_base_source(velocity_c0_before, hit_normal, 0.6, 1.5, 1.5);
    let mut source = base.velocity_source;
    let speed = length3_sum(velocity_c0_before);
    let drive_direction = if drive_raw_aa0 == 1.0 {
        1.0
    } else if drive_raw_aa0 == -1.0 {
        -1.0
    } else {
        0.0
    };
    let optional_active = speed < OPTIONAL_SPEED_THRESHOLD
        && !selector_af4
        && flag_188
        && drive_selected_slot == 1.0
        && drive_direction != 0.0;
    let mut angular_z_delta = 0.0;
    let mut optional = None;
    if optional_active {
        let normal_component = base.normal_component;
        let normal_length = length3_sum(normal_component);
        let normal_direction = if normal_length != 0.0 {
            [
                normal_component[0] / normal_length,
                normal_component[1] / normal_length,
                normal_component[2] / normal_length,
            ]
        } else {
            [0.0, 0.0, 0.0]
        };
        let neg_ad0_length = length3_sum(basis_ad0);
        if neg_ad0_length == 0.0 {
            // Python `-value / neg_ad0_length` has no zero guard here.
            return Err("float division by zero".to_string());
        }
        let neg_ad0 = [
            -basis_ad0[0] / neg_ad0_length,
            -basis_ad0[1] / neg_ad0_length,
            -basis_ad0[2] / neg_ad0_length,
        ];
        let mut phase = dot3_sum(hit_normal, neg_ad0);
        let side_test = sum_f64(&[
            hit_normal[0] * basis_ab8[0] * drive_direction,
            hit_normal[1] * basis_ab8[1] * drive_direction,
            hit_normal[2] * basis_ab8[2] * drive_direction,
        ]);
        let cubic;
        if side_test < 0.0 {
            phase = 2.0 - phase;
            // Runtime optional-contact rows clamp this branch at 1.5:
            // phases 1.47, 1.36, 1.27, and 1.19 all write exactly +11
            // (6*1.5+2), while phases above 1.5 remain linear.
            cubic = max2_f64(phase, OPTIONAL_NEGATIVE_SIDE_PHASE_FLOOR);
        } else {
            cubic = phase * phase * phase;
        }
        let scale = OPTIONAL_PHASE_GAIN * phase + 1.0;
        let turn_sign = drive_direction * if drive_sign_aac { -1.0 } else { 1.0 };
        let extra = [
            -normal_direction[0] * scale + basis_ab8[0] * scale * turn_sign,
            -normal_direction[1] * scale + basis_ab8[1] * scale * turn_sign,
            -normal_direction[2] * scale + basis_ab8[2] * scale * turn_sign,
        ];
        source = [
            source[0] + extra[0],
            source[1] + extra[1],
            source[2] + extra[2],
        ];
        angular_z_delta =
            (OPTIONAL_ANGULAR_CUBIC_GAIN * cubic + OPTIONAL_ANGULAR_BASE) * turn_sign;
        optional = Some(SoftContactOptional {
            phase,
            side_test,
            scale,
            turn_sign,
            velocity_extra: extra,
            angular_z_delta,
        });
    }
    Ok(SoftContactSource {
        velocity_source: source,
        normal_component: base.normal_component,
        tangent_component: base.tangent_component,
        tangent_remove: base.tangent_remove,
        formula: base.formula,
        optional_active,
        optional,
        angular_z_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_source_head_on_wall() {
        // Constants computed with the Python authority:
        //   evaluate_soft_contact_base_source(
        //     velocity_c0_before=[-3.0,1.0,0.0], hit_normal=[1.0,0.0,0.0])
        let r = evaluate_soft_contact_base_source([-3.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.6, 1.5, 1.5);
        assert_eq!(r.normal_component, [-3.0, -0.0, -0.0]);
        assert_eq!(r.tangent_component, [0.0, 1.0, 0.0]);
        assert_eq!(r.tangent_remove, 0.6);
        assert_eq!(r.velocity_source, [4.5, -0.6, 0.0]);
        assert_eq!(r.formula, FORMULA);
    }

    #[test]
    fn base_source_zero_tangent_keeps_zero_direction() {
        // Pure normal approach: tangent is zero, direction falls back to
        // [0,0,0] and tangent_remove is 0.
        let r = evaluate_soft_contact_base_source([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.6, 1.5, 1.5);
        assert_eq!(r.tangent_remove, 0.0);
        assert_eq!(r.velocity_source, [3.0, 0.0, 0.0]);
    }

    #[test]
    fn optional_source_activates_below_speed_threshold() {
        // Constants computed with the Python authority:
        //   evaluate_soft_contact_source(
        //     velocity_c0_before=[-3.0,1.0,0.0], hit_normal=[1.0,0.0,0.0],
        //     basis_ad0=[0.0,-1.0,0.0], basis_ab8=[0.0,0.0,1.0],
        //     drive_selected_slot=1.0, drive_raw_aa0=1.0, drive_sign_aac=False,
        //     selector_af4=False, flag_188=True)
        let r = evaluate_soft_contact_source(
            [-3.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
            1.0,
            false,
            false,
            true,
        )
        .expect("unit basis does not divide by zero");
        assert!(r.optional_active);
        let optional = r.optional.expect("optional payload present");
        assert_eq!(optional.phase, 0.0);
        assert_eq!(optional.side_test, 0.0);
        assert_eq!(optional.scale, 1.0);
        assert_eq!(optional.turn_sign, 1.0);
        assert_eq!(optional.velocity_extra, [1.0, 0.0, 1.0]);
        assert_eq!(r.velocity_source, [5.5, -0.6, 1.0]);
        assert_eq!(r.angular_z_delta, 2.0);
    }

    #[test]
    fn optional_source_inactive_without_drive() {
        let r = evaluate_soft_contact_source(
            [-3.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
            0.0,
            false,
            false,
            true,
        )
        .expect("unit basis does not divide by zero");
        assert!(!r.optional_active);
        assert!(r.optional.is_none());
        assert_eq!(r.angular_z_delta, 0.0);
        assert_eq!(r.velocity_source, [4.5, -0.6, 0.0]);
    }

    #[test]
    fn optional_source_negative_side_clamps_cubic() {
        // side_test < 0 branch: phase = 2 - phase, cubic = max(phase, 1.5).
        //   evaluate_soft_contact_source(
        //     velocity_c0_before=[0.0,-3.0,0.5], hit_normal=[0.0,1.0,0.0],
        //     basis_ad0=[1.0,0.0,0.0], basis_ab8=[0.0,1.0,0.0],
        //     drive_selected_slot=1.0, drive_raw_aa0=-1.0, drive_sign_aac=False,
        //     selector_af4=False, flag_188=True)
        let r = evaluate_soft_contact_source(
            [0.0, -3.0, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            1.0,
            -1.0,
            false,
            false,
            true,
        )
        .expect("unit basis does not divide by zero");
        let optional = r.optional.expect("optional payload present");
        assert_eq!(optional.phase, 2.0); // 2.0 - (+0.0)
        assert!(optional.side_test.is_sign_negative());
        assert_eq!(optional.side_test, -1.0);
        assert_eq!(r.angular_z_delta, -14.0); // (6*2+2)*(-1)
    }

    #[test]
    fn zero_basis_ad0_is_division_by_zero() {
        // Python raises ZeroDivisionError("float division by zero") on the
        // unguarded neg_ad0 normalize.
        let error = evaluate_soft_contact_source(
            [-3.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            1.0,
            1.0,
            false,
            false,
            true,
        )
        .expect_err("zero basis must error");
        assert_eq!(error, "float division by zero");
    }
}
