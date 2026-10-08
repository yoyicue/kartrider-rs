//! Port of `scripts/evaluate_post_bs_reset_constraint.py` — executable
//! no-contact fallback constraint producer sub_B2D3C0 plus the UpdateTick
//! selection gate at B261EF..B26358. Pure f64 (no numeric_mode switch);
//! this module's `_add` is a plain `a[i] + b[i]`.
//!
//! The historical ``post_bs_reset`` name is retained for compatibility, but
//! UpdateTick selects this body only on the branch opposite the
//! contact/trigger handlers (including BS). It is therefore never a
//! same-substep post-BS producer.

pub const SELECT_ORDERING: &str =
    "contact/trigger path (including BS/JM/DJ handlers) wins; otherwise non-rail substep calls B2D3C0";
pub const EVALUATE_ORDERING: &str = "clear AF4/AFC/B08 -> add (604*obf660)/E50 to +0x614 -> subtract CC*30 from +0x620 -> optional ACC/AC0/AD8 orientation torque -> clear AA8";

// static_source provenance (Python result key `static_source`).
pub const STATIC_OUTER_IF_VA: &str = "0xB261EF";
pub const STATIC_RAIL_ELSE_GATE_VA: &str = "0xB26353";
pub const STATIC_CALL_VA: &str = "0xB26358";
pub const STATIC_FUNCTION_VA: &str = "0xB2D3C0";
pub const STATIC_FORCE_SINK_VA: &str = "0xB2D3ED-0xB2D43C";
pub const STATIC_TORQUE_SINK_VA: &str = "0xB2D441-0xB2D46E";
pub const STATIC_ORIENTATION_BODY_VA: &str = "0xB2D4A5-0xB2D5B8";

/// Python `select_post_bs_reset_constraint(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct PostBsResetSelection {
    pub selected: bool,
    pub contact_or_trigger: bool,
    pub rail_active: bool,
    pub ordering: &'static str,
}

/// `select_post_bs_reset_constraint(...)` — exact UpdateTick gate at
/// B261EF..B26358 for sub_B2D3C0.
pub fn select_post_bs_reset_constraint(
    current_contact_updated: bool,
    contact_trigger_e38: bool,
    contact_trigger_e4c: bool,
    rail_state_1cc: i64,
) -> PostBsResetSelection {
    let contact_or_trigger =
        current_contact_updated || contact_trigger_e38 || contact_trigger_e4c;
    let rail_active = matches!(rail_state_1cc, 2 | 3);
    let selected = !contact_or_trigger && !rail_active;
    PostBsResetSelection {
        selected,
        contact_or_trigger,
        rail_active,
        ordering: SELECT_ORDERING,
    }
}

/// Python `state_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PostBsResetStateAfter {
    pub selector_af4: bool,
    pub decoded_afc: i64,
    pub selector_b08: bool,
    pub field_aa8: f64,
}

/// Python `evaluate_post_bs_reset_constraint(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct PostBsResetConstraintResult {
    pub force_delta: [f64; 3],
    pub base_torque_delta: [f64; 3],
    pub orientation_torque_delta: [f64; 3],
    pub torque_delta: [f64; 3],
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub state_after: PostBsResetStateAfter,
    pub ordering: &'static str,
}

/// `evaluate_post_bs_reset_constraint(...)`.
pub fn evaluate_post_bs_reset_constraint(
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    vector_604: [f64; 3],
    sub_660: f64,
    divisor_e50: f64,
    angular_velocity_cc: [f64; 3],
    special_orientation_gate: bool,
    field_acc: f64,
    field_ac0: f64,
    field_ad8: f64,
) -> Result<PostBsResetConstraintResult, String> {
    if divisor_e50 == 0.0 {
        return Err("raw(+0xE50) divisor must be nonzero".to_string());
    }
    let force_delta = [
        vector_604[0] * sub_660 / divisor_e50,
        vector_604[1] * sub_660 / divisor_e50,
        vector_604[2] * sub_660 / divisor_e50,
    ];
    let base_torque_delta = [
        -angular_velocity_cc[0] * 30.0,
        -angular_velocity_cc[1] * 30.0,
        -angular_velocity_cc[2] * 30.0,
    ];
    let mut torque_delta = base_torque_delta;
    let mut orientation_delta = [0.0, 0.0, 0.0];
    if special_orientation_gate {
        if field_acc < 0.5 {
            let y_gain = if field_ac0 <= 0.0 { -90.0 } else { 90.0 };
            orientation_delta[1] += (1.0 - field_acc) * y_gain;
        }
        if field_ad8 > 0.5 {
            orientation_delta[0] += (field_ad8 + 1.0) * 90.0;
        } else if field_ad8 < -0.5 {
            orientation_delta[0] -= (1.0 - field_ad8) * 90.0;
        }
        // Plain a[i] + b[i], matching this module's Python `_add`.
        torque_delta = [
            torque_delta[0] + orientation_delta[0],
            torque_delta[1] + orientation_delta[1],
            torque_delta[2] + orientation_delta[2],
        ];
    }
    Ok(PostBsResetConstraintResult {
        force_delta,
        base_torque_delta,
        orientation_torque_delta: orientation_delta,
        torque_delta,
        accumulator_614_before,
        accumulator_614_after: [
            accumulator_614_before[0] + force_delta[0],
            accumulator_614_before[1] + force_delta[1],
            accumulator_614_before[2] + force_delta[2],
        ],
        accumulator_620_before,
        accumulator_620_after: [
            accumulator_620_before[0] + torque_delta[0],
            accumulator_620_before[1] + torque_delta[1],
            accumulator_620_before[2] + torque_delta[2],
        ],
        state_after: PostBsResetStateAfter {
            selector_af4: false,
            decoded_afc: 0,
            selector_b08: false,
            field_aa8: 0.0,
        },
        ordering: EVALUATE_ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_gate() {
        // Constants computed with the Python authority.
        assert!(select_post_bs_reset_constraint(false, false, false, 0).selected);
        assert!(select_post_bs_reset_constraint(false, false, false, 6).selected);
        assert!(!select_post_bs_reset_constraint(true, false, false, 0).selected);
        assert!(!select_post_bs_reset_constraint(false, false, false, 2).selected);
        assert!(!select_post_bs_reset_constraint(false, false, false, 3).selected);
        let r = select_post_bs_reset_constraint(false, true, false, 0);
        assert!(r.contact_or_trigger);
        assert!(!r.selected);
    }

    #[test]
    fn orientation_gate_both_terms() {
        // Constants computed with the Python authority:
        //   evaluate_post_bs_reset_constraint(..., special_orientation_gate=True,
        //     field_acc=0.2, field_ac0=1.0, field_ad8=0.7)
        let r = evaluate_post_bs_reset_constraint(
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 0.25],
            [0.5, -1.0, 2.0],
            80.0,
            4.0,
            [0.1, -0.2, 0.3],
            true,
            0.2,
            1.0,
            0.7,
        )
        .unwrap();
        assert_eq!(r.force_delta, [10.0, -20.0, 40.0]);
        assert_eq!(r.base_torque_delta, [-3.0, 6.0, -9.0]);
        assert_eq!(r.orientation_torque_delta, [153.0, 72.0, 0.0]);
        assert_eq!(r.torque_delta, [150.0, 78.0, -9.0]);
        assert_eq!(r.accumulator_614_after, [11.0, -18.0, 43.0]);
        assert_eq!(r.accumulator_620_after, [150.5, 77.5, -8.75]);
    }

    #[test]
    fn error_guard_matches_python_message() {
        assert_eq!(
            evaluate_post_bs_reset_constraint(
                [0.0; 3], [0.0; 3], [0.0; 3], 80.0, 0.0, [0.0; 3], false, 0.0, 0.0, 0.0,
            )
            .unwrap_err(),
            "raw(+0xE50) divisor must be nonzero"
        );
    }
}
