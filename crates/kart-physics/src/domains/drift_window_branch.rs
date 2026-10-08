//! Port of `scripts/evaluate_drift_window_branch.py` — executable +0xB08
//! drift-window branch inside ground-drive forces (B2B074..B2B321). Pure
//! f64 (the Python module has no numeric_mode switch).

use crate::num::max2_f64;

pub const ORDERING: &str = "entry B08 selects drift force -> submit force/torque -> initialize B0C or max(0,B0C-dt) -> on expiry clear B08 and set B1C settlement";

// static_source provenance (Python result key `static_source`).
pub const STATIC_BODY_VA: &str = "0xB2B074-0xB2B321";
pub const STATIC_FORCE_SINK_VA: &str = "0xB2C111";
pub const STATIC_TORQUE_SINK_VA: &str = "0xB2C177";

/// Python `state_after` dict (`None` fields become `Option`).
#[derive(Debug, Clone, PartialEq)]
pub struct DriftWindowStateAfter {
    pub blend_ab0: f64,
    pub blend_ab4: f64,
    pub selector_b08: bool,
    pub timer_b0c: f64,
    pub field_b00: Option<f64>,
    pub snapshot_5d0: Option<f64>,
    pub b1c_charge_settlement: bool,
    pub b20: Option<f64>,
    pub b2c: Option<f64>,
}

/// Python result dict for the selected path, minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftWindowSelected {
    pub longitudinal: f64,
    pub lateral: f64,
    pub local_force: [f64; 3],
    pub local_torque: [f64; 3],
    pub af0_torque_gate: f64,
    pub timer_initialized_after_force: bool,
    pub timer_expired_after_force: bool,
    pub state_after: DriftWindowStateAfter,
    pub ordering: &'static str,
}

/// Python result: either `{"selected": False, "reason": "selector_b08_clear"}`
/// or the full selected-path row.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftWindowBranchResult {
    /// `{"selected": False, "reason": "selector_b08_clear"}`
    NotSelected,
    /// `{"selected": True, ...}`
    Selected(DriftWindowSelected),
}

/// `evaluate_drift_window_branch(...)`.
pub fn evaluate_drift_window_branch(
    selector_b08: bool,
    timer_b0c: f64,
    dt: f64,
    configured_duration_734: f64,
    mass_scale: f64,
    base_longitudinal_gain: f64,
    signed_axis: f64,
    scaled_tune_660: f64,
    drift_gain_728: f64,
    tune_758: f64,
    axis_lat: f64,
    drive_raw_aa0: f64,
    debug_force_scale: f64,
) -> DriftWindowBranchResult {
    if !selector_b08 {
        return DriftWindowBranchResult::NotSelected;
    }

    let longitudinal = 0.0;
    let lateral = -mass_scale
        * base_longitudinal_gain
        * signed_axis
        * scaled_tune_660
        * drift_gain_728
        * debug_force_scale;
    let force_x = lateral;
    let timer_initialized = timer_b0c <= 0.0;
    let timer_after = if timer_initialized {
        configured_duration_734
    } else {
        max2_f64(0.0, timer_b0c - dt)
    };
    let expired_after_force = !timer_initialized && timer_b0c > 0.0 && timer_after == 0.0;
    let state_after = DriftWindowStateAfter {
        blend_ab0: 0.0,
        blend_ab4: 0.0,
        selector_b08: !expired_after_force,
        timer_b0c: timer_after,
        field_b00: if timer_initialized {
            Some(configured_duration_734 * 2.0)
        } else {
            None
        },
        snapshot_5d0: if timer_initialized {
            Some(drive_raw_aa0)
        } else {
            None
        },
        b1c_charge_settlement: expired_after_force,
        b20: if expired_after_force { Some(0.0) } else { None },
        b2c: if expired_after_force { Some(0.0) } else { None },
    };
    DriftWindowBranchResult::Selected(DriftWindowSelected {
        longitudinal,
        lateral,
        local_force: [force_x, 0.0, 0.0],
        // This branch clears +0xAF0 before the common torque block, so the
        // AF0<=10 half-scale gate is always selected.
        local_torque: [0.0, -force_x * tune_758 * 0.5, -axis_lat * lateral],
        af0_torque_gate: 0.5,
        timer_initialized_after_force: timer_initialized,
        timer_expired_after_force: expired_after_force,
        state_after,
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(timer_b0c: f64) -> DriftWindowBranchResult {
        evaluate_drift_window_branch(
            true, timer_b0c, 0.002, 0.25, 80.0, 2.0, 1.0, 0.5, 1.5, 0.8, 0.5, 1.0, 1.0,
        )
    }

    #[test]
    fn timer_initializes() {
        // Constants computed with the Python authority:
        //   evaluate_drift_window_branch(selector_b08=True, timer_b0c=0.0, ...)
        let DriftWindowBranchResult::Selected(r) = call(0.0) else {
            panic!("selector set must select the branch")
        };
        assert_eq!(r.lateral, -120.0);
        assert_eq!(r.local_torque, [0.0, 48.0, 60.0]);
        assert!(r.timer_initialized_after_force);
        assert!(!r.timer_expired_after_force);
        assert_eq!(r.state_after.timer_b0c, 0.25);
        assert_eq!(r.state_after.field_b00, Some(0.5));
        assert_eq!(r.state_after.snapshot_5d0, Some(1.0));
        assert_eq!(r.state_after.b20, None);
    }

    #[test]
    fn timer_expires() {
        // Constants computed with the Python authority (timer_b0c=0.001).
        let DriftWindowBranchResult::Selected(r) = call(0.001) else {
            panic!("selector set must select the branch")
        };
        assert!(r.timer_expired_after_force);
        assert!(!r.state_after.selector_b08);
        assert!(r.state_after.b1c_charge_settlement);
        assert_eq!(r.state_after.field_b00, None);
        assert_eq!(r.state_after.b20, Some(0.0));
        assert_eq!(r.state_after.b2c, Some(0.0));
    }

    #[test]
    fn selector_clear_not_selected() {
        assert_eq!(
            evaluate_drift_window_branch(
                false, 0.0, 0.002, 0.25, 80.0, 2.0, 1.0, 0.5, 1.5, 0.8, 0.5, 1.0, 1.0,
            ),
            DriftWindowBranchResult::NotSelected
        );
    }
}
