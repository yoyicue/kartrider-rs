//! Port of `scripts/evaluate_mz_magnet_trigger.py` — exact MZ
//! authored-trigger velocity replacement and state16 transition.

/// Python result key `policy`.
pub const POLICY: &str = "The MZ token writes +0xC0 before testing whether state13..16 is already active; the gate suppresses only FX/state16, not the velocity replacement.";

/// Python result key `static_order`.
pub const STATIC_ORDER: &str = "MZ parse -> target-current -> z=0 -> scale -> direct C0 write 0xB2C838..0xB2C843 -> special-family gate -> optional state16/3000ms 0xB2C870";

/// Python `state_transition` sub-dict: `{"state": 16, "remaining_ms": 3000}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MzStateTransition {
    pub state: i64,
    pub remaining_ms: i64,
}

/// Python `evaluate_mz_magnet_trigger(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct MzMagnetTrigger {
    pub accepted: bool,
    pub velocity_c0_after: [f64; 3],
    /// Present only when accepted (Python `None` otherwise).
    pub state_transition: Option<MzStateTransition>,
    pub velocity_write_occurs_before_state_family_gate: bool,
    pub policy: &'static str,
    pub static_order: &'static str,
}

/// `evaluate_mz_magnet_trigger(...)` — B2C802 computes authored target minus
/// sub_6F7C50(this), B2C80D forces local z to zero, and B2C825 scales before
/// the direct +0xC0 write. Component-count validation in Python is
/// unreachable here (fixed `[f64; 3]` inputs).
pub fn evaluate_mz_magnet_trigger(
    current_reference_xyz: [f64; 3],
    authored_target_xyz: [f64; 3],
    velocity_scale: f64,
    special_state_family_already_active: bool,
) -> MzMagnetTrigger {
    let velocity = [
        (authored_target_xyz[0] - current_reference_xyz[0]) * velocity_scale,
        (authored_target_xyz[1] - current_reference_xyz[1]) * velocity_scale,
        0.0,
    ];
    let accepted = !special_state_family_already_active;
    MzMagnetTrigger {
        accepted,
        velocity_c0_after: velocity,
        state_transition: if accepted {
            Some(MzStateTransition { state: 16, remaining_ms: 3000 })
        } else {
            None
        },
        velocity_write_occurs_before_state_family_gate: true,
        policy: POLICY,
        static_order: STATIC_ORDER,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_velocity_before_family_gate() {
        // Constants computed with the Python authority:
        //   evaluate_mz_magnet_trigger(current_reference_xyz=[1,2,3],
        //     authored_target_xyz=[4,8,9], velocity_scale=2.0,
        //     special_state_family_already_active=False)
        let r = evaluate_mz_magnet_trigger([1.0, 2.0, 3.0], [4.0, 8.0, 9.0], 2.0, false);
        assert_eq!(r.velocity_c0_after, [6.0, 12.0, 0.0]);
        assert!(r.accepted);
        assert_eq!(r.state_transition, Some(MzStateTransition { state: 16, remaining_ms: 3000 }));
        //   Gate active: velocity still replaced, transition suppressed.
        let r = evaluate_mz_magnet_trigger([1.0, 2.0, 3.0], [4.0, 8.0, 9.0], 2.0, true);
        assert_eq!(r.velocity_c0_after, [6.0, 12.0, 0.0]);
        assert!(!r.accepted);
        assert_eq!(r.state_transition, None);
    }
}
