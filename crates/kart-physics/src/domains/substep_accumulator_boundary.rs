//! Port of `scripts/evaluate_substep_accumulator_boundary.py` — exact
//! pending-to-active accumulator transfer at sub_B28B50.

pub const OPERATION: &str =
    "replace active accumulators from pending B74/B80, then clear B74/B80/B90";

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB28B50";
pub const STATIC_FUNCTION_RVA: &str = "0x728B50";
pub const STATIC_TRANSFER_VAS: &str = "0xB28B5C-0xB28B99";
pub const STATIC_CLEAR_VAS: &str = "0xB28B9F-0xB28BFE";

/// Python result dict minus `static_source`. The three-component length
/// guard ("pending vectors must have three components") is compile-time in
/// Rust.
#[derive(Debug, Clone, PartialEq)]
pub struct SubstepAccumulatorBoundaryResult {
    pub active_force_614_before: Option<[f64; 3]>,
    pub active_torque_620_before: Option<[f64; 3]>,
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub pending_force_b74_after: [f64; 3],
    pub pending_torque_b80_after: [f64; 3],
    pub pending_b90_after: [f64; 3],
    pub operation: &'static str,
}

/// `evaluate_substep_accumulator_boundary(...)`.
pub fn evaluate_substep_accumulator_boundary(
    pending_force_b74: [f64; 3],
    pending_torque_b80: [f64; 3],
    pending_b90: [f64; 3],
    active_force_614_before: Option<[f64; 3]>,
    active_torque_620_before: Option<[f64; 3]>,
) -> SubstepAccumulatorBoundaryResult {
    // pending_b90 is only cleared by this boundary, never read (same as the
    // Python authority, which accepts it and emits the zeroed-after value).
    let _ = pending_b90;
    SubstepAccumulatorBoundaryResult {
        active_force_614_before,
        active_torque_620_before,
        accumulator_614_after: pending_force_b74,
        accumulator_620_after: pending_torque_b80,
        pending_force_b74_after: [0.0, 0.0, 0.0],
        pending_torque_b80_after: [0.0, 0.0, 0.0],
        pending_b90_after: [0.0, 0.0, 0.0],
        operation: OPERATION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_and_clears() {
        // Constants computed with the Python authority:
        //   evaluate_substep_accumulator_boundary(pending_force_b74=[1,-2,3],
        //     pending_torque_b80=[0.5,-0.5,0.25], pending_b90=[4,5,6],
        //     active_force_614_before=[7,8,9], active_torque_620_before=[-1,-2,-3])
        let r = evaluate_substep_accumulator_boundary(
            [1.0, -2.0, 3.0],
            [0.5, -0.5, 0.25],
            [4.0, 5.0, 6.0],
            Some([7.0, 8.0, 9.0]),
            Some([-1.0, -2.0, -3.0]),
        );
        assert_eq!(r.accumulator_614_after, [1.0, -2.0, 3.0]);
        assert_eq!(r.accumulator_620_after, [0.5, -0.5, 0.25]);
        assert_eq!(r.pending_force_b74_after, [0.0, 0.0, 0.0]);
        assert_eq!(r.pending_torque_b80_after, [0.0, 0.0, 0.0]);
        assert_eq!(r.pending_b90_after, [0.0, 0.0, 0.0]);
        assert_eq!(r.active_force_614_before, Some([7.0, 8.0, 9.0]));
    }
}
