//! Port of `scripts/evaluate_fall_clamp_state_replacement.py` — exact
//! immediate fall-clamp state replacement from BD7840 -> B16290.

/// Python result key `predicate`.
pub const PREDICATE: &str = "position_90_before.z < drop_threshold";

// source provenance (Python result key `source`).
pub const SOURCE_SELECTOR_VA: &str = "0xBD7840";
pub const SOURCE_WRITER_VA: &str = "0xB16290";
pub const SOURCE_WRITER_RVA: &str = "0x716290";
pub const SOURCE_CALLSITE_VA: &str = "0xBD788C";
pub const SOURCE_CALLSITE_RVA: &str = "0x7D788C";

/// Python `evaluate_fall_clamp_state_replacement(...)` result dict minus
/// `source`.
#[derive(Debug, Clone, PartialEq)]
pub struct FallClampStateReplacement {
    pub selected: bool,
    pub terminal_replacement: bool,
    pub predicate: &'static str,
    pub drop_threshold: f64,
    pub position_90_before: [f64; 3],
    pub position_90_after: [f64; 3],
    /// `replacement_position` when selected, else Python `None`.
    pub previous_position_70_after: Option<[f64; 3]>,
    pub matrix_9c_before: [f64; 9],
    pub matrix_9c_after: [f64; 9],
    pub velocity_c0_before: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub angular_velocity_cc_before: [f64; 3],
    pub angular_velocity_cc_after: [f64; 3],
    pub checkpoint_respawn_modeled: bool,
}

/// `evaluate_fall_clamp_state_replacement(...)` — apply the exact immediate
/// drop branch, including its terminal writes. Component-count validation in
/// Python is unreachable here (fixed-size inputs).
pub fn evaluate_fall_clamp_state_replacement(
    position_90_before: [f64; 3],
    matrix_9c_before: [f64; 9],
    velocity_c0_before: [f64; 3],
    angular_velocity_cc_before: [f64; 3],
    drop_threshold: f64,
    clamp_z: f64,
) -> FallClampStateReplacement {
    let position = position_90_before;
    let velocity = velocity_c0_before;
    let angular_velocity = angular_velocity_cc_before;
    let matrix = matrix_9c_before;

    let selected = position[2] < drop_threshold;
    let replacement_position =
        if selected { [position[0], position[1], clamp_z] } else { position };
    FallClampStateReplacement {
        selected,
        terminal_replacement: selected,
        predicate: PREDICATE,
        drop_threshold,
        position_90_before: position,
        position_90_after: replacement_position,
        previous_position_70_after: if selected { Some(replacement_position) } else { None },
        matrix_9c_before: matrix,
        matrix_9c_after: matrix,
        velocity_c0_before: velocity,
        velocity_c0_after: if selected { [0.0, 0.0, 0.0] } else { velocity },
        angular_velocity_cc_before: angular_velocity,
        angular_velocity_cc_after: if selected { [0.0, 0.0, 0.0] } else { angular_velocity },
        checkpoint_respawn_modeled: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_threshold_replaces_and_clears() {
        // Constants computed with the Python authority:
        //   evaluate_fall_clamp_state_replacement(
        //     position_90_before=[1,2,-6], matrix_9c_before=[0..8]=1..9,
        //     velocity_c0_before=[3,4,5], angular_velocity_cc_before=[6,7,8])
        let r = evaluate_fall_clamp_state_replacement(
            [1.0, 2.0, -6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            -5.0,
            0.0,
        );
        assert!(r.selected);
        assert_eq!(r.position_90_after, [1.0, 2.0, 0.0]);
        assert_eq!(r.previous_position_70_after, Some([1.0, 2.0, 0.0]));
        assert_eq!(r.velocity_c0_after, [0.0, 0.0, 0.0]);
        assert_eq!(r.angular_velocity_cc_after, [0.0, 0.0, 0.0]);
        assert_eq!(r.matrix_9c_after, r.matrix_9c_before);
    }

    #[test]
    fn at_threshold_not_selected() {
        // Python authority: `position.z < drop_threshold` is strict.
        let r = evaluate_fall_clamp_state_replacement(
            [1.0, 2.0, -5.0],
            [0.0; 9],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            -5.0,
            0.0,
        );
        assert!(!r.selected);
        assert_eq!(r.position_90_after, [1.0, 2.0, -5.0]);
        assert_eq!(r.previous_position_70_after, None);
        assert_eq!(r.velocity_c0_after, [3.0, 4.0, 5.0]);
    }
}
