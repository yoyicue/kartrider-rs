//! Port of `scripts/evaluate_checkpoint_respawn_state_replacement.py` —
//! mode-specific checkpoint restore reached from BD7840 vtable slot +0x164.

use crate::num::sum_f64;

// stage_offsets provenance (Python result key `stage_offsets`).
pub const OFFSETS_C016_FUNCTION_VA: &str = "0xC01690";
pub const OFFSETS_C016_GATE: &str = "0xAE4";
pub const OFFSETS_C016_TARGET: &str = "0xAE8";
pub const OFFSETS_C72C_FUNCTION_VA: &str = "0xC72C80";
pub const OFFSETS_C72C_GATE: &str = "0xAA8";
pub const OFFSETS_C72C_TARGET: &str = "0xAAC";

// source provenance (Python result key `source`).
pub const SOURCE_OUTER_SELECTOR_VA: &str = "0xBD7840";
pub const SOURCE_VIRTUAL_SLOT: &str = "0x164";
pub const SOURCE_WRITER_VA: &str = "0xB16290";
pub const SOURCE_VECTOR_SUBTRACT_VA: &str = "0x6F34E0";
pub const SOURCE_CROSS_VA: &str = "0x6F3680";
pub const SOURCE_NORMALIZE_VA: &str = "0x6F42E0";
pub const SOURCE_MATRIX_COLUMN_SETTER_VA: &str = "0x6F7D60";

/// Python `stage_offsets` sub-dict for one mode family.
#[derive(Debug, Clone, PartialEq)]
pub struct StageOffsets {
    pub function_va: &'static str,
    pub gate: &'static str,
    pub target: &'static str,
}

/// Python `evaluate_checkpoint_respawn_state_replacement(...)` result dict
/// minus `source`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointRespawnStateReplacement {
    pub selected: bool,
    pub terminal_replacement: bool,
    pub mode_family: String,
    pub checkpoint_gate: bool,
    pub sample_result_zero: bool,
    pub target_position: [f64; 3],
    pub course_reference_position: [f64; 3],
    pub forward_axis1: [f64; 3],
    pub right_axis0: [f64; 3],
    pub up_axis2: [f64; 3],
    /// `target` when selected, else Python `None`.
    pub position_90_after: Option<[f64; 3]>,
    /// `target` when selected, else Python `None`.
    pub previous_position_70_after: Option<[f64; 3]>,
    /// Row-major matrix when selected, else Python `None`.
    pub matrix_9c_after: Option<[f64; 9]>,
    pub velocity_c0_before: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub angular_velocity_cc_before: [f64; 3],
    pub angular_velocity_cc_after: [f64; 3],
    pub stage_offsets: StageOffsets,
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

/// `_normalize`: length is `math.sqrt(sum(c*c))` — CPython 3.12+ `sum()`
/// (Neumaier, crate::num::sum_f64); exactly-zero length falls back.
fn normalize(value: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    let length =
        sum_f64(&[value[0] * value[0], value[1] * value[1], value[2] * value[2]]).sqrt();
    if length == 0.0 {
        return fallback;
    }
    [value[0] / length, value[1] / length, value[2] / length]
}

/// `evaluate_checkpoint_respawn_state_replacement(...)` — both mode
/// overrides use the same math; they differ only in the stage offsets
/// holding the gate and target position. Component-count validation in
/// Python is unreachable here (fixed-size inputs).
pub fn evaluate_checkpoint_respawn_state_replacement(
    mode_family: &str,
    checkpoint_gate: bool,
    sample_result_zero: bool,
    target_position: [f64; 3],
    course_reference_position: [f64; 3],
    velocity_c0_before: [f64; 3],
    angular_velocity_cc_before: [f64; 3],
) -> Result<CheckpointRespawnStateReplacement, String> {
    let stage_offsets = match mode_family {
        "c016" => StageOffsets {
            function_va: OFFSETS_C016_FUNCTION_VA,
            gate: OFFSETS_C016_GATE,
            target: OFFSETS_C016_TARGET,
        },
        "c72c" => StageOffsets {
            function_va: OFFSETS_C72C_FUNCTION_VA,
            gate: OFFSETS_C72C_GATE,
            target: OFFSETS_C72C_TARGET,
        },
        _ => return Err("mode_family must be 'c016' or 'c72c'".to_string()),
    };
    let target = target_position;
    let reference = course_reference_position;
    let velocity = velocity_c0_before;
    let angular_velocity = angular_velocity_cc_before;
    let selected = checkpoint_gate && sample_result_zero;

    // sub_6F34E0(target, reference), normalize, then construct columns:
    // axis1=forward, axis0=normalize(forward x world_z), axis2=axis0 x axis1.
    let forward = normalize(
        [target[0] - reference[0], target[1] - reference[1], target[2] - reference[2]],
        [0.0, 1.0, 0.0],
    );
    let axis0 = normalize(cross(forward, [0.0, 0.0, 1.0]), [1.0, 0.0, 0.0]);
    let axis2 = cross(axis0, forward);
    let matrix = [
        axis0[0], forward[0], axis2[0], axis0[1], forward[1], axis2[1], axis0[2], forward[2],
        axis2[2],
    ];
    Ok(CheckpointRespawnStateReplacement {
        selected,
        terminal_replacement: selected,
        mode_family: mode_family.to_string(),
        checkpoint_gate,
        sample_result_zero,
        target_position: target,
        course_reference_position: reference,
        forward_axis1: forward,
        right_axis0: axis0,
        up_axis2: axis2,
        position_90_after: if selected { Some(target) } else { None },
        previous_position_70_after: if selected { Some(target) } else { None },
        matrix_9c_after: if selected { Some(matrix) } else { None },
        velocity_c0_before: velocity,
        velocity_c0_after: if selected { [0.0, 0.0, 0.0] } else { velocity },
        angular_velocity_cc_before: angular_velocity,
        angular_velocity_cc_after: if selected { [0.0, 0.0, 0.0] } else { angular_velocity },
        stage_offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_family_is_validated() {
        assert_eq!(
            evaluate_checkpoint_respawn_state_replacement(
                "c999", true, true, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3],
            )
            .unwrap_err(),
            "mode_family must be 'c016' or 'c72c'"
        );
    }

    #[test]
    fn gated_respawn_builds_orthonormal_columns() {
        // Constants computed with the Python authority:
        //   evaluate_checkpoint_respawn_state_replacement(
        //     mode_family="c016", checkpoint_gate=True, sample_result_zero=True,
        //     target_position=[10,0,0], course_reference_position=[0,0,0],
        //     velocity_c0_before=[1,2,3], angular_velocity_cc_before=[4,5,6])
        //   forward=[1,0,0], axis0=[0,-1,0], axis2=[-0.0,0,1].
        let r = evaluate_checkpoint_respawn_state_replacement(
            "c016",
            true,
            true,
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        )
        .unwrap();
        assert!(r.selected);
        assert_eq!(r.forward_axis1, [1.0, 0.0, 0.0]);
        assert_eq!(r.right_axis0, [0.0, -1.0, 0.0]);
        assert_eq!(r.up_axis2, [-0.0, 0.0, 1.0]);
        assert!(r.up_axis2[0].is_sign_negative());
        assert_eq!(
            r.matrix_9c_after,
            Some([0.0, 1.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        );
        assert_eq!(r.position_90_after, Some([10.0, 0.0, 0.0]));
        assert_eq!(r.velocity_c0_after, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn degenerate_target_uses_fallbacks() {
        // Python authority: target == reference -> forward fallback [0,1,0],
        // axis0 fallback [1,0,0].
        let r = evaluate_checkpoint_respawn_state_replacement(
            "c72c", false, false, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0; 3], [0.0; 3],
        )
        .unwrap();
        assert!(!r.selected);
        assert_eq!(r.forward_axis1, [0.0, 1.0, 0.0]);
        assert_eq!(r.right_axis0, [1.0, 0.0, 0.0]);
        assert_eq!(r.up_axis2, [0.0, 0.0, 1.0]);
        assert_eq!(r.matrix_9c_after, None);
        assert_eq!(r.stage_offsets.function_va, "0xC72C80");
    }
}
