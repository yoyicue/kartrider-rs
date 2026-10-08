//! Port of `scripts/evaluate_network_correction_topology.py` — authoritative
//! rigid-body state reconciliation for network correction.
//!
//! Pure f64 (the Python leaf never quantizes through `_f32`). The cosine and
//! magnitude reductions are CPython `sum()` sites, so they route through
//! [`sum_f64`] (Neumaier), per the dot3 audit in `crate::num`; `min`/`max`
//! clamps use the CPython two-operand helpers. `math.sqrt`/`math.acos`/
//! `math.sin` map to the native f64 methods (same libm, matching the
//! established pattern at engine.rs:2849).

use crate::num::{max2_f64, min2_f64, sum_f64};
use crate::state::{JsonValue, NetworkCorrectionState, RigidStateSnapshot};

/// Python `RIGID_FIELDS` (evaluate_network_correction_topology.py:9).
pub const RIGID_FIELDS: [&str; 4] = ["position", "velocity", "angular_velocity", "matrix"];
/// Python `SEQUENCE_BITS` (evaluate_network_correction_topology.py:10).
pub const SEQUENCE_BITS: u32 = 32;

/// `evaluate_network_sequence(...)` result dict
/// (evaluate_network_correction_topology.py:13-77). Integer fields are `i64`
/// (Python `int`; the `bits` parameter allows up to 63, so values reach
/// 2^63-1).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkSequenceDecision {
    pub accepted: bool,
    pub reason: &'static str,
    pub candidate: Option<i64>,
    pub last_accepted_before: Option<i64>,
    pub last_accepted_after: Option<i64>,
    pub forward_distance: Option<i64>,
    pub estimated_packets_lost: i64,
    pub ordered_authority: bool,
}

/// Interpret a Python `int | None` argument carried as JSON: `null` -> None,
/// an integral number -> Some(value), a bool or non-integral number -> the
/// Python `isinstance(candidate, int)` failure (`error_message`). Values for
/// `bits` above 52 are only exact up to 2^53 through JSON; the parity harness
/// keeps generated in-range sequences below that bound (out-of-range values
/// only need the range error, which the caller computes in f64).
fn sequence_arg(value: &JsonValue, error_message: &str) -> Result<Option<f64>, String> {
    match value {
        JsonValue::Null => Ok(None),
        JsonValue::Number(number) if number.is_finite() && number.fract() == 0.0 => {
            Ok(Some(*number))
        }
        _ => Err(error_message.to_string()),
    }
}

/// Python `0 <= value <= maximum` range check in f64. `maximum as f64 + 1.0`
/// is the exact acceptance ceiling: for `bits` < 63 the maximum is exactly
/// representable; for `bits` == 63 it rounds to 2^63, which is also the
/// first f64 above the maximum (the largest f64 below 2^63 is 2^63 - 512,
/// in range), so the boundary lands exactly right in both cases.
fn sequence_in_range(value: f64, maximum: u64) -> bool {
    value >= 0.0 && value < maximum as f64 + 1.0
}

/// `evaluate_network_sequence(candidate, last_accepted, bits)` — apply modulo
/// serial-number ordering to one authoritative snapshot. `ValueError` maps to
/// `Err(String)` with the Python message.
pub fn evaluate_network_sequence(
    candidate: &JsonValue,
    last_accepted: &JsonValue,
    bits: u32,
) -> Result<NetworkSequenceDecision, String> {
    if !(2..=63).contains(&bits) {
        return Err("sequence bits must be in [2, 63]".to_string());
    }
    let modulus: u64 = 1u64 << bits;
    let maximum: u64 = modulus - 1;
    let candidate = sequence_arg(candidate, "network sequence must be an integer or None")?;
    let Some(candidate_f64) = candidate else {
        return Ok(NetworkSequenceDecision {
            accepted: true,
            reason: "unsequenced_compatibility_packet",
            candidate: None,
            last_accepted_before: sequence_arg(last_accepted, "unused")
                .unwrap_or(None)
                .map(|value| value as i64),
            last_accepted_after: sequence_arg(last_accepted, "unused")
                .unwrap_or(None)
                .map(|value| value as i64),
            forward_distance: None,
            estimated_packets_lost: 0,
            ordered_authority: false,
        });
    };
    if !sequence_in_range(candidate_f64, maximum) {
        return Err(format!("network sequence must be in [0, {maximum}]"));
    }
    // In-range integral f64 values are below 2^63, so this is exact.
    let candidate = candidate_f64 as i64;
    let last_accepted_value = match last_accepted {
        JsonValue::Null => None,
        other => sequence_arg(other, "last accepted network sequence must be an integer")?,
    };
    let Some(last_f64) = last_accepted_value else {
        return Ok(NetworkSequenceDecision {
            accepted: true,
            reason: "first_ordered_packet",
            candidate: Some(candidate),
            last_accepted_before: None,
            last_accepted_after: Some(candidate),
            forward_distance: None,
            estimated_packets_lost: 0,
            ordered_authority: true,
        });
    };
    if !sequence_in_range(last_f64, maximum) {
        return Err(format!("last accepted sequence must be in [0, {maximum}]"));
    }
    let last = last_f64 as i64;
    // Python `(candidate - last_accepted) & maximum`: the mask keeps the low
    // `bits` bits of the two's-complement difference.
    let distance = (candidate.wrapping_sub(last) as u64) & maximum;
    let (accepted, reason) = if distance == 0 {
        (false, "duplicate_sequence")
    } else if distance < modulus / 2 {
        (true, "newer_sequence")
    } else {
        (false, "stale_or_reordered_sequence")
    };
    Ok(NetworkSequenceDecision {
        accepted,
        reason,
        candidate: Some(candidate),
        last_accepted_before: Some(last),
        last_accepted_after: if accepted { Some(candidate) } else { Some(last) },
        forward_distance: Some(distance as i64),
        estimated_packets_lost: if accepted {
            (distance as i64 - 1).max(0)
        } else {
            0
        },
        ordered_authority: true,
    })
}

/// `_columns` (evaluate_network_correction_topology.py:91-96): row-major
/// storage; columns are the vehicle axes. Same layout as
/// `crate::num::matrix_columns`.
fn columns(matrix: [f64; 9]) -> [[f64; 3]; 3] {
    [
        [matrix[0], matrix[3], matrix[6]],
        [matrix[1], matrix[4], matrix[7]],
        [matrix[2], matrix[5], matrix[8]],
    ]
}

/// `_axes_to_quaternion` (evaluate_network_correction_topology.py:99-138).
/// Returns `(w, x, y, z)` normalized by `math.sqrt(sum(v*v))` (Neumaier).
fn axes_to_quaternion(matrix: [f64; 9]) -> [f64; 4] {
    let axes = columns(matrix);
    let [m00, m10, m20] = axes[0];
    let [m01, m11, m21] = axes[1];
    let [m02, m12, m22] = axes[2];
    let trace = m00 + m11 + m22;
    let result = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        [
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        ]
    } else if m00 > m11 && m00 > m22 {
        let scale = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        ]
    } else if m11 > m22 {
        let scale = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        ]
    } else {
        let scale = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        ]
    };
    let magnitude = sum_f64(&[
        result[0] * result[0],
        result[1] * result[1],
        result[2] * result[2],
        result[3] * result[3],
    ])
    .sqrt();
    [
        result[0] / magnitude,
        result[1] / magnitude,
        result[2] / magnitude,
        result[3] / magnitude,
    ]
}

/// `_quaternion_to_matrix` (evaluate_network_correction_topology.py:141-152):
/// `(w, x, y, z)` -> row-major flattened matrix (columns are the axes).
fn quaternion_to_matrix(quaternion: [f64; 4]) -> [f64; 9] {
    let [w, x, y, z] = quaternion;
    let columns = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut result = [0.0; 9];
    for row in 0..3 {
        for (column_index, column) in columns.iter().enumerate() {
            result[row * 3 + column_index] = column[row];
        }
    }
    result
}

/// `_slerp` (evaluate_network_correction_topology.py:155-177). The cosine
/// and the near-parallel magnitude are CPython `sum()` reductions
/// ([`sum_f64`], NOT a naive fold).
fn slerp(left: [f64; 4], right: [f64; 4], alpha: f64) -> [f64; 4] {
    let mut cosine = sum_f64(&[
        left[0] * right[0],
        left[1] * right[1],
        left[2] * right[2],
        left[3] * right[3],
    ]);
    let mut right = right;
    if cosine < 0.0 {
        right = [-right[0], -right[1], -right[2], -right[3]];
        cosine = -cosine;
    }
    cosine = max2_f64(-1.0, min2_f64(1.0, cosine));
    if cosine > 0.9995 {
        let mut result = [0.0; 4];
        for index in 0..4 {
            result[index] = left[index] + alpha * (right[index] - left[index]);
        }
        let magnitude = sum_f64(&[
            result[0] * result[0],
            result[1] * result[1],
            result[2] * result[2],
            result[3] * result[3],
        ])
        .sqrt();
        return [
            result[0] / magnitude,
            result[1] / magnitude,
            result[2] / magnitude,
            result[3] / magnitude,
        ];
    }
    let angle = cosine.acos();
    let denominator = angle.sin();
    let left_scale = ((1.0 - alpha) * angle).sin();
    let right_scale = (alpha * angle).sin();
    let mut result = [0.0; 4];
    for index in 0..4 {
        result[index] =
            (left_scale * left[index] + right_scale * right[index]) / denominator;
    }
    result
}

/// `begin_network_correction(...)` (evaluate_network_correction_topology.py
/// :180-210): create one immutable reconciliation transaction. The Python
/// `_validate_state` presence/length checks are type-enforced by
/// [`RigidStateSnapshot`]; `ValueError` maps to `Err(String)`.
pub fn begin_network_correction(
    predicted_state: &RigidStateSnapshot,
    authoritative_state: &RigidStateSnapshot,
    mode: &str,
    duration_ms: f64,
    sequence: Option<u32>,
) -> Result<NetworkCorrectionState, String> {
    if mode != "snap" && mode != "blend" {
        return Err("network correction mode must be snap or blend".to_string());
    }
    if mode == "blend" && duration_ms <= 0.0 {
        return Err("blend correction requires positive duration_ms".to_string());
    }
    if mode == "snap" && duration_ms != 0.0 {
        return Err("snap correction duration_ms must be zero".to_string());
    }
    Ok(NetworkCorrectionState {
        mode: mode.to_string(),
        duration_ms,
        elapsed_ms: 0.0,
        sequence,
        initial_state: predicted_state.clone(),
        target_state: authoritative_state.clone(),
        complete: false,
        policy_authority: Some("explicit_network_reconciliation_policy".to_string()),
        resume_topology: None,
    })
}

/// `evaluate_network_correction_step(...)` result dict
/// (evaluate_network_correction_topology.py:246-253).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionStepResult {
    pub state_after: RigidStateSnapshot,
    pub correction_after: NetworkCorrectionState,
    pub alpha: f64,
    pub complete: bool,
    pub mode: String,
    pub sequence: Option<u32>,
    /// Python constant `generic_authoritative_reconciliation_executable`.
    pub closure_status: &'static str,
}

/// `evaluate_network_correction_step(correction, elapsed_ms)` — advance one
/// snap/blend transaction toward its authoritative target.
pub fn evaluate_network_correction_step(
    correction: &NetworkCorrectionState,
    elapsed_ms: f64,
) -> Result<NetworkCorrectionStepResult, String> {
    if elapsed_ms <= 0.0 {
        return Err("elapsed_ms must be positive".to_string());
    }
    let initial = &correction.initial_state;
    let target = &correction.target_state;
    let next_elapsed = correction.elapsed_ms + elapsed_ms;
    let alpha = if correction.mode == "snap" {
        1.0
    } else {
        min2_f64(1.0, next_elapsed / correction.duration_ms)
    };
    let mut state = RigidStateSnapshot::default();
    for index in 0..3 {
        state.position[index] =
            initial.position[index] + alpha * (target.position[index] - initial.position[index]);
        state.velocity[index] =
            initial.velocity[index] + alpha * (target.velocity[index] - initial.velocity[index]);
        state.angular_velocity[index] = initial.angular_velocity[index]
            + alpha * (target.angular_velocity[index] - initial.angular_velocity[index]);
    }
    state.matrix = quaternion_to_matrix(slerp(
        axes_to_quaternion(initial.matrix),
        axes_to_quaternion(target.matrix),
        alpha,
    ));
    let complete = alpha >= 1.0;
    let mut correction_after = correction.clone();
    correction_after.elapsed_ms = next_elapsed;
    correction_after.complete = complete;
    Ok(NetworkCorrectionStepResult {
        state_after: state,
        correction_after,
        alpha,
        complete,
        mode: correction.mode.clone(),
        sequence: correction.sequence,
        closure_status: "generic_authoritative_reconciliation_executable",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_int(value: i64) -> JsonValue {
        JsonValue::Number(value as f64)
    }

    #[test]
    fn sequence_wrap_around_half_range() {
        // Python authority: candidate 2 after last 2^32-2 is newer
        // (forward distance 4, 3 estimated lost).
        let decision = evaluate_network_sequence(
            &json_int(2),
            &json_int((1i64 << 32) - 2),
            SEQUENCE_BITS,
        )
        .unwrap();
        assert!(decision.accepted);
        assert_eq!(decision.reason, "newer_sequence");
        assert_eq!(decision.forward_distance, Some(4));
        assert_eq!(decision.estimated_packets_lost, 3);
        assert_eq!(decision.last_accepted_after, Some(2));
        // The mirror direction is stale.
        let decision = evaluate_network_sequence(
            &json_int((1i64 << 32) - 2),
            &json_int(2),
            SEQUENCE_BITS,
        )
        .unwrap();
        assert!(!decision.accepted);
        assert_eq!(decision.reason, "stale_or_reordered_sequence");
        assert_eq!(decision.last_accepted_after, Some(2));
    }

    #[test]
    fn sequence_duplicate_and_unsequenced() {
        let decision =
            evaluate_network_sequence(&json_int(7), &json_int(7), SEQUENCE_BITS).unwrap();
        assert!(!decision.accepted);
        assert_eq!(decision.reason, "duplicate_sequence");
        assert_eq!(decision.forward_distance, Some(0));
        let decision =
            evaluate_network_sequence(&JsonValue::Null, &json_int(7), SEQUENCE_BITS).unwrap();
        assert!(decision.accepted);
        assert_eq!(decision.reason, "unsequenced_compatibility_packet");
        assert!(!decision.ordered_authority);
        assert_eq!(decision.last_accepted_after, Some(7));
        let decision =
            evaluate_network_sequence(&json_int(7), &JsonValue::Null, SEQUENCE_BITS).unwrap();
        assert_eq!(decision.reason, "first_ordered_packet");
    }

    #[test]
    fn sequence_validation_messages() {
        assert_eq!(
            evaluate_network_sequence(&json_int(1), &json_int(1), 1),
            Err("sequence bits must be in [2, 63]".to_string())
        );
        assert_eq!(
            evaluate_network_sequence(&JsonValue::Bool(true), &JsonValue::Null, SEQUENCE_BITS),
            Err("network sequence must be an integer or None".to_string())
        );
        assert_eq!(
            evaluate_network_sequence(
                &JsonValue::Number(1.5),
                &JsonValue::Null,
                SEQUENCE_BITS
            ),
            Err("network sequence must be an integer or None".to_string())
        );
        assert_eq!(
            evaluate_network_sequence(&json_int(1 << 32), &JsonValue::Null, SEQUENCE_BITS),
            Err("network sequence must be in [0, 4294967295]".to_string())
        );
        assert_eq!(
            evaluate_network_sequence(&json_int(1), &JsonValue::Bool(false), SEQUENCE_BITS),
            Err("last accepted network sequence must be an integer".to_string())
        );
        assert_eq!(
            evaluate_network_sequence(&json_int(1), &json_int(-1), SEQUENCE_BITS),
            Err("last accepted sequence must be in [0, 4294967295]".to_string())
        );
    }

    #[test]
    fn begin_validates_mode_and_duration() {
        let state = RigidStateSnapshot::default();
        assert_eq!(
            begin_network_correction(&state, &state, "lerp", 0.0, None),
            Err("network correction mode must be snap or blend".to_string())
        );
        assert_eq!(
            begin_network_correction(&state, &state, "blend", 0.0, None),
            Err("blend correction requires positive duration_ms".to_string())
        );
        assert_eq!(
            begin_network_correction(&state, &state, "snap", 2.0, None),
            Err("snap correction duration_ms must be zero".to_string())
        );
        let correction =
            begin_network_correction(&state, &state, "snap", 0.0, Some(9)).unwrap();
        assert_eq!(correction.mode, "snap");
        assert_eq!(correction.sequence, Some(9));
        assert!(!correction.complete);
        assert_eq!(
            correction.policy_authority.as_deref(),
            Some("explicit_network_reconciliation_policy")
        );
    }

    #[test]
    fn quaternion_round_trip_identity() {
        // Constants computed with the Python authority: identity matrix ->
        // quaternion (1, 0, 0, 0) -> identity matrix (bit-exact).
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let quaternion = axes_to_quaternion(identity);
        assert_eq!(quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(quaternion_to_matrix(quaternion), identity);
    }

    #[test]
    fn snap_step_completes_in_one_substep() {
        let predicted = RigidStateSnapshot {
            position: [1.0, 2.0, 3.0],
            velocity: [0.5, 0.0, -0.25],
            ..RigidStateSnapshot::default()
        };
        let target = RigidStateSnapshot {
            position: [4.0, 5.0, 6.0],
            velocity: [1.5, 2.0, 0.75],
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..RigidStateSnapshot::default()
        };
        let correction =
            begin_network_correction(&predicted, &target, "snap", 0.0, None).unwrap();
        let result = evaluate_network_correction_step(&correction, 2.0).unwrap();
        assert!(result.complete);
        assert_eq!(result.alpha, 1.0);
        assert_eq!(result.state_after.position, target.position);
        assert_eq!(result.state_after.velocity, target.velocity);
        assert_eq!(result.correction_after.elapsed_ms, 2.0);
        assert_eq!(
            result.closure_status,
            "generic_authoritative_reconciliation_executable"
        );
        assert_eq!(
            evaluate_network_correction_step(&correction, 0.0),
            Err("elapsed_ms must be positive".to_string())
        );
    }

    #[test]
    fn blend_step_alpha_is_elapsed_ratio() {
        let predicted = RigidStateSnapshot {
            position: [0.0, 0.0, 0.0],
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..RigidStateSnapshot::default()
        };
        let target = RigidStateSnapshot {
            position: [10.0, 0.0, 0.0],
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            ..RigidStateSnapshot::default()
        };
        let correction =
            begin_network_correction(&predicted, &target, "blend", 6.0, None).unwrap();
        let result = evaluate_network_correction_step(&correction, 2.0).unwrap();
        assert!(!result.complete);
        // Python: alpha = min(1.0, 2.0/6.0); x = 0 + alpha * 10.
        assert_eq!(result.alpha, 2.0 / 6.0);
        assert_eq!(result.state_after.position[0], (2.0 / 6.0) * 10.0);
        let second = evaluate_network_correction_step(&result.correction_after, 2.0).unwrap();
        assert!(!second.complete);
        assert_eq!(second.correction_after.elapsed_ms, 4.0);
        let third = evaluate_network_correction_step(&second.correction_after, 2.0).unwrap();
        assert!(third.complete);
        assert_eq!(third.state_after.position[0], 10.0);
    }
}
