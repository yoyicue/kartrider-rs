//! Port of `scripts/evaluate_position_integrator.py` — exact linear-position
//! edge of sub_B2FD80. Plain f64 math throughout; no f32 quantization points.
//!
//! The Python length guards are compile-time in Rust (arrays are `[f64; 3]`),
//! so only the branch-axis guard remains a runtime `Err`.

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionIntegratorResult {
    pub position_90_before: [f64; 3],
    pub velocity_c0_before_b2fd80: [f64; 3],
    pub pre_position_branch_selected: bool,
    pub pre_position_branch_delta: [f64; 3],
    pub velocity_c0_after_pre_position_branch: [f64; 3],
    pub latch_658_after: bool,
    pub pre_position_velocity_delta: [f64; 3],
    pub velocity_c0_for_position: [f64; 3],
    pub position_90_after: [f64; 3],
    pub dt: f64,
    pub scope: &'static str,
}

pub const SCOPE: &str =
    "exact position edge; quaternion/orientation update is composed by evaluate_orientation_integrator.py";

/// `evaluate_position_integrator(...)`.
pub fn evaluate_position_integrator(
    position_90_before: [f64; 3],
    velocity_c0_before: [f64; 3],
    dt: f64,
    pre_position_velocity_delta: Option<[f64; 3]>,
    state_1d0: i64,
    latch_658: bool,
    obf_b44: f64,
    axis_ab8: Option<[f64; 3]>,
    axis_a48: Option<[f64; 3]>,
) -> Result<PositionIntegratorResult, String> {
    let branch_selected = state_1d0 == 2 && latch_658 && obf_b44 > 0.0;
    let mut velocity_after_branch = velocity_c0_before;
    let mut branch_delta = [0.0, 0.0, 0.0];
    if branch_selected {
        let (Some(ab8), Some(a48)) = (axis_ab8, axis_a48) else {
            return Err(
                "selected pre-position branch requires three-component AB8/A48 axes".to_string(),
            );
        };
        let cross = [
            ab8[1] * a48[2] - ab8[2] * a48[1],
            ab8[2] * a48[0] - ab8[0] * a48[2],
            ab8[0] * a48[1] - ab8[1] * a48[0],
        ];
        // Python: math.sqrt(sum(value * value for value in velocity_c0_before)).
        // CPython 3.12+ sum() is Neumaier-compensated (see crate::num::sum_f64),
        // not a naive fold — this changes the last ulp of `speed`.
        let speed = crate::num::sum_f64(&[
            velocity_c0_before[0] * velocity_c0_before[0],
            velocity_c0_before[1] * velocity_c0_before[1],
            velocity_c0_before[2] * velocity_c0_before[2],
        ])
        .sqrt();
        for i in 0..3 {
            velocity_after_branch[i] = 0.5 * velocity_c0_before[i] + 0.5 * speed * cross[i];
            branch_delta[i] = velocity_after_branch[i] - velocity_c0_before[i];
        }
    }
    let delta = pre_position_velocity_delta.unwrap_or([0.0, 0.0, 0.0]);
    let mut velocity_for_position = [0.0, 0.0, 0.0];
    let mut position_after = [0.0, 0.0, 0.0];
    for i in 0..3 {
        velocity_for_position[i] = velocity_after_branch[i] + delta[i];
        position_after[i] = position_90_before[i] + velocity_for_position[i] * dt;
    }
    Ok(PositionIntegratorResult {
        position_90_before,
        velocity_c0_before_b2fd80: velocity_c0_before,
        pre_position_branch_selected: branch_selected,
        pre_position_branch_delta: branch_delta,
        velocity_c0_after_pre_position_branch: velocity_after_branch,
        latch_658_after: false,
        pre_position_velocity_delta: delta,
        velocity_c0_for_position: velocity_for_position,
        position_90_after: position_after,
        dt,
        scope: SCOPE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_off_plain_integration() {
        // Constants computed with the Python authority:
        //   evaluate_position_integrator(position_90_before=[1.0, -2.0, 3.5],
        //     velocity_c0_before=[10.0, 0.5, -4.25], dt=0.002)
        let r = evaluate_position_integrator(
            [1.0, -2.0, 3.5],
            [10.0, 0.5, -4.25],
            0.002,
            None,
            0,
            false,
            0.0,
            None,
            None,
        )
        .unwrap();
        assert!(!r.pre_position_branch_selected);
        assert_eq!(r.position_90_after, [1.02, -1.999, 3.4915]);
        assert_eq!(r.velocity_c0_for_position, [10.0, 0.5, -4.25]);
        assert!(!r.latch_658_after);
    }

    #[test]
    fn branch_on_cross_product() {
        // Constants computed with the Python authority:
        //   evaluate_position_integrator(position_90_before=[0.0, 0.0, 0.0],
        //     velocity_c0_before=[3.0, 4.0, 0.0], dt=0.001, state_1d0=2,
        //     latch_658=True, obf_b44=1.5, axis_ab8=[1.0, 0.0, 0.0],
        //     axis_a48=[0.0, 1.0, 0.0])
        let r = evaluate_position_integrator(
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            0.001,
            None,
            2,
            true,
            1.5,
            Some([1.0, 0.0, 0.0]),
            Some([0.0, 1.0, 0.0]),
        )
        .unwrap();
        assert!(r.pre_position_branch_selected);
        // cross = [0, 0, 1]; speed = 5.0
        assert_eq!(
            r.velocity_c0_after_pre_position_branch,
            [1.5, 2.0, 2.5]
        );
        assert_eq!(r.pre_position_branch_delta, [-1.5, -2.0, 2.5]);
        assert_eq!(r.position_90_after, [0.0015, 0.002, 0.0025]);
    }

    #[test]
    fn branch_requires_axes() {
        let err = evaluate_position_integrator(
            [0.0; 3],
            [1.0, 0.0, 0.0],
            0.001,
            None,
            2,
            true,
            1.0,
            None,
            Some([0.0, 1.0, 0.0]),
        )
        .unwrap_err();
        assert_eq!(
            err,
            "selected pre-position branch requires three-component AB8/A48 axes"
        );
    }

    #[test]
    fn obf_b44_zero_disables_branch() {
        let r = evaluate_position_integrator(
            [0.0; 3],
            [1.0, 2.0, 3.0],
            0.001,
            None,
            2,
            true,
            0.0,
            None,
            None,
        )
        .unwrap();
        assert!(!r.pre_position_branch_selected);
    }
}
