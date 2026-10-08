//! Port of `scripts/evaluate_linear_velocity_integrator.py` — exact linear
//! half of sub_B2D940 accumulator integration. Both `numeric_mode` paths.

use crate::num::{fadd, fdiv, fmul};

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearVelocityIntegratorResult {
    pub velocity_c0_before: [f64; 3],
    pub post_bs_later_delta: [f64; 3],
    pub post_bs_later_delta_policy: &'static str,
    pub accumulator_614_before_bac: [f64; 3],
    pub bac_force: [f64; 3],
    pub accumulator_614_after_bac: [f64; 3],
    pub accumulator_zero_cleared_velocity: bool,
    pub delta_velocity: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub numeric_mode: String,
    pub ordering: &'static str,
}

pub const POST_BS_LATER_DELTA_POLICY: &str = "compatibility field fixed at zero; measured residuals are diagnostics and never mutate predicted state";
pub const ORDERING: &str =
    "+0x614 += +0xBAC -> if total zero set +0xC0=0 -> +0xC0 += (+0x614/ObfFloat(+0x660))*dt";

/// Python `_add(a, b)` uses `sum(...)` per component, which starts at int 0:
/// `0 + a[i] + b[i]`. CPython 3.12+ sum() is Neumaier-compensated
/// (crate::num::sum_f64); for two elements it coincides with the plain sum
/// but keeps the load-bearing `-0.0` -> `+0.0` start semantics.
#[inline]
fn sum2_f64(a: f64, b: f64) -> f64 {
    crate::num::sum_f64(&[a, b])
}

fn add(a: [f64; 3], b: [f64; 3], numeric_mode: &str) -> [f64; 3] {
    if numeric_mode == "float32" {
        return [
            fadd(a[0], b[0]),
            fadd(a[1], b[1]),
            fadd(a[2], b[2]),
        ];
    }
    [
        sum2_f64(a[0], b[0]),
        sum2_f64(a[1], b[1]),
        sum2_f64(a[2], b[2]),
    ]
}

/// `evaluate_linear_velocity_integrator(...)`.
pub fn evaluate_linear_velocity_integrator(
    velocity_c0_before: [f64; 3],
    accumulator_614_before: [f64; 3],
    bac_force: [f64; 3],
    sub_660: f64,
    dt: f64,
    post_bs_later_delta: Option<[f64; 3]>,
    numeric_mode: &str,
) -> Result<LinearVelocityIntegratorResult, String> {
    if sub_660 == 0.0 {
        return Err("decoded ObfFloat(+0x660) must be nonzero".to_string());
    }
    let post_bs = post_bs_later_delta.unwrap_or([0.0, 0.0, 0.0]);
    if post_bs.iter().any(|&value| value != 0.0) {
        return Err("post_bs_later_delta has no enumerated static producer and must remain zero; keep measured residuals outside state prediction".to_string());
    }
    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    let accumulator_before_bac = add(accumulator_614_before, post_bs, numeric_mode);
    let accumulator_after_bac = add(accumulator_before_bac, bac_force, numeric_mode);
    let accumulator_is_zero = accumulator_after_bac.iter().all(|&value| value == 0.0);
    let velocity_base = if accumulator_is_zero {
        [0.0, 0.0, 0.0]
    } else {
        velocity_c0_before
    };
    let mut delta_velocity = [0.0, 0.0, 0.0];
    if numeric_mode == "float32" {
        for i in 0..3 {
            delta_velocity[i] = fmul(fdiv(accumulator_after_bac[i], sub_660), dt);
        }
    } else {
        for i in 0..3 {
            delta_velocity[i] = accumulator_after_bac[i] / sub_660 * dt;
        }
    }
    let velocity_after = add(velocity_base, delta_velocity, numeric_mode);
    Ok(LinearVelocityIntegratorResult {
        velocity_c0_before,
        post_bs_later_delta: post_bs,
        post_bs_later_delta_policy: POST_BS_LATER_DELTA_POLICY,
        accumulator_614_before_bac: accumulator_before_bac,
        bac_force,
        accumulator_614_after_bac: accumulator_after_bac,
        accumulator_zero_cleared_velocity: accumulator_is_zero,
        delta_velocity,
        velocity_c0_after: velocity_after,
        numeric_mode: numeric_mode.to_string(),
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float64_nonzero_accumulator() {
        // Constants computed with the Python authority:
        //   evaluate_linear_velocity_integrator(velocity_c0_before=[10.0, -2.0, 0.5],
        //     accumulator_614_before=[3.0, 0.0, -1.5], bac_force=[0.5, -0.25, 2.0],
        //     sub_660=80.0, dt=0.002, numeric_mode="float64")
        let r = evaluate_linear_velocity_integrator(
            [10.0, -2.0, 0.5],
            [3.0, 0.0, -1.5],
            [0.5, -0.25, 2.0],
            80.0,
            0.002,
            None,
            "float64",
        )
        .unwrap();
        assert!(!r.accumulator_zero_cleared_velocity);
        assert_eq!(r.accumulator_614_after_bac, [3.5, -0.25, 0.5]);
        assert_eq!(r.delta_velocity, [8.75e-5, -6.25e-6, 1.25e-5]);
        assert_eq!(r.velocity_c0_after, [10.0000875, -2.00000625, 0.5000125]);
    }

    #[test]
    fn float32_nonzero_accumulator() {
        // Constants computed with the Python authority (same inputs, float32).
        let r = evaluate_linear_velocity_integrator(
            [10.0, -2.0, 0.5],
            [3.0, 0.0, -1.5],
            [0.5, -0.25, 2.0],
            80.0,
            0.002,
            None,
            "float32",
        )
        .unwrap();
        assert_eq!(
            r.velocity_c0_after,
            [10.00008773803711, -2.0000061988830566, 0.5000125169754028]
        );
    }

    #[test]
    fn zero_accumulator_clears_velocity() {
        // Integer-valued forces cancel exactly in both numeric modes.
        let r = evaluate_linear_velocity_integrator(
            [10.0, -2.0, 0.5],
            [1.0, -2.0, 3.0],
            [-1.0, 2.0, -3.0],
            80.0,
            0.002,
            Some([0.0, 0.0, 0.0]),
            "float32",
        )
        .unwrap();
        assert!(r.accumulator_zero_cleared_velocity);
        assert_eq!(r.velocity_c0_after, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            evaluate_linear_velocity_integrator(
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                0.0,
                0.002,
                None,
                "float64",
            )
            .unwrap_err(),
            "decoded ObfFloat(+0x660) must be nonzero"
        );
        assert_eq!(
            evaluate_linear_velocity_integrator(
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                80.0,
                0.002,
                Some([0.0, 1.0, 0.0]),
                "float64",
            )
            .unwrap_err(),
            "post_bs_later_delta has no enumerated static producer and must remain zero; keep measured residuals outside state prediction"
        );
        assert_eq!(
            evaluate_linear_velocity_integrator(
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                80.0,
                0.002,
                None,
                "float16",
            )
            .unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }

    #[test]
    fn negative_zero_accumulator_sum_stays_positive_zero() {
        // Python _add starts sum() at int 0, so 0 + (-0.0) + (-0.0) == +0.0;
        // a Rust a+b would keep -0.0. The explicit start is load-bearing.
        let r = evaluate_linear_velocity_integrator(
            [1.0, 2.0, 3.0],
            [-0.0, 0.0, 0.0],
            [-0.0, 0.0, 0.0],
            80.0,
            0.002,
            None,
            "float64",
        )
        .unwrap();
        assert!(r.accumulator_zero_cleared_velocity);
        assert!(r.accumulator_614_after_bac[0].is_sign_positive());
        assert_eq!(r.velocity_c0_after, [0.0, 0.0, 0.0]);
    }
}
