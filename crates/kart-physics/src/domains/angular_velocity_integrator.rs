//! Port of `scripts/evaluate_angular_velocity_integrator.py` — exact angular
//! half of sub_B2D940. Both `numeric_mode` paths.

use super::ground_accumulator_sink::transform_row_major;
use crate::num::{fadd, fmul, fsub};

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct AngularVelocityIntegratorResult {
    pub angular_velocity_cc_before: [f64; 3],
    pub mapped_omega: [f64; 3],
    pub gyroscopic_term: [f64; 3],
    pub accumulator_620: [f64; 3],
    pub torque_minus_gyro: [f64; 3],
    pub angular_acceleration: [f64; 3],
    pub delta_angular_velocity: [f64; 3],
    pub angular_velocity_cc_after: [f64; 3],
    pub numeric_mode: String,
    pub ordering: &'static str,
}

pub const ORDERING: &str =
    "M9C4*CC -> CC cross mapped_CC -> M9C4*(+0x620-gyro) -> multiply dt -> add to +0xCC";

fn cross(a: [f64; 3], b: [f64; 3], numeric_mode: &str) -> [f64; 3] {
    if numeric_mode == "float32" {
        return [
            fsub(fmul(a[1], b[2]), fmul(a[2], b[1])),
            fsub(fmul(a[2], b[0]), fmul(a[0], b[2])),
            fsub(fmul(a[0], b[1]), fmul(a[1], b[0])),
        ];
    }
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3], numeric_mode: &str) -> [f64; 3] {
    if numeric_mode == "float32" {
        return [
            fsub(a[0], b[0]),
            fsub(a[1], b[1]),
            fsub(a[2], b[2]),
        ];
    }
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// `evaluate_angular_velocity_integrator(...)`.
pub fn evaluate_angular_velocity_integrator(
    angular_velocity_cc_before: [f64; 3],
    accumulator_620: [f64; 3],
    matrix_9c4: [f64; 9],
    dt: f64,
    numeric_mode: &str,
) -> Result<AngularVelocityIntegratorResult, String> {
    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    let mapped_omega =
        transform_row_major(&matrix_9c4, &angular_velocity_cc_before, numeric_mode)?;
    let gyroscopic = cross(angular_velocity_cc_before, mapped_omega, numeric_mode);
    let torque_minus_gyro = sub(accumulator_620, gyroscopic, numeric_mode);
    let angular_acceleration =
        transform_row_major(&matrix_9c4, &torque_minus_gyro, numeric_mode)?;
    let mut delta = [0.0, 0.0, 0.0];
    let mut after = [0.0, 0.0, 0.0];
    if numeric_mode == "float32" {
        for i in 0..3 {
            delta[i] = fmul(angular_acceleration[i], dt);
            after[i] = fadd(angular_velocity_cc_before[i], delta[i]);
        }
    } else {
        for i in 0..3 {
            delta[i] = angular_acceleration[i] * dt;
            after[i] = angular_velocity_cc_before[i] + delta[i];
        }
    }
    Ok(AngularVelocityIntegratorResult {
        angular_velocity_cc_before,
        mapped_omega,
        gyroscopic_term: gyroscopic,
        accumulator_620,
        torque_minus_gyro,
        angular_acceleration,
        delta_angular_velocity: delta,
        angular_velocity_cc_after: after,
        numeric_mode: numeric_mode.to_string(),
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OMEGA: [f64; 3] = [1.5, -0.75, 2.25];
    const ACC: [f64; 3] = [0.5, 1.25, -2.0];
    const M: [f64; 9] = [
        0.36, 0.48, -0.8, //
        -0.8, 0.6, 0.0, //
        0.48, 0.64, 0.6,
    ];

    #[test]
    fn float64_mode() {
        // Constants computed with the Python authority:
        //   evaluate_angular_velocity_integrator(angular_velocity_cc_before=OMEGA,
        //     accumulator_620=ACC, matrix_9c4=M, dt=0.002, numeric_mode="float64")
        let r =
            evaluate_angular_velocity_integrator(OMEGA, ACC, M, 0.002, "float64").unwrap();
        assert_eq!(r.mapped_omega, [-1.62, -1.6500000000000001, 1.5899999999999999]);
        assert_eq!(r.gyroscopic_term, [2.5200000000000005, -6.03, -3.6900000000000004]);
        assert_eq!(r.torque_minus_gyro, [-2.0200000000000005, 7.28, 1.6900000000000004]);
        assert_eq!(
            r.angular_velocity_cc_after,
            [1.5028304, -0.738032, 2.2594072]
        );
    }

    #[test]
    fn float32_mode() {
        // Constants computed with the Python authority (same inputs, float32).
        let r =
            evaluate_angular_velocity_integrator(OMEGA, ACC, M, 0.002, "float32").unwrap();
        assert_eq!(
            r.mapped_omega,
            [-1.6200000047683716, -1.6500000953674316, 1.590000033378601]
        );
        assert_eq!(
            r.angular_velocity_cc_after,
            [1.5028303861618042, -0.7380319833755493, 2.2594072818756104]
        );
    }

    #[test]
    fn rejects_unknown_mode() {
        assert_eq!(
            evaluate_angular_velocity_integrator(OMEGA, ACC, M, 0.002, "float16").unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }
}
