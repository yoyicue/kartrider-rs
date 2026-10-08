//! Port of `scripts/evaluate_rail_constraint_orientation.py` — RAIL-relative
//! orientation producer used by `sub_B2DA80`.
//!
//! State 3 relaxes the relative C1C quaternion toward identity (B29350),
//! integrates that relative quaternion (B2DA80), and replaces the primary
//! matrix with `BF8 * C1C`. Other active RAIL states use the ordinary
//! primary integrator and then regenerate C1C relative to BF8.

use crate::num::{f32q, sum_f64};

use super::orientation_integrator::{
    evaluate_orientation_integrator, matrix_to_quat, normalize, quat_to_matrix,
    runtime_integrate_upright, runtime_matrix_to_quat_upright, runtime_quat_to_matrix,
    OrientationIntegratorResult,
};

/// Python result key `writer` on each branch.
pub const WRITER_STATE3: &str = "state3_B2DA80_direct_BF8_times_C1C";
pub const WRITER_NON_STATE3: &str = "non_state3_primary_integrate_then_relative_C1C";

/// Python result key `static_order` on each branch.
pub const STATIC_ORDER_STATE3: &str =
    "B29350_C1C_slerp_identity -> B2DA80_C1C_integrate -> primary_9C=BF8*C1C";
pub const STATIC_ORDER_NON_STATE3: &str =
    "ordinary_primary_integration -> C1C=inverse(BF8)*primary_9C";

/// Python `_matmul(a, b, float32=...)`: row-major 3x3 product.
fn matmul(a: [f64; 9], b: [f64; 9], float32: bool) -> [f64; 9] {
    let mut result = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            let value = if float32 {
                let mut value = f32q(0.0);
                for index in 0..3 {
                    value = f32q(
                        value + f32q(f32q(a[row * 3 + index]) * f32q(b[index * 3 + column])),
                    );
                }
                value
            } else {
                // Python sum(...) over three products: CPython Neumaier.
                sum_f64(&[
                    a[row * 3] * b[column],
                    a[row * 3 + 1] * b[3 + column],
                    a[row * 3 + 2] * b[6 + column],
                ])
            };
            result[row * 3 + column] = value;
        }
    }
    result
}

/// Python `_transpose(matrix)`.
fn transpose(matrix: [f64; 9]) -> [f64; 9] {
    let mut result = [0.0; 9];
    for row in 0..3 {
        for column in 0..3 {
            result[row * 3 + column] = matrix[column * 3 + row];
        }
    }
    result
}

/// Python `_slerp_identity(quaternion, amount, float32=...)`: shortest-arc
/// SLERP from `quaternion` to `[1, 0, 0, 0]` (f64 reference path only; the
/// runtime path uses [`runtime_interpolate_identity`]).
fn slerp_identity(quaternion: [f64; 4], amount: f64, float32: bool) -> Result<[f64; 4], String> {
    let mut q = normalize(quaternion)?;
    if q[0] < 0.0 {
        q = [-q[0], -q[1], -q[2], -q[3]];
    }
    let t = crate::num::max2_f64(0.0, crate::num::min2_f64(1.0, amount));
    let dot = crate::num::max2_f64(-1.0, crate::num::min2_f64(1.0, q[0]));
    let result = if dot > 0.9995 {
        let mixed = [
            q[0] + t * (1.0 - q[0]),
            q[1] * (1.0 - t),
            q[2] * (1.0 - t),
            q[3] * (1.0 - t),
        ];
        normalize(mixed)?
    } else {
        let theta = dot.acos();
        let sine = theta.sin();
        let left = ((1.0 - t) * theta).sin() / sine;
        let right = (t * theta).sin() / sine;
        [left * q[0] + right, left * q[1], left * q[2], left * q[3]]
    };
    Ok(if float32 {
        [f32q(result[0]), f32q(result[1]), f32q(result[2]), f32q(result[3])]
    } else {
        result
    })
}

/// Python `_runtime_interpolate_identity(quaternion, amount)`: exact
/// float32 `sub_01039610` approximation toward identity. Not a
/// trigonometric SLERP: warps the lerp fraction with two fitted constants,
/// lerps the four components, then applies a piecewise approximate
/// reciprocal square root.
fn runtime_interpolate_identity(quaternion: [f64; 4], amount: f64) -> [f64; 4] {
    let mut current = [
        f32q(quaternion[0]),
        f32q(quaternion[1]),
        f32q(quaternion[2]),
        f32q(quaternion[3]),
    ];
    if current[0] < 0.0 {
        current = [
            f32q(-current[0]),
            f32q(-current[1]),
            f32q(-current[2]),
            f32q(-current[3]),
        ];
    }
    let t = f32q(crate::num::max2_f64(0.0, crate::num::min2_f64(1.0, amount)));

    // 01039623..0103967D. Preserve the executable's scalar SSE order.
    let mut dot = f32q(f32q(0.0 * current[1]) + f32q(1.0 * current[0]));
    dot = f32q(dot + f32q(0.0 * current[2]));
    dot = f32q(dot + f32q(0.0 * current[3]));
    let mut curvature = f32q(1.0 - f32q(dot * f32q(0.8227968811988831)));
    curvature = f32q(curvature * curvature);
    curvature = f32q(curvature * f32q(0.5854921936988831));

    let factor;
    if t <= f32q(0.5) {
        let factor_c = f32q(curvature * t);
        let mut polynomial = f32q(t + t);
        polynomial = f32q(polynomial - f32q(3.0));
        polynomial = f32q(polynomial * factor_c);
        polynomial = f32q(polynomial + f32q(1.0));
        polynomial = f32q(polynomial + curvature);
        factor = f32q(polynomial * t);
    } else {
        let complement = f32q(1.0 - t);
        let factor_c = f32q(curvature * complement);
        let mut polynomial = f32q(complement + complement);
        polynomial = f32q(polynomial - f32q(3.0));
        polynomial = f32q(polynomial * factor_c);
        polynomial = f32q(polynomial + f32q(1.0));
        polynomial = f32q(polynomial + curvature);
        let complement_factor = f32q(polynomial * complement);
        factor = f32q(1.0 - complement_factor);
    }

    let identity = [f32q(1.0), f32q(0.0), f32q(0.0), f32q(0.0)];
    let mut mixed = [0.0; 4];
    for index in 0..4 {
        mixed[index] = f32q(current[index] + f32q(f32q(identity[index] - current[index]) * factor));
    }

    let mut norm_sq = f32q(mixed[0] * mixed[0]);
    norm_sq = f32q(norm_sq + f32q(mixed[1] * mixed[1]));
    norm_sq = f32q(norm_sq + f32q(mixed[2] * mixed[2]));
    norm_sq = f32q(norm_sq + f32q(mixed[3] * mixed[3]));
    // Runtime-initialized globals 0x013BFD1C / 0x013BFD18.
    let reciprocal_slope = f32q(-0.5325155854225159);
    let reciprocal_intercept = f32q(1.0214351415634155);
    let mut inverse = f32q(norm_sq - f32q(0.9590659737586975));
    inverse = f32q(inverse * reciprocal_slope);
    inverse = f32q(inverse + reciprocal_intercept);

    fn newton(value: f64, norm_sq: f64, reciprocal_slope: f64, reciprocal_intercept: f64) -> f64 {
        let mut correction = f32q(value * value);
        correction = f32q(correction * norm_sq);
        correction = f32q(correction - f32q(1.0));
        correction = f32q(correction * reciprocal_slope);
        correction = f32q(correction + reciprocal_intercept);
        f32q(value * correction)
    }

    if norm_sq <= f32q(0.9152119755744934) {
        inverse = newton(inverse, norm_sq, reciprocal_slope, reciprocal_intercept);
        if norm_sq <= f32q(0.6521196961402893) {
            inverse = newton(inverse, norm_sq, reciprocal_slope, reciprocal_intercept);
        }
    }
    [
        f32q(mixed[0] * inverse),
        f32q(mixed[1] * inverse),
        f32q(mixed[2] * inverse),
        f32q(mixed[3] * inverse),
    ]
}

/// Python `evaluate_rail_constraint_orientation(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailConstraintOrientation {
    pub primary_matrix_after: [f64; 9],
    pub rail_base_bf8_after: [f64; 9],
    pub rail_constraint_c1c_after: [f64; 9],
    /// State-3 branch keys (None on the non-state3 branch).
    pub constraint_quaternion_before: Option<[f64; 4]>,
    pub constraint_quaternion_relaxed: Option<[f64; 4]>,
    pub constraint_quaternion_after: Option<[f64; 4]>,
    pub relaxation_amount: Option<f64>,
    /// Non-state3 branch key: the ordinary integrator result.
    pub orientation_integrator: Option<OrientationIntegratorResult>,
    pub writer: &'static str,
    pub static_order: &'static str,
}

/// `evaluate_rail_constraint_orientation(...)`. Matrix/vector length checks
/// are type-enforced in Rust.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rail_constraint_orientation(
    primary_matrix_before: [f64; 9],
    rail_base_bf8: [f64; 9],
    rail_constraint_c1c_before: [f64; 9],
    angular_velocity: [f64; 3],
    dt: f64,
    rail_state: i64,
    runtime_float32_exact: bool,
) -> Result<RailConstraintOrientation, String> {
    if !(1..=6).contains(&rail_state) {
        return Err("active RAIL state must be in 1..6".to_string());
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err("dt must be finite and positive".to_string());
    }

    if rail_state == 3 {
        let quaternion_before = if runtime_float32_exact {
            runtime_matrix_to_quat_upright(rail_constraint_c1c_before)?
        } else {
            matrix_to_quat(rail_constraint_c1c_before)?
        };
        let relaxation_amount = crate::num::min2_f64(dt * 3.0, 1.0);
        let relaxed_quaternion = if runtime_float32_exact {
            runtime_interpolate_identity(quaternion_before, relaxation_amount)
        } else {
            slerp_identity(quaternion_before, relaxation_amount, false)?
        };
        let relaxed_constraint = if runtime_float32_exact {
            runtime_quat_to_matrix(relaxed_quaternion)
        } else {
            quat_to_matrix(relaxed_quaternion)
        };
        let (integrated_quaternion, constraint_after);
        if runtime_float32_exact {
            let (_, integrated, matrix) =
                runtime_integrate_upright(relaxed_constraint, angular_velocity, dt)?;
            integrated_quaternion = integrated;
            constraint_after = matrix;
        } else {
            let derivative = [
                -relaxed_quaternion[1] * angular_velocity[0]
                    - relaxed_quaternion[2] * angular_velocity[1]
                    - relaxed_quaternion[3] * angular_velocity[2],
                relaxed_quaternion[0] * angular_velocity[0]
                    + relaxed_quaternion[2] * angular_velocity[2]
                    - relaxed_quaternion[3] * angular_velocity[1],
                relaxed_quaternion[0] * angular_velocity[1]
                    + relaxed_quaternion[3] * angular_velocity[0]
                    - relaxed_quaternion[1] * angular_velocity[2],
                relaxed_quaternion[0] * angular_velocity[2]
                    + relaxed_quaternion[1] * angular_velocity[1]
                    - relaxed_quaternion[2] * angular_velocity[0],
            ];
            let mut integrated = [0.0; 4];
            for index in 0..4 {
                integrated[index] = relaxed_quaternion[index] + derivative[index] * (dt * 0.5);
            }
            integrated_quaternion = normalize(integrated)?;
            constraint_after = quat_to_matrix(integrated_quaternion);
        }
        let primary_after = matmul(rail_base_bf8, constraint_after, runtime_float32_exact);
        return Ok(RailConstraintOrientation {
            primary_matrix_after: primary_after,
            rail_base_bf8_after: rail_base_bf8,
            rail_constraint_c1c_after: constraint_after,
            constraint_quaternion_before: Some(quaternion_before),
            constraint_quaternion_relaxed: Some(relaxed_quaternion),
            constraint_quaternion_after: Some(integrated_quaternion),
            relaxation_amount: Some(relaxation_amount),
            orientation_integrator: None,
            writer: WRITER_STATE3,
            static_order: STATIC_ORDER_STATE3,
        });
    }

    let ordinary = evaluate_orientation_integrator(
        primary_matrix_before,
        angular_velocity,
        dt,
        false,
        false,
        rail_state,
        runtime_float32_exact,
    )?;
    let primary_after = ordinary.matrix_9c_after;
    let constraint_after = matmul(transpose(rail_base_bf8), primary_after, runtime_float32_exact);
    Ok(RailConstraintOrientation {
        primary_matrix_after: primary_after,
        rail_base_bf8_after: rail_base_bf8,
        rail_constraint_c1c_after: constraint_after,
        constraint_quaternion_before: None,
        constraint_quaternion_relaxed: None,
        constraint_quaternion_after: None,
        relaxation_amount: None,
        orientation_integrator: Some(ordinary),
        writer: WRITER_NON_STATE3,
        static_order: STATIC_ORDER_NON_STATE3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn argument_validation() {
        assert_eq!(
            evaluate_rail_constraint_orientation(IDENTITY, IDENTITY, IDENTITY, [0.0; 3], 0.002, 0, true)
                .unwrap_err(),
            "active RAIL state must be in 1..6"
        );
        assert_eq!(
            evaluate_rail_constraint_orientation(IDENTITY, IDENTITY, IDENTITY, [0.0; 3], 0.0, 3, true)
                .unwrap_err(),
            "dt must be finite and positive"
        );
    }

    #[test]
    fn state3_identity_stays_identity_both_modes() {
        // Python authority: identity C1C relaxed toward identity and
        // integrated with zero angular velocity stays identity; primary
        // after = BF8 * I = BF8.
        for float32 in [true, false] {
            let result = evaluate_rail_constraint_orientation(
                IDENTITY, IDENTITY, IDENTITY, [0.0; 3], 0.002, 3, float32,
            )
            .unwrap();
            assert_eq!(result.writer, WRITER_STATE3);
            assert_eq!(result.relaxation_amount, Some(0.006));
            for index in 0..9 {
                let expected = IDENTITY[index];
                let actual = result.primary_matrix_after[index];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "index {index}: {actual} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn non_state3_regenerates_c1c_from_bf8() {
        // Python authority: with BF8 = identity and primary_before = identity,
        // zero angular velocity leaves primary identity and C1C identity.
        for float32 in [true, false] {
            let result = evaluate_rail_constraint_orientation(
                IDENTITY, IDENTITY, IDENTITY, [0.0; 3], 0.002, 1, float32,
            )
            .unwrap();
            assert_eq!(result.writer, WRITER_NON_STATE3);
            let ordinary = result.orientation_integrator.unwrap();
            for index in 0..9 {
                assert!((ordinary.matrix_9c_after[index] - IDENTITY[index]).abs() < 1e-9);
                assert!(
                    (result.rail_constraint_c1c_after[index] - IDENTITY[index]).abs() < 1e-9
                );
            }
        }
    }
}
