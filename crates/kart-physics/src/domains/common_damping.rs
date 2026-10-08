//! Port of `scripts/evaluate_common_damping.py` — executable common
//! linear/angular damping producer sub_B2D5D0. Both `numeric_mode` paths.
//!
//! Note this module's `_add` is a plain `a[i] + b[i]` (NOT a `sum()`), so
//! the float64 path keeps plain additions and `-0.0` is preserved.

use crate::num::{f32q, fadd, fdiv, fmul, sum_f64};

pub const ORDERING: &str =
    "build -C0*obf700 and -CC*obf700; optional subtract normalize(C0)*obf70C*obfBA0*token; add to +0x614/+0x620";

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2D5D0";
pub const STATIC_BASE_DAMPING_VA: &str = "0xB2D5FD-0xB2D659";
pub const STATIC_SPECIAL_DIRECTIONAL_VA: &str = "0xB2D675-0xB2D88C";
pub const STATIC_SINK_VA: &str = "0xB2D899-0xB2D8B8";

/// Python `_normalize` (float64 path): `math.sqrt(sum(v * v))` then divide.
fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = sum_f64(&[
        vector[0] * vector[0],
        vector[1] * vector[1],
        vector[2] * vector[2],
    ])
    .sqrt();
    if length <= 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [
            vector[0] / length,
            vector[1] / length,
            vector[2] / length,
        ]
    }
}

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct CommonDampingResult {
    pub linear_damping_delta: [f64; 3],
    pub angular_damping_delta: [f64; 3],
    pub normalized_velocity_direction: [f64; 3],
    pub special_directional_delta: [f64; 3],
    pub directional_multiplier: f64,
    pub directional_multiplier_source: &'static str,
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub numeric_mode: String,
    pub ordering: &'static str,
}

/// `evaluate_common_damping(...)`.
pub fn evaluate_common_damping(
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    velocity_c0: [f64; 3],
    angular_velocity_cc: [f64; 3],
    damping_700: f64,
    special_gate: bool,
    directional_70c: f64,
    runtime_ba0: f64,
    token_multiplier: Option<f64>,
    numeric_mode: &str,
) -> Result<CommonDampingResult, String> {
    if numeric_mode != "float32" && numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    let float32_mode = numeric_mode == "float32";
    let mut linear_delta = [0.0, 0.0, 0.0];
    let mut angular_delta = [0.0, 0.0, 0.0];
    let speed_sq: f64;
    let direction: [f64; 3];
    if float32_mode {
        for i in 0..3 {
            linear_delta[i] = fmul(-f32q(velocity_c0[i]), damping_700);
            angular_delta[i] = fmul(-f32q(angular_velocity_cc[i]), damping_700);
        }
        let mut sq = f32q(0.0);
        for value in velocity_c0 {
            sq = fadd(sq, fmul(value, value));
        }
        speed_sq = sq;
        let length = f32q(speed_sq.sqrt());
        direction = if length <= 0.0 {
            [0.0, 0.0, 0.0]
        } else {
            [
                fdiv(velocity_c0[0], length),
                fdiv(velocity_c0[1], length),
                fdiv(velocity_c0[2], length),
            ]
        };
    } else {
        for i in 0..3 {
            linear_delta[i] = -velocity_c0[i] * damping_700;
            angular_delta[i] = -angular_velocity_cc[i] * damping_700;
        }
        speed_sq = sum_f64(&[
            velocity_c0[0] * velocity_c0[0],
            velocity_c0[1] * velocity_c0[1],
            velocity_c0[2] * velocity_c0[2],
        ]);
        direction = normalize(velocity_c0);
    }
    let mut directional_delta = [0.0, 0.0, 0.0];
    if special_gate {
        // Default loc_B2D819 uses |C0|^2. The alternate configuration-token
        // path supplies an explicit parsed multiplier instead.
        let effective_multiplier = token_multiplier.unwrap_or(speed_sq);
        if float32_mode {
            let scale = fmul(fmul(directional_70c, runtime_ba0), effective_multiplier);
            for i in 0..3 {
                directional_delta[i] = fmul(-f32q(direction[i]), scale);
                linear_delta[i] = fadd(linear_delta[i], directional_delta[i]);
            }
        } else {
            let scale = directional_70c * runtime_ba0 * effective_multiplier;
            for i in 0..3 {
                directional_delta[i] = -direction[i] * scale;
                linear_delta[i] += directional_delta[i];
            }
        }
    }
    let add = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        if float32_mode {
            [
                fadd(a[0], b[0]),
                fadd(a[1], b[1]),
                fadd(a[2], b[2]),
            ]
        } else {
            // Plain a[i] + b[i], matching this module's Python `_add`.
            [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
        }
    };
    Ok(CommonDampingResult {
        linear_damping_delta: linear_delta,
        angular_damping_delta: angular_delta,
        normalized_velocity_direction: direction,
        special_directional_delta: directional_delta,
        directional_multiplier: token_multiplier.unwrap_or(speed_sq),
        directional_multiplier_source: if token_multiplier.is_some() {
            "configuration_token_override"
        } else {
            "speed_squared_default"
        },
        accumulator_614_before,
        accumulator_614_after: add(accumulator_614_before, linear_delta),
        accumulator_620_before,
        accumulator_620_after: add(accumulator_620_before, angular_delta),
        numeric_mode: numeric_mode.to_string(),
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(special_gate: bool, numeric_mode: &str) -> CommonDampingResult {
        evaluate_common_damping(
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 0.25],
            [3.0, 0.0, 4.0],
            [0.1, -0.2, 0.3],
            0.05,
            special_gate,
            0.1,
            1.5,
            None,
            numeric_mode,
        )
        .unwrap()
    }

    #[test]
    fn float32_with_special_gate() {
        // Constants computed with the Python authority:
        //   evaluate_common_damping(..., special_gate=True, numeric_mode="float32")
        let r = call(true, "float32");
        assert_eq!(
            r.linear_damping_delta,
            [-2.4000003337860107, -0.0, -3.200000286102295]
        );
        assert!(r.linear_damping_delta[1].is_sign_negative());
        assert_eq!(r.directional_multiplier, 25.0);
        assert_eq!(r.directional_multiplier_source, "speed_squared_default");
        assert_eq!(
            r.normalized_velocity_direction,
            [0.6000000238418579, 0.0, 0.800000011920929]
        );
        assert_eq!(
            r.accumulator_614_after,
            [-1.4000003337860107, 2.0, -0.20000028610229492]
        );
        assert_eq!(
            r.accumulator_620_after,
            [0.4950000047683716, -0.49000000953674316, 0.23499999940395355]
        );
    }

    #[test]
    fn float64_with_special_gate() {
        // Constants computed with the Python authority (same inputs, float64).
        let r = call(true, "float64");
        assert_eq!(r.linear_damping_delta, [-2.4, -0.0, -3.2000000000000006]);
        assert_eq!(r.normalized_velocity_direction, [0.6, 0.0, 0.8]);
        assert_eq!(r.special_directional_delta, [-2.25, -0.0, -3.0000000000000004]);
        assert_eq!(r.accumulator_614_after, [-1.4, 2.0, -0.20000000000000062]);
        assert_eq!(r.accumulator_620_after, [0.495, -0.49, 0.235]);
    }

    #[test]
    fn error_guard_matches_python_message() {
        assert_eq!(
            evaluate_common_damping(
                [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3], 0.05, false, 0.0, 0.0, None, "float16",
            )
            .unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }
}
