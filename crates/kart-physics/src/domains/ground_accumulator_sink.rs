//! Port of `scripts/evaluate_ground_accumulator_sink.py` — exact common
//! ground-force accumulator sink at B2C111..B2C186
//! (`transform_row_major` + `apply_ground_accumulator_sink`). Row-major
//! matrix(+0x9C) * vector.
//!
//! Note this module's `_add` is a plain `a[i] + b[i]` (NOT a `sum()`), so
//! the float64 path keeps plain additions and `-0.0` is preserved.

use crate::num::{fadd, fmul};

pub const ORDERING: &str =
    "row-major matrix(+0x9C) * local_force -> add to +0x614; add untransformed local_torque to +0x620";

// static_source provenance (Python result key `static_source`).
pub const STATIC_TRANSFORM_FUNCTION_VA: &str = "0x72C030";
pub const STATIC_FORCE_SINK_VA: &str = "0xB2C111-0xB2C13A";
pub const STATIC_TORQUE_SINK_VA: &str = "0xB2C177-0xB2C186";

/// `transform_row_major(matrix_9c, vector, numeric_mode=...)`.
///
/// The Python length guard ("matrix_9c must have 9 floats and vector must
/// have 3") is compile-time in Rust; only the numeric_mode guard remains.
pub fn transform_row_major(
    matrix_9c: &[f64; 9],
    vector: &[f64; 3],
    numeric_mode: &str,
) -> Result<[f64; 3], String> {
    if numeric_mode == "float32" {
        let mut out = [0.0, 0.0, 0.0];
        for row in 0..3 {
            out[row] = fadd(
                fadd(
                    fmul(matrix_9c[row * 3], vector[0]),
                    fmul(matrix_9c[row * 3 + 1], vector[1]),
                ),
                fmul(matrix_9c[row * 3 + 2], vector[2]),
            );
        }
        return Ok(out);
    }
    if numeric_mode != "float64" {
        return Err("numeric_mode must be float32 or float64".to_string());
    }
    Ok([
        matrix_9c[0] * vector[0] + matrix_9c[1] * vector[1] + matrix_9c[2] * vector[2],
        matrix_9c[3] * vector[0] + matrix_9c[4] * vector[1] + matrix_9c[5] * vector[2],
        matrix_9c[6] * vector[0] + matrix_9c[7] * vector[1] + matrix_9c[8] * vector[2],
    ])
}

/// Python `apply_ground_accumulator_sink(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundAccumulatorSinkResult {
    pub local_force: [f64; 3],
    pub world_force: [f64; 3],
    pub local_torque: [f64; 3],
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub numeric_mode: String,
    pub ordering: &'static str,
}

/// `apply_ground_accumulator_sink(...)`.
pub fn apply_ground_accumulator_sink(
    local_force: [f64; 3],
    local_torque: [f64; 3],
    matrix_9c: [f64; 9],
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    numeric_mode: &str,
) -> Result<GroundAccumulatorSinkResult, String> {
    // Python evaluates the transform first, so an unknown numeric_mode fails
    // with transform_row_major's message before any add runs.
    let world_force = transform_row_major(&matrix_9c, &local_force, numeric_mode)?;
    let add = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        if numeric_mode == "float32" {
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
    Ok(GroundAccumulatorSinkResult {
        local_force,
        world_force,
        local_torque,
        accumulator_614_before,
        accumulator_614_after: add(accumulator_614_before, world_force),
        accumulator_620_before,
        accumulator_620_after: add(accumulator_620_before, local_torque),
        numeric_mode: numeric_mode.to_string(),
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: [f64; 9] = [
        0.36, 0.48, -0.8, //
        -0.8, 0.6, 0.0, //
        0.48, 0.64, 0.6,
    ];
    const V: [f64; 3] = [1.5, -2.25, 0.75];

    #[test]
    fn float64_mode() {
        // Constants computed with the Python authority:
        //   transform_row_major(M, V, numeric_mode="float64")
        let r = transform_row_major(&M, &V, "float64").unwrap();
        assert_eq!(r, [-1.1400000000000001, -2.55, -0.27]);
    }

    #[test]
    fn float32_mode() {
        // Constants computed with the Python authority:
        //   transform_row_major(M, V, numeric_mode="float32")
        let r = transform_row_major(&M, &V, "float32").unwrap();
        assert_eq!(
            r,
            [
                -1.1399998664855957,
                -2.5500001907348633,
                -0.2699999511241913
            ]
        );
    }

    #[test]
    fn rejects_unknown_mode() {
        assert_eq!(
            transform_row_major(&M, &V, "float16").unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }

    #[test]
    fn sink_float32() {
        // Constants computed with the Python authority:
        //   apply_ground_accumulator_sink(local_force=[10,-5,0],
        //     local_torque=[0,1.5,-0.5], matrix_9c=M,
        //     accumulator_614_before=[1,2,3], accumulator_620_before=[0.5,-0.5,0.25],
        //     numeric_mode="float32")
        let r = apply_ground_accumulator_sink(
            [10.0, -5.0, 0.0],
            [0.0, 1.5, -0.5],
            M,
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 0.25],
            "float32",
        )
        .unwrap();
        assert_eq!(r.world_force, [1.200000286102295, -11.0, 1.5999999046325684]);
        assert_eq!(
            r.accumulator_614_after,
            [2.200000286102295, -9.0, 4.599999904632568]
        );
        assert_eq!(r.accumulator_620_after, [0.5, 1.0, -0.25]);
    }

    #[test]
    fn sink_propagates_transform_mode_error() {
        assert_eq!(
            apply_ground_accumulator_sink(
                [0.0; 3],
                [0.0; 3],
                M,
                [0.0; 3],
                [0.0; 3],
                "float16",
            )
            .unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }
}
