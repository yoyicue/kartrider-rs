//! Port of `scripts/evaluate_rail_base_orientation.py` — the RAIL
//! `GoPlayKart+0xBF8` path-frame matrix (straight-line body at
//! `sub_B30150+0x177..+0x207`).
//!
//! column 1 receives the negated interpolated secondary vector, column 0
//! receives `cross(column1, tertiary)`, column 2 receives
//! `cross(column0, column1)`. The `runtime_float32_exact` path quantizes
//! through the module-local `_f32`/`_cross_f32` chain; the fallback path is
//! plain f64.

use crate::num::f32q;

/// Python result key `closure_status`.
pub const CLOSURE_STATUS: &str = "exact_static_B30150_BF8_path_frame_body";

/// Python result key `formula`.
pub const FORMULA_COLUMN1: &str = "-secondary";
pub const FORMULA_COLUMN0: &str = "cross(column1, tertiary)";
pub const FORMULA_COLUMN2: &str = "cross(column0, column1)";

/// Python `_cross_f32(a, b)`: quantized cross product with fixed operand
/// order (note: `_f32(_f32(a)*_f32(b))` per product, then `_f32` of the
/// difference — identical to `fsub(fmul(...), fmul(...))` since the operands
/// are already f32-quantized).
fn cross_f32(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    crate::num::cross3_f32(a, b)
}

/// Python result dict minus `static_source` (provenance constants below).
#[derive(Debug, Clone, PartialEq)]
pub struct RailBaseOrientation {
    pub matrix_bf8: [f64; 9],
    pub column0: [f64; 3],
    pub column1: [f64; 3],
    pub column2: [f64; 3],
    pub secondary_endpoint: [f64; 3],
    pub tertiary_endpoint: [f64; 3],
    pub runtime_float32_exact: bool,
    pub closure_status: &'static str,
}

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB30150";
pub const STATIC_NEGATE_SECONDARY_VA: &str = "0xB302C7-0xB302DF";
pub const STATIC_COLUMN0_CROSS_VA: &str = "0xB302E4-0xB30315";
pub const STATIC_COLUMN2_CROSS_VA: &str = "0xB3031A-0xB30357";
pub const STATIC_COLUMN_GET_VA: &str = "0x6F7B70";
pub const STATIC_COLUMN_SET_VA: &str = "0x6F7D60";
pub const STATIC_CROSS_VA: &str = "0x6F3680";

/// `evaluate_rail_base_orientation(secondary_endpoint, tertiary_endpoint,
/// runtime_float32_exact=True)`. The three-component length checks are
/// type-enforced in Rust.
pub fn evaluate_rail_base_orientation(
    secondary_endpoint: [f64; 3],
    tertiary_endpoint: [f64; 3],
    runtime_float32_exact: bool,
) -> Result<RailBaseOrientation, String> {
    if !secondary_endpoint
        .iter()
        .chain(tertiary_endpoint.iter())
        .all(|value| value.is_finite())
    {
        return Err("RAIL path-frame vectors must be finite".to_string());
    }
    let secondary = secondary_endpoint;
    let tertiary = tertiary_endpoint;
    let (column1, column0, column2);
    if runtime_float32_exact {
        column1 = [f32q(-f32q(secondary[0])), f32q(-f32q(secondary[1])), f32q(-f32q(secondary[2]))];
        column0 = cross_f32(column1, tertiary);
        column2 = cross_f32(column0, column1);
    } else {
        column1 = [-secondary[0], -secondary[1], -secondary[2]];
        column0 = [
            column1[1] * tertiary[2] - column1[2] * tertiary[1],
            column1[2] * tertiary[0] - column1[0] * tertiary[2],
            column1[0] * tertiary[1] - column1[1] * tertiary[0],
        ];
        column2 = [
            column0[1] * column1[2] - column0[2] * column1[1],
            column0[2] * column1[0] - column0[0] * column1[2],
            column0[0] * column1[1] - column0[1] * column1[0],
        ];
    }
    let matrix = [
        column0[0], column1[0], column2[0], //
        column0[1], column1[1], column2[1], //
        column0[2], column1[2], column2[2],
    ];
    Ok(RailBaseOrientation {
        matrix_bf8: matrix,
        column0,
        column1,
        column2,
        secondary_endpoint: secondary,
        tertiary_endpoint: tertiary,
        runtime_float32_exact,
        closure_status: CLOSURE_STATUS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_finite_inputs() {
        assert_eq!(
            evaluate_rail_base_orientation([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0], true)
                .unwrap_err(),
            "RAIL path-frame vectors must be finite"
        );
    }

    #[test]
    fn axes_match_python_authority_both_modes() {
        // Constants computed with the Python authority:
        //   evaluate_rail_base_orientation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        // column1 = [-1,0,0]; column0 = cross([-1,0,0],[0,0,1]) = [0,1,0];
        // column2 = cross([0,1,0],[-1,0,0]) = [0,0,1].
        let result =
            evaluate_rail_base_orientation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], true).unwrap();
        assert_eq!(result.column1, [-1.0, 0.0, 0.0]);
        assert_eq!(result.column0, [0.0, 1.0, 0.0]);
        assert_eq!(result.column2, [0.0, 0.0, 1.0]);
        assert_eq!(
            result.matrix_bf8,
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        let plain =
            evaluate_rail_base_orientation([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], false).unwrap();
        assert_eq!(plain.matrix_bf8, result.matrix_bf8);
    }
}
