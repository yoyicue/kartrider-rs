//! Port of `scripts/evaluate_bs_zone_override.py` — executable BSx.y
//! booster-zone replacement barrier for +0x614.

use crate::num::sum_f64;

/// Python result key `ordering`.
pub const ORDERING: &str = "compute BS vector -> replace all three +0x614 components -> accepted trigger SetCurrentState(1000,13); +0x620 is untouched";

// static_source provenance (Python result key `static_source`).
pub const STATIC_HANDLER_VA: &str = "0xB2C890";
pub const STATIC_REPLACEMENT_WRITES_VA: &str = "0xB2CAC3/0xB2CAC8/0xB2CACE";
pub const STATIC_STATE13_CALL_VA: &str = "0xB2CAFB";

/// Python `state_after` dict: `{"decoded_state": 13, "remaining_ms": 1000}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BsZoneStateAfter {
    pub decoded_state: i64,
    pub remaining_ms: i64,
}

/// Python `evaluate_bs_zone_override(...)` result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct BsZoneOverrideResult {
    pub token: String,
    pub token_value: f64,
    pub magnitude: f64,
    pub velocity_direction: [f64; 3],
    pub bs_vector: [f64; 3],
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub discarded_accumulator_614: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub state_after: BsZoneStateAfter,
    pub ordering: &'static str,
}

/// `_normalize`: `math.sqrt(sum(v*v))` is CPython 3.12+ `sum()` (Neumaier,
/// crate::num::sum_f64); zero length maps to the zero vector.
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
        [vector[0] / length, vector[1] / length, vector[2] / length]
    }
}

/// `evaluate_bs_zone_override(...)`.
pub fn evaluate_bs_zone_override(
    digit_x: i64,
    digit_y: i64,
    flag_64c: bool,
    field_190: [f64; 3],
    velocity_c0: [f64; 3],
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
) -> Result<BsZoneOverrideResult, String> {
    if !((0..=9).contains(&digit_x) && (0..=9).contains(&digit_y)) {
        return Err("BS digits must be in 0..9".to_string());
    }
    let token_value = digit_x as f64 * 10000.0 + digit_y as f64 * 1000.0;
    let magnitude = token_value * (if flag_64c { 6.0 } else { 3.0 });
    let velocity_direction = normalize(velocity_c0);
    let mut vector = [0.0; 3];
    for i in 0..3 {
        vector[i] = magnitude * field_190[i] - 0.7 * magnitude * velocity_direction[i];
    }
    Ok(BsZoneOverrideResult {
        token: format!("BS{digit_x}.{digit_y}"),
        token_value,
        magnitude,
        velocity_direction,
        bs_vector: vector,
        accumulator_614_before,
        accumulator_614_after: vector,
        discarded_accumulator_614: accumulator_614_before,
        accumulator_620_before,
        accumulator_620_after: accumulator_620_before,
        state_after: BsZoneStateAfter { decoded_state: 13, remaining_ms: 1000 },
        ordering: ORDERING,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bs23_flag_set_replaces_accumulator() {
        // Constants computed with the Python authority:
        //   evaluate_bs_zone_override(digit_x=2, digit_y=3, flag_64c=True,
        //     field_190=[0.5,-0.5,0.25], velocity_c0=[3.0,0.0,4.0],
        //     accumulator_614_before=[1,2,3], accumulator_620_before=[4,5,6])
        let r = evaluate_bs_zone_override(
            2,
            3,
            true,
            [0.5, -0.5, 0.25],
            [3.0, 0.0, 4.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        )
        .unwrap();
        assert_eq!(r.token, "BS2.3");
        assert_eq!(r.token_value, 23000.0);
        assert_eq!(r.magnitude, 138000.0);
        assert_eq!(r.velocity_direction, [0.6, 0.0, 0.8]);
        assert_eq!(r.state_after, BsZoneStateAfter { decoded_state: 13, remaining_ms: 1000 });
        assert_eq!(r.accumulator_614_after, r.bs_vector);
        assert_eq!(r.accumulator_620_after, [4.0, 5.0, 6.0]);
    }

    #[test]
    fn digit_range_is_validated() {
        assert_eq!(
            evaluate_bs_zone_override(10, 0, false, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3])
                .unwrap_err(),
            "BS digits must be in 0..9"
        );
        assert!(evaluate_bs_zone_override(-1, 0, false, [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3])
            .is_err());
    }

    #[test]
    fn zero_velocity_normalizes_to_zero() {
        // Python authority: zero-length velocity -> direction [0,0,0].
        let r = evaluate_bs_zone_override(
            0,
            1,
            false,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 3],
        )
        .unwrap();
        assert_eq!(r.velocity_direction, [0.0, 0.0, 0.0]);
        assert_eq!(r.magnitude, 3000.0);
        assert_eq!(r.bs_vector, [3000.0, 0.0, 0.0]);
    }
}
