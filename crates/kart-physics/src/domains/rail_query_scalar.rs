//! Port of `scripts/evaluate_rail_query_scalar.py` — executable RAIL
//! per-kart path scalar recovered from `sub_A4F600`.
//!
//! The scalar is a clamped arc coordinate. All arithmetic runs through the
//! module-local `_f32`/`_sub`/`_dot`/`_length` f32 chain (mapped to
//! crate::num `f32q`/`dot3_f32`); see the Python module for provenance.

use crate::num::{dot3_f32, f32q, max2_f64, min2_f64};

/// Python module constant `MIN_SEGMENT_LENGTH`
/// (already the f32 rounding of 0.1; `_f32` is the identity on it).
pub const MIN_SEGMENT_LENGTH: f64 = 0.10000000149011612;

/// Python result key `authority`.
pub const AUTHORITY: &str = "executable_sub_A4F600_equivalent_path_coordinate";

/// One decoded path record: Python resolves `position` (fallback `primary`)
/// and `secondary` (fallback `direction`) from the mapping; the caller owns
/// that key resolution. Components are quantized to f32 and must be finite
/// (Python `_vec3`).
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryScalarRecord {
    pub position: [f64; 3],
    pub direction: [f64; 3],
}

/// Python `evaluate_rail_query_scalar(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryScalarResult {
    pub query_scalar: f64,
    pub unclamped_query_scalar: f64,
    pub total_length: f64,
    pub selected_record_index: i64,
    pub accumulated_before_record: f64,
    pub local_projection: f64,
    pub local_direction: [f64; 3],
    pub local_direction_source: &'static str,
    pub authority: &'static str,
}

/// Python `_vec3(value, name)`: elementwise f32 quantization + finite check.
/// The three-component length check is type-enforced in Rust.
fn vec3(value: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    let result = [f32q(value[0]), f32q(value[1]), f32q(value[2])];
    if !result.iter().all(|component| component.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    Ok(result)
}

/// Python `_sub(left, right)`: elementwise `_f32(a - b)`. Operands are
/// already f32-quantized at every call site, so `_f32(a) - _f32(b)` is the
/// plain subtraction.
fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        f32q(left[0] - right[0]),
        f32q(left[1] - right[1]),
        f32q(left[2] - right[2]),
    ]
}

/// Python `_dot(left, right)`: `_f32(l0*r0)`, then two `_f32(acc + _f32(p))`
/// steps. Identical to the `dot3_f32` accumulator (its explicit `+0.0` start
/// folds away into the first product).
fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot3_f32(left, right)
}

/// Python `_length(vector)`: `_f32(sqrt(max(0.0, _dot(v, v))))`.
fn length(vector: [f64; 3]) -> f64 {
    f32q(max2_f64(0.0, dot(vector, vector)).sqrt())
}

/// `evaluate_rail_query_scalar(records, query_position, total_length=None)`.
pub fn evaluate_rail_query_scalar(
    records: &[RailQueryScalarRecord],
    query_position: [f64; 3],
    total_length: Option<f64>,
) -> Result<RailQueryScalarResult, String> {
    if records.is_empty() {
        return Err("rail query scalar requires at least one record".to_string());
    }
    let query = vec3(query_position, "query_position")?;
    let mut decoded = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        decoded.push(RailQueryScalarRecord {
            position: vec3(record.position, &format!("record {index} position"))?,
            direction: vec3(record.direction, &format!("record {index} secondary"))?,
        });
    }

    let mut selected_index = 0usize;
    for (index, record) in decoded.iter().enumerate() {
        let relative = sub(query, record.position);
        if dot(record.direction, relative) < 0.0 {
            break;
        }
        selected_index = index;
    }

    let mut accumulated = f32q(0.0);
    for index in 0..selected_index {
        accumulated = f32q(
            accumulated
                + length(sub(
                    decoded[index + 1].position,
                    decoded[index].position,
                )),
        );
    }

    let selected = &decoded[selected_index];
    let mut local_direction = selected.direction;
    let mut direction_source = "authored_secondary";
    if selected_index + 1 < decoded.len() {
        let segment = sub(decoded[selected_index + 1].position, selected.position);
        let segment_length = length(segment);
        if segment_length > f32q(MIN_SEGMENT_LENGTH) {
            local_direction = [
                f32q(segment[0] / segment_length),
                f32q(segment[1] / segment_length),
                f32q(segment[2] / segment_length),
            ];
            direction_source = "normalized_next_primary_segment";
        }
    }

    let local_projection_raw = dot(sub(query, selected.position), local_direction);
    let local_projection = max2_f64(f32q(0.0), local_projection_raw);
    let unclamped = f32q(accumulated + local_projection);

    let total = match total_length {
        None => {
            let mut total = f32q(0.0);
            for index in 0..decoded.len().saturating_sub(1) {
                total = f32q(
                    total + length(sub(decoded[index + 1].position, decoded[index].position)),
                );
            }
            total
        }
        Some(total_length) => {
            let total = f32q(total_length);
            if !total.is_finite() || total < 0.0 {
                return Err("total_length must be finite and non-negative".to_string());
            }
            total
        }
    };

    let scalar = min2_f64(unclamped, total);
    Ok(RailQueryScalarResult {
        query_scalar: scalar,
        unclamped_query_scalar: unclamped,
        total_length: total,
        selected_record_index: selected_index as i64,
        accumulated_before_record: accumulated,
        local_projection,
        local_direction,
        local_direction_source: direction_source,
        authority: AUTHORITY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(position: [f64; 3], direction: [f64; 3]) -> RailQueryScalarRecord {
        RailQueryScalarRecord { position, direction }
    }

    #[test]
    fn requires_at_least_one_record() {
        assert_eq!(
            evaluate_rail_query_scalar(&[], [0.0; 3], None).unwrap_err(),
            "rail query scalar requires at least one record"
        );
    }

    #[test]
    fn rejects_non_finite_components() {
        assert_eq!(
            evaluate_rail_query_scalar(
                &[record([f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0])],
                [0.0; 3],
                None,
            )
            .unwrap_err(),
            "record 0 position must be finite"
        );
        assert_eq!(
            evaluate_rail_query_scalar(
                &[record([0.0; 3], [1.0, 0.0, 0.0])],
                [0.0; 3],
                Some(-1.0),
            )
            .unwrap_err(),
            "total_length must be finite and non-negative"
        );
    }

    #[test]
    fn straight_path_matches_python_authority() {
        // Constants computed with the Python authority:
        //   evaluate_rail_query_scalar(
        //     records=[{"position": [0,0,0], "secondary": [1,0,0]},
        //              {"position": [10,0,0], "secondary": [1,0,0]},
        //              {"position": [20,0,0], "secondary": [1,0,0]}],
        //     query_position=[13.5, 0.25, -0.5])
        let records = vec![
            record([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            record([10.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            record([20.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let result =
            evaluate_rail_query_scalar(&records, [13.5, 0.25, -0.5], None).unwrap();
        assert_eq!(result.selected_record_index, 1);
        assert_eq!(result.accumulated_before_record, 10.0);
        assert_eq!(result.local_projection, 3.5);
        assert_eq!(result.unclamped_query_scalar, 13.5);
        assert_eq!(result.query_scalar, 13.5);
        assert_eq!(result.total_length, 20.0);
        assert_eq!(result.local_direction_source, "normalized_next_primary_segment");
        assert_eq!(result.local_direction, [1.0, 0.0, 0.0]);
        // Last-record branch uses the authored secondary.
        let last =
            evaluate_rail_query_scalar(&records, [25.0, 0.0, 0.0], None).unwrap();
        assert_eq!(last.selected_record_index, 2);
        assert_eq!(last.local_direction_source, "authored_secondary");
        assert_eq!(last.query_scalar, 20.0); // clamped to total length
    }
}
