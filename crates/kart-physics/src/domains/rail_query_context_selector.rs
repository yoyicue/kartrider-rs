//! Port of `scripts/evaluate_rail_query_context_selector.py` — executable
//! RAIL path-context selector recovered from `sub_A4EA20`.
//!
//! Branch-ordered endpoint/line projection over consecutive path records,
//! then the recovered one-sided normal predicate. All arithmetic is the
//! module-local `_f32` chain (crate::num `f32q`/`dot3_f32`).

use crate::num::{dot3_f32, f32q};
use crate::state::JsonValue;

/// Python module constants.
pub const INITIAL_BEST_DISTANCE_SQUARED: f64 = 999999.0;
pub const NORMAL_SCALE: f64 = 0.20000000298023224;
pub const NORMAL_DOT_THRESHOLD: f64 = -0.10000000149011612;

/// Python result keys `authority` / `asset_boundary`.
pub const AUTHORITY: &str = "executable_sub_A4EA20_equivalent_registered_path_contexts";
pub const ASSET_BOUNDARY: &str =
    "ordered context points and authored +0x18 vectors must come from autonomous scene/path asset registration";

/// One decoded context point: Python resolves `position` and `normal` from
/// the mapping (caller owns key resolution); components are f32-quantized
/// and finite-checked (Python `_vec3`).
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryContextPoint {
    pub position: [f64; 3],
    pub normal: [f64; 3],
}

/// One registered path context. `context_id` mirrors Python
/// `context.get("context_id", context_index)`: `None` resolves to the
/// enumeration index.
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryContext {
    pub context_id: Option<JsonValue>,
    pub points: Vec<RailQueryContextPoint>,
}

/// Python `selected_detail` dict (present only while a candidate beat the
/// running distance, including normal-predicate rejections).
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryContextSelectionDetail {
    pub context_id: JsonValue,
    pub context_index: i64,
    pub segment_index: i64,
    pub region: &'static str,
    pub closest_point: [f64; 3],
    pub distance_squared: f64,
    pub normal_dot: f64,
    pub normal_predicate_passed: bool,
}

/// Python `select_rail_query_context(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailQueryContextSelection {
    pub selected: bool,
    pub selected_context: Option<JsonValue>,
    pub best_distance_squared: f64,
    pub selected_detail: Option<RailQueryContextSelectionDetail>,
    pub visited_contexts: i64,
    pub visited_segments: i64,
    pub authority: &'static str,
    pub asset_boundary: &'static str,
}

/// Python `_vec3(value, name)` (length check type-enforced in Rust).
fn vec3(value: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    let result = [f32q(value[0]), f32q(value[1]), f32q(value[2])];
    if !result.iter().all(|component| component.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    Ok(result)
}

/// Python `_sub(left, right)`.
fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        f32q(left[0] - right[0]),
        f32q(left[1] - right[1]),
        f32q(left[2] - right[2]),
    ]
}

/// Python `_scale(vector, scalar)`: `_f32(scalar)` then `_f32(c * scalar)`.
fn scale(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    let scalar = f32q(scalar);
    [
        f32q(vector[0] * scalar),
        f32q(vector[1] * scalar),
        f32q(vector[2] * scalar),
    ]
}

/// Python `_divide(vector, scalar)`: `_f32(scalar)` then `_f32(c / scalar)`.
fn divide(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    let scalar = f32q(scalar);
    [
        f32q(vector[0] / scalar),
        f32q(vector[1] / scalar),
        f32q(vector[2] / scalar),
    ]
}

/// Python `_dot(left, right)` (same accumulator shape as `dot3_f32`).
fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    dot3_f32(left, right)
}

/// `select_rail_query_context(contexts, kart_position)`.
pub fn select_rail_query_context(
    contexts: &[RailQueryContext],
    kart_position: [f64; 3],
) -> Result<RailQueryContextSelection, String> {
    let position = vec3(kart_position, "kart_position")?;

    let mut best_distance_squared = f32q(INITIAL_BEST_DISTANCE_SQUARED);
    let mut selected_context: Option<JsonValue> = None;
    let mut selected_detail: Option<RailQueryContextSelectionDetail> = None;
    let mut visited_segments: i64 = 0;

    for (context_index, context) in contexts.iter().enumerate() {
        let mut points = Vec::with_capacity(context.points.len());
        for (point_index, point) in context.points.iter().enumerate() {
            points.push(RailQueryContextPoint {
                position: vec3(
                    point.position,
                    &format!("context {context_index} point {point_index} position"),
                )?,
                normal: vec3(
                    point.normal,
                    &format!("context {context_index} point {point_index} normal"),
                )?,
            });
        }
        let context_id = context
            .context_id
            .clone()
            .unwrap_or(JsonValue::Number(context_index as f64));
        for segment_index in 0..points.len().saturating_sub(1) {
            visited_segments += 1;
            let start = &points[segment_index];
            let end = &points[segment_index + 1];
            let direction = sub(end.position, start.position);
            let end_to_kart = sub(end.position, position);
            let end_projection = dot(direction, end_to_kart);

            let (closest, normal, region): ([f64; 3], [f64; 3], &'static str);
            if end_projection < 0.0 {
                closest = end.position;
                normal = end.normal;
                region = "end_endpoint";
            } else {
                let start_to_kart = sub(start.position, position);
                if dot(direction, start_to_kart) < 0.0 {
                    closest = start.position;
                    normal = start.normal;
                    region = "start_endpoint";
                } else {
                    let direction_norm_squared = dot(direction, direction);
                    if direction_norm_squared == 0.0 {
                        // SSE division produces NaN here. The following COMISS
                        // is unordered and its JBE skips the candidate. Real
                        // factory_I02 contexts contain duplicate records, so
                        // preserving that non-throwing behavior is required.
                        continue;
                    }
                    let offset = divide(scale(direction, end_projection), direction_norm_squared);
                    closest = sub(end.position, offset);
                    // The executable uses the second point's +0x18 vector for
                    // an interior projection.
                    normal = end.normal;
                    region = "projected_line";
                }
            }

            let residual = sub(position, closest);
            let distance_squared = dot(residual, residual);
            if best_distance_squared <= distance_squared {
                continue;
            }

            let normal_dot = dot(residual, scale(normal, NORMAL_SCALE));
            let accepted = normal_dot > NORMAL_DOT_THRESHOLD;
            selected_context = if accepted { Some(context_id.clone()) } else { None };
            best_distance_squared = distance_squared;
            selected_detail = Some(RailQueryContextSelectionDetail {
                context_id: context_id.clone(),
                context_index: context_index as i64,
                segment_index: segment_index as i64,
                region,
                closest_point: closest,
                distance_squared,
                normal_dot,
                normal_predicate_passed: accepted,
            });
        }
    }

    Ok(RailQueryContextSelection {
        selected: selected_context.is_some(),
        selected_context,
        best_distance_squared,
        selected_detail,
        visited_contexts: contexts.len() as i64,
        visited_segments,
        authority: AUTHORITY,
        asset_boundary: ASSET_BOUNDARY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f64; 3], normal: [f64; 3]) -> RailQueryContextPoint {
        RailQueryContextPoint { position, normal }
    }

    #[test]
    fn selects_nearest_segment_with_normal_predicate() {
        // Constants computed with the Python authority:
        //   select_rail_query_context(
        //     contexts=[{"context_id": 7, "points": [
        //       {"position": [0,0,0], "normal": [0,1,0]},
        //       {"position": [10,0,0], "normal": [0,1,0]}]}],
        //     kart_position=[5.0, 0.5, 0.25])
        let contexts = vec![RailQueryContext {
            context_id: Some(JsonValue::Number(7.0)),
            points: vec![point([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), point(
                [10.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            )],
        }];
        let result = select_rail_query_context(&contexts, [5.0, 0.5, 0.25]).unwrap();
        // Python authority: end_projection >= 0 and
        // dot(direction, start - kart) < 0 select the "start_endpoint" arm
        // with closest = start.position, distance_squared 25.3125,
        // normal_dot 0.1 > -0.1 (accepted).
        assert!(result.selected);
        assert_eq!(result.selected_context, Some(JsonValue::Number(7.0)));
        assert_eq!(result.visited_contexts, 1);
        assert_eq!(result.visited_segments, 1);
        let detail = result.selected_detail.unwrap();
        assert_eq!(detail.region, "start_endpoint");
        assert_eq!(detail.closest_point, [0.0, 0.0, 0.0]);
        assert!(detail.normal_predicate_passed);
        assert_eq!(result.best_distance_squared, 25.3125);
    }

    #[test]
    fn normal_predicate_rejection_clears_selection() {
        // A kart far below the path: residual dot scaled normal is very
        // negative, so the nearest candidate clears the selection.
        let contexts = vec![RailQueryContext {
            context_id: None,
            points: vec![point([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), point(
                [10.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            )],
        }];
        let result = select_rail_query_context(&contexts, [5.0, -3.0, 0.0]).unwrap();
        assert!(!result.selected);
        assert_eq!(result.selected_context, None);
        // The detail is still recorded (best distance was updated).
        // Python authority: distance_squared 34.0, normal_dot -0.6.
        let detail = result.selected_detail.unwrap();
        assert!(!detail.normal_predicate_passed);
        assert_eq!(detail.normal_dot, -0.6000000238418579);
        assert_eq!(result.best_distance_squared, 34.0);
    }

    #[test]
    fn duplicate_records_skip_nan_segment() {
        // Zero-length segment: direction_norm_squared == 0.0 -> continue.
        let contexts = vec![RailQueryContext {
            context_id: Some(JsonValue::Number(3.0)),
            points: vec![
                point([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                point([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                point([10.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ],
        }];
        let result = select_rail_query_context(&contexts, [5.0, 0.5, 0.0]).unwrap();
        assert!(result.selected);
        assert_eq!(result.visited_segments, 2);
    }

    #[test]
    fn non_finite_point_is_an_error() {
        let contexts = vec![RailQueryContext {
            context_id: None,
            points: vec![point([f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0])],
        }];
        assert_eq!(
            select_rail_query_context(&contexts, [0.0; 3]).unwrap_err(),
            "context 0 point 0 position must be finite"
        );
    }
}
