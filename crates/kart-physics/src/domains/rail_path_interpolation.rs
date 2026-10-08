//! Port of `scripts/evaluate_rail_path_interpolation.py` — executable exact
//! interpolation body recovered from `sub_A4C5C0`.
//!
//! Pure f64 arithmetic; every `math.sqrt(sum(...))` routes through
//! crate::num `sum_f64` (CPython Neumaier `sum()`).

use crate::num::{max2_f64, min2_f64, sum_f64};

/// Python result keys `selection_policy` / `closure_status`.
pub const SELECTION_POLICY_ACCEPTED: &str = "earliest_inclusive_accumulated_primary_segment";
pub const SELECTION_POLICY_REJECTED: &str = "no_inclusive_candidate_interval";
pub const CLOSURE_STATUS: &str = "exact_sub_A4C5C0_interpolation_ordered_candidates";

/// One decoded candidate: the recovered 36-byte vector payload (`primary`
/// at +0x00, `secondary` at +0x0C, optional `tertiary` at +0x18). Components
/// are `float()`-converted and finite-checked (Python `_vector3`).
#[derive(Debug, Clone, PartialEq)]
pub struct RailPathCandidate {
    pub primary: [f64; 3],
    pub secondary: [f64; 3],
    pub tertiary: Option<[f64; 3]>,
}

/// Python `interpolate_rail_path_candidate(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailPathInterpolation {
    pub accepted: bool,
    pub segment_index: Option<i64>,
    pub accumulated_start: Option<f64>,
    pub accumulated_end: Option<f64>,
    pub segment_length: Option<f64>,
    pub query_metric: f64,
    pub fraction: Option<f64>,
    pub primary: Option<[f64; 3]>,
    pub secondary: Option<[f64; 3]>,
    pub tertiary: Option<[f64; 3]>,
    /// Rejection-only key: final accumulated length.
    pub total_length: Option<f64>,
    pub selection_policy: &'static str,
    pub closure_status: &'static str,
}

/// Python `_vector3(value, name)`: `float()` per component + finite check.
/// The three-component length check is type-enforced in Rust.
fn vector3(value: [f64; 3], name: &str) -> Result<[f64; 3], String> {
    if !value.iter().all(|component| component.is_finite()) {
        return Err(format!("{name} must be finite"));
    }
    Ok(value)
}

/// Python `_lerp(a, b, t)`.
fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// `math.sqrt(sum(component * component ...))` — CPython `sum()` is
/// Neumaier-compensated (see crate::num::sum_f64), NOT a naive fold.
fn length3_sum(values: [f64; 3]) -> f64 {
    sum_f64(&[
        values[0] * values[0],
        values[1] * values[1],
        values[2] * values[2],
    ])
    .sqrt()
}

/// `interpolate_rail_path_candidate(candidates, query_metric=..., query_vector=...)`.
pub fn interpolate_rail_path_candidate(
    candidates: &[RailPathCandidate],
    query_metric: Option<f64>,
    query_vector: Option<[f64; 3]>,
) -> Result<RailPathInterpolation, String> {
    if query_metric.is_some() && query_vector.is_some() {
        return Err("provide query_metric or query_vector, not both".to_string());
    }
    let query_metric = match query_metric {
        None => {
            let Some(query_vector) = query_vector else {
                return Err("query_metric is required".to_string());
            };
            let query = vector3(query_vector, "query_vector")?;
            length3_sum(query)
        }
        Some(query_metric) => {
            if !query_metric.is_finite() || query_metric < 0.0 {
                return Err("query_metric must be finite and non-negative".to_string());
            }
            query_metric
        }
    };
    if candidates.len() < 2 {
        return Err("rail interpolation requires at least two candidates".to_string());
    }
    let mut decoded = Vec::with_capacity(candidates.len());
    let mut tertiary_presence = Vec::with_capacity(candidates.len());
    for (index, candidate) in candidates.iter().enumerate() {
        let primary = vector3(candidate.primary, &format!("candidate {index} primary"))?;
        let secondary = vector3(candidate.secondary, &format!("candidate {index} secondary"))?;
        tertiary_presence.push(candidate.tertiary.is_some());
        decoded.push(RailPathCandidate {
            primary,
            secondary,
            tertiary: match candidate.tertiary {
                Some(tertiary) => {
                    Some(vector3(tertiary, &format!("candidate {index} tertiary"))?)
                }
                None => None,
            },
        });
    }
    if tertiary_presence.iter().any(|&present| present)
        && !tertiary_presence.iter().all(|&present| present)
    {
        return Err("tertiary vector must be present for every candidate or none".to_string());
    }

    let mut accumulated_start = 0.0_f64;
    for index in 0..decoded.len() - 1 {
        let start = &decoded[index];
        let end = &decoded[index + 1];
        let segment_length = length3_sum([
            end.primary[0] - start.primary[0],
            end.primary[1] - start.primary[1],
            end.primary[2] - start.primary[2],
        ]);
        let accumulated_end = accumulated_start + segment_length;
        // A4F600 clamps against context+0x2C accumulated in float32, while
        // this interpolator reconstructs segment totals in Python double.
        // The mismatch grows with cumulative length, not only with the final
        // segment length (context29: 2.47e-5 m at 42.03 m).
        let endpoint_tolerance = max2_f64(1.0e-6, accumulated_end * 1.0e-6);
        if segment_length > 0.0
            && query_metric >= accumulated_start
            && query_metric <= accumulated_end + endpoint_tolerance
        {
            let bounded_metric = min2_f64(query_metric, accumulated_end);
            let fraction = (bounded_metric - accumulated_start) / segment_length;
            return Ok(RailPathInterpolation {
                accepted: true,
                segment_index: Some(index as i64),
                accumulated_start: Some(accumulated_start),
                accumulated_end: Some(accumulated_end),
                segment_length: Some(segment_length),
                query_metric,
                fraction: Some(fraction),
                primary: Some(lerp(start.primary, end.primary, fraction)),
                secondary: Some(lerp(start.secondary, end.secondary, fraction)),
                tertiary: start
                    .tertiary
                    .map(|tertiary| lerp(tertiary, end.tertiary.unwrap(), fraction)),
                total_length: None,
                selection_policy: SELECTION_POLICY_ACCEPTED,
                closure_status: CLOSURE_STATUS,
            });
        }
        accumulated_start = accumulated_end;
    }
    Ok(RailPathInterpolation {
        accepted: false,
        segment_index: None,
        accumulated_start: None,
        accumulated_end: None,
        segment_length: None,
        query_metric,
        fraction: None,
        primary: None,
        secondary: None,
        tertiary: None,
        total_length: Some(accumulated_start),
        selection_policy: SELECTION_POLICY_REJECTED,
        closure_status: CLOSURE_STATUS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(
        primary: [f64; 3],
        secondary: [f64; 3],
        tertiary: Option<[f64; 3]>,
    ) -> RailPathCandidate {
        RailPathCandidate { primary, secondary, tertiary }
    }

    #[test]
    fn argument_validation() {
        let candidates = vec![
            candidate([0.0; 3], [1.0, 0.0, 0.0], None),
            candidate([10.0, 0.0, 0.0], [1.0, 0.0, 0.0], None),
        ];
        assert_eq!(
            interpolate_rail_path_candidate(&candidates, Some(1.0), Some([0.0; 3])).unwrap_err(),
            "provide query_metric or query_vector, not both"
        );
        assert_eq!(
            interpolate_rail_path_candidate(&candidates, None, None).unwrap_err(),
            "query_metric is required"
        );
        assert_eq!(
            interpolate_rail_path_candidate(&candidates, Some(-1.0), None).unwrap_err(),
            "query_metric must be finite and non-negative"
        );
        assert_eq!(
            interpolate_rail_path_candidate(&candidates[..1], Some(1.0), None).unwrap_err(),
            "rail interpolation requires at least two candidates"
        );
        let mixed = vec![
            candidate([0.0; 3], [1.0, 0.0, 0.0], Some([0.0, 0.0, 1.0])),
            candidate([10.0, 0.0, 0.0], [1.0, 0.0, 0.0], None),
        ];
        assert_eq!(
            interpolate_rail_path_candidate(&mixed, Some(1.0), None).unwrap_err(),
            "tertiary vector must be present for every candidate or none"
        );
    }

    #[test]
    fn midpoint_interpolation_matches_python_authority() {
        // Python authority:
        //   interpolate_rail_path_candidate(
        //     candidates=[{"primary": [0,0,0], "secondary": [1,0,0]},
        //                 {"primary": [10,0,0], "secondary": [0,1,0]}],
        //     query_metric=2.5)
        let candidates = vec![
            candidate([0.0; 3], [1.0, 0.0, 0.0], None),
            candidate([10.0, 0.0, 0.0], [0.0, 1.0, 0.0], None),
        ];
        let result = interpolate_rail_path_candidate(&candidates, Some(2.5), None).unwrap();
        assert!(result.accepted);
        assert_eq!(result.segment_index, Some(0));
        assert_eq!(result.fraction, Some(0.25));
        assert_eq!(result.primary, Some([2.5, 0.0, 0.0]));
        assert_eq!(result.secondary, Some([0.75, 0.25, 0.0]));
        // Past the end: rejected with the total length.
        let rejected =
            interpolate_rail_path_candidate(&candidates, Some(42.0), None).unwrap();
        assert!(!rejected.accepted);
        assert_eq!(rejected.total_length, Some(10.0));
        // Endpoint tolerance admits a query marginally past the end.
        let edge =
            interpolate_rail_path_candidate(&candidates, Some(10.0 + 5e-7), None).unwrap();
        assert!(edge.accepted);
        assert_eq!(edge.fraction, Some(1.0));
    }
}
