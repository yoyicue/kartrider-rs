//! Port of `scripts/evaluate_rail_state1_query.py` — autonomous RAIL
//! state-1 query from registered path contexts and pose.
//!
//! Executes `A4EA20 -> A4F600 -> A4C990/A4C5C0` without replay fields.
//! `math.dist` routes through crate::num `cpython_dist3` (CPython
//! `vector_norm`, NOT `sqrt(sum())`); `math.sqrt(sum(...))` routes through
//! `sum_f64` (Neumaier).

use crate::num::{cpython_dist3, max2_f64, min2_f64, sum_f64};
use crate::state::JsonValue;

use super::rail_base_orientation::{evaluate_rail_base_orientation, RailBaseOrientation};
use super::rail_path_interpolation::{
    interpolate_rail_path_candidate, RailPathCandidate, RailPathInterpolation,
};
use super::rail_query_context_selector::{
    select_rail_query_context, RailQueryContext, RailQueryContextPoint, RailQueryContextSelection,
};
use super::rail_query_scalar::{
    evaluate_rail_query_scalar, RailQueryScalarRecord, RailQueryScalarResult,
};

/// Python result key `authority` (also the gate's `authority`).
pub const AUTHORITY: &str = "autonomous_A4EA20_A4F600_A4C990_equivalent_query";
/// Python gate key `rail_base_orientation_authority`.
pub const RAIL_BASE_ORIENTATION_AUTHORITY: &str = "B30150_one_metre_lookahead_interpolation";
/// Python synthesized retained-selection `authority`.
pub const RETAINED_AUTHORITY: &str = "retained_RAIL_constraint_owner_context";
/// Python velocity-transfer result key `authority`.
pub const VELOCITY_TRANSFER_AUTHORITY: &str =
    "autonomous_path_scalar_plus_one_metre_and_decoded_rail_min_velocity";

/// One decoded context point: Python resolves `position` (fallback
/// `primary`), `secondary` (fallback `direction`), `normal` (fallback
/// `normal_or_side`); the caller owns that key resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1Point {
    pub position: [f64; 3],
    pub secondary: [f64; 3],
    pub normal: [f64; 3],
}

/// One registered path context. `context_id` resolution mirrors Python
/// `context.get("context_id", context.get("context_index", context_index))`:
/// `context_id`, then `context_index`, then the enumeration index.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1Context {
    pub context_id: Option<JsonValue>,
    pub context_index: Option<i64>,
    /// Python `successor_ids` (fallback `successor_indices`).
    pub successor_ids: Option<Vec<JsonValue>>,
    pub road: Option<String>,
    pub total_length: Option<f64>,
    pub points: Vec<RailState1Point>,
}

/// Python `context_selection` payload: either the real `sub_A4EA20`
/// selection or the synthesized retained-owner selection.
#[derive(Debug, Clone, PartialEq)]
pub enum RailState1ContextSelection {
    Selector(Box<RailQueryContextSelection>),
    /// `{"selected": True, "selected_context": id, "authority":
    /// "retained_RAIL_constraint_owner_context", "visited_contexts": 1}`.
    Retained { selected_context: JsonValue },
}

impl RailState1ContextSelection {
    /// Python `selection["selected"]` / `selection["selected_context"]`.
    pub fn selected_context(&self) -> Option<&JsonValue> {
        match self {
            RailState1ContextSelection::Selector(selection) => {
                selection.selected_context.as_ref()
            }
            RailState1ContextSelection::Retained { selected_context } => Some(selected_context),
        }
    }

    /// Python `selection["selected"]`.
    pub fn selected(&self) -> bool {
        match self {
            RailState1ContextSelection::Selector(selection) => selection.selected,
            RailState1ContextSelection::Retained { .. } => true,
        }
    }
}

/// One `successor_discontinuities` entry.
#[derive(Debug, Clone, PartialEq)]
pub struct RailSuccessorDiscontinuity {
    pub context_id: JsonValue,
    pub endpoint_gap: f64,
}

/// Python `state1_gate` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1Gate {
    pub authority: &'static str,
    pub contact_exists: bool,
    pub contact_identifier_nonempty: bool,
    pub path_metric: f64,
    pub metric_threshold: f64,
    pub path_z: f64,
    pub kart_z_plus_one: f64,
    pub alignment_base_ms: f64,
    pub alignment_primary_scale_ms: Option<f64>,
    pub alignment_stable_initial_ms: Option<f64>,
    pub selected_context: JsonValue,
    pub query_scalar: f64,
    pub primary_endpoint: [f64; 3],
    pub lookahead_distance: f64,
    pub lookahead_endpoint: [f64; 3],
    pub lookahead_context: JsonValue,
    pub successor_authority: String,
    pub successor_discontinuities: Vec<RailSuccessorDiscontinuity>,
    pub advance_context: Option<JsonValue>,
    pub terminal_lookahead: bool,
    pub terminal_distance_remaining: f64,
    pub terminal_state6_edge: bool,
    pub secondary_endpoint: [f64; 3],
    pub tertiary_endpoint: Option<[f64; 3]>,
    pub lookahead_secondary_endpoint: [f64; 3],
    pub lookahead_tertiary_endpoint: Option<[f64; 3]>,
    pub rail_base_orientation: Option<RailBaseOrientation>,
    pub rail_base_orientation_authority: &'static str,
}

/// Python `evaluate_rail_state1_query(...)` result dict. Key presence rules:
/// `query_scalar` / `path_interpolation` are `None` only when the context
/// selection already failed (Python omits those keys there).
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1QueryResult {
    pub selected: bool,
    pub context_selection: RailState1ContextSelection,
    pub query_scalar: Option<RailQueryScalarResult>,
    pub path_interpolation: Option<RailPathInterpolation>,
    pub state1_gate: Option<RailState1Gate>,
    pub authority: &'static str,
}

/// Python `evaluate_rail_state1_velocity_transfer(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1VelocityTransfer {
    pub position: [f64; 3],
    pub lookahead_endpoint: [f64; 3],
    pub min_velocity: f64,
    pub max_velocity: f64,
    pub current_speed: f64,
    pub selected_speed: f64,
    pub velocity_after: [f64; 3],
    pub velocity_magnitude: f64,
    pub authority: &'static str,
}

/// `repr()` of a context id for the Python error message
/// `f"rail successor {candidate_id!r} is not registered"`.
fn py_repr_id(value: &JsonValue) -> String {
    match value {
        JsonValue::Number(number) => {
            if number.is_finite() && number.fract() == 0.0 && number.abs() < 9.0e15 {
                format!("{}", *number as i64)
            } else {
                format!("{number:?}")
            }
        }
        JsonValue::String(text) => format!("'{text}'"),
        other => format!("{other:?}"),
    }
}

/// `math.sqrt(sum(value * value for value in v))` — CPython Neumaier sum.
fn length3_sum(values: [f64; 3]) -> f64 {
    sum_f64(&[
        values[0] * values[0],
        values[1] * values[1],
        values[2] * values[2],
    ])
    .sqrt()
}

/// `evaluate_rail_state1_velocity_transfer(...)`. Vector length checks are
/// type-enforced in Rust.
pub fn evaluate_rail_state1_velocity_transfer(
    kart_position: [f64; 3],
    lookahead_endpoint: [f64; 3],
    min_velocity: f64,
    current_velocity: Option<[f64; 3]>,
    max_velocity: Option<f64>,
) -> Result<RailState1VelocityTransfer, String> {
    let position = kart_position;
    let endpoint = lookahead_endpoint;
    let minimum = min_velocity;
    if !position.iter().chain(endpoint.iter()).all(|value| value.is_finite()) {
        return Err("rail velocity transfer points must be finite".to_string());
    }
    if !minimum.is_finite() || minimum <= 0.0 {
        return Err("rail minimum velocity must be finite and positive".to_string());
    }
    let maximum = match max_velocity {
        None => minimum,
        Some(max_velocity) => max_velocity,
    };
    if !maximum.is_finite() || maximum < minimum {
        return Err("rail maximum velocity must be finite and at least the minimum".to_string());
    }
    let current_speed = match current_velocity {
        None => minimum,
        Some(velocity_before) => {
            if !velocity_before.iter().all(|value| value.is_finite()) {
                return Err("current rail velocity must be a finite 3D vector".to_string());
            }
            length3_sum(velocity_before)
        }
    };
    let speed = min2_f64(maximum, max2_f64(minimum, current_speed));
    let delta = [
        endpoint[0] - position[0],
        endpoint[1] - position[1],
        endpoint[2] - position[2],
    ];
    let length = length3_sum(delta);
    if length <= 1.0e-12 {
        return Err("rail lookahead endpoint equals kart position".to_string());
    }
    let velocity = [
        delta[0] / length * speed,
        delta[1] / length * speed,
        delta[2] / length * speed,
    ];
    Ok(RailState1VelocityTransfer {
        position,
        lookahead_endpoint: endpoint,
        min_velocity: minimum,
        max_velocity: maximum,
        current_speed,
        selected_speed: speed,
        velocity_after: velocity,
        velocity_magnitude: length3_sum(velocity),
        authority: VELOCITY_TRANSFER_AUTHORITY,
    })
}

/// `evaluate_rail_state1_query(...)`.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rail_state1_query(
    contexts: &[RailState1Context],
    kart_position: [f64; 3],
    contact_exists: bool,
    contact_identifier_nonempty: bool,
    metric_threshold: f64,
    height_bias: f64,
    alignment_base_ms: f64,
    alignment_primary_scale_ms: Option<f64>,
    alignment_stable_initial_ms: Option<f64>,
    retained_context_id: Option<JsonValue>,
) -> Result<RailState1QueryResult, String> {
    let position = kart_position;
    if !position.iter().all(|value| value.is_finite()) {
        return Err("kart_position must contain three finite components".to_string());
    }
    let threshold = metric_threshold;
    let bias = height_bias;
    if !threshold.is_finite() || threshold < 0.0 {
        return Err("metric_threshold must be finite and non-negative".to_string());
    }
    if !bias.is_finite() {
        return Err("height_bias must be finite".to_string());
    }

    // (context_id, context) pairs in Python dict insertion order.
    let mut normalized: Vec<RailQueryContext> = Vec::with_capacity(contexts.len());
    let mut by_id: Vec<(JsonValue, &RailState1Context)> = Vec::with_capacity(contexts.len());
    for (context_index, context) in contexts.iter().enumerate() {
        let context_id = context
            .context_id
            .clone()
            .or_else(|| context.context_index.map(|index| JsonValue::Number(index as f64)))
            .unwrap_or(JsonValue::Number(context_index as f64));
        normalized.push(RailQueryContext {
            context_id: Some(context_id.clone()),
            points: context
                .points
                .iter()
                .map(|point| RailQueryContextPoint {
                    position: point.position,
                    normal: point.normal,
                })
                .collect(),
        });
        by_id.push((context_id, context));
    }

    let selection = match &retained_context_id {
        None => RailState1ContextSelection::Selector(Box::new(select_rail_query_context(
            &normalized,
            position,
        )?)),
        Some(retained_context_id) => {
            if !by_id.iter().any(|(id, _)| id == retained_context_id) {
                return Err("retained rail context is not registered".to_string());
            }
            RailState1ContextSelection::Retained {
                selected_context: retained_context_id.clone(),
            }
        }
    };
    if !selection.selected() {
        return Ok(RailState1QueryResult {
            selected: false,
            context_selection: selection,
            query_scalar: None,
            path_interpolation: None,
            state1_gate: None,
            authority: AUTHORITY,
        });
    }

    let context_id = selection.selected_context().unwrap().clone();
    let context = by_id
        .iter()
        .find(|(id, _)| *id == context_id)
        .map(|(_, context)| *context)
        .unwrap();
    let points = &context.points;
    let scalar = evaluate_rail_query_scalar(
        &points
            .iter()
            .map(|point| RailQueryScalarRecord {
                position: point.position,
                direction: point.secondary,
            })
            .collect::<Vec<_>>(),
        position,
        context.total_length,
    )?;
    let interpolation = interpolate_rail_path_candidate(
        &points
            .iter()
            .map(|point| RailPathCandidate {
                primary: point.position,
                secondary: point.secondary,
                tertiary: Some(point.normal),
            })
            .collect::<Vec<_>>(),
        Some(scalar.query_scalar),
        None,
    )?;
    if !interpolation.accepted {
        return Ok(RailState1QueryResult {
            selected: false,
            context_selection: selection,
            query_scalar: Some(scalar),
            path_interpolation: Some(interpolation),
            state1_gate: None,
            authority: AUTHORITY,
        });
    }

    let primary = interpolation.primary.unwrap();
    let lookahead_distance = 1.0_f64;
    let mut lookahead_metric = scalar.query_scalar + lookahead_distance;
    let mut lookahead_context_id = context_id.clone();
    let mut advance_context_id: Option<JsonValue> = None;
    let mut lookahead_points: &[RailState1Point] = points;
    let mut terminal_lookahead = false;
    let mut successor_discontinuities: Vec<RailSuccessorDiscontinuity> = Vec::new();
    let mut successor_authority: String;
    if lookahead_metric > scalar.total_length {
        let endpoint = points[points.len() - 1].position;
        let mut successors: Vec<(JsonValue, &RailState1Context)> = Vec::new();
        if let Some(successor_ids) = &context.successor_ids {
            successor_authority = "context_plus_0x08_successor_vector".to_string();
            for candidate_id in successor_ids {
                let Some((_, candidate_context)) =
                    by_id.iter().find(|(id, _)| id == candidate_id)
                else {
                    return Err(format!(
                        "rail successor {} is not registered",
                        py_repr_id(candidate_id)
                    ));
                };
                let candidate_points = &candidate_context.points;
                let endpoint_gap = if candidate_points.is_empty() {
                    f64::INFINITY
                } else {
                    cpython_dist3(endpoint, candidate_points[0].position)
                };
                // +0x08 is an authored ownership graph, not proof that the
                // next context is a continuous interpolation segment. Only
                // zero-gap successors may own the one-metre cross-context
                // lookahead (see the Python module comment).
                if endpoint_gap <= 0.05 {
                    successors.push((candidate_id.clone(), candidate_context));
                } else {
                    successor_discontinuities.push(RailSuccessorDiscontinuity {
                        context_id: candidate_id.clone(),
                        endpoint_gap,
                    });
                }
            }
        } else {
            // Compatibility for the original v1 snapshot, which predates the
            // context+0x08 vector capture. New closure evidence must use v2.
            successor_authority = "legacy_endpoint_geometry_fallback".to_string();
            for (candidate_id, candidate_context) in &by_id {
                if *candidate_id == context_id || candidate_context.points.is_empty() {
                    continue;
                }
                if candidate_context.road != context.road {
                    continue;
                }
                if cpython_dist3(endpoint, candidate_context.points[0].position) <= 0.05 {
                    successors.push((candidate_id.clone(), candidate_context));
                }
            }
        }
        if successors.len() > 1 {
            let current_index = context.context_index;
            let ordered: Vec<(JsonValue, &RailState1Context)> = successors
                .iter()
                .filter(|(_, candidate_context)| {
                    current_index.is_some()
                        && candidate_context.context_index == current_index.map(|index| index + 1)
                })
                .cloned()
                .collect();
            if ordered.len() == 1 {
                successors = ordered;
            }
        }
        if successors.len() == 1 {
            lookahead_context_id = successors[0].0.clone();
            lookahead_points = successors[0].1.points.as_slice();
            lookahead_metric -= scalar.total_length;
            if scalar.total_length - scalar.query_scalar <= 1.0e-3 {
                advance_context_id = Some(lookahead_context_id.clone());
            }
        } else {
            lookahead_metric = scalar.total_length;
            terminal_lookahead = true;
            successor_authority = if context.successor_ids.is_some() {
                "context_plus_0x08_successor_vector_terminal".to_string()
            } else {
                "legacy_endpoint_geometry_fallback_terminal".to_string()
            };
        }
    } else {
        successor_authority = "same_context_no_successor_lookup".to_string();
    }
    let lookahead = interpolate_rail_path_candidate(
        &lookahead_points
            .iter()
            .map(|point| RailPathCandidate {
                primary: point.position,
                secondary: point.secondary,
                tertiary: Some(point.normal),
            })
            .collect::<Vec<_>>(),
        Some(lookahead_metric),
        None,
    )?;
    if !lookahead.accepted {
        // Python raises RuntimeError here (not ValueError); the message is
        // the parity contract.
        return Err(
            "selected rail context rejected its clamped one-metre lookahead".to_string()
        );
    }
    let path_metric = cpython_dist3(position, primary);
    let tertiary = interpolation.tertiary;
    let lookahead_tertiary = lookahead.tertiary;
    let rail_base_orientation = match lookahead_tertiary {
        Some(lookahead_tertiary) => Some(evaluate_rail_base_orientation(
            lookahead.secondary.unwrap(),
            lookahead_tertiary,
            true,
        )?),
        None => None,
    };
    let gate = RailState1Gate {
        authority: AUTHORITY,
        contact_exists,
        contact_identifier_nonempty,
        path_metric,
        metric_threshold: threshold,
        path_z: primary[2],
        kart_z_plus_one: position[2] + bias,
        alignment_base_ms,
        alignment_primary_scale_ms,
        alignment_stable_initial_ms,
        selected_context: context_id,
        query_scalar: scalar.query_scalar,
        primary_endpoint: primary,
        lookahead_distance,
        lookahead_endpoint: lookahead.primary.unwrap(),
        lookahead_context: lookahead_context_id,
        successor_authority,
        successor_discontinuities,
        advance_context: advance_context_id,
        terminal_lookahead,
        terminal_distance_remaining: max2_f64(
            0.0,
            scalar.total_length - scalar.query_scalar,
        ),
        terminal_state6_edge: terminal_lookahead
            && (scalar.total_length - scalar.query_scalar <= lookahead_distance + 1.0e-3),
        secondary_endpoint: interpolation.secondary.unwrap(),
        tertiary_endpoint: tertiary,
        lookahead_secondary_endpoint: lookahead.secondary.unwrap(),
        lookahead_tertiary_endpoint: lookahead_tertiary,
        rail_base_orientation,
        rail_base_orientation_authority: RAIL_BASE_ORIENTATION_AUTHORITY,
    };
    Ok(RailState1QueryResult {
        selected: true,
        context_selection: selection,
        query_scalar: Some(scalar),
        path_interpolation: Some(interpolation),
        state1_gate: Some(gate),
        authority: AUTHORITY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(position: [f64; 3], secondary: [f64; 3], normal: [f64; 3]) -> RailState1Point {
        RailState1Point { position, secondary, normal }
    }

    fn straight_context(id: f64) -> RailState1Context {
        RailState1Context {
            context_id: Some(JsonValue::Number(id)),
            context_index: None,
            successor_ids: None,
            road: None,
            total_length: None,
            points: vec![
                point([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                point([10.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                point([20.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ],
        }
    }

    #[test]
    fn selects_and_builds_gate() {
        // Python authority spot values for this geometry (see the parity
        // harness for the full bit-exact sweep).
        let result = evaluate_rail_state1_query(
            &[straight_context(7.0)],
            [5.0, 0.1, 0.2],
            true,
            true,
            2.0,
            1.0,
            0.0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(result.selected);
        let gate = result.state1_gate.unwrap();
        assert_eq!(gate.selected_context, JsonValue::Number(7.0));
        assert_eq!(gate.query_scalar, 5.0);
        assert_eq!(gate.primary_endpoint, [5.0, 0.0, 0.0]);
        assert_eq!(gate.lookahead_endpoint, [6.0, 0.0, 0.0]);
        assert_eq!(gate.lookahead_distance, 1.0);
        assert_eq!(gate.successor_authority, "same_context_no_successor_lookup");
        assert!(!gate.terminal_lookahead);
        assert!(gate.rail_base_orientation.is_some());
        // path_metric = math.dist([5,0.1,0.2], [5,0,0]) = hypot-like 0.2236...
        assert!((gate.path_metric - 0.22360679774997897).abs() < 1e-15);
    }

    #[test]
    fn unselected_when_normal_predicate_fails() {
        // Kart 50 m below the path (z): normal_dot = -10 < -0.1 rejects the
        // only context. (A y-offset is NOT rejected here: the authored
        // normal is [0,0,1], so a y residual has a zero normal dot.)
        let result = evaluate_rail_state1_query(
            &[straight_context(7.0)],
            [5.0, 0.0, -50.0],
            false,
            false,
            2.0,
            1.0,
            0.0,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(!result.selected);
        assert!(result.query_scalar.is_none());
        assert!(result.path_interpolation.is_none());
        assert!(result.state1_gate.is_none());
    }

    #[test]
    fn retained_context_short_circuits_selector() {
        let result = evaluate_rail_state1_query(
            &[straight_context(9.0)],
            [5.0, 0.0, 0.0],
            true,
            true,
            2.0,
            1.0,
            0.0,
            None,
            None,
            Some(JsonValue::Number(9.0)),
        )
        .unwrap();
        assert!(result.selected);
        assert!(matches!(
            result.context_selection,
            RailState1ContextSelection::Retained { .. }
        ));
        assert_eq!(
            evaluate_rail_state1_query(
                &[straight_context(9.0)],
                [5.0, 0.0, 0.0],
                true,
                true,
                2.0,
                1.0,
                0.0,
                None,
                None,
                Some(JsonValue::Number(4.0)),
            )
            .unwrap_err(),
            "retained rail context is not registered"
        );
    }

    #[test]
    fn terminal_lookahead_without_successor() {
        // Query near the end of the only context: lookahead clamps to the
        // total length and the state6 terminal edge arm fires.
        let result = evaluate_rail_state1_query(
            &[straight_context(7.0)],
            [19.5, 0.0, 0.0],
            true,
            true,
            2.0,
            1.0,
            0.0,
            None,
            None,
            None,
        )
        .unwrap();
        let gate = result.state1_gate.unwrap();
        assert!(gate.terminal_lookahead);
        assert_eq!(gate.successor_authority, "legacy_endpoint_geometry_fallback_terminal");
        assert_eq!(gate.lookahead_endpoint, [20.0, 0.0, 0.0]);
        assert!(gate.terminal_state6_edge);
        assert_eq!(gate.terminal_distance_remaining, 0.5);
    }

    #[test]
    fn velocity_transfer_matches_python_authority() {
        // Python authority:
        //   evaluate_rail_state1_velocity_transfer([0,0,0], [3,0,4], 10.0)
        let result = evaluate_rail_state1_velocity_transfer(
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 4.0],
            10.0,
            None,
            None,
        )
        .unwrap();
        assert_eq!(result.selected_speed, 10.0);
        assert_eq!(result.velocity_after, [6.0, 0.0, 8.0]);
        assert_eq!(result.velocity_magnitude, 10.0);
        assert_eq!(
            evaluate_rail_state1_velocity_transfer([0.0; 3], [0.0; 3], 10.0, None, None)
                .unwrap_err(),
            "rail lookahead endpoint equals kart position"
        );
        assert_eq!(
            evaluate_rail_state1_velocity_transfer([0.0; 3], [1.0, 0.0, 0.0], 0.0, None, None)
                .unwrap_err(),
            "rail minimum velocity must be finite and positive"
        );
    }
}
