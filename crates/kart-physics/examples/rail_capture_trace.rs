//! RAIL capture parity trace tool (leaf modules + capture selector machinery).
//!
//! Reads a JSON spec from stdin: `{"cases": [...]}`. Case shapes:
//! - `{"kind": "leaf", "name": ..., "function": <leaf name>, "args": {...}}`
//!   drives one RAIL leaf function directly.
//! - `{"kind": "engine", "name": ..., "initial_state": ..., "parameters":
//!   ..., "numeric_mode": ..., "steps": [...]}` drives the engine capture
//!   machinery. Step shapes:
//!   - `{"kind": "step", "keys": [...], "world": <null|scripted>}` runs
//!     `step_raw_topology` (rail sampling/latch/capture paths).
//!   - `{"kind": "advance_selector", "args": {...}}` runs
//!     `advance_rail_capture_selector`.
//!   - `{"kind": "inspect", "world": <scripted>}` runs
//!     `inspect_autonomous_rail_capture` (no mutation).
//!   - `{"kind": "acquire", "keys": [...], "world": <scripted>}` runs
//!     `acquire_autonomous_rail` with `support=None` (rail entry through
//!     `step_raw_rail`; the part-2 stepper error is expected AFTER entry).
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! method name and bit-exact arguments, and returns the recorded response
//! (same contract as raw_frame_trace.rs). Every f64 is emitted as its
//! IEEE-754 bit pattern (`{:016x}`); a domain `Err` is emitted as
//! `{"error": <message>}` and the row still carries the post-call state.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;

use kart_physics::domains::rail_base_orientation::{
    evaluate_rail_base_orientation, RailBaseOrientation,
};
use kart_physics::domains::rail_capture_selector::{
    evaluate_rail_capture_selector, RailCaptureSelectorArgs, RailCaptureSelectorGates,
    RailCaptureSelectorResult, RailCaptureSelectorRuntimeAfter,
};
use kart_physics::domains::rail_constraint_orientation::{
    evaluate_rail_constraint_orientation, RailConstraintOrientation,
};
use kart_physics::domains::rail_path_interpolation::{
    interpolate_rail_path_candidate, RailPathCandidate, RailPathInterpolation,
};
use kart_physics::domains::rail_query_context_selector::{
    select_rail_query_context, RailQueryContext, RailQueryContextPoint,
    RailQueryContextSelection,
};
use kart_physics::domains::rail_query_scalar::{
    evaluate_rail_query_scalar, RailQueryScalarRecord, RailQueryScalarResult,
};
use kart_physics::domains::rail_state1_query::{
    evaluate_rail_state1_query, evaluate_rail_state1_velocity_transfer, RailState1Context,
    RailState1ContextSelection, RailState1Point, RailState1QueryResult, RailState1VelocityTransfer,
};
use kart_physics::domains::rail_topology::{evaluate_rail_step, RailStepArgs, RailStepResult};
use kart_physics::engine::{
    AdvanceRailCaptureSelectorArgs, KartRiderVehicleEngine, NumericMode, RailCaptureInspection,
    RailState1GroundWriter,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{JsonValue, VehicleState};
use kart_physics::topology::{TopologyState};
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Bit-pattern encoding helpers (same conventions as raw_frame_trace.rs)
// ---------------------------------------------------------------------------

fn fbits(value: f64) -> Value {
    Value::String(format!("{:016x}", value.to_bits()))
}

/// Walk a serialized value, replacing every JSON float with its f64 bit
/// pattern. JSON integers (Python `int` fields) pass through untouched.
fn bits_json(value: &Value) -> Value {
    match value {
        Value::Number(number) => {
            if number.is_f64() {
                fbits(number.as_f64().expect("is_f64 implies as_f64"))
            } else {
                value.clone()
            }
        }
        Value::Array(items) => Value::Array(items.iter().map(bits_json).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), bits_json(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn arr_fbits(values: [f64; 3]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn arr9_fbits(values: [f64; 9]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn opt_arr_fbits(values: Option<[f64; 3]>) -> Value {
    values.map_or(Value::Null, arr_fbits)
}

fn opt_fbits(value: Option<f64>) -> Value {
    value.map_or(Value::Null, fbits)
}

/// `JsonValue` numbers are f64, but Python context ids are plain `int`s
/// (JSON integers) — integral numbers are emitted as JSON integers so the
/// Python `int` and the Rust `JsonValue::Number` compare equal. Numbers
/// with a fraction keep the f64 bit-pattern string.
fn json_value_encoded(value: &JsonValue) -> Value {
    match value {
        JsonValue::Number(number)
            if number.is_finite() && number.fract() == 0.0 && number.abs() < 9.0e15 =>
        {
            json!(*number as i64)
        }
        _ => bits_json(&serde_json::to_value(value).expect("JsonValue serializes")),
    }
}

// ---------------------------------------------------------------------------
// Leaf result encoders (mirror the Python verify script exactly)
// ---------------------------------------------------------------------------

fn encode_scalar_result(result: &RailQueryScalarResult) -> Value {
    json!({
        "query_scalar": fbits(result.query_scalar),
        "unclamped_query_scalar": fbits(result.unclamped_query_scalar),
        "total_length": fbits(result.total_length),
        "selected_record_index": result.selected_record_index,
        "accumulated_before_record": fbits(result.accumulated_before_record),
        "local_projection": fbits(result.local_projection),
        "local_direction": arr_fbits(result.local_direction),
        "local_direction_source": result.local_direction_source,
        "authority": result.authority,
    })
}

fn encode_context_selection(selection: &RailQueryContextSelection) -> Value {
    json!({
        "selected": selection.selected,
        "selected_context": selection.selected_context.as_ref().map(json_value_encoded).unwrap_or(Value::Null),
        "best_distance_squared": fbits(selection.best_distance_squared),
        "selected_detail": selection.selected_detail.as_ref().map(|detail| json!({
            "context_id": json_value_encoded(&detail.context_id),
            "context_index": detail.context_index,
            "segment_index": detail.segment_index,
            "region": detail.region,
            "closest_point": arr_fbits(detail.closest_point),
            "distance_squared": fbits(detail.distance_squared),
            "normal_dot": fbits(detail.normal_dot),
            "normal_predicate_passed": detail.normal_predicate_passed,
        })).unwrap_or(Value::Null),
        "visited_contexts": selection.visited_contexts,
        "visited_segments": selection.visited_segments,
        "authority": selection.authority,
        "asset_boundary": selection.asset_boundary,
    })
}

fn encode_interpolation(result: &RailPathInterpolation) -> Value {
    if result.accepted {
        let mut row = Map::new();
        row.insert("accepted".to_string(), json!(true));
        row.insert("segment_index".to_string(), json!(result.segment_index.unwrap()));
        row.insert("accumulated_start".to_string(), fbits(result.accumulated_start.unwrap()));
        row.insert("accumulated_end".to_string(), fbits(result.accumulated_end.unwrap()));
        row.insert("segment_length".to_string(), fbits(result.segment_length.unwrap()));
        row.insert("query_metric".to_string(), fbits(result.query_metric));
        row.insert("fraction".to_string(), fbits(result.fraction.unwrap()));
        row.insert("primary".to_string(), arr_fbits(result.primary.unwrap()));
        row.insert("secondary".to_string(), arr_fbits(result.secondary.unwrap()));
        row.insert("selection_policy".to_string(), json!(result.selection_policy));
        row.insert("closure_status".to_string(), json!(result.closure_status));
        if let Some(tertiary) = result.tertiary {
            row.insert("tertiary".to_string(), arr_fbits(tertiary));
        }
        Value::Object(row)
    } else {
        json!({
            "accepted": false,
            "query_metric": fbits(result.query_metric),
            "total_length": fbits(result.total_length.unwrap()),
            "selection_policy": result.selection_policy,
            "closure_status": result.closure_status,
        })
    }
}

fn encode_base_orientation(result: &RailBaseOrientation) -> Value {
    json!({
        "matrix_bf8": arr9_fbits(result.matrix_bf8),
        "column0": arr_fbits(result.column0),
        "column1": arr_fbits(result.column1),
        "column2": arr_fbits(result.column2),
        "secondary_endpoint": arr_fbits(result.secondary_endpoint),
        "tertiary_endpoint": arr_fbits(result.tertiary_endpoint),
        "runtime_float32_exact": result.runtime_float32_exact,
        "formula": {
            "column1": kart_physics::domains::rail_base_orientation::FORMULA_COLUMN1,
            "column0": kart_physics::domains::rail_base_orientation::FORMULA_COLUMN0,
            "column2": kart_physics::domains::rail_base_orientation::FORMULA_COLUMN2,
        },
        "closure_status": result.closure_status,
    })
}

fn quat_fbits(values: [f64; 4]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn opt_quat_fbits(values: Option<[f64; 4]>) -> Value {
    values.map_or(Value::Null, quat_fbits)
}

/// Projected `evaluate_orientation_integrator` result (the parity harness
/// compares this projection on both sides).
fn encode_orientation_integrator_projected(
    result: &kart_physics::domains::orientation_integrator::OrientationIntegratorResult,
) -> Value {
    json!({
        "quaternion_after": quat_fbits(result.quaternion_after),
        "matrix_9c_after": arr9_fbits(result.matrix_9c_after),
        "axis2_z_after": fbits(result.axis2_z_after),
    })
}

fn encode_constraint_orientation(result: &RailConstraintOrientation) -> Value {
    let mut row = Map::new();
    row.insert("primary_matrix_after".to_string(), arr9_fbits(result.primary_matrix_after));
    row.insert("rail_base_bf8_after".to_string(), arr9_fbits(result.rail_base_bf8_after));
    row.insert(
        "rail_constraint_c1c_after".to_string(),
        arr9_fbits(result.rail_constraint_c1c_after),
    );
    if let Some(ordinary) = &result.orientation_integrator {
        row.insert(
            "orientation_integrator".to_string(),
            encode_orientation_integrator_projected(ordinary),
        );
    }
    if result.constraint_quaternion_before.is_some() {
        row.insert(
            "constraint_quaternion_before".to_string(),
            opt_quat_fbits(result.constraint_quaternion_before),
        );
        row.insert(
            "constraint_quaternion_relaxed".to_string(),
            opt_quat_fbits(result.constraint_quaternion_relaxed),
        );
        row.insert(
            "constraint_quaternion_after".to_string(),
            opt_quat_fbits(result.constraint_quaternion_after),
        );
        row.insert("relaxation_amount".to_string(), opt_fbits(result.relaxation_amount));
    }
    row.insert("writer".to_string(), json!(result.writer));
    row.insert("static_order".to_string(), json!(result.static_order));
    Value::Object(row)
}

fn encode_capture_selector_result(result: &RailCaptureSelectorResult) -> Value {
    let mut row = Map::new();
    row.insert("state_before".to_string(), json!(result.state_before));
    row.insert("state_after".to_string(), json!(result.state_after));
    row.insert("transitioned".to_string(), json!(result.transitioned));
    row.insert("reason".to_string(), json!(result.reason));
    if let Some(elapsed) = result.elapsed_in_state_ms {
        row.insert("elapsed_in_state_ms".to_string(), fbits(elapsed));
    }
    if let Some(delay) = result.state1_contact_delay_ms {
        row.insert("state1_contact_delay_ms".to_string(), fbits(delay));
    }
    if let Some(timeout) = result.state1_timeout_ms {
        row.insert("state1_timeout_ms".to_string(), fbits(timeout));
    }
    if let Some(exit_ms) = result.state5_exit_ms {
        row.insert("state5_exit_ms".to_string(), fbits(exit_ms));
    }
    match &result.gates {
        Some(RailCaptureSelectorGates::State1(gates)) => {
            row.insert("gates".to_string(), json!({
                "contact_delay_pass": gates.contact_delay_pass,
                "contact_exists": gates.contact_exists,
                "contact_identifier_nonempty": gates.contact_identifier_nonempty,
                "metric_pass": gates.metric_pass,
                "height_pass": gates.height_pass,
            }));
        }
        Some(RailCaptureSelectorGates::State2(gates)) => {
            row.insert("gates".to_string(), json!({
                "alignment_ratio": fbits(gates.alignment_ratio),
                "forward_speed": fbits(gates.forward_speed),
                "lateral_speed_abs": fbits(gates.lateral_speed_abs),
                "alignment_pass": gates.alignment_pass,
                "primary_timer_strictly_elapsed": gates.primary_timer_strictly_elapsed,
                "stable_timer_strictly_elapsed": gates.stable_timer_strictly_elapsed,
            }));
        }
        Some(RailCaptureSelectorGates::State3(gates)) => {
            row.insert("gates".to_string(), json!({
                "trigger_cooldown_zero": gates.trigger_cooldown_zero,
                "authored_trigger_accepted": gates.authored_trigger_accepted,
            }));
        }
        Some(RailCaptureSelectorGates::State6(gates)) => {
            row.insert("gates".to_string(), json!({
                "contact_candidate_valid": gates.contact_candidate_valid,
                "contact_timer_strictly_elapsed": gates.contact_timer_strictly_elapsed,
            }));
        }
        None => {}
    }
    match &result.runtime_after {
        Some(RailCaptureSelectorRuntimeAfter::State2(runtime)) => {
            row.insert("runtime_after".to_string(), json!({
                "alignment_primary_remaining_ms": fbits(runtime.alignment_primary_remaining_ms),
                "alignment_stable_remaining_ms": fbits(runtime.alignment_stable_remaining_ms),
            }));
        }
        Some(RailCaptureSelectorRuntimeAfter::State3(runtime)) => {
            row.insert("runtime_after".to_string(), json!({
                "trigger_cooldown_remaining_ms": fbits(runtime.trigger_cooldown_remaining_ms),
            }));
        }
        Some(RailCaptureSelectorRuntimeAfter::State6(runtime)) => {
            row.insert("runtime_after".to_string(), json!({
                "state6_contact_remaining_ms": fbits(runtime.state6_contact_remaining_ms),
                "state6_contact_raw_after_ms": fbits(runtime.state6_contact_raw_after_ms),
            }));
        }
        None => {}
    }
    row.insert("closure_status".to_string(), json!(result.closure_status));
    Value::Object(row)
}

fn encode_state1_query_result(result: &RailState1QueryResult) -> Value {
    let mut row = Map::new();
    row.insert("selected".to_string(), json!(result.selected));
    let context_selection = match &result.context_selection {
        RailState1ContextSelection::Selector(selection) => encode_context_selection(selection),
        RailState1ContextSelection::Retained { selected_context } => json!({
            "selected": true,
            "selected_context": json_value_encoded(selected_context),
            "authority": kart_physics::domains::rail_state1_query::RETAINED_AUTHORITY,
            "visited_contexts": 1,
        }),
    };
    row.insert("context_selection".to_string(), context_selection);
    if let Some(scalar) = &result.query_scalar {
        row.insert("query_scalar".to_string(), encode_scalar_result(scalar));
    }
    if let Some(interpolation) = &result.path_interpolation {
        row.insert("path_interpolation".to_string(), encode_interpolation(interpolation));
    }
    match &result.state1_gate {
        None => {
            row.insert("state1_gate".to_string(), Value::Null);
        }
        Some(gate) => {
            row.insert("state1_gate".to_string(), json!({
                "authority": gate.authority,
                "contact_exists": gate.contact_exists,
                "contact_identifier_nonempty": gate.contact_identifier_nonempty,
                "path_metric": fbits(gate.path_metric),
                "metric_threshold": fbits(gate.metric_threshold),
                "path_z": fbits(gate.path_z),
                "kart_z_plus_one": fbits(gate.kart_z_plus_one),
                "alignment_base_ms": fbits(gate.alignment_base_ms),
                "alignment_primary_scale_ms": opt_fbits(gate.alignment_primary_scale_ms),
                "alignment_stable_initial_ms": opt_fbits(gate.alignment_stable_initial_ms),
                "selected_context": json_value_encoded(&gate.selected_context),
                "query_scalar": fbits(gate.query_scalar),
                "primary_endpoint": arr_fbits(gate.primary_endpoint),
                "lookahead_distance": fbits(gate.lookahead_distance),
                "lookahead_endpoint": arr_fbits(gate.lookahead_endpoint),
                "lookahead_context": json_value_encoded(&gate.lookahead_context),
                "successor_authority": gate.successor_authority,
                "successor_discontinuities": gate.successor_discontinuities.iter().map(|entry| json!({
                    "context_id": json_value_encoded(&entry.context_id),
                    "endpoint_gap": fbits(entry.endpoint_gap),
                })).collect::<Vec<_>>(),
                "advance_context": gate.advance_context.as_ref().map(json_value_encoded).unwrap_or(Value::Null),
                "terminal_lookahead": gate.terminal_lookahead,
                "terminal_distance_remaining": fbits(gate.terminal_distance_remaining),
                "terminal_state6_edge": gate.terminal_state6_edge,
                "secondary_endpoint": arr_fbits(gate.secondary_endpoint),
                "tertiary_endpoint": opt_arr_fbits(gate.tertiary_endpoint),
                "lookahead_secondary_endpoint": arr_fbits(gate.lookahead_secondary_endpoint),
                "lookahead_tertiary_endpoint": opt_arr_fbits(gate.lookahead_tertiary_endpoint),
                "rail_base_orientation": gate.rail_base_orientation.as_ref().map(encode_base_orientation).unwrap_or(Value::Null),
                "rail_base_orientation_authority": gate.rail_base_orientation_authority,
            }));
        }
    }
    row.insert("authority".to_string(), json!(result.authority));
    Value::Object(row)
}

fn encode_velocity_transfer(result: &RailState1VelocityTransfer) -> Value {
    json!({
        "position": arr_fbits(result.position),
        "lookahead_endpoint": arr_fbits(result.lookahead_endpoint),
        "min_velocity": fbits(result.min_velocity),
        "max_velocity": fbits(result.max_velocity),
        "current_speed": fbits(result.current_speed),
        "selected_speed": fbits(result.selected_speed),
        "velocity_after": arr_fbits(result.velocity_after),
        "velocity_magnitude": fbits(result.velocity_magnitude),
        "authority": result.authority,
    })
}

/// Projected `evaluate_rail_step` result (mirrors the Python encoder).
fn encode_rail_step_result(result: &RailStepResult) -> Value {
    let state2 = result.rail_state2_alignment_torque.as_ref().map(|alignment| json!({
        "torque_before": arr_fbits(alignment.torque_before),
        "torque_after": arr_fbits(alignment.torque_after),
        "signed_steering_axis": fbits(alignment.signed_steering_axis),
        "drive_tune_718": fbits(alignment.drive_tune_718),
        "angular_scale_pi_tune_over_180": fbits(alignment.angular_scale_pi_tune_over_180),
        "alignment_failed": alignment.alignment_failed,
        "fields": {
            "adc": fbits(alignment.fields.adc),
            "ae0": fbits(alignment.fields.ae0),
            "aec": fbits(alignment.fields.aec),
        },
        "events": alignment.events.iter().map(|event| json!({
            "name": event.name,
            "target": event.target,
            "operation": event.operation,
            "source": arr_fbits(event.source),
            "static_va": event.static_va,
        })).collect::<Vec<_>>(),
    }));
    let steering = result.rail_orientation_steering_torque.as_ref().map(|steering| json!({
        "torque_before": arr_fbits(steering.torque_before),
        "torque_after": arr_fbits(steering.torque_after),
        "signed_steering_axis": fbits(steering.signed_steering_axis),
        "drive_tune_718": fbits(steering.drive_tune_718),
        "angular_scale_pi_tune_over_180": fbits(steering.angular_scale_pi_tune_over_180),
        "events": steering.events.iter().map(|event| json!({
            "name": event.name,
            "target": event.target,
            "operation": event.operation,
            "source": arr_fbits(event.source),
            "static_va": event.static_va,
        })).collect::<Vec<_>>(),
    }));
    json!({
        "position_after": arr_fbits(result.position_after),
        "velocity_after": arr_fbits(result.velocity_after),
        "angular_velocity_after": arr_fbits(result.angular_velocity_after),
        "matrix_after": arr9_fbits(result.matrix_after),
        "force_accumulator": arr_fbits(result.force_accumulator),
        "torque_accumulator": arr_fbits(result.torque_accumulator),
        "velocity_before_damping": arr_fbits(result.velocity_before_damping),
        "pose_rail_state": result.pose_rail_state,
        "rail_orientation_dt": fbits(result.rail_orientation_dt),
        "rail_orientation_dt_authority": result.rail_orientation_dt_authority,
        "ordering": result.ordering,
        "closure_status": result.closure_status,
        "rail_selector": {
            "enabled_kinds": result.rail_selector.enabled_kinds,
            "force_branch": result.rail_selector.force_branch,
            "runtime_policy": result.rail_selector.runtime_policy,
        },
        "rail_accumulator": {
            "enabled_kinds": result.rail_accumulator.enabled_kinds,
            "events": result.rail_accumulator.events.iter().map(|event| json!({
                "kind": event.kind,
                "target": event.target,
                "operation": event.operation,
                "source": arr_fbits(event.source),
            })).collect::<Vec<_>>(),
            "accumulator_614_before": arr_fbits(result.rail_accumulator.accumulator_614_before),
            "accumulator_614_after": arr_fbits(result.rail_accumulator.accumulator_614_after),
            "accumulator_620_before": arr_fbits(result.rail_accumulator.accumulator_620_before),
            "accumulator_620_after": arr_fbits(result.rail_accumulator.accumulator_620_after),
            "gate_policy": result.rail_accumulator.gate_policy,
        },
        "rail_state2_alignment_torque": state2.unwrap_or(Value::Null),
        "rail_orientation_steering_torque": steering.unwrap_or(Value::Null),
        "post_rail_accumulator": {
            "force": opt_arr_fbits(result.post_rail_accumulator.force),
            "torque": opt_arr_fbits(result.post_rail_accumulator.torque),
            "force_before_damping": arr_fbits(result.post_rail_accumulator.force_before_damping),
            "torque_before_damping": arr_fbits(result.post_rail_accumulator.torque_before_damping),
            "authority": result.post_rail_accumulator.authority,
        },
        "common_damping": {
            "linear_damping_delta": arr_fbits(result.common_damping.linear_damping_delta),
            "angular_damping_delta": arr_fbits(result.common_damping.angular_damping_delta),
            "directional_multiplier": fbits(result.common_damping.directional_multiplier),
            "accumulator_614_after": arr_fbits(result.common_damping.accumulator_614_after),
            "accumulator_620_after": arr_fbits(result.common_damping.accumulator_620_after),
        },
        "linear_integrator": {
            "velocity_c0_after": arr_fbits(result.linear_integrator.velocity_c0_after),
        },
        "angular_integrator": {
            "angular_velocity_cc_after": arr_fbits(result.angular_integrator.angular_velocity_cc_after),
        },
        "position_integrator": {
            "position_90_after": arr_fbits(result.position_integrator.position_90_after),
        },
        "orientation_integrator": result.orientation_integrator.as_ref().map(encode_orientation_integrator_projected).unwrap_or(Value::Null),
        "rail_constraint_orientation": result.rail_constraint_orientation.as_ref().map(encode_constraint_orientation).unwrap_or(Value::Null),
    })
}

// ---------------------------------------------------------------------------
// Spec decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Spec {
    cases: Vec<SpecCase>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum SpecCase {
    #[serde(rename = "leaf")]
    Leaf { name: String, function: String, args: Value },
    #[serde(rename = "engine")]
    Engine {
        name: String,
        initial_state: Box<VehicleState>,
        parameters: Box<VehicleParameters>,
        numeric_mode: NumericMode,
        steps: Vec<SpecStep>,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum SpecStep {
    #[serde(rename = "step")]
    Step {
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "advance_selector")]
    AdvanceSelector { args: Value },
    #[serde(rename = "inspect")]
    Inspect { world: SpecWorld },
    #[serde(rename = "acquire")]
    Acquire {
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        part2_stepper_boundary: bool,
    },
}

#[derive(Deserialize, Clone)]
struct SpecWorld {
    capabilities: SpecWorldCapabilities,
    call_log: Vec<SpecCall>,
}

#[derive(Deserialize, Clone, Default)]
struct SpecWorldCapabilities {
    #[serde(default)]
    query_rail_contact: bool,
    #[serde(default)]
    sweep_obb_rotating_all: bool,
}

#[derive(Deserialize, Clone)]
struct SpecCall {
    method: String,
    args: Value,
    response: Value,
}

// ---------------------------------------------------------------------------
// Scripted replay world (same contract as raw_frame_trace.rs)
// ---------------------------------------------------------------------------

struct ScriptedWorld {
    capabilities: WorldCapabilities,
    call_log: Vec<SpecCall>,
    cursor: RefCell<usize>,
    mismatches: RefCell<Vec<String>>,
}

impl ScriptedWorld {
    fn next_call(&self, method: &str, args: Value) -> Option<Value> {
        let mut cursor = self.cursor.borrow_mut();
        let index = *cursor;
        *cursor += 1;
        if index >= self.call_log.len() {
            self.mismatches
                .borrow_mut()
                .push(format!("call {index}: unexpected extra call {method}"));
            return None;
        }
        let entry = &self.call_log[index];
        if entry.method != method {
            self.mismatches
                .borrow_mut()
                .push(format!("call {index}: method {} != {method}", entry.method));
        }
        if bits_json(&entry.args) != bits_json(&args) {
            self.mismatches.borrow_mut().push(format!(
                "call {index}: args mismatch for {method}: recorded {} vs engine {}",
                bits_json(&entry.args),
                bits_json(&args),
            ));
        }
        Some(entry.response.clone())
    }

    fn status(&self) -> Value {
        let cursor = *self.cursor.borrow();
        let mismatches = self.mismatches.borrow();
        json!({
            "consumed": cursor,
            "total": self.call_log.len(),
            "mismatches": mismatches.iter().map(|m| json!(m)).collect::<Vec<_>>(),
        })
    }
}

impl World for ScriptedWorld {
    fn capabilities(&self) -> WorldCapabilities {
        self.capabilities
    }

    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult {
        let args = json!({
            "position": arr_fbits(position),
            "up": arr_fbits(up),
            "min_dot": fbits(min_dot),
        });
        let response = self.next_call("query_ground", args).unwrap_or(Value::Null);
        serde_json::from_value(response)
            .unwrap_or_else(|error| panic!("recorded query_ground response does not decode: {error}"))
    }

    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<kart_physics::world::WorldObbOverlap>> {
        let args = json!({
            "start": arr_fbits(start),
            "end": arr_fbits(end),
            "start_matrix": Value::Array(start_matrix.iter().map(|&v| fbits(v)).collect()),
            "end_matrix": Value::Array(end_matrix.iter().map(|&v| fbits(v)).collect()),
            "half_extents": arr_fbits(half_extents),
        });
        let response = self.next_call("sweep_obb_rotating_all", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded sweep_obb_rotating_all response decodes"),
        )
    }

    fn query_rail_contact(
        &self,
        position: [f64; 3],
        previous_rail_state: i64,
        owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let args = json!({
            "position": arr_fbits(position),
            "previous_rail_state": previous_rail_state,
            "owner_id": owner_id,
        });
        let response = self.next_call("query_rail_contact", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_rail_contact response decodes"),
        )
    }
}

// ---------------------------------------------------------------------------
// Engine row encoders (mirror the Python verify script exactly)
// ---------------------------------------------------------------------------

fn topology_row(topology: &TopologyState) -> Value {
    bits_json(&serde_json::to_value(topology).expect("topology serializes"))
}

fn state_row(state: &VehicleState) -> Value {
    json!({
        "position": arr_fbits(state.position),
        "velocity": arr_fbits(state.velocity),
        "angular_velocity": arr_fbits(state.angular_velocity),
        "matrix": arr9_fbits(state.matrix),
        "boost_state": state.boost_state,
        "special_trigger_cooldown_ms": fbits(state.special_trigger_cooldown_ms),
        "surface_mode_latched": state.surface_mode_latched,
        "surface_selector_elapsed_ms": fbits(state.surface_selector_elapsed_ms),
        "rail_selector_runtime": state.rail_selector_runtime.as_ref().map(|runtime| {
            bits_json(&serde_json::to_value(runtime).expect("runtime serializes"))
        }).unwrap_or(Value::Null),
        "rail_capture_latch": state.rail_capture_latch.as_ref().map(|latch| {
            bits_json(&serde_json::to_value(latch).expect("latch serializes"))
        }).unwrap_or(Value::Null),
    })
}

fn ground_writer_row(writer: &RailState1GroundWriter) -> Value {
    // Python `_rail_state1_ground_writer_result` only includes the
    // `selector_phase` key when it is supplied.
    let mut row = Map::new();
    row.insert("selected".to_string(), json!(writer.selected));
    row.insert("state_after".to_string(), json!(writer.state_after));
    row.insert(
        "support_contact_count".to_string(),
        json!(writer.support_contact_count),
    );
    if let Some(selector_phase) = writer.selector_phase {
        row.insert("selector_phase".to_string(), json!(selector_phase));
    }
    row.insert("authority".to_string(), json!(writer.authority));
    Value::Object(row)
}

fn inspection_row(inspection: &RailCaptureInspection) -> Value {
    json!({
        "contact_active": inspection.contact_active,
        "rail_state": inspection.rail_state,
        "selector": inspection.selector.as_ref().map(|selector| {
            bits_json(&serde_json::to_value(selector).expect("selector serializes"))
        }).unwrap_or(Value::Null),
        "authority": inspection.authority,
    })
}

// ---------------------------------------------------------------------------
// Leaf argument decoding + dispatch
// ---------------------------------------------------------------------------

/// Spec floats are JSON numbers; non-finite values (JSON cannot carry
/// them) travel as "NaN" / "Infinity" / "-Infinity" strings.
fn spec_f64(value: &Value) -> f64 {
    match value {
        Value::Number(number) => number.as_f64().expect("f64"),
        Value::String(text) => match text.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => panic!("unknown non-finite token {other}"),
        },
        _ => panic!("f64 argument"),
    }
}

fn arr3(value: &Value) -> [f64; 3] {
    let array = value.as_array().expect("vec3 argument");
    [spec_f64(&array[0]), spec_f64(&array[1]), spec_f64(&array[2])]
}

fn arr9(value: &Value) -> [f64; 9] {
    let array = value.as_array().expect("vec9 argument");
    let mut result = [0.0; 9];
    for (index, item) in array.iter().enumerate().take(9) {
        result[index] = spec_f64(item);
    }
    result
}

fn opt_arr3(value: &Value) -> Option<[f64; 3]> {
    if value.is_null() { None } else { Some(arr3(value)) }
}

fn opt_f64(value: &Value) -> Option<f64> {
    if value.is_null() { None } else { Some(spec_f64(value)) }
}

fn opt_i64(value: &Value) -> Option<i64> {
    if value.is_null() { None } else { Some(value.as_i64().expect("i64")) }
}

fn opt_json_value(value: &Value) -> Option<JsonValue> {
    if value.is_null() { None } else { Some(serde_json::from_value(value.clone()).expect("JsonValue decodes")) }
}

fn run_leaf(function: &str, args: &Value) -> Result<Value, String> {
    match function {
        "evaluate_rail_query_scalar" => {
            let records: Vec<RailQueryScalarRecord> = args["records"]
                .as_array()
                .expect("records list")
                .iter()
                .map(|record| RailQueryScalarRecord {
                    position: arr3(&record["position"]),
                    direction: arr3(&record["direction"]),
                })
                .collect();
            let result = evaluate_rail_query_scalar(
                &records,
                arr3(&args["query_position"]),
                opt_f64(&args["total_length"]),
            )?;
            Ok(encode_scalar_result(&result))
        }
        "select_rail_query_context" => {
            let contexts: Vec<RailQueryContext> = args["contexts"]
                .as_array()
                .expect("contexts list")
                .iter()
                .map(|context| RailQueryContext {
                    context_id: opt_json_value(&context["context_id"]),
                    points: context["points"]
                        .as_array()
                        .expect("points list")
                        .iter()
                        .map(|point| RailQueryContextPoint {
                            position: arr3(&point["position"]),
                            normal: arr3(&point["normal"]),
                        })
                        .collect(),
                })
                .collect();
            let result = select_rail_query_context(&contexts, arr3(&args["kart_position"]))?;
            Ok(encode_context_selection(&result))
        }
        "interpolate_rail_path_candidate" => {
            let candidates: Vec<RailPathCandidate> = args["candidates"]
                .as_array()
                .expect("candidates list")
                .iter()
                .map(|candidate| RailPathCandidate {
                    primary: arr3(&candidate["primary"]),
                    secondary: arr3(&candidate["secondary"]),
                    tertiary: opt_arr3(&candidate["tertiary"]),
                })
                .collect();
            let result = interpolate_rail_path_candidate(
                &candidates,
                opt_f64(&args["query_metric"]),
                opt_arr3(&args["query_vector"]),
            )?;
            Ok(encode_interpolation(&result))
        }
        "evaluate_rail_base_orientation" => {
            let result = evaluate_rail_base_orientation(
                arr3(&args["secondary_endpoint"]),
                arr3(&args["tertiary_endpoint"]),
                args["runtime_float32_exact"].as_bool().expect("bool"),
            )?;
            Ok(encode_base_orientation(&result))
        }
        "evaluate_rail_constraint_orientation" => {
            let result = evaluate_rail_constraint_orientation(
                arr9(&args["primary_matrix_before"]),
                arr9(&args["rail_base_bf8"]),
                arr9(&args["rail_constraint_c1c_before"]),
                arr3(&args["angular_velocity"]),
                spec_f64(&args["dt"]),
                args["rail_state"].as_i64().expect("i64"),
                args["runtime_float32_exact"].as_bool().expect("bool"),
            )?;
            Ok(encode_constraint_orientation(&result))
        }
        "evaluate_rail_state1_query" => {
            let contexts: Vec<RailState1Context> = args["contexts"]
                .as_array()
                .expect("contexts list")
                .iter()
                .map(|context| RailState1Context {
                    context_id: opt_json_value(&context["context_id"]),
                    context_index: opt_i64(&context["context_index"]),
                    successor_ids: if context["successor_ids"].is_null() {
                        None
                    } else {
                        Some(
                            context["successor_ids"]
                                .as_array()
                                .expect("successor_ids list")
                                .iter()
                                .map(|id| serde_json::from_value(id.clone()).expect("JsonValue"))
                                .collect(),
                        )
                    },
                    road: if context["road"].is_null() {
                        None
                    } else {
                        Some(context["road"].as_str().expect("road").to_string())
                    },
                    total_length: opt_f64(&context["total_length"]),
                    points: context["points"]
                        .as_array()
                        .expect("points list")
                        .iter()
                        .map(|point| RailState1Point {
                            position: arr3(&point["position"]),
                            secondary: arr3(&point["secondary"]),
                            normal: arr3(&point["normal"]),
                        })
                        .collect(),
                })
                .collect();
            let result = evaluate_rail_state1_query(
                &contexts,
                arr3(&args["kart_position"]),
                args["contact_exists"].as_bool().expect("bool"),
                args["contact_identifier_nonempty"].as_bool().expect("bool"),
                spec_f64(&args["metric_threshold"]),
                spec_f64(&args["height_bias"]),
                spec_f64(&args["alignment_base_ms"]),
                opt_f64(&args["alignment_primary_scale_ms"]),
                opt_f64(&args["alignment_stable_initial_ms"]),
                opt_json_value(&args["retained_context_id"]),
            )?;
            Ok(encode_state1_query_result(&result))
        }
        "evaluate_rail_state1_velocity_transfer" => {
            let result = evaluate_rail_state1_velocity_transfer(
                arr3(&args["kart_position"]),
                arr3(&args["lookahead_endpoint"]),
                spec_f64(&args["min_velocity"]),
                opt_arr3(&args["current_velocity"]),
                opt_f64(&args["max_velocity"]),
            )?;
            Ok(encode_velocity_transfer(&result))
        }
        "evaluate_rail_capture_selector" => {
            let selector_args = RailCaptureSelectorArgs {
                state_before: args["state_before"].as_i64().expect("i64"),
                elapsed_in_state_ms: spec_f64(&args["elapsed_in_state_ms"]),
                field_af0: spec_f64(&args["field_af0"]),
                contact_exists: args["contact_exists"].as_bool().expect("bool"),
                contact_identifier_nonempty: args["contact_identifier_nonempty"]
                    .as_bool()
                    .expect("bool"),
                path_metric: opt_f64(&args["path_metric"]),
                metric_threshold: opt_f64(&args["metric_threshold"]),
                path_z: opt_f64(&args["path_z"]),
                kart_z_plus_one: opt_f64(&args["kart_z_plus_one"]),
                delta_ms: opt_f64(&args["delta_ms"]),
                alignment_primary_remaining_ms: opt_f64(&args["alignment_primary_remaining_ms"]),
                alignment_stable_remaining_ms: opt_f64(&args["alignment_stable_remaining_ms"]),
                forward_speed: opt_f64(&args["forward_speed"]),
                lateral_speed: opt_f64(&args["lateral_speed"]),
                trigger_cooldown_remaining_ms: opt_f64(&args["trigger_cooldown_remaining_ms"]),
                authored_trigger_accepted: args["authored_trigger_accepted"]
                    .as_bool()
                    .expect("bool"),
                state6_contact_remaining_ms: opt_f64(&args["state6_contact_remaining_ms"]),
                state6_contact_candidate_valid: args["state6_contact_candidate_valid"]
                    .as_bool()
                    .expect("bool"),
            };
            let result = evaluate_rail_capture_selector(&selector_args)?;
            Ok(encode_capture_selector_result(&result))
        }
        "evaluate_rail_step" => {
            let step_args: RailStepArgsWire =
                serde_json::from_value(args.clone()).expect("rail step args decode");
            let result = evaluate_rail_step(&step_args.into_args())?;
            Ok(encode_rail_step_result(&result))
        }
        _ => Err(format!("unknown leaf function {function}")),
    }
}

/// Serde wire shape for `RailStepArgs` (field names match the Python kwargs).
#[derive(Deserialize)]
struct RailStepArgsWire {
    position_before: [f64; 3],
    velocity_before: [f64; 3],
    angular_velocity_before: [f64; 3],
    matrix_before: [f64; 9],
    force_accumulator_before: [f64; 3],
    torque_accumulator_before: [f64; 3],
    rail_contact: bool,
    rail_state: i64,
    gate_1c0_decoded: bool,
    positive_a80: f64,
    negative_a80: f64,
    boost_family_active: bool,
    boost_special_gate_9b8: bool,
    boost_special_gate_6e4: bool,
    attached_object_state: i64,
    rail_dir: [f64; 3],
    accel_be0: f64,
    obf_6e8: f64,
    obf_808: f64,
    obf_814: f64,
    grip_720: f64,
    field_604: [f64; 3],
    mass: f64,
    gravity_be8: f64,
    adc: f64,
    ae0: f64,
    #[serde(default)]
    aec: f64,
    #[serde(default)]
    signed_steering_axis: f64,
    #[serde(default = "default_drive_tune")]
    drive_tune_718: f64,
    #[serde(default)]
    post_rail_force_accumulator: Option<[f64; 3]>,
    #[serde(default)]
    post_rail_torque_accumulator: Option<[f64; 3]>,
    inverse_inertia_matrix: [f64; 9],
    damping: f64,
    directional_damping: f64,
    damping_runtime_multiplier: f64,
    damping_special_gate: bool,
    dt: f64,
    numeric_mode: String,
    #[serde(default)]
    pre_damping_velocity_override: Option<[f64; 3]>,
    #[serde(default)]
    pose_rail_state: Option<i64>,
    #[serde(default)]
    rail_base_bf8: Option<[f64; 9]>,
    #[serde(default)]
    rail_constraint_c1c_before: Option<[f64; 9]>,
    #[serde(default)]
    rail_orientation_dt: Option<f64>,
}

fn default_drive_tune() -> f64 {
    10.0
}

impl RailStepArgsWire {
    fn into_args(self) -> RailStepArgs {
        RailStepArgs {
            position_before: self.position_before,
            velocity_before: self.velocity_before,
            angular_velocity_before: self.angular_velocity_before,
            matrix_before: self.matrix_before,
            force_accumulator_before: self.force_accumulator_before,
            torque_accumulator_before: self.torque_accumulator_before,
            rail_contact: self.rail_contact,
            rail_state: self.rail_state,
            gate_1c0_decoded: self.gate_1c0_decoded,
            positive_a80: self.positive_a80,
            negative_a80: self.negative_a80,
            boost_family_active: self.boost_family_active,
            boost_special_gate_9b8: self.boost_special_gate_9b8,
            boost_special_gate_6e4: self.boost_special_gate_6e4,
            attached_object_state: self.attached_object_state,
            rail_dir: self.rail_dir,
            accel_be0: self.accel_be0,
            obf_6e8: self.obf_6e8,
            obf_808: self.obf_808,
            obf_814: self.obf_814,
            grip_720: self.grip_720,
            field_604: self.field_604,
            mass: self.mass,
            gravity_be8: self.gravity_be8,
            adc: self.adc,
            ae0: self.ae0,
            aec: self.aec,
            signed_steering_axis: self.signed_steering_axis,
            drive_tune_718: self.drive_tune_718,
            post_rail_force_accumulator: self.post_rail_force_accumulator,
            post_rail_torque_accumulator: self.post_rail_torque_accumulator,
            inverse_inertia_matrix: self.inverse_inertia_matrix,
            damping: self.damping,
            directional_damping: self.directional_damping,
            damping_runtime_multiplier: self.damping_runtime_multiplier,
            damping_special_gate: self.damping_special_gate,
            dt: self.dt,
            numeric_mode: self.numeric_mode,
            pre_damping_velocity_override: self.pre_damping_velocity_override,
            pose_rail_state: self.pose_rail_state,
            rail_base_bf8: self.rail_base_bf8,
            rail_constraint_c1c_before: self.rail_constraint_c1c_before,
            rail_orientation_dt: self.rail_orientation_dt,
        }
    }
}

// ---------------------------------------------------------------------------
// Engine step drivers
// ---------------------------------------------------------------------------

fn build_world(spec: &Option<SpecWorld>) -> Option<ScriptedWorld> {
    spec.as_ref().map(|world| ScriptedWorld {
        capabilities: WorldCapabilities {
            query_rail_contact: world.capabilities.query_rail_contact,
            sweep_obb_rotating_all: world.capabilities.sweep_obb_rotating_all,
            ..WorldCapabilities::default()
        },
        call_log: world.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
    })
}

fn world_status(world: Option<&ScriptedWorld>) -> Value {
    world.map_or(
        json!({"consumed": 0, "total": 0, "mismatches": Value::Array(vec![])}),
        |world| world.status(),
    )
}

fn run_engine_case(
    name: &str,
    initial_state: &VehicleState,
    parameters: &VehicleParameters,
    numeric_mode: NumericMode,
    steps: &[SpecStep],
) -> Value {
    let mut engine = KartRiderVehicleEngine::new(initial_state, numeric_mode)
        .expect("initial state validates");
    let mut rows = Vec::new();
    for step in steps {
        match step {
            SpecStep::Step { keys, world, parameters: step_parameters } => {
                let scripted = build_world(world);
                let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
                let result = engine.step_raw_topology(
                    keys,
                    step_parameters.as_deref().unwrap_or(parameters),
                    world_ref,
                    None,
                    None,
                );
                let mut row = Map::new();
                row.insert("kind".to_string(), json!("step"));
                match &result {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert("rail_captured".to_string(), json!(outcome.rail_captured));
                        row.insert(
                            "executor".to_string(),
                            json!(match &outcome.executor {
                                kart_physics::engine::TopologyExecutorOutcome::Ground(_) => "ground",
                                kart_physics::engine::TopologyExecutorOutcome::Airborne(_) => "airborne",
                                kart_physics::engine::TopologyExecutorOutcome::ResetSurface(_) => "reset_surface",
                                kart_physics::engine::TopologyExecutorOutcome::RailCapture(_) => "rail_capture",
                                kart_physics::engine::TopologyExecutorOutcome::MovingContactCapture(_) => "moving_contact_capture",
                                kart_physics::engine::TopologyExecutorOutcome::NetworkCorrection(_) => "network_correction",
                            }),
                        );
                        row.insert(
                            "support_contact_count".to_string(),
                            json!(outcome.support_observation.support_contact_count),
                        );
                        row.insert(
                            "rail_state1_ground_writer".to_string(),
                            outcome
                                .rail_state1_ground_writer
                                .as_ref()
                                .map(ground_writer_row)
                                .unwrap_or(Value::Null),
                        );
                        row.insert(
                            "post_ground_rail_capture".to_string(),
                            outcome
                                .post_ground_rail_capture
                                .as_ref()
                                .map(|capture| json!({
                                    "selector_phase": capture.selector_phase,
                                    "shared_ground_epoch_count": capture.shared_ground_epoch_count,
                                    "authority": capture.authority,
                                }))
                                .unwrap_or(Value::Null),
                        );
                        row.insert(
                            "autonomous_rail_selector".to_string(),
                            outcome
                                .autonomous_rail_selector
                                .as_ref()
                                .map(|selector| {
                                    bits_json(
                                        &serde_json::to_value(selector)
                                            .expect("selector serializes"),
                                    )
                                })
                                .unwrap_or(Value::Null),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("rail_captured".to_string(), json!(false));
                        row.insert("executor".to_string(), Value::Null);
                        row.insert("support_contact_count".to_string(), Value::Null);
                        row.insert("rail_state1_ground_writer".to_string(), Value::Null);
                        row.insert("post_ground_rail_capture".to_string(), Value::Null);
                        row.insert("autonomous_rail_selector".to_string(), Value::Null);
                    }
                }
                row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
                row.insert("state".to_string(), state_row(&engine.state));
                row.insert("world_status".to_string(), world_status(scripted.as_ref()));
                rows.push(Value::Object(row));
            }
            SpecStep::AdvanceSelector { args } => {
                let selector_args: AdvanceSelectorArgsWire =
                    serde_json::from_value(args.clone()).expect("advance selector args decode");
                let result =
                    engine.advance_rail_capture_selector(&selector_args.into_args());
                let mut row = Map::new();
                row.insert("kind".to_string(), json!("advance_selector"));
                match &result {
                    Ok(advance) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert(
                            "selector".to_string(),
                            encode_capture_selector_result(&advance.selector),
                        );
                        row.insert(
                            "topology_transition".to_string(),
                            json!(advance.topology_transition.is_some()),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("selector".to_string(), Value::Null);
                        row.insert("topology_transition".to_string(), Value::Null);
                    }
                }
                row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
                row.insert("state".to_string(), state_row(&engine.state));
                rows.push(Value::Object(row));
            }
            SpecStep::Inspect { world } => {
                let scripted = build_world(&Some(world.clone()));
                let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
                let result = engine.inspect_autonomous_rail_capture(world_ref);
                let mut row = Map::new();
                row.insert("kind".to_string(), json!("inspect"));
                match &result {
                    Ok(inspection) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert("inspection".to_string(), inspection_row(inspection));
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("inspection".to_string(), Value::Null);
                    }
                }
                row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
                row.insert("state".to_string(), state_row(&engine.state));
                row.insert("world_status".to_string(), world_status(scripted.as_ref()));
                rows.push(Value::Object(row));
            }
            SpecStep::Acquire { keys, world, part2_stepper_boundary } => {
                let scripted = build_world(world);
                let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
                let result = engine.acquire_autonomous_rail(
                    keys,
                    parameters,
                    world_ref,
                    None,
                    None,
                );
                let mut row = Map::new();
                row.insert("kind".to_string(), json!("acquire"));
                match &result {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert("captured".to_string(), json!(outcome.is_some()));
                        row.insert(
                            "rail_state1_ground_writer".to_string(),
                            outcome
                                .as_ref()
                                .and_then(|execution| match execution {
                                    kart_physics::engine::RailCaptureExecution::GroundWriter(capture) => {
                                        Some(ground_writer_row(&capture.rail_state1_ground_writer))
                                    }
                                    kart_physics::engine::RailCaptureExecution::RawRail(_) => None,
                                })
                                .unwrap_or(Value::Null),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("captured".to_string(), Value::Null);
                        row.insert("rail_state1_ground_writer".to_string(), Value::Null);
                    }
                }
                if *part2_stepper_boundary {
                    // Pinned part-1 contract: this scenario was captured
                    // while the RAIL domain stepper still returned the
                    // structured part-2 error. The Python harness overrides
                    // its row the same way and compares only the
                    // entry-determined topology fields, so the full writer
                    // chain (now ported on both sides) is masked out here.
                    row.insert("error".to_string(), json!(
                        "autonomous RAIL stepping (step_autonomous_rail) is not yet ported (M2 part 2 scope)"
                    ));
                    row.insert("captured".to_string(), Value::Null);
                    row.insert("rail_state1_ground_writer".to_string(), Value::Null);
                    let topology = &engine.state.topology;
                    row.insert("entered_topology".to_string(), json!({
                        "domain": topology.domain.value(),
                        "previous_domain": topology.previous_domain.map(|domain| domain.value()),
                        "return_domain": topology.return_domain.map(|domain| domain.value()),
                        "transition_index": topology.transition_index,
                        "entry_reason": topology.entry_reason,
                        "constraint_owner_id": topology.constraint_owner_id,
                        "rail_state": topology.rail_state,
                    }));
                    row.insert("topology_after".to_string(), Value::Null);
                    row.insert("state".to_string(), Value::Null);
                } else {
                    row.insert("entered_topology".to_string(), Value::Null);
                    row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
                    row.insert("state".to_string(), state_row(&engine.state));
                }
                row.insert("world_status".to_string(), world_status(scripted.as_ref()));
                rows.push(Value::Object(row));
            }
        }
    }
    json!({"name": name, "rows": rows})
}

/// Serde wire shape for `AdvanceRailCaptureSelectorArgs`.
#[derive(Deserialize, Default)]
struct AdvanceSelectorArgsWire {
    #[serde(default)]
    elapsed_in_state_ms: f64,
    #[serde(default)]
    field_af0: f64,
    #[serde(default)]
    contact_exists: bool,
    #[serde(default)]
    contact_identifier_nonempty: bool,
    #[serde(default)]
    path_metric: Option<f64>,
    #[serde(default)]
    metric_threshold: Option<f64>,
    #[serde(default)]
    path_z: Option<f64>,
    #[serde(default)]
    kart_z_plus_one: Option<f64>,
    #[serde(default)]
    delta_ms: Option<f64>,
    #[serde(default)]
    alignment_base_ms: Option<f64>,
    #[serde(default)]
    alignment_primary_scale_ms: Option<f64>,
    #[serde(default)]
    alignment_stable_initial_ms: Option<f64>,
    #[serde(default)]
    alignment_primary_remaining_ms: Option<f64>,
    #[serde(default)]
    alignment_stable_remaining_ms: Option<f64>,
    #[serde(default)]
    forward_speed: Option<f64>,
    #[serde(default)]
    lateral_speed: Option<f64>,
    #[serde(default)]
    trigger_cooldown_remaining_ms: Option<f64>,
    #[serde(default)]
    authored_trigger_accepted: bool,
    #[serde(default)]
    release_support_contact_count: i64,
    #[serde(default)]
    release_surface_mode: i64,
}

impl AdvanceSelectorArgsWire {
    fn into_args(self) -> AdvanceRailCaptureSelectorArgs {
        AdvanceRailCaptureSelectorArgs {
            elapsed_in_state_ms: self.elapsed_in_state_ms,
            field_af0: self.field_af0,
            contact_exists: self.contact_exists,
            contact_identifier_nonempty: self.contact_identifier_nonempty,
            path_metric: self.path_metric,
            metric_threshold: self.metric_threshold,
            path_z: self.path_z,
            kart_z_plus_one: self.kart_z_plus_one,
            delta_ms: self.delta_ms,
            alignment_base_ms: self.alignment_base_ms,
            alignment_primary_scale_ms: self.alignment_primary_scale_ms,
            alignment_stable_initial_ms: self.alignment_stable_initial_ms,
            alignment_primary_remaining_ms: self.alignment_primary_remaining_ms,
            alignment_stable_remaining_ms: self.alignment_stable_remaining_ms,
            forward_speed: self.forward_speed,
            lateral_speed: self.lateral_speed,
            trigger_cooldown_remaining_ms: self.trigger_cooldown_remaining_ms,
            authored_trigger_accepted: self.authored_trigger_accepted,
            release_support_contact_count: self.release_support_contact_count,
            release_surface_mode: self.release_surface_mode,
        }
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Spec = serde_json::from_str(&input).expect("spec decodes");
    let mut out = Vec::new();
    for case in &spec.cases {
        match case {
            SpecCase::Leaf { name, function, args } => {
                let row = match run_leaf(function, args) {
                    Ok(result) => json!({"name": name, "rows": [{
                        "kind": "leaf",
                        "function": function,
                        "error": Value::Null,
                        "result": result,
                    }]}),
                    Err(error) => json!({"name": name, "rows": [{
                        "kind": "leaf",
                        "function": function,
                        "error": error,
                        "result": Value::Null,
                    }]}),
                };
                out.push(row);
            }
            SpecCase::Engine { name, initial_state, parameters, numeric_mode, steps } => {
                out.push(run_engine_case(name, initial_state, parameters, *numeric_mode, steps));
            }
        }
    }
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
}
