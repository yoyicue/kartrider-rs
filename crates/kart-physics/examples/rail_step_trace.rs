//! RAIL domain stepper parity trace tool (M2 part 2).
//!
//! Reads a JSON spec from stdin: `{"cases": [...]}`. Every case is
//! `{"kind": "engine", "name": ..., "initial_state": ..., "parameters":
//! ..., "numeric_mode": ..., "steps": [...]}` and drives the engine-level
//! RAIL steppers. Step shapes:
//! - `{"kind": "rail_step", "keys": [...], "parameters": <opt>, "world":
//!   <scripted>}` runs `step_autonomous_rail`.
//! - `{"kind": "state6_tail", "keys": [...], "parameters": <opt>, "world":
//!   <scripted>}` runs `_step_rail_state6_tail` directly (function-level
//!   cases over constructed engines).
//! - `{"kind": "raw_motion", "keys": [...], "parameters": <opt>, "world":
//!   <scripted>}` runs `step_raw_motion` (covers the
//!   `_step_raw_motion_once` RAIL dispatch arm).
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! method name and bit-exact arguments, and returns the recorded response
//! (same contract as rail_capture_trace.rs). Every f64 is emitted as its
//! IEEE-754 bit pattern (`{:016x}`); a domain `Err` is emitted as
//! `{"error": <message>}` and the row still carries the post-call state.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;

use kart_physics::domains::rail_capture_selector::{
    RailCaptureSelectorResult, RailCaptureSelectorRuntimeAfter,
};
use kart_physics::engine::{
    AutonomousRailOutcome, AutonomousRailSelectorAdvance, KartRiderVehicleEngine, NumericMode,
    RailState6TailOutcome,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{JsonValue, RailPathContextId, RailSelectorRuntime, VehicleState};
use kart_physics::topology::TopologyState;
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Bit-pattern encoding helpers (same conventions as rail_capture_trace.rs)
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

fn opt_arr9_fbits(values: Option<[f64; 9]>) -> Value {
    values.map_or(Value::Null, arr9_fbits)
}

fn opt_fbits(value: Option<f64>) -> Value {
    value.map_or(Value::Null, fbits)
}

/// Open-payload encoding (Python `encode_open_payload_preserve_ints` on the
/// harness side): integral numbers are emitted as JSON integers so a Python
/// `int` and the Rust `JsonValue::Number` compare equal; numbers with a
/// fraction keep the f64 bit-pattern string.
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

fn json_map_encoded(map: &BTreeMap<String, JsonValue>) -> Value {
    Value::Object(
        map.iter()
            .map(|(key, value)| (key.clone(), json_value_encoded(value)))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// State / topology row encoders (mirror the Python verify script exactly)
// ---------------------------------------------------------------------------

fn topology_row(topology: &TopologyState) -> Value {
    bits_json(&serde_json::to_value(topology).expect("topology serializes"))
}

/// `rail_selector_runtime` encoding: every field except `path_context_id`
/// is an f64 on the Rust side (Python stores floats; the state6 int
/// counters are encoded as their f64 bit pattern on the Python side too).
fn runtime_encoded(runtime: &RailSelectorRuntime) -> Value {
    let mut row = Map::new();
    if let Some(value) = runtime.alignment_primary_remaining_ms {
        row.insert("alignment_primary_remaining_ms".to_string(), fbits(value));
    }
    if let Some(value) = runtime.alignment_stable_remaining_ms {
        row.insert("alignment_stable_remaining_ms".to_string(), fbits(value));
    }
    if let Some(id) = &runtime.path_context_id {
        row.insert(
            "path_context_id".to_string(),
            match id {
                RailPathContextId::Index(index) => json!(index),
                RailPathContextId::Name(name) => json!(name),
            },
        );
    }
    if let Some(value) = runtime.state6_contact_remaining_ms {
        row.insert("state6_contact_remaining_ms".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_contact_candidate_latched {
        row.insert("state6_contact_candidate_latched".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_zero_boundary_release_delay_substeps {
        row.insert(
            "state6_zero_boundary_release_delay_substeps".to_string(),
            fbits(value),
        );
    }
    if let Some(value) = runtime.state6_entry_force_pending_x {
        row.insert("state6_entry_force_pending_x".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_entry_force_pending_y {
        row.insert("state6_entry_force_pending_y".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_entry_force_pending_z {
        row.insert("state6_entry_force_pending_z".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_entry_force_delay_substeps {
        row.insert("state6_entry_force_delay_substeps".to_string(), fbits(value));
    }
    if let Some(value) = runtime.state6_entry_common_tail_delay_substeps {
        row.insert(
            "state6_entry_common_tail_delay_substeps".to_string(),
            fbits(value),
        );
    }
    Value::Object(row)
}

fn state_row(state: &VehicleState) -> Value {
    json!({
        "position": arr_fbits(state.position),
        "velocity": arr_fbits(state.velocity),
        "angular_velocity": arr_fbits(state.angular_velocity),
        "matrix": arr9_fbits(state.matrix),
        "force_accumulator": arr_fbits(state.force_accumulator),
        "torque_accumulator": arr_fbits(state.torque_accumulator),
        "boost_state": state.boost_state,
        "boost_remaining_ms": state.boost_remaining_ms,
        "drive_selected_slot": fbits(state.drive_selected_slot),
        "contact_compression": Value::Array(state.contact_compression.iter().map(|&v| fbits(v)).collect()),
        "soft_flag_188": state.soft_flag_188,
        "special_trigger_cooldown_ms": fbits(state.special_trigger_cooldown_ms),
        "authored_trigger_contact_id": state.authored_trigger_contact_id,
        "surface_mode_latched": state.surface_mode_latched,
        "surface_selector_elapsed_ms": fbits(state.surface_selector_elapsed_ms),
        "rail_selector_runtime": state.rail_selector_runtime.as_ref().map(runtime_encoded).unwrap_or(Value::Null),
        "rail_base_bf8": opt_arr9_fbits(state.rail_base_bf8),
        "rail_constraint_c1c": opt_arr9_fbits(state.rail_constraint_c1c),
        "rail_orientation_dt_latch": opt_fbits(state.rail_orientation_dt_latch),
        "rail_capture_latch": state.rail_capture_latch.as_ref().map(|latch| json!({
            "contact_active": latch.contact_active,
            "rail_state": latch.rail_state,
            "selector": json_map_encoded(&latch.selector),
            "authority": latch.authority,
        })).unwrap_or(Value::Null),
        "time_ms": state.time_ms,
        "step_index": state.step_index,
    })
}

// ---------------------------------------------------------------------------
// Outcome projection encoders (mirror the Python verify script exactly)
// ---------------------------------------------------------------------------

fn selector_projection(selector: &RailCaptureSelectorResult) -> Value {
    let runtime_after = match &selector.runtime_after {
        None => Value::Null,
        Some(RailCaptureSelectorRuntimeAfter::State2(runtime)) => json!({
            "kind": "state2",
            "alignment_primary_remaining_ms": fbits(runtime.alignment_primary_remaining_ms),
            "alignment_stable_remaining_ms": fbits(runtime.alignment_stable_remaining_ms),
        }),
        Some(RailCaptureSelectorRuntimeAfter::State3(runtime)) => json!({
            "kind": "state3",
            "trigger_cooldown_remaining_ms": fbits(runtime.trigger_cooldown_remaining_ms),
        }),
        Some(RailCaptureSelectorRuntimeAfter::State6(runtime)) => json!({
            "kind": "state6",
            "state6_contact_remaining_ms": fbits(runtime.state6_contact_remaining_ms),
            "state6_contact_raw_after_ms": fbits(runtime.state6_contact_raw_after_ms),
        }),
    };
    json!({
        "state_before": selector.state_before,
        "state_after": selector.state_after,
        "transitioned": selector.transitioned,
        "reason": selector.reason,
        "runtime_after": runtime_after,
    })
}

fn advance_projection(advance: &AutonomousRailSelectorAdvance) -> Value {
    json!({
        "selector": selector_projection(&advance.advance.selector),
        "topology_transition": advance.advance.topology_transition.is_some(),
        "has_autonomous_state1_query": advance.autonomous_state1_query.is_some(),
    })
}

fn state6_tail_projection(outcome: &RailState6TailOutcome) -> Value {
    let bridge = &outcome.rail_state6_partial_wheel_bridge;
    json!({
        "outcome_kind": "state6_tail",
        "rail_released": outcome.rail_released,
        "release_destination": outcome.release_destination,
        "support_force_branch": outcome.rail_state6_support_force_branch,
        "support_force": {
            "force": opt_arr_fbits(outcome.rail_state6_support_force.force),
            "torque_after_state6_angular": arr_fbits(outcome.rail_state6_support_force.torque_after_state6_angular),
        },
        "contact_physics": outcome.rail_state6_contact_physics,
        "common_angular": {
            "coefficient": fbits(outcome.rail_state6_common_angular_damping.coefficient),
            "final_coefficient_after_common_damping": fbits(outcome.rail_state6_common_angular_damping.final_coefficient_after_common_damping),
            "torque_delta": arr_fbits(outcome.rail_state6_common_angular_damping.torque_delta),
        },
        "collision_schedule": outcome.rail_state6_collision_schedule,
        "collision_chain": outcome.rail_state6_collision_chain.iter().map(|response| json!({
            "loop_index": response.loop_index,
            "branch": response.branch,
            "velocity_after": arr_fbits(response.velocity_after),
        })).collect::<Vec<_>>(),
        "partial_wheel_bridge": {
            "selected": bridge.selected,
            "contact_188": bridge.contact_188,
            "active_slots": Value::Array(bridge.active_slots.iter().map(|v| json!(v)).collect()),
            "active_count": bridge.active_count,
            "velocity_delta": arr_fbits(bridge.velocity_delta),
            "missing_wheel_queries": bridge.missing_wheel_queries.iter().map(|query| json!({
                "slot": query.slot,
                "start": arr_fbits(query.start),
                "end": arr_fbits(query.end),
                "hit": query.hit,
                "candidate": json_map_encoded(&query.candidate),
            })).collect::<Vec<_>>(),
        },
        "exit": {
            "selector": selector_projection(&outcome.rail_state6_exit.selector),
            "float_zero_boundary_writer_committed": outcome.rail_state6_exit.float_zero_boundary_writer_committed,
        },
        "contact_slots": Value::Array(outcome.state6_contact_slots.iter().map(|v| json!(v)).collect()),
        "contact_candidate": {
            "segment_start": arr_fbits(outcome.state6_contact_candidate.segment_start),
            "segment_end": arr_fbits(outcome.state6_contact_candidate.segment_end),
            "valid": outcome.state6_contact_candidate.valid,
            "selected_source": outcome.state6_contact_candidate.selected_source,
            "contact_188_before": outcome.state6_contact_candidate.contact_188_before,
            "contact_188_after": outcome.state6_contact_candidate.contact_188_after,
            "candidate": json_map_encoded(&outcome.state6_contact_candidate.candidate),
        },
        "entry_force_transaction": outcome.rail_state6_entry_force_transaction.as_ref().map(|transaction| json!({
            "phase": transaction.phase,
            "delay_substeps_after": transaction.delay_substeps_after,
            "force": arr_fbits(transaction.force),
        })).unwrap_or(Value::Null),
        "release_pending_producer": outcome.rail_release_pending_producer.as_ref().map(|pending| json!({
            "destination": pending.destination,
            "force_accumulator": arr_fbits(pending.force_accumulator),
            "torque_accumulator": arr_fbits(pending.torque_accumulator),
            "contact_compression": pending.contact_compression.map(fbits),
            "takeoff_separation_pending_before_release": pending.takeoff_separation_pending_before_release,
            "takeoff_separation_pending_after_release": pending.takeoff_separation_pending_after_release,
        })).unwrap_or(Value::Null),
        "airborne": {
            "position_after": arr_fbits(outcome.airborne.position_after),
            "velocity_after": arr_fbits(outcome.airborne.velocity_after),
            "angular_velocity_after": arr_fbits(outcome.airborne.angular_velocity_after),
            "matrix_after": arr9_fbits(outcome.airborne.matrix_after),
            "force_accumulator": arr_fbits(outcome.airborne.force_accumulator),
            "torque_accumulator": arr_fbits(outcome.airborne.torque_accumulator),
        },
    })
}

fn rail_outcome_projection(outcome: &AutonomousRailOutcome) -> Value {
    match outcome {
        AutonomousRailOutcome::State6Tail(tail) => state6_tail_projection(tail),
        AutonomousRailOutcome::ResetRelease(release) => json!({
            "outcome_kind": "reset_release",
            "release_destination": "RESET",
            "reset_surface_entry": release.reset.reset_surface_entry,
        }),
        AutonomousRailOutcome::ContactRelease(release) => json!({
            "outcome_kind": "contact_release",
            "release_destination": release.release.release_destination,
            "rail_selector": {
                "rail_contact": release.release.rail_selector.rail_contact,
                "rail_state_before": release.release.rail_selector.rail_state_before,
                "support_contact_count": release.release.rail_selector.support_contact_count,
                "surface_mode": release.release.rail_selector.surface_mode,
            },
            "context_authority": release.release.rail_contract.rail_context_authority,
        }),
        AutonomousRailOutcome::SelectorRelease(release) => json!({
            "outcome_kind": "selector_release",
            "release_destination": release.release_destination,
            "rail_state_transition": release.rail_state_transition.as_ref().map(advance_projection).unwrap_or(Value::Null),
        }),
        AutonomousRailOutcome::State1GroundWriter(state1) => json!({
            "outcome_kind": "state1_ground_writer",
            "writer": {
                "selected": state1.rail_state1_ground_writer.selected,
                "state_after": state1.rail_state1_ground_writer.state_after,
                "support_contact_count": state1.rail_state1_ground_writer.support_contact_count,
                "selector_phase": state1.rail_state1_ground_writer.selector_phase,
                "authority": state1.rail_state1_ground_writer.authority,
            },
            "post_linear": state1.rail_state1_post_linear_collision_response.as_ref().map(|response| json!({
                "velocity_delta": arr_fbits(response.velocity_delta),
                "path_speed": fbits(response.path_speed),
                "gain_per_second": fbits(response.gain_per_second),
                "low_speed_limit": fbits(response.low_speed_limit),
            })).unwrap_or(Value::Null),
            "world_sweep": state1.ground.world_sweep.as_ref().map(|sweep| json!({
                "dispatch_reason": sweep.dispatch_reason,
                "response_dispatched": sweep.response_dispatched,
            })).unwrap_or(Value::Null),
            "rail_state_transition": state1.rail_state_transition.as_ref().map(advance_projection).unwrap_or(Value::Null),
        }),
        AutonomousRailOutcome::Active(active) => {
            let rail_step = &active.rail.rail_step;
            json!({
                "outcome_kind": "active",
                "rail_step": {
                    "position_after": arr_fbits(rail_step.position_after),
                    "velocity_after": arr_fbits(rail_step.velocity_after),
                    "angular_velocity_after": arr_fbits(rail_step.angular_velocity_after),
                    "matrix_after": arr9_fbits(rail_step.matrix_after),
                    "force_accumulator": arr_fbits(rail_step.force_accumulator),
                    "torque_accumulator": arr_fbits(rail_step.torque_accumulator),
                    "pose_rail_state": rail_step.pose_rail_state,
                    "rail_constraint_orientation": rail_step.rail_constraint_orientation.as_ref().map(|orientation| json!({
                        "rail_base_bf8_after": arr9_fbits(orientation.rail_base_bf8_after),
                        "rail_constraint_c1c_after": arr9_fbits(orientation.rail_constraint_c1c_after),
                    })).unwrap_or(Value::Null),
                },
                "accumulator_clear": {
                    "force_before_clear": arr_fbits(active.rail.accumulator_clear.force_before_clear),
                    "torque_before_clear": arr_fbits(active.rail.accumulator_clear.torque_before_clear),
                },
                "contract_context_authority": active.rail.rail_contract.rail_context_authority,
                "rail_terminal_position_phase": active.rail_terminal_position_phase.as_ref().map(|phase| json!({
                    "published_position": arr_fbits(phase.published_position),
                    "suppressed_post_setter_position": arr_fbits(phase.suppressed_post_setter_position),
                    "published_velocity": arr_fbits(phase.published_velocity),
                    "suppressed_post_setter_velocity": arr_fbits(phase.suppressed_post_setter_velocity),
                })).unwrap_or(Value::Null),
                "rail_support_observation_present": active.rail_support_observation.is_some(),
                "rail_authored_trigger": active.rail_authored_trigger.as_ref().map(|edge| json!({
                    "transitioned": edge.selector.transitioned,
                    "state_after": edge.selector.state_after,
                    "cooldown_remaining_after": match &edge.selector.runtime_after {
                        Some(RailCaptureSelectorRuntimeAfter::State3(runtime)) => fbits(runtime.trigger_cooldown_remaining_ms),
                        _ => Value::Null,
                    },
                    "velocity_c0_after": arr_fbits(edge.trigger.velocity_c0_after),
                    "authored_token": edge.authored_token,
                    "state_transition": edge.trigger.state_transition.as_ref().map(|transition| json!({
                        "state": transition.state,
                        "remaining_ms": transition.remaining_ms,
                    })).unwrap_or(Value::Null),
                })).unwrap_or(Value::Null),
                "rail_authored_trigger_geometry": active.rail_authored_trigger_geometry.as_ref().map(|geometry| json!({
                    "authority": geometry.authority,
                    "center": opt_arr_fbits(geometry.center),
                })).unwrap_or(Value::Null),
                "rail_terminal_endpoint_transition": active.rail_terminal_endpoint_transition.as_ref().map(|transition| json!({
                    "selected_context": transition.selected_context.as_ref().map(json_value_encoded).unwrap_or(Value::Null),
                    "query_scalar": opt_fbits(transition.query_scalar),
                    "terminal_distance_remaining": opt_fbits(transition.terminal_distance_remaining),
                })).unwrap_or(Value::Null),
                "rail_state6_entry_force": active.rail_state6_entry_force.as_ref().map(|force| json!({
                    "force_accumulator_b74": arr_fbits(force.force_accumulator_b74),
                })).unwrap_or(Value::Null),
                "rail_state_transition": active.rail_state_transition.as_ref().map(advance_projection).unwrap_or(Value::Null),
                "rail_path_velocity_transfer": active.rail_path_velocity_transfer.as_ref().map(|transfer| json!({
                    "state_before": transfer.state_before,
                    "state_after": transfer.state_after,
                    "velocity_after": arr_fbits(transfer.transfer.velocity_after),
                    "selected_speed": fbits(transfer.transfer.selected_speed),
                    "lookahead_distance": opt_fbits(transfer.lookahead_distance),
                    "velocity_before": opt_arr_fbits(transfer.velocity_before),
                    "query_scalar": opt_fbits(transfer.query_scalar),
                    "selected_context": transfer.selected_context.as_ref().map(json_value_encoded).unwrap_or(Value::Null),
                    "context_transfer": transfer.context_transfer.as_ref().map(|context| json!({
                        "from": json_value_encoded(&context.from),
                        "to": json_value_encoded(&context.to),
                    })).unwrap_or(Value::Null),
                    "rail_writer_direction": arr_fbits(transfer.rail_writer_direction),
                })).unwrap_or(Value::Null),
            })
        }
    }
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
    #[serde(rename = "rail_step")]
    RailStep {
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "state6_tail")]
    State6Tail {
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "raw_motion")]
    RawMotion {
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
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
    #[serde(default)]
    sweep_point: bool,
    #[serde(default)]
    overlap_obb_all: bool,
    #[serde(default)]
    query_at_step_start: bool,
}

#[derive(Deserialize, Clone)]
struct SpecCall {
    method: String,
    args: Value,
    response: Value,
}

// ---------------------------------------------------------------------------
// Scripted replay world (same contract as rail_capture_trace.rs)
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

    fn sweep_point(
        &self,
        start: [f64; 3],
        end: [f64; 3],
    ) -> Option<BTreeMap<String, JsonValue>> {
        let args = json!({
            "start": arr_fbits(start),
            "end": arr_fbits(end),
        });
        let response = self.next_call("sweep_point", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded sweep_point response decodes"),
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

    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<kart_physics::world::WorldObbOverlap>> {
        let args = json!({
            "center": arr_fbits(center),
            "matrix": Value::Array(matrix.iter().map(|&v| fbits(v)).collect()),
            "half_extents": arr_fbits(half_extents),
        });
        let response = self.next_call("overlap_obb_all", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded overlap_obb_all response decodes"),
        )
    }
}

fn build_world(spec: &Option<SpecWorld>) -> Option<ScriptedWorld> {
    spec.as_ref().map(|world| ScriptedWorld {
        capabilities: WorldCapabilities {
            query_rail_contact: world.capabilities.query_rail_contact,
            sweep_obb_rotating_all: world.capabilities.sweep_obb_rotating_all,
            sweep_point: world.capabilities.sweep_point,
            overlap_obb_all: world.capabilities.overlap_obb_all,
            query_at_step_start: world.capabilities.query_at_step_start,
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

// ---------------------------------------------------------------------------
// Engine case runner
// ---------------------------------------------------------------------------

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
        let (kind, keys, world, step_parameters) = match step {
            SpecStep::RailStep { keys, world, parameters } => {
                ("rail_step", keys, world, parameters)
            }
            SpecStep::State6Tail { keys, world, parameters } => {
                ("state6_tail", keys, world, parameters)
            }
            SpecStep::RawMotion { keys, world, parameters } => {
                ("raw_motion", keys, world, parameters)
            }
        };
        let scripted = build_world(world);
        let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
        let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
        let mut row = Map::new();
        row.insert("kind".to_string(), json!(kind));
        match kind {
            "rail_step" => {
                match engine.step_autonomous_rail(keys, step_parameters, world_ref) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert("outcome".to_string(), rail_outcome_projection(&outcome));
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            "state6_tail" => {
                match engine.step_rail_state6_tail(keys, step_parameters, world_ref) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert("outcome".to_string(), state6_tail_projection(&outcome));
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            "raw_motion" => {
                match engine.step_raw_motion(keys, step_parameters, world_ref, None, None) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        let rail = match &outcome.kind {
                            kart_physics::engine::RawMotionKind::Rail(rail) => {
                                rail_outcome_projection(rail)
                            }
                            _ => Value::Null,
                        };
                        row.insert(
                            "outcome".to_string(),
                            json!({
                                "outcome_kind": "raw_motion",
                                "raw_motion_executor": outcome.raw_motion_executor,
                                "raw_motion_dispatch_reason": outcome.raw_motion_dispatch_reason,
                                "rail": rail,
                            }),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            _ => unreachable!("step kinds are exhaustive"),
        }
        row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
        row.insert("state".to_string(), state_row(&engine.state));
        row.insert("world_status".to_string(), world_status(scripted.as_ref()));
        rows.push(Value::Object(row));
    }
    json!({"name": name, "rows": rows})
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
            SpecCase::Engine { name, initial_state, parameters, numeric_mode, steps } => {
                out.push(run_engine_case(name, initial_state, parameters, *numeric_mode, steps));
            }
        }
    }
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
}
