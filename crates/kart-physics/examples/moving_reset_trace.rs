//! MOVING-CONTACT + RESET domain parity trace tool (M2 part 3).
//!
//! Reads a JSON spec from stdin: `{"cases": [...]}`. Case shapes:
//! - `{"kind": "engine", "name", "initial_state", "parameters",
//!   "numeric_mode", "steps": [...]}` drives the engine-level steppers. Step
//!   kinds: `moving_step` (`step_autonomous_moving_contact`), `reset_step`
//!   (`step_autonomous_reset`), `moving_acquire`
//!   (`acquire_autonomous_moving_contact`), `raw_motion` (`step_raw_motion`,
//!   covers the `_step_raw_motion_once` MOVING-CONTACT / RESET dispatch arms
//!   and the `step_raw_topology` initial moving-contact capture).
//! - `{"kind": "leaf_acquire"|"leaf_selector"|"leaf_step", "name", "args"}`
//!   drives the `domains::moving_contact_topology` leaf functions directly.
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! method name and bit-exact arguments, and returns the recorded response
//! (same contract as rail_step_trace.rs). Every f64 is emitted as its
//! IEEE-754 bit pattern (`{:016x}`); a domain `Err` is emitted as
//! `{"error": <message>}` and the row still carries the post-call state.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;

use kart_physics::domains::moving_contact_topology::{
    acquire_moving_contact_frame, evaluate_moving_contact_selector,
    evaluate_moving_contact_step, MovingContactFrame, MovingContactSelectorResult,
    MovingContactStep,
};
use kart_physics::domains::orientation_integrator::OrientationIntegratorResult;
use kart_physics::engine::{
    AcquireAutonomousMovingContactOutcome, AutonomousMovingContactOutcome,
    AutonomousResetOutcome, KartRiderVehicleEngine, NumericMode, RawMotionKind,
    TopologyExecutorOutcome,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{JsonValue, MovingContactAnchor, VehicleState};
use kart_physics::topology::TopologyState;
use kart_physics::world::{
    CheckpointRespawnCandidate, World, WorldCapabilities, WorldGroundResult,
};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Bit-pattern encoding helpers (same conventions as rail_step_trace.rs)
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

/// Open-payload encoding (Python `encode_open_scalar` on the harness side).
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
// State / anchor / outcome projections (mirror the Python verify script)
// ---------------------------------------------------------------------------

fn topology_row(topology: &TopologyState) -> Value {
    bits_json(&serde_json::to_value(topology).expect("topology serializes"))
}

fn anchor_projection(anchor: &MovingContactAnchor) -> Value {
    json!({
        "object_id": anchor.object_id,
        "local_position": arr_fbits(anchor.local_position),
        "local_velocity": arr_fbits(anchor.local_velocity),
        "relative_matrix": arr9_fbits(anchor.relative_matrix),
        "local_angular_velocity": arr_fbits(anchor.local_angular_velocity),
        "local_contact_normal": arr_fbits(anchor.local_contact_normal),
        "local_plane_offset": fbits(anchor.local_plane_offset),
        "discarded_normal_velocity": opt_fbits(anchor.discarded_normal_velocity),
        "object_point_velocity": opt_arr_fbits(anchor.object_point_velocity),
        "kinematic_contract": anchor.kinematic_contract,
    })
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
        "takeoff_separation_pending": state.takeoff_separation_pending,
        "special_trigger_cooldown_ms": fbits(state.special_trigger_cooldown_ms),
        "authored_trigger_contact_id": state.authored_trigger_contact_id,
        "surface_mode_latched": state.surface_mode_latched,
        "surface_selector_elapsed_ms": fbits(state.surface_selector_elapsed_ms),
        "checkpoint_mode_family": state.checkpoint_mode_family,
        "checkpoint_target": arr_fbits(state.checkpoint_target),
        "checkpoint_gate": state.checkpoint_gate,
        "checkpoint_gate_armed_tick_ms": state.checkpoint_gate_armed_tick_ms,
        "manual_restart_count": state.manual_restart_count,
        "moving_contact_anchor": state.moving_contact_anchor.as_ref().map(anchor_projection).unwrap_or(Value::Null),
        "time_ms": state.time_ms,
        "step_index": state.step_index,
    })
}

fn selector_projection(selector: &MovingContactSelectorResult) -> Value {
    json!({
        "sustain": selector.sustain,
        "reason": selector.reason,
        "relative_normal_velocity": opt_fbits(selector.relative_normal_velocity),
        "separation_tolerance": opt_fbits(selector.separation_tolerance),
        "normal_alignment": opt_fbits(selector.normal_alignment),
        "normal_alignment_tolerance": opt_fbits(selector.normal_alignment_tolerance),
        "object_point_velocity": opt_arr_fbits(selector.object_point_velocity),
        "authority": selector.authority,
    })
}

fn integrator_projection(result: &OrientationIntegratorResult) -> Value {
    json!({
        "quaternion_after": Value::Array(result.quaternion_after.iter().map(|&v| fbits(v)).collect()),
        "matrix_9c_after": arr9_fbits(result.matrix_9c_after),
        "angular_velocity_cc_after": arr_fbits(result.angular_velocity_cc_after),
    })
}

fn step_projection(step: &MovingContactStep) -> Value {
    json!({
        "position_after": arr_fbits(step.position_after),
        "velocity_after": arr_fbits(step.velocity_after),
        "matrix_after": arr9_fbits(step.matrix_after),
        "angular_velocity_after": arr_fbits(step.angular_velocity_after),
        "anchor_after": anchor_projection(&step.anchor_after),
        "relative_orientation_integrator": integrator_projection(&step.relative_orientation_integrator),
        "object_orientation_integrator": step.object_orientation_integrator.as_ref().map(integrator_projection).unwrap_or(Value::Null),
        "object_position_after": arr_fbits(step.object_position_after),
        "object_matrix_after": arr9_fbits(step.object_matrix_after),
        "object_state_phase": step.object_state_phase,
        "object_point_velocity": arr_fbits(step.object_point_velocity),
        "reference_frame_owner": step.reference_frame_owner,
        "contact_plane_error_before_projection": fbits(step.contact_plane_error_before_projection),
        "closure_status": step.closure_status,
    })
}

fn frame_projection(frame: &MovingContactFrame) -> Value {
    json!({
        "object_id": frame.object_id,
        "local_position": arr_fbits(frame.local_position),
        "local_velocity": arr_fbits(frame.local_velocity),
        "local_contact_normal": arr_fbits(frame.local_contact_normal),
        "local_plane_offset": fbits(frame.local_plane_offset),
        "discarded_normal_velocity": fbits(frame.discarded_normal_velocity),
        "relative_matrix": arr9_fbits(frame.relative_matrix),
        "local_angular_velocity": arr_fbits(frame.local_angular_velocity),
        "object_point_velocity": arr_fbits(frame.object_point_velocity),
        "kinematic_contract": frame.kinematic_contract,
    })
}

fn moving_outcome_projection(outcome: &AutonomousMovingContactOutcome) -> Value {
    match outcome {
        AutonomousMovingContactOutcome::Sustained(sustained) => json!({
            "outcome_kind": "sustained",
            "released": sustained.released,
            "selector": selector_projection(&sustained.moving_contact_selector),
            "step": step_projection(&sustained.step.step),
            "contract": {
                "reference_frame_authority": sustained.step.moving_contact_contract.reference_frame_authority,
                "object_motion_authority": sustained.step.moving_contact_contract.object_motion_authority,
            },
        }),
        AutonomousMovingContactOutcome::ResetRelease(release) => json!({
            "outcome_kind": "reset_release",
            "released": release.released,
            "selector": selector_projection(&release.moving_contact_selector),
            "reset_surface_entry": release.reset.reset_surface_entry,
            "exit_authority": release.exit_authority,
        }),
        AutonomousMovingContactOutcome::Released(release) => json!({
            "outcome_kind": "released",
            "released": release.released,
            "selector": selector_projection(&release.moving_contact_selector),
            "support_contact_count": release.support_observation.support_contact_count,
            "surface_mode": release.support_observation.surface_mode,
            "exit_authority": release.exit_authority,
        }),
    }
}

fn replacement_projection(
    replacement: &kart_physics::domains::checkpoint_respawn_state_replacement::CheckpointRespawnStateReplacement,
) -> Value {
    json!({
        "selected": replacement.selected,
        "mode_family": replacement.mode_family,
        "checkpoint_gate": replacement.checkpoint_gate,
        "sample_result_zero": replacement.sample_result_zero,
        "forward_axis1": arr_fbits(replacement.forward_axis1),
        "right_axis0": arr_fbits(replacement.right_axis0),
        "up_axis2": arr_fbits(replacement.up_axis2),
        "position_90_after": opt_arr_fbits(replacement.position_90_after),
        "matrix_9c_after": opt_arr9_fbits(replacement.matrix_9c_after),
        "velocity_c0_after": arr_fbits(replacement.velocity_c0_after),
        "angular_velocity_cc_after": arr_fbits(replacement.angular_velocity_cc_after),
    })
}

fn reset_outcome_projection(outcome: &AutonomousResetOutcome) -> Value {
    match outcome {
        AutonomousResetOutcome::NotReady(not_ready) => json!({
            "outcome_kind": "not_ready",
            "selected": not_ready.selected,
            "reason": not_ready.reason,
            "selector_authority": not_ready.selector_authority,
        }),
        AutonomousResetOutcome::Applied(applied) => json!({
            "outcome_kind": "applied",
            "replacement": replacement_projection(&applied.respawn.replacement),
            "selector": {
                "course_reference_position": arr_fbits(applied.checkpoint_respawn_selector.course_reference_position),
                "sample_result_zero": applied.checkpoint_respawn_selector.sample_result_zero,
                "authority": applied.checkpoint_respawn_selector.authority,
            },
            "selector_authority": applied.selector_authority,
            "support_contact_count": applied.respawn.support_observation.as_ref().map(|support| json!(support.support_contact_count)).unwrap_or(Value::Null),
            "topology_transition_reason": applied.respawn.topology_transition.as_ref().map(|decision| json!(decision.reason)).unwrap_or(Value::Null),
        }),
    }
}

fn acquire_outcome_projection(outcome: &AcquireAutonomousMovingContactOutcome) -> Value {
    match outcome {
        AcquireAutonomousMovingContactOutcome::NotActive { .. } => json!({
            "acquired": false,
        }),
        AcquireAutonomousMovingContactOutcome::Acquired(acquired) => json!({
            "acquired": true,
            "anchor": anchor_projection(&acquired.acquisition.anchor),
            "entry_authority": acquired.entry_authority,
            "entry_transfer": acquired.acquisition.entry_transfer,
        }),
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
    #[serde(rename = "leaf_acquire")]
    LeafAcquire { name: String, args: Value },
    #[serde(rename = "leaf_selector")]
    LeafSelector { name: String, args: Value },
    #[serde(rename = "leaf_step")]
    LeafStep { name: String, args: Value },
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum SpecStep {
    #[serde(rename = "moving_step")]
    MovingStep {
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "reset_step")]
    ResetStep {
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "moving_acquire")]
    MovingAcquire {
        #[serde(default)]
        world: Option<SpecWorld>,
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
    query_moving_contact: bool,
    #[serde(default)]
    query_checkpoint_respawn: bool,
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
// Scripted replay world (same contract as rail_step_trace.rs)
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

    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let args = json!({
            "position": arr_fbits(position),
            "time_ms": time_ms,
            "object_id": object_id,
            "delta_ms": delta_ms,
            "max_distance": fbits(max_distance),
        });
        let response = self.next_call("query_moving_contact", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_moving_contact response decodes"),
        )
    }

    fn query_checkpoint_respawn(
        &self,
        position: [f64; 3],
        checkpoint_target: [f64; 3],
        checkpoint_gate: bool,
        mode_family: &str,
        max_course_distance: f64,
    ) -> Option<Option<CheckpointRespawnCandidate>> {
        let args = json!({
            "position": arr_fbits(position),
            "checkpoint_target": arr_fbits(checkpoint_target),
            "checkpoint_gate": checkpoint_gate,
            "mode_family": mode_family,
            "max_course_distance": fbits(max_course_distance),
        });
        let response = self.next_call("query_checkpoint_respawn", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_checkpoint_respawn response decodes"),
        )
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
}

fn build_world(spec: &Option<SpecWorld>) -> Option<ScriptedWorld> {
    spec.as_ref().map(|world| ScriptedWorld {
        capabilities: WorldCapabilities {
            query_moving_contact: world.capabilities.query_moving_contact,
            query_checkpoint_respawn: world.capabilities.query_checkpoint_respawn,
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
        let scripted = match step {
            SpecStep::MovingStep { world, .. }
            | SpecStep::ResetStep { world, .. }
            | SpecStep::MovingAcquire { world }
            | SpecStep::RawMotion { world, .. } => build_world(world),
        };
        let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
        let mut row = Map::new();
        match step {
            SpecStep::MovingStep { parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("moving_step"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match engine.step_autonomous_moving_contact(step_parameters, world_ref) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert(
                            "outcome".to_string(),
                            moving_outcome_projection(&outcome),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            SpecStep::ResetStep { parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("reset_step"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match engine.step_autonomous_reset(step_parameters, world_ref) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert(
                            "outcome".to_string(),
                            reset_outcome_projection(&outcome),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            SpecStep::MovingAcquire { .. } => {
                row.insert("kind".to_string(), json!("moving_acquire"));
                match engine.acquire_autonomous_moving_contact(world_ref) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        row.insert(
                            "outcome".to_string(),
                            acquire_outcome_projection(&outcome),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
            SpecStep::RawMotion { keys, parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("raw_motion"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match engine.step_raw_motion(keys, step_parameters, world_ref, None, None) {
                    Ok(outcome) => {
                        row.insert("error".to_string(), Value::Null);
                        let detail = match &outcome.kind {
                            RawMotionKind::Topology(topology) => match &topology.executor {
                                TopologyExecutorOutcome::MovingContactCapture(capture) => json!({
                                    "kind": "moving_capture",
                                    "step": step_projection(&capture.step.step),
                                    "anchor": anchor_projection(&capture.acquisition.anchor),
                                    "selector": json_map_encoded(&capture.moving_contact_selector),
                                    "contract": {
                                        "reference_frame_authority": capture.step.moving_contact_contract.reference_frame_authority,
                                        "object_motion_authority": capture.step.moving_contact_contract.object_motion_authority,
                                    },
                                }),
                                TopologyExecutorOutcome::Ground(_) => json!({"kind": "topology", "executor": "ground"}),
                                TopologyExecutorOutcome::Airborne(_) => json!({"kind": "topology", "executor": "airborne"}),
                                TopologyExecutorOutcome::ResetSurface(_) => json!({"kind": "topology", "executor": "reset_surface"}),
                                TopologyExecutorOutcome::RailCapture(_) => json!({"kind": "topology", "executor": "rail_capture"}),
                                TopologyExecutorOutcome::NetworkCorrection(_) => json!({"kind": "topology", "executor": "network_correction"}),
                            },
                            RawMotionKind::MovingContact(moving) => json!({
                                "kind": "moving_step",
                                "moving": moving_outcome_projection(moving),
                            }),
                            RawMotionKind::Reset(reset) => json!({
                                "kind": "reset_step",
                                "reset": reset_outcome_projection(reset),
                            }),
                            RawMotionKind::FallClampReset(_) => json!({"kind": "fall_clamp"}),
                            RawMotionKind::ManualRestart(_) => json!({"kind": "manual_restart"}),
                            RawMotionKind::Rail(_) => json!({"kind": "rail"}),
                            RawMotionKind::NetworkCorrection(_) | RawMotionKind::NetworkCorrectionPreemption(_) => {
                                json!({"kind": "network_correction"})
                            }
                        };
                        row.insert(
                            "outcome".to_string(),
                            json!({
                                "outcome_kind": "raw_motion",
                                "raw_motion_executor": outcome.raw_motion_executor,
                                "raw_motion_dispatch_reason": outcome.raw_motion_dispatch_reason,
                                "detail": detail,
                            }),
                        );
                    }
                    Err(error) => {
                        row.insert("error".to_string(), json!(error));
                        row.insert("outcome".to_string(), Value::Null);
                    }
                }
            }
        }
        row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
        row.insert("state".to_string(), state_row(&engine.state));
        row.insert("world_status".to_string(), world_status(scripted.as_ref()));
        rows.push(Value::Object(row));
    }
    json!({"name": name, "rows": rows})
}

// ---------------------------------------------------------------------------
// Leaf case runners
// ---------------------------------------------------------------------------

fn value_f64(value: &Value, key: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("leaf arg {key} must be a number"))
}

fn value_vec3(value: &Value, key: &str) -> [f64; 3] {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("leaf arg {key} must be an array"));
    assert!(items.len() == 3, "leaf arg {key} must have three components");
    [
        value_f64(&items[0], key),
        value_f64(&items[1], key),
        value_f64(&items[2], key),
    ]
}

fn value_mat9(value: &Value, key: &str) -> [f64; 9] {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("leaf arg {key} must be an array"));
    assert!(items.len() == 9, "leaf arg {key} must have nine components");
    let mut out = [0.0; 9];
    for (index, item) in items.iter().enumerate() {
        out[index] = value_f64(item, key);
    }
    out
}

fn arg<'a>(args: &'a Value, key: &str) -> &'a Value {
    args.get(key)
        .unwrap_or_else(|| panic!("leaf args missing {key}"))
}

fn run_leaf_acquire(name: &str, args: &Value) -> Value {
    let result = acquire_moving_contact_frame(
        value_vec3(arg(args, "vehicle_position"), "vehicle_position"),
        value_vec3(arg(args, "vehicle_velocity"), "vehicle_velocity"),
        value_mat9(arg(args, "vehicle_matrix"), "vehicle_matrix"),
        value_vec3(arg(args, "vehicle_angular_velocity"), "vehicle_angular_velocity"),
        arg(args, "object_id").as_str().expect("object_id is a string"),
        value_vec3(arg(args, "object_position"), "object_position"),
        value_vec3(arg(args, "object_velocity"), "object_velocity"),
        value_mat9(arg(args, "object_matrix"), "object_matrix"),
        value_vec3(arg(args, "object_angular_velocity"), "object_angular_velocity"),
        value_vec3(arg(args, "contact_normal_world"), "contact_normal_world"),
        value_f64(arg(args, "separation_tolerance"), "separation_tolerance"),
    );
    let (error, outcome) = match result {
        Ok(frame) => (Value::Null, frame_projection(&frame)),
        Err(message) => (json!(message), Value::Null),
    };
    json!({"name": name, "rows": [{"kind": "leaf_acquire", "error": error, "outcome": outcome}]})
}

fn run_leaf_selector(name: &str, args: &Value) -> Value {
    let anchor: MovingContactAnchor =
        serde_json::from_value(arg(args, "anchor").clone()).expect("anchor decodes");
    let result = evaluate_moving_contact_selector(
        &anchor,
        value_vec3(arg(args, "vehicle_position"), "vehicle_position"),
        value_vec3(arg(args, "vehicle_velocity"), "vehicle_velocity"),
        arg(args, "object_id").as_str().expect("object_id is a string"),
        value_vec3(arg(args, "object_position"), "object_position"),
        value_vec3(arg(args, "object_velocity"), "object_velocity"),
        value_mat9(arg(args, "object_matrix"), "object_matrix"),
        value_vec3(arg(args, "object_angular_velocity"), "object_angular_velocity"),
        value_vec3(arg(args, "contact_normal_world"), "contact_normal_world"),
        arg(args, "contact_active").as_bool().expect("contact_active is a bool"),
        value_f64(arg(args, "separation_tolerance"), "separation_tolerance"),
        value_f64(
            arg(args, "normal_alignment_tolerance"),
            "normal_alignment_tolerance",
        ),
    );
    let (error, outcome) = match result {
        Ok(selector) => (Value::Null, selector_projection(&selector)),
        Err(message) => (json!(message), Value::Null),
    };
    json!({"name": name, "rows": [{"kind": "leaf_selector", "error": error, "outcome": outcome}]})
}

fn run_leaf_step(name: &str, args: &Value) -> Value {
    let anchor: MovingContactAnchor =
        serde_json::from_value(arg(args, "anchor").clone()).expect("anchor decodes");
    let result = evaluate_moving_contact_step(
        &anchor,
        arg(args, "object_id").as_str().expect("object_id is a string"),
        value_vec3(arg(args, "object_position"), "object_position"),
        value_vec3(arg(args, "object_velocity"), "object_velocity"),
        value_mat9(arg(args, "object_matrix"), "object_matrix"),
        value_vec3(arg(args, "object_angular_velocity"), "object_angular_velocity"),
        value_f64(arg(args, "dt"), "dt"),
        arg(args, "numeric_mode")
            .as_str()
            .expect("numeric_mode is a string"),
        arg(args, "object_state_phase")
            .as_str()
            .expect("object_state_phase is a string"),
    );
    let (error, outcome) = match result {
        Ok(step) => (Value::Null, step_projection(&step)),
        Err(message) => (json!(message), Value::Null),
    };
    json!({"name": name, "rows": [{"kind": "leaf_step", "error": error, "outcome": outcome}]})
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
            SpecCase::LeafAcquire { name, args } => {
                out.push(run_leaf_acquire(name, args));
            }
            SpecCase::LeafSelector { name, args } => {
                out.push(run_leaf_selector(name, args));
            }
            SpecCase::LeafStep { name, args } => {
                out.push(run_leaf_step(name, args));
            }
        }
    }
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
}
