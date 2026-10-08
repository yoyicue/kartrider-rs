//! Engine control-path parity trace tool (step_control / step_raw_control).
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"initial_state": ..., "parameters": ..., "numeric_mode":
//! "float32"|"float64", "control_model": <trace response model payload>,
//! "steps": [...]}...]}`.
//!
//! Step shapes:
//! - `{"kind": "raw_control", "keys": [...], "substep_ms": N,
//!   "world": <null|{"capabilities": {...}, "call_log": [...]}>}` — drives
//!   `step_raw_control` (world=None or the scripted replay world).
//! - `{"kind": "control", "frame": <replay frame dict including the
//!   "control" payload>}` — drives `step_control` (world=None only; the
//!   step_world control delegation is exercised through `raw_control`
//!   steps).
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! the method name and bit-exact arguments, and returns the recorded
//! response (same contract as raw_frame_trace.rs). Every f64 is emitted as
//! its IEEE-754 bit pattern; a domain `Err` is emitted as
//! `{"kind": "error", "error": <message>}` and ends the case.

use std::cell::RefCell;
use std::io::Read;

use kart_physics::domains::control_response_input::ControlResponseInputModel;
use kart_physics::domains::drive_basis_input::DriveBasisInput;
use kart_physics::engine::{
    ControlDriveTarget, ControlPayload, ControlPrediction, KartRiderVehicleEngine, NumericMode,
    StepFrame, WorldSweepInfo,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use kart_physics::topology::TopologyDecision;
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult, WorldObbOverlap};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Spec decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Spec {
    cases: Vec<SpecCase>,
}

#[derive(Deserialize)]
struct SpecCase {
    initial_state: VehicleState,
    parameters: VehicleParameters,
    numeric_mode: NumericMode,
    control_model: Value,
    steps: Vec<SpecStep>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum SpecStep {
    #[serde(rename = "raw_control")]
    RawControl {
        keys: Vec<String>,
        substep_ms: i64,
        #[serde(default)]
        world: Option<SpecWorld>,
    },
    #[serde(rename = "control")]
    Control { frame: Value },
}

#[derive(Deserialize, Clone)]
struct SpecWorld {
    #[serde(default)]
    capabilities: SpecCapabilities,
    #[serde(default)]
    call_log: Vec<SpecCall>,
}

#[derive(Deserialize, Clone, Default)]
struct SpecCapabilities {
    #[serde(default)]
    sweep_obb_rotating_all: bool,
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
// Encoding helpers (same conventions as the other trace examples)
// ---------------------------------------------------------------------------

fn fbits(value: f64) -> Value {
    Value::String(format!("{:016x}", value.to_bits()))
}

fn arr<const N: usize>(values: &[f64; N]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
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
                .map(|(key, item)| (key.clone(), bits_json(item)))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn encode_state(state: &VehicleState) -> Value {
    bits_json(&serde_json::to_value(state).expect("VehicleState serializes"))
}

fn encode_decision(decision: &TopologyDecision) -> Value {
    json!({
        "selected_domain": decision.selected_domain.value(),
        "transitioned": decision.transitioned,
        "reason": decision.reason,
        "domain_before": decision.state_before.domain.value(),
        "domain_after": decision.state_after.domain.value(),
    })
}

fn encode_world_sweep(sweep: Option<&WorldSweepInfo>) -> Value {
    match sweep {
        None => Value::Null,
        Some(sweep) => json!({
            "dispatch_reason": sweep.dispatch_reason,
            "response_dispatched": sweep.response_dispatched,
            "candidate_count": sweep.candidates.len(),
            "topology_transition": match &sweep.topology_transition {
                Some(decision) => encode_decision(decision),
                None => Value::Null,
            },
        }),
    }
}

fn encode_drive_basis(basis: &DriveBasisInput) -> Value {
    json!({
        "drive_raw_aa0": fbits(basis.drive_raw_aa0),
        "drive_sign_aac": basis.drive_sign_aac,
        "drive_signed_a80": fbits(basis.drive_signed_a80),
        "drive_slot0_a84": fbits(basis.drive_slot0_a84),
        "drive_slot1_a90": fbits(basis.drive_slot1_a90),
        "drive_slot_selector_a9c": basis.drive_slot_selector_a9c,
        "drive_selected_slot": fbits(basis.drive_selected_slot),
        "drive_opposite_slot": fbits(basis.drive_opposite_slot),
        "dispatch_cases": {
            "left_right_signed_axis": basis.dispatch_left_right_signed_axis,
            "up_slot0": basis.dispatch_up_slot0,
            "down_slot1": basis.dispatch_down_slot1,
        },
        "selection_authority": kart_physics::domains::drive_basis_input::SELECTION_AUTHORITY,
        "scope": kart_physics::domains::drive_basis_input::SCOPE,
    })
}

fn encode_drive_target(target: &ControlDriveTarget) -> Value {
    let state = match target {
        ControlDriveTarget::State(result) => result,
        ControlDriveTarget::Runtime(result) => &result.state,
    };
    let mut map = Map::new();
    map.insert("raw_target".to_string(), fbits(state.raw_target));
    map.insert("exponent_input".to_string(), fbits(state.exponent_input));
    map.insert("attenuation".to_string(), fbits(state.attenuation));
    map.insert("candidate_aa4".to_string(), fbits(state.candidate_aa4));
    map.insert("field_aa4_after".to_string(), fbits(state.field_aa4_after));
    map.insert("field_aa8_after".to_string(), fbits(state.field_aa8_after));
    map.insert(
        "overshoot_clamped_to_aa8".to_string(),
        json!(state.overshoot_clamped_to_aa8),
    );
    map.insert("ordering".to_string(), json!(state.ordering));
    if let ControlDriveTarget::Runtime(result) = target {
        map.insert("field_718".to_string(), fbits(result.field_718));
        map.insert("scaled_tune_660".to_string(), fbits(result.scaled_tune_660));
        map.insert("tune_accessor".to_string(), json!(result.tune_accessor));
    }
    Value::Object(map)
}

fn encode_prediction(prediction: &ControlPrediction) -> Value {
    let mut map = Map::new();
    map.insert("action".to_string(), json!(prediction.action));
    if let Some(band) = prediction.speed_band {
        map.insert("speed_band".to_string(), json!(band));
    }
    if let Some(speed) = prediction.speed {
        map.insert("speed".to_string(), fbits(speed));
    }
    if let Some(accel) = prediction.longitudinal_accel_mps2 {
        map.insert("longitudinal_accel_mps2".to_string(), fbits(accel));
    }
    if let Some(rate) = prediction.heading_rate_rad_s {
        map.insert("heading_rate_rad_s".to_string(), fbits(rate));
    }
    map.insert(
        "steering_selected".to_string(),
        json!(prediction.steering_selected),
    );
    if let Some(world) = &prediction.acceleration_world {
        map.insert("acceleration_world".to_string(), arr(world));
    }
    if let Some(force) = &prediction.force_accumulator_614 {
        map.insert("force_accumulator_614".to_string(), arr(force));
    }
    map.insert("source".to_string(), json!(prediction.source));
    if let Some(scope) = prediction.heading_rate_scope {
        map.insert("heading_rate_scope".to_string(), json!(scope));
    }
    map.insert("scope".to_string(), json!(prediction.scope));
    map.insert(
        "drive_basis".to_string(),
        encode_drive_basis(&prediction.drive_basis),
    );
    map.insert("input_mode".to_string(), json!(prediction.input_mode));
    if let Some(target) = &prediction.drive_target {
        map.insert("drive_target".to_string(), encode_drive_target(target));
    }
    if let Some(adapter) = prediction.ground_runtime_adapter {
        map.insert("ground_runtime_adapter".to_string(), json!(adapter));
    }
    if let Some(branch) = prediction.ground_runtime_branch {
        map.insert("ground_runtime_branch".to_string(), json!(branch));
    }
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// Scripted world (same contract as raw_frame_trace.rs)
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
            self.mismatches.borrow_mut().push(format!(
                "call {index}: method {} != {method}",
                entry.method
            ));
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
            "mismatches": Value::Array(mismatches.iter().map(|m| json!(m)).collect()),
        })
    }
}

impl World for ScriptedWorld {
    fn capabilities(&self) -> WorldCapabilities {
        self.capabilities
    }

    fn query_ground(&self, _position: [f64; 3], _up: [f64; 3], _min_dot: f64) -> WorldGroundResult {
        self.mismatches
            .borrow_mut()
            .push("query_ground called on the control-path scripted world".to_string());
        WorldGroundResult::default()
    }

    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let args = json!({
            "start": arr(&start),
            "end": arr(&end),
            "start_matrix": arr(&start_matrix),
            "end_matrix": arr(&end_matrix),
            "half_extents": arr(&half_extents),
        });
        let response = self.next_call("sweep_obb_rotating_all", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded sweep_obb_rotating_all response decodes"),
        )
    }
}

fn empty_world_log() -> Value {
    json!({"consumed": 0, "total": 0, "mismatches": Value::Array(vec![])})
}

// ---------------------------------------------------------------------------
// Step dispatch
// ---------------------------------------------------------------------------

fn run_case(case: &SpecCase) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    let mut engine = match KartRiderVehicleEngine::new(&case.initial_state, case.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => {
            return json!({"rows": [{"kind": "error", "error": message}]});
        }
    };
    let model = match ControlResponseInputModel::from_json_str(&case.control_model.to_string()) {
        Ok(model) => model,
        Err(message) => {
            return json!({"rows": [{"kind": "error", "error": message}]});
        }
    };
    for step in &case.steps {
        match step {
            SpecStep::RawControl {
                keys,
                substep_ms,
                world,
            } => {
                let parameters = VehicleParameters {
                    substep_ms: *substep_ms,
                    ..case.parameters.clone()
                };
                let scripted = world.as_ref().map(|world| ScriptedWorld {
                    capabilities: WorldCapabilities {
                        sweep_obb_rotating_all: world.capabilities.sweep_obb_rotating_all,
                        query_at_step_start: world.capabilities.query_at_step_start,
                        ..WorldCapabilities::default()
                    },
                    call_log: world.call_log.clone(),
                    cursor: RefCell::new(0),
                    mismatches: RefCell::new(Vec::new()),
                });
                let world_ref: Option<&dyn World> = match &scripted {
                    Some(scripted) => Some(scripted),
                    None => None,
                };
                match engine.step_raw_control(keys, &parameters, &model, world_ref) {
                    Ok(outcome) => {
                        let world_log = match &scripted {
                            Some(scripted) => scripted.status(),
                            None => empty_world_log(),
                        };
                        rows.push(json!({
                            "kind": "raw_control",
                            "state_before": encode_state(&outcome.step.state_before),
                            "state_after": encode_state(&outcome.step.state_after),
                            "engine_state_after": encode_state(&engine.state),
                            "teacher_forced_before": outcome.step.teacher_forced_before,
                            "accumulator_input_applied": outcome.step.accumulator_input_applied,
                            "kinematic_input_applied": outcome.step.kinematic_input_applied,
                            "control_prediction": encode_prediction(&outcome.control_prediction),
                            "raw_control_pose_contract": match &outcome.raw_control_pose_contract {
                                Some(contract) => bits_json(
                                    &serde_json::to_value(contract)
                                        .expect("pose contract serializes"),
                                ),
                                None => Value::Null,
                            },
                            "world_sweep": encode_world_sweep(outcome.world_sweep.as_ref()),
                            "world_log": world_log,
                        }));
                    }
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                }
            }
            SpecStep::Control { frame } => {
                let control_value = frame.get("control").cloned().unwrap_or(Value::Null);
                let control: ControlPayload = match serde_json::from_value(control_value) {
                    Ok(control) => control,
                    Err(error) => {
                        rows.push(json!({"kind": "error", "error": format!("control payload decode: {error}")}));
                        break;
                    }
                };
                let frame: StepFrame = match serde_json::from_value(frame.clone()) {
                    Ok(frame) => frame,
                    Err(error) => {
                        rows.push(json!({"kind": "error", "error": format!("frame decode: {error}")}));
                        break;
                    }
                };
                match engine.step_control(&frame, &control, Some(&model)) {
                    Ok(outcome) => {
                        rows.push(json!({
                            "kind": "control",
                            "state_before": encode_state(&outcome.step.state_before),
                            "state_after": encode_state(&outcome.step.state_after),
                            "engine_state_after": encode_state(&engine.state),
                            "teacher_forced_before": outcome.step.teacher_forced_before,
                            "accumulator_input_applied": outcome.step.accumulator_input_applied,
                            "kinematic_input_applied": outcome.step.kinematic_input_applied,
                            "control_prediction": encode_prediction(&outcome.control_prediction),
                            "world_sweep": encode_world_sweep(outcome.world_sweep.as_ref()),
                            "world_log": empty_world_log(),
                        }));
                    }
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                }
            }
        }
    }
    json!({"rows": rows})
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Spec = serde_json::from_str(&input).expect("parse spec JSON");
    let rows: Vec<Value> = spec.cases.iter().map(run_case).collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
