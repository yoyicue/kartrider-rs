//! Engine CONTACT-domain parity trace tool (step_world collision chain).
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"initial_state": ..., "parameters": ..., "numeric_mode":
//! "float32"|"float64", "control_model": <trace response model payload|null>,
//! "steps": [...]}...]}`.
//!
//! Step shapes:
//! - `{"kind": "raw_control", "keys": [...], "substep_ms": N,
//!   "world": <null|{"capabilities": {...}, "call_log": [...]}>}` — drives
//!   `step_raw_control` (world=None or the scripted replay world), whose GND
//!   path dispatches through `step_world`.
//! - `{"kind": "world_step", "frame": <replay frame dict>, "world": {...},
//!   "collision_selector_state": <overlay|null>}` — drives `step_world`
//!   directly with a caller-built frame (covers frames that already carry a
//!   `model.hard_contact`/`model.soft_contact` payload).
//! - `{"kind": "chain", ...}` — drives `_apply_collision_chain` directly
//!   (synthetic candidate lists; mixed/pure-soft/pure-hard/state6
//!   parameterizations).
//! - `{"kind": "finalize", "frame": ..., "chain": {...},
//!   "finalize": {...}}` — runs `step`, then `_apply_collision_chain`, then
//!   `_finalize_collision_chain` with the explicit finalize parameters.
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! the method name and bit-exact arguments, and returns the recorded
//! response (same contract as control_trace.rs). Every f64 is emitted as
//! its IEEE-754 bit pattern; a domain `Err` is emitted as
//! `{"kind": "error", "error": <message>}` and ends the case.

use std::cell::RefCell;
use std::io::Read;

use kart_physics::domains::control_response_input::ControlResponseInputModel;
use kart_physics::domains::hard_contact_angular_response::HardContactAngularResponse;
use kart_physics::domains::hard_contact_velocity_response::HardContactVelocityResponse;
use kart_physics::domains::soft_contact_angular_response::SoftContactAngularSource;
use kart_physics::domains::soft_contact_source::{SoftContactBaseSource, SoftContactSource};
use kart_physics::engine::{
    CollisionChainAngularResponse, CollisionChainOutcome, CollisionChainResponse,
    CollisionChainSelectorState, CollisionChainVelocityResponse, CollisionDispatchSelection,
    CollisionSelectorState, ControlPayload, KartRiderVehicleEngine, NumericMode, SoftBaseResponse,
    StepFrame, WorldSweepInfo,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use kart_physics::topology::TopologyDecision;
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult, WorldObbOverlap};
use serde::Deserialize;
use serde_json::{json, Value};

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
    #[serde(default)]
    control_model: Option<Value>,
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
    #[serde(rename = "world_step")]
    WorldStep {
        frame: Value,
        #[serde(default)]
        world: SpecWorld,
        #[serde(default)]
        collision_selector_state: Option<CollisionSelectorState>,
    },
    #[serde(rename = "chain")]
    Chain { chain: ChainSpec },
    #[serde(rename = "finalize")]
    Finalize {
        frame: Value,
        chain: ChainSpec,
        finalize: Box<FinalizeSpec>,
    },
}

#[derive(Deserialize, Clone)]
struct ChainSpec {
    candidates: Vec<WorldObbOverlap>,
    start_matrix: [f64; 9],
    end_matrix: [f64; 9],
    velocity: [f64; 3],
    angular_velocity: [f64; 3],
    selector_state: CollisionChainSelectorState,
    #[serde(default)]
    allowed_branch: Option<String>,
    stop_on_disallowed: bool,
    quantize_each: bool,
}

#[derive(Deserialize, Clone)]
struct FinalizeSpec {
    position_before: [f64; 3],
    matrix_before: [f64; 9],
    end_matrix: [f64; 9],
    end_position: [f64; 3],
    candidates: Vec<WorldObbOverlap>,
    chain_key: String,
    dispatch_reason: String,
    composition_authority: String,
    soft_base_from_last: bool,
    quantize_final: bool,
}

#[derive(Deserialize, Clone, Default)]
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

fn encode_candidate(candidate: &WorldObbOverlap) -> Value {
    bits_json(&serde_json::to_value(candidate).expect("WorldObbOverlap serializes"))
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

fn encode_selection(selection: &CollisionDispatchSelection) -> Value {
    json!({
        "selected": selection.selected,
        "reason": selection.reason,
        "branch": selection.branch,
        "approach": selection.approach.map(fbits),
        "normal": selection.normal.map(|normal| arr(&normal)),
        "selector": selection.selector.as_ref().map(|selector| json!({
            "branch": selector.branch,
            "normal_z": fbits(selector.normal_z),
            "arg_threshold": fbits(selector.arg_threshold),
            "e38": selector.e38,
            "e4c": selector.e4c,
            "static_condition": selector.static_condition,
        })),
    })
}

fn encode_hard_velocity(response: &HardContactVelocityResponse) -> Value {
    json!({
        "velocity_c0_before": arr(&response.velocity_c0_before),
        "velocity_c0_after": arr(&response.velocity_c0_after),
        "normal_dot_velocity": fbits(response.normal_dot_velocity),
        "normal_component": arr(&response.normal_component),
        "tangent_velocity": arr(&response.tangent_velocity),
        "basis_ad0_dot_normal": fbits(response.basis_ad0_dot_normal),
        "reflected_scale": fbits(response.reflected_scale),
        "direct_velocity_replacement": response.direct_velocity_replacement,
        "runtime_evidence": response.runtime_evidence,
    })
}

fn encode_soft_source(response: &SoftContactSource) -> Value {
    json!({
        "velocity_source": arr(&response.velocity_source),
        "normal_component": arr(&response.normal_component),
        "tangent_component": arr(&response.tangent_component),
        "tangent_remove": fbits(response.tangent_remove),
        "formula": response.formula,
        "optional_active": response.optional_active,
        "optional": response.optional.as_ref().map(|optional| json!({
            "phase": fbits(optional.phase),
            "side_test": fbits(optional.side_test),
            "scale": fbits(optional.scale),
            "turn_sign": fbits(optional.turn_sign),
            "velocity_extra": arr(&optional.velocity_extra),
            "angular_z_delta": fbits(optional.angular_z_delta),
        })),
        "angular_z_delta": fbits(response.angular_z_delta),
    })
}

fn encode_soft_base(response: &SoftContactBaseSource) -> Value {
    json!({
        "velocity_source": arr(&response.velocity_source),
        "normal_component": arr(&response.normal_component),
        "tangent_component": arr(&response.tangent_component),
        "tangent_remove": fbits(response.tangent_remove),
        "formula": response.formula,
    })
}

fn encode_hard_angular(response: &HardContactAngularResponse) -> Value {
    json!({
        "angular_velocity_before": arr(&response.angular_velocity_before),
        "angular_velocity_after": arr(&response.angular_velocity_after),
        "angular_delta": arr(&response.angular_delta),
        "damping": fbits(response.damping),
    })
}

fn encode_soft_angular(response: &SoftContactAngularSource) -> Value {
    json!({
        "kind": response.kind,
        "source": arr(&response.source),
        "writeback_selected": response.writeback_selected,
        "gate_dot": fbits(response.gate_dot),
        "writeback_gate": response.writeback_gate,
        "normal_ad0": fbits(response.normal_ad0),
        "normal_ab8": fbits(response.normal_ab8),
        "normal_velocity": fbits(response.normal_velocity),
        "scale": fbits(response.scale),
        "numeric_mode": response.numeric_mode,
    })
}

fn encode_chain_velocity_response(response: &CollisionChainVelocityResponse) -> Value {
    match response {
        CollisionChainVelocityResponse::Hard(value) => {
            json!({"type": "hard", "value": encode_hard_velocity(value)})
        }
        CollisionChainVelocityResponse::Soft(value) => {
            json!({"type": "soft", "value": encode_soft_source(value)})
        }
    }
}

fn encode_angular_response(response: &CollisionChainAngularResponse) -> Value {
    match response {
        CollisionChainAngularResponse::Hard(value) => {
            json!({"type": "hard", "value": encode_hard_angular(value)})
        }
        CollisionChainAngularResponse::Soft(value) => {
            json!({"type": "soft", "value": encode_soft_angular(value)})
        }
    }
}

fn encode_soft_base_response(response: &SoftBaseResponse) -> Value {
    match response {
        SoftBaseResponse::Full(value) => json!({"type": "full", "value": encode_soft_source(value)}),
        SoftBaseResponse::Base(value) => json!({"type": "base", "value": encode_soft_base(value)}),
    }
}

fn encode_chain_response(response: &CollisionChainResponse) -> Value {
    json!({
        "loop_index": response.loop_index,
        "branch": response.branch,
        "candidate": encode_candidate(&response.candidate),
        "selection": encode_selection(&response.selection),
        "velocity_before": arr(&response.velocity_before),
        "velocity_after": arr(&response.velocity_after),
        "angular_velocity_before": arr(&response.angular_velocity_before),
        "angular_velocity_after": arr(&response.angular_velocity_after),
        "velocity_response": encode_chain_velocity_response(&response.velocity_response),
        "angular_response": encode_angular_response(&response.angular_response),
    })
}

fn encode_chain_outcome(outcome: &CollisionChainOutcome) -> Value {
    json!({
        "responses": outcome.responses.iter().map(encode_chain_response).collect::<Vec<_>>(),
        "branches": outcome.branches.iter().collect::<Vec<_>>(),
        "crossed_branch": outcome.crossed_branch,
        "velocity_after": arr(&outcome.velocity_after),
        "angular_velocity_after": arr(&outcome.angular_velocity_after),
    })
}

fn encode_world_sweep(sweep: &WorldSweepInfo) -> Value {
    json!({
        "shape": sweep.shape,
        "start": arr(&sweep.start),
        "end": arr(&sweep.end),
        "candidate": sweep.candidate.as_ref().map(encode_candidate),
        "candidates": sweep.candidates.iter().map(encode_candidate).collect::<Vec<_>>(),
        "response_dispatched": sweep.response_dispatched,
        "response_complete": sweep.response_complete,
        "dispatch_reason": sweep.dispatch_reason,
        "dispatch_authority": sweep.dispatch_authority,
        "angular_response": sweep.angular_response.as_ref().map(encode_angular_response),
        "soft_base_response": sweep.soft_base_response.as_ref().map(encode_soft_base_response),
        "chain": sweep.chain.as_ref().map(|chain| json!({
            "chain_key": chain.chain_key,
            "responses": chain.responses.iter().map(encode_chain_response).collect::<Vec<_>>(),
            "composition_authority": chain.composition_authority,
            "hard_response_chain_empty": chain.hard_response_chain_empty,
        })),
        "topology_transition": sweep.topology_transition.as_ref().map(encode_decision),
    })
}

// ---------------------------------------------------------------------------
// Scripted world (same contract as control_trace.rs)
// ---------------------------------------------------------------------------

struct ScriptedWorld {
    capabilities: WorldCapabilities,
    call_log: Vec<SpecCall>,
    cursor: RefCell<usize>,
    mismatches: RefCell<Vec<String>>,
}

impl ScriptedWorld {
    fn from_spec(spec: &SpecWorld) -> Self {
        ScriptedWorld {
            capabilities: WorldCapabilities {
                sweep_obb_rotating_all: spec.capabilities.sweep_obb_rotating_all,
                query_at_step_start: spec.capabilities.query_at_step_start,
                ..WorldCapabilities::default()
            },
            call_log: spec.call_log.clone(),
            cursor: RefCell::new(0),
            mismatches: RefCell::new(Vec::new()),
        }
    }

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
            "mismatches": Value::Array(mismatches.iter().map(|m| json!(m)).collect::<Vec<_>>()),
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
            .push("query_ground called on the collision scripted world".to_string());
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

fn control_model_for(case: &SpecCase) -> Result<Option<ControlResponseInputModel>, String> {
    case.control_model
        .as_ref()
        .map(|payload| ControlResponseInputModel::from_json_str(&payload.to_string()))
        .transpose()
}

fn run_chain(engine: &KartRiderVehicleEngine, spec: &ChainSpec) -> Value {
    match engine.apply_collision_chain(
        &spec.candidates,
        spec.start_matrix,
        spec.end_matrix,
        spec.velocity,
        spec.angular_velocity,
        &spec.selector_state,
        spec.allowed_branch.as_deref(),
        spec.stop_on_disallowed,
        spec.quantize_each,
    ) {
        Ok(outcome) => json!({"kind": "chain", "outcome": encode_chain_outcome(&outcome)}),
        Err(message) => json!({"kind": "error", "error": message}),
    }
}

fn run_case(case: &SpecCase) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    let mut engine = match KartRiderVehicleEngine::new(&case.initial_state, case.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => {
            return json!({"rows": [{"kind": "error", "error": message}]});
        }
    };
    let model = match control_model_for(case) {
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
                let scripted = world.as_ref().map(ScriptedWorld::from_spec);
                let world_ref: Option<&dyn World> = scripted.as_ref().map(|s| s as &dyn World);
                match engine.step_raw_control(
                    keys,
                    &parameters,
                    model.as_ref().expect("raw_control steps require control_model"),
                    world_ref,
                ) {
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
                            "world_sweep": outcome.world_sweep.as_ref().map(encode_world_sweep),
                            "world_log": world_log,
                        }));
                    }
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                }
            }
            SpecStep::WorldStep {
                frame,
                world,
                collision_selector_state,
            } => {
                let control_value = frame.get("control").cloned();
                let control: Option<ControlPayload> = match control_value {
                    Some(value) => match serde_json::from_value(value) {
                        Ok(control) => Some(control),
                        Err(error) => {
                            rows.push(json!({"kind": "error", "error": format!("control payload decode: {error}")}));
                            break;
                        }
                    },
                    None => None,
                };
                let frame: StepFrame = match serde_json::from_value(frame.clone()) {
                    Ok(frame) => frame,
                    Err(error) => {
                        rows.push(json!({"kind": "error", "error": format!("frame decode: {error}")}));
                        break;
                    }
                };
                let scripted = ScriptedWorld::from_spec(world);
                let result = engine.step_world(
                    &frame,
                    &scripted,
                    case.parameters.collision_half_extents,
                    case.parameters.collision_center_up,
                    collision_selector_state.unwrap_or_default(),
                    false,
                    control
                        .as_ref()
                        .map(|payload| (payload, model.as_ref())),
                );
                match result {
                    Ok(outcome) => {
                        rows.push(json!({
                            "kind": "world_step",
                            "state_before": encode_state(&outcome.step.state_before),
                            "state_after": encode_state(&outcome.step.state_after),
                            "engine_state_after": encode_state(&engine.state),
                            "world_sweep": encode_world_sweep(&outcome.world_sweep),
                            "world_log": scripted.status(),
                        }));
                    }
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                }
            }
            SpecStep::Chain { chain } => {
                let row = run_chain(&engine, chain);
                let stop = row.get("kind") == Some(&json!("error"));
                rows.push(row);
                if stop {
                    break;
                }
            }
            SpecStep::Finalize {
                frame,
                chain,
                finalize,
            } => {
                let frame: StepFrame = match serde_json::from_value(frame.clone()) {
                    Ok(frame) => frame,
                    Err(error) => {
                        rows.push(json!({"kind": "error", "error": format!("frame decode: {error}")}));
                        break;
                    }
                };
                let step_outcome = match engine.step(&frame) {
                    Ok(outcome) => outcome,
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                };
                let chain_outcome = match engine.apply_collision_chain(
                    &chain.candidates,
                    chain.start_matrix,
                    chain.end_matrix,
                    chain.velocity,
                    chain.angular_velocity,
                    &chain.selector_state,
                    chain.allowed_branch.as_deref(),
                    chain.stop_on_disallowed,
                    chain.quantize_each,
                ) {
                    Ok(outcome) => outcome,
                    Err(message) => {
                        rows.push(json!({"kind": "error", "error": message}));
                        break;
                    }
                };
                let soft_base_response = if finalize.soft_base_from_last {
                    chain_outcome.responses.last().map(|response| {
                        match &response.velocity_response {
                            CollisionChainVelocityResponse::Soft(source) => {
                                SoftBaseResponse::Full(source.clone())
                            }
                            CollisionChainVelocityResponse::Hard(_) => {
                                unreachable!("soft_base_from_last requires a soft chain")
                            }
                        }
                    })
                } else {
                    None
                };
                match engine.finalize_collision_chain(
                    step_outcome,
                    None,
                    &chain_outcome,
                    &frame,
                    finalize.position_before,
                    finalize.matrix_before,
                    chain.start_matrix,
                    finalize.end_matrix,
                    finalize.end_position,
                    case.parameters.collision_half_extents,
                    case.parameters.collision_center_up,
                    &finalize.candidates,
                    &finalize.chain_key,
                    &finalize.dispatch_reason,
                    &finalize.composition_authority,
                    soft_base_response,
                    finalize.quantize_final,
                    // Not the rail-state1 ground writer path.
                    false,
                ) {
                    Ok(outcome) => {
                        rows.push(json!({
                            "kind": "finalize",
                            "state_after": encode_state(&outcome.step.state_after),
                            "engine_state_after": encode_state(&engine.state),
                            "world_sweep": encode_world_sweep(&outcome.world_sweep),
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
