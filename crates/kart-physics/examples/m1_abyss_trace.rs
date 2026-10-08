//! M1 free-running parity replay tool (step_raw_control over a recorded
//! probe-capture scenario).
//!
//! Reads the case spec produced by
//! `scripts/verify_kart_physics_m1_abyss_parity.py` from stdin:
//! `{"scenario": {...}, "initial_state": ..., "parameters": ...,
//! "numeric_mode": "float32", "control_model": <trace response model
//! payload>, "ops": [...]}`.
//!
//! Ops:
//! - `{"op": "set_state", "state": <VehicleState>}` — engine-state restore
//!   points recorded from the Python runner (torque bootstrap writes and
//!   world-refinement chunk rollbacks).
//! - `{"op": "step_raw_control", "keys": [...], "substep_ms": N,
//!   "world": <null|{"margin": f, "call_log": [...]}>,
//!   "state_after": <VehicleState>}` — drives `step_raw_control`. With a
//!   world, the scripted `_InstantaneousObbWorld` analogue maps the engine's
//!   `sweep_obb_rotating_all` call (start pose at both endpoints,
//!   `query_at_step_start`) onto the recorded `overlap_obb_all` entry,
//!   asserting bit-exact margined arguments and returning the recorded
//!   candidates.
//! - `{"op": "step_raw_ground_correction", ...}` — drives the authoritative
//!   GND producer/integrator with recorded current and pending wheel planes.
//! - `{"op": "error", ...step inputs..., "error": <message>}` — the Python
//!   engine raised on this call; the replay must raise the same message.
//!
//! Every f64 is emitted as its IEEE-754 bit pattern; an engine `Err` is
//! emitted as `{"op": "error", "index": N, "error": <message>}` and ends
//! the replay (the M2 CONTACT-acquisition structured error is the expected
//! stop when the scenario dispatches a selected collision candidate).

use std::cell::RefCell;
use std::io::Read;

use kart_physics::domains::control_response_input::ControlResponseInputModel;
use kart_physics::engine::{
    GroundPlaneSet, GroundStepOptions, KartRiderVehicleEngine, NumericMode,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult, WorldObbOverlap};
use serde::Deserialize;
use serde_json::{json, Value};

/// Python `parameters.substep_ms` may be an integer-valued float
/// (`_captured_substep_ms` decodes to float); accept both JSON forms.
fn de_i64_lenient<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)? as i64)
}

fn de_opt_i64_lenient<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f64>::deserialize(deserializer)?.map(|value| value as i64))
}

// ---------------------------------------------------------------------------
// Spec decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Spec {
    initial_state: VehicleState,
    parameters: VehicleParameters,
    numeric_mode: NumericMode,
    control_model: Value,
    ops: Vec<SpecOp>,
}

#[derive(Deserialize)]
#[serde(tag = "op")]
enum SpecOp {
    #[serde(rename = "set_state")]
    SetState { state: Value },
    #[serde(rename = "step_raw_control")]
    Step {
        keys: Vec<String>,
        #[serde(deserialize_with = "de_i64_lenient")]
        substep_ms: i64,
        #[serde(default)]
        world: Option<SpecWorld>,
    },
    #[serde(rename = "step_raw_ground_correction")]
    GroundStep {
        keys: Vec<String>,
        #[serde(deserialize_with = "de_i64_lenient")]
        substep_ms: i64,
        #[serde(default)]
        ground_plane: Option<GroundPlaneSet>,
        #[serde(default)]
        #[serde(deserialize_with = "de_opt_i64_lenient")]
        pending_substep_ms: Option<i64>,
        #[serde(default)]
        pending_provider_present: bool,
        #[serde(default)]
        pending_ground_plane: Option<GroundPlaneSet>,
        #[serde(default = "default_true")]
        refresh_pending: bool,
        #[serde(default = "default_true")]
        advance_surface_selector: bool,
        #[serde(default)]
        support_active: Option<bool>,
        #[serde(default)]
        world: Option<SpecWorld>,
    },
    #[serde(rename = "error")]
    ErrorStep {
        keys: Vec<String>,
        #[serde(deserialize_with = "de_i64_lenient")]
        substep_ms: i64,
        #[serde(default)]
        world: Option<SpecWorld>,
    },
    #[serde(rename = "error_ground_correction")]
    ErrorGroundStep {
        keys: Vec<String>,
        #[serde(deserialize_with = "de_i64_lenient")]
        substep_ms: i64,
        #[serde(default)]
        ground_plane: Option<GroundPlaneSet>,
        #[serde(default)]
        #[serde(deserialize_with = "de_opt_i64_lenient")]
        pending_substep_ms: Option<i64>,
        #[serde(default)]
        pending_provider_present: bool,
        #[serde(default)]
        pending_ground_plane: Option<GroundPlaneSet>,
        #[serde(default = "default_true")]
        refresh_pending: bool,
        #[serde(default = "default_true")]
        advance_surface_selector: bool,
        #[serde(default)]
        support_active: Option<bool>,
        #[serde(default)]
        world: Option<SpecWorld>,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Deserialize, Clone)]
struct SpecWorld {
    margin: f64,
    #[serde(default)]
    call_log: Vec<SpecCall>,
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

// ---------------------------------------------------------------------------
// Scripted instantaneous world (`_InstantaneousObbWorld` analogue)
// ---------------------------------------------------------------------------

struct InstantaneousScriptedWorld {
    margin: f64,
    call_log: Vec<SpecCall>,
    cursor: RefCell<usize>,
    mismatches: RefCell<Vec<String>>,
}

impl InstantaneousScriptedWorld {
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

impl World for InstantaneousScriptedWorld {
    fn capabilities(&self) -> WorldCapabilities {
        WorldCapabilities {
            sweep_obb_rotating_all: true,
            query_at_step_start: true,
            ..WorldCapabilities::default()
        }
    }

    fn query_ground(&self, _position: [f64; 3], _up: [f64; 3], _min_dot: f64) -> WorldGroundResult {
        self.mismatches
            .borrow_mut()
            .push("query_ground called on the instantaneous scripted world".to_string());
        WorldGroundResult::default()
    }

    /// `_InstantaneousObbWorld.sweep_obb_rotating_all`
    /// (run_raw_control_pose_prediction.py:298-301): delegates to the
    /// recorded `overlap_obb_all(start, start_matrix, margined extents)`.
    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        _end: [f64; 3],
        start_matrix: [f64; 9],
        _end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let margined = [
            half_extents[0] + self.margin,
            half_extents[1] + self.margin,
            half_extents[2] + self.margin,
        ];
        let args = json!({
            "center": Value::Array(start.iter().map(|&v| fbits(v)).collect()),
            "matrix": Value::Array(start_matrix.iter().map(|&v| fbits(v)).collect()),
            "half_extents": Value::Array(margined.iter().map(|&v| fbits(v)).collect()),
        });
        let response = self.next_call("overlap_obb_all", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded overlap_obb_all response decodes"),
        )
    }
}

fn empty_world_log() -> Value {
    json!({"consumed": 0, "total": 0, "mismatches": Value::Array(vec![])})
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

fn run_step(
    engine: &mut KartRiderVehicleEngine,
    model: &ControlResponseInputModel,
    base_parameters: &VehicleParameters,
    keys: &[String],
    substep_ms: i64,
    world: &Option<SpecWorld>,
    index: usize,
) -> Value {
    let parameters = VehicleParameters {
        substep_ms,
        ..base_parameters.clone()
    };
    let scripted = world.as_ref().map(|world| InstantaneousScriptedWorld {
        margin: world.margin,
        call_log: world.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
    });
    let world_ref: Option<&dyn World> = match &scripted {
        Some(scripted) => Some(scripted),
        None => None,
    };
    match engine.step_raw_control(keys, &parameters, model, world_ref) {
        Ok(_outcome) => {
            let world_log = match &scripted {
                Some(scripted) => scripted.status(),
                None => empty_world_log(),
            };
            json!({
                "op": "step",
                "index": index,
                "state_after": encode_state(&engine.state),
                "world_log": world_log,
            })
        }
        Err(message) => json!({
            "op": "error",
            "index": index,
            "error": message,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_ground_step(
    engine: &mut KartRiderVehicleEngine,
    base_parameters: &VehicleParameters,
    keys: &[String],
    substep_ms: i64,
    ground_plane: &Option<GroundPlaneSet>,
    pending_substep_ms: Option<i64>,
    pending_provider_present: bool,
    pending_ground_plane: &Option<GroundPlaneSet>,
    refresh_pending: bool,
    advance_surface_selector: bool,
    support_active: Option<bool>,
    world: &Option<SpecWorld>,
    index: usize,
) -> Value {
    let parameters = VehicleParameters {
        substep_ms,
        ..base_parameters.clone()
    };
    let pending_parameters = pending_substep_ms.map(|pending_substep_ms| VehicleParameters {
        substep_ms: pending_substep_ms,
        ..base_parameters.clone()
    });
    let scripted = world.as_ref().map(|world| InstantaneousScriptedWorld {
        margin: world.margin,
        call_log: world.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
    });
    let world_ref: Option<&dyn World> = match &scripted {
        Some(scripted) => Some(scripted),
        None => None,
    };
    let provider =
        |_state: &VehicleState, _parameters: &VehicleParameters| pending_ground_plane.clone();
    let options = GroundStepOptions {
        refresh_pending,
        pending_parameters,
        pending_ground_plane_provider: if pending_provider_present {
            Some(&provider)
        } else {
            None
        },
        advance_surface_selector,
        support_active,
        ..GroundStepOptions::default()
    };
    match engine.step_raw_ground_correction(
        keys,
        &parameters,
        world_ref,
        ground_plane.as_ref(),
        &options,
    ) {
        Ok(_outcome) => {
            let world_log = match &scripted {
                Some(scripted) => scripted.status(),
                None => empty_world_log(),
            };
            json!({
                "op": "step",
                "index": index,
                "state_after": encode_state(&engine.state),
                "world_log": world_log,
            })
        }
        Err(message) => json!({
            "op": "error",
            "index": index,
            "error": message,
        }),
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Spec = serde_json::from_str(&input).expect("parse spec JSON");
    let model = ControlResponseInputModel::from_json_str(&spec.control_model.to_string())
        .expect("control model payload decodes");
    let mut engine = match KartRiderVehicleEngine::new(&spec.initial_state, spec.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => {
            println!(
                "{}",
                serde_json::to_string(&json!({"rows": [{"op": "error", "index": -1, "error": message}]}))
                .unwrap()
            );
            return;
        }
    };
    let mut rows: Vec<Value> = Vec::new();
    for (index, op) in spec.ops.iter().enumerate() {
        match op {
            SpecOp::SetState { state } => {
                engine.state =
                    serde_json::from_value(state.clone()).expect("set_state payload decodes");
                rows.push(json!({
                    "op": "set_state",
                    "index": index,
                    "state": encode_state(&engine.state),
                }));
            }
            SpecOp::Step {
                keys,
                substep_ms,
                world,
            } => {
                let row = run_step(
                    &mut engine,
                    &model,
                    &spec.parameters,
                    keys,
                    *substep_ms,
                    world,
                    index,
                );
                let stop = row.get("op") == Some(&Value::String("error".to_string()));
                rows.push(row);
                if stop {
                    break;
                }
            }
            SpecOp::GroundStep {
                keys,
                substep_ms,
                ground_plane,
                pending_substep_ms,
                pending_provider_present,
                pending_ground_plane,
                refresh_pending,
                advance_surface_selector,
                support_active,
                world,
            } => {
                let row = run_ground_step(
                    &mut engine,
                    &spec.parameters,
                    keys,
                    *substep_ms,
                    ground_plane,
                    *pending_substep_ms,
                    *pending_provider_present,
                    pending_ground_plane,
                    *refresh_pending,
                    *advance_surface_selector,
                    *support_active,
                    world,
                    index,
                );
                let stop = row.get("op") == Some(&Value::String("error".to_string()));
                rows.push(row);
                if stop {
                    break;
                }
            }
            SpecOp::ErrorStep {
                keys,
                substep_ms,
                world,
            } => {
                let row = run_step(
                    &mut engine,
                    &model,
                    &spec.parameters,
                    keys,
                    *substep_ms,
                    world,
                    index,
                );
                rows.push(row);
                break;
            }
            SpecOp::ErrorGroundStep {
                keys,
                substep_ms,
                ground_plane,
                pending_substep_ms,
                pending_provider_present,
                pending_ground_plane,
                refresh_pending,
                advance_surface_selector,
                support_active,
                world,
            } => {
                let row = run_ground_step(
                    &mut engine,
                    &spec.parameters,
                    keys,
                    *substep_ms,
                    ground_plane,
                    *pending_substep_ms,
                    *pending_provider_present,
                    pending_ground_plane,
                    *refresh_pending,
                    *advance_surface_selector,
                    *support_active,
                    world,
                    index,
                );
                rows.push(row);
                break;
            }
        }
    }
    println!("{}", serde_json::to_string(&json!({"rows": rows})).unwrap());
}
