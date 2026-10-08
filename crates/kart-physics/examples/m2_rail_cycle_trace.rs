//! M2 RAIL free-running cycle parity replay tool (`step_raw_topology` before
//! RAIL capture + `step_autonomous_rail` under RAIL ownership over a recorded
//! probe-capture scenario).
//!
//! Reads the case spec produced by
//! `scripts/verify_kart_physics_m2_rail_cycle_parity.py` from stdin:
//! `{"scenario": {...}, "initial_state": ..., "numeric_mode": "float32",
//! "ops": [...]}`.
//!
//! Ops:
//! - `{"op": "set_state", "state": <VehicleState>}` — engine-state restore
//!   points recorded from the Python runner (none are expected in
//!   `run_rail_free_running_cycle`, but the M1 contract is kept).
//! - `{"op": "step", "method": "step_raw_topology"|"step_autonomous_rail",
//!   "keys": [...], "parameters": <VehicleParameters>,
//!   "world": <null|{"capabilities": {...}, "call_log": [...]}>,
//!   "state_after": <VehicleState>}` — drives the named engine stepper. The
//!   scripted world pops one recorded entry per engine world call, asserts
//!   the method name and bit-exact arguments, and returns the recorded
//!   response (same contract as rail_step_trace.rs).
//! - `{"op": "error", ...step inputs..., "error": <message>}` — the Python
//!   engine raised on this call; the replay must raise the same message.
//!
//! Every f64 is emitted as its IEEE-754 bit pattern; an engine `Err` is
//! emitted as `{"op": "error", "index": N, "error": <message>}` and ends the
//! replay (the M3 NET-CORRECTION structured error is the only acceptable
//! stop).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;
use std::rc::Rc;

use kart_physics::engine::{GroundPlaneSet, KartRiderVehicleEngine, NumericMode};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{JsonValue, VehicleState};
use kart_physics::world::{
    CheckpointRespawnCandidate, CheckpointTargetCandidate, TakeoffLandingContact, World,
    WorldCapabilities, WorldGroundResult, WorldObbOverlap,
};
use serde::Deserialize;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Spec decoding
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Spec {
    initial_state: VehicleState,
    numeric_mode: NumericMode,
    ops: Vec<SpecOp>,
}

#[derive(Deserialize)]
#[serde(tag = "op")]
enum SpecOp {
    #[serde(rename = "set_state")]
    SetState { state: Value },
    #[serde(rename = "step")]
    Step {
        method: String,
        keys: Vec<String>,
        parameters: Box<VehicleParameters>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        ground_planes: Option<Vec<WorldGroundResult>>,
        #[serde(default)]
        provider_calls: Vec<SpecProviderCall>,
    },
    #[serde(rename = "error")]
    ErrorStep {
        method: String,
        keys: Vec<String>,
        parameters: Box<VehicleParameters>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        ground_planes: Option<Vec<WorldGroundResult>>,
        #[serde(default)]
        provider_calls: Vec<SpecProviderCall>,
    },
}

/// One recorded `_pending_ground_plane_provider(state, pending)` invocation
/// (run_raw_control_pose_prediction's `_AdaptiveWheelGround.query` seam):
/// `query_count` recorded `query_ground` world entries popped from the shared
/// call cursor (the runner-side cache arguments are not recomputed; order and
/// method are asserted), then the recorded plane set is returned verbatim.
#[derive(Deserialize, Clone)]
struct SpecProviderCall {
    query_count: usize,
    result: GroundPlaneSet,
}

#[derive(Deserialize, Clone)]
struct SpecWorld {
    #[serde(default)]
    capabilities: SpecWorldCapabilities,
    #[serde(default)]
    call_log: Vec<SpecCall>,
}

/// Optional-capability flags recorded from the Python `hasattr` probes on the
/// recording KartTrackWorld subclass.
#[derive(Deserialize, Clone, Default)]
struct SpecWorldCapabilities {
    #[serde(default)]
    overlap_obb_all: bool,
    #[serde(default)]
    query_takeoff_contact: bool,
    #[serde(default)]
    query_landing_contact: bool,
    #[serde(default)]
    query_rail_contact: bool,
    #[serde(default)]
    query_rail_entry_boundary: bool,
    #[serde(default)]
    query_moving_contact: bool,
    #[serde(default)]
    query_checkpoint_target_candidate: bool,
    #[serde(default)]
    query_checkpoint_respawn: bool,
    #[serde(default)]
    sweep_obb_rotating_all: bool,
    #[serde(default)]
    sweep_obb_rotating: bool,
    #[serde(default)]
    sweep_point: bool,
    #[serde(default)]
    query_at_step_start: bool,
    #[serde(default)]
    rail_query_contexts: bool,
    #[serde(default)]
    rail_entry_contexts: bool,
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

fn arr_fbits(values: [f64; 3]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn arr9_fbits(values: [f64; 9]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn encode_state(state: &VehicleState) -> Value {
    bits_json(&serde_json::to_value(state).expect("VehicleState serializes"))
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
            "mismatches": Value::Array(mismatches.iter().map(|m| json!(m)).collect()),
        })
    }

    /// Pop one recorded `query_ground` entry for a provider invocation
    /// without re-deriving its arguments: the runner-side `_AdaptiveWheelGround`
    /// cache arguments are not recomputed on the Rust side (they depend on
    /// runner-only ray/triangle state), but the method name and the shared
    /// cursor order are still asserted.
    fn pop_provider_query(&self) {
        let mut cursor = self.cursor.borrow_mut();
        let index = *cursor;
        *cursor += 1;
        if index >= self.call_log.len() {
            self.mismatches.borrow_mut().push(format!(
                "call {index}: unexpected extra provider query_ground"
            ));
            return;
        }
        let entry = &self.call_log[index];
        if entry.method != "query_ground" {
            self.mismatches.borrow_mut().push(format!(
                "call {index}: provider expected query_ground, recorded {}",
                entry.method
            ));
        }
    }
}

impl World for ScriptedWorld {
    fn capabilities(&self) -> WorldCapabilities {
        self.capabilities
    }

    fn query_ground(&self, position: [f64; 3], up: [f64; 3], _min_dot: f64) -> WorldGroundResult {
        // The `step_raw_topology` support call site invokes
        // `query_ground(position, up)` and relies on the Python default
        // `min_dot=0.65` (scripts/kart_track_world.py:1046), so the recorded
        // bound arguments carry only position/up.
        let args = json!({
            "position": arr_fbits(position),
            "up": arr_fbits(up),
        });
        let response = self.next_call("query_ground", args).unwrap_or(Value::Null);
        match serde_json::from_value(response) {
            Ok(result) => result,
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_ground response does not decode: {error}"
                ));
                WorldGroundResult::default()
            }
        }
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
            "start": arr_fbits(start),
            "end": arr_fbits(end),
            "start_matrix": arr9_fbits(start_matrix),
            "end_matrix": arr9_fbits(end_matrix),
            "half_extents": arr_fbits(half_extents),
        });
        let response = self.next_call("sweep_obb_rotating_all", args)?;
        match serde_json::from_value(response) {
            Ok(candidates) => Some(candidates),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded sweep_obb_rotating_all response does not decode: {error}"
                ));
                Some(Vec::new())
            }
        }
    }

    fn sweep_point(&self, start: [f64; 3], end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        let args = json!({
            "start": arr_fbits(start),
            "end": arr_fbits(end),
        });
        let response = self.next_call("sweep_point", args)?;
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded sweep_point response does not decode: {error}"
                ));
                Some(BTreeMap::new())
            }
        }
    }

    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let args = json!({
            "center": arr_fbits(center),
            "matrix": arr9_fbits(matrix),
            "half_extents": arr_fbits(half_extents),
        });
        let response = self.next_call("overlap_obb_all", args)?;
        match serde_json::from_value(response) {
            Ok(candidates) => Some(candidates),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded overlap_obb_all response does not decode: {error}"
                ));
                Some(Vec::new())
            }
        }
    }

    fn query_takeoff_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        // The Python authority names the intentionally unused positional
        // parameter `_position`; the recorder persists bound parameter names
        // verbatim, so the replay protocol mirrors that spelling.
        let args = json!({
            "_position": arr_fbits(position),
            "velocity": arr_fbits(velocity),
            "support_planes": bits_json(
                &serde_json::to_value(support_planes).expect("planes serialize")
            ),
            "active_slots": Value::Array(
                active_slots.iter().map(|&value| json!(value != 0)).collect()
            ),
        });
        let response = self.next_call("query_takeoff_contact", args)?;
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_takeoff_contact response does not decode: {error}"
                ));
                Some(None)
            }
        }
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
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_rail_contact response does not decode: {error}"
                ));
                Some(None)
            }
        }
    }

    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        _max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        // The topology probe call site passes no `max_distance` kwarg
        // (kartrider_vehicle_engine.py:4126-4134).
        let args = json!({
            "position": arr_fbits(position),
            "time_ms": time_ms,
            "object_id": object_id,
            "delta_ms": delta_ms,
        });
        let response = self.next_call("query_moving_contact", args)?;
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_moving_contact response does not decode: {error}"
                ));
                Some(None)
            }
        }
    }

    fn query_checkpoint_target_candidate(
        &self,
        position: [f64; 3],
        _radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        // The engine call site passes only `position`
        // (kartrider_vehicle_engine.py:1525).
        let args = json!({
            "position": arr_fbits(position),
        });
        let response = self.next_call("query_checkpoint_target_candidate", args)?;
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_checkpoint_target_candidate response does not decode: {error}"
                ));
                Some(None)
            }
        }
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
        match serde_json::from_value(response) {
            Ok(record) => Some(record),
            Err(error) => {
                self.mismatches.borrow_mut().push(format!(
                    "recorded query_checkpoint_respawn response does not decode: {error}"
                ));
                Some(None)
            }
        }
    }
}

fn build_world(spec: &Option<SpecWorld>) -> Option<ScriptedWorld> {
    spec.as_ref().map(|world| ScriptedWorld {
        capabilities: WorldCapabilities {
            query_banana_trigger: false,
            overlap_obb_all: world.capabilities.overlap_obb_all,
            query_event_obstacle_contacts: false,
            query_takeoff_contact: world.capabilities.query_takeoff_contact,
            query_landing_contact: world.capabilities.query_landing_contact,
            query_rail_contact: world.capabilities.query_rail_contact,
            query_rail_entry_boundary: world.capabilities.query_rail_entry_boundary,
            query_moving_contact: world.capabilities.query_moving_contact,
            query_checkpoint_target_candidate: world.capabilities.query_checkpoint_target_candidate,
            query_checkpoint_respawn: world.capabilities.query_checkpoint_respawn,
            sweep_obb_rotating_all: world.capabilities.sweep_obb_rotating_all,
            sweep_obb_rotating: world.capabilities.sweep_obb_rotating,
            sweep_point: world.capabilities.sweep_point,
            query_at_step_start: world.capabilities.query_at_step_start,
            rail_query_contexts: world.capabilities.rail_query_contexts,
            rail_entry_contexts: world.capabilities.rail_entry_contexts,
            // The M2 rail-cycle scenarios never publish network corrections.
            query_network_correction: false,
            query_distinct_water_volume: false,
        },
        call_log: world.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
    })
}

fn empty_world_log() -> Value {
    json!({"consumed": 0, "total": 0, "mismatches": Value::Array(vec![])})
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

fn run_step(
    engine: &mut KartRiderVehicleEngine,
    method: &str,
    keys: &[String],
    parameters: &VehicleParameters,
    world: &Option<SpecWorld>,
    ground_planes: &Option<Vec<WorldGroundResult>>,
    provider_calls: &[SpecProviderCall],
    index: usize,
) -> Value {
    let scripted = build_world(world).map(Rc::new);
    let world_ref: Option<&dyn World> = match &scripted {
        Some(scripted) => Some(scripted.as_ref()),
        None => None,
    };
    // Scripted `_pending_ground_plane_provider`: pop the recorded provider
    // invocations in order, consume each invocation's recorded query_ground
    // entries from the shared world cursor, and return the recorded planes.
    let provider_cursor: RefCell<usize> = RefCell::new(0);
    let provider = |_: &VehicleState, _: &VehicleParameters| -> Option<GroundPlaneSet> {
        let mut cursor = provider_cursor.borrow_mut();
        let call_index = *cursor;
        *cursor += 1;
        let Some(call) = provider_calls.get(call_index) else {
            if let Some(scripted) = &scripted {
                scripted.mismatches.borrow_mut().push(format!(
                    "engine made unexpected extra provider call {call_index}"
                ));
            }
            return None;
        };
        if let Some(scripted) = &scripted {
            for _ in 0..call.query_count {
                scripted.pop_provider_query();
            }
        }
        Some(call.result.clone())
    };
    let provider_ref = if provider_calls.is_empty() {
        None
    } else {
        Some(&provider as &dyn Fn(&VehicleState, &VehicleParameters) -> Option<GroundPlaneSet>)
    };
    let result = match method {
        "step_raw_topology" => engine
            .step_raw_topology(
                keys,
                parameters,
                world_ref,
                ground_planes.as_deref(),
                provider_ref,
            )
            .map(|_| ()),
        "step_autonomous_rail" => engine
            .step_autonomous_rail(keys, parameters, world_ref)
            .map(|_| ()),
        other => Err(format!("unknown recorded step method {other}")),
    };
    match result {
        Ok(()) => {
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
    let mut engine = match KartRiderVehicleEngine::new(&spec.initial_state, spec.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => {
            println!(
                "{}",
                serde_json::to_string(
                    &json!({"rows": [{"op": "error", "index": -1, "error": message}]})
                )
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
                method,
                keys,
                parameters,
                world,
                ground_planes,
                provider_calls,
            } => {
                let row = run_step(
                    &mut engine,
                    method,
                    keys,
                    parameters,
                    world,
                    ground_planes,
                    provider_calls,
                    index,
                );
                let stop = row.get("op") == Some(&Value::String("error".to_string()));
                rows.push(row);
                if stop {
                    break;
                }
            }
            SpecOp::ErrorStep {
                method,
                keys,
                parameters,
                world,
                ground_planes,
                provider_calls,
            } => {
                let row = run_step(
                    &mut engine,
                    method,
                    keys,
                    parameters,
                    world,
                    ground_planes,
                    provider_calls,
                    index,
                );
                rows.push(row);
                break;
            }
        }
    }
    println!("{}", serde_json::to_string(&json!({"rows": rows})).unwrap());
}
