//! W2/W3 shared live-world parity replay: the `m2_rail_cycle_trace` contract
//! with the recorded/replayed world answers replaced by LIVE [`TrackWorld`]
//! queries.
//!
//! Consumes the same case spec as `kart-physics/examples/m2_rail_cycle_trace.rs`
//! (produced by `scripts/verify_kart_physics_m2_rail_cycle_parity.py`),
//! instantiates one `TrackWorld` for `scenario.track`, and drives the engine
//! with it. Two comparison channels, both reported:
//!
//! 1. State trajectory: per-step emitted state, diffed downstream by
//!    `scripts/verify_kart_physics_live_world_parity.py` (native) and
//!    `scripts/verify_kart_physics_wasm_parity.py` (wasm) against the recorded
//!    `state_after` with the exact row shape and budget logic of the
//!    recorded-replay verify script.
//! 2. World-response diff: every engine world call ALSO pops the recorded
//!    call_log entry — the method name and bit-exact arguments are asserted
//!    exactly like the ScriptedWorld contract — and the live response is
//!    diffed against the recorded response field-by-field (f64 bit-exact;
//!    JSON numbers compare by bit pattern, `null`/absent keys are
//!    normalized). Per-method identical/divergent counts plus the first
//!    divergence samples are emitted under `world_response_diff`.
//!
//! Capability contract: the world presents the RECORDED capability flags to
//! the engine. Every enabled capability, including the asset-backed
//! `query_takeoff_contact`, is answered live and diffed against the recorded
//! Python response; capabilities absent from the recording remain dark.
//!
//! Provider seam: `run_raw_control_pose_prediction`'s `_AdaptiveWheelGround`
//! composition is runner-side logic the Rust engine does not reimplement, so
//! the recorded `GroundPlaneSet` is still returned verbatim — but each
//! underlying recorded `query_ground(position, up)` is ALSO answered live and
//! diffed (counted under `query_ground(provider)`).
//!
//! [`run_live_world_replay`] returns `{"rows": [...], "world_response_diff":
//! {...}}` where `rows` is the exact `m2_rail_cycle_trace` row shape. Both
//! the native example (`examples/live_world_trace.rs`) and the wasm export
//! (`kart-wasm`'s `kart_wasm_run_replay`) call this one implementation, so
//! the W3 native-vs-wasm gate compares two executions of literally the same
//! code. For the wasm path the track bytes are pre-registered under
//! `scenario.track` via [`crate::track::register_virtual_bytes`], so the
//! `TrackWorld::from_path` call below transparently loads from bytes.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;
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

use crate::world::TrackWorld;

// ---------------------------------------------------------------------------
// Spec decoding (same shapes as m2_rail_cycle_trace.rs)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Spec {
    scenario: SpecScenario,
    initial_state: VehicleState,
    numeric_mode: NumericMode,
    ops: Vec<SpecOp>,
}

#[derive(Deserialize)]
struct SpecScenario {
    track: PathBuf,
    #[serde(default)]
    #[allow(dead_code)]
    name: Option<String>,
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
// Recorded-vs-live response diff
// ---------------------------------------------------------------------------

/// Drop `null`-valued object entries recursively. The Python client omits
/// optional keys (e.g. `triangle`, `contact_basis_44`) where the typed Rust
/// structs serialize an explicit `null`; both spellings mean absence.
fn strip_nulls(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(_, item)| !item.is_null())
                .map(|(key, item)| (key.clone(), strip_nulls(item)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(strip_nulls).collect()),
        other => other.clone(),
    }
}

/// Field-by-field diff; every JSON number compares by its f64 bit pattern
/// (Python ints and Rust `JsonValue::Number` both land on the same f64).
fn diff_response(recorded: &Value, live: &Value, path: &str, out: &mut Vec<String>, limit: usize) {
    if out.len() >= limit {
        return;
    }
    match (recorded, live) {
        (Value::Object(left), Value::Object(right)) => {
            for key in left.keys() {
                if !right.contains_key(key) {
                    out.push(format!("{path}.{key}: missing in live response"));
                    if out.len() >= limit {
                        return;
                    }
                }
            }
            for (key, right_value) in right {
                match left.get(key) {
                    Some(left_value) => diff_response(
                        left_value,
                        right_value,
                        &format!("{path}.{key}"),
                        out,
                        limit,
                    ),
                    None => {
                        out.push(format!("{path}.{key}: unexpected key in live response"));
                    }
                }
                if out.len() >= limit {
                    return;
                }
            }
        }
        (Value::Array(left), Value::Array(right)) => {
            if left.len() != right.len() {
                out.push(format!(
                    "{path}: array length live {} != recorded {}",
                    right.len(),
                    left.len()
                ));
                return;
            }
            for (index, (left_value, right_value)) in left.iter().zip(right.iter()).enumerate() {
                diff_response(
                    left_value,
                    right_value,
                    &format!("{path}[{index}]"),
                    out,
                    limit,
                );
                if out.len() >= limit {
                    return;
                }
            }
        }
        (Value::Number(left), Value::Number(right)) => {
            let left = left.as_f64().expect("JSON number as f64");
            let right = right.as_f64().expect("JSON number as f64");
            if left.to_bits() != right.to_bits() {
                out.push(format!(
                    "{path}: live {right:e} ({:016x}) != recorded {left:e} ({:016x})",
                    right.to_bits(),
                    left.to_bits(),
                ));
            }
        }
        _ => {
            if recorded != live {
                out.push(format!("{path}: live {live} != recorded {recorded}"));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-method response-diff statistics
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MethodStats {
    calls: u64,
    identical: u64,
    divergent: u64,
    samples: Vec<Value>,
}

type SharedStats = Rc<RefCell<BTreeMap<String, MethodStats>>>;

fn stats_json(stats: &SharedStats) -> Value {
    let stats = stats.borrow();
    Value::Object(
        stats
            .iter()
            .map(|(method, entry)| {
                (
                    method.clone(),
                    json!({
                        "calls": entry.calls,
                        "identical": entry.identical,
                        "divergent": entry.divergent,
                        "samples": entry.samples,
                    }),
                )
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Live replay world: TrackWorld answers + recorded-call assertion + diff
// ---------------------------------------------------------------------------

struct LiveWorld {
    world: Rc<TrackWorld>,
    capabilities: WorldCapabilities,
    call_log: Vec<SpecCall>,
    cursor: RefCell<usize>,
    mismatches: RefCell<Vec<String>>,
    step_index: usize,
    stats: SharedStats,
}

impl LiveWorld {
    fn record_diff(
        &self,
        method: &str,
        index: Option<usize>,
        recorded: Option<&Value>,
        live: Value,
    ) {
        let mut stats = self.stats.borrow_mut();
        let entry = stats.entry(method.to_string()).or_default();
        entry.calls += 1;
        let paths = match recorded {
            Some(recorded) => {
                let mut out = Vec::new();
                diff_response(
                    &strip_nulls(recorded),
                    &strip_nulls(&live),
                    "$",
                    &mut out,
                    5,
                );
                out
            }
            None => vec!["no recorded call_log entry for this live call".to_string()],
        };
        if paths.is_empty() {
            entry.identical += 1;
        } else {
            entry.divergent += 1;
            if entry.samples.len() < 3 {
                entry.samples.push(json!({
                    "step": self.step_index,
                    "call": index,
                    "paths": paths,
                }));
            }
        }
    }

    /// Pop the next recorded call_log entry, asserting the method name and
    /// bit-exact arguments exactly like `ScriptedWorld::next_call`, then diff
    /// the live response against the recorded response.
    fn exchange(&self, method: &str, args: Value, live: Value) {
        let index = {
            let mut cursor = self.cursor.borrow_mut();
            let index = *cursor;
            *cursor += 1;
            index
        };
        if index >= self.call_log.len() {
            self.mismatches
                .borrow_mut()
                .push(format!("call {index}: unexpected extra call {method}"));
            self.record_diff(method, None, None, live);
            return;
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
        let recorded = entry.response.clone();
        self.record_diff(method, Some(index), Some(&recorded), live);
    }

    /// Pop one recorded provider `query_ground` entry (no argument
    /// recomputation — the runner-side `_AdaptiveWheelGround` cache arguments
    /// are recorded, not re-derived), answer it LIVE with the recorded
    /// position/up, and diff the response.
    fn pop_provider_query(&self) {
        let index = {
            let mut cursor = self.cursor.borrow_mut();
            let index = *cursor;
            *cursor += 1;
            index
        };
        if index >= self.call_log.len() {
            self.mismatches.borrow_mut().push(format!(
                "call {index}: unexpected extra provider query_ground"
            ));
            self.record_diff("query_ground(provider)", None, None, Value::Null);
            return;
        }
        let entry = &self.call_log[index];
        if entry.method != "query_ground" {
            self.mismatches.borrow_mut().push(format!(
                "call {index}: provider expected query_ground, recorded {}",
                entry.method
            ));
            self.record_diff(
                "query_ground(provider)",
                Some(index),
                Some(&entry.response.clone()),
                Value::Null,
            );
            return;
        }
        let position: [f64; 3] =
            serde_json::from_value(entry.args.get("position").cloned().unwrap_or(Value::Null))
                .expect("recorded provider query_ground position decodes");
        let up: [f64; 3] =
            serde_json::from_value(entry.args.get("up").cloned().unwrap_or(Value::Null))
                .expect("recorded provider query_ground up decodes");
        // Python `query_ground(position, up)` default (kart_track_world.py:1046).
        let live = self.world.query_ground(position, up, 0.65);
        let live_json = serde_json::to_value(&live).expect("WorldGroundResult serializes");
        let recorded = entry.response.clone();
        self.record_diff(
            "query_ground(provider)",
            Some(index),
            Some(&recorded),
            live_json,
        );
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

impl World for LiveWorld {
    fn capabilities(&self) -> WorldCapabilities {
        self.capabilities
    }

    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult {
        // Recorded bound arguments carry only position/up (the engine relies
        // on the Python default `min_dot=0.65`); see m2_rail_cycle_trace.rs.
        let args = json!({
            "position": arr_fbits(position),
            "up": arr_fbits(up),
        });
        let live = self.world.query_ground(position, up, min_dot);
        let live_json = serde_json::to_value(&live).expect("WorldGroundResult serializes");
        self.exchange("query_ground", args, live_json);
        live
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
        let live =
            self.world
                .sweep_obb_rotating_all(start, end, start_matrix, end_matrix, half_extents);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live sweep_obb_rotating_all capability absent".to_string());
        }
        let live_json = match &live {
            Some(candidates) => serde_json::to_value(candidates).expect("overlaps serialize"),
            None => Value::Null,
        };
        self.exchange("sweep_obb_rotating_all", args, live_json);
        live
    }

    fn sweep_point(&self, start: [f64; 3], end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        let args = json!({
            "start": arr_fbits(start),
            "end": arr_fbits(end),
        });
        let live = self.world.sweep_point(start, end);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live sweep_point capability absent".to_string());
        }
        let live_json = match &live {
            Some(record) => serde_json::to_value(record).expect("sweep_point payload serializes"),
            None => Value::Null,
        };
        self.exchange("sweep_point", args, live_json);
        live
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
        let live = self.world.overlap_obb_all(center, matrix, half_extents);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live overlap_obb_all capability absent".to_string());
        }
        let live_json = match &live {
            Some(candidates) => serde_json::to_value(candidates).expect("overlaps serialize"),
            None => Value::Null,
        };
        self.exchange("overlap_obb_all", args, live_json);
        live
    }

    fn query_takeoff_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        // The Python method names its unused first parameter `_position`, and
        // the recorder persists bound argument names verbatim.
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
        let live =
            self.world
                .query_takeoff_contact(position, velocity, support_planes, active_slots);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live query_takeoff_contact capability absent".to_string());
        }
        let live_json = match &live {
            Some(Some(candidate)) => {
                serde_json::to_value(candidate).expect("takeoff candidate serializes")
            }
            Some(None) | None => Value::Null,
        };
        self.exchange("query_takeoff_contact", args, live_json);
        live
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
        let live = self
            .world
            .query_rail_contact(position, previous_rail_state, owner_id);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live query_rail_contact capability absent".to_string());
        }
        let live_json = match &live {
            Some(Some(record)) => serde_json::to_value(record).expect("rail selector serializes"),
            Some(None) | None => Value::Null,
        };
        self.exchange("query_rail_contact", args, live_json);
        live
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
        let live =
            self.world
                .query_moving_contact(position, time_ms, object_id, delta_ms, _max_distance);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live query_moving_contact capability absent".to_string());
        }
        let live_json = match &live {
            Some(Some(record)) => {
                serde_json::to_value(record).expect("moving-contact payload serializes")
            }
            Some(None) | None => Value::Null,
        };
        self.exchange("query_moving_contact", args, live_json);
        live
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
        let live = self
            .world
            .query_checkpoint_target_candidate(position, _radius);
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live query_checkpoint_target_candidate capability absent".to_string());
        }
        let live_json = match &live {
            Some(Some(candidate)) => {
                serde_json::to_value(candidate).expect("checkpoint candidate serializes")
            }
            Some(None) | None => Value::Null,
        };
        self.exchange("query_checkpoint_target_candidate", args, live_json);
        live
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
        let live = self.world.query_checkpoint_respawn(
            position,
            checkpoint_target,
            checkpoint_gate,
            mode_family,
            max_course_distance,
        );
        if live.is_none() {
            self.mismatches
                .borrow_mut()
                .push("live query_checkpoint_respawn capability absent".to_string());
        }
        let live_json = match &live {
            Some(Some(candidate)) => {
                serde_json::to_value(candidate).expect("respawn candidate serializes")
            }
            Some(None) | None => Value::Null,
        };
        self.exchange("query_checkpoint_respawn", args, live_json);
        live
    }
}

fn build_world(
    spec: &Option<SpecWorld>,
    world: &Rc<TrackWorld>,
    step_index: usize,
    stats: &SharedStats,
) -> Option<LiveWorld> {
    spec.as_ref().map(|world_spec| LiveWorld {
        world: Rc::clone(world),
        capabilities: WorldCapabilities {
            overlap_obb_all: world_spec.capabilities.overlap_obb_all,
            query_event_obstacle_contacts: false,
            query_banana_trigger: false,
            query_takeoff_contact: world_spec.capabilities.query_takeoff_contact,
            query_landing_contact: world_spec.capabilities.query_landing_contact,
            query_rail_contact: world_spec.capabilities.query_rail_contact,
            query_rail_entry_boundary: world_spec.capabilities.query_rail_entry_boundary,
            query_moving_contact: world_spec.capabilities.query_moving_contact,
            query_checkpoint_target_candidate: world_spec
                .capabilities
                .query_checkpoint_target_candidate,
            query_checkpoint_respawn: world_spec.capabilities.query_checkpoint_respawn,
            sweep_obb_rotating_all: world_spec.capabilities.sweep_obb_rotating_all,
            sweep_obb_rotating: world_spec.capabilities.sweep_obb_rotating,
            sweep_point: world_spec.capabilities.sweep_point,
            query_at_step_start: world_spec.capabilities.query_at_step_start,
            rail_query_contexts: world_spec.capabilities.rail_query_contexts,
            rail_entry_contexts: world_spec.capabilities.rail_entry_contexts,
            // The M2 rail-cycle scenarios never publish network corrections.
            query_network_correction: false,
            query_distinct_water_volume: false,
        },
        call_log: world_spec.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
        step_index,
        stats: Rc::clone(stats),
    })
}

fn empty_world_log() -> Value {
    json!({"consumed": 0, "total": 0, "mismatches": Value::Array(vec![])})
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_step(
    engine: &mut KartRiderVehicleEngine,
    method: &str,
    keys: &[String],
    parameters: &VehicleParameters,
    world: &Option<SpecWorld>,
    ground_planes: &Option<Vec<WorldGroundResult>>,
    provider_calls: &[SpecProviderCall],
    index: usize,
    track_world: &Rc<TrackWorld>,
    stats: &SharedStats,
) -> Value {
    let live = build_world(world, track_world, index, stats).map(Rc::new);
    let world_ref: Option<&dyn World> = match &live {
        Some(live) => Some(live.as_ref()),
        None => None,
    };
    // Scripted `_pending_ground_plane_provider`: the recorded plane sets are
    // returned verbatim (the `_AdaptiveWheelGround` composition is
    // runner-side logic), but each recorded underlying `query_ground` is
    // ALSO answered live and diffed (see `pop_provider_query`).
    let provider_cursor: RefCell<usize> = RefCell::new(0);
    let provider = |_: &VehicleState, _: &VehicleParameters| -> Option<GroundPlaneSet> {
        let mut cursor = provider_cursor.borrow_mut();
        let call_index = *cursor;
        *cursor += 1;
        let Some(call) = provider_calls.get(call_index) else {
            if let Some(live) = &live {
                live.mismatches.borrow_mut().push(format!(
                    "engine made unexpected extra provider call {call_index}"
                ));
            }
            return None;
        };
        if let Some(live) = &live {
            for _ in 0..call.query_count {
                live.pop_provider_query();
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
            let world_log = match &live {
                Some(live) => live.status(),
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

/// Executes one W2-style live-world parity spec end-to-end and returns the
/// output JSON document (`{"rows": [...], "world_response_diff": {...}}`).
/// Error rows follow the same shape as the historical example main: a
/// single `{"op": "error", "index": -1, "error": ...}` row.
pub fn run_live_world_replay(input: &str) -> String {
    let error_document = |message: String| {
        serde_json::to_string(&json!({"rows": [{"op": "error", "index": -1, "error": message}]}))
            .expect("error document serializes")
    };
    let spec: Spec = match serde_json::from_str(input) {
        Ok(spec) => spec,
        Err(error) => return error_document(format!("parse spec JSON: {error}")),
    };
    let track_world = match TrackWorld::from_path(&spec.scenario.track) {
        Ok(world) => Rc::new(world),
        Err(message) => return error_document(message),
    };
    let stats: SharedStats = Rc::new(RefCell::new(BTreeMap::new()));
    let mut engine = match KartRiderVehicleEngine::new(&spec.initial_state, spec.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => return error_document(message),
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
                    &track_world,
                    &stats,
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
                    &track_world,
                    &stats,
                );
                rows.push(row);
                break;
            }
        }
    }
    serde_json::to_string(&json!({
        "rows": rows,
        "world_response_diff": stats_json(&stats),
    }))
    .expect("replay output serializes")
}
