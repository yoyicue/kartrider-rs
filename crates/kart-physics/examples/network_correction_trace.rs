//! NET-CORRECTION (M3) parity trace tool.
//!
//! Reads a JSON spec from stdin: `{"cases": [...]}`. Case shapes:
//! - `{"kind": "engine", "name", "initial_state", "parameters",
//!   "numeric_mode", "steps": [...]}` drives the engine-level steppers. Step
//!   kinds: `raw_motion` (`step_raw_motion`: preemption arms +
//!   NET-CORRECTION domain arm + history semantics), `network_begin`
//!   (`begin_network_correction`), `network_step` (`step_network_correction`
//!   with explicit resume selectors), `network_autonomous_step`
//!   (`step_autonomous_network_correction`), `network_inspect`
//!   (`inspect_autonomous_network_correction`), `network_begin_and_step`
//!   (`begin_and_step_autonomous_network_correction`), `network_rollback`
//!   (`rollback_and_resimulate_network_correction`).
//! - `{"kind": "leaf_sequence"|"leaf_begin"|"leaf_step", "name", "args"}`
//!   drives the `domains::network_correction_topology` leaf functions
//!   directly.
//!
//! The scripted world pops one recorded entry per engine world call, asserts
//! method name and bit-exact arguments, and returns the recorded response
//! (same contract as moving_reset_trace.rs). Every f64 is emitted as its
//! IEEE-754 bit pattern (`{:016x}`); a domain `Err` is emitted as
//! `{"error": <message>}` and the row still carries the post-call state.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;

use kart_physics::domains::network_correction_topology::{
    begin_network_correction, evaluate_network_correction_step, evaluate_network_sequence,
    NetworkCorrectionStepResult, NetworkSequenceDecision,
};
use kart_physics::engine::{
    AutonomousCorrectionEntry, AutonomousNetworkCorrectionOutcome, AutonomousResumeSelectors,
    KartRiderVehicleEngine, NetworkCorrectionContract, NetworkCorrectionEntryOutcome,
    NetworkCorrectionInspection, NetworkCorrectionRecord, NetworkCorrectionResume,
    NetworkCorrectionRollbackOutcome, NetworkCorrectionStepOutcome, NumericMode,
    RawMotionKind, RawMotionOutcome, RestoredConstraintSnapshot, TopologyExecutorOutcome,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{
    JsonValue, MovingContactAnchor, NetworkCorrectionState, RigidStateSnapshot, VehicleState,
};
use kart_physics::topology::{MotionDomain, TopologyDecision, TopologyState};
use kart_physics::world::{World, WorldCapabilities, WorldGroundResult, WorldObbOverlap};
use serde::Deserialize;
use serde_json::{json, Map, Value};

// ---------------------------------------------------------------------------
// Bit-pattern encoding helpers (same conventions as moving_reset_trace.rs)
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

fn opt_fbits(value: Option<f64>) -> Value {
    value.map_or(Value::Null, fbits)
}

/// Open-payload encoding (Python `encode_open_scalar` on the harness side):
/// the TOP-LEVEL rule — integral numbers stay JSON ints, everything else
/// takes the f64 bit pattern. Used for the scalar `distinct_water_volume`
/// echo.
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

/// Python `encode_open_scalar(dict)`: the dict branch encodes every VALUE
/// with the nested rule (`encode_open_nested`), where every number — even
/// integral ones — becomes its f64 bit pattern.
fn json_map_encoded(map: &BTreeMap<String, JsonValue>) -> Value {
    Value::Object(
        map.iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    bits_json(&serde_json::to_value(value).expect("JsonValue serializes")),
                )
            })
            .collect(),
    )
}

fn opt_json_map_encoded(map: &Option<BTreeMap<String, JsonValue>>) -> Value {
    map.as_ref().map(json_map_encoded).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// State / correction / outcome projections (mirror the Python verify script)
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
        "object_point_velocity": anchor.object_point_velocity.map(arr_fbits).unwrap_or(Value::Null),
        "kinematic_contract": anchor.kinematic_contract,
    })
}

fn rigid_projection(rigid: &RigidStateSnapshot) -> Value {
    json!({
        "position": arr_fbits(rigid.position),
        "velocity": arr_fbits(rigid.velocity),
        "angular_velocity": arr_fbits(rigid.angular_velocity),
        "matrix": arr9_fbits(rigid.matrix),
    })
}

fn correction_projection(correction: &NetworkCorrectionState) -> Value {
    json!({
        "mode": correction.mode,
        "duration_ms": fbits(correction.duration_ms),
        "elapsed_ms": fbits(correction.elapsed_ms),
        "sequence": correction.sequence,
        "initial_state": rigid_projection(&correction.initial_state),
        "target_state": rigid_projection(&correction.target_state),
        "complete": correction.complete,
        "policy_authority": correction.policy_authority,
        "resume_topology": correction
            .resume_topology
            .as_ref()
            .map(topology_row)
            .unwrap_or(Value::Null),
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
        "network_correction": state.network_correction.as_ref().map(correction_projection).unwrap_or(Value::Null),
        "last_network_sequence": state.last_network_sequence,
        "network_packets_accepted": state.network_packets_accepted,
        "network_packets_rejected": state.network_packets_rejected,
        "network_packets_lost_estimate": state.network_packets_lost_estimate,
        "time_ms": state.time_ms,
        "step_index": state.step_index,
    })
}

fn decision_projection(decision: &NetworkSequenceDecision) -> Value {
    json!({
        "accepted": decision.accepted,
        "reason": decision.reason,
        "candidate": decision.candidate,
        "last_accepted_before": decision.last_accepted_before,
        "last_accepted_after": decision.last_accepted_after,
        "forward_distance": decision.forward_distance,
        "estimated_packets_lost": decision.estimated_packets_lost,
        "ordered_authority": decision.ordered_authority,
    })
}

fn topology_decision_projection(decision: &TopologyDecision) -> Value {
    bits_json(&serde_json::to_value(decision).expect("decision serializes"))
}

fn contract_projection(contract: &NetworkCorrectionContract) -> Value {
    json!({
        "target_authority": contract.target_authority,
        "resume_selector_authority": contract.resume_selector_authority,
        "moving_constraint_resume_authority": contract.moving_constraint_resume_authority,
        "preserved_local_state": contract.preserved_local_state,
        "open_boundary": contract.open_boundary,
    })
}

fn restored_snapshot_projection(snapshot: &RestoredConstraintSnapshot) -> Value {
    json!({
        "domain": snapshot.domain.value(),
        "return_domain": snapshot.return_domain.map(MotionDomain::value),
        "constraint_owner_id": snapshot.constraint_owner_id,
        "rail_state": snapshot.rail_state,
    })
}

/// Python `topology_transition` on a `step_network_correction` result: the
/// (possibly state_after-overwritten) decision dict with
/// `restored_constraint_snapshot` embedded when the resume restored a
/// constraint (:4895-4903).
fn network_step_transition_projection(outcome: &NetworkCorrectionStepOutcome) -> Value {
    let Some(decision) = &outcome.topology_transition else {
        return Value::Null;
    };
    let mut value = topology_decision_projection(decision);
    if let Some(snapshot) = &outcome.restored_constraint_snapshot {
        value["restored_constraint_snapshot"] = restored_snapshot_projection(snapshot);
    }
    value
}

fn network_step_projection(outcome: &NetworkCorrectionStepOutcome) -> Value {
    json!({
        "state_after": rigid_projection(&outcome.step.state_after),
        "correction_after": correction_projection(&outcome.step.correction_after),
        "alpha": fbits(outcome.step.alpha),
        "complete": outcome.step.complete,
        "mode": outcome.step.mode,
        "sequence": outcome.step.sequence,
        "closure_status": outcome.step.closure_status,
        "topology_transition": network_step_transition_projection(outcome),
        "topology_after": topology_row(&outcome.topology_after),
        "network_correction_contract": contract_projection(&outcome.network_correction_contract),
    })
}

fn selectors_projection(selectors: &AutonomousResumeSelectors) -> Value {
    json!({
        "candidate_position": arr_fbits(selectors.candidate_position),
        "support": selectors
            .support
            .as_ref()
            .map(|support| json!({
                "support_contact_count": support.support_contact_count,
                "surface_mode": support.surface_mode,
                "active_slots": support.active_slots,
            }))
            .unwrap_or(Value::Null),
        "moving_contact": opt_json_map_encoded(&selectors.moving_contact),
        "rail_contact": opt_json_map_encoded(&selectors.rail_contact),
        "distinct_water_volume": selectors
            .distinct_water_volume
            .as_ref()
            .map(json_value_encoded)
            .unwrap_or(Value::Null),
        "reset_surface": selectors.reset_surface,
        "authority": selectors.authority,
    })
}

fn entry_projection(outcome: &NetworkCorrectionEntryOutcome) -> Value {
    json!({
        "correction": correction_projection(&outcome.correction),
        "topology_transition": topology_decision_projection(&outcome.topology_transition),
        "entry_transfer": outcome.entry_transfer,
        "sequence_decision": decision_projection(&outcome.sequence_decision),
    })
}

fn autonomous_projection(outcome: &AutonomousNetworkCorrectionOutcome) -> Value {
    json!({
        "step": network_step_projection(&outcome.step),
        "autonomous_resume_selectors": selectors_projection(&outcome.autonomous_resume_selectors),
        "network_correction_acquired": outcome.network_correction_acquired,
        "network_correction_entry": outcome
            .entry
            .as_ref()
            .map(|entry| entry_projection(entry))
            .unwrap_or(Value::Null),
        "network_correction_selector": opt_json_map_encoded(&outcome.selector),
    })
}

fn history_entry_projection(entry: &kart_physics::engine::PredictionHistoryEntry) -> Value {
    json!({
        "time_ms_before": entry.time_ms_before,
        "time_ms_after": entry.time_ms_after,
        "keys": entry.keys,
        "has_pending_ground_plane_provider": entry.has_pending_ground_plane_provider,
        "state_before": state_row(&entry.state_before),
        "state_after": state_row(&entry.state_after),
    })
}

fn rollback_projection(outcome: &NetworkCorrectionRollbackOutcome) -> Value {
    json!({
        "network_correction_acquired": true,
        "rollback_resimulated": true,
        "history_committed_by_rollback": true,
        "authoritative_time_ms": fbits(outcome.authoritative_time_ms),
        "replayed_substep_count": outcome.replayed_substep_count,
        "sequence_decision": decision_projection(&outcome.sequence_decision),
        "replay_results": Value::Array(
            outcome.replay_results.iter().map(replay_row_projection).collect(),
        ),
        "state_after": state_row(&outcome.state_after),
        "topology_after": topology_row(&outcome.topology_after),
        "complete": true,
        "network_correction_contract": {
            "target_authority": "ordered_authoritative_historical_rigid_body_snapshot",
            "rollback_authority": "retained_substep_boundary_and_raw_control_history",
            "resimulation_authority": "raw_keys_current_world_selectors_network_query_suppressed",
            "resume_selector_authority": "recomputed_during_each_replayed_substep",
            "open_boundary": "game_packet_timestamp_origin_and_independent_runtime_cycles",
        },
    })
}

fn correction_entry_projection(entry: &AutonomousCorrectionEntry) -> Value {
    match entry {
        AutonomousCorrectionEntry::Forward(forward) => json!({
            "entry_kind": "forward",
            "result": autonomous_projection(forward),
        }),
        AutonomousCorrectionEntry::Rollback(rollback) => json!({
            "entry_kind": "rollback",
            "result": rollback_projection(rollback),
        }),
    }
}

/// The `detail` payload of a raw-motion row (Python `raw_motion_detail`).
fn raw_motion_detail(outcome: &RawMotionOutcome) -> Value {
    match &outcome.kind {
        RawMotionKind::Topology(topology) => match &topology.executor {
            TopologyExecutorOutcome::Ground(_) => json!({"kind": "topology", "executor": "ground"}),
            TopologyExecutorOutcome::Airborne(_) => json!({"kind": "topology", "executor": "airborne"}),
            TopologyExecutorOutcome::ResetSurface(_) => json!({"kind": "topology", "executor": "reset_surface"}),
            TopologyExecutorOutcome::RailCapture(_) => json!({"kind": "topology", "executor": "rail_capture"}),
            TopologyExecutorOutcome::MovingContactCapture(_) => json!({"kind": "topology", "executor": "moving_contact_capture"}),
            TopologyExecutorOutcome::NetworkCorrection(preemption) => json!({
                "kind": "network_preemption",
                "entry": correction_entry_projection(&preemption.entry),
                "entry_support_observation": {
                    "support_contact_count": preemption.entry_support_observation.support_contact_count,
                    "surface_mode": preemption.entry_support_observation.surface_mode,
                },
                "raw_topology_contract": {
                    "control_authority": preemption.raw_topology_contract.control_authority,
                    "geometry_authority": preemption.raw_topology_contract.geometry_authority,
                    "selector_authority": preemption.raw_topology_contract.selector_authority,
                    "scheduler_authority": preemption.raw_topology_contract.scheduler_authority,
                    "landing_transfer_authority": preemption.raw_topology_contract.landing_transfer_authority,
                    "takeoff_transfer_authority": preemption.raw_topology_contract.takeoff_transfer_authority,
                },
            }),
        },
        RawMotionKind::NetworkCorrection(step) => json!({
            "kind": "network_step",
            "result": autonomous_projection(step),
        }),
        RawMotionKind::NetworkCorrectionPreemption(entry) => json!({
            "kind": "network_preemption",
            "entry": correction_entry_projection(entry),
        }),
        RawMotionKind::Rail(_) => json!({"kind": "rail"}),
        RawMotionKind::MovingContact(_) => json!({"kind": "moving_contact"}),
        RawMotionKind::Reset(_) => json!({"kind": "reset"}),
        RawMotionKind::FallClampReset(_) => json!({"kind": "fall_clamp"}),
        RawMotionKind::ManualRestart(_) => json!({"kind": "manual_restart"}),
    }
}

/// One replayed raw-control substep row (Python `replay_row_projection` over
/// the `_step_raw_motion_once` result dicts inside `replay_results`; the
/// Python dicts carry no `state_before` — the spliced `history_after`
/// projection is the before/after trajectory evidence).
fn replay_row_projection(outcome: &RawMotionOutcome) -> Value {
    json!({
        "raw_motion_executor": outcome.raw_motion_executor,
        "raw_motion_dispatch_reason": outcome.raw_motion_dispatch_reason,
        "state_after": state_row(&outcome.state_after),
        "detail": raw_motion_detail(outcome),
    })
}

fn inspect_projection(inspection: &NetworkCorrectionInspection) -> Value {
    json!({
        "outcome_kind": "network_inspect",
        "correction_active": inspection.correction_active,
        "authority": inspection.authority,
        "record": opt_json_map_encoded(&inspection.record),
        "sequence_decision": inspection
            .sequence_decision
            .as_ref()
            .map(decision_projection)
            .unwrap_or(Value::Null),
        "rejection_reason": inspection.rejection_reason,
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
    #[serde(rename = "engine")]
    Engine {
        name: String,
        initial_state: Box<VehicleState>,
        parameters: Box<VehicleParameters>,
        numeric_mode: NumericMode,
        steps: Vec<SpecStep>,
    },
    #[serde(rename = "leaf_sequence")]
    LeafSequence { name: String, args: Value },
    #[serde(rename = "leaf_begin")]
    LeafBegin { name: String, args: Value },
    #[serde(rename = "leaf_step")]
    LeafStep { name: String, args: Value },
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
enum SpecStep {
    #[serde(rename = "raw_motion")]
    RawMotion {
        #[serde(default)]
        keys: Vec<String>,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "network_begin")]
    NetworkBegin { args: Value },
    #[serde(rename = "network_step")]
    NetworkStep {
        #[serde(default)]
        resume: Option<Value>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "network_autonomous_step")]
    NetworkAutonomousStep {
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "network_inspect")]
    NetworkInspect {
        #[serde(default)]
        world: Option<SpecWorld>,
    },
    #[serde(rename = "network_begin_and_step")]
    NetworkBeginAndStep {
        record: Value,
        #[serde(default)]
        world: Option<SpecWorld>,
        #[serde(default)]
        parameters: Option<Box<VehicleParameters>>,
    },
    #[serde(rename = "network_rollback")]
    NetworkRollback {
        record: Value,
        #[serde(default)]
        world: Option<SpecWorld>,
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
    query_rail_contact: bool,
    #[serde(default)]
    query_network_correction: bool,
    #[serde(default)]
    query_distinct_water_volume: bool,
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
// Scripted replay world (same contract as moving_reset_trace.rs)
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
    ) -> Option<Vec<WorldObbOverlap>> {
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

    fn query_network_correction(
        &self,
        position: [f64; 3],
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let args = json!({
            "position": arr_fbits(position),
        });
        let response = self.next_call("query_network_correction", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_network_correction response decodes"),
        )
    }

    fn query_distinct_water_volume(&self, position: [f64; 3]) -> Option<Option<JsonValue>> {
        let args = json!({
            "position": arr_fbits(position),
        });
        let response = self.next_call("query_distinct_water_volume", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_distinct_water_volume response decodes"),
        )
    }
}

fn build_world(spec: &Option<SpecWorld>) -> Option<ScriptedWorld> {
    spec.as_ref().map(|world| ScriptedWorld {
        capabilities: WorldCapabilities {
            query_moving_contact: world.capabilities.query_moving_contact,
            query_rail_contact: world.capabilities.query_rail_contact,
            query_network_correction: world.capabilities.query_network_correction,
            query_distinct_water_volume: world.capabilities.query_distinct_water_volume,
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
// Spec argument decoding helpers
// ---------------------------------------------------------------------------

fn value_f64(value: &Value, key: &str) -> f64 {
    value
        .as_f64()
        .unwrap_or_else(|| panic!("arg {key} must be a number"))
}

fn value_vec3(value: &Value, key: &str) -> [f64; 3] {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("arg {key} must be an array"));
    assert!(items.len() == 3, "arg {key} must have three components");
    [
        value_f64(&items[0], key),
        value_f64(&items[1], key),
        value_f64(&items[2], key),
    ]
}

fn value_mat9(value: &Value, key: &str) -> [f64; 9] {
    let items = value
        .as_array()
        .unwrap_or_else(|| panic!("arg {key} must be an array"));
    assert!(items.len() == 9, "arg {key} must have nine components");
    let mut out = [0.0; 9];
    for (index, item) in items.iter().enumerate() {
        out[index] = value_f64(item, key);
    }
    out
}

fn arg<'a>(args: &'a Value, key: &str) -> &'a Value {
    args.get(key)
        .unwrap_or_else(|| panic!("args missing {key}"))
}

fn value_rigid(value: &Value, _key: &str) -> RigidStateSnapshot {
    RigidStateSnapshot {
        position: value_vec3(arg(value, "position"), "position"),
        velocity: value_vec3(arg(value, "velocity"), "velocity"),
        angular_velocity: value_vec3(arg(value, "angular_velocity"), "angular_velocity"),
        matrix: value_mat9(arg(value, "matrix"), "matrix"),
    }
}

fn value_opt_sequence(value: &Value) -> Option<i64> {
    if value.is_null() {
        return None;
    }
    let number = value
        .as_i64()
        .unwrap_or_else(|| panic!("sequence must be an integer or null"));
    Some(number)
}

fn json_value_from(value: &Value) -> JsonValue {
    serde_json::from_value(value.clone()).expect("arg decodes as JsonValue")
}

fn resume_from_value(value: &Option<Value>) -> NetworkCorrectionResume {
    let Some(value) = value else {
        return NetworkCorrectionResume::default();
    };
    let moving_object_state = match value.get("moving_object_state") {
        None | Some(Value::Null) => None,
        Some(mapping) => Some(
            serde_json::from_value::<BTreeMap<String, JsonValue>>(mapping.clone())
                .expect("moving_object_state decodes"),
        ),
    };
    NetworkCorrectionResume {
        support_contact_count: value
            .get("support_contact_count")
            .and_then(Value::as_i64)
            .unwrap_or(0),
        surface_mode: value
            .get("surface_mode")
            .and_then(Value::as_i64)
            .unwrap_or(0),
        rail_state: value
            .get("rail_state")
            .and_then(Value::as_i64)
            .unwrap_or(0),
        moving_object_state,
        moving_contact_observed: value
            .get("moving_contact_observed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        distinct_water_volume: value
            .get("distinct_water_volume")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        reset_requested: value
            .get("reset_requested")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

fn record_from_value(value: &Value) -> Result<NetworkCorrectionRecord, String> {
    let record: BTreeMap<String, JsonValue> =
        serde_json::from_value(value.clone()).expect("record decodes as a mapping");
    NetworkCorrectionRecord::from_record(&record)
}

// ---------------------------------------------------------------------------
// Engine case runner
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
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
            SpecStep::RawMotion { world, .. }
            | SpecStep::NetworkAutonomousStep { world, .. }
            | SpecStep::NetworkInspect { world }
            | SpecStep::NetworkBeginAndStep { world, .. }
            | SpecStep::NetworkRollback { world, .. } => build_world(world),
            SpecStep::NetworkBegin { .. } | SpecStep::NetworkStep { .. } => None,
        };
        let world_ref: Option<&dyn World> = scripted.as_ref().map(|w| w as &dyn World);
        let mut row = Map::new();
        let mut error: Option<String> = None;
        let mut outcome = Value::Null;
        match step {
            SpecStep::RawMotion { keys, parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("raw_motion"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match engine.step_raw_motion(keys, step_parameters, world_ref, None, None) {
                    Ok(result) => {
                        let mut projection = json!({
                            "outcome_kind": "raw_motion",
                            "raw_motion_executor": result.raw_motion_executor,
                            "raw_motion_dispatch_reason": result.raw_motion_dispatch_reason,
                            "detail": raw_motion_detail(&result),
                        });
                        if result.history_committed_by_rollback() {
                            // The spliced history (prefix + replayed suffix)
                            // is the M3 acceptance trajectory evidence.
                            projection["detail"]["entry"]["result"]["history_after"] =
                                Value::Array(
                                    engine
                                        .prediction_history
                                        .iter()
                                        .map(history_entry_projection)
                                        .collect(),
                                );
                        }
                        outcome = projection;
                    }
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkBegin { args } => {
                row.insert("kind".to_string(), json!("network_begin"));
                let authoritative = value_rigid(arg(args, "authoritative"), "authoritative");
                let mode = arg(args, "mode").as_str().expect("mode is a string");
                let duration_ms = value_f64(arg(args, "duration_ms"), "duration_ms");
                let sequence = value_opt_sequence(arg(args, "sequence"));
                match engine.begin_network_correction(&authoritative, mode, duration_ms, sequence)
                {
                    Ok(result) => {
                        outcome = json!({
                            "outcome_kind": "network_begin",
                            "entry": entry_projection(&result),
                        });
                    }
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkStep { resume, parameters: step_parameters } => {
                row.insert("kind".to_string(), json!("network_step"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                let resume = resume_from_value(resume);
                match engine.step_network_correction(step_parameters, &resume) {
                    Ok(result) => {
                        outcome = json!({
                            "outcome_kind": "network_step",
                            "result": network_step_projection(&result),
                        });
                    }
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkAutonomousStep { parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("network_autonomous_step"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match engine.step_autonomous_network_correction(step_parameters, world_ref) {
                    Ok(result) => {
                        outcome = json!({
                            "outcome_kind": "network_autonomous_step",
                            "result": autonomous_projection(&result),
                        });
                    }
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkInspect { .. } => {
                row.insert("kind".to_string(), json!("network_inspect"));
                match engine.inspect_autonomous_network_correction(world_ref) {
                    Ok(result) => outcome = inspect_projection(&result),
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkBeginAndStep { record, parameters: step_parameters, .. } => {
                row.insert("kind".to_string(), json!("network_begin_and_step"));
                let step_parameters = step_parameters.as_deref().unwrap_or(parameters);
                match record_from_value(record).and_then(|record| {
                    engine.begin_and_step_autonomous_network_correction(
                        step_parameters,
                        world_ref,
                        &record,
                    )
                }) {
                    Ok(result) => {
                        outcome = json!({
                            "outcome_kind": "network_begin_and_step",
                            "entry": correction_entry_projection(&result),
                        });
                    }
                    Err(message) => error = Some(message),
                }
            }
            SpecStep::NetworkRollback { record, .. } => {
                row.insert("kind".to_string(), json!("network_rollback"));
                match record_from_value(record)
                    .and_then(|record| engine.rollback_and_resimulate_network_correction(world_ref, &record))
                {
                    Ok(result) => {
                        let mut projection = rollback_projection(&result);
                        // The spliced history (prefix + replayed suffix) is
                        // the M3 acceptance trajectory evidence.
                        projection["history_after"] = Value::Array(
                            engine
                                .prediction_history
                                .iter()
                                .map(history_entry_projection)
                                .collect(),
                        );
                        outcome = json!({
                            "outcome_kind": "network_rollback",
                            "result": projection,
                        });
                    }
                    Err(message) => error = Some(message),
                }
            }
        }
        row.insert(
            "error".to_string(),
            error.map_or(Value::Null, |message| json!(message)),
        );
        row.insert("outcome".to_string(), outcome);
        row.insert("topology_after".to_string(), topology_row(&engine.state.topology));
        row.insert("state".to_string(), state_row(&engine.state));
        row.insert(
            "history_depth".to_string(),
            json!(engine.prediction_history.len()),
        );
        row.insert("world_status".to_string(), world_status(scripted.as_ref()));
        rows.push(Value::Object(row));
    }
    json!({"name": name, "rows": rows})
}

// ---------------------------------------------------------------------------
// Leaf case runners
// ---------------------------------------------------------------------------

fn leaf_step_projection(result: &NetworkCorrectionStepResult) -> Value {
    json!({
        "state_after": rigid_projection(&result.state_after),
        "correction_after": correction_projection(&result.correction_after),
        "alpha": fbits(result.alpha),
        "complete": result.complete,
        "mode": result.mode,
        "sequence": result.sequence,
        "closure_status": result.closure_status,
    })
}

fn run_leaf_sequence(name: &str, args: &Value) -> Value {
    let candidate = json_value_from(arg(args, "candidate"));
    let last_accepted = json_value_from(arg(args, "last_accepted"));
    let bits = arg(args, "bits")
        .as_u64()
        .expect("bits is an unsigned integer") as u32;
    let (error, outcome) = match evaluate_network_sequence(&candidate, &last_accepted, bits) {
        Ok(decision) => (Value::Null, decision_projection(&decision)),
        Err(message) => (json!(message), Value::Null),
    };
    json!({"name": name, "rows": [{"kind": "leaf_sequence", "error": error, "outcome": outcome}]})
}

fn run_leaf_begin(name: &str, args: &Value) -> Value {
    let predicted = value_rigid(arg(args, "predicted_state"), "predicted_state");
    let authoritative = value_rigid(arg(args, "authoritative_state"), "authoritative_state");
    let mode = arg(args, "mode").as_str().expect("mode is a string");
    let duration_ms = value_f64(arg(args, "duration_ms"), "duration_ms");
    let sequence = value_opt_sequence(arg(args, "sequence")).map(|value| value as u32);
    let (error, outcome) =
        match begin_network_correction(&predicted, &authoritative, mode, duration_ms, sequence) {
            Ok(correction) => (Value::Null, correction_projection(&correction)),
            Err(message) => (json!(message), Value::Null),
        };
    json!({"name": name, "rows": [{"kind": "leaf_begin", "error": error, "outcome": outcome}]})
}

fn run_leaf_step(name: &str, args: &Value) -> Value {
    let correction_arg = arg(args, "correction");
    let correction = NetworkCorrectionState {
        mode: arg(correction_arg, "mode")
            .as_str()
            .expect("mode is a string")
            .to_string(),
        duration_ms: value_f64(arg(correction_arg, "duration_ms"), "duration_ms"),
        elapsed_ms: value_f64(arg(correction_arg, "elapsed_ms"), "elapsed_ms"),
        sequence: value_opt_sequence(arg(correction_arg, "sequence")).map(|value| value as u32),
        initial_state: value_rigid(arg(correction_arg, "initial_state"), "initial_state"),
        target_state: value_rigid(arg(correction_arg, "target_state"), "target_state"),
        complete: arg(correction_arg, "complete")
            .as_bool()
            .expect("complete is a bool"),
        policy_authority: correction_arg
            .get("policy_authority")
            .and_then(|value| value.as_str().map(str::to_string)),
        resume_topology: correction_arg
            .get("resume_topology")
            .filter(|value| !value.is_null())
            .map(|value| {
                serde_json::from_value::<TopologyState>(value.clone())
                    .expect("resume_topology decodes")
            }),
    };
    let elapsed_ms = value_f64(arg(args, "elapsed_ms"), "elapsed_ms");
    let (error, outcome) = match evaluate_network_correction_step(&correction, elapsed_ms) {
        Ok(result) => (Value::Null, leaf_step_projection(&result)),
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
            SpecCase::LeafSequence { name, args } => {
                out.push(run_leaf_sequence(name, args));
            }
            SpecCase::LeafBegin { name, args } => {
                out.push(run_leaf_begin(name, args));
            }
            SpecCase::LeafStep { name, args } => {
                out.push(run_leaf_step(name, args));
            }
        }
    }
    println!("{}", serde_json::to_string(&out).expect("output serializes"));
}
