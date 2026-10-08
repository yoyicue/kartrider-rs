//! Engine raw-frame parity trace tool (frame loop + topology dispatch).
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"initial_state": ..., "parameters": ..., "numeric_mode":
//! "float32"|"float64", "frames": [{"tick_ms": N, "keys": [...]}...],
//! "ground_planes": <null|[4 world-plane rows]>,
//! "world": <null|{"capabilities": {...}, "call_log": [{"method": ...,
//! "args": {...}, "response": ...}...]}>}...]}`.
//!
//! For each case a `KartRiderVehicleEngine` is built and each frame drives
//! `step_raw_frame`. `world: null` is the legal world=None path (only valid
//! with `ground_planes` supplied); otherwise a SCRIPTED world replays the
//! recorded call log produced by the Python side: each engine world call
//! pops the next log entry, asserts the method name and the bit-exact
//! arguments (every f64 compared as its IEEE-754 bit pattern), and returns
//! the recorded response. Mismatches and unconsumed entries are reported in
//! the per-case `world_log` block instead of aborting the run, so the first
//! divergence point is visible in the diff.
//!
//! Every f64 is emitted as its bit pattern (`{:016x}`, matching the Python
//! side's `struct.pack('<d')`); bools/ints/strings/None pass through. A
//! domain `Err` is emitted as `{"kind": "error", "error": <message>}` and
//! ends the case, mirroring the Python side's ValueError handling.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Read;

use kart_physics::engine::{
    AirborneProducerContract, AuthoredTriggerEdge, BoostGaugeGrantTransition,
    CheckpointTargetEvent, KartRiderVehicleEngine, NumericMode, RawFrameOutcome, RawMotionKind,
    RawMotionOutcome, RawTopologyOutcome, SupportObservation, TakeoffLandingTransfer,
    TopologyExecutorOutcome, WorldSweepInfo,
};
use kart_physics::params::VehicleParameters;
use kart_physics::scheduler::FrameSchedule;
use kart_physics::state::{JsonValue, VehicleState};
use kart_physics::topology::{TopologyDecision, TopologyState};
use kart_physics::world::{
    CheckpointTargetCandidate, TakeoffLandingContact, World, WorldCapabilities, WorldGroundResult,
    WorldObbOverlap,
};
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
    frames: Vec<SpecFrame>,
    #[serde(default)]
    ground_planes: Option<Vec<WorldGroundResult>>,
    #[serde(default)]
    world: Option<SpecWorld>,
}

#[derive(Deserialize)]
struct SpecFrame {
    tick_ms: u32,
    #[serde(default)]
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct SpecWorld {
    #[serde(default)]
    capabilities: SpecCapabilities,
    #[serde(default)]
    call_log: Vec<SpecCall>,
}

#[derive(Deserialize, Default)]
struct SpecCapabilities {
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
    query_at_step_start: bool,
}

#[derive(Deserialize, Clone)]
struct SpecCall {
    method: String,
    args: Value,
    response: Value,
}

// ---------------------------------------------------------------------------
// Encoding helpers (same conventions as engine_gnd_trace.rs)
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

fn encode_topology(topology: &TopologyState) -> Value {
    bits_json(&serde_json::to_value(topology).expect("TopologyState serializes"))
}

fn encode_scheduler(schedule: &FrameSchedule) -> Value {
    json!({
        "previous_tick_ms": schedule.previous_tick_ms,
        "current_tick_ms": schedule.current_tick_ms,
        "raw_frame_delta_ms": schedule.raw_frame_delta_ms,
        "capped_frame_delta_ms": schedule.capped_frame_delta_ms,
        "frame_delta_capped": schedule.frame_delta_capped,
        "substeps_ms": schedule.substeps_ms,
        "substeps_seconds_f32": Value::Array(
            schedule.substeps_seconds_f32.iter().map(|&v| fbits(v)).collect()
        ),
    })
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

fn encode_support(support: &SupportObservation) -> Value {
    json!({
        "active_slots": support.active_slots,
        "support_contact_count": support.support_contact_count,
        "surface_mode": support.surface_mode,
        "contact_compression_after": Value::Array(
            support.producer.contact_compression_after.iter().map(|&v| fbits(v)).collect()
        ),
    })
}

fn encode_transfer(transfer: &TakeoffLandingTransfer) -> Value {
    json!({
        "role": transfer.role,
        "selected": transfer.selected,
        "reason": transfer.reason,
        "velocity_after": match &transfer.velocity_after {
            Some(velocity) => Value::Array(velocity.iter().map(|&v| fbits(v)).collect()),
            None => Value::Null,
        },
        "hit_normal": Value::Array(
            transfer.hit_normal.iter().map(|&v| fbits(v)).collect()
        ),
        "approach_velocity": fbits(transfer.approach_velocity),
        "selector_authority": transfer.selector_authority,
        "active_support_slot_count": transfer.active_support_slot_count,
    })
}

fn encode_authored_trigger(edge: &AuthoredTriggerEdge) -> Value {
    json!({
        "selected": edge.selected,
        "reason": edge.reason,
        "contact_id": edge.contact_id,
        "velocity_c0_after": match &edge.trigger {
            Some(trigger) => Value::Array(
                trigger.velocity_c0_after.iter().map(|&v| fbits(v)).collect()
            ),
            None => Value::Null,
        },
        "state_transition": match &edge.trigger {
            Some(trigger) => match &trigger.state_transition {
                Some(transition) => json!({
                    "state": transition.state,
                    "remaining_ms": transition.remaining_ms,
                }),
                None => Value::Null,
            },
            None => Value::Null,
        },
    })
}

fn encode_checkpoint_events(events: &[CheckpointTargetEvent]) -> Value {
    Value::Array(
        events
            .iter()
            .map(|event| {
                json!({
                    "accepted": event.selector.accepted,
                    "target_after": Value::Array(
                        event.selector.target_after.iter().map(|&v| fbits(v)).collect()
                    ),
                    "gate_after": event.selector.gate_after,
                    "gate_written": event.selector.gate_written,
                    "target_written": event.selector.target_written,
                    "checkpoint_gate_armed_tick_ms": event.checkpoint_gate_armed_tick_ms,
                    "marker_id": event.marker_candidate.marker_id,
                    "marker_ordinal": event.marker_candidate.marker_ordinal,
                })
            })
            .collect(),
    )
}

fn encode_world_sweep(sweep: &WorldSweepInfo) -> Value {
    json!({
        "dispatch_reason": sweep.dispatch_reason,
        "response_dispatched": sweep.response_dispatched,
        "candidate_count": sweep.candidates.len(),
        "topology_transition": match &sweep.topology_transition {
            Some(decision) => encode_decision(decision),
            None => Value::Null,
        },
    })
}

fn encode_airborne_contract(contract: &AirborneProducerContract) -> Value {
    json!({
        "control_authority": contract.control_authority,
        "support_authority": contract.support_authority,
        "contact_slots": contract.contact_slots,
        "common_gravity": contract.common_gravity,
        "vehicle_producers": contract.vehicle_producers,
    })
}

// ---------------------------------------------------------------------------
// Scripted world
// ---------------------------------------------------------------------------

/// Replay world: each engine call pops the next recorded entry, asserts
/// method + bit-exact args, and returns the recorded response.
struct ScriptedWorld {
    capabilities: WorldCapabilities,
    call_log: Vec<SpecCall>,
    cursor: RefCell<usize>,
    mismatches: RefCell<Vec<String>>,
}

/// The world=None stand-in: every capability absent; `query_ground` records
/// a mismatch (the engine must not call it on the world=None paths).
struct NullWorld {
    mismatches: RefCell<Vec<String>>,
}

impl World for NullWorld {
    fn query_ground(&self, _position: [f64; 3], _up: [f64; 3], _min_dot: f64) -> WorldGroundResult {
        self.mismatches
            .borrow_mut()
            .push("query_ground called on world=None".to_string());
        WorldGroundResult::default()
    }
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
            "mismatches": Value::Array(
                mismatches.iter().map(|m| json!(m)).collect()
            ),
        })
    }
}

fn arr_fbits(values: [f64; 3]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
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
        serde_json::from_value(response).unwrap_or_else(|error| {
            panic!("recorded query_ground response does not decode: {error}")
        })
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

    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        let args = json!({
            "center": arr_fbits(center),
            "matrix": Value::Array(matrix.iter().map(|&v| fbits(v)).collect()),
            "half_extents": arr_fbits(half_extents),
        });
        let response = self.next_call("overlap_obb_all", args)?;
        Some(serde_json::from_value(response).expect("recorded overlap_obb_all response decodes"))
    }

    fn query_takeoff_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let args = json!({
            "position": arr_fbits(position),
            "velocity": arr_fbits(velocity),
            // Python records the support planes as their original dicts and
            // the active slots as JSON booleans.
            "support_planes": bits_json(
                &serde_json::to_value(support_planes).expect("planes serialize")
            ),
            "active_slots": Value::Array(
                active_slots.iter().map(|&v| json!(v != 0)).collect()
            ),
        });
        let response = self.next_call("query_takeoff_contact", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_takeoff_contact response decodes"),
        )
    }

    fn query_landing_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        let args = json!({
            "position": arr_fbits(position),
            "velocity": arr_fbits(velocity),
            "support_planes": bits_json(
                &serde_json::to_value(support_planes).expect("planes serialize")
            ),
            "active_slots": Value::Array(
                active_slots.iter().map(|&v| json!(v != 0)).collect()
            ),
        });
        let response = self.next_call("query_landing_contact", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_landing_contact response decodes"),
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
            serde_json::from_value(response).expect("recorded query_rail_contact response decodes"),
        )
    }

    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        _max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        let args = json!({
            "position": arr_fbits(position),
            "time_ms": time_ms,
            "object_id": object_id,
            "delta_ms": delta_ms,
        });
        let response = self.next_call("query_moving_contact", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_moving_contact response decodes"),
        )
    }

    fn query_checkpoint_target_candidate(
        &self,
        position: [f64; 3],
        _radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        let args = json!({
            "position": arr_fbits(position),
        });
        let response = self.next_call("query_checkpoint_target_candidate", args)?;
        Some(
            serde_json::from_value(response)
                .expect("recorded query_checkpoint_target_candidate response decodes"),
        )
    }
}

// ---------------------------------------------------------------------------
// Row encoding
// ---------------------------------------------------------------------------

fn encode_topology_outcome(outcome: &RawTopologyOutcome, row: &mut Map<String, Value>) {
    row.insert(
        "support".to_string(),
        encode_support(&outcome.support_observation),
    );
    row.insert(
        "topology_decision".to_string(),
        encode_decision(&outcome.topology_decision),
    );
    row.insert(
        "takeoff_transfer".to_string(),
        match &outcome.takeoff_state_transfer {
            Some(transfer) => encode_transfer(transfer),
            None => Value::Null,
        },
    );
    row.insert(
        "landing_transfer".to_string(),
        match &outcome.landing_state_transfer {
            Some(transfer) => encode_transfer(transfer),
            None => Value::Null,
        },
    );
    row.insert(
        "authored_special_trigger".to_string(),
        match &outcome.authored_special_trigger {
            Some(edge) => encode_authored_trigger(edge),
            None => Value::Null,
        },
    );
    match &outcome.executor {
        TopologyExecutorOutcome::Ground(ground) => {
            row.insert("executor_kind".to_string(), json!("ground"));
            row.insert(
                "world_sweep".to_string(),
                match &ground.world_sweep {
                    Some(sweep) => encode_world_sweep(sweep),
                    None => Value::Null,
                },
            );
            row.insert(
                "surface_mode".to_string(),
                json!(ground.surface_selection.surface_mode),
            );
        }
        TopologyExecutorOutcome::Airborne(airborne) => {
            row.insert("executor_kind".to_string(), json!("airborne"));
            row.insert(
                "airborne_producer_contract".to_string(),
                encode_airborne_contract(&airborne.airborne_producer_contract),
            );
            row.insert(
                "world_sweep".to_string(),
                match &airborne.ground.world_sweep {
                    Some(sweep) => encode_world_sweep(sweep),
                    None => Value::Null,
                },
            );
            row.insert(
                "surface_mode".to_string(),
                json!(airborne.ground.surface_selection.surface_mode),
            );
        }
        TopologyExecutorOutcome::ResetSurface(reset) => {
            row.insert("executor_kind".to_string(), json!("reset_surface"));
            row.insert("world_sweep".to_string(), Value::Null);
            row.insert(
                "reset_surface_entry".to_string(),
                json!(reset.reset_surface_entry),
            );
        }
        TopologyExecutorOutcome::RailCapture(execution) => {
            row.insert("executor_kind".to_string(), json!("rail_capture"));
            row.insert("world_sweep".to_string(), Value::Null);
            row.insert(
                "rail_state1_ground_writer".to_string(),
                match execution.as_ref() {
                    kart_physics::engine::RailCaptureExecution::GroundWriter(capture) => {
                        json!(capture.rail_state1_ground_writer.state_after)
                    }
                    kart_physics::engine::RailCaptureExecution::RawRail(_) => Value::Null,
                },
            );
        }
        TopologyExecutorOutcome::MovingContactCapture(_) => {
            // Exercised by moving_reset_trace; the raw-frame scenarios never
            // capture MOVING-CONTACT ownership.
            row.insert("executor_kind".to_string(), json!("moving_contact_capture"));
            row.insert("world_sweep".to_string(), Value::Null);
            row.insert("moving_contact_acquired".to_string(), json!(true));
        }
        TopologyExecutorOutcome::NetworkCorrection(_) => {
            // Exercised by network_correction_trace; the raw-frame scenarios
            // never publish authoritative correction packets.
            row.insert("executor_kind".to_string(), json!("network_correction"));
            row.insert("world_sweep".to_string(), Value::Null);
            row.insert("network_correction_acquired".to_string(), json!(true));
        }
    }
}

fn encode_motion(outcome: &RawMotionOutcome) -> Value {
    let mut row = Map::new();
    row.insert("executor".to_string(), json!(outcome.raw_motion_executor));
    row.insert(
        "dispatch_reason".to_string(),
        json!(outcome.raw_motion_dispatch_reason),
    );
    row.insert(
        "state_before".to_string(),
        encode_state(&outcome.state_before),
    );
    row.insert(
        "state_after".to_string(),
        encode_state(&outcome.state_after),
    );
    row.insert(
        "topology_after".to_string(),
        encode_topology(&outcome.topology_after),
    );
    row.insert(
        "checkpoint_target_events".to_string(),
        encode_checkpoint_events(&outcome.checkpoint_target_events),
    );
    row.insert(
        "item_use_transition".to_string(),
        match &outcome.item_use_transition {
            None => Value::Null,
            Some(transition) => {
                let mut encoded = Map::new();
                encoded.insert("consumed".to_string(), json!(transition.consumed));
                encoded.insert(
                    "handler_accepted".to_string(),
                    json!(transition.handler_accepted),
                );
                encoded.insert("state_changed".to_string(), json!(transition.state_changed));
                encoded.insert("reason".to_string(), json!(transition.reason));
                encoded.insert("slots_after".to_string(), json!(transition.slots_after));
                for (key, value) in [
                    ("item_id", transition.item_id),
                    ("consumed_slot_index", transition.consumed_slot_index),
                    ("target_state", transition.target_state),
                    ("duration_ms", transition.duration_ms),
                    ("state_after", transition.state_after),
                    ("remaining_ms_after", transition.remaining_ms_after),
                ] {
                    if let Some(value) = value {
                        encoded.insert(key.to_string(), json!(value));
                    }
                }
                if let Some(value) = transition.runtime_order {
                    encoded.insert("runtime_order".to_string(), json!(value));
                }
                if let Some(value) = transition.authority {
                    encoded.insert("authority".to_string(), json!(value));
                }
                if let Some(value) = &transition.domain {
                    encoded.insert("domain".to_string(), json!(value));
                }
                Value::Object(encoded)
            }
        },
    );
    row.insert(
        "boost_gauge_grant_transition".to_string(),
        match &outcome.boost_gauge_grant_transition {
            None => Value::Null,
            Some(BoostGaugeGrantTransition::SameSubstepItemUseUnadjudicated {
                accum_b10_after,
                slots_after,
            }) => json!({
                "granted": false,
                "reason": "same_substep_item_use_and_grant_order_unadjudicated",
                "accum_b10_after": fbits(*accum_b10_after),
                "slots_after": slots_after,
            }),
            Some(BoostGaugeGrantTransition::Evaluated(grant)) => {
                let mut encoded = Map::new();
                encoded.insert(
                    "committed_b10".to_string(),
                    fbits(grant.snapshot.committed_b10),
                );
                encoded.insert("pending_b2c".to_string(), fbits(grant.snapshot.pending_b2c));
                encoded.insert("raw_760".to_string(), fbits(grant.snapshot.raw_760));
                encoded.insert("cap".to_string(), fbits(grant.snapshot.cap));
                encoded.insert(
                    "accum_clamped".to_string(),
                    fbits(grant.snapshot.accum_clamped),
                );
                encoded.insert(
                    "final_eligible".to_string(),
                    json!(grant.snapshot.final_eligible),
                );
                encoded.insert("granted".to_string(), json!(grant.granted));
                encoded.insert("reason".to_string(), json!(grant.reason));
                encoded.insert(
                    "selected_slot_index".to_string(),
                    json!(grant.selected_slot_index),
                );
                encoded.insert("accum_b10_after".to_string(), fbits(grant.accum_b10_after));
                encoded.insert("slots_after".to_string(), json!(grant.slots_after));
                if let Some(value) = grant.runtime_order {
                    encoded.insert("runtime_order".to_string(), json!(value));
                }
                if let Some(value) = grant.authority {
                    encoded.insert("authority".to_string(), json!(value));
                }
                Value::Object(encoded)
            }
        },
    );
    row.insert(
        "banana_effect_transition".to_string(),
        match &outcome.banana_effect_transition {
            None => Value::Null,
            Some(candidate) => json!({
                "activated": true,
                "object_id": candidate.object_id,
                "distance": fbits(candidate.distance),
                "radius": fbits(candidate.radius),
                "duration_ms": 2000,
                "authority": "GoItemBanana_slot1_to_slot2_kind1_B9C",
            }),
        },
    );
    row.insert(
        "banana_effect_remaining_ms".to_string(),
        json!(outcome.banana_effect_remaining_ms),
    );
    match &outcome.kind {
        RawMotionKind::Topology(topology) => {
            row.insert("kind".to_string(), json!("topology"));
            encode_topology_outcome(topology, &mut row);
        }
        RawMotionKind::FallClampReset(fall) => {
            row.insert("kind".to_string(), json!("fall_clamp"));
            row.insert(
                "topology_decision".to_string(),
                encode_decision(&fall.topology_transition),
            );
        }
        RawMotionKind::ManualRestart(restart) => {
            row.insert("kind".to_string(), json!("manual_restart"));
            row.insert(
                "topology_decision".to_string(),
                encode_decision(&restart.topology_transition),
            );
            row.insert(
                "checkpoint_gate_after".to_string(),
                json!(restart.checkpoint_gate_after),
            );
        }
        RawMotionKind::Rail(_) => {
            // The RAIL domain stepper is exercised by rail_step_trace; the
            // raw-frame scenarios never enter RAIL ownership.
            row.insert("kind".to_string(), json!("rail"));
        }
        RawMotionKind::MovingContact(_) => {
            // Exercised by moving_reset_trace.
            row.insert("kind".to_string(), json!("moving_contact"));
        }
        RawMotionKind::Reset(_) => {
            // Exercised by moving_reset_trace.
            row.insert("kind".to_string(), json!("reset"));
        }
        RawMotionKind::NetworkCorrection(_) | RawMotionKind::NetworkCorrectionPreemption(_) => {
            // Exercised by network_correction_trace; the raw-frame scenarios
            // never publish authoritative correction packets.
            row.insert("kind".to_string(), json!("network_correction"));
        }
    }
    Value::Object(row)
}

fn encode_frame(outcome: &RawFrameOutcome) -> Value {
    json!({
        "clock_initialized": outcome.clock_initialized,
        "physics_substep_count": outcome.physics_substep_count,
        "scheduler": encode_scheduler(&outcome.frame_scheduler),
        "substeps": Value::Array(
            outcome.physics_substeps.iter().map(encode_motion).collect()
        ),
        "state_after": encode_state(&outcome.state_after),
    })
}

fn run_case(case: &SpecCase) -> Value {
    let mut rows: Vec<Value> = Vec::new();
    let mut engine = match KartRiderVehicleEngine::new(&case.initial_state, case.numeric_mode) {
        Ok(engine) => engine,
        Err(message) => {
            return json!({
                "rows": [{"kind": "error", "error": message}],
                "world_log": Value::Null,
            });
        }
    };
    let scripted = case.world.as_ref().map(|world| ScriptedWorld {
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
            sweep_point: false,
            query_at_step_start: world.capabilities.query_at_step_start,
            rail_query_contexts: false,
            rail_entry_contexts: false,
            // The raw-frame scenarios never publish network corrections.
            query_network_correction: false,
            query_distinct_water_volume: false,
        },
        call_log: world.call_log.clone(),
        cursor: RefCell::new(0),
        mismatches: RefCell::new(Vec::new()),
    });
    let null_world = NullWorld {
        mismatches: RefCell::new(Vec::new()),
    };
    let world: Option<&dyn World> = match &scripted {
        Some(scripted) => Some(scripted),
        None => None,
    };
    let _ = &null_world;
    for frame in &case.frames {
        match engine.step_raw_frame(
            frame.tick_ms,
            &frame.keys,
            &case.parameters,
            world,
            case.ground_planes.as_deref(),
            None,
        ) {
            Ok(outcome) => rows.push(encode_frame(&outcome)),
            Err(message) => {
                rows.push(json!({"kind": "error", "error": message}));
                break;
            }
        }
    }
    let world_log = match &scripted {
        Some(scripted) => scripted.status(),
        None => json!({
            "consumed": 0,
            "total": 0,
            "mismatches": Value::Array(
                null_world.mismatches.borrow().iter().map(|m| json!(m)).collect()
            ),
        }),
    };
    json!({
        "rows": rows,
        "world_log": world_log,
        "prediction_history_len": engine.prediction_history.len(),
    })
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
