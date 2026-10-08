//! Engine GND-chain parity trace tool.
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"initial_state": ..., "parameters": ..., "numeric_mode":
//! "float32"|"float64", "steps": [{"keys": [...], "substep_ms": N,
//! "ground_plane": <null|plane|[4 planes]>, "support_active": <null|bool>,
//! "producer": <bool>, "refresh_pending": <bool>, "pending_substep_ms":
//! <null|int>}...]}...]}`.
//!
//! For each case a `KartRiderVehicleEngine` is built and each step drives
//! `step_raw_ground_correction` (world=None) or, when `"producer": true`,
//! `produce_ground_accumulators`. Prints one JSON object per case with the
//! per-step rows. Every f64 is emitted as its IEEE-754 bit pattern
//! (`format!("{:016x}", v.to_bits())`, matching the Python side's
//! `struct.pack('<d')`); bools/ints/strings/None pass through. A domain
//! `Err` is emitted as `{"kind": "error", "error": <message>}` and ends the
//! case, mirroring the Python side's `ValueError` handling.

use std::io::Read;

use kart_physics::domains::booster_vehicle_step::{BoosterVehicleStepResult, InventoryUseResult};
use kart_physics::domains::branch_aware_vehicle_step::BranchAwareStepResult;
use kart_physics::domains::common_damping::CommonDampingResult;
use kart_physics::domains::contact_accumulator_producer::ContactAccumulatorProducerResult;
use kart_physics::domains::ground_drive_force_body::GroundDriveForceBodyResult;
use kart_physics::domains::ground_state_scalar::{
    FallbackLateralActions, GroundCorrectionPipelineResult, PipelineEvent, TraceStep,
};
use kart_physics::domains::low_projection_ground_branch::LowProjectionBranchResult;
use kart_physics::engine::{
    ContactProbeDebug, GroundStepOptions, KartRiderVehicleEngine, NumericMode, SurfaceSelection,
};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
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
    steps: Vec<SpecStep>,
}

#[derive(Deserialize)]
struct SpecStep {
    keys: Vec<String>,
    substep_ms: i64,
    #[serde(default)]
    ground_plane: Option<kart_physics::engine::GroundPlaneSet>,
    #[serde(default)]
    support_active: Option<bool>,
    #[serde(default)]
    producer: bool,
    #[serde(default = "default_true")]
    refresh_pending: bool,
    #[serde(default)]
    pending_substep_ms: Option<i64>,
}

fn default_true() -> bool {
    true
}

// ---------------------------------------------------------------------------
// Encoding helpers (same conventions as the leaf/orchestrator trace examples)
// ---------------------------------------------------------------------------

fn fbits(value: f64) -> Value {
    Value::String(format!("{:016x}", value.to_bits()))
}

fn opt_fbits(value: Option<f64>) -> Value {
    match value {
        Some(value) => fbits(value),
        None => Value::Null,
    }
}

fn arr<const N: usize>(values: &[f64; N]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn opt_arr<const N: usize>(values: &Option<[f64; N]>) -> Value {
    match values {
        Some(values) => arr(values),
        None => Value::Null,
    }
}

fn opt_leaf<T>(value: &Option<T>, encode: fn(&T) -> Value) -> Value {
    match value {
        Some(value) => encode(value),
        None => Value::Null,
    }
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
// Evaluator sub-object encoders (subset copied from branch_step_trace.rs;
// the Python side pops every `static_source` key recursively before diffing)
// ---------------------------------------------------------------------------

fn encode_trace(trace: &[TraceStep]) -> Value {
    Value::Array(
        trace
            .iter()
            .map(|step| {
                json!({
                    "step": step.step,
                    "operation": step.operation,
                    "operand": fbits(step.operand),
                    "result": fbits(step.result),
                })
            })
            .collect(),
    )
}

fn encode_inventory_use(r: &InventoryUseResult) -> Value {
    let mut row = json!({
        "consumed": r.consumed,
        "handler_accepted": r.handler_accepted,
        "state_changed": r.state_changed,
        "reason": r.reason,
        "slots_after": Value::Array(r.slots_after.iter().map(|slot| json!({
            "item_id": slot.item_id,
            "quantity": slot.quantity,
        })).collect()),
    });
    let map = row.as_object_mut().expect("row is an object");
    if let Some(item_id) = r.item_id {
        map.insert("item_id".to_string(), json!(item_id));
    }
    if let Some(state_setter_attempted) = r.state_setter_attempted {
        map.insert("state_setter_attempted".to_string(), json!(state_setter_attempted));
    }
    if let Some(target_state) = r.target_state {
        map.insert("target_state".to_string(), json!(target_state));
    }
    if let Some(duration_ms) = r.duration_ms {
        map.insert("duration_ms".to_string(), json!(duration_ms));
    }
    if let Some(state_after) = r.state_after {
        map.insert("state_after".to_string(), json!(state_after));
    }
    if let Some(remaining_ms_after) = r.remaining_ms_after {
        map.insert("remaining_ms_after".to_string(), json!(remaining_ms_after));
    }
    if let Some(static_order) = r.static_order {
        map.insert("static_order".to_string(), json!(static_order));
    }
    row
}

fn encode_boost(r: &BoosterVehicleStepResult) -> Value {
    json!({
        "state_before": r.state_before,
        "inventory_transition": match &r.inventory_transition {
            Some(transition) => encode_inventory_use(transition),
            None => Value::Null,
        },
        "state_before_timer": r.state_before_timer,
        "state_for_force": r.state_for_force,
        "boost_family_for_force": r.boost_family_for_force,
        "remaining_before_ms": r.remaining_before_ms,
        "state_scalar": {
            "state_mult": fbits(r.state_scalar.state_mult),
            "trace": encode_trace(&r.state_scalar.trace),
        },
        "primary_force": {
            "force_delta": arr(&r.primary_force.force_delta),
            "scalar": fbits(r.primary_force.scalar),
            "scope": r.primary_force.scope,
        },
        "timer_after": {
            "state": r.timer_after.state,
            "remaining_ms": r.timer_after.remaining_ms,
            "frame_ms": r.timer_after.frame_ms,
            "consumed_ms": r.timer_after.consumed_ms,
            "clears_on_tick": r.timer_after.clears_on_tick,
        },
        "ordering": r.ordering,
        "scope": r.scope,
    })
}

fn encode_ground_drive_force(r: &GroundDriveForceBodyResult) -> Value {
    json!({
        "path": r.path,
        "surface_mode": r.surface_mode,
        "surface_profile_source": r.surface_profile_source,
        "slip_factor": opt_fbits(r.slip_factor),
        "forward_target": fbits(r.forward_target),
        "longitudinal": fbits(r.longitudinal),
        "lateral": fbits(r.lateral),
        "local_force": arr(&r.local_force),
        "local_torque": arr(&r.local_torque),
        "torque_y_gain": fbits(r.torque_y_gain),
        "numeric_mode": r.numeric_mode,
        "state_after": {
            "state_5c8": fbits(r.state_after.state_5c8),
            "field_5d4": fbits(r.state_after.field_5d4),
            "b1c_charge_settlement": r.state_after.b1c_charge_settlement,
            "b10": fbits(r.state_after.b10),
            "b20": fbits(r.state_after.b20),
            "b38": fbits(r.state_after.b38),
            "b2c": fbits(r.state_after.b2c),
        },
        "ordering": r.ordering,
        "context_policy": r.context_policy,
    })
}

fn encode_low_projection(r: &LowProjectionBranchResult) -> Value {
    match r {
        LowProjectionBranchResult::NotSelected { projection_len } => json!({
            "selected": false,
            "reason": "projection_above_threshold",
            "projection_len": fbits(*projection_len),
        }),
        LowProjectionBranchResult::Selected(r) => json!({
            "selected": true,
            "reason": "projection_at_or_below_5",
            "projection_len": fbits(r.projection_len),
            "forward_target": fbits(r.forward_target),
            "normalized_velocity": fbits(r.normalized_velocity),
            "longitudinal_correction": fbits(r.longitudinal_correction),
            "lateral_correction": fbits(r.lateral_correction),
            "longitudinal": fbits(r.longitudinal),
            "lateral": fbits(r.lateral),
            "local_force": arr(&r.local_force),
            "local_torque": arr(&r.local_torque),
            "numeric_mode": r.numeric_mode,
            "state_after": {
                "state_5c8": fbits(r.state_after.state_5c8),
                "field_5d4": fbits(r.state_after.field_5d4),
                "selector_af4": r.state_after.selector_af4,
                "field_af8": fbits(r.state_after.field_af8),
                "selector_b08": r.state_after.selector_b08,
                "field_b0c": fbits(r.state_after.field_b0c),
                "selector_b5c": r.state_after.selector_b5c,
                "b1c_charge_settlement": r.state_after.b1c_charge_settlement,
                "b10": fbits(r.state_after.b10),
                "b20": fbits(r.state_after.b20),
                "b38": fbits(r.state_after.b38),
                "b2c": fbits(r.state_after.b2c),
            },
            "ordering": r.ordering,
        }),
    }
}

fn encode_contact(r: &ContactAccumulatorProducerResult) -> Value {
    json!({
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "slots": Value::Array(r.slots.iter().map(|slot| json!({
            "slot": slot.slot,
            "active": slot.active,
            "rate_gain": opt_fbits(slot.rate_gain),
            "raw_force": fbits(slot.raw_force),
            "alignment": opt_fbits(slot.alignment),
            "normal_force": fbits(slot.normal_force),
            "force_delta": arr(&slot.force_delta),
            "lever": arr(&slot.lever),
            "torque_delta": arr(&slot.torque_delta),
        })).collect()),
        "common_force_delta": arr(&r.common_force_delta),
    })
}

fn encode_damping(r: &CommonDampingResult) -> Value {
    json!({
        "linear_damping_delta": arr(&r.linear_damping_delta),
        "angular_damping_delta": arr(&r.angular_damping_delta),
        "normalized_velocity_direction": arr(&r.normalized_velocity_direction),
        "special_directional_delta": arr(&r.special_directional_delta),
        "directional_multiplier": fbits(r.directional_multiplier),
        "directional_multiplier_source": r.directional_multiplier_source,
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "numeric_mode": r.numeric_mode,
        "ordering": r.ordering,
    })
}

fn encode_selector(selector: &FallbackLateralActions) -> Value {
    json!({
        "actions_in_program_order": selector.actions_in_program_order,
        "timer_a80_after": fbits(selector.timer_a80_after),
        "primary_path_selected": selector.primary_path_selected,
        "fallback_path_selected": selector.fallback_path_selected,
        "fallback_initial": selector.fallback_initial,
        "fallback_before_lateral": selector.fallback_before_lateral,
        "skip_lateral": selector.skip_lateral,
        "evidence": selector.evidence,
    })
}

fn encode_event(event: &PipelineEvent) -> Value {
    match event {
        PipelineEvent::Primary {
            force_delta,
            scalar,
            scope,
        } => json!({
            "kind": "primary",
            "force_delta": arr(force_delta),
            "scalar": fbits(*scalar),
            "scope": scope,
        }),
        PipelineEvent::NegativeAdc {
            kind,
            branch,
            force_delta,
            scalar,
            evidence,
        } => json!({
            "kind": kind,
            "branch": branch,
            "force_delta": arr(force_delta),
            "scalar": fbits(*scalar),
            "evidence": evidence,
        }),
        PipelineEvent::FallbackOpposite {
            branch,
            force_delta,
            scalar,
            evidence,
        } => json!({
            "kind": 33,
            "branch": branch,
            "force_delta": arr(force_delta),
            "scalar": fbits(*scalar),
            "evidence": evidence,
        }),
        PipelineEvent::LateralSubtract {
            kind,
            branch,
            source,
            accumulator_delta,
            lateral_unit_residual,
            evidence,
        } => json!({
            "kind": kind,
            "branch": branch,
            "source": arr(source),
            "accumulator_delta": arr(accumulator_delta),
            "lateral_unit_residual": arr(lateral_unit_residual),
            "evidence": evidence,
        }),
        PipelineEvent::VelocityZero {
            kind,
            branch,
            pre_velocity,
            post_velocity,
            terminal_for_current_correction_path,
            evidence,
        } => json!({
            "kind": kind,
            "branch": branch,
            "pre_velocity": arr(pre_velocity),
            "post_velocity": arr(post_velocity),
            "terminal_for_current_correction_path": terminal_for_current_correction_path,
            "evidence": evidence,
        }),
    }
}

fn encode_pipeline(r: &GroundCorrectionPipelineResult) -> Value {
    let events = Value::Array(r.events.iter().map(encode_event).collect());
    if r.disabled {
        return json!({
            "disabled": true,
            "events": events,
            "timer_a80_after": fbits(r.timer_a80_after),
            "accumulator_614_before": arr(&r.accumulator_614_before),
            "accumulator_614_after": arr(&r.accumulator_614_after),
            "accumulator_620_before": arr(&r.accumulator_620_before),
            "accumulator_620_after": arr(&r.accumulator_620_after),
            "velocity_c0_before": arr(&r.velocity_c0_before),
            "velocity_c0_after": arr(&r.velocity_c0_after),
            "velocity_direct_write": r.velocity_direct_write,
        });
    }
    json!({
        "disabled": false,
        "selector": encode_selector(r.selector.as_ref().expect("enabled row has a selector")),
        "events": events,
        "timer_a80_after": fbits(r.timer_a80_after),
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "velocity_direct_write": r.velocity_direct_write,
    })
}

// ---------------------------------------------------------------------------
// Engine-level encoders
// ---------------------------------------------------------------------------

fn encode_drive_target(r: &kart_physics::domains::drive_target_state::DriveTargetRuntimeResult) -> Value {
    json!({
        "raw_target": fbits(r.state.raw_target),
        "exponent_input": fbits(r.state.exponent_input),
        "attenuation": fbits(r.state.attenuation),
        "candidate_aa4": fbits(r.state.candidate_aa4),
        "field_aa4_after": fbits(r.state.field_aa4_after),
        "field_aa8_after": fbits(r.state.field_aa8_after),
        "overshoot_clamped_to_aa8": r.state.overshoot_clamped_to_aa8,
        "ordering": r.state.ordering,
        "field_718": fbits(r.field_718),
        "scaled_tune_660": fbits(r.scaled_tune_660),
        "tune_accessor": r.tune_accessor,
    })
}

fn encode_surface_selection(selection: &SurfaceSelection) -> Value {
    let mut map = Map::new();
    map.insert("surface_mode".to_string(), json!(selection.surface_mode));
    map.insert("authority".to_string(), json!(selection.authority));
    if let Some(rule) = selection.selection_rule {
        map.insert("selection_rule".to_string(), json!(rule));
    }
    if let Some(tokens) = &selection.road_surface_tokens {
        map.insert("road_surface_tokens".to_string(), json!(tokens));
    }
    if let Some(counts) = &selection.mode_counts {
        // serde_json stringifies integer map keys, matching the JSON
        // round-trip the Python dict keys take on the way into the diff.
        map.insert("mode_counts".to_string(), json!(counts));
    }
    if let Some(branches) = &selection.dynamics_branches {
        map.insert("dynamics_branches".to_string(), json!(branches));
    }
    if let Some(payloads) = &selection.df_payloads {
        map.insert(
            "df_payloads".to_string(),
            Value::Array(
                payloads
                    .iter()
                    .map(|payload| payload.map_or(Value::Null, fbits))
                    .collect(),
            ),
        );
    }
    if let Some(branch) = &selection.selected_dynamics_branch {
        map.insert("selected_dynamics_branch".to_string(), json!(branch));
    }
    map.insert(
        "sampled_surface_mode".to_string(),
        json!(selection.sampled_surface_mode),
    );
    map.insert(
        "selector_refreshed".to_string(),
        json!(selection.selector_refreshed),
    );
    map.insert(
        "selector_period_ms".to_string(),
        fbits(selection.selector_period_ms),
    );
    map.insert(
        "selector_elapsed_ms".to_string(),
        fbits(selection.selector_elapsed_ms),
    );
    Value::Object(map)
}

fn encode_probe_debug(debug: &[ContactProbeDebug]) -> Value {
    Value::Array(
        debug
            .iter()
            .map(|entry| {
                json!({
                    "origin": arr(&entry.origin),
                    "ray": arr(&entry.ray),
                    "ray_t": opt_fbits(entry.ray_t),
                    "hit_point": arr(&entry.hit_point),
                    "hit_projection": opt_fbits(entry.hit_projection),
                    "position_projection": opt_fbits(entry.position_projection),
                    "compression": fbits(entry.compression),
                    "plane_normal": arr(&entry.plane_normal),
                    "plane_point": arr(&entry.plane_point),
                    "plane_node": match &entry.plane_node {
                        Some(node) => bits_json(
                            &serde_json::to_value(node).expect("JsonValue serializes"),
                        ),
                        None => Value::Null,
                    },
                    "plane_tri_index": entry.plane_tri_index,
                    "plane_surface": entry.plane_surface,
                })
            })
            .collect(),
    )
}

/// The `pending_producer_refresh` evaluator sub-keys
/// (kartrider_vehicle_engine.py:5984-5993).
fn encode_refresh_evaluator(evaluator: &BranchAwareStepResult) -> Vec<(&'static str, Value)> {
    vec![
        ("ground_drive_force", opt_leaf(&evaluator.ground_force, encode_ground_drive_force)),
        ("low_projection_branch", opt_leaf(&evaluator.low_projection, encode_low_projection)),
        ("boost_ground_primary", encode_boost(&evaluator.boost)),
        ("contact_accumulator", opt_leaf(&evaluator.contact, encode_contact)),
        ("accumulator_614_after_bs", opt_arr(&evaluator.accumulator_614_after_bs)),
        (
            "accumulator_614_after_post_bs_reset",
            opt_arr(&evaluator.accumulator_614_after_post_bs_reset),
        ),
        ("common_damping", opt_leaf(&evaluator.damping, encode_damping)),
        ("ground_correction", opt_leaf(&evaluator.correction, encode_pipeline)),
    ]
}

/// The producer-row evaluator subset: the refresh keys plus the persisted
/// accumulators and the selected producer name.
fn encode_producer_eval(evaluator: &BranchAwareStepResult) -> Value {
    let mut map = Map::new();
    for (key, value) in encode_refresh_evaluator(evaluator) {
        map.insert(key.to_string(), value);
    }
    map.insert(
        "accumulator_614_after_damping".to_string(),
        opt_arr(&evaluator.accumulator_614_after_damping),
    );
    map.insert(
        "accumulator_620_after_damping".to_string(),
        opt_arr(&evaluator.accumulator_620_after_damping),
    );
    map.insert(
        "selected_ground_force_producer".to_string(),
        match evaluator.selected_ground_force_producer {
            Some(name) => json!(name),
            None => Value::Null,
        },
    );
    Value::Object(map)
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
    for step in &case.steps {
        let parameters = VehicleParameters {
            substep_ms: step.substep_ms,
            ..case.parameters.clone()
        };
        if step.producer {
            match engine.produce_ground_accumulators(
                &step.keys,
                &parameters,
                step.ground_plane.as_ref(),
                step.support_active,
                false,
            ) {
                Ok(output) => rows.push(json!({
                    "kind": "producer",
                    "state_after": encode_state(&engine.state),
                    "surface_selection": encode_surface_selection(&output.surface_selection),
                    "drive_target": encode_drive_target(&output.drive_target),
                    "contact_probe_debug": encode_probe_debug(&output.contact_probe_debug),
                    "contact_compression_after": arr(&output.contact_compression_after),
                    "producer_contract": {
                        "state_mutation": output.producer_contract.state_mutation,
                        "integration_stages": output.producer_contract.integration_stages,
                        "authority": output.producer_contract.authority,
                    },
                    "producer_eval": encode_producer_eval(&output.evaluator),
                })),
                Err(message) => {
                    rows.push(json!({"kind": "error", "error": message}));
                    break;
                }
            }
            continue;
        }
        let pending_parameters = step.pending_substep_ms.map(|substep_ms| VehicleParameters {
            substep_ms,
            ..parameters.clone()
        });
        let options = GroundStepOptions {
            refresh_pending: step.refresh_pending,
            pending_parameters,
            support_active: step.support_active,
            ..Default::default()
        };
        match engine.step_raw_ground_correction(
            &step.keys,
            &parameters,
            None,
            step.ground_plane.as_ref(),
            &options,
        ) {
            Ok(outcome) => {
                let pending_refresh = match &outcome.pending_producer_refresh {
                    Some(refresh) => {
                        let mut map = Map::new();
                        map.insert("authority".to_string(), json!(refresh.authority));
                        map.insert(
                            "force_accumulator".to_string(),
                            arr(&refresh.force_accumulator),
                        );
                        map.insert(
                            "torque_accumulator".to_string(),
                            arr(&refresh.torque_accumulator),
                        );
                        map.insert(
                            "contact_compression".to_string(),
                            arr(&refresh.contact_compression),
                        );
                        map.insert(
                            "contact_probe_debug".to_string(),
                            encode_probe_debug(&refresh.producer.contact_probe_debug),
                        );
                        for (key, value) in
                            encode_refresh_evaluator(&refresh.producer.evaluator)
                        {
                            map.insert(key.to_string(), value);
                        }
                        Value::Object(map)
                    }
                    None => Value::Null,
                };
                rows.push(json!({
                    "kind": "integrate",
                    "state_before": encode_state(&outcome.step.state_before),
                    "state_after": encode_state(&outcome.step.state_after),
                    "teacher_forced_before": outcome.step.teacher_forced_before,
                    "accumulator_input_applied": outcome.step.accumulator_input_applied,
                    "kinematic_input_applied": outcome.step.kinematic_input_applied,
                    "prediction_errors": bits_json(
                        &serde_json::to_value(&outcome.step.prediction_errors)
                            .expect("prediction errors serialize"),
                    ),
                    "selected_ground_force_producer":
                        match outcome.step.selected_ground_force_producer {
                            Some(name) => json!(name),
                            None => Value::Null,
                        },
                    "surface_selection": encode_surface_selection(&outcome.surface_selection),
                    "drive_target": encode_drive_target(&outcome.drive_target),
                    "contact_probe_debug": encode_probe_debug(&outcome.contact_probe_debug),
                    "pending_producer_refresh": pending_refresh,
                    "raw_control_pose_contract": bits_json(
                        &serde_json::to_value(&outcome.raw_control_pose_contract)
                            .expect("pose contract serializes"),
                    ),
                }));
            }
            Err(message) => {
                rows.push(json!({"kind": "error", "error": message}));
                break;
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
