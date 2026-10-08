//! CTRL -> drift-selector parity trace.
//!
//! Reads `{"cases":[...]}` from stdin and emits one JSON row per case.
//! Every f64 is encoded as its IEEE-754 bit pattern so the Python driver can
//! require exact state and branch-output parity.

use std::io::Read;

use kart_physics::domains::branch_aware_vehicle_step::RecoveryScheduleOutcome;
use kart_physics::domains::drift_recovery_timer::{
    SelectorState, SelectorTransitionResult, Surface1RecoveryForceResult,
};
use kart_physics::domains::drift_window_branch::DriftWindowBranchResult;
use kart_physics::domains::low_projection_ground_branch::LowProjectionBranchResult;
use kart_physics::engine::{GroundStepOptions, KartRiderVehicleEngine, NumericMode};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct Spec {
    cases: Vec<SpecCase>,
}

#[derive(Deserialize)]
struct SpecCase {
    name: String,
    initial_state: VehicleState,
    parameters: VehicleParameters,
    numeric_mode: NumericMode,
    steps: Vec<Vec<String>>,
}

fn fbits(value: f64) -> Value {
    Value::String(format!("{:016x}", value.to_bits()))
}

fn arr(values: &[f64; 3]) -> Value {
    Value::Array(values.iter().map(|value| fbits(*value)).collect())
}

fn bits_json(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            fbits(number.as_f64().expect("f64 JSON number"))
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

fn encode_state(state: &VehicleState) -> Value {
    bits_json(&serde_json::to_value(state).expect("VehicleState serializes"))
}

fn encode_selector(state: SelectorState) -> Value {
    json!({
        "selector_af4": state.selector_af4,
        "selector_b08": state.selector_b08,
        "selector_b5c": state.selector_b5c,
        "selector_f08": state.selector_f08,
        "selector_fec": state.selector_fec,
        "selector_ff8": state.selector_ff8,
        "recovery_timer": fbits(state.recovery_timer),
    })
}

fn encode_transition(result: &Option<SelectorTransitionResult>) -> Value {
    match result {
        Some(result) => json!({
            "applied": result.applied,
            "reason": result.reason,
            "before": encode_selector(result.before),
            "after": encode_selector(result.after),
        }),
        None => Value::Null,
    }
}

fn encode_transitions(results: &[SelectorTransitionResult]) -> Value {
    Value::Array(
        results
            .iter()
            .map(|result| {
                json!({
                    "applied": result.applied,
                    "reason": result.reason,
                    "before": encode_selector(result.before),
                    "after": encode_selector(result.after),
                })
            })
            .collect(),
    )
}

fn encode_drift(result: &Option<DriftWindowBranchResult>) -> Value {
    match result {
        None => Value::Null,
        Some(DriftWindowBranchResult::NotSelected) => json!({"selected": false}),
        Some(DriftWindowBranchResult::Selected(result)) => json!({
            "selected": true,
            "local_force": arr(&result.local_force),
            "local_torque": arr(&result.local_torque),
            "timer_initialized_after_force": result.timer_initialized_after_force,
            "timer_expired_after_force": result.timer_expired_after_force,
            "state_after": {
                "selector_b08": result.state_after.selector_b08,
                "timer_b0c": fbits(result.state_after.timer_b0c),
                "field_b00": result.state_after.field_b00.map(fbits),
                "snapshot_5d0": result.state_after.snapshot_5d0.map(fbits),
                "b1c_charge_settlement": result.state_after.b1c_charge_settlement,
            },
        }),
    }
}

fn encode_recovery(result: &RecoveryScheduleOutcome) -> Value {
    match result {
        RecoveryScheduleOutcome::Suppressed { reason } => {
            json!({"suppressed": true, "reason": reason})
        }
        RecoveryScheduleOutcome::Evaluated(result) => json!({
            "suppressed": false,
            "branch_for_force": result.branch_for_force,
            "recovery_timer_before": fbits(result.recovery_timer_before),
            "recovery_timer_after": fbits(result.recovery_timer_after),
            "field_af8_before": fbits(result.field_af8_before),
            "field_af8_after": fbits(result.field_af8_after),
        }),
    }
}

fn encode_recovery_force(result: &Option<Surface1RecoveryForceResult>) -> Value {
    match result {
        None => Value::Null,
        Some(result) => json!({
            "path": result.path,
            "blend_ab0_after": fbits(result.blend_ab0_after),
            "blend_ab4_after": fbits(result.blend_ab4_after),
            "local_force": arr(&result.local_force),
            "local_torque": arr(&result.local_torque),
        }),
    }
}

fn encode_low_projection(result: &Option<LowProjectionBranchResult>) -> Value {
    match result {
        None => Value::Null,
        Some(LowProjectionBranchResult::NotSelected { .. }) => {
            json!({"selected": false})
        }
        Some(LowProjectionBranchResult::Selected(_)) => json!({"selected": true}),
    }
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Spec = serde_json::from_str(&input).expect("parse spec");
    let mut emitted = Vec::with_capacity(spec.cases.len());
    for case in spec.cases {
        let mut engine =
            KartRiderVehicleEngine::new(&case.initial_state, case.numeric_mode).expect("state");
        let options = GroundStepOptions {
            refresh_pending: false,
            ..Default::default()
        };
        let mut rows = Vec::with_capacity(case.steps.len());
        for keys in case.steps {
            let before = engine.state.clone();
            match engine.step_raw_ground_correction(&keys, &case.parameters, None, None, &options) {
                Ok(outcome) => rows.push(json!({
                    "state_before": encode_state(&before),
                    "state_after": encode_state(&engine.state),
                    "drift_selector_transition": encode_transition(
                        &outcome.drift_selector_transition
                    ),
                    "drift_selector_transitions": encode_transitions(
                        &outcome.drift_selector_transitions
                    ),
                    "selected_ground_force_producer":
                        outcome.step.evaluator.selected_ground_force_producer,
                    "low_projection_branch":
                        encode_low_projection(&outcome.step.evaluator.low_projection),
                    "drift_window_branch":
                        encode_drift(&outcome.step.evaluator.drift_window),
                    "surface1_recovery_schedule":
                        encode_recovery(&outcome.step.evaluator.recovery_schedule),
                    "surface1_recovery_force":
                        encode_recovery_force(&outcome.step.evaluator.recovery_force),
                })),
                Err(error) => {
                    rows.push(json!({"error": error}));
                    break;
                }
            }
        }
        emitted.push(json!({"name": case.name, "rows": rows}));
    }
    println!(
        "{}",
        serde_json::to_string(&json!({"cases": emitted})).expect("serialize output")
    );
}
