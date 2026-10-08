//! Branch-aware vehicle step (orchestrator) parity trace tool.
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"kwargs": {...}}, ...]}` where each `kwargs` object mirrors
//! the Python `evaluate_branch_aware_step(**kwargs)` signature. Prints a
//! JSON array of result rows, one per case. Every f64 is emitted as its
//! IEEE-754 bit pattern (`format!("{:016x}", v.to_bits())`, matching the
//! Python side's `struct.pack('<d')`); bools/ints/strings/None pass
//! through. A domain `Err` is emitted as `{"error": <message>}` so the
//! Python side can diff `ValueError` texts bit-for-bit.
//!
//! The emitted row reproduces the Python orchestrator's ordered result
//! dict, minus every `static_source` key (the established leaf convention:
//! the Python side pops them recursively before comparing).

use std::io::Read;

use kart_physics::domains::booster_resource::InventorySlot;
use kart_physics::domains::booster_vehicle_step::{
    BoosterVehicleStepResult, InventoryUseResult,
};
use kart_physics::domains::branch_aware_vehicle_step::{
    self, evaluate_branch_aware_step, BranchAwareStepArgs, BranchAwareStepResult,
    PostBsResetSelectorOutcome, RailSelectorOutcome, RecoveryScheduleOutcome, SoftContactOutcome,
};
use kart_physics::domains::bs_zone_override::BsZoneOverrideResult;
use kart_physics::domains::checkpoint_respawn_state_replacement::{
    self, CheckpointRespawnStateReplacement,
};
use kart_physics::domains::checkpoint_target_selector::{
    self, CheckpointTargetSelectorResult,
};
use kart_physics::domains::common_damping::CommonDampingResult;
use kart_physics::domains::contact_accumulator_producer::ContactAccumulatorProducerResult;
use kart_physics::domains::drift_recovery_timer::{
    RecoverySubstepResult, Surface1RecoveryForceResult,
};
use kart_physics::domains::drift_window_branch::DriftWindowBranchResult;
use kart_physics::domains::fall_clamp_state_replacement::{
    self, FallClampStateReplacement,
};
use kart_physics::domains::ground_accumulator_sink::GroundAccumulatorSinkResult;
use kart_physics::domains::ground_drive_force_body::GroundDriveForceBodyResult;
use kart_physics::domains::ground_state_scalar::{
    FallbackLateralActions, GroundCorrectionPipelineResult, PipelineEvent, TraceStep,
};
use kart_physics::domains::hard_contact_velocity_response::{
    self, HardContactVelocityResponse,
};
use kart_physics::domains::jm_dj_trigger::JmDjTrigger;
use kart_physics::domains::low_projection_ground_branch::LowProjectionBranchResult;
use kart_physics::domains::mz_magnet_trigger::{self, MzMagnetTrigger};
use kart_physics::domains::pit_resource_step::PitResourceStep;
use kart_physics::domains::post_bs_reset_constraint::PostBsResetConstraintResult;
use kart_physics::domains::rail_accumulator_producers::{
    RailAccumulatorKinds, RailAccumulatorProducersResult,
};
use kart_physics::domains::soft_contact_velocity_response::{
    self, SoftContactVelocityResponse,
};
use kart_physics::domains::speed_threshold_latch::SpeedThresholdLatchResult;
use kart_physics::domains::substep_accumulator_boundary::SubstepAccumulatorBoundaryResult;
use kart_physics::domains::tune_boost_gauge::{self, BchargeSurfaceStep};
use kart_physics::domains::angular_velocity_integrator::AngularVelocityIntegratorResult;
use kart_physics::domains::linear_velocity_integrator::LinearVelocityIntegratorResult;
use kart_physics::domains::orientation_integrator::OrientationIntegratorResult;
use kart_physics::domains::position_integrator::PositionIntegratorResult;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Encoding helpers (same conventions as the leaf trace examples)
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

fn opt_str(value: &Option<String>) -> Value {
    match value {
        Some(value) => json!(value),
        None => Value::Null,
    }
}

fn slots_json(slots: &[InventorySlot]) -> Value {
    Value::Array(
        slots
            .iter()
            .map(|slot| json!({"item_id": slot.item_id, "quantity": slot.quantity}))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Leaf encoders (mirrors of the leaf trace examples; Python dicts minus
// `static_source`, which the Python side pops recursively).
// ---------------------------------------------------------------------------

fn encode_inventory_use(r: &InventoryUseResult) -> Value {
    let mut row = json!({
        "consumed": r.consumed,
        "handler_accepted": r.handler_accepted,
        "state_changed": r.state_changed,
        "reason": r.reason,
        "slots_after": slots_json(&r.slots_after),
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

fn encode_pit_resource(r: &PitResourceStep) -> Value {
    json!({
        "active_before_cleanup": r.active_before_cleanup,
        "rate_before_cleanup": fbits(r.rate_before_cleanup),
        "mode_before_cleanup": r.mode_before_cleanup,
        "accum_before": fbits(r.accum_before),
        "accum_increment": fbits(r.accum_increment),
        "accum_after": fbits(r.accum_after),
        "fx_selected": r.fx_selected,
        "active_after_cleanup": r.active_after_cleanup,
        "rate_after_cleanup": fbits(r.rate_after_cleanup),
        "mode_after_cleanup": r.mode_after_cleanup,
        "vehicle_force_delta": arr(&r.vehicle_force_delta),
        "vehicle_torque_delta": arr(&r.vehicle_torque_delta),
        "dynamics_role": r.dynamics_role,
        "drain_consumer": r.drain_consumer,
    })
}

fn encode_bcharge(r: &BchargeSurfaceStep) -> Value {
    json!({
        "active_before_cleanup": r.active_before_cleanup,
        "accum_b10_before": fbits(r.accum_b10_before),
        "increment": fbits(r.increment),
        "accum_b10_after_add": fbits(r.accum_b10_after_add),
        "cap": fbits(r.cap),
        "accum_b10_after": fbits(r.accum_b10_after),
        "active_after_cleanup": r.active_after_cleanup,
        "vehicle_force_delta": arr(&r.vehicle_force_delta),
        "vehicle_torque_delta": arr(&r.vehicle_torque_delta),
        "dynamics_role": tune_boost_gauge::DYNAMICS_ROLE,
        "scheduler_phase": tune_boost_gauge::SCHEDULER_PHASE,
    })
}

fn encode_recovery_substep(r: &RecoverySubstepResult) -> Value {
    json!({
        "branch_for_force": r.branch_for_force,
        "recovery_timer_before": fbits(r.recovery_timer_before),
        "recovery_timer_after": fbits(r.recovery_timer_after),
        "timer_expired_after_force": r.timer_expired_after_force,
        "timer_expired_before_sink": r.timer_expired_before_sink,
        "field_af8_before": fbits(r.field_af8_before),
        "field_af8_after": fbits(r.field_af8_after),
        "field_af8_expired_after_force": r.field_af8_expired_after_force,
        "field_af8_expired_before_sink": r.field_af8_expired_before_sink,
        "ordering": r.ordering,
        "scope": r.scope,
    })
}

fn encode_recovery_schedule(outcome: &RecoveryScheduleOutcome) -> Value {
    match outcome {
        RecoveryScheduleOutcome::Suppressed { reason } => json!({
            "branch_for_force": false,
            "suppressed": true,
            "reason": reason,
        }),
        RecoveryScheduleOutcome::Evaluated(result) => encode_recovery_substep(result),
    }
}

fn encode_surface1_recovery_force(r: &Surface1RecoveryForceResult) -> Value {
    json!({
        "path": r.path,
        "surface_mode": r.surface_mode,
        "surface_profile_source": r.surface_profile_source,
        "surface_force_scale_read": fbits(r.surface_force_scale_read),
        "surface_force_scale_applied": r.surface_force_scale_applied,
        "speed_ratio_sq": fbits(r.speed_ratio_sq),
        "blend_ab0_before": fbits(r.blend_ab0_before),
        "blend_ab0_after": fbits(r.blend_ab0_after),
        "blend_ab4_before": fbits(r.blend_ab4_before),
        "blend_ab4_after": fbits(r.blend_ab4_after),
        "forward_target": fbits(r.forward_target),
        "longitudinal": fbits(r.longitudinal),
        "lateral": fbits(r.lateral),
        "local_force": arr(&r.local_force),
        "local_torque": arr(&r.local_torque),
        "af0_torque_gate": fbits(r.af0_torque_gate),
        "ordering": r.ordering,
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

fn encode_drift_window(r: &DriftWindowBranchResult) -> Value {
    match r {
        DriftWindowBranchResult::NotSelected => json!({
            "selected": false,
            "reason": "selector_b08_clear",
        }),
        DriftWindowBranchResult::Selected(r) => json!({
            "selected": true,
            "longitudinal": fbits(r.longitudinal),
            "lateral": fbits(r.lateral),
            "local_force": arr(&r.local_force),
            "local_torque": arr(&r.local_torque),
            "af0_torque_gate": fbits(r.af0_torque_gate),
            "timer_initialized_after_force": r.timer_initialized_after_force,
            "timer_expired_after_force": r.timer_expired_after_force,
            "state_after": {
                "blend_ab0": fbits(r.state_after.blend_ab0),
                "blend_ab4": fbits(r.state_after.blend_ab4),
                "selector_b08": r.state_after.selector_b08,
                "timer_b0c": fbits(r.state_after.timer_b0c),
                "field_b00": opt_fbits(r.state_after.field_b00),
                "snapshot_5d0": opt_fbits(r.state_after.snapshot_5d0),
                "b1c_charge_settlement": r.state_after.b1c_charge_settlement,
                "b20": opt_fbits(r.state_after.b20),
                "b2c": opt_fbits(r.state_after.b2c),
            },
            "ordering": r.ordering,
        }),
    }
}

fn encode_boundary(r: &SubstepAccumulatorBoundaryResult) -> Value {
    json!({
        "active_force_614_before": opt_arr(&r.active_force_614_before),
        "active_torque_620_before": opt_arr(&r.active_torque_620_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "pending_force_b74_after": arr(&r.pending_force_b74_after),
        "pending_torque_b80_after": arr(&r.pending_torque_b80_after),
        "pending_b90_after": arr(&r.pending_b90_after),
        "operation": r.operation,
    })
}

fn encode_rail_kinds(r: &RailAccumulatorKinds) -> Value {
    json!({
        "enabled_kinds": r.enabled_kinds,
        "force_branch": r.force_branch,
        "runtime_policy": r.runtime_policy,
    })
}

fn encode_rail_selector(outcome: &RailSelectorOutcome) -> Value {
    match outcome {
        RailSelectorOutcome::Exact(kinds) => {
            let mut row = encode_rail_kinds(kinds);
            row.as_object_mut()
                .expect("row is an object")
                .insert("selection_authority".to_string(), json!("exact_selector"));
            row
        }
        RailSelectorOutcome::Observed { enabled_kinds } => json!({
            "enabled_kinds": enabled_kinds,
            "selection_authority": "observed_probe_rows",
            "runtime_policy": "replay only; explicit kinds must come from authoritative probe_rail writer rows",
        }),
    }
}

fn encode_rail_producers(r: &RailAccumulatorProducersResult) -> Value {
    json!({
        "enabled_kinds": r.enabled_kinds,
        "events": Value::Array(r.events.iter().map(|event| json!({
            "kind": event.kind,
            "target": event.target,
            "operation": event.operation,
            "source": arr(&event.source),
        })).collect()),
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "gate_policy": r.gate_policy,
    })
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

fn encode_hard_contact(r: &HardContactVelocityResponse) -> Value {
    json!({
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "normal_dot_velocity": fbits(r.normal_dot_velocity),
        "normal_component": arr(&r.normal_component),
        "tangent_velocity": arr(&r.tangent_velocity),
        "basis_ad0_dot_normal": fbits(r.basis_ad0_dot_normal),
        "reflected_scale": fbits(r.reflected_scale),
        "direct_velocity_replacement": r.direct_velocity_replacement,
        "runtime_evidence": hard_contact_velocity_response::RUNTIME_EVIDENCE,
    })
}

fn encode_soft_contact(r: &SoftContactVelocityResponse) -> Value {
    json!({
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_delta": arr(&r.velocity_delta),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "direct_velocity_add": r.direct_velocity_add,
        "ordering": soft_contact_velocity_response::ORDERING,
        "runtime_authority": soft_contact_velocity_response::RUNTIME_AUTHORITY,
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

fn encode_sink(r: &GroundAccumulatorSinkResult) -> Value {
    json!({
        "local_force": arr(&r.local_force),
        "world_force": arr(&r.world_force),
        "local_torque": arr(&r.local_torque),
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "numeric_mode": r.numeric_mode,
        "ordering": r.ordering,
    })
}

fn encode_bs_zone(r: &BsZoneOverrideResult) -> Value {
    json!({
        "token": r.token,
        "token_value": fbits(r.token_value),
        "magnitude": fbits(r.magnitude),
        "velocity_direction": arr(&r.velocity_direction),
        "bs_vector": arr(&r.bs_vector),
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "discarded_accumulator_614": arr(&r.discarded_accumulator_614),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "state_after": {
            "decoded_state": r.state_after.decoded_state,
            "remaining_ms": r.state_after.remaining_ms,
        },
        "ordering": r.ordering,
    })
}

fn encode_mz_magnet(r: &MzMagnetTrigger) -> Value {
    json!({
        "accepted": r.accepted,
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "state_transition": match &r.state_transition {
            Some(t) => json!({"state": t.state, "remaining_ms": t.remaining_ms}),
            None => Value::Null,
        },
        "velocity_write_occurs_before_state_family_gate": r.velocity_write_occurs_before_state_family_gate,
        "policy": mz_magnet_trigger::POLICY,
        "static_order": mz_magnet_trigger::STATIC_ORDER,
    })
}

fn encode_jm_dj(r: &JmDjTrigger) -> Value {
    json!({
        "variant": r.variant,
        "branch": r.branch,
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "accepted": r.accepted,
        "state_transition": match &r.state_transition {
            Some(t) => json!({"state": t.state, "remaining_ms": t.remaining_ms}),
            None => Value::Null,
        },
        "cooldown_648_after": fbits(r.cooldown_648_after),
        "velocity_write_occurs_before_state_family_gate": r.velocity_write_occurs_before_state_family_gate,
        "velocity_operation": r.velocity_operation,
        "policy": r.policy,
    })
}

fn encode_post_bs_reset_selector(outcome: &PostBsResetSelectorOutcome) -> Value {
    match outcome {
        PostBsResetSelectorOutcome::Exact(r) => json!({
            "selected": r.selected,
            "contact_or_trigger": r.contact_or_trigger,
            "rail_active": r.rail_active,
            "ordering": r.ordering,
            "selection_authority": "exact_selector",
        }),
        PostBsResetSelectorOutcome::ObservedProbeRows => json!({
            "selected": true,
            "selection_authority": "observed_probe_rows",
        }),
    }
}

fn encode_post_bs_reset(r: &PostBsResetConstraintResult) -> Value {
    json!({
        "force_delta": arr(&r.force_delta),
        "base_torque_delta": arr(&r.base_torque_delta),
        "orientation_torque_delta": arr(&r.orientation_torque_delta),
        "torque_delta": arr(&r.torque_delta),
        "accumulator_614_before": arr(&r.accumulator_614_before),
        "accumulator_614_after": arr(&r.accumulator_614_after),
        "accumulator_620_before": arr(&r.accumulator_620_before),
        "accumulator_620_after": arr(&r.accumulator_620_after),
        "state_after": {
            "selector_af4": r.state_after.selector_af4,
            "decoded_afc": r.state_after.decoded_afc,
            "selector_b08": r.state_after.selector_b08,
            "field_aa8": fbits(r.state_after.field_aa8),
        },
        "ordering": r.ordering,
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

fn encode_speed_threshold(r: &SpeedThresholdLatchResult) -> Value {
    json!({
        "threshold_b8c_before": fbits(r.threshold_b8c_before),
        "speed_sq": fbits(r.speed_sq),
        "threshold_sq": fbits(r.threshold_sq),
        "clears": r.clears,
        "threshold_b8c_after": fbits(r.threshold_b8c_after),
        "ordering": r.ordering,
    })
}

fn encode_linear(r: &LinearVelocityIntegratorResult) -> Value {
    json!({
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "post_bs_later_delta": arr(&r.post_bs_later_delta),
        "post_bs_later_delta_policy": r.post_bs_later_delta_policy,
        "accumulator_614_before_bac": arr(&r.accumulator_614_before_bac),
        "bac_force": arr(&r.bac_force),
        "accumulator_614_after_bac": arr(&r.accumulator_614_after_bac),
        "accumulator_zero_cleared_velocity": r.accumulator_zero_cleared_velocity,
        "delta_velocity": arr(&r.delta_velocity),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "numeric_mode": r.numeric_mode,
        "ordering": r.ordering,
    })
}

fn encode_angular(r: &AngularVelocityIntegratorResult) -> Value {
    json!({
        "angular_velocity_cc_before": arr(&r.angular_velocity_cc_before),
        "mapped_omega": arr(&r.mapped_omega),
        "gyroscopic_term": arr(&r.gyroscopic_term),
        "accumulator_620": arr(&r.accumulator_620),
        "torque_minus_gyro": arr(&r.torque_minus_gyro),
        "angular_acceleration": arr(&r.angular_acceleration),
        "delta_angular_velocity": arr(&r.delta_angular_velocity),
        "angular_velocity_cc_after": arr(&r.angular_velocity_cc_after),
        "numeric_mode": r.numeric_mode,
        "ordering": r.ordering,
    })
}

fn encode_position(r: &PositionIntegratorResult) -> Value {
    json!({
        "position_90_before": arr(&r.position_90_before),
        "velocity_c0_before_b2fd80": arr(&r.velocity_c0_before_b2fd80),
        "pre_position_branch_selected": r.pre_position_branch_selected,
        "pre_position_branch_delta": arr(&r.pre_position_branch_delta),
        "velocity_c0_after_pre_position_branch": arr(&r.velocity_c0_after_pre_position_branch),
        "latch_658_after": r.latch_658_after,
        "pre_position_velocity_delta": arr(&r.pre_position_velocity_delta),
        "velocity_c0_for_position": arr(&r.velocity_c0_for_position),
        "position_90_after": arr(&r.position_90_after),
        "dt": fbits(r.dt),
        "scope": r.scope,
    })
}

fn encode_orientation(r: &OrientationIntegratorResult) -> Value {
    let retries: Vec<Value> = r
        .stabilization_retries
        .iter()
        .map(|retry| {
            json!({
                "attempt": retry.attempt,
                "axis2_z_before": fbits(retry.axis2_z_before),
                "operation": retry.operation,
                "angular_velocity_after_operation": arr(&retry.angular_velocity_after_operation),
                "axis2_z_after": fbits(retry.axis2_z_after),
            })
        })
        .collect();
    json!({
        "matrix_9c_before": arr(&r.matrix_9c_before),
        "quaternion_backup": arr(&r.quaternion_backup),
        "quaternion_after": arr(&r.quaternion_after),
        "matrix_9c_after": arr(&r.matrix_9c_after),
        "angular_velocity_cc_before": arr(&r.angular_velocity_cc_before),
        "angular_velocity_cc_after": arr(&r.angular_velocity_cc_after),
        "pre_clear_xy": r.pre_clear_xy,
        "stabilization_enabled": r.stabilization_enabled,
        "stabilization_retries": retries,
        "axis2_z_after": fbits(r.axis2_z_after),
    })
}

fn encode_fall_clamp(r: &FallClampStateReplacement) -> Value {
    json!({
        "selected": r.selected,
        "terminal_replacement": r.terminal_replacement,
        "predicate": r.predicate,
        "drop_threshold": fbits(r.drop_threshold),
        "position_90_before": arr(&r.position_90_before),
        "position_90_after": arr(&r.position_90_after),
        "previous_position_70_after": opt_arr(&r.previous_position_70_after),
        "matrix_9c_before": arr(&r.matrix_9c_before),
        "matrix_9c_after": arr(&r.matrix_9c_after),
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "angular_velocity_cc_before": arr(&r.angular_velocity_cc_before),
        "angular_velocity_cc_after": arr(&r.angular_velocity_cc_after),
        "checkpoint_respawn_modeled": r.checkpoint_respawn_modeled,
        "source": {
            "selector_va": fall_clamp_state_replacement::SOURCE_SELECTOR_VA,
            "writer_va": fall_clamp_state_replacement::SOURCE_WRITER_VA,
            "writer_rva": fall_clamp_state_replacement::SOURCE_WRITER_RVA,
            "callsite_va": fall_clamp_state_replacement::SOURCE_CALLSITE_VA,
            "callsite_rva": fall_clamp_state_replacement::SOURCE_CALLSITE_RVA,
        },
    })
}

fn encode_checkpoint_respawn(r: &CheckpointRespawnStateReplacement) -> Value {
    json!({
        "selected": r.selected,
        "terminal_replacement": r.terminal_replacement,
        "mode_family": r.mode_family,
        "checkpoint_gate": r.checkpoint_gate,
        "sample_result_zero": r.sample_result_zero,
        "target_position": arr(&r.target_position),
        "course_reference_position": arr(&r.course_reference_position),
        "forward_axis1": arr(&r.forward_axis1),
        "right_axis0": arr(&r.right_axis0),
        "up_axis2": arr(&r.up_axis2),
        "position_90_after": opt_arr(&r.position_90_after),
        "previous_position_70_after": opt_arr(&r.previous_position_70_after),
        "matrix_9c_after": opt_arr(&r.matrix_9c_after),
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "angular_velocity_cc_before": arr(&r.angular_velocity_cc_before),
        "angular_velocity_cc_after": arr(&r.angular_velocity_cc_after),
        "stage_offsets": {
            "function_va": r.stage_offsets.function_va,
            "gate": r.stage_offsets.gate,
            "target": r.stage_offsets.target,
        },
        "source": {
            "outer_selector_va": checkpoint_respawn_state_replacement::SOURCE_OUTER_SELECTOR_VA,
            "virtual_slot": checkpoint_respawn_state_replacement::SOURCE_VIRTUAL_SLOT,
            "writer_va": checkpoint_respawn_state_replacement::SOURCE_WRITER_VA,
            "vector_subtract_va": checkpoint_respawn_state_replacement::SOURCE_VECTOR_SUBTRACT_VA,
            "cross_va": checkpoint_respawn_state_replacement::SOURCE_CROSS_VA,
            "normalize_va": checkpoint_respawn_state_replacement::SOURCE_NORMALIZE_VA,
            "matrix_column_setter_va": checkpoint_respawn_state_replacement::SOURCE_MATRIX_COLUMN_SETTER_VA,
        },
    })
}

fn encode_checkpoint_target(r: &CheckpointTargetSelectorResult) -> Value {
    json!({
        "mode_family": r.mode_family,
        "event": r.event,
        "accepted": r.accepted,
        "target_before": arr(&r.target_before),
        "target_after": arr(&r.target_after),
        "target_written": r.target_written,
        "gate_before": r.gate_before,
        "gate_after": r.gate_after,
        "gate_written": r.gate_written,
        "source": {
            "c016_init_va": checkpoint_target_selector::SOURCE_C016_INIT_VA,
            "c016_progress_snapshot_va": checkpoint_target_selector::SOURCE_C016_PROGRESS_SNAPSHOT_VA,
            "c016_gate_writers": checkpoint_target_selector::SOURCE_C016_GATE_WRITERS,
            "c72c_init_va": checkpoint_target_selector::SOURCE_C72C_INIT_VA,
            "c72c_progress_snapshot_va": checkpoint_target_selector::SOURCE_C72C_PROGRESS_SNAPSHOT_VA,
        },
        "scope": r.scope,
    })
}

// ---------------------------------------------------------------------------
// Top-level orchestrator encoder: Python `evaluate_branch_aware_step`
// result dict (minus every `static_source` key, popped recursively on the
// Python side before comparison).
// ---------------------------------------------------------------------------

fn opt_leaf<T>(value: &Option<T>, encode: fn(&T) -> Value) -> Value {
    match value {
        Some(value) => encode(value),
        None => Value::Null,
    }
}

fn encode_soft_outcome(outcome: &SoftContactOutcome) -> Value {
    match outcome {
        SoftContactOutcome::NotSelected => Value::Null,
        SoftContactOutcome::Pending {
            velocity_c0_before,
            velocity_delta,
            selection_authority,
        } => {
            let mut pending = serde_json::Map::new();
            if let Some(velocity) = velocity_c0_before {
                pending.insert("velocity_c0_before".to_string(), arr(velocity));
            }
            pending.insert("velocity_delta".to_string(), arr(velocity_delta));
            json!({
                "pending_args": Value::Object(pending),
                "selection_authority": selection_authority,
            })
        }
        SoftContactOutcome::Evaluated {
            result,
            selection_authority,
            scheduler_phase,
        } => {
            let mut row = encode_soft_contact(result);
            let map = row.as_object_mut().expect("row is an object");
            map.insert(
                "selection_authority".to_string(),
                json!(selection_authority),
            );
            map.insert("scheduler_phase".to_string(), json!(scheduler_phase));
            row
        }
    }
}

fn encode_hard_outcome(result: &BranchAwareStepResult) -> Value {
    match &result.hard_contact {
        Some(hard) => {
            let mut row = encode_hard_contact(hard);
            let map = row.as_object_mut().expect("row is an object");
            map.insert(
                "selection_authority".to_string(),
                opt_str(&result.hard_contact_selection_authority),
            );
            map.insert(
                "scheduler_phase".to_string(),
                opt_str(&result.hard_contact_scheduler_phase),
            );
            row
        }
        None => Value::Null,
    }
}

fn encode_bs_outcome(result: &BranchAwareStepResult) -> Value {
    match &result.bs_zone {
        Some(bs) => {
            let mut row = encode_bs_zone(bs);
            row.as_object_mut().expect("row is an object").insert(
                "selection_authority".to_string(),
                opt_str(&result.bs_zone_selection_authority),
            );
            row
        }
        None => Value::Null,
    }
}

fn encode_mz_outcome(result: &BranchAwareStepResult) -> Value {
    match &result.mz_magnet {
        Some(mz) => {
            let mut row = encode_mz_magnet(mz);
            row.as_object_mut().expect("row is an object").insert(
                "selection_authority".to_string(),
                opt_str(&result.mz_magnet_selection_authority),
            );
            row
        }
        None => Value::Null,
    }
}

fn encode_jm_dj_outcome(result: &BranchAwareStepResult) -> Value {
    match &result.jm_dj {
        Some(jm) => {
            let mut row = encode_jm_dj(jm);
            row.as_object_mut().expect("row is an object").insert(
                "selection_authority".to_string(),
                opt_str(&result.jm_dj_selection_authority),
            );
            row
        }
        None => Value::Null,
    }
}

fn encode_static_sources() -> Value {
    let mut map = serde_json::Map::new();
    for (name, vas) in branch_aware_vehicle_step::STATIC_SOURCES {
        let value = if vas.len() == 1 {
            json!(vas[0])
        } else {
            json!(vas)
        };
        map.insert(name.to_string(), value);
    }
    Value::Object(map)
}

fn encode_result(r: &BranchAwareStepResult) -> Value {
    let low_selected = matches!(
        &r.low_projection,
        Some(kart_physics::domains::low_projection_ground_branch::LowProjectionBranchResult::Selected(_))
    );
    let drift_selected = matches!(
        &r.drift_window,
        Some(DriftWindowBranchResult::Selected(_))
    );
    let low_projection_status = if low_selected {
        "selected"
    } else if r.low_projection.is_some() {
        "not_selected"
    } else {
        "parameters_missing"
    };
    let drift_window_status = if drift_selected {
        "selected"
    } else if r.drift_window.is_some() {
        "not_selected"
    } else if low_selected {
        "suppressed_by_low_projection"
    } else {
        "parameters_missing"
    };
    let post_bs_selector = opt_leaf(&r.post_bs_reset_selector, encode_post_bs_reset_selector);
    let post_bs_constraint = opt_leaf(&r.post_bs_reset, encode_post_bs_reset);
    let mut map = serde_json::Map::new();
    macro_rules! put {
        ($key:literal, $value:expr) => {
            map.insert($key.to_string(), serde_json::json!($value))
        };
    }

    put!("elapsed_ms", r.elapsed_ms);
    put!("substep_accumulator_boundary", opt_leaf(&r.boundary, encode_boundary));
    put!("substep_accumulator_boundary_status", if r.boundary.is_some() {
            "evaluated_transfer"
        } else {
            "caller_supplied_active_accumulator"
        });
    put!("boost_ground_primary", encode_boost(&r.boost));
    put!("state18_feedback_selected", r.state18_feedback_selected);
    put!("state18_feedback_status", r.state18_feedback_status);
    put!("pit_resource", opt_leaf(&r.pit_resource, encode_pit_resource));
    put!("pit_resource_status", if r.pit_resource.is_some() {
            "integrated_resource_no_force"
        } else {
            "not_requested"
        });
    put!("bcharge_resource", opt_leaf(&r.bcharge_resource, encode_bcharge));
    put!("bcharge_resource_status", if r.bcharge_resource.is_some() {
            "integrated_resource_no_force"
        } else {
            "not_requested"
        });
    put!("surface1_recovery_schedule", encode_recovery_schedule(&r.recovery_schedule));
    put!("surface1_recovery_force", opt_leaf(&r.recovery_force, encode_surface1_recovery_force));
    put!("surface1_recovery_force_status", r.surface1_recovery_force_status);
    put!("ground_drive_force", opt_leaf(&r.ground_force, encode_ground_drive_force));
    put!("ground_drive_force_status", r.ground_drive_force_status);
    put!("low_projection_branch", opt_leaf(&r.low_projection, encode_low_projection));
    put!("low_projection_status", low_projection_status);
    put!("drift_window_branch", opt_leaf(&r.drift_window, encode_drift_window));
    put!("drift_window_status", drift_window_status);
    put!("selected_ground_force_producer", match r.selected_ground_force_producer {
            Some(name) => json!(name),
            None => Value::Null,
        });
    put!("ground_accumulator", opt_leaf(&r.accumulator, encode_sink));
    put!("ground_accumulator_status", r.ground_accumulator_status);
    put!("rail_accumulator", opt_leaf(&r.rail, encode_rail_producers));
    put!("rail_selector", opt_leaf(&r.rail_selector, encode_rail_selector));
    put!("rail_accumulator_status", r.rail_accumulator_status);
    put!("contact_accumulator", opt_leaf(&r.contact, encode_contact));
    put!("contact_accumulator_status", r.contact_accumulator_status);
    put!("hard_contact_velocity", encode_hard_outcome(r));
    put!("hard_contact_velocity_status", if r.hard_contact.is_some() {
            "evaluated_replacement"
        } else {
            "not_selected"
        });
    put!("hard_contact_selection_authority", opt_str(&r.hard_contact_selection_authority));
    put!("hard_contact_scheduler_phase", opt_str(&r.hard_contact_scheduler_phase));
    put!("soft_contact_velocity", encode_soft_outcome(&r.soft_contact));
    put!("soft_contact_velocity_status", if r.soft_contact.evaluated_velocity().is_some() {
            "evaluated_add"
        } else {
            "not_selected"
        });
    put!("soft_contact_scheduler_phase", opt_str(&r.soft_contact_scheduler_phase));
    put!("ground_correction", opt_leaf(&r.correction, encode_pipeline));
    put!("ground_correction_status", r.ground_correction_status);
    put!("bs_zone", encode_bs_outcome(r));
    put!("bs_zone_status", r.bs_zone_status);
    put!("mz_magnet", encode_mz_outcome(r));
    put!("mz_magnet_status", r.mz_magnet_status);
    put!("jm_dj_trigger", encode_jm_dj_outcome(r));
    put!("jm_dj_trigger_status", r.jm_dj_trigger_status);
    put!("accumulator_614_after_bs", opt_arr(&r.accumulator_614_after_bs));
    put!("accumulator_620_after_bs", opt_arr(&r.accumulator_620_after_bs));
    put!("accumulator_614_after_post_bs_reset", opt_arr(&r.accumulator_614_after_post_bs_reset));
    put!("accumulator_620_after_post_bs_reset", opt_arr(&r.accumulator_620_after_post_bs_reset));
    put!("common_damping", opt_leaf(&r.damping, encode_damping));
    put!("common_damping_status", r.common_damping_status);
    put!("accumulator_614_after_damping", opt_arr(&r.accumulator_614_after_damping));
    put!("accumulator_620_after_damping", opt_arr(&r.accumulator_620_after_damping));
    put!("post_bs_reset_constraint", post_bs_constraint.clone());
    put!("post_bs_reset_selector", post_bs_selector.clone());
    put!("post_bs_reset_constraint_status", r.post_bs_reset_constraint_status);
    put!("no_contact_fallback_constraint", post_bs_constraint);
    put!("no_contact_fallback_selector", post_bs_selector);
    put!("no_contact_fallback_constraint_status", r.post_bs_reset_constraint_status);
    put!("speed_threshold_latch", opt_leaf(&r.speed_threshold, encode_speed_threshold));
    put!("speed_threshold_latch_status", if r.speed_threshold.is_some() {
            "evaluated"
        } else {
            "parameters_missing"
        });
    put!("linear_integrator", opt_leaf(&r.linear, encode_linear));
    put!("linear_integrator_status", r.linear_integrator_status);
    put!("angular_integrator", opt_leaf(&r.angular, encode_angular));
    put!("angular_integrator_status", r.angular_integrator_status);
    put!("position_integrator", opt_leaf(&r.position, encode_position));
    put!("position_integrator_status", r.position_integrator_status);
    put!("orientation_integrator", opt_leaf(&r.orientation, encode_orientation));
    put!("orientation_integrator_status", r.orientation_integrator_status);
    put!("velocity_c0_after_pose_step", opt_arr(&r.velocity_c0_after_pose_step));
    put!("angular_velocity_cc_after_pose_step", opt_arr(&r.angular_velocity_cc_after_pose_step));
    put!("terminal_fall_reset", opt_leaf(&r.fall_reset, encode_fall_clamp));
    put!("terminal_fall_reset_status", r.terminal_fall_reset_status);
    put!("terminal_checkpoint_respawn", opt_leaf(&r.checkpoint_respawn, encode_checkpoint_respawn));
    put!("terminal_checkpoint_respawn_status", r.terminal_checkpoint_respawn_status);
    put!("checkpoint_target_selector_events", Value::Array(
            r.checkpoint_target_events
                .iter()
                .map(encode_checkpoint_target)
                .collect(),
        ));
    put!("checkpoint_target_selector_final", opt_leaf(
            &r.checkpoint_target_final,
            encode_checkpoint_target,
        ));
    put!("position_90_final", opt_arr(&r.position_90_final));
    put!("matrix_9c_final", opt_arr(&r.matrix_9c_final));
    put!("velocity_c0_final", opt_arr(&r.velocity_c0_final));
    put!("angular_velocity_cc_final", opt_arr(&r.angular_velocity_cc_final));
    put!("ordered_events", branch_aware_vehicle_step::ORDERED_EVENTS);
    put!("composition_policy", branch_aware_vehicle_step::COMPOSITION_POLICY);
    put!("static_sources", encode_static_sources());
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Value = serde_json::from_str(&input).expect("parse spec JSON");
    let cases = spec["cases"].as_array().expect("spec.cases must be an array");
    let rows: Vec<Value> = cases
        .iter()
        .map(|case| {
            let args: BranchAwareStepArgs = serde_json::from_value(case["kwargs"].clone())
                .unwrap_or_else(|error| panic!("decode kwargs: {error}"));
            match evaluate_branch_aware_step(&args) {
                Ok(result) => encode_result(&result),
                Err(message) => json!({"error": message}),
            }
        })
        .collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
