//! Ground force-producer parity trace tool.
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"fn": "<module>/<function>", "args": {...}}, ...]}`
//! and prints a JSON array of result rows, one per case. Every f64 is emitted
//! as its IEEE-754 bit pattern (`format!("{:016x}", v.to_bits())`, matching
//! the Python side's `struct.pack('<d')`); bools/ints/strings/None pass
//! through. A domain `Err` is emitted as `{"error": <message>}` so the
//! Python side can diff `ValueError` texts bit-for-bit.
//!
//! Covers the GND force-producer leaf modules:
//! ground_drive_force_body, ground_state_scalar, low_projection_ground_branch,
//! drift_window_branch, drift_recovery_timer, common_damping,
//! ground_accumulator_sink, substep_accumulator_boundary,
//! speed_threshold_latch, post_bs_reset_constraint,
//! contact_accumulator_producer, drive_target_state.

use std::io::Read;

use kart_physics::domains::common_damping::evaluate_common_damping;
use kart_physics::domains::contact_accumulator_producer::evaluate_contact_accumulator_producer;
use kart_physics::domains::drift_recovery_timer::{
    apply_selector_transition, evaluate_recovery_substep, evaluate_surface1_recovery_force,
    seed_recovery_timer, SelectorState,
};
use kart_physics::domains::drift_window_branch::{evaluate_drift_window_branch, DriftWindowBranchResult};
use kart_physics::domains::drive_target_state::{
    evaluate_drive_target_runtime, evaluate_drive_target_state,
};
use kart_physics::domains::ground_accumulator_sink::{
    apply_ground_accumulator_sink, transform_row_major,
};
use kart_physics::domains::ground_drive_force_body::evaluate_ground_drive_force_body;
use kart_physics::domains::ground_state_scalar::{
    evaluate_direction_gain, evaluate_fallback_opposite_add, evaluate_ground_correction_pipeline,
    evaluate_lateral_subtract, evaluate_negative_adc_addon, evaluate_primary_force_delta,
    evaluate_state_mult, evaluate_velocity_zero_action, select_fallback_lateral_actions,
    FallbackLateralActions, PipelineEvent, TraceStep,
};
use kart_physics::domains::low_projection_ground_branch::{
    evaluate_low_projection_branch, LowProjectionBranchResult,
};
use kart_physics::domains::post_bs_reset_constraint::{
    evaluate_post_bs_reset_constraint, select_post_bs_reset_constraint,
};
use kart_physics::domains::speed_threshold_latch::evaluate_speed_threshold_latch;
use kart_physics::domains::substep_accumulator_boundary::evaluate_substep_accumulator_boundary;
use kart_physics::domains::surface_tune_defaults::{
    GroundSurfaceProfile, NonSurfaceProfile, RecoverySurfaceProfile,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Encoding helpers
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

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------

fn get_f64(args: &Value, key: &str) -> f64 {
    args[key].as_f64().unwrap_or_else(|| panic!("{key} must be a float"))
}

fn get_opt_f64(args: &Value, key: &str) -> Option<f64> {
    if args[key].is_null() {
        None
    } else {
        Some(get_f64(args, key))
    }
}

fn get_i64(args: &Value, key: &str) -> i64 {
    args[key].as_i64().unwrap_or_else(|| panic!("{key} must be an int"))
}

fn get_bool(args: &Value, key: &str) -> bool {
    args[key].as_bool().unwrap_or_else(|| panic!("{key} must be a bool"))
}

fn get_arr<const N: usize>(args: &Value, key: &str) -> [f64; N] {
    let items = args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"));
    let mut out = [0.0_f64; N];
    assert_eq!(items.len(), N, "{key} must have {N} components");
    for (index, item) in items.iter().enumerate() {
        out[index] = item.as_f64().expect("array components must be floats");
    }
    out
}

fn get_opt_arr<const N: usize>(args: &Value, key: &str) -> Option<[f64; N]> {
    if args[key].is_null() {
        None
    } else {
        Some(get_arr(args, key))
    }
}

fn get_bool4(args: &Value, key: &str) -> [bool; 4] {
    let items = args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"));
    assert_eq!(items.len(), 4, "{key} must have 4 entries");
    let mut out = [false; 4];
    for (index, item) in items.iter().enumerate() {
        out[index] = item.as_bool().expect("entries must be bools");
    }
    out
}

fn get_vec3_4(args: &Value, key: &str) -> [[f64; 3]; 4] {
    let items = args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"));
    assert_eq!(items.len(), 4, "{key} must have 4 entries");
    let mut out = [[0.0_f64; 3]; 4];
    for (index, item) in items.iter().enumerate() {
        let comps = item.as_array().expect("entries must be arrays");
        assert_eq!(comps.len(), 3, "{key} entries must have 3 components");
        for (c, value) in comps.iter().enumerate() {
            out[index][c] = value.as_f64().expect("components must be floats");
        }
    }
    out
}

fn get_ground_profile(args: &Value, key: &str) -> Option<GroundSurfaceProfile> {
    if args[key].is_null() {
        return None;
    }
    let p = &args[key];
    Some(GroundSurfaceProfile {
        speed_denominator: get_f64(p, "speed_denominator"),
        speed_quadratic_gain: get_f64(p, "speed_quadratic_gain"),
        speed_bias: get_f64(p, "speed_bias"),
        force_scale: get_f64(p, "force_scale"),
    })
}

fn get_recovery_profile(args: &Value, key: &str) -> Option<RecoverySurfaceProfile> {
    if args[key].is_null() {
        return None;
    }
    let p = &args[key];
    Some(RecoverySurfaceProfile {
        longitudinal_gain_offset: get_f64(p, "longitudinal_gain_offset"),
        lateral_gain_offset: get_f64(p, "lateral_gain_offset"),
        force_scale: get_f64(p, "force_scale"),
        forward_target_gain: get_f64(p, "forward_target_gain"),
    })
}

fn get_non_surface_profile(args: &Value, key: &str) -> Option<NonSurfaceProfile> {
    if args[key].is_null() {
        return None;
    }
    let p = &args[key];
    Some(NonSurfaceProfile {
        longitudinal_gain_offset: get_f64(p, "longitudinal_gain_offset"),
        lateral_gain_offset: get_f64(p, "lateral_gain_offset"),
        force_scale: get_f64(p, "force_scale"),
    })
}

// ---------------------------------------------------------------------------
// Per-function tracers
// ---------------------------------------------------------------------------

fn trace_ground_drive_force_body(args: &Value) -> Value {
    match evaluate_ground_drive_force_body(
        get_i64(args, "surface_mode"),
        get_f64(args, "speed_magnitude"),
        get_f64(args, "ground_forward_term"),
        get_f64(args, "contact_basis_product"),
        get_f64(args, "drive_direction"),
        get_f64(args, "normalized_velocity"),
        get_f64(args, "longitudinal_correction"),
        get_f64(args, "lateral_correction"),
        get_f64(args, "mass_scale"),
        get_f64(args, "base_longitudinal_gain"),
        get_f64(args, "base_lateral_gain"),
        get_f64(args, "ground_force_multiplier"),
        get_f64(args, "settlement_multiplier"),
        get_f64(args, "tune_75c"),
        get_f64(args, "axis_long"),
        get_f64(args, "axis_lat"),
        get_ground_profile(args, "surface_profile"),
        get_f64(args, "torque_y_scale"),
        get_opt_f64(args, "torque_y_gain"),
        get_f64(args, "state_5c8"),
        get_bool(args, "b1c_charge_settlement"),
        get_f64(args, "b10"),
        get_f64(args, "b20"),
        get_f64(args, "b2c"),
        get_f64(args, "b38"),
        get_f64(args, "cap_760"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_primary_force_delta(args: &Value) -> Value {
    let r = evaluate_primary_force_delta(
        get_arr(args, "correction_dir"),
        get_f64(args, "direction_gain"),
        get_f64(args, "selected_slot"),
        get_f64(args, "state_mult"),
    );
    json!({
        "force_delta": arr(&r.force_delta),
        "scalar": fbits(r.scalar),
        "scope": r.scope,
    })
}

fn trace_negative_adc_addon(args: &Value) -> Value {
    let r = evaluate_negative_adc_addon(
        get_f64(args, "field_adc"),
        get_bool(args, "flag_af4"),
        get_bool(args, "flag_afc"),
        get_arr(args, "sub_ad0"),
        get_f64(args, "sub_af0"),
        get_f64(args, "field_9f0"),
        get_f64(args, "cap"),
    );
    json!({
        "branch": r.branch,
        "force_delta": arr(&r.force_delta),
        "scalar": fbits(r.scalar),
        "evidence": r.evidence,
    })
}

fn trace_fallback_opposite_add(args: &Value) -> Value {
    let r = evaluate_fallback_opposite_add(
        get_arr(args, "sub_ad0"),
        get_f64(args, "sub_6f4"),
        get_f64(args, "sub_bc4"),
        get_f64(args, "drive_slot1_a90"),
    );
    json!({
        "branch": r.branch,
        "force_delta": arr(&r.force_delta),
        "scalar": fbits(r.scalar),
        "evidence": r.evidence,
    })
}

fn trace_lateral_subtract(args: &Value) -> Value {
    match evaluate_lateral_subtract(
        get_i64(args, "kind"),
        get_arr(args, "velocity_c0"),
        get_arr(args, "axis_ac4"),
        get_f64(args, "sub_720"),
        get_f64(args, "sub_724"),
    ) {
        Ok(r) => json!({
            "branch": r.branch,
            "source": arr(&r.source),
            "accumulator_delta": arr(&r.accumulator_delta),
            "lateral_unit_residual": arr(&r.lateral_unit_residual),
            "evidence": r.evidence,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_velocity_zero_action(args: &Value) -> Value {
    match evaluate_velocity_zero_action(get_i64(args, "kind"), get_arr(args, "velocity_c0")) {
        Ok(r) => json!({
            "branch": r.branch,
            "pre_velocity": arr(&r.pre_velocity),
            "post_velocity": arr(&r.post_velocity),
            "terminal_for_current_correction_path": r.terminal_for_current_correction_path,
            "evidence": r.evidence,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn encode_selector(s: &FallbackLateralActions) -> Value {
    json!({
        "actions_in_program_order": s.actions_in_program_order,
        "timer_a80_after": fbits(s.timer_a80_after),
        "primary_path_selected": s.primary_path_selected,
        "fallback_path_selected": s.fallback_path_selected,
        "fallback_initial": s.fallback_initial,
        "fallback_before_lateral": s.fallback_before_lateral,
        "skip_lateral": s.skip_lateral,
        "evidence": s.evidence,
    })
}

fn trace_select_fallback_lateral(args: &Value) -> Value {
    let r = select_fallback_lateral_actions(
        get_i64(args, "selected_slot"),
        get_i64(args, "opposite_slot"),
        get_bool(args, "gate_1c0"),
        get_bool(args, "gate_6f3b00"),
        get_bool(args, "gate_6f3ae0"),
        get_f64(args, "field_adc"),
        get_f64(args, "timer_a80_before"),
        get_f64(args, "dt"),
        get_f64(args, "field_ae0"),
        get_f64(args, "b8c"),
        get_f64(args, "velocity_length_sq"),
        get_f64(args, "lateral_dot"),
    );
    encode_selector(&r)
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

fn trace_direction_gain(args: &Value) -> Value {
    let r = evaluate_direction_gain(
        get_f64(args, "base_gain"),
        get_i64(args, "decoded_state"),
        get_bool(args, "byte_62c"),
        get_bool(args, "byte_9b8"),
        get_bool(args, "flag_afc"),
        get_f64(args, "obf_838"),
        get_f64(args, "obf_844"),
        get_f64(args, "obf_74c"),
    );
    json!({
        "direction_gain": fbits(r.direction_gain),
        "trace": encode_trace(&r.trace),
    })
}

fn trace_state_mult(args: &Value) -> Value {
    let r = evaluate_state_mult(
        get_i64(args, "decoded_state"),
        get_bool(args, "boost_family"),
        get_i64(args, "attached_state"),
        get_bool(args, "e8c"),
        get_bool(args, "efc"),
        get_bool(args, "flag_5bc"),
        get_f64(args, "raw_144"),
        get_f64(args, "obf_808"),
        get_f64(args, "obf_814"),
        get_f64(args, "obf_69c"),
        get_f64(args, "obf_684"),
        get_f64(args, "obf_8ac"),
        get_f64(args, "obf_8e8"),
        get_f64(args, "obf_94c"),
    );
    json!({
        "state_mult": fbits(r.state_mult),
        "trace": encode_trace(&r.trace),
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

fn trace_ground_correction_pipeline(args: &Value) -> Value {
    let r = evaluate_ground_correction_pipeline(
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        get_arr(args, "velocity_c0"),
        get_bool(args, "disabled_b9c"),
        get_i64(args, "selected_slot"),
        get_i64(args, "opposite_slot"),
        get_bool(args, "gate_1c0"),
        get_bool(args, "gate_6f3b00"),
        get_bool(args, "gate_6f3ae0"),
        get_f64(args, "field_adc"),
        get_f64(args, "timer_a80_before"),
        get_f64(args, "dt"),
        get_f64(args, "field_ae0"),
        get_f64(args, "b8c"),
        get_f64(args, "lateral_dot"),
        get_arr(args, "correction_dir"),
        get_f64(args, "direction_gain"),
        get_f64(args, "state_mult"),
        get_bool(args, "flag_af4"),
        get_bool(args, "flag_afc"),
        get_arr(args, "sub_ad0"),
        get_f64(args, "sub_af0"),
        get_f64(args, "field_9f0"),
        get_f64(args, "sub_6f4"),
        get_f64(args, "sub_bc4"),
        get_f64(args, "drive_slot1_a90"),
        get_arr(args, "axis_ac4"),
        get_f64(args, "sub_720"),
        get_f64(args, "sub_724"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    );
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

fn trace_low_projection_branch(args: &Value) -> Value {
    match evaluate_low_projection_branch(
        get_f64(args, "projection_len"),
        get_f64(args, "ground_forward_term"),
        get_f64(args, "drive_direction"),
        get_f64(args, "normalized_velocity_raw"),
        get_f64(args, "correction_raw"),
        get_f64(args, "mass_scale"),
        get_f64(args, "base_longitudinal_gain"),
        get_f64(args, "base_lateral_gain"),
        get_f64(args, "axis_long"),
        get_f64(args, "axis_lat"),
        get_f64(args, "state_5c8"),
        get_bool(args, "b1c_charge_settlement"),
        get_f64(args, "b10"),
        get_f64(args, "b20"),
        get_f64(args, "b2c"),
        get_f64(args, "b38"),
        get_f64(args, "cap_760"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(LowProjectionBranchResult::NotSelected { projection_len }) => json!({
            "selected": false,
            "reason": "projection_above_threshold",
            "projection_len": fbits(projection_len),
        }),
        Ok(LowProjectionBranchResult::Selected(r)) => json!({
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
        Err(message) => json!({"error": message}),
    }
}

fn trace_drift_window_branch(args: &Value) -> Value {
    match evaluate_drift_window_branch(
        get_bool(args, "selector_b08"),
        get_f64(args, "timer_b0c"),
        get_f64(args, "dt"),
        get_f64(args, "configured_duration_734"),
        get_f64(args, "mass_scale"),
        get_f64(args, "base_longitudinal_gain"),
        get_f64(args, "signed_axis"),
        get_f64(args, "scaled_tune_660"),
        get_f64(args, "drift_gain_728"),
        get_f64(args, "tune_758"),
        get_f64(args, "axis_lat"),
        get_f64(args, "drive_raw_aa0"),
        get_f64(args, "debug_force_scale"),
    ) {
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

fn trace_surface1_recovery_force(args: &Value) -> Value {
    match evaluate_surface1_recovery_force(
        get_bool(args, "selector_af4"),
        get_f64(args, "speed_magnitude"),
        get_f64(args, "speed_profile_denominator"),
        get_f64(args, "blend_ab0"),
        get_f64(args, "blend_ab4"),
        get_f64(args, "active_ab0_rate"),
        get_f64(args, "active_ab4_rate"),
        get_f64(args, "timer_ab0_rate"),
        get_f64(args, "contact_basis_product"),
        get_f64(args, "ground_forward_term"),
        get_f64(args, "drive_direction"),
        get_f64(args, "normalized_velocity"),
        get_f64(args, "longitudinal_correction"),
        get_f64(args, "lateral_correction"),
        get_f64(args, "mass_scale"),
        get_f64(args, "base_longitudinal_gain"),
        get_f64(args, "base_lateral_gain"),
        get_f64(args, "ground_force_multiplier"),
        get_f64(args, "tune_758"),
        get_f64(args, "axis_long"),
        get_f64(args, "axis_lat"),
        get_recovery_profile(args, "active_surface"),
        get_recovery_profile(args, "timer_surface"),
        get_f64(args, "field_af0"),
        get_f64(args, "torque_y_scale"),
        get_i64(args, "surface_mode"),
        get_non_surface_profile(args, "non_surface_profile"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_seed_recovery_timer(args: &Value) -> Value {
    json!({"value": fbits(seed_recovery_timer(get_f64(args, "field_af0")))})
}

fn encode_selector_state(state: &SelectorState) -> Value {
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

fn trace_apply_selector_transition(args: &Value) -> Value {
    let r = apply_selector_transition(
        get_bool(args, "enable"),
        get_i64(args, "decoded_e8c"),
        get_i64(args, "surface_mode"),
        get_bool(args, "selector_af4"),
        get_bool(args, "selector_b08"),
        get_bool(args, "selector_b5c"),
        get_bool(args, "selector_f08"),
        get_i64(args, "selector_fec"),
        get_i64(args, "selector_ff8"),
        get_f64(args, "field_af8"),
        get_f64(args, "field_adc"),
        get_bool(args, "contact_188"),
        get_f64(args, "drive_raw_aa0"),
        get_f64(args, "field_af0"),
        get_f64(args, "recovery_timer"),
    );
    json!({
        "applied": r.applied,
        "reason": r.reason,
        "before": encode_selector_state(&r.before),
        "after": encode_selector_state(&r.after),
    })
}

fn trace_recovery_substep(args: &Value) -> Value {
    let r = evaluate_recovery_substep(
        get_i64(args, "surface_mode"),
        get_bool(args, "selector_af4"),
        get_f64(args, "field_af8"),
        get_f64(args, "recovery_timer"),
        get_bool(args, "other_recovery_gate"),
        get_f64(args, "dt"),
    );
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

fn trace_common_damping(args: &Value) -> Value {
    match evaluate_common_damping(
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        get_arr(args, "velocity_c0"),
        get_arr(args, "angular_velocity_cc"),
        get_f64(args, "damping_700"),
        get_bool(args, "special_gate"),
        get_f64(args, "directional_70c"),
        get_f64(args, "runtime_ba0"),
        get_opt_f64(args, "token_multiplier"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_transform_row_major(args: &Value) -> Value {
    match transform_row_major(
        &get_arr(args, "matrix_9c"),
        &get_arr(args, "vector"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({"value": arr(&r)}),
        Err(message) => json!({"error": message}),
    }
}

fn trace_apply_ground_accumulator_sink(args: &Value) -> Value {
    match apply_ground_accumulator_sink(
        get_arr(args, "local_force"),
        get_arr(args, "local_torque"),
        get_arr(args, "matrix_9c"),
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({
            "local_force": arr(&r.local_force),
            "world_force": arr(&r.world_force),
            "local_torque": arr(&r.local_torque),
            "accumulator_614_before": arr(&r.accumulator_614_before),
            "accumulator_614_after": arr(&r.accumulator_614_after),
            "accumulator_620_before": arr(&r.accumulator_620_before),
            "accumulator_620_after": arr(&r.accumulator_620_after),
            "numeric_mode": r.numeric_mode,
            "ordering": r.ordering,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_substep_accumulator_boundary(args: &Value) -> Value {
    let r = evaluate_substep_accumulator_boundary(
        get_arr(args, "pending_force_b74"),
        get_arr(args, "pending_torque_b80"),
        get_arr(args, "pending_b90"),
        get_opt_arr(args, "active_force_614_before"),
        get_opt_arr(args, "active_torque_620_before"),
    );
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

fn trace_speed_threshold_latch(args: &Value) -> Value {
    let r = evaluate_speed_threshold_latch(get_f64(args, "threshold_b8c"), get_arr(args, "velocity_c0"));
    json!({
        "threshold_b8c_before": fbits(r.threshold_b8c_before),
        "speed_sq": fbits(r.speed_sq),
        "threshold_sq": fbits(r.threshold_sq),
        "clears": r.clears,
        "threshold_b8c_after": fbits(r.threshold_b8c_after),
        "ordering": r.ordering,
    })
}

fn trace_select_post_bs_reset(args: &Value) -> Value {
    let r = select_post_bs_reset_constraint(
        get_bool(args, "current_contact_updated"),
        get_bool(args, "contact_trigger_e38"),
        get_bool(args, "contact_trigger_e4c"),
        get_i64(args, "rail_state_1cc"),
    );
    json!({
        "selected": r.selected,
        "contact_or_trigger": r.contact_or_trigger,
        "rail_active": r.rail_active,
        "ordering": r.ordering,
    })
}

fn trace_evaluate_post_bs_reset(args: &Value) -> Value {
    match evaluate_post_bs_reset_constraint(
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        get_arr(args, "vector_604"),
        get_f64(args, "sub_660"),
        get_f64(args, "divisor_e50"),
        get_arr(args, "angular_velocity_cc"),
        get_bool(args, "special_orientation_gate"),
        get_f64(args, "field_acc"),
        get_f64(args, "field_ac0"),
        get_f64(args, "field_ad8"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_contact_accumulator_producer(args: &Value) -> Value {
    match evaluate_contact_accumulator_producer(
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        get_f64(args, "elapsed_seconds"),
        get_bool4(args, "contact_active"),
        get_arr(args, "compression_a54"),
        get_arr(args, "compression_rate_a64"),
        get_vec3_4(args, "contact_vectors_a14"),
        get_arr(args, "contact_normal_ac4"),
        get_f64(args, "spring_7c0"),
        get_f64(args, "rebound_7c4"),
        get_f64(args, "bump_7c8"),
        get_arr(args, "lever_x_9f4"),
        get_arr(args, "lever_y_9f8"),
        get_f64(args, "obf_66c"),
        get_f64(args, "obf_678"),
        get_arr(args, "field_604"),
        get_f64(args, "obf_660"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn encode_drive_target_state(r: &kart_physics::domains::drive_target_state::DriveTargetStateResult) -> Value {
    json!({
        "raw_target": fbits(r.raw_target),
        "exponent_input": fbits(r.exponent_input),
        "attenuation": fbits(r.attenuation),
        "candidate_aa4": fbits(r.candidate_aa4),
        "field_aa4_after": fbits(r.field_aa4_after),
        "field_aa8_after": fbits(r.field_aa8_after),
        "overshoot_clamped_to_aa8": r.overshoot_clamped_to_aa8,
        "ordering": r.ordering,
    })
}

fn trace_drive_target_state(args: &Value) -> Value {
    match evaluate_drive_target_state(
        get_f64(args, "signed_axis"),
        get_f64(args, "scaled_tune_660"),
        get_f64(args, "field_adc"),
        get_f64(args, "field_71c"),
        get_f64(args, "decoded_bb8"),
        get_f64(args, "selected_slot"),
        get_f64(args, "field_aa8_before"),
    ) {
        Ok(r) => encode_drive_target_state(&r),
        Err(message) => json!({"error": message}),
    }
}

fn trace_drive_target_runtime(args: &Value) -> Value {
    match evaluate_drive_target_runtime(
        get_f64(args, "signed_axis"),
        get_f64(args, "field_718"),
        get_f64(args, "field_adc"),
        get_f64(args, "field_71c"),
        get_f64(args, "decoded_bb8"),
        get_f64(args, "selected_slot"),
        get_f64(args, "field_aa8_before"),
    ) {
        Ok(r) => {
            let mut row = encode_drive_target_state(&r.state);
            let map = row.as_object_mut().expect("state row is an object");
            map.insert("field_718".to_string(), fbits(r.field_718));
            map.insert("scaled_tune_660".to_string(), fbits(r.scaled_tune_660));
            map.insert("tune_accessor".to_string(), json!(r.tune_accessor));
            row
        }
        Err(message) => json!({"error": message}),
    }
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
            let args = &case["args"];
            match case["fn"].as_str().expect("case.fn must be a string") {
                "ground_drive_force_body/evaluate_ground_drive_force_body" => {
                    trace_ground_drive_force_body(args)
                }
                "ground_state_scalar/evaluate_primary_force_delta" => trace_primary_force_delta(args),
                "ground_state_scalar/evaluate_negative_adc_addon" => trace_negative_adc_addon(args),
                "ground_state_scalar/evaluate_fallback_opposite_add" => {
                    trace_fallback_opposite_add(args)
                }
                "ground_state_scalar/evaluate_lateral_subtract" => trace_lateral_subtract(args),
                "ground_state_scalar/evaluate_velocity_zero_action" => {
                    trace_velocity_zero_action(args)
                }
                "ground_state_scalar/select_fallback_lateral_actions" => {
                    trace_select_fallback_lateral(args)
                }
                "ground_state_scalar/evaluate_direction_gain" => trace_direction_gain(args),
                "ground_state_scalar/evaluate_state_mult" => trace_state_mult(args),
                "ground_state_scalar/evaluate_ground_correction_pipeline" => {
                    trace_ground_correction_pipeline(args)
                }
                "low_projection_ground_branch/evaluate_low_projection_branch" => {
                    trace_low_projection_branch(args)
                }
                "drift_window_branch/evaluate_drift_window_branch" => {
                    trace_drift_window_branch(args)
                }
                "drift_recovery_timer/evaluate_surface1_recovery_force" => {
                    trace_surface1_recovery_force(args)
                }
                "drift_recovery_timer/seed_recovery_timer" => trace_seed_recovery_timer(args),
                "drift_recovery_timer/apply_selector_transition" => {
                    trace_apply_selector_transition(args)
                }
                "drift_recovery_timer/evaluate_recovery_substep" => trace_recovery_substep(args),
                "common_damping/evaluate_common_damping" => trace_common_damping(args),
                "ground_accumulator_sink/transform_row_major" => trace_transform_row_major(args),
                "ground_accumulator_sink/apply_ground_accumulator_sink" => {
                    trace_apply_ground_accumulator_sink(args)
                }
                "substep_accumulator_boundary/evaluate_substep_accumulator_boundary" => {
                    trace_substep_accumulator_boundary(args)
                }
                "speed_threshold_latch/evaluate_speed_threshold_latch" => {
                    trace_speed_threshold_latch(args)
                }
                "post_bs_reset_constraint/select_post_bs_reset_constraint" => {
                    trace_select_post_bs_reset(args)
                }
                "post_bs_reset_constraint/evaluate_post_bs_reset_constraint" => {
                    trace_evaluate_post_bs_reset(args)
                }
                "contact_accumulator_producer/evaluate_contact_accumulator_producer" => {
                    trace_contact_accumulator_producer(args)
                }
                "drive_target_state/evaluate_drive_target_state" => trace_drive_target_state(args),
                "drive_target_state/evaluate_drive_target_runtime" => {
                    trace_drive_target_runtime(args)
                }
                other => panic!("unknown case fn {other:?}"),
            }
        })
        .collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
