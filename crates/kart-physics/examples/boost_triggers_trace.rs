//! Boost / collision / checkpoint / trigger parity trace tool.
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"fn": "<module>/<function>", "args": {...}}, ...]}`
//! and prints a JSON array of result rows, one per case. Every f64 is emitted
//! as its IEEE-754 bit pattern (`format!("{:016x}", v.to_bits())`, matching
//! the Python side's `struct.pack('<d')`); bools/ints/strings/None pass
//! through. A domain `Err` is emitted as `{"error": <message>}` so the
//! Python side can diff `ValueError` texts bit-for-bit.
//!
//! Covers the boost/trigger leaf modules: booster_resource,
//! booster_vehicle_step, bs_zone_override, hard_contact_velocity_response,
//! soft_contact_velocity_response, rail_accumulator_producers,
//! fall_clamp_state_replacement, checkpoint_respawn_state_replacement,
//! checkpoint_target_selector, state18_feedback, pit_resource_step,
//! tune_boost_gauge, mz_magnet_trigger, jm_dj_trigger.

use std::io::Read;

use kart_physics::domains::booster_resource::{
    activation_for_item, add_inventory_item, advance_state_timer, consume_inventory_item,
    tune_time_ms, InventorySlot,
};
use kart_physics::domains::booster_vehicle_step::{
    activate_from_inventory, evaluate_booster_inventory_use, evaluate_vehicle_step,
    ActivateFromInventoryResult, InventoryUseInput, InventoryUseResult,
};
use kart_physics::domains::bs_zone_override::{
    evaluate_bs_zone_override, BsZoneOverrideResult,
};
use kart_physics::domains::checkpoint_respawn_state_replacement::{
    self, evaluate_checkpoint_respawn_state_replacement, CheckpointRespawnStateReplacement,
};
use kart_physics::domains::checkpoint_target_selector::{
    self, evaluate_checkpoint_target_selector, CheckpointTargetSelectorResult,
};
use kart_physics::domains::fall_clamp_state_replacement::{
    self, evaluate_fall_clamp_state_replacement, FallClampStateReplacement,
};
use kart_physics::domains::event_obstacle_contact::apply_event_obstacle_contact_source;
use kart_physics::domains::hard_contact_velocity_response::{
    self, evaluate_hard_contact_velocity_response, HardContactVelocityResponse,
};
use kart_physics::domains::jm_dj_trigger::{
    evaluate_jm_dj_trigger, parse_jm_dj_surface_token, JmDjSurfaceToken, JmDjTrigger,
};
use kart_physics::domains::mz_magnet_trigger::{
    self, evaluate_mz_magnet_trigger, MzMagnetTrigger,
};
use kart_physics::domains::pit_resource_step::{
    drain_pit_accumulator, evaluate_pit_resource_step, PitAccumulatorDrain, PitResourceStep,
};
use kart_physics::domains::rail_accumulator_producers::{
    evaluate_rail_accumulator_producers, evaluate_rail_orientation_steering_torque,
    evaluate_rail_state2_alignment_torque, select_rail_accumulator_kinds,
    RailAccumulatorKinds, RailAccumulatorProducersResult, RailOrientationSteeringTorque,
    RailState2AlignmentTorque, RailTorqueEvent, RailWriterEvent,
};
use kart_physics::domains::soft_contact_velocity_response::{
    self, evaluate_soft_contact_velocity_response, SoftContactVelocityResponse,
};
use kart_physics::domains::state18_feedback::{
    representative_stroke_dispatch, select_state18_feedback, timer_expiry_next_state,
};
use kart_physics::domains::tune_boost_gauge::{
    self, accumulate_pending, commit_pending, evaluate_bcharge_surface_step, gauge_accum_clamped,
    gauge_cap, gauge_ratio, mq_display_target, pending_accum_delta, state34_drift_gauge_event,
    state34_drift_gauge_window_can_activate, state34_drift_gauge_window_expired,
    BchargeSurfaceStep,
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

fn slots_json(slots: &[InventorySlot]) -> Value {
    Value::Array(
        slots
            .iter()
            .map(|slot| json!({"item_id": slot.item_id, "quantity": slot.quantity}))
            .collect(),
    )
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

fn get_opt_i64(args: &Value, key: &str) -> Option<i64> {
    if args[key].is_null() {
        None
    } else {
        Some(get_i64(args, key))
    }
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

fn get_slots(args: &Value, key: &str) -> Vec<InventorySlot> {
    let items = args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"));
    items
        .iter()
        .map(|item| InventorySlot {
            item_id: item["item_id"].as_i64().expect("slot.item_id must be an int"),
            quantity: item["quantity"].as_i64().expect("slot.quantity must be an int"),
        })
        .collect()
}

fn get_i64_list(args: &Value, key: &str) -> Vec<i64> {
    args[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} must be an array"))
        .iter()
        .map(|item| item.as_i64().expect("entries must be ints"))
        .collect()
}

// ---------------------------------------------------------------------------
// Per-function tracers
// ---------------------------------------------------------------------------

fn trace_tune_time_ms(args: &Value) -> Value {
    json!({"value": tune_time_ms(get_f64(args, "value"))})
}

fn trace_activation_for_item(args: &Value) -> Value {
    let value = activation_for_item(
        get_i64(args, "item_id"),
        get_f64(args, "normal_booster_time"),
        get_f64(args, "item_booster_time"),
        get_f64(args, "animal_booster_time"),
    );
    match value {
        Some((state, duration_ms)) => json!({"value": [state, duration_ms]}),
        None => json!({"value": Value::Null}),
    }
}

fn trace_advance_state_timer(args: &Value) -> Value {
    match advance_state_timer(
        get_i64(args, "state"),
        get_i64(args, "remaining_ms"),
        get_i64(args, "elapsed_ms"),
    ) {
        Ok(step) => json!({
            "state": step.state,
            "remaining_ms": step.remaining_ms,
            "frame_ms": step.frame_ms,
            "consumed_ms": step.consumed_ms,
            "clears_on_tick": step.clears_on_tick,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_add_inventory_item(args: &Value) -> Value {
    let (added, slots_after) = add_inventory_item(
        &get_slots(args, "slots"),
        get_i64(args, "item_id"),
        get_i64(args, "quantity"),
        get_i64(args, "capacity"),
    );
    json!({"added": added, "slots_after": slots_json(&slots_after)})
}

fn trace_consume_inventory_item(args: &Value) -> Value {
    let (item_id, slots_after) =
        consume_inventory_item(&get_slots(args, "slots"), get_i64(args, "index"));
    json!({"item_id": item_id, "slots_after": slots_json(&slots_after)})
}

fn encode_activate_from_inventory(r: &ActivateFromInventoryResult) -> Value {
    let mut row = json!({
        "activated": r.activated,
        "slots": slots_json(&r.slots),
    });
    let map = row.as_object_mut().expect("row is an object");
    if let Some(reason) = r.reason {
        map.insert("reason".to_string(), json!(reason));
    }
    if let Some(item_id) = r.item_id {
        map.insert("item_id".to_string(), json!(item_id));
    }
    if let Some(state) = r.state {
        map.insert("state".to_string(), json!(state));
    }
    if let Some(remaining_ms) = r.remaining_ms {
        map.insert("remaining_ms".to_string(), json!(remaining_ms));
    }
    row
}

fn trace_activate_from_inventory(args: &Value) -> Value {
    let r = activate_from_inventory(
        &get_slots(args, "slots"),
        get_f64(args, "normal_booster_time"),
        get_f64(args, "item_booster_time"),
        get_f64(args, "animal_booster_time"),
    );
    encode_activate_from_inventory(&r)
}

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

fn get_inventory_use_input(args: &Value, key: &str) -> Option<InventoryUseInput> {
    if args[key].is_null() {
        return None;
    }
    let input = &args[key];
    Some(InventoryUseInput {
        slots: get_slots(input, "slots"),
        stage_mode_id: get_i64(input, "stage_mode_id"),
        allow_refresh_in_mode72: get_bool(input, "allow_refresh_in_mode72"),
        positive_a80: get_f64(input, "positive_a80"),
        state_setter_environment_allowed: get_bool(input, "state_setter_environment_allowed"),
        normal_booster_time: get_f64(input, "normal_booster_time"),
        item_booster_time: get_f64(input, "item_booster_time"),
        animal_booster_time: get_f64(input, "animal_booster_time"),
    })
}

fn trace_evaluate_booster_inventory_use(args: &Value) -> Value {
    let input = InventoryUseInput {
        slots: get_slots(args, "slots"),
        stage_mode_id: get_i64(args, "stage_mode_id"),
        allow_refresh_in_mode72: get_bool(args, "allow_refresh_in_mode72"),
        positive_a80: get_f64(args, "positive_a80"),
        state_setter_environment_allowed: get_bool(args, "state_setter_environment_allowed"),
        normal_booster_time: get_f64(args, "normal_booster_time"),
        item_booster_time: get_f64(args, "item_booster_time"),
        animal_booster_time: get_f64(args, "animal_booster_time"),
    };
    let r = evaluate_booster_inventory_use(get_i64(args, "current_state"), &input);
    encode_inventory_use(&r)
}

fn trace_evaluate_vehicle_step(args: &Value) -> Value {
    let inventory_use = get_inventory_use_input(args, "inventory_use");
    match evaluate_vehicle_step(
        get_i64(args, "state"),
        get_i64(args, "remaining_ms"),
        get_i64(args, "elapsed_ms"),
        get_arr(args, "correction_dir"),
        get_f64(args, "direction_gain"),
        get_f64(args, "selected_slot"),
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
        inventory_use.as_ref(),
    ) {
        Ok(r) => json!({
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
                "trace": Value::Array(r.state_scalar.trace.iter().map(|step| json!({
                    "step": step.step,
                    "operation": step.operation,
                    "operand": fbits(step.operand),
                    "result": fbits(step.result),
                })).collect()),
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
        }),
        Err(message) => json!({"error": message}),
    }
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

fn trace_bs_zone_override(args: &Value) -> Value {
    match evaluate_bs_zone_override(
        get_i64(args, "digit_x"),
        get_i64(args, "digit_y"),
        get_bool(args, "flag_64c"),
        get_arr(args, "field_190"),
        get_arr(args, "velocity_c0"),
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
    ) {
        Ok(r) => encode_bs_zone(&r),
        Err(message) => json!({"error": message}),
    }
}

fn trace_hard_contact(args: &Value) -> Value {
    let r: HardContactVelocityResponse = evaluate_hard_contact_velocity_response(
        get_arr(args, "velocity_c0_before"),
        get_arr(args, "hit_normal"),
        get_arr(args, "basis_ad0"),
        get_f64(args, "restitution"),
    );
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

fn trace_soft_contact(args: &Value) -> Value {
    let r: SoftContactVelocityResponse = evaluate_soft_contact_velocity_response(
        get_arr(args, "velocity_c0_before"),
        get_arr(args, "velocity_delta"),
    );
    json!({
        "velocity_c0_before": arr(&r.velocity_c0_before),
        "velocity_delta": arr(&r.velocity_delta),
        "velocity_c0_after": arr(&r.velocity_c0_after),
        "direct_velocity_add": r.direct_velocity_add,
        "ordering": soft_contact_velocity_response::ORDERING,
        "runtime_authority": soft_contact_velocity_response::RUNTIME_AUTHORITY,
    })
}

fn encode_torque_events(events: &[RailTorqueEvent]) -> Value {
    Value::Array(
        events
            .iter()
            .map(|event| {
                json!({
                    "name": event.name,
                    "target": event.target,
                    "operation": event.operation,
                    "source": arr(&event.source),
                    "static_va": event.static_va,
                })
            })
            .collect(),
    )
}

fn trace_rail_state2_alignment_torque(args: &Value) -> Value {
    match evaluate_rail_state2_alignment_torque(
        get_arr(args, "torque_before"),
        get_f64(args, "signed_steering_axis"),
        get_f64(args, "drive_tune_718"),
        get_f64(args, "adc"),
        get_f64(args, "ae0"),
        get_f64(args, "aec"),
        get_bool(args, "alignment_failed"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(RailState2AlignmentTorque {
            torque_before,
            torque_after,
            events,
            signed_steering_axis,
            drive_tune_718,
            angular_scale_pi_tune_over_180,
            alignment_failed,
            fields,
        }) => json!({
            "torque_before": arr(&torque_before),
            "torque_after": arr(&torque_after),
            "events": encode_torque_events(&events),
            "signed_steering_axis": fbits(signed_steering_axis),
            "drive_tune_718": fbits(drive_tune_718),
            "angular_scale_pi_tune_over_180": fbits(angular_scale_pi_tune_over_180),
            "alignment_failed": alignment_failed,
            "fields": {
                "adc": fbits(fields.adc),
                "ae0": fbits(fields.ae0),
                "aec": fbits(fields.aec),
            },
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_rail_orientation_steering_torque(args: &Value) -> Value {
    match evaluate_rail_orientation_steering_torque(
        get_arr(args, "torque_before"),
        get_f64(args, "signed_steering_axis"),
        get_f64(args, "drive_tune_718"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(RailOrientationSteeringTorque {
            torque_before,
            torque_after,
            events,
            signed_steering_axis,
            drive_tune_718,
            angular_scale_pi_tune_over_180,
        }) => json!({
            "torque_before": arr(&torque_before),
            "torque_after": arr(&torque_after),
            "events": encode_torque_events(&events),
            "signed_steering_axis": fbits(signed_steering_axis),
            "drive_tune_718": fbits(drive_tune_718),
            "angular_scale_pi_tune_over_180": fbits(angular_scale_pi_tune_over_180),
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_select_rail_kinds(args: &Value) -> Value {
    let r: RailAccumulatorKinds = select_rail_accumulator_kinds(
        get_bool(args, "rail_contact"),
        get_i64(args, "rail_state_1cc"),
        get_bool(args, "gate_1c0_decoded"),
        get_f64(args, "positive_a80"),
        get_f64(args, "negative_a80"),
        get_bool(args, "boost_family_active"),
        get_bool(args, "boost_special_gate_9b8"),
        get_bool(args, "boost_special_gate_6e4"),
        get_i64(args, "attached_object_state"),
        get_f64(args, "abs_ae0"),
        get_f64(args, "adc"),
    );
    json!({
        "enabled_kinds": r.enabled_kinds,
        "force_branch": r.force_branch,
        "runtime_policy": r.runtime_policy,
    })
}

fn trace_rail_accumulator_producers(args: &Value) -> Value {
    match evaluate_rail_accumulator_producers(
        &get_i64_list(args, "enabled_kinds"),
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "accumulator_620_before"),
        get_arr(args, "rail_dir"),
        get_f64(args, "selected_a80"),
        get_f64(args, "accel_be0"),
        get_f64(args, "obf_6e8"),
        get_f64(args, "obf_808"),
        get_f64(args, "obf_814"),
        get_f64(args, "grip_720"),
        get_arr(args, "angular_velocity_cc"),
        get_arr(args, "field_604"),
        get_f64(args, "obf_660"),
        get_f64(args, "gravity_be8"),
    ) {
        Ok(RailAccumulatorProducersResult {
            enabled_kinds,
            events,
            accumulator_614_before,
            accumulator_614_after,
            accumulator_620_before,
            accumulator_620_after,
            gate_policy,
        }) => json!({
            "enabled_kinds": enabled_kinds,
            "events": Value::Array(events.iter().map(|event: &RailWriterEvent| json!({
                "kind": event.kind,
                "target": event.target,
                "operation": event.operation,
                "source": arr(&event.source),
            })).collect()),
            "accumulator_614_before": arr(&accumulator_614_before),
            "accumulator_614_after": arr(&accumulator_614_after),
            "accumulator_620_before": arr(&accumulator_620_before),
            "accumulator_620_after": arr(&accumulator_620_after),
            "gate_policy": gate_policy,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_fall_clamp(args: &Value) -> Value {
    let r: FallClampStateReplacement = evaluate_fall_clamp_state_replacement(
        get_arr(args, "position_90_before"),
        get_arr(args, "matrix_9c_before"),
        get_arr(args, "velocity_c0_before"),
        get_arr(args, "angular_velocity_cc_before"),
        get_f64(args, "drop_threshold"),
        get_f64(args, "clamp_z"),
    );
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

fn trace_checkpoint_respawn(args: &Value) -> Value {
    match evaluate_checkpoint_respawn_state_replacement(
        args["mode_family"].as_str().expect("mode_family must be a string"),
        get_bool(args, "checkpoint_gate"),
        get_bool(args, "sample_result_zero"),
        get_arr(args, "target_position"),
        get_arr(args, "course_reference_position"),
        get_arr(args, "velocity_c0_before"),
        get_arr(args, "angular_velocity_cc_before"),
    ) {
        Ok(CheckpointRespawnStateReplacement {
            selected,
            terminal_replacement,
            mode_family,
            checkpoint_gate,
            sample_result_zero,
            target_position,
            course_reference_position,
            forward_axis1,
            right_axis0,
            up_axis2,
            position_90_after,
            previous_position_70_after,
            matrix_9c_after,
            velocity_c0_before,
            velocity_c0_after,
            angular_velocity_cc_before,
            angular_velocity_cc_after,
            stage_offsets,
        }) => json!({
            "selected": selected,
            "terminal_replacement": terminal_replacement,
            "mode_family": mode_family,
            "checkpoint_gate": checkpoint_gate,
            "sample_result_zero": sample_result_zero,
            "target_position": arr(&target_position),
            "course_reference_position": arr(&course_reference_position),
            "forward_axis1": arr(&forward_axis1),
            "right_axis0": arr(&right_axis0),
            "up_axis2": arr(&up_axis2),
            "position_90_after": opt_arr(&position_90_after),
            "previous_position_70_after": opt_arr(&previous_position_70_after),
            "matrix_9c_after": opt_arr(&matrix_9c_after),
            "velocity_c0_before": arr(&velocity_c0_before),
            "velocity_c0_after": arr(&velocity_c0_after),
            "angular_velocity_cc_before": arr(&angular_velocity_cc_before),
            "angular_velocity_cc_after": arr(&angular_velocity_cc_after),
            "stage_offsets": {
                "function_va": stage_offsets.function_va,
                "gate": stage_offsets.gate,
                "target": stage_offsets.target,
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_checkpoint_target_selector(args: &Value) -> Value {
    match evaluate_checkpoint_target_selector(
        args["mode_family"].as_str().expect("mode_family must be a string"),
        get_arr(args, "target_before"),
        get_bool(args, "gate_before"),
        args["event"].as_str().expect("event must be a string"),
        get_opt_arr(args, "current_kart_position"),
        get_bool(args, "config_gate"),
        get_opt_i64(args, "mode_state"),
        get_opt_i64(args, "mission_index"),
        get_opt_i64(args, "mission_count"),
        get_bool(args, "retire_detected"),
        get_bool(args, "reset_suppressed"),
        get_opt_f64(args, "trace_value"),
        get_opt_f64(args, "trace_min"),
        get_opt_f64(args, "trace_max"),
        get_bool(args, "peer_present"),
        get_bool(args, "kart_event_accepts"),
        get_bool(args, "stage_input_blocked"),
        get_opt_i64(args, "tick_ms"),
        get_opt_i64(args, "gate_armed_tick_ms"),
    ) {
        Ok(CheckpointTargetSelectorResult {
            mode_family,
            event,
            accepted,
            target_before,
            target_after,
            target_written,
            gate_before,
            gate_after,
            gate_written,
            scope,
        }) => json!({
            "mode_family": mode_family,
            "event": event,
            "accepted": accepted,
            "target_before": arr(&target_before),
            "target_after": arr(&target_after),
            "target_written": target_written,
            "gate_before": gate_before,
            "gate_after": gate_after,
            "gate_written": gate_written,
            "source": {
                "c016_init_va": checkpoint_target_selector::SOURCE_C016_INIT_VA,
                "c016_progress_snapshot_va": checkpoint_target_selector::SOURCE_C016_PROGRESS_SNAPSHOT_VA,
                "c016_gate_writers": checkpoint_target_selector::SOURCE_C016_GATE_WRITERS,
                "c72c_init_va": checkpoint_target_selector::SOURCE_C72C_INIT_VA,
                "c72c_progress_snapshot_va": checkpoint_target_selector::SOURCE_C72C_PROGRESS_SNAPSHOT_VA,
            },
            "scope": scope,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_pit_resource_step(args: &Value) -> Value {
    match evaluate_pit_resource_step(
        get_bool(args, "active"),
        get_f64(args, "rate"),
        get_i64(args, "mode"),
        get_f64(args, "accum_before"),
        get_f64(args, "dt"),
    ) {
        Ok(PitResourceStep {
            active_before_cleanup,
            rate_before_cleanup,
            mode_before_cleanup,
            accum_before,
            accum_increment,
            accum_after,
            fx_selected,
            active_after_cleanup,
            rate_after_cleanup,
            mode_after_cleanup,
            vehicle_force_delta,
            vehicle_torque_delta,
            dynamics_role,
            drain_consumer,
        }) => json!({
            "active_before_cleanup": active_before_cleanup,
            "rate_before_cleanup": fbits(rate_before_cleanup),
            "mode_before_cleanup": mode_before_cleanup,
            "accum_before": fbits(accum_before),
            "accum_increment": fbits(accum_increment),
            "accum_after": fbits(accum_after),
            "fx_selected": fx_selected,
            "active_after_cleanup": active_after_cleanup,
            "rate_after_cleanup": fbits(rate_after_cleanup),
            "mode_after_cleanup": mode_after_cleanup,
            "vehicle_force_delta": arr(&vehicle_force_delta),
            "vehicle_torque_delta": arr(&vehicle_torque_delta),
            "dynamics_role": dynamics_role,
            "drain_consumer": drain_consumer,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_drain_pit_accumulator(args: &Value) -> Value {
    let r: PitAccumulatorDrain = drain_pit_accumulator(
        get_f64(args, "accum_before"),
        get_f64(args, "race_progress_before"),
    );
    json!({
        "drained": fbits(r.drained),
        "accum_after": fbits(r.accum_after),
        "race_progress_after": fbits(r.race_progress_after),
        "vehicle_state_mutated": r.vehicle_state_mutated,
    })
}

fn trace_bcharge_surface_step(args: &Value) -> Value {
    let r: BchargeSurfaceStep = evaluate_bcharge_surface_step(
        get_bool(args, "active"),
        get_f64(args, "accum_b10"),
        get_f64(args, "raw_760"),
    );
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

fn trace_commit_pending(args: &Value) -> Value {
    let value = commit_pending(
        get_f64(args, "accum_b10"),
        get_f64(args, "accum_b2c"),
        get_f64(args, "raw_760"),
    );
    match value {
        Some(r) => json!({
            "value": {
                "accum_b10": fbits(r.accum_b10),
                "pending_b1c": r.pending_b1c,
                "progress_b20": fbits(r.progress_b20),
                "history_b38": fbits(r.history_b38),
                "accum_b2c": fbits(r.accum_b2c),
            }
        }),
        None => json!({"value": Value::Null}),
    }
}

fn trace_mz_magnet_trigger(args: &Value) -> Value {
    let r: MzMagnetTrigger = evaluate_mz_magnet_trigger(
        get_arr(args, "current_reference_xyz"),
        get_arr(args, "authored_target_xyz"),
        get_f64(args, "velocity_scale"),
        get_bool(args, "special_state_family_already_active"),
    );
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

fn trace_parse_jm_dj_token(args: &Value) -> Value {
    match parse_jm_dj_surface_token(args["token"].as_str().expect("token must be a string")) {
        Ok(JmDjSurfaceToken::Jm { jm_horizontal_param, jm_basis_param, token }) => json!({
            "variant": "JM",
            "jm_horizontal_param": fbits(jm_horizontal_param),
            "jm_basis_param": fbits(jm_basis_param),
            "token": token,
        }),
        Ok(JmDjSurfaceToken::Dj { dj_vector, token }) => json!({
            "variant": "DJ",
            "dj_vector": arr(&dj_vector),
            "token": token,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_jm_dj_trigger(args: &Value) -> Value {
    match evaluate_jm_dj_trigger(
        args["variant"].as_str().expect("variant must be a string"),
        get_bool(args, "special_state_family_already_active"),
        get_opt_arr(args, "velocity_c0_before"),
        get_opt_arr(args, "jm_basis_190"),
        get_opt_f64(args, "jm_horizontal_param"),
        get_opt_f64(args, "jm_basis_param"),
        get_opt_arr(args, "dj_vector"),
        get_bool(args, "boost_multiplier_64c"),
    ) {
        Ok(JmDjTrigger {
            variant,
            branch,
            velocity_c0_after,
            accepted,
            state_transition,
            cooldown_648_after,
            velocity_write_occurs_before_state_family_gate,
            velocity_operation,
            policy,
        }) => json!({
            "variant": variant,
            "branch": branch,
            "velocity_c0_after": arr(&velocity_c0_after),
            "accepted": accepted,
            "state_transition": match &state_transition {
                Some(t) => json!({"state": t.state, "remaining_ms": t.remaining_ms}),
                None => Value::Null,
            },
            "cooldown_648_after": fbits(cooldown_648_after),
            "velocity_write_occurs_before_state_family_gate": velocity_write_occurs_before_state_family_gate,
            "velocity_operation": velocity_operation,
            "policy": policy,
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_event_obstacle_contact(args: &Value) -> Value {
    match apply_event_obstacle_contact_source(
        get_arr(args, "velocity"),
        get_arr(args, "normal"),
        get_arr(args, "obstacle_motion"),
    ) {
        Ok(response) => {
            let source = &response.source;
            json!({
                "branch": source.branch,
                "motion_normal_dot_f32": fbits(source.motion_normal_dot_f32),
                "approaching_dot_f32": fbits(source.approaching_dot_f32),
                "approaching_gate": source.approaching_gate,
                "side_normal_z_gate": source.side_normal_z_gate,
                "normal_component": arr(&source.normal_component),
                "tangent": arr(&source.tangent),
                "tangent_length_f32": fbits(source.tangent_length_f32),
                "normal_component_length_f32": fbits(
                    source.normal_component_length_f32
                ),
                "response_scale_f32": fbits(source.response_scale_f32),
                "velocity_source": arr(&source.velocity_source),
                "selected": response.selected,
                "velocity_before": arr(&response.velocity_before),
                "velocity_after": arr(&response.velocity_after),
            })
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
                "booster_resource/tune_time_ms" => trace_tune_time_ms(args),
                "booster_resource/activation_for_item" => trace_activation_for_item(args),
                "booster_resource/advance_state_timer" => trace_advance_state_timer(args),
                "booster_resource/add_inventory_item" => trace_add_inventory_item(args),
                "booster_resource/consume_inventory_item" => trace_consume_inventory_item(args),
                "booster_vehicle_step/activate_from_inventory" => trace_activate_from_inventory(args),
                "booster_vehicle_step/evaluate_booster_inventory_use" => {
                    trace_evaluate_booster_inventory_use(args)
                }
                "booster_vehicle_step/evaluate_vehicle_step" => trace_evaluate_vehicle_step(args),
                "bs_zone_override/evaluate_bs_zone_override" => trace_bs_zone_override(args),
                "hard_contact_velocity_response/evaluate_hard_contact_velocity_response" => {
                    trace_hard_contact(args)
                }
                "soft_contact_velocity_response/evaluate_soft_contact_velocity_response" => {
                    trace_soft_contact(args)
                }
                "rail_accumulator_producers/evaluate_rail_state2_alignment_torque" => {
                    trace_rail_state2_alignment_torque(args)
                }
                "rail_accumulator_producers/evaluate_rail_orientation_steering_torque" => {
                    trace_rail_orientation_steering_torque(args)
                }
                "rail_accumulator_producers/select_rail_accumulator_kinds" => {
                    trace_select_rail_kinds(args)
                }
                "rail_accumulator_producers/evaluate_rail_accumulator_producers" => {
                    trace_rail_accumulator_producers(args)
                }
                "fall_clamp_state_replacement/evaluate_fall_clamp_state_replacement" => {
                    trace_fall_clamp(args)
                }
                "checkpoint_respawn_state_replacement/evaluate_checkpoint_respawn_state_replacement" => {
                    trace_checkpoint_respawn(args)
                }
                "checkpoint_target_selector/evaluate_checkpoint_target_selector" => {
                    trace_checkpoint_target_selector(args)
                }
                "state18_feedback/representative_stroke_dispatch" => {
                    json!({"value": representative_stroke_dispatch(
                        get_i64(args, "event_type"), get_bool(args, "kart_a9c"))})
                }
                "state18_feedback/select_state18_feedback" => {
                    json!({"value": select_state18_feedback(
                        get_bool(args, "stroke_dispatch_selected"),
                        get_bool(args, "control_lock_1c0"),
                        get_bool(args, "event_active"),
                        get_bool(args, "effect_feedback_enabled_258"))})
                }
                "state18_feedback/timer_expiry_next_state" => {
                    json!({"value": timer_expiry_next_state(
                        get_i64(args, "current_state"), get_bool(args, "timer_active"))})
                }
                "pit_resource_step/evaluate_pit_resource_step" => trace_pit_resource_step(args),
                "pit_resource_step/drain_pit_accumulator" => trace_drain_pit_accumulator(args),
                "tune_boost_gauge/evaluate_bcharge_surface_step" => {
                    trace_bcharge_surface_step(args)
                }
                "tune_boost_gauge/gauge_cap" => {
                    json!({"value": fbits(gauge_cap(get_f64(args, "raw_760")))})
                }
                "tune_boost_gauge/gauge_accum_clamped" => {
                    json!({"value": fbits(gauge_accum_clamped(
                        get_f64(args, "accum_b10"),
                        get_f64(args, "accum_b2c"),
                        get_f64(args, "raw_760")))})
                }
                "tune_boost_gauge/pending_accum_delta" => {
                    json!({"value": opt_fbits(pending_accum_delta(
                        get_f64(args, "dt"),
                        get_f64(args, "progress_b20"),
                        get_f64(args, "conditional_multiplier_994")))})
                }
                "tune_boost_gauge/accumulate_pending" => {
                    json!({"value": opt_fbits(accumulate_pending(
                        get_f64(args, "accum_b2c"),
                        get_f64(args, "dt"),
                        get_f64(args, "progress_b20"),
                        get_f64(args, "conditional_multiplier_994")))})
                }
                "tune_boost_gauge/state34_drift_gauge_event" => {
                    let (total, cycle) = state34_drift_gauge_event(
                        get_i64(args, "state"),
                        get_bool(args, "suppress_5e8"),
                        get_bool(args, "eligible_fba"),
                        get_bool(args, "active_fbb"),
                        get_i64(args, "total_count"),
                        get_i64(args, "cycle_count"),
                    );
                    json!({"value": [total, cycle]})
                }
                "tune_boost_gauge/state34_drift_gauge_window_can_activate" => {
                    json!({"value": state34_drift_gauge_window_can_activate(
                        get_bool(args, "eligible_fba"),
                        get_bool(args, "active_fbb"),
                        get_i64(args, "divisor"),
                        get_i64(args, "total_count"),
                        get_i64(args, "cycle_count"),
                        get_i64(args, "activation_count"),
                        get_i64(args, "current_tick"),
                        get_i64(args, "previous_expiry_tick"))})
                }
                "tune_boost_gauge/state34_drift_gauge_window_expired" => {
                    json!({"value": state34_drift_gauge_window_expired(
                        get_bool(args, "active_fbb"),
                        get_i64(args, "current_tick"),
                        get_i64(args, "expiry_tick"))})
                }
                "tune_boost_gauge/commit_pending" => trace_commit_pending(args),
                "tune_boost_gauge/gauge_ratio" => {
                    json!({"value": opt_fbits(gauge_ratio(
                        get_f64(args, "accum_b10"),
                        get_f64(args, "accum_b2c"),
                        get_f64(args, "raw_760")))})
                }
                "tune_boost_gauge/mq_display_target" => {
                    json!({"value": fbits(mq_display_target(get_f64(args, "ratio")))})
                }
                "mz_magnet_trigger/evaluate_mz_magnet_trigger" => trace_mz_magnet_trigger(args),
                "jm_dj_trigger/parse_jm_dj_surface_token" => trace_parse_jm_dj_token(args),
                "jm_dj_trigger/evaluate_jm_dj_trigger" => trace_jm_dj_trigger(args),
                "event_obstacle_contact/apply_event_obstacle_contact_source" => {
                    trace_event_obstacle_contact(args)
                }
                other => panic!("unknown case fn {other:?}"),
            }
        })
        .collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
