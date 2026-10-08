//! Integrator parity trace tool.
//!
//! Reads a JSON spec from stdin:
//! `{"cases": [{"fn": "position"|"orientation"|"angular"|"linear"|"integrator_dt", "args": {...}}, ...]}`
//! and prints a JSON array of result rows, one per case. Every f64 is emitted
//! as its IEEE-754 bit pattern (`format!("{:016x}", v.to_bits())`, matching
//! the Python side's `struct.pack('<d')`); bools/ints/strings pass through.
//! A domain `Err` is emitted as `{"error": <message>}` so the Python side can
//! diff `ValueError` texts bit-for-bit.

use std::io::Read;

use kart_physics::domains::angular_velocity_integrator::evaluate_angular_velocity_integrator;
use kart_physics::domains::integrator_dt::{select_integrator_dt, LegacyDtRow};
use kart_physics::domains::linear_velocity_integrator::evaluate_linear_velocity_integrator;
use kart_physics::domains::orientation_integrator::evaluate_orientation_integrator;
use kart_physics::domains::position_integrator::evaluate_position_integrator;
use serde_json::{json, Value};

fn fbits(value: f64) -> Value {
    Value::String(format!("{:016x}", value.to_bits()))
}

fn arr<const N: usize>(values: &[f64; N]) -> Value {
    Value::Array(values.iter().map(|&v| fbits(v)).collect())
}

fn get_f64(args: &Value, key: &str) -> f64 {
    args[key].as_f64().unwrap_or_else(|| panic!("{key} must be a float"))
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

fn trace_position(args: &Value) -> Value {
    match evaluate_position_integrator(
        get_arr(args, "position_90_before"),
        get_arr(args, "velocity_c0_before"),
        get_f64(args, "dt"),
        get_opt_arr(args, "pre_position_velocity_delta"),
        args["state_1d0"].as_i64().expect("state_1d0 must be an int"),
        args["latch_658"].as_bool().expect("latch_658 must be a bool"),
        get_f64(args, "obf_b44"),
        get_opt_arr(args, "axis_ab8"),
        get_opt_arr(args, "axis_a48"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_orientation(args: &Value) -> Value {
    match evaluate_orientation_integrator(
        get_arr(args, "matrix_9c_before"),
        get_arr(args, "angular_velocity_cc_before"),
        get_f64(args, "dt"),
        args["pre_clear_xy"].as_bool().expect("pre_clear_xy must be a bool"),
        args["stabilization_disabled_b70"]
            .as_bool()
            .expect("stabilization_disabled_b70 must be a bool"),
        args["rail_state_1cc"]
            .as_i64()
            .expect("rail_state_1cc must be an int"),
        args["runtime_float32_exact"]
            .as_bool()
            .expect("runtime_float32_exact must be a bool"),
    ) {
        Ok(r) => {
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
        Err(message) => json!({"error": message}),
    }
}

fn trace_angular(args: &Value) -> Value {
    match evaluate_angular_velocity_integrator(
        get_arr(args, "angular_velocity_cc_before"),
        get_arr(args, "accumulator_620"),
        get_arr(args, "matrix_9c4"),
        get_f64(args, "dt"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_linear(args: &Value) -> Value {
    match evaluate_linear_velocity_integrator(
        get_arr(args, "velocity_c0_before"),
        get_arr(args, "accumulator_614_before"),
        get_arr(args, "bac_force"),
        get_f64(args, "sub_660"),
        get_f64(args, "dt"),
        get_opt_arr(args, "post_bs_later_delta"),
        args["numeric_mode"].as_str().expect("numeric_mode must be a string"),
    ) {
        Ok(r) => json!({
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
        }),
        Err(message) => json!({"error": message}),
    }
}

fn trace_integrator_dt(args: &Value) -> Value {
    let captured = if args["captured"].is_null() {
        None
    } else {
        Some(get_f64(args, "captured"))
    };
    let total_force = get_opt_arr(args, "total_force");
    let legacy = if args["legacy"].is_null() {
        None
    } else {
        let row = &args["legacy"];
        Some(LegacyDtRow {
            pre_xyz: get_arr(row, "pre_xyz"),
            post_xyz: get_arr(row, "post_xyz"),
            sub_660: get_f64(row, "sub_660"),
        })
    };
    match select_integrator_dt(captured, total_force, legacy) {
        Ok((value, source)) => json!({"value": fbits(value), "source": source}),
        Err(message) => json!({"error": message}),
    }
}

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
                "position" => trace_position(args),
                "orientation" => trace_orientation(args),
                "angular" => trace_angular(args),
                "linear" => trace_linear(args),
                "integrator_dt" => trace_integrator_dt(args),
                other => panic!("unknown case fn {other:?}"),
            }
        })
        .collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
