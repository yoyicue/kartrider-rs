//! Port of `scripts/evaluate_rail_accumulator_producers.py` — executable
//! seven-writer rail accumulator formulas and the exact B29350 dispatch.

use crate::num::f32q;

/// Python result key `runtime_policy` on `select_rail_accumulator_kinds`.
pub const RUNTIME_POLICY: &str = "Static dispatch is exact. Runtime reachability of kinds 20/21/23 is proven by rail_runtime_coverage_20260715.json; autonomous rail-path geometry and decoded per-step context remain separate.";

/// Python result key `gate_policy` on `evaluate_rail_accumulator_producers`.
pub const GATE_POLICY: &str = "enabled_kinds come from the exact B29350 selector; runtime promotion rail_runtime_promotion_20260723.json proves kinds20/21/23 activation without aliasing generic state";

// static_source provenance shared by all four evaluators.
pub const STATIC_FUNCTION_VA: &str = "0xB29350";
pub const STATIC_SIGNED_AXIS_GETTER_VA: &str = "0x6F3C80";
pub const STATIC_ANGULAR_SCALE_GETTER_VA: &str = "0x6F3B80";
pub const STATIC_STATE2_QUALIFICATION: &str =
    "pre-kind24 accumulator producer; distinct from visible kind20..26 writer hooks";
pub const STATIC_ORIENTATION_SCALAR_VA: &str = "0x11ECB04";
pub const STATIC_ORIENTATION_SCALAR: f64 = 200.0;
pub const STATIC_ORIENTATION_QUALIFICATION: &str = "pre-kind25 accumulator producer";
pub const STATIC_RAIL_CONTACT_GATE_VA: &str = "0xB2936F-0xB29374";
pub const STATIC_FORCE_DISPATCH_VA: &str = "0xB29588-0xB29890";
pub const STATIC_STATE2_FAST_TORQUE_GATE_VA: &str = "0xB29969-0xB29C71";
pub const STATIC_ORIENTATION_TORQUE_ELSE_VA: &str = "0xB29C89-0xB29CFF";
pub const STATIC_GRAVITY_WRITER_VA: &str = "0xB29E1B-0xB29E64";
pub const STATIC_STATE2_THRESHOLD_CONSTANT_VA: &str = "0x11ECAE8";
pub const STATIC_STATE2_THRESHOLD_CONSTANT: f64 = 0.2;

/// KartRider's .rdata literal at 0x11ECAFC is slightly below the usual
/// float32 rounding of Python's pi; use the executable's exact constant.
const EXECUTABLE_PI: f64 = 3.141592025756836;

/// One torque-writer event dict from the two alignment/steering evaluators:
/// `{"name", "target", "operation", "source", "static_va"}`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailTorqueEvent {
    pub name: &'static str,
    pub target: &'static str,
    pub operation: &'static str,
    pub source: [f64; 3],
    pub static_va: &'static str,
}

/// Python `fields` sub-dict on the state2 evaluator.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState2Fields {
    pub adc: f64,
    pub ae0: f64,
    pub aec: f64,
}

/// Python `evaluate_rail_state2_alignment_torque(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState2AlignmentTorque {
    pub torque_before: [f64; 3],
    pub torque_after: [f64; 3],
    pub events: Vec<RailTorqueEvent>,
    pub signed_steering_axis: f64,
    pub drive_tune_718: f64,
    pub angular_scale_pi_tune_over_180: f64,
    pub alignment_failed: bool,
    pub fields: RailState2Fields,
}

/// Numeric-mode dispatcher: Python `cast = _f32 if numeric_mode ==
/// "float32" else float`, where `float` is the identity on Python floats.
fn cast_fn(numeric_mode: &str) -> Result<fn(f64) -> f64, String> {
    match numeric_mode {
        "float32" => Ok(f32q),
        "float64" => Ok(|v| v),
        _ => Err("numeric_mode must be float32 or float64".to_string()),
    }
}

/// `evaluate_rail_state2_alignment_torque(...)` — B299B5-B29C38 before the
/// visible kind-24 damping writer. `sub_6F3B80(object+0x660)` is not the
/// vehicle mass getter; it returns `pi * object[0x718] / 180`.
pub fn evaluate_rail_state2_alignment_torque(
    torque_before: [f64; 3],
    signed_steering_axis: f64,
    drive_tune_718: f64,
    adc: f64,
    ae0: f64,
    aec: f64,
    alignment_failed: bool,
    numeric_mode: &str,
) -> Result<RailState2AlignmentTorque, String> {
    let cast = cast_fn(numeric_mode)?;
    let mut torque = [cast(torque_before[0]), cast(torque_before[1]), cast(torque_before[2])];
    let executable_pi = cast(EXECUTABLE_PI);
    let angular_scale = cast(cast(executable_pi * cast(drive_tune_718)) / cast(180.0));
    let steering_z = cast(cast(cast(signed_steering_axis) * angular_scale) * cast(600.0));
    torque[2] = cast(torque[2] + steering_z);
    let mut events = vec![RailTorqueEvent {
        name: "state2_signed_steering_z",
        target: "torque",
        operation: "add",
        source: [0.0, 0.0, steering_z],
        static_va: "0xB299B5-0xB29A0B",
    }];
    if alignment_failed {
        let lateral_z = cast(cast(ae0) * cast(1.600000023841858));
        let vertical_x = cast(cast(aec) * cast(if cast(adc) > 0.0 { -2.0 } else { 2.0 }));
        torque[0] = cast(torque[0] + vertical_x);
        torque[2] = cast(torque[2] + lateral_z);
        events.push(RailTorqueEvent {
            name: "state2_failed_alignment_xz",
            target: "torque",
            operation: "add",
            source: [vertical_x, 0.0, lateral_z],
            static_va: "0xB29BAB-0xB29C38",
        });
    }
    Ok(RailState2AlignmentTorque {
        torque_before,
        torque_after: torque,
        events,
        signed_steering_axis: cast(signed_steering_axis),
        drive_tune_718: cast(drive_tune_718),
        angular_scale_pi_tune_over_180: angular_scale,
        alignment_failed,
        fields: RailState2Fields { adc: cast(adc), ae0: cast(ae0), aec: cast(aec) },
    })
}

/// Python `evaluate_rail_orientation_steering_torque(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailOrientationSteeringTorque {
    pub torque_before: [f64; 3],
    pub torque_after: [f64; 3],
    pub events: Vec<RailTorqueEvent>,
    pub signed_steering_axis: f64,
    pub drive_tune_718: f64,
    pub angular_scale_pi_tune_over_180: f64,
}

/// `evaluate_rail_orientation_steering_torque(...)` — B29C7B-B29CC7 before
/// the visible kind-25 writer. Every non-state2 RAIL contact branch adds the
/// signed steering axis times `pi*tune718/180` and the executable's 200
/// scalar to torque.y.
pub fn evaluate_rail_orientation_steering_torque(
    torque_before: [f64; 3],
    signed_steering_axis: f64,
    drive_tune_718: f64,
    numeric_mode: &str,
) -> Result<RailOrientationSteeringTorque, String> {
    let cast = cast_fn(numeric_mode)?;
    let mut torque = [cast(torque_before[0]), cast(torque_before[1]), cast(torque_before[2])];
    let executable_pi = cast(EXECUTABLE_PI);
    let angular_scale = cast(cast(executable_pi * cast(drive_tune_718)) / cast(180.0));
    let steering_y = cast(cast(cast(signed_steering_axis) * angular_scale) * cast(200.0));
    torque[1] = cast(torque[1] + steering_y);
    Ok(RailOrientationSteeringTorque {
        torque_before,
        torque_after: torque,
        events: vec![RailTorqueEvent {
            name: "non_state2_signed_steering_y",
            target: "torque",
            operation: "add",
            source: [0.0, steering_y, 0.0],
            static_va: "0xB29C7B-0xB29CC7",
        }],
        signed_steering_axis: cast(signed_steering_axis),
        drive_tune_718: cast(drive_tune_718),
        angular_scale_pi_tune_over_180: angular_scale,
    })
}

/// Python `select_rail_accumulator_kinds(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailAccumulatorKinds {
    pub enabled_kinds: Vec<i64>,
    pub force_branch: &'static str,
    pub runtime_policy: &'static str,
}

/// `select_rail_accumulator_kinds(...)` — exact B29350 rail writer dispatch,
/// separate from writer arithmetic.
pub fn select_rail_accumulator_kinds(
    rail_contact: bool,
    rail_state_1cc: i64,
    gate_1c0_decoded: bool,
    positive_a80: f64,
    negative_a80: f64,
    boost_family_active: bool,
    boost_special_gate_9b8: bool,
    boost_special_gate_6e4: bool,
    attached_object_state: i64,
    abs_ae0: f64,
    adc: f64,
) -> RailAccumulatorKinds {
    let mut kinds = Vec::new();
    let mut force_branch = "none";
    if rail_contact && rail_state_1cc == 3 && !gate_1c0_decoded {
        if positive_a80 > 0.0 {
            let special = boost_special_gate_9b8
                && boost_special_gate_6e4
                && matches!(attached_object_state, 2 | 4 | 5 | 6)
                && boost_family_active;
            if special {
                kinds.push(20);
                force_branch = "boost_special_positive";
            } else if boost_family_active {
                kinds.push(21);
                force_branch = "boost_positive";
            } else {
                kinds.push(22);
                force_branch = "drive_positive";
            }
        } else if negative_a80 != 0.0 {
            kinds.push(23);
            force_branch = "brake_negative";
        }
    }
    if rail_contact {
        if rail_state_1cc == 2 {
            if abs_ae0 >= adc * 0.2 {
                kinds.push(24);
            }
        } else {
            kinds.push(25);
        }
        kinds.push(26);
    }
    RailAccumulatorKinds {
        enabled_kinds: kinds,
        force_branch,
        runtime_policy: RUNTIME_POLICY,
    }
}

/// One writer event dict from `evaluate_rail_accumulator_producers`:
/// `{"kind", "target", "operation", "source"}`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailWriterEvent {
    pub kind: i64,
    pub target: &'static str,
    pub operation: &'static str,
    pub source: [f64; 3],
}

/// Python `evaluate_rail_accumulator_producers(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailAccumulatorProducersResult {
    pub enabled_kinds: Vec<i64>,
    pub events: Vec<RailWriterEvent>,
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub gate_policy: &'static str,
}

fn scale(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

/// `_apply(target, source, subtract)` — `target[i] + sign * source[i]` with
/// `sign = -1.0` on the subtract path (a multiply, matching Python).
fn apply(target: [f64; 3], source: [f64; 3], subtract: bool) -> [f64; 3] {
    let sign = if subtract { -1.0 } else { 1.0 };
    [
        target[0] + sign * source[0],
        target[1] + sign * source[1],
        target[2] + sign * source[2],
    ]
}

/// `evaluate_rail_accumulator_producers(...)` — the seven-writer loop.
pub fn evaluate_rail_accumulator_producers(
    enabled_kinds: &[i64],
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    rail_dir: [f64; 3],
    selected_a80: f64,
    accel_be0: f64,
    obf_6e8: f64,
    obf_808: f64,
    obf_814: f64,
    grip_720: f64,
    angular_velocity_cc: [f64; 3],
    field_604: [f64; 3],
    obf_660: f64,
    gravity_be8: f64,
) -> Result<RailAccumulatorProducersResult, String> {
    let mut force = accumulator_614_before;
    let mut torque = accumulator_620_before;
    let mut events = Vec::new();
    for &kind in enabled_kinds {
        let (source, target, subtract): ([f64; 3], &'static str, bool) = match kind {
            20 => (scale(rail_dir, selected_a80 * accel_be0 * obf_6e8 * obf_808), "force", false),
            21 => (scale(rail_dir, selected_a80 * accel_be0 * obf_6e8 * obf_814), "force", false),
            22 => (scale(rail_dir, selected_a80 * accel_be0 * obf_6e8), "force", false),
            23 => (scale(rail_dir, accel_be0 * grip_720), "force", true),
            24 => (scale(angular_velocity_cc, 20.0), "torque", true),
            25 => (scale(angular_velocity_cc, 30.0), "torque", true),
            26 => (scale(field_604, obf_660 * gravity_be8), "force", false),
            _ => return Err(format!("unsupported rail writer kind {kind}")),
        };
        if target == "force" {
            force = apply(force, source, subtract);
        } else {
            torque = apply(torque, source, subtract);
        }
        events.push(RailWriterEvent {
            kind,
            target,
            operation: if subtract { "subtract" } else { "add" },
            source,
        });
    }
    Ok(RailAccumulatorProducersResult {
        enabled_kinds: enabled_kinds.to_vec(),
        events,
        accumulator_614_before,
        accumulator_614_after: force,
        accumulator_620_before,
        accumulator_620_after: torque,
        gate_policy: GATE_POLICY,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state2_torque_factory_capture_float32() {
        // Constants computed with the Python authority (numeric_mode="float32"):
        //   evaluate_rail_state2_alignment_torque(torque_before=[0,0,0],
        //     signed_steering_axis=-1.0, drive_tune_718=10.0, adc=0, ae0=0,
        //     aec=0, alignment_failed=False)
        //   -> torque_after[2] == -104.71973419189453 (the observed -104.7197).
        let r = evaluate_rail_state2_alignment_torque(
            [0.0, 0.0, 0.0],
            -1.0,
            10.0,
            0.0,
            0.0,
            0.0,
            false,
            "float32",
        )
        .unwrap();
        assert_eq!(r.torque_after[2], -104.71973419189453_f64);
        assert_eq!(r.events.len(), 1);
        assert_eq!(r.events[0].name, "state2_signed_steering_z");
    }

    #[test]
    fn orientation_torque_run98_state3_float32() {
        // Constants computed with the Python authority (numeric_mode="float32"):
        //   evaluate_rail_orientation_steering_torque(torque_before=[0,0,0],
        //     signed_steering_axis=-1.0, drive_tune_718=10.0)
        //   -> torque_after[1] == -34.906578063964844 (observed -34.9065781).
        let r = evaluate_rail_orientation_steering_torque(
            [0.0, 0.0, 0.0],
            -1.0,
            10.0,
            "float32",
        )
        .unwrap();
        assert_eq!(r.torque_after[1], -34.906578063964844_f64);
        assert_eq!(r.events[0].operation, "add");
    }

    #[test]
    fn numeric_mode_is_validated() {
        assert_eq!(
            evaluate_rail_orientation_steering_torque([0.0; 3], 1.0, 1.0, "float16").unwrap_err(),
            "numeric_mode must be float32 or float64"
        );
    }

    #[test]
    fn kind_dispatch_and_writer_loop() {
        // Python authority:
        //   select_rail_accumulator_kinds(rail_contact=True, rail_state_1cc=3,
        //     gate_1c0_decoded=False, positive_a80=1.0, negative_a80=0.0,
        //     boost_family_active=False, ...) -> kinds [22, 25, 26]
        let kinds = select_rail_accumulator_kinds(
            true, 3, false, 1.0, 0.0, false, false, false, 0, 0.0, 0.0,
        );
        assert_eq!(kinds.enabled_kinds, vec![22, 25, 26]);
        assert_eq!(kinds.force_branch, "drive_positive");
        //   evaluate_rail_accumulator_producers(enabled_kinds=[22],
        //     accumulator_614_before=[1,2,3], accumulator_620_before=[4,5,6],
        //     rail_dir=[1,0,0], selected_a80=2.0, accel_be0=3.0, obf_6e8=4.0, ...)
        //   -> force [25,2,3], torque unchanged.
        let r = evaluate_rail_accumulator_producers(
            &[22],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0, 0.0, 0.0],
            2.0,
            3.0,
            4.0,
            0.0,
            0.0,
            0.0,
            [0.0; 3],
            [0.0; 3],
            0.0,
            0.0,
        )
        .unwrap();
        assert_eq!(r.accumulator_614_after, [25.0, 2.0, 3.0]);
        assert_eq!(r.accumulator_620_after, [4.0, 5.0, 6.0]);
        assert_eq!(
            evaluate_rail_accumulator_producers(
                &[99],
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                [0.0; 3],
                [0.0; 3],
                0.0,
                0.0,
            )
            .unwrap_err(),
            "unsupported rail writer kind 99"
        );
    }
}
