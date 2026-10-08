//! Port of `scripts/evaluate_rail_topology.py` — executable rail-constrained
//! rigid-body producer chain (`evaluate_rail_step`).
//!
//! Owns writer selection, accumulator composition, damping, and rigid-body
//! integration by composing the already-ported leaf modules. Ordering:
//! `B29350 rail dispatch/writers -> B2D5D0 damping -> B2D940 velocity ->
//! B2DA80 rail pose / B2FD80 ordinary pose`.

use super::angular_velocity_integrator::{
    evaluate_angular_velocity_integrator, AngularVelocityIntegratorResult,
};
use super::common_damping::{evaluate_common_damping, CommonDampingResult};
use super::linear_velocity_integrator::{
    evaluate_linear_velocity_integrator, LinearVelocityIntegratorResult,
};
use super::orientation_integrator::{
    evaluate_orientation_integrator, OrientationIntegratorResult,
};
use super::position_integrator::{evaluate_position_integrator, PositionIntegratorResult};
use super::rail_accumulator_producers::{
    evaluate_rail_accumulator_producers, evaluate_rail_orientation_steering_torque,
    evaluate_rail_state2_alignment_torque, select_rail_accumulator_kinds, RailAccumulatorKinds,
    RailAccumulatorProducersResult, RailOrientationSteeringTorque, RailState2AlignmentTorque,
};
use super::rail_constraint_orientation::{
    evaluate_rail_constraint_orientation, RailConstraintOrientation,
};

/// Python result key `ordering`.
pub const ORDERING: &str = "B29350 rail dispatch/writers -> B2D5D0 damping -> B2D940 velocity -> B2DA80 rail pose / B2FD80 ordinary pose";

/// Python result key `closure_status` per branch.
pub const CLOSURE_WITH_CONSTRAINT: &str = "exact_rail_producer_and_constraint_pose_chain";
pub const CLOSURE_WITHOUT_CONSTRAINT: &str =
    "rail_producer_chain_without_explicit_constraint_pose_state";

/// Python `post_rail_accumulator` sub-dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PostRailAccumulator {
    pub force: Option<[f64; 3]>,
    pub torque: Option<[f64; 3]>,
    pub force_before_damping: [f64; 3],
    pub torque_before_damping: [f64; 3],
    pub authority: &'static str,
}

pub const POST_RAIL_AUTHORITY: &str = "static_B29350_then_B29E70_at_0xB261D1_0xB262D3";

/// Python `evaluate_rail_step(...)` result dict. `orientation_integrator` is
/// the ordinary integrator result (non-constraint branch) or the ordinary
/// half of the constraint result (state != 3 constraint branch); it is
/// `None` only on the state-3 constraint branch (Python `.get()`).
#[derive(Debug, Clone, PartialEq)]
pub struct RailStepResult {
    pub position_after: [f64; 3],
    pub velocity_after: [f64; 3],
    pub angular_velocity_after: [f64; 3],
    pub matrix_after: [f64; 9],
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub rail_selector: RailAccumulatorKinds,
    pub velocity_before_damping: [f64; 3],
    pub pose_rail_state: i64,
    pub rail_accumulator: RailAccumulatorProducersResult,
    pub rail_state2_alignment_torque: Option<RailState2AlignmentTorque>,
    pub rail_orientation_steering_torque: Option<RailOrientationSteeringTorque>,
    pub post_rail_accumulator: PostRailAccumulator,
    pub common_damping: CommonDampingResult,
    pub linear_integrator: LinearVelocityIntegratorResult,
    pub angular_integrator: AngularVelocityIntegratorResult,
    pub position_integrator: PositionIntegratorResult,
    pub orientation_integrator: Option<OrientationIntegratorResult>,
    pub rail_constraint_orientation: Option<RailConstraintOrientation>,
    pub rail_orientation_dt: f64,
    pub rail_orientation_dt_authority: &'static str,
    pub ordering: &'static str,
    pub closure_status: &'static str,
}

/// Keyword bundle for `evaluate_rail_step` (Python kwargs).
#[derive(Debug, Clone, PartialEq)]
pub struct RailStepArgs {
    pub position_before: [f64; 3],
    pub velocity_before: [f64; 3],
    pub angular_velocity_before: [f64; 3],
    pub matrix_before: [f64; 9],
    pub force_accumulator_before: [f64; 3],
    pub torque_accumulator_before: [f64; 3],
    pub rail_contact: bool,
    pub rail_state: i64,
    pub gate_1c0_decoded: bool,
    pub positive_a80: f64,
    pub negative_a80: f64,
    pub boost_family_active: bool,
    pub boost_special_gate_9b8: bool,
    pub boost_special_gate_6e4: bool,
    pub attached_object_state: i64,
    pub rail_dir: [f64; 3],
    pub accel_be0: f64,
    pub obf_6e8: f64,
    pub obf_808: f64,
    pub obf_814: f64,
    pub grip_720: f64,
    pub field_604: [f64; 3],
    pub mass: f64,
    pub gravity_be8: f64,
    pub adc: f64,
    pub ae0: f64,
    pub aec: f64,
    pub signed_steering_axis: f64,
    pub drive_tune_718: f64,
    pub post_rail_force_accumulator: Option<[f64; 3]>,
    pub post_rail_torque_accumulator: Option<[f64; 3]>,
    pub inverse_inertia_matrix: [f64; 9],
    pub damping: f64,
    pub directional_damping: f64,
    pub damping_runtime_multiplier: f64,
    pub damping_special_gate: bool,
    pub dt: f64,
    pub numeric_mode: String,
    pub pre_damping_velocity_override: Option<[f64; 3]>,
    pub pose_rail_state: Option<i64>,
    pub rail_base_bf8: Option<[f64; 9]>,
    pub rail_constraint_c1c_before: Option<[f64; 9]>,
    pub rail_orientation_dt: Option<f64>,
}

/// `evaluate_rail_step(...)`. Vector/matrix length checks are type-enforced
/// in Rust.
pub fn evaluate_rail_step(args: &RailStepArgs) -> Result<RailStepResult, String> {
    if args.rail_base_bf8.is_some() != args.rail_constraint_c1c_before.is_some() {
        return Err(
            "rail_base_bf8 and rail_constraint_c1c_before must be supplied together".to_string()
        );
    }
    if !(1..=6).contains(&args.rail_state) {
        return Err("active rail_state must be in 1..6".to_string());
    }
    if !args.rail_contact {
        return Err("rail step requires an active rail contact".to_string());
    }
    if args.mass <= 0.0 || args.dt <= 0.0 {
        return Err("mass and dt must be positive".to_string());
    }
    if let Some(rail_orientation_dt) = args.rail_orientation_dt {
        if rail_orientation_dt <= 0.0 {
            return Err("rail_orientation_dt must be positive".to_string());
        }
    }
    let effective_pose_state = args.pose_rail_state.unwrap_or(args.rail_state);
    if !(1..=6).contains(&effective_pose_state) {
        return Err("pose_rail_state must be in 1..6".to_string());
    }

    let selection = select_rail_accumulator_kinds(
        args.rail_contact,
        args.rail_state,
        args.gate_1c0_decoded,
        args.positive_a80,
        args.negative_a80,
        args.boost_family_active,
        args.boost_special_gate_9b8,
        args.boost_special_gate_6e4,
        args.attached_object_state,
        args.ae0.abs(),
        args.adc,
    );
    let mut state2_alignment: Option<RailState2AlignmentTorque> = None;
    let mut orientation_steering: Option<RailOrientationSteeringTorque> = None;
    let force_before_producers = args.force_accumulator_before;
    let mut torque_before_producers = args.torque_accumulator_before;
    if args.rail_state == 2 {
        let alignment = evaluate_rail_state2_alignment_torque(
            torque_before_producers,
            args.signed_steering_axis,
            args.drive_tune_718,
            args.adc,
            args.ae0,
            args.aec,
            selection.enabled_kinds.contains(&24),
            &args.numeric_mode,
        )?;
        torque_before_producers = alignment.torque_after;
        state2_alignment = Some(alignment);
    } else if args.rail_contact {
        let steering = evaluate_rail_orientation_steering_torque(
            torque_before_producers,
            args.signed_steering_axis,
            args.drive_tune_718,
            &args.numeric_mode,
        )?;
        torque_before_producers = steering.torque_after;
        orientation_steering = Some(steering);
    }
    let producers = evaluate_rail_accumulator_producers(
        &selection.enabled_kinds,
        force_before_producers,
        torque_before_producers,
        args.rail_dir,
        args.positive_a80,
        args.accel_be0,
        args.obf_6e8,
        args.obf_808,
        args.obf_814,
        args.grip_720,
        args.angular_velocity_before,
        args.field_604,
        args.mass,
        args.gravity_be8,
    )?;
    let mut force_before_damping = producers.accumulator_614_after;
    let mut torque_before_damping = producers.accumulator_620_after;
    if let Some(post_rail_force) = args.post_rail_force_accumulator {
        for index in 0..3 {
            force_before_damping[index] += post_rail_force[index];
        }
    }
    if let Some(post_rail_torque) = args.post_rail_torque_accumulator {
        for index in 0..3 {
            torque_before_damping[index] += post_rail_torque[index];
        }
    }
    let velocity_before_damping = args.pre_damping_velocity_override.unwrap_or(args.velocity_before);
    let damped = evaluate_common_damping(
        force_before_damping,
        torque_before_damping,
        velocity_before_damping,
        args.angular_velocity_before,
        args.damping,
        args.damping_special_gate,
        args.directional_damping,
        args.damping_runtime_multiplier,
        None,
        &args.numeric_mode,
    )?;
    let linear = evaluate_linear_velocity_integrator(
        velocity_before_damping,
        damped.accumulator_614_after,
        [0.0, 0.0, 0.0],
        args.mass,
        args.dt,
        None,
        &args.numeric_mode,
    )?;
    let angular = evaluate_angular_velocity_integrator(
        args.angular_velocity_before,
        damped.accumulator_620_after,
        args.inverse_inertia_matrix,
        args.dt,
        &args.numeric_mode,
    )?;
    let position = evaluate_position_integrator(
        args.position_before,
        linear.velocity_c0_after,
        args.dt,
        None,
        0,
        false,
        0.0,
        None,
        None,
    )?;
    let mut rail_constraint_orientation: Option<RailConstraintOrientation> = None;
    let effective_orientation_dt = args.rail_orientation_dt.unwrap_or(args.dt);
    let matrix_after;
    let orientation: Option<OrientationIntegratorResult>;
    if let Some(rail_base_bf8) = args.rail_base_bf8 {
        let constraint = evaluate_rail_constraint_orientation(
            args.matrix_before,
            rail_base_bf8,
            args.rail_constraint_c1c_before.unwrap(),
            angular.angular_velocity_cc_after,
            effective_orientation_dt,
            effective_pose_state,
            args.numeric_mode == "float32",
        )?;
        matrix_after = constraint.primary_matrix_after;
        orientation = constraint.orientation_integrator.clone();
        rail_constraint_orientation = Some(constraint);
    } else {
        let ordinary = evaluate_orientation_integrator(
            args.matrix_before,
            angular.angular_velocity_cc_after,
            args.dt,
            false,
            false,
            effective_pose_state,
            args.numeric_mode == "float32",
        )?;
        matrix_after = ordinary.matrix_9c_after;
        orientation = Some(ordinary);
    }
    let has_constraint = rail_constraint_orientation.is_some();
    Ok(RailStepResult {
        position_after: position.position_90_after,
        velocity_after: linear.velocity_c0_after,
        angular_velocity_after: angular.angular_velocity_cc_after,
        matrix_after,
        force_accumulator: damped.accumulator_614_after,
        torque_accumulator: damped.accumulator_620_after,
        rail_selector: selection,
        velocity_before_damping,
        pose_rail_state: effective_pose_state,
        rail_accumulator: producers,
        rail_state2_alignment_torque: state2_alignment,
        rail_orientation_steering_torque: orientation_steering,
        post_rail_accumulator: PostRailAccumulator {
            force: args.post_rail_force_accumulator,
            torque: args.post_rail_torque_accumulator,
            force_before_damping,
            torque_before_damping,
            authority: POST_RAIL_AUTHORITY,
        },
        common_damping: damped,
        linear_integrator: linear,
        angular_integrator: angular,
        position_integrator: position,
        orientation_integrator: orientation,
        rail_constraint_orientation,
        rail_orientation_dt: effective_orientation_dt,
        rail_orientation_dt_authority: if args.rail_orientation_dt.is_some() {
            "prior_writer_epoch_latch"
        } else {
            "current_dt_compatibility_default"
        },
        ordering: ORDERING,
        closure_status: if has_constraint {
            CLOSURE_WITH_CONSTRAINT
        } else {
            CLOSURE_WITHOUT_CONSTRAINT
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> RailStepArgs {
        RailStepArgs {
            position_before: [0.0; 3],
            velocity_before: [10.0, 0.0, 0.0],
            angular_velocity_before: [0.0; 3],
            matrix_before: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            force_accumulator_before: [0.0; 3],
            torque_accumulator_before: [0.0; 3],
            rail_contact: true,
            rail_state: 3,
            gate_1c0_decoded: false,
            positive_a80: 1.0,
            negative_a80: 0.0,
            boost_family_active: false,
            boost_special_gate_9b8: false,
            boost_special_gate_6e4: false,
            attached_object_state: 0,
            rail_dir: [1.0, 0.0, 0.0],
            accel_be0: 4608.0,
            obf_6e8: 1.0,
            obf_808: 1.0,
            obf_814: 1.0,
            grip_720: 1.0,
            field_604: [0.0, 0.0, -1.0],
            mass: 100.0,
            gravity_be8: 1.0,
            adc: 0.0,
            ae0: 0.0,
            aec: 0.0,
            signed_steering_axis: 0.0,
            drive_tune_718: 10.0,
            post_rail_force_accumulator: None,
            post_rail_torque_accumulator: None,
            inverse_inertia_matrix: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            damping: 0.0,
            directional_damping: 0.0,
            damping_runtime_multiplier: 0.0,
            damping_special_gate: false,
            dt: 0.002,
            numeric_mode: "float32".to_string(),
            pre_damping_velocity_override: None,
            pose_rail_state: None,
            rail_base_bf8: None,
            rail_constraint_c1c_before: None,
            rail_orientation_dt: None,
        }
    }

    #[test]
    fn argument_validation() {
        let mut args = base_args();
        args.rail_contact = false;
        assert_eq!(
            evaluate_rail_step(&args).unwrap_err(),
            "rail step requires an active rail contact"
        );
        let mut args = base_args();
        args.rail_state = 7;
        assert_eq!(
            evaluate_rail_step(&args).unwrap_err(),
            "active rail_state must be in 1..6"
        );
        let mut args = base_args();
        args.rail_base_bf8 = Some([0.0; 9]);
        assert_eq!(
            evaluate_rail_step(&args).unwrap_err(),
            "rail_base_bf8 and rail_constraint_c1c_before must be supplied together"
        );
        let mut args = base_args();
        args.mass = 0.0;
        assert_eq!(evaluate_rail_step(&args).unwrap_err(), "mass and dt must be positive");
    }

    #[test]
    fn state3_drive_positive_chain_float32() {
        // Python authority spot values (numeric_mode="float32"): kinds
        // [22, 25, 26], force x = 4608 * 1, velocity advances by
        // f32(4608/100*0.002), position by velocity*dt.
        let args = base_args();
        let result = evaluate_rail_step(&args).unwrap();
        assert_eq!(result.rail_selector.enabled_kinds, vec![22, 25, 26]);
        assert_eq!(result.pose_rail_state, 3);
        assert!(result.rail_constraint_orientation.is_none());
        assert_eq!(result.closure_status, CLOSURE_WITHOUT_CONSTRAINT);
        // velocity_after[0] = 10 + f32-chain of (4608/100)*0.002.
        assert!((result.velocity_after[0] - 10.09216).abs() < 1e-4);
        assert!(result.position_after[0] > 0.02);
    }

    #[test]
    fn constraint_branch_runs_state3_writer() {
        let mut args = base_args();
        args.rail_base_bf8 = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        args.rail_constraint_c1c_before = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let result = evaluate_rail_step(&args).unwrap();
        let constraint = result.rail_constraint_orientation.unwrap();
        assert_eq!(
            constraint.writer,
            super::super::rail_constraint_orientation::WRITER_STATE3
        );
        assert_eq!(result.closure_status, CLOSURE_WITH_CONSTRAINT);
        assert_eq!(result.rail_orientation_dt_authority, "current_dt_compatibility_default");
    }
}
