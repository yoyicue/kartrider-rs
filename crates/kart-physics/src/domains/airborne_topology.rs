//! Port of `scripts/evaluate_airborne_topology.py` — executable takeoff,
//! airborne integration, and landing physics composition.
//!
//! `evaluate_airborne_step` advances one unconstrained 6-DoF airborne substep
//! from the exact producers: `sub_B29E70` adds `field_604 * mass` after its
//! four contact slots even when every slot is inactive (AIR therefore retains
//! the common gravity producer but contributes no suspension force/torque),
//! `sub_B2D5D0` applies the ordinary linear/angular damping, and both halves
//! of `sub_B2D940` integrate velocity before `sub_B2FD80` writes the pose.

use super::angular_velocity_integrator::{
    evaluate_angular_velocity_integrator, AngularVelocityIntegratorResult,
};
use super::common_damping::{evaluate_common_damping, CommonDampingResult};
use super::contact_accumulator_producer::{
    evaluate_contact_accumulator_producer, ContactAccumulatorProducerResult,
};
use super::hard_contact_velocity_response::{
    evaluate_hard_contact_velocity_response, HardContactVelocityResponse,
};
use super::linear_velocity_integrator::{
    evaluate_linear_velocity_integrator, LinearVelocityIntegratorResult,
};
use super::orientation_integrator::{
    evaluate_orientation_integrator, OrientationIntegratorResult,
};
use super::position_integrator::{evaluate_position_integrator, PositionIntegratorResult};

/// Python `evaluate_takeoff_or_landing_contact(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct TakeoffOrLandingContact {
    pub role: String,
    pub velocity_before: [f64; 3],
    pub velocity_after: [f64; 3],
    pub hit_normal: [f64; 3],
    pub basis_ad0: [f64; 3],
    pub hard_contact_response: HardContactVelocityResponse,
    pub topology_edge: &'static str,
    pub evidence_scope: &'static str,
}

/// `evaluate_takeoff_or_landing_contact(...)` — apply the exact B2DC80
/// contact velocity replacement at a topology edge. Component-count
/// validation in Python is unreachable here (fixed `[f64; 3]` inputs);
/// Python's default `restitution=0.7` is passed explicitly.
pub fn evaluate_takeoff_or_landing_contact(
    velocity_before: [f64; 3],
    hit_normal: [f64; 3],
    basis_ad0: [f64; 3],
    role: &str,
) -> Result<TakeoffOrLandingContact, String> {
    if role != "takeoff" && role != "landing" {
        return Err("role must be takeoff or landing".to_string());
    }
    let response = evaluate_hard_contact_velocity_response(
        velocity_before,
        hit_normal,
        basis_ad0,
        0.7,
    );
    Ok(TakeoffOrLandingContact {
        role: role.to_string(),
        velocity_before,
        velocity_after: response.velocity_c0_after,
        hit_normal,
        basis_ad0,
        hard_contact_response: response,
        topology_edge: if role == "takeoff" {
            "GND->AIR-ENTRY"
        } else {
            "AIR->LAND"
        },
        evidence_scope: "desert_I08 sampled takeoff and landing pairs",
    })
}

/// Python `evaluate_airborne_step(...)` result dict minus the static
/// `ordering`/`closure_status` documentation keys.
#[derive(Debug, Clone, PartialEq)]
pub struct AirborneStep {
    pub position_before: [f64; 3],
    pub position_after: [f64; 3],
    pub velocity_before: [f64; 3],
    pub velocity_after: [f64; 3],
    pub angular_velocity_before: [f64; 3],
    pub angular_velocity_after: [f64; 3],
    pub matrix_before: [f64; 9],
    pub matrix_after: [f64; 9],
    pub gravity_per_mass: [f64; 3],
    pub additional_force_world: [f64; 3],
    pub additional_torque_world: [f64; 3],
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub contact_common_gravity: ContactAccumulatorProducerResult,
    pub common_damping: CommonDampingResult,
    pub linear_integrator: LinearVelocityIntegratorResult,
    pub angular_integrator: AngularVelocityIntegratorResult,
    pub position_integrator: PositionIntegratorResult,
    pub orientation_integrator: OrientationIntegratorResult,
    pub rail_state_1cc: i64,
}

/// `evaluate_airborne_step(...)` — advance one unconstrained 6-DoF airborne
/// substep from exact producers. Component-count validation in Python is
/// unreachable here (fixed-size inputs).
#[allow(clippy::too_many_arguments)]
pub fn evaluate_airborne_step(
    position_before: [f64; 3],
    velocity_before: [f64; 3],
    angular_velocity_before: [f64; 3],
    matrix_before: [f64; 9],
    mass: f64,
    dt: f64,
    gravity_per_mass: [f64; 3],
    inverse_inertia_matrix: [f64; 9],
    damping: f64,
    directional_damping: f64,
    damping_runtime_multiplier: f64,
    damping_special_gate: bool,
    additional_force_world: Option<[f64; 3]>,
    additional_torque_world: Option<[f64; 3]>,
    rail_state_1cc: i64,
    numeric_mode: &str,
) -> Result<AirborneStep, String> {
    let additional_force_world = additional_force_world.unwrap_or([0.0, 0.0, 0.0]);
    let additional_torque_world = additional_torque_world.unwrap_or([0.0, 0.0, 0.0]);
    if mass <= 0.0 || dt <= 0.0 {
        return Err("mass and dt must be positive".to_string());
    }
    if rail_state_1cc != 0 && rail_state_1cc != 4 && rail_state_1cc != 6 {
        return Err("airborne rail_state_1cc must be 0, 4, or 6".to_string());
    }

    let contact = evaluate_contact_accumulator_producer(
        additional_force_world,
        additional_torque_world,
        dt,
        [false; 4],
        [0.0; 4],
        [0.0; 4],
        [[0.0, 0.0, 0.0]; 4],
        [0.0, 0.0, 1.0],
        0.0,
        0.0,
        0.0,
        [0.0; 4],
        [0.0; 4],
        0.0,
        0.0,
        gravity_per_mass,
        mass,
    )?;
    let damping_result = evaluate_common_damping(
        contact.accumulator_614_after,
        contact.accumulator_620_after,
        velocity_before,
        angular_velocity_before,
        damping,
        damping_special_gate,
        directional_damping,
        damping_runtime_multiplier,
        None,
        numeric_mode,
    )?;
    let linear = evaluate_linear_velocity_integrator(
        velocity_before,
        damping_result.accumulator_614_after,
        [0.0, 0.0, 0.0],
        mass,
        dt,
        None,
        numeric_mode,
    )?;
    let angular = evaluate_angular_velocity_integrator(
        angular_velocity_before,
        damping_result.accumulator_620_after,
        inverse_inertia_matrix,
        dt,
        numeric_mode,
    )?;
    let position = evaluate_position_integrator(
        position_before,
        linear.velocity_c0_after,
        dt,
        None,
        0,
        false,
        0.0,
        None,
        None,
    )?;
    let orientation = evaluate_orientation_integrator(
        matrix_before,
        angular.angular_velocity_cc_after,
        dt,
        false,
        false,
        rail_state_1cc,
        numeric_mode == "float32",
    )?;
    Ok(AirborneStep {
        position_before,
        position_after: position.position_90_after,
        velocity_before,
        velocity_after: linear.velocity_c0_after,
        angular_velocity_before,
        angular_velocity_after: angular.angular_velocity_cc_after,
        matrix_before,
        matrix_after: orientation.matrix_9c_after,
        gravity_per_mass,
        additional_force_world,
        additional_torque_world,
        force_accumulator: damping_result.accumulator_614_after,
        torque_accumulator: damping_result.accumulator_620_after,
        contact_common_gravity: contact,
        common_damping: damping_result,
        linear_integrator: linear,
        angular_integrator: angular,
        position_integrator: position,
        orientation_integrator: orientation,
        rail_state_1cc,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn takeoff_contact_matches_python() {
        // Constants computed with the Python authority:
        //   evaluate_takeoff_or_landing_contact(
        //     velocity_before=[0.0, 0.0, -5.0], hit_normal=[0.0, 0.0, 1.0],
        //     basis_ad0=[0.0, 1.0, 0.0], role="takeoff")
        let r = evaluate_takeoff_or_landing_contact(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            "takeoff",
        )
        .unwrap();
        assert_eq!(r.topology_edge, "GND->AIR-ENTRY");
        // The tangent component is zero and basis_ad0 is orthogonal to the
        // normal (reflected_scale == -0.0), so the replacement is exact zero.
        assert_eq!(r.velocity_after, [0.0, 0.0, 0.0]);
        assert_eq!(
            evaluate_takeoff_or_landing_contact(
                [0.0; 3],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                "hop",
            )
            .unwrap_err(),
            "role must be takeoff or landing"
        );
    }

    #[test]
    fn airborne_step_free_fall_float32() {
        // Constants computed with the Python authority
        // (scripts/evaluate_airborne_topology.py):
        //   evaluate_airborne_step(
        //     position_before=[0,0,10], velocity_before=[0,0,0],
        //     angular_velocity_before=[0,0,0],
        //     matrix_before=[1,0,0,0,1,0,0,0,1],
        //     mass=100.0, dt=0.002, gravity_per_mass=[0,0,-58.8000031],
        //     inverse_inertia_matrix=[0.119999997,0,0,0,0.119999997,0,0,0,0.119999997],
        //     damping=3.0, directional_damping=0.666999996,
        //     damping_runtime_multiplier=1.0, numeric_mode="float32")
        let r = evaluate_airborne_step(
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            100.0,
            0.002,
            [0.0, 0.0, -58.8000031],
            [0.119999997, 0.0, 0.0, 0.0, 0.119999997, 0.0, 0.0, 0.0, 0.119999997],
            3.0,
            0.666999996,
            1.0,
            false,
            None,
            None,
            0,
            "float32",
        )
        .unwrap();
        assert_eq!(r.velocity_after[2].to_bits(), 0xbfbe_1b08_c000_0000);
        assert_eq!(r.position_after[2].to_bits(), 0x4023_ffe1_2bff_3b64);
        assert_eq!(r.force_accumulator[2].to_bits(), 0xc0b6_f800_2000_0000);
        assert_eq!(r.torque_accumulator, [0.0, 0.0, 0.0]);
        assert_eq!(
            evaluate_airborne_step(
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                0.0,
                0.002,
                [0.0, 0.0, -58.8],
                [0.12, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.12],
                3.0,
                0.0,
                0.0,
                false,
                None,
                None,
                0,
                "float32",
            )
            .unwrap_err(),
            "mass and dt must be positive"
        );
        assert_eq!(
            evaluate_airborne_step(
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                100.0,
                0.002,
                [0.0, 0.0, -58.8],
                [0.12, 0.0, 0.0, 0.0, 0.12, 0.0, 0.0, 0.0, 0.12],
                3.0,
                0.0,
                0.0,
                false,
                None,
                None,
                5,
                "float32",
            )
            .unwrap_err(),
            "airborne rail_state_1cc must be 0, 4, or 6"
        );
    }
}
