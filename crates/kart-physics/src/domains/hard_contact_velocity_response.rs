//! Port of `scripts/evaluate_hard_contact_velocity_response.py` — exact
//! hard-contact direct velocity replacement used by walls/jump contacts.

use crate::num::dot3_sum;

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2DC80";
pub const STATIC_FUNCTION_RVA: &str = "0x72DC80";
pub const STATIC_VELOCITY_WRITE_VAS: [&str; 3] = ["0xB2DF38", "0xB2DF3D", "0xB2DF43"];
pub const STATIC_RESTITUTION_GLOBAL_VA: &str = "0x11ED2FC";

/// Python result key `runtime_evidence`.
pub const RUNTIME_EVIDENCE: &str = "jump takeoff/landing exact pairs; wall_hard.post equals a later-inner-iteration kind40.pre within the same UpdateTick call";

/// Python `evaluate_hard_contact_velocity_response(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct HardContactVelocityResponse {
    pub velocity_c0_before: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub normal_dot_velocity: f64,
    pub normal_component: [f64; 3],
    pub tangent_velocity: [f64; 3],
    pub basis_ad0_dot_normal: f64,
    pub reflected_scale: f64,
    pub direct_velocity_replacement: bool,
    pub runtime_evidence: &'static str,
}

/// `evaluate_hard_contact_velocity_response(...)`. Component-count
/// validation in Python is unreachable here (fixed `[f64; 3]` inputs).
pub fn evaluate_hard_contact_velocity_response(
    velocity_c0_before: [f64; 3],
    hit_normal: [f64; 3],
    basis_ad0: [f64; 3],
    restitution: f64,
) -> HardContactVelocityResponse {
    let normal_dot_velocity = dot3_sum(velocity_c0_before, hit_normal);
    let normal_component = [
        normal_dot_velocity * hit_normal[0],
        normal_dot_velocity * hit_normal[1],
        normal_dot_velocity * hit_normal[2],
    ];
    let tangent_velocity = [
        velocity_c0_before[0] - normal_component[0],
        velocity_c0_before[1] - normal_component[1],
        velocity_c0_before[2] - normal_component[2],
    ];
    let basis_projection = dot3_sum(basis_ad0, hit_normal);
    let reflected_scale = -basis_projection.abs() * restitution;
    let velocity_after = [
        tangent_velocity[0] + reflected_scale * normal_component[0],
        tangent_velocity[1] + reflected_scale * normal_component[1],
        tangent_velocity[2] + reflected_scale * normal_component[2],
    ];
    HardContactVelocityResponse {
        velocity_c0_before,
        velocity_c0_after: velocity_after,
        normal_dot_velocity,
        normal_component,
        tangent_velocity,
        basis_ad0_dot_normal: basis_projection,
        reflected_scale,
        direct_velocity_replacement: true,
        runtime_evidence: RUNTIME_EVIDENCE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_wall_reflects_normal_component() {
        // Constants computed with the Python authority:
        //   evaluate_hard_contact_velocity_response(
        //     velocity_c0_before=[3.0,1.0,0.0], hit_normal=[1.0,0.0,0.0],
        //     basis_ad0=[1.0,0.0,0.0])
        let r = evaluate_hard_contact_velocity_response(
            [3.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.7,
        );
        assert_eq!(r.normal_dot_velocity, 3.0);
        assert_eq!(r.normal_component, [3.0, 0.0, 0.0]);
        assert_eq!(r.tangent_velocity, [0.0, 1.0, 0.0]);
        assert_eq!(r.basis_ad0_dot_normal, 1.0);
        assert_eq!(r.reflected_scale, -0.7);
        assert_eq!(r.velocity_c0_after, [-2.0999999999999996, 1.0, 0.0]);
        assert!(r.direct_velocity_replacement);
    }

    #[test]
    fn basis_projection_signed_zero_matches_cpython_sum() {
        // Python `_dot` is `sum(a[i] * b[i])`: with products
        // [-0.0, -0.0, -0.0] CPython sum() yields +0.0 (accumulator starts
        // at int 0); a naive fold would keep -0.0.
        let r = evaluate_hard_contact_velocity_response(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.0, -1.0, -1.0],
            0.7,
        );
        assert_eq!(r.basis_ad0_dot_normal.to_bits(), 0.0_f64.to_bits());
        assert!(r.basis_ad0_dot_normal.is_sign_positive());
    }

    #[test]
    fn default_restitution_is_0p7() {
        // Python authority default: restitution=0.7; basis parallel to the
        // normal reflects the normal component.
        let r = evaluate_hard_contact_velocity_response(
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            0.7,
        );
        assert_eq!(r.velocity_c0_after, [0.0, 0.0, 3.5]);
    }
}
