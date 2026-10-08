//! Port of `scripts/evaluate_hard_contact_angular_response.py` — exact
//! hard-contact angular writeback from the GoPlayKart wall response.
//!
//! Numeric discipline: plain f64 throughout (the Python module has no `_f32`
//! helpers); every `_dot` is CPython `sum()` (Neumaier,
//! [`crate::num::dot3_sum`]), `_cross` is the plain left-assoc f64 cross.

use crate::num::dot3_sum;

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2DD00";
pub const STATIC_WRITE_VA: &str = "0xB2E013";

/// Module-local `_cross`: plain f64, fixed operand order.
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Python `evaluate_hard_contact_angular_response(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct HardContactAngularResponse {
    pub angular_velocity_before: [f64; 3],
    pub angular_velocity_after: [f64; 3],
    pub angular_delta: [f64; 3],
    pub damping: f64,
}

/// `evaluate_hard_contact_angular_response(...)`. Component-count
/// validation in Python is unreachable here (fixed `[f64; 3]` inputs).
pub fn evaluate_hard_contact_angular_response(
    angular_velocity_before: [f64; 3],
    hit_normal: [f64; 3],
    basis_ac4: [f64; 3],
    basis_ab8: [f64; 3],
    basis_ad0: [f64; 3],
    damping: f64,
) -> HardContactAngularResponse {
    let delta_x = -damping * dot3_sum(cross3(hit_normal, basis_ac4), basis_ab8);
    let delta_y = -damping * dot3_sum(cross3(hit_normal, basis_ad0), basis_ac4);
    let after = [
        angular_velocity_before[0] + delta_x,
        angular_velocity_before[1] + delta_y,
        angular_velocity_before[2],
    ];
    HardContactAngularResponse {
        angular_velocity_before,
        angular_velocity_after: after,
        angular_delta: [delta_x, delta_y, 0.0],
        damping,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn head_on_wall_damps_roll_and_pitch() {
        // Constants computed with the Python authority:
        //   evaluate_hard_contact_angular_response(
        //     angular_velocity_before=[0.5,-0.25,1.0], hit_normal=[1.0,0.0,0.0],
        //     basis_ac4=[0.0,0.0,1.0], basis_ab8=[0.0,1.0,0.0],
        //     basis_ad0=[0.0,-1.0,0.0])
        let r = evaluate_hard_contact_angular_response(
            [0.5, -0.25, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            0.1,
        );
        assert_eq!(r.angular_delta, [0.1, 0.1, 0.0]);
        assert_eq!(r.angular_velocity_after, [0.6, -0.15, 1.0]);
        assert_eq!(r.damping, 0.1);
    }

    #[test]
    fn z_component_is_never_written() {
        // Constants computed with the Python authority:
        //   evaluate_hard_contact_angular_response(
        //     angular_velocity_before=[0.0,0.0,3.0], hit_normal=[0.0,1.0,0.0],
        //     basis_ac4=[1.0,0.0,0.0], basis_ab8=[0.0,0.0,1.0],
        //     basis_ad0=[0.0,0.0,-1.0])
        let r = evaluate_hard_contact_angular_response(
            [0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            0.1,
        );
        assert_eq!(r.angular_delta, [0.1, 0.1, 0.0]);
        assert_eq!(r.angular_velocity_after, [0.1, 0.1, 3.0]);
    }
}
