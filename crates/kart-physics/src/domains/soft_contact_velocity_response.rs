//! Port of `scripts/evaluate_soft_contact_velocity_response.py` — soft-wall
//! additive direct velocity writer.

/// Python result key `ordering`.
pub const ORDERING: &str = "post-linear soft-wall kind8 source add -> pose integrator";

/// Python result key `runtime_authority`.
pub const RUNTIME_AUTHORITY: &str = "probe_wall_hard kind8 pre/source/post exact additive write";

/// Python `evaluate_soft_contact_velocity_response(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftContactVelocityResponse {
    pub velocity_c0_before: [f64; 3],
    pub velocity_delta: [f64; 3],
    pub velocity_c0_after: [f64; 3],
    pub direct_velocity_add: bool,
    pub ordering: &'static str,
    pub runtime_authority: &'static str,
}

/// `evaluate_soft_contact_velocity_response(...)`. Component-count
/// validation in Python is unreachable here (fixed `[f64; 3]` inputs).
pub fn evaluate_soft_contact_velocity_response(
    velocity_c0_before: [f64; 3],
    velocity_delta: [f64; 3],
) -> SoftContactVelocityResponse {
    let after = [
        velocity_c0_before[0] + velocity_delta[0],
        velocity_c0_before[1] + velocity_delta[1],
        velocity_c0_before[2] + velocity_delta[2],
    ];
    SoftContactVelocityResponse {
        velocity_c0_before,
        velocity_delta,
        velocity_c0_after: after,
        direct_velocity_add: true,
        ordering: ORDERING,
        runtime_authority: RUNTIME_AUTHORITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn additive_write() {
        // Constants computed with the Python authority:
        //   evaluate_soft_contact_velocity_response(
        //     velocity_c0_before=[1.0,-2.0,0.5], velocity_delta=[0.25,0.5,-0.75])
        let r = evaluate_soft_contact_velocity_response([1.0, -2.0, 0.5], [0.25, 0.5, -0.75]);
        assert_eq!(r.velocity_c0_after, [1.25, -1.5, -0.25]);
        assert!(r.direct_velocity_add);
    }
}
