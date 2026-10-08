//! Port of `scripts/evaluate_collision_response_selector.py` — exact
//! hard/soft branch selector in GoPlayKart_UpdateWallCollisionResponse.

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionResponseSelector {
    pub branch: &'static str,
    pub normal_z: f64,
    pub arg_threshold: f64,
    pub e38: bool,
    pub e4c: bool,
    pub static_condition: &'static str,
}

/// `evaluate_collision_response_selector(...)`. The three-component
/// `hit_normal` length check is type-enforced in Rust.
pub fn evaluate_collision_response_selector(
    hit_normal: [f64; 3],
    arg_threshold: f64,
    e38: bool,
    e4c: bool,
) -> CollisionResponseSelector {
    let normal_z = hit_normal[2];
    let branch = if normal_z > arg_threshold || e38 || e4c {
        "hard_contact"
    } else {
        "soft_contact"
    };
    CollisionResponseSelector {
        branch,
        normal_z,
        arg_threshold,
        e38,
        e4c,
        static_condition: "hit_normal.z > arg_threshold || E38 != 0 || E4C != 0",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branch_matches_python() {
        // Constants from the Python authority:
        //   evaluate_collision_response_selector(hit_normal=[0, 0, 1])
        //     -> branch "hard_contact", normal_z 1.0
        //   evaluate_collision_response_selector(hit_normal=[0, 1, 0])
        //     -> branch "soft_contact"
        //   evaluate_collision_response_selector(hit_normal=[0, 1, 0], e38=True)
        //     -> branch "hard_contact"
        let hard = evaluate_collision_response_selector([0.0, 0.0, 1.0], 0.65, false, false);
        assert_eq!(hard.branch, "hard_contact");
        assert_eq!(hard.normal_z, 1.0);
        let soft = evaluate_collision_response_selector([0.0, 1.0, 0.0], 0.65, false, false);
        assert_eq!(soft.branch, "soft_contact");
        let override_e38 =
            evaluate_collision_response_selector([0.0, 1.0, 0.0], 0.65, true, false);
        assert_eq!(override_e38.branch, "hard_contact");
        let override_e4c =
            evaluate_collision_response_selector([0.0, 1.0, 0.0], 0.65, false, true);
        assert_eq!(override_e4c.branch, "hard_contact");
        // Boundary: normal_z == threshold is soft (strict >).
        let edge = evaluate_collision_response_selector([0.0, 0.0, 0.65], 0.65, false, false);
        assert_eq!(edge.branch, "soft_contact");
    }
}
