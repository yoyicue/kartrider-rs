//! Port of `scripts/evaluate_soft_contact_angular_response.py` — exact
//! kind9/kind10 soft-contact angular source from sub_B2DC80.
//!
//! Numeric discipline: the float32 dot is the module-local `_dot`
//! (`_fadd(_fadd(_fmul(a0,b0), _fmul(a1,b1)), _fmul(a2,b2))`) — it starts
//! directly from the first product with NO leading `+0.0` accumulator, so it
//! differs from [`crate::num::dot3_f32`] on all-`-0.0` product lists
//! (`fadd(0.0, -0.0) == +0.0`). The float64 dot is CPython `sum()`
//! (Neumaier, [`crate::num::dot3_sum`]). `min`/`max` are CPython two-operand
//! semantics ([`crate::num::min2_f64`]/[`crate::num::max2_f64`]).

use crate::num::{dot3_sum, fadd, fmul, max2_f64, min2_f64};

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2DC80";
pub const STATIC_KIND9_RANGE_VA: &str = "0xB2E768-0xB2E890";
pub const STATIC_KIND10_RANGE_VA: &str = "0xB2E89A-0xB2E9C3";

/// Python result key `writeback_gate`.
pub const WRITEBACK_GATE: &str = "dot(angular_velocity_cc, source) < 1.0";

/// Module-local `_dot(a, b, numeric_mode=...)`: float32 chains `_fadd` from
/// the first `_fmul` product (no zero start); float64 is CPython `sum()`.
fn dot(a: [f64; 3], b: [f64; 3], float32_mode: bool) -> f64 {
    if float32_mode {
        fadd(
            fadd(fmul(a[0], b[0]), fmul(a[1], b[1])),
            fmul(a[2], b[2]),
        )
    } else {
        dot3_sum(a, b)
    }
}

/// Python `evaluate_soft_contact_angular_source(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftContactAngularSource {
    pub kind: i64,
    pub source: [f64; 3],
    pub writeback_selected: bool,
    pub gate_dot: f64,
    pub writeback_gate: &'static str,
    pub normal_ad0: f64,
    pub normal_ab8: f64,
    pub normal_velocity: f64,
    pub scale: f64,
    pub numeric_mode: &'static str,
}

/// `evaluate_soft_contact_angular_source(...)`. Component-count and
/// numeric-mode validation in Python are unreachable here (fixed `[f64; 3]`
/// inputs and a boolean mode switch).
pub fn evaluate_soft_contact_angular_source(
    velocity_c0: [f64; 3],
    angular_velocity_cc: [f64; 3],
    hit_normal: [f64; 3],
    basis_ad0: [f64; 3],
    basis_ab8: [f64; 3],
    float32_mode: bool,
) -> SoftContactAngularSource {
    let normal_ad0 = dot(hit_normal, basis_ad0, float32_mode);
    let normal_ab8 = dot(hit_normal, basis_ab8, float32_mode);
    let normal_velocity = dot(hit_normal, velocity_c0, float32_mode);
    let scale = if float32_mode {
        max2_f64(1.0, min2_f64(fmul(normal_velocity.abs(), 0.5), 30.0))
    } else {
        max2_f64(1.0, min2_f64(normal_velocity.abs() * 0.5, 30.0))
    };
    let weighted_ad0 = if float32_mode {
        fmul(normal_ad0, 0.8)
    } else {
        normal_ad0 * 0.8
    };
    let kind;
    let z;
    if weighted_ad0.abs() > normal_ab8.abs() {
        kind = 9;
        let sign = if normal_ad0 > 0.0 { 1.0 } else { -1.0 };
        z = if float32_mode {
            fmul(fmul(normal_ab8, sign), scale)
        } else {
            normal_ab8 * sign * scale
        };
    } else {
        kind = 10;
        let sign = if normal_ab8 > 0.0 { -1.0 } else { 1.0 };
        z = if float32_mode {
            fmul(fmul(normal_ad0, sign), scale)
        } else {
            normal_ad0 * sign * scale
        };
    }
    let source = [0.0, 0.0, z];
    // sub_B2DC80 only enters the selected kind9/kind10 Vec3Add when the
    // candidate does not increase the angular/source projection past 1.0.
    // This gate separates all 224 paired kind8 contacts in the abyss_I03
    // 2026-07-21 run (151 writes, 73 skips), including the first wall hit.
    let gate_dot = dot(angular_velocity_cc, source, float32_mode);
    let writeback_selected = gate_dot < 1.0;
    SoftContactAngularSource {
        kind,
        source,
        writeback_selected,
        gate_dot,
        writeback_gate: WRITEBACK_GATE,
        normal_ad0,
        normal_ab8,
        normal_velocity,
        scale,
        numeric_mode: if float32_mode { "float32" } else { "float64" },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind9_selected_when_ad0_dominates() {
        // Constants computed with the Python authority:
        //   evaluate_soft_contact_angular_source(
        //     velocity_c0=[-3.0,1.0,0.0], angular_velocity_cc=[0.0,0.0,0.0],
        //     hit_normal=[1.0,0.0,0.0], basis_ad0=[0.0,-1.0,0.0],
        //     basis_ab8=[0.0,0.0,1.0], numeric_mode="float64")
        let r = evaluate_soft_contact_angular_source(
            [-3.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            false,
        );
        assert_eq!(r.kind, 10);
        assert_eq!(r.normal_ad0, 0.0);
        assert_eq!(r.normal_ab8, 0.0);
        assert_eq!(r.normal_velocity, -3.0);
        assert_eq!(r.scale, 1.5);
        assert_eq!(r.source, [0.0, 0.0, 0.0]);
        assert!(r.writeback_selected);
        assert_eq!(r.gate_dot, 0.0);
    }

    #[test]
    fn kind9_sign_and_scale() {
        // Constants computed with the Python authority:
        //   evaluate_soft_contact_angular_source(
        //     velocity_c0=[-10.0,0.0,0.0], angular_velocity_cc=[0.0,0.0,0.0],
        //     hit_normal=[0.6,0.8,0.0], basis_ad0=[0.0,1.0,0.0],
        //     basis_ab8=[1.0,0.0,0.0], numeric_mode="float64")
        let r = evaluate_soft_contact_angular_source(
            [-10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            false,
        );
        assert_eq!(r.kind, 9);
        assert_eq!(r.normal_ad0, 0.8);
        assert_eq!(r.normal_ab8, 0.6);
        assert_eq!(r.normal_velocity, -6.0);
        assert_eq!(r.scale, 3.0);
        assert_eq!(r.source[2], 1.7999999999999998);
        assert!(r.writeback_selected);
        assert_eq!(r.gate_dot, 0.0);
    }

    #[test]
    fn writeback_gate_blocks_at_one() {
        // gate_dot = angular_velocity_cc . source >= 1.0 -> no writeback.
        //   evaluate_soft_contact_angular_source(
        //     velocity_c0=[-10.0,0.0,0.0], angular_velocity_cc=[0.0,0.0,2.0],
        //     hit_normal=[0.6,0.8,0.0], basis_ad0=[0.0,1.0,0.0],
        //     basis_ab8=[1.0,0.0,0.0], numeric_mode="float64")
        let r = evaluate_soft_contact_angular_source(
            [-10.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.6, 0.8, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            false,
        );
        assert_eq!(r.kind, 9);
        assert_eq!(r.source[2], 1.7999999999999998);
        assert!(!r.writeback_selected);
        assert_eq!(r.gate_dot, 3.5999999999999996);
    }

    #[test]
    fn float32_dot_has_no_leading_zero_accumulator() {
        // Module-local float32 `_dot` starts from the first product: with all
        // products -0.0 it yields -0.0 where dot3_f32 (leading +0.0) yields
        // +0.0.
        assert!(dot([-0.0, -1.0, -1.0], [1.0, 0.0, 0.0], true).is_sign_negative());
        assert!(!crate::num::dot3_f32([-0.0, -1.0, -1.0], [1.0, 0.0, 0.0]).is_sign_negative());
    }
}
