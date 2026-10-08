//! Port of `scripts/evaluate_contact_accumulator_producer.py` — executable
//! four-slot contact accumulator producer from sub_B29E70. Every helper in
//! the Python module is f32-quantized (`_fadd`/`_fsub`/`_fmul`/`_fdiv`);
//! there is no numeric_mode switch.
//!
//! Numeric note: this module's `_dot` is `_fadd(_fadd(m0, m1), m2)` with NO
//! explicit zero start, unlike `num::dot3_f32` (which starts the
//! accumulator at +0.0 and therefore maps a leading -0.0 product to +0.0).
//! The local `dot` below keeps the module's exact chain.

use crate::num::{cross3_f32, f32q, fadd, fdiv, fmul};

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB29E70";
pub const STATIC_FUNCTION_RVA: &str = "0x729E70";
pub const STATIC_FORCE_SLOT_SINK_VA: &str = "0xB29FD9";
pub const STATIC_TORQUE_SLOT_SINK_VA: &str = "0xB2A0AA";
pub const STATIC_TORQUE_SCALAR_PROJECTION_STORE_VA: &str = "0xB29FA5";
pub const STATIC_TORQUE_SCALAR_LOAD_VA: &str = "0xB2A05B";
pub const STATIC_COMMON_FORCE_SINK_VA: &str = "0xB2A0E6";

/// This module's `_add`: elementwise `_fadd`.
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        fadd(a[0], b[0]),
        fadd(a[1], b[1]),
        fadd(a[2], b[2]),
    ]
}

/// This module's `_scale`: elementwise `_fmul`.
fn scale(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [
        fmul(vector[0], scalar),
        fmul(vector[1], scalar),
        fmul(vector[2], scalar),
    ]
}

/// This module's `_dot`: `_fadd(_fadd(m0, m1), m2)` — no zero start (see
/// module doc).
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    fadd(
        fadd(fmul(a[0], b[0]), fmul(a[1], b[1])),
        fmul(a[2], b[2]),
    )
}

/// One entry of the Python `slots` list.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSlot {
    pub slot: i64,
    pub active: bool,
    pub rate_gain: Option<f64>,
    pub raw_force: f64,
    pub alignment: Option<f64>,
    pub normal_force: f64,
    pub force_delta: [f64; 3],
    pub lever: [f64; 3],
    pub torque_delta: [f64; 3],
}

/// Python result dict minus `static_source`. The four-entry slot-array
/// guard ("contact slot arrays must each contain exactly four entries") is
/// compile-time in Rust.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactAccumulatorProducerResult {
    pub accumulator_614_before: [f64; 3],
    pub accumulator_614_after: [f64; 3],
    pub accumulator_620_before: [f64; 3],
    pub accumulator_620_after: [f64; 3],
    pub slots: Vec<ContactSlot>,
    pub common_force_delta: [f64; 3],
}

/// `evaluate_contact_accumulator_producer(...)` — apply exact contact
/// writes in slot order, followed by the common 604 term.
pub fn evaluate_contact_accumulator_producer(
    accumulator_614_before: [f64; 3],
    accumulator_620_before: [f64; 3],
    elapsed_seconds: f64,
    contact_active: [bool; 4],
    compression_a54: [f64; 4],
    compression_rate_a64: [f64; 4],
    contact_vectors_a14: [[f64; 3]; 4],
    contact_normal_ac4: [f64; 3],
    spring_7c0: f64,
    rebound_7c4: f64,
    bump_7c8: f64,
    lever_x_9f4: [f64; 4],
    lever_y_9f8: [f64; 4],
    obf_66c: f64,
    obf_678: f64,
    field_604: [f64; 3],
    obf_660: f64,
) -> Result<ContactAccumulatorProducerResult, String> {
    if elapsed_seconds <= 0.0 {
        return Err("elapsed_seconds must be positive".to_string());
    }
    let mut force = accumulator_614_before;
    let mut torque = accumulator_620_before;
    let mut slots = Vec::with_capacity(4);
    for i in 0..4 {
        let mut raw_force = 0.0;
        if contact_active[i] {
            let rate_gain = if compression_rate_a64[i] > 0.0 {
                rebound_7c4
            } else {
                bump_7c8
            };
            raw_force = fadd(
                fmul(spring_7c0, compression_a54[i]),
                fmul(fdiv(compression_rate_a64[i], elapsed_seconds), rate_gain),
            );
        }

        let mut normal_force = 0.0;
        let mut alignment: Option<f64> = None;
        if raw_force > 0.0 {
            let value = dot(contact_vectors_a14[i], contact_normal_ac4);
            alignment = Some(value);
            normal_force = fmul(value, raw_force);
        }

        let force_delta = scale(contact_normal_ac4, normal_force);
        force = add(force, force_delta);

        // lever=(obf66c*9F4, -obf678*9F8, 0), normal=(0,0,F).
        let lever = [
            fmul(obf_66c, lever_x_9f4[i]),
            fmul(f32q(-obf_678), lever_y_9f8[i]),
            0.0,
        ];
        // B29FA5 overwrites [ebp-0x0C] with the AC4-projected scalar before
        // B2A05B reloads it for the lever cross product.
        let torque_delta = scale(
            cross3_f32(lever, [0.0, 0.0, normal_force]),
            f32q(0.1),
        );
        torque = add(torque, torque_delta);
        slots.push(ContactSlot {
            slot: i as i64,
            active: contact_active[i],
            rate_gain: if contact_active[i] {
                Some(if compression_rate_a64[i] > 0.0 {
                    rebound_7c4
                } else {
                    bump_7c8
                })
            } else {
                None
            },
            raw_force,
            alignment,
            normal_force,
            force_delta,
            lever,
            torque_delta,
        });
    }

    let common_force_delta = scale(field_604, obf_660);
    force = add(force, common_force_delta);
    Ok(ContactAccumulatorProducerResult {
        accumulator_614_before,
        accumulator_614_after: force,
        accumulator_620_before,
        accumulator_620_after: torque,
        slots,
        common_force_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_slots_and_common_term() {
        // Constants computed with the Python authority:
        //   evaluate_contact_accumulator_producer(contact_active=[T,F,T,F],
        //     compression_a54=[0.5,0,-0.1,0], compression_rate_a64=[0.1,0,0.2,0], ...)
        let r = evaluate_contact_accumulator_producer(
            [1.0, 2.0, 3.0],
            [0.5, -0.5, 0.25],
            0.002,
            [true, false, true, false],
            [0.5, 0.0, -0.1, 0.0],
            [0.1, 0.0, 0.2, 0.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            [0.0, 0.0, 1.0],
            100.0,
            20.0,
            10.0,
            [0.5, -0.5, 0.25, -0.25],
            [0.1, 0.2, -0.1, -0.2],
            0.4,
            0.3,
            [0.5, -1.0, 2.0],
            80.0,
        )
        .unwrap();
        assert_eq!(r.accumulator_614_after, [41.0, -78.0, 2153.0]);
        assert_eq!(
            r.accumulator_620_after,
            [6.470000267028809, -20.399999618530273, 0.25]
        );
        // Slot 0: raw force positive but alignment zero -> no force write.
        assert_eq!(r.slots[0].raw_force, 1050.0);
        assert_eq!(r.slots[0].alignment, Some(0.0));
        assert_eq!(r.slots[0].normal_force, 0.0);
        // Slot 1: inactive -> no rate gain, no alignment.
        assert_eq!(r.slots[1].rate_gain, None);
        assert_eq!(r.slots[1].alignment, None);
        // Slot 2: rebound gain, full alignment.
        assert_eq!(r.slots[2].rate_gain, Some(20.0));
        assert_eq!(r.slots[2].raw_force, 1990.0);
        assert_eq!(r.slots[2].force_delta, [0.0, 0.0, 1990.0]);
        assert_eq!(
            r.slots[2].torque_delta,
            [5.970000267028809, -19.899999618530273, 0.0]
        );
        assert_eq!(r.common_force_delta, [40.0, -80.0, 160.0]);
    }

    #[test]
    fn error_guard_matches_python_message() {
        assert_eq!(
            evaluate_contact_accumulator_producer(
                [0.0; 3],
                [0.0; 3],
                0.0,
                [false; 4],
                [0.0; 4],
                [0.0; 4],
                [[0.0; 3]; 4],
                [0.0; 3],
                100.0,
                20.0,
                10.0,
                [0.0; 4],
                [0.0; 4],
                0.4,
                0.3,
                [0.0; 3],
                80.0,
            )
            .unwrap_err(),
            "elapsed_seconds must be positive"
        );
    }
}
