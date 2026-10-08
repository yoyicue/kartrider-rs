//! Port of `scripts/evaluate_tune_boost_gauge.py` — executable static
//! formula for the GoPlayKart -> BoostGauge bridge.

use crate::domains::booster_resource::{
    add_current_manager_item, validate_current_manager_slots, InventorySlot,
    CURRENT_MANAGER_EMPTY_ITEM_ID,
};
use crate::num::{f32q, fadd, fdiv, fmul, max2_f64, min2_f64};

/// Python module constant `BCHARGE_SURFACE_INCREMENT`.
pub const BCHARGE_SURFACE_INCREMENT: f64 = 1.600000023841858;

/// Python result key `dynamics_role` on `evaluate_bcharge_surface_step`.
pub const DYNAMICS_ROLE: &str = "none_proven_booster_acquisition_gauge_only";

/// Python result key `scheduler_phase`.
pub const SCHEDULER_PHASE: &str = "post_ground_and_contact_updates_pre_tick_cleanup";

pub const CURRENT_ITEM6_GRANT_RUNTIME_ORDER: &str = "C5759D committed-B10 equality -> C575C6 window reset -> C575CE B10-cap -> C575F4 fixed-slot item6 grant";
pub const CURRENT_ITEM6_GRANT_AUTHORITY: &str = "run23_gate_and_same_tick_slot1_grant";

/// Python `model["boost_gauge_producers"]`: dynamic operands plus the
/// current-family tune fields. Gauge state itself remains engine-owned.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct CurrentFamilyGaugeProducersArgs {
    pub airborne: bool,
    pub contact_188: bool,
    pub selector_e8c: bool,
    pub state_1a8: bool,
    pub pending_b1c: bool,
    pub window_fba: bool,
    pub window_fbb: bool,
    pub plain_adc: f64,
    pub field_ae0: f64,
    pub speed_144: f64,
    pub threshold_54c: f64,
    pub tune_6b4: f64,
    pub tune_100c: f64,
    pub tune_988: f64,
    pub tune_994: f64,
    pub raw_760: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeProducerState {
    pub b10: f64,
    pub b20: f64,
    pub b2c: f64,
    pub b38: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingGaugeProducerResult {
    pub applied: bool,
    pub reason: &'static str,
    pub state_before: GaugeProducerState,
    pub state_after: GaugeProducerState,
    pub progress: f64,
    pub delta: f64,
    pub branch: Option<&'static str>,
    pub numeric_mode: &'static str,
    pub authority: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrimaryGaugeProducerResult {
    pub applied: bool,
    pub reason: &'static str,
    pub state_before: GaugeProducerState,
    pub state_after: GaugeProducerState,
    pub rate: f64,
    pub delta: f64,
    pub cap: f64,
    pub primary_f09: bool,
    pub numeric_mode: &'static str,
    pub authority: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurrentFamilyGaugeProducerTransition {
    pub pending_b2c: PendingGaugeProducerResult,
    pub primary_b10_b20: PrimaryGaugeProducerResult,
    pub state_after_producers: GaugeProducerState,
    pub ordering: &'static str,
    pub authority: &'static str,
}

fn store(value: f64, numeric_mode: &str) -> Result<f64, String> {
    match numeric_mode {
        "float32" => Ok(f32q(value)),
        "float64" => Ok(value),
        _ => Err("numeric_mode must be float32 or float64".to_string()),
    }
}

fn add(left: f64, right: f64, numeric_mode: &str) -> f64 {
    if numeric_mode == "float32" {
        fadd(left, right)
    } else {
        left + right
    }
}

fn mul(left: f64, right: f64, numeric_mode: &str) -> f64 {
    if numeric_mode == "float32" {
        fmul(left, right)
    } else {
        left * right
    }
}

fn div(left: f64, right: f64, numeric_mode: &str) -> f64 {
    if numeric_mode == "float32" {
        fdiv(left, right)
    } else {
        left / right
    }
}

fn validate_finite(operands: &[(&str, f64)]) -> Result<(), String> {
    for (name, value) in operands {
        if !value.is_finite() {
            return Err(format!("{name} must be finite"));
        }
    }
    Ok(())
}

/// Rust mirror of Python
/// `evaluate_current_family_pending_gauge_producer` / fixcrt35 B26E00.
pub fn evaluate_current_family_pending_gauge_producer(
    args: &CurrentFamilyGaugeProducersArgs,
    state: GaugeProducerState,
    dt: f64,
    numeric_mode: &str,
) -> Result<PendingGaugeProducerResult, String> {
    validate_finite(&[
        ("dt", dt),
        ("plain_adc", args.plain_adc),
        ("field_ae0", args.field_ae0),
        ("tune_994", args.tune_994),
        ("accum_b10", state.b10),
        ("progress_b20", state.b20),
        ("accum_b2c", state.b2c),
        ("history_b38", state.b38),
    ])?;
    let before = GaugeProducerState {
        b10: store(state.b10, numeric_mode)?,
        b20: store(state.b20, numeric_mode)?,
        b2c: store(state.b2c, numeric_mode)?,
        b38: store(state.b38, numeric_mode)?,
    };
    let zero = store(0.0, numeric_mode)?;
    let gate = (args.airborne || args.contact_188)
        && !args.selector_e8c
        && args.pending_b1c
        && store(args.plain_adc, numeric_mode)? >= zero;
    if !gate {
        return Ok(PendingGaugeProducerResult {
            applied: false,
            reason: "pending_gate_false",
            state_before: before,
            state_after: before,
            progress: zero,
            delta: zero,
            branch: None,
            numeric_mode: if numeric_mode == "float32" {
                "float32"
            } else {
                "float64"
            },
            authority: None,
        });
    }
    let mut progress = mul(
        mul(args.field_ae0, dt, numeric_mode),
        args.field_ae0,
        numeric_mode,
    );
    if args.airborne {
        progress = mul(progress, 2.0, numeric_mode);
    }
    let b20_after = add(before.b20, dt, numeric_mode);
    let (mut delta, mut branch) = if b20_after < store(0.2, numeric_mode)? {
        (mul(3.0, progress, numeric_mode), "b20_lt_0_2")
    } else if b20_after < store(0.5, numeric_mode)? {
        (mul(1.5, progress, numeric_mode), "b20_lt_0_5")
    } else {
        (
            div(
                progress,
                mul(2.0, b20_after, numeric_mode),
                numeric_mode,
            ),
            "b20_ge_0_5",
        )
    };
    if args.window_fba && args.window_fbb {
        delta = mul(args.tune_994, delta, numeric_mode);
        branch = match branch {
            "b20_lt_0_2" => "b20_lt_0_2_state34_factor",
            "b20_lt_0_5" => "b20_lt_0_5_state34_factor",
            _ => "b20_ge_0_5_state34_factor",
        };
    }
    let after = GaugeProducerState {
        b10: before.b10,
        b20: b20_after,
        b2c: add(before.b2c, delta, numeric_mode),
        b38: before.b38,
    };
    Ok(PendingGaugeProducerResult {
        applied: true,
        reason: "pending_gate_true",
        state_before: before,
        state_after: after,
        progress,
        delta,
        branch: Some(branch),
        numeric_mode: if numeric_mode == "float32" {
            "float32"
        } else {
            "float64"
        },
        authority: Some("run25_476_of_476_bit_exact"),
    })
}

/// Rust mirror of Python
/// `evaluate_current_family_primary_gauge_producer` / fixcrt35 B270C0.
pub fn evaluate_current_family_primary_gauge_producer(
    args: &CurrentFamilyGaugeProducersArgs,
    state: GaugeProducerState,
    dt: f64,
    numeric_mode: &str,
) -> Result<PrimaryGaugeProducerResult, String> {
    validate_finite(&[
        ("dt", dt),
        ("speed_144", args.speed_144),
        ("threshold_54c", args.threshold_54c),
        ("tune_6b4", args.tune_6b4),
        ("tune_100c", args.tune_100c),
        ("tune_988", args.tune_988),
        ("raw_760", args.raw_760),
        ("accum_b10", state.b10),
        ("progress_b20", state.b20),
        ("accum_b2c", state.b2c),
        ("history_b38", state.b38),
    ])?;
    let before = GaugeProducerState {
        b10: store(state.b10, numeric_mode)?,
        b20: store(state.b20, numeric_mode)?,
        b2c: store(state.b2c, numeric_mode)?,
        b38: store(state.b38, numeric_mode)?,
    };
    let zero = store(0.0, numeric_mode)?;
    let one = store(1.0, numeric_mode)?;
    let tune_base = store(args.tune_6b4, numeric_mode)?;
    let cap = max2_f64(store(args.raw_760, numeric_mode)?, one);
    let gate = tune_base != zero
        && cap != one
        && (args.airborne || args.contact_188)
        && !args.state_1a8
        && !args.pending_b1c
        && store(args.speed_144, numeric_mode)? >= store(args.threshold_54c, numeric_mode)?;
    if !gate {
        return Ok(PrimaryGaugeProducerResult {
            applied: false,
            reason: "primary_gate_false",
            state_before: before,
            state_after: before,
            rate: zero,
            delta: zero,
            cap,
            primary_f09: false,
            numeric_mode: if numeric_mode == "float32" {
                "float32"
            } else {
                "float64"
            },
            authority: None,
        });
    }
    let override_rate = store(args.tune_100c, numeric_mode)?;
    let mut rate = if override_rate > zero {
        override_rate
    } else {
        tune_base
    };
    if args.window_fba && args.window_fbb {
        rate = add(rate, args.tune_988, numeric_mode);
    }
    let mut delta = mul(dt, rate, numeric_mode);
    if args.airborne {
        delta = mul(delta, 2.0, numeric_mode);
    }
    let added = add(before.b10, delta, numeric_mode);
    let b10_after = if added >= cap { cap } else { added };
    let after = GaugeProducerState {
        b10: b10_after,
        b20: add(before.b20, dt, numeric_mode),
        b2c: before.b2c,
        b38: before.b38,
    };
    Ok(PrimaryGaugeProducerResult {
        applied: true,
        reason: "primary_gate_true",
        state_before: before,
        state_after: after,
        rate,
        delta,
        cap,
        primary_f09: true,
        numeric_mode: if numeric_mode == "float32" {
            "float32"
        } else {
            "float64"
        },
        authority: Some("run25_128_of_128_bit_exact"),
    })
}

// static_source provenance (Python result key `static_source`).
pub const STATIC_STEP_VA: &str = "0xB24DE0";
pub const STATIC_CALL_VA: &str = "0xB26453";
pub const STATIC_CLEANUP_VA: &str = "0xB268FD";
pub const STATIC_GAUGE_FIELD: &str = "GoPlayKart+0xB10";
pub const STATIC_CAP_FIELD: &str = "KartTuneSourceBlock+0x100 (GoPlayKart+0x760)";

#[derive(Debug, Clone, PartialEq)]
pub struct CurrentFamilyGaugeSnapshot {
    pub committed_b10: f64,
    pub pending_b2c: f64,
    pub raw_760: f64,
    pub cap: f64,
    pub accum_clamped: f64,
    pub final_eligible: bool,
}

/// Current fixcrt35 C57450 gate operands.  Each observable store is
/// binary32-quantized exactly where Python calls `_f32`.
pub fn current_family_gauge_snapshot(
    accum_b10: f64,
    accum_b2c: f64,
    raw_760: f64,
) -> Result<CurrentFamilyGaugeSnapshot, String> {
    if !accum_b10.is_finite() || !accum_b2c.is_finite() || !raw_760.is_finite() {
        return Err("current gauge operands must be finite".to_string());
    }
    let committed = f32q(accum_b10);
    let pending = f32q(accum_b2c);
    let raw = f32q(raw_760);
    let cap = f32q(max2_f64(raw, f32q(1.0)));
    let accum = f32q(min2_f64(raw, f32q(committed + pending)));
    Ok(CurrentFamilyGaugeSnapshot {
        committed_b10: committed,
        pending_b2c: pending,
        raw_760: raw,
        cap,
        accum_clamped: accum,
        final_eligible: committed == cap,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct CurrentFamilyItem6Grant {
    pub snapshot: CurrentFamilyGaugeSnapshot,
    pub granted: bool,
    pub reason: &'static str,
    pub selected_slot_index: Option<i64>,
    pub accum_b10_after: f64,
    pub slots_after: Vec<InventorySlot>,
    pub runtime_order: Option<&'static str>,
    pub authority: Option<&'static str>,
}

pub fn evaluate_current_family_item6_grant(
    accum_b10: f64,
    accum_b2c: f64,
    raw_760: f64,
    slots: &[InventorySlot],
) -> Result<CurrentFamilyItem6Grant, String> {
    validate_current_manager_slots(slots)?;
    let snapshot = current_family_gauge_snapshot(accum_b10, accum_b2c, raw_760)?;
    if !snapshot.final_eligible {
        return Ok(CurrentFamilyItem6Grant {
            accum_b10_after: snapshot.committed_b10,
            snapshot,
            granted: false,
            reason: "committed_b10_not_at_cap",
            selected_slot_index: None,
            slots_after: slots.to_vec(),
            runtime_order: None,
            authority: None,
        });
    }
    if slots
        .iter()
        .all(|slot| slot.item_id != CURRENT_MANAGER_EMPTY_ITEM_ID)
    {
        return Ok(CurrentFamilyItem6Grant {
            accum_b10_after: snapshot.committed_b10,
            snapshot,
            granted: false,
            reason: "full_inventory_grant_outcome_unadjudicated",
            selected_slot_index: None,
            slots_after: slots.to_vec(),
            runtime_order: None,
            authority: None,
        });
    }
    let (selected_slot_index, slots_after) = add_current_manager_item(slots, 6, 1)?;
    assert!(
        selected_slot_index >= 0,
        "empty current-manager slot was not selected"
    );
    let accum_b10_after = f32q(snapshot.committed_b10 - snapshot.cap);
    Ok(CurrentFamilyItem6Grant {
        snapshot,
        granted: true,
        reason: "committed_b10_cap_item6_granted",
        selected_slot_index: Some(selected_slot_index),
        accum_b10_after,
        slots_after,
        runtime_order: Some(CURRENT_ITEM6_GRANT_RUNTIME_ORDER),
        authority: Some(CURRENT_ITEM6_GRANT_AUTHORITY),
    })
}

/// `gauge_cap(raw_760)` — CPython two-operand `max` (crate::num::max2_f64).
pub fn gauge_cap(raw_760: f64) -> f64 {
    max2_f64(raw_760, 1.0)
}

/// `gauge_accum_clamped(accum_b10, accum_b2c, raw_760)`.
pub fn gauge_accum_clamped(accum_b10: f64, accum_b2c: f64, raw_760: f64) -> f64 {
    min2_f64(raw_760, accum_b10 + accum_b2c)
}

/// Python `evaluate_bcharge_surface_step(...)` result dict minus
/// `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct BchargeSurfaceStep {
    pub active_before_cleanup: bool,
    pub accum_b10_before: f64,
    pub increment: f64,
    pub accum_b10_after_add: f64,
    pub cap: f64,
    pub accum_b10_after: f64,
    pub active_after_cleanup: bool,
    pub vehicle_force_delta: [f64; 3],
    pub vehicle_torque_delta: [f64; 3],
    pub dynamics_role: &'static str,
    pub scheduler_phase: &'static str,
}

/// `evaluate_bcharge_surface_step(...)` — model B24DE0 followed by
/// UpdateTick's B268FD active-flag clear. Despite its historical IDA name,
/// B24DE0 has no force/torque sink.
pub fn evaluate_bcharge_surface_step(
    active: bool,
    accum_b10: f64,
    raw_760: f64,
) -> BchargeSurfaceStep {
    let cap = gauge_cap(raw_760);
    let before = accum_b10;
    let after_add = if active {
        before + BCHARGE_SURFACE_INCREMENT
    } else {
        before
    };
    let after = if active {
        min2_f64(after_add, cap)
    } else {
        before
    };
    BchargeSurfaceStep {
        active_before_cleanup: active,
        accum_b10_before: before,
        increment: if active {
            BCHARGE_SURFACE_INCREMENT
        } else {
            0.0
        },
        accum_b10_after_add: after_add,
        cap,
        accum_b10_after: after,
        active_after_cleanup: false,
        vehicle_force_delta: [0.0, 0.0, 0.0],
        vehicle_torque_delta: [0.0, 0.0, 0.0],
        dynamics_role: DYNAMICS_ROLE,
        scheduler_phase: SCHEDULER_PHASE,
    }
}

/// Archived visible-suffix helper retained for legacy parity fixtures. It
/// omits the protected AE0-squared prefix and is not the complete producer;
/// engine code uses [`evaluate_current_family_pending_gauge_producer`].
pub fn pending_accum_delta(
    dt: f64,
    progress_b20: f64,
    conditional_multiplier_994: f64,
) -> Option<f64> {
    if !dt.is_finite() || !progress_b20.is_finite() || !conditional_multiplier_994.is_finite() {
        return None;
    }
    let base = if progress_b20 < 0.2 {
        3.0 * dt
    } else if progress_b20 < 0.5 {
        1.5 * dt
    } else if progress_b20 == 0.0 {
        return None;
    } else {
        dt / (2.0 * progress_b20)
    };
    Some(base * conditional_multiplier_994)
}

/// `accumulate_pending(accum_b2c, dt, progress_b20,
/// conditional_multiplier_994=1.0)`.
pub fn accumulate_pending(
    accum_b2c: f64,
    dt: f64,
    progress_b20: f64,
    conditional_multiplier_994: f64,
) -> Option<f64> {
    let delta = pending_accum_delta(dt, progress_b20, conditional_multiplier_994)?;
    if !accum_b2c.is_finite() {
        return None;
    }
    Some(accum_b2c + delta)
}

/// `state34_drift_gauge_event(...)` — model the state-3/4 counters updated
/// by GoPlayKart_SetCurrentState. Returns `(total_count, cycle_count)`.
pub fn state34_drift_gauge_event(
    state: i64,
    suppress_5e8: bool,
    eligible_fba: bool,
    active_fbb: bool,
    total_count: i64,
    cycle_count: i64,
) -> (i64, i64) {
    if suppress_5e8 || !matches!(state, 3 | 4) {
        return (total_count, cycle_count);
    }
    let total_count = total_count + 1;
    let cycle_count = if eligible_fba && !active_fbb {
        cycle_count + 1
    } else {
        cycle_count
    };
    (total_count, cycle_count)
}

/// `state34_drift_gauge_window_can_activate(...)` — static activation gate
/// spanning SetCurrentState and the FBB setter. `total_count // divisor` is
/// CPython floor division; the `divisor > 0`/`total_count > 0` guards make
/// it coincide with truncating division.
pub fn state34_drift_gauge_window_can_activate(
    eligible_fba: bool,
    active_fbb: bool,
    divisor: i64,
    total_count: i64,
    cycle_count: i64,
    activation_count: i64,
    current_tick: i64,
    previous_expiry_tick: i64,
) -> bool {
    eligible_fba
        && !active_fbb
        && divisor > 0
        && total_count > 0
        && total_count / divisor > activation_count
        && current_tick >= previous_expiry_tick
        && cycle_count == divisor
}

/// `state34_drift_gauge_window_expired(...)`.
pub fn state34_drift_gauge_window_expired(
    active_fbb: bool,
    current_tick: i64,
    expiry_tick: i64,
) -> bool {
    active_fbb && current_tick > expiry_tick
}

/// Python `commit_pending(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitPending {
    pub accum_b10: f64,
    pub pending_b1c: i64,
    pub progress_b20: f64,
    pub history_b38: f64,
    pub accum_b2c: f64,
}

/// `commit_pending(accum_b10, accum_b2c, raw_760)` — None when any input is
/// non-finite.
pub fn commit_pending(accum_b10: f64, accum_b2c: f64, raw_760: f64) -> Option<CommitPending> {
    if !accum_b10.is_finite() || !accum_b2c.is_finite() || !raw_760.is_finite() {
        return None;
    }
    Some(CommitPending {
        accum_b10: gauge_accum_clamped(accum_b10, accum_b2c, raw_760),
        pending_b1c: 0,
        progress_b20: 0.0,
        history_b38: accum_b2c,
        accum_b2c: 0.0,
    })
}

/// `gauge_ratio(accum_b10, accum_b2c, raw_760)` — None on non-finite inputs
/// or a zero cap.
pub fn gauge_ratio(accum_b10: f64, accum_b2c: f64, raw_760: f64) -> Option<f64> {
    let cap = gauge_cap(raw_760);
    let accum = gauge_accum_clamped(accum_b10, accum_b2c, raw_760);
    if !accum_b10.is_finite()
        || !accum_b2c.is_finite()
        || !raw_760.is_finite()
        || !cap.is_finite()
        || !accum.is_finite()
        || cap == 0.0
    {
        return None;
    }
    Some(accum / cap)
}

/// `mq_display_target(ratio)` — MqBoostGauge vtable +0x108 scales its input
/// by 0.99.
pub fn mq_display_target(ratio: f64) -> f64 {
    ratio * 0.9900000095367432
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_from_bits(bits: u32) -> f64 {
        f32::from_bits(bits) as f64
    }

    fn f32_bits(value: f64) -> u32 {
        (value as f32).to_bits()
    }

    fn producer_args() -> CurrentFamilyGaugeProducersArgs {
        CurrentFamilyGaugeProducersArgs {
            airborne: false,
            contact_188: true,
            selector_e8c: false,
            state_1a8: false,
            pending_b1c: true,
            window_fba: false,
            window_fbb: false,
            plain_adc: f32_from_bits(0x41B8_9493),
            field_ae0: 0.0,
            speed_144: f32_from_bits(0x4320_1743),
            threshold_54c: f32_from_bits(0x4320_0000),
            tune_6b4: f32_from_bits(0x43AF_0000),
            tune_100c: 0.0,
            tune_988: 0.0,
            tune_994: 0.0,
            raw_760: f32_from_bits(0x4571_4000),
        }
    }

    #[test]
    fn run25_pending_rows_cover_all_three_branches_bit_exactly() {
        let rows = [
            (
                [0x0000_0000, 0x0000_0000, 0x0000_0000, 0x0000_0000],
                [0x0000_0000, 0x3B03_126F, 0x3E0B_790E, 0x0000_0000],
                0xC098_76F5,
                "b20_lt_0_2",
            ),
            (
                [0x0000_0000, 0x3E4B_C6AB, 0x4258_CD96, 0x0000_0000],
                [0x0000_0000, 0x3E4D_D2F5, 0x425B_24A2, 0x0000_0000],
                0xC15F_6DE8,
                "b20_lt_0_5",
            ),
            (
                [0x0000_0000, 0x3EFF_7CFE, 0x4376_C089, 0x0000_0000],
                [0x0000_0000, 0x3F00_4191, 0x4378_0025, 0x0000_0000],
                0xC1C8_13DE,
                "b20_ge_0_5",
            ),
        ];
        for (before, after, ae0, branch) in rows {
            let mut args = producer_args();
            args.field_ae0 = f32_from_bits(ae0);
            let result = evaluate_current_family_pending_gauge_producer(
                &args,
                GaugeProducerState {
                    b10: f32_from_bits(before[0]),
                    b20: f32_from_bits(before[1]),
                    b2c: f32_from_bits(before[2]),
                    b38: f32_from_bits(before[3]),
                },
                f32_from_bits(0x3B03_126F),
                "float32",
            )
            .unwrap();
            assert!(result.applied);
            assert_eq!(result.branch, Some(branch));
            assert_eq!(
                [
                    f32_bits(result.state_after.b10),
                    f32_bits(result.state_after.b20),
                    f32_bits(result.state_after.b2c),
                    f32_bits(result.state_after.b38),
                ],
                after
            );
        }
    }

    #[test]
    fn run25_primary_row_is_bit_exact() {
        let mut args = producer_args();
        args.pending_b1c = false;
        let result = evaluate_current_family_primary_gauge_producer(
            &args,
            GaugeProducerState {
                b10: f32_from_bits(0x4296_7FFF),
                b20: 0.0,
                b2c: 0.0,
                b38: 0.0,
            },
            f32_from_bits(0x3B03_126F),
            "float32",
        )
        .unwrap();
        assert!(result.applied);
        assert!(result.primary_f09);
        assert_eq!(
            [
                f32_bits(result.state_after.b10),
                f32_bits(result.state_after.b20),
                f32_bits(result.state_after.b2c),
                f32_bits(result.state_after.b38),
            ],
            [0x4297_E665, 0x3B03_126F, 0x0000_0000, 0x0000_0000]
        );
    }

    #[test]
    fn bcharge_step_adds_and_clamps() {
        // Constants computed with the Python authority:
        //   evaluate_bcharge_surface_step(active=True, accum_b10=0.5, raw_760=0.8)
        //   -> after_add 2.100000023841858, cap 1.0, after 1.0.
        let r = evaluate_bcharge_surface_step(true, 0.5, 0.8);
        assert_eq!(r.accum_b10_after_add, 2.100000023841858_f64);
        assert_eq!(r.cap, 1.0);
        assert_eq!(r.accum_b10_after, 1.0);
        assert_eq!(r.increment, BCHARGE_SURFACE_INCREMENT);
        //   inactive -> before passes through untouched.
        let r = evaluate_bcharge_surface_step(false, 0.5, 0.8);
        assert_eq!(r.accum_b10_after_add, 0.5);
        assert_eq!(r.accum_b10_after, 0.5);
        assert_eq!(r.increment, 0.0);
        //   raw cap above 1.0 stays authoritative (gauge_cap max).
        let r = evaluate_bcharge_surface_step(true, 0.0, 5.0);
        assert_eq!(r.cap, 5.0);
        assert_eq!(r.accum_b10_after, BCHARGE_SURFACE_INCREMENT);
    }

    #[test]
    fn current_family_gate_uses_committed_b10_and_grants_fixed_slot() {
        let slots = vec![
            InventorySlot {
                item_id: 6,
                quantity: 1,
            },
            InventorySlot {
                item_id: CURRENT_MANAGER_EMPTY_ITEM_ID,
                quantity: 0,
            },
        ];
        let boundary = evaluate_current_family_item6_grant(
            3711.87060546875,
            299.3129577636719,
            3860.0,
            &slots,
        )
        .unwrap();
        assert_eq!(boundary.snapshot.accum_clamped, 3860.0);
        assert!(!boundary.snapshot.final_eligible);
        assert!(!boundary.granted);

        let granted = evaluate_current_family_item6_grant(3860.0, 0.0, 3860.0, &slots).unwrap();
        assert!(granted.granted);
        assert_eq!(granted.selected_slot_index, Some(1));
        assert_eq!(granted.accum_b10_after.to_bits(), 0.0_f64.to_bits());
        assert_eq!(granted.slots_after[1].item_id, 6);
    }

    #[test]
    fn pending_delta_piecewise() {
        // Constants computed with the Python authority:
        //   pending_accum_delta(0.016, 0.1) == 0.048; (0.016, 0.3) == 0.024;
        //   (0.016, 0.8) == 0.016/(2*0.8); multiplier scales last.
        assert_eq!(pending_accum_delta(0.016, 0.1, 1.0), Some(0.048));
        assert_eq!(pending_accum_delta(0.016, 0.3, 1.0), Some(0.024));
        assert_eq!(pending_accum_delta(0.016, 0.8, 1.0), Some(0.016 / 1.6));
        assert_eq!(pending_accum_delta(0.016, 0.1, 2.5), Some(0.048 * 2.5));
        assert_eq!(pending_accum_delta(f64::NAN, 0.1, 1.0), None);
    }

    #[test]
    fn gauge_ratio_and_display_target() {
        // Constants computed with the Python authority:
        //   gauge_ratio(0.25, 0.25, 1.0) == 0.5; mq_display_target(0.5) ==
        //   0.4950000047683716; raw cap zero still uses denominator 1.0.
        assert_eq!(gauge_ratio(0.25, 0.25, 1.0), Some(0.5));
        assert_eq!(mq_display_target(0.5), 0.4950000047683716_f64);
        assert_eq!(gauge_ratio(0.25, 0.25, 0.0), Some(0.0));
        assert_eq!(gauge_ratio(0.25, 0.25, -1.0), Some(-1.0));
    }

    #[test]
    fn drift_gauge_counters_and_window() {
        // Python authority:
        //   state34_drift_gauge_event(3, False, True, False, 7, 2) == (8, 3)
        //   state34_drift_gauge_event(5, False, True, False, 7, 2) == (7, 2)
        assert_eq!(
            state34_drift_gauge_event(3, false, true, false, 7, 2),
            (8, 3)
        );
        assert_eq!(
            state34_drift_gauge_event(5, false, true, false, 7, 2),
            (7, 2)
        );
        assert_eq!(
            state34_drift_gauge_event(3, true, true, false, 7, 2),
            (7, 2)
        );
        //   state34_drift_gauge_window_can_activate(eligible=True, active=False,
        //     divisor=3, total=7, cycle=3, activation=1, tick=10, prev_expiry=9) -> True
        assert!(state34_drift_gauge_window_can_activate(
            true, false, 3, 7, 3, 1, 10, 9
        ));
        assert!(!state34_drift_gauge_window_can_activate(
            true, false, 3, 7, 3, 2, 10, 9
        ));
        assert!(state34_drift_gauge_window_expired(true, 11, 10));
        assert!(!state34_drift_gauge_window_expired(true, 10, 10));
    }

    #[test]
    fn commit_pending_resets() {
        // Constants computed with the Python authority:
        //   commit_pending(0.4, 0.3, 1.0) -> {"accum_b10": 0.7, "pending_b1c": 0,
        //     "progress_b20": 0.0, "history_b38": 0.3, "accum_b2c": 0.0}
        let r = commit_pending(0.4, 0.3, 1.0).unwrap();
        assert_eq!(r.accum_b10, 0.7);
        assert_eq!(r.pending_b1c, 0);
        assert_eq!(r.history_b38, 0.3);
        assert_eq!(r.accum_b2c, 0.0);
        assert!(commit_pending(f64::INFINITY, 0.3, 1.0).is_none());
    }
}
