//! Port of `scripts/evaluate_booster_vehicle_step.py` — executable bridge
//! from booster inventory/timer state to the ground force branch. Reuses the
//! Rust ports of `booster_resource` (helper module) and
//! `ground_state_scalar` (`evaluate_state_mult`/`evaluate_primary_force_delta`).

use crate::domains::booster_resource::{
    activation_for_item, advance_state_timer, consume_current_manager_item, consume_inventory_item,
    validate_current_manager_slots, InventorySlot, StateTimerStep, CURRENT_MANAGER_EMPTY_ITEM_ID,
};
use crate::domains::ground_state_scalar::{
    evaluate_primary_force_delta, evaluate_state_mult, PrimaryForceDelta, StateMultResult,
};

/// `evaluate_booster_inventory_use` static order provenance (Python key
/// `static_order`, present on the accepted path only).
pub const STATIC_ORDER: &str = "GoItemSlot peek -> 0x722D50 rejection gate -> 0xB1F7A0 SetCurrentState attempt -> common GoItemSlot consume -> later UpdateTick timer decrement";

/// `evaluate_vehicle_step` ordering provenance (Python key `ordering`).
pub const ORDERING: &str = "optional inventory peek/activation/accepted-consume 0x720590+0x722D50 -> timer decrement 0xB26101 -> optional state0 clear 0xB2611F -> ground scalar call 0xB262E6";

/// `evaluate_vehicle_step` scope provenance (Python key `scope`).
pub const SCOPE: &str = "ground primary correction term only; BS state13, rail, wall, water, air, landing, and optional ground add-ons remain separate producers";

/// Python `activate_from_inventory(...)` result dict. The Python dict has no
/// `reason` key on the activated path, so `reason` is `None` there;
/// `item_id` is absent only on the empty-inventory path, and
/// `state`/`remaining_ms` are present only when activated.
#[derive(Debug, Clone, PartialEq)]
pub struct ActivateFromInventoryResult {
    pub activated: bool,
    pub reason: Option<&'static str>,
    /// Present unless `reason == "empty_inventory"`.
    pub item_id: Option<i64>,
    /// Present only when activated.
    pub state: Option<i64>,
    /// Present only when activated.
    pub remaining_ms: Option<i64>,
    pub slots: Vec<InventorySlot>,
}

/// `activate_from_inventory(slots, ...)` — consume first, then gate on the
/// activation table; the returned `slots` are always the post-consume list.
pub fn activate_from_inventory(
    slots: &[InventorySlot],
    normal_booster_time: f64,
    item_booster_time: f64,
    animal_booster_time: f64,
) -> ActivateFromInventoryResult {
    let (item_id, slots_after) = consume_inventory_item(slots, 0);
    let Some(item_id) = item_id else {
        return ActivateFromInventoryResult {
            activated: false,
            reason: Some("empty_inventory"),
            item_id: None,
            state: None,
            remaining_ms: None,
            slots: slots_after,
        };
    };
    let activation = activation_for_item(
        item_id,
        normal_booster_time,
        item_booster_time,
        animal_booster_time,
    );
    match activation {
        None => ActivateFromInventoryResult {
            activated: false,
            reason: Some("unsupported_item"),
            item_id: Some(item_id),
            state: None,
            remaining_ms: None,
            slots: slots_after,
        },
        Some((state, remaining_ms)) => ActivateFromInventoryResult {
            activated: true,
            reason: None,
            item_id: Some(item_id),
            state: Some(state),
            remaining_ms: Some(remaining_ms),
            slots: slots_after,
        },
    }
}

/// Keyword bundle for the Python `inventory_use` dict argument.
#[derive(Debug, Clone, PartialEq)]
pub struct InventoryUseInput {
    pub slots: Vec<InventorySlot>,
    pub stage_mode_id: i64,
    pub allow_refresh_in_mode72: bool,
    pub positive_a80: f64,
    pub state_setter_environment_allowed: bool,
    pub normal_booster_time: f64,
    pub item_booster_time: f64,
    pub animal_booster_time: f64,
}

/// Python `evaluate_booster_inventory_use(...)` result dict. Optional fields
/// mirror the Python key-presence rules (see the trace example encoder).
#[derive(Debug, Clone, PartialEq)]
pub struct InventoryUseResult {
    pub consumed: bool,
    pub handler_accepted: bool,
    pub state_changed: bool,
    pub reason: String,
    /// Absent only on the `empty_inventory` path.
    pub item_id: Option<i64>,
    /// Present only on the accepted path.
    pub state_setter_attempted: Option<bool>,
    /// Present only on the accepted path.
    pub target_state: Option<i64>,
    /// Present only on the accepted path.
    pub duration_ms: Option<i64>,
    /// Present only on the accepted path.
    pub state_after: Option<i64>,
    /// Present only on the accepted path.
    pub remaining_ms_after: Option<i64>,
    pub slots_after: Vec<InventorySlot>,
    /// Present only on the accepted path (Python key `static_order`).
    pub static_order: Option<&'static str>,
}

/// `evaluate_booster_inventory_use(...)` — peek, gate activation, then
/// consume only after 722D50 accepts.
pub fn evaluate_booster_inventory_use(
    current_state: i64,
    input: &InventoryUseInput,
) -> InventoryUseResult {
    let slots_after = input.slots.clone();
    let item_id = slots_after.first().map(|slot| slot.item_id);
    let Some(item_id) = item_id else {
        return InventoryUseResult {
            consumed: false,
            handler_accepted: false,
            state_changed: false,
            reason: "empty_inventory".to_string(),
            item_id: None,
            state_setter_attempted: None,
            target_state: None,
            duration_ms: None,
            state_after: None,
            remaining_ms_after: None,
            slots_after,
            static_order: None,
        };
    };
    let activation = activation_for_item(
        item_id,
        input.normal_booster_time,
        input.item_booster_time,
        input.animal_booster_time,
    );
    let Some((target_state, duration_ms)) = activation else {
        return InventoryUseResult {
            consumed: false,
            handler_accepted: false,
            state_changed: false,
            reason: "non_booster_or_invalid_tuning".to_string(),
            item_id: Some(item_id),
            state_setter_attempted: None,
            target_state: None,
            duration_ms: None,
            state_after: None,
            remaining_ms_after: None,
            slots_after,
            static_order: None,
        };
    };
    let boost_family_active = (1..12).contains(&current_state);
    let reject_reason = if (13..17).contains(&current_state) {
        Some("current_state_13_16_block")
    } else if input.positive_a80 <= 0.0 {
        Some("positive_a80_not_active")
    } else if boost_family_active && input.stage_mode_id == 72 && !input.allow_refresh_in_mode72 {
        Some("mode72_boost_refresh_not_allowed")
    } else if boost_family_active && input.stage_mode_id != 72 {
        Some("non_mode72_existing_boost")
    } else {
        None
    };
    if let Some(reject_reason) = reject_reason {
        return InventoryUseResult {
            consumed: false,
            handler_accepted: false,
            state_changed: false,
            reason: reject_reason.to_string(),
            item_id: Some(item_id),
            state_setter_attempted: None,
            target_state: None,
            duration_ms: None,
            state_after: None,
            remaining_ms_after: None,
            slots_after,
            static_order: None,
        };
    }
    let state_changed = input.state_setter_environment_allowed;
    let (consumed_item, slots_after) = consume_inventory_item(&slots_after, 0);
    assert_eq!(
        consumed_item,
        Some(item_id),
        "peek/consume inventory identity mismatch"
    );
    InventoryUseResult {
        consumed: true,
        handler_accepted: true,
        state_changed,
        reason: if state_changed {
            "activated".to_string()
        } else {
            "setter_environment_rejected_after_handler_accept".to_string()
        },
        item_id: Some(item_id),
        state_setter_attempted: Some(true),
        target_state: Some(target_state),
        duration_ms: Some(duration_ms),
        state_after: Some(if state_changed { target_state } else { 0 }),
        remaining_ms_after: Some(if state_changed { duration_ms } else { 0 }),
        slots_after,
        static_order: Some(STATIC_ORDER),
    }
}

pub const CURRENT_ITEM6_RUNTIME_ORDER: &str = "C56F70 -> BD3860 -> BB4F40 highest occupied fixed slot -> C57136 state3/timer setter -> UpdateTick timer decrement";
pub const CURRENT_ITEM6_AUTHORITY: &str =
    "run22_double_item6_highest_slot_consume_and_run19_run20_state3_2900ms_ground_chain";

/// Current-family fixed-slot item6 action.  Optional fields mirror Python's
/// key-presence rules so trace encoders can remain structurally exact.
#[derive(Debug, Clone, PartialEq)]
pub struct CurrentItem6InventoryUseResult {
    pub consumed: bool,
    pub handler_accepted: bool,
    pub state_changed: bool,
    pub reason: String,
    pub item_id: Option<i64>,
    pub consumed_slot_index: Option<i64>,
    pub target_state: Option<i64>,
    pub duration_ms: Option<i64>,
    pub state_after: Option<i64>,
    pub remaining_ms_after: Option<i64>,
    pub slots_after: Vec<InventorySlot>,
    pub runtime_order: Option<&'static str>,
    pub authority: Option<&'static str>,
    /// Present only for the engine's non-GND fail-safe projection.
    pub domain: Option<String>,
}

pub fn evaluate_current_item6_inventory_use(
    slots: &[InventorySlot],
    current_state: i64,
    duration_ms: i64,
) -> Result<CurrentItem6InventoryUseResult, String> {
    validate_current_manager_slots(slots)?;
    if duration_ms <= 0 {
        return Err("current item6 duration_ms must be a positive integer".to_string());
    }
    let base = || CurrentItem6InventoryUseResult {
        consumed: false,
        handler_accepted: false,
        state_changed: false,
        reason: String::new(),
        item_id: None,
        consumed_slot_index: None,
        target_state: None,
        duration_ms: None,
        state_after: None,
        remaining_ms_after: None,
        slots_after: slots.to_vec(),
        runtime_order: None,
        authority: None,
        domain: None,
    };
    let occupied: Vec<_> = slots
        .iter()
        .filter(|slot| slot.item_id != CURRENT_MANAGER_EMPTY_ITEM_ID)
        .collect();
    if occupied.is_empty() {
        let mut result = base();
        result.reason = "empty_inventory".to_string();
        return Ok(result);
    }
    if occupied.iter().any(|slot| slot.item_id != 6) {
        let mut result = base();
        result.reason = "mixed_or_unadjudicated_item_inventory".to_string();
        return Ok(result);
    }
    if current_state != 0 {
        let mut result = base();
        result.reason = "nonzero_boost_state_refresh_unadjudicated".to_string();
        result.item_id = Some(6);
        return Ok(result);
    }
    let (consumed, slots_after) = consume_current_manager_item(slots)?;
    let consumed = consumed.expect("occupied item6 inventory must consume");
    assert_eq!(
        consumed.item_id, 6,
        "current item6 peek/consume identity mismatch"
    );
    Ok(CurrentItem6InventoryUseResult {
        consumed: true,
        handler_accepted: true,
        state_changed: true,
        reason: "activated".to_string(),
        item_id: Some(6),
        consumed_slot_index: Some(consumed.slot_index),
        target_state: Some(3),
        duration_ms: Some(duration_ms),
        state_after: Some(3),
        remaining_ms_after: Some(duration_ms),
        slots_after,
        runtime_order: Some(CURRENT_ITEM6_RUNTIME_ORDER),
        authority: Some(CURRENT_ITEM6_AUTHORITY),
        domain: None,
    })
}

/// Python `evaluate_vehicle_step(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct BoosterVehicleStepResult {
    pub state_before: i64,
    pub inventory_transition: Option<InventoryUseResult>,
    pub state_before_timer: i64,
    pub state_for_force: i64,
    pub boost_family_for_force: bool,
    pub remaining_before_ms: i64,
    pub state_scalar: StateMultResult,
    pub primary_force: PrimaryForceDelta,
    pub timer_after: StateTimerStep,
    pub ordering: &'static str,
    pub scope: &'static str,
}

/// `evaluate_vehicle_step(...)` — GoPlayKart_UpdateTick decrements +0x630 at
/// 0xB26101 and clears state at 0xB2611F before the ground scalar call at
/// 0xB262E6, so the force selector observes the post-timer state.
pub fn evaluate_vehicle_step(
    state: i64,
    remaining_ms: i64,
    elapsed_ms: i64,
    correction_dir: [f64; 3],
    direction_gain: f64,
    selected_slot: f64,
    attached_state: i64,
    e8c: bool,
    efc: bool,
    flag_5bc: bool,
    raw_144: f64,
    obf_808: f64,
    obf_814: f64,
    obf_69c: f64,
    obf_684: f64,
    obf_8ac: f64,
    obf_8e8: f64,
    obf_94c: f64,
    inventory_use: Option<&InventoryUseInput>,
) -> Result<BoosterVehicleStepResult, String> {
    let mut inventory_transition = None;
    let mut state_before_timer = state;
    let mut remaining_before_timer = remaining_ms;
    if let Some(input) = inventory_use {
        let transition = evaluate_booster_inventory_use(state, input);
        if transition.state_changed || transition.handler_accepted {
            // B1F7A0 clears state when its environment/caller gate rejects.
            state_before_timer = transition
                .state_after
                .expect("accepted row has state_after");
            remaining_before_timer = transition
                .remaining_ms_after
                .expect("accepted row has remaining_ms_after");
        }
        inventory_transition = Some(transition);
    }
    let timer = advance_state_timer(state_before_timer, remaining_before_timer, elapsed_ms)?;
    let effective_state = timer.state;
    let boost_family = (1..12).contains(&effective_state);
    let scalar = evaluate_state_mult(
        effective_state,
        boost_family,
        attached_state,
        e8c,
        efc,
        flag_5bc,
        raw_144,
        obf_808,
        obf_814,
        obf_69c,
        obf_684,
        obf_8ac,
        obf_8e8,
        obf_94c,
    );
    let force = evaluate_primary_force_delta(
        correction_dir,
        direction_gain,
        selected_slot,
        scalar.state_mult,
    );
    Ok(BoosterVehicleStepResult {
        state_before: state,
        inventory_transition,
        state_before_timer,
        state_for_force: effective_state,
        boost_family_for_force: boost_family,
        remaining_before_ms: remaining_before_timer,
        state_scalar: scalar,
        primary_force: force,
        timer_after: timer,
        ordering: ORDERING,
        scope: SCOPE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn use_input() -> InventoryUseInput {
        InventoryUseInput {
            slots: vec![InventorySlot {
                item_id: 6,
                quantity: 1,
            }],
            stage_mode_id: 0,
            allow_refresh_in_mode72: false,
            positive_a80: 1.0,
            state_setter_environment_allowed: true,
            normal_booster_time: 2600.0,
            item_booster_time: 3000.0,
            animal_booster_time: 4000.0,
        }
    }

    #[test]
    fn inventory_use_rejection_gate_order() {
        // Python authority: state 14 blocks before the a80 gate is consulted.
        let r = evaluate_booster_inventory_use(14, &use_input());
        assert_eq!(r.reason, "current_state_13_16_block");
        assert!(!r.consumed);
        // Python authority: existing boost outside mode72 rejects.
        let r = evaluate_booster_inventory_use(3, &use_input());
        assert_eq!(r.reason, "non_mode72_existing_boost");
        // Python authority: fresh activation from state 0 consumes slot 0.
        let r = evaluate_booster_inventory_use(0, &use_input());
        assert!(r.consumed && r.state_changed);
        assert_eq!(r.target_state, Some(3));
        assert_eq!(r.duration_ms, Some(3000));
        assert!(r.slots_after.is_empty());
    }

    #[test]
    fn vehicle_step_observes_post_timer_state() {
        // Python authority:
        //   evaluate_vehicle_step(state=3, remaining_ms=1, elapsed_ms=16,
        //     correction_dir=[1,0,0], direction_gain=1.5, selected_slot=2.0,
        //     attached_state=0, e8c=False, efc=False, flag_5bc=False,
        //     raw_144=1.0, obf_808=1.1, obf_814=1.2, obf_69c=1.3, obf_684=1.4,
        //     obf_8ac=1.5, obf_8e8=1.6, obf_94c=2.0)
        //   -> state_for_force 0 (timer expired same tick), state_mult 1.0,
        //      force scalar 3.0.
        let r = evaluate_vehicle_step(
            3,
            1,
            16,
            [1.0, 0.0, 0.0],
            1.5,
            2.0,
            0,
            false,
            false,
            false,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            2.0,
            None,
        )
        .unwrap();
        assert_eq!(r.state_for_force, 0);
        assert!(!r.boost_family_for_force);
        assert_eq!(r.state_scalar.state_mult, 1.0);
        assert_eq!(r.primary_force.scalar, 3.0);
        assert_eq!(r.primary_force.force_delta, [3.0, 0.0, 0.0]);
        assert!(r.timer_after.clears_on_tick);
        assert!(evaluate_vehicle_step(
            0, -1, 0, [0.0; 3], 1.0, 0.0, 0, false, false, false, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, None,
        )
        .is_err());
    }

    #[test]
    fn vehicle_step_inventory_accept_updates_timer_inputs() {
        // Python authority: accepted activation seeds state/remaining before
        // the timer decrement runs.
        let input = use_input();
        let r = evaluate_vehicle_step(
            0,
            0,
            100,
            [0.0, 1.0, 0.0],
            2.0,
            1.0,
            0,
            false,
            false,
            false,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            2.0,
            Some(&input),
        )
        .unwrap();
        assert_eq!(r.state_before_timer, 3);
        assert_eq!(r.remaining_before_ms, 3000);
        assert_eq!(r.state_for_force, 3);
        assert!(r.boost_family_for_force);
        assert_eq!(r.state_scalar.state_mult, 1.2);
        assert_eq!(r.timer_after.remaining_ms, 2900);
    }

    #[test]
    fn current_item6_uses_highest_fixed_slot() {
        let slots = vec![
            InventorySlot {
                item_id: 6,
                quantity: 1,
            },
            InventorySlot {
                item_id: 6,
                quantity: 1,
            },
        ];
        let result = evaluate_current_item6_inventory_use(&slots, 0, 2900).unwrap();
        assert!(result.state_changed);
        assert_eq!(result.consumed_slot_index, Some(1));
        assert_eq!(result.remaining_ms_after, Some(2900));
        assert_eq!(result.slots_after[0].item_id, 6);
        assert_eq!(result.slots_after[1].item_id, CURRENT_MANAGER_EMPTY_ITEM_ID);

        let refused = evaluate_current_item6_inventory_use(&slots, 3, 2900).unwrap();
        assert_eq!(refused.reason, "nonzero_boost_state_refresh_unadjudicated");
        assert_eq!(refused.slots_after, slots);
    }
}
