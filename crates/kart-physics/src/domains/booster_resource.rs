//! Port of `scripts/evaluate_booster_resource.py` — booster inventory and
//! activation timing model recovered from GoItemSlot and GoPlayKart. This is
//! the small pure helper module imported by `booster_vehicle_step`.

/// One GoItemSlot entry: `{"item_id": int, "quantity": int}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InventorySlot {
    pub item_id: i64,
    pub quantity: i64,
}

/// Current fixcrt35 C57450-family manager layout.  Unlike the historical
/// compact vector above, these are two stable slot addresses with 0xffff as
/// the empty sentinel (run21d/run22).
pub const CURRENT_MANAGER_CAPACITY: usize = 2;
pub const CURRENT_MANAGER_EMPTY_ITEM_ID: i64 = 0xffff;

pub fn empty_current_manager_slots() -> Vec<InventorySlot> {
    vec![
        InventorySlot {
            item_id: CURRENT_MANAGER_EMPTY_ITEM_ID,
            quantity: 0,
        };
        CURRENT_MANAGER_CAPACITY
    ]
}

pub fn validate_current_manager_slots(slots: &[InventorySlot]) -> Result<(), String> {
    if slots.len() != CURRENT_MANAGER_CAPACITY {
        return Err("current booster inventory must contain exactly two slots".to_string());
    }
    for (index, slot) in slots.iter().enumerate() {
        if !(0..=0xffff).contains(&slot.item_id) {
            return Err(format!(
                "current booster slot {index} item_id must be uint16"
            ));
        }
        if slot.quantity < 0 {
            return Err(format!(
                "current booster slot {index} quantity must be nonnegative"
            ));
        }
        if slot.item_id == CURRENT_MANAGER_EMPTY_ITEM_ID {
            if slot.quantity != 0 {
                return Err(format!(
                    "current booster slot {index} empty sentinel requires quantity 0"
                ));
            }
        } else if slot.quantity == 0 {
            return Err(format!(
                "current booster slot {index} occupied item requires positive quantity"
            ));
        }
    }
    Ok(())
}

/// Functional Rust form of Python's mutating `add_current_manager_item`.
/// Returns the selected slot index (`-1` when full) and the fixed layout.
pub fn add_current_manager_item(
    slots: &[InventorySlot],
    item_id: i64,
    quantity: i64,
) -> Result<(i64, Vec<InventorySlot>), String> {
    validate_current_manager_slots(slots)?;
    if item_id != 6 {
        return Err("current booster grant is only adjudicated for item6".to_string());
    }
    if quantity != 1 {
        return Err("current item6 grant is only adjudicated for quantity 1".to_string());
    }
    let mut slots_after = slots.to_vec();
    for (index, slot) in slots_after.iter_mut().enumerate() {
        if slot.item_id == CURRENT_MANAGER_EMPTY_ITEM_ID {
            *slot = InventorySlot { item_id, quantity };
            return Ok((index as i64, slots_after));
        }
    }
    Ok((-1, slots_after))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentManagerConsumed {
    pub item_id: i64,
    pub slot_index: i64,
}

/// Functional Rust form of Python's highest-occupied-slot-first consumer.
pub fn consume_current_manager_item(
    slots: &[InventorySlot],
) -> Result<(Option<CurrentManagerConsumed>, Vec<InventorySlot>), String> {
    validate_current_manager_slots(slots)?;
    if slots
        .iter()
        .any(|slot| slot.item_id != 6 && slot.item_id != CURRENT_MANAGER_EMPTY_ITEM_ID)
    {
        return Err("current booster consumption is only adjudicated for item6 slots".to_string());
    }
    let mut slots_after = slots.to_vec();
    for index in (0..CURRENT_MANAGER_CAPACITY).rev() {
        let slot = slots_after[index];
        if slot.item_id == CURRENT_MANAGER_EMPTY_ITEM_ID {
            continue;
        }
        if slot.quantity > 1 {
            slots_after[index].quantity -= 1;
        } else {
            slots_after[index] = InventorySlot {
                item_id: CURRENT_MANAGER_EMPTY_ITEM_ID,
                quantity: 0,
            };
        }
        return Ok((
            Some(CurrentManagerConsumed {
                item_id: slot.item_id,
                slot_index: index as i64,
            }),
            slots_after,
        ));
    }
    Ok((None, slots_after))
}

/// `tune_time_ms(value)`: non-finite -> None, negative -> 0, else `int(value)`
/// (truncation toward zero; `value` is non-negative on this path).
pub fn tune_time_ms(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }
    Some(if value < 0.0 { 0 } else { value as i64 })
}

/// `activation_for_item(item_id, ...)` -> `(state, duration_ms)` or None.
pub fn activation_for_item(
    item_id: i64,
    normal_booster_time: f64,
    item_booster_time: f64,
    animal_booster_time: f64,
) -> Option<(i64, i64)> {
    let normal_ms = tune_time_ms(normal_booster_time)?;
    let item_ms = tune_time_ms(item_booster_time)?;
    let animal_ms = tune_time_ms(animal_booster_time)?;
    if item_id == 6 {
        return Some((3, normal_ms.max(item_ms)));
    }
    if item_id == 31 {
        return Some((5, animal_ms));
    }
    if item_id == 60 {
        return Some((6, animal_ms));
    }
    None
}

/// `advance_state_timer(...)` result dict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateTimerStep {
    pub state: i64,
    pub remaining_ms: i64,
    pub frame_ms: i64,
    pub consumed_ms: i64,
    pub clears_on_tick: bool,
}

/// `advance_state_timer(state, remaining_ms, elapsed_ms)` — UpdateTick's
/// capped frame delta and same-tick state clear.
pub fn advance_state_timer(
    state: i64,
    remaining_ms: i64,
    elapsed_ms: i64,
) -> Result<StateTimerStep, String> {
    if remaining_ms < 0 {
        return Err("remaining_ms must be nonnegative".to_string());
    }
    let frame_ms = elapsed_ms.clamp(0, 500); // Python min(max(elapsed_ms, 0), 500)
    let consumed_ms = remaining_ms.min(frame_ms);
    let remaining_after = remaining_ms - consumed_ms;
    let clears_on_tick = remaining_after == 0 && !matches!(state, 0 | 2 | 19);
    Ok(StateTimerStep {
        state: if clears_on_tick { 0 } else { state },
        remaining_ms: remaining_after,
        frame_ms,
        consumed_ms,
        clears_on_tick,
    })
}

/// `add_inventory_item(slots, item_id, quantity, capacity=2)` — mutating in
/// Python; here it returns `(added, slots_after)`.
pub fn add_inventory_item(
    slots: &[InventorySlot],
    item_id: i64,
    quantity: i64,
    capacity: i64,
) -> (bool, Vec<InventorySlot>) {
    if quantity <= 0 {
        return (false, slots.to_vec());
    }
    let mut slots_after = slots.to_vec();
    for slot in slots_after.iter_mut() {
        if slot.item_id == item_id && slot.quantity < 10 {
            slot.quantity += quantity;
            return (true, slots_after);
        }
    }
    if slots_after.len() as i64 >= capacity {
        return (false, slots_after);
    }
    slots_after.push(InventorySlot { item_id, quantity });
    (true, slots_after)
}

/// `consume_inventory_item(slots, index=0)` — mutating in Python; here it
/// returns `(item_id, slots_after)` with `item_id == None` on the miss path.
pub fn consume_inventory_item(
    slots: &[InventorySlot],
    index: i64,
) -> (Option<i64>, Vec<InventorySlot>) {
    if index < 0 || index as usize >= slots.len() {
        return (None, slots.to_vec());
    }
    let mut slots_after = slots.to_vec();
    let slot = slots_after[index as usize];
    let item_id = slot.item_id;
    if slot.quantity > 1 {
        slots_after[index as usize].quantity -= 1;
    } else {
        slots_after.remove(index as usize);
    }
    (Some(item_id), slots_after)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tune_time_ms_matches_python_int_truncation() {
        // Python authority: tune_time_ms(2500.9) == 2500, tune_time_ms(-1.0) == 0.
        assert_eq!(tune_time_ms(2500.9), Some(2500));
        assert_eq!(tune_time_ms(-1.0), Some(0));
        assert_eq!(tune_time_ms(f64::NAN), None);
        assert_eq!(tune_time_ms(f64::INFINITY), None);
    }

    #[test]
    fn activation_for_item_table() {
        // Python authority:
        //   activation_for_item(6, normal=2600.0, item=3000.0, animal=4000.0) == (3, 3000)
        //   activation_for_item(31, ...) == (5, 4000); (60, ...) == (6, 4000); (7, ...) is None
        assert_eq!(
            activation_for_item(6, 2600.0, 3000.0, 4000.0),
            Some((3, 3000))
        );
        assert_eq!(
            activation_for_item(31, 2600.0, 3000.0, 4000.0),
            Some((5, 4000))
        );
        assert_eq!(
            activation_for_item(60, 2600.0, 3000.0, 4000.0),
            Some((6, 4000))
        );
        assert_eq!(activation_for_item(7, 2600.0, 3000.0, 4000.0), None);
    }

    #[test]
    fn advance_state_timer_caps_and_clears() {
        // Python authority: advance_state_timer(3, 10, 600) ->
        //   {"state": 0, "remaining_ms": 0, "frame_ms": 500, "consumed_ms": 10, "clears_on_tick": True}
        let step = advance_state_timer(3, 10, 600).unwrap();
        assert_eq!(
            step,
            StateTimerStep {
                state: 0,
                remaining_ms: 0,
                frame_ms: 500,
                consumed_ms: 10,
                clears_on_tick: true
            }
        );
        // state 2 never clears on tick.
        let held = advance_state_timer(2, 4, 100).unwrap();
        assert_eq!(held.state, 2);
        assert!(!held.clears_on_tick);
        assert_eq!(held.remaining_ms, 0);
        assert!(advance_state_timer(3, -1, 0).is_err());
    }

    #[test]
    fn inventory_add_and_consume() {
        let slots = vec![InventorySlot {
            item_id: 6,
            quantity: 1,
        }];
        // Python authority: add stacks below quantity 10, then new slots fill to capacity.
        let (added, after) = add_inventory_item(&slots, 6, 2, 2);
        assert!(added);
        assert_eq!(
            after,
            vec![InventorySlot {
                item_id: 6,
                quantity: 3
            }]
        );
        let (item, after) = consume_inventory_item(&after, 0);
        assert_eq!(item, Some(6));
        assert_eq!(
            after,
            vec![InventorySlot {
                item_id: 6,
                quantity: 2
            }]
        );
        let (item, after) = consume_inventory_item(&slots, 0);
        assert_eq!(item, Some(6));
        assert!(after.is_empty());
        assert_eq!(consume_inventory_item(&slots, 5).0, None);
    }

    #[test]
    fn current_manager_preserves_fixed_slots_and_consumes_highest() {
        let empty = empty_current_manager_slots();
        validate_current_manager_slots(&empty).unwrap();
        let (index, one) = add_current_manager_item(&empty, 6, 1).unwrap();
        assert_eq!(index, 0);
        let (index, two) = add_current_manager_item(&one, 6, 1).unwrap();
        assert_eq!(index, 1);
        let (consumed, after) = consume_current_manager_item(&two).unwrap();
        assert_eq!(consumed.unwrap().slot_index, 1);
        assert_eq!(after[0].item_id, 6);
        assert_eq!(after[1].item_id, CURRENT_MANAGER_EMPTY_ITEM_ID);
        assert_eq!(after.len(), CURRENT_MANAGER_CAPACITY);
        assert_eq!(add_current_manager_item(&two, 6, 1).unwrap().0, -1);
    }
}
