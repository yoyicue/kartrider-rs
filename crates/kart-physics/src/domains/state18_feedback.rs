//! Port of `scripts/evaluate_state18_feedback.py` — executable reference for
//! the proven state18 stroke-feedback selector.

/// `representative_stroke_dispatch(...)` — exact outer branch used by the
/// inspected GoStroke dispatch family.
pub fn representative_stroke_dispatch(event_type: i64, kart_a9c: bool) -> bool {
    (event_type == 2 && !kart_a9c) || (event_type == 3 && kart_a9c)
}

/// `select_state18_feedback(...)` — true means request GoPlayKart state18
/// for 1000 ms.
pub fn select_state18_feedback(
    stroke_dispatch_selected: bool,
    control_lock_1c0: bool,
    event_active: bool,
    effect_feedback_enabled_258: bool,
) -> bool {
    stroke_dispatch_selected
        && control_lock_1c0
        && event_active
        && effect_feedback_enabled_258
}

/// `timer_expiry_next_state(...)` — GoPlayKart_UpdateTick's generic +0x630
/// timer-expiry transition.
pub fn timer_expiry_next_state(current_state: i64, timer_active: bool) -> i64 {
    if !timer_active && !matches!(current_state, 0 | 2 | 19) {
        return 0;
    }
    current_state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stroke_dispatch_table() {
        // Python authority: representative_stroke_dispatch over the 2x2 gate.
        assert!(representative_stroke_dispatch(2, false));
        assert!(!representative_stroke_dispatch(2, true));
        assert!(representative_stroke_dispatch(3, true));
        assert!(!representative_stroke_dispatch(3, false));
        assert!(!representative_stroke_dispatch(1, false));
    }

    #[test]
    fn feedback_requires_all_gates() {
        // Python authority: select_state18_feedback(True, True, True, True) is True.
        assert!(select_state18_feedback(true, true, true, true));
        assert!(!select_state18_feedback(true, true, true, false));
        assert!(!select_state18_feedback(false, true, true, true));
    }

    #[test]
    fn timer_expiry_holds_states_0_2_19() {
        // Python authority: timer_expiry_next_state over the exception set.
        assert_eq!(timer_expiry_next_state(18, false), 0);
        assert_eq!(timer_expiry_next_state(18, true), 18);
        assert_eq!(timer_expiry_next_state(0, false), 0);
        assert_eq!(timer_expiry_next_state(2, false), 2);
        assert_eq!(timer_expiry_next_state(19, false), 19);
    }
}
