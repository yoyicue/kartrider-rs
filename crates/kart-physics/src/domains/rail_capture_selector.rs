//! Port of `scripts/evaluate_rail_capture_selector.py` — executable
//! statically recovered subset of the RAIL state machine (states 1/2/3/5/6).
//!
//! The state-6 confirmation countdown performs the native binary32 seconds
//! subtraction of `sub_B27820` (movss/subss/movss) and converts back to
//! milliseconds in f64 — independent frame deltas totaling exactly 100 ms
//! land on a tiny negative binary32 value, not +1.49e-6 ms.

use crate::num::{f32q, max2_f64};

/// Python module constants.
pub const STATE1_CONTACT_DELAY_MS: f64 = 10.0;
pub const STATE1_TIMEOUT_BASE_MS: f64 = 1618.0;
pub const STATE1_AF0_SCALE: f64 = 3.6;
pub const STATE1_TIMEOUT_TARGET: f64 = 100.0;
pub const STATE1_TIMEOUT_DIVISOR: f64 = 1000.0;
pub const STATE5_EXIT_MS: f64 = 1000.0;
pub const STATE2_ALIGNMENT_RATIO: f64 = 0.20000000298023224;
pub const SPECIAL_TRIGGER_COOLDOWN_MS: f64 = 300.0;
pub const STATE6_CONTACT_CONFIRM_MS: f64 = 100.00000149011612;
/// `struct.unpack("<f", bytes.fromhex("6f12833a"))[0]` — the binary32
/// milliseconds-to-seconds factor (0.001 rounded to f32, exact decimal
/// expansion).
pub const MILLISECONDS_TO_SECONDS_F32: f64 = 0.001000000047497451305389404296875;

/// Python `state1_timeout_ms(field_af0)`.
pub fn state1_timeout_ms(field_af0: f64) -> f64 {
    let extra = max2_f64(0.0, STATE1_TIMEOUT_TARGET - STATE1_AF0_SCALE * field_af0);
    STATE1_TIMEOUT_BASE_MS + extra * extra
}

/// Python state-1 `gates` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1Gates {
    pub contact_delay_pass: bool,
    pub contact_exists: bool,
    pub contact_identifier_nonempty: bool,
    pub metric_pass: bool,
    pub height_pass: bool,
}

/// Python state-2 `gates` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState2Gates {
    pub alignment_ratio: f64,
    pub forward_speed: f64,
    pub lateral_speed_abs: f64,
    pub alignment_pass: bool,
    pub primary_timer_strictly_elapsed: bool,
    pub stable_timer_strictly_elapsed: bool,
}

/// Python state-3 `gates` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState3Gates {
    pub trigger_cooldown_zero: bool,
    pub authored_trigger_accepted: bool,
}

/// Python state-6 `gates` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6Gates {
    pub contact_candidate_valid: bool,
    pub contact_timer_strictly_elapsed: bool,
}

/// Python per-state `gates` dict (absent on states 1 and 5).
#[derive(Debug, Clone, PartialEq)]
pub enum RailCaptureSelectorGates {
    State1(RailState1Gates),
    State2(RailState2Gates),
    State3(RailState3Gates),
    State6(RailState6Gates),
}

/// Python state-2 `runtime_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState2RuntimeAfter {
    pub alignment_primary_remaining_ms: f64,
    pub alignment_stable_remaining_ms: f64,
}

/// Python state-3 `runtime_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState3RuntimeAfter {
    pub trigger_cooldown_remaining_ms: f64,
}

/// Python state-6 `runtime_after` dict.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6RuntimeAfter {
    pub state6_contact_remaining_ms: f64,
    pub state6_contact_raw_after_ms: f64,
}

/// Python per-state `runtime_after` dict (absent on states 1 and 5).
#[derive(Debug, Clone, PartialEq)]
pub enum RailCaptureSelectorRuntimeAfter {
    State2(RailState2RuntimeAfter),
    State3(RailState3RuntimeAfter),
    State6(RailState6RuntimeAfter),
}

/// Python `evaluate_rail_capture_selector(...)` result dict. State-specific
/// keys: `elapsed_in_state_ms` / `state1_contact_delay_ms` /
/// `state1_timeout_ms` only on state 1; `state5_exit_ms` only on state 5.
#[derive(Debug, Clone, PartialEq)]
pub struct RailCaptureSelectorResult {
    pub state_before: i64,
    pub state_after: i64,
    pub transitioned: bool,
    pub reason: &'static str,
    pub gates: Option<RailCaptureSelectorGates>,
    pub runtime_after: Option<RailCaptureSelectorRuntimeAfter>,
    pub elapsed_in_state_ms: Option<f64>,
    pub state1_contact_delay_ms: Option<f64>,
    pub state1_timeout_ms: Option<f64>,
    pub state5_exit_ms: Option<f64>,
    pub closure_status: &'static str,
}

/// Keyword bundle for `evaluate_rail_capture_selector` (Python kwargs;
/// length/type checks are type-enforced in Rust).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RailCaptureSelectorArgs {
    pub state_before: i64,
    pub elapsed_in_state_ms: f64,
    pub field_af0: f64,
    pub contact_exists: bool,
    pub contact_identifier_nonempty: bool,
    pub path_metric: Option<f64>,
    pub metric_threshold: Option<f64>,
    pub path_z: Option<f64>,
    pub kart_z_plus_one: Option<f64>,
    pub delta_ms: Option<f64>,
    pub alignment_primary_remaining_ms: Option<f64>,
    pub alignment_stable_remaining_ms: Option<f64>,
    pub forward_speed: Option<f64>,
    pub lateral_speed: Option<f64>,
    pub trigger_cooldown_remaining_ms: Option<f64>,
    pub authored_trigger_accepted: bool,
    pub state6_contact_remaining_ms: Option<f64>,
    pub state6_contact_candidate_valid: bool,
}

/// `evaluate_rail_capture_selector(...)`.
pub fn evaluate_rail_capture_selector(
    args: &RailCaptureSelectorArgs,
) -> Result<RailCaptureSelectorResult, String> {
    let state_before = args.state_before;
    if !matches!(state_before, 1 | 2 | 3 | 5 | 6) {
        return Err(
            "recovered rail capture selector supports state 1, 2, 3, 5, or 6".to_string()
        );
    }
    if args.elapsed_in_state_ms < 0.0 {
        return Err("elapsed_in_state_ms must be nonnegative".to_string());
    }

    if state_before == 5 {
        let exits = args.elapsed_in_state_ms >= STATE5_EXIT_MS;
        return Ok(RailCaptureSelectorResult {
            state_before: 5,
            state_after: if exits { 0 } else { 5 },
            transitioned: exits,
            reason: if exits { "state5_timeout_exit" } else { "state5_timer_active" },
            gates: None,
            runtime_after: None,
            elapsed_in_state_ms: None,
            state1_contact_delay_ms: None,
            state1_timeout_ms: None,
            state5_exit_ms: Some(STATE5_EXIT_MS),
            closure_status: "runtime_validated_state5_timer_exit",
        });
    }

    if state_before == 6 {
        let (Some(delta_ms), Some(state6_contact_remaining_ms)) =
            (args.delta_ms, args.state6_contact_remaining_ms)
        else {
            return Err(
                "state6 exit requires delta_ms and state6_contact_remaining_ms".to_string()
            );
        };
        if delta_ms < 0.0 || state6_contact_remaining_ms < 0.0 {
            return Err("state6 contact timer and delta_ms must be nonnegative".to_string());
        }
        let mut remaining_after = state6_contact_remaining_ms;
        if args.state6_contact_candidate_valid {
            // B27820 is ``movss/subss/movss`` over the float32 C60 seconds
            // field. Keep the public/runtime representation in milliseconds,
            // but perform the native binary32 seconds subtraction.
            let remaining_seconds = f32q(state6_contact_remaining_ms / 1000.0);
            let delta_seconds = f32q(f32q(delta_ms) * MILLISECONDS_TO_SECONDS_F32);
            remaining_after = f32q(remaining_seconds - delta_seconds) * 1000.0;
        }
        let exits = remaining_after < 0.0;
        return Ok(RailCaptureSelectorResult {
            state_before: 6,
            state_after: if exits { 0 } else { 6 },
            transitioned: exits,
            reason: if exits {
                "state6_contact_confirmation_elapsed"
            } else {
                "state6_waiting_for_contact_confirmation"
            },
            gates: Some(RailCaptureSelectorGates::State6(RailState6Gates {
                contact_candidate_valid: args.state6_contact_candidate_valid,
                contact_timer_strictly_elapsed: remaining_after < 0.0,
            })),
            runtime_after: Some(RailCaptureSelectorRuntimeAfter::State6(
                RailState6RuntimeAfter {
                    state6_contact_remaining_ms: max2_f64(0.0, remaining_after),
                    state6_contact_raw_after_ms: remaining_after,
                },
            )),
            elapsed_in_state_ms: None,
            state1_contact_delay_ms: None,
            state1_timeout_ms: None,
            state5_exit_ms: None,
            closure_status: "static_exact_state6_contact_timer_exit",
        });
    }

    if state_before == 2 {
        let mut missing: Vec<&str> = Vec::new();
        if args.delta_ms.is_none() {
            missing.push("delta_ms");
        }
        if args.alignment_primary_remaining_ms.is_none() {
            missing.push("alignment_primary_remaining_ms");
        }
        if args.alignment_stable_remaining_ms.is_none() {
            missing.push("alignment_stable_remaining_ms");
        }
        if args.forward_speed.is_none() {
            missing.push("forward_speed");
        }
        if args.lateral_speed.is_none() {
            missing.push("lateral_speed");
        }
        if !missing.is_empty() {
            // Python: ", ".join(sorted(missing)).
            missing.sort_unstable();
            return Err(format!("state2 alignment requires {}", missing.join(", ")));
        }
        let delta_ms = args.delta_ms.unwrap();
        if delta_ms < 0.0 {
            return Err("delta_ms must be nonnegative".to_string());
        }
        let forward_speed = args.forward_speed.unwrap();
        let lateral_speed = args.lateral_speed.unwrap();
        let primary_after = args.alignment_primary_remaining_ms.unwrap() - delta_ms;
        let alignment_pass = STATE2_ALIGNMENT_RATIO * forward_speed > lateral_speed.abs();
        let mut stable_after = args.alignment_stable_remaining_ms.unwrap();
        if alignment_pass {
            stable_after -= delta_ms;
        }
        let promotes = primary_after < 0.0 && stable_after < 0.0;
        return Ok(RailCaptureSelectorResult {
            state_before: 2,
            state_after: if promotes { 3 } else { 2 },
            transitioned: promotes,
            reason: if promotes {
                "state2_alignment_timers_elapsed"
            } else {
                "state2_alignment_active"
            },
            gates: Some(RailCaptureSelectorGates::State2(RailState2Gates {
                alignment_ratio: STATE2_ALIGNMENT_RATIO,
                forward_speed,
                lateral_speed_abs: lateral_speed.abs(),
                alignment_pass,
                primary_timer_strictly_elapsed: primary_after < 0.0,
                stable_timer_strictly_elapsed: stable_after < 0.0,
            })),
            runtime_after: Some(RailCaptureSelectorRuntimeAfter::State2(
                RailState2RuntimeAfter {
                    alignment_primary_remaining_ms: primary_after,
                    alignment_stable_remaining_ms: stable_after,
                },
            )),
            elapsed_in_state_ms: None,
            state1_contact_delay_ms: None,
            state1_timeout_ms: None,
            state5_exit_ms: None,
            closure_status: "static_exact_state2_alignment_predicate",
        });
    }

    if state_before == 3 {
        let (Some(delta_ms), Some(trigger_cooldown_remaining_ms)) =
            (args.delta_ms, args.trigger_cooldown_remaining_ms)
        else {
            return Err(
                "state3 authored-trigger selection requires delta_ms and \
                 trigger_cooldown_remaining_ms"
                    .to_string(),
            );
        };
        if delta_ms < 0.0 || trigger_cooldown_remaining_ms < 0.0 {
            return Err("state3 trigger cooldown and delta_ms must be nonnegative".to_string());
        }
        let cooldown_after_countdown =
            max2_f64(0.0, trigger_cooldown_remaining_ms - delta_ms);
        let promotes = cooldown_after_countdown == 0.0 && args.authored_trigger_accepted;
        return Ok(RailCaptureSelectorResult {
            state_before: 3,
            state_after: if promotes { 6 } else { 3 },
            transitioned: promotes,
            reason: if promotes {
                "state3_jm_dj_trigger_accepted"
            } else {
                "state3_trigger_not_accepted"
            },
            gates: Some(RailCaptureSelectorGates::State3(RailState3Gates {
                trigger_cooldown_zero: cooldown_after_countdown == 0.0,
                authored_trigger_accepted: args.authored_trigger_accepted,
            })),
            runtime_after: Some(RailCaptureSelectorRuntimeAfter::State3(
                RailState3RuntimeAfter {
                    trigger_cooldown_remaining_ms: if promotes {
                        SPECIAL_TRIGGER_COOLDOWN_MS
                    } else {
                        cooldown_after_countdown
                    },
                },
            )),
            elapsed_in_state_ms: None,
            state1_contact_delay_ms: None,
            state1_timeout_ms: None,
            state5_exit_ms: None,
            closure_status: "static_exact_state3_jm_dj_trigger_edge",
        });
    }

    // state_before == 1
    let contact_delay_pass = args.elapsed_in_state_ms >= STATE1_CONTACT_DELAY_MS;
    if contact_delay_pass
        && args.contact_exists
        && args.contact_identifier_nonempty
        && (args.path_metric.is_none()
            || args.metric_threshold.is_none()
            || args.path_z.is_none()
            || args.kart_z_plus_one.is_none())
    {
        return Err(
            "accepted rail contact requires path metric/threshold/height fields".to_string()
        );
    }
    let identifier_pass = args.contact_exists && args.contact_identifier_nonempty;
    let metric_pass = identifier_pass
        && args.metric_threshold.is_some()
        && args.path_metric.is_some()
        && args.metric_threshold.unwrap() > args.path_metric.unwrap();
    let height_pass = identifier_pass
        && args.kart_z_plus_one.is_some()
        && args.path_z.is_some()
        && args.kart_z_plus_one.unwrap() > args.path_z.unwrap();
    let contact_accepts = contact_delay_pass && identifier_pass && metric_pass && height_pass;
    let timeout = state1_timeout_ms(args.field_af0);
    let timeout_elapsed = args.elapsed_in_state_ms >= timeout;
    if contact_accepts && timeout_elapsed {
        return Err(
            "simultaneous state1 contact acceptance and timeout priority is unproven"
                .to_string(),
        );
    }
    let (state_after, reason): (i64, &'static str) = if contact_accepts {
        (2, "state1_contact_path_accepted")
    } else if timeout_elapsed {
        (5, "state1_timeout_bypass")
    } else {
        (1, "state1_waiting")
    };
    Ok(RailCaptureSelectorResult {
        state_before: 1,
        state_after,
        transitioned: state_after != 1,
        reason,
        gates: Some(RailCaptureSelectorGates::State1(RailState1Gates {
            contact_delay_pass,
            contact_exists: args.contact_exists,
            contact_identifier_nonempty: args.contact_identifier_nonempty,
            metric_pass,
            height_pass,
        })),
        runtime_after: None,
        elapsed_in_state_ms: Some(args.elapsed_in_state_ms),
        state1_contact_delay_ms: Some(STATE1_CONTACT_DELAY_MS),
        state1_timeout_ms: Some(timeout),
        state5_exit_ms: None,
        closure_status: "runtime_validated_state1_contact_and_timeout_subset",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(state_before: i64, elapsed: f64) -> RailCaptureSelectorArgs {
        RailCaptureSelectorArgs {
            state_before,
            elapsed_in_state_ms: elapsed,
            ..Default::default()
        }
    }

    #[test]
    fn state_validation() {
        assert_eq!(
            evaluate_rail_capture_selector(&args(4, 0.0)).unwrap_err(),
            "recovered rail capture selector supports state 1, 2, 3, 5, or 6"
        );
        assert_eq!(
            evaluate_rail_capture_selector(&args(1, -1.0)).unwrap_err(),
            "elapsed_in_state_ms must be nonnegative"
        );
    }

    #[test]
    fn state5_timeout_exit() {
        let waiting = evaluate_rail_capture_selector(&args(5, 999.0)).unwrap();
        assert_eq!(waiting.state_after, 5);
        assert_eq!(waiting.reason, "state5_timer_active");
        let exits = evaluate_rail_capture_selector(&args(5, 1000.0)).unwrap();
        assert_eq!(exits.state_after, 0);
        assert!(exits.transitioned);
        assert_eq!(exits.reason, "state5_timeout_exit");
    }

    #[test]
    fn state6_binary32_seconds_countdown() {
        // Python authority: a single exact-100 ms delta lands on exactly
        // 0.0 (not negative), so the timer holds; one more 1 ms delta with a
        // valid candidate then exits with raw -1.0000000474974513 ms.
        let mut arguments = args(6, 0.0);
        arguments.delta_ms = Some(100.0);
        arguments.state6_contact_remaining_ms = Some(STATE6_CONTACT_CONFIRM_MS);
        arguments.state6_contact_candidate_valid = true;
        let held = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(held.state_after, 6);
        assert_eq!(held.reason, "state6_waiting_for_contact_confirmation");
        match held.runtime_after.unwrap() {
            RailCaptureSelectorRuntimeAfter::State6(runtime) => {
                assert_eq!(runtime.state6_contact_remaining_ms, 0.0);
                assert_eq!(runtime.state6_contact_raw_after_ms, 0.0);
            }
            _ => panic!("expected state6 runtime_after"),
        }
        // The documented exact-100 ms total split across independent frame
        // deltas (60 + 40): f64 arithmetic would leave +1.49e-6 ms; the
        // binary32 seconds subtraction exits at raw -3.725290298461914e-6.
        let mut first = args(6, 0.0);
        first.delta_ms = Some(60.0);
        first.state6_contact_remaining_ms = Some(STATE6_CONTACT_CONFIRM_MS);
        first.state6_contact_candidate_valid = true;
        let first = evaluate_rail_capture_selector(&first).unwrap();
        let RailCaptureSelectorRuntimeAfter::State6(runtime) =
            first.runtime_after.clone().unwrap()
        else {
            panic!("expected state6 runtime_after")
        };
        let mut second = args(6, 0.0);
        second.delta_ms = Some(40.0);
        second.state6_contact_remaining_ms = Some(runtime.state6_contact_remaining_ms);
        second.state6_contact_candidate_valid = true;
        let second = evaluate_rail_capture_selector(&second).unwrap();
        assert_eq!(second.state_after, 0);
        assert_eq!(second.reason, "state6_contact_confirmation_elapsed");
        match second.runtime_after.unwrap() {
            RailCaptureSelectorRuntimeAfter::State6(runtime) => {
                assert_eq!(runtime.state6_contact_raw_after_ms, -3.725290298461914e-6);
            }
            _ => panic!("expected state6 runtime_after"),
        }
        // Without a valid candidate the timer holds.
        let mut arguments = args(6, 0.0);
        arguments.delta_ms = Some(100.0);
        arguments.state6_contact_remaining_ms = Some(STATE6_CONTACT_CONFIRM_MS);
        arguments.state6_contact_candidate_valid = false;
        let held = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(held.state_after, 6);
    }

    #[test]
    fn state2_alignment_promotion() {
        let mut arguments = args(2, 0.0);
        arguments.delta_ms = Some(50.0);
        arguments.alignment_primary_remaining_ms = Some(40.0);
        arguments.alignment_stable_remaining_ms = Some(30.0);
        arguments.forward_speed = Some(10.0);
        arguments.lateral_speed = Some(1.0);
        let result = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(result.state_after, 3);
        assert_eq!(result.reason, "state2_alignment_timers_elapsed");
        // Stable timer freezes while alignment fails.
        arguments.alignment_primary_remaining_ms = Some(100.0);
        arguments.lateral_speed = Some(9.0);
        let held = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(held.state_after, 2);
        match held.runtime_after.unwrap() {
            RailCaptureSelectorRuntimeAfter::State2(runtime) => {
                assert_eq!(runtime.alignment_primary_remaining_ms, 50.0);
                assert_eq!(runtime.alignment_stable_remaining_ms, 30.0);
            }
            _ => panic!("expected state2 runtime_after"),
        }
    }

    #[test]
    fn state3_trigger_edge_and_cooldown_reset() {
        let mut arguments = args(3, 0.0);
        arguments.delta_ms = Some(20.0);
        arguments.trigger_cooldown_remaining_ms = Some(20.0);
        arguments.authored_trigger_accepted = true;
        let result = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(result.state_after, 6);
        match result.runtime_after.unwrap() {
            RailCaptureSelectorRuntimeAfter::State3(runtime) => {
                assert_eq!(runtime.trigger_cooldown_remaining_ms, SPECIAL_TRIGGER_COOLDOWN_MS);
            }
            _ => panic!("expected state3 runtime_after"),
        }
        // Cooldown still running: no promotion even with an accepted trigger.
        arguments.trigger_cooldown_remaining_ms = Some(20.1);
        let held = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(held.state_after, 3);
    }

    #[test]
    fn state1_contact_and_timeout_subset() {
        // Contact path acceptance.
        let mut arguments = args(1, 10.0);
        arguments.contact_exists = true;
        arguments.contact_identifier_nonempty = true;
        arguments.path_metric = Some(0.5);
        arguments.metric_threshold = Some(1.0);
        arguments.path_z = Some(0.0);
        arguments.kart_z_plus_one = Some(1.0);
        let accepted = evaluate_rail_capture_selector(&arguments).unwrap();
        assert_eq!(accepted.state_after, 2);
        assert_eq!(accepted.reason, "state1_contact_path_accepted");
        // Missing path fields on an otherwise-accepting contact: ValueError.
        arguments.path_metric = None;
        assert_eq!(
            evaluate_rail_capture_selector(&arguments).unwrap_err(),
            "accepted rail contact requires path metric/threshold/height fields"
        );
        // Timeout bypass (Python authority: timeout is 11618 ms at
        // field_af0 = 0, so 12000 ms elapses it).
        let mut timeout_args = args(1, 12000.0);
        timeout_args.field_af0 = 0.0;
        let timeout = evaluate_rail_capture_selector(&timeout_args).unwrap();
        assert_eq!(timeout.state_after, 5);
        assert_eq!(timeout.reason, "state1_timeout_bypass");
        assert_eq!(timeout.state1_timeout_ms, Some(11618.0));
        // Waiting.
        let waiting = evaluate_rail_capture_selector(&args(1, 5.0)).unwrap();
        assert_eq!(waiting.state_after, 1);
        assert_eq!(waiting.reason, "state1_waiting");
        assert!(!waiting.transitioned);
    }
}
