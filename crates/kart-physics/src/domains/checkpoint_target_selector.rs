//! Port of `scripts/evaluate_checkpoint_target_selector.py` — ordered
//! saved-checkpoint target/gate writers for C016 and C72C modes.

// source provenance (Python result key `source`).
pub const SOURCE_C016_INIT_VA: &str = "0xC01D30";
pub const SOURCE_C016_PROGRESS_SNAPSHOT_VA: &str = "0xC01810";
pub const SOURCE_C016_GATE_WRITERS: [&str; 3] = ["0xC02760", "0xC03BD0", "0xC04E50"];
pub const SOURCE_C72C_INIT_VA: &str = "0xC70E30";
pub const SOURCE_C72C_PROGRESS_SNAPSHOT_VA: &str = "0xC72270";

/// Python result key `scope`.
pub const SCOPE: &str = "exact local writer ordering and C016 arm/250ms-clear predicates; event 25/26 identity remains caller-supplied to c016_state_event_arm";

/// Python `evaluate_checkpoint_target_selector(...)` result dict minus
/// `source`.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointTargetSelectorResult {
    pub mode_family: String,
    pub event: String,
    pub accepted: bool,
    pub target_before: [f64; 3],
    pub target_after: [f64; 3],
    pub target_written: bool,
    pub gate_before: bool,
    pub gate_after: bool,
    pub gate_written: bool,
    pub scope: &'static str,
}

/// `evaluate_checkpoint_target_selector(...)` — apply one verified
/// target/gate writer event in scheduler order. Component-count validation
/// in Python is unreachable here (fixed `[f64; 3]` inputs).
pub fn evaluate_checkpoint_target_selector(
    mode_family: &str,
    target_before: [f64; 3],
    gate_before: bool,
    event: &str,
    current_kart_position: Option<[f64; 3]>,
    config_gate: bool,
    mode_state: Option<i64>,
    mission_index: Option<i64>,
    mission_count: Option<i64>,
    retire_detected: bool,
    reset_suppressed: bool,
    trace_value: Option<f64>,
    trace_min: Option<f64>,
    trace_max: Option<f64>,
    peer_present: bool,
    kart_event_accepts: bool,
    stage_input_blocked: bool,
    tick_ms: Option<i64>,
    gate_armed_tick_ms: Option<i64>,
) -> Result<CheckpointTargetSelectorResult, String> {
    if mode_family != "c016" && mode_family != "c72c" {
        return Err("mode_family must be 'c016' or 'c72c'".to_string());
    }
    let c016_events = [
        "c016_retire_arm",
        "c016_trace_bounds_arm",
        "c016_state_event_arm",
        "c016_timeout_clear",
    ];
    if event != "mode_init" && event != "mission_progress" && !c016_events.contains(&event) {
        return Err("unsupported checkpoint target event".to_string());
    }
    if c016_events.contains(&event) && mode_family != "c016" {
        return Err(format!("{event} is valid only for c016"));
    }
    let mut target = target_before;
    let mut gate = gate_before;
    let mut target_written = false;
    let mut gate_written = false;
    let accepted;

    if event == "mode_init" {
        if mode_family == "c72c" {
            let Some(position) = current_kart_position else {
                return Err("c72c mode_init requires current_kart_position".to_string());
            };
            target = position;
            gate = config_gate;
        } else {
            target = [0.0, 0.0, 0.0];
            gate = false;
        }
        target_written = true;
        gate_written = true;
        accepted = true;
    } else if event == "mission_progress" {
        let required_state = if mode_family == "c016" { 5 } else { 1 };
        if current_kart_position.is_none()
            || mission_index.is_none()
            || mission_count.is_none()
            || mode_state.is_none()
        {
            return Err(
                "mission_progress requires position, mode_state, mission_index, and mission_count"
                    .to_string(),
            );
        }
        accepted = mode_state.expect("checked") == required_state
            && mission_index.expect("checked") - 1 <= mission_count.expect("checked");
        if accepted {
            target = current_kart_position.expect("checked");
            target_written = true;
        }
    } else if event == "c016_retire_arm" {
        accepted = retire_detected && !reset_suppressed;
        if accepted {
            gate = true;
            gate_written = true;
        }
    } else if event == "c016_trace_bounds_arm" {
        if trace_value.is_none()
            || trace_min.is_none()
            || trace_max.is_none()
            || mode_state.is_none()
        {
            return Err("trace arm requires value, min, max, and mode_state".to_string());
        }
        let trace_value = trace_value.expect("checked");
        let trace_min = trace_min.expect("checked");
        let trace_max = trace_max.expect("checked");
        accepted = peer_present
            && mode_state.expect("checked") == 2
            && !(trace_min <= trace_value && trace_value <= trace_max);
        if accepted {
            gate = true;
            gate_written = true;
        }
    } else if event == "c016_state_event_arm" {
        accepted = mode_state.unwrap_or(0) == 2
            && !gate
            && kart_event_accepts
            && !stage_input_blocked;
        if accepted {
            gate = true;
            gate_written = true;
        }
    } else {
        if tick_ms.is_none() || gate_armed_tick_ms.is_none() {
            return Err("timeout clear requires tick_ms and gate_armed_tick_ms".to_string());
        }
        accepted =
            gate && tick_ms.expect("checked") - gate_armed_tick_ms.expect("checked") >= 250;
        if accepted {
            gate = false;
            gate_written = true;
        }
    }

    Ok(CheckpointTargetSelectorResult {
        mode_family: mode_family.to_string(),
        event: event.to_string(),
        accepted,
        target_before,
        target_after: target,
        target_written,
        gate_before,
        gate_after: gate,
        gate_written,
        scope: SCOPE,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_family_validation() {
        assert_eq!(
            evaluate_checkpoint_target_selector(
                "c999", [0.0; 3], false, "mode_init", None, false, None, None, None, false,
                false, None, None, None, false, false, false, None, None,
            )
            .unwrap_err(),
            "mode_family must be 'c016' or 'c72c'"
        );
        assert_eq!(
            evaluate_checkpoint_target_selector(
                "c72c", [0.0; 3], false, "c016_retire_arm", None, false, None, None, None,
                true, false, None, None, None, false, false, false, None, None,
            )
            .unwrap_err(),
            "c016_retire_arm is valid only for c016"
        );
        assert_eq!(
            evaluate_checkpoint_target_selector(
                "c016", [0.0; 3], false, "bogus", None, false, None, None, None, false,
                false, None, None, None, false, false, false, None, None,
            )
            .unwrap_err(),
            "unsupported checkpoint target event"
        );
    }

    #[test]
    fn mode_init_and_mission_progress() {
        // Python authority:
        //   evaluate_checkpoint_target_selector(mode_family="c72c", event="mode_init",
        //     target_before=[1,2,3], gate_before=True,
        //     current_kart_position=[4,5,6], config_gate=True)
        //   -> accepted, target=[4,5,6], gate=True, both written.
        let r = evaluate_checkpoint_target_selector(
            "c72c",
            [1.0, 2.0, 3.0],
            true,
            "mode_init",
            Some([4.0, 5.0, 6.0]),
            true,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
        )
        .unwrap();
        assert!(r.accepted && r.target_written && r.gate_written);
        assert_eq!(r.target_after, [4.0, 5.0, 6.0]);
        assert!(r.gate_after);
        //   c016 mission_progress: mode_state 5 and mission_index-1 <= count
        //   snapshots the position without touching the gate.
        let r = evaluate_checkpoint_target_selector(
            "c016",
            [1.0, 2.0, 3.0],
            false,
            "mission_progress",
            Some([7.0, 8.0, 9.0]),
            false,
            Some(5),
            Some(3),
            Some(2),
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false,
            None,
            None,
        )
        .unwrap();
        assert!(r.accepted && r.target_written);
        assert!(!r.gate_written);
        assert_eq!(r.target_after, [7.0, 8.0, 9.0]);
    }

    #[test]
    fn timeout_clear_threshold() {
        // Python authority: accepted iff gate armed and elapsed >= 250 ms.
        let r = evaluate_checkpoint_target_selector(
            "c016",
            [0.0; 3],
            true,
            "c016_timeout_clear",
            None,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false,
            Some(1250),
            Some(1000),
        )
        .unwrap();
        assert!(r.accepted);
        assert!(!r.gate_after);
        let r = evaluate_checkpoint_target_selector(
            "c016",
            [0.0; 3],
            true,
            "c016_timeout_clear",
            None,
            false,
            None,
            None,
            None,
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false,
            Some(1249),
            Some(1000),
        )
        .unwrap();
        assert!(!r.accepted);
        assert!(r.gate_after);
    }
}
