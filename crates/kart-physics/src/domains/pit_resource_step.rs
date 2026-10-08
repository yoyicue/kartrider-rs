//! Port of `scripts/evaluate_pit_resource_step.py` — exact pit-surface
//! resource step and its non-dynamics boundary.

/// Python result key `dynamics_role`.
pub const DYNAMICS_ROLE: &str = "none_proven_resource_and_fx_only";

/// Python result key `drain_consumer`.
pub const DRAIN_CONSUMER: &str = "race_mode_progress_B942D0";

/// Python `evaluate_pit_resource_step(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PitResourceStep {
    pub active_before_cleanup: bool,
    pub rate_before_cleanup: f64,
    pub mode_before_cleanup: i64,
    pub accum_before: f64,
    pub accum_increment: f64,
    pub accum_after: f64,
    pub fx_selected: bool,
    pub active_after_cleanup: bool,
    pub rate_after_cleanup: f64,
    pub mode_after_cleanup: i64,
    pub vehicle_force_delta: [f64; 3],
    pub vehicle_torque_delta: [f64; 3],
    pub dynamics_role: &'static str,
    pub drain_consumer: &'static str,
}

/// `evaluate_pit_resource_step(...)` — model UpdateTick's pit accumulator,
/// FX snapshot, and end-tick clear.
pub fn evaluate_pit_resource_step(
    active: bool,
    rate: f64,
    mode: i64,
    accum_before: f64,
    dt: f64,
) -> Result<PitResourceStep, String> {
    if dt < 0.0 {
        return Err("dt must be non-negative".to_string());
    }
    if !(0..=2).contains(&mode) {
        return Err("pit mode must be 0, 1, or 2".to_string());
    }
    let increment = if active { dt * rate } else { 0.0 };
    Ok(PitResourceStep {
        active_before_cleanup: active,
        rate_before_cleanup: rate,
        mode_before_cleanup: mode,
        accum_before,
        accum_increment: increment,
        accum_after: accum_before + increment,
        fx_selected: active,
        active_after_cleanup: false,
        rate_after_cleanup: 0.0,
        mode_after_cleanup: 2,
        vehicle_force_delta: [0.0, 0.0, 0.0],
        vehicle_torque_delta: [0.0, 0.0, 0.0],
        dynamics_role: DYNAMICS_ROLE,
        drain_consumer: DRAIN_CONSUMER,
    })
}

/// Python `drain_pit_accumulator(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct PitAccumulatorDrain {
    pub drained: f64,
    pub accum_after: f64,
    pub race_progress_after: f64,
    pub vehicle_state_mutated: bool,
}

/// `drain_pit_accumulator(...)` — model B24BA0's read-and-zero followed by
/// B942D0's progress add.
pub fn drain_pit_accumulator(accum_before: f64, race_progress_before: f64) -> PitAccumulatorDrain {
    let drained = accum_before;
    PitAccumulatorDrain {
        drained,
        accum_after: 0.0,
        race_progress_after: race_progress_before + drained,
        vehicle_state_mutated: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_step_accumulates_then_cleans_up() {
        // Constants computed with the Python authority:
        //   evaluate_pit_resource_step(active=True, rate=0.3, mode=1,
        //     accum_before=0.5, dt=0.016)
        let r = evaluate_pit_resource_step(true, 0.3, 1, 0.5, 0.016).unwrap();
        assert_eq!(r.accum_increment, 0.3 * 0.016);
        assert_eq!(r.accum_after, 0.5 + 0.3 * 0.016);
        assert!(r.fx_selected);
        assert!(!r.active_after_cleanup);
        assert_eq!(r.rate_after_cleanup, 0.0);
        assert_eq!(r.mode_after_cleanup, 2);
    }

    #[test]
    fn validation_errors() {
        assert_eq!(
            evaluate_pit_resource_step(true, 0.3, 1, 0.5, -0.001).unwrap_err(),
            "dt must be non-negative"
        );
        assert_eq!(
            evaluate_pit_resource_step(true, 0.3, 3, 0.5, 0.016).unwrap_err(),
            "pit mode must be 0, 1, or 2"
        );
    }

    #[test]
    fn inactive_step_adds_exactly_zero() {
        // Python authority: inactive -> increment 0.0, accum unchanged.
        let r = evaluate_pit_resource_step(false, 9.0, 0, 1.25, 0.016).unwrap();
        assert_eq!(r.accum_increment, 0.0);
        assert_eq!(r.accum_after, 1.25);
        assert!(!r.fx_selected);
    }

    #[test]
    fn drain_reads_and_zeros() {
        // Python authority: drain_pit_accumulator(accum_before=1.25,
        //   race_progress_before=2.5) -> drained 1.25, progress 3.75.
        let r = drain_pit_accumulator(1.25, 2.5);
        assert_eq!(r.drained, 1.25);
        assert_eq!(r.accum_after, 0.0);
        assert_eq!(r.race_progress_after, 3.75);
        assert!(!r.vehicle_state_mutated);
    }
}
