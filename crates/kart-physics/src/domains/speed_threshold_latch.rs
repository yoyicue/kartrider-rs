//! Port of `scripts/evaluate_speed_threshold_latch.py` — evaluate the
//! +0xB8C speed-threshold latch at sub_B2D8D0.

use crate::num::sum_f64;

pub const ORDERING: &str = "after common damping producers and before velocity integration, clear +0xB8C when positive and lengthSq(+0xC0)>=B8C^2";

// static_source provenance (Python result key `static_source`).
pub const STATIC_FUNCTION_VA: &str = "0xB2D8D0";
pub const STATIC_BODY_VA: &str = "0xB2D8EB-0xB2D92E";

/// Python result dict minus `static_source`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeedThresholdLatchResult {
    pub threshold_b8c_before: f64,
    pub speed_sq: f64,
    pub threshold_sq: f64,
    pub clears: bool,
    pub threshold_b8c_after: f64,
    pub ordering: &'static str,
}

/// `evaluate_speed_threshold_latch(...)`.
pub fn evaluate_speed_threshold_latch(
    threshold_b8c: f64,
    velocity_c0: [f64; 3],
) -> SpeedThresholdLatchResult {
    // Python `sum(value * value ...)` is Neumaier-compensated
    // (crate::num::sum_f64).
    let speed_sq = sum_f64(&[
        velocity_c0[0] * velocity_c0[0],
        velocity_c0[1] * velocity_c0[1],
        velocity_c0[2] * velocity_c0[2],
    ]);
    let threshold_sq = threshold_b8c * threshold_b8c;
    let clears = threshold_b8c > 0.0 && speed_sq >= threshold_sq;
    SpeedThresholdLatchResult {
        threshold_b8c_before: threshold_b8c,
        speed_sq,
        threshold_sq,
        clears,
        threshold_b8c_after: if clears { 0.0 } else { threshold_b8c },
        ordering: ORDERING,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_equality_clears() {
        // Constants computed with the Python authority:
        //   evaluate_speed_threshold_latch(threshold_b8c=5.0, velocity_c0=[3,0,4])
        let r = evaluate_speed_threshold_latch(5.0, [3.0, 0.0, 4.0]);
        assert_eq!(r.speed_sq, 25.0);
        assert_eq!(r.threshold_sq, 25.0);
        assert!(r.clears);
        assert_eq!(r.threshold_b8c_after, 0.0);
    }

    #[test]
    fn non_positive_threshold_never_clears() {
        let r = evaluate_speed_threshold_latch(-3.0, [3.0, 0.0, 4.0]);
        assert!(!r.clears);
        assert_eq!(r.threshold_b8c_after, -3.0);
    }
}
