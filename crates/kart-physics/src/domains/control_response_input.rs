//! Port of `scripts/evaluate_control_response_input.py` — raw key input
//! adapter backed by the quality-gated trace response model.
//!
//! The production model is `analysis/input_response/trace_response_model.json`
//! (`verdict == "PASS"`). All arithmetic is plain f64 in Python's evaluation
//! order; the module performs no float32 quantization.

use std::collections::BTreeMap;

use crate::num::sum_f64;

/// Python prediction key `heading_rate_scope`.
pub const HEADING_RATE_SCOPE: &str = "empirical prior only for LEFT/RIGHT actions; straight/coast actions are forced to zero rather than inheriting trace yaw noise";
/// Python prediction key `scope`.
pub const PREDICT_SCOPE: &str = "quality-gated empirical input response used as an upstream accumulator producer; exact internal branch evaluator remains authoritative downstream";

/// Python `speed_band(speed)`.
pub fn speed_band(speed: f64) -> &'static str {
    if speed < 0.2 {
        "stationary"
    } else if speed < 2.0 {
        "low_0_2"
    } else if speed < 5.0 {
        "low_mid_2_5"
    } else if speed < 10.0 {
        "mid_5_10"
    } else {
        "high_10_15"
    }
}

/// Python `canonical_action(keys)`.
pub fn canonical_action(keys: &[String]) -> String {
    let active: std::collections::BTreeSet<String> =
        keys.iter().map(|key| key.to_uppercase()).collect();
    if active.contains("UP") && active.contains("LEFT") {
        return "UP+LEFT".to_string();
    }
    if active.contains("UP") && active.contains("RIGHT") {
        return "UP+RIGHT".to_string();
    }
    for key in ["UP", "DOWN", "LEFT", "RIGHT"] {
        if active.contains(key) {
            return key.to_string();
        }
    }
    "COAST".to_string()
}

/// One `response_curves.<action>.<band>` entry.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ResponseCurve {
    pub accel_mps2_median: f64,
    pub heading_rate_rad_s_median: f64,
}

/// Python `ControlResponseInputModel.predict(...)` result dict.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlModelPrediction {
    pub action: String,
    pub speed_band: &'static str,
    pub longitudinal_accel_mps2: f64,
    pub heading_rate_rad_s: f64,
    pub steering_selected: bool,
    pub acceleration_world: [f64; 3],
    pub force_accumulator_614: [f64; 3],
    pub source: String,
}

/// Python `ControlResponseInputModel`. Only the fields the engine reads are
/// retained: `vehicle_model.coast_decel` (default 0.0) and
/// `vehicle_model.response_curves`.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlResponseInputModel {
    pub coast_decel: f64,
    pub response_curves: BTreeMap<String, BTreeMap<String, ResponseCurve>>,
}

/// Serde view of the model JSON payload (`verdict` +
/// `vehicle_model.{coast_decel,response_curves}`). `verdict` is `Option` so
/// a missing key takes the same "must have verdict PASS" error as a
/// non-PASS value (Python `payload.get("verdict") != "PASS"`).
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct ModelPayload {
    verdict: Option<String>,
    vehicle_model: VehicleModelPayload,
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct VehicleModelPayload {
    #[serde(default)]
    coast_decel: f64,
    response_curves: BTreeMap<String, BTreeMap<String, ResponseCurve>>,
}

impl ControlResponseInputModel {
    /// Build from already-decoded fields, applying the Python verdict gate.
    pub fn new(
        verdict: Option<&str>,
        coast_decel: f64,
        response_curves: BTreeMap<String, BTreeMap<String, ResponseCurve>>,
    ) -> Result<Self, String> {
        if verdict != Some("PASS") {
            return Err("trace response model must have verdict PASS".to_string());
        }
        Ok(ControlResponseInputModel {
            coast_decel,
            response_curves,
        })
    }

    /// `ControlResponseInputModel(payload)` from a JSON document.
    #[cfg(feature = "serde")]
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let payload: ModelPayload = serde_json::from_str(text)
            .map_err(|error| format!("failed to parse trace response model: {error}"))?;
        ControlResponseInputModel::new(
            payload.verdict.as_deref(),
            payload.vehicle_model.coast_decel,
            payload.vehicle_model.response_curves,
        )
    }

    /// `ControlResponseInputModel.from_path(path)`.
    #[cfg(feature = "serde")]
    pub fn from_path(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("failed to read trace response model {}: {error}", path.display()))?;
        ControlResponseInputModel::from_json_str(&text)
    }

    /// `predict(keys=..., speed=..., forward_axis=..., sub_660=...)`.
    pub fn predict(
        &self,
        keys: &[String],
        speed: f64,
        forward_axis: [f64; 3],
        sub_660: f64,
    ) -> Result<ControlModelPrediction, String> {
        if sub_660 == 0.0 {
            return Err(
                "forward_axis must have 3 components and sub_660 must be nonzero".to_string(),
            );
        }
        let length = sum_f64(&[
            forward_axis[0] * forward_axis[0],
            forward_axis[1] * forward_axis[1],
            forward_axis[2] * forward_axis[2],
        ])
        .sqrt();
        if length == 0.0 {
            return Err("forward_axis must be nonzero".to_string());
        }
        let action = canonical_action(keys);
        let band = speed_band(speed);
        let (accel, mut heading_rate, source) = if action == "COAST" {
            let accel = if speed > 0.0 { -self.coast_decel } else { 0.0 };
            (accel, 0.0, "vehicle_model.coast_decel".to_string())
        } else {
            let curve = self
                .response_curves
                .get(&action)
                .and_then(|bands| bands.get(band))
                .ok_or_else(|| {
                    format!("no accepted response curve for action={action} band={band}")
                })?;
            (
                curve.accel_mps2_median,
                curve.heading_rate_rad_s_median,
                format!("response_curves.{action}.{band}"),
            )
        };
        let steering_selected = action.contains("LEFT") || action.contains("RIGHT");
        if !steering_selected {
            // Straight/coast traces can contain yaw wrapping, reset motion, or
            // residual angular velocity. They are not steering commands and
            // must not become a synthetic steering prior.
            heading_rate = 0.0;
        }
        let axis = [
            forward_axis[0] / length,
            forward_axis[1] / length,
            forward_axis[2] / length,
        ];
        let acceleration_world = [axis[0] * accel, axis[1] * accel, axis[2] * accel];
        Ok(ControlModelPrediction {
            action,
            speed_band: band,
            longitudinal_accel_mps2: accel,
            heading_rate_rad_s: heading_rate,
            steering_selected,
            acceleration_world,
            force_accumulator_614: [
                acceleration_world[0] * sub_660,
                acceleration_world[1] * sub_660,
                acceleration_world[2] * sub_660,
            ],
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| (*name).to_string()).collect()
    }

    fn model() -> ControlResponseInputModel {
        let mut bands = BTreeMap::new();
        bands.insert(
            "mid_5_10".to_string(),
            ResponseCurve {
                accel_mps2_median: 2.5,
                heading_rate_rad_s_median: 0.75,
            },
        );
        let mut curves = BTreeMap::new();
        curves.insert("UP".to_string(), bands.clone());
        curves.insert("UP+LEFT".to_string(), bands);
        ControlResponseInputModel::new(Some("PASS"), 1.25, curves).unwrap()
    }

    #[test]
    fn canonical_action_priority_matches_python() {
        assert_eq!(canonical_action(&keys(&["up", "left"])), "UP+LEFT");
        assert_eq!(canonical_action(&keys(&["UP", "RIGHT"])), "UP+RIGHT");
        assert_eq!(canonical_action(&keys(&["DOWN", "LEFT"])), "DOWN");
        assert_eq!(canonical_action(&keys(&["RIGHT"])), "RIGHT");
        assert_eq!(canonical_action(&keys(&[])), "COAST");
    }

    #[test]
    fn speed_band_edges_match_python() {
        assert_eq!(speed_band(0.0), "stationary");
        assert_eq!(speed_band(0.2), "low_0_2");
        assert_eq!(speed_band(1.99), "low_0_2");
        assert_eq!(speed_band(2.0), "low_mid_2_5");
        assert_eq!(speed_band(5.0), "mid_5_10");
        assert_eq!(speed_band(10.0), "high_10_15");
    }

    #[test]
    fn predict_straight_forces_zero_heading_rate() {
        // Computed with the Python authority:
        //   model.predict(keys=["UP"], speed=7.5, forward_axis=[0.6, -0.8, 0.0],
        //                 sub_660=100.0)
        let prediction = model()
            .predict(&keys(&["UP"]), 7.5, [0.6, -0.8, 0.0], 100.0)
            .unwrap();
        assert_eq!(prediction.action, "UP");
        assert_eq!(prediction.speed_band, "mid_5_10");
        assert_eq!(prediction.longitudinal_accel_mps2, 2.5);
        assert_eq!(prediction.heading_rate_rad_s, 0.0);
        assert!(!prediction.steering_selected);
        assert_eq!(prediction.acceleration_world, [1.5, -2.0, 0.0]);
        assert_eq!(prediction.force_accumulator_614, [150.0, -200.0, 0.0]);
        assert_eq!(prediction.source, "response_curves.UP.mid_5_10");
    }

    #[test]
    fn predict_steering_keeps_empirical_heading_rate() {
        let prediction = model()
            .predict(&keys(&["UP", "LEFT"]), 7.5, [0.6, -0.8, 0.0], 100.0)
            .unwrap();
        assert_eq!(prediction.action, "UP+LEFT");
        assert!(prediction.steering_selected);
        assert_eq!(prediction.heading_rate_rad_s, 0.75);
        assert_eq!(prediction.source, "response_curves.UP+LEFT.mid_5_10");
    }

    #[test]
    fn predict_coast_uses_coast_decel() {
        // model.predict(keys=[], speed=3.0, forward_axis=[1.0, 0.0, 0.0],
        //               sub_660=100.0)
        let prediction = model()
            .predict(&keys(&[]), 3.0, [1.0, 0.0, 0.0], 100.0)
            .unwrap();
        assert_eq!(prediction.action, "COAST");
        assert_eq!(prediction.speed_band, "low_mid_2_5");
        assert_eq!(prediction.longitudinal_accel_mps2, -1.25);
        assert_eq!(prediction.acceleration_world, [-1.25, -0.0, -0.0]);
        assert_eq!(prediction.source, "vehicle_model.coast_decel");
        // Stationary coast has no decel.
        let stopped = model()
            .predict(&keys(&[]), 0.0, [1.0, 0.0, 0.0], 100.0)
            .unwrap();
        assert_eq!(stopped.longitudinal_accel_mps2, 0.0);
    }

    #[test]
    fn error_guards_match_python_messages() {
        assert_eq!(
            model().predict(&keys(&[]), 1.0, [1.0, 0.0, 0.0], 0.0).unwrap_err(),
            "forward_axis must have 3 components and sub_660 must be nonzero"
        );
        assert_eq!(
            model().predict(&keys(&[]), 1.0, [0.0, 0.0, 0.0], 1.0).unwrap_err(),
            "forward_axis must be nonzero"
        );
        assert_eq!(
            model().predict(&keys(&["UP"]), 0.1, [1.0, 0.0, 0.0], 1.0).unwrap_err(),
            "no accepted response curve for action=UP band=stationary"
        );
        assert_eq!(
            ControlResponseInputModel::new(Some("FAIL"), 0.0, BTreeMap::new()).unwrap_err(),
            "trace response model must have verdict PASS"
        );
        assert_eq!(
            ControlResponseInputModel::new(None, 0.0, BTreeMap::new()).unwrap_err(),
            "trace response model must have verdict PASS"
        );
    }
}
