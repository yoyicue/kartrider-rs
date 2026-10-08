//! Rail runtime context (port of `RailRuntimeContext` in
//! `scripts/kartrider_vehicle_engine.py:600-632`).

/// `RailRuntimeContext`: decoded rail-relative fields that are not derivable
/// from rigid-body state.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RailRuntimeContext {
    pub rail_dir: [f64; 3],
    pub accel_be0: f64,
    pub obf_6e8: f64,
    pub grip_720: f64,
    pub gravity_be8: f64,
    pub gate_1c0_decoded: bool,
    pub boost_special_gate_9b8: bool,
    pub boost_special_gate_6e4: bool,
    pub attached_object_state: i64,
    pub obf_808: f64,
    pub obf_814: f64,
    pub min_velocity: Option<f64>,
    pub max_velocity: Option<f64>,
}

impl Default for RailRuntimeContext {
    /// Python defaults for the defaulted fields; the four required
    /// constructor arguments (`rail_dir`, `accel_be0`, `obf_6e8`, `grip_720`)
    /// are zero-filled and fail `validate` until set, mirroring the Python
    /// behavior of requiring them at construction time.
    fn default() -> Self {
        RailRuntimeContext {
            rail_dir: [0.0; 3],
            accel_be0: 0.0,
            obf_6e8: 0.0,
            grip_720: 0.0,
            gravity_be8: 1.0,
            gate_1c0_decoded: false,
            boost_special_gate_9b8: false,
            boost_special_gate_6e4: false,
            attached_object_state: 0,
            obf_808: 1.0,
            obf_814: 1.0,
            min_velocity: None,
            max_velocity: None,
        }
    }
}

impl RailRuntimeContext {
    /// Port of `RailRuntimeContext.validate`
    /// (kartrider_vehicle_engine.py:618-632), `ValueError` -> `Err(String)`.
    /// The three-component `rail_dir` length check is type-enforced.
    pub fn validate(&self) -> Result<(), String> {
        // Python: math.sqrt(sum(float(value) ** 2 for value in self.rail_dir))
        // (kartrider_vehicle_engine.py:621) — CPython sum() is
        // Neumaier-compensated, so this is NOT a naive fold (see
        // crate::num::sum_f64).
        let magnitude = crate::num::sum_f64(&[
            self.rail_dir[0] * self.rail_dir[0],
            self.rail_dir[1] * self.rail_dir[1],
            self.rail_dir[2] * self.rail_dir[2],
        ])
        .sqrt();
        if !(0.999..=1.001).contains(&magnitude) {
            return Err("rail_dir must be normalized".to_string());
        }
        if let Some(min_velocity) = self.min_velocity {
            if min_velocity <= 0.0 {
                return Err(
                    "rail min_velocity must be positive when supplied".to_string()
                );
            }
        }
        if let Some(max_velocity) = self.max_velocity {
            if self.min_velocity.is_none() || max_velocity < self.min_velocity.unwrap() {
                return Err(
                    "rail max_velocity requires min_velocity and cannot be lower"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> RailRuntimeContext {
        RailRuntimeContext {
            rail_dir: [0.0, 1.0, 0.0],
            accel_be0: 4608.0,
            obf_6e8: 1.0,
            grip_720: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn defaults_match_python_dataclass() {
        let default = RailRuntimeContext::default();
        assert_eq!(default.gravity_be8, 1.0);
        assert!(!default.gate_1c0_decoded);
        assert!(!default.boost_special_gate_9b8);
        assert!(!default.boost_special_gate_6e4);
        assert_eq!(default.attached_object_state, 0);
        assert_eq!(default.obf_808, 1.0);
        assert_eq!(default.obf_814, 1.0);
        assert_eq!(default.min_velocity, None);
        assert_eq!(default.max_velocity, None);
        context().validate().unwrap();
    }

    #[test]
    fn validate_rejects_unnormalized_rail_dir() {
        assert_eq!(
            RailRuntimeContext::default().validate(),
            Err("rail_dir must be normalized".to_string())
        );
        let mut oversized = context();
        oversized.rail_dir = [0.0, 1.01, 0.0];
        assert_eq!(
            oversized.validate(),
            Err("rail_dir must be normalized".to_string())
        );
        // Boundary magnitudes 0.999 / 1.001 are accepted.
        let mut lower = context();
        lower.rail_dir = [0.0, 0.999, 0.0];
        lower.validate().unwrap();
        let mut upper = context();
        upper.rail_dir = [0.0, 1.001, 0.0];
        upper.validate().unwrap();
    }

    #[test]
    fn validate_velocity_window() {
        let mut zero_min = context();
        zero_min.min_velocity = Some(0.0);
        assert_eq!(
            zero_min.validate(),
            Err("rail min_velocity must be positive when supplied".to_string())
        );
        let mut max_without_min = context();
        max_without_min.max_velocity = Some(100.0);
        assert_eq!(
            max_without_min.validate(),
            Err("rail max_velocity requires min_velocity and cannot be lower"
                .to_string())
        );
        let mut inverted = context();
        inverted.min_velocity = Some(100.0);
        inverted.max_velocity = Some(50.0);
        assert_eq!(
            inverted.validate(),
            Err("rail max_velocity requires min_velocity and cannot be lower"
                .to_string())
        );
        let mut window = context();
        window.min_velocity = Some(83.3333359);
        window.max_velocity = Some(83.3333359);
        window.validate().unwrap();
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    fn base() -> RailRuntimeContext {
        RailRuntimeContext {
            rail_dir: [0.0, 1.0, 0.0],
            accel_be0: 4608.0,
            obf_6e8: 1.0,
            grip_720: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn rail_runtime_context_round_trip() {
        let context = RailRuntimeContext {
            min_velocity: Some(83.3333359),
            max_velocity: Some(120.0),
            ..base()
        };
        let json = serde_json::to_value(&context).unwrap();
        assert_eq!(json["gravity_be8"], 1.0);
        assert_eq!(json["obf_808"], 1.0);
        assert_eq!(json["min_velocity"], 83.3333359);
        assert_eq!(json["max_velocity"], 120.0);
        let round_trip: RailRuntimeContext = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, context);
        round_trip.validate().unwrap();
        // None velocity window serializes as null.
        let json = serde_json::to_value(base()).unwrap();
        assert_eq!(json["min_velocity"], serde_json::Value::Null);
        assert_eq!(json["max_velocity"], serde_json::Value::Null);
    }
}
