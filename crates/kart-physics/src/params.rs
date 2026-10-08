//! Vehicle parameters (port of `VehicleParameters` and
//! `_default_boost_parameters` in `scripts/kartrider_vehicle_engine.py`).
//!
//! Authority: `scripts/kartrider_vehicle_engine.py:454-597`. Field names and
//! evidence comments are preserved verbatim — this is a reverse-engineering
//! archive, not a tunable parameter set.

/// `_default_boost_parameters()` (kartrider_vehicle_engine.py:454-461) as a
/// struct with the exact dict key set.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct BoostParameters {
    pub correction_dir: [f64; 3],
    pub direction_gain: f64,
    pub selected_slot: f64,
    pub attached_state: i64,
    pub e8c: bool,
    pub efc: bool,
    pub flag_5bc: bool,
    pub raw_144: f64,
    pub obf_808: f64,
    pub obf_814: f64,
    pub obf_69c: f64,
    pub obf_684: f64,
    pub obf_8ac: f64,
    pub obf_8e8: f64,
    pub obf_94c: f64,
}

impl Default for BoostParameters {
    fn default() -> Self {
        BoostParameters {
            correction_dir: [1.0, 0.0, 0.0],
            direction_gain: 1.0,
            selected_slot: 0.0,
            attached_state: 0,
            e8c: false,
            efc: false,
            flag_5bc: false,
            raw_144: 0.0,
            obf_808: 1.0,
            obf_814: 1.0,
            obf_69c: 1.0,
            obf_684: 1.0,
            obf_8ac: 1.0,
            obf_8e8: 1.0,
            obf_94c: 0.0,
        }
    }
}

/// `VehicleParameters` (kartrider_vehicle_engine.py:464-549): run-invariant
/// vehicle parameters for autonomous raw-control stepping.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct VehicleParameters {
    pub substep_ms: i64,
    /// Some executable selectors are scheduled once per outer game frame even
    /// though their geometry predicates are sampled by every physics substep.
    /// ``None`` preserves direct-substep callers by using ``substep_ms``;
    /// frame-owned entry points publish the full frame delta at the recovered
    /// selector phase (the second writer when present) and zero elsewhere.
    pub topology_selector_delta_ms: Option<f64>,
    /// Initial highSpeed ownership is committed at the outer-frame sink, after
    /// substep-frequency entry-plane sampling has had a chance to latch a
    /// crossing. ``None`` preserves direct-substep callers; frame-owned entry
    /// points publish False for intermediate substeps and True for the final
    /// substep.
    pub initial_rail_capture_commit: Option<bool>,
    pub mass: f64,
    pub common_damping: f64,
    /// In the state6 common tail, the angular accumulator observed at kind36
    /// is exactly -30*CC while linear damping continues to use +0x700 = 3.
    /// This is distinct from B29350 kind25, which is absent in state6.
    /// Runtime kind36 observes the final -30*CC torque. B29E70's ordinary
    /// common damping already contributes -3*CC later in the same evaluator,
    /// so the state6-only residual producer must contribute -27*CC.
    pub state6_residual_angular_damping: f64,
    pub directional_damping: f64,
    pub damping_runtime_multiplier: f64,
    pub surface_mode: i64,
    pub forward_axis_column: i64,
    pub forward_axis_sign: f64,
    pub ground_projection_floor: f64,
    pub ground_angular_correction_scale: f64,
    pub primary_ground_direction_gain: f64,
    pub fallback_ground_direction_gain: f64,
    pub negative_adc_force_scale: f64,
    pub lateral_high_dot_scale: f64,
    pub lateral_low_dot_scale: f64,
    pub ground_force_multiplier: f64,
    pub ground_longitudinal_gain: f64,
    pub ground_lateral_gain: f64,
    pub ground_axis_long: f64,
    pub ground_axis_lat: f64,
    pub ground_settlement_multiplier: f64,
    pub ground_torque_y_gain: f64,
    pub drift_window_duration_734: f64,
    pub drift_gain_728: f64,
    pub drift_torque_y_gain_758: f64,
    pub drift_speed_profile_denominator: f64,
    pub drift_active_ab0_rate: f64,
    pub drift_active_ab4_rate: f64,
    pub drift_timer_ab0_rate: f64,
    /// Active abyss_I03 mode-2 rows use one fifth of the reset-table 0.5
    /// force scale.  The surface forward basis is AA4 times 180/100.
    pub generic_slip_force_scale: f64,
    /// Runtime mode-2 LEFT rows hold this steering/contact-basis magnitude
    /// independently of AA4. Eight consecutive rows resolve 0.1745316..9
    /// (approximately the configured 10-degree basis angle in radians); the
    /// same-epoch LEFT release resets the signed term to zero.
    pub generic_slip_steering_basis: f64,
    pub contact_rest_compression: f64,
    pub contact_spring: f64,
    pub contact_rebound: f64,
    pub contact_bump: f64,
    pub contact_compression_quantum: f64,
    /// Decoded GoPlayKart+0x4DC, read by B28DF3-B28E07 and applied to both
    /// wheel-probe span axes.  The clean runtime captures 0.800000012; the
    /// optional +0x21C multiplier is 1.0 in this configuration.
    pub contact_probe_span_scale: f64,
    /// Runtime +0x604 is already -58.8; sub_B29E70 multiplies it by +0x660
    /// (100), balancing four 2940*0.5 suspension slots at rest.
    pub contact_gravity_per_mass: f64,
    pub drive_tune_718: f64,
    pub drive_speed_scale_71c: f64,
    pub drive_attenuation_bb8: f64,
    /// GoPlayKart +0x9C4, sampled directly at the B2D940 angular integrator.
    /// The clean-kart runtime matrix is diagonal 0.12 (see
    /// analysis/goplaykart_angular_hw_samples.tsv).
    pub inverse_inertia_matrix: [f64; 9],
    pub boost: BoostParameters,
    /// Current fixcrt35 ordinary item6 setter argument, runtime-certified by
    /// run19/run20/run22. Item31/item60 remain outside this ODD.
    pub item6_boost_duration_ms: i64,
    /// Captured current-vehicle GoPlayKart+0x760 tune value.
    pub boost_gauge_raw_760: f64,
    /// Current-family raw gauge producer tune fields, runtime-certified by
    /// run25. Other vehicle profiles must supply their decoded values.
    pub boost_gauge_primary_rate_6b4: f64,
    pub boost_gauge_primary_override_100c: f64,
    pub boost_gauge_window_add_988: f64,
    pub boost_gauge_window_factor_994: f64,
    pub boost_gauge_speed_threshold_54c: f64,
    pub collision_half_extents: [f64; 3],
    /// Runtime probe_wall_hard origin is pose + 1.0*AC4; this is independent
    /// of the 0.7 vertical half extent.
    pub collision_center_up: f64,
    /// B276F0 builds its A5B840 state4/5/6 contact segment from
    /// position+A74*AC4 to position-A74*AC4.  All 407 run119 state-6 pose
    /// observations carry sub_a74=float32(0.5); the earlier 1.0 value was an
    /// unverified geometry-audit assumption and selected landing 20ms early.
    pub state6_contact_half_span: f64,
}

impl Default for VehicleParameters {
    fn default() -> Self {
        VehicleParameters {
            substep_ms: 2,
            topology_selector_delta_ms: None,
            initial_rail_capture_commit: None,
            mass: 100.0,
            common_damping: 3.0,
            state6_residual_angular_damping: 27.0,
            directional_damping: 0.666999996,
            damping_runtime_multiplier: 1.0,
            surface_mode: 0,
            forward_axis_column: 1,
            forward_axis_sign: -1.0,
            ground_projection_floor: 5.0,
            ground_angular_correction_scale: 0.5,
            primary_ground_direction_gain: 2304.0,
            fallback_ground_direction_gain: 1825.0,
            negative_adc_force_scale: 980.0,
            lateral_high_dot_scale: 2070.0,
            lateral_low_dot_scale: 1415.0,
            ground_force_multiplier: 980.0,
            ground_longitudinal_gain: 5.0,
            ground_lateral_gain: 5.0,
            ground_axis_long: 0.5,
            ground_axis_lat: 0.5,
            ground_settlement_multiplier: 0.254000008,
            ground_torque_y_gain: 0.00999999978,
            drift_window_duration_734: 0.25,
            drift_gain_728: 1.5,
            drift_torque_y_gain_758: 0.0599999987,
            drift_speed_profile_denominator: 340.0,
            drift_active_ab0_rate: 0.1,
            drift_active_ab4_rate: 0.2,
            drift_timer_ab0_rate: 0.05,
            generic_slip_force_scale: 0.1,
            generic_slip_steering_basis: 0.17453176,
            contact_rest_compression: 0.5,
            contact_spring: 2940.00024,
            contact_rebound: 0.0,
            contact_bump: 588.000061,
            contact_compression_quantum: 0.0,
            contact_probe_span_scale: 0.800000012,
            contact_gravity_per_mass: -58.8000031,
            drive_tune_718: 10.0,
            drive_speed_scale_71c: 24.6100006,
            drive_attenuation_bb8: 1.0,
            inverse_inertia_matrix: [
                0.119999997,
                0.0,
                0.0,
                0.0,
                0.119999997,
                0.0,
                0.0,
                0.0,
                0.119999997,
            ],
            boost: BoostParameters::default(),
            item6_boost_duration_ms: 2900,
            boost_gauge_raw_760: 3860.0,
            boost_gauge_primary_rate_6b4: 350.0,
            boost_gauge_primary_override_100c: 0.0,
            boost_gauge_window_add_988: 0.0,
            boost_gauge_window_factor_994: 0.0,
            boost_gauge_speed_threshold_54c: 160.0,
            collision_half_extents: [0.766791165, 0.817228198, 0.699999988],
            collision_center_up: 1.0,
            state6_contact_half_span: 0.5,
        }
    }
}

impl VehicleParameters {
    /// Port of `VehicleParameters.validate`
    /// (kartrider_vehicle_engine.py:551-597), `ValueError` -> `Err(String)`.
    /// The `inverse_inertia_matrix` nine-component check is type-enforced.
    pub fn validate(&self) -> Result<(), String> {
        // Python additionally enforces an integer-valued float; the Rust
        // field is an integer type, so only the positivity check remains.
        if self.substep_ms <= 0 {
            return Err("substep_ms must be a positive integer millisecond count".to_string());
        }
        if let Some(delta_ms) = self.topology_selector_delta_ms {
            if delta_ms < 0.0 {
                return Err("topology_selector_delta_ms must be nonnegative".to_string());
            }
        }
        if self.mass <= 0.0 {
            return Err("mass must be positive".to_string());
        }
        if self.state6_residual_angular_damping < 0.0 {
            return Err("state6_residual_angular_damping must be nonnegative".to_string());
        }
        if self
            .collision_half_extents
            .iter()
            .any(|value| *value <= 0.0)
        {
            return Err("collision_half_extents must contain three positive values".to_string());
        }
        if self.collision_center_up < 0.0 {
            return Err("collision_center_up must be non-negative".to_string());
        }
        if self.state6_contact_half_span <= 0.0 {
            return Err("state6_contact_half_span must be positive".to_string());
        }
        if self.item6_boost_duration_ms <= 0 {
            return Err("item6_boost_duration_ms must be a positive integer".to_string());
        }
        if !self.boost_gauge_raw_760.is_finite() || self.boost_gauge_raw_760 <= 0.0 {
            return Err("boost_gauge_raw_760 must be finite and positive".to_string());
        }
        for (name, value) in [
            (
                "boost_gauge_primary_rate_6b4",
                self.boost_gauge_primary_rate_6b4,
            ),
            (
                "boost_gauge_primary_override_100c",
                self.boost_gauge_primary_override_100c,
            ),
            (
                "boost_gauge_window_add_988",
                self.boost_gauge_window_add_988,
            ),
            (
                "boost_gauge_window_factor_994",
                self.boost_gauge_window_factor_994,
            ),
            (
                "boost_gauge_speed_threshold_54c",
                self.boost_gauge_speed_threshold_54c,
            ),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.boost_gauge_primary_rate_6b4 < 0.0
            || self.boost_gauge_primary_override_100c < 0.0
            || self.boost_gauge_speed_threshold_54c < 0.0
        {
            return Err(
                "boost gauge primary rates and threshold must be nonnegative".to_string(),
            );
        }
        if !(0..=2).contains(&self.forward_axis_column)
            || (self.forward_axis_sign != -1.0 && self.forward_axis_sign != 1.0)
        {
            return Err("forward axis requires a matrix column in 0..2 and sign +/-1".to_string());
        }
        if self.ground_projection_floor <= 0.0 {
            return Err("ground_projection_floor must be positive".to_string());
        }
        if self.drift_window_duration_734 <= 0.0 {
            return Err("drift_window_duration_734 must be positive".to_string());
        }
        if self.drift_speed_profile_denominator <= 0.0 {
            return Err("drift_speed_profile_denominator must be positive".to_string());
        }
        if self.drift_active_ab0_rate < 0.0
            || self.drift_active_ab4_rate < 0.0
            || self.drift_timer_ab0_rate < 0.0
        {
            return Err("drift blend rates must be nonnegative".to_string());
        }
        if self.contact_compression_quantum < 0.0 {
            return Err("contact_compression_quantum must be non-negative".to_string());
        }
        if self.contact_probe_span_scale <= 0.0 {
            return Err("contact_probe_span_scale must be positive".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python_dataclass() {
        let parameters = VehicleParameters::default();
        assert_eq!(parameters.substep_ms, 2);
        assert_eq!(parameters.mass, 100.0);
        assert_eq!(parameters.state6_residual_angular_damping, 27.0);
        assert_eq!(parameters.directional_damping, 0.666999996);
        assert_eq!(parameters.forward_axis_column, 1);
        assert_eq!(parameters.forward_axis_sign, -1.0);
        assert_eq!(parameters.generic_slip_steering_basis, 0.17453176);
        assert_eq!(parameters.contact_probe_span_scale, 0.800000012);
        assert_eq!(parameters.contact_gravity_per_mass, -58.8000031);
        assert_eq!(
            parameters.inverse_inertia_matrix,
            [
                0.119999997,
                0.0,
                0.0,
                0.0,
                0.119999997,
                0.0,
                0.0,
                0.0,
                0.119999997,
            ]
        );
        assert_eq!(
            parameters.collision_half_extents,
            [0.766791165, 0.817228198, 0.699999988]
        );
        assert_eq!(parameters.boost, BoostParameters::default());
        assert_eq!(parameters.boost.correction_dir, [1.0, 0.0, 0.0]);
        assert_eq!(parameters.boost.obf_808, 1.0);
        assert_eq!(parameters.boost.obf_94c, 0.0);
        assert_eq!(parameters.item6_boost_duration_ms, 2900);
        assert_eq!(parameters.boost_gauge_raw_760, 3860.0);
        assert_eq!(parameters.boost_gauge_primary_rate_6b4, 350.0);
        assert_eq!(parameters.boost_gauge_primary_override_100c, 0.0);
        assert_eq!(parameters.boost_gauge_window_add_988, 0.0);
        assert_eq!(parameters.boost_gauge_window_factor_994, 0.0);
        assert_eq!(parameters.boost_gauge_speed_threshold_54c, 160.0);
        parameters.validate().unwrap();
    }

    #[test]
    fn validate_rejects_python_cases() {
        let cases: [(VehicleParameters, &str); 11] = [
            (
                VehicleParameters {
                    substep_ms: 0,
                    ..Default::default()
                },
                "substep_ms must be a positive integer millisecond count",
            ),
            (
                VehicleParameters {
                    topology_selector_delta_ms: Some(-1.0),
                    ..Default::default()
                },
                "topology_selector_delta_ms must be nonnegative",
            ),
            (
                VehicleParameters {
                    mass: 0.0,
                    ..Default::default()
                },
                "mass must be positive",
            ),
            (
                VehicleParameters {
                    state6_residual_angular_damping: -1.0,
                    ..Default::default()
                },
                "state6_residual_angular_damping must be nonnegative",
            ),
            (
                VehicleParameters {
                    collision_half_extents: [1.0, 0.0, 1.0],
                    ..Default::default()
                },
                "collision_half_extents must contain three positive values",
            ),
            (
                VehicleParameters {
                    collision_center_up: -1.0,
                    ..Default::default()
                },
                "collision_center_up must be non-negative",
            ),
            (
                VehicleParameters {
                    state6_contact_half_span: 0.0,
                    ..Default::default()
                },
                "state6_contact_half_span must be positive",
            ),
            (
                VehicleParameters {
                    forward_axis_column: 3,
                    ..Default::default()
                },
                "forward axis requires a matrix column in 0..2 and sign +/-1",
            ),
            (
                VehicleParameters {
                    forward_axis_sign: 0.0,
                    ..Default::default()
                },
                "forward axis requires a matrix column in 0..2 and sign +/-1",
            ),
            (
                VehicleParameters {
                    ground_projection_floor: 0.0,
                    ..Default::default()
                },
                "ground_projection_floor must be positive",
            ),
            (
                VehicleParameters {
                    contact_compression_quantum: -1.0,
                    ..Default::default()
                },
                "contact_compression_quantum must be non-negative",
            ),
        ];
        for (parameters, message) in cases {
            assert_eq!(parameters.validate(), Err(message.to_string()));
        }
        assert_eq!(
            VehicleParameters {
                contact_probe_span_scale: 0.0,
                ..Default::default()
            }
            .validate(),
            Err("contact_probe_span_scale must be positive".to_string())
        );
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn boost_parameters_keys_match_python_dict() {
        let json = serde_json::to_value(BoostParameters::default()).unwrap();
        let object = json.as_object().unwrap();
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "attached_state",
                "correction_dir",
                "direction_gain",
                "e8c",
                "efc",
                "flag_5bc",
                "obf_684",
                "obf_69c",
                "obf_808",
                "obf_814",
                "obf_8ac",
                "obf_8e8",
                "obf_94c",
                "raw_144",
                "selected_slot",
            ]
        );
        assert_eq!(json["correction_dir"], serde_json::json!([1.0, 0.0, 0.0]));
        assert_eq!(json["obf_808"], 1.0);
        assert_eq!(json["obf_94c"], 0.0);
        assert_eq!(json["e8c"], false);
        let round_trip: BoostParameters = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, BoostParameters::default());
    }

    #[test]
    fn vehicle_parameters_round_trip() {
        let parameters = VehicleParameters::default();
        let json = serde_json::to_value(&parameters).unwrap();
        assert_eq!(json["topology_selector_delta_ms"], serde_json::Value::Null);
        assert_eq!(json["initial_rail_capture_commit"], serde_json::Value::Null);
        assert_eq!(json["substep_ms"], 2);
        let round_trip: VehicleParameters = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, parameters);
        round_trip.validate().unwrap();
    }
}
