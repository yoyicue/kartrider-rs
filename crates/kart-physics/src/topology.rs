//! Executable motion-topology ownership state machine.
//!
//! Faithful port of `scripts/motion_topology.py`. This module chooses which
//! constraint/reference-frame domain owns a vehicle substep. It deliberately
//! does not pretend that choosing a domain implements that domain's forces.
//! Domain-specific producers are dispatched separately by the vehicle engine.
//!
//! String values match the persisted domain registry
//! (`analysis/motion_topology/domain_registry_20260723.json`) exactly.

/// `MotionDomain(str, Enum)`: exact persisted domain identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MotionDomain {
    #[cfg_attr(feature = "serde", serde(rename = "GND"))]
    Ground,
    #[cfg_attr(feature = "serde", serde(rename = "GND-SLIP"))]
    GroundSlip,
    #[cfg_attr(feature = "serde", serde(rename = "CONTACT"))]
    Contact,
    #[cfg_attr(feature = "serde", serde(rename = "AIR-ENTRY"))]
    AirEntry,
    #[cfg_attr(feature = "serde", serde(rename = "AIR"))]
    Air,
    #[cfg_attr(feature = "serde", serde(rename = "LAND"))]
    Land,
    #[cfg_attr(feature = "serde", serde(rename = "RAIL"))]
    Rail,
    #[cfg_attr(feature = "serde", serde(rename = "MOVING-CONTACT"))]
    MovingContact,
    #[cfg_attr(feature = "serde", serde(rename = "RESET"))]
    Reset,
    #[cfg_attr(feature = "serde", serde(rename = "WATER?"))]
    Water,
    #[cfg_attr(feature = "serde", serde(rename = "NET-CORRECTION"))]
    NetworkCorrection,
}

impl MotionDomain {
    /// The persisted string value (`MotionDomain.value` in Python).
    pub fn value(self) -> &'static str {
        match self {
            MotionDomain::Ground => "GND",
            MotionDomain::GroundSlip => "GND-SLIP",
            MotionDomain::Contact => "CONTACT",
            MotionDomain::AirEntry => "AIR-ENTRY",
            MotionDomain::Air => "AIR",
            MotionDomain::Land => "LAND",
            MotionDomain::Rail => "RAIL",
            MotionDomain::MovingContact => "MOVING-CONTACT",
            MotionDomain::Reset => "RESET",
            MotionDomain::Water => "WATER?",
            MotionDomain::NetworkCorrection => "NET-CORRECTION",
        }
    }

    /// `MotionDomain(self.value)`: parse a persisted value back.
    pub fn from_value(value: &str) -> Option<MotionDomain> {
        Some(match value {
            "GND" => MotionDomain::Ground,
            "GND-SLIP" => MotionDomain::GroundSlip,
            "CONTACT" => MotionDomain::Contact,
            "AIR-ENTRY" => MotionDomain::AirEntry,
            "AIR" => MotionDomain::Air,
            "LAND" => MotionDomain::Land,
            "RAIL" => MotionDomain::Rail,
            "MOVING-CONTACT" => MotionDomain::MovingContact,
            "RESET" => MotionDomain::Reset,
            "WATER?" => MotionDomain::Water,
            "NET-CORRECTION" => MotionDomain::NetworkCorrection,
            _ => return None,
        })
    }

    fn is_ground(self) -> bool {
        matches!(self, MotionDomain::Ground | MotionDomain::GroundSlip)
    }

    fn is_air(self) -> bool {
        matches!(self, MotionDomain::AirEntry | MotionDomain::Air)
    }
}

/// `GROUND_DOMAINS = {GROUND, GROUND_SLIP}` membership test.
pub fn is_ground_domain(domain: MotionDomain) -> bool {
    domain.is_ground()
}

/// `AIR_DOMAINS = {AIR_ENTRY, AIR}` membership test.
pub fn is_air_domain(domain: MotionDomain) -> bool {
    domain.is_air()
}

/// `TopologyObservation`: engine-owned selectors visible at one topology
/// boundary. The derived `Default` matches the Python dataclass defaults
/// (zero counts, false flags, `None` identifiers).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct TopologyObservation {
    pub support_contact_count: i64,
    pub surface_mode: i64,
    pub collision_contact_count: i64,
    pub rail_contact: bool,
    pub rail_state: i64,
    pub rail_owner_id: Option<String>,
    pub moving_contact: bool,
    pub moving_object_id: Option<String>,
    pub launch_requested: bool,
    pub reset_requested: bool,
    pub reset_complete: bool,
    pub network_correction_active: bool,
    pub network_correction_complete: bool,
    pub resume_domain: Option<MotionDomain>,
    pub distinct_water_volume: bool,
}

impl TopologyObservation {
    /// Port of `TopologyObservation.validate` (ValueError -> Err(String)).
    pub fn validate(&self) -> Result<(), String> {
        if self.support_contact_count < 0 {
            return Err("support_contact_count must be nonnegative".to_string());
        }
        if self.collision_contact_count < 0 {
            return Err("collision_contact_count must be nonnegative".to_string());
        }
        if self.surface_mode != 0 && self.surface_mode != 2 {
            return Err("only proven surface modes 0 and 2 are supported".to_string());
        }
        if self.rail_state < 0 {
            return Err("rail_state must be nonnegative".to_string());
        }
        if let Some(owner) = &self.rail_owner_id {
            if owner.is_empty() {
                return Err("rail_owner_id must be nonempty when present".to_string());
            }
        }
        if self.moving_contact
            && self.moving_object_id.as_deref().unwrap_or("").is_empty()
        {
            return Err("moving_contact requires moving_object_id".to_string());
        }
        if let Some(candidate) = self.resume_domain {
            if matches!(
                candidate,
                MotionDomain::Reset | MotionDomain::NetworkCorrection
            ) {
                return Err(
                    "resume_domain must be a continuous physical domain".to_string()
                );
            }
        }
        Ok(())
    }
}

/// `TopologyState`: persisted topology ownership state.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct TopologyState {
    pub domain: MotionDomain,
    pub previous_domain: Option<MotionDomain>,
    pub return_domain: Option<MotionDomain>,
    pub transition_index: i64,
    pub domain_step_index: i64,
    pub rail_state_elapsed_ms: f64,
    pub entry_reason: String,
    pub constraint_owner_id: Option<String>,
    pub rail_state: i64,
}

impl Default for TopologyState {
    fn default() -> Self {
        TopologyState {
            domain: MotionDomain::Ground,
            previous_domain: None,
            return_domain: None,
            transition_index: 0,
            domain_step_index: 0,
            rail_state_elapsed_ms: 0.0,
            entry_reason: "initialization".to_string(),
            constraint_owner_id: None,
            rail_state: 0,
        }
    }
}

impl TopologyState {
    /// Port of `TopologyState.validate` (ValueError -> Err(String)).
    ///
    /// The Python checks that `domain`/`previous_domain`/`return_domain` name
    /// a registered `MotionDomain` are enforced by the Rust type system.
    pub fn validate(&self) -> Result<(), String> {
        if self.transition_index < 0 || self.domain_step_index < 0 {
            return Err("topology indices must be nonnegative".to_string());
        }
        if !self.rail_state_elapsed_ms.is_finite() || self.rail_state_elapsed_ms < 0.0
        {
            return Err(
                "rail_state_elapsed_ms must be finite and nonnegative".to_string()
            );
        }
        if self.domain != MotionDomain::Rail && self.rail_state_elapsed_ms != 0.0 {
            return Err("rail_state_elapsed_ms must be zero outside RAIL".to_string());
        }
        if self.domain == MotionDomain::Contact {
            let return_domain = self.return_domain.ok_or_else(|| {
                "CONTACT ownership requires return_domain".to_string()
            })?;
            if matches!(
                return_domain,
                MotionDomain::Contact | MotionDomain::Reset | MotionDomain::NetworkCorrection
            ) {
                return Err(
                    "CONTACT return_domain must be a physical continuous domain"
                        .to_string(),
                );
            }
        }
        Ok(())
    }
}

/// `_ground_domain(surface_mode)`.
fn ground_domain(surface_mode: i64) -> MotionDomain {
    if surface_mode == 2 {
        MotionDomain::GroundSlip
    } else {
        MotionDomain::Ground
    }
}

/// `_resume_domain(state, observation)`.
fn resume_domain(state: &TopologyState, observation: &TopologyObservation) -> MotionDomain {
    let candidate = observation.resume_domain.or(state.return_domain);
    if observation.support_contact_count > 0 {
        // A correction transaction does not execute contact impulses. Preserve
        // an airborne/landing source for the next physics substep, where
        // autonomous wheel normals can run the LAND transfer.
        if candidate.is_some_and(is_air_domain) || candidate == Some(MotionDomain::Land) {
            return candidate.unwrap();
        }
        return ground_domain(observation.surface_mode);
    }
    if let Some(candidate) = candidate {
        // CONTACT is evidence-owned, not a persistent constraint.  A network
        // correction may preempt it, but completion must not resurrect the
        // old contact after the collision selector reports no active contact.
        // The generic network completion observation currently carries no
        // nested CONTACT return provenance, so unsupported release falls back
        // to free flight; supported release was handled above.
        if candidate == MotionDomain::Contact {
            return MotionDomain::Air;
        }
        // Exclusive constraints are reacquired only by their current selector
        // branches above. A pre-correction domain name is not evidence that
        // the rail/path, moving body, or water volume still owns the corrected
        // pose.
        if matches!(
            candidate,
            MotionDomain::Rail | MotionDomain::MovingContact | MotionDomain::Water
        ) {
            return MotionDomain::Air;
        }
        if candidate.is_ground() {
            return MotionDomain::Air;
        }
        return candidate;
    }
    MotionDomain::Air
}

/// `select_motion_domain(state, observation)`: select the next exclusive
/// topology owner and transition reason.
///
/// Priority follows authority strength: reset/network replacement, exclusive
/// path/reference-frame capture, explicit water volume, landing/support,
/// collision overlay, then free flight.
///
/// Returns `(selected, reason, return_domain)` mirroring the Python tuple.
pub fn select_motion_domain(
    state: &TopologyState,
    observation: &TopologyObservation,
) -> Result<(MotionDomain, &'static str, Option<MotionDomain>), String> {
    state.validate()?;
    observation.validate()?;
    let current = state.domain;

    if observation.reset_requested {
        return Ok((MotionDomain::Reset, "reset_requested", Some(current)));
    }
    if current == MotionDomain::Reset && !observation.reset_complete {
        return Ok((
            MotionDomain::Reset,
            "reset_in_progress",
            state.return_domain,
        ));
    }

    if observation.network_correction_active {
        return Ok((
            MotionDomain::NetworkCorrection,
            "network_correction_active",
            Some(current),
        ));
    }
    if current == MotionDomain::NetworkCorrection && !observation.network_correction_complete
    {
        return Ok((
            MotionDomain::NetworkCorrection,
            "network_correction_in_progress",
            state.return_domain,
        ));
    }

    // Runtime writer evidence observes the complete owned cycle
    // 0 -> 1 -> 2 -> 3 -> 6 -> 0.  State 1 is capture/setup and state 6 is
    // release/setup; both still retain the rail contact/reference frame.
    if observation.rail_contact && matches!(observation.rail_state, 1..=6) {
        return Ok((MotionDomain::Rail, "rail_capture_or_sustain", Some(current)));
    }
    if observation.moving_contact {
        return Ok((
            MotionDomain::MovingContact,
            "moving_reference_frame_acquired",
            Some(current),
        ));
    }
    if observation.distinct_water_volume {
        return Ok((
            MotionDomain::Water,
            "distinct_water_volume_selected",
            Some(current),
        ));
    }

    if observation.launch_requested && current.is_ground() {
        return Ok((MotionDomain::AirEntry, "explicit_launch_producer", None));
    }

    if current == MotionDomain::Reset && observation.reset_complete {
        let resumed = if observation.support_contact_count > 0 {
            ground_domain(observation.surface_mode)
        } else {
            MotionDomain::Air
        };
        return Ok((resumed, "reset_complete", None));
    }
    if current == MotionDomain::NetworkCorrection && observation.network_correction_complete {
        let resumed = resume_domain(state, observation);
        return Ok((resumed, "network_correction_complete", None));
    }

    if matches!(
        current,
        MotionDomain::Rail | MotionDomain::MovingContact | MotionDomain::Water
    ) {
        if observation.support_contact_count > 0 {
            return Ok((
                ground_domain(observation.surface_mode),
                "exclusive_constraint_released_to_ground",
                None,
            ));
        }
        return Ok((
            MotionDomain::AirEntry,
            "exclusive_constraint_released_to_air",
            None,
        ));
    }

    if observation.support_contact_count > 0 {
        if current == MotionDomain::Contact
            && matches!(
                state.return_domain,
                Some(MotionDomain::Air) | Some(MotionDomain::AirEntry)
            )
        {
            return Ok((
                MotionDomain::Land,
                "support_reacquired_from_airborne_contact",
                None,
            ));
        }
        if current.is_air() {
            return Ok((MotionDomain::Land, "support_contact_reacquired", None));
        }
        if current == MotionDomain::Land && observation.collision_contact_count > 0 {
            return Ok((
                MotionDomain::Contact,
                "landing_collision_contact_active",
                Some(ground_domain(observation.surface_mode)),
            ));
        }
        if current == MotionDomain::Land {
            return Ok((
                ground_domain(observation.surface_mode),
                "landing_stabilized",
                None,
            ));
        }
        if observation.collision_contact_count > 0 {
            return Ok((
                MotionDomain::Contact,
                if current == MotionDomain::Contact {
                    "grounded_collision_contact_sustained"
                } else {
                    "non_support_collision_contact_active"
                },
                if current == MotionDomain::Contact {
                    state.return_domain
                } else {
                    Some(current)
                },
            ));
        }
        return Ok((
            ground_domain(observation.surface_mode),
            "ground_support_active",
            None,
        ));
    }

    if current == MotionDomain::Contact {
        if observation.collision_contact_count > 0 {
            return Ok((
                MotionDomain::Contact,
                "airborne_collision_contact_sustained",
                state.return_domain,
            ));
        }
        if matches!(
            state.return_domain,
            Some(MotionDomain::Air) | Some(MotionDomain::AirEntry)
        ) {
            return Ok((MotionDomain::Air, "airborne_contact_released_to_air", None));
        }
        return Ok((
            MotionDomain::AirEntry,
            "contact_released_without_support",
            None,
        ));
    }
    if observation.collision_contact_count > 0 {
        return Ok((
            MotionDomain::Contact,
            "airborne_collision_contact_active",
            Some(current),
        ));
    }
    if current == MotionDomain::Land {
        return Ok((MotionDomain::Air, "landing_bounce", None));
    }
    if current.is_ground() {
        return Ok((MotionDomain::AirEntry, "support_contact_lost", None));
    }
    if current == MotionDomain::AirEntry {
        return Ok((MotionDomain::Air, "takeoff_transfer_complete", None));
    }
    Ok((MotionDomain::Air, "no_support_contact", None))
}

/// Result of `advance_topology` (port of the Python decision dict).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TopologyDecision {
    pub selected_domain: MotionDomain,
    pub transitioned: bool,
    pub reason: String,
    pub state_before: TopologyState,
    pub state_after: TopologyState,
    pub observation: TopologyObservation,
    pub authority: String,
}

/// `advance_topology(state, observation)`: advance topology ownership without
/// mutating caller-owned state.
pub fn advance_topology(
    state: &TopologyState,
    observation: &TopologyObservation,
) -> Result<TopologyDecision, String> {
    let (selected, reason, return_domain) = select_motion_domain(state, observation)?;
    let current = state.domain;
    let transitioned = selected != current;
    let rail_state_changed =
        selected == MotionDomain::Rail && observation.rail_state != state.rail_state;
    let mut next_state = TopologyState {
        domain: selected,
        previous_domain: if transitioned {
            Some(current)
        } else {
            state.previous_domain
        },
        return_domain: if transitioned {
            return_domain
        } else {
            state.return_domain
        },
        transition_index: state.transition_index + i64::from(transitioned),
        domain_step_index: if transitioned {
            0
        } else {
            state.domain_step_index + 1
        },
        rail_state_elapsed_ms: if transitioned || rail_state_changed {
            0.0
        } else {
            state.rail_state_elapsed_ms
        },
        entry_reason: if transitioned {
            reason.to_string()
        } else {
            state.entry_reason.clone()
        },
        constraint_owner_id: if selected == MotionDomain::MovingContact {
            observation.moving_object_id.clone()
        } else if selected == MotionDomain::Rail {
            // RAIL transition-step ownership rule (motion_topology.py:313-319):
            // a transition substep entering RAIL must not inherit the previous
            // dynamic body's constraint_owner_id.
            observation.rail_owner_id.clone().or(if !transitioned {
                state.constraint_owner_id.clone()
            } else {
                None
            })
        } else {
            None
        },
        rail_state: if selected == MotionDomain::Rail {
            observation.rail_state
        } else {
            0
        },
    };
    // CONTACT is an exclusive scheduler owner here, but remembers the domain
    // to which separation should return. Reset/network replacement also retain
    // their preemption target.
    if matches!(
        selected,
        MotionDomain::Contact | MotionDomain::Reset | MotionDomain::NetworkCorrection
    ) {
        next_state.return_domain = return_domain.or(state.return_domain);
    }
    next_state.validate()?;
    Ok(TopologyDecision {
        selected_domain: selected,
        transitioned,
        reason: reason.to_string(),
        state_before: state.clone(),
        state_after: next_state,
        observation: observation.clone(),
        authority: "engine_owned_topology_selectors".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(
        state: &TopologyState,
        observation: TopologyObservation,
    ) -> (TopologyState, TopologyDecision) {
        let decision = advance_topology(state, &observation).expect("advance_topology");
        (decision.state_after.clone(), decision)
    }

    #[test]
    fn domain_values_match_persisted_registry() {
        let values = [
            MotionDomain::Ground,
            MotionDomain::GroundSlip,
            MotionDomain::Contact,
            MotionDomain::AirEntry,
            MotionDomain::Air,
            MotionDomain::Land,
            MotionDomain::Rail,
            MotionDomain::MovingContact,
            MotionDomain::Reset,
            MotionDomain::Water,
            MotionDomain::NetworkCorrection,
        ]
        .map(MotionDomain::value);
        assert_eq!(
            values,
            [
                "GND",
                "GND-SLIP",
                "CONTACT",
                "AIR-ENTRY",
                "AIR",
                "LAND",
                "RAIL",
                "MOVING-CONTACT",
                "RESET",
                "WATER?",
                "NET-CORRECTION",
            ]
        );
        for domain in [
            MotionDomain::Ground,
            MotionDomain::GroundSlip,
            MotionDomain::Contact,
            MotionDomain::AirEntry,
            MotionDomain::Air,
            MotionDomain::Land,
            MotionDomain::Rail,
            MotionDomain::MovingContact,
            MotionDomain::Reset,
            MotionDomain::Water,
            MotionDomain::NetworkCorrection,
        ] {
            assert_eq!(MotionDomain::from_value(domain.value()), Some(domain));
        }
        assert_eq!(MotionDomain::from_value("bogus"), None);
    }

    #[test]
    fn default_state_and_observation_validate() {
        TopologyState::default().validate().unwrap();
        TopologyObservation::default().validate().unwrap();
    }

    #[test]
    fn observation_validation_rejects_python_cases() {
        let cases: [(TopologyObservation, &str); 6] = [
            (
                TopologyObservation {
                    support_contact_count: -1,
                    ..Default::default()
                },
                "support_contact_count must be nonnegative",
            ),
            (
                TopologyObservation {
                    collision_contact_count: -1,
                    ..Default::default()
                },
                "collision_contact_count must be nonnegative",
            ),
            (
                TopologyObservation {
                    surface_mode: 1,
                    ..Default::default()
                },
                "only proven surface modes 0 and 2 are supported",
            ),
            (
                TopologyObservation {
                    rail_state: -1,
                    ..Default::default()
                },
                "rail_state must be nonnegative",
            ),
            (
                TopologyObservation {
                    rail_owner_id: Some(String::new()),
                    ..Default::default()
                },
                "rail_owner_id must be nonempty when present",
            ),
            (
                TopologyObservation {
                    moving_contact: true,
                    ..Default::default()
                },
                "moving_contact requires moving_object_id",
            ),
        ];
        for (observation, message) in cases {
            assert_eq!(observation.validate(), Err(message.to_string()));
        }
        assert_eq!(
            TopologyObservation {
                resume_domain: Some(MotionDomain::Reset),
                ..Default::default()
            }
            .validate(),
            Err("resume_domain must be a continuous physical domain".to_string())
        );
        assert_eq!(
            TopologyObservation {
                resume_domain: Some(MotionDomain::NetworkCorrection),
                ..Default::default()
            }
            .validate(),
            Err("resume_domain must be a continuous physical domain".to_string())
        );
    }

    #[test]
    fn state_validation_rejects_python_cases() {
        assert_eq!(
            TopologyState {
                transition_index: -1,
                ..Default::default()
            }
            .validate(),
            Err("topology indices must be nonnegative".to_string())
        );
        assert_eq!(
            TopologyState {
                domain_step_index: -1,
                ..Default::default()
            }
            .validate(),
            Err("topology indices must be nonnegative".to_string())
        );
        assert_eq!(
            TopologyState {
                rail_state_elapsed_ms: f64::NAN,
                ..Default::default()
            }
            .validate(),
            Err("rail_state_elapsed_ms must be finite and nonnegative".to_string())
        );
        assert_eq!(
            TopologyState {
                rail_state_elapsed_ms: -0.5,
                ..Default::default()
            }
            .validate(),
            Err("rail_state_elapsed_ms must be finite and nonnegative".to_string())
        );
        assert_eq!(
            TopologyState {
                rail_state_elapsed_ms: 1.0,
                ..Default::default()
            }
            .validate(),
            Err("rail_state_elapsed_ms must be zero outside RAIL".to_string())
        );
        assert_eq!(
            TopologyState {
                domain: MotionDomain::Contact,
                ..Default::default()
            }
            .validate(),
            Err("CONTACT ownership requires return_domain".to_string())
        );
        for invalid in [
            MotionDomain::Contact,
            MotionDomain::Reset,
            MotionDomain::NetworkCorrection,
        ] {
            assert_eq!(
                TopologyState {
                    domain: MotionDomain::Contact,
                    return_domain: Some(invalid),
                    ..Default::default()
                }
                .validate(),
                Err("CONTACT return_domain must be a physical continuous domain"
                    .to_string())
            );
        }
    }

    #[test]
    fn ground_slip_air_entry_air_land_ground_cycle() {
        let state = TopologyState::default();
        let (state, _) = step(
            &state,
            TopologyObservation {
                support_contact_count: 4,
                surface_mode: 2,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::GroundSlip);
        let (state, _) = step(&state, TopologyObservation::default());
        assert_eq!(state.domain, MotionDomain::AirEntry);
        let (state, _) = step(&state, TopologyObservation::default());
        assert_eq!(state.domain, MotionDomain::Air);
        let (state, _) = step(
            &state,
            TopologyObservation {
                support_contact_count: 2,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::Land);
        let (state, _) = step(
            &state,
            TopologyObservation {
                support_contact_count: 4,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::Ground);
    }

    #[test]
    fn explicit_launch_producer_preempts_current_ground_support() {
        let (state, decision) = step(
            &TopologyState::default(),
            TopologyObservation {
                support_contact_count: 4,
                launch_requested: true,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::AirEntry);
        assert_eq!(decision.reason, "explicit_launch_producer");
    }

    #[test]
    fn landing_collision_precedes_ground_stabilization() {
        let (state, decision) = step(
            &TopologyState {
                domain: MotionDomain::Land,
                ..Default::default()
            },
            TopologyObservation {
                support_contact_count: 4,
                surface_mode: 2,
                collision_contact_count: 1,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::Contact);
        assert_eq!(state.return_domain, Some(MotionDomain::GroundSlip));
        assert_eq!(decision.reason, "landing_collision_contact_active");
    }

    #[test]
    fn rail_capture_and_release_to_air() {
        for rail_state in [1, 2, 3, 4, 5, 6] {
            let (state, _) = step(
                &TopologyState::default(),
                TopologyObservation {
                    rail_contact: true,
                    rail_state,
                    ..Default::default()
                },
            );
            assert_eq!(state.domain, MotionDomain::Rail);
            assert_eq!(state.rail_state, rail_state);
            let (state, _) = step(&state, TopologyObservation::default());
            assert_eq!(state.domain, MotionDomain::AirEntry);
        }
    }

    #[test]
    fn rail_transition_step_does_not_inherit_moving_owner() {
        // motion_topology.py:313-319: entering RAIL on a transition substep
        // must not carry over the previous constraint owner.
        let moving = TopologyState {
            domain: MotionDomain::MovingContact,
            constraint_owner_id: Some("roadobj:owner".to_string()),
            ..Default::default()
        };
        let (state, _) = step(
            &moving,
            TopologyObservation {
                rail_contact: true,
                rail_state: 3,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::Rail);
        assert_eq!(state.constraint_owner_id, None);
        // A rail_owner_id in the observation wins on the transition step.
        let (state, _) = step(
            &moving,
            TopologyObservation {
                rail_contact: true,
                rail_state: 3,
                rail_owner_id: Some("rail:path".to_string()),
                ..Default::default()
            },
        );
        assert_eq!(
            state.constraint_owner_id,
            Some("rail:path".to_string())
        );
        // Sustained RAIL without an observation owner keeps the stored owner.
        let (state, _) = step(
            &state,
            TopologyObservation {
                rail_contact: true,
                rail_state: 3,
                ..Default::default()
            },
        );
        assert_eq!(state.domain, MotionDomain::Rail);
        assert_eq!(
            state.constraint_owner_id,
            Some("rail:path".to_string())
        );
    }

    #[test]
    fn exhaustive_domain_observation_combinations_preserve_invariants() {
        // Mirror of scripts/test_motion_topology.py
        // test_all_domain_observation_combinations_preserve_state_invariants.
        let mut states = Vec::new();
        for domain in [
            MotionDomain::Ground,
            MotionDomain::GroundSlip,
            MotionDomain::Contact,
            MotionDomain::AirEntry,
            MotionDomain::Air,
            MotionDomain::Land,
            MotionDomain::Rail,
            MotionDomain::MovingContact,
            MotionDomain::Reset,
            MotionDomain::Water,
            MotionDomain::NetworkCorrection,
        ] {
            let mut state = TopologyState {
                domain,
                ..Default::default()
            };
            if domain == MotionDomain::Contact {
                state.return_domain = Some(MotionDomain::Ground);
            }
            if domain == MotionDomain::Rail {
                state.rail_state = 3;
            }
            if domain == MotionDomain::MovingContact {
                state.constraint_owner_id = Some("roadobj:owner".to_string());
            }
            if matches!(
                domain,
                MotionDomain::Reset | MotionDomain::NetworkCorrection
            ) {
                state.return_domain = Some(MotionDomain::Ground);
            }
            state.validate().unwrap();
            states.push(state);
        }

        let mut checked = 0u32;
        for state in &states {
            for flags in 0..(1u32 << 9) {
                let flag = |bit: u32| flags & (1 << bit) != 0;
                let (support, collision, reset_requested, reset_complete) =
                    (flag(0), flag(1), flag(2), flag(3));
                let (network_active, network_complete, rail_contact) =
                    (flag(4), flag(5), flag(6));
                let (moving_contact, water) = (flag(7), flag(8));
                let observation = TopologyObservation {
                    support_contact_count: i64::from(support),
                    collision_contact_count: i64::from(collision),
                    reset_requested,
                    reset_complete,
                    network_correction_active: network_active,
                    network_correction_complete: network_complete,
                    rail_contact,
                    rail_state: if rail_contact { 3 } else { 0 },
                    moving_contact,
                    moving_object_id: if moving_contact {
                        Some("roadobj:new".to_string())
                    } else {
                        None
                    },
                    distinct_water_volume: water,
                    ..Default::default()
                };
                let decision = advance_topology(state, &observation).unwrap();
                decision.state_after.validate().unwrap();
                checked += 1;
            }
        }
        assert_eq!(checked, 11 * 512);
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn motion_domain_serializes_as_persisted_value() {
        for domain in [
            MotionDomain::Ground,
            MotionDomain::GroundSlip,
            MotionDomain::Contact,
            MotionDomain::AirEntry,
            MotionDomain::Air,
            MotionDomain::Land,
            MotionDomain::Rail,
            MotionDomain::MovingContact,
            MotionDomain::Reset,
            MotionDomain::Water,
            MotionDomain::NetworkCorrection,
        ] {
            let json = serde_json::to_string(&domain).unwrap();
            assert_eq!(json, format!("\"{}\"", domain.value()));
            let round_trip: MotionDomain = serde_json::from_str(&json).unwrap();
            assert_eq!(round_trip, domain);
        }
    }

    #[test]
    fn topology_state_round_trip_with_nulls() {
        let state = TopologyState {
            domain: MotionDomain::Rail,
            previous_domain: Some(MotionDomain::Air),
            rail_state: 3,
            rail_state_elapsed_ms: 10.0,
            domain_step_index: 5,
            constraint_owner_id: Some("rail:exact-state1-gate".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["domain"], "RAIL");
        assert_eq!(json["previous_domain"], "AIR");
        assert_eq!(json["return_domain"], serde_json::Value::Null);
        let round_trip: TopologyState = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, state);
        round_trip.validate().unwrap();
    }
}
