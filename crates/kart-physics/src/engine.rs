//! Engine layer (port of `KartRiderVehicleEngine` in
//! `scripts/kartrider_vehicle_engine.py`) — the GND chain plus the frame
//! loop and topology dispatch.
//!
//! Ported scope (docs/kart-physics-rust-handoff-2026-07-28.md §3):
//! - module helpers `_validate_raw_control_keys` (:88),
//!   `select_surface_mode_from_contacts` (:97), `_wheel_anchor` (:703,
//!   f32/f64 dual path), `derive_autonomous_ground_projection` (:761),
//!   `query_wheel_ground` (:178), `_axes_to_quaternion` /
//!   `_quaternion_to_axes` / `_interpolate_axes` (:831-1021),
//!   `select_collision_dispatch` (:1024);
//! - the replay-path `step` (:5106-5191) and `_bind_state` (:5068-5104);
//! - the control path `step_control` (:5193-5287) and `step_raw_control`
//!   (:5289-5368), including the `exact_drive_basis` input mode with the
//!   drive-target evaluator selection and the `ground_runtime` adapter
//!   wiring, and the `step_world` control delegation (:7217-7222);
//! - `step_raw_ground_correction` (:5370-6006), the GND producer/integrator
//!   body, and `produce_ground_accumulators` (:6008-6028);
//! - `step_world` (:7170) full CONTACT scope: the broad-phase sweep, the
//!   mixed/pure-soft/pure-hard collision chains, single-contact hard/soft
//!   dispatch with frame replay, and the CONTACT topology commit/release;
//! - `step_airborne` (:1751), `step_raw_airborne` (:6030),
//!   `inspect_autonomous_support` (:6121), the authored reset-surface entry
//!   (:6179-6226), `step_raw_topology` (:6228), the capture/selector
//!   inspections (`_select_current_pose_hard_overlap` :1293,
//!   `_apply_autonomous_authored_trigger` :3589 support + rail query paths,
//!   `inspect_autonomous_rail_capture` :3753,
//!   `inspect_autonomous_moving_contact` :4117,
//!   `poll_autonomous_checkpoint_target` :1521,
//!   `apply_fall_clamp_reset` :1559, `apply_manual_restart_request` :1595),
//!   and the frame loop `step_raw_topology`/`_step_raw_motion_once`/
//!   `step_raw_motion`/`step_raw_frame` (:6919-7168);
//! - the RAIL domain steppers (M2 part 2): `step_raw_rail` full body
//!   (:2454-2660, active writer chain + continuous release),
//!   `_step_rail_state6_tail` (:1798-2452, the B276F0
//!   contact-confirmation tail with the binary32 zero-boundary writer
//!   phase), `step_autonomous_rail` (:2661-3313, the RAIL domain owner
//!   stepper incl. the state1 ground-writer path, path velocity transfers,
//!   terminal endpoint transition, and state6 entry-force staging),
//!   `_advance_autonomous_rail_selector` (:3315-3432),
//!   `_query_autonomous_rail_authored_trigger` (:3434-3527),
//!   `_evaluate_rail_authored_trigger` (:3529-3587), the
//!   `step_world` rail-state1 ground-writer mode (:7181-7211) and the
//!   `_step_raw_motion_once` RAIL dispatch arm (:7128-7129, :7147-7151);
//! - the MOVING-CONTACT domain steppers (M2 part 3):
//!   `acquire_moving_contact` (:4058),
//!   `acquire_autonomous_moving_contact` (:4170),
//!   `acquire_and_step_autonomous_moving_contact` (:4192),
//!   `step_moving_contact` (:4235), `step_autonomous_moving_contact`
//!   (:4289), `release_moving_contact` (:4372), the initial-capture arm of
//!   `step_raw_topology` (:6375-6388) and the `_step_raw_motion_once`
//!   MOVING-CONTACT dispatch arm (:7130-7131, :7153-7157);
//! - the RESET domain owner (M2 part 3): `apply_checkpoint_respawn` (:1637)
//!   and `step_autonomous_reset` (:1704) plus the `_step_raw_motion_once`
//!   RESET dispatch arm (:7132-7133);
//! - the NET-CORRECTION domain (M3): `begin_network_correction` (:4403),
//!   `inspect_autonomous_network_correction` (:4472),
//!   `begin_and_step_autonomous_network_correction` (:4560),
//!   `rollback_and_resimulate_network_correction` (:4608),
//!   `step_network_correction` (:4740), `step_autonomous_network_correction`
//!   (:4938), the preemption arms (:6294-6312, :7093-7111) and the
//!   `_step_raw_motion_once` NET-CORRECTION dispatch arm (:7134-7135), with
//!   the `step_raw_motion` history clear/commit semantics (:6933-6962).
//!
//! Hard rules carried over from the handoff:
//! - every cross-substep latch lives in [`VehicleState`]; this struct holds
//!   only state + numeric mode + the (M3) prediction history;
//! - the producer/integrator split is expressed at the type level:
//!   [`KartRiderVehicleEngine::produce_ground_accumulators`] takes `&self`
//!   (the zero-state-write contract is unrepresentable as a mutation), the
//!   integrating [`KartRiderVehicleEngine::step_raw_ground_correction`] takes
//!   `&mut self`;
//! - the 60 Hz surface selector latch commits on the integrating path only;
//!   producer probes read the same epoch without advancing it
//!   (`docs/kart-physics-rust-handoff-2026-07-28.md`, §6 item 3);
//! - the collision chain is ONE implementation (`_apply_collision_chain` +
//!   `_finalize_collision_chain`); the four chain types (mixed / pure-soft /
//!   pure-hard / state6) are parameter differences — branch filter,
//!   per-candidate quantization, quantize_final (§3, §6.5);
//! - the producer path has no world parameter at all (Python:
//!   `ValueError("producer-only ground evaluation cannot dispatch world
//!   collisions")`, here type-enforced).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::domains::airborne_topology::{
    evaluate_airborne_step, evaluate_takeoff_or_landing_contact, AirborneStep,
    TakeoffOrLandingContact,
};
use crate::domains::booster_resource::InventorySlot;
use crate::domains::booster_vehicle_step::{
    evaluate_current_item6_inventory_use, CurrentItem6InventoryUseResult,
};
use crate::domains::branch_aware_vehicle_step::{
    evaluate_branch_aware_step, AccumulatorArgs, AngularIntegratorArgs, BoostArgs,
    BranchAwareStepArgs, BranchAwareStepResult, CommonDampingArgs, ContactArgs, DriftWindowArgs,
    GroundCorrectionArgs, GroundDriveForceArgs, GroundSurfaceProfileArgs, HardContactArgs,
    JmDjArgs, LinearIntegratorArgs, LowProjectionArgs, OrientationIntegratorArgs,
    PositionIntegratorArgs, RecoveryArgs, RecoveryScheduleArgs, RecoveryScheduleOutcome,
    SoftContactArgs, SubstepBoundaryArgs, Surface1RecoveryForceArgs, ORDERED_EVENTS,
};
use crate::domains::checkpoint_respawn_state_replacement::{
    evaluate_checkpoint_respawn_state_replacement, CheckpointRespawnStateReplacement,
};
use crate::domains::checkpoint_target_selector::{
    evaluate_checkpoint_target_selector, CheckpointTargetSelectorResult,
};
use crate::domains::collision_response_selector::{
    evaluate_collision_response_selector, CollisionResponseSelector,
};
use crate::domains::control_response_input::{canonical_action, ControlResponseInputModel};
use crate::domains::drift_recovery_timer::{apply_selector_transition, SelectorTransitionResult};
use crate::domains::drift_window_branch::DriftWindowBranchResult;
use crate::domains::drive_basis_input::{evaluate_drive_basis_input, DriveBasisInput};
use crate::domains::drive_target_state::{
    evaluate_drive_target_runtime, evaluate_drive_target_state, DriveTargetRuntimeResult,
    DriveTargetStateResult,
};
use crate::domains::event_obstacle_contact::{
    apply_event_obstacle_contact_source, EventObstacleContactResponse,
};
use crate::domains::fall_clamp_state_replacement::{
    evaluate_fall_clamp_state_replacement, FallClampStateReplacement,
};
use crate::domains::ground_runtime_input::{select_ground_runtime_input, GroundRuntimeSelection};
use crate::domains::hard_contact_angular_response::{
    evaluate_hard_contact_angular_response, HardContactAngularResponse,
};
use crate::domains::hard_contact_velocity_response::{
    evaluate_hard_contact_velocity_response, HardContactVelocityResponse,
};
use crate::domains::jm_dj_trigger::{
    evaluate_jm_dj_trigger, parse_jm_dj_surface_token, JmDjSurfaceToken, JmDjTrigger,
};
use crate::domains::low_projection_ground_branch::LowProjectionBranchResult;
use crate::domains::moving_contact_topology::{
    acquire_moving_contact_frame, evaluate_moving_contact_selector, evaluate_moving_contact_step,
    py_float_repr, MovingContactSelectorResult, MovingContactStep,
};
use crate::domains::network_correction_topology::{
    begin_network_correction as evaluate_begin_network_correction,
    evaluate_network_correction_step, evaluate_network_sequence, NetworkCorrectionStepResult,
    NetworkSequenceDecision, SEQUENCE_BITS,
};
use crate::domains::orientation_integrator::evaluate_orientation_integrator;
use crate::domains::position_integrator::evaluate_position_integrator;
use crate::domains::rail_capture_selector::{
    evaluate_rail_capture_selector, RailCaptureSelectorArgs, RailCaptureSelectorResult,
    RailCaptureSelectorRuntimeAfter, STATE6_CONTACT_CONFIRM_MS,
};
use crate::domains::rail_state1_query::{
    evaluate_rail_state1_query, evaluate_rail_state1_velocity_transfer, RailState1Context,
    RailState1Gate, RailState1Point, RailState1QueryResult, RailState1VelocityTransfer,
};
use crate::domains::rail_topology::{evaluate_rail_step, RailStepArgs, RailStepResult};
use crate::domains::soft_contact_angular_response::{
    evaluate_soft_contact_angular_source, SoftContactAngularSource,
};
use crate::domains::soft_contact_source::{
    evaluate_soft_contact_source, SoftContactBaseSource, SoftContactSource,
};
use crate::domains::tune_boost_gauge::{
    evaluate_current_family_item6_grant, evaluate_current_family_pending_gauge_producer,
    evaluate_current_family_primary_gauge_producer, CurrentFamilyGaugeProducerTransition,
    CurrentFamilyGaugeProducersArgs, CurrentFamilyItem6Grant, GaugeProducerState,
};
use crate::num::{
    cpython_hypot2, cross3_f32, dot3_f32, dot3_sum, f32_vector, f32q, fadd, fdiv, fmul, fsub,
    length3, length3_f32, matrix_columns, max2_f64, min2_f64, sub3_f32, sum_f64,
};
use crate::params::VehicleParameters;
use crate::rail::RailRuntimeContext;
use crate::scheduler::{evaluate_physics_frame_scheduler, FrameSchedule};
use crate::state::{
    JsonValue, MovingContactAnchor, NetworkCorrectionState, RailPathContextId, RigidStateSnapshot,
    VehicleState,
};
use crate::topology::{
    advance_topology, MotionDomain, TopologyDecision, TopologyObservation, TopologyState,
};
use crate::world::{
    CheckpointRespawnCandidate, CheckpointTargetCandidate, EventObstacleContact, World,
    WorldGroundResult, WorldObbOverlap,
};

/// `MILLISECONDS_PER_SECOND` (kartrider_vehicle_engine.py:78).
pub const MILLISECONDS_PER_SECOND: f64 = 1000.0;
/// `SURFACE_SELECTOR_HZ` (kartrider_vehicle_engine.py:79).
pub const SURFACE_SELECTOR_HZ: f64 = 60.0;
/// `AUTHORED_TRIGGER_COOLDOWN_MS` (kartrider_vehicle_engine.py:84).
pub const AUTHORED_TRIGGER_COOLDOWN_MS: f64 = 300.0;
/// GoItemBanana kind-1 writes GoPlayKart+0xB9C for exactly 2000 ms.
pub const BANANA_EFFECT_DURATION_MS: i64 = 2000;
/// `RAIL_CAPTURE_POSITION_GAIN_PER_SECOND` (kartrider_vehicle_engine.py:81).
pub const RAIL_CAPTURE_POSITION_GAIN_PER_SECOND: f64 = 2.5;
/// `RAIL_STATE6_ENTRY_FORCE_SCALE` (kartrider_vehicle_engine.py:82).
pub const RAIL_STATE6_ENTRY_FORCE_SCALE: f64 = 600000.0;
/// `STATE6_PARTIAL_WHEEL_BRIDGE_DELTA_Z` (kartrider_vehicle_engine.py:83).
pub const STATE6_PARTIAL_WHEEL_BRIDGE_DELTA_Z: f64 = 0.180000007;
/// Python `_prediction_history_limit` (kartrider_vehicle_engine.py:1071).
pub const PREDICTION_HISTORY_LIMIT: usize = 512;

/// Python `numeric_mode` string ("float32" production / "float64" research).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericMode {
    Float32,
    Float64,
}

impl NumericMode {
    pub fn as_str(self) -> &'static str {
        match self {
            NumericMode::Float32 => "float32",
            NumericMode::Float64 => "float64",
        }
    }

    pub fn is_float32(self) -> bool {
        matches!(self, NumericMode::Float32)
    }

    /// Python `KartRiderVehicleEngine.__init__` mode guard
    /// (kartrider_vehicle_engine.py:1064-1065).
    pub fn parse(value: &str) -> Result<NumericMode, String> {
        match value {
            "float32" => Ok(NumericMode::Float32),
            "float64" => Ok(NumericMode::Float64),
            _ => Err("numeric_mode must be float32 or float64".to_string()),
        }
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for NumericMode {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for NumericMode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = <String as serde::Deserialize>::deserialize(deserializer)?;
        NumericMode::parse(&value).map_err(serde::de::Error::custom)
    }
}

/// One wheel ground plane dict as consumed by `step_raw_ground_correction`
/// (`normal`, `point`, `triangle`, `road_surface`, `node`, `tri_index`;
/// extra keys such as `surface_id` are ignored).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GroundPlane {
    pub normal: [f64; 3],
    pub point: [f64; 3],
    /// Authored triangle vertices; selects the float32 Moller-Trumbore ray
    /// branch (CollisionCollection_SweepMainFaces, A5B840) in float32 mode.
    pub triangle: Option<[[f64; 3]; 3]>,
    pub road_surface: Option<String>,
    pub node: Option<JsonValue>,
    pub tri_index: Option<i64>,
}

/// Python `ground_plane` argument shape: either a single plane dict
/// (broadcast to all four wheels by the contact probe loop) or a sequence
/// of exactly four wheel planes.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum GroundPlaneSet {
    Single(GroundPlane),
    Sequence(Vec<GroundPlane>),
}

impl GroundPlaneSet {
    /// Python `list(ground_plane) if isinstance(...) else [ground_plane]`:
    /// the surface selector sees one token for a single dict.
    fn selector_planes(&self) -> Vec<&GroundPlane> {
        match self {
            GroundPlaneSet::Single(plane) => vec![plane],
            GroundPlaneSet::Sequence(planes) => planes.iter().collect(),
        }
    }

    /// The 4-wheel probe loop view: a single dict broadcasts to all four
    /// slots; a sequence must contain exactly four wheel planes
    /// (ValueError "ground_plane sequence must contain four wheel planes").
    fn probe_planes(&self) -> Result<[&GroundPlane; 4], String> {
        match self {
            GroundPlaneSet::Single(plane) => Ok([plane, plane, plane, plane]),
            GroundPlaneSet::Sequence(planes) => {
                if planes.len() != 4 {
                    return Err("ground_plane sequence must contain four wheel planes".to_string());
                }
                Ok([&planes[0], &planes[1], &planes[2], &planes[3]])
            }
        }
    }
}

/// `_validate_raw_control_keys` (kartrider_vehicle_engine.py:88-94).
pub fn validate_raw_control_keys(keys: &[String]) -> Result<BTreeSet<String>, String> {
    let active: BTreeSet<String> = keys.iter().map(|key| key.to_uppercase()).collect();
    if active.contains("UP") && active.contains("DOWN") {
        return Err("simultaneous UP+DOWN depends on dispatcher event order".to_string());
    }
    if active.contains("LEFT") && active.contains("RIGHT") {
        return Err("simultaneous LEFT+RIGHT depends on dispatcher event order".to_string());
    }
    Ok(active)
}

/// `select_surface_mode_from_contacts` result (kartrider_vehicle_engine.py:97-175).
/// Fields absent from the Python dict in the `vehicle_parameters` fallback
/// path are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct ContactSurfaceSelection {
    pub surface_mode: i64,
    pub authority: &'static str,
    pub selection_rule: Option<&'static str>,
    pub road_surface_tokens: Option<Vec<String>>,
    pub mode_counts: Option<BTreeMap<i64, i64>>,
    pub dynamics_branches: Option<Vec<Option<String>>>,
    pub df_payloads: Option<Vec<Option<f64>>>,
    /// Key present (possibly null) only on the autonomous path.
    pub selected_dynamics_branch: Option<Option<String>>,
}

/// Python `parse_df_surface_token(...)` result.
#[derive(Debug, Clone, PartialEq)]
pub struct DfSurfaceToken {
    pub variant: &'static str,
    pub payload: f64,
    pub token: String,
}

/// Parse one complete `[+-]?\d+(?:\.\d+)?` literal.
fn parse_df_number_literal(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut index = 0;
    if bytes
        .first()
        .is_some_and(|byte| *byte == b'+' || *byte == b'-')
    {
        index += 1;
    }
    let digits_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    if index == digits_start {
        return None;
    }
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fraction_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == fraction_start {
            return None;
        }
    }
    if index != bytes.len() {
        return None;
    }
    let value: f64 = text.parse().ok()?;
    value.is_finite().then_some(value)
}

/// Python `parse_df_surface_token(token)` — decode the complete DF numeric
/// payload before the scoped selector decides whether game evidence covers
/// that value.
pub fn parse_df_surface_token(token: &str) -> Result<DfSurfaceToken, String> {
    let normalized = token.trim().to_uppercase();
    let Some(payload_text) = normalized.strip_prefix("DF") else {
        return Err(format!("unsupported DF authored token: {token}"));
    };
    let payload_text = payload_text.trim_start();
    let Some(payload) = parse_df_number_literal(payload_text) else {
        return Err(format!("unsupported DF authored token: {token}"));
    };
    Ok(DfSurfaceToken {
        variant: "DF",
        payload,
        token: normalized,
    })
}

/// `select_surface_mode_from_contacts(ground_planes, fallback_mode=...)`.
///
/// Empty `road_surface` and game-adjudicated `DF2.0` are ordinary mode 0,
/// `jump`/`점프` and authored `JM`/`DJ` payloads are explicit dynamics
/// branches inside mode 0, `리셋` is an authoritative RESET branch, and
/// `slip` is generic mode 2. Unknown tokens and unadjudicated DF payloads are
/// rejected.
pub fn select_surface_mode_from_contacts(
    ground_plane: Option<&GroundPlaneSet>,
    fallback_mode: i64,
) -> Result<ContactSurfaceSelection, String> {
    let Some(ground_plane) = ground_plane else {
        return Ok(ContactSurfaceSelection {
            surface_mode: fallback_mode,
            authority: "vehicle_parameters",
            selection_rule: None,
            road_surface_tokens: None,
            mode_counts: None,
            dynamics_branches: None,
            df_payloads: None,
            selected_dynamics_branch: None,
        });
    };
    let planes = ground_plane.selector_planes();
    let mut modes: Vec<i64> = Vec::with_capacity(planes.len());
    let mut tokens: Vec<String> = Vec::with_capacity(planes.len());
    let mut dynamics_branches: Vec<Option<String>> = Vec::with_capacity(planes.len());
    let mut df_payloads: Vec<Option<f64>> = Vec::with_capacity(planes.len());
    for plane in planes {
        let token = plane
            .road_surface
            .as_deref()
            .unwrap_or("")
            .trim()
            .to_lowercase();
        tokens.push(token.clone());
        if token.is_empty() {
            modes.push(0);
            dynamics_branches.push(None);
            df_payloads.push(None);
        } else if token == "slip" {
            modes.push(2);
            dynamics_branches.push(None);
            df_payloads.push(None);
        } else if token == "jump" || token == "점프" {
            modes.push(0);
            dynamics_branches.push(Some("jump".to_string()));
            df_payloads.push(None);
        } else if token.starts_with("jm") || token.starts_with("dj") {
            // Validate the complete authored payload here. Prefix matching
            // alone would silently admit malformed map tokens.
            let decoded = parse_jm_dj_surface_token(&token)?;
            let variant = match &decoded {
                JmDjSurfaceToken::Jm { .. } => "jm",
                JmDjSurfaceToken::Dj { .. } => "dj",
            };
            modes.push(0);
            dynamics_branches.push(Some(format!("authored_{variant}_trigger")));
            df_payloads.push(None);
        } else if token.starts_with("df") {
            let decoded = parse_df_surface_token(&token)?;
            if decoded.payload != 2.0 {
                return Err(format!(
                    "unsupported DF authored payload without game adjudication: {}",
                    py_float_repr(decoded.payload)
                ));
            }
            modes.push(0);
            dynamics_branches.push(None);
            df_payloads.push(Some(decoded.payload));
        } else if token == "리셋" {
            modes.push(0);
            dynamics_branches.push(Some("reset".to_string()));
            df_payloads.push(None);
        } else {
            return Err(format!(
                "unsupported autonomous road_surface token: {token}"
            ));
        }
    }
    let special_branches: Vec<&Option<String>> = dynamics_branches
        .iter()
        .filter(|branch| branch.as_deref().is_some_and(|branch| branch != "reset"))
        .collect();
    if !dynamics_branches
        .iter()
        .any(|branch| branch.as_deref() == Some("reset"))
        && modes.contains(&2)
        && !special_branches.is_empty()
    {
        return Err(
            "simultaneous jump and slip or authored-trigger and slip surface \
             precedence is unproven"
                .to_string(),
        );
    }
    let mut counts: BTreeMap<i64, i64> = BTreeMap::new();
    for mode in &modes {
        *counts.entry(*mode).or_insert(0) += 1;
    }
    let has_reset = dynamics_branches
        .iter()
        .any(|branch| branch.as_deref() == Some("reset"));
    let has_jump = dynamics_branches
        .iter()
        .any(|branch| branch.as_deref() == Some("jump"));
    let selected_branch: Option<String> = if has_reset {
        Some("reset".to_string())
    } else if has_jump {
        Some("jump".to_string())
    } else {
        special_branches
            .first()
            .and_then(|branch| (*branch).clone())
    };
    let selected_mode = if selected_branch.as_deref() == Some("reset") {
        0
    } else if counts.get(&2).copied().unwrap_or(0) > 0 {
        2
    } else {
        0
    };
    Ok(ContactSurfaceSelection {
        surface_mode: selected_mode,
        authority: "autonomous_world_contacts",
        selection_rule: Some(if selected_branch.as_deref() == Some("reset") {
            "reset_precedes_all_surface_dynamics"
        } else {
            "slip_precedes_ordinary_contacts"
        }),
        road_surface_tokens: Some(tokens),
        mode_counts: Some(counts),
        dynamics_branches: Some(dynamics_branches),
        df_payloads: Some(df_payloads),
        selected_dynamics_branch: Some(selected_branch),
    })
}

/// `_wheel_anchor` (kartrider_vehicle_engine.py:703-758): build one B28D30
/// wheel anchor with the selected store discipline.
pub fn wheel_anchor(
    position: [f64; 3],
    lateral: [f64; 3],
    forward: [f64; 3],
    up: [f64; 3],
    parameters: &VehicleParameters,
    lateral_sign: f64,
    forward_sign: f64,
    numeric_mode: NumericMode,
) -> [f64; 3] {
    let mut result = [0.0; 3];
    for index in 0..3 {
        if numeric_mode.is_float32() {
            let base = fadd(
                position[index],
                fmul(up[index], parameters.contact_rest_compression),
            );
            let lateral_offset = fmul(
                fmul(
                    fmul(lateral[index], parameters.collision_half_extents[0]),
                    parameters.contact_probe_span_scale,
                ),
                lateral_sign,
            );
            let forward_offset = fmul(
                fmul(
                    fmul(forward[index], parameters.collision_half_extents[1]),
                    parameters.contact_probe_span_scale,
                ),
                forward_sign,
            );
            result[index] = fadd(fadd(base, lateral_offset), forward_offset);
        } else {
            result[index] = position[index]
                + up[index] * parameters.contact_rest_compression
                + lateral[index]
                    * parameters.collision_half_extents[0]
                    * parameters.contact_probe_span_scale
                    * lateral_sign
                + forward[index]
                    * parameters.collision_half_extents[1]
                    * parameters.contact_probe_span_scale
                    * forward_sign;
        }
    }
    result
}

/// `derive_autonomous_ground_projection` result
/// (kartrider_vehicle_engine.py:803-810; the static `authority` and
/// `runtime_correspondence` metadata keys are not ported).
#[derive(Debug, Clone, PartialEq)]
pub struct GroundProjection {
    pub field_adc: f64,
    pub field_ae0: f64,
    pub projection_len: f64,
    pub normalized_lateral_velocity: f64,
    pub ground_normalized_lateral_velocity: f64,
    pub angular_correction: f64,
    pub longitudinal_angular_correction: f64,
    pub lateral_angular_correction: f64,
    pub correction_denominator: f64,
    pub low_projection_adapter: bool,
    pub drive_direction: f64,
    pub axis_forward: [f64; 3],
    pub axis_lateral: [f64; 3],
}

/// `derive_autonomous_ground_projection(state, parameters, numeric_mode=...)`
/// (kartrider_vehicle_engine.py:761-828): derive the B2A100/ground-drive
/// velocity projection from owned state.
pub fn derive_autonomous_ground_projection(
    state: &VehicleState,
    parameters: &VehicleParameters,
    numeric_mode: NumericMode,
) -> Result<GroundProjection, String> {
    parameters.validate()?;
    let columns = matrix_columns(state.matrix);
    let axis_forward: [f64; 3] = {
        let column = columns[parameters.forward_axis_column as usize];
        [
            parameters.forward_axis_sign * column[0],
            parameters.forward_axis_sign * column[1],
            parameters.forward_axis_sign * column[2],
        ]
    };
    let lateral_column = if parameters.forward_axis_column == 1 {
        0
    } else {
        1
    };
    let axis_lateral = columns[lateral_column];
    let (adc, ae0, projection_len) = if numeric_mode.is_float32() {
        let adc = dot3_f32(state.velocity, axis_forward);
        let ae0 = dot3_f32(state.velocity, axis_lateral);
        (adc, ae0, length3_f32([adc, ae0, 0.0]))
    } else {
        let adc = dot3_sum(state.velocity, axis_forward);
        let ae0 = dot3_sum(state.velocity, axis_lateral);
        // Python math.hypot: CPython's double-double vector_norm, not the
        // platform libm hypot (see num::cpython_hypot2).
        (adc, ae0, cpython_hypot2(adc, ae0))
    };
    // GoPlayKart_ApplyGroundDriveForces uses the true projection length above
    // 5, and its 5-unit adapter below that boundary. +0xD4 is omega.z;
    // dword_11CEAFC is 0.5. Thus both high- and low-speed branches can be
    // derived from engine-owned rigid-body state without a replay scalar.
    let correction_denominator = max2_f64(projection_len, parameters.ground_projection_floor);
    let angular_correction = if numeric_mode.is_float32() {
        fdiv(
            fmul(
                state.angular_velocity[2],
                parameters.ground_angular_correction_scale,
            ),
            correction_denominator,
        )
    } else {
        state.angular_velocity[2] * parameters.ground_angular_correction_scale
            / correction_denominator
    };
    let normalized_lateral = if numeric_mode.is_float32() && projection_len != 0.0 {
        fdiv(ae0, projection_len)
    } else if projection_len != 0.0 {
        ae0 / projection_len
    } else {
        0.0
    };
    let ground_normalized_lateral = if numeric_mode.is_float32() {
        f32q(f32q(ae0) / f32q(correction_denominator))
    } else {
        ae0 / correction_denominator
    };
    Ok(GroundProjection {
        field_adc: adc,
        field_ae0: ae0,
        projection_len,
        normalized_lateral_velocity: normalized_lateral,
        ground_normalized_lateral_velocity: ground_normalized_lateral,
        angular_correction,
        longitudinal_angular_correction: angular_correction,
        lateral_angular_correction: angular_correction,
        correction_denominator,
        low_projection_adapter: projection_len <= parameters.ground_projection_floor,
        drive_direction: if adc >= 0.0 { 1.0 } else { -1.0 },
        axis_forward,
        axis_lateral,
    })
}

/// `query_wheel_ground(world, state, parameters)`
/// (kartrider_vehicle_engine.py:178-204): query the four B28D30 suspension
/// rays from engine-owned pose. The Python `world.query_ground` presence
/// guard is type-enforced by the [`World`] trait; the engine calls the
/// world with its default `min_dot=0.65`.
pub fn query_wheel_ground(
    world: &dyn World,
    state: &VehicleState,
    parameters: &VehicleParameters,
) -> Vec<WorldGroundResult> {
    let columns = matrix_columns(state.matrix);
    let lateral = columns[0];
    let forward = [-columns[1][0], -columns[1][1], -columns[1][2]];
    let up = columns[2];
    let mut results = Vec::with_capacity(4);
    for (lateral_sign, forward_sign) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        // Python validates numeric position components here; the Rust type
        // enforces it. The anchor always uses the float32 store discipline
        // (numeric_mode="float32" at :199).
        let origin = wheel_anchor(
            state.position,
            lateral,
            forward,
            up,
            parameters,
            lateral_sign,
            forward_sign,
            NumericMode::Float32,
        );
        results.push(world.query_ground(origin, up, 0.65));
    }
    results
}

/// `_norm_delta` (kartrider_vehicle_engine.py:635-636).
fn norm_delta(a: &[f64], b: &[f64]) -> f64 {
    let squares: Vec<f64> = a
        .iter()
        .zip(b.iter())
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .collect();
    sum_f64(&squares).sqrt()
}

// Engine-layer Python `_dot3` call sites use `num::dot3_sum`
// (kartrider_vehicle_engine.py:663-664 is `sum(a[i] * b[i])`, i.e. CPython
// Neumaier summation, NOT a naive left fold). The distinction is
// observable: with `forward_axis_sign = -1.0` the forward axis carries
// `-0.0` components and `sum([...])` starting at int `0` yields `+0.0`
// where a naive fold yields `-0.0` (this broke
// `drive_target.exponent_input` sign parity in float64 mode).

/// `frame["accumulator_input"]` (`{"force": [...], "torque": [...]}`).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct AccumulatorInput {
    pub force: Option<[f64; 3]>,
    pub torque: Option<[f64; 3]>,
}

/// `frame["kinematic_input"]` (`{"velocity": [...], "angular_velocity": [...]}`).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct KinematicInput {
    pub velocity: Option<[f64; 3]>,
    pub angular_velocity: Option<[f64; 3]>,
}

/// `frame["observed_after"]`: observed post-step values compared against
/// the integrated state for `prediction_errors`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ObservedAfter {
    pub position: Option<[f64; 3]>,
    pub velocity: Option<[f64; 3]>,
    pub angular_velocity: Option<[f64; 3]>,
    pub matrix: Option<[f64; 9]>,
}

/// Python replay frame dict consumed by `step` (kartrider_vehicle_engine.py:5106).
/// `model` is the branch-aware payload; `step` binds the engine-owned
/// fields (elapsed_ms, boost state, accumulator/integrator befores) exactly
/// like `_bind_state`, overwriting whatever the model carries in those
/// slots.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct StepFrame {
    /// Positive integer millisecond count (validated; Python accepts an
    /// integer-valued float).
    pub elapsed_ms: f64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub authoritative_before: Option<BTreeMap<String, JsonValue>>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub accumulator_input: Option<AccumulatorInput>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub kinematic_input: Option<KinematicInput>,
    pub model: BranchAwareStepArgs,
    #[cfg_attr(feature = "serde", serde(default))]
    pub observed_after: Option<ObservedAfter>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub source: Option<BTreeMap<String, JsonValue>>,
}

/// Python `step` result dict (kartrider_vehicle_engine.py:5177-5191).
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutcome {
    pub step_index: u64,
    pub time_ms: u64,
    pub numeric_mode: NumericMode,
    pub teacher_forced_before: bool,
    pub accumulator_input_applied: bool,
    pub kinematic_input_applied: bool,
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub prediction_errors: BTreeMap<String, f64>,
    pub source: Option<BTreeMap<String, JsonValue>>,
    pub selected_ground_force_producer: Option<&'static str>,
    pub boost_gauge_producer_transition: Option<CurrentFamilyGaugeProducerTransition>,
    pub ordered_events: &'static [&'static str],
    pub evaluator: BranchAwareStepResult,
}

// ---------------------------------------------------------------------------
// Control path (step_control / step_raw_control, kartrider_vehicle_engine.py
// :5193-5368)
// ---------------------------------------------------------------------------

/// Python `control["drive_target_parameters"]`: the kwargs of
/// `evaluate_drive_target_state` minus the engine-supplied positionals
/// (`signed_axis`, `selected_slot`, `field_aa8_before`).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct DriveTargetStateParameters {
    pub scaled_tune_660: f64,
    pub field_adc: f64,
    pub field_71c: f64,
    pub decoded_bb8: f64,
}

/// Python `control["drive_target_runtime"]`: the kwargs of
/// `evaluate_drive_target_runtime` minus the engine-supplied positionals.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct DriveTargetRuntimeParameters {
    pub field_718: f64,
    pub field_adc: f64,
    pub field_71c: f64,
    pub decoded_bb8: f64,
}

/// Python `control["ground_runtime"]`: the three positional runtime locals of
/// `select_ground_runtime_input` plus its `**fields`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct GroundRuntimeControl {
    pub field_adc: f64,
    pub field_ae0: f64,
    pub field_d4: f64,
    #[cfg_attr(feature = "serde", serde(flatten))]
    pub fields: crate::domains::ground_runtime_input::GroundRuntimeFields,
}

/// Python `frame["control"]` payload consumed by `step_control`
/// (kartrider_vehicle_engine.py:5195). `input_mode` defaults to
/// `"empirical_response"` exactly like `control.get("input_mode", ...)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct ControlPayload {
    #[cfg_attr(feature = "serde", serde(default))]
    pub keys: Vec<String>,
    #[cfg_attr(feature = "serde", serde(default = "default_control_input_mode"))]
    pub input_mode: String,
    #[cfg_attr(feature = "serde", serde(default))]
    pub forward_axis: Option<[f64; 3]>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sub_660: Option<f64>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub selector_a9c: i64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sign_aac: i64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub sign_abc: i64,
    #[cfg_attr(feature = "serde", serde(default))]
    pub drive_target_parameters: Option<DriveTargetStateParameters>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub drive_target_runtime: Option<DriveTargetRuntimeParameters>,
    #[cfg_attr(feature = "serde", serde(default))]
    pub ground_runtime: Option<GroundRuntimeControl>,
}

#[cfg(feature = "serde")]
fn default_control_input_mode() -> String {
    "empirical_response".to_string()
}

/// Python `prediction["drive_target"]`: either evaluator's result dict,
/// selected by which of `drive_target_parameters` / `drive_target_runtime`
/// the control payload carried.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlDriveTarget {
    /// `evaluate_drive_target_state(...)` result.
    State(DriveTargetStateResult),
    /// `evaluate_drive_target_runtime(...)` result.
    Runtime(DriveTargetRuntimeResult),
}

impl ControlDriveTarget {
    /// `drive_target["field_aa4_after"]`.
    pub fn field_aa4_after(&self) -> f64 {
        match self {
            ControlDriveTarget::State(result) => result.field_aa4_after,
            ControlDriveTarget::Runtime(result) => result.state.field_aa4_after,
        }
    }

    /// `drive_target["field_aa8_after"]`.
    pub fn field_aa8_after(&self) -> f64 {
        match self {
            ControlDriveTarget::State(result) => result.field_aa8_after,
            ControlDriveTarget::Runtime(result) => result.state.field_aa8_after,
        }
    }
}

/// Python `result["control_prediction"]` (kartrider_vehicle_engine.py:5286).
/// Mode-specific keys are `Option`: the empirical prediction carries
/// `speed_band`/`longitudinal_accel_mps2`/`heading_rate_rad_s`/
/// `acceleration_world`/`force_accumulator_614`/`heading_rate_scope`; the
/// exact prediction carries `speed`; the exact drive-target transaction
/// carries `drive_target` and optionally `ground_runtime_adapter` /
/// `ground_runtime_branch`.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlPrediction {
    pub action: String,
    pub speed_band: Option<&'static str>,
    pub speed: Option<f64>,
    pub longitudinal_accel_mps2: Option<f64>,
    pub heading_rate_rad_s: Option<f64>,
    pub steering_selected: bool,
    pub acceleration_world: Option<[f64; 3]>,
    pub force_accumulator_614: Option<[f64; 3]>,
    pub source: String,
    pub heading_rate_scope: Option<&'static str>,
    pub scope: &'static str,
    pub drive_basis: DriveBasisInput,
    pub input_mode: &'static str,
    pub drive_target: Option<ControlDriveTarget>,
    pub ground_runtime_adapter: Option<&'static str>,
    pub ground_runtime_branch: Option<&'static str>,
}

/// Python `step_control` / `step_raw_control` result: the `step` outcome
/// plus the control prediction, the world sweep when dispatched through
/// `step_world`, and the `raw_control_pose_contract` source snapshot
/// appended by `step_raw_control` (kartrider_vehicle_engine.py:5367).
#[derive(Debug, Clone, PartialEq)]
pub struct ControlStepOutcome {
    pub step: StepOutcome,
    pub control_prediction: ControlPrediction,
    pub world_sweep: Option<WorldSweepInfo>,
    pub raw_control_pose_contract: Option<BTreeMap<String, JsonValue>>,
}

/// `frame["authoritative_before"]["velocity"]`: decode a JSON 3-vector
/// (Python passes the replay list through unchanged).
fn json_value_vec3(value: &JsonValue) -> Result<[f64; 3], String> {
    match value {
        JsonValue::Array(items) if items.len() == 3 => {
            let mut result = [0.0; 3];
            for (index, item) in items.iter().enumerate() {
                result[index] = json_value_f64(item)?;
            }
            Ok(result)
        }
        _ => Err("authoritative velocity must be a 3-component array".to_string()),
    }
}

/// `float(authoritative_before.get("drive_aa8", ...))`.
fn json_value_f64(value: &JsonValue) -> Result<f64, String> {
    match value {
        JsonValue::Number(number) => Ok(*number),
        _ => Err("authoritative numeric field must be a number".to_string()),
    }
}

/// Engine frame-level surface selection: the sampled
/// [`ContactSurfaceSelection`] plus the 60 Hz selector latch fields added at
/// kartrider_vehicle_engine.py:5410-5417.
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceSelection {
    pub surface_mode: i64,
    pub authority: &'static str,
    pub selection_rule: Option<&'static str>,
    pub road_surface_tokens: Option<Vec<String>>,
    pub mode_counts: Option<BTreeMap<i64, i64>>,
    pub dynamics_branches: Option<Vec<Option<String>>>,
    pub df_payloads: Option<Vec<Option<f64>>>,
    pub selected_dynamics_branch: Option<Option<String>>,
    pub sampled_surface_mode: i64,
    pub selector_refreshed: bool,
    pub selector_period_ms: f64,
    pub selector_elapsed_ms: f64,
}

/// One `contact_probe_debug` entry (kartrider_vehicle_engine.py:5590-5601).
#[derive(Debug, Clone, PartialEq)]
pub struct ContactProbeDebug {
    pub origin: [f64; 3],
    pub ray: [f64; 3],
    pub ray_t: Option<f64>,
    pub hit_point: [f64; 3],
    pub hit_projection: Option<f64>,
    pub position_projection: Option<f64>,
    pub compression: f64,
    pub plane_normal: [f64; 3],
    pub plane_point: [f64; 3],
    pub plane_node: Option<JsonValue>,
    pub plane_tri_index: Option<i64>,
    pub plane_surface: String,
}

/// Python `producer_contract` dict (kartrider_vehicle_engine.py:5904-5910).
#[derive(Debug, Clone, PartialEq)]
pub struct ProducerContract {
    pub state_mutation: &'static str,
    pub integration_stages: &'static str,
    pub authority: &'static str,
}

impl Default for ProducerContract {
    fn default() -> Self {
        ProducerContract {
            state_mutation: "none",
            integration_stages: "not_scheduled",
            authority: "pure_producer_frame_from_explicit_vehicle_state",
        }
    }
}

/// `produce_ground_accumulators` result (kartrider_vehicle_engine.py:5898-5911).
#[derive(Debug, Clone, PartialEq)]
pub struct GroundProducerOutput {
    pub evaluator: BranchAwareStepResult,
    pub surface_selection: SurfaceSelection,
    pub contact_compression_after: [f64; 4],
    pub contact_probe_debug: Vec<ContactProbeDebug>,
    pub drive_target: DriveTargetRuntimeResult,
    pub producer_contract: ProducerContract,
}

/// Python `pending_producer_refresh` payload (kartrider_vehicle_engine.py:5976-5994).
/// The evaluator sub-keys (`ground_drive_force`, `low_projection_branch`,
/// `boost_ground_primary`, `contact_accumulator`, the post-BS accumulators,
/// `common_damping`, `ground_correction`) are read out of
/// [`GroundProducerOutput::evaluator`].
#[derive(Debug, Clone, PartialEq)]
pub struct PendingProducerRefresh {
    pub authority: &'static str,
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub contact_compression: [f64; 4],
    pub producer: GroundProducerOutput,
}

/// `step_raw_ground_correction` result (kartrider_vehicle_engine.py:5934-6006):
/// the `step` outcome plus the GND frame keys.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundStepOutcome {
    pub step: StepOutcome,
    pub surface_selection: SurfaceSelection,
    pub pending_producer_refresh: Option<PendingProducerRefresh>,
    pub drive_target: DriveTargetRuntimeResult,
    pub drift_selector_transition: Option<SelectorTransitionResult>,
    pub drift_selector_transitions: Vec<SelectorTransitionResult>,
    pub contact_probe_debug: Vec<ContactProbeDebug>,
    /// Python `raw_control_pose_contract` (deepcopy of `frame["source"]`).
    pub raw_control_pose_contract: BTreeMap<String, JsonValue>,
    /// Python `world_sweep` key, present only when a world was dispatched
    /// through `step_world`.
    pub world_sweep: Option<WorldSweepInfo>,
}

/// Python `_pending_ground_plane_provider(state, pending_parameters)`,
/// borrowed form. The frame-loop entry points thread a provider reference
/// down to `step_raw_ground_correction`; boxed ownership cannot be cloned
/// through those layers.
pub type PendingGroundPlaneProviderRef<'a> =
    &'a (dyn Fn(&VehicleState, &VehicleParameters) -> Option<GroundPlaneSet> + 'a);

/// Keyword switches of `step_raw_ground_correction`
/// (kartrider_vehicle_engine.py:5370-5378). `_rail_state1_ground_writer` is
/// threaded into `step_world`, which keeps its structured RAIL error while
/// the rail-state1 ground writer is out of scope.
pub struct GroundStepOptions<'a> {
    pub refresh_pending: bool,
    pub pending_parameters: Option<VehicleParameters>,
    pub pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'a>>,
    pub advance_surface_selector: bool,
    pub support_active: Option<bool>,
    pub authored_trigger_model: Option<JmDjArgs>,
    pub rail_state1_ground_writer: bool,
}

impl Default for GroundStepOptions<'_> {
    fn default() -> Self {
        GroundStepOptions {
            refresh_pending: true,
            pending_parameters: None,
            pending_ground_plane_provider: None,
            advance_surface_selector: true,
            support_active: None,
            authored_trigger_model: None,
            rail_state1_ground_writer: false,
        }
    }
}

/// Intermediate products of the GND frame construction shared by the
/// producer and integrator paths.
struct GroundFrameParts {
    args: BranchAwareStepArgs,
    source: BTreeMap<String, JsonValue>,
    surface_selection: SurfaceSelection,
    surface_mode_latched: i64,
    surface_selector_elapsed_ms: f64,
    drive_target: DriveTargetRuntimeResult,
    contact_probe_debug: Vec<ContactProbeDebug>,
    next_contact_compression: Option<[f64; 4]>,
    soft_contact_drive_slot: f64,
    signed_axis: f64,
    projection: GroundProjection,
    speed: f64,
}

/// One `_prediction_history` entry appended by `step_raw_motion`
/// (kartrider_vehicle_engine.py:6946-6957). `parameters` and
/// `ground_planes` are deep copies on the Python side; the Rust struct owns
/// them. `has_pending_ground_plane_provider` records provider *presence*
/// only, exactly like Python (closures are not snapshot-able).
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionHistoryEntry {
    pub time_ms_before: u64,
    pub time_ms_after: u64,
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub keys: Vec<String>,
    pub parameters: VehicleParameters,
    pub ground_planes: Option<Vec<WorldGroundResult>>,
    pub has_pending_ground_plane_provider: bool,
}

/// `KartRiderVehicleEngine` (kartrider_vehicle_engine.py:1060-1071): owns
/// cross-substep state; frame payloads contain selectors and parameters.
/// No physical latch lives here — everything cross-substep is in
/// [`VehicleState`] (handoff §6.2).
#[derive(Debug, Clone, PartialEq)]
pub struct KartRiderVehicleEngine {
    pub state: VehicleState,
    pub numeric_mode: NumericMode,
    /// Python `_prediction_history` (capped at [`PREDICTION_HISTORY_LIMIT`]);
    /// appended by `step_raw_motion` and cleared on NET-CORRECTION entry.
    pub prediction_history: VecDeque<PredictionHistoryEntry>,
    /// Python `_network_replay_active` (kartrider_vehicle_engine.py:1070):
    /// true only on the transient resimulation engine inside
    /// `rollback_and_resimulate_network_correction`; suppresses the world
    /// network-packet query and the history append during replay.
    pub network_replay_active: bool,
}

impl KartRiderVehicleEngine {
    /// Python `__init__` (:1063-1071): validate mode and state, then
    /// deepcopy the handed-in state (Rust `clone`).
    pub fn new(state: &VehicleState, numeric_mode: NumericMode) -> Result<Self, String> {
        state.validate()?;
        Ok(KartRiderVehicleEngine {
            state: state.clone(),
            numeric_mode,
            prediction_history: VecDeque::new(),
            network_replay_active: false,
        })
    }

    /// `_bind_state` (kartrider_vehicle_engine.py:5068-5104): inject the
    /// engine-owned fields into the model payload. Later producers in the
    /// same simulated transaction consume engine-owned velocity, never the
    /// captured replay value.
    fn bind_state(&self, args: &mut BranchAwareStepArgs, elapsed_ms: i64) {
        args.elapsed_ms = elapsed_ms;
        args.boost.state = self.state.boost_state;
        args.boost.remaining_ms = self.state.boost_remaining_ms;
        if let Some(accumulator) = &mut args.accumulator {
            accumulator.accumulator_614_before = self.state.force_accumulator;
            accumulator.accumulator_620_before = self.state.torque_accumulator;
        }
        if let Some(common) = &mut args.common_damping {
            common.velocity_c0 = Some(self.state.velocity);
            common.angular_velocity_cc = self.state.angular_velocity;
        }
        if let Some(ground_correction) = &mut args.ground_correction {
            // sub_B2A100 is an earlier producer in the same simulated
            // transaction. Its direct C0 writes and lateral selector must
            // consume engine-owned velocity, never the captured replay value.
            ground_correction.velocity_c0 = self.state.velocity;
        }
        if let Some(linear) = &mut args.linear_integrator {
            linear.velocity_c0_before = Some(self.state.velocity);
        }
        if let Some(angular) = &mut args.angular_integrator {
            angular.angular_velocity_cc_before = self.state.angular_velocity;
        }
        if let Some(position) = &mut args.position_integrator {
            position.position_90_before = self.state.position;
            position.velocity_c0_before = Some(self.state.velocity);
        }
        if let Some(orientation) = &mut args.orientation_integrator {
            orientation.matrix_9c_before = self.state.matrix;
            orientation.angular_velocity_cc_before = Some(self.state.angular_velocity);
        }
    }

    /// Apply `frame["authoritative_before"]` key by key
    /// (kartrider_vehicle_engine.py:5120-5123), rejecting unknown fields
    /// like Python's `hasattr` guard.
    #[cfg(feature = "serde")]
    fn apply_authoritative_before(
        &mut self,
        patch: &BTreeMap<String, JsonValue>,
    ) -> Result<(), String> {
        let mut value = serde_json::to_value(&self.state)
            .map_err(|error| format!("failed to snapshot state: {error}"))?;
        let object = value
            .as_object_mut()
            .expect("VehicleState serializes to a JSON object");
        for (key, field) in patch {
            if !object.contains_key(key) {
                return Err(format!("unknown authoritative state field: {key}"));
            }
            let encoded = serde_json::to_value(field)
                .map_err(|error| format!("failed to encode authoritative field {key}: {error}"))?;
            object.insert(key.clone(), encoded);
        }
        self.state = serde_json::from_value(value)
            .map_err(|error| format!("invalid authoritative_before payload: {error}"))?;
        Ok(())
    }

    #[cfg(not(feature = "serde"))]
    fn apply_authoritative_before(
        &mut self,
        patch: &BTreeMap<String, JsonValue>,
    ) -> Result<(), String> {
        if patch.is_empty() {
            return Ok(());
        }
        Err("authoritative_before application requires the serde feature".to_string())
    }

    /// `step(frame)` (kartrider_vehicle_engine.py:5106-5191): the
    /// replay-path step — validate integer elapsed_ms, apply the input
    /// overlays, evaluate the bound model, write back, quantize in float32
    /// mode, advance the integer clock and the RAIL-owned elapsed
    /// accumulator (actual widths only, handoff §6.1).
    pub fn step(&mut self, frame: &StepFrame) -> Result<StepOutcome, String> {
        let raw_elapsed_ms = frame.elapsed_ms;
        if !raw_elapsed_ms.is_finite() || raw_elapsed_ms <= 0.0 || raw_elapsed_ms.fract() != 0.0 {
            return Err("elapsed_ms must be a positive integer millisecond count".to_string());
        }
        let elapsed_ms = raw_elapsed_ms as i64;
        let teacher_forced = frame
            .authoritative_before
            .as_ref()
            .is_some_and(|patch| !patch.is_empty());
        if let Some(patch) = &frame.authoritative_before {
            self.apply_authoritative_before(patch)?;
        }
        let accumulator_input_applied = frame
            .accumulator_input
            .as_ref()
            .is_some_and(|input| input.force.is_some() || input.torque.is_some());
        if let Some(input) = &frame.accumulator_input {
            if let Some(force) = input.force {
                self.state.force_accumulator = force;
            }
            if let Some(torque) = input.torque {
                self.state.torque_accumulator = torque;
            }
        }
        let kinematic_input_applied = frame
            .kinematic_input
            .as_ref()
            .is_some_and(|input| input.velocity.is_some() || input.angular_velocity.is_some());
        if let Some(input) = &frame.kinematic_input {
            if let Some(velocity) = input.velocity {
                self.state.velocity = velocity;
            }
            if let Some(angular_velocity) = input.angular_velocity {
                self.state.angular_velocity = angular_velocity;
            }
        }
        self.state.validate()?;
        let before = self.state.clone();
        let mut args = frame.model.clone();
        let mut boost_gauge_producer_transition = None;
        if let Some(producer) = args.boost_gauge_producers.clone() {
            let dt = if self.numeric_mode.is_float32() {
                fdiv(elapsed_ms as f64, 1000.0)
            } else {
                elapsed_ms as f64 / 1000.0
            };
            let gauge_before = GaugeProducerState {
                b10: self.state.boost_gauge_committed_b10,
                b20: self.state.boost_gauge_progress_b20,
                b2c: self.state.boost_gauge_pending_b2c,
                b38: self.state.boost_gauge_history_b38,
            };
            let pending = evaluate_current_family_pending_gauge_producer(
                &producer,
                gauge_before,
                dt,
                self.numeric_mode.as_str(),
            )?;
            self.state.boost_gauge_committed_b10 = pending.state_after.b10;
            self.state.boost_gauge_progress_b20 = pending.state_after.b20;
            self.state.boost_gauge_pending_b2c = pending.state_after.b2c;
            self.state.boost_gauge_history_b38 = pending.state_after.b38;
            let primary = evaluate_current_family_primary_gauge_producer(
                &producer,
                pending.state_after,
                dt,
                self.numeric_mode.as_str(),
            )?;
            self.state.boost_gauge_committed_b10 = primary.state_after.b10;
            self.state.boost_gauge_progress_b20 = primary.state_after.b20;
            self.state.boost_gauge_pending_b2c = primary.state_after.b2c;
            self.state.boost_gauge_history_b38 = primary.state_after.b38;
            let state_after_producers = primary.state_after;
            for branch in [
                args.recovery.low_projection.as_mut().map(|value| {
                    (
                        &mut value.b10,
                        &mut value.b20,
                        &mut value.b2c,
                        &mut value.b38,
                    )
                }),
                args.recovery.ground_force.as_mut().map(|value| {
                    (
                        &mut value.b10,
                        &mut value.b20,
                        &mut value.b2c,
                        &mut value.b38,
                    )
                }),
            ]
            .into_iter()
            .flatten()
            {
                *branch.0 = Some(state_after_producers.b10);
                *branch.1 = Some(state_after_producers.b20);
                *branch.2 = Some(state_after_producers.b2c);
                *branch.3 = Some(state_after_producers.b38);
            }
            boost_gauge_producer_transition = Some(CurrentFamilyGaugeProducerTransition {
                pending_b2c: pending,
                primary_b10_b20: primary,
                state_after_producers,
                ordering: "pending_B26E00 -> primary_B270C0 -> accepted_B33800_noop_current_scope -> ground settlement",
                authority: "run25_bit_exact_current_family",
            });
        }
        self.bind_state(&mut args, elapsed_ms);
        let result = evaluate_branch_aware_step(&args)?;

        if let Some(position) = result.position_90_final {
            self.state.position = position;
        }
        if let Some(velocity) = result.velocity_c0_final {
            self.state.velocity = velocity;
        }
        if let Some(matrix) = result.matrix_9c_final {
            self.state.matrix = matrix;
        }
        if let Some(angular_velocity) = result.angular_velocity_cc_final {
            self.state.angular_velocity = angular_velocity;
        }
        if let Some(force) = result.accumulator_614_after_damping {
            self.state.force_accumulator = force;
        }
        if let Some(torque) = result.accumulator_620_after_damping {
            self.state.torque_accumulator = torque;
        }
        if let Some(correction) = &result.correction {
            self.state.ground_timer_a80 = correction.timer_a80_after;
        }
        if let Some(LowProjectionBranchResult::Selected(selected)) = &result.low_projection {
            let drift_state = &selected.state_after;
            self.state.soft_selector_af4 = drift_state.selector_af4;
            self.state.field_af8 = drift_state.field_af8;
            self.state.selector_b08 = drift_state.selector_b08;
            self.state.drift_timer_b0c = drift_state.field_b0c;
            self.state.selector_b5c = drift_state.selector_b5c;
            self.state.drift_b1c_charge_settlement = drift_state.b1c_charge_settlement;
            self.commit_gauge_settlement_state(
                drift_state.b10,
                drift_state.b20,
                drift_state.b2c,
                drift_state.b38,
            );
        } else {
            if let Some(DriftWindowBranchResult::Selected(selected)) = &result.drift_window {
                let drift_state = &selected.state_after;
                self.state.selector_b08 = drift_state.selector_b08;
                self.state.drift_timer_b0c = drift_state.timer_b0c;
                if let Some(field_b00) = drift_state.field_b00 {
                    self.state.drift_field_b00 = field_b00;
                }
                if let Some(snapshot_5d0) = drift_state.snapshot_5d0 {
                    self.state.drift_snapshot_5d0 = snapshot_5d0;
                }
                self.state.drift_b1c_charge_settlement = drift_state.b1c_charge_settlement;
            } else if let Some(ground_force) = &result.ground_force {
                let gauge_state = &ground_force.state_after;
                self.state.drift_b1c_charge_settlement = gauge_state.b1c_charge_settlement;
                self.commit_gauge_settlement_state(
                    gauge_state.b10,
                    gauge_state.b20,
                    gauge_state.b2c,
                    gauge_state.b38,
                );
            }
            if let RecoveryScheduleOutcome::Evaluated(recovery) = &result.recovery_schedule {
                self.state.recovery_timer = recovery.recovery_timer_after;
                self.state.field_af8 = recovery.field_af8_after;
            }
            if let Some(recovery_force) = &result.recovery_force {
                self.state.drift_blend_ab0 = recovery_force.blend_ab0_after;
                self.state.drift_blend_ab4 = recovery_force.blend_ab4_after;
            }
        }
        if self.numeric_mode.is_float32() {
            f32_vector(&mut self.state.position);
            f32_vector(&mut self.state.velocity);
            f32_vector(&mut self.state.angular_velocity);
            f32_vector(&mut self.state.matrix);
            f32_vector(&mut self.state.force_accumulator);
            f32_vector(&mut self.state.torque_accumulator);
            self.state.drive_aa4 = f32q(self.state.drive_aa4);
            self.state.drive_aa8 = f32q(self.state.drive_aa8);
            self.state.ground_timer_a80 = f32q(self.state.ground_timer_a80);
        }

        let timer = &result.boost.timer_after;
        self.state.boost_state = timer.state;
        self.state.boost_remaining_ms = timer.remaining_ms;
        // The Python trigger loop also reads bs_zone, but the bs_zone
        // evaluator result carries no `state_transition` key, so only
        // mz_magnet / jm_dj can override the timer here.
        if let Some(mz_magnet) = &result.mz_magnet {
            if let Some(transition) = &mz_magnet.state_transition {
                self.state.boost_state = transition.state;
                self.state.boost_remaining_ms = transition.remaining_ms;
            }
        }
        if let Some(jm_dj) = &result.jm_dj {
            if let Some(transition) = &jm_dj.state_transition {
                self.state.boost_state = transition.state;
                self.state.boost_remaining_ms = transition.remaining_ms;
            }
        }
        self.state.time_ms += elapsed_ms as u64;
        self.state.step_index += 1;
        if self.state.topology.domain == MotionDomain::Rail {
            self.state.topology.rail_state_elapsed_ms += elapsed_ms as f64;
        }

        let mut errors = BTreeMap::new();
        if let Some(observed) = &frame.observed_after {
            if let Some(position) = &observed.position {
                errors.insert(
                    "position_l2".to_string(),
                    norm_delta(&self.state.position, position),
                );
            }
            if let Some(velocity) = &observed.velocity {
                errors.insert(
                    "velocity_l2".to_string(),
                    norm_delta(&self.state.velocity, velocity),
                );
            }
            if let Some(angular_velocity) = &observed.angular_velocity {
                errors.insert(
                    "angular_velocity_l2".to_string(),
                    norm_delta(&self.state.angular_velocity, angular_velocity),
                );
            }
            if let Some(matrix) = &observed.matrix {
                errors.insert(
                    "matrix_l2".to_string(),
                    norm_delta(&self.state.matrix, matrix),
                );
            }
        }
        Ok(StepOutcome {
            step_index: self.state.step_index,
            time_ms: self.state.time_ms,
            numeric_mode: self.numeric_mode,
            teacher_forced_before: teacher_forced,
            accumulator_input_applied,
            kinematic_input_applied,
            state_before: before,
            state_after: self.state.clone(),
            prediction_errors: errors,
            source: frame.source.clone(),
            selected_ground_force_producer: result.selected_ground_force_producer,
            boost_gauge_producer_transition,
            ordered_events: ORDERED_EVENTS,
            evaluator: result,
        })
    }

    fn commit_gauge_settlement_state(&mut self, b10: f64, b20: f64, b2c: f64, b38: f64) {
        if self.numeric_mode.is_float32() {
            self.state.boost_gauge_committed_b10 = f32q(b10);
            self.state.boost_gauge_progress_b20 = f32q(b20);
            self.state.boost_gauge_pending_b2c = f32q(b2c);
            self.state.boost_gauge_history_b38 = f32q(b38);
        } else {
            self.state.boost_gauge_committed_b10 = b10;
            self.state.boost_gauge_progress_b20 = b20;
            self.state.boost_gauge_pending_b2c = b2c;
            self.state.boost_gauge_history_b38 = b38;
        }
    }

    /// `step_control(frame, control_model)` (kartrider_vehicle_engine.py
    /// :5193-5287): drive one replay-path step from a raw-key control
    /// payload. The empirical mode injects the trace-response-model force
    /// (and steering heading rate) as accumulator/kinematic inputs; the
    /// `exact_drive_basis` mode runs the stateful AA4/AA8 producer and wires
    /// its output into the recovery ground body instead.
    ///
    /// `control_model` may be `None` only for `exact_drive_basis` frames;
    /// Python raises `AttributeError` on `None.predict` for empirical frames
    /// (the `step_world` guard message is used here so both entries report
    /// the same contract).
    pub fn step_control(
        &mut self,
        frame: &StepFrame,
        control: &ControlPayload,
        control_model: Option<&ControlResponseInputModel>,
    ) -> Result<ControlStepOutcome, String> {
        // payload = copy.deepcopy(frame); control = payload.pop("control")
        let mut payload = frame.clone();
        let authoritative_velocity = frame
            .authoritative_before
            .as_ref()
            .and_then(|patch| patch.get("velocity"))
            .map(json_value_vec3)
            .transpose()?;
        let drive_aa8_before = frame
            .authoritative_before
            .as_ref()
            .and_then(|patch| patch.get("drive_aa8"))
            .map(json_value_f64)
            .transpose()?
            .unwrap_or(self.state.drive_aa8);
        let control_velocity = authoritative_velocity.unwrap_or(self.state.velocity);
        let speed = length3(control_velocity, self.numeric_mode.is_float32());
        let empirical = match control.input_mode.as_str() {
            "empirical_response" => true,
            "exact_drive_basis" => false,
            _ => {
                return Err(
                    "control input_mode must be empirical_response or exact_drive_basis"
                        .to_string(),
                )
            }
        };
        let mut prediction = if empirical {
            let model = control_model
                .ok_or_else(|| "empirical control frames require a control_model".to_string())?;
            let forward_axis = control
                .forward_axis
                .ok_or_else(|| "KeyError: 'forward_axis'".to_string())?;
            let sub_660 = control
                .sub_660
                .ok_or_else(|| "KeyError: 'sub_660'".to_string())?;
            let predicted = model.predict(&control.keys, speed, forward_axis, sub_660)?;
            ControlPrediction {
                action: predicted.action,
                speed_band: Some(predicted.speed_band),
                speed: None,
                longitudinal_accel_mps2: Some(predicted.longitudinal_accel_mps2),
                heading_rate_rad_s: Some(predicted.heading_rate_rad_s),
                steering_selected: predicted.steering_selected,
                acceleration_world: Some(predicted.acceleration_world),
                force_accumulator_614: Some(predicted.force_accumulator_614),
                source: predicted.source,
                heading_rate_scope: Some(
                    crate::domains::control_response_input::HEADING_RATE_SCOPE,
                ),
                scope: crate::domains::control_response_input::PREDICT_SCOPE,
                drive_basis: DriveBasisInput {
                    drive_raw_aa0: 0.0,
                    drive_sign_aac: 0,
                    drive_signed_a80: 0.0,
                    drive_slot0_a84: 0.0,
                    drive_slot1_a90: 0.0,
                    drive_slot_selector_a9c: 0,
                    drive_selected_slot: 0.0,
                    drive_opposite_slot: 0.0,
                    dispatch_left_right_signed_axis: None,
                    dispatch_up_slot0: None,
                    dispatch_down_slot1: None,
                },
                input_mode: "empirical_response",
                drive_target: None,
                ground_runtime_adapter: None,
                ground_runtime_branch: None,
            }
        } else {
            let action = canonical_action(&control.keys);
            ControlPrediction {
                steering_selected: action.contains("LEFT") || action.contains("RIGHT"),
                action,
                speed_band: None,
                speed: Some(speed),
                longitudinal_accel_mps2: None,
                heading_rate_rad_s: None,
                acceleration_world: None,
                force_accumulator_614: None,
                source: "exact_drive_basis_dispatch".to_string(),
                heading_rate_scope: None,
                scope: "exact arrow-key dispatcher and stateful AA4 producer; no empirical force or heading-rate injection",
                drive_basis: DriveBasisInput {
                    drive_raw_aa0: 0.0,
                    drive_sign_aac: 0,
                    drive_signed_a80: 0.0,
                    drive_slot0_a84: 0.0,
                    drive_slot1_a90: 0.0,
                    drive_slot_selector_a9c: 0,
                    drive_selected_slot: 0.0,
                    drive_opposite_slot: 0.0,
                    dispatch_left_right_signed_axis: None,
                    dispatch_up_slot0: None,
                    dispatch_down_slot1: None,
                },
                input_mode: "exact_drive_basis",
                drive_target: None,
                ground_runtime_adapter: None,
                ground_runtime_branch: None,
            }
        };
        let basis = evaluate_drive_basis_input(
            &control.keys,
            control.selector_a9c,
            control.sign_aac,
            control.sign_abc,
        )?;
        prediction.drive_basis = basis;
        if !empirical {
            let drive_target = match (
                &control.drive_target_parameters,
                &control.drive_target_runtime,
            ) {
                (Some(parameters), None) => ControlDriveTarget::State(
                    evaluate_drive_target_state(
                        prediction.drive_basis.drive_signed_a80,
                        parameters.scaled_tune_660,
                        parameters.field_adc,
                        parameters.field_71c,
                        parameters.decoded_bb8,
                        prediction.drive_basis.drive_selected_slot,
                        drive_aa8_before,
                    )?,
                ),
                (None, Some(parameters)) => ControlDriveTarget::Runtime(
                    evaluate_drive_target_runtime(
                        prediction.drive_basis.drive_signed_a80,
                        parameters.field_718,
                        parameters.field_adc,
                        parameters.field_71c,
                        parameters.decoded_bb8,
                        prediction.drive_basis.drive_selected_slot,
                        drive_aa8_before,
                    )?,
                ),
                _ => {
                    return Err(
                        "exact_drive_basis requires exactly one of drive_target_parameters or drive_target_runtime"
                            .to_string(),
                    )
                }
            };
            let field_aa4_after = drive_target.field_aa4_after();
            match &control.ground_runtime {
                Some(ground_runtime) => {
                    if payload.model.recovery.ground_force.is_some() {
                        return Err(
                            "exact_drive_basis cannot combine ground_runtime with model.recovery.ground_force"
                                .to_string(),
                        );
                    }
                    let selection = select_ground_runtime_input(
                        field_aa4_after,
                        ground_runtime.field_adc,
                        ground_runtime.field_ae0,
                        ground_runtime.field_d4,
                        &ground_runtime.fields,
                    )?;
                    prediction.ground_runtime_adapter = Some(selection.runtime_adapter());
                    prediction.ground_runtime_branch = Some(selection.branch());
                    match selection {
                        GroundRuntimeSelection::LowProjection(selected) => {
                            payload.model.recovery.low_projection = Some(LowProjectionArgs {
                                projection_len: selected.projection_len,
                                ground_forward_term: selected.ground_forward_term,
                                drive_direction: selected.drive_direction,
                                normalized_velocity_raw: selected.normalized_velocity_raw,
                                correction_raw: selected.correction_raw,
                                mass_scale: selected.mass_scale,
                                base_longitudinal_gain: selected.base_longitudinal_gain,
                                base_lateral_gain: selected.base_lateral_gain,
                                axis_long: Some(selected.axis_long),
                                axis_lat: Some(selected.axis_lat),
                                state_5c8: None,
                                b1c_charge_settlement: None,
                                b10: None,
                                b20: None,
                                b2c: None,
                                b38: None,
                                cap_760: None,
                                numeric_mode: Some("float32".to_string()),
                            });
                        }
                        GroundRuntimeSelection::GroundForce(selected) => {
                            payload.model.recovery.ground_force = Some(GroundDriveForceArgs {
                                surface_mode: Some(selected.surface_mode),
                                speed_magnitude: selected.speed_magnitude,
                                ground_forward_term: selected.ground_forward_term,
                                contact_basis_product: selected.contact_basis_product,
                                drive_direction: selected.drive_direction,
                                normalized_velocity: selected.normalized_velocity,
                                longitudinal_correction: selected.longitudinal_correction,
                                lateral_correction: selected.lateral_correction,
                                mass_scale: selected.mass_scale,
                                base_longitudinal_gain: selected.base_longitudinal_gain,
                                base_lateral_gain: selected.base_lateral_gain,
                                ground_force_multiplier: selected.ground_force_multiplier,
                                settlement_multiplier: selected.settlement_multiplier,
                                tune_75c: selected.tune_75c,
                                axis_long: selected.axis_long,
                                axis_lat: selected.axis_lat,
                                surface_profile: None,
                                torque_y_scale: None,
                                torque_y_gain: Some(selected.torque_y_gain),
                                state_5c8: None,
                                b1c_charge_settlement: None,
                                b10: None,
                                b20: None,
                                b2c: None,
                                b38: None,
                                cap_760: None,
                                numeric_mode: Some("float32".to_string()),
                            });
                        }
                    }
                }
                None => match &mut payload.model.recovery.ground_force {
                    None => {
                        return Err(
                            "exact_drive_basis requires control.ground_runtime or model.recovery.ground_force"
                                .to_string(),
                        );
                    }
                    Some(ground_force) => {
                        ground_force.ground_forward_term = field_aa4_after;
                    }
                },
            }
            prediction.drive_target = Some(drive_target);
        }
        // Python truthiness: a present-but-empty dict is falsy. The typed
        // payload cannot carry a key-less dict, so presence of any field is
        // the conflict condition.
        if payload
            .accumulator_input
            .as_ref()
            .is_some_and(|input| input.force.is_some() || input.torque.is_some())
        {
            return Err(
                "control-derived and explicit accumulator_input cannot be combined".to_string(),
            );
        }
        if empirical {
            payload.accumulator_input = Some(AccumulatorInput {
                force: prediction.force_accumulator_614,
                torque: None,
            });
            if prediction.steering_selected {
                if payload.kinematic_input.as_ref().is_some_and(|input| {
                    input.velocity.is_some() || input.angular_velocity.is_some()
                }) {
                    return Err(
                        "control-derived and explicit kinematic_input cannot be combined"
                            .to_string(),
                    );
                }
                payload.kinematic_input = Some(KinematicInput {
                    velocity: None,
                    angular_velocity: Some([
                        0.0,
                        0.0,
                        prediction
                            .heading_rate_rad_s
                            .expect("empirical prediction carries the heading rate"),
                    ]),
                });
            }
        }
        let mut outcome = self.step(&payload)?;
        // Post-step drive selector commits (kartrider_vehicle_engine.py
        // :5270-5278). The result's state_after is refreshed only when the
        // drive-target transaction ran, exactly like Python.
        self.state.drive_selected_slot = prediction.drive_basis.drive_selected_slot;
        self.state.drive_raw_aa0 = prediction.drive_basis.drive_raw_aa0;
        self.state.drive_sign_aac = prediction.drive_basis.drive_sign_aac != 0;
        if let Some(schedule) = &payload.model.recovery.schedule {
            if let Some(selector_af4) = schedule.selector_af4 {
                self.state.soft_selector_af4 = selector_af4;
            }
        }
        if let Some(common_damping) = &payload.model.common_damping {
            self.state.soft_flag_188 = common_damping.special_gate;
        }
        if let Some(drive_target) = &prediction.drive_target {
            // AA4/AA8 belong to the same simulated transaction as the force
            // step.  Do not expose or retain them if downstream evaluation
            // raises, and keep state_before as the genuine pre-step snapshot.
            self.state.drive_aa4 = drive_target.field_aa4_after();
            self.state.drive_aa8 = drive_target.field_aa8_after();
            outcome.state_after = self.state.clone();
        }
        Ok(ControlStepOutcome {
            step: outcome,
            control_prediction: prediction,
            world_sweep: None,
            raw_control_pose_contract: None,
        })
    }

    /// `step_raw_control(keys, parameters, control_model, world=None)`
    /// (kartrider_vehicle_engine.py:5289-5368): advance one autonomous fixed
    /// substep from raw keys and owned state. The control frame is the
    /// run-invariant empirical-model payload (:5301-5357); with `world:
    /// Some` the step dispatches through `step_world` (M1 scope).
    pub fn step_raw_control(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        control_model: &ControlResponseInputModel,
        world: Option<&dyn World>,
    ) -> Result<ControlStepOutcome, String> {
        parameters.validate()?;
        let columns = matrix_columns(self.state.matrix);
        let axis_column = columns[parameters.forward_axis_column as usize];
        let forward_axis = [
            parameters.forward_axis_sign * axis_column[0],
            parameters.forward_axis_sign * axis_column[1],
            parameters.forward_axis_sign * axis_column[2],
        ];
        let numeric_mode = Some(self.numeric_mode.as_str().to_string());
        let mut source = BTreeMap::new();
        source.insert(
            "control_authority".to_string(),
            JsonValue::String("raw_keys".to_string()),
        );
        source.insert(
            "ground_input_authority".to_string(),
            JsonValue::String("run_invariant_empirical_control_model".to_string()),
        );
        source.insert(
            "scheduler_authority".to_string(),
            JsonValue::String("engine_fixed_substep".to_string()),
        );
        source.insert(
            "collision_authority".to_string(),
            JsonValue::String(
                if world.is_some() {
                    "autonomous_world_dispatch"
                } else {
                    "disabled"
                }
                .to_string(),
            ),
        );
        let frame = StepFrame {
            elapsed_ms: parameters.substep_ms as f64,
            authoritative_before: None,
            accumulator_input: None,
            kinematic_input: None,
            model: BranchAwareStepArgs {
                // Bound from parameters.substep_ms by `bind_state`.
                elapsed_ms: 0,
                boost: BoostArgs {
                    // state/remaining_ms are bound from engine state.
                    state: 0,
                    remaining_ms: 0,
                    correction_dir: parameters.boost.correction_dir,
                    direction_gain: parameters.boost.direction_gain,
                    selected_slot: parameters.boost.selected_slot,
                    attached_state: parameters.boost.attached_state,
                    e8c: parameters.boost.e8c,
                    efc: parameters.boost.efc,
                    flag_5bc: parameters.boost.flag_5bc,
                    raw_144: parameters.boost.raw_144,
                    obf_808: parameters.boost.obf_808,
                    obf_814: parameters.boost.obf_814,
                    obf_69c: parameters.boost.obf_69c,
                    obf_684: parameters.boost.obf_684,
                    obf_8ac: parameters.boost.obf_8ac,
                    obf_8e8: parameters.boost.obf_8e8,
                    obf_94c: parameters.boost.obf_94c,
                    inventory_use: None,
                },
                boost_gauge_producers: None,
                recovery: RecoveryArgs {
                    disabled_b9c: false,
                    schedule: Some(RecoveryScheduleArgs {
                        surface_mode: Some(parameters.surface_mode),
                        selector_af4: Some(self.state.soft_selector_af4),
                        field_af8: Some(0.0),
                        recovery_timer: Some(0.0),
                        other_recovery_gate: Some(false),
                    }),
                    low_projection: None,
                    drift_window: None,
                    force: None,
                    ground_force: None,
                },
                state18_feedback: None,
                pit_resource: None,
                bcharge_resource: None,
                accumulator: Some(AccumulatorArgs {
                    // The befores are bound from engine state.
                    accumulator_614_before: [0.0; 3],
                    accumulator_620_before: [0.0; 3],
                    matrix_9c: self.state.matrix,
                    numeric_mode: numeric_mode.clone(),
                }),
                substep_boundary: None,
                rail: None,
                contact: None,
                hard_contact: None,
                soft_contact: None,
                ground_correction: None,
                bs_zone: None,
                mz_magnet: None,
                jm_dj_trigger: None,
                post_bs_reset: None,
                common_damping: Some(CommonDampingArgs {
                    // velocity_c0/angular_velocity_cc are bound from engine state.
                    velocity_c0: None,
                    angular_velocity_cc: [0.0; 3],
                    damping_700: parameters.common_damping,
                    special_gate: self.state.soft_flag_188,
                    directional_70c: Some(parameters.directional_damping),
                    runtime_ba0: Some(parameters.damping_runtime_multiplier),
                    token_multiplier: None,
                    numeric_mode: numeric_mode.clone(),
                }),
                speed_threshold: None,
                linear_integrator: Some(LinearIntegratorArgs {
                    // velocity_c0_before is bound from engine state.
                    velocity_c0_before: None,
                    accumulator_614_override: None,
                    bac_force: [0.0; 3],
                    sub_660: parameters.mass,
                    post_bs_later_delta: None,
                    numeric_mode: numeric_mode.clone(),
                }),
                angular_integrator: Some(AngularIntegratorArgs {
                    angular_velocity_cc_before: self.state.angular_velocity,
                    accumulator_620_override: None,
                    matrix_9c4: parameters.inverse_inertia_matrix,
                    numeric_mode: numeric_mode.clone(),
                }),
                position_integrator: Some(PositionIntegratorArgs {
                    // Both befores are bound from engine state.
                    position_90_before: [0.0; 3],
                    velocity_c0_before: None,
                    pre_position_velocity_delta: None,
                    state_1d0: None,
                    latch_658: None,
                    obf_b44: None,
                    axis_ab8: None,
                    axis_a48: None,
                }),
                orientation_integrator: Some(OrientationIntegratorArgs {
                    // Both befores are bound from engine state.
                    matrix_9c_before: [0.0; 9],
                    angular_velocity_cc_before: None,
                    pre_clear_xy: Some(false),
                    stabilization_disabled_b70: Some(false),
                    rail_state_1cc: Some(0),
                    runtime_float32_exact: Some(self.numeric_mode.is_float32()),
                }),
                terminal_fall_reset: None,
                checkpoint_target_selector: None,
                terminal_checkpoint_respawn: None,
            },
            observed_after: None,
            source: Some(source.clone()),
        };
        let control = ControlPayload {
            keys: keys.to_vec(),
            input_mode: "empirical_response".to_string(),
            forward_axis: Some(forward_axis),
            sub_660: Some(parameters.mass),
            selector_a9c: 0,
            sign_aac: 0,
            sign_abc: 0,
            drive_target_parameters: None,
            drive_target_runtime: None,
            ground_runtime: None,
        };
        let mut outcome = match world {
            None => self.step_control(&frame, &control, Some(control_model))?,
            Some(world) => {
                // Python step_raw_control passes NO collision_selector_state
                // (kartrider_vehicle_engine.py:5361-5366): step_world's
                // selector defaults are the engine-state fields, which
                // step_control has just persisted from THIS transaction's
                // drive basis (:5270-5272). Overlaying the pre-step state
                // here would freeze the selector one transaction behind.
                let world_outcome = self.step_world(
                    &frame,
                    world,
                    parameters.collision_half_extents,
                    parameters.collision_center_up,
                    CollisionSelectorState::default(),
                    false,
                    Some((&control, Some(control_model))),
                )?;
                ControlStepOutcome {
                    step: world_outcome.step,
                    control_prediction: world_outcome.control_prediction.ok_or_else(|| {
                        "control-dispatched step_world missing control_prediction".to_string()
                    })?,
                    world_sweep: Some(world_outcome.world_sweep),
                    raw_control_pose_contract: None,
                }
            }
        };
        // `result["raw_control_pose_contract"] = deepcopy(frame["source"])`
        // (kartrider_vehicle_engine.py:5367).
        outcome.raw_control_pose_contract = Some(source);
        Ok(outcome)
    }

    /// The GND frame construction shared by the producer and integrator
    /// paths (kartrider_vehicle_engine.py:5387-5871). Takes `&self`: the
    /// selector epoch is computed locally and committed by the integrating
    /// path only.
    #[allow(clippy::too_many_lines)]
    fn build_ground_frame(
        &self,
        keys: &[String],
        parameters: &VehicleParameters,
        ground_plane: Option<&GroundPlaneSet>,
        advance_surface_selector: bool,
        support_active: Option<bool>,
        authored_trigger_model: Option<&JmDjArgs>,
    ) -> Result<GroundFrameParts, String> {
        parameters.validate()?;
        let active = validate_raw_control_keys(keys)?;
        let sampled_selection =
            select_surface_mode_from_contacts(ground_plane, parameters.surface_mode)?;
        let selector_period_ms = MILLISECONDS_PER_SECOND / SURFACE_SELECTOR_HZ;
        let mut surface_mode_latched = self.state.surface_mode_latched;
        let mut surface_selector_elapsed_ms = self.state.surface_selector_elapsed_ms;
        let mut selector_refreshed = surface_mode_latched.is_none();
        if advance_surface_selector && !selector_refreshed {
            surface_selector_elapsed_ms += parameters.substep_ms as f64;
            if surface_selector_elapsed_ms + 1e-9 >= selector_period_ms {
                surface_selector_elapsed_ms -= selector_period_ms;
                selector_refreshed = true;
            }
        }
        if selector_refreshed {
            surface_mode_latched = Some(sampled_selection.surface_mode);
        }
        let surface_mode_latched = surface_mode_latched.expect("refresh latch is always set");
        let surface_selection = SurfaceSelection {
            surface_mode: surface_mode_latched,
            authority: sampled_selection.authority,
            selection_rule: sampled_selection.selection_rule,
            road_surface_tokens: sampled_selection.road_surface_tokens,
            mode_counts: sampled_selection.mode_counts,
            dynamics_branches: sampled_selection.dynamics_branches,
            df_payloads: sampled_selection.df_payloads,
            selected_dynamics_branch: sampled_selection.selected_dynamics_branch,
            sampled_surface_mode: sampled_selection.surface_mode,
            selector_refreshed,
            selector_period_ms,
            selector_elapsed_ms: surface_selector_elapsed_ms,
        };
        let effective_surface_mode = surface_mode_latched;
        let projection =
            derive_autonomous_ground_projection(&self.state, parameters, self.numeric_mode)?;
        let columns = matrix_columns(self.state.matrix);
        let lateral_axis = columns[0];
        let axis_up = columns[2];
        let axis_forward = projection.axis_forward;
        let speed = length3(self.state.velocity, self.numeric_mode.is_float32());
        let unit_velocity = if speed != 0.0 {
            if self.numeric_mode.is_float32() {
                [
                    fdiv(self.state.velocity[0], speed),
                    fdiv(self.state.velocity[1], speed),
                    fdiv(self.state.velocity[2], speed),
                ]
            } else {
                [
                    self.state.velocity[0] / speed,
                    self.state.velocity[1] / speed,
                    self.state.velocity[2] / speed,
                ]
            }
        } else {
            [0.0, 0.0, 0.0]
        };
        let selected_slot = if active.contains("UP") { 1.0 } else { 0.0 };
        let opposite_slot = if active.contains("DOWN") { 1.0 } else { 0.0 };
        let signed_axis = if active.contains("LEFT") {
            1.0
        } else if active.contains("RIGHT") {
            -1.0
        } else {
            0.0
        };
        // The soft-contact optional source reads drive slot A84 (UP), while
        // raw_aa0 carries the independent LEFT/RIGHT steering axis.
        let soft_contact_drive_slot = selected_slot;
        let drive_target = evaluate_drive_target_runtime(
            signed_axis,
            parameters.drive_tune_718,
            projection.field_adc,
            parameters.drive_speed_scale_71c,
            parameters.drive_attenuation_bb8,
            // B2AB55's overshoot clamp reads the active longitudinal drive
            // slot (A84/UP). A9C remains a separate dispatcher selector.
            selected_slot,
            self.state.drive_aa8,
        )?;
        let ground_forward_term = drive_target.state.field_aa4_after;

        let mut contact_payload: Option<ContactArgs> = None;
        let mut next_contact_compression: Option<[f64; 4]> = None;
        let mut contact_probe_debug: Vec<ContactProbeDebug> = Vec::new();
        let mut contact_average_normal = axis_up;
        if let Some(ground_plane) = ground_plane {
            let planes = ground_plane.probe_planes()?;
            let forward_axis = [-columns[1][0], -columns[1][1], -columns[1][2]];
            let signs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
            let mut compression_slots = [0.0; 4];
            let mut contact_normals: [[f64; 3]; 4] = [[0.0; 3]; 4];
            for (slot, (lateral_sign, forward_sign)) in signs.iter().enumerate() {
                let slot_plane = planes[slot];
                let normal = slot_plane.normal;
                let point = slot_plane.point;
                contact_normals[slot] = normal;
                // B28D30 stores each scalar multiply and both vector adds.
                let origin = wheel_anchor(
                    self.state.position,
                    lateral_axis,
                    forward_axis,
                    axis_up,
                    parameters,
                    *lateral_sign,
                    *forward_sign,
                    self.numeric_mode,
                );
                let ray = if self.numeric_mode.is_float32() {
                    [
                        fsub(0.0, axis_up[0]),
                        fsub(0.0, axis_up[1]),
                        fsub(0.0, axis_up[2]),
                    ]
                } else {
                    [-axis_up[0], -axis_up[1], -axis_up[2]]
                };
                let denominator = if self.numeric_mode.is_float32() {
                    dot3_f32(ray, normal)
                } else {
                    dot3_sum(ray, normal)
                };
                let mut ray_t = None;
                let mut hit_point = origin;
                let mut hit_projection = None;
                let mut position_projection = None;
                let denominator_epsilon = if self.numeric_mode.is_float32() {
                    f32q(1e-9)
                } else {
                    1e-9
                };
                let mut compression;
                if denominator.abs() <= denominator_epsilon {
                    compression = 0.0;
                } else {
                    if self.numeric_mode.is_float32() {
                        let delta = sub3_f32(point, origin);
                        if let Some(triangle) = &slot_plane.triangle {
                            // CollisionCollection_SweepMainFaces (A5B840)
                            // uses the float32 Moller-Trumbore sequence, then
                            // constructs hit = origin + ray*t with the common
                            // vector helpers. A plane quotient is
                            // geometrically equivalent but not bit-equivalent.
                            let [v0, v1, v2] = triangle;
                            let v0 = [f32q(v0[0]), f32q(v0[1]), f32q(v0[2])];
                            let v1 = [f32q(v1[0]), f32q(v1[1]), f32q(v1[2])];
                            let v2 = [f32q(v2[0]), f32q(v2[1]), f32q(v2[2])];
                            let edge1 = sub3_f32(v1, v0);
                            let edge2 = sub3_f32(v2, v0);
                            let pvec = cross3_f32(ray, edge2);
                            let inv_det = fdiv(1.0, dot3_f32(edge1, pvec));
                            let tvec = sub3_f32(origin, v0);
                            let qvec = cross3_f32(tvec, edge1);
                            ray_t = Some(fmul(dot3_f32(edge2, qvec), inv_det));
                        } else {
                            ray_t = Some(fdiv(dot3_f32(delta, normal), denominator));
                        }
                        let ray_t = ray_t.expect("ray_t set above");
                        hit_point = [
                            fadd(origin[0], fmul(ray[0], ray_t)),
                            fadd(origin[1], fmul(ray[1], ray_t)),
                            fadd(origin[2], fmul(ray[2], ray_t)),
                        ];
                        let hit_projection_value = dot3_f32(hit_point, axis_up);
                        let position_projection_value = dot3_f32(self.state.position, axis_up);
                        hit_projection = Some(hit_projection_value);
                        position_projection = Some(position_projection_value);
                        compression = fsub(
                            hit_projection_value,
                            fsub(
                                position_projection_value,
                                parameters.contact_rest_compression,
                            ),
                        );
                    } else {
                        let delta = [
                            point[0] - origin[0],
                            point[1] - origin[1],
                            point[2] - origin[2],
                        ];
                        let ray_t_value = dot3_sum(delta, normal) / denominator;
                        ray_t = Some(ray_t_value);
                        hit_point = [
                            origin[0] + ray[0] * ray_t_value,
                            origin[1] + ray[1] * ray_t_value,
                            origin[2] + ray[2] * ray_t_value,
                        ];
                        compression = dot3_sum(hit_point, axis_up)
                            - (dot3_sum(self.state.position, axis_up)
                                - parameters.contact_rest_compression);
                    }
                    // TrackWorld's ordinary GND query retains a nearest-face
                    // fallback when the finite -up segment misses every
                    // triangle. Re-evaluating that face as an infinite plane
                    // can put the intersection behind the wheel anchor and
                    // manufacture full suspension compression. Keep this
                    // guard scoped to ground ownership: the accepted RAIL
                    // endpoint handoff uses the legacy signed-plane result.
                    let ordinary_ground_owner = matches!(
                        self.state.topology.domain,
                        MotionDomain::Ground | MotionDomain::GroundSlip
                    );
                    let rail_endpoint_handoff =
                        self.state.topology.previous_domain == Some(MotionDomain::Rail);
                    compression = if ordinary_ground_owner
                        && !rail_endpoint_handoff
                        && ray_t.is_some_and(|value| value < 0.0)
                    {
                        0.0
                    } else if self.numeric_mode.is_float32() {
                        min2_f64(
                            fmul(parameters.contact_rest_compression, 2.0),
                            max2_f64(f32q(0.0), f32q(compression)),
                        )
                    } else {
                        min2_f64(
                            parameters.contact_rest_compression * 2.0,
                            max2_f64(0.0, compression),
                        )
                    };
                }
                if self.numeric_mode.is_float32() {
                    compression = f32q(compression);
                }
                if parameters.contact_compression_quantum > 0.0 {
                    if self.numeric_mode.is_float32() {
                        let quantum = f32q(parameters.contact_compression_quantum);
                        compression = fmul(round_py(fdiv(compression, quantum)), quantum);
                    } else {
                        compression =
                            round_py(compression / parameters.contact_compression_quantum)
                                * parameters.contact_compression_quantum;
                    }
                }
                compression_slots[slot] = compression;
                contact_probe_debug.push(ContactProbeDebug {
                    origin,
                    ray,
                    ray_t,
                    hit_point,
                    hit_projection,
                    position_projection,
                    compression,
                    plane_normal: normal,
                    plane_point: point,
                    plane_node: slot_plane.node.clone(),
                    plane_tri_index: slot_plane.tri_index,
                    plane_surface: slot_plane.road_surface.clone().unwrap_or_default(),
                });
            }
            let active_normals: Vec<[f64; 3]> = (0..4)
                .filter(|index| compression_slots[*index] > 0.0)
                .map(|index| contact_normals[index])
                .collect();
            if !active_normals.is_empty() {
                if self.numeric_mode.is_float32() {
                    let mut summed = [f32q(0.0), f32q(0.0), f32q(0.0)];
                    for normal in &active_normals {
                        for (index, sum) in summed.iter_mut().enumerate() {
                            *sum = f32q(*sum + f32q(normal[index]));
                        }
                    }
                    let count = f32q(active_normals.len() as f64);
                    contact_average_normal = [
                        f32q(summed[0] / count),
                        f32q(summed[1] / count),
                        f32q(summed[2] / count),
                    ];
                } else {
                    for (index, component) in contact_average_normal.iter_mut().enumerate() {
                        let values: Vec<f64> =
                            active_normals.iter().map(|normal| normal[index]).collect();
                        *component = sum_f64(&values) / active_normals.len() as f64;
                    }
                }
            }
            let rates = if self.numeric_mode.is_float32() {
                [
                    fsub(compression_slots[0], self.state.contact_compression[0]),
                    fsub(compression_slots[1], self.state.contact_compression[1]),
                    fsub(compression_slots[2], self.state.contact_compression[2]),
                    fsub(compression_slots[3], self.state.contact_compression[3]),
                ]
            } else {
                [
                    compression_slots[0] - self.state.contact_compression[0],
                    compression_slots[1] - self.state.contact_compression[1],
                    compression_slots[2] - self.state.contact_compression[2],
                    compression_slots[3] - self.state.contact_compression[3],
                ]
            };
            next_contact_compression = Some(compression_slots);
            contact_payload = Some(ContactArgs {
                elapsed_seconds: if self.numeric_mode.is_float32() {
                    fdiv(parameters.substep_ms as f64, 1000.0)
                } else {
                    parameters.substep_ms as f64 / 1000.0
                },
                contact_active: [
                    compression_slots[0] > 0.0,
                    compression_slots[1] > 0.0,
                    compression_slots[2] > 0.0,
                    compression_slots[3] > 0.0,
                ],
                compression_a54: compression_slots,
                compression_rate_a64: rates,
                contact_vectors_a14: contact_normals,
                contact_normal_ac4: axis_up,
                spring_7c0: parameters.contact_spring,
                rebound_7c4: parameters.contact_rebound,
                bump_7c8: parameters.contact_bump,
                lever_x_9f4: [1.0, -1.0, 1.0, -1.0],
                lever_y_9f8: [1.0, 1.0, -1.0, -1.0],
                obf_66c: parameters.collision_half_extents[0],
                obf_678: parameters.collision_half_extents[1],
                field_604: [0.0, 0.0, parameters.contact_gravity_per_mass],
                obf_660: parameters.mass,
            });
        } else if support_active == Some(false) {
            // AIR still calls sub_B29E70. Its four suspension slots are
            // inactive, but the field_604*mass common producer after the loop
            // remains unconditional and supplies gravity.
            next_contact_compression = Some([0.0; 4]);
            contact_payload = Some(ContactArgs {
                elapsed_seconds: if self.numeric_mode.is_float32() {
                    fdiv(parameters.substep_ms as f64, 1000.0)
                } else {
                    parameters.substep_ms as f64 / 1000.0
                },
                contact_active: [false; 4],
                compression_a54: [0.0; 4],
                compression_rate_a64: if self.numeric_mode.is_float32() {
                    [
                        fsub(0.0, self.state.contact_compression[0]),
                        fsub(0.0, self.state.contact_compression[1]),
                        fsub(0.0, self.state.contact_compression[2]),
                        fsub(0.0, self.state.contact_compression[3]),
                    ]
                } else {
                    [
                        -self.state.contact_compression[0],
                        -self.state.contact_compression[1],
                        -self.state.contact_compression[2],
                        -self.state.contact_compression[3],
                    ]
                },
                contact_vectors_a14: [[0.0; 3]; 4],
                contact_normal_ac4: axis_up,
                spring_7c0: parameters.contact_spring,
                rebound_7c4: parameters.contact_rebound,
                bump_7c8: parameters.contact_bump,
                lever_x_9f4: [1.0, -1.0, 1.0, -1.0],
                lever_y_9f8: [1.0, 1.0, -1.0, -1.0],
                obf_66c: parameters.collision_half_extents[0],
                obf_678: parameters.collision_half_extents[1],
                field_604: [0.0, 0.0, parameters.contact_gravity_per_mass],
                obf_660: parameters.mass,
            });
        }
        let primary_correction_dir = if self.numeric_mode.is_float32() {
            [
                f32q(
                    f32q(f32q(lateral_axis[1]) * f32q(contact_average_normal[2]))
                        - f32q(f32q(lateral_axis[2]) * f32q(contact_average_normal[1])),
                ),
                f32q(
                    f32q(f32q(lateral_axis[2]) * f32q(contact_average_normal[0]))
                        - f32q(f32q(lateral_axis[0]) * f32q(contact_average_normal[2])),
                ),
                f32q(
                    f32q(f32q(lateral_axis[0]) * f32q(contact_average_normal[1]))
                        - f32q(f32q(lateral_axis[1]) * f32q(contact_average_normal[0])),
                ),
            ]
        } else {
            [
                lateral_axis[1] * contact_average_normal[2]
                    - lateral_axis[2] * contact_average_normal[1],
                lateral_axis[2] * contact_average_normal[0]
                    - lateral_axis[0] * contact_average_normal[2],
                lateral_axis[0] * contact_average_normal[1]
                    - lateral_axis[1] * contact_average_normal[0],
            ]
        };

        let numeric_mode = Some(self.numeric_mode.as_str().to_string());
        let support_inactive = support_active == Some(false);
        let args = BranchAwareStepArgs {
            // Bound from parameters.substep_ms by `bind_state`.
            elapsed_ms: 0,
            boost: BoostArgs {
                // state/remaining_ms are bound from engine state.
                state: 0,
                remaining_ms: 0,
                correction_dir: parameters.boost.correction_dir,
                direction_gain: parameters.boost.direction_gain,
                selected_slot: parameters.boost.selected_slot,
                attached_state: parameters.boost.attached_state,
                e8c: parameters.boost.e8c,
                efc: parameters.boost.efc,
                flag_5bc: parameters.boost.flag_5bc,
                raw_144: parameters.boost.raw_144,
                obf_808: parameters.boost.obf_808,
                obf_814: parameters.boost.obf_814,
                obf_69c: parameters.boost.obf_69c,
                obf_684: parameters.boost.obf_684,
                obf_8ac: parameters.boost.obf_8ac,
                obf_8e8: parameters.boost.obf_8e8,
                obf_94c: parameters.boost.obf_94c,
                inventory_use: None,
            },
            boost_gauge_producers: Some(CurrentFamilyGaugeProducersArgs {
                airborne: support_inactive,
                contact_188: self.state.soft_flag_188,
                selector_e8c: parameters.boost.e8c,
                state_1a8: false,
                pending_b1c: self.state.drift_b1c_charge_settlement,
                window_fba: false,
                window_fbb: false,
                plain_adc: projection.field_adc,
                field_ae0: projection.field_ae0,
                speed_144: speed,
                threshold_54c: parameters.boost_gauge_speed_threshold_54c,
                tune_6b4: parameters.boost_gauge_primary_rate_6b4,
                tune_100c: parameters.boost_gauge_primary_override_100c,
                tune_988: parameters.boost_gauge_window_add_988,
                tune_994: parameters.boost_gauge_window_factor_994,
                raw_760: parameters.boost_gauge_raw_760,
            }),
            recovery: RecoveryArgs {
                disabled_b9c: self.state.banana_effect_remaining_ms > 0,
                schedule: Some(RecoveryScheduleArgs {
                    surface_mode: Some(effective_surface_mode),
                    selector_af4: Some(self.state.soft_selector_af4),
                    field_af8: Some(self.state.field_af8),
                    recovery_timer: Some(self.state.recovery_timer),
                    other_recovery_gate: Some(false),
                }),
                low_projection: if support_inactive {
                    // No-support AIR is a distinct topology owner. Keeping
                    // the normal/slip drive body or B2A100 ground correction
                    // scheduled here invents large horizontal acceleration
                    // after a JM launch.
                    None
                } else {
                    Some(LowProjectionArgs {
                        projection_len: projection.projection_len,
                        ground_forward_term,
                        drive_direction: projection.drive_direction,
                        normalized_velocity_raw: projection.field_ae0,
                        correction_raw: self.state.angular_velocity[2],
                        mass_scale: parameters.ground_force_multiplier,
                        base_longitudinal_gain: parameters.ground_longitudinal_gain,
                        base_lateral_gain: parameters.ground_lateral_gain,
                        axis_long: Some(parameters.ground_axis_long),
                        axis_lat: Some(parameters.ground_axis_lat),
                        state_5c8: None,
                        b1c_charge_settlement: Some(self.state.drift_b1c_charge_settlement),
                        b10: Some(self.state.boost_gauge_committed_b10),
                        b20: Some(self.state.boost_gauge_progress_b20),
                        b2c: Some(self.state.boost_gauge_pending_b2c),
                        b38: Some(self.state.boost_gauge_history_b38),
                        cap_760: Some(parameters.boost_gauge_raw_760),
                        numeric_mode: numeric_mode.clone(),
                    })
                },
                drift_window: if support_inactive {
                    None
                } else {
                    Some(DriftWindowArgs {
                        selector_b08: self.state.selector_b08,
                        timer_b0c: self.state.drift_timer_b0c,
                        configured_duration_734: parameters.drift_window_duration_734,
                        mass_scale: parameters.mass,
                        base_longitudinal_gain: parameters.ground_longitudinal_gain,
                        signed_axis: match self.state.selector_ff8 {
                            1 => 1.0,
                            2 => -1.0,
                            _ => signed_axis,
                        },
                        scaled_tune_660: drive_target.scaled_tune_660,
                        drift_gain_728: parameters.drift_gain_728,
                        tune_758: parameters.drift_torque_y_gain_758,
                        axis_lat: parameters.ground_axis_lat,
                        drive_raw_aa0: signed_axis,
                        debug_force_scale: None,
                    })
                },
                force: if support_inactive {
                    None
                } else {
                    Some(Surface1RecoveryForceArgs {
                        selector_af4: Some(self.state.soft_selector_af4),
                        speed_magnitude: speed,
                        speed_profile_denominator: parameters.drift_speed_profile_denominator,
                        blend_ab0: self.state.drift_blend_ab0,
                        blend_ab4: self.state.drift_blend_ab4,
                        active_ab0_rate: parameters.drift_active_ab0_rate,
                        active_ab4_rate: parameters.drift_active_ab4_rate,
                        timer_ab0_rate: parameters.drift_timer_ab0_rate,
                        contact_basis_product: ground_forward_term,
                        ground_forward_term,
                        drive_direction: projection.drive_direction,
                        normalized_velocity: projection.ground_normalized_lateral_velocity,
                        longitudinal_correction: projection.longitudinal_angular_correction,
                        lateral_correction: projection.lateral_angular_correction,
                        mass_scale: 1.0,
                        base_longitudinal_gain: parameters.ground_longitudinal_gain,
                        base_lateral_gain: parameters.ground_lateral_gain,
                        ground_force_multiplier: parameters.ground_force_multiplier,
                        tune_758: parameters.drift_torque_y_gain_758,
                        axis_long: parameters.ground_axis_long,
                        axis_lat: parameters.ground_axis_lat,
                        active_surface: None,
                        timer_surface: None,
                        field_af0: Some(speed),
                        torque_y_scale: Some(1.0),
                        surface_mode: Some(effective_surface_mode),
                        non_surface_profile: None,
                    })
                },
                ground_force: if support_inactive {
                    None
                } else {
                    Some(GroundDriveForceArgs {
                        surface_mode: Some(effective_surface_mode),
                        speed_magnitude: speed,
                        ground_forward_term,
                        contact_basis_product: if effective_surface_mode == 2 {
                            signed_axis * parameters.generic_slip_steering_basis
                        } else {
                            0.0
                        },
                        drive_direction: projection.drive_direction,
                        normalized_velocity: projection.ground_normalized_lateral_velocity,
                        longitudinal_correction: projection.longitudinal_angular_correction,
                        lateral_correction: projection.lateral_angular_correction,
                        mass_scale: 1.0,
                        base_longitudinal_gain: parameters.ground_longitudinal_gain,
                        base_lateral_gain: parameters.ground_lateral_gain,
                        ground_force_multiplier: parameters.ground_force_multiplier,
                        settlement_multiplier: parameters.ground_settlement_multiplier,
                        tune_75c: parameters.ground_torque_y_gain,
                        axis_long: parameters.ground_axis_long,
                        axis_lat: parameters.ground_axis_lat,
                        surface_profile: if effective_surface_mode == 2 {
                            Some(GroundSurfaceProfileArgs {
                                speed_denominator: 180.0,
                                speed_quadratic_gain: 2.0,
                                speed_bias: 0.5,
                                force_scale: parameters.generic_slip_force_scale,
                            })
                        } else {
                            None
                        },
                        torque_y_scale: None,
                        torque_y_gain: None,
                        state_5c8: None,
                        b1c_charge_settlement: Some(self.state.drift_b1c_charge_settlement),
                        b10: Some(self.state.boost_gauge_committed_b10),
                        b20: Some(self.state.boost_gauge_progress_b20),
                        b2c: Some(self.state.boost_gauge_pending_b2c),
                        b38: Some(self.state.boost_gauge_history_b38),
                        cap_760: Some(parameters.boost_gauge_raw_760),
                        numeric_mode: numeric_mode.clone(),
                    })
                },
            },
            state18_feedback: None,
            pit_resource: None,
            bcharge_resource: None,
            accumulator: Some(AccumulatorArgs {
                // The befores are bound from engine state.
                accumulator_614_before: [0.0; 3],
                accumulator_620_before: [0.0; 3],
                matrix_9c: self.state.matrix,
                numeric_mode: numeric_mode.clone(),
            }),
            substep_boundary: Some(SubstepBoundaryArgs {
                pending_force_b74: [0.0; 3],
                pending_torque_b80: [0.0; 3],
                pending_b90: [0.0; 3],
            }),
            rail: None,
            contact: contact_payload,
            hard_contact: None,
            soft_contact: None,
            ground_correction: Some(GroundCorrectionArgs {
                // velocity_c0 is bound from engine state.
                velocity_c0: [0.0; 3],
                disabled_b9c: support_inactive || self.state.banana_effect_remaining_ms > 0,
                selected_slot: selected_slot as i64,
                opposite_slot: opposite_slot as i64,
                // RAIL state 6 keeps the B2A100 fallback path active while
                // its carried contact record remains live.
                gate_1c0: self.state.topology.domain == MotionDomain::Rail
                    && self.state.topology.rail_state == 6,
                gate_6f3b00: false,
                gate_6f3ae0: false,
                field_adc: projection.field_adc,
                timer_a80_before: self.state.ground_timer_a80,
                field_ae0: projection.field_ae0,
                b8c: 0.0,
                lateral_dot: if self.numeric_mode.is_float32() {
                    dot3_f32(unit_velocity, axis_forward)
                } else {
                    dot3_sum(unit_velocity, axis_forward)
                },
                correction_dir: primary_correction_dir,
                direction_gain: parameters.primary_ground_direction_gain,
                state_mult: 1.0,
                flag_af4: self.state.soft_selector_af4,
                flag_afc: false,
                sub_ad0: axis_forward,
                // +0xAF0 is the current 3D speed magnitude. On negative ADC
                // kind31/32 it scales the additional AD0 response.
                sub_af0: speed,
                field_9f0: parameters.negative_adc_force_scale,
                sub_6f4: parameters.fallback_ground_direction_gain,
                sub_bc4: 1.0,
                drive_slot1_a90: opposite_slot,
                axis_ac4: axis_up,
                sub_720: parameters.lateral_high_dot_scale,
                sub_724: parameters.lateral_low_dot_scale,
                numeric_mode: numeric_mode.clone(),
            }),
            bs_zone: None,
            mz_magnet: None,
            jm_dj_trigger: authored_trigger_model.cloned(),
            post_bs_reset: None,
            common_damping: Some(CommonDampingArgs {
                // velocity_c0/angular_velocity_cc are bound from engine state.
                velocity_c0: None,
                angular_velocity_cc: [0.0; 3],
                damping_700: parameters.common_damping,
                special_gate: self.state.soft_flag_188,
                directional_70c: Some(parameters.directional_damping),
                runtime_ba0: Some(parameters.damping_runtime_multiplier),
                token_multiplier: None,
                numeric_mode: numeric_mode.clone(),
            }),
            speed_threshold: None,
            linear_integrator: Some(LinearIntegratorArgs {
                // velocity_c0_before is bound from engine state.
                velocity_c0_before: None,
                // +0x614/+0x620 consumed by kind40/B2D940 are the accumulator
                // snapshot produced by the preceding epoch. The boundary
                // clears that active snapshot before this epoch's
                // contact/ground producers build the value persisted for the
                // next epoch.
                accumulator_614_override: Some(self.state.force_accumulator),
                bac_force: [0.0; 3],
                sub_660: parameters.mass,
                post_bs_later_delta: None,
                numeric_mode: numeric_mode.clone(),
            }),
            angular_integrator: Some(AngularIntegratorArgs {
                // angular_velocity_cc_before is bound from engine state.
                angular_velocity_cc_before: [0.0; 3],
                accumulator_620_override: Some(self.state.torque_accumulator),
                matrix_9c4: parameters.inverse_inertia_matrix,
                numeric_mode: numeric_mode.clone(),
            }),
            position_integrator: Some(PositionIntegratorArgs {
                // Both befores are bound from engine state.
                position_90_before: [0.0; 3],
                velocity_c0_before: None,
                pre_position_velocity_delta: None,
                state_1d0: None,
                latch_658: None,
                obf_b44: None,
                axis_ab8: None,
                axis_a48: None,
            }),
            orientation_integrator: Some(OrientationIntegratorArgs {
                // Both befores are bound from engine state.
                matrix_9c_before: [0.0; 9],
                angular_velocity_cc_before: None,
                pre_clear_xy: Some(false),
                stabilization_disabled_b70: Some(false),
                rail_state_1cc: Some(0),
                runtime_float32_exact: Some(self.numeric_mode.is_float32()),
            }),
            terminal_fall_reset: None,
            checkpoint_target_selector: None,
            terminal_checkpoint_respawn: None,
        };

        let mut source = BTreeMap::new();
        source.insert(
            "control_authority".to_string(),
            JsonValue::String("raw_keys".to_string()),
        );
        source.insert(
            "ground_input_authority".to_string(),
            JsonValue::String(format!("exact_ground_body_{}", surface_selection.authority)),
        );
        source.insert(
            "scheduler_authority".to_string(),
            JsonValue::String("engine_fixed_substep".to_string()),
        );
        // collision_authority is inserted by the caller, which knows whether
        // a world was supplied.
        source.insert(
            "scope".to_string(),
            JsonValue::String(
                "normal/nonzero ground force body plus normal-state correction and \
                 M9C4 angular integration; autonomous surface selection open"
                    .to_string(),
            ),
        );

        Ok(GroundFrameParts {
            args,
            source,
            surface_selection,
            surface_mode_latched,
            surface_selector_elapsed_ms,
            drive_target,
            contact_probe_debug,
            next_contact_compression,
            soft_contact_drive_slot,
            signed_axis,
            projection,
            speed,
        })
    }

    /// `produce_ground_accumulators` (kartrider_vehicle_engine.py:6008-6028):
    /// evaluate the ground producer transaction without state integration.
    /// Takes `&self` — the zero-state-write contract is enforced at the type
    /// level. There is deliberately no world parameter: Python raises
    /// `ValueError("producer-only ground evaluation cannot dispatch world
    /// collisions")` for that combination (:5873-5877).
    pub fn produce_ground_accumulators(
        &self,
        keys: &[String],
        parameters: &VehicleParameters,
        ground_plane: Option<&GroundPlaneSet>,
        support_active: Option<bool>,
        advance_surface_selector: bool,
    ) -> Result<GroundProducerOutput, String> {
        let mut parts = self.build_ground_frame(
            keys,
            parameters,
            ground_plane,
            advance_surface_selector,
            support_active,
            None,
        )?;
        // Producer contract: the integration stages are not scheduled.
        parts.args.linear_integrator = None;
        parts.args.angular_integrator = None;
        parts.args.position_integrator = None;
        parts.args.orientation_integrator = None;
        parts.args.boost_gauge_producers = None;
        let mut args = parts.args;
        self.bind_state(&mut args, parameters.substep_ms);
        let evaluator = evaluate_branch_aware_step(&args)?;
        let mut compression_after = parts
            .next_contact_compression
            .unwrap_or(self.state.contact_compression);
        if self.numeric_mode.is_float32() {
            f32_vector(&mut compression_after);
        }
        Ok(GroundProducerOutput {
            evaluator,
            surface_selection: parts.surface_selection,
            contact_compression_after: compression_after,
            contact_probe_debug: parts.contact_probe_debug,
            drive_target: parts.drive_target,
            producer_contract: ProducerContract::default(),
        })
    }

    /// `step_raw_ground_correction` (kartrider_vehicle_engine.py:5370-6006):
    /// advance the raw-key ground producer with an autonomous
    /// contact-selected surface mode.
    ///
    /// With `world: Some` the frame dispatches through [`Self::step_world`]
    /// (M1 scope: no-hit/separating sweeps; approaching-candidate dispatch
    /// is a structured M2 error), after the selector latch commit, matching
    /// the Python ordering (:5914-5918). `options.rail_state1_ground_writer`
    /// is accepted for API parity and currently unused.
    #[allow(clippy::too_many_lines)]
    pub fn step_raw_ground_correction(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_plane: Option<&GroundPlaneSet>,
        options: &GroundStepOptions<'_>,
    ) -> Result<GroundStepOutcome, String> {
        let mut parts = self.build_ground_frame(
            keys,
            parameters,
            ground_plane,
            options.advance_surface_selector,
            options.support_active,
            options.authored_trigger_model.as_ref(),
        )?;
        let drift_down = validate_raw_control_keys(keys)?.contains("CTRL");
        let drift_steer = if parts.signed_axis > 0.0 {
            1
        } else if parts.signed_axis < 0.0 {
            -1
        } else {
            0
        };
        // `drive_raw_aa0` is the already-persisted previous input axis at
        // this pre-producer phase, so it is also the cross-substep steering
        // event latch; no new C-ABI state field is required.
        let previous_steer = if self.state.drive_raw_aa0 > 0.0 {
            1
        } else if self.state.drive_raw_aa0 < 0.0 {
            -1
        } else {
            0
        };
        let was_active = self.state.drift_key_latched && previous_steer != 0;
        let is_active = drift_down && drift_steer != 0;
        let direction_changed = was_active && is_active && drift_steer != previous_steer;
        let mut dispatches = Vec::with_capacity(2);
        if was_active && (!is_active || direction_changed) {
            // The runtime releases the old steering case before pressing the
            // opposite case, so the intermediate disable sees raw AA0 == 0.
            dispatches.push((
                false,
                if direction_changed {
                    0.0
                } else {
                    parts.signed_axis
                },
            ));
        }
        if is_active && (!was_active || direction_changed) {
            dispatches.push((true, parts.signed_axis));
        }
        let mut drift_selector_transitions = Vec::with_capacity(dispatches.len());
        for (enable, selector_drive_raw) in dispatches {
            let transition = apply_selector_transition(
                enable,
                i64::from(parameters.boost.e8c),
                parts.surface_mode_latched,
                self.state.soft_selector_af4,
                self.state.selector_b08,
                self.state.selector_b5c,
                self.state.selector_f08,
                self.state.selector_fec,
                self.state.selector_ff8,
                self.state.field_af8,
                parts.projection.field_adc,
                self.state.soft_flag_188,
                selector_drive_raw,
                parts.speed,
                self.state.recovery_timer,
            );
            if transition.applied {
                let after = transition.after;
                self.state.soft_selector_af4 = after.selector_af4;
                self.state.selector_b08 = after.selector_b08;
                self.state.selector_b5c = after.selector_b5c;
                self.state.selector_f08 = after.selector_f08;
                self.state.selector_fec = after.selector_fec;
                self.state.selector_ff8 = after.selector_ff8;
                self.state.recovery_timer = after.recovery_timer;
            }
            drift_selector_transitions.push(transition);
        }
        self.state.drift_key_latched = drift_down;
        let drift_selector_transition = drift_selector_transitions.last().cloned();
        if !drift_selector_transitions.is_empty() {
            // Rebuild after the complete input event transaction so the
            // same substep consumes the final selector snapshot.
            parts = self.build_ground_frame(
                keys,
                parameters,
                ground_plane,
                options.advance_surface_selector,
                options.support_active,
                options.authored_trigger_model.as_ref(),
            )?;
        }
        // Collision topology commits inside step_world and must observe the
        // same selector epoch as this producer frame. The latch commits on
        // the integrating path only by the producer/integrator type contract.
        self.state.surface_mode_latched = Some(parts.surface_mode_latched);
        self.state.surface_selector_elapsed_ms = parts.surface_selector_elapsed_ms;
        let mut source = parts.source.clone();
        // kartrider_vehicle_engine.py:5849.
        source.insert(
            "collision_authority".to_string(),
            JsonValue::String(
                if world.is_some() {
                    "autonomous_world_dispatch"
                } else {
                    "disabled"
                }
                .to_string(),
            ),
        );
        let frame = StepFrame {
            elapsed_ms: parameters.substep_ms as f64,
            authoritative_before: None,
            accumulator_input: None,
            kinematic_input: None,
            model: parts.args,
            observed_after: None,
            source: Some(source.clone()),
        };
        // `result = self.step(frame) if world is None else self.step_world(...)`
        // (kartrider_vehicle_engine.py:5918-5933). Both paths return the
        // `step` outcome; step_world additionally sweeps the vehicle volume.
        let (mut outcome, world_sweep) = match world {
            None => (self.step(&frame)?, None),
            Some(world) => {
                let world_outcome = self.step_world(
                    &frame,
                    world,
                    parameters.collision_half_extents,
                    parameters.collision_center_up,
                    CollisionSelectorState {
                        drive_selected_slot: Some(parts.soft_contact_drive_slot),
                        drive_raw_aa0: Some(parts.signed_axis),
                        ..CollisionSelectorState::default()
                    },
                    options.rail_state1_ground_writer,
                    None,
                )?;
                (world_outcome.step, Some(world_outcome.world_sweep))
            }
        };
        if options.authored_trigger_model.is_some() {
            self.state.special_trigger_cooldown_ms = AUTHORED_TRIGGER_COOLDOWN_MS;
        }
        // Producers persisted for the next kind40 observe this epoch's final
        // pose/C0/CC, including a wall writer. Evaluate the producer frame
        // directly from that finalized state; no shadow engine or speculative
        // state integration participates in the transaction.
        let mut pending_refresh = None;
        if options.refresh_pending {
            let pending_parameters = options
                .pending_parameters
                .clone()
                .unwrap_or_else(|| parameters.clone());
            pending_parameters.validate()?;
            let pending_ground_plane = match &options.pending_ground_plane_provider {
                Some(provider) => provider(&self.state, &pending_parameters),
                None => ground_plane.cloned(),
            };
            let producer = self.produce_ground_accumulators(
                keys,
                &pending_parameters,
                pending_ground_plane.as_ref(),
                options.support_active,
                false,
            )?;
            self.state.force_accumulator = producer
                .evaluator
                .accumulator_614_after_damping
                .ok_or_else(|| {
                    "producer evaluator missing accumulator_614_after_damping".to_string()
                })?;
            self.state.torque_accumulator = producer
                .evaluator
                .accumulator_620_after_damping
                .ok_or_else(|| {
                    "producer evaluator missing accumulator_620_after_damping".to_string()
                })?;
            // UpdateCurrentContactRecord runs after this epoch's pose writer.
            // The producer-only transaction therefore owns the compression
            // persisted into the next epoch along with +0x614/+0x620.
            self.state.contact_compression = producer.contact_compression_after;
            if self.numeric_mode.is_float32() {
                f32_vector(&mut self.state.force_accumulator);
                f32_vector(&mut self.state.torque_accumulator);
                f32_vector(&mut self.state.contact_compression);
            }
            pending_refresh = Some(PendingProducerRefresh {
                authority: "pure_producer_frame_from_engine_owned_post_pose_state",
                force_accumulator: self.state.force_accumulator,
                torque_accumulator: self.state.torque_accumulator,
                contact_compression: self.state.contact_compression,
                producer,
            });
        }
        self.state.drive_raw_aa0 = parts.signed_axis;
        // Persist the current UP/A84 slot consumed by later soft contacts.
        self.state.drive_selected_slot = parts.soft_contact_drive_slot;
        self.state.drive_aa4 = parts.drive_target.state.field_aa4_after;
        self.state.drive_aa8 = parts.drive_target.state.field_aa8_after;
        if let Some(next_compression) = parts.next_contact_compression {
            if !options.refresh_pending {
                // Python quantizes unconditionally here
                // (kartrider_vehicle_engine.py:6000-6001).
                let mut compression = next_compression;
                f32_vector(&mut compression);
                self.state.contact_compression = compression;
            }
        }
        outcome.state_after = self.state.clone();
        Ok(GroundStepOutcome {
            step: outcome,
            surface_selection: parts.surface_selection,
            pending_producer_refresh: pending_refresh,
            drive_target: parts.drive_target,
            drift_selector_transition,
            drift_selector_transitions,
            contact_probe_debug: parts.contact_probe_debug,
            raw_control_pose_contract: source,
            world_sweep,
        })
    }
}

// ---------------------------------------------------------------------------
// Frame loop + topology dispatch (step_raw_frame / step_raw_motion /
// _step_raw_motion_once / step_raw_topology / step_world / AIR family).
// ---------------------------------------------------------------------------

/// `_axes_to_quaternion` (kartrider_vehicle_engine.py:831-913): axes are the
/// matrix columns. Returns `[w, x, y, z]`.
fn axes_to_quaternion(axes: [[f64; 3]; 3], numeric_mode: NumericMode) -> Result<[f64; 4], String> {
    if numeric_mode.is_float32() {
        let matrix = [
            axes[0][0], axes[1][0], axes[2][0], axes[0][1], axes[1][1], axes[2][1], axes[0][2],
            axes[1][2], axes[2][2],
        ];
        let trace = fadd(fadd(matrix[0], matrix[4]), matrix[8]);
        let mut quaternion = [0.0; 4];
        if trace > 0.0 {
            let root = f32q(fadd(trace, 1.0).sqrt());
            let scale = fdiv(0.5, root);
            quaternion = [
                fmul(root, 0.5),
                fmul(fsub(matrix[7], matrix[5]), scale),
                fmul(fsub(matrix[2], matrix[6]), scale),
                fmul(fsub(matrix[3], matrix[1]), scale),
            ];
        } else {
            let diagonal = [matrix[0], matrix[4], matrix[8]];
            // Python `max(range(3), key=diagonal.__getitem__)`: first max.
            let mut index = 0usize;
            for candidate in 1..3 {
                if diagonal[candidate] > diagonal[index] {
                    index = candidate;
                }
            }
            let second = (index + 1) % 3;
            let third = (index + 2) % 3;
            let root_argument = fadd(
                fsub(fsub(diagonal[index], diagonal[second]), diagonal[third]),
                1.0,
            );
            let root = f32q(max2_f64(0.0, root_argument).sqrt());
            if root == 0.0 {
                return Err("degenerate float32 rotation matrix".to_string());
            }
            let scale = fdiv(0.5, root);
            quaternion[index + 1] = fmul(root, 0.5);
            quaternion[0] = fmul(
                fsub(matrix[3 * third + second], matrix[3 * second + third]),
                scale,
            );
            quaternion[second + 1] = fmul(
                fadd(matrix[3 * index + second], matrix[3 * second + index]),
                scale,
            );
            quaternion[third + 1] = fmul(
                fadd(matrix[3 * index + third], matrix[3 * third + index]),
                scale,
            );
        }
        let mut length_squared = f32q(0.0);
        for value in quaternion {
            length_squared = fadd(length_squared, fmul(value, value));
        }
        let length = f32q(length_squared.sqrt());
        if length == 0.0 {
            return Err("cannot normalize zero float32 quaternion".to_string());
        }
        return Ok([
            fdiv(quaternion[0], length),
            fdiv(quaternion[1], length),
            fdiv(quaternion[2], length),
            fdiv(quaternion[3], length),
        ]);
    }
    let [m00, m10, m20] = axes[0];
    let [m01, m11, m21] = axes[1];
    let [m02, m12, m22] = axes[2];
    let trace = m00 + m11 + m22;
    let q: [f64; 4];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        q = [0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s];
    } else if m00 > m11 && m00 > m22 {
        let s = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        q = [(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s];
    } else if m11 > m22 {
        let s = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        q = [(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s];
    } else {
        let s = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        q = [(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s];
    }
    // Python math.sqrt(sum(value * value ...)): CPython sum() is Neumaier
    // (crate::num::sum_f64); math.sqrt is the correctly-rounded C sqrt.
    let norm = sum_f64(&[q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3]]).sqrt();
    Ok([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])
}

/// `_quaternion_to_axes` (kartrider_vehicle_engine.py:916-940): returns the
/// three matrix columns from `[w, x, y, z]`.
fn quaternion_to_axes(q: [f64; 4], numeric_mode: NumericMode) -> [[f64; 3]; 3] {
    if numeric_mode.is_float32() {
        let [w, x, y, z] = [f32q(q[0]), f32q(q[1]), f32q(q[2]), f32q(q[3])];
        let (tx, ty, tz) = (fmul(2.0, x), fmul(2.0, y), fmul(2.0, z));
        let (xw, yw, zw) = (fmul(tx, w), fmul(ty, w), fmul(tz, w));
        let (xx, yx, zx) = (fmul(tx, x), fmul(ty, x), fmul(tz, x));
        let (yy, zy, zz) = (fmul(ty, y), fmul(tz, y), fmul(tz, z));
        let row_major = [
            fsub(1.0, fadd(yy, zz)),
            fsub(yx, zw),
            fadd(zx, yw),
            fadd(yx, zw),
            fsub(1.0, fadd(xx, zz)),
            fsub(zy, xw),
            fsub(zx, yw),
            fadd(zy, xw),
            fsub(1.0, fadd(xx, yy)),
        ];
        return matrix_columns(row_major);
    }
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// `_interpolate_axes` (kartrider_vehicle_engine.py:943-1021): slerp (with a
/// linear near-threshold branch) between two rotation matrices, expressed as
/// column axes. Python `math.acos`/`math.sin` map to the platform libm, the
/// same functions Rust's `f64::acos`/`f64::sin` call.
fn interpolate_axes(
    start_matrix: [f64; 9],
    end_matrix: [f64; 9],
    fraction: f64,
    numeric_mode: NumericMode,
) -> Result<[[f64; 3]; 3], String> {
    let q0 = axes_to_quaternion(matrix_columns(start_matrix), numeric_mode)?;
    let mut q1 = axes_to_quaternion(matrix_columns(end_matrix), numeric_mode)?;
    let mut cosine;
    let mut fraction = fraction;
    if numeric_mode.is_float32() {
        fraction = f32q(fraction);
        cosine = f32q(0.0);
        for index in 0..4 {
            cosine = fadd(cosine, fmul(q0[index], q1[index]));
        }
    } else {
        cosine = sum_f64(&[q0[0] * q1[0], q0[1] * q1[1], q0[2] * q1[2], q0[3] * q1[3]]);
    }
    if cosine < 0.0 {
        for value in q1.iter_mut() {
            *value = if numeric_mode.is_float32() {
                f32q(-*value)
            } else {
                -*value
            };
        }
        cosine = if numeric_mode.is_float32() {
            f32q(-cosine)
        } else {
            -cosine
        };
    }
    let linear_threshold = if numeric_mode.is_float32() {
        f32q(0.9995)
    } else {
        0.9995
    };
    if cosine > linear_threshold {
        let mut q = [0.0; 4];
        for index in 0..4 {
            q[index] = if numeric_mode.is_float32() {
                fadd(q0[index], fmul(fsub(q1[index], q0[index]), fraction))
            } else {
                q0[index] + (q1[index] - q0[index]) * fraction
            };
        }
        let normalized = if numeric_mode.is_float32() {
            let mut norm_squared = f32q(0.0);
            for value in q {
                norm_squared = fadd(norm_squared, fmul(value, value));
            }
            let norm = f32q(norm_squared.sqrt());
            [
                fdiv(q[0], norm),
                fdiv(q[1], norm),
                fdiv(q[2], norm),
                fdiv(q[3], norm),
            ]
        } else {
            let norm = sum_f64(&[q[0] * q[0], q[1] * q[1], q[2] * q[2], q[3] * q[3]]).sqrt();
            [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
        };
        return Ok(quaternion_to_axes(normalized, numeric_mode));
    }
    let angle = max2_f64(-1.0, min2_f64(1.0, cosine)).acos();
    let mut q = [0.0; 4];
    if numeric_mode.is_float32() {
        let angle = f32q(angle);
        let denominator = f32q(angle.sin());
        let left_angle = fmul(fsub(1.0, fraction), angle);
        let right_angle = fmul(fraction, angle);
        let left = fdiv(f32q(left_angle.sin()), denominator);
        let right = fdiv(f32q(right_angle.sin()), denominator);
        for index in 0..4 {
            q[index] = fadd(fmul(q0[index], left), fmul(q1[index], right));
        }
    } else {
        let denominator = angle.sin();
        for index in 0..4 {
            q[index] = q0[index] * ((1.0 - fraction) * angle).sin() / denominator
                + q1[index] * (fraction * angle).sin() / denominator;
        }
    }
    Ok(quaternion_to_axes(q, numeric_mode))
}

/// `select_collision_dispatch` result (kartrider_vehicle_engine.py:1024-1057).
/// Fields absent from the Python dict in the non-selected paths are `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionDispatchSelection {
    pub selected: bool,
    pub reason: String,
    pub branch: Option<&'static str>,
    /// Key absent on the `no_hit` path.
    pub approach: Option<f64>,
    /// Quantized (float32 mode) hit normal; present only when selected.
    pub normal: Option<[f64; 3]>,
    pub selector: Option<CollisionResponseSelector>,
}

/// `select_collision_dispatch(candidate, velocity, collision_axes, ...)`
/// (kartrider_vehicle_engine.py:1024-1057): the exact approach gate and
/// hard/soft branch selector. The returned axes payload (`basis_ad0`/
/// `basis_ab8`/`basis_ac4`) is consumed only by the M2 response dispatch and
/// is therefore not carried here.
pub fn select_collision_dispatch(
    candidate: &WorldObbOverlap,
    velocity: [f64; 3],
    arg_threshold: f64,
    e38: bool,
    e4c: bool,
    numeric_mode: NumericMode,
) -> CollisionDispatchSelection {
    if !candidate.hit {
        return CollisionDispatchSelection {
            selected: false,
            reason: "no_hit".to_string(),
            branch: None,
            approach: None,
            normal: None,
            selector: None,
        };
    }
    let normal = if numeric_mode.is_float32() {
        let mut normal = candidate.normal;
        f32_vector(&mut normal);
        normal
    } else {
        candidate.normal
    };
    let arg_threshold = if numeric_mode.is_float32() {
        f32q(arg_threshold)
    } else {
        arg_threshold
    };
    let approach = if numeric_mode.is_float32() {
        dot3_f32(velocity, normal)
    } else {
        dot3_sum(velocity, normal)
    };
    if approach >= 0.0 {
        return CollisionDispatchSelection {
            selected: false,
            reason: "separating_or_tangent".to_string(),
            branch: None,
            approach: Some(approach),
            normal: None,
            selector: None,
        };
    }
    let selector = evaluate_collision_response_selector(normal, arg_threshold, e38, e4c);
    CollisionDispatchSelection {
        selected: true,
        reason: format!("approaching_{}", selector.branch),
        branch: Some(selector.branch),
        approach: Some(approach),
        normal: Some(normal),
        selector: Some(selector),
    }
}

/// Python `collision_selector_state` overlay dict. The only production
/// caller passes `drive_selected_slot` / `drive_raw_aa0` of the current
/// input transaction (kartrider_vehicle_engine.py:5929-5932), but the
/// `selector_state.update(collision_selector_state or {})` merges in
/// `step_world` (:7288, :7461) accept any of the listed keys, so each field
/// is an `Option`: `None` keeps the engine-state default exactly like a
/// missing dict key.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct CollisionSelectorState {
    pub arg_threshold: Option<f64>,
    pub e38: Option<bool>,
    pub e4c: Option<bool>,
    pub drive_selected_slot: Option<f64>,
    pub drive_raw_aa0: Option<f64>,
    pub drive_sign_aac: Option<bool>,
    pub selector_af4: Option<bool>,
    pub flag_188: Option<bool>,
    /// Python `selector_state.get("soft_optional_active")` (:7481): no
    /// engine-state default exists, so the un-overlaid value is `None`.
    pub soft_optional_active: Option<bool>,
}

/// The merged `selector_state` dict of the collision loop
/// (:7278-7288/:7451-7461): engine-state defaults with the
/// [`CollisionSelectorState`] overlay applied, so every key the chain and
/// the single-contact soft path read is concrete. This is also the
/// `selector_state` argument of the direct `_apply_collision_chain` calls
/// (mixed/pure-soft/pure-hard/state6 parameterizations).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct CollisionChainSelectorState {
    pub arg_threshold: f64,
    pub e38: bool,
    pub e4c: bool,
    pub drive_selected_slot: f64,
    pub drive_raw_aa0: f64,
    pub drive_sign_aac: bool,
    pub selector_af4: bool,
    pub flag_188: bool,
}

/// `velocity_response` payload of one `_apply_collision_chain` response:
/// the hard direct replacement or the soft kind8 source.
#[derive(Debug, Clone, PartialEq)]
pub enum CollisionChainVelocityResponse {
    Hard(HardContactVelocityResponse),
    Soft(SoftContactSource),
}

/// `angular_response` payload of one `_apply_collision_chain` response (also
/// the `world_sweep["angular_response"]` of the single-contact paths).
#[derive(Debug, Clone, PartialEq)]
pub enum CollisionChainAngularResponse {
    Hard(HardContactAngularResponse),
    Soft(SoftContactAngularSource),
}

/// One `responses` entry of `_apply_collision_chain`
/// (kartrider_vehicle_engine.py:1192-1205).
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionChainResponse {
    pub loop_index: i64,
    pub branch: &'static str,
    pub candidate: WorldObbOverlap,
    pub selection: CollisionDispatchSelection,
    pub velocity_before: [f64; 3],
    pub velocity_after: [f64; 3],
    pub angular_velocity_before: [f64; 3],
    pub angular_velocity_after: [f64; 3],
    pub velocity_response: CollisionChainVelocityResponse,
    pub angular_response: CollisionChainAngularResponse,
}

/// `_apply_collision_chain` result (kartrider_vehicle_engine.py:1206-1212).
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionChainOutcome {
    pub responses: Vec<CollisionChainResponse>,
    /// Python `branches` set; `BTreeSet` keeps the equality semantics of
    /// `mixed_chain["branches"] == {"hard_contact", "soft_contact"}` (:7301).
    pub branches: BTreeSet<&'static str>,
    pub crossed_branch: bool,
    pub velocity_after: [f64; 3],
    pub angular_velocity_after: [f64; 3],
}

/// Python `world_sweep["soft_base_response"]` (:7619): the full kind8
/// source (exact selector state) or the base-only source.
#[derive(Debug, Clone, PartialEq)]
pub enum SoftBaseResponse {
    Full(SoftContactSource),
    Base(SoftContactBaseSource),
}

/// The chain payload `world_sweep[chain_key]` plus its composition
/// authority (kartrider_vehicle_engine.py:1285-1290).
#[derive(Debug, Clone, PartialEq)]
pub struct CollisionChainPayload {
    pub chain_key: String,
    pub responses: Vec<CollisionChainResponse>,
    pub composition_authority: String,
    /// Python adds `"hard_response_chain": []` alongside the soft chain
    /// (:1289-1290).
    pub hard_response_chain_empty: bool,
}

/// One selected P1 dynamic-event writer in authored candidate order.
#[derive(Debug, Clone, PartialEq)]
pub struct EventObstacleChainResponse {
    pub loop_index: i64,
    pub candidate: EventObstacleContact,
    pub response: EventObstacleContactResponse,
}

/// Dynamic event candidates plus the selected terminal-writer chain.
#[derive(Debug, Clone, PartialEq)]
pub struct EventObstacleContactInfo {
    pub candidates: Vec<EventObstacleContact>,
    pub responses: Vec<EventObstacleChainResponse>,
}

/// Python `world_sweep` payload (kartrider_vehicle_engine.py:7604-7622 and
/// the chain-finalize shape :1268-1288). The `radius` / `half_extents` /
/// `center_up` / `orientation_basis_*` keys are not carried (the only
/// production sweep shape is `obb_rotating` and parity harnesses project
/// the fields below).
#[derive(Debug, Clone, PartialEq)]
pub struct WorldSweepInfo {
    pub shape: &'static str,
    pub start: [f64; 3],
    pub end: [f64; 3],
    /// Python `candidate`: the last chain response's candidate on finalize
    /// paths, the dispatched/fallback candidate otherwise (`None` on
    /// `no_hit`).
    pub candidate: Option<WorldObbOverlap>,
    pub candidates: Vec<WorldObbOverlap>,
    pub response_dispatched: bool,
    pub response_complete: bool,
    pub dispatch_reason: String,
    /// Python `dispatch_authority`: `"collision_solver_dispatch"` when a
    /// response dispatched, else `None`.
    pub dispatch_authority: Option<&'static str>,
    pub angular_response: Option<CollisionChainAngularResponse>,
    pub soft_base_response: Option<SoftBaseResponse>,
    /// Present on the chain-finalize paths only.
    pub chain: Option<CollisionChainPayload>,
    /// Present when the world exposes the P1 dynamic-event query.
    pub event_obstacle_contact: Option<EventObstacleContactInfo>,
    pub topology_transition: Option<TopologyDecision>,
}

/// `step_world` result: the `step` outcome plus the `world_sweep` payload.
/// `control_prediction` is present when the frame carried a `control`
/// payload (Python dispatches those to `step_control`, whose result dict
/// carries the prediction).
#[derive(Debug, Clone, PartialEq)]
pub struct WorldStepOutcome {
    pub step: StepOutcome,
    pub world_sweep: WorldSweepInfo,
    pub control_prediction: Option<ControlPrediction>,
}

/// `inspect_autonomous_support` result (kartrider_vehicle_engine.py:6121-6177).
/// The Python `surface_selection`, `contact_probe_debug`,
/// `contact_compression_after`, `contact_accumulator`, and the four
/// accumulator rows are all read out of [`SupportObservation::producer`]
/// (`surface_selection`, `contact_probe_debug`, `contact_compression_after`,
/// `evaluator.contact`, `evaluator.accumulator_614_after_post_bs_reset`,
/// `evaluator.correction`).
#[derive(Debug, Clone, PartialEq)]
pub struct SupportObservation {
    /// The four wheel planes (query rows or caller-supplied planes).
    pub planes: Vec<WorldGroundResult>,
    pub active_slots: [bool; 4],
    pub support_contact_count: i64,
    pub surface_mode: i64,
    pub producer: GroundProducerOutput,
    pub authority: &'static str,
}

/// Convert world query rows to the `GroundPlaneSet` consumed by the ground
/// producer. The world row always carries `node`/`tri_index`
/// (`parse_world_query_tsv`, scripts/kart_track_world.py:57-77), so the
/// converted plane keeps them as present values.
fn ground_plane_set_from_world_planes(planes: &[WorldGroundResult]) -> GroundPlaneSet {
    GroundPlaneSet::Sequence(
        planes
            .iter()
            .map(|plane| GroundPlane {
                normal: plane.normal,
                point: plane.point,
                triangle: plane.triangle,
                road_surface: Some(plane.road_surface.clone()),
                node: Some(JsonValue::String(plane.node.clone())),
                tri_index: Some(plane.tri_index),
            })
            .collect(),
    )
}

/// One committed takeoff/landing hard velocity transfer, or the
/// not-approaching landing placeholder (kartrider_vehicle_engine.py:6670-6696,
/// :6748-6789).
#[derive(Debug, Clone, PartialEq)]
pub struct TakeoffLandingTransfer {
    pub role: String,
    /// Python `selected` key: absent on hard-response transfers (`None`),
    /// `Some(false)` on the not-approaching landing placeholder.
    pub selected: Option<bool>,
    /// Python `reason` key (placeholder only).
    pub reason: Option<&'static str>,
    /// Keys absent on the not-approaching landing placeholder.
    pub velocity_before: Option<[f64; 3]>,
    pub velocity_after: Option<[f64; 3]>,
    pub hit_normal: [f64; 3],
    pub basis_ad0: [f64; 3],
    pub approach_velocity: f64,
    pub selector_authority: String,
    pub active_support_slot_count: Option<i64>,
    /// The B2DC80 response composition (hard-response transfers only).
    pub response: Option<TakeoffOrLandingContact>,
}

/// `_apply_autonomous_authored_trigger` result
/// (kartrider_vehicle_engine.py:3589-3730), support path only.
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredTriggerEdge {
    pub selected: bool,
    /// Python `reason` key on the already-latched edge.
    pub reason: Option<&'static str>,
    pub contact_id: String,
    /// The evaluated JM/DJ trigger (selected edges only).
    pub trigger: Option<JmDjTrigger>,
    /// The `model_args` payload fed into the ground frame (selected edges).
    pub model_args: Option<JmDjArgs>,
    pub authority: Option<&'static str>,
}

/// `_select_current_pose_hard_overlap` result
/// (kartrider_vehicle_engine.py:1293-1339): the world overlap row plus the
/// engine-assigned `selected`/`source`/`collision_dispatch` keys.
#[derive(Debug, Clone, PartialEq)]
pub struct HardOverlapCandidate {
    pub overlap: WorldObbOverlap,
    pub source: String,
    pub selection: CollisionDispatchSelection,
}

/// `inspect_autonomous_rail_capture` result
/// (kartrider_vehicle_engine.py:3753-3791). Mirrors the engine-produced
/// latch dict: `rail_state` is present only when `contact_active`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailCaptureInspection {
    pub contact_active: bool,
    pub rail_state: Option<i64>,
    pub selector: Option<BTreeMap<String, JsonValue>>,
    pub authority: &'static str,
}

/// `_rail_state1_ground_writer_result` (kartrider_vehicle_engine.py:1342-1377):
/// the shared state-1 ground-writer evidence payload. The deep-copied
/// evaluator evidence (`ground_correction`, `ground_force`,
/// `linear_integrator`, `world_sweep`, `post_linear_collision_response`) is
/// carried by the enclosing typed [`GroundStepOutcome`] instead of being
/// duplicated into an open mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1GroundWriter {
    /// Python `selected` key; always `true` when this payload exists.
    pub selected: bool,
    pub state_after: i64,
    pub support_contact_count: i64,
    pub selector_phase: Option<&'static str>,
    pub authority: &'static str,
}

/// Python `post_ground_rail_capture` dict
/// (kartrider_vehicle_engine.py:6858-6866).
#[derive(Debug, Clone, PartialEq)]
pub struct PostGroundRailCapture {
    pub contact: RailCaptureInspection,
    pub topology_decision: TopologyDecision,
    pub selector_phase: &'static str,
    pub shared_ground_epoch_count: i64,
    pub authority: &'static str,
}

/// Python `rail_contract` dict on rail capture results
/// (kartrider_vehicle_engine.py:3856-3873).
#[derive(Debug, Clone, PartialEq)]
pub struct RailContract {
    pub control_authority: &'static str,
    pub rigid_body_state_authority: &'static str,
    pub writer_selection_authority: &'static str,
    pub rail_context_authority: &'static str,
    pub capture_authority: &'static str,
    pub open_boundary: &'static str,
}

/// `acquire_autonomous_rail` state-1 ground-writer capture outcome
/// (kartrider_vehicle_engine.py:3823-3874).
#[derive(Debug, Clone, PartialEq)]
pub struct RailCaptureOutcome {
    pub ground: Box<GroundStepOutcome>,
    pub rail_state1_ground_writer: RailState1GroundWriter,
    pub topology_decision: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub autonomous_rail_selector: BTreeMap<String, JsonValue>,
    pub rail_contract: RailContract,
}

/// `step_raw_rail` `rail_contract` dict (five keys; unlike
/// [`RailContract`] it carries no `capture_authority`).
#[derive(Debug, Clone, PartialEq)]
pub struct RailStepContract {
    pub control_authority: &'static str,
    pub rigid_body_state_authority: &'static str,
    pub writer_selection_authority: &'static str,
    pub rail_context_authority: &'static str,
    pub open_boundary: &'static str,
}

/// `step_raw_rail` release-path `rail_selector` dict
/// (kartrider_vehicle_engine.py:2510-2515).
#[derive(Debug, Clone, PartialEq)]
pub struct RailReleaseSelector {
    pub rail_contact: bool,
    pub rail_state_before: i64,
    pub support_contact_count: i64,
    pub surface_mode: i64,
}

/// `_step_rail_state6_tail` post-release next-epoch producer payload
/// (kartrider_vehicle_engine.py:2491-2542). The active state-6 accumulator
/// was consumed by the release epoch; this payload is produced from the
/// finalized pose and committed for the first GND/AIR successor.
#[derive(Debug, Clone, PartialEq)]
pub struct RailReleasePendingProducer {
    pub destination: String,
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub contact_compression: [f64; 4],
    pub takeoff_separation_pending_before_release: bool,
    pub takeoff_separation_pending_after_release: bool,
    pub authority: &'static str,
}

/// `step_raw_rail(rail_contact=False)` outcome
/// (kartrider_vehicle_engine.py:2499-2530): continuous release with no rail
/// writer. `release_destination` is the Python `MotionDomain.value` string.
#[derive(Debug, Clone, PartialEq)]
pub struct RailReleaseOutcome {
    pub release_destination: String,
    pub rigid_state_transfer: &'static str,
    pub rail_selector: RailReleaseSelector,
    pub topology_decision: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub rail_contract: RailStepContract,
    pub rail_release_pending_producer: Option<RailReleasePendingProducer>,
}

/// `step_raw_rail` `accumulator_clear` dict (kartrider_vehicle_engine.py:2633-2639).
/// The after-clear accumulators are constant zeros.
#[derive(Debug, Clone, PartialEq)]
pub struct RailAccumulatorClear {
    pub force_before_clear: [f64; 3],
    pub torque_before_clear: [f64; 3],
    pub authority: &'static str,
}

/// `step_raw_rail(rail_contact=True)` outcome
/// (kartrider_vehicle_engine.py:2531-2659): the B29350 writer chain result
/// plus the committed state/topology.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRailActiveOutcome {
    pub rail_step: RailStepResult,
    pub accumulator_clear: RailAccumulatorClear,
    pub topology_decision: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub rail_contract: RailStepContract,
}

/// `step_raw_rail` outcome: either the active writer chain or the release.
#[derive(Debug, Clone, PartialEq)]
pub enum RawRailOutcome {
    Active(Box<RawRailActiveOutcome>),
    Released(Box<RailReleaseOutcome>),
}

/// `acquire_autonomous_rail` capture through `step_raw_rail`
/// (kartrider_vehicle_engine.py:3875-3895): rail_state 2..6, or state 1
/// without a support observation.
#[derive(Debug, Clone, PartialEq)]
pub struct RailCaptureRawRailOutcome {
    pub rail: RawRailActiveOutcome,
    pub autonomous_rail_selector: BTreeMap<String, JsonValue>,
    pub rail_contract: RailContract,
}

/// `acquire_autonomous_rail` result when the selector offered a rail
/// (kartrider_vehicle_engine.py:3793-3895).
#[derive(Debug, Clone, PartialEq)]
pub enum RailCaptureExecution {
    /// State-1 capture with a support observation (ground-writer path).
    GroundWriter(Box<RailCaptureOutcome>),
    /// Capture through the `step_raw_rail` writer chain.
    RawRail(Box<RailCaptureRawRailOutcome>),
}

/// Unified view of the `state1_gate` mapping consumed by
/// `_advance_autonomous_rail_selector` and `step_autonomous_rail`: either the
/// open JSON mapping carried by the world selector payload or the typed
/// [`RailState1Gate`] produced by `evaluate_rail_state1_query`, with the
/// Python `dict.get` defaults applied at construction. An empty view is the
/// Python `{}` fallback (`gate = {}`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RailState1GateView {
    /// Python dict truthiness (`if gate and ...`); the typed query gate is
    /// always non-empty.
    pub nonempty: bool,
    pub authority: Option<String>,
    pub contact_exists: bool,
    pub contact_identifier_nonempty: bool,
    pub path_metric: Option<f64>,
    pub metric_threshold: Option<f64>,
    pub path_z: Option<f64>,
    pub kart_z_plus_one: Option<f64>,
    pub alignment_base_ms: Option<f64>,
    pub alignment_primary_scale_ms: Option<f64>,
    pub alignment_stable_initial_ms: Option<f64>,
    pub selected_context: Option<JsonValue>,
    pub query_scalar: Option<f64>,
    pub lookahead_distance: Option<f64>,
    /// Raw lookahead endpoint; a present-but-malformed JSON value decodes to
    /// an empty vector so the consumer raises the same "requires the
    /// autonomous one-metre lookahead endpoint" ValueError Python raises
    /// for `not isinstance(...) or len(...) != 3`.
    pub lookahead_endpoint: Option<Vec<f64>>,
    pub advance_context: Option<JsonValue>,
    pub terminal_state6_edge: bool,
    pub terminal_distance_remaining: Option<f64>,
    /// `rail_base_orientation` presence and its `matrix_bf8`; `Some(None)`
    /// reproduces the Python "autonomous rail path query returned invalid
    /// BF8 matrix" ValueError at consumption time.
    pub rail_base_orientation_matrix_bf8: Option<Option<[f64; 9]>>,
}

/// `_advance_autonomous_rail_selector` result
/// (kartrider_vehicle_engine.py:3315-3432): the
/// [`RailCaptureSelectorAdvance`] plus the autonomous state-1 query when the
/// selector payload carried `registered_query_contexts` (Python
/// `transition["autonomous_state1_query"]`).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailSelectorAdvance {
    pub advance: RailCaptureSelectorAdvance,
    pub autonomous_state1_query: Option<RailState1QueryResult>,
}

/// `_query_autonomous_rail_authored_trigger` result
/// (kartrider_vehicle_engine.py:3434-3527). The `scope` metadata key is not
/// ported (static provenance).
#[derive(Debug, Clone, PartialEq)]
pub struct RailAuthoredTriggerGeometry {
    pub authored_trigger: Option<BTreeMap<String, JsonValue>>,
    pub authority: String,
    /// Absent on the capability-absent row (Python omits the `center` key).
    pub center: Option<[f64; 3]>,
    pub candidate: Option<WorldObbOverlap>,
    /// Absent unless the no-match row carried rejections (Python key).
    pub rejected_jm_without_contact_basis: Option<i64>,
}

/// One matched JM/DJ overlap entry of
/// [`KartRiderVehicleEngine::query_autonomous_rail_authored_trigger`]:
/// (decoded token, surface_id) key and (candidate, trigger payload, variant).
type AuthoredTriggerMatch = (
    (String, i64),
    (WorldObbOverlap, BTreeMap<String, JsonValue>, &'static str),
);

/// `_evaluate_rail_authored_trigger` result
/// (kartrider_vehicle_engine.py:3529-3587). The `ordering` and
/// `static_source` metadata keys are static provenance and are not ported;
/// `authored_token` / `selection_authority` fold into the trigger payload on
/// the Python side and live here as fields.
#[derive(Debug, Clone, PartialEq)]
pub struct RailAuthoredTriggerEdge {
    pub selector: RailCaptureSelectorResult,
    pub trigger: JmDjTrigger,
    pub authored_token: String,
    pub selection_authority: &'static str,
}

/// `step_autonomous_rail` state1->2 / sustain path velocity transfer payload
/// (kartrider_vehicle_engine.py:2964-2974 / :3030-3053). The merged
/// `**velocity_transfer` keys live in [`RailState1VelocityTransfer`];
/// `static_source` is static provenance and folds into `velocity_writer`.
#[derive(Debug, Clone, PartialEq)]
pub struct RailPathVelocityTransfer {
    pub state_before: i64,
    pub state_after: i64,
    pub transfer: RailState1VelocityTransfer,
    pub lookahead_distance: Option<f64>,
    /// Present only on the state1->2 transfer (Python key).
    pub velocity_before: Option<[f64; 3]>,
    /// Present only on the state2/state3 sustain transfer (Python keys).
    pub query_scalar: Option<f64>,
    pub selected_context: Option<JsonValue>,
    /// Python `context_transfer` sub-dict (sustain path only).
    pub context_transfer: Option<RailPathContextTransfer>,
    /// Python `rail_writer_direction.rail_dir` (the normalized writer
    /// direction written back into the decoded runtime context).
    pub rail_writer_direction: [f64; 3],
    pub velocity_writer: &'static str,
}

/// Python `context_transfer` sub-dict of the sustain path transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct RailPathContextTransfer {
    pub from: JsonValue,
    pub to: JsonValue,
    pub authority: &'static str,
}

/// `step_autonomous_rail` `rail_state1_post_linear_collision_response` dict
/// (kartrider_vehicle_engine.py:2820-2834). `selected` is constant true when
/// this payload exists; `writer` / `scheduler_phase` / `authority` are
/// static provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState1PostLinearCollisionResponse {
    pub velocity_delta: [f64; 3],
    pub path_speed: f64,
    pub gain_per_second: f64,
    pub low_speed_limit: f64,
}

/// `step_autonomous_rail` `rail_terminal_position_phase` dict
/// (kartrider_vehicle_engine.py:3205-3213).
#[derive(Debug, Clone, PartialEq)]
pub struct RailTerminalPositionPhase {
    pub published_position: [f64; 3],
    pub suppressed_post_setter_position: [f64; 3],
    pub published_velocity: [f64; 3],
    pub suppressed_post_setter_velocity: [f64; 3],
}

/// `step_autonomous_rail` `rail_terminal_endpoint_transition` dict
/// (kartrider_vehicle_engine.py:3260-3274). `transitioned` / `state_before`
/// / `state_after` are constant (true / 3 / 6); `velocity_replacement` is
/// constant false; `authority` is static provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct RailTerminalEndpointTransition {
    pub selected_context: Option<JsonValue>,
    pub query_scalar: Option<f64>,
    pub terminal_distance_remaining: Option<f64>,
}

/// `step_autonomous_rail` `rail_state6_entry_force` dict
/// (kartrider_vehicle_engine.py:3288-3296). `force_accumulator_614` is
/// constant zeros and `delay_substeps_before_614_consumption` is constant 1.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6EntryForce {
    pub force_accumulator_b74: [f64; 3],
}

/// `step_autonomous_rail` outcome for the state-1 ground-writer path
/// (kartrider_vehicle_engine.py:2731-2894).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailState1Outcome {
    pub ground: GroundStepOutcome,
    pub rail_state1_post_linear_collision_response: Option<RailState1PostLinearCollisionResponse>,
    pub rail_state1_ground_writer: RailState1GroundWriter,
    pub autonomous_rail_selector: Option<BTreeMap<String, JsonValue>>,
    pub support_observation: SupportObservation,
    pub rail_state_transition: Option<AutonomousRailSelectorAdvance>,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub rail_contract: RailStepContract,
}

/// `step_autonomous_rail` outcome when the world selector reports no active
/// rail contact and the support manifold enters the authored reset surface
/// (kartrider_vehicle_engine.py:2686-2707). `rail_released` is constant
/// true; `release_destination` is constant `"RESET"`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailResetReleaseOutcome {
    pub reset: AuthoredResetSurfaceOutcome,
    pub autonomous_rail_selector: Option<BTreeMap<String, JsonValue>>,
    pub rail_contract: RailStepContract,
}

/// `step_autonomous_rail` outcome when the world selector reports no active
/// rail contact and the ordinary release path runs
/// (kartrider_vehicle_engine.py:2708-2721).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailContactReleaseOutcome {
    pub release: RailReleaseOutcome,
    pub autonomous_rail_selector: Option<BTreeMap<String, JsonValue>>,
    pub support_observation: SupportObservation,
}

/// `step_autonomous_rail` outcome when the state machine itself releases
/// (selector transitioned to state 0; kartrider_vehicle_engine.py:2899-2922).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailSelectorReleaseOutcome {
    pub release_destination: String,
    pub autonomous_rail_selector: BTreeMap<String, JsonValue>,
    pub rail_state_transition: Option<AutonomousRailSelectorAdvance>,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub rail_contract: RailStepContract,
}

/// `step_autonomous_rail` outcome for the active sustain/transition path
/// (kartrider_vehicle_engine.py:2923-3313). `topology_after` / `state_after`
/// are the FINAL refreshed snapshots (Python overwrites the `step_raw_rail`
/// result keys in place); the `rail` payload keeps the as-committed
/// `step_raw_rail` snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousRailActiveOutcome {
    pub rail: RawRailActiveOutcome,
    pub rail_terminal_position_phase: Option<RailTerminalPositionPhase>,
    pub rail_support_observation: Option<SupportObservation>,
    pub rail_authored_trigger: Option<RailAuthoredTriggerEdge>,
    pub rail_authored_trigger_geometry: Option<RailAuthoredTriggerGeometry>,
    pub rail_terminal_endpoint_transition: Option<RailTerminalEndpointTransition>,
    pub rail_state6_entry_force: Option<RailState6EntryForce>,
    pub autonomous_rail_selector: BTreeMap<String, JsonValue>,
    pub rail_state_transition: Option<AutonomousRailSelectorAdvance>,
    pub rail_path_velocity_transfer: Option<RailPathVelocityTransfer>,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
}

/// `_step_rail_state6_tail` `rail_state6_support_force` dict
/// (kartrider_vehicle_engine.py:2380-2386).
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6SupportForce {
    pub force: Option<[f64; 3]>,
    pub torque_after_state6_angular: [f64; 3],
}

/// `_step_rail_state6_tail` `rail_state6_entry_force_transaction` dict
/// (kartrider_vehicle_engine.py:2124-2153).
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6EntryForceTransaction {
    pub phase: &'static str,
    pub delay_substeps_after: Option<i64>,
    pub force: [f64; 3],
}

/// `_step_rail_state6_tail` `rail_state6_common_angular_damping` dict
/// (kartrider_vehicle_engine.py:1982-2019). The `runtime_evidence` /
/// `closure_status` metadata keys are static provenance and are not ported.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6CommonAngular {
    pub angular_velocity_before: [f64; 3],
    pub coefficient: f64,
    pub final_coefficient_after_common_damping: f64,
    pub torque_delta: [f64; 3],
}

/// One `missing_wheel_queries` entry of the B2FB00 partial-wheel bridge
/// (kartrider_vehicle_engine.py:2328-2334).
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6MissingWheelQuery {
    pub slot: usize,
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub candidate: BTreeMap<String, JsonValue>,
    pub hit: bool,
}

/// `_step_rail_state6_tail` `rail_state6_partial_wheel_bridge` dict
/// (kartrider_vehicle_engine.py:2276-2352). The `static_source` metadata key
/// is not ported.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6PartialWheelBridge {
    pub selected: bool,
    pub contact_188: bool,
    pub active_slots: [bool; 4],
    pub active_count: i64,
    pub missing_wheel_queries: Vec<RailState6MissingWheelQuery>,
    pub velocity_delta: [f64; 3],
}

/// `_step_rail_state6_tail` `state6_contact_candidate` dict
/// (kartrider_vehicle_engine.py:2401-2413). The `authority` metadata key is
/// static provenance and is not ported.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6ContactCandidate {
    pub segment_start: [f64; 3],
    pub segment_end: [f64; 3],
    pub candidate: BTreeMap<String, JsonValue>,
    pub valid: bool,
    pub selected_source: &'static str,
    pub contact_188_before: bool,
    pub contact_188_after: bool,
}

/// `_step_rail_state6_tail` `rail_state6_exit` payload: the (possibly
/// overridden) state-6 selector plus the
/// `gates["float_zero_boundary_writer_committed"]` override flag the Python
/// code writes into the gates dict (kartrider_vehicle_engine.py:1913-1964).
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6Exit {
    pub selector: RailCaptureSelectorResult,
    pub float_zero_boundary_writer_committed: Option<bool>,
}

/// `_step_rail_state6_tail` outcome (kartrider_vehicle_engine.py:1798-2452).
/// `rail_state6_tail` is constant true.
#[derive(Debug, Clone, PartialEq)]
pub struct RailState6TailOutcome {
    pub airborne: AirborneStep,
    pub rail_state6_contact_physics: bool,
    pub rail_state6_common_angular_damping: RailState6CommonAngular,
    pub rail_state6_support_force_branch: &'static str,
    pub rail_state6_support_force: RailState6SupportForce,
    pub rail_state6_entry_force_transaction: Option<RailState6EntryForceTransaction>,
    pub rail_state6_collision_schedule: &'static str,
    pub rail_state6_collision_chain: Vec<CollisionChainResponse>,
    pub rail_state6_partial_wheel_bridge: RailState6PartialWheelBridge,
    pub rail_state6_exit: RailState6Exit,
    pub support_observation: SupportObservation,
    pub state6_contact_slots: [bool; 4],
    pub state6_contact_candidate: RailState6ContactCandidate,
    pub rail_released: bool,
    pub release_destination: Option<String>,
    pub topology_decision: Option<TopologyDecision>,
    pub rail_release: Option<Box<RailReleaseOutcome>>,
    pub rail_release_pending_producer: Option<RailReleasePendingProducer>,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub rail_contract: RailStepContract,
}

/// `step_autonomous_rail` outcome (kartrider_vehicle_engine.py:2661-3313).
#[derive(Debug, Clone, PartialEq)]
pub enum AutonomousRailOutcome {
    /// rail_state 6: the B276F0 contact-confirmation tail.
    State6Tail(Box<RailState6TailOutcome>),
    /// Negative world selector + authored reset support.
    ResetRelease(Box<AutonomousRailResetReleaseOutcome>),
    /// Negative world selector + ordinary release.
    ContactRelease(Box<AutonomousRailContactReleaseOutcome>),
    /// State machine released to state 0.
    SelectorRelease(Box<AutonomousRailSelectorReleaseOutcome>),
    /// State-1 ground-writer capture bookkeeping.
    State1GroundWriter(Box<AutonomousRailState1Outcome>),
    /// Active sustain/transition through `step_raw_rail`.
    Active(Box<AutonomousRailActiveOutcome>),
}

/// Keyword bundle of `advance_rail_capture_selector`
/// (kartrider_vehicle_engine.py:3897-3920).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdvanceRailCaptureSelectorArgs {
    pub elapsed_in_state_ms: f64,
    pub field_af0: f64,
    pub contact_exists: bool,
    pub contact_identifier_nonempty: bool,
    pub path_metric: Option<f64>,
    pub metric_threshold: Option<f64>,
    pub path_z: Option<f64>,
    pub kart_z_plus_one: Option<f64>,
    pub delta_ms: Option<f64>,
    pub alignment_base_ms: Option<f64>,
    pub alignment_primary_scale_ms: Option<f64>,
    pub alignment_stable_initial_ms: Option<f64>,
    pub alignment_primary_remaining_ms: Option<f64>,
    pub alignment_stable_remaining_ms: Option<f64>,
    pub forward_speed: Option<f64>,
    pub lateral_speed: Option<f64>,
    pub trigger_cooldown_remaining_ms: Option<f64>,
    pub authored_trigger_accepted: bool,
    pub release_support_contact_count: i64,
    pub release_surface_mode: i64,
}

/// `advance_rail_capture_selector` result
/// (kartrider_vehicle_engine.py:4044-4056).
#[derive(Debug, Clone, PartialEq)]
pub struct RailCaptureSelectorAdvance {
    pub selector: RailCaptureSelectorResult,
    pub topology_transition: Option<TopologyDecision>,
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub selector_authority: &'static str,
    pub open_boundary: &'static str,
}

/// `inspect_autonomous_moving_contact` result
/// (kartrider_vehicle_engine.py:4117-4168).
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactInspection {
    pub contact_active: bool,
    pub record: Option<BTreeMap<String, JsonValue>>,
    pub object_id: Option<String>,
    pub authority: &'static str,
}

/// `acquire_moving_contact` result (kartrider_vehicle_engine.py:4058-4115).
/// `entry_transfer` is constant
/// `"continuous_world_state_to_object_local_state"`.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactAcquisition {
    pub anchor: MovingContactAnchor,
    pub topology_transition: TopologyDecision,
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub entry_transfer: &'static str,
}

/// `acquire_autonomous_moving_contact` result
/// (kartrider_vehicle_engine.py:4170-4190).
#[derive(Debug, Clone, PartialEq)]
pub enum AcquireAutonomousMovingContactOutcome {
    /// No active dynamic-contact record: `acquired` is false and the state
    /// is untouched.
    NotActive {
        selector: MovingContactInspection,
        state_after: Box<VehicleState>,
    },
    /// Acquired MOVING-CONTACT ownership (`acquired` true,
    /// `entry_authority="autonomous_world_dynamic_contact"`).
    Acquired(Box<AutonomousMovingContactAcquisition>),
}

/// The `Acquired` payload of [`AcquireAutonomousMovingContactOutcome`].
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousMovingContactAcquisition {
    pub acquisition: MovingContactAcquisition,
    pub selector: MovingContactInspection,
    pub entry_authority: &'static str,
}

/// Python `moving_contact_contract` dict
/// (kartrider_vehicle_engine.py:4276-4286).
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactStepContract {
    /// `"engine_owned_acquired_anchor"`, overridden to
    /// `"autonomous_world_dynamic_contact_initial_capture"` by
    /// `acquire_and_step_autonomous_moving_contact` (:4230-4232).
    pub reference_frame_authority: String,
    pub object_motion_authority: &'static str,
    pub open_boundary: &'static str,
}

/// `step_moving_contact` result (kartrider_vehicle_engine.py:4235-4287): the
/// evaluator composition plus the committed state/topology.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactStepOutcome {
    pub step: MovingContactStep,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub moving_contact_contract: MovingContactStepContract,
}

/// `acquire_and_step_autonomous_moving_contact` result
/// (kartrider_vehicle_engine.py:4192-4233). `moving_contact_acquired` is
/// constant true on this path.
#[derive(Debug, Clone, PartialEq)]
pub struct AcquireAndStepMovingContactOutcome {
    pub step: MovingContactStepOutcome,
    pub acquisition: MovingContactAcquisition,
    /// Python `moving_contact_selector`: the deep-copied validated record.
    pub moving_contact_selector: BTreeMap<String, JsonValue>,
}

/// `release_moving_contact` result (kartrider_vehicle_engine.py:4372-4401).
/// `exit_transfer` is constant
/// `"continuous_object_local_state_to_world_state"`.
#[derive(Debug, Clone, PartialEq)]
pub struct MovingContactReleaseOutcome {
    pub anchor_before: Option<MovingContactAnchor>,
    pub topology_transition: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub exit_transfer: &'static str,
}

/// `step_autonomous_moving_contact` sustained path
/// (kartrider_vehicle_engine.py:4325-4349). `released` is constant false.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousMovingContactSustainOutcome {
    pub step: MovingContactStepOutcome,
    pub moving_contact_selector: MovingContactSelectorResult,
    pub released: bool,
}

/// `step_autonomous_moving_contact` release-onto-reset-surface path
/// (kartrider_vehicle_engine.py:4351-4359). `released` is constant true;
/// `exit_authority` is constant
/// `"autonomous_dynamic_contact_release_to_reset_surface"`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousMovingContactResetReleaseOutcome {
    pub reset: AuthoredResetSurfaceOutcome,
    pub moving_contact_selector: MovingContactSelectorResult,
    pub released: bool,
    pub exit_authority: &'static str,
}

/// `step_autonomous_moving_contact` ordinary release path
/// (kartrider_vehicle_engine.py:4360-4370). `released` is constant true;
/// `exit_authority` is constant
/// `"autonomous_dynamic_contact_and_four_wheel_support_geometry"`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousMovingContactReleaseOutcome {
    pub release: MovingContactReleaseOutcome,
    pub moving_contact_selector: MovingContactSelectorResult,
    pub support_observation: SupportObservation,
    pub released: bool,
    pub exit_authority: &'static str,
}

/// `step_autonomous_moving_contact` outcome
/// (kartrider_vehicle_engine.py:4289-4370).
#[derive(Debug, Clone, PartialEq)]
pub enum AutonomousMovingContactOutcome {
    /// Selector sustained the moving frame; the substep was integrated.
    Sustained(Box<AutonomousMovingContactSustainOutcome>),
    /// Released onto the authored 리셋 surface (RESET ownership).
    ResetRelease(Box<AutonomousMovingContactResetReleaseOutcome>),
    /// Ordinary release through the four-wheel support geometry.
    Released(Box<AutonomousMovingContactReleaseOutcome>),
}

/// `apply_checkpoint_respawn` result (kartrider_vehicle_engine.py:1637-1702).
/// `support_observation` / `topology_transition` / `state_after` are present
/// only when the replacement was selected (Python adds the keys on that
/// branch).
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointRespawnApplyOutcome {
    pub replacement: CheckpointRespawnStateReplacement,
    pub topology_after: TopologyState,
    pub support_observation: Option<SupportObservation>,
    pub topology_transition: Option<TopologyDecision>,
    pub state_after: Option<VehicleState>,
}

/// `step_autonomous_reset` result when the world selector is not ready
/// (kartrider_vehicle_engine.py:1720-1727). `selected` is constant false;
/// `reason` is constant `"checkpoint_respawn_not_ready"`;
/// `selector_authority` is constant `"world_query_checkpoint_respawn"`.
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousResetNotReadyOutcome {
    pub selected: bool,
    pub reason: &'static str,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub selector_authority: &'static str,
}

/// `step_autonomous_reset` result when the world selector produced a
/// candidate (kartrider_vehicle_engine.py:1739-1748).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousResetAppliedOutcome {
    pub respawn: CheckpointRespawnApplyOutcome,
    pub checkpoint_respawn_selector: CheckpointRespawnCandidate,
    pub selector_authority: String,
}

/// `step_autonomous_reset` outcome (kartrider_vehicle_engine.py:1704-1749).
#[derive(Debug, Clone, PartialEq)]
pub enum AutonomousResetOutcome {
    /// World selector returned `None`; RESET ownership is retained.
    NotReady(Box<AutonomousResetNotReadyOutcome>),
    /// The respawn replacement ran (selected or gate-blocked).
    Applied(Box<AutonomousResetAppliedOutcome>),
}

/// `poll_autonomous_checkpoint_target` event
/// (kartrider_vehicle_engine.py:1496-1557): the selector result plus the
/// committed checkpoint fields and the marker provenance keys.
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointTargetEvent {
    pub selector: CheckpointTargetSelectorResult,
    pub checkpoint_mode_family: String,
    pub checkpoint_target: [f64; 3],
    pub checkpoint_gate: bool,
    pub checkpoint_gate_armed_tick_ms: Option<i64>,
    pub marker_candidate: CheckpointTargetCandidate,
    pub producer_authority: &'static str,
    pub identity_status: &'static str,
}

/// `apply_fall_clamp_reset` result when selected
/// (kartrider_vehicle_engine.py:1559-1593).
#[derive(Debug, Clone, PartialEq)]
pub struct FallClampResetOutcome {
    pub replacement: FallClampStateReplacement,
    pub topology_transition: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
}

/// `apply_manual_restart_request` result (kartrider_vehicle_engine.py:1595-1635).
#[derive(Debug, Clone, PartialEq)]
pub struct ManualRestartOutcome {
    pub selected: bool,
    pub edge: &'static str,
    pub checkpoint_target: [f64; 3],
    pub checkpoint_gate_after: bool,
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub topology_transition: TopologyDecision,
    pub topology_after: TopologyState,
    pub producer_authority: &'static str,
}

/// Python `raw_topology_contract` dict: six authority strings.
#[derive(Debug, Clone, PartialEq)]
pub struct RawTopologyContract {
    pub control_authority: String,
    pub geometry_authority: String,
    pub selector_authority: String,
    pub scheduler_authority: String,
    pub landing_transfer_authority: String,
    pub takeoff_transfer_authority: String,
}

/// `_enter_authored_reset_surface` result (kartrider_vehicle_engine.py:6194-6226).
#[derive(Debug, Clone, PartialEq)]
pub struct AuthoredResetSurfaceOutcome {
    pub reset_surface_entry: bool,
    pub reset_surface_token: &'static str,
    pub support_observation: SupportObservation,
    pub topology_decision: TopologyDecision,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub raw_topology_contract: RawTopologyContract,
}

/// Python `airborne_producer_contract` dict
/// (kartrider_vehicle_engine.py:6102-6116).
#[derive(Debug, Clone, PartialEq)]
pub struct AirborneProducerContract {
    pub control_authority: &'static str,
    pub support_authority: &'static str,
    pub contact_slots: &'static str,
    pub common_gravity: &'static str,
    pub vehicle_producers: &'static str,
}

/// `step_raw_airborne` result (kartrider_vehicle_engine.py:6030-6119).
#[derive(Debug, Clone, PartialEq)]
pub struct RawAirborneOutcome {
    pub ground: GroundStepOutcome,
    pub airborne_producer_contract: AirborneProducerContract,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
}

/// `step_airborne` result (kartrider_vehicle_engine.py:1751-1796): the
/// evaluator composition plus the committed state/topology.
#[derive(Debug, Clone, PartialEq)]
pub struct AirborneStepOutcome {
    pub evaluator: AirborneStep,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
}

/// Which executor produced a `step_raw_topology` result.
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyExecutorOutcome {
    /// GND / GND-SLIP / LAND ground-writer result.
    Ground(Box<GroundStepOutcome>),
    /// AIR-ENTRY / AIR producer result.
    Airborne(Box<RawAirborneOutcome>),
    /// Authored 리셋 surface entry (no continuous integration).
    ResetSurface(Box<AuthoredResetSurfaceOutcome>),
    /// Initial RAIL capture (state-1 ground-writer or `step_raw_rail` path).
    RailCapture(Box<RailCaptureExecution>),
    /// Initial MOVING-CONTACT capture
    /// (`acquire_and_step_autonomous_moving_contact`).
    MovingContactCapture(Box<AcquireAndStepMovingContactOutcome>),
    /// Network-correction preemption
    /// (`begin_and_step_autonomous_network_correction`,
    /// kartrider_vehicle_engine.py:6294-6312).
    NetworkCorrection(Box<NetworkCorrectionPreemption>),
}

/// `step_raw_topology` result (kartrider_vehicle_engine.py:6228-6917).
#[derive(Debug, Clone, PartialEq)]
pub struct RawTopologyOutcome {
    pub executor: TopologyExecutorOutcome,
    pub support_observation: SupportObservation,
    pub topology_decision: TopologyDecision,
    /// Python `rail_captured` key.
    pub rail_captured: bool,
    /// Python `post_ground_rail_capture` key (present only after the
    /// shared-epoch ground-sink commit).
    pub post_ground_rail_capture: Option<PostGroundRailCapture>,
    /// Python `rail_state1_ground_writer` key (present on both capture
    /// paths).
    pub rail_state1_ground_writer: Option<RailState1GroundWriter>,
    /// Python `autonomous_rail_selector` key (the deep-copied selector).
    pub autonomous_rail_selector: Option<BTreeMap<String, JsonValue>>,
    pub takeoff_state_transfer: Option<TakeoffLandingTransfer>,
    pub landing_state_transfer: Option<TakeoffLandingTransfer>,
    pub authored_special_trigger: Option<AuthoredTriggerEdge>,
    pub topology_after: TopologyState,
    pub state_after: VehicleState,
    pub raw_topology_contract: RawTopologyContract,
}

/// The dispatched body of one `_step_raw_motion_once` result.
#[derive(Debug, Clone, PartialEq)]
pub enum RawMotionKind {
    Topology(Box<RawTopologyOutcome>),
    Rail(Box<AutonomousRailOutcome>),
    MovingContact(Box<AutonomousMovingContactOutcome>),
    Reset(Box<AutonomousResetOutcome>),
    FallClampReset(Box<FallClampResetOutcome>),
    ManualRestart(Box<ManualRestartOutcome>),
    /// NET-CORRECTION domain-owner steady stepping
    /// (`step_autonomous_network_correction`, :7134-7135).
    NetworkCorrection(Box<AutonomousNetworkCorrectionOutcome>),
    /// Authoritative preemption from RAIL / MOVING-CONTACT / WATER
    /// (`begin_and_step_autonomous_network_correction`, :7093-7111).
    NetworkCorrectionPreemption(Box<AutonomousCorrectionEntry>),
}

/// Python's `boost_gauge_grant_transition` has a compact fail-safe row when
/// ITEM_USE rises in the same substep and a full evaluated row otherwise.
#[derive(Debug, Clone, PartialEq)]
pub enum BoostGaugeGrantTransition {
    SameSubstepItemUseUnadjudicated {
        accum_b10_after: f64,
        slots_after: Vec<InventorySlot>,
    },
    Evaluated(CurrentFamilyItem6Grant),
}

/// `step_raw_motion` result (kartrider_vehicle_engine.py:6919-6963).
#[derive(Debug, Clone, PartialEq)]
pub struct RawMotionOutcome {
    pub state_before: VehicleState,
    pub state_after: VehicleState,
    pub topology_after: TopologyState,
    pub raw_motion_executor: String,
    pub raw_motion_dispatch_reason: String,
    pub checkpoint_target_events: Vec<CheckpointTargetEvent>,
    pub item_use_transition: Option<CurrentItem6InventoryUseResult>,
    pub boost_gauge_grant_transition: Option<BoostGaugeGrantTransition>,
    /// Python `banana_effect_transition`: the accepted world candidate for
    /// the substep that latched the B9C effect, otherwise `None`.
    pub banana_effect_transition: Option<crate::world::BananaTrigger>,
    /// Python `banana_effect_remaining_ms`, published after this substep's
    /// actual elapsed-ms countdown.
    pub banana_effect_remaining_ms: i64,
    pub kind: RawMotionKind,
}

impl RawMotionOutcome {
    /// Python `result.get("network_correction_acquired")`
    /// (kartrider_vehicle_engine.py:6938): set by the
    /// `begin_and_step_autonomous_network_correction` wrapper (:4600) on
    /// every preemption/entry result.
    pub fn network_correction_acquired(&self) -> bool {
        match &self.kind {
            RawMotionKind::NetworkCorrectionPreemption(_) => true,
            RawMotionKind::Topology(outcome) => matches!(
                &outcome.executor,
                TopologyExecutorOutcome::NetworkCorrection(_)
            ),
            _ => false,
        }
    }

    /// Python `result.get("history_committed_by_rollback") is True`
    /// (kartrider_vehicle_engine.py:6935): the rollback path splices the
    /// prediction history itself, so `step_raw_motion` must not append.
    pub fn history_committed_by_rollback(&self) -> bool {
        match &self.kind {
            RawMotionKind::NetworkCorrectionPreemption(entry) => {
                matches!(entry.as_ref(), AutonomousCorrectionEntry::Rollback(_))
            }
            RawMotionKind::Topology(outcome) => match &outcome.executor {
                TopologyExecutorOutcome::NetworkCorrection(preemption) => {
                    matches!(preemption.entry, AutonomousCorrectionEntry::Rollback(_))
                }
                _ => false,
            },
            _ => false,
        }
    }
}

/// `step_raw_frame` result (kartrider_vehicle_engine.py:6965-7036).
#[derive(Debug, Clone, PartialEq)]
pub struct RawFrameOutcome {
    pub frame_scheduler: FrameSchedule,
    pub physics_substep_count: usize,
    pub physics_substeps: Vec<RawMotionOutcome>,
    pub clock_initialized: bool,
    pub state_after: VehicleState,
}

/// `inspect_autonomous_network_correction` result
/// (kartrider_vehicle_engine.py:4472-4558). `record` is the merged candidate
/// mapping (`{**record, "correction_active": ..., "authority": ...}` plus
/// `sequence_decision` / `rejection_reason` when present); it is `None` for
/// the early inactive rows that carry no world record (replay suppression,
/// missing capability, empty query).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionInspection {
    pub correction_active: bool,
    pub authority: &'static str,
    pub record: Option<BTreeMap<String, JsonValue>>,
    pub sequence_decision: Option<NetworkSequenceDecision>,
    pub rejection_reason: Option<&'static str>,
}

/// Typed extraction of one active network-correction record, consumed by
/// `begin_and_step_autonomous_network_correction` and
/// `rollback_and_resimulate_network_correction` (the Python code reads the
/// same keys off the merged candidate/record dict). `raw` keeps the merged
/// mapping for the `network_correction_selector` echo (:4602).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionRecord {
    pub correction_active: bool,
    pub authoritative_position: [f64; 3],
    pub authoritative_velocity: [f64; 3],
    pub authoritative_matrix: [f64; 9],
    pub authoritative_angular_velocity: [f64; 3],
    pub mode: String,
    pub duration_ms: f64,
    /// Python `int | None`; validated against the 32-bit window by
    /// `evaluate_network_sequence` (range errors surface with the Python
    /// message, so this stays `i64` rather than `u32`).
    pub sequence: Option<i64>,
    pub authoritative_time_ms: Option<f64>,
    pub raw: BTreeMap<String, JsonValue>,
}

/// `step_network_correction` resume selectors
/// (kartrider_vehicle_engine.py:4744-4750 keyword arguments; the derived
/// `Default` matches the Python defaults).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NetworkCorrectionResume {
    pub support_contact_count: i64,
    pub surface_mode: i64,
    pub rail_state: i64,
    /// `resume_moving_object_state`: the open world record mapping (Python
    /// `dict | None`).
    pub moving_object_state: Option<BTreeMap<String, JsonValue>>,
    pub moving_contact_observed: bool,
    pub distinct_water_volume: bool,
    pub reset_requested: bool,
}

/// `begin_network_correction` result (kartrider_vehicle_engine.py:4403-4470).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionEntryOutcome {
    pub correction: NetworkCorrectionState,
    pub topology_transition: TopologyDecision,
    pub state_after: VehicleState,
    /// Python constant `predicted_and_authoritative_rigid_state_captured`.
    pub entry_transfer: &'static str,
    pub sequence_decision: NetworkSequenceDecision,
}

/// Python `topology["restored_constraint_snapshot"]`
/// (kartrider_vehicle_engine.py:4896-4903).
#[derive(Debug, Clone, PartialEq)]
pub struct RestoredConstraintSnapshot {
    pub domain: MotionDomain,
    pub return_domain: Option<MotionDomain>,
    pub constraint_owner_id: Option<String>,
    pub rail_state: i64,
}

/// Python `network_correction_contract` (kartrider_vehicle_engine.py
/// :4919-4935); `target_authority` / `resume_selector_authority` are
/// overwritten by the `begin_and_step` / autonomous wrappers (:4603, :4980,
/// :5063).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionContract {
    pub target_authority: String,
    pub resume_selector_authority: String,
    pub moving_constraint_resume_authority: String,
    pub preserved_local_state: &'static str,
    pub open_boundary: &'static str,
}

/// `step_network_correction` result (kartrider_vehicle_engine.py:4740-4936).
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionStepOutcome {
    /// The leaf step result: Python keys `state_after` (rigid),
    /// `correction_after`, `alpha`, `complete`, `mode`, `sequence`,
    /// `closure_status`.
    pub step: NetworkCorrectionStepResult,
    /// The rebased MOVING-CONTACT anchor (`acquire_moving_contact_frame`
    /// against the current object context), present only when a completing
    /// correction resumes MOVING-CONTACT ownership.
    pub rebased_moving_anchor: Option<MovingContactAnchor>,
    /// Python `topology_transition` (None while the correction is in
    /// progress).
    pub topology_transition: Option<TopologyDecision>,
    /// Python `topology["restored_constraint_snapshot"]`.
    pub restored_constraint_snapshot: Option<RestoredConstraintSnapshot>,
    pub topology_after: TopologyState,
    /// Python `engine_state_after`.
    pub engine_state_after: VehicleState,
    pub network_correction_contract: NetworkCorrectionContract,
}

/// Python `autonomous_resume_selectors` (kartrider_vehicle_engine.py
/// :4972-4979 deferred row, :5054-5062 converged row). The deferred row
/// carries no `reset_surface` key (`None` here).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousResumeSelectors {
    pub candidate_position: [f64; 3],
    pub support: Option<SupportObservation>,
    pub moving_contact: Option<BTreeMap<String, JsonValue>>,
    pub rail_contact: Option<BTreeMap<String, JsonValue>>,
    pub distinct_water_volume: Option<JsonValue>,
    pub reset_surface: Option<bool>,
    pub authority: &'static str,
}

/// `step_autonomous_network_correction` result
/// (kartrider_vehicle_engine.py:4938-5066); the `begin_and_step` wrapper
/// adds `network_correction_acquired`, `network_correction_entry` and
/// `network_correction_selector` (:4600-4602).
#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousNetworkCorrectionOutcome {
    pub step: NetworkCorrectionStepOutcome,
    pub autonomous_resume_selectors: AutonomousResumeSelectors,
    pub network_correction_acquired: bool,
    /// Python `network_correction_entry`.
    pub entry: Option<Box<NetworkCorrectionEntryOutcome>>,
    /// Python `network_correction_selector` (the merged record echo).
    pub selector: Option<BTreeMap<String, JsonValue>>,
}

/// `rollback_and_resimulate_network_correction` result
/// (kartrider_vehicle_engine.py:4608-4738). The Python constant flags
/// (`network_correction_acquired` / `rollback_resimulated` /
/// `history_committed_by_rollback` / `complete`, all true) are implied by the
/// type. The `network_correction_contract` string set is constant
/// (:4721-4737) and re-embedded by the parity projections.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionRollbackOutcome {
    pub authoritative_time_ms: f64,
    pub replayed_substep_count: usize,
    pub sequence_decision: NetworkSequenceDecision,
    pub replay_results: Vec<RawMotionOutcome>,
    pub state_after: VehicleState,
    pub topology_after: TopologyState,
}

/// `begin_and_step_autonomous_network_correction` result
/// (kartrider_vehicle_engine.py:4560-4606): the forward path (begin + first
/// substep) or the rollback path (past `authoritative_time_ms`).
#[derive(Debug, Clone, PartialEq)]
pub enum AutonomousCorrectionEntry {
    Forward(Box<AutonomousNetworkCorrectionOutcome>),
    Rollback(Box<NetworkCorrectionRollbackOutcome>),
}

/// The `step_raw_topology` network-correction preemption result payload
/// (kartrider_vehicle_engine.py:6294-6312): the `begin_and_step` result plus
/// `entry_support_observation` and the preempted contract strings.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkCorrectionPreemption {
    pub entry: AutonomousCorrectionEntry,
    /// Python `entry_support_observation`.
    pub entry_support_observation: SupportObservation,
    /// Python `raw_topology_contract` with the preempted authority strings.
    pub raw_topology_contract: RawTopologyContract,
}

/// Capability probe for an optional world reference: Python's
/// `hasattr(None, ...)` is always False, so `None` reports all-absent.
fn world_caps(world: Option<&dyn World>) -> crate::world::WorldCapabilities {
    world.map_or_else(crate::world::WorldCapabilities::default, |world| {
        world.capabilities()
    })
}

/// Python `str(selector["path_id"]) if selector.get("path_id") is not None
/// else None` — the rail owner id carried into `TopologyObservation`.
fn selector_path_id(selector: &BTreeMap<String, JsonValue>) -> Option<String> {
    match selector.get("path_id") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::String(value)) => Some(value.clone()),
        Some(JsonValue::Number(value)) => {
            if value.is_finite() && value.fract() == 0.0 && value.abs() < 9.0e15 {
                Some(format!("{}", *value as i64))
            } else {
                Some(format!("{value:?}"))
            }
        }
        Some(JsonValue::Bool(value)) => Some(if *value { "True" } else { "False" }.to_string()),
        Some(other) => Some(format!("{other:?}")),
    }
}

/// Python `float(value)` coercion for selector payload numbers (used after
/// the explicit length check; missing/non-numeric entries cannot reach the
/// consumers in validated payloads).
fn number_from_json(value: Option<&JsonValue>) -> f64 {
    match value {
        Some(JsonValue::Number(value)) => *value,
        Some(JsonValue::Bool(value)) => f64::from(*value),
        _ => 0.0,
    }
}

/// Python truthiness for the open-ended selector payload values
/// (`bool(selector.get("contact_active", True))` and friends).
fn py_truthy(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => false,
        JsonValue::Bool(value) => *value,
        JsonValue::Number(value) => *value != 0.0,
        JsonValue::String(value) => !value.is_empty(),
        JsonValue::Array(values) => !values.is_empty(),
        JsonValue::Object(entries) => !entries.is_empty(),
    }
}

/// `state.last_network_sequence` as the `last_accepted` argument of
/// `evaluate_network_sequence` (Python `int | None`).
fn last_network_sequence_json(value: Option<u32>) -> JsonValue {
    value.map_or(JsonValue::Null, |value| JsonValue::Number(f64::from(value)))
}

/// The Python `sequence_decision` dict merged into the inspection candidate
/// (kartrider_vehicle_engine.py:4507).
fn sequence_decision_json(decision: &NetworkSequenceDecision) -> JsonValue {
    let int_or_null =
        |value: Option<i64>| value.map_or(JsonValue::Null, |value| JsonValue::Number(value as f64));
    JsonValue::Object(BTreeMap::from([
        ("accepted".to_string(), JsonValue::Bool(decision.accepted)),
        (
            "reason".to_string(),
            JsonValue::String(decision.reason.to_string()),
        ),
        ("candidate".to_string(), int_or_null(decision.candidate)),
        (
            "last_accepted_before".to_string(),
            int_or_null(decision.last_accepted_before),
        ),
        (
            "last_accepted_after".to_string(),
            int_or_null(decision.last_accepted_after),
        ),
        (
            "forward_distance".to_string(),
            int_or_null(decision.forward_distance),
        ),
        (
            "estimated_packets_lost".to_string(),
            JsonValue::Number(decision.estimated_packets_lost as f64),
        ),
        (
            "ordered_authority".to_string(),
            JsonValue::Bool(decision.ordered_authority),
        ),
    ]))
}

/// One rigid vector field of a network-correction record, with the Python
/// `_validate_state("authoritative_state", ...)` length message
/// (evaluate_network_correction_topology.py:84-86).
fn network_rigid_vec3(record: &BTreeMap<String, JsonValue>, key: &str) -> Result<[f64; 3], String> {
    let field = key.strip_prefix("authoritative_").unwrap_or(key);
    match record.get(key) {
        Some(JsonValue::Array(items)) if items.len() == 3 => {
            let mut result = [0.0; 3];
            for (index, item) in items.iter().enumerate() {
                result[index] = json_value_f64(item)?;
            }
            Ok(result)
        }
        _ => Err(format!(
            "authoritative_state.{field} must have three components"
        )),
    }
}

/// The rigid matrix field of a network-correction record
/// (`authoritative_state.matrix must have nine components`,
/// evaluate_network_correction_topology.py:87-88). Moving-object contexts
/// reuse the same helper; their Python failure mode is a `TypeError` the
/// parity harness never drives.
fn network_rigid_mat9(record: &BTreeMap<String, JsonValue>, key: &str) -> Result<[f64; 9], String> {
    let field = key.strip_prefix("authoritative_").unwrap_or(key);
    match record.get(key) {
        Some(JsonValue::Array(items)) if items.len() == 9 => {
            let mut result = [0.0; 9];
            for (index, item) in items.iter().enumerate() {
                result[index] = json_value_f64(item)?;
            }
            Ok(result)
        }
        _ => Err(format!(
            "authoritative_state.{field} must have nine components"
        )),
    }
}

/// `candidate["mode"]` (post required-fields check): Python raises a
/// `TypeError` inside `begin_network_correction`'s set membership test for
/// non-string modes; the leaf's mode validation covers every string value,
/// so only the string shape is decoded here.
fn network_record_mode(record: &BTreeMap<String, JsonValue>) -> Result<&str, String> {
    match record.get("mode") {
        Some(JsonValue::String(mode)) => Ok(mode),
        _ => Err("network correction mode must be snap or blend".to_string()),
    }
}

/// `record.get("sequence")` as Python `int | None` (`i64` so out-of-window
/// values still reach `evaluate_network_sequence`'s range error).
fn network_record_sequence(record: &BTreeMap<String, JsonValue>) -> Result<Option<i64>, String> {
    match record.get("sequence") {
        None | Some(JsonValue::Null) => Ok(None),
        Some(JsonValue::Number(value))
            if value.is_finite() && value.fract() == 0.0 && value.abs() < 9.0e15 =>
        {
            Ok(Some(*value as i64))
        }
        _ => Err("network sequence must be an integer or None".to_string()),
    }
}

/// `context["object_id"]` for the MOVING-CONTACT rebase (post
/// required-fields check; Python `acquire_moving_contact_frame` rejects a
/// missing/empty id with `object_id is required`).
fn network_record_object_id(record: &BTreeMap<String, JsonValue>) -> Result<&str, String> {
    match record.get("object_id") {
        Some(JsonValue::String(object_id)) => Ok(object_id),
        _ => Err("object_id is required".to_string()),
    }
}

impl NetworkCorrectionRecord {
    /// Build the typed record from the open world/caller mapping (the same
    /// keys Python reads in `begin_and_step_autonomous_network_correction`
    /// and `rollback_and_resimulate_network_correction`). The authoritative
    /// fields decode with the `_validate_state` length messages when
    /// `strict_shapes` is set (the inspection path validates the complete
    /// transaction before this runs, so decoding cannot fail there).
    pub fn from_record(record: &BTreeMap<String, JsonValue>) -> Result<Self, String> {
        let authoritative_time_ms = match record.get("authoritative_time_ms") {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::Number(value)) if value.is_finite() => Some(*value),
            _ => return Err("authoritative_time_ms must be a finite numeric boundary".to_string()),
        };
        Ok(NetworkCorrectionRecord {
            correction_active: record.get("correction_active").is_some_and(py_truthy),
            authoritative_position: network_rigid_vec3(record, "authoritative_position")?,
            authoritative_velocity: network_rigid_vec3(record, "authoritative_velocity")?,
            authoritative_matrix: network_rigid_mat9(record, "authoritative_matrix")?,
            authoritative_angular_velocity: network_rigid_vec3(
                record,
                "authoritative_angular_velocity",
            )?,
            mode: network_record_mode(record)?.to_string(),
            duration_ms: record.get("duration_ms").map_or(Ok(0.0), json_value_f64)?,
            sequence: network_record_sequence(record)?,
            authoritative_time_ms,
            raw: record.clone(),
        })
    }

    /// Build the typed record from a merged inspection candidate
    /// (kartrider_vehicle_engine.py:7102, :6297 pass the candidate through).
    pub fn from_inspection(inspection: &NetworkCorrectionInspection) -> Result<Self, String> {
        let record = inspection
            .record
            .as_ref()
            .ok_or_else(|| "active network-correction inspection requires a record".to_string())?;
        let mut typed = Self::from_record(record)?;
        typed.correction_active = inspection.correction_active;
        Ok(typed)
    }
}

/// The Python merged moving-contact inspection mapping: `{**record,
/// "contact_active": contact_active, "authority":
/// "world_query_moving_contact"}` (kartrider_vehicle_engine.py:4144-4148).
/// The engine consumes this merged shape in `acquire_and_step_*` and
/// `step_autonomous_moving_contact` (selector record + `object_state_after`).
fn merged_moving_contact_record(
    inspection: &MovingContactInspection,
) -> Result<BTreeMap<String, JsonValue>, String> {
    let mut merged = inspection
        .record
        .clone()
        .ok_or_else(|| "active moving-contact inspection requires a record".to_string())?;
    merged.insert(
        "contact_active".to_string(),
        JsonValue::Bool(inspection.contact_active),
    );
    merged.insert(
        "authority".to_string(),
        JsonValue::String(inspection.authority.to_string()),
    );
    Ok(merged)
}

/// Python `record["object_id"]` (a validated nonempty string).
fn moving_record_string(record: &BTreeMap<String, JsonValue>, key: &str) -> Result<String, String> {
    match record.get(key) {
        Some(JsonValue::String(value)) => Ok(value.clone()),
        _ => Err(format!(
            "moving-contact record field {key} must be a string"
        )),
    }
}

fn json_array_numbers(value: &JsonValue, key: &str, len: usize) -> Result<Vec<f64>, String> {
    match value {
        JsonValue::Array(items) if items.len() == len => items
            .iter()
            .map(|item| match item {
                JsonValue::Number(number) => Ok(*number),
                _ => Err(format!(
                    "moving-contact record field {key} must be a {len}-component array"
                )),
            })
            .collect(),
        _ => Err(format!(
            "moving-contact record field {key} must be a {len}-component array"
        )),
    }
}

/// Python `record[key]` for a validated 3-component vector field.
fn moving_record_vec3(record: &BTreeMap<String, JsonValue>, key: &str) -> Result<[f64; 3], String> {
    let value = record
        .get(key)
        .ok_or_else(|| format!("moving-contact record missing field {key}"))?;
    let numbers = json_array_numbers(value, key, 3)?;
    Ok([numbers[0], numbers[1], numbers[2]])
}

/// Python `record[key]` for a validated 9-component matrix field.
fn moving_record_mat9(record: &BTreeMap<String, JsonValue>, key: &str) -> Result<[f64; 9], String> {
    let value = record
        .get(key)
        .ok_or_else(|| format!("moving-contact record missing field {key}"))?;
    let numbers = json_array_numbers(value, key, 9)?;
    let mut out = [0.0; 9];
    out.copy_from_slice(&numbers);
    Ok(out)
}

/// Python `step_record.get("position", record["object_position"])`.
fn moving_record_vec3_fallback(
    record: &BTreeMap<String, JsonValue>,
    key: &str,
    fallback: [f64; 3],
) -> Result<[f64; 3], String> {
    match record.get(key) {
        Some(value) => {
            let numbers = json_array_numbers(value, key, 3)?;
            Ok([numbers[0], numbers[1], numbers[2]])
        }
        None => Ok(fallback),
    }
}

/// Python `step_record.get("matrix", record["object_matrix"])`.
fn moving_record_mat9_fallback(
    record: &BTreeMap<String, JsonValue>,
    key: &str,
    fallback: [f64; 9],
) -> Result<[f64; 9], String> {
    match record.get(key) {
        Some(value) => {
            let numbers = json_array_numbers(value, key, 9)?;
            let mut out = [0.0; 9];
            out.copy_from_slice(&numbers);
            Ok(out)
        }
        None => Ok(fallback),
    }
}

/// Python `str(value)` for the scalar JSON payloads the rail engine coerces
/// (`str(candidate.get("road_rail") or "")`, `str(trigger["road_surface"])`).
/// `None` maps to `"None"` like CPython; callers apply the `or ""` fallback
/// before calling this where Python does.
fn json_py_str(value: &JsonValue) -> String {
    match value {
        JsonValue::Null => "None".to_string(),
        JsonValue::Bool(value) => if *value { "True" } else { "False" }.to_string(),
        JsonValue::Number(value) => {
            if value.is_finite() && value.fract() == 0.0 && value.abs() < 9.0e15 {
                format!("{}", *value as i64)
            } else {
                format!("{value:?}")
            }
        }
        JsonValue::String(value) => value.clone(),
        other => format!("{other:?}"),
    }
}

/// Python `float(selector.get(key, default))`.
fn number_or(value: Option<&JsonValue>, default: f64) -> f64 {
    match value {
        None | Some(JsonValue::Null) => default,
        some => number_from_json(some),
    }
}

/// Python `int(selector.get(key, 0))` (truncation toward zero).
fn json_int_or(value: Option<&JsonValue>) -> i64 {
    number_or(value, 0.0) as i64
}

/// Python `str(x or "")` for token reads on open payloads.
fn json_token_or_empty(value: Option<&JsonValue>) -> String {
    match value {
        None | Some(JsonValue::Null) => String::new(),
        Some(v) if !py_truthy(v) => String::new(),
        Some(v) => json_py_str(v),
    }
}

/// `RailPathContextId` as the JSON id carried in selector payloads
/// (`path_context_id` round-trips ints and strings; a non-integral number
/// falls back to its Python `str` form, unreachable in validated payloads).
fn path_context_id_to_json(id: &RailPathContextId) -> JsonValue {
    match id {
        RailPathContextId::Index(index) => JsonValue::Number(*index as f64),
        RailPathContextId::Name(name) => JsonValue::String(name.clone()),
    }
}

/// JSON id -> `RailPathContextId` (the `rail_selector_runtime` typed latch).
fn rail_path_context_id_from_json(value: &JsonValue) -> RailPathContextId {
    match value {
        JsonValue::Number(number)
            if number.is_finite() && number.fract() == 0.0 && number.abs() < 9.0e15 =>
        {
            RailPathContextId::Index(*number as i64)
        }
        JsonValue::String(name) => RailPathContextId::Name(name.clone()),
        other => RailPathContextId::Name(json_py_str(other)),
    }
}

/// Decode the `registered_query_contexts` selector payload into the typed
/// [`RailState1Context`] list consumed by `evaluate_rail_state1_query`.
/// Mirrors the Python engine's duck-typed resolution (`_points` in
/// scripts/evaluate_rail_state1_query.py:27-52 and the context loop at
/// :149-170): `position`/`secondary`/`normal` fall back to
/// `primary`/`direction`/`normal_or_side`. Point vectors that are missing or
/// malformed decode to zeros — Python would raise a `TypeError` downstream
/// on such payloads, which the parity harness never exercises.
fn rail_state1_contexts_from_json(value: &JsonValue) -> Result<Vec<RailState1Context>, String> {
    let JsonValue::Array(contexts) = value else {
        return Err("contexts must be a list".to_string());
    };
    let decode3 = |value: Option<&JsonValue>| -> [f64; 3] {
        match value {
            Some(JsonValue::Array(values)) if values.len() == 3 => [
                number_from_json(values.first()),
                number_from_json(values.get(1)),
                number_from_json(values.get(2)),
            ],
            _ => [0.0, 0.0, 0.0],
        }
    };
    let mut decoded = Vec::with_capacity(contexts.len());
    for (index, context) in contexts.iter().enumerate() {
        let JsonValue::Object(map) = context else {
            return Err(format!("context {index} must be a mapping"));
        };
        let points_value = match map.get("points") {
            Some(value) => Some(value),
            None => map.get("records"),
        };
        let Some(JsonValue::Array(points)) = points_value else {
            return Err(format!("context {index} requires points or records"));
        };
        let mut decoded_points = Vec::with_capacity(points.len());
        for (point_index, point) in points.iter().enumerate() {
            let JsonValue::Object(point) = point else {
                return Err(format!(
                    "context {index} point {point_index} must be a mapping"
                ));
            };
            decoded_points.push(RailState1Point {
                position: decode3(point.get("position").or_else(|| point.get("primary"))),
                secondary: decode3(point.get("secondary").or_else(|| point.get("direction"))),
                normal: decode3(point.get("normal").or_else(|| point.get("normal_or_side"))),
            });
        }
        let successor_ids = match map
            .get("successor_ids")
            .or_else(|| map.get("successor_indices"))
        {
            None | Some(JsonValue::Null) => None,
            Some(JsonValue::Array(values)) => Some(values.clone()),
            Some(_) => return Err("rail successor ids must be a list".to_string()),
        };
        decoded.push(RailState1Context {
            context_id: map.get("context_id").cloned(),
            context_index: match map.get("context_index") {
                Some(JsonValue::Number(value)) => Some(*value as i64),
                _ => None,
            },
            successor_ids,
            road: match map.get("road") {
                Some(JsonValue::String(value)) => Some(value.clone()),
                _ => None,
            },
            total_length: match map.get("total_length") {
                Some(JsonValue::Number(value)) => Some(*value),
                _ => None,
            },
            points: decoded_points,
        });
    }
    Ok(decoded)
}

/// [`RailState1GateView`] from the open JSON `state1_gate` mapping.
fn rail_state1_gate_view_from_json(value: &JsonValue) -> Result<RailState1GateView, String> {
    let JsonValue::Object(map) = value else {
        return Err("state1_gate must be a mapping".to_string());
    };
    let json_number_opt = |key: &str| -> Option<f64> {
        match map.get(key) {
            Some(JsonValue::Number(value)) => Some(*value),
            Some(JsonValue::Bool(value)) => Some(f64::from(*value)),
            _ => None,
        }
    };
    let lookahead_endpoint = match map.get("lookahead_endpoint") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Array(values)) => Some(
            values
                .iter()
                .map(|value| number_from_json(Some(value)))
                .collect(),
        ),
        // Malformed value: force the downstream
        // "requires the autonomous one-metre lookahead endpoint" ValueError.
        Some(_) => Some(Vec::new()),
    };
    let rail_base_orientation_matrix_bf8 = match map.get("rail_base_orientation") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Object(base)) => match base.get("matrix_bf8") {
            Some(JsonValue::Array(values)) if values.len() == 9 => Some(Some([
                number_from_json(values.first()),
                number_from_json(values.get(1)),
                number_from_json(values.get(2)),
                number_from_json(values.get(3)),
                number_from_json(values.get(4)),
                number_from_json(values.get(5)),
                number_from_json(values.get(6)),
                number_from_json(values.get(7)),
                number_from_json(values.get(8)),
            ])),
            _ => Some(None),
        },
        Some(_) => Some(None),
    };
    Ok(RailState1GateView {
        nonempty: !map.is_empty(),
        authority: map.get("authority").map(json_py_str),
        contact_exists: map.get("contact_exists").is_some_and(py_truthy),
        contact_identifier_nonempty: map
            .get("contact_identifier_nonempty")
            .is_some_and(py_truthy),
        path_metric: json_number_opt("path_metric"),
        metric_threshold: json_number_opt("metric_threshold"),
        path_z: json_number_opt("path_z"),
        kart_z_plus_one: json_number_opt("kart_z_plus_one"),
        alignment_base_ms: json_number_opt("alignment_base_ms"),
        alignment_primary_scale_ms: json_number_opt("alignment_primary_scale_ms"),
        alignment_stable_initial_ms: json_number_opt("alignment_stable_initial_ms"),
        selected_context: match map.get("selected_context") {
            None | Some(JsonValue::Null) => None,
            Some(value) => Some(value.clone()),
        },
        query_scalar: json_number_opt("query_scalar"),
        lookahead_distance: json_number_opt("lookahead_distance"),
        lookahead_endpoint,
        advance_context: match map.get("advance_context") {
            None | Some(JsonValue::Null) => None,
            Some(value) => Some(value.clone()),
        },
        terminal_state6_edge: map.get("terminal_state6_edge").is_some_and(py_truthy),
        terminal_distance_remaining: json_number_opt("terminal_distance_remaining"),
        rail_base_orientation_matrix_bf8,
    })
}

/// [`RailState1GateView`] from the typed `evaluate_rail_state1_query` gate.
fn rail_state1_gate_view_from_query(gate: &RailState1Gate) -> RailState1GateView {
    RailState1GateView {
        nonempty: true,
        authority: Some(gate.authority.to_string()),
        contact_exists: gate.contact_exists,
        contact_identifier_nonempty: gate.contact_identifier_nonempty,
        path_metric: Some(gate.path_metric),
        metric_threshold: Some(gate.metric_threshold),
        path_z: Some(gate.path_z),
        kart_z_plus_one: Some(gate.kart_z_plus_one),
        alignment_base_ms: Some(gate.alignment_base_ms),
        alignment_primary_scale_ms: gate.alignment_primary_scale_ms,
        alignment_stable_initial_ms: gate.alignment_stable_initial_ms,
        selected_context: Some(gate.selected_context.clone()),
        query_scalar: Some(gate.query_scalar),
        lookahead_distance: Some(gate.lookahead_distance),
        lookahead_endpoint: Some(gate.lookahead_endpoint.to_vec()),
        advance_context: gate.advance_context.clone(),
        terminal_state6_edge: gate.terminal_state6_edge,
        terminal_distance_remaining: Some(gate.terminal_distance_remaining),
        rail_base_orientation_matrix_bf8: gate
            .rail_base_orientation
            .as_ref()
            .map(|base| Some(base.matrix_bf8)),
    }
}

/// The `state1_gate` authority allow-list of `_advance_autonomous_rail_selector`
/// (kartrider_vehicle_engine.py:3364-3368).
const STATE1_GATE_AUTHORITIES: [&str; 3] = [
    "sub_A4C990_equivalent_candidate_interpolation",
    "runtime_sub_A4C990_capture",
    "autonomous_A4EA20_A4F600_A4C990_equivalent_query",
];

impl KartRiderVehicleEngine {
    /// `advance_motion_topology` (kartrider_vehicle_engine.py:1379-1390):
    /// commit one engine-owned constraint-ownership decision.
    pub fn advance_motion_topology(
        &mut self,
        observation: &TopologyObservation,
    ) -> Result<TopologyDecision, String> {
        let domain_before = self.state.topology.domain;
        let result = advance_topology(&self.state.topology, observation)?;
        self.state.topology = result.state_after.clone();
        let domain_after = self.state.topology.domain;
        if domain_before == MotionDomain::Rail && domain_after != MotionDomain::Rail {
            self.state.rail_selector_runtime = None;
            self.state.rail_base_bf8 = None;
            self.state.rail_constraint_c1c = None;
            self.state.rail_orientation_dt_latch = None;
        }
        Ok(result)
    }

    /// `apply_fall_clamp_reset` (kartrider_vehicle_engine.py:1559-1593):
    /// apply the exact BD7840 fall clamp and enter RESET ownership. Returns
    /// `None` when the clamp does not select (Python's `selected: False`
    /// early return).
    pub fn apply_fall_clamp_reset(
        &mut self,
        drop_threshold: f64,
        clamp_z: f64,
    ) -> Result<Option<FallClampResetOutcome>, String> {
        let replacement = evaluate_fall_clamp_state_replacement(
            self.state.position,
            self.state.matrix,
            self.state.velocity,
            self.state.angular_velocity,
            drop_threshold,
            clamp_z,
        );
        if !replacement.selected {
            return Ok(None);
        }
        let topology = self.advance_motion_topology(&TopologyObservation {
            reset_requested: true,
            ..Default::default()
        })?;
        self.state.position = replacement.position_90_after;
        self.state.matrix = replacement.matrix_9c_after;
        self.state.velocity = replacement.velocity_c0_after;
        self.state.angular_velocity = replacement.angular_velocity_cc_after;
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        // RESET is an authoritative replacement barrier. Any local-frame or
        // network interpolation payload from the pre-reset pose is stale at
        // the replacement pose and must not survive under RESET ownership.
        self.state.moving_contact_anchor = None;
        self.state.network_correction = None;
        self.state.contact_compression = [0.0; 4];
        self.state.validate()?;
        Ok(Some(FallClampResetOutcome {
            replacement,
            topology_transition: topology,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
        }))
    }

    /// `apply_manual_restart_request` (kartrider_vehicle_engine.py:1595-1635):
    /// enter RESET on one raw R-key rising edge using the saved target.
    pub fn apply_manual_restart_request(&mut self) -> Result<ManualRestartOutcome, String> {
        let current = self.state.topology.domain;
        if current == MotionDomain::Reset {
            return Err("manual restart cannot re-enter active RESET".to_string());
        }
        if current == MotionDomain::NetworkCorrection {
            return Err("manual restart cannot preempt an active network transaction".to_string());
        }
        let before = self.state.clone();
        let topology = self.advance_motion_topology(&TopologyObservation {
            reset_requested: true,
            ..Default::default()
        })?;
        if topology.selected_domain != MotionDomain::Reset {
            return Err("manual restart did not acquire RESET ownership".to_string());
        }
        self.state.checkpoint_gate = true;
        self.state.checkpoint_gate_armed_tick_ms = Some(self.state.time_ms as i64);
        self.state.manual_restart_count += 1;
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.contact_compression = [0.0; 4];
        self.state.takeoff_separation_pending = false;
        self.state.moving_contact_anchor = None;
        self.state.rail_selector_runtime = None;
        self.state.rail_base_bf8 = None;
        self.state.rail_constraint_c1c = None;
        self.state.rail_orientation_dt_latch = None;
        self.state.network_correction = None;
        self.state.validate()?;
        Ok(ManualRestartOutcome {
            selected: true,
            edge: "R_rising",
            checkpoint_target: self.state.checkpoint_target,
            checkpoint_gate_after: self.state.checkpoint_gate,
            state_before: before,
            state_after: self.state.clone(),
            topology_transition: topology,
            topology_after: self.state.topology.clone(),
            producer_authority: "engine_owned_raw_R_key_rising_edge",
        })
    }

    /// `apply_checkpoint_target_event(event="mission_progress", ...)` as
    /// driven by `poll_autonomous_checkpoint_target`
    /// (kartrider_vehicle_engine.py:1496-1519, :1542-1549).
    fn apply_checkpoint_progress_event(
        &mut self,
        candidate: &CheckpointTargetCandidate,
    ) -> Result<CheckpointTargetEvent, String> {
        let marker_ordinal = candidate.marker_ordinal;
        let result = evaluate_checkpoint_target_selector(
            &self.state.checkpoint_mode_family,
            self.state.checkpoint_target,
            self.state.checkpoint_gate,
            "mission_progress",
            Some(self.state.position),
            false,
            Some(if self.state.checkpoint_mode_family == "c016" {
                5
            } else {
                1
            }),
            Some(marker_ordinal + 1),
            Some(marker_ordinal),
            false,
            false,
            None,
            None,
            None,
            false,
            false,
            false,
            Some(self.state.time_ms as i64),
            None,
        )?;
        self.state.checkpoint_target = result.target_after;
        self.state.checkpoint_gate = result.gate_after;
        if result.gate_written && result.gate_after {
            self.state.checkpoint_gate_armed_tick_ms = Some(self.state.time_ms as i64);
        } else if result.gate_written && !result.gate_after {
            self.state.checkpoint_gate_armed_tick_ms = None;
        }
        Ok(CheckpointTargetEvent {
            selector: result,
            checkpoint_mode_family: self.state.checkpoint_mode_family.clone(),
            checkpoint_target: self.state.checkpoint_target,
            checkpoint_gate: self.state.checkpoint_gate,
            checkpoint_gate_armed_tick_ms: self.state.checkpoint_gate_armed_tick_ms,
            marker_candidate: candidate.clone(),
            producer_authority: "decoded_KAP_marker_proximity_edge_candidate",
            identity_status: "game_checkpoint_event_identity_unproven",
        })
    }

    /// `poll_autonomous_checkpoint_target` (kartrider_vehicle_engine.py:1521-1557):
    /// convert a KAP marker proximity edge into a candidate target event.
    pub fn poll_autonomous_checkpoint_target(
        &mut self,
        world: Option<&dyn World>,
    ) -> Result<Vec<CheckpointTargetEvent>, String> {
        if !world_caps(world).query_checkpoint_target_candidate {
            return Ok(Vec::new());
        }
        let world = world.expect("capability probe implies a world");
        let candidate = world
            .query_checkpoint_target_candidate(self.state.position, 2.0)
            .ok_or_else(|| {
                "query_checkpoint_target_candidate capability reported present \
                 but returned no result"
                    .to_string()
            })?;
        let Some(candidate) = candidate else {
            self.state.checkpoint_marker_active_id = None;
            return Ok(Vec::new());
        };
        // Python str(candidate.get("marker_id") or "").
        let marker_id = candidate.marker_id.clone();
        if marker_id.is_empty() {
            return Err("checkpoint target candidate requires marker_id".to_string());
        }
        if self.state.checkpoint_marker_active_id.as_ref() == Some(&marker_id) {
            return Ok(Vec::new());
        }
        self.state.checkpoint_marker_active_id = Some(marker_id);
        let event = self.apply_checkpoint_progress_event(&candidate)?;
        Ok(vec![event])
    }

    /// `inspect_autonomous_support` (kartrider_vehicle_engine.py:6121-6177):
    /// sample four wheel rays and classify support without mutating state.
    pub fn inspect_autonomous_support(
        &self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_planes: Option<&[WorldGroundResult]>,
    ) -> Result<SupportObservation, String> {
        let planes = match ground_planes {
            Some(planes) => planes.to_vec(),
            None => query_wheel_ground(
                world.ok_or_else(|| {
                    "autonomous topology stepping requires world.query_ground".to_string()
                })?,
                &self.state,
                parameters,
            ),
        };
        if planes.len() != 4 {
            return Err("autonomous support requires four wheel planes".to_string());
        }
        let plane_set = ground_plane_set_from_world_planes(&planes);
        let sampled =
            self.produce_ground_accumulators(keys, parameters, Some(&plane_set), None, true)?;
        if sampled.evaluator.contact.is_none() {
            // Python RuntimeError.
            return Err("wheel support probe did not schedule contact accumulator".to_string());
        }
        // Topology ownership observes B28D30's fresh contact record, while
        // B29E70 force production consumes the A54/A64 state carried into
        // this epoch. They intentionally differ for one boundary epoch.
        let next_compression = sampled.contact_compression_after;
        let active_slots = [
            next_compression[0] > 0.0,
            next_compression[1] > 0.0,
            next_compression[2] > 0.0,
            next_compression[3] > 0.0,
        ];
        let support_contact_count = active_slots.iter().filter(|active| **active).count() as i64;
        let surface_mode = sampled.surface_selection.surface_mode;
        Ok(SupportObservation {
            planes,
            active_slots,
            support_contact_count,
            surface_mode,
            producer: sampled,
            authority: "engine_owned_B28D30_wheel_rays",
        })
    }

    /// `_support_selects_authored_reset` (kartrider_vehicle_engine.py:6179-6192):
    /// true only for an active wheel slot on the authored reset token.
    pub fn support_selects_authored_reset(support: &SupportObservation) -> bool {
        let selection = &support.producer.surface_selection;
        let branch_reset = selection
            .selected_dynamics_branch
            .as_ref()
            .and_then(Option::as_deref)
            == Some("reset");
        branch_reset
            && support
                .active_slots
                .iter()
                .enumerate()
                .any(|(index, active)| {
                    *active && support.planes[index].road_surface.trim().to_lowercase() == "리셋"
                })
    }

    /// `_enter_authored_reset_surface` (kartrider_vehicle_engine.py:6194-6226):
    /// commit the discrete RESET owner without integrating continuous state.
    pub fn enter_authored_reset_surface(
        &mut self,
        support: SupportObservation,
    ) -> Result<AuthoredResetSurfaceOutcome, String> {
        if !Self::support_selects_authored_reset(&support) {
            return Err("authored reset entry requires an active 리셋 wheel contact".to_string());
        }
        let topology_decision = self.advance_motion_topology(&TopologyObservation {
            reset_requested: true,
            ..Default::default()
        })?;
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.contact_compression = [0.0; 4];
        self.state.moving_contact_anchor = None;
        self.state.network_correction = None;
        self.state.validate()?;
        Ok(AuthoredResetSurfaceOutcome {
            reset_surface_entry: true,
            reset_surface_token: "리셋",
            support_observation: support,
            topology_decision,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
            raw_topology_contract: RawTopologyContract {
                control_authority: "preempted_by_reset_transaction".to_string(),
                geometry_authority: "world_query_ground_four_wheel_rays".to_string(),
                selector_authority: "authored_reset_surface_hash_103940721".to_string(),
                scheduler_authority: "motion_topology_state_machine".to_string(),
                landing_transfer_authority: "not_selected".to_string(),
                takeoff_transfer_authority: "not_selected".to_string(),
            },
        })
    }

    /// `inspect_autonomous_network_correction`
    /// (kartrider_vehicle_engine.py:4472-4558): query one already-accepted
    /// authoritative correction packet. The world capability probe mirrors
    /// Python's `hasattr(world, "query_network_correction")` (:4479).
    pub fn inspect_autonomous_network_correction(
        &mut self,
        world: Option<&dyn World>,
    ) -> Result<NetworkCorrectionInspection, String> {
        if self.network_replay_active {
            return Ok(NetworkCorrectionInspection {
                correction_active: false,
                authority: "network_packet_query_suppressed_during_resimulation",
                record: None,
                sequence_decision: None,
                rejection_reason: None,
            });
        }
        if !world_caps(world).query_network_correction {
            return Ok(NetworkCorrectionInspection {
                correction_active: false,
                authority: "world_has_no_network_correction_selector",
                record: None,
                sequence_decision: None,
                rejection_reason: None,
            });
        }
        let record = world
            .expect("capability probe implies a world")
            .query_network_correction(self.state.position)
            .ok_or_else(|| {
                "query_network_correction capability reported present but returned no result"
                    .to_string()
            })?;
        let Some(record) = record else {
            return Ok(NetworkCorrectionInspection {
                correction_active: false,
                authority: "world_query_network_correction",
                record: None,
                sequence_decision: None,
                rejection_reason: None,
            });
        };
        // Python requires the record to be a mapping
        // (`ValueError("query_network_correction must return a mapping or
        // None")`); the trait return type enforces it.
        // candidate = {**deepcopy(record), "correction_active":
        // bool(record.get("correction_active", True)), "authority":
        // "world_query_network_correction"}.
        let correction_active = record.get("correction_active").is_none_or(py_truthy);
        if !correction_active {
            let mut candidate = record;
            candidate.insert("correction_active".to_string(), JsonValue::Bool(false));
            candidate.insert(
                "authority".to_string(),
                JsonValue::String("world_query_network_correction".to_string()),
            );
            return Ok(NetworkCorrectionInspection {
                correction_active: false,
                authority: "world_query_network_correction",
                record: Some(candidate),
                sequence_decision: None,
                rejection_reason: None,
            });
        }
        let sequence_decision = evaluate_network_sequence(
            record.get("sequence").unwrap_or(&JsonValue::Null),
            &last_network_sequence_json(self.state.last_network_sequence),
            SEQUENCE_BITS,
        )?;
        if !sequence_decision.accepted {
            self.state.network_packets_rejected += 1;
            let mut candidate = record;
            candidate.insert("correction_active".to_string(), JsonValue::Bool(false));
            candidate.insert(
                "sequence_decision".to_string(),
                sequence_decision_json(&sequence_decision),
            );
            candidate.insert(
                "rejection_reason".to_string(),
                JsonValue::String(sequence_decision.reason.to_string()),
            );
            candidate.insert(
                "authority".to_string(),
                JsonValue::String("engine_owned_modulo_sequence_acceptance_window".to_string()),
            );
            return Ok(NetworkCorrectionInspection {
                correction_active: false,
                authority: "engine_owned_modulo_sequence_acceptance_window",
                record: Some(candidate),
                rejection_reason: Some(sequence_decision.reason),
                sequence_decision: Some(sequence_decision),
            });
        }
        const REQUIRED: [&str; 5] = [
            "authoritative_angular_velocity",
            "authoritative_matrix",
            "authoritative_position",
            "authoritative_velocity",
            "mode",
        ];
        let missing: Vec<&str> = REQUIRED
            .iter()
            .copied()
            .filter(|field| !record.contains_key(*field))
            .collect();
        if !missing.is_empty() {
            // REQUIRED is pre-sorted, matching Python `sorted(missing)`.
            let rendered = missing
                .iter()
                .map(|field| format!("'{field}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "active network-correction record missing fields: [{rendered}]"
            ));
        }
        if let Some(value) = record.get("authoritative_time_ms") {
            let valid = match value {
                JsonValue::Null => true,
                JsonValue::Number(number) => number.is_finite(),
                _ => false,
            };
            if !valid {
                return Err("authoritative_time_ms must be a finite numeric boundary".to_string());
            }
        }
        // Validate the complete transaction without changing topology.
        evaluate_begin_network_correction(
            &RigidStateSnapshot {
                position: self.state.position,
                velocity: self.state.velocity,
                matrix: self.state.matrix,
                angular_velocity: self.state.angular_velocity,
            },
            &RigidStateSnapshot {
                position: network_rigid_vec3(&record, "authoritative_position")?,
                velocity: network_rigid_vec3(&record, "authoritative_velocity")?,
                matrix: network_rigid_mat9(&record, "authoritative_matrix")?,
                angular_velocity: network_rigid_vec3(&record, "authoritative_angular_velocity")?,
            },
            network_record_mode(&record)?,
            record.get("duration_ms").map_or(Ok(0.0), json_value_f64)?,
            // The accepted sequence decision above bounds the value to the
            // 32-bit window (or None), so this conversion cannot wrap.
            network_record_sequence(&record)?.map(|value| value as u32),
        )?;
        let mut candidate = record;
        candidate.insert("correction_active".to_string(), JsonValue::Bool(true));
        candidate.insert(
            "sequence_decision".to_string(),
            sequence_decision_json(&sequence_decision),
        );
        candidate.insert(
            "authority".to_string(),
            JsonValue::String("world_query_network_correction".to_string()),
        );
        Ok(NetworkCorrectionInspection {
            correction_active: true,
            authority: "world_query_network_correction",
            record: Some(candidate),
            sequence_decision: Some(sequence_decision),
            rejection_reason: None,
        })
    }

    /// `begin_network_correction` (kartrider_vehicle_engine.py:4403-4470):
    /// preempt continuous integration with an authoritative transaction.
    pub fn begin_network_correction(
        &mut self,
        authoritative_state: &RigidStateSnapshot,
        mode: &str,
        duration_ms: f64,
        sequence: Option<i64>,
    ) -> Result<NetworkCorrectionEntryOutcome, String> {
        let current_domain = self.state.topology.domain;
        if current_domain == MotionDomain::Reset {
            return Err(
                "network correction cannot preempt an incomplete RESET transaction".to_string(),
            );
        }
        if self.state.network_correction.is_some() {
            return Err("a network correction transaction is already active".to_string());
        }
        let sequence_decision = evaluate_network_sequence(
            &sequence.map_or(JsonValue::Null, |value| JsonValue::Number(value as f64)),
            &last_network_sequence_json(self.state.last_network_sequence),
            SEQUENCE_BITS,
        )?;
        if !sequence_decision.accepted {
            return Err(format!(
                "network correction packet rejected: {}",
                sequence_decision.reason
            ));
        }
        let predicted = RigidStateSnapshot {
            position: self.state.position,
            velocity: self.state.velocity,
            matrix: self.state.matrix,
            angular_velocity: self.state.angular_velocity,
        };
        // The accepted decision bounds the sequence to the 32-bit window (or
        // None), so the u32 store in the transaction cannot wrap.
        let sequence_u32 = sequence.map(|value| value as u32);
        let mut correction = evaluate_begin_network_correction(
            &predicted,
            authoritative_state,
            mode,
            duration_ms,
            sequence_u32,
        )?;
        correction.resume_topology = Some(self.state.topology.clone());
        let topology = self.advance_motion_topology(&TopologyObservation {
            network_correction_active: true,
            ..Default::default()
        })?;
        if topology.selected_domain != MotionDomain::NetworkCorrection {
            // Python RuntimeError.
            return Err("network correction did not acquire ownership".to_string());
        }
        self.state.network_correction = Some(correction.clone());
        self.state.last_network_sequence = sequence_decision
            .last_accepted_after
            .map(|value| value as u32);
        self.state.network_packets_accepted += 1;
        self.state.network_packets_lost_estimate += sequence_decision.estimated_packets_lost;
        self.state.validate()?;
        Ok(NetworkCorrectionEntryOutcome {
            correction,
            topology_transition: topology,
            state_after: self.state.clone(),
            entry_transfer: "predicted_and_authoritative_rigid_state_captured",
            sequence_decision,
        })
    }

    /// `begin_and_step_autonomous_network_correction`
    /// (kartrider_vehicle_engine.py:4560-4606): atomically begin an accepted
    /// packet and execute its first substep (or roll back when the packet is
    /// older than the current clock).
    pub fn begin_and_step_autonomous_network_correction(
        &mut self,
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        record: &NetworkCorrectionRecord,
    ) -> Result<AutonomousCorrectionEntry, String> {
        if !record.correction_active {
            return Err("network correction entry requires an active record".to_string());
        }
        if let Some(authoritative_time_ms) = record.authoritative_time_ms {
            if authoritative_time_ms < self.state.time_ms as f64 {
                return Ok(AutonomousCorrectionEntry::Rollback(Box::new(
                    self.rollback_and_resimulate_network_correction(world, record)?,
                )));
            }
            if authoritative_time_ms > self.state.time_ms as f64 {
                return Err("network authoritative_time_ms cannot be in the future".to_string());
            }
        }
        let authoritative = RigidStateSnapshot {
            position: record.authoritative_position,
            velocity: record.authoritative_velocity,
            matrix: record.authoritative_matrix,
            angular_velocity: record.authoritative_angular_velocity,
        };
        let mut candidate_engine = KartRiderVehicleEngine::new(&self.state, self.numeric_mode)?;
        let entry = candidate_engine.begin_network_correction(
            &authoritative,
            &record.mode,
            record.duration_ms,
            record.sequence,
        )?;
        let mut result = candidate_engine.step_autonomous_network_correction(parameters, world)?;
        self.state = candidate_engine.state;
        result.network_correction_acquired = true;
        result.entry = Some(Box::new(entry));
        result.selector = Some(record.raw.clone());
        result.step.network_correction_contract.target_authority =
            "world_query_accepted_authoritative_rigid_body_snapshot".to_string();
        Ok(AutonomousCorrectionEntry::Forward(Box::new(result)))
    }

    /// `rollback_and_resimulate_network_correction`
    /// (kartrider_vehicle_engine.py:4608-4738): replace one retained boundary
    /// and deterministically replay the recorded raw controls.
    pub fn rollback_and_resimulate_network_correction(
        &mut self,
        world: Option<&dyn World>,
        record: &NetworkCorrectionRecord,
    ) -> Result<NetworkCorrectionRollbackOutcome, String> {
        let Some(authoritative_time_ms) = record.authoritative_time_ms else {
            return Err("rollback correction requires numeric authoritative_time_ms".to_string());
        };
        if !authoritative_time_ms.is_finite() {
            return Err("authoritative_time_ms must be finite".to_string());
        }
        if authoritative_time_ms >= self.state.time_ms as f64 {
            return Err("rollback authoritative_time_ms must precede current state".to_string());
        }
        // Python `frame["time_ms_before"] == authoritative_time_ms`: int ==
        // float equality; both sides are exact below 2^53.
        let history_index = self
            .prediction_history
            .iter()
            .position(|frame| frame.time_ms_before as f64 == authoritative_time_ms);
        let Some(history_index) = history_index else {
            return Err(
                "authoritative_time_ms is outside retained prediction history or not \
                 aligned to a substep boundary"
                    .to_string(),
            );
        };
        let replay_frames: Vec<PredictionHistoryEntry> = self
            .prediction_history
            .iter()
            .skip(history_index)
            .cloned()
            .collect();
        if replay_frames
            .iter()
            .any(|frame| frame.has_pending_ground_plane_provider)
        {
            return Err(
                "rollback cannot replay an opaque pending ground-plane provider".to_string(),
            );
        }
        let sequence_decision = evaluate_network_sequence(
            &record
                .sequence
                .map_or(JsonValue::Null, |value| JsonValue::Number(value as f64)),
            &last_network_sequence_json(self.state.last_network_sequence),
            SEQUENCE_BITS,
        )?;
        if !sequence_decision.accepted {
            return Err(format!(
                "network correction packet rejected: {}",
                sequence_decision.reason
            ));
        }
        let mut base_state = replay_frames[0].state_before.clone();
        base_state.position = record.authoritative_position;
        base_state.velocity = record.authoritative_velocity;
        base_state.matrix = record.authoritative_matrix;
        base_state.angular_velocity = record.authoritative_angular_velocity;
        base_state.last_network_sequence = sequence_decision
            .last_accepted_after
            .map(|value| value as u32);
        base_state.network_packets_accepted = self.state.network_packets_accepted + 1;
        base_state.network_packets_rejected = self.state.network_packets_rejected;
        base_state.network_packets_lost_estimate =
            self.state.network_packets_lost_estimate + sequence_decision.estimated_packets_lost;
        base_state.network_correction = None;
        base_state.validate()?;

        let mut replay_engine = KartRiderVehicleEngine::new(&base_state, self.numeric_mode)?;
        replay_engine.network_replay_active = true;
        let mut replayed_history: Vec<PredictionHistoryEntry> = Vec::new();
        let mut replay_results: Vec<RawMotionOutcome> = Vec::new();
        for frame in &replay_frames {
            let before = replay_engine.state.clone();
            let replay_result = replay_engine.step_raw_motion_once(
                &frame.keys,
                &frame.parameters,
                world,
                frame.ground_planes.as_deref(),
                None,
            )?;
            replay_results.push(replay_result);
            replayed_history.push(PredictionHistoryEntry {
                time_ms_before: before.time_ms,
                time_ms_after: replay_engine.state.time_ms,
                state_before: before,
                state_after: replay_engine.state.clone(),
                keys: frame.keys.clone(),
                parameters: frame.parameters.clone(),
                ground_planes: frame.ground_planes.clone(),
                has_pending_ground_plane_provider: frame.has_pending_ground_plane_provider,
            });
        }
        if replay_engine.state.time_ms != self.state.time_ms {
            // Python RuntimeError.
            return Err("network rollback replay did not return to current time".to_string());
        }
        replay_engine.state.validate()?;
        let replayed_substep_count = replayed_history.len();
        let prefix: Vec<PredictionHistoryEntry> = self
            .prediction_history
            .iter()
            .take(history_index)
            .cloned()
            .collect();
        self.state = replay_engine.state;
        self.prediction_history = prefix.into_iter().chain(replayed_history).collect();
        while self.prediction_history.len() > PREDICTION_HISTORY_LIMIT {
            self.prediction_history.pop_front();
        }
        Ok(NetworkCorrectionRollbackOutcome {
            authoritative_time_ms,
            replayed_substep_count,
            sequence_decision,
            replay_results,
            state_after: self.state.clone(),
            topology_after: self.state.topology.clone(),
        })
    }

    /// `step_network_correction` (kartrider_vehicle_engine.py:4740-4936):
    /// advance authoritative snap/blend and autonomously leave on
    /// convergence. Python's keyword resume selectors are the
    /// [`NetworkCorrectionResume`] struct.
    #[allow(clippy::too_many_lines)]
    pub fn step_network_correction(
        &mut self,
        parameters: &VehicleParameters,
        resume: &NetworkCorrectionResume,
    ) -> Result<NetworkCorrectionStepOutcome, String> {
        parameters.validate()?;
        if resume.support_contact_count < 0 {
            return Err("resume_support_contact_count must be nonnegative".to_string());
        }
        if !(0..=6).contains(&resume.rail_state) {
            return Err("resume_rail_state must be 0 or an active state in 1..6".to_string());
        }
        if self.state.topology.domain != MotionDomain::NetworkCorrection {
            return Err("step requires NET-CORRECTION topology".to_string());
        }
        let Some(correction) = self.state.network_correction.clone() else {
            return Err("network correction transaction is missing".to_string());
        };
        let resume_snapshot = correction.resume_topology.clone();
        let substep_ms = parameters.substep_ms as f64;
        let correction_completes = correction.mode == "snap"
            || correction.elapsed_ms + substep_ms >= correction.duration_ms;
        let moving_resume_candidate = !resume.reset_requested
            && resume_snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.domain == MotionDomain::MovingContact)
            && resume.support_contact_count == 0;
        let resumes_moving_contact =
            moving_resume_candidate && resume.moving_object_state.is_some();
        if correction_completes && resumes_moving_contact {
            let state = resume
                .moving_object_state
                .as_ref()
                .expect("resumes_moving_contact implies a record");
            let snapshot_owner = resume_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.constraint_owner_id.as_deref());
            let identity_unchanged = match (state.get("object_id"), snapshot_owner) {
                (Some(JsonValue::String(record_id)), Some(owner)) => record_id == owner,
                // Python `.get("object_id") != .get("constraint_owner_id")`:
                // None == None (absent or explicit null on the record side).
                (None, None) | (Some(JsonValue::Null), None) => true,
                _ => false,
            };
            if !identity_unchanged {
                return Err(
                    "MOVING-CONTACT correction completion object identity changed".to_string(),
                );
            }
        }
        if correction_completes
            && moving_resume_candidate
            && !resume.moving_contact_observed
            && resume.moving_object_state.is_none()
        {
            return Err(
                "MOVING-CONTACT correction completion requires current object context \
                 or an explicit negative selector observation"
                    .to_string(),
            );
        }
        if correction_completes && resumes_moving_contact {
            const REQUIRED: [&str; 6] = [
                "contact_normal_world",
                "object_angular_velocity",
                "object_id",
                "object_matrix",
                "object_position",
                "object_velocity",
            ];
            let state = resume
                .moving_object_state
                .as_ref()
                .expect("resumes_moving_contact implies a record");
            let missing: Vec<&str> = REQUIRED
                .iter()
                .copied()
                .filter(|field| !state.contains_key(*field))
                .collect();
            if !missing.is_empty() {
                // REQUIRED is pre-sorted, matching Python `sorted(missing)`.
                let rendered = missing
                    .iter()
                    .map(|field| format!("'{field}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "MOVING-CONTACT correction completion requires current object \
                     context; missing [{rendered}]"
                ));
            }
        }
        let result = evaluate_network_correction_step(&correction, substep_ms)?;
        let rigid = result.state_after.clone();
        let mut rebased_moving_anchor: Option<MovingContactAnchor> = None;
        if result.complete && resumes_moving_contact {
            let context = resume
                .moving_object_state
                .as_ref()
                .expect("resumes_moving_contact implies a record");
            let frame = acquire_moving_contact_frame(
                rigid.position,
                rigid.velocity,
                rigid.matrix,
                rigid.angular_velocity,
                network_record_object_id(context)?,
                network_rigid_vec3(context, "object_position")?,
                network_rigid_vec3(context, "object_velocity")?,
                network_rigid_mat9(context, "object_matrix")?,
                network_rigid_vec3(context, "object_angular_velocity")?,
                network_rigid_vec3(context, "contact_normal_world")?,
                1e-6,
            )?;
            rebased_moving_anchor = Some(frame.anchor());
        }
        self.state.position = rigid.position;
        self.state.velocity = rigid.velocity;
        self.state.matrix = rigid.matrix;
        self.state.angular_velocity = rigid.angular_velocity;
        if self.numeric_mode.is_float32() {
            f32_vector(&mut self.state.position);
            f32_vector(&mut self.state.velocity);
            f32_vector(&mut self.state.matrix);
            f32_vector(&mut self.state.angular_velocity);
        }
        self.state.network_correction = Some(result.correction_after.clone());
        self.state.time_ms += parameters.substep_ms as u64;
        self.state.step_index += 1;
        self.state.topology.domain_step_index += 1;
        let mut topology: Option<TopologyDecision> = None;
        let mut restored_constraint_snapshot: Option<RestoredConstraintSnapshot> = None;
        if result.complete {
            let resume_topology = self
                .state
                .network_correction
                .as_ref()
                .and_then(|active| active.resume_topology.clone());
            let mut resume_domain: Option<MotionDomain> = None;
            if let Some(snapshot) = &resume_topology {
                resume_domain = Some(snapshot.domain);
                if snapshot.domain == MotionDomain::Contact && snapshot.return_domain.is_some() {
                    resume_domain = snapshot.return_domain;
                }
            }
            let decision = self.advance_motion_topology(&TopologyObservation {
                reset_requested: resume.reset_requested,
                network_correction_complete: true,
                resume_domain,
                support_contact_count: resume.support_contact_count,
                surface_mode: resume.surface_mode,
                rail_contact: resume.rail_state != 0,
                rail_state: resume.rail_state,
                moving_contact: rebased_moving_anchor.is_some(),
                moving_object_id: rebased_moving_anchor
                    .as_ref()
                    .map(|anchor| anchor.object_id.clone()),
                distinct_water_volume: resume.distinct_water_volume,
                ..Default::default()
            })?;
            topology = Some(decision);
            if let Some(snapshot) = &resume_topology {
                let resumed_domain = self.state.topology.domain;
                let original_domain = snapshot.domain;
                if resumed_domain == original_domain {
                    self.state.topology.return_domain = snapshot.return_domain;
                    self.state.topology.constraint_owner_id = snapshot.constraint_owner_id.clone();
                    if resumed_domain != MotionDomain::Rail {
                        self.state.topology.rail_state = snapshot.rail_state;
                    }
                    if resumed_domain == MotionDomain::MovingContact {
                        self.state.moving_contact_anchor = rebased_moving_anchor.clone();
                    }
                    // Python `topology["state_after"] = asdict(self.state.topology)`
                    // (:4895): the decision carried in the result reflects the
                    // restored constraint state, not the raw advance output.
                    if let Some(decision) = topology.as_mut() {
                        decision.state_after = self.state.topology.clone();
                    }
                    restored_constraint_snapshot = Some(RestoredConstraintSnapshot {
                        domain: original_domain,
                        return_domain: self.state.topology.return_domain,
                        constraint_owner_id: self.state.topology.constraint_owner_id.clone(),
                        rail_state: self.state.topology.rail_state,
                    });
                } else if original_domain == MotionDomain::MovingContact {
                    self.state.moving_contact_anchor = None;
                }
            }
            if resume.reset_requested {
                self.state.force_accumulator = [0.0, 0.0, 0.0];
                self.state.torque_accumulator = [0.0, 0.0, 0.0];
                self.state.contact_compression = [0.0; 4];
                self.state.moving_contact_anchor = None;
            } else if resume.support_contact_count > 0 {
                self.state.surface_mode_latched = Some(resume.surface_mode);
                self.state.surface_selector_elapsed_ms = 0.0;
            }
            self.state.network_correction = None;
        }
        self.state.validate()?;
        Ok(NetworkCorrectionStepOutcome {
            step: result,
            rebased_moving_anchor,
            topology_transition: topology,
            restored_constraint_snapshot,
            topology_after: self.state.topology.clone(),
            engine_state_after: self.state.clone(),
            network_correction_contract: NetworkCorrectionContract {
                target_authority: "authoritative_rigid_body_snapshot".to_string(),
                resume_selector_authority:
                    "caller_supplied_current support/rail/moving/water selector result".to_string(),
                moving_constraint_resume_authority: if resume.moving_object_state.is_some() {
                    "current_object_state_rebased_anchor".to_string()
                } else {
                    "not_applicable".to_string()
                },
                preserved_local_state:
                    "force/torque accumulators, controls, boost, resources, and \
                     checkpoints; current support reinitializes surface selector",
                open_boundary: "game_sequence_width_timestamp_origin_and_rollback_equivalence",
            },
        })
    }

    /// `step_autonomous_network_correction`
    /// (kartrider_vehicle_engine.py:4938-5066): advance the correction and
    /// derive resume selectors at the candidate pose. Queries run against a
    /// detached preview of this correction substep.
    #[allow(clippy::too_many_lines)]
    pub fn step_autonomous_network_correction(
        &mut self,
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
    ) -> Result<AutonomousNetworkCorrectionOutcome, String> {
        parameters.validate()?;
        if self.state.topology.domain != MotionDomain::NetworkCorrection {
            return Err("autonomous network step requires NET-CORRECTION topology".to_string());
        }
        let Some(correction) = self.state.network_correction.clone() else {
            return Err("network correction transaction is missing".to_string());
        };
        let preview = evaluate_network_correction_step(&correction, parameters.substep_ms as f64)?;
        let mut candidate_state = self.state.clone();
        candidate_state.position = preview.state_after.position;
        candidate_state.velocity = preview.state_after.velocity;
        candidate_state.matrix = preview.state_after.matrix;
        candidate_state.angular_velocity = preview.state_after.angular_velocity;
        let candidate_engine = KartRiderVehicleEngine::new(&candidate_state, self.numeric_mode)?;
        if !preview.complete {
            let mut result =
                self.step_network_correction(parameters, &NetworkCorrectionResume::default())?;
            result.network_correction_contract.resume_selector_authority =
                "deferred_until_correction_converges".to_string();
            return Ok(AutonomousNetworkCorrectionOutcome {
                step: result,
                autonomous_resume_selectors: AutonomousResumeSelectors {
                    candidate_position: candidate_state.position,
                    support: None,
                    moving_contact: None,
                    rail_contact: None,
                    distinct_water_volume: None,
                    reset_surface: None,
                    authority: "deferred_until_correction_converges",
                },
                network_correction_acquired: false,
                entry: None,
                selector: None,
            });
        }
        let support = candidate_engine.inspect_autonomous_support(&[], parameters, world, None)?;
        let reset_requested = Self::support_selects_authored_reset(&support);
        let resume_snapshot = correction.resume_topology.clone();
        let original_domain = resume_snapshot.as_ref().map(|snapshot| snapshot.domain);
        let mut moving_record: Option<BTreeMap<String, JsonValue>> = None;
        if original_domain == Some(MotionDomain::MovingContact)
            && support.support_contact_count == 0
        {
            let inspection = candidate_engine.inspect_autonomous_moving_contact(
                world,
                resume_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.constraint_owner_id.as_deref()),
                None,
            )?;
            if inspection.contact_active {
                // Python passes the merged inspection mapping
                // ({**record, "contact_active": True, "authority": ...}).
                moving_record = Some(merged_moving_contact_record(&inspection)?);
            }
        }
        let mut resume_rail_state: i64 = 0;
        let mut rail_selector: Option<BTreeMap<String, JsonValue>> = None;
        if original_domain == Some(MotionDomain::Rail) && !reset_requested {
            if !world_caps(world).query_rail_contact {
                return Err("autonomous RAIL resume requires world.query_rail_contact".to_string());
            }
            let selector = world
                .expect("capability probe implies a world")
                .query_rail_contact(
                    candidate_state.position,
                    resume_snapshot
                        .as_ref()
                        .map_or(0, |snapshot| snapshot.rail_state),
                    resume_snapshot
                        .as_ref()
                        .and_then(|snapshot| snapshot.constraint_owner_id.as_deref()),
                )
                .ok_or_else(|| {
                    "query_rail_contact capability reported present but returned no result"
                        .to_string()
                })?;
            if let Some(selector) = selector {
                // Python `bool(rail_selector.get("contact_active", True))`
                // and `int(rail_selector.get("rail_state", 0))`.
                if selector.get("contact_active").is_none_or(py_truthy) {
                    resume_rail_state = match selector.get("rail_state") {
                        Some(JsonValue::Number(value)) => *value as i64,
                        _ => 0,
                    };
                }
                rail_selector = Some(selector);
            }
        }
        let mut distinct_water = false;
        let mut water_selector: Option<JsonValue> = None;
        if original_domain == Some(MotionDomain::Water) && !reset_requested {
            if !world_caps(world).query_distinct_water_volume {
                return Err(
                    "autonomous WATER resume requires world.query_distinct_water_volume"
                        .to_string(),
                );
            }
            let selector = world
                .expect("capability probe implies a world")
                .query_distinct_water_volume(candidate_state.position)
                .ok_or_else(|| {
                    "query_distinct_water_volume capability reported present but returned \
                     no result"
                        .to_string()
                })?;
            distinct_water = selector.as_ref().is_some_and(py_truthy);
            water_selector = selector;
        }
        let mut result = self.step_network_correction(
            parameters,
            &NetworkCorrectionResume {
                support_contact_count: support.support_contact_count,
                surface_mode: support.surface_mode,
                rail_state: resume_rail_state,
                moving_object_state: moving_record.clone(),
                moving_contact_observed: original_domain == Some(MotionDomain::MovingContact)
                    && support.support_contact_count == 0,
                distinct_water_volume: distinct_water,
                reset_requested,
            },
        )?;
        result.network_correction_contract.resume_selector_authority =
            "autonomous_world_queries_at_post_correction_candidate_pose".to_string();
        Ok(AutonomousNetworkCorrectionOutcome {
            step: result,
            autonomous_resume_selectors: AutonomousResumeSelectors {
                candidate_position: candidate_state.position,
                support: Some(support),
                moving_contact: moving_record,
                rail_contact: rail_selector,
                distinct_water_volume: water_selector,
                reset_surface: Some(reset_requested),
                authority: "world_queries_at_post_correction_candidate_pose",
            },
            network_correction_acquired: false,
            entry: None,
            selector: None,
        })
    }

    /// `_decode_active_rail_runtime` (kartrider_vehicle_engine.py:3733-3751):
    /// validate the decoded writer context carried by an active rail hit.
    /// Python also accepts a `RailRuntimeContext` instance; the Rust
    /// selector payload is always the decoded JSON mapping, so only the
    /// `dict` branch is representable. Dataclass `TypeError`s map to
    /// `invalid decoded rail runtime context: <typeerror>` with the CPython
    /// message; the field validation itself lives in
    /// [`RailRuntimeContext::validate`].
    pub fn decode_active_rail_runtime(
        selector: &BTreeMap<String, JsonValue>,
    ) -> Result<RailRuntimeContext, String> {
        let Some(JsonValue::Object(runtime_payload)) = selector.get("runtime") else {
            return Err(
                "active rail selector requires runtime mapping or RailRuntimeContext".to_string(),
            );
        };
        const KNOWN_KEYS: [&str; 13] = [
            "rail_dir",
            "accel_be0",
            "obf_6e8",
            "grip_720",
            "gravity_be8",
            "gate_1c0_decoded",
            "boost_special_gate_9b8",
            "boost_special_gate_6e4",
            "attached_object_state",
            "obf_808",
            "obf_814",
            "min_velocity",
            "max_velocity",
        ];
        let type_error =
            |message: String| format!("invalid decoded rail runtime context: {message}");
        for key in runtime_payload.keys() {
            if !KNOWN_KEYS.contains(&key.as_str()) {
                return Err(type_error(format!(
                    "RailRuntimeContext.__init__() got an unexpected keyword argument '{key}'"
                )));
            }
        }
        let number = |key: &str| -> Option<f64> {
            match runtime_payload.get(key) {
                Some(JsonValue::Number(value)) => Some(*value),
                Some(JsonValue::Bool(value)) => Some(f64::from(*value)),
                _ => None,
            }
        };
        let boolean = |key: &str| -> bool { runtime_payload.get(key).is_some_and(py_truthy) };
        let mut missing: Vec<&str> = Vec::new();
        let rail_dir = match runtime_payload.get("rail_dir") {
            Some(JsonValue::Array(values)) => Some(values.clone()),
            _ => None,
        };
        if rail_dir.is_none() {
            missing.push("rail_dir");
        }
        for key in ["accel_be0", "obf_6e8", "grip_720"] {
            if number(key).is_none() {
                missing.push(key);
            }
        }
        if !missing.is_empty() {
            let quoted: Vec<String> = missing.iter().map(|key| format!("'{key}'")).collect();
            let joined = match quoted.len() {
                1 => quoted[0].clone(),
                2 => format!("{} and {}", quoted[0], quoted[1]),
                _ => format!(
                    "{}, and {}",
                    quoted[..quoted.len() - 1].join(", "),
                    quoted[quoted.len() - 1]
                ),
            };
            return Err(type_error(format!(
                "RailRuntimeContext.__init__() missing {} required positional arguments: {joined}",
                missing.len()
            )));
        }
        let rail_dir_values = rail_dir.unwrap();
        // Python stores the list verbatim and the three-component check is a
        // `RailRuntimeContext.validate` ValueError (unwrapped), not a
        // dataclass TypeError.
        if rail_dir_values.len() != 3 {
            return Err("rail_dir must have three components".to_string());
        }
        let runtime = RailRuntimeContext {
            rail_dir: [
                number_from_json(rail_dir_values.first()),
                number_from_json(rail_dir_values.get(1)),
                number_from_json(rail_dir_values.get(2)),
            ],
            accel_be0: number("accel_be0").unwrap(),
            obf_6e8: number("obf_6e8").unwrap(),
            grip_720: number("grip_720").unwrap(),
            gravity_be8: number("gravity_be8").unwrap_or(1.0),
            gate_1c0_decoded: boolean("gate_1c0_decoded"),
            boost_special_gate_9b8: boolean("boost_special_gate_9b8"),
            boost_special_gate_6e4: boolean("boost_special_gate_6e4"),
            attached_object_state: number("attached_object_state").unwrap_or(0.0) as i64,
            obf_808: number("obf_808").unwrap_or(1.0),
            obf_814: number("obf_814").unwrap_or(1.0),
            min_velocity: number("min_velocity"),
            max_velocity: number("max_velocity"),
        };
        runtime.validate()?;
        Ok(runtime)
    }

    /// `inspect_autonomous_rail_capture` (kartrider_vehicle_engine.py:3753-3791):
    /// query and validate a possible initial rail capture without mutation.
    pub fn inspect_autonomous_rail_capture(
        &self,
        world: Option<&dyn World>,
    ) -> Result<RailCaptureInspection, String> {
        if !world_caps(world).query_rail_contact {
            return Ok(RailCaptureInspection {
                contact_active: false,
                rail_state: None,
                selector: None,
                authority: "world_has_no_rail_selector",
            });
        }
        let selector = world
            .expect("capability probe implies a world")
            .query_rail_contact(self.state.position, 0, None)
            .ok_or_else(|| {
                "query_rail_contact capability reported present but returned no result".to_string()
            })?;
        let Some(selector) = selector else {
            return Ok(RailCaptureInspection {
                contact_active: false,
                rail_state: None,
                selector: None,
                authority: "world_query_rail_contact",
            });
        };
        // Python bool(selector.get("contact_active", True)).
        let contact_active = selector.get("contact_active").is_none_or(py_truthy);
        if !contact_active {
            return Ok(RailCaptureInspection {
                contact_active: false,
                rail_state: None,
                selector: Some(selector),
                authority: "world_query_rail_contact",
            });
        }
        // Python int(selector.get("rail_state", 1)).
        let rail_state = match selector.get("rail_state") {
            None => 1,
            Some(JsonValue::Number(value)) => *value as i64,
            Some(JsonValue::Bool(value)) => i64::from(*value),
            Some(_) => {
                return Err("active rail capture requires runtime state in 1..6".to_string());
            }
        };
        if !(1..=6).contains(&rail_state) {
            return Err("active rail capture requires runtime state in 1..6".to_string());
        }
        Self::decode_active_rail_runtime(&selector)?;
        Ok(RailCaptureInspection {
            contact_active: true,
            rail_state: Some(rail_state),
            selector: Some(selector),
            authority: "world_query_rail_contact",
        })
    }

    /// `step_raw_rail` (kartrider_vehicle_engine.py:2454-2660): advance one
    /// rail-owned substep from raw keys and decoded rail context, or run the
    /// continuous release when `rail_contact` is false.
    #[allow(clippy::too_many_arguments)]
    pub fn step_raw_rail(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        runtime: Option<&RailRuntimeContext>,
        rail_contact: bool,
        rail_state: Option<i64>,
        release_support_contact_count: i64,
        release_surface_mode: i64,
        pre_damping_velocity_override: Option<[f64; 3]>,
        post_rail_force_accumulator: Option<[f64; 3]>,
        post_rail_torque_accumulator: Option<[f64; 3]>,
        pose_rail_state: Option<i64>,
        rail_owner_id: Option<String>,
    ) -> Result<RawRailOutcome, String> {
        parameters.validate()?;
        if release_support_contact_count < 0 {
            return Err("release_support_contact_count must be nonnegative".to_string());
        }
        let active = validate_raw_control_keys(keys)?;
        let current_state = rail_state.unwrap_or(self.state.topology.rail_state);
        let current_domain = self.state.topology.domain;
        if !rail_contact && current_domain != MotionDomain::Rail {
            return Err("rail release requires current RAIL ownership".to_string());
        }
        if rail_contact && !(1..=6).contains(&current_state) {
            return Err("rail capture/sustain requires an active state in 1..6".to_string());
        }
        if rail_contact {
            let Some(runtime) = runtime else {
                return Err("active rail contact requires decoded runtime context".to_string());
            };
            runtime.validate()?;
        }
        let topology = self.advance_motion_topology(&TopologyObservation {
            rail_contact,
            rail_state: current_state,
            rail_owner_id,
            support_contact_count: release_support_contact_count,
            surface_mode: release_surface_mode,
            ..TopologyObservation::default()
        })?;
        if !rail_contact {
            let destination = self.state.topology.domain;
            if destination == MotionDomain::Rail {
                // Python RuntimeError.
                return Err("inactive rail contact retained RAIL ownership".to_string());
            }
            if release_support_contact_count > 0 {
                self.state.surface_mode_latched = Some(release_surface_mode);
                self.state.surface_selector_elapsed_ms = 0.0;
            }
            return Ok(RawRailOutcome::Released(Box::new(RailReleaseOutcome {
                release_destination: destination.value().to_string(),
                rigid_state_transfer: "continuous_no_rail_writer",
                rail_selector: RailReleaseSelector {
                    rail_contact: false,
                    rail_state_before: current_state,
                    support_contact_count: release_support_contact_count,
                    surface_mode: release_surface_mode,
                },
                topology_decision: topology,
                topology_after: self.state.topology.clone(),
                state_after: self.state.clone(),
                rail_contract: RailStepContract {
                    control_authority: "raw_keys",
                    rigid_body_state_authority: "engine_owned_continuous_release",
                    writer_selection_authority: "no_rail_writer_after_release",
                    rail_context_authority: "current_contact_and_support_selectors",
                    open_boundary:
                        "autonomous map-side rail and JM/DJ contact candidate generation",
                },
                rail_release_pending_producer: None,
            })));
        }
        if self.state.topology.domain != MotionDomain::Rail {
            return Err("step_raw_rail requires an active rail state/contact".to_string());
        }
        let runtime = runtime.expect("active rail contact validates runtime above");
        let columns = matrix_columns(self.state.matrix);
        let forward_axis = columns[parameters.forward_axis_column as usize];
        let forward = [
            parameters.forward_axis_sign * forward_axis[0],
            parameters.forward_axis_sign * forward_axis[1],
            parameters.forward_axis_sign * forward_axis[2],
        ];
        let lateral = columns[0];
        let vertical = columns[2];
        let (adc, ae0, aec) = if self.numeric_mode.is_float32() {
            (
                dot3_f32(self.state.velocity, forward),
                dot3_f32(self.state.velocity, lateral),
                dot3_f32(self.state.velocity, vertical),
            )
        } else {
            (
                dot3_sum(self.state.velocity, forward),
                dot3_sum(self.state.velocity, lateral),
                dot3_sum(self.state.velocity, vertical),
            )
        };
        let signed_steering_axis = if active.contains("LEFT") {
            1.0
        } else if active.contains("RIGHT") {
            -1.0
        } else {
            0.0
        };
        let positive_a80 = if active.contains("UP") { 1.0 } else { 0.0 };
        let negative_a80 = if active.contains("DOWN") { -1.0 } else { 0.0 };
        let result = evaluate_rail_step(&RailStepArgs {
            position_before: self.state.position,
            velocity_before: self.state.velocity,
            angular_velocity_before: self.state.angular_velocity,
            matrix_before: self.state.matrix,
            force_accumulator_before: self.state.force_accumulator,
            torque_accumulator_before: self.state.torque_accumulator,
            rail_contact,
            rail_state: current_state,
            gate_1c0_decoded: runtime.gate_1c0_decoded,
            positive_a80,
            negative_a80,
            boost_family_active: self.state.boost_state != 0,
            boost_special_gate_9b8: runtime.boost_special_gate_9b8,
            boost_special_gate_6e4: runtime.boost_special_gate_6e4,
            attached_object_state: runtime.attached_object_state,
            rail_dir: runtime.rail_dir,
            accel_be0: runtime.accel_be0,
            obf_6e8: runtime.obf_6e8,
            obf_808: runtime.obf_808,
            obf_814: runtime.obf_814,
            grip_720: runtime.grip_720,
            field_604: [0.0, 0.0, parameters.contact_gravity_per_mass],
            mass: parameters.mass,
            gravity_be8: runtime.gravity_be8,
            adc,
            ae0,
            aec,
            signed_steering_axis,
            drive_tune_718: parameters.drive_tune_718,
            post_rail_force_accumulator,
            post_rail_torque_accumulator,
            inverse_inertia_matrix: parameters.inverse_inertia_matrix,
            damping: parameters.common_damping,
            directional_damping: parameters.directional_damping,
            damping_runtime_multiplier: parameters.damping_runtime_multiplier,
            damping_special_gate: self.state.soft_flag_188,
            dt: parameters.substep_ms as f64 / 1000.0,
            numeric_mode: self.numeric_mode.as_str().to_string(),
            pre_damping_velocity_override,
            pose_rail_state,
            rail_base_bf8: self.state.rail_base_bf8,
            rail_constraint_c1c_before: self.state.rail_constraint_c1c,
            rail_orientation_dt: Some(
                self.state
                    .rail_orientation_dt_latch
                    .unwrap_or(parameters.substep_ms as f64 / 1000.0),
            ),
        })?;
        self.state.position = result.position_after;
        self.state.velocity = result.velocity_after;
        self.state.angular_velocity = result.angular_velocity_after;
        self.state.matrix = result.matrix_after;
        if let Some(rail_orientation) = &result.rail_constraint_orientation {
            self.state.rail_base_bf8 = Some(rail_orientation.rail_base_bf8_after);
            self.state.rail_constraint_c1c = Some(rail_orientation.rail_constraint_c1c_after);
        }
        // B29350's C1C pose writer consumes the preceding writer epoch's
        // scheduler dt while using the current epoch's angular state
        // (:2622-2627).
        self.state.rail_orientation_dt_latch = Some(parameters.substep_ms as f64 / 1000.0);
        // 614/620 are per-integration-epoch accumulators: every next kind10
        // writer begins from exact zero (:2628-2641).
        let accumulator_clear = RailAccumulatorClear {
            force_before_clear: result.force_accumulator,
            torque_before_clear: result.torque_accumulator,
            authority: "runtime_next_linear_write_epoch_zero_boundary",
        };
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.drive_selected_slot = positive_a80;
        self.state.time_ms += parameters.substep_ms as u64;
        self.state.step_index += 1;
        self.state.topology.domain_step_index += 1;
        self.state.topology.rail_state_elapsed_ms += parameters.substep_ms as f64;
        Ok(RawRailOutcome::Active(Box::new(RawRailActiveOutcome {
            rail_step: result,
            accumulator_clear,
            topology_decision: topology,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
            rail_contract: RailStepContract {
                control_authority: "raw_keys",
                rigid_body_state_authority: "engine_owned",
                writer_selection_authority: "exact_B29350_dispatch",
                rail_context_authority: "explicit_decoded_runtime_context",
                open_boundary: "autonomous_rail_path_geometry_and_state_transitions",
            },
        })))
    }

    /// `acquire_autonomous_rail` (kartrider_vehicle_engine.py:3793-3895):
    /// capture a newly selected rail and execute its first owned substep.
    /// `Ok(None)` means the current world selector did not offer a rail.
    pub fn acquire_autonomous_rail(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        inspected: Option<RailCaptureInspection>,
        support: Option<&SupportObservation>,
    ) -> Result<Option<RailCaptureExecution>, String> {
        let domain = self.state.topology.domain;
        if matches!(
            domain,
            MotionDomain::Reset
                | MotionDomain::NetworkCorrection
                | MotionDomain::Rail
                | MotionDomain::MovingContact
                | MotionDomain::Water
        ) {
            return Err(format!(
                "cannot acquire rail while {} owns state",
                domain.value()
            ));
        }
        let capture = match inspected {
            None => self.inspect_autonomous_rail_capture(world)?,
            Some(inspected) => inspected,
        };
        if !capture.contact_active {
            return Ok(None);
        }
        let selector = capture.selector.clone().unwrap_or_default();
        let rail_state = capture.rail_state.unwrap();
        let runtime = Self::decode_active_rail_runtime(&selector)?;
        let rail_owner_id = selector_path_id(&selector);
        if let (1, Some(support)) = (rail_state, support) {
            let topology = self.advance_motion_topology(&TopologyObservation {
                rail_contact: true,
                rail_state: 1,
                rail_owner_id,
                ..TopologyObservation::default()
            })?;
            let ground = self.step_raw_ground_correction(
                keys,
                parameters,
                None,
                Some(&ground_plane_set_from_world_planes(&support.planes)),
                &GroundStepOptions::default(),
            )?;
            self.state.topology.domain_step_index += 1;
            let outcome = RailCaptureOutcome {
                ground: Box::new(ground),
                rail_state1_ground_writer: RailState1GroundWriter {
                    selected: true,
                    state_after: 1,
                    support_contact_count: support.support_contact_count,
                    selector_phase: None,
                    authority: "run119_state1_capture_store_then_ground_kind42_30_5_1_6_2_36",
                },
                topology_decision: topology,
                topology_after: self.state.topology.clone(),
                state_after: self.state.clone(),
                autonomous_rail_selector: selector,
                rail_contract: RailContract {
                    control_authority: "raw_keys",
                    rigid_body_state_authority: "engine_owned_state1_ground_capture_chain",
                    writer_selection_authority: "runtime_state1_ground_after_capture_store",
                    rail_context_authority: "world_query_rail_contact_initial_capture",
                    capture_authority: "active_geometry_selector_state_1_to_6",
                    open_boundary: "autonomous state1 path candidate list generation",
                },
            };
            return Ok(Some(RailCaptureExecution::GroundWriter(Box::new(outcome))));
        }
        let raw = self.step_raw_rail(
            keys,
            parameters,
            Some(&runtime),
            true,
            Some(rail_state),
            0,
            0,
            None,
            None,
            None,
            None,
            rail_owner_id,
        )?;
        let RawRailOutcome::Active(active) = raw else {
            // Unreachable: rail_contact=true always yields the active writer
            // chain (Python sets `result["rail_captured"] = True` on it).
            return Err("rail capture through step_raw_rail released ownership".to_string());
        };
        // Python overwrites two rail_contract keys on the capture result
        // (:3889-3894); the remaining three are the step_raw_rail values.
        Ok(Some(RailCaptureExecution::RawRail(Box::new(
            RailCaptureRawRailOutcome {
                rail: *active,
                autonomous_rail_selector: selector,
                rail_contract: RailContract {
                    control_authority: "raw_keys",
                    rigid_body_state_authority: "engine_owned",
                    writer_selection_authority: "exact_B29350_dispatch",
                    rail_context_authority: "world_query_rail_contact_initial_capture",
                    capture_authority: "active_geometry_selector_state_1_to_6",
                    open_boundary: "autonomous_rail_path_geometry_and_state_transitions",
                },
            },
        ))))
    }

    /// `_step_rail_state6_tail` (kartrider_vehicle_engine.py:1798-2452): run
    /// the B276F0 state-6 airborne/contact-confirmation tail.
    #[allow(clippy::too_many_lines)]
    pub fn step_rail_state6_tail(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
    ) -> Result<RailState6TailOutcome, String> {
        let active_keys = validate_raw_control_keys(keys)?;
        let state_at_step_start = self.state.clone();
        let support = self.inspect_autonomous_support(keys, parameters, world, None)?;
        let world_ref = world.expect("support inspection implies a world");
        let mut state6_contact_slots = [false; 4];
        for (index, active) in support.active_slots.iter().enumerate() {
            state6_contact_slots[index] =
                *active && support.planes[index].road_rail.trim().to_lowercase() != "highspeed";
        }
        let contact_188_before = self.state.soft_flag_188;
        let contact_record_18c_valid = state6_contact_slots.iter().any(|valid| *valid);
        // +0x188 denotes the existence of a retained current contact record,
        // not a four-wheel unanimity bit (:1814-1819).
        let contact_188_after = support.active_slots.iter().any(|active| *active);
        if !self.state.soft_flag_188 && !world_caps(world).sweep_point {
            return Err("state6 B276F0 tail requires world.sweep_point".to_string());
        }
        let state6_up = matrix_columns(self.state.matrix)[2];
        // f64 intermediates with explicit _f32 quantization at each Python
        // site (handoff §6.4); these quantize unconditionally in both modes.
        let mut state6_segment_start = [0.0; 3];
        let mut state6_segment_end = [0.0; 3];
        for index in 0..3 {
            state6_segment_start[index] = f32q(
                f32q(self.state.position[index])
                    + f32q(f32q(state6_up[index]) * f32q(parameters.state6_contact_half_span)),
            );
            state6_segment_end[index] = f32q(
                f32q(self.state.position[index])
                    - f32q(f32q(state6_up[index]) * f32q(parameters.state6_contact_half_span)),
            );
        }
        let state6_contact_candidate: BTreeMap<String, JsonValue>;
        let state6_contact_valid: bool;
        let state6_contact_source: &'static str;
        if contact_188_before {
            state6_contact_candidate = BTreeMap::from([
                ("hit".to_string(), JsonValue::Null),
                (
                    "reason".to_string(),
                    JsonValue::String("contact_188_selects_current_record_18c".to_string()),
                ),
            ]);
            state6_contact_valid = contact_record_18c_valid;
            state6_contact_source = "contact_188_current_record_18c";
        } else {
            state6_contact_candidate = world_ref
                .sweep_point(state6_segment_start, state6_segment_end)
                .ok_or_else(|| {
                    "sweep_point capability reported present but returned no result".to_string()
                })?;
            state6_contact_valid = state6_contact_candidate.get("hit").is_some_and(py_truthy)
                && json_token_or_empty(state6_contact_candidate.get("road_rail"))
                    .trim()
                    .to_lowercase()
                    != "highspeed";
            state6_contact_source = "A5B840_center_segment_query";
        }
        let mut runtime = self.state.rail_selector_runtime.clone().unwrap_or_default();
        let zero_boundary_release_delay_before = runtime
            .state6_zero_boundary_release_delay_substeps
            .unwrap_or(0.0) as i64;
        // B276F0 samples the candidate at physics-substep frequency, but its
        // C60 confirmation timer belongs to the outer-frame selector
        // (:1880-1889).
        let state6_contact_latched =
            runtime.state6_contact_candidate_latched.unwrap_or(0.0) != 0.0 || state6_contact_valid;
        runtime.state6_contact_candidate_latched =
            Some(if state6_contact_latched { 1.0 } else { 0.0 });
        let remaining = runtime
            .state6_contact_remaining_ms
            .unwrap_or(STATE6_CONTACT_CONFIRM_MS);
        let selector_delta_ms = parameters
            .topology_selector_delta_ms
            .unwrap_or(parameters.substep_ms as f64);
        let mut selector = evaluate_rail_capture_selector(&RailCaptureSelectorArgs {
            state_before: 6,
            elapsed_in_state_ms: self.state.topology.rail_state_elapsed_ms,
            field_af0: length3(self.state.velocity, self.numeric_mode.is_float32()),
            delta_ms: Some(selector_delta_ms),
            state6_contact_remaining_ms: Some(remaining),
            state6_contact_candidate_valid: state6_contact_latched,
            ..RailCaptureSelectorArgs::default()
        })?;
        let mut float_zero_boundary_writer_committed: Option<bool> = None;
        if zero_boundary_release_delay_before > 0 {
            let release_delay_after = zero_boundary_release_delay_before - 1;
            if release_delay_after == 0 {
                runtime.state6_zero_boundary_release_delay_substeps = None;
                selector.state_after = 0;
                selector.transitioned = true;
                selector.reason = "state6_float_zero_boundary_writer_committed";
                float_zero_boundary_writer_committed = Some(true);
            } else {
                runtime.state6_zero_boundary_release_delay_substeps =
                    Some(release_delay_after as f64);
                selector.state_after = 6;
                selector.transitioned = false;
                selector.reason = "state6_float_zero_boundary_writer_pending";
                float_zero_boundary_writer_committed = Some(false);
            }
        } else if selector.transitioned {
            let raw_after_ms = match &selector.runtime_after {
                Some(RailCaptureSelectorRuntimeAfter::State6(runtime_after)) => {
                    runtime_after.state6_contact_raw_after_ms
                }
                _ => 0.0,
            };
            // run129 lands on the float32 zero boundary at the frame-owned
            // countdown call; preserve that writer phase instead of treating
            // a float32 rounding crumb as a whole-frame delay (:1947-1964).
            if -0.001 < raw_after_ms && raw_after_ms < 0.0 {
                runtime.state6_zero_boundary_release_delay_substeps = Some(2.0);
                selector.state_after = 6;
                selector.transitioned = false;
                selector.reason = "state6_float_zero_boundary_writer_pending";
                float_zero_boundary_writer_committed = Some(false);
            }
        }
        if let Some(RailCaptureSelectorRuntimeAfter::State6(runtime_after)) =
            &selector.runtime_after
        {
            runtime.state6_contact_remaining_ms = Some(runtime_after.state6_contact_remaining_ms);
        }
        self.state.rail_selector_runtime = Some(runtime.clone());
        let contact_accumulator_present = support.active_slots.iter().any(|active| *active)
            && support.producer.evaluator.contact.is_some();
        // State 6 no longer calls B29350; its common tail contains the
        // observed -30*CC angular producer before B29E70 damping
        // (:1975-2019). The float32 branch is `_fmul(_f32(-value), damping)`;
        // the float64 branch keeps the pre-audit `_f32(-value * damping)`
        // single quantization — both preserve the unary-minus negative zero.
        let mut torque_delta = [0.0; 3];
        for (index, value) in self.state.angular_velocity.iter().enumerate() {
            torque_delta[index] = if self.numeric_mode.is_float32() {
                fmul(f32q(-value), parameters.state6_residual_angular_damping)
            } else {
                f32q(-value * parameters.state6_residual_angular_damping)
            };
        }
        let state6_common_angular = RailState6CommonAngular {
            angular_velocity_before: self.state.angular_velocity,
            coefficient: parameters.state6_residual_angular_damping,
            final_coefficient_after_common_damping: if self.numeric_mode.is_float32() {
                fadd(
                    parameters.state6_residual_angular_damping,
                    parameters.common_damping,
                )
            } else {
                parameters.state6_residual_angular_damping + parameters.common_damping
            },
            torque_delta,
        };
        let mut additional_torque = torque_delta;
        let mut gravity_per_mass = [
            0.0,
            0.0,
            if self.numeric_mode.is_float32() {
                f32q(parameters.contact_gravity_per_mass)
            } else {
                parameters.contact_gravity_per_mass
            },
        ];
        let mut additional_force: Option<[f64; 3]> = None;
        let mut state6_support_force_branch: &'static str = "no_active_support";
        if contact_accumulator_present {
            let entry_common_tail_delay = runtime
                .state6_entry_common_tail_delay_substeps
                .unwrap_or(0.0) as i64;
            if entry_common_tail_delay > 0 {
                runtime.state6_entry_common_tail_delay_substeps =
                    Some((entry_common_tail_delay - 1) as f64);
                state6_support_force_branch = "entry_common_tail_delay";
            } else {
                let ground_614 = support
                    .producer
                    .evaluator
                    .accumulator_614_after_post_bs_reset
                    .expect("contact support carries the post-BS 614 accumulator");
                let ground_620 = support
                    .producer
                    .evaluator
                    .accumulator_620_after_post_bs_reset
                    .expect("contact support carries the post-BS 620 accumulator");
                let support_torque;
                if active_keys.contains("UP") || active_keys.contains("DOWN") {
                    additional_force = Some(ground_614);
                    support_torque = ground_620;
                    state6_support_force_branch = "primary_drive_ground_correction_and_contact";
                } else {
                    // Remove only the correction's horizontal linear
                    // contribution; keep the sampled Z and torque
                    // (:2061-2086).
                    let correction_614 = support
                        .producer
                        .evaluator
                        .correction
                        .as_ref()
                        .expect("contact support carries the ground correction pipeline")
                        .accumulator_614_after;
                    let mut force = ground_614;
                    for index in 0..2 {
                        force[index] = if self.numeric_mode.is_float32() {
                            fsub(force[index], correction_614[index])
                        } else {
                            force[index] - correction_614[index]
                        };
                    }
                    additional_force = Some(force);
                    support_torque = ground_620;
                    state6_support_force_branch = "contact_only_without_primary_drive";
                }
                for index in 0..3 {
                    additional_torque[index] = if self.numeric_mode.is_float32() {
                        fadd(additional_torque[index], support_torque[index])
                    } else {
                        additional_torque[index] + support_torque[index]
                    };
                }
            }
            gravity_per_mass = [0.0, 0.0, 0.0];
        }
        let pending_entry_force = if let Some(pending_x) = runtime.state6_entry_force_pending_x {
            Some([
                pending_x,
                runtime.state6_entry_force_pending_y.unwrap_or(0.0),
                runtime.state6_entry_force_pending_z.unwrap_or(0.0),
            ])
        } else {
            None
        };
        let pending_entry_delay = runtime.state6_entry_force_delay_substeps.unwrap_or(0.0) as i64;
        let mut entry_force_transaction: Option<RailState6EntryForceTransaction> = None;
        if let Some(pending_force) = pending_entry_force {
            if pending_entry_delay > 0 {
                runtime.state6_entry_force_delay_substeps = Some((pending_entry_delay - 1) as f64);
                entry_force_transaction = Some(RailState6EntryForceTransaction {
                    phase: "B74_staged_waiting_for_kind34_fold",
                    delay_substeps_after: Some(pending_entry_delay - 1),
                    force: pending_force,
                });
            } else {
                let base_force = additional_force.unwrap_or([0.0, 0.0, 0.0]);
                let mut folded = [0.0; 3];
                for index in 0..3 {
                    folded[index] = if self.numeric_mode.is_float32() {
                        fadd(base_force[index], pending_force[index])
                    } else {
                        base_force[index] + pending_force[index]
                    };
                }
                additional_force = Some(folded);
                runtime.state6_entry_force_pending_x = None;
                runtime.state6_entry_force_pending_y = None;
                runtime.state6_entry_force_pending_z = None;
                runtime.state6_entry_force_delay_substeps = None;
                entry_force_transaction = Some(RailState6EntryForceTransaction {
                    phase: "B74_folded_by_kind34_into_614_and_consumed",
                    delay_substeps_after: None,
                    force: pending_force,
                });
            }
        }
        self.state.rail_selector_runtime = Some(runtime);
        let result = evaluate_airborne_step(
            self.state.position,
            self.state.velocity,
            self.state.angular_velocity,
            self.state.matrix,
            parameters.mass,
            parameters.substep_ms as f64 / 1000.0,
            gravity_per_mass,
            parameters.inverse_inertia_matrix,
            parameters.common_damping,
            parameters.directional_damping,
            parameters.damping_runtime_multiplier,
            // The B29E70 call reads the contact bit after this substep's
            // support refresh (:2170-2175).
            contact_188_after,
            additional_force,
            Some(additional_torque),
            6,
            self.numeric_mode.as_str(),
        )?;
        let mut velocity_after = result.velocity_after;
        let mut angular_velocity_after = result.angular_velocity_after;
        let mut position_after = result.position_after;
        let mut matrix_after = result.matrix_after;
        let mut collision_chain: Vec<CollisionChainResponse> = Vec::new();
        if world_caps(world).sweep_obb_rotating_all {
            let start_columns = matrix_columns(state_at_step_start.matrix);
            let end_columns = matrix_columns(matrix_after);
            let mut sweep_start = [0.0; 3];
            let mut sweep_end = [0.0; 3];
            for index in 0..3 {
                sweep_start[index] = state_at_step_start.position[index]
                    + start_columns[2][index] * parameters.collision_center_up;
                sweep_end[index] =
                    position_after[index] + end_columns[2][index] * parameters.collision_center_up;
            }
            let query_at_start = world_caps(world).query_at_step_start;
            let collision_candidates = world_ref
                .sweep_obb_rotating_all(
                    sweep_start,
                    if query_at_start {
                        sweep_start
                    } else {
                        sweep_end
                    },
                    state_at_step_start.matrix,
                    if query_at_start {
                        state_at_step_start.matrix
                    } else {
                        matrix_after
                    },
                    parameters.collision_half_extents,
                )
                .ok_or_else(|| {
                    "sweep_obb_rotating_all capability reported present but returned no result"
                        .to_string()
                })?;
            let state6_chain = self.apply_collision_chain(
                &collision_candidates,
                state_at_step_start.matrix,
                matrix_after,
                velocity_after,
                angular_velocity_after,
                &CollisionChainSelectorState {
                    arg_threshold: state_at_step_start.collision_arg_threshold,
                    e38: state_at_step_start.collision_e38,
                    e4c: state_at_step_start.collision_e4c,
                    drive_selected_slot: state_at_step_start.drive_selected_slot,
                    drive_raw_aa0: state_at_step_start.drive_raw_aa0,
                    drive_sign_aac: state_at_step_start.drive_sign_aac,
                    selector_af4: state_at_step_start.soft_selector_af4,
                    flag_188: state_at_step_start.soft_flag_188,
                },
                Some("hard_contact"),
                false,
                true,
            )?;
            collision_chain = state6_chain.responses;
            if !collision_chain.is_empty() {
                let dt = parameters.substep_ms as f64 / 1000.0;
                velocity_after = state6_chain.velocity_after;
                angular_velocity_after = state6_chain.angular_velocity_after;
                position_after = evaluate_position_integrator(
                    state_at_step_start.position,
                    velocity_after,
                    dt,
                    None,
                    0,
                    false,
                    0.0,
                    None,
                    None,
                )?
                .position_90_after;
                matrix_after = evaluate_orientation_integrator(
                    state_at_step_start.matrix,
                    angular_velocity_after,
                    dt,
                    false,
                    false,
                    6,
                    self.numeric_mode.is_float32(),
                )?
                .matrix_9c_after;
            }
        }
        // sub_B2FB00 partial-wheel bridge (:2269-2352): sweep the missing
        // wheel anchors vertically through A5B840; when every missing slot
        // finds a candidate, write the literal (0, 0, 0.180000007) to C0.
        let active_count = support
            .active_slots
            .iter()
            .filter(|active| **active)
            .count() as i64;
        let mut partial_wheel_bridge = RailState6PartialWheelBridge {
            selected: false,
            contact_188: contact_188_after,
            active_slots: support.active_slots,
            active_count,
            missing_wheel_queries: Vec::new(),
            velocity_delta: [0.0, 0.0, 0.0],
        };
        if contact_188_after && active_count > 0 && active_count < 4 {
            if !world_caps(world).sweep_point {
                return Err("B2FB00 partial-wheel bridge requires world.sweep_point".to_string());
            }
            let bridge_columns = matrix_columns(matrix_after);
            let bridge_lateral = bridge_columns[0];
            let bridge_forward = [
                -bridge_columns[1][0],
                -bridge_columns[1][1],
                -bridge_columns[1][2],
            ];
            let bridge_up = bridge_columns[2];
            let signs = [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)];
            let mut all_missing_hit = true;
            for (slot, (lateral_sign, forward_sign)) in signs.iter().enumerate() {
                if support.active_slots[slot] {
                    continue;
                }
                let anchor = wheel_anchor(
                    position_after,
                    bridge_lateral,
                    bridge_forward,
                    bridge_up,
                    parameters,
                    *lateral_sign,
                    *forward_sign,
                    self.numeric_mode,
                );
                let (start, end) = if self.numeric_mode.is_float32() {
                    let mut start = anchor;
                    let mut end = anchor;
                    start[2] = fadd(start[2], 1.0);
                    end[2] = fsub(end[2], 1.0);
                    (start, end)
                } else {
                    (
                        [anchor[0], anchor[1], anchor[2] + 1.0],
                        [anchor[0], anchor[1], anchor[2] - 1.0],
                    )
                };
                let candidate = world_ref.sweep_point(start, end).ok_or_else(|| {
                    "sweep_point capability reported present but returned no result".to_string()
                })?;
                let hit = candidate.get("hit").is_some_and(py_truthy);
                all_missing_hit = all_missing_hit && hit;
                partial_wheel_bridge
                    .missing_wheel_queries
                    .push(RailState6MissingWheelQuery {
                        slot,
                        start,
                        end,
                        candidate,
                        hit,
                    });
            }
            if all_missing_hit {
                let velocity_delta_z = if self.numeric_mode.is_float32() {
                    f32q(STATE6_PARTIAL_WHEEL_BRIDGE_DELTA_Z)
                } else {
                    STATE6_PARTIAL_WHEEL_BRIDGE_DELTA_Z
                };
                velocity_after[2] = if self.numeric_mode.is_float32() {
                    f32q(f32q(velocity_after[2]) + velocity_delta_z)
                } else {
                    velocity_after[2] + velocity_delta_z
                };
                partial_wheel_bridge.selected = true;
                partial_wheel_bridge.velocity_delta = [0.0, 0.0, velocity_delta_z];
            }
        }
        self.state.position = position_after;
        self.state.velocity = velocity_after;
        self.state.angular_velocity = angular_velocity_after;
        self.state.matrix = matrix_after;
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.contact_compression = support.producer.contact_compression_after;
        self.state.time_ms += parameters.substep_ms as u64;
        self.state.step_index += 1;
        self.state.topology.domain_step_index += 1;
        self.state.topology.rail_state_elapsed_ms += parameters.substep_ms as f64;
        self.state.soft_flag_188 = contact_188_after;
        // Python mutates the evaluator result dict in place through the
        // collision chain and the partial-wheel bridge (:2239-2267, :2341):
        // the published `position_after`/`velocity_after`/
        // `angular_velocity_after`/`matrix_after` keys are the final values.
        let mut result = result;
        result.position_after = position_after;
        result.velocity_after = velocity_after;
        result.angular_velocity_after = angular_velocity_after;
        result.matrix_after = matrix_after;
        let mut rail_release: Option<Box<RailReleaseOutcome>> = None;
        let mut release_destination: Option<String> = None;
        let mut release_topology_decision: Option<TopologyDecision> = None;
        let mut rail_release_pending_producer: Option<RailReleasePendingProducer> = None;
        if selector.transitioned {
            let release = self.step_raw_rail(
                keys,
                parameters,
                None,
                false,
                None,
                support.support_contact_count,
                support.surface_mode,
                None,
                None,
                None,
                None,
                None,
            )?;
            let RawRailOutcome::Released(mut release) = release else {
                // Unreachable: rail_contact=false yields the release outcome.
                return Err("state6 release produced an active rail outcome".to_string());
            };

            // A completed exclusive RAIL constraint supersedes a hard-
            // takeoff separation transaction retained from before capture.
            // Leaving it armed suppresses the freshly sampled four-wheel
            // manifold on the next topology step and invents GND->AIR.
            let takeoff_separation_pending_before_release = self.state.takeoff_separation_pending;
            self.state.takeoff_separation_pending = false;

            // B2D940 consumes the accumulator published by the preceding
            // epoch. State 6 consumed its inline support producer and cleared
            // the active snapshot above, so publish the next producer from
            // the finalized pose exactly like Python's post-pose ground
            // refresh. Otherwise the first post-RAIL step sees literal zero
            // +0x614 and selects B2D940's C0-clear branch.
            let destination = self.state.topology.domain;
            let grounded_release = matches!(
                destination,
                MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
            );
            let plane_set =
                grounded_release.then(|| ground_plane_set_from_world_planes(&support.planes));
            let pending = self.produce_ground_accumulators(
                keys,
                parameters,
                plane_set.as_ref(),
                Some(grounded_release),
                false,
            )?;
            self.state.force_accumulator = pending
                .evaluator
                .accumulator_614_after_damping
                .ok_or_else(|| {
                    "release producer missing accumulator_614_after_damping".to_string()
                })?;
            self.state.torque_accumulator = pending
                .evaluator
                .accumulator_620_after_damping
                .ok_or_else(|| {
                    "release producer missing accumulator_620_after_damping".to_string()
                })?;
            self.state.contact_compression = pending.contact_compression_after;
            if self.numeric_mode.is_float32() {
                f32_vector(&mut self.state.force_accumulator);
                f32_vector(&mut self.state.torque_accumulator);
                f32_vector(&mut self.state.contact_compression);
            }
            let pending_payload = RailReleasePendingProducer {
                destination: destination.value().to_string(),
                force_accumulator: self.state.force_accumulator,
                torque_accumulator: self.state.torque_accumulator,
                contact_compression: self.state.contact_compression,
                takeoff_separation_pending_before_release,
                takeoff_separation_pending_after_release: false,
                authority: "post_pose_next_epoch_producer_and_exclusive_constraint_handoff",
            };
            release.state_after = self.state.clone();
            release.rail_release_pending_producer = Some(pending_payload.clone());
            release_destination = Some(release.release_destination.clone());
            release_topology_decision = Some(release.topology_decision.clone());
            rail_release_pending_producer = Some(pending_payload);
            rail_release = Some(release);
        }
        Ok(RailState6TailOutcome {
            rail_state6_contact_physics: contact_accumulator_present,
            rail_state6_common_angular_damping: state6_common_angular,
            rail_state6_support_force_branch: state6_support_force_branch,
            rail_state6_support_force: RailState6SupportForce {
                force: additional_force,
                torque_after_state6_angular: additional_torque,
            },
            rail_state6_entry_force_transaction: entry_force_transaction,
            rail_state6_collision_schedule: if collision_chain.is_empty() {
                "autonomous_rotating_obb_no_hard_contact"
            } else {
                "autonomous_rotating_obb_hard_contact_chain"
            },
            rail_state6_collision_chain: collision_chain,
            rail_state6_partial_wheel_bridge: partial_wheel_bridge,
            rail_state6_exit: RailState6Exit {
                selector,
                float_zero_boundary_writer_committed,
            },
            state6_contact_slots,
            state6_contact_candidate: RailState6ContactCandidate {
                segment_start: state6_segment_start,
                segment_end: state6_segment_end,
                candidate: state6_contact_candidate,
                valid: state6_contact_valid,
                selected_source: state6_contact_source,
                contact_188_before,
                contact_188_after,
            },
            rail_released: rail_release.is_some(),
            release_destination,
            topology_decision: release_topology_decision,
            rail_release,
            rail_release_pending_producer,
            airborne: result,
            support_observation: support,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
            rail_contract: RailStepContract {
                control_authority: "raw_keys",
                rigid_body_state_authority: "engine_owned_common_airborne_chain",
                writer_selection_authority: "rail_state6_B276F0_contact_timer_tail",
                rail_context_authority:
                    "autonomous_B276F0_contact188_18c_or_A5B840_center_segment_candidate_until_contact_timer_exit",
                open_boundary:
                    "game_A5B840_candidate_identity_runtime_equivalence_and_time_varying_contact_188_runtime_BA0_selector",
            },
        })
    }

    /// `_advance_autonomous_rail_selector`
    /// (kartrider_vehicle_engine.py:3315-3432): run state1/2/5 transitions
    /// from current engine/world evidence.
    pub fn advance_autonomous_rail_selector(
        &mut self,
        selector: &BTreeMap<String, JsonValue>,
        parameters: &VehicleParameters,
    ) -> Result<Option<AutonomousRailSelectorAdvance>, String> {
        let state = self.state.topology.rail_state;
        if !matches!(state, 1 | 2 | 5) {
            return Ok(None);
        }
        let speed = length3(self.state.velocity, self.numeric_mode.is_float32());
        let elapsed = self.state.topology.rail_state_elapsed_ms;
        if state == 1 {
            // Python authority: the B29350 state-1 decision is published
            // after this substep's ground writer. The retained clock contains
            // completed widths only, so the 10 ms gate observes the current
            // substep as well.
            let selector_elapsed = elapsed + parameters.substep_ms as f64;
            let mut autonomous_query: Option<RailState1QueryResult> = None;
            let mut gate: Option<RailState1GateView> = match selector.get("state1_gate") {
                None | Some(JsonValue::Null) => None,
                Some(value) => Some(rail_state1_gate_view_from_json(value)?),
            };
            if gate.is_none()
                && selector
                    .get("registered_query_contexts")
                    .is_some_and(|value| !matches!(value, JsonValue::Null))
            {
                let contexts = rail_state1_contexts_from_json(
                    selector
                        .get("registered_query_contexts")
                        .expect("checked is_some"),
                )?;
                let query = evaluate_rail_state1_query(
                    &contexts,
                    self.state.position,
                    selector.get("contact_exists").is_none_or(py_truthy),
                    selector
                        .get("contact_identifier_nonempty")
                        .is_none_or(py_truthy),
                    number_or(selector.get("metric_threshold"), 4.0),
                    number_or(selector.get("height_bias"), 1.0),
                    number_or(selector.get("alignment_base_ms"), 0.0),
                    match selector.get("alignment_primary_scale_ms") {
                        Some(JsonValue::Number(value)) => Some(*value),
                        _ => None,
                    },
                    match selector.get("alignment_stable_initial_ms") {
                        Some(JsonValue::Number(value)) => Some(*value),
                        _ => None,
                    },
                    None,
                )?;
                gate = query
                    .state1_gate
                    .as_ref()
                    .map(rail_state1_gate_view_from_query);
                autonomous_query = Some(query);
            }
            let gate = gate.unwrap_or_default();
            if gate.nonempty
                && !gate
                    .authority
                    .as_deref()
                    .is_some_and(|authority| STATE1_GATE_AUTHORITIES.contains(&authority))
            {
                return Err(
                    "state1_gate requires exact candidate interpolation authority".to_string(),
                );
            }
            let mut transition =
                self.advance_rail_capture_selector(&AdvanceRailCaptureSelectorArgs {
                    elapsed_in_state_ms: selector_elapsed,
                    field_af0: speed,
                    contact_exists: gate.contact_exists,
                    contact_identifier_nonempty: gate.contact_identifier_nonempty,
                    path_metric: gate.path_metric,
                    metric_threshold: gate.metric_threshold,
                    path_z: gate.path_z,
                    kart_z_plus_one: gate.kart_z_plus_one,
                    alignment_base_ms: gate.alignment_base_ms,
                    alignment_primary_scale_ms: gate.alignment_primary_scale_ms,
                    alignment_stable_initial_ms: gate.alignment_stable_initial_ms,
                    ..AdvanceRailCaptureSelectorArgs::default()
                })?;
            if let Some(query) = &autonomous_query {
                let query_gate = query
                    .state1_gate
                    .as_ref()
                    .map(rail_state1_gate_view_from_query)
                    .unwrap_or_default();
                if transition.selector.state_after == 2
                    && self.state.rail_selector_runtime.is_some()
                {
                    if let Some(runtime) = self.state.rail_selector_runtime.as_mut() {
                        runtime.path_context_id = query_gate
                            .selected_context
                            .as_ref()
                            .map(rail_path_context_id_from_json);
                    }
                    // Python `transition["state_after"] = asdict(self.state)`
                    // (:3402): refresh after the path_context_id write.
                    transition.state_after = self.state.clone();
                }
            }
            return Ok(Some(AutonomousRailSelectorAdvance {
                advance: transition,
                autonomous_state1_query: autonomous_query,
            }));
        }
        if state == 2 {
            let columns = matrix_columns(self.state.matrix);
            let forward_axis = columns[parameters.forward_axis_column as usize];
            let forward = [
                parameters.forward_axis_sign * forward_axis[0],
                parameters.forward_axis_sign * forward_axis[1],
                parameters.forward_axis_sign * forward_axis[2],
            ];
            let lateral = columns[0];
            let (forward_speed, lateral_speed) = if self.numeric_mode.is_float32() {
                (
                    dot3_f32(self.state.velocity, forward),
                    dot3_f32(self.state.velocity, lateral),
                )
            } else {
                (
                    dot3_sum(self.state.velocity, forward),
                    dot3_sum(self.state.velocity, lateral),
                )
            };
            let transition =
                self.advance_rail_capture_selector(&AdvanceRailCaptureSelectorArgs {
                    elapsed_in_state_ms: elapsed,
                    field_af0: speed,
                    delta_ms: Some(parameters.substep_ms as f64),
                    forward_speed: Some(forward_speed),
                    lateral_speed: Some(lateral_speed),
                    ..AdvanceRailCaptureSelectorArgs::default()
                })?;
            return Ok(Some(AutonomousRailSelectorAdvance {
                advance: transition,
                autonomous_state1_query: None,
            }));
        }
        let transition = self.advance_rail_capture_selector(&AdvanceRailCaptureSelectorArgs {
            elapsed_in_state_ms: elapsed,
            field_af0: speed,
            release_support_contact_count: json_int_or(
                selector.get("release_support_contact_count"),
            ),
            release_surface_mode: json_int_or(selector.get("release_surface_mode")),
            ..AdvanceRailCaptureSelectorArgs::default()
        })?;
        Ok(Some(AutonomousRailSelectorAdvance {
            advance: transition,
            autonomous_state1_query: None,
        }))
    }

    /// `_query_autonomous_rail_authored_trigger`
    /// (kartrider_vehicle_engine.py:3434-3527): generate an exact JM/DJ
    /// rail-exit trigger from current OBB overlaps.
    pub fn query_autonomous_rail_authored_trigger(
        &self,
        world: Option<&dyn World>,
        parameters: &VehicleParameters,
        center_up: Option<f64>,
    ) -> Result<RailAuthoredTriggerGeometry, String> {
        if !world_caps(world).overlap_obb_all {
            return Ok(RailAuthoredTriggerGeometry {
                authored_trigger: None,
                authority: "world_has_no_obb_overlap_query".to_string(),
                center: None,
                candidate: None,
                rejected_jm_without_contact_basis: None,
            });
        }
        let columns = matrix_columns(self.state.matrix);
        let up = columns[2];
        let center_offset = center_up.unwrap_or(parameters.collision_center_up);
        let center = [
            self.state.position[0] + up[0] * center_offset,
            self.state.position[1] + up[1] * center_offset,
            self.state.position[2] + up[2] * center_offset,
        ];
        let candidates = world
            .expect("capability probe implies a world")
            .overlap_obb_all(center, self.state.matrix, parameters.collision_half_extents)
            .ok_or_else(|| {
                "overlap_obb_all capability reported present but returned no result".to_string()
            })?;
        let mut matches: Vec<AuthoredTriggerMatch> = Vec::new();
        let mut rejected_jm_without_contact_basis = 0_i64;
        for candidate in &candidates {
            if !candidate.hit {
                continue;
            }
            let token = candidate.road_surface.trim().to_string();
            let Ok(decoded) = parse_jm_dj_surface_token(&token) else {
                continue;
            };
            let mut trigger = BTreeMap::new();
            trigger.insert(
                "road_surface".to_string(),
                JsonValue::String(candidate.road_surface.clone()),
            );
            trigger.insert(
                "boost_multiplier_64c".to_string(),
                JsonValue::Bool(candidate.boost_multiplier_64c),
            );
            let (token_key, variant): (String, &str) = match &decoded {
                JmDjSurfaceToken::Jm { token, .. } => {
                    // A JM candidate is eligible only with the exact B28D30
                    // CollisionFace+0x44 contact-record basis (:3482-3497).
                    match candidate.contact_basis_44 {
                        Some(basis) if basis.iter().all(|value| value.is_finite()) => {
                            trigger.insert(
                                "jm_basis_190".to_string(),
                                JsonValue::Array(
                                    basis
                                        .iter()
                                        .map(|value| JsonValue::Number(*value))
                                        .collect(),
                                ),
                            );
                        }
                        _ => {
                            rejected_jm_without_contact_basis += 1;
                            continue;
                        }
                    }
                    (token.clone(), "JM")
                }
                JmDjSurfaceToken::Dj { token, .. } => (token.clone(), "DJ"),
            };
            let key = (token_key, candidate.surface_id);
            // Python `matches.setdefault(key, ...)`: the first candidate for
            // a key wins; dict insertion order is preserved.
            if !matches.iter().any(|(existing, _)| *existing == key) {
                matches.push((key, (candidate.clone(), trigger, variant)));
            }
        }
        if matches.len() > 1 {
            return Err("multiple distinct autonomous rail DJ triggers overlap".to_string());
        }
        if matches.is_empty() {
            return Ok(RailAuthoredTriggerGeometry {
                authored_trigger: None,
                authority: "map_obb_overlap_no_exact_jm_dj".to_string(),
                center: Some(center),
                candidate: None,
                rejected_jm_without_contact_basis: Some(rejected_jm_without_contact_basis),
            });
        }
        let (_, (candidate, trigger, variant)) =
            matches.into_iter().next().expect("non-empty matches");
        Ok(RailAuthoredTriggerGeometry {
            authored_trigger: Some(trigger),
            authority: if variant == "DJ" {
                "map_obb_overlap_authored_dj".to_string()
            } else {
                "map_obb_overlap_authored_jm_contact_basis_44".to_string()
            },
            center: Some(center),
            candidate: Some(candidate),
            rejected_jm_without_contact_basis: None,
        })
    }

    /// `_evaluate_rail_authored_trigger`
    /// (kartrider_vehicle_engine.py:3529-3587): decode an autonomous JM/DJ
    /// contact and evaluate the state3 edge. The `isinstance(trigger, dict)`
    /// guard is type-enforced by the `&BTreeMap` parameter.
    pub fn evaluate_rail_authored_trigger(
        &self,
        trigger: &BTreeMap<String, JsonValue>,
        delta_ms: f64,
    ) -> Result<RailAuthoredTriggerEdge, String> {
        let token = match trigger.get("road_surface") {
            None | Some(JsonValue::Null) => {
                return Err("rail authored_trigger requires road_surface token".to_string());
            }
            Some(value) => json_py_str(value),
        };
        let decoded = parse_jm_dj_surface_token(&token)?;
        let special_state_family_already_active = (13..17).contains(&self.state.boost_state);
        let boost_multiplier_64c = trigger.get("boost_multiplier_64c").is_some_and(py_truthy);
        let (
            variant,
            velocity_c0_before,
            jm_basis_190,
            jm_horizontal_param,
            jm_basis_param,
            dj_vector,
            authored_token,
        ) = match &decoded {
            JmDjSurfaceToken::Jm {
                jm_horizontal_param,
                jm_basis_param,
                token,
            } => {
                let basis = match trigger.get("jm_basis_190") {
                    Some(JsonValue::Array(values)) if values.len() == 3 => [
                        number_from_json(values.first()),
                        number_from_json(values.get(1)),
                        number_from_json(values.get(2)),
                    ],
                    _ => {
                        return Err("JM rail trigger requires decoded jm_basis_190".to_string());
                    }
                };
                (
                    "JM",
                    Some(self.state.velocity),
                    Some(basis),
                    Some(*jm_horizontal_param),
                    Some(*jm_basis_param),
                    None,
                    token.clone(),
                )
            }
            JmDjSurfaceToken::Dj { dj_vector, token } => (
                "DJ",
                None,
                None,
                None,
                None,
                Some(*dj_vector),
                token.clone(),
            ),
        };
        let selector_result = evaluate_rail_capture_selector(&RailCaptureSelectorArgs {
            state_before: 3,
            elapsed_in_state_ms: 0.0,
            field_af0: length3(self.state.velocity, self.numeric_mode.is_float32()),
            delta_ms: Some(delta_ms),
            trigger_cooldown_remaining_ms: Some(self.state.special_trigger_cooldown_ms),
            authored_trigger_accepted: true,
            ..RailCaptureSelectorArgs::default()
        })?;
        let trigger_result = evaluate_jm_dj_trigger(
            variant,
            special_state_family_already_active,
            velocity_c0_before,
            jm_basis_190,
            jm_horizontal_param,
            jm_basis_param,
            dj_vector,
            boost_multiplier_64c,
        )?;
        Ok(RailAuthoredTriggerEdge {
            selector: selector_result,
            trigger: trigger_result,
            authored_token,
            selection_authority: "world_query_rail_contact_authored_trigger",
        })
    }

    /// `step_autonomous_rail` (kartrider_vehicle_engine.py:2661-3313):
    /// advance or release current RAIL ownership from world selectors.
    #[allow(clippy::too_many_lines)]
    pub fn step_autonomous_rail(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
    ) -> Result<AutonomousRailOutcome, String> {
        if self.state.topology.domain != MotionDomain::Rail {
            return Err("autonomous rail step requires RAIL topology".to_string());
        }
        if !world_caps(world).query_rail_contact {
            return Err("autonomous rail step requires world.query_rail_contact".to_string());
        }
        validate_raw_control_keys(keys)?;
        if self.state.topology.rail_state == 6 {
            return Ok(AutonomousRailOutcome::State6Tail(Box::new(
                self.step_rail_state6_tail(keys, parameters, world)?,
            )));
        }
        let world_ref = world.expect("capability probe implies a world");
        let selector = world_ref
            .query_rail_contact(
                self.state.position,
                self.state.topology.rail_state,
                self.state.topology.constraint_owner_id.as_deref(),
            )
            .ok_or_else(|| {
                "query_rail_contact capability reported present but returned no result".to_string()
            })?;
        let active = selector
            .as_ref()
            .is_some_and(|selector| selector.get("contact_active").is_none_or(py_truthy));
        if !active {
            let support = self.inspect_autonomous_support(&[], parameters, world, None)?;
            if Self::support_selects_authored_reset(&support) {
                let reset = self.enter_authored_reset_surface(support)?;
                return Ok(AutonomousRailOutcome::ResetRelease(Box::new(
                    AutonomousRailResetReleaseOutcome {
                        reset,
                        autonomous_rail_selector: selector,
                        rail_contract: RailStepContract {
                            control_authority: "raw_keys",
                            rigid_body_state_authority: "authoritative_reset_transaction",
                            writer_selection_authority: "no_rail_writer_after_release",
                            rail_context_authority:
                                "world_query_rail_contact_negative_then_reset_support",
                            open_boundary:
                                "autonomous map-side rail and JM/DJ contact candidate generation",
                        },
                    },
                )));
            }
            let raw = self.step_raw_rail(
                keys,
                parameters,
                None,
                false,
                None,
                support.support_contact_count,
                support.surface_mode,
                None,
                None,
                None,
                None,
                None,
            )?;
            let RawRailOutcome::Released(mut release) = raw else {
                // Unreachable: rail_contact=false yields the release outcome.
                return Err("rail release produced an active rail outcome".to_string());
            };
            release.rail_contract.rail_context_authority =
                "world_query_rail_contact_negative_then_four_wheel_support";
            return Ok(AutonomousRailOutcome::ContactRelease(Box::new(
                AutonomousRailContactReleaseOutcome {
                    release: *release,
                    autonomous_rail_selector: selector,
                    support_observation: support,
                },
            )));
        }
        let selector = selector.expect("active implies a selector mapping");
        let rail_state_queried = match selector.get("rail_state") {
            None => self.state.topology.rail_state,
            Some(JsonValue::Number(value)) => *value as i64,
            Some(JsonValue::Bool(value)) => i64::from(*value),
            Some(_) => self.state.topology.rail_state,
        };
        if rail_state_queried != self.state.topology.rail_state {
            return Err(
                "world rail selector cannot publish a state transition; supply current-state \
                 gate inputs instead"
                    .to_string(),
            );
        }
        if rail_state_queried == 1 && world.is_some() {
            // State 1 owns capture bookkeeping while the ordinary ground
            // chain remains the physical writer; the state-2 predicate is
            // evaluated *after* that writer (:2731-2740).
            let support = self.inspect_autonomous_support(keys, parameters, world, None)?;
            let pending_provider = |state: &VehicleState, pending: &VehicleParameters| {
                Some(ground_plane_set_from_world_planes(&query_wheel_ground(
                    world_ref, state, pending,
                )))
            };
            let ground = self.step_raw_ground_correction(
                keys,
                parameters,
                world,
                Some(&ground_plane_set_from_world_planes(&support.planes)),
                &GroundStepOptions {
                    pending_ground_plane_provider: Some(&pending_provider),
                    rail_state1_ground_writer: true,
                    ..GroundStepOptions::default()
                },
            )?;
            // B2EA20 post-linear/pre-pose low-speed horizontal transaction
            // (:2754-2835). All quantization here is unconditional _f32.
            let mut post_linear: Option<RailState1PostLinearCollisionResponse> = None;
            let world_sweep = ground.world_sweep.as_ref();
            let empty_runtime = BTreeMap::new();
            let rail_runtime = match selector.get("runtime") {
                Some(JsonValue::Object(map)) => map,
                _ => &empty_runtime,
            };
            let rail_direction: Vec<f64> = match rail_runtime.get("rail_dir") {
                Some(JsonValue::Array(values)) => values
                    .iter()
                    .map(|value| number_from_json(Some(value)))
                    .collect(),
                _ => Vec::new(),
            };
            let horizontal_direction_norm = if rail_direction.len() == 3 {
                cpython_hypot2(rail_direction[0], rail_direction[1])
            } else {
                0.0
            };
            let path_speed = if horizontal_direction_norm > 1.0e-9 {
                (self.state.velocity[0] * rail_direction[0]
                    + self.state.velocity[1] * rail_direction[1])
                    / horizontal_direction_norm
            } else {
                0.0
            };
            let low_speed_limit = number_or(rail_runtime.get("min_velocity"), 0.0) * 0.1;
            if world_sweep
                .and_then(|sweep| sweep.candidate.as_ref())
                .is_some_and(|candidate| candidate.road_rail == "highSpeed")
                && !world_sweep.is_some_and(|sweep| sweep.response_dispatched)
                && world_sweep.is_some_and(|sweep| sweep.dispatch_reason == "separating_or_tangent")
                && low_speed_limit > 0.0
                && path_speed.abs() < low_speed_limit
            {
                let dt = f32q(parameters.substep_ms as f64 / 1000.0);
                let gain = f32q(dt * RAIL_CAPTURE_POSITION_GAIN_PER_SECOND);
                let delta_magnitude = f32q(path_speed * gain);
                let direction = [
                    f32q(rail_direction[0] / horizontal_direction_norm),
                    f32q(rail_direction[1] / horizontal_direction_norm),
                    0.0,
                ];
                let mut velocity_delta = [
                    direction[0] * delta_magnitude,
                    direction[1] * delta_magnitude,
                    0.0,
                ];
                f32_vector(&mut velocity_delta);
                let mut velocity = [
                    self.state.velocity[0] + velocity_delta[0],
                    self.state.velocity[1] + velocity_delta[1],
                    self.state.velocity[2] + velocity_delta[2],
                ];
                f32_vector(&mut velocity);
                self.state.velocity = velocity;
                // This writer precedes B2FD80, so its C0 delta contributes
                // to the current pose transaction as well (:2813-2819).
                let mut position = [
                    self.state.position[0] + velocity_delta[0] * dt,
                    self.state.position[1] + velocity_delta[1] * dt,
                    self.state.position[2] + velocity_delta[2] * dt,
                ];
                f32_vector(&mut position);
                self.state.position = position;
                post_linear = Some(RailState1PostLinearCollisionResponse {
                    velocity_delta,
                    path_speed,
                    gain_per_second: RAIL_CAPTURE_POSITION_GAIN_PER_SECOND,
                    low_speed_limit,
                });
            }
            self.state.topology.domain_step_index += 1;
            let post_selector = world_ref
                .query_rail_contact(
                    self.state.position,
                    1,
                    self.state.topology.constraint_owner_id.as_deref(),
                )
                .ok_or_else(|| {
                    "query_rail_contact capability reported present but returned no result"
                        .to_string()
                })?;
            let post_active = post_selector
                .as_ref()
                .is_some_and(|selector| selector.get("contact_active").is_none_or(py_truthy));
            if !post_active {
                // Python RuntimeError.
                return Err(
                    "state1 ground writer lost retained rail contact before post-writer \
                     selector evaluation"
                        .to_string(),
                );
            }
            let post_selector = post_selector.expect("post_active implies a selector");
            let post_rail_state = match post_selector.get("rail_state") {
                None => 1,
                Some(JsonValue::Number(value)) => *value as i64,
                Some(JsonValue::Bool(value)) => i64::from(*value),
                Some(_) => 1,
            };
            if post_rail_state != 1 {
                return Err(
                    "post-writer world rail selector cannot publish a state transition; \
                     supply current-state gate inputs instead"
                        .to_string(),
                );
            }
            let selector_transition =
                self.advance_autonomous_rail_selector(&post_selector, parameters)?;
            let selected_state_after = self.state.topology.rail_state;
            return Ok(AutonomousRailOutcome::State1GroundWriter(Box::new(
                AutonomousRailState1Outcome {
                    ground,
                    rail_state1_post_linear_collision_response: post_linear,
                    rail_state1_ground_writer: RailState1GroundWriter {
                        selected: true,
                        state_after: selected_state_after,
                        support_contact_count: support.support_contact_count,
                        selector_phase: Some("post_ground_writer"),
                        authority:
                            "run119_state1_ground_kind42_30_5_1_6_2_36_then_same_epoch_state2_store",
                    },
                    autonomous_rail_selector: Some(post_selector),
                    support_observation: support,
                    rail_state_transition: selector_transition,
                    topology_after: self.state.topology.clone(),
                    state_after: self.state.clone(),
                    rail_contract: RailStepContract {
                        control_authority: "raw_keys",
                        rigid_body_state_authority: "engine_owned_state1_ground_capture_chain",
                        writer_selection_authority:
                            "runtime_state1_ground_then_post_writer_state2_store",
                        rail_context_authority:
                            "post_writer_world_query_inputs_to_exact_rail_selector",
                        open_boundary: "autonomous state1 path candidate list generation",
                    },
                },
            )));
        }
        let selector_transition = self.advance_autonomous_rail_selector(&selector, parameters)?;
        let rail_state = self.state.topology.rail_state;
        if rail_state == 0 {
            return Ok(AutonomousRailOutcome::SelectorRelease(Box::new(
                AutonomousRailSelectorReleaseOutcome {
                    release_destination: self.state.topology.domain.value().to_string(),
                    autonomous_rail_selector: selector,
                    rail_state_transition: selector_transition,
                    topology_after: self.state.topology.clone(),
                    state_after: self.state.clone(),
                    rail_contract: RailStepContract {
                        control_authority: "raw_keys",
                        rigid_body_state_authority: "engine_owned_selector_release",
                        writer_selection_authority: "no_rail_writer_after_state_machine_release",
                        rail_context_authority: "world_query_inputs_to_exact_rail_selector",
                        open_boundary: "autonomous state1 path candidate list generation",
                    },
                },
            )));
        }
        let mut runtime = Self::decode_active_rail_runtime(&selector)?;
        let mut path_velocity_transfer: Option<RailPathVelocityTransfer> = None;
        let mut path_velocity_override: Option<[f64; 3]> = None;
        let mut path_gate: Option<RailState1GateView> = None;
        let transition_is_1_to_2 = selector_transition.as_ref().is_some_and(|transition| {
            transition.advance.selector.state_before == 1
                && transition.advance.selector.state_after == 2
        });
        if transition_is_1_to_2 {
            let transition = selector_transition.as_ref().expect("checked is_some");
            // Python `gate = query.get("state1_gate") or
            // selector.get("state1_gate") or {}` (:2936-2941): the first
            // present-and-nonempty gate wins.
            let query_gate = transition
                .autonomous_state1_query
                .as_ref()
                .and_then(|query| query.state1_gate.as_ref())
                .map(rail_state1_gate_view_from_query);
            let json_gate = match selector.get("state1_gate") {
                None | Some(JsonValue::Null) => None,
                Some(value) => Some(rail_state1_gate_view_from_json(value)?),
            };
            let gate = match (query_gate, json_gate) {
                (Some(gate), _) if gate.nonempty => gate,
                (_, Some(gate)) if gate.nonempty => gate,
                _ => RailState1GateView::default(),
            };
            path_gate = Some(gate.clone());
            let Some(min_velocity) = runtime.min_velocity else {
                return Err(
                    "state1-to-state2 velocity transfer requires decoded rail min_velocity"
                        .to_string(),
                );
            };
            let lookahead = match &gate.lookahead_endpoint {
                Some(values) if values.len() == 3 => [values[0], values[1], values[2]],
                _ => {
                    return Err(
                        "state1-to-state2 velocity transfer requires the autonomous \
                         one-metre lookahead endpoint"
                            .to_string(),
                    );
                }
            };
            let transfer = evaluate_rail_state1_velocity_transfer(
                self.state.position,
                lookahead,
                min_velocity,
                Some(self.state.velocity),
                runtime.max_velocity,
            )?;
            path_velocity_override = Some(transfer.velocity_after);
            path_velocity_transfer = Some(RailPathVelocityTransfer {
                state_before: 1,
                state_after: 2,
                transfer,
                lookahead_distance: gate.lookahead_distance,
                velocity_before: Some(self.state.velocity),
                query_scalar: None,
                selected_context: None,
                context_transfer: None,
                rail_writer_direction: [0.0, 0.0, 0.0],
                velocity_writer: "state1_to_state2_B29350_entry",
            });
        } else if matches!(rail_state, 2 | 3)
            && selector
                .get("registered_query_contexts")
                .is_some_and(|value| !matches!(value, JsonValue::Null))
        {
            let contexts = rail_state1_contexts_from_json(
                selector
                    .get("registered_query_contexts")
                    .expect("checked is_some"),
            )?;
            let owned_context = self
                .state
                .rail_selector_runtime
                .as_ref()
                .and_then(|runtime| runtime.path_context_id.clone());
            let Some(owned_context) = owned_context else {
                return Err("state2/state3 rail sustain has no retained path context".to_string());
            };
            let owned_json = path_context_id_to_json(&owned_context);
            let owned_matches: Vec<&RailState1Context> = contexts
                .iter()
                .filter(|context| {
                    context.context_id.clone().or_else(|| {
                        context
                            .context_index
                            .map(|index| JsonValue::Number(index as f64))
                    }) == Some(owned_json.clone())
                })
                .collect();
            if owned_matches.len() != 1 {
                return Err("retained rail path context must resolve exactly once".to_string());
            }
            let query = evaluate_rail_state1_query(
                &contexts,
                self.state.position,
                true,
                true,
                number_or(selector.get("metric_threshold"), 4.0),
                number_or(selector.get("height_bias"), 1.0),
                0.0,
                None,
                None,
                Some(owned_json.clone()),
            )?;
            let gate = query
                .state1_gate
                .as_ref()
                .map(rail_state1_gate_view_from_query)
                .unwrap_or_default();
            path_gate = Some(gate.clone());
            let Some(min_velocity) = runtime.min_velocity else {
                return Err("state2/state3 rail sustain requires decoded min_velocity".to_string());
            };
            let lookahead = match &gate.lookahead_endpoint {
                Some(values) if values.len() == 3 => [values[0], values[1], values[2]],
                _ => {
                    return Err(
                        "state2/state3 rail sustain requires an autonomous one-metre \
                         lookahead endpoint"
                            .to_string(),
                    );
                }
            };
            let transfer = evaluate_rail_state1_velocity_transfer(
                self.state.position,
                lookahead,
                min_velocity,
                Some(self.state.velocity),
                runtime.max_velocity,
            )?;
            path_velocity_override = Some(transfer.velocity_after);
            let mut context_transfer = None;
            if let Some(advance_context) = &gate.advance_context {
                if let Some(runtime_state) = self.state.rail_selector_runtime.as_mut() {
                    runtime_state.path_context_id =
                        Some(rail_path_context_id_from_json(advance_context));
                }
                context_transfer = Some(RailPathContextTransfer {
                    from: owned_json,
                    to: advance_context.clone(),
                    authority: "authored_same_road_endpoint_continuity",
                });
            }
            path_velocity_transfer = Some(RailPathVelocityTransfer {
                state_before: rail_state,
                state_after: rail_state,
                transfer,
                lookahead_distance: gate.lookahead_distance,
                velocity_before: None,
                query_scalar: gate.query_scalar,
                selected_context: gate.selected_context.clone(),
                context_transfer,
                rail_writer_direction: [0.0, 0.0, 0.0],
                velocity_writer: "B29350_continuous_kind11",
            });
        }
        if let Some(transfer) = path_velocity_transfer.as_mut() {
            let path_velocity = transfer.transfer.velocity_after;
            let path_speed = length3(path_velocity, self.numeric_mode.is_float32());
            if path_speed <= 0.0 {
                return Err("autonomous rail path velocity must be nonzero".to_string());
            }
            runtime.rail_dir = if self.numeric_mode.is_float32() {
                [
                    fdiv(path_velocity[0], path_speed),
                    fdiv(path_velocity[1], path_speed),
                    fdiv(path_velocity[2], path_speed),
                ]
            } else {
                [
                    path_velocity[0] / path_speed,
                    path_velocity[1] / path_speed,
                    path_velocity[2] / path_speed,
                ]
            };
            runtime.validate()?;
            transfer.rail_writer_direction = runtime.rail_dir;
        }
        if let Some(gate) = &path_gate {
            if let Some(base_matrix) = &gate.rail_base_orientation_matrix_bf8 {
                let Some(base_matrix) = base_matrix else {
                    return Err(
                        "autonomous rail path query returned invalid BF8 matrix".to_string()
                    );
                };
                self.state.rail_base_bf8 = Some(*base_matrix);
                if self.state.rail_constraint_c1c.is_none() {
                    // Before the first relative writer, preserve the current
                    // primary pose by expressing it in the new path frame
                    // (:3088-3106). `sum(...)` is CPython Neumaier.
                    let base = base_matrix;
                    let primary = self.state.matrix;
                    let mut c1c = [0.0; 9];
                    for row in 0..3 {
                        for column in 0..3 {
                            c1c[row * 3 + column] = sum_f64(&[
                                base[row] * primary[column],
                                base[3 + row] * primary[3 + column],
                                base[6 + row] * primary[6 + column],
                            ]);
                        }
                    }
                    self.state.rail_constraint_c1c = Some(c1c);
                }
            }
        }
        let mut authored_trigger: Option<BTreeMap<String, JsonValue>> =
            match selector.get("authored_trigger") {
                None | Some(JsonValue::Null) => None,
                Some(JsonValue::Object(map)) => Some(map.clone()),
                Some(_) => {
                    if rail_state == 3 {
                        return Err("rail authored_trigger must be a mapping".to_string());
                    }
                    None
                }
            };
        let mut trigger_geometry: Option<RailAuthoredTriggerGeometry> = None;
        if rail_state == 3 && authored_trigger.is_none() {
            let geometry = self.query_autonomous_rail_authored_trigger(world, parameters, None)?;
            authored_trigger = geometry.authored_trigger.clone();
            trigger_geometry = Some(geometry);
        }
        let mut trigger_edge: Option<RailAuthoredTriggerEdge> = None;
        if rail_state == 3 {
            if let Some(trigger) = &authored_trigger {
                trigger_edge = Some(
                    self.evaluate_rail_authored_trigger(trigger, parameters.substep_ms as f64)?,
                );
            }
        }
        let terminal_endpoint_edge = rail_state == 3
            && path_gate
                .as_ref()
                .is_some_and(|gate| gate.terminal_state6_edge);
        if terminal_endpoint_edge
            && trigger_edge
                .as_ref()
                .is_some_and(|edge| edge.selector.transitioned)
        {
            return Err(
                "ambiguous rail state3 exit: authored JM/DJ trigger and terminal path \
                 endpoint selected in the same substep"
                    .to_string(),
            );
        }
        let enters_state6 = terminal_endpoint_edge
            || trigger_edge
                .as_ref()
                .is_some_and(|edge| edge.selector.transitioned);
        let mut state6_entry_force: Option<[f64; 3]> = None;
        let mut rail_support: Option<SupportObservation> = None;
        if matches!(rail_state, 1..=3) && world.is_some() {
            // B28D30 still refreshes the wheel/contact manifold while RAIL
            // owns motion, but B29E70 does not publish its ordinary-ground
            // force/torque into the active accumulator (:3140-3150).
            rail_support = Some(self.inspect_autonomous_support(keys, parameters, world, None)?);
        }
        if enters_state6 {
            let up_axis = matrix_columns(self.state.matrix)[2];
            state6_entry_force = Some([
                fmul(up_axis[0], RAIL_STATE6_ENTRY_FORCE_SCALE),
                fmul(up_axis[1], RAIL_STATE6_ENTRY_FORCE_SCALE),
                fmul(up_axis[2], RAIL_STATE6_ENTRY_FORCE_SCALE),
            ]);
        }
        let terminal_position_store = if terminal_endpoint_edge {
            Some(self.state.position)
        } else {
            None
        };
        let terminal_velocity_store = if terminal_endpoint_edge {
            Some(self.state.velocity)
        } else {
            None
        };
        let trigger_transitioned = trigger_edge
            .as_ref()
            .is_some_and(|edge| edge.selector.transitioned);
        let pre_damping_velocity_override = if trigger_transitioned {
            Some(
                trigger_edge
                    .as_ref()
                    .expect("transitioned implies an edge")
                    .trigger
                    .velocity_c0_after,
            )
        } else if terminal_endpoint_edge {
            None
        } else {
            path_velocity_override
        };
        let pose_rail_state = if trigger_transitioned {
            Some(
                trigger_edge
                    .as_ref()
                    .expect("transitioned implies an edge")
                    .selector
                    .state_after,
            )
        } else if terminal_endpoint_edge {
            Some(6)
        } else {
            None
        };
        let rail_owner_id =
            selector_path_id(&selector).or_else(|| self.state.topology.constraint_owner_id.clone());
        let raw = self.step_raw_rail(
            keys,
            parameters,
            Some(&runtime),
            true,
            Some(rail_state),
            0,
            0,
            pre_damping_velocity_override,
            None,
            None,
            pose_rail_state,
            rail_owner_id,
        )?;
        let RawRailOutcome::Active(active) = raw else {
            // Unreachable: rail_contact=true yields the active writer chain.
            return Err("active rail sustain released ownership".to_string());
        };
        let mut active = *active;
        let mut rail_terminal_position_phase: Option<RailTerminalPositionPhase> = None;
        if let (Some(store_position), Some(store_velocity)) =
            (terminal_position_store, terminal_velocity_store)
        {
            // run119's epoch-156924 state setter is observed after the
            // terminal kind10 selector row but before the next position
            // writer (:3194-3214).
            let integrated_position = self.state.position;
            let integrated_velocity = self.state.velocity;
            self.state.position = store_position;
            self.state.velocity = store_velocity;
            active.rail_step.position_after = store_position;
            active.rail_step.velocity_after = store_velocity;
            rail_terminal_position_phase = Some(RailTerminalPositionPhase {
                published_position: store_position,
                suppressed_post_setter_position: integrated_position,
                published_velocity: store_velocity,
                suppressed_post_setter_velocity: integrated_velocity,
            });
        }
        if let Some(support) = &rail_support {
            self.state.contact_compression = support.producer.contact_compression_after;
        }
        if let Some(edge) = &trigger_edge {
            if let Some(RailCaptureSelectorRuntimeAfter::State3(runtime_after)) =
                &edge.selector.runtime_after
            {
                self.state.special_trigger_cooldown_ms =
                    runtime_after.trigger_cooldown_remaining_ms;
            }
            if edge.selector.transitioned {
                self.state.topology.rail_state = edge.selector.state_after;
                self.state.topology.rail_state_elapsed_ms = 0.0;
                let mut runtime_state =
                    self.state.rail_selector_runtime.clone().unwrap_or_default();
                runtime_state.state6_contact_remaining_ms = Some(STATE6_CONTACT_CONFIRM_MS);
                self.state.rail_selector_runtime = Some(runtime_state);
                if let Some(transition) = &edge.trigger.state_transition {
                    self.state.boost_state = transition.state;
                    self.state.boost_remaining_ms = transition.remaining_ms;
                }
            }
        }
        let mut rail_terminal_endpoint_transition: Option<RailTerminalEndpointTransition> = None;
        if terminal_endpoint_edge {
            self.state.topology.rail_state = 6;
            self.state.topology.rail_state_elapsed_ms = 0.0;
            let mut runtime_state = self.state.rail_selector_runtime.clone().unwrap_or_default();
            runtime_state.state6_contact_remaining_ms = Some(STATE6_CONTACT_CONFIRM_MS);
            self.state.rail_selector_runtime = Some(runtime_state);
            let gate = path_gate
                .as_ref()
                .expect("terminal edge implies a path gate");
            rail_terminal_endpoint_transition = Some(RailTerminalEndpointTransition {
                selected_context: gate.selected_context.clone(),
                query_scalar: gate.query_scalar,
                terminal_distance_remaining: gate.terminal_distance_remaining,
            });
        }
        let mut rail_state6_entry_force: Option<RailState6EntryForce> = None;
        if enters_state6 {
            if let Some(force) = state6_entry_force {
                let mut runtime_state =
                    self.state.rail_selector_runtime.clone().unwrap_or_default();
                runtime_state.state6_entry_force_pending_x = Some(force[0]);
                runtime_state.state6_entry_force_pending_y = Some(force[1]);
                runtime_state.state6_entry_force_pending_z = Some(force[2]);
                runtime_state.state6_entry_force_delay_substeps = Some(1.0);
                runtime_state.state6_entry_common_tail_delay_substeps = Some(1.0);
                self.state.rail_selector_runtime = Some(runtime_state);
                rail_state6_entry_force = Some(RailState6EntryForce {
                    force_accumulator_b74: force,
                });
            }
        }
        active.rail_contract.rail_context_authority =
            "world_query_inputs_to_exact_rail_selector_and_current_decoded_context";
        // Python publishes the geometry only inside the
        // `if trigger_edge is not None:` block (:3247-3249).
        let rail_authored_trigger_geometry = if trigger_edge.is_some() {
            trigger_geometry
        } else {
            None
        };
        Ok(AutonomousRailOutcome::Active(Box::new(
            AutonomousRailActiveOutcome {
                rail: active,
                rail_terminal_position_phase,
                rail_support_observation: rail_support,
                rail_authored_trigger: trigger_edge,
                rail_authored_trigger_geometry,
                rail_terminal_endpoint_transition,
                rail_state6_entry_force,
                autonomous_rail_selector: selector,
                rail_state_transition: selector_transition,
                rail_path_velocity_transfer: path_velocity_transfer,
                topology_after: self.state.topology.clone(),
                state_after: self.state.clone(),
            },
        )))
    }

    /// `advance_rail_capture_selector`
    /// (kartrider_vehicle_engine.py:3897-4056): advance the statically
    /// recovered RAIL selector subset. Probe-without-side-effects contract
    /// (handoff §6.3): only this integrating path commits selector state.
    pub fn advance_rail_capture_selector(
        &mut self,
        args: &AdvanceRailCaptureSelectorArgs,
    ) -> Result<RailCaptureSelectorAdvance, String> {
        if self.state.topology.domain != MotionDomain::Rail {
            return Err("rail capture selector requires RAIL ownership".to_string());
        }
        if args.release_support_contact_count < 0 {
            return Err("release_support_contact_count must be nonnegative".to_string());
        }
        let state_before = self.state.clone();
        let rail_state_before = self.state.topology.rail_state;
        let runtime_before = self.state.rail_selector_runtime.clone();
        let mut alignment_primary_remaining_ms = args.alignment_primary_remaining_ms;
        let mut alignment_stable_remaining_ms = args.alignment_stable_remaining_ms;
        let mut trigger_cooldown_remaining_ms = args.trigger_cooldown_remaining_ms;
        if rail_state_before == 2 {
            if alignment_primary_remaining_ms.is_none() {
                alignment_primary_remaining_ms = runtime_before
                    .as_ref()
                    .and_then(|runtime| runtime.alignment_primary_remaining_ms);
            }
            if alignment_stable_remaining_ms.is_none() {
                alignment_stable_remaining_ms = runtime_before
                    .as_ref()
                    .and_then(|runtime| runtime.alignment_stable_remaining_ms);
            }
        } else if rail_state_before == 3 && trigger_cooldown_remaining_ms.is_none() {
            trigger_cooldown_remaining_ms = Some(self.state.special_trigger_cooldown_ms);
        }
        let selector = evaluate_rail_capture_selector(&RailCaptureSelectorArgs {
            state_before: rail_state_before,
            elapsed_in_state_ms: args.elapsed_in_state_ms,
            field_af0: args.field_af0,
            contact_exists: args.contact_exists,
            contact_identifier_nonempty: args.contact_identifier_nonempty,
            path_metric: args.path_metric,
            metric_threshold: args.metric_threshold,
            path_z: args.path_z,
            kart_z_plus_one: args.kart_z_plus_one,
            delta_ms: args.delta_ms,
            alignment_primary_remaining_ms,
            alignment_stable_remaining_ms,
            forward_speed: args.forward_speed,
            lateral_speed: args.lateral_speed,
            trigger_cooldown_remaining_ms,
            authored_trigger_accepted: args.authored_trigger_accepted,
            state6_contact_remaining_ms: None,
            state6_contact_candidate_valid: false,
        })?;
        let mut topology_transition: Option<TopologyDecision> = None;
        let rail_state_after = selector.state_after;
        if rail_state_after == 0 {
            topology_transition = Some(self.advance_motion_topology(&TopologyObservation {
                rail_contact: false,
                rail_state: 0,
                support_contact_count: args.release_support_contact_count,
                surface_mode: args.release_surface_mode,
                ..TopologyObservation::default()
            })?);
            if args.release_support_contact_count > 0 {
                self.state.surface_mode_latched = Some(args.release_surface_mode);
                self.state.surface_selector_elapsed_ms = 0.0;
            }
        } else {
            self.state.topology.rail_state = rail_state_after;
            if rail_state_after != rail_state_before {
                self.state.topology.rail_state_elapsed_ms = 0.0;
            }
            if rail_state_before == 1 && rail_state_after == 2 {
                if args.alignment_base_ms.is_none()
                    && args.alignment_primary_scale_ms.is_none()
                    && args.alignment_stable_initial_ms.is_none()
                {
                    self.state.rail_selector_runtime = None;
                } else {
                    if let Some(alignment_base_ms) = args.alignment_base_ms {
                        if alignment_base_ms < 0.0 {
                            return Err("alignment_base_ms must be nonnegative".to_string());
                        }
                    }
                    let primary_scale = match args.alignment_primary_scale_ms {
                        None => args.alignment_base_ms,
                        Some(alignment_primary_scale_ms) => Some(alignment_primary_scale_ms),
                    };
                    let stable_initial = match args.alignment_stable_initial_ms {
                        None => args.alignment_base_ms.map(|base| base * 0.5),
                        Some(alignment_stable_initial_ms) => Some(alignment_stable_initial_ms),
                    };
                    let (Some(primary_scale), Some(stable_initial)) =
                        (primary_scale, stable_initial)
                    else {
                        return Err(
                            "rail alignment timers require both primary scale and stable \
                             initial values"
                                .to_string(),
                        );
                    };
                    if primary_scale < 0.0 || stable_initial < 0.0 {
                        return Err("rail alignment timer values must be nonnegative".to_string());
                    }
                    self.state.rail_selector_runtime = Some(crate::state::RailSelectorRuntime {
                        alignment_primary_remaining_ms: Some(primary_scale * args.field_af0),
                        alignment_stable_remaining_ms: Some(stable_initial),
                        ..crate::state::RailSelectorRuntime::default()
                    });
                }
            } else if rail_state_before == 2 {
                let mut next_runtime = crate::state::RailSelectorRuntime::default();
                if let Some(RailCaptureSelectorRuntimeAfter::State2(runtime_after)) =
                    selector.runtime_after.clone()
                {
                    next_runtime.alignment_primary_remaining_ms =
                        Some(runtime_after.alignment_primary_remaining_ms);
                    next_runtime.alignment_stable_remaining_ms =
                        Some(runtime_after.alignment_stable_remaining_ms);
                }
                if let Some(path_context_id) = runtime_before
                    .as_ref()
                    .and_then(|runtime| runtime.path_context_id.clone())
                {
                    next_runtime.path_context_id = Some(path_context_id);
                }
                self.state.rail_selector_runtime = Some(next_runtime);
            } else if rail_state_before == 3 {
                if let Some(RailCaptureSelectorRuntimeAfter::State3(runtime_after)) =
                    selector.runtime_after.clone()
                {
                    self.state.special_trigger_cooldown_ms =
                        runtime_after.trigger_cooldown_remaining_ms;
                }
                let mut next_runtime = crate::state::RailSelectorRuntime::default();
                if let Some(path_context_id) = runtime_before
                    .as_ref()
                    .and_then(|runtime| runtime.path_context_id.clone())
                {
                    next_runtime.path_context_id = Some(path_context_id);
                }
                self.state.rail_selector_runtime = Some(next_runtime);
            } else if matches!(rail_state_after, 1 | 5) {
                self.state.rail_selector_runtime = None;
            }
        }
        self.state.validate()?;
        Ok(RailCaptureSelectorAdvance {
            selector,
            topology_transition,
            state_before,
            state_after: self.state.clone(),
            selector_authority: "decoded_current_contact_identifier_path_and_timer_state",
            open_boundary: "autonomous JM/DJ authored-contact candidate construction and map-side \
                 rail path geometry",
        })
    }

    /// `inspect_autonomous_moving_contact`
    /// (kartrider_vehicle_engine.py:4117-4168): query and validate one
    /// current dynamic-contact geometry record. The `world is None` guard is
    /// type-enforced; the capability-absent guard keeps the Python message.
    pub fn inspect_autonomous_moving_contact(
        &self,
        world: Option<&dyn World>,
        object_id: Option<&str>,
        delta_ms: Option<i64>,
    ) -> Result<MovingContactInspection, String> {
        if !world_caps(world).query_moving_contact {
            return Err(
                "autonomous moving contact requires world.query_moving_contact".to_string(),
            );
        }
        let record = world
            .expect("capability probe implies a world")
            .query_moving_contact(
                self.state.position,
                self.state.time_ms as i64,
                object_id,
                delta_ms.unwrap_or(1),
                1.5,
            )
            .ok_or_else(|| {
                "query_moving_contact capability reported present but returned no result"
                    .to_string()
            })?;
        let Some(record) = record else {
            return Ok(MovingContactInspection {
                contact_active: false,
                record: None,
                object_id: object_id.map(str::to_string),
                authority: "world_query_moving_contact",
            });
        };
        // Python bool(record.get("contact_active", True)).
        let contact_active = record.get("contact_active").is_none_or(py_truthy);
        if !contact_active {
            return Ok(MovingContactInspection {
                contact_active: false,
                record: Some(record),
                object_id: object_id.map(str::to_string),
                authority: "world_query_moving_contact",
            });
        }
        const REQUIRED: [&str; 6] = [
            "contact_normal_world",
            "object_angular_velocity",
            "object_id",
            "object_matrix",
            "object_position",
            "object_velocity",
        ];
        let missing: Vec<&str> = REQUIRED
            .iter()
            .copied()
            .filter(|field| !record.contains_key(*field))
            .collect();
        if !missing.is_empty() {
            let rendered = missing
                .iter()
                .map(|field| format!("'{field}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "active moving-contact record missing fields: [{rendered}]"
            ));
        }
        if !record.get("object_id").is_some_and(py_truthy) {
            return Err("active moving-contact record requires object_id".to_string());
        }
        Ok(MovingContactInspection {
            contact_active: true,
            record: Some(record),
            object_id: object_id.map(str::to_string),
            authority: "world_query_moving_contact",
        })
    }

    /// `acquire_moving_contact` (kartrider_vehicle_engine.py:4058-4115):
    /// acquire a dynamic body's reference frame without a pose discontinuity.
    pub fn acquire_moving_contact(
        &mut self,
        object_id: &str,
        object_position: [f64; 3],
        object_velocity: [f64; 3],
        object_matrix: [f64; 9],
        object_angular_velocity: [f64; 3],
        contact_normal_world: [f64; 3],
    ) -> Result<MovingContactAcquisition, String> {
        let current_domain = self.state.topology.domain;
        if matches!(
            current_domain,
            MotionDomain::Reset
                | MotionDomain::NetworkCorrection
                | MotionDomain::Rail
                | MotionDomain::Water
        ) {
            return Err(format!(
                "cannot acquire moving contact while {} owns state",
                current_domain.value()
            ));
        }
        if self.state.moving_contact_anchor.is_some() {
            return Err("a moving-contact anchor is already active".to_string());
        }
        let before = self.state.clone();
        let frame = acquire_moving_contact_frame(
            self.state.position,
            self.state.velocity,
            self.state.matrix,
            self.state.angular_velocity,
            object_id,
            object_position,
            object_velocity,
            object_matrix,
            object_angular_velocity,
            contact_normal_world,
            1e-6,
        )?;
        let topology = self.advance_motion_topology(&TopologyObservation {
            moving_contact: true,
            moving_object_id: Some(object_id.to_string()),
            ..Default::default()
        })?;
        if topology.selected_domain != MotionDomain::MovingContact {
            // Python RuntimeError.
            return Err("moving-contact selector did not acquire ownership".to_string());
        }
        let anchor = frame.anchor();
        self.state.moving_contact_anchor = Some(anchor.clone());
        // MOVING-CONTACT uses its own kinematic reference-frame integrator
        // and never consumes the common previous-epoch accumulators.
        // Retaining a ground/contact producer here would freeze it for the
        // whole platform ride and apply a stale impulse after release.
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.contact_compression = [0.0; 4];
        self.state.validate()?;
        Ok(MovingContactAcquisition {
            anchor,
            topology_transition: topology,
            state_before: before,
            state_after: self.state.clone(),
            entry_transfer: "continuous_world_state_to_object_local_state",
        })
    }

    /// `acquire_autonomous_moving_contact`
    /// (kartrider_vehicle_engine.py:4170-4190): acquire MOVING-CONTACT solely
    /// from the current world selector.
    pub fn acquire_autonomous_moving_contact(
        &mut self,
        world: Option<&dyn World>,
    ) -> Result<AcquireAutonomousMovingContactOutcome, String> {
        let record = self.inspect_autonomous_moving_contact(world, None, None)?;
        if !record.contact_active {
            return Ok(AcquireAutonomousMovingContactOutcome::NotActive {
                selector: record,
                state_after: Box::new(self.state.clone()),
            });
        }
        let merged = merged_moving_contact_record(&record)?;
        let acquisition = self.acquire_moving_contact(
            &moving_record_string(&merged, "object_id")?,
            moving_record_vec3(&merged, "object_position")?,
            moving_record_vec3(&merged, "object_velocity")?,
            moving_record_mat9(&merged, "object_matrix")?,
            moving_record_vec3(&merged, "object_angular_velocity")?,
            moving_record_vec3(&merged, "contact_normal_world")?,
        )?;
        Ok(AcquireAutonomousMovingContactOutcome::Acquired(Box::new(
            AutonomousMovingContactAcquisition {
                acquisition,
                selector: record,
                entry_authority: "autonomous_world_dynamic_contact",
            },
        )))
    }

    /// `acquire_and_step_autonomous_moving_contact`
    /// (kartrider_vehicle_engine.py:4192-4233): acquire one validated dynamic
    /// contact and execute its first substep. `record` is the merged
    /// inspection mapping (world record + `contact_active` / `authority`).
    pub fn acquire_and_step_autonomous_moving_contact(
        &mut self,
        parameters: &VehicleParameters,
        record: &BTreeMap<String, JsonValue>,
    ) -> Result<AcquireAndStepMovingContactOutcome, String> {
        if !record.get("contact_active").is_some_and(py_truthy) {
            return Err("moving-contact capture requires an active record".to_string());
        }
        let object_id = moving_record_string(record, "object_id")?;
        let object_position = moving_record_vec3(record, "object_position")?;
        let object_velocity = moving_record_vec3(record, "object_velocity")?;
        let object_matrix = moving_record_mat9(record, "object_matrix")?;
        let object_angular_velocity = moving_record_vec3(record, "object_angular_velocity")?;
        let contact_normal_world = moving_record_vec3(record, "contact_normal_world")?;
        let acquisition = self.acquire_moving_contact(
            &object_id,
            object_position,
            object_velocity,
            object_matrix,
            object_angular_velocity,
            contact_normal_world,
        )?;
        let object_state_after = match record.get("object_state_after") {
            Some(JsonValue::Object(map)) => Some(map),
            _ => None,
        };
        // Python `step_record = object_state_after or record`: an empty
        // mapping is falsy and falls back to the capture record.
        let step_record: &BTreeMap<String, JsonValue> = match object_state_after {
            Some(map) if !map.is_empty() => map,
            _ => record,
        };
        let phase = if object_state_after.is_some() {
            "end"
        } else {
            "start"
        };
        let mut step = self.step_moving_contact(
            parameters,
            &object_id,
            moving_record_vec3_fallback(step_record, "position", object_position)?,
            moving_record_vec3_fallback(step_record, "velocity", object_velocity)?,
            moving_record_mat9_fallback(step_record, "matrix", object_matrix)?,
            moving_record_vec3_fallback(step_record, "angular_velocity", object_angular_velocity)?,
            phase,
        )?;
        step.moving_contact_contract.reference_frame_authority =
            "autonomous_world_dynamic_contact_initial_capture".to_string();
        Ok(AcquireAndStepMovingContactOutcome {
            step,
            acquisition,
            moving_contact_selector: record.clone(),
        })
    }

    /// `step_moving_contact` (kartrider_vehicle_engine.py:4235-4287): advance
    /// a vehicle constrained to an acquired moving rigid frame.
    pub fn step_moving_contact(
        &mut self,
        parameters: &VehicleParameters,
        object_id: &str,
        object_position: [f64; 3],
        object_velocity: [f64; 3],
        object_matrix: [f64; 9],
        object_angular_velocity: [f64; 3],
        object_state_phase: &str,
    ) -> Result<MovingContactStepOutcome, String> {
        parameters.validate()?;
        if self.state.topology.domain != MotionDomain::MovingContact {
            return Err("step_moving_contact requires MOVING-CONTACT topology".to_string());
        }
        let Some(anchor) = self.state.moving_contact_anchor.clone() else {
            return Err("moving-contact topology requires an acquired anchor".to_string());
        };
        let result = evaluate_moving_contact_step(
            &anchor,
            object_id,
            object_position,
            object_velocity,
            object_matrix,
            object_angular_velocity,
            parameters.substep_ms as f64 / 1000.0,
            self.numeric_mode.as_str(),
            object_state_phase,
        )?;
        self.state.position = result.position_after;
        self.state.velocity = result.velocity_after;
        self.state.matrix = result.matrix_after;
        self.state.angular_velocity = result.angular_velocity_after;
        self.state.moving_contact_anchor = Some(result.anchor_after.clone());
        if self.numeric_mode.is_float32() {
            f32_vector(&mut self.state.position);
            f32_vector(&mut self.state.velocity);
            f32_vector(&mut self.state.matrix);
            f32_vector(&mut self.state.angular_velocity);
        }
        self.state.time_ms += parameters.substep_ms as u64;
        self.state.step_index += 1;
        self.state.topology.domain_step_index += 1;
        Ok(MovingContactStepOutcome {
            moving_contact_contract: MovingContactStepContract {
                reference_frame_authority: "engine_owned_acquired_anchor".to_string(),
                object_motion_authority: if object_state_phase == "end" {
                    "caller_supplied_dynamic_body_exact_end_state"
                } else {
                    "caller_supplied_dynamic_body_start_state_integrated_by_engine"
                },
                open_boundary: "game_specific_dynamic_object_selector_and_contact_forces",
            },
            step: result,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
        })
    }

    /// `step_autonomous_moving_contact`
    /// (kartrider_vehicle_engine.py:4289-4370): sustain or release the moving
    /// frame from current world geometry.
    pub fn step_autonomous_moving_contact(
        &mut self,
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
    ) -> Result<AutonomousMovingContactOutcome, String> {
        if self.state.topology.domain != MotionDomain::MovingContact {
            return Err(
                "autonomous moving-contact step requires MOVING-CONTACT topology".to_string(),
            );
        }
        let Some(anchor) = self.state.moving_contact_anchor.clone() else {
            return Err("moving-contact topology requires an acquired anchor".to_string());
        };
        let record = self.inspect_autonomous_moving_contact(
            world,
            Some(&anchor.object_id),
            Some(parameters.substep_ms),
        )?;
        let selector = if record.contact_active {
            let merged = merged_moving_contact_record(&record)?;
            evaluate_moving_contact_selector(
                &anchor,
                self.state.position,
                self.state.velocity,
                &moving_record_string(&merged, "object_id")?,
                moving_record_vec3(&merged, "object_position")?,
                moving_record_vec3(&merged, "object_velocity")?,
                moving_record_mat9(&merged, "object_matrix")?,
                moving_record_vec3(&merged, "object_angular_velocity")?,
                moving_record_vec3(&merged, "contact_normal_world")?,
                true,
                1e-6,
                1e-5,
            )?
        } else {
            MovingContactSelectorResult {
                sustain: false,
                reason: "contact_geometry_released".to_string(),
                relative_normal_velocity: None,
                separation_tolerance: None,
                normal_alignment: None,
                normal_alignment_tolerance: None,
                object_point_velocity: None,
                authority: Some("world_query_moving_contact"),
            }
        };
        if selector.sustain {
            let merged = merged_moving_contact_record(&record)?;
            let object_id = moving_record_string(&merged, "object_id")?;
            let object_position = moving_record_vec3(&merged, "object_position")?;
            let object_velocity = moving_record_vec3(&merged, "object_velocity")?;
            let object_matrix = moving_record_mat9(&merged, "object_matrix")?;
            let object_angular_velocity = moving_record_vec3(&merged, "object_angular_velocity")?;
            let object_state_after = match merged.get("object_state_after") {
                Some(JsonValue::Object(map)) => Some(map),
                _ => None,
            };
            let step_record: &BTreeMap<String, JsonValue> = match object_state_after {
                Some(map) if !map.is_empty() => map,
                _ => &merged,
            };
            let phase = if object_state_after.is_some() {
                "end"
            } else {
                "start"
            };
            let step = self.step_moving_contact(
                parameters,
                &object_id,
                moving_record_vec3_fallback(step_record, "position", object_position)?,
                moving_record_vec3_fallback(step_record, "velocity", object_velocity)?,
                moving_record_mat9_fallback(step_record, "matrix", object_matrix)?,
                moving_record_vec3_fallback(
                    step_record,
                    "angular_velocity",
                    object_angular_velocity,
                )?,
                phase,
            )?;
            return Ok(AutonomousMovingContactOutcome::Sustained(Box::new(
                AutonomousMovingContactSustainOutcome {
                    step,
                    moving_contact_selector: selector,
                    released: false,
                },
            )));
        }
        let support = self.inspect_autonomous_support(&[], parameters, world, None)?;
        if Self::support_selects_authored_reset(&support) {
            let reset = self.enter_authored_reset_surface(support)?;
            return Ok(AutonomousMovingContactOutcome::ResetRelease(Box::new(
                AutonomousMovingContactResetReleaseOutcome {
                    reset,
                    moving_contact_selector: selector,
                    released: true,
                    exit_authority: "autonomous_dynamic_contact_release_to_reset_surface",
                },
            )));
        }
        let release =
            self.release_moving_contact(support.support_contact_count, support.surface_mode)?;
        Ok(AutonomousMovingContactOutcome::Released(Box::new(
            AutonomousMovingContactReleaseOutcome {
                release,
                moving_contact_selector: selector,
                support_observation: support,
                released: true,
                exit_authority: "autonomous_dynamic_contact_and_four_wheel_support_geometry",
            },
        )))
    }

    /// `release_moving_contact` (kartrider_vehicle_engine.py:4372-4401):
    /// release the moving frame while preserving reconstructed world state.
    pub fn release_moving_contact(
        &mut self,
        support_contact_count: i64,
        surface_mode: i64,
    ) -> Result<MovingContactReleaseOutcome, String> {
        if self.state.topology.domain != MotionDomain::MovingContact {
            return Err("release requires MOVING-CONTACT topology".to_string());
        }
        let topology = self.advance_motion_topology(&TopologyObservation {
            support_contact_count,
            surface_mode,
            moving_contact: false,
            ..Default::default()
        })?;
        let anchor_before = self.state.moving_contact_anchor.take();
        // The release boundary has queried destination support but has not
        // executed a destination producer epoch. Start that owner with no
        // stale moving-frame or pre-capture accumulator history.
        self.state.force_accumulator = [0.0, 0.0, 0.0];
        self.state.torque_accumulator = [0.0, 0.0, 0.0];
        self.state.contact_compression = [0.0; 4];
        if support_contact_count > 0 {
            self.state.surface_mode_latched = Some(surface_mode);
            self.state.surface_selector_elapsed_ms = 0.0;
        }
        self.state.validate()?;
        Ok(MovingContactReleaseOutcome {
            anchor_before,
            topology_transition: topology,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
            exit_transfer: "continuous_object_local_state_to_world_state",
        })
    }

    /// `apply_checkpoint_respawn` (kartrider_vehicle_engine.py:1637-1702):
    /// apply checkpoint pose replacement and autonomously leave RESET.
    pub fn apply_checkpoint_respawn(
        &mut self,
        course_reference_position: [f64; 3],
        sample_result_zero: bool,
        world: Option<&dyn World>,
        parameters: &VehicleParameters,
    ) -> Result<CheckpointRespawnApplyOutcome, String> {
        let result = evaluate_checkpoint_respawn_state_replacement(
            &self.state.checkpoint_mode_family,
            self.state.checkpoint_gate,
            sample_result_zero,
            self.state.checkpoint_target,
            course_reference_position,
            self.state.velocity,
            self.state.angular_velocity,
        )?;
        if !result.selected {
            return Ok(CheckpointRespawnApplyOutcome {
                replacement: result,
                topology_after: self.state.topology.clone(),
                support_observation: None,
                topology_transition: None,
                state_after: None,
            });
        }
        let mut candidate = self.state.clone();
        if candidate.topology.domain != MotionDomain::Reset {
            let reset_entry = advance_topology(
                &candidate.topology,
                &TopologyObservation {
                    reset_requested: true,
                    ..Default::default()
                },
            )?;
            candidate.topology = reset_entry.state_after;
        }
        candidate.moving_contact_anchor = None;
        candidate.network_correction = None;
        candidate.position = result
            .position_90_after
            .expect("selected replacement carries a pose");
        candidate.matrix = result
            .matrix_9c_after
            .expect("selected replacement carries a matrix");
        candidate.velocity = result.velocity_c0_after;
        candidate.angular_velocity = result.angular_velocity_cc_after;
        candidate.force_accumulator = [0.0, 0.0, 0.0];
        candidate.torque_accumulator = [0.0, 0.0, 0.0];
        candidate.contact_compression = [0.0; 4];
        candidate.surface_mode_latched = None;
        candidate.surface_selector_elapsed_ms = 0.0;
        candidate.validate()?;
        let candidate_engine = KartRiderVehicleEngine::new(&candidate, self.numeric_mode)?;
        let support = candidate_engine.inspect_autonomous_support(&[], parameters, world, None)?;
        if Self::support_selects_authored_reset(&support) {
            return Err(
                "checkpoint respawn destination remains on authored reset surface".to_string(),
            );
        }
        let topology = advance_topology(
            &candidate.topology,
            &TopologyObservation {
                reset_complete: true,
                support_contact_count: support.support_contact_count,
                surface_mode: support.surface_mode,
                ..Default::default()
            },
        )?;
        candidate.topology = topology.state_after.clone();
        candidate.surface_mode_latched = if support.support_contact_count > 0 {
            Some(support.surface_mode)
        } else {
            None
        };
        candidate.surface_selector_elapsed_ms = 0.0;
        candidate.checkpoint_gate = false;
        candidate.checkpoint_gate_armed_tick_ms = None;
        candidate.validate()?;
        self.state = candidate;
        Ok(CheckpointRespawnApplyOutcome {
            replacement: result,
            topology_after: self.state.topology.clone(),
            support_observation: Some(support),
            topology_transition: Some(topology),
            state_after: Some(self.state.clone()),
        })
    }

    /// `step_autonomous_reset` (kartrider_vehicle_engine.py:1704-1749):
    /// complete an owned RESET from the current checkpoint world selector.
    pub fn step_autonomous_reset(
        &mut self,
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
    ) -> Result<AutonomousResetOutcome, String> {
        if self.state.topology.domain != MotionDomain::Reset {
            return Err("autonomous reset step requires RESET topology".to_string());
        }
        if !world_caps(world).query_checkpoint_respawn {
            return Err(
                "autonomous reset step requires world.query_checkpoint_respawn".to_string(),
            );
        }
        let selector = world
            .expect("capability probe implies a world")
            .query_checkpoint_respawn(
                self.state.position,
                self.state.checkpoint_target,
                self.state.checkpoint_gate,
                &self.state.checkpoint_mode_family,
                20.0,
            )
            .ok_or_else(|| {
                "query_checkpoint_respawn capability reported present but returned no result"
                    .to_string()
            })?;
        let Some(selector) = selector else {
            return Ok(AutonomousResetOutcome::NotReady(Box::new(
                AutonomousResetNotReadyOutcome {
                    selected: false,
                    reason: "checkpoint_respawn_not_ready",
                    topology_after: self.state.topology.clone(),
                    state_after: self.state.clone(),
                    selector_authority: "world_query_checkpoint_respawn",
                },
            )));
        };
        let selector_authority = selector
            .authority
            .clone()
            .unwrap_or_else(|| "world_query_checkpoint_respawn".to_string());
        let respawn = self.apply_checkpoint_respawn(
            selector.course_reference_position,
            selector.sample_result_zero,
            world,
            parameters,
        )?;
        Ok(AutonomousResetOutcome::Applied(Box::new(
            AutonomousResetAppliedOutcome {
                respawn,
                checkpoint_respawn_selector: selector,
                selector_authority,
            },
        )))
    }

    /// `_apply_autonomous_authored_trigger`
    /// (kartrider_vehicle_engine.py:3589-3730): apply one map-owned JM/DJ
    /// contact edge outside exclusive RAIL. The collision dispatcher
    /// consumes JM/DJ on contact entry, not once per overlapping physics
    /// substep; the overlap identity latch reproduces that edge contract.
    /// `support=Some(..)` takes the B28D30 slot-0 contact-record path;
    /// `support=None` takes the rail OBB-overlap query path
    /// (`_query_autonomous_rail_authored_trigger` with `center_up=0.0`).
    pub fn apply_autonomous_authored_trigger(
        &mut self,
        world: Option<&dyn World>,
        parameters: &VehicleParameters,
        commit: bool,
        support: Option<&SupportObservation>,
    ) -> Result<Option<AuthoredTriggerEdge>, String> {
        // (road_surface, jm_basis_190, boost_multiplier_64c, contact_id).
        let (road_surface, jm_basis_190, boost_multiplier_64c, contact_id) = if let Some(support) =
            support
        {
            // B2CB10 consumes the current B28D30 contact record; only
            // slot 0's active contact is a selector candidate (:3602-3622).
            let mut unique: Option<((String, i64, i64, String), usize)> = None;
            for (index, active) in support.active_slots.iter().enumerate() {
                if index != 0 || !active {
                    continue;
                }
                let plane = &support.planes[index];
                if parse_jm_dj_surface_token(&plane.road_surface).is_err() {
                    continue;
                }
                if plane.contact_basis_44.is_none() {
                    return Err(
                        "active JM/DJ wheel contact requires exact CollisionFace+0x44 \
                             basis"
                            .to_string(),
                    );
                }
                let key = (
                    plane.node.clone(),
                    plane.tri_index,
                    plane.surface_id,
                    plane.road_surface.clone(),
                );
                match &unique {
                    Some((existing, _)) if *existing != key => {
                        return Err(
                            "multiple distinct authored JM/DJ wheel contacts overlap".to_string()
                        );
                    }
                    Some(_) => {}
                    None => unique = Some((key, index)),
                }
            }
            let Some((_, candidate_index)) = unique else {
                // geometry = {"authored_trigger": None,
                //             "authority": "active_wheel_contacts_no_jm_dj"}
                self.state.authored_trigger_contact_id = None;
                return Ok(None);
            };
            let candidate = &support.planes[candidate_index];
            let mut basis = candidate
                .contact_basis_44
                .expect("checked in the candidate loop");
            // CollisionFace+0x44 is an unoriented tangent; B28D30 orients
            // the contact record along travel before B2CB10 consumes it
            // (:3641-3652). The rail query path does NOT orient.
            let basis_approach = if self.numeric_mode.is_float32() {
                dot3_f32(basis, self.state.velocity)
            } else {
                dot3_sum(basis, self.state.velocity)
            };
            if basis_approach < 0.0 {
                basis = [-basis[0], -basis[1], -basis[2]];
            }
            let contact_id = format!(
                "{}|{}|{}|{}",
                candidate.node, candidate.tri_index, candidate.surface_id, candidate.road_surface
            );
            (
                candidate.road_surface.clone(),
                Some(basis),
                false,
                contact_id,
            )
        } else {
            let geometry =
                self.query_autonomous_rail_authored_trigger(world, parameters, Some(0.0))?;
            let Some(trigger) = &geometry.authored_trigger else {
                self.state.authored_trigger_contact_id = None;
                return Ok(None);
            };
            let candidate = geometry.candidate.clone().unwrap_or_default();
            let road_surface = trigger
                .get("road_surface")
                .map_or_else(String::new, json_py_str);
            let contact_id = format!(
                "{}|{}|{}|{}",
                candidate.node, candidate.tri_index, candidate.surface_id, road_surface
            );
            let jm_basis_190 = match trigger.get("jm_basis_190") {
                Some(JsonValue::Array(values)) if values.len() == 3 => Some([
                    number_from_json(values.first()),
                    number_from_json(values.get(1)),
                    number_from_json(values.get(2)),
                ]),
                _ => None,
            };
            let boost_multiplier_64c = trigger.get("boost_multiplier_64c").is_some_and(py_truthy);
            (road_surface, jm_basis_190, boost_multiplier_64c, contact_id)
        };
        if self.state.authored_trigger_contact_id.as_deref() == Some(contact_id.as_str()) {
            return Ok(Some(AuthoredTriggerEdge {
                selected: false,
                reason: Some("authored_trigger_contact_already_latched"),
                contact_id,
                trigger: None,
                model_args: None,
                authority: None,
            }));
        }
        let decoded = parse_jm_dj_surface_token(&road_surface)?;
        let special_state_family_already_active = (13..17).contains(&self.state.boost_state);
        let (
            variant,
            velocity_c0_before,
            jm_basis_190,
            jm_horizontal_param,
            jm_basis_param,
            dj_vector,
        ) = match &decoded {
            JmDjSurfaceToken::Jm {
                jm_horizontal_param,
                jm_basis_param,
                ..
            } => (
                "JM",
                Some(self.state.velocity),
                jm_basis_190,
                Some(*jm_horizontal_param),
                Some(*jm_basis_param),
                None,
            ),
            JmDjSurfaceToken::Dj { dj_vector, .. } => {
                ("DJ", None, None, None, None, Some(*dj_vector))
            }
        };
        let evaluated = evaluate_jm_dj_trigger(
            variant,
            special_state_family_already_active,
            velocity_c0_before,
            jm_basis_190,
            jm_horizontal_param,
            jm_basis_param,
            dj_vector,
            boost_multiplier_64c,
        )?;
        if commit {
            self.state.velocity = evaluated.velocity_c0_after;
            // Python quantizes unconditionally here
            // (kartrider_vehicle_engine.py:3706).
            f32_vector(&mut self.state.velocity);
            self.state.special_trigger_cooldown_ms =
                evaluated.cooldown_648_after * MILLISECONDS_PER_SECOND;
            if let Some(transition) = &evaluated.state_transition {
                self.state.boost_state = transition.state;
                self.state.boost_remaining_ms = transition.remaining_ms;
            }
        }
        self.state.authored_trigger_contact_id = Some(contact_id.clone());
        Ok(Some(AuthoredTriggerEdge {
            selected: true,
            reason: None,
            contact_id,
            trigger: Some(evaluated),
            model_args: Some(JmDjArgs {
                variant: variant.to_string(),
                special_state_family_already_active: Some(special_state_family_already_active),
                velocity_c0_before,
                jm_basis_190,
                jm_horizontal_param,
                jm_basis_param,
                dj_vector,
                boost_multiplier_64c: Some(boost_multiplier_64c),
                selection_authority: Some("authored_trigger_dispatch".to_string()),
            }),
            authority: Some("map_obb_overlap_authored_jm_dj_contact_entry"),
        }))
    }

    /// `_select_current_pose_hard_overlap`
    /// (kartrider_vehicle_engine.py:1293-1339): return the first exact
    /// approaching hard overlap at the current pose.
    pub fn select_current_pose_hard_overlap(
        &self,
        world: Option<&dyn World>,
        parameters: &VehicleParameters,
        source: &str,
        required_face_id: Option<&str>,
    ) -> Result<Option<HardOverlapCandidate>, String> {
        if !world_caps(world).overlap_obb_all {
            return Ok(None);
        }
        let world = world.expect("capability probe implies a world");
        let columns = matrix_columns(self.state.matrix);
        let up = columns[2];
        let center = [
            self.state.position[0] + up[0] * parameters.collision_center_up,
            self.state.position[1] + up[1] * parameters.collision_center_up,
            self.state.position[2] + up[2] * parameters.collision_center_up,
        ];
        let overlaps = world
            .overlap_obb_all(center, self.state.matrix, parameters.collision_half_extents)
            .ok_or_else(|| {
                "overlap_obb_all capability reported present but returned no result".to_string()
            })?;
        for overlap in &overlaps {
            if required_face_id.is_some_and(|required| {
                format!("{}:{}", overlap.node, overlap.tri_index) != required
            }) {
                continue;
            }
            let selection = select_collision_dispatch(
                overlap,
                self.state.velocity,
                self.state.collision_arg_threshold,
                self.state.collision_e38,
                self.state.collision_e4c,
                self.numeric_mode,
            );
            if selection.selected && selection.branch == Some("hard_contact") {
                return Ok(Some(HardOverlapCandidate {
                    overlap: overlap.clone(),
                    source: source.to_string(),
                    selection,
                }));
            }
        }
        Ok(None)
    }
}

impl KartRiderVehicleEngine {
    /// `_release_collision_contact_topology`
    /// (kartrider_vehicle_engine.py:1431-1449): release CONTACT when
    /// autonomous geometry has no active response.
    fn release_collision_contact_topology(&mut self) -> Result<Option<TopologyDecision>, String> {
        if self.state.topology.domain != MotionDomain::Contact {
            return Ok(None);
        }
        let return_domain = self
            .state
            .topology
            .return_domain
            .unwrap_or(MotionDomain::Air);
        let supported = matches!(
            return_domain,
            MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
        );
        Ok(Some(self.advance_motion_topology(
            &TopologyObservation {
                support_contact_count: i64::from(supported),
                surface_mode: if return_domain == MotionDomain::GroundSlip {
                    2
                } else {
                    0
                },
                ..Default::default()
            },
        )?))
    }

    /// The `commit_collision_topology` closure of `step_world`
    /// (kartrider_vehicle_engine.py:7198-7212): on the rail-state1 ground
    /// writer path it returns the synthetic ownership-retaining payload
    /// instead of committing CONTACT. The Python synthetic dict also carries
    /// `contact_count`; [`TopologyDecision`] has no such field, and the
    /// parity harness compares the projected keys.
    fn commit_collision_topology_for(
        &mut self,
        contact_count: i64,
        rail_state1_ground_writer: bool,
    ) -> Result<TopologyDecision, String> {
        if rail_state1_ground_writer {
            let snapshot = self.state.topology.clone();
            return Ok(TopologyDecision {
                selected_domain: MotionDomain::Rail,
                transitioned: false,
                reason: "rail_state1_ground_writer_retains_ownership".to_string(),
                state_before: snapshot.clone(),
                state_after: snapshot,
                observation: TopologyObservation::default(),
                authority: "state1_ground_physics_with_rail_capture_ownership".to_string(),
            });
        }
        self.commit_collision_contact_topology(contact_count)
    }

    /// `_commit_collision_contact_topology`
    /// (kartrider_vehicle_engine.py:1392-1429): transfer scheduler ownership
    /// to CONTACT after a dispatched response.
    fn commit_collision_contact_topology(
        &mut self,
        contact_count: i64,
    ) -> Result<TopologyDecision, String> {
        if contact_count <= 0 {
            return Err("contact_count must be positive".to_string());
        }
        let current = self.state.topology.domain;
        let mut return_domain = if current == MotionDomain::Contact {
            self.state.topology.return_domain.unwrap_or(current)
        } else {
            current
        };
        let supported = matches!(
            return_domain,
            MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
        );
        if supported && current != MotionDomain::Contact {
            // By the time collision dispatch runs, this substep's support and
            // surface producer has already executed. LAND is therefore
            // reconciled to a grounded return owner, and a same-step surface
            // change must not be lost behind the older topology domain name.
            return_domain = if self.state.surface_mode_latched == Some(2)
                || (self.state.surface_mode_latched.is_none()
                    && current == MotionDomain::GroundSlip)
            {
                MotionDomain::GroundSlip
            } else {
                MotionDomain::Ground
            };
        }
        self.advance_motion_topology(&TopologyObservation {
            support_contact_count: i64::from(supported),
            surface_mode: if return_domain == MotionDomain::GroundSlip {
                2
            } else {
                0
            },
            collision_contact_count: contact_count,
            ..Default::default()
        })
    }

    /// Merge the engine-state selector defaults with the
    /// `collision_selector_state` overlay
    /// (`selector_state.update(collision_selector_state or {})`,
    /// kartrider_vehicle_engine.py:7288/:7461).
    fn resolve_collision_selector_state(
        &self,
        overlay: CollisionSelectorState,
    ) -> CollisionChainSelectorState {
        CollisionChainSelectorState {
            arg_threshold: overlay
                .arg_threshold
                .unwrap_or(self.state.collision_arg_threshold),
            e38: overlay.e38.unwrap_or(self.state.collision_e38),
            e4c: overlay.e4c.unwrap_or(self.state.collision_e4c),
            drive_selected_slot: overlay
                .drive_selected_slot
                .unwrap_or(self.state.drive_selected_slot),
            drive_raw_aa0: overlay.drive_raw_aa0.unwrap_or(self.state.drive_raw_aa0),
            drive_sign_aac: overlay.drive_sign_aac.unwrap_or(self.state.drive_sign_aac),
            selector_af4: overlay.selector_af4.unwrap_or(self.state.soft_selector_af4),
            flag_188: overlay.flag_188.unwrap_or(self.state.soft_flag_188),
        }
    }

    /// `_apply_collision_chain` (kartrider_vehicle_engine.py:1073-1212):
    /// apply authored collision candidates through one response loop.
    ///
    /// SINGLE implementation for the four chain types (mixed / pure-soft /
    /// pure-hard / state6) — they differ only in `allowed_branch`,
    /// `stop_on_disallowed`, and `quantize_each` (handoff §3; the call sites
    /// are :7289 mixed, :7331 soft, :7374 hard, :2212 state6). The Python
    /// `candidate_error` isinstance guard is type-enforced in Rust. Takes
    /// `&self`: the chain works on local velocity copies and never writes
    /// engine state (the caller commits through `_finalize_collision_chain`
    /// or the single-contact replay).
    #[allow(clippy::too_many_arguments)]
    pub fn apply_collision_chain(
        &self,
        candidates: &[WorldObbOverlap],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        velocity: [f64; 3],
        angular_velocity: [f64; 3],
        selector_state: &CollisionChainSelectorState,
        allowed_branch: Option<&str>,
        stop_on_disallowed: bool,
        quantize_each: bool,
    ) -> Result<CollisionChainOutcome, String> {
        let mut chain_velocity = velocity;
        let mut chain_angular_velocity = angular_velocity;
        let mut responses = Vec::new();
        let mut branches = BTreeSet::new();
        let mut crossed_branch = false;
        for candidate in candidates {
            let axes = interpolate_axes(
                start_matrix,
                end_matrix,
                candidate.fraction,
                self.numeric_mode,
            )?;
            let selection = select_collision_dispatch(
                candidate,
                chain_velocity,
                selector_state.arg_threshold,
                selector_state.e38,
                selector_state.e4c,
                self.numeric_mode,
            );
            if !selection.selected {
                continue;
            }
            let branch = selection.branch.expect("selected implies a branch");
            if let Some(allowed) = allowed_branch {
                if branch != allowed {
                    crossed_branch = true;
                    if stop_on_disallowed {
                        break;
                    }
                    continue;
                }
            }
            branches.insert(branch);
            // Runtime collision bases are AB8=matrix column 0,
            // AD0=-matrix column 1, AC4=matrix column 2.
            let basis_ab8 = axes[0];
            let basis_ad0 = [-axes[1][0], -axes[1][1], -axes[1][2]];
            let basis_ac4 = axes[2];
            let normal = selection.normal.expect("selected implies a normal");
            let before_velocity = chain_velocity;
            let before_angular = chain_angular_velocity;
            let velocity_response;
            let angular_response;
            if branch == "hard_contact" {
                let hard_velocity = evaluate_hard_contact_velocity_response(
                    before_velocity,
                    normal,
                    basis_ad0,
                    0.7,
                );
                let hard_angular = evaluate_hard_contact_angular_response(
                    before_angular,
                    normal,
                    basis_ac4,
                    basis_ab8,
                    basis_ad0,
                    0.1,
                );
                chain_velocity = hard_velocity.velocity_c0_after;
                chain_angular_velocity = hard_angular.angular_velocity_after;
                velocity_response = CollisionChainVelocityResponse::Hard(hard_velocity);
                angular_response = CollisionChainAngularResponse::Hard(hard_angular);
            } else {
                let soft_source = evaluate_soft_contact_source(
                    before_velocity,
                    normal,
                    basis_ad0,
                    basis_ab8,
                    selector_state.drive_selected_slot,
                    selector_state.drive_raw_aa0,
                    selector_state.drive_sign_aac,
                    selector_state.selector_af4,
                    selector_state.flag_188,
                )?;
                chain_velocity = [
                    before_velocity[0] + soft_source.velocity_source[0],
                    before_velocity[1] + soft_source.velocity_source[1],
                    before_velocity[2] + soft_source.velocity_source[2],
                ];
                chain_angular_velocity = before_angular;
                if soft_source.optional_active {
                    chain_angular_velocity[2] += soft_source.angular_z_delta;
                }
                let soft_angular = evaluate_soft_contact_angular_source(
                    chain_velocity,
                    chain_angular_velocity,
                    normal,
                    basis_ad0,
                    basis_ab8,
                    self.numeric_mode.is_float32(),
                );
                if soft_angular.writeback_selected {
                    chain_angular_velocity = [
                        chain_angular_velocity[0] + soft_angular.source[0],
                        chain_angular_velocity[1] + soft_angular.source[1],
                        chain_angular_velocity[2] + soft_angular.source[2],
                    ];
                }
                velocity_response = CollisionChainVelocityResponse::Soft(soft_source);
                angular_response = CollisionChainAngularResponse::Soft(soft_angular);
            }
            // Per-candidate quantization is a chain-type parameter
            // (quantize_each; handoff §6.5): mixed/soft/state6 quantize,
            // the pure-hard chain defers to quantize_final.
            if quantize_each && self.numeric_mode.is_float32() {
                f32_vector(&mut chain_velocity);
                f32_vector(&mut chain_angular_velocity);
            }
            responses.push(CollisionChainResponse {
                // Python `candidate.get("loop_index", candidate_index)`; the
                // world row always carries loop_index (typed field).
                loop_index: candidate.loop_index,
                branch,
                candidate: candidate.clone(),
                selection,
                velocity_before: before_velocity,
                velocity_after: chain_velocity,
                angular_velocity_before: before_angular,
                angular_velocity_after: chain_angular_velocity,
                velocity_response,
                angular_response,
            });
        }
        Ok(CollisionChainOutcome {
            responses,
            branches,
            crossed_branch,
            velocity_after: chain_velocity,
            angular_velocity_after: chain_angular_velocity,
        })
    }

    /// `_finalize_collision_chain` (kartrider_vehicle_engine.py:1214-1291):
    /// commit one completed chain and build its common sweep payload. The
    /// Python `commit_collision_topology` callable is
    /// `_commit_collision_contact_topology` on every path except the
    /// rail-state1 ground writer, where it is the synthetic
    /// ownership-retaining payload (:7199-7211) selected here by
    /// `rail_state1_ground_writer`. The `_start_matrix` / `_end_matrix` /
    /// `_collision_half_extents` / `_collision_center_up` parameters feed
    /// only the Python world_sweep payload keys that [`WorldSweepInfo`]
    /// deliberately does not carry.
    #[allow(clippy::too_many_arguments)]
    pub fn finalize_collision_chain(
        &mut self,
        mut step: StepOutcome,
        control_prediction: Option<ControlPrediction>,
        chain: &CollisionChainOutcome,
        frame: &StepFrame,
        position_before: [f64; 3],
        matrix_before: [f64; 9],
        _start_matrix: [f64; 9],
        _end_matrix: [f64; 9],
        end_position: [f64; 3],
        _collision_half_extents: [f64; 3],
        _collision_center_up: f64,
        candidates: &[WorldObbOverlap],
        chain_key: &str,
        dispatch_reason: &str,
        composition_authority: &str,
        soft_base_response: Option<SoftBaseResponse>,
        quantize_final: bool,
        rail_state1_ground_writer: bool,
    ) -> Result<WorldStepOutcome, String> {
        let mut velocity = chain.velocity_after;
        let mut angular_velocity = chain.angular_velocity_after;
        if quantize_final && self.numeric_mode.is_float32() {
            f32_vector(&mut velocity);
            f32_vector(&mut angular_velocity);
        }
        let dt = frame.elapsed_ms / 1000.0;
        let position = evaluate_position_integrator(
            position_before,
            velocity,
            dt,
            None,
            0,
            false,
            0.0,
            None,
            None,
        )?
        .position_90_after;
        let matrix = evaluate_orientation_integrator(
            matrix_before,
            angular_velocity,
            dt,
            false,
            false,
            self.state.topology.rail_state,
            self.numeric_mode.is_float32(),
        )?
        .matrix_9c_after;
        self.state.velocity = velocity;
        self.state.angular_velocity = angular_velocity;
        self.state.position = position;
        self.state.matrix = matrix;
        if self.numeric_mode.is_float32() {
            f32_vector(&mut self.state.position);
            f32_vector(&mut self.state.matrix);
        }
        let topology_transition = self.commit_collision_topology_for(
            chain.responses.len() as i64,
            rail_state1_ground_writer,
        )?;
        step.state_after = self.state.clone();
        let last_response = chain.responses.last();
        Ok(WorldStepOutcome {
            step,
            world_sweep: WorldSweepInfo {
                shape: "obb_rotating",
                start: position_before,
                end: end_position,
                candidate: last_response.map(|response| response.candidate.clone()),
                candidates: candidates.to_vec(),
                response_dispatched: true,
                response_complete: true,
                dispatch_reason: dispatch_reason.to_string(),
                dispatch_authority: Some("collision_solver_dispatch"),
                angular_response: last_response.map(|response| response.angular_response.clone()),
                soft_base_response,
                chain: Some(CollisionChainPayload {
                    chain_key: chain_key.to_string(),
                    responses: chain.responses.clone(),
                    composition_authority: composition_authority.to_string(),
                    hard_response_chain_empty: chain_key == "soft_response_chain",
                }),
                event_obstacle_contact: None,
                topology_transition: Some(topology_transition),
            },
            control_prediction,
        })
    }

    fn apply_event_obstacle_contact_chain(
        &self,
        candidates: &[EventObstacleContact],
        velocity: [f64; 3],
    ) -> Result<(Vec<EventObstacleChainResponse>, [f64; 3]), String> {
        let mut chain_velocity = velocity;
        let mut responses = Vec::new();
        for candidate in candidates {
            if candidate.event_index != 1743 || candidate.object_id != "event:1743" {
                return Err(
                    "event-obstacle physics is only adjudicated for factory_I02 event 1743"
                        .to_string(),
                );
            }
            let response = apply_event_obstacle_contact_source(
                chain_velocity,
                candidate.normal,
                candidate.motion,
            )?;
            if !response.selected {
                continue;
            }
            chain_velocity = response.velocity_after;
            responses.push(EventObstacleChainResponse {
                loop_index: candidate.loop_index,
                candidate: candidate.clone(),
                response,
            });
        }
        Ok((responses, chain_velocity))
    }

    /// `step_world` (kartrider_vehicle_engine.py:7170-7623): advance one
    /// step, sweep the vehicle volume, and dispatch CONTACT. Full CONTACT
    /// scope: the mixed/pure-soft/pure-hard chains finalize through
    /// [`Self::finalize_collision_chain`]; a single approaching candidate
    /// replays the frame with the hard/soft direct writer injected; with no
    /// selected candidate the sweep only reports the fallback reason and
    /// releases CONTACT.
    #[allow(clippy::too_many_lines)]
    pub fn step_world(
        &mut self,
        frame: &StepFrame,
        world: &dyn World,
        collision_half_extents: [f64; 3],
        collision_center_up: f64,
        collision_selector_state: CollisionSelectorState,
        allow_rail_state1_ground_writer: bool,
        control: Option<(&ControlPayload, Option<&ControlResponseInputModel>)>,
    ) -> Result<WorldStepOutcome, String> {
        let active_domain = self.state.topology.domain;
        let rail_state1_ground_writer = allow_rail_state1_ground_writer
            && active_domain == MotionDomain::Rail
            && self.state.topology.rail_state == 1;
        if matches!(
            active_domain,
            MotionDomain::Reset
                | MotionDomain::NetworkCorrection
                | MotionDomain::Rail
                | MotionDomain::MovingContact
                | MotionDomain::Water
        ) && !rail_state1_ground_writer
        {
            return Err(format!(
                "step_world cannot bypass exclusive {} executor",
                active_domain.value()
            ));
        }
        let state_at_frame_start = self.state.clone();
        let start = self.state.position;
        let start_matrix = self.state.matrix;
        // `result = self.step_control(frame, control_model)` when the frame
        // carries a control payload, else `self.step(frame)` (:7217-7222).
        let (mut outcome, mut control_prediction) = match control {
            Some((control_payload, control_model)) => {
                if control_model.is_none() && control_payload.input_mode != "exact_drive_basis" {
                    return Err("empirical control frames require a control_model".to_string());
                }
                let control_outcome = self.step_control(frame, control_payload, control_model)?;
                (
                    control_outcome.step,
                    Some(control_outcome.control_prediction),
                )
            }
            None => (self.step(frame)?, None),
        };
        let end = self.state.position;
        let end_matrix = self.state.matrix;
        let start_up = matrix_columns(start_matrix)[2];
        let end_up = matrix_columns(end_matrix)[2];
        let sweep_start = [
            start[0] + start_up[0] * collision_center_up,
            start[1] + start_up[1] * collision_center_up,
            start[2] + start_up[2] * collision_center_up,
        ];
        let sweep_end = [
            end[0] + end_up[0] * collision_center_up,
            end[1] + end_up[1] * collision_center_up,
            end[2] + end_up[2] * collision_center_up,
        ];
        // collision_half_extents is always provided by the only caller, so
        // the sweep shape is always "obb_rotating" (:7248).
        let capabilities = world.capabilities();
        let candidates: Vec<WorldObbOverlap> = if capabilities.sweep_obb_rotating_all {
            let query_at_start = capabilities.query_at_step_start;
            world
                .sweep_obb_rotating_all(
                    sweep_start,
                    if query_at_start {
                        sweep_start
                    } else {
                        sweep_end
                    },
                    start_matrix,
                    if query_at_start {
                        start_matrix
                    } else {
                        end_matrix
                    },
                    collision_half_extents,
                )
                .ok_or_else(|| {
                    "sweep_obb_rotating_all capability reported present but returned no \
                     result"
                        .to_string()
                })?
        } else if capabilities.sweep_obb_rotating {
            let candidate = world
                .sweep_obb_rotating(
                    sweep_start,
                    sweep_end,
                    start_matrix,
                    end_matrix,
                    collision_half_extents,
                )
                .ok_or_else(|| {
                    "sweep_obb_rotating capability reported present but returned no result"
                        .to_string()
                })?;
            if candidate.hit {
                vec![candidate]
            } else {
                Vec::new()
            }
        } else {
            // Python raises AttributeError here; a structured error is the
            // honest Rust analog (parity scenarios always provide the sweep).
            return Err(
                "world collision sweep requires world.sweep_obb_rotating_all or \
                 world.sweep_obb_rotating"
                    .to_string(),
            );
        };
        let shape = "obb_rotating";
        let event_obstacle_contact = if capabilities.query_event_obstacle_contacts {
            let time_ms = i64::try_from(state_at_frame_start.time_ms)
                .map_err(|_| "event-obstacle time_ms exceeds i64".to_string())?;
            let delta_ms = frame.elapsed_ms as i64;
            let event_candidates = world
                .query_event_obstacle_contacts(
                    sweep_start,
                    start_matrix,
                    collision_half_extents,
                    time_ms,
                    delta_ms,
                )
                .ok_or_else(|| {
                    "query_event_obstacle_contacts capability reported present but returned no result"
                        .to_string()
                })?;
            let (responses, velocity_after) = self.apply_event_obstacle_contact_chain(
                &event_candidates,
                outcome.state_after.velocity,
            )?;
            let info = EventObstacleContactInfo {
                candidates: event_candidates,
                responses,
            };
            if !info.responses.is_empty() {
                let position = evaluate_position_integrator(
                    start,
                    velocity_after,
                    frame.elapsed_ms / 1000.0,
                    None,
                    0,
                    false,
                    0.0,
                    None,
                    None,
                )?
                .position_90_after;
                self.state.velocity = velocity_after;
                self.state.position = position;
                if self.numeric_mode.is_float32() {
                    f32_vector(&mut self.state.velocity);
                    f32_vector(&mut self.state.position);
                }
                let topology_transition = self.commit_collision_topology_for(
                    info.responses.len() as i64,
                    rail_state1_ground_writer,
                )?;
                outcome.state_after = self.state.clone();
                return Ok(WorldStepOutcome {
                    step: outcome,
                    world_sweep: WorldSweepInfo {
                        shape,
                        start,
                        end,
                        candidate: None,
                        candidates,
                        response_dispatched: true,
                        response_complete: true,
                        dispatch_reason: "approaching_event_dynamic_obstacle_contact".to_string(),
                        dispatch_authority: Some("runtime_GoItemObstacle_kind90_kind91"),
                        angular_response: None,
                        soft_base_response: None,
                        chain: None,
                        event_obstacle_contact: Some(info),
                        topology_transition: Some(topology_transition),
                    },
                    control_prediction,
                });
            }
            Some(info)
        } else {
            None
        };
        let mut candidate: Option<WorldObbOverlap> = None;
        let dispatch_reason: String;
        let mut angular_response: Option<CollisionChainAngularResponse> = None;
        let mut soft_base_response: Option<SoftBaseResponse> = None;
        let mut response_dispatched = false;
        let mut response_complete = false;
        // collision_dispatch == "hard_solid" from the only caller (:5921);
        // collision_half_extents is always Some on this path, so the
        // "requires_obb_vehicle_volume" reason (:7263-7264) is unreachable.
        if candidates.is_empty() {
            dispatch_reason = "no_hit".to_string();
        } else {
            // Runtime multi-contact epochs chain wall velocity writers in
            // authored loop order: writer[n].post == writer[n+1].pre.
            // (:7269-7297)
            let selector_state = self.resolve_collision_selector_state(collision_selector_state);
            let chain_velocity = outcome.state_after.velocity;
            let chain_angular_velocity = outcome.state_after.angular_velocity;
            let mixed_chain = self.apply_collision_chain(
                &candidates,
                start_matrix,
                end_matrix,
                chain_velocity,
                chain_angular_velocity,
                &selector_state,
                None,
                true,
                true,
            )?;
            if mixed_chain.responses.len() >= 2
                && mixed_chain.branches == BTreeSet::from(["hard_contact", "soft_contact"])
            {
                return self.finalize_collision_chain(
                    outcome,
                    control_prediction,
                    &mixed_chain,
                    frame,
                    start,
                    start_matrix,
                    start_matrix,
                    end_matrix,
                    end,
                    collision_half_extents,
                    collision_center_up,
                    &candidates,
                    "mixed_response_chain",
                    "approaching_mixed_contact_chain",
                    "runtime_mixed_kind4_kind8_same_epoch_post_to_next_pre",
                    None,
                    false,
                    rail_state1_ground_writer,
                );
            }
            let soft_chain = self.apply_collision_chain(
                &candidates,
                start_matrix,
                end_matrix,
                chain_velocity,
                chain_angular_velocity,
                &selector_state,
                Some("soft_contact"),
                true,
                true,
            )?;
            if soft_chain.responses.len() >= 2 && !soft_chain.crossed_branch {
                let soft_base = match &soft_chain
                    .responses
                    .last()
                    .expect("len >= 2 implies a last response")
                    .velocity_response
                {
                    CollisionChainVelocityResponse::Soft(source) => source.clone(),
                    CollisionChainVelocityResponse::Hard(_) => {
                        unreachable!("a soft-only chain carries no hard response")
                    }
                };
                return self.finalize_collision_chain(
                    outcome,
                    control_prediction,
                    &soft_chain,
                    frame,
                    start,
                    start_matrix,
                    start_matrix,
                    end_matrix,
                    end,
                    collision_half_extents,
                    collision_center_up,
                    &candidates,
                    "soft_response_chain",
                    "approaching_soft_contact_chain",
                    "runtime_kind8_same_epoch_post_to_next_pre",
                    Some(SoftBaseResponse::Full(soft_base)),
                    false,
                    rail_state1_ground_writer,
                );
            }
            let hard_chain = self.apply_collision_chain(
                &candidates,
                start_matrix,
                end_matrix,
                chain_velocity,
                chain_angular_velocity,
                &selector_state,
                Some("hard_contact"),
                true,
                false,
            )?;
            if hard_chain.responses.len() >= 2 && !hard_chain.crossed_branch {
                return self.finalize_collision_chain(
                    outcome,
                    control_prediction,
                    &hard_chain,
                    frame,
                    start,
                    start_matrix,
                    start_matrix,
                    end_matrix,
                    end,
                    collision_half_extents,
                    collision_center_up,
                    &candidates,
                    "hard_response_chain",
                    "approaching_hard_contact_chain",
                    "runtime_same_epoch_writer_post_to_next_pre",
                    None,
                    true,
                    rail_state1_ground_writer,
                );
            }
            // Single-candidate selection loop (:7416-7441): the first
            // approaching candidate wins; the interpolated axes are evaluated
            // for every candidate before its gate (degenerate-matrix errors
            // propagate exactly like Python). This loop observes the state
            // defaults WITHOUT the collision_selector_state overlay.
            let mut axes: Option<[[f64; 3]; 3]> = None;
            for current_candidate in &candidates {
                let current_axes = interpolate_axes(
                    start_matrix,
                    end_matrix,
                    current_candidate.fraction,
                    self.numeric_mode,
                )?;
                let current_selection = select_collision_dispatch(
                    current_candidate,
                    outcome.state_after.velocity,
                    self.state.collision_arg_threshold,
                    self.state.collision_e38,
                    self.state.collision_e4c,
                    self.numeric_mode,
                );
                if current_selection.selected {
                    // Python reassigns `candidate["loop_index"] =
                    // current_candidate.get("loop_index", candidate_index)`
                    // (:7438) — identity for typed world rows.
                    candidate = Some(current_candidate.clone());
                    axes = Some(current_axes);
                    break;
                }
            }
            if candidate.is_none() {
                let fallback = candidates[0].clone();
                axes = Some(interpolate_axes(
                    start_matrix,
                    end_matrix,
                    fallback.fraction,
                    self.numeric_mode,
                )?);
                candidate = Some(fallback);
            }
            let axes = axes.expect("axes assigned on both paths");
            // Re-dispatch with the merged selector state (:7451-7473).
            let selection = select_collision_dispatch(
                candidate
                    .as_ref()
                    .expect("candidate assigned on both paths"),
                outcome.state_after.velocity,
                selector_state.arg_threshold,
                selector_state.e38,
                selector_state.e4c,
                self.numeric_mode,
            );
            if !selection.selected {
                dispatch_reason = selection.reason.clone();
            } else if frame.model.hard_contact.is_some() || frame.model.soft_contact.is_some() {
                dispatch_reason = "frame_already_selected_contact_response".to_string();
            } else if selection.branch == Some("soft_contact") {
                let normal = selection.normal.expect("selected implies a normal");
                let basis_ab8 = axes[0];
                let basis_ad0 = [-axes[1][0], -axes[1][1], -axes[1][2]];
                let mut soft_optional_active = collision_selector_state.soft_optional_active;
                // `exact_soft_keys.issubset(selector_state)` (:7486): the
                // merged selector state always carries the five exact keys
                // (the overlay cannot remove a key), so this is constant.
                let exact_soft_state = true;
                if soft_optional_active.is_none() {
                    let schedule_selector_af4 = frame
                        .model
                        .recovery
                        .schedule
                        .as_ref()
                        .and_then(|schedule| schedule.selector_af4);
                    let damping_special_gate = frame
                        .model
                        .common_damping
                        .as_ref()
                        .map(|damping| damping.special_gate);
                    if schedule_selector_af4 == Some(true) || damping_special_gate == Some(false) {
                        soft_optional_active = Some(false);
                    }
                }
                if exact_soft_state || soft_optional_active == Some(false) {
                    // `if exact_soft_state:` (:7493) is constant (see above),
                    // so the base-only source (:7504) is unreachable on this
                    // path.
                    let soft_source = evaluate_soft_contact_source(
                        outcome.state_after.velocity,
                        normal,
                        basis_ad0,
                        basis_ab8,
                        selector_state.drive_selected_slot,
                        selector_state.drive_raw_aa0,
                        selector_state.drive_sign_aac,
                        selector_state.selector_af4,
                        selector_state.flag_188,
                    )?;
                    soft_base_response = Some(SoftBaseResponse::Full(soft_source.clone()));
                    // Runtime ordering is mixed: the collision C0 write is
                    // visible to the current position writer even though it
                    // occurs after the ordinary linear integration. Replay
                    // that transaction first; CC is committed separately to
                    // the current orientation below. (:7507-7523)
                    let matrix_after_no_contact = self.state.matrix;
                    let mut replay_frame = frame.clone();
                    replay_frame.model.soft_contact = Some(SoftContactArgs {
                        velocity_c0_before: Some(outcome.state_after.velocity),
                        velocity_delta: soft_source.velocity_source,
                        selection_authority: Some("collision_solver_dispatch".to_string()),
                        scheduler_phase: Some("post_linear_pre_pose".to_string()),
                    });
                    self.state = state_at_frame_start.clone();
                    let (replayed, replayed_prediction) = match control {
                        Some((control_payload, control_model)) => {
                            let control_outcome =
                                self.step_control(&replay_frame, control_payload, control_model)?;
                            (
                                control_outcome.step,
                                Some(control_outcome.control_prediction),
                            )
                        }
                        None => (self.step(&replay_frame)?, None),
                    };
                    outcome = replayed;
                    control_prediction = replayed_prediction;
                    self.state.matrix = matrix_after_no_contact;
                    if exact_soft_state && soft_source.optional_active {
                        self.state.angular_velocity[2] += soft_source.angular_z_delta;
                        if self.numeric_mode.is_float32() {
                            f32_vector(&mut self.state.angular_velocity);
                        }
                    }
                    let soft_angular = evaluate_soft_contact_angular_source(
                        self.state.velocity,
                        self.state.angular_velocity,
                        normal,
                        basis_ad0,
                        basis_ab8,
                        self.numeric_mode.is_float32(),
                    );
                    if soft_angular.writeback_selected {
                        self.state.angular_velocity = [
                            self.state.angular_velocity[0] + soft_angular.source[0],
                            self.state.angular_velocity[1] + soft_angular.source[1],
                            self.state.angular_velocity[2] + soft_angular.source[2],
                        ];
                        if self.numeric_mode.is_float32() {
                            f32_vector(&mut self.state.angular_velocity);
                        }
                    }
                    // The wall CC write precedes the current orientation
                    // commit even though its linear C0 write is scheduled
                    // after the ordinary linear integrator. Rebuild the
                    // current matrix from the frame backup with final CC;
                    // otherwise every contact leaves pose one angular epoch
                    // behind while CC itself remains exact. (:7540-7557)
                    let orientation_after_contact = evaluate_orientation_integrator(
                        start_matrix,
                        self.state.angular_velocity,
                        frame.elapsed_ms / 1000.0,
                        false,
                        false,
                        0,
                        self.numeric_mode.is_float32(),
                    )?;
                    self.state.matrix = orientation_after_contact.matrix_9c_after;
                    if self.numeric_mode.is_float32() {
                        f32_vector(&mut self.state.matrix);
                    }
                    angular_response = Some(CollisionChainAngularResponse::Soft(soft_angular));
                    outcome.state_after = self.state.clone();
                    dispatch_reason = if exact_soft_state && soft_source.optional_active {
                        "approaching_soft_contact_full_response".to_string()
                    } else {
                        "approaching_soft_contact_base_response".to_string()
                    };
                    response_dispatched = true;
                    response_complete = exact_soft_state;
                } else {
                    dispatch_reason =
                        "approaching_soft_contact_optional_or_unknown_source_open".to_string();
                }
            } else {
                let normal = selection.normal.expect("selected implies a normal");
                let basis_ab8 = axes[0];
                let basis_ad0 = [-axes[1][0], -axes[1][1], -axes[1][2]];
                let basis_ac4 = axes[2];
                let speculative_angular_velocity = outcome.state_after.angular_velocity;
                let mut replay_frame = frame.clone();
                replay_frame.model.hard_contact = Some(HardContactArgs {
                    velocity_c0_before: outcome.state_after.velocity,
                    hit_normal: normal,
                    basis_ad0,
                    restitution: None,
                    selection_authority: Some("collision_solver_dispatch".to_string()),
                    scheduler_phase: Some("post_linear_pre_pose".to_string()),
                });
                self.state = state_at_frame_start.clone();
                let (replayed, replayed_prediction) = match control {
                    Some((control_payload, control_model)) => {
                        let control_outcome =
                            self.step_control(&replay_frame, control_payload, control_model)?;
                        (
                            control_outcome.step,
                            Some(control_outcome.control_prediction),
                        )
                    }
                    None => (self.step(&replay_frame)?, None),
                };
                outcome = replayed;
                control_prediction = replayed_prediction;
                let hard_angular = evaluate_hard_contact_angular_response(
                    speculative_angular_velocity,
                    normal,
                    basis_ac4,
                    basis_ab8,
                    basis_ad0,
                    0.1,
                );
                self.state.angular_velocity = hard_angular.angular_velocity_after;
                if self.numeric_mode.is_float32() {
                    f32_vector(&mut self.state.angular_velocity);
                }
                angular_response = Some(CollisionChainAngularResponse::Hard(hard_angular));
                outcome.state_after = self.state.clone();
                dispatch_reason = "approaching_hard_contact".to_string();
                response_dispatched = true;
                response_complete = true;
            }
        }
        // (:7594-7601): the rail-state1 ground writer commits the synthetic
        // ownership-retaining payload on dispatch and keeps ownership (None)
        // otherwise; every other path commits/releases CONTACT.
        let topology_transition = if response_dispatched {
            Some(self.commit_collision_topology_for(1, rail_state1_ground_writer)?)
        } else if rail_state1_ground_writer {
            None
        } else {
            self.release_collision_contact_topology()?
        };
        // `result["state_after"] = asdict(self.state)` (:7603) is
        // unconditional: the CONTACT commit/release above can advance the
        // topology after the step snapshot was taken.
        outcome.state_after = self.state.clone();
        Ok(WorldStepOutcome {
            step: outcome,
            world_sweep: WorldSweepInfo {
                shape,
                start,
                end,
                candidate,
                candidates,
                response_dispatched,
                response_complete,
                dispatch_reason,
                dispatch_authority: if response_dispatched {
                    Some("collision_solver_dispatch")
                } else {
                    None
                },
                angular_response,
                soft_base_response,
                chain: None,
                event_obstacle_contact,
                topology_transition,
            },
            control_prediction,
        })
    }

    /// `step_airborne` (kartrider_vehicle_engine.py:1751-1796): advance AIR
    /// from engine-owned gravity, damping, and rigid-body state.
    pub fn step_airborne(
        &mut self,
        parameters: &VehicleParameters,
        additional_force_world: Option<[f64; 3]>,
        additional_torque_world: Option<[f64; 3]>,
    ) -> Result<AirborneStepOutcome, String> {
        let domain = self.state.topology.domain;
        if !matches!(domain, MotionDomain::AirEntry | MotionDomain::Air) {
            return Err("step_airborne requires AIR-ENTRY or AIR topology".to_string());
        }
        let result = evaluate_airborne_step(
            self.state.position,
            self.state.velocity,
            self.state.angular_velocity,
            self.state.matrix,
            parameters.mass,
            parameters.substep_ms as f64 / 1000.0,
            [0.0, 0.0, parameters.contact_gravity_per_mass],
            parameters.inverse_inertia_matrix,
            parameters.common_damping,
            parameters.directional_damping,
            parameters.damping_runtime_multiplier,
            self.state.soft_flag_188,
            additional_force_world,
            additional_torque_world,
            0,
            self.numeric_mode.as_str(),
        )?;
        self.state.position = result.position_after;
        self.state.velocity = result.velocity_after;
        self.state.angular_velocity = result.angular_velocity_after;
        self.state.matrix = result.matrix_after;
        self.state.force_accumulator = result.force_accumulator;
        self.state.torque_accumulator = result.torque_accumulator;
        if domain == MotionDomain::AirEntry {
            self.state.topology.previous_domain = Some(MotionDomain::AirEntry);
            self.state.topology.domain = MotionDomain::Air;
            self.state.topology.transition_index += 1;
            self.state.topology.domain_step_index = 0;
            self.state.topology.rail_state_elapsed_ms = 0.0;
            self.state.topology.entry_reason = "takeoff_transfer_complete".to_string();
        } else {
            self.state.topology.domain_step_index += 1;
        }
        self.state.time_ms += parameters.substep_ms as u64;
        self.state.step_index += 1;
        Ok(AirborneStepOutcome {
            evaluator: result,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
        })
    }

    /// `step_raw_airborne` (kartrider_vehicle_engine.py:6030-6119): advance
    /// AIR from raw keys with autonomous no-support producers.
    pub fn step_raw_airborne(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        takeoff_contact_bridge_planes: Option<&[WorldGroundResult]>,
        pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'_>>,
    ) -> Result<RawAirborneOutcome, String> {
        let domain_before = self.state.topology.domain;
        let contact_return = if domain_before == MotionDomain::Contact {
            self.state.topology.return_domain
        } else {
            None
        };
        let airborne_contact = domain_before == MotionDomain::Contact
            && matches!(
                contact_return,
                Some(MotionDomain::AirEntry) | Some(MotionDomain::Air)
            );
        if !matches!(domain_before, MotionDomain::AirEntry | MotionDomain::Air) && !airborne_contact
        {
            return Err("step_raw_airborne requires AIR-ENTRY/AIR or airborne CONTACT".to_string());
        }
        if domain_before == MotionDomain::Contact && world.is_none() {
            return Err("airborne CONTACT stepping requires world collision geometry".to_string());
        }
        let bridge_set = takeoff_contact_bridge_planes.map(ground_plane_set_from_world_planes);
        let ground = self.step_raw_ground_correction(
            keys,
            parameters,
            world,
            bridge_set.as_ref(),
            &GroundStepOptions {
                support_active: Some(takeoff_contact_bridge_planes.is_some()),
                pending_ground_plane_provider: if takeoff_contact_bridge_planes.is_some() {
                    pending_ground_plane_provider
                } else {
                    None
                },
                ..Default::default()
            },
        )?;
        let domain_after_producers = self.state.topology.domain;
        if domain_before == MotionDomain::Contact {
            if !matches!(
                domain_after_producers,
                MotionDomain::Contact | MotionDomain::Air
            ) {
                // Python RuntimeError.
                return Err(format!(
                    "airborne CONTACT producer changed topology to unsupported owner {}",
                    domain_after_producers.value()
                ));
            }
        } else if domain_after_producers == MotionDomain::Contact {
            // The autonomous collision executor acquired ownership during the
            // first or a sustained airborne substep. (Unreachable in the M1
            // no-dispatch world: CONTACT acquisition errors in step_world.)
        } else if domain_after_producers != domain_before {
            // Python RuntimeError.
            return Err(format!(
                "airborne producer changed topology to unsupported owner {}",
                domain_after_producers.value()
            ));
        } else if domain_before == MotionDomain::AirEntry {
            self.state.topology.previous_domain = Some(MotionDomain::AirEntry);
            self.state.topology.domain = MotionDomain::Air;
            self.state.topology.transition_index += 1;
            self.state.topology.domain_step_index = 0;
            self.state.topology.rail_state_elapsed_ms = 0.0;
            self.state.topology.entry_reason = "takeoff_transfer_complete".to_string();
        } else {
            self.state.topology.domain_step_index += 1;
        }
        let bridged = takeoff_contact_bridge_planes.is_some();
        Ok(RawAirborneOutcome {
            ground,
            airborne_producer_contract: AirborneProducerContract {
                control_authority: "raw_keys",
                support_authority: if bridged {
                    "explicit_launch_retained_wheel_contact_bridge"
                } else {
                    "engine_explicit_no_support"
                },
                contact_slots: if bridged {
                    "autonomous_world_contacts"
                } else {
                    "four_inactive"
                },
                common_gravity: "B29E70_field_604_times_mass",
                vehicle_producers: "runtime_common_drive_and_correction_chain",
            },
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
        })
    }
}

/// Takeoff candidate view shared by the two selection paths in
/// `step_raw_topology` (authored jump manifold and the
/// `world.query_takeoff_contact` selector, optionally confirmed by an exact
/// current-pose hard OBB face).
struct TakeoffCandidate {
    normal: Option<[f64; 3]>,
    source: Option<String>,
}

/// Landing geometry view (kartrider_vehicle_engine.py:6520-6627).
struct LandingGeometry {
    normal: [f64; 3],
    selector_authority: &'static str,
    active_support_slot_count: i64,
}

impl KartRiderVehicleEngine {
    /// `step_raw_topology` (kartrider_vehicle_engine.py:6228-6917):
    /// autonomously select and execute grounded/airborne topology.
    ///
    /// Initial RAIL capture acquires ownership through
    /// [`Self::acquire_autonomous_rail`] (state-1 ground-writer or the
    /// `step_raw_rail` writer chain); MOVING-CONTACT capture acquires and
    /// steps through [`Self::acquire_and_step_autonomous_moving_contact`].
    /// An accepted authoritative network packet preempts the selector
    /// through [`Self::begin_and_step_autonomous_network_correction`]
    /// (:6294-6312).
    #[allow(clippy::too_many_lines)]
    pub fn step_raw_topology(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_planes: Option<&[WorldGroundResult]>,
        pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'_>>,
    ) -> Result<RawTopologyOutcome, String> {
        let current_owner = self.state.topology.domain;
        if matches!(
            current_owner,
            MotionDomain::Reset
                | MotionDomain::NetworkCorrection
                | MotionDomain::Rail
                | MotionDomain::MovingContact
                | MotionDomain::Water
        ) {
            return Err(format!(
                "step_raw_topology cannot bypass exclusive {} executor",
                current_owner.value()
            ));
        }
        validate_raw_control_keys(keys)?;
        let support = self.inspect_autonomous_support(keys, parameters, world, ground_planes)?;
        let current = self.state.topology.domain;
        let mut topology_support_contact_count = support.support_contact_count;
        let mut pending_landing_candidate: Option<HardOverlapCandidate> = None;
        if self.state.takeoff_separation_pending {
            // A hard takeoff and landing are the paired B2DC80 OBB
            // transactions observed by the runtime. Suspension rays can
            // remain geometrically reachable throughout this authored jump,
            // so they cannot release the takeoff latch on their own.
            if !world_caps(world).overlap_obb_all {
                // Legacy/minimal worlds own takeoff and landing through
                // query_takeoff_contact/query_landing_contact plus the wheel
                // manifold. They cannot prove the paired hard-OBB release,
                // so retaining the latch would suppress support forever
                // (handoff §4; kartrider_vehicle_engine.py:6263-6268).
                self.state.takeoff_separation_pending = false;
            } else {
                topology_support_contact_count = 0;
                if matches!(current, MotionDomain::AirEntry | MotionDomain::Air)
                    && self.state.velocity[2] < 0.0
                {
                    pending_landing_candidate = self.select_current_pose_hard_overlap(
                        world,
                        parameters,
                        "autonomous_world_hard_landing_overlap",
                        None,
                    )?;
                    if pending_landing_candidate.is_some() {
                        topology_support_contact_count = support.support_contact_count.max(1);
                        self.state.takeoff_separation_pending = false;
                    }
                }
            }
        }
        if Self::support_selects_authored_reset(&support) {
            let reset = self.enter_authored_reset_surface(support)?;
            return Ok(RawTopologyOutcome {
                support_observation: reset.support_observation.clone(),
                topology_decision: reset.topology_decision.clone(),
                topology_after: reset.topology_after.clone(),
                state_after: reset.state_after.clone(),
                raw_topology_contract: reset.raw_topology_contract.clone(),
                executor: TopologyExecutorOutcome::ResetSurface(Box::new(reset)),
                rail_captured: false,
                post_ground_rail_capture: None,
                rail_state1_ground_writer: None,
                autonomous_rail_selector: None,
                takeoff_state_transfer: None,
                landing_state_transfer: None,
                authored_special_trigger: None,
            });
        }
        let network_candidate = self.inspect_autonomous_network_correction(world)?;
        if network_candidate.correction_active {
            // kartrider_vehicle_engine.py:6294-6312: preempt the grounded
            // topology selector with the accepted authoritative transaction.
            let record = NetworkCorrectionRecord::from_inspection(&network_candidate)?;
            let entry =
                self.begin_and_step_autonomous_network_correction(parameters, world, &record)?;
            let preempted_contract = || RawTopologyContract {
                control_authority: "preempted_by_authoritative_state".to_string(),
                geometry_authority: "candidate_pose_world_selectors_on_convergence".to_string(),
                selector_authority: "world_query_network_correction".to_string(),
                scheduler_authority: "motion_topology_state_machine".to_string(),
                landing_transfer_authority: "deferred_after_air_resume".to_string(),
                takeoff_transfer_authority: "not_selected".to_string(),
            };
            // The Python network result carries no `topology_decision` key;
            // the parity projections never read this filler. The forward
            // entry's begin transition is real data; the rollback path has
            // no single transition, so it is marked as a placeholder.
            let topology_decision = match &entry {
                AutonomousCorrectionEntry::Forward(forward) => forward
                    .entry
                    .as_ref()
                    .expect("begin_and_step sets network_correction_entry")
                    .topology_transition
                    .clone(),
                AutonomousCorrectionEntry::Rollback(rollback) => TopologyDecision {
                    selected_domain: rollback.topology_after.domain,
                    transitioned: true,
                    reason: "network_correction_rollback_resimulated".to_string(),
                    state_before: rollback.topology_after.clone(),
                    state_after: rollback.topology_after.clone(),
                    observation: TopologyObservation::default(),
                    authority: "rust_placeholder_no_python_counterpart".to_string(),
                },
            };
            return Ok(RawTopologyOutcome {
                executor: TopologyExecutorOutcome::NetworkCorrection(Box::new(
                    NetworkCorrectionPreemption {
                        entry,
                        entry_support_observation: support.clone(),
                        raw_topology_contract: preempted_contract(),
                    },
                )),
                support_observation: support,
                topology_decision,
                rail_captured: false,
                post_ground_rail_capture: None,
                rail_state1_ground_writer: None,
                autonomous_rail_selector: None,
                takeoff_state_transfer: None,
                landing_state_transfer: None,
                authored_special_trigger: None,
                topology_after: self.state.topology.clone(),
                state_after: self.state.clone(),
                raw_topology_contract: preempted_contract(),
            });
        }
        // Initial RAIL ownership is selected by the outer-frame topology
        // selector, not by every 1/2 ms integration substep. The candidate
        // is sampled every substep, latched, and consumed only when the
        // direct frame schedule publishes a commit (:6313-6338).
        let rail_selector_due = parameters.initial_rail_capture_commit.unwrap_or(true);
        let observed_rail_candidate = self.inspect_autonomous_rail_capture(world)?;
        if observed_rail_candidate.contact_active {
            self.state.rail_capture_latch = Some(crate::state::RailCaptureLatch {
                contact_active: true,
                rail_state: observed_rail_candidate.rail_state,
                selector: observed_rail_candidate.selector.clone().unwrap_or_default(),
                authority: Some(observed_rail_candidate.authority.to_string()),
            });
        }
        let mut rail_candidate_rail_state: Option<i64> = None;
        let mut rail_candidate_selector: BTreeMap<String, JsonValue> = BTreeMap::new();
        let rail_candidate_active = if rail_selector_due && self.state.rail_capture_latch.is_some()
        {
            // Python deep-copies the latch into `rail_candidate` and clears
            // the latch; the selector/rail_state feed
            // `acquire_autonomous_rail(inspected=rail_candidate)`.
            let latch = self.state.rail_capture_latch.take().unwrap();
            rail_candidate_rail_state = latch.rail_state;
            rail_candidate_selector = latch.selector.clone();
            latch.contact_active
        } else {
            false
        };
        let moving_candidate = if world_caps(world).query_moving_contact {
            self.inspect_autonomous_moving_contact(world, None, Some(parameters.substep_ms))?
        } else {
            MovingContactInspection {
                contact_active: false,
                record: None,
                object_id: None,
                authority: "world_has_no_moving_contact_selector",
            }
        };
        if rail_candidate_active && moving_candidate.contact_active {
            return Err(
                "simultaneous initial RAIL and MOVING-CONTACT precedence is unproven".to_string(),
            );
        }
        if rail_candidate_active {
            // Python acquires RAIL ownership here (`acquire_autonomous_rail`
            // always captures on an active inspected candidate).
            let inspected = RailCaptureInspection {
                contact_active: true,
                rail_state: rail_candidate_rail_state,
                selector: Some(rail_candidate_selector.clone()),
                authority: "world_query_rail_contact",
            };
            if let Some(execution) = self.acquire_autonomous_rail(
                keys,
                parameters,
                world,
                Some(inspected),
                Some(&support),
            )? {
                let contract = RawTopologyContract {
                    control_authority: "raw_keys".to_string(),
                    geometry_authority: "world_query_rail_contact".to_string(),
                    selector_authority: "active_decoded_rail_context".to_string(),
                    scheduler_authority: "motion_topology_state_machine".to_string(),
                    landing_transfer_authority: "not_selected".to_string(),
                    takeoff_transfer_authority: "not_selected".to_string(),
                };
                return Ok(match execution {
                    RailCaptureExecution::GroundWriter(rail_capture) => RawTopologyOutcome {
                        support_observation: support.clone(),
                        topology_decision: rail_capture.topology_decision.clone(),
                        topology_after: rail_capture.topology_after.clone(),
                        state_after: rail_capture.state_after.clone(),
                        rail_captured: true,
                        post_ground_rail_capture: None,
                        rail_state1_ground_writer: Some(
                            rail_capture.rail_state1_ground_writer.clone(),
                        ),
                        autonomous_rail_selector: Some(
                            rail_capture.autonomous_rail_selector.clone(),
                        ),
                        takeoff_state_transfer: None,
                        landing_state_transfer: None,
                        authored_special_trigger: None,
                        raw_topology_contract: contract,
                        executor: TopologyExecutorOutcome::RailCapture(Box::new(
                            RailCaptureExecution::GroundWriter(rail_capture),
                        )),
                    },
                    RailCaptureExecution::RawRail(raw_capture) => RawTopologyOutcome {
                        support_observation: support.clone(),
                        topology_decision: raw_capture.rail.topology_decision.clone(),
                        topology_after: raw_capture.rail.topology_after.clone(),
                        state_after: raw_capture.rail.state_after.clone(),
                        rail_captured: true,
                        post_ground_rail_capture: None,
                        rail_state1_ground_writer: None,
                        autonomous_rail_selector: Some(
                            raw_capture.autonomous_rail_selector.clone(),
                        ),
                        takeoff_state_transfer: None,
                        landing_state_transfer: None,
                        authored_special_trigger: None,
                        raw_topology_contract: contract,
                        executor: TopologyExecutorOutcome::RailCapture(Box::new(
                            RailCaptureExecution::RawRail(raw_capture),
                        )),
                    },
                });
            }
        }
        if moving_candidate.contact_active {
            // Python `acquire_and_step_autonomous_moving_contact(parameters,
            // moving_candidate)` consumes the merged inspection mapping.
            let record = merged_moving_contact_record(&moving_candidate)?;
            let capture = self.acquire_and_step_autonomous_moving_contact(parameters, &record)?;
            let contract = RawTopologyContract {
                control_authority: "raw_keys_no_moving_frame_drive_writer".to_string(),
                geometry_authority: "world_query_moving_contact".to_string(),
                selector_authority: "active_dynamic_body_contact".to_string(),
                scheduler_authority: "motion_topology_state_machine".to_string(),
                landing_transfer_authority: "not_selected".to_string(),
                takeoff_transfer_authority: "not_selected".to_string(),
            };
            return Ok(RawTopologyOutcome {
                support_observation: support.clone(),
                topology_decision: capture.acquisition.topology_transition.clone(),
                topology_after: capture.step.topology_after.clone(),
                state_after: capture.step.state_after.clone(),
                rail_captured: false,
                post_ground_rail_capture: None,
                rail_state1_ground_writer: None,
                autonomous_rail_selector: None,
                takeoff_state_transfer: None,
                landing_state_transfer: None,
                authored_special_trigger: None,
                raw_topology_contract: contract,
                executor: TopologyExecutorOutcome::MovingContactCapture(Box::new(capture)),
            });
        }
        let authored_trigger_edge =
            self.apply_autonomous_authored_trigger(world, parameters, false, Some(&support))?;
        let mut takeoff_geometry: Option<([f64; 3], f64)> = None;
        let mut candidate: Option<TakeoffCandidate> = None;
        if matches!(current, MotionDomain::Ground | MotionDomain::GroundSlip) {
            let mut jump_normals: Vec<[f64; 3]> = Vec::new();
            for (index, active) in support.active_slots.iter().enumerate() {
                let token = support.planes[index].road_surface.trim().to_lowercase();
                if *active && (token == "jump" || token == "점프") {
                    jump_normals.push(support.planes[index].normal);
                }
            }
            if !jump_normals.is_empty() {
                for normal in &jump_normals {
                    let length = length3(*normal, self.numeric_mode.is_float32());
                    if !(0.999..=1.001).contains(&length) {
                        return Err("jump support normal must be normalized".to_string());
                    }
                }
                let coherent_threshold = if self.numeric_mode.is_float32() {
                    fsub(1.0, 1e-5)
                } else {
                    1.0 - 1e-5
                };
                let coherent = jump_normals[1..].iter().all(|normal| {
                    let dot = if self.numeric_mode.is_float32() {
                        dot3_f32(jump_normals[0], *normal)
                    } else {
                        dot3_sum(jump_normals[0], *normal)
                    };
                    dot >= coherent_threshold
                });
                if coherent {
                    candidate = Some(TakeoffCandidate {
                        normal: Some(jump_normals[0]),
                        source: Some("authored_jump_support_manifold".to_string()),
                    });
                }
            }
            if candidate.is_none() && world_caps(world).query_takeoff_contact {
                let active_slots: Vec<i64> = support
                    .active_slots
                    .iter()
                    .map(|active| i64::from(*active))
                    .collect();
                let contact = world
                    .expect("capability probe implies a world")
                    .query_takeoff_contact(
                        self.state.position,
                        self.state.velocity,
                        &support.planes,
                        &active_slots,
                    )
                    .ok_or_else(|| {
                        "query_takeoff_contact capability reported present but returned no \
                         result"
                            .to_string()
                    })?;
                if let Some(contact) = contact {
                    // Python bool(candidate.get("selected",
                    // candidate.get("hit", True))).
                    let selected = contact.selected.or(contact.hit).unwrap_or(true);
                    if selected {
                        let selector_source = contact
                            .source
                            .unwrap_or_else(|| "world_query_takeoff_contact".to_string());
                        candidate = if let (Some(face_id), true) = (
                            contact.face_id.as_deref(),
                            world_caps(world).overlap_obb_all,
                        ) {
                            self.select_current_pose_hard_overlap(
                                world,
                                parameters,
                                &selector_source,
                                Some(face_id),
                            )?
                            .map(|overlap| TakeoffCandidate {
                                normal: Some(overlap.overlap.normal),
                                source: Some(overlap.source),
                            })
                        } else {
                            Some(TakeoffCandidate {
                                normal: contact.normal,
                                source: Some(selector_source),
                            })
                        };
                    }
                }
            }
        }
        if let Some(candidate) = &candidate {
            let Some(normal) = candidate.normal else {
                return Err("selected takeoff candidate requires normal".to_string());
            };
            let length = length3(normal, self.numeric_mode.is_float32());
            if length == 0.0 {
                return Err("takeoff candidate normal must be nonzero".to_string());
            }
            if !(0.999..=1.001).contains(&length) {
                return Err("takeoff candidate normal must be normalized".to_string());
            }
            let approach = if self.numeric_mode.is_float32() {
                dot3_f32(self.state.velocity, normal)
            } else {
                dot3_sum(self.state.velocity, normal)
            };
            if approach < 0.0 {
                takeoff_geometry = Some((normal, approach));
            } else if candidate.source.as_deref() != Some("authored_jump_support_manifold") {
                return Err("selected takeoff candidate must be approaching".to_string());
            }
        }
        let observation = TopologyObservation {
            support_contact_count: topology_support_contact_count,
            surface_mode: support.surface_mode,
            launch_requested: takeoff_geometry.is_some(),
            ..Default::default()
        };
        let mut landing_geometry: Option<LandingGeometry> = None;
        let landing_source = matches!(current, MotionDomain::AirEntry | MotionDomain::Air)
            || (current == MotionDomain::Contact
                && matches!(
                    self.state.topology.return_domain,
                    Some(MotionDomain::AirEntry) | Some(MotionDomain::Air)
                ));
        if landing_source && topology_support_contact_count > 0 {
            if let Some(pending) = &pending_landing_candidate {
                let normal = pending.overlap.normal;
                let length = length3(normal, self.numeric_mode.is_float32());
                if !(0.999..=1.001).contains(&length) {
                    return Err("landing hard-contact normal must be normalized Vec3".to_string());
                }
                landing_geometry = Some(LandingGeometry {
                    normal,
                    selector_authority: "autonomous_landing_hard_contact_overlap",
                    active_support_slot_count: support.support_contact_count,
                });
            } else {
                let mut active_normals: Vec<[f64; 3]> = Vec::new();
                for (index, active) in support.active_slots.iter().enumerate() {
                    if !active {
                        continue;
                    }
                    let normal = support.planes[index].normal;
                    let length = length3(normal, self.numeric_mode.is_float32());
                    if length == 0.0 {
                        return Err("landing support normal must be nonzero".to_string());
                    }
                    active_normals.push(if self.numeric_mode.is_float32() {
                        [
                            fdiv(normal[0], length),
                            fdiv(normal[1], length),
                            fdiv(normal[2], length),
                        ]
                    } else {
                        [normal[0] / length, normal[1] / length, normal[2] / length]
                    });
                }
                let reference = active_normals[0];
                let coherent_threshold = if self.numeric_mode.is_float32() {
                    fsub(1.0, 1e-5)
                } else {
                    1.0 - 1e-5
                };
                let coherent = active_normals[1..].iter().all(|normal| {
                    let dot = if self.numeric_mode.is_float32() {
                        dot3_f32(reference, *normal)
                    } else {
                        dot3_sum(reference, *normal)
                    };
                    dot >= coherent_threshold
                });
                if coherent {
                    landing_geometry = Some(LandingGeometry {
                        normal: reference,
                        selector_authority: "autonomous_coherent_wheel_contact_manifold",
                        active_support_slot_count: active_normals.len() as i64,
                    });
                } else {
                    if !world_caps(world).query_landing_contact {
                        return Err("divergent landing support normals require \
                             world.query_landing_contact"
                            .to_string());
                    }
                    let active_slots: Vec<i64> = support
                        .active_slots
                        .iter()
                        .map(|active| i64::from(*active))
                        .collect();
                    let contact = world
                        .expect("capability probe implies a world")
                        .query_landing_contact(
                            self.state.position,
                            self.state.velocity,
                            &support.planes,
                            &active_slots,
                        )
                        .ok_or_else(|| {
                            "query_landing_contact capability reported present but returned \
                             no result"
                                .to_string()
                        })?;
                    let Some(contact) = contact else {
                        return Err(
                            "query_landing_contact must return a candidate mapping with \
                             normal"
                                .to_string(),
                        );
                    };
                    let Some(normal) = contact.normal else {
                        return Err(
                            "query_landing_contact must return a candidate mapping with \
                             normal"
                                .to_string(),
                        );
                    };
                    let length = length3(normal, self.numeric_mode.is_float32());
                    if length == 0.0 {
                        return Err(
                            "landing collision candidate normal must be nonzero".to_string()
                        );
                    }
                    if !(0.999..=1.001).contains(&length) {
                        return Err(
                            "landing collision candidate normal must be normalized".to_string()
                        );
                    }
                    landing_geometry = Some(LandingGeometry {
                        normal,
                        selector_authority: "autonomous_landing_collision_candidate",
                        active_support_slot_count: active_normals.len() as i64,
                    });
                }
            }
        }
        let mut deferred_contact_owner = false;
        let mut executor_domain: Option<MotionDomain> = None;
        let topology_decision: TopologyDecision;
        if current == MotionDomain::Contact {
            let return_domain = self
                .state
                .topology
                .return_domain
                .ok_or_else(|| "CONTACT ownership requires return_domain".to_string())?;
            let grounded_return = matches!(
                return_domain,
                MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
            );
            let airborne_return =
                matches!(return_domain, MotionDomain::AirEntry | MotionDomain::Air);
            if (grounded_return && support.support_contact_count > 0)
                || (airborne_return && support.support_contact_count == 0)
            {
                deferred_contact_owner = true;
                executor_domain = Some(if grounded_return {
                    MotionDomain::Ground
                } else {
                    MotionDomain::Air
                });
                let snapshot = self.state.topology.clone();
                topology_decision = TopologyDecision {
                    selected_domain: MotionDomain::Contact,
                    transitioned: false,
                    reason: "contact_release_deferred_to_collision_geometry".to_string(),
                    state_before: snapshot.clone(),
                    state_after: snapshot,
                    observation: observation.clone(),
                    authority: "engine_owned_topology_selectors".to_string(),
                };
            } else {
                topology_decision = self.advance_motion_topology(&observation)?;
            }
        } else {
            topology_decision = self.advance_motion_topology(&observation)?;
        }
        let selected = if deferred_contact_owner {
            executor_domain.expect("set when deferred")
        } else {
            self.state.topology.domain
        };
        let mut landing_transfer: Option<TakeoffLandingTransfer> = None;
        let mut takeoff_transfer: Option<TakeoffLandingTransfer> = None;
        let executor: TopologyExecutorOutcome;
        if matches!(selected, MotionDomain::AirEntry | MotionDomain::Air) {
            if selected == MotionDomain::AirEntry
                && topology_decision.reason == "explicit_launch_producer"
            {
                if let Some((normal, approach)) = takeoff_geometry {
                    let columns = matrix_columns(self.state.matrix);
                    let basis_ad0 = [-columns[1][0], -columns[1][1], -columns[1][2]];
                    let response = evaluate_takeoff_or_landing_contact(
                        self.state.velocity,
                        normal,
                        basis_ad0,
                        "takeoff",
                    )?;
                    takeoff_transfer = Some(TakeoffLandingTransfer {
                        role: "takeoff".to_string(),
                        selected: None,
                        reason: None,
                        velocity_before: Some(self.state.velocity),
                        velocity_after: Some(response.velocity_after),
                        hit_normal: normal,
                        basis_ad0,
                        approach_velocity: approach,
                        selector_authority: "autonomous_takeoff_collision_candidate".to_string(),
                        active_support_slot_count: None,
                        response: Some(response.clone()),
                    });
                    self.state.velocity = response.velocity_after;
                    self.state.takeoff_separation_pending = true;
                }
            }
            let takeoff_contact_bridge =
                self.state.takeoff_separation_pending && support.support_contact_count > 0;
            // Explicit desert_I08 launch retains +0x188 and all four wheel
            // producer slots throughout the authored takeoff/landing pair.
            // Ordinary support-loss AIR keeps the no-support producer.
            self.state.soft_flag_188 = takeoff_contact_bridge;
            let airborne = self.step_raw_airborne(
                keys,
                parameters,
                // The selected takeoff candidate has already executed this
                // substep's B2DC80 hard response above (handoff §6.6a).
                if takeoff_transfer.is_some() {
                    None
                } else {
                    world
                },
                if takeoff_contact_bridge {
                    Some(support.planes.as_slice())
                } else {
                    None
                },
                pending_ground_plane_provider,
            )?;
            executor = TopologyExecutorOutcome::Airborne(Box::new(airborne));
        } else if matches!(
            selected,
            MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
        ) {
            self.state.soft_flag_188 = support.support_contact_count > 0;
            if selected == MotionDomain::Land
                && topology_decision.transitioned
                && (matches!(
                    topology_decision.state_before.domain,
                    MotionDomain::AirEntry | MotionDomain::Air
                ) || (topology_decision.state_before.domain == MotionDomain::Contact
                    && matches!(
                        topology_decision.state_before.return_domain,
                        Some(MotionDomain::AirEntry) | Some(MotionDomain::Air)
                    )))
            {
                let Some(geometry) = &landing_geometry else {
                    // Python RuntimeError.
                    return Err("landing geometry was not selected".to_string());
                };
                let landing_normal = geometry.normal;
                let approach = if self.numeric_mode.is_float32() {
                    dot3_f32(self.state.velocity, landing_normal)
                } else {
                    dot3_sum(self.state.velocity, landing_normal)
                };
                if approach < 0.0 {
                    let columns = matrix_columns(self.state.matrix);
                    let basis_ad0 = [-columns[1][0], -columns[1][1], -columns[1][2]];
                    let response = evaluate_takeoff_or_landing_contact(
                        self.state.velocity,
                        landing_normal,
                        basis_ad0,
                        "landing",
                    )?;
                    landing_transfer = Some(TakeoffLandingTransfer {
                        role: "landing".to_string(),
                        selected: None,
                        reason: None,
                        velocity_before: Some(self.state.velocity),
                        velocity_after: Some(response.velocity_after),
                        hit_normal: landing_normal,
                        basis_ad0,
                        approach_velocity: approach,
                        selector_authority: geometry.selector_authority.to_string(),
                        active_support_slot_count: Some(geometry.active_support_slot_count),
                        response: Some(response.clone()),
                    });
                    self.state.velocity = response.velocity_after;
                } else {
                    landing_transfer = Some(TakeoffLandingTransfer {
                        role: "landing".to_string(),
                        selected: Some(false),
                        reason: Some("support_reacquired_while_not_approaching"),
                        velocity_before: None,
                        velocity_after: None,
                        hit_normal: landing_normal,
                        basis_ad0: [0.0, 0.0, 0.0],
                        approach_velocity: approach,
                        selector_authority: geometry.selector_authority.to_string(),
                        active_support_slot_count: None,
                        response: None,
                    });
                }
            }
            // The paired landing hard response above already consumes this
            // current-pose OBB candidate (handoff §6.6b).
            let suppress_world = landing_transfer.as_ref().is_some_and(|transfer| {
                transfer.selected.unwrap_or(true)
                    && transfer.selector_authority == "autonomous_landing_hard_contact_overlap"
            });
            let ground = self.step_raw_ground_correction(
                keys,
                parameters,
                if suppress_world { None } else { world },
                Some(&ground_plane_set_from_world_planes(&support.planes)),
                &GroundStepOptions {
                    pending_ground_plane_provider,
                    authored_trigger_model: authored_trigger_edge.as_ref().and_then(|edge| {
                        if edge.selected {
                            edge.model_args.clone()
                        } else {
                            None
                        }
                    }),
                    ..Default::default()
                },
            )?;
            executor = TopologyExecutorOutcome::Ground(Box::new(ground));
        } else {
            return Err(format!(
                "step_raw_topology has no executor for selected domain {}",
                selected.value()
            ));
        }
        let mut post_ground_rail_capture: Option<PostGroundRailCapture> = None;
        let mut rail_state1_ground_writer: Option<RailState1GroundWriter> = None;
        let mut autonomous_rail_selector: Option<BTreeMap<String, JsonValue>> = None;
        if matches!(
            selected,
            MotionDomain::Ground | MotionDomain::GroundSlip | MotionDomain::Land
        ) && !rail_candidate_active
        {
            // The initial highSpeed selector store shares the final ordinary
            // ground epoch (:6826-6884). Capture is committed after the
            // writer without executing a second producer.
            let candidate_after = self.inspect_autonomous_rail_capture(world)?;
            if candidate_after.contact_active {
                self.state.rail_capture_latch = Some(crate::state::RailCaptureLatch {
                    contact_active: true,
                    rail_state: candidate_after.rail_state,
                    selector: candidate_after.selector.clone().unwrap_or_default(),
                    authority: Some(candidate_after.authority.to_string()),
                });
            }
            if candidate_after.contact_active && rail_selector_due {
                self.state.rail_capture_latch = None;
                let selector_after = candidate_after.selector.clone().unwrap_or_default();
                let rail_state_after = candidate_after.rail_state.unwrap();
                let topology_after_ground = self.advance_motion_topology(&TopologyObservation {
                    rail_contact: true,
                    rail_state: rail_state_after,
                    rail_owner_id: selector_path_id(&selector_after),
                    ..TopologyObservation::default()
                })?;
                // Capture occurs before the shared epoch's ground body and
                // is observed at its sink, so one selector-delay quantum has
                // already elapsed at this published boundary.
                self.state.topology.domain_step_index = 1;
                self.state.topology.rail_state_elapsed_ms = parameters.substep_ms as f64;
                post_ground_rail_capture = Some(PostGroundRailCapture {
                    contact: candidate_after.clone(),
                    topology_decision: topology_after_ground,
                    selector_phase: "post_ground_writer_sink",
                    shared_ground_epoch_count: 1,
                    authority: "run119_epoch155433_ground_sink_initial_state1_store",
                });
                rail_state1_ground_writer = Some(RailState1GroundWriter {
                    selected: true,
                    state_after: rail_state_after,
                    support_contact_count: support.support_contact_count,
                    selector_phase: Some("post_ground_writer_sink"),
                    authority: "initial_state1_store_shares_completed_ground_epoch",
                });
                autonomous_rail_selector = Some(selector_after);
            }
        }
        let topology_decision = post_ground_rail_capture
            .as_ref()
            .map_or(topology_decision, |capture| {
                capture.topology_decision.clone()
            });
        Ok(RawTopologyOutcome {
            executor,
            support_observation: support,
            topology_decision,
            rail_captured: post_ground_rail_capture.is_some(),
            post_ground_rail_capture: post_ground_rail_capture.clone(),
            rail_state1_ground_writer,
            autonomous_rail_selector,
            takeoff_state_transfer: takeoff_transfer.clone(),
            landing_state_transfer: landing_transfer.clone(),
            authored_special_trigger: authored_trigger_edge,
            topology_after: self.state.topology.clone(),
            state_after: self.state.clone(),
            raw_topology_contract: RawTopologyContract {
                control_authority: "raw_keys".to_string(),
                geometry_authority: "world_query_ground_four_wheel_rays".to_string(),
                selector_authority: if post_ground_rail_capture.is_some() {
                    "post_ground_world_query_rail_contact".to_string()
                } else {
                    "engine_owned_support_contact_count".to_string()
                },
                scheduler_authority: "motion_topology_state_machine".to_string(),
                landing_transfer_authority: if landing_transfer
                    .as_ref()
                    .is_some_and(|transfer| transfer.selected.unwrap_or(true))
                {
                    "B2DC80_from_autonomous_contact_manifold".to_string()
                } else {
                    "not_selected".to_string()
                },
                takeoff_transfer_authority: if takeoff_transfer.is_some() {
                    "B2DC80_from_autonomous_takeoff_collision_candidate".to_string()
                } else {
                    "not_selected".to_string()
                },
            },
        })
    }
}

impl KartRiderVehicleEngine {
    /// Latch the P1-closed GoItemBanana run-over effect. Animated placement
    /// and the exact 2 m gate remain world-query responsibilities; all
    /// cross-substep state is committed to `VehicleState`.
    pub fn poll_autonomous_banana_effect(
        &mut self,
        world: Option<&dyn World>,
    ) -> Result<Option<crate::world::BananaTrigger>, String> {
        if self.state.banana_effect_remaining_ms > 0 || !world_caps(world).query_banana_trigger {
            return Ok(None);
        }
        let world = world.expect("capability probe implies a world");
        let Some(candidate) =
            world.query_banana_trigger(self.state.position, self.state.time_ms as i64)
        else {
            return Ok(None);
        };
        if candidate.object_id.is_empty() {
            return Err("banana trigger candidate requires object_id".to_string());
        }
        if !candidate.distance.is_finite()
            || !candidate.radius.is_finite()
            || candidate.radius != 2.0
            || candidate.distance > candidate.radius
        {
            return Err(
                "banana trigger candidate must carry a finite accepted distance and the adjudicated 2.0 m radius"
                    .to_string(),
            );
        }
        self.state.banana_effect_remaining_ms = BANANA_EFFECT_DURATION_MS;
        self.state.banana_effect_owner_id = Some(candidate.object_id.clone());
        Ok(Some(candidate))
    }

    fn finish_raw_motion_outcome(
        &mut self,
        mut outcome: RawMotionOutcome,
        domain_before: MotionDomain,
        time_ms_before: u64,
        parameters: &VehicleParameters,
        item_use_rising: bool,
        item_use_transition: Option<CurrentItem6InventoryUseResult>,
    ) -> Result<RawMotionOutcome, String> {
        let elapsed_ms = self.state.time_ms.saturating_sub(time_ms_before);
        if self.state.banana_effect_remaining_ms > 0 && elapsed_ms > 0 {
            self.state.banana_effect_remaining_ms = self
                .state
                .banana_effect_remaining_ms
                .saturating_sub(i64::try_from(elapsed_ms).unwrap_or(i64::MAX));
            if self.state.banana_effect_remaining_ms == 0 {
                self.state.banana_effect_owner_id = None;
            }
        }
        let reset_surface_entry = matches!(
            &outcome.kind,
            RawMotionKind::Topology(topology)
                if matches!(topology.executor, TopologyExecutorOutcome::ResetSurface(_))
        );
        let mut grant_transition = None;
        if domain_before == MotionDomain::Ground
            && self.state.time_ms > time_ms_before
            && !reset_surface_entry
        {
            if item_use_rising {
                grant_transition =
                    Some(BoostGaugeGrantTransition::SameSubstepItemUseUnadjudicated {
                        accum_b10_after: self.state.boost_gauge_committed_b10,
                        slots_after: self.state.booster_inventory_slots.clone(),
                    });
            } else {
                let grant = evaluate_current_family_item6_grant(
                    self.state.boost_gauge_committed_b10,
                    self.state.boost_gauge_pending_b2c,
                    parameters.boost_gauge_raw_760,
                    &self.state.booster_inventory_slots,
                )?;
                if grant.granted {
                    self.state.boost_gauge_committed_b10 = grant.accum_b10_after;
                    self.state.booster_inventory_slots = grant.slots_after.clone();
                }
                grant_transition = Some(BoostGaugeGrantTransition::Evaluated(grant));
            }
        }
        outcome.item_use_transition = item_use_transition;
        outcome.boost_gauge_grant_transition = grant_transition;
        outcome.banana_effect_remaining_ms = self.state.banana_effect_remaining_ms;
        outcome.state_after = self.state.clone();
        outcome.topology_after = self.state.topology.clone();
        let refresh_nested_state = outcome.boost_gauge_grant_transition.is_some();
        if let RawMotionKind::Topology(topology) = &mut outcome.kind {
            topology.state_after = self.state.clone();
            topology.topology_after = self.state.topology.clone();
            if let TopologyExecutorOutcome::Ground(ground) = &mut topology.executor {
                ground.step.state_after = self.state.clone();
            }
            if refresh_nested_state {
                if let TopologyExecutorOutcome::NetworkCorrection(preemption) =
                    &mut topology.executor
                {
                    if let AutonomousCorrectionEntry::Forward(entry) = &mut preemption.entry {
                        // Python's open result mapping uses one `state_after`
                        // key for both the leaf correction row and raw-motion
                        // publication. B2b's GND finish overwrites that key with
                        // the binary32-quantized engine state. Mirror that
                        // observable alias/collision while keeping the typed
                        // Rust engine state and unquantized preview distinct.
                        entry.step.step.state_after = RigidStateSnapshot {
                            position: self.state.position,
                            velocity: self.state.velocity,
                            matrix: self.state.matrix,
                            angular_velocity: self.state.angular_velocity,
                        };
                    }
                }
            }
        }
        Ok(outcome)
    }

    /// `_step_raw_motion_once` (kartrider_vehicle_engine.py:7038-7168):
    /// route one raw-control substep to the current topology owner.
    ///
    /// GND/GND-SLIP/CONTACT/AIR-ENTRY/AIR/LAND dispatch through
    /// `step_raw_topology`; RAIL dispatches through `step_autonomous_rail`;
    /// MOVING-CONTACT through `step_autonomous_moving_contact`; RESET
    /// through `step_autonomous_reset`; NET-CORRECTION through
    /// `step_autonomous_network_correction`. WATER keeps Python's plain
    /// `ValueError`. The preemption logic before dispatch (checkpoint poll,
    /// fall clamp, R rising edge, authoritative network correction) is fully
    /// ported.
    pub fn step_raw_motion_once(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_planes: Option<&[WorldGroundResult]>,
        pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'_>>,
    ) -> Result<RawMotionOutcome, String> {
        parameters.validate()?;
        let active_keys = validate_raw_control_keys(keys)?;
        let time_ms_before = self.state.time_ms;
        // Python step_raw_motion captures state_before before the dispatcher
        // runs, i.e. before the restart-latch write and the checkpoint poll.
        let state_before = self.state.clone();
        let restart_down = active_keys.contains("R");
        let restart_rising = restart_down && !self.state.manual_restart_key_latched;
        self.state.manual_restart_key_latched = restart_down;
        let item_use_down = active_keys.contains("ITEM_USE");
        let item_use_rising = item_use_down && !self.state.item_use_key_latched;
        self.state.item_use_key_latched = item_use_down;
        let mut item_use_transition = None;
        let domain = self.state.topology.domain;
        let banana_effect_transition = if !matches!(
            domain,
            MotionDomain::Reset | MotionDomain::NetworkCorrection
        ) {
            self.poll_autonomous_banana_effect(world)?
        } else {
            None
        };
        let checkpoint_target_events = if matches!(
            domain,
            MotionDomain::Reset | MotionDomain::NetworkCorrection
        ) {
            Vec::new()
        } else {
            self.poll_autonomous_checkpoint_target(world)?
        };
        if !matches!(
            domain,
            MotionDomain::Reset | MotionDomain::NetworkCorrection
        ) {
            if let Some(fall) = self.apply_fall_clamp_reset(-5.0, 0.0)? {
                let outcome = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: MotionDomain::Reset.value().to_string(),
                    raw_motion_dispatch_reason: "fall_clamp_reset_entry".to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition: banana_effect_transition.clone(),
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::FallClampReset(Box::new(fall)),
                };
                return self.finish_raw_motion_outcome(
                    outcome,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                );
            }
        }
        if restart_rising
            && !matches!(
                domain,
                MotionDomain::Reset | MotionDomain::NetworkCorrection
            )
        {
            let restart = self.apply_manual_restart_request()?;
            let outcome = RawMotionOutcome {
                state_before,
                state_after: self.state.clone(),
                topology_after: self.state.topology.clone(),
                raw_motion_executor: MotionDomain::Reset.value().to_string(),
                raw_motion_dispatch_reason: "raw_R_rising_manual_restart_entry".to_string(),
                checkpoint_target_events,
                item_use_transition: None,
                boost_gauge_grant_transition: None,
                banana_effect_transition: banana_effect_transition.clone(),
                banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                kind: RawMotionKind::ManualRestart(Box::new(restart)),
            };
            return self.finish_raw_motion_outcome(
                outcome,
                domain,
                time_ms_before,
                parameters,
                item_use_rising,
                item_use_transition,
            );
        }
        if matches!(
            domain,
            MotionDomain::Rail | MotionDomain::MovingContact | MotionDomain::Water
        ) {
            // kartrider_vehicle_engine.py:7093-7111: an accepted
            // authoritative packet preempts the exclusive owners.
            let network_candidate = self.inspect_autonomous_network_correction(world)?;
            if network_candidate.correction_active {
                let record = NetworkCorrectionRecord::from_inspection(&network_candidate)?;
                let entry =
                    self.begin_and_step_autonomous_network_correction(parameters, world, &record)?;
                let outcome = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: MotionDomain::NetworkCorrection.value().to_string(),
                    raw_motion_dispatch_reason: "autonomous_authoritative_correction_entry"
                        .to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition: banana_effect_transition.clone(),
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::NetworkCorrectionPreemption(Box::new(entry)),
                };
                return self.finish_raw_motion_outcome(
                    outcome,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                );
            }
        }
        if item_use_rising {
            if domain != MotionDomain::Ground {
                item_use_transition = Some(CurrentItem6InventoryUseResult {
                    consumed: false,
                    handler_accepted: false,
                    state_changed: false,
                    reason: "domain_not_current_item6_adjudicated".to_string(),
                    item_id: None,
                    consumed_slot_index: None,
                    target_state: None,
                    duration_ms: None,
                    state_after: None,
                    remaining_ms_after: None,
                    slots_after: self.state.booster_inventory_slots.clone(),
                    runtime_order: None,
                    authority: None,
                    domain: Some(domain.value().to_string()),
                });
            } else {
                let transition = evaluate_current_item6_inventory_use(
                    &self.state.booster_inventory_slots,
                    self.state.boost_state,
                    parameters.item6_boost_duration_ms,
                )?;
                if transition.state_changed {
                    self.state.booster_inventory_slots = transition.slots_after.clone();
                    self.state.boost_state = transition
                        .state_after
                        .expect("changed current item6 row has state_after");
                    self.state.boost_remaining_ms = transition
                        .remaining_ms_after
                        .expect("changed current item6 row has remaining_ms_after");
                }
                item_use_transition = Some(transition);
            }
        }
        if matches!(
            domain,
            MotionDomain::Ground
                | MotionDomain::GroundSlip
                | MotionDomain::Contact
                | MotionDomain::AirEntry
                | MotionDomain::Air
                | MotionDomain::Land
        ) {
            let outcome = self.step_raw_topology(
                keys,
                parameters,
                world,
                ground_planes,
                pending_ground_plane_provider,
            )?;
            // Python (:7143-7158): reset-surface entries report RESET; an
            // initial rail capture reports RAIL with the capture reason; an
            // initial moving-contact capture reports MOVING-CONTACT with the
            // capture reason.
            let (raw_motion_executor, raw_motion_dispatch_reason) =
                if let TopologyExecutorOutcome::ResetSurface(_) = &outcome.executor {
                    (
                        MotionDomain::Reset.value().to_string(),
                        "current_topology_owner".to_string(),
                    )
                } else if outcome.rail_captured {
                    (
                        MotionDomain::Rail.value().to_string(),
                        "autonomous_initial_rail_capture".to_string(),
                    )
                } else if matches!(
                    &outcome.executor,
                    TopologyExecutorOutcome::MovingContactCapture(_)
                ) {
                    (
                        MotionDomain::MovingContact.value().to_string(),
                        "autonomous_initial_moving_contact_capture".to_string(),
                    )
                } else if matches!(
                    &outcome.executor,
                    TopologyExecutorOutcome::NetworkCorrection(_)
                ) {
                    // Python :7159-7166 (`network_correction_acquired`).
                    (
                        MotionDomain::NetworkCorrection.value().to_string(),
                        "autonomous_authoritative_correction_entry".to_string(),
                    )
                } else {
                    (
                        domain.value().to_string(),
                        "current_topology_owner".to_string(),
                    )
                };
            let outcome = RawMotionOutcome {
                state_before,
                state_after: self.state.clone(),
                topology_after: self.state.topology.clone(),
                raw_motion_executor,
                raw_motion_dispatch_reason,
                checkpoint_target_events,
                item_use_transition: None,
                boost_gauge_grant_transition: None,
                banana_effect_transition: banana_effect_transition.clone(),
                banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                kind: RawMotionKind::Topology(Box::new(outcome)),
            };
            return self.finish_raw_motion_outcome(
                outcome,
                domain,
                time_ms_before,
                parameters,
                item_use_rising,
                item_use_transition,
            );
        }
        match domain {
            MotionDomain::Rail => {
                let outcome = self.step_autonomous_rail(keys, parameters, world)?;
                // Python: `result["raw_motion_executor"] = RESET.value if
                // result.get("reset_surface_entry") else domain.value`
                // (:7143-7146); the autonomous rail outcomes never carry
                // rail_captured / moving_contact / network-correction keys.
                let reset_surface_entry =
                    matches!(&outcome, AutonomousRailOutcome::ResetRelease(_));
                let result = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: if reset_surface_entry {
                        MotionDomain::Reset.value().to_string()
                    } else {
                        domain.value().to_string()
                    },
                    raw_motion_dispatch_reason: "current_topology_owner".to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition: banana_effect_transition.clone(),
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::Rail(Box::new(outcome)),
                };
                self.finish_raw_motion_outcome(
                    result,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                )
            }
            MotionDomain::MovingContact => {
                let outcome = self.step_autonomous_moving_contact(parameters, world)?;
                // Python (:7143-7146): `result["raw_motion_executor"] =
                // RESET.value if result.get("reset_surface_entry") else
                // domain.value`; only the reset-release path carries
                // `reset_surface_entry`.
                let reset_surface_entry =
                    matches!(&outcome, AutonomousMovingContactOutcome::ResetRelease(_));
                let result = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: if reset_surface_entry {
                        MotionDomain::Reset.value().to_string()
                    } else {
                        domain.value().to_string()
                    },
                    raw_motion_dispatch_reason: "current_topology_owner".to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition: banana_effect_transition.clone(),
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::MovingContact(Box::new(outcome)),
                };
                self.finish_raw_motion_outcome(
                    result,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                )
            }
            MotionDomain::Reset => {
                let outcome = self.step_autonomous_reset(parameters, world)?;
                let result = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: domain.value().to_string(),
                    raw_motion_dispatch_reason: "current_topology_owner".to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition: banana_effect_transition.clone(),
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::Reset(Box::new(outcome)),
                };
                self.finish_raw_motion_outcome(
                    result,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                )
            }
            MotionDomain::NetworkCorrection => {
                let outcome = self.step_autonomous_network_correction(parameters, world)?;
                let result = RawMotionOutcome {
                    state_before,
                    state_after: self.state.clone(),
                    topology_after: self.state.topology.clone(),
                    raw_motion_executor: domain.value().to_string(),
                    raw_motion_dispatch_reason: "current_topology_owner".to_string(),
                    checkpoint_target_events,
                    item_use_transition: None,
                    boost_gauge_grant_transition: None,
                    banana_effect_transition,
                    banana_effect_remaining_ms: self.state.banana_effect_remaining_ms,
                    kind: RawMotionKind::NetworkCorrection(Box::new(outcome)),
                };
                self.finish_raw_motion_outcome(
                    result,
                    domain,
                    time_ms_before,
                    parameters,
                    item_use_rising,
                    item_use_transition,
                )
            }
            MotionDomain::Water => Err(
                "distinct WATER owner has no proven executable model; validated \
                 abyss_I03 water-themed motion is GND-SLIP"
                    .to_string(),
            ),
            _ => Err(format!("no raw-motion executor for {}", domain.value())),
        }
    }

    /// `step_raw_motion` (kartrider_vehicle_engine.py:6919-6963): execute and
    /// retain one replayable raw-control prediction boundary.
    pub fn step_raw_motion(
        &mut self,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_planes: Option<&[WorldGroundResult]>,
        pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'_>>,
    ) -> Result<RawMotionOutcome, String> {
        let state_before = self.state.clone();
        let outcome = self.step_raw_motion_once(
            keys,
            parameters,
            world,
            ground_planes,
            pending_ground_plane_provider,
        )?;
        // Python :6933-6962: the resimulation engine never appends; a
        // rollback result already spliced the history; a correction
        // entry/step clears it.
        if !self.network_replay_active && !outcome.history_committed_by_rollback() {
            if outcome.network_correction_acquired()
                || state_before.topology.domain == MotionDomain::NetworkCorrection
                || self.state.topology.domain == MotionDomain::NetworkCorrection
            {
                self.prediction_history.clear();
                return Ok(outcome);
            }
            self.prediction_history.push_back(PredictionHistoryEntry {
                time_ms_before: state_before.time_ms,
                time_ms_after: self.state.time_ms,
                state_before,
                state_after: self.state.clone(),
                keys: keys.to_vec(),
                parameters: parameters.clone(),
                ground_planes: ground_planes.map(<[_]>::to_vec),
                has_pending_ground_plane_provider: pending_ground_plane_provider.is_some(),
            });
            while self.prediction_history.len() > PREDICTION_HISTORY_LIMIT {
                self.prediction_history.pop_front();
            }
        }
        Ok(outcome)
    }

    /// `step_raw_frame` (kartrider_vehicle_engine.py:6965-7036): advance one
    /// game frame through the executable's physics scheduler. The engine
    /// owns delta calculation, the 500 ms stall cap, and the ordered 1/2 ms
    /// substep schedule; a single initial tick establishes the clock without
    /// advancing physics.
    pub fn step_raw_frame(
        &mut self,
        frame_tick_ms: u32,
        keys: &[String],
        parameters: &VehicleParameters,
        world: Option<&dyn World>,
        ground_planes: Option<&[WorldGroundResult]>,
        pending_ground_plane_provider: Option<PendingGroundPlaneProviderRef<'_>>,
    ) -> Result<RawFrameOutcome, String> {
        parameters.validate()?;
        validate_raw_control_keys(keys)?;
        let Some(previous_tick_ms) = self.state.scheduler_previous_tick_ms else {
            let schedule = evaluate_physics_frame_scheduler(frame_tick_ms, frame_tick_ms);
            self.state.scheduler_previous_tick_ms = Some(frame_tick_ms);
            self.state.validate()?;
            return Ok(RawFrameOutcome {
                frame_scheduler: schedule,
                physics_substep_count: 0,
                physics_substeps: Vec::new(),
                clock_initialized: true,
                state_after: self.state.clone(),
            });
        };
        let schedule = evaluate_physics_frame_scheduler(previous_tick_ms, frame_tick_ms);
        let substep_count = schedule.substeps_ms.len();
        // Python min(1, len(substeps_ms) - 1); with zero substeps the loop
        // body never runs, so the saturating form is equivalent.
        let selector_substep_index = 1usize.min(substep_count.saturating_sub(1));
        let mut substep_results = Vec::with_capacity(substep_count);
        for (substep_index, substep_ms) in schedule.substeps_ms.iter().enumerate() {
            let substep_parameters = VehicleParameters {
                substep_ms: i64::from(*substep_ms),
                topology_selector_delta_ms: Some(if substep_index == selector_substep_index {
                    f64::from(schedule.capped_frame_delta_ms)
                } else {
                    0.0
                }),
                initial_rail_capture_commit: Some(substep_index == substep_count - 1),
                ..parameters.clone()
            };
            substep_results.push(self.step_raw_motion(
                keys,
                &substep_parameters,
                world,
                ground_planes,
                pending_ground_plane_provider,
            )?);
        }
        self.state.scheduler_previous_tick_ms = Some(frame_tick_ms);
        self.state.validate()?;
        Ok(RawFrameOutcome {
            frame_scheduler: schedule,
            physics_substep_count: substep_results.len(),
            physics_substeps: substep_results,
            clock_initialized: false,
            state_after: self.state.clone(),
        })
    }
}

/// CPython `round(x)` (no ndigits): round-half-to-even returning an int.
/// `f64::round_ties_even` matches except that it returns `-0.0` for
/// `round(-0.5)`, while Python's int result widens to `+0.0`; normalize the
/// zero sign.
fn round_py(value: f64) -> f64 {
    let rounded = value.round_ties_even();
    if rounded == 0.0 {
        0.0
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|name| name.to_string()).collect()
    }

    fn slip_plane() -> GroundPlane {
        GroundPlane {
            normal: [0.0, 0.0, 1.0],
            point: [0.0, 0.0, 0.0],
            road_surface: Some("slip".to_string()),
            ..Default::default()
        }
    }

    #[test]
    fn ground_projection_adc_signed_zero_matches_cpython_sum() {
        // Python: _dot3(state.velocity, forward) with velocity [0,0,0] and
        // forward = -[m1,m4,m7] = [-0.0, -1.0, -0.0] (identity matrix,
        // forward_axis_column=1, forward_axis_sign=-1.0) has products
        // [-0.0, -0.0, -0.0]. CPython sum() starts at int 0 and yields +0.0;
        // a naive fold would keep -0.0 (this broke
        // drive_target.exponent_input sign parity in float64 mode).
        let projection = derive_autonomous_ground_projection(
            &VehicleState::default(),
            &VehicleParameters::default(),
            NumericMode::Float64,
        )
        .unwrap();
        assert_eq!(projection.field_adc.to_bits(), 0.0_f64.to_bits());
        assert!(projection.field_adc.is_sign_positive());
    }

    #[test]
    fn raw_control_key_validation_matches_python() {
        assert_eq!(
            validate_raw_control_keys(&keys(&["UP", "DOWN"])).unwrap_err(),
            "simultaneous UP+DOWN depends on dispatcher event order"
        );
        assert_eq!(
            validate_raw_control_keys(&keys(&["left", "RIGHT"])).unwrap_err(),
            "simultaneous LEFT+RIGHT depends on dispatcher event order"
        );
        let active = validate_raw_control_keys(&keys(&["up", "Left"])).unwrap();
        assert!(active.contains("UP") && active.contains("LEFT"));
    }

    #[test]
    fn surface_selector_tokens_and_precedence() {
        let decoded = parse_df_surface_token(" DF 2.00 ").unwrap();
        assert_eq!(
            decoded,
            DfSurfaceToken {
                variant: "DF",
                payload: 2.0,
                token: "DF 2.00".to_string(),
            }
        );
        for malformed in [
            "DF",
            "DF2.",
            "DF.2",
            "DFnan",
            "DF2.0junk",
            "DF٢.٠",
            "DF１２.０",
            "DF𝟚.𝟘",
        ] {
            assert_eq!(
                parse_df_surface_token(malformed).unwrap_err(),
                format!("unsupported DF authored token: {malformed}")
            );
        }

        // Fallback when no contacts are supplied.
        let selection = select_surface_mode_from_contacts(None, 2).unwrap();
        assert_eq!(selection.surface_mode, 2);
        assert_eq!(selection.authority, "vehicle_parameters");
        assert!(selection.selection_rule.is_none());

        // Slip precedence over ordinary contacts.
        let mixed = GroundPlaneSet::Sequence(vec![
            GroundPlane::default(),
            GroundPlane::default(),
            slip_plane(),
            slip_plane(),
        ]);
        let selection = select_surface_mode_from_contacts(Some(&mixed), 0).unwrap();
        assert_eq!(selection.surface_mode, 2);
        assert_eq!(
            selection.selection_rule,
            Some("slip_precedes_ordinary_contacts")
        );

        // Reset precedes all surface dynamics.
        let with_reset = GroundPlaneSet::Sequence(vec![
            slip_plane(),
            GroundPlane {
                road_surface: Some("점프".to_string()),
                ..Default::default()
            },
            GroundPlane {
                road_surface: Some("리셋".to_string()),
                ..Default::default()
            },
        ]);
        let selection = select_surface_mode_from_contacts(Some(&with_reset), 0).unwrap();
        assert_eq!(selection.surface_mode, 0);
        assert_eq!(
            selection.selected_dynamics_branch,
            Some(Some("reset".to_string()))
        );
        assert_eq!(
            selection.selection_rule,
            Some("reset_precedes_all_surface_dynamics")
        );

        // Jump/slip coexistence is unproven and rejected.
        let jump_slip = GroundPlaneSet::Sequence(vec![
            GroundPlane {
                road_surface: Some("jump".to_string()),
                ..Default::default()
            },
            slip_plane(),
        ]);
        assert!(select_surface_mode_from_contacts(Some(&jump_slip), 0)
            .unwrap_err()
            .contains("precedence is unproven"));

        // JM authored payload stays mode 0 with an explicit branch.
        let jm = GroundPlaneSet::Single(GroundPlane {
            road_surface: Some("JM2.2/0.2".to_string()),
            ..Default::default()
        });
        let selection = select_surface_mode_from_contacts(Some(&jm), 0).unwrap();
        assert_eq!(selection.surface_mode, 0);
        assert_eq!(
            selection.selected_dynamics_branch,
            Some(Some("authored_jm_trigger".to_string()))
        );

        // Game-adjudicated DF2.0 stays ordinary mode 0 while publishing the
        // fully decoded numeric payload.
        let df = GroundPlaneSet::Sequence(vec![
            GroundPlane::default(),
            GroundPlane {
                road_surface: Some("DF2.0".to_string()),
                ..Default::default()
            },
            GroundPlane {
                road_surface: Some("df 2.00".to_string()),
                ..Default::default()
            },
            GroundPlane::default(),
        ]);
        let selection = select_surface_mode_from_contacts(Some(&df), 0).unwrap();
        assert_eq!(selection.surface_mode, 0);
        assert_eq!(
            selection.df_payloads,
            Some(vec![None, Some(2.0), Some(2.0), None])
        );
        assert_eq!(selection.selected_dynamics_branch, Some(None));

        let unverified = GroundPlaneSet::Single(GroundPlane {
            road_surface: Some("DF3.0".to_string()),
            ..Default::default()
        });
        assert_eq!(
            select_surface_mode_from_contacts(Some(&unverified), 0).unwrap_err(),
            "unsupported DF authored payload without game adjudication: 3.0"
        );
        for (token, expected_payload) in [("DF10000000000000000", "1e+16"), ("DF0.00001", "1e-05")]
        {
            let plane = GroundPlaneSet::Single(GroundPlane {
                road_surface: Some(token.to_string()),
                ..Default::default()
            });
            assert_eq!(
                select_surface_mode_from_contacts(Some(&plane), 0).unwrap_err(),
                format!(
                    "unsupported DF authored payload without game adjudication: \
                     {expected_payload}"
                )
            );
        }

        // Unknown tokens are rejected with the Python message.
        let unknown = GroundPlaneSet::Single(GroundPlane {
            road_surface: Some("dirt".to_string()),
            ..Default::default()
        });
        assert_eq!(
            select_surface_mode_from_contacts(Some(&unknown), 0).unwrap_err(),
            "unsupported autonomous road_surface token: dirt"
        );
    }

    #[test]
    fn surface_selector_latch_refreshes_at_60hz_epoch() {
        // Mirror of test_surface_selector_is_latched_until_next_outer_tick.
        let state = VehicleState {
            surface_mode_latched: Some(0),
            surface_selector_elapsed_ms: 4.0,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let parameters = VehicleParameters::default();
        let plane = GroundPlaneSet::Single(slip_plane());
        for _ in 0..6 {
            let outcome = engine
                .step_raw_ground_correction(
                    &keys(&[]),
                    &parameters,
                    None,
                    Some(&plane),
                    &GroundStepOptions::default(),
                )
                .unwrap();
            assert_eq!(outcome.surface_selection.surface_mode, 0);
            assert!(!outcome.surface_selection.selector_refreshed);
        }
        let outcome = engine
            .step_raw_ground_correction(
                &keys(&[]),
                &parameters,
                None,
                Some(&plane),
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(outcome.surface_selection.surface_mode, 2);
        assert!(outcome.surface_selection.selector_refreshed);
        assert_eq!(engine.state.surface_mode_latched, Some(2));
    }

    #[test]
    fn producer_only_call_has_zero_state_writes() {
        let state = VehicleState {
            velocity: [0.0, 10.0, 0.0],
            ..Default::default()
        };
        let engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let plane = GroundPlaneSet::Single(slip_plane());
        let output = engine
            .produce_ground_accumulators(
                &keys(&["UP", "LEFT", "CTRL"]),
                &VehicleParameters::default(),
                Some(&plane),
                None,
                false,
            )
            .unwrap();
        assert_eq!(engine.state, state);
        assert_eq!(output.producer_contract.state_mutation, "none");
        assert_eq!(output.producer_contract.integration_stages, "not_scheduled");
        assert_eq!(
            output.producer_contract.authority,
            "pure_producer_frame_from_explicit_vehicle_state"
        );
        // Producer evaluator: integration stages not scheduled.
        assert!(output.evaluator.linear.is_none());
        assert!(output.evaluator.position.is_none());
        assert!(output.evaluator.position_90_final.is_none());
        assert!(output.evaluator.accumulator_614_after_damping.is_some());
    }

    #[test]
    fn ground_probe_rejects_plane_intersection_behind_ray_origin() {
        let state = VehicleState::default();
        let engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let plane = GroundPlaneSet::Single(GroundPlane {
            normal: [0.0, 0.0, 1.0],
            point: [0.0, 0.0, 2.0],
            triangle: Some([[-10.0, -10.0, 2.0], [10.0, -10.0, 2.0], [0.0, 10.0, 2.0]]),
            ..Default::default()
        });
        let output = engine
            .produce_ground_accumulators(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&plane),
                None,
                false,
            )
            .unwrap();
        assert!(output
            .contact_probe_debug
            .iter()
            .all(|probe| probe.ray_t.is_some_and(|ray_t| ray_t < 0.0)));
        assert_eq!(output.contact_compression_after, [0.0; 4]);
    }

    #[test]
    fn ground_probe_preserves_evidence_backed_rail_endpoint_handoff() {
        let state = VehicleState {
            topology: TopologyState {
                previous_domain: Some(MotionDomain::Rail),
                ..TopologyState::default()
            },
            ..VehicleState::default()
        };
        let engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let plane = GroundPlaneSet::Single(GroundPlane {
            normal: [0.0, 0.0, 1.0],
            point: [0.0, 0.0, 2.0],
            triangle: Some([[-10.0, -10.0, 2.0], [10.0, -10.0, 2.0], [0.0, 10.0, 2.0]]),
            ..Default::default()
        });
        let output = engine
            .produce_ground_accumulators(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&plane),
                None,
                false,
            )
            .unwrap();
        assert_eq!(output.contact_compression_after, [1.0; 4]);
    }

    #[test]
    fn drift_input_event_drives_selector_before_same_substep_ground_producer() {
        let state = VehicleState {
            velocity: [0.0, -10.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let parameters = VehicleParameters {
            drift_window_duration_734: 0.002,
            ..Default::default()
        };
        let pressed = engine
            .step_raw_ground_correction(
                &keys(&["LEFT", "CTRL"]),
                &parameters,
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let transition = pressed
            .drift_selector_transition
            .expect("CTRL down edge must invoke the selector helper");
        assert!(transition.applied);
        assert_eq!(transition.reason, "enable");
        assert!(matches!(
            pressed.step.evaluator.drift_window,
            Some(DriftWindowBranchResult::Selected(_))
        ));
        assert!(engine.state.selector_b08);
        assert_eq!(engine.state.selector_ff8, 1);
        assert_eq!(engine.state.drift_timer_b0c, 0.002);
        assert!(engine.state.drift_key_latched);
        assert_eq!(engine.state.drive_raw_aa0, 1.0);

        // Preserve a >5 projection so the next step exercises the existing
        // B08 window rather than the low-projection clear branch.
        engine.state.velocity = [0.0, -10.0, 0.0];
        let released = engine
            .step_raw_ground_correction(
                &keys(&[]),
                &parameters,
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let transition = released
            .drift_selector_transition
            .expect("CTRL up edge must invoke the selector helper");
        assert!(transition.applied);
        assert_eq!(transition.reason, "disable");
        assert!(!engine.state.drift_key_latched);
        assert_eq!(engine.state.drive_raw_aa0, 0.0);
        assert!(!engine.state.selector_b08);
        assert_eq!(engine.state.drift_timer_b0c, 0.0);
        assert!(engine.state.drift_b1c_charge_settlement);
    }

    #[test]
    fn ctrl_without_steering_waits_for_steering_event() {
        let state = VehicleState {
            velocity: [0.0, -10.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let ctrl_only = engine
            .step_raw_ground_correction(
                &keys(&["CTRL"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert!(ctrl_only.drift_selector_transition.is_none());
        assert!(ctrl_only.drift_selector_transitions.is_empty());
        assert!(engine.state.drift_key_latched);
        assert_eq!(engine.state.drive_raw_aa0, 0.0);

        let steered = engine
            .step_raw_ground_correction(
                &keys(&["CTRL", "LEFT"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(steered.drift_selector_transitions.len(), 1);
        assert_eq!(steered.drift_selector_transitions[0].reason, "enable");
        assert_eq!(engine.state.drive_raw_aa0, 1.0);
    }

    #[test]
    fn held_ctrl_direction_switch_dispatches_disable_then_enable() {
        let state = VehicleState {
            velocity: [0.0, -10.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        engine
            .step_raw_ground_correction(
                &keys(&["CTRL", "LEFT"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        engine.state.field_af8 = 0.25;
        engine.state.selector_b08 = false;

        let switched = engine
            .step_raw_ground_correction(
                &keys(&["CTRL", "RIGHT"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(
            switched
                .drift_selector_transitions
                .iter()
                .map(|row| row.reason)
                .collect::<Vec<_>>(),
            vec!["disable", "enable"]
        );
        assert_eq!(engine.state.selector_ff8, 2);
        assert_eq!(engine.state.selector_fec, 1);
        assert!(!engine.state.selector_b08);
        assert_eq!(engine.state.drive_raw_aa0, -1.0);
    }

    #[test]
    fn af8_positive_press_is_not_repolled_while_input_is_stable() {
        let state = VehicleState {
            velocity: [0.0, -10.0, 0.0],
            field_af8: 0.003,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let pressed = engine
            .step_raw_ground_correction(
                &keys(&["CTRL", "LEFT"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(pressed.drift_selector_transitions.len(), 1);
        assert_eq!(pressed.drift_selector_transitions[0].reason, "enable");
        assert!(!engine.state.selector_b08);

        for _ in 0..3 {
            let held = engine
                .step_raw_ground_correction(
                    &keys(&["CTRL", "LEFT"]),
                    &VehicleParameters::default(),
                    None,
                    None,
                    &GroundStepOptions::default(),
                )
                .unwrap();
            assert!(held.drift_selector_transitions.is_empty());
        }
        assert_eq!(engine.state.field_af8, 0.0);
        assert!(!engine.state.selector_b08);
    }

    #[test]
    fn ctrl_right_sets_ff8_and_surface1_release_seeds_recovery_timer() {
        let state = VehicleState {
            velocity: [0.0, 10.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float64).unwrap();
        let parameters = VehicleParameters {
            surface_mode: 1,
            ..Default::default()
        };
        let pressed = engine
            .step_raw_ground_correction(
                &keys(&["RIGHT", "CTRL"]),
                &parameters,
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(
            pressed
                .drift_selector_transition
                .expect("CTRL down transition")
                .after
                .selector_ff8,
            2
        );
        assert!(!engine.state.selector_b5c);
        engine.state.velocity = [0.0, 10.0, 0.0];
        let released = engine
            .step_raw_ground_correction(
                &keys(&[]),
                &parameters,
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let transition = released
            .drift_selector_transition
            .expect("CTRL up transition");
        assert_eq!(transition.after.recovery_timer, 100.0 / 1440.0);
        assert!(engine.state.recovery_timer > 0.0);
        assert!(!engine.state.soft_selector_af4);
    }

    /// Flat-ground world with a no-hit OBB sweep for the `step_world`
    /// no-dispatch path.
    struct FlatSweepWorld;
    impl World for FlatSweepWorld {
        fn capabilities(&self) -> crate::world::WorldCapabilities {
            crate::world::WorldCapabilities {
                sweep_obb_rotating_all: true,
                ..Default::default()
            }
        }
        fn query_ground(
            &self,
            _position: [f64; 3],
            _up: [f64; 3],
            _min_dot: f64,
        ) -> WorldGroundResult {
            WorldGroundResult {
                normal: [0.0, 0.0, 1.0],
                point: [0.0, 0.0, 0.0],
                ..Default::default()
            }
        }
        fn sweep_obb_rotating_all(
            &self,
            _start: [f64; 3],
            _end: [f64; 3],
            _start_matrix: [f64; 9],
            _end_matrix: [f64; 9],
            _half_extents: [f64; 3],
        ) -> Option<Vec<WorldObbOverlap>> {
            Some(Vec::new())
        }
    }

    fn current_item6_slots(first_item_id: i64, second_item_id: i64) -> Vec<InventorySlot> {
        [first_item_id, second_item_id]
            .into_iter()
            .map(|item_id| InventorySlot {
                item_id,
                quantity: i64::from(item_id != 0xffff),
            })
            .collect()
    }

    #[test]
    fn current_item_use_edge_consumes_slot1_and_enters_ground_chain() {
        let state = VehicleState {
            booster_inventory_slots: current_item6_slots(6, 6),
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let parameters = VehicleParameters {
            substep_ms: 2,
            ..Default::default()
        };
        let world = FlatSweepWorld;

        let activated = engine
            .step_raw_motion(
                &keys(&["UP", "ITEM_USE"]),
                &parameters,
                Some(&world),
                None,
                None,
            )
            .unwrap();
        let transition = activated
            .item_use_transition
            .expect("ITEM_USE rising edge is reported");
        assert!(transition.state_changed);
        assert_eq!(transition.consumed_slot_index, Some(1));
        assert_eq!(
            engine.state.booster_inventory_slots,
            current_item6_slots(6, 0xffff)
        );
        assert_eq!(engine.state.boost_state, 3);
        assert_eq!(engine.state.boost_remaining_ms, 2898);
        assert!(engine.state.item_use_key_latched);
        assert_eq!(
            engine.prediction_history[0]
                .state_before
                .booster_inventory_slots,
            current_item6_slots(6, 6)
        );

        let held = engine
            .step_raw_motion(
                &keys(&["UP", "ITEM_USE"]),
                &parameters,
                Some(&world),
                None,
                None,
            )
            .unwrap();
        assert!(held.item_use_transition.is_none());
        assert_eq!(engine.state.boost_remaining_ms, 2896);
        assert_eq!(
            engine.state.booster_inventory_slots,
            current_item6_slots(6, 0xffff)
        );
    }

    #[test]
    fn current_gauge_settlement_grants_item6_in_same_ground_substep() {
        let state = VehicleState {
            booster_inventory_slots: current_item6_slots(6, 0xffff),
            drift_b1c_charge_settlement: true,
            boost_gauge_committed_b10: 3800.0,
            boost_gauge_progress_b20: 0.4,
            boost_gauge_pending_b2c: 60.0,
            boost_gauge_history_b38: 12.0,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = FlatSweepWorld;

        let result = engine
            .step_raw_motion(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&world),
                None,
                None,
            )
            .unwrap();
        let BoostGaugeGrantTransition::Evaluated(grant) = result
            .boost_gauge_grant_transition
            .expect("ground substep evaluates the current-family grant")
        else {
            panic!("ordinary gauge settlement must use the evaluated grant row");
        };
        assert!(grant.granted);
        assert_eq!(grant.selected_slot_index, Some(1));
        assert_eq!(
            engine.state.booster_inventory_slots,
            current_item6_slots(6, 6)
        );
        assert_eq!(engine.state.boost_gauge_committed_b10, 0.0);
        assert_eq!(engine.state.boost_gauge_pending_b2c, 0.0);
        assert_eq!(engine.state.boost_gauge_progress_b20, 0.0);
        assert_eq!(engine.state.boost_gauge_history_b38, 60.0);
        assert!(!engine.state.drift_b1c_charge_settlement);
        assert_eq!(
            engine.prediction_history[0]
                .state_after
                .booster_inventory_slots,
            engine.state.booster_inventory_slots
        );
    }

    #[test]
    fn current_gauge_primary_producer_runs_before_ground_body() {
        let state = VehicleState {
            velocity: [0.0, -161.0, 0.0],
            soft_flag_188: true,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let outcome = engine
            .step_raw_ground_correction(
                &keys(&[]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let producer = outcome
            .step
            .boost_gauge_producer_transition
            .expect("raw ground step applies gauge producers");
        assert!(!producer.pending_b2c.applied);
        assert!(producer.primary_b10_b20.applied);
        assert_eq!(producer.state_after_producers.b10, 0.7000000476837158);
        assert_eq!(producer.state_after_producers.b20, 0.0020000000949949026);
        assert_eq!(engine.state.boost_gauge_committed_b10, 0.7000000476837158);
        assert_eq!(engine.state.boost_gauge_progress_b20, 0.0020000000949949026);
    }

    #[test]
    fn current_gauge_pending_producer_precedes_same_step_settlement() {
        let state = VehicleState {
            velocity: [10.0, -20.0, 0.0],
            soft_flag_188: true,
            drift_b1c_charge_settlement: true,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let outcome = engine
            .step_raw_ground_correction(
                &keys(&[]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let producer = outcome
            .step
            .boost_gauge_producer_transition
            .expect("raw ground step applies gauge producers");
        assert!(producer.pending_b2c.applied);
        assert!(!producer.primary_b10_b20.applied);
        assert_eq!(producer.pending_b2c.state_after.b2c, 0.6000000238418579);
        assert_eq!(engine.state.boost_gauge_committed_b10, 0.6000000238418579);
        assert_eq!(engine.state.boost_gauge_history_b38, 0.6000000238418579);
        assert_eq!(engine.state.boost_gauge_pending_b2c, 0.0);
    }

    #[test]
    fn force_producer_only_does_not_write_boost_gauge() {
        let state = VehicleState {
            velocity: [0.0, -161.0, 0.0],
            soft_flag_188: true,
            boost_gauge_committed_b10: 10.0,
            boost_gauge_progress_b20: 0.5,
            ..Default::default()
        };
        let engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let before = engine.state.clone();
        let producer = engine
            .produce_ground_accumulators(
                &keys(&[]),
                &VehicleParameters::default(),
                None,
                None,
                true,
            )
            .unwrap();
        assert_eq!(engine.state, before);
        assert!(producer.producer_contract.state_mutation == "none");
    }

    #[test]
    fn current_item_use_and_grant_same_substep_stays_fail_safe() {
        let state = VehicleState {
            booster_inventory_slots: current_item6_slots(6, 0xffff),
            boost_gauge_committed_b10: 3860.0,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let parameters = VehicleParameters::default();
        let world = FlatSweepWorld;

        let overlap = engine
            .step_raw_motion(
                &keys(&["UP", "ITEM_USE"]),
                &parameters,
                Some(&world),
                None,
                None,
            )
            .unwrap();
        assert!(overlap
            .item_use_transition
            .as_ref()
            .is_some_and(|transition| transition.consumed));
        assert!(matches!(
            overlap.boost_gauge_grant_transition,
            Some(BoostGaugeGrantTransition::SameSubstepItemUseUnadjudicated { .. })
        ));
        assert_eq!(engine.state.boost_gauge_committed_b10, 3860.0);
        assert_eq!(
            engine.state.booster_inventory_slots,
            current_item6_slots(0xffff, 0xffff)
        );

        let granted = engine
            .step_raw_motion(&keys(&["UP"]), &parameters, Some(&world), None, None)
            .unwrap();
        assert!(matches!(
            granted.boost_gauge_grant_transition,
            Some(BoostGaugeGrantTransition::Evaluated(
                CurrentFamilyItem6Grant { granted: true, .. }
            ))
        ));
        assert_eq!(
            engine.state.booster_inventory_slots,
            current_item6_slots(6, 0xffff)
        );
    }

    #[test]
    fn world_dispatch_no_hit_sweep_behaves_like_plain_step() {
        // M1 step_world: with an empty sweep the outcome is the plain `step`
        // result plus the world_sweep payload and a CONTACT release probe.
        let state = VehicleState::default();
        let mut with_world = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let mut without_world = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = FlatSweepWorld;
        let outcome = with_world
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                Some(&world),
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let reference = without_world
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                None,
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        assert_eq!(with_world.state, without_world.state);
        let sweep = outcome.world_sweep.expect("world path sweeps");
        assert_eq!(sweep.shape, "obb_rotating");
        assert_eq!(sweep.dispatch_reason, "no_hit");
        assert!(!sweep.response_dispatched);
        assert!(sweep.topology_transition.is_none());
        assert!(reference.world_sweep.is_none());
        // The world path tags the collision authority
        // (kartrider_vehicle_engine.py:5849).
        assert_eq!(
            outcome.raw_control_pose_contract.get("collision_authority"),
            Some(&JsonValue::String("autonomous_world_dispatch".to_string()))
        );
    }

    struct BigGearSweepWorld;
    impl World for BigGearSweepWorld {
        fn capabilities(&self) -> crate::world::WorldCapabilities {
            crate::world::WorldCapabilities {
                sweep_obb_rotating_all: true,
                query_event_obstacle_contacts: true,
                ..Default::default()
            }
        }

        fn query_ground(
            &self,
            _position: [f64; 3],
            _up: [f64; 3],
            _min_dot: f64,
        ) -> WorldGroundResult {
            WorldGroundResult {
                normal: [0.0, 0.0, 1.0],
                point: [0.0, 0.0, 0.0],
                ..Default::default()
            }
        }

        fn sweep_obb_rotating_all(
            &self,
            _start: [f64; 3],
            _end: [f64; 3],
            _start_matrix: [f64; 9],
            _end_matrix: [f64; 9],
            _half_extents: [f64; 3],
        ) -> Option<Vec<WorldObbOverlap>> {
            Some(Vec::new())
        }

        fn query_event_obstacle_contacts(
            &self,
            _center: [f64; 3],
            _matrix: [f64; 9],
            _half_extents: [f64; 3],
            time_ms: i64,
            delta_ms: i64,
        ) -> Option<Vec<EventObstacleContact>> {
            assert_eq!(time_ms, 0);
            assert_eq!(delta_ms, 2);
            Some(vec![EventObstacleContact {
                event_index: 1743,
                object_id: "event:1743".to_string(),
                node: "Object12_Geom".to_string(),
                tri_index: 2,
                loop_index: 14,
                centroid: [162.638_184, 844.211_121, 68.182_640_1],
                normal: [-0.904_148_161, 0.427_219_033, 0.000_005_808_697_1],
                motion: [-12.069_702_1, 4.006_778_72, 0.0],
                obstacle_position: [158.490_265, 831.380_615, 68.262_833],
            }])
        }
    }

    #[test]
    fn p1_big_gear_world_query_dispatches_terminal_contact_writer() {
        let state = VehicleState {
            velocity: [34.823_501_6, -22.845_993, 0.015_695_085_7],
            ..Default::default()
        };
        let mut baseline = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let mut event = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let baseline_outcome = baseline
            .step_raw_ground_correction(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&FlatSweepWorld),
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let event_outcome = event
            .step_raw_ground_correction(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&BigGearSweepWorld),
                None,
                &GroundStepOptions::default(),
            )
            .unwrap();
        let expected = apply_event_obstacle_contact_source(
            baseline_outcome.step.state_after.velocity,
            [-0.904_148_161, 0.427_219_033, 0.000_005_808_697_1],
            [-12.069_702_1, 4.006_778_72, 0.0],
        )
        .unwrap();
        assert_eq!(event.state.velocity, expected.velocity_after);
        let sweep = event_outcome.world_sweep.expect("world path sweeps");
        assert_eq!(
            sweep.dispatch_reason,
            "approaching_event_dynamic_obstacle_contact"
        );
        assert_eq!(
            sweep.dispatch_authority,
            Some("runtime_GoItemObstacle_kind90_kind91")
        );
        let event_info = sweep
            .event_obstacle_contact
            .expect("event candidates are published");
        assert_eq!(event_info.candidates.len(), 1);
        assert_eq!(event_info.responses.len(), 1);
        assert_eq!(event.state.topology.domain, MotionDomain::Contact);
    }

    #[test]
    fn banana_trigger_latches_b9c_and_suppresses_ground_producers() {
        struct BananaWorld;
        impl World for BananaWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    query_banana_trigger: true,
                    ..Default::default()
                }
            }

            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult::default()
            }

            fn query_banana_trigger(
                &self,
                position: [f64; 3],
                _time_ms: i64,
            ) -> Option<crate::world::BananaTrigger> {
                Some(crate::world::BananaTrigger {
                    object_id: "banana:0xF0000039".to_string(),
                    event_index: 1762,
                    position,
                    distance: 1.5,
                    radius: 2.0,
                    authority: "test".to_string(),
                })
            }
        }

        let mut engine =
            KartRiderVehicleEngine::new(&VehicleState::default(), NumericMode::Float32).unwrap();
        let hit = engine
            .poll_autonomous_banana_effect(Some(&BananaWorld))
            .unwrap()
            .expect("accepted banana");
        assert_eq!(hit.event_index, 1762);
        assert_eq!(engine.state.banana_effect_remaining_ms, 2000);
        assert_eq!(
            engine.state.banana_effect_owner_id.as_deref(),
            Some("banana:0xF0000039")
        );

        let produced = engine
            .produce_ground_accumulators(
                &["UP".to_string(), "LEFT".to_string()],
                &VehicleParameters {
                    substep_ms: 2,
                    ..Default::default()
                },
                Some(&GroundPlaneSet::Single(slip_plane())),
                Some(true),
                false,
            )
            .unwrap();
        assert!(matches!(
            produced.evaluator.recovery_schedule,
            RecoveryScheduleOutcome::Suppressed {
                reason: "disabled_b9c"
            }
        ));
        assert!(produced.evaluator.ground_force.is_none());
        assert_eq!(produced.evaluator.ground_drive_force_status, "disabled_b9c");
        assert!(produced.evaluator.correction.unwrap().disabled);
    }

    #[test]
    fn world_dispatch_selected_candidate_dispatches_soft_contact() {
        struct HitWorld;
        impl World for HitWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    sweep_obb_rotating_all: true,
                    ..Default::default()
                }
            }
            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult::default()
            }
            fn sweep_obb_rotating_all(
                &self,
                _start: [f64; 3],
                _end: [f64; 3],
                _start_matrix: [f64; 9],
                _end_matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(vec![WorldObbOverlap {
                    hit: true,
                    normal: [0.0, 1.0, 0.0],
                    ..Default::default()
                }])
            }
        }
        // The first step primes the drive accumulator (B2D940 clears C0
        // while the total is zero); the second step's post-step forward
        // velocity approaches the +y wall normal, and the selected soft
        // branch dispatches the CONTACT single-contact soft response.
        let state = VehicleState::default();
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let flat = GroundPlaneSet::Single(GroundPlane {
            normal: [0.0, 0.0, 1.0],
            point: [0.0, 0.0, 0.0],
            ..Default::default()
        });
        engine
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                None,
                Some(&flat),
                &GroundStepOptions::default(),
            )
            .unwrap();
        let world = HitWorld;
        let outcome = engine
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                Some(&world),
                Some(&flat),
                &GroundStepOptions::default(),
            )
            .unwrap();
        let sweep = outcome.world_sweep.expect("world path sweeps");
        assert!(sweep.response_dispatched);
        assert!(sweep.response_complete);
        assert!(
            sweep.dispatch_reason == "approaching_soft_contact_full_response"
                || sweep.dispatch_reason == "approaching_soft_contact_base_response",
            "{}",
            sweep.dispatch_reason
        );
        assert_eq!(sweep.dispatch_authority, Some("collision_solver_dispatch"));
        assert!(matches!(
            sweep.angular_response,
            Some(CollisionChainAngularResponse::Soft(_))
        ));
        assert!(matches!(
            sweep.soft_base_response,
            Some(SoftBaseResponse::Full(_))
        ));
        // CONTACT acquisition commits the topology with the grounded return
        // owner (surface_mode_latched == 0 -> GROUND).
        let transition = sweep.topology_transition.expect("dispatch commits");
        assert_eq!(transition.selected_domain, MotionDomain::Contact);
        assert_eq!(
            engine.state.topology.domain,
            MotionDomain::Contact,
            "engine topology after dispatch"
        );
        assert_eq!(
            engine.state.topology.return_domain,
            Some(MotionDomain::Ground)
        );
        // The next no-hit step releases CONTACT back to the return owner.
        let flat_world = FlatSweepWorld;
        let released = engine
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                Some(&flat_world),
                Some(&flat),
                &GroundStepOptions::default(),
            )
            .unwrap();
        let release = released
            .world_sweep
            .expect("world path sweeps")
            .topology_transition
            .expect("release commits when CONTACT is active");
        assert_eq!(release.selected_domain, MotionDomain::Ground);
        assert_eq!(engine.state.topology.domain, MotionDomain::Ground);
    }

    #[test]
    fn takeoff_latch_released_without_overlap_obb_all_capability() {
        // Handoff §4 / kartrider_vehicle_engine.py:6263-6268: with the latch
        // set and a world that lacks overlap_obb_all, step_raw_topology must
        // release takeoff_separation_pending immediately and fall back to
        // wheel-manifold support instead of suppressing support forever.
        struct NoObbWorld;
        impl World for NoObbWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    sweep_obb_rotating_all: true,
                    ..Default::default()
                }
            }
            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult {
                    normal: [0.0, 0.0, 1.0],
                    point: [0.0, 0.0, 0.0],
                    ..Default::default()
                }
            }
            fn sweep_obb_rotating_all(
                &self,
                _start: [f64; 3],
                _end: [f64; 3],
                _start_matrix: [f64; 9],
                _end_matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(Vec::new())
            }
        }
        let state = VehicleState {
            takeoff_separation_pending: true,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = NoObbWorld;
        let outcome = engine
            .step_raw_topology(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&world),
                None,
                None,
            )
            .unwrap();
        // The latch was released and the wheel manifold kept GND ownership.
        assert!(!engine.state.takeoff_separation_pending);
        assert_eq!(engine.state.topology.domain, MotionDomain::Ground);
        assert_eq!(outcome.support_observation.support_contact_count, 4);
        assert!(matches!(
            outcome.executor,
            TopologyExecutorOutcome::Ground(_)
        ));
    }

    #[test]
    fn ordinary_hard_support_overlap_does_not_request_takeoff() {
        struct FactoryRampWorld;
        impl World for FactoryRampWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    overlap_obb_all: true,
                    sweep_obb_rotating_all: true,
                    query_takeoff_contact: true,
                    ..Default::default()
                }
            }
            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult {
                    node: "RoadObj01@s2".to_string(),
                    tri_index: 11,
                    normal: [0.0831683353, -0.0000397030781, 0.99653548],
                    ..Default::default()
                }
            }
            fn overlap_obb_all(
                &self,
                _center: [f64; 3],
                _matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(vec![WorldObbOverlap {
                    hit: true,
                    node: "RoadObj01@s2".to_string(),
                    tri_index: 11,
                    normal: [0.0831683353, -0.0000397030781, 0.99653548],
                    ..Default::default()
                }])
            }
            fn sweep_obb_rotating_all(
                &self,
                _start: [f64; 3],
                _end: [f64; 3],
                _start_matrix: [f64; 9],
                _end_matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(Vec::new())
            }
            fn query_takeoff_contact(
                &self,
                _position: [f64; 3],
                _velocity: [f64; 3],
                _support_planes: &[WorldGroundResult],
                _active_slots: &[i64],
            ) -> Option<Option<crate::world::TakeoffLandingContact>> {
                Some(None)
            }
        }

        let state = VehicleState {
            velocity: [-45.2024803, 0.967220724, 1.04835331],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let outcome = engine
            .step_raw_topology(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&FactoryRampWorld),
                None,
                None,
            )
            .unwrap();
        assert!(outcome.takeoff_state_transfer.is_none());
        assert_ne!(engine.state.topology.domain, MotionDomain::Air);
        assert!(!engine.state.takeoff_separation_pending);
    }

    #[test]
    fn takeoff_latch_suppresses_support_with_overlap_obb_all() {
        // With the capability present the latch holds: the support count fed
        // to the topology observation is forced to zero, so a GND kart with
        // reachable wheel rays transitions AIR-ENTRY instead of staying GND.
        struct ObbWorld;
        impl World for ObbWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    overlap_obb_all: true,
                    sweep_obb_rotating_all: true,
                    ..Default::default()
                }
            }
            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult {
                    normal: [0.0, 0.0, 1.0],
                    point: [0.0, 0.0, 0.0],
                    ..Default::default()
                }
            }
            fn overlap_obb_all(
                &self,
                _center: [f64; 3],
                _matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(Vec::new())
            }
            fn sweep_obb_rotating_all(
                &self,
                _start: [f64; 3],
                _end: [f64; 3],
                _start_matrix: [f64; 9],
                _end_matrix: [f64; 9],
                _half_extents: [f64; 3],
            ) -> Option<Vec<WorldObbOverlap>> {
                Some(Vec::new())
            }
        }
        let state = VehicleState {
            takeoff_separation_pending: true,
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = ObbWorld;
        engine
            .step_raw_topology(
                &keys(&[]),
                &VehicleParameters::default(),
                Some(&world),
                None,
                None,
            )
            .unwrap();
        // Latch retained (no landing overlap candidate). The substep entered
        // AIR-ENTRY and completed the takeoff transfer within the same
        // substep, so the published domain is AIR.
        assert!(engine.state.takeoff_separation_pending);
        assert_eq!(engine.state.topology.domain, MotionDomain::Air);
        assert_eq!(
            engine.state.topology.entry_reason,
            "takeoff_transfer_complete"
        );
    }

    #[test]
    fn frame_loop_clock_init_then_substeps() {
        let state = VehicleState::default();
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = FlatSweepWorld;
        let parameters = VehicleParameters::default();
        let first = engine
            .step_raw_frame(1000, &keys(&["UP"]), &parameters, Some(&world), None, None)
            .unwrap();
        assert!(first.clock_initialized);
        assert_eq!(first.physics_substep_count, 0);
        assert_eq!(engine.state.time_ms, 0);
        let second = engine
            .step_raw_frame(1005, &keys(&["UP"]), &parameters, Some(&world), None, None)
            .unwrap();
        assert!(!second.clock_initialized);
        assert_eq!(second.frame_scheduler.substeps_ms, vec![2, 2, 1]);
        assert_eq!(second.physics_substep_count, 3);
        assert_eq!(engine.state.time_ms, 5);
        // Every substep appended one prediction-history entry.
        assert_eq!(engine.prediction_history.len(), 3);
        // Zero-delta frame steps zero physics.
        let stalled = engine
            .step_raw_frame(1005, &keys(&[]), &parameters, Some(&world), None, None)
            .unwrap();
        assert_eq!(stalled.physics_substep_count, 0);
        assert_eq!(engine.prediction_history.len(), 3);
        assert_eq!(engine.state.time_ms, 5);
    }

    #[test]
    fn r_rising_edge_enters_reset_and_latches_key() {
        let state = VehicleState::default();
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = FlatSweepWorld;
        let parameters = VehicleParameters {
            substep_ms: 2,
            ..Default::default()
        };
        let outcome = engine
            .step_raw_motion(&keys(&["R"]), &parameters, Some(&world), None, None)
            .unwrap();
        assert_eq!(outcome.raw_motion_executor, "RESET");
        assert_eq!(
            outcome.raw_motion_dispatch_reason,
            "raw_R_rising_manual_restart_entry"
        );
        assert_eq!(engine.state.topology.domain, MotionDomain::Reset);
        assert_eq!(engine.state.manual_restart_count, 1);
        assert!(engine.state.manual_restart_key_latched);
        // Holding R is not a second edge: the RESET dispatch guard fires
        // instead (world lacks query_checkpoint_respawn).
        let held = engine.step_raw_motion(&keys(&["R"]), &parameters, Some(&world), None, None);
        assert_eq!(
            held.unwrap_err(),
            "autonomous reset step requires world.query_checkpoint_respawn"
        );
    }

    #[test]
    fn fall_clamp_reset_enters_reset() {
        let state = VehicleState {
            position: [0.0, 0.0, -6.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let world = FlatSweepWorld;
        let parameters = VehicleParameters::default();
        let outcome = engine
            .step_raw_motion(&keys(&[]), &parameters, Some(&world), None, None)
            .unwrap();
        assert_eq!(outcome.raw_motion_executor, "RESET");
        assert_eq!(outcome.raw_motion_dispatch_reason, "fall_clamp_reset_entry");
        assert_eq!(engine.state.topology.domain, MotionDomain::Reset);
        assert_eq!(engine.state.position[2], 0.0);
    }

    #[test]
    fn support_active_false_builds_air_payload() {
        let state = VehicleState {
            velocity: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        let options = GroundStepOptions {
            support_active: Some(false),
            ..Default::default()
        };
        let outcome = engine
            .step_raw_ground_correction(
                &keys(&["UP"]),
                &VehicleParameters::default(),
                None,
                None,
                &options,
            )
            .unwrap();
        let contact = outcome
            .step
            .evaluator
            .contact
            .expect("AIR payload still schedules the contact producer");
        assert_eq!(contact.slots.len(), 4);
        // Recovery drive bodies were popped and B2A100 disabled.
        assert!(outcome.step.evaluator.ground_force.is_none());
        let correction = outcome
            .step
            .evaluator
            .correction
            .expect("correction scheduled");
        assert!(correction.disabled);
    }

    #[test]
    fn query_wheel_ground_uses_float32_anchors() {
        struct RecordingWorld {
            origins: std::cell::RefCell<Vec<[f64; 3]>>,
        }
        impl World for RecordingWorld {
            fn query_ground(
                &self,
                position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                self.origins.borrow_mut().push(position);
                WorldGroundResult::default()
            }
        }
        let world = RecordingWorld {
            origins: std::cell::RefCell::new(Vec::new()),
        };
        let state = VehicleState::default();
        let results = query_wheel_ground(&world, &state, &VehicleParameters::default());
        assert_eq!(results.len(), 4);
        assert_eq!(world.origins.borrow().len(), 4);
    }

    // --- MOVING-CONTACT / RESET domain owner tests (M2 part 3) ---

    struct MovingResetWorld {
        moving_record: Option<BTreeMap<String, JsonValue>>,
        respawn: Option<CheckpointRespawnCandidate>,
    }

    impl World for MovingResetWorld {
        fn capabilities(&self) -> crate::world::WorldCapabilities {
            crate::world::WorldCapabilities {
                query_moving_contact: self.moving_record.is_some(),
                query_checkpoint_respawn: true,
                ..Default::default()
            }
        }

        fn query_ground(
            &self,
            _position: [f64; 3],
            _up: [f64; 3],
            _min_dot: f64,
        ) -> WorldGroundResult {
            WorldGroundResult {
                normal: [0.0, 0.0, 1.0],
                point: [0.0, 0.0, 0.0],
                ..Default::default()
            }
        }

        fn query_moving_contact(
            &self,
            _position: [f64; 3],
            _time_ms: i64,
            _object_id: Option<&str>,
            _delta_ms: i64,
            _max_distance: f64,
        ) -> Option<Option<BTreeMap<String, JsonValue>>> {
            Some(self.moving_record.clone())
        }

        fn query_checkpoint_respawn(
            &self,
            _position: [f64; 3],
            _checkpoint_target: [f64; 3],
            _checkpoint_gate: bool,
            _mode_family: &str,
            _max_course_distance: f64,
        ) -> Option<Option<CheckpointRespawnCandidate>> {
            Some(self.respawn.clone())
        }
    }

    fn moving_record(object_id: &str, velocity: [f64; 3]) -> BTreeMap<String, JsonValue> {
        BTreeMap::from([
            (
                "object_id".to_string(),
                JsonValue::String(object_id.to_string()),
            ),
            (
                "object_position".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(0.0),
                    JsonValue::Number(0.0),
                    JsonValue::Number(0.0),
                ]),
            ),
            (
                "object_velocity".to_string(),
                JsonValue::Array(velocity.iter().map(|v| JsonValue::Number(*v)).collect()),
            ),
            (
                "object_matrix".to_string(),
                JsonValue::Array(
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                        .iter()
                        .map(|v| JsonValue::Number(*v))
                        .collect(),
                ),
            ),
            (
                "object_angular_velocity".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(0.0),
                    JsonValue::Number(0.0),
                    JsonValue::Number(0.0),
                ]),
            ),
            (
                "contact_normal_world".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(0.0),
                    JsonValue::Number(0.0),
                    JsonValue::Number(1.0),
                ]),
            ),
            ("contact_active".to_string(), JsonValue::Bool(true)),
        ])
    }

    #[test]
    fn moving_contact_acquire_step_release_cycle_validates() {
        // Acquire on GND, ride one 2 ms substep on a +x conveyor, then
        // release back to full support. The anchor/owner invariants of
        // `VehicleState.validate` hold at every commit point.
        let state = VehicleState {
            position: [0.25, -0.5, 0.5],
            velocity: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float64).unwrap();
        let acquisition = engine
            .acquire_moving_contact(
                "roadobj:7:node3",
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            )
            .unwrap();
        assert_eq!(engine.state.topology.domain, MotionDomain::MovingContact);
        assert_eq!(
            engine.state.topology.constraint_owner_id.as_deref(),
            Some("roadobj:7:node3")
        );
        assert_eq!(acquisition.anchor.local_position, [0.25, -0.5, 0.5]);
        assert_eq!(engine.state.force_accumulator, [0.0; 3]);
        assert_eq!(engine.state.contact_compression, [0.0; 4]);
        engine.state.validate().unwrap();
        // A second acquisition is rejected while the anchor is active.
        assert_eq!(
            engine
                .acquire_moving_contact(
                    "roadobj:7:node3",
                    [0.0; 3],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0; 3],
                    [0.0, 0.0, 1.0],
                )
                .unwrap_err(),
            "a moving-contact anchor is already active"
        );
        let parameters = VehicleParameters::default();
        let step = engine
            .step_moving_contact(
                &parameters,
                "roadobj:7:node3",
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                "start",
            )
            .unwrap();
        assert_eq!(
            step.moving_contact_contract.object_motion_authority,
            "caller_supplied_dynamic_body_start_state_integrated_by_engine"
        );
        assert_eq!(engine.state.time_ms, 2);
        assert_eq!(engine.state.step_index, 1);
        assert_eq!(engine.state.topology.domain_step_index, 1);
        engine.state.validate().unwrap();
        let release = engine.release_moving_contact(4, 0).unwrap();
        assert_eq!(engine.state.topology.domain, MotionDomain::Ground);
        assert!(engine.state.moving_contact_anchor.is_none());
        assert_eq!(engine.state.surface_mode_latched, Some(0));
        assert!(release.anchor_before.is_some());
        engine.state.validate().unwrap();
        // Release outside MOVING-CONTACT is rejected.
        assert_eq!(
            engine.release_moving_contact(4, 0).unwrap_err(),
            "release requires MOVING-CONTACT topology"
        );
    }

    #[test]
    fn acquire_moving_contact_domain_guards() {
        let state = VehicleState {
            topology: crate::topology::TopologyState {
                domain: MotionDomain::Reset,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).unwrap();
        assert_eq!(
            engine
                .acquire_moving_contact(
                    "roadobj:7:node3",
                    [0.0; 3],
                    [0.0; 3],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0; 3],
                    [0.0, 0.0, 1.0],
                )
                .unwrap_err(),
            "cannot acquire moving contact while RESET owns state"
        );
        let mut engine =
            KartRiderVehicleEngine::new(&VehicleState::default(), NumericMode::Float32).unwrap();
        assert_eq!(
            engine
                .step_moving_contact(
                    &VehicleParameters::default(),
                    "roadobj:7:node3",
                    [0.0; 3],
                    [0.0; 3],
                    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0; 3],
                    "start",
                )
                .unwrap_err(),
            "step_moving_contact requires MOVING-CONTACT topology"
        );
        assert_eq!(
            engine
                .step_autonomous_moving_contact(&VehicleParameters::default(), None)
                .unwrap_err(),
            "autonomous moving-contact step requires MOVING-CONTACT topology"
        );
    }

    #[test]
    fn autonomous_moving_contact_sustain_then_release() {
        let state = VehicleState {
            position: [0.25, -0.5, 0.5],
            velocity: [1.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float64).unwrap();
        let world = MovingResetWorld {
            moving_record: Some(moving_record("roadobj:7:node3", [1.0, 0.0, 0.0])),
            respawn: None,
        };
        let parameters = VehicleParameters::default();
        let acquired = engine
            .acquire_autonomous_moving_contact(Some(&world))
            .unwrap();
        assert!(matches!(
            acquired,
            AcquireAutonomousMovingContactOutcome::Acquired(_)
        ));
        let sustained = engine
            .step_autonomous_moving_contact(&parameters, Some(&world))
            .unwrap();
        let AutonomousMovingContactOutcome::Sustained(sustained) = sustained else {
            panic!("expected a sustained moving-contact step");
        };
        assert!(!sustained.released);
        assert_eq!(
            sustained.moving_contact_selector.reason,
            "contact_active_nonseparating"
        );
        assert_eq!(engine.state.time_ms, 2);
        engine.state.validate().unwrap();
        // World drops the contact: the next step releases to full support.
        let world = MovingResetWorld {
            moving_record: None,
            respawn: None,
        };
        // The capability flag tracks record presence, so build a capability-
        // present world whose selector returns None.
        struct AbsentContactWorld;
        impl World for AbsentContactWorld {
            fn capabilities(&self) -> crate::world::WorldCapabilities {
                crate::world::WorldCapabilities {
                    query_moving_contact: true,
                    ..Default::default()
                }
            }
            fn query_ground(
                &self,
                _position: [f64; 3],
                _up: [f64; 3],
                _min_dot: f64,
            ) -> WorldGroundResult {
                WorldGroundResult {
                    normal: [0.0, 0.0, 1.0],
                    point: [0.0, 0.0, 0.0],
                    ..Default::default()
                }
            }
            fn query_moving_contact(
                &self,
                _position: [f64; 3],
                _time_ms: i64,
                _object_id: Option<&str>,
                _delta_ms: i64,
                _max_distance: f64,
            ) -> Option<Option<BTreeMap<String, JsonValue>>> {
                Some(None)
            }
        }
        let _ = world;
        let released = engine
            .step_autonomous_moving_contact(&parameters, Some(&AbsentContactWorld))
            .unwrap();
        let AutonomousMovingContactOutcome::Released(released) = released else {
            panic!("expected a moving-contact release");
        };
        assert!(released.released);
        assert_eq!(
            released.moving_contact_selector.reason,
            "contact_geometry_released"
        );
        assert_eq!(
            released.support_observation.support_contact_count,
            released
                .release
                .topology_transition
                .observation
                .support_contact_count
        );
        // The synthetic flat plane yields no active wheel slots, so the
        // release destination is AIR-ENTRY (support_contact_count=0).
        assert_eq!(engine.state.topology.domain, MotionDomain::AirEntry);
        assert!(engine.state.moving_contact_anchor.is_none());
        engine.state.validate().unwrap();
    }

    #[test]
    fn autonomous_reset_not_ready_then_respawn_to_gnd() {
        let state = VehicleState {
            topology: crate::topology::TopologyState {
                domain: MotionDomain::Reset,
                ..Default::default()
            },
            checkpoint_gate: true,
            checkpoint_target: [10.0, 0.0, 0.0],
            ..Default::default()
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float64).unwrap();
        let parameters = VehicleParameters::default();
        let world = MovingResetWorld {
            moving_record: None,
            respawn: None,
        };
        let not_ready = engine
            .step_autonomous_reset(&parameters, Some(&world))
            .unwrap();
        let AutonomousResetOutcome::NotReady(not_ready) = not_ready else {
            panic!("expected a not-ready reset outcome");
        };
        assert!(!not_ready.selected);
        assert_eq!(not_ready.reason, "checkpoint_respawn_not_ready");
        assert_eq!(engine.state.topology.domain, MotionDomain::Reset);
        let world = MovingResetWorld {
            moving_record: None,
            respawn: Some(CheckpointRespawnCandidate {
                course_reference_position: [9.0, 0.0, 0.0],
                sample_result_zero: true,
                authority: None,
                ..Default::default()
            }),
        };
        let applied = engine
            .step_autonomous_reset(&parameters, Some(&world))
            .unwrap();
        let AutonomousResetOutcome::Applied(applied) = applied else {
            panic!("expected an applied reset outcome");
        };
        assert_eq!(applied.selector_authority, "world_query_checkpoint_respawn");
        assert!(applied.respawn.replacement.selected);
        assert_eq!(engine.state.position, [10.0, 0.0, 0.0]);
        assert!(!engine.state.checkpoint_gate);
        assert_eq!(engine.state.checkpoint_gate_armed_tick_ms, None);
        assert_eq!(engine.state.topology.domain, MotionDomain::Ground);
        engine.state.validate().unwrap();
    }

    #[test]
    fn autonomous_reset_domain_guard() {
        let mut engine =
            KartRiderVehicleEngine::new(&VehicleState::default(), NumericMode::Float32).unwrap();
        assert_eq!(
            engine
                .step_autonomous_reset(&VehicleParameters::default(), None)
                .unwrap_err(),
            "autonomous reset step requires RESET topology"
        );
    }
}
