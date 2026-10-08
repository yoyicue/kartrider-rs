//! Port of `scripts/evaluate_branch_aware_vehicle_step.py` — the ordered
//! branch/timer scheduler for one vehicle physics substep. This is the
//! central orchestrator: it composes every leaf producer/sink in the proven
//! UpdateTick selector order without merging unlike coordinate sinks.
//!
//! The Python entry point is `evaluate_branch_aware_step(**kwargs)`; here the
//! kwargs become [`BranchAwareStepArgs`] (one typed struct per payload,
//! `Option` for the optional payloads) and the ordered result dict becomes
//! [`BranchAwareStepResult`]. Status strings stay `&'static str`; selection
//! authorities and scheduler phases are caller-supplied `String`s.
//!
//! Python `TypeError`/`KeyError` crash paths (missing required leaf kwargs)
//! map to `Err` with a `missing ...` message; the parity harness never
//! generates them because the Python authority dies before producing a
//! comparable result.

use crate::domains::angular_velocity_integrator::{
    evaluate_angular_velocity_integrator, AngularVelocityIntegratorResult,
};
use crate::domains::booster_resource::InventorySlot;
use crate::domains::booster_vehicle_step::{
    evaluate_vehicle_step, BoosterVehicleStepResult, InventoryUseInput,
};
use crate::domains::bs_zone_override::{evaluate_bs_zone_override, BsZoneOverrideResult};
use crate::domains::checkpoint_respawn_state_replacement::{
    evaluate_checkpoint_respawn_state_replacement, CheckpointRespawnStateReplacement,
};
use crate::domains::checkpoint_target_selector::{
    evaluate_checkpoint_target_selector, CheckpointTargetSelectorResult,
};
use crate::domains::common_damping::{evaluate_common_damping, CommonDampingResult};
use crate::domains::contact_accumulator_producer::{
    evaluate_contact_accumulator_producer, ContactAccumulatorProducerResult,
};
use crate::domains::drift_recovery_timer::{
    evaluate_recovery_substep, evaluate_surface1_recovery_force, RecoverySubstepResult,
    Surface1RecoveryForceResult,
};
use crate::domains::drift_window_branch::{evaluate_drift_window_branch, DriftWindowBranchResult};
use crate::domains::fall_clamp_state_replacement::{
    evaluate_fall_clamp_state_replacement, FallClampStateReplacement,
};
use crate::domains::ground_accumulator_sink::{
    apply_ground_accumulator_sink, GroundAccumulatorSinkResult,
};
use crate::domains::ground_drive_force_body::{
    evaluate_ground_drive_force_body, GroundDriveForceBodyResult,
};
use crate::domains::ground_state_scalar::{
    evaluate_ground_correction_pipeline, GroundCorrectionPipelineResult,
};
use crate::domains::hard_contact_velocity_response::{
    evaluate_hard_contact_velocity_response, HardContactVelocityResponse,
};
use crate::domains::jm_dj_trigger::{evaluate_jm_dj_trigger, JmDjTrigger};
use crate::domains::linear_velocity_integrator::{
    evaluate_linear_velocity_integrator, LinearVelocityIntegratorResult,
};
use crate::domains::low_projection_ground_branch::{
    evaluate_low_projection_branch, LowProjectionBranchResult,
};
use crate::domains::mz_magnet_trigger::{evaluate_mz_magnet_trigger, MzMagnetTrigger};
use crate::domains::orientation_integrator::{
    evaluate_orientation_integrator, OrientationIntegratorResult,
};
use crate::domains::pit_resource_step::{evaluate_pit_resource_step, PitResourceStep};
use crate::domains::position_integrator::{evaluate_position_integrator, PositionIntegratorResult};
use crate::domains::post_bs_reset_constraint::{
    evaluate_post_bs_reset_constraint, select_post_bs_reset_constraint,
    PostBsResetConstraintResult, PostBsResetSelection,
};
use crate::domains::rail_accumulator_producers::{
    evaluate_rail_accumulator_producers, select_rail_accumulator_kinds, RailAccumulatorKinds,
    RailAccumulatorProducersResult,
};
use crate::domains::soft_contact_velocity_response::{
    evaluate_soft_contact_velocity_response, SoftContactVelocityResponse,
};
use crate::domains::speed_threshold_latch::{
    evaluate_speed_threshold_latch, SpeedThresholdLatchResult,
};
use crate::domains::state18_feedback::select_state18_feedback;
use crate::domains::substep_accumulator_boundary::{
    evaluate_substep_accumulator_boundary, SubstepAccumulatorBoundaryResult,
};
use crate::domains::surface_tune_defaults::{
    GroundSurfaceProfile, NonSurfaceProfile, RecoverySurfaceProfile,
};
use crate::domains::tune_boost_gauge::{
    evaluate_bcharge_surface_step, BchargeSurfaceStep, CurrentFamilyGaugeProducersArgs,
};
use crate::num::f32q;

#[cfg(feature = "serde")]
use serde::Deserialize;

// ---------------------------------------------------------------------------
// Static metadata (Python `ordered_events`, `composition_policy`,
// `static_sources` result keys).
// ---------------------------------------------------------------------------

/// Python `ordered_events` result key, verbatim.
pub const ORDERED_EVENTS: &[&str] = &[
    "replace active +0x614/+0x620 from one-shot pending +0xB74/+0xB80 and clear B74/B80/B90",
    "if an accepted stroke-feedback event precedes UpdateTick, request decoded state18 with +0x630=1000 ms; do not add a state18 force term",
    "decrement boost +0x630 and clear expired decoded state",
    "when rail state2/3 gate is active, apply explicit rail writer kinds before contact/ground producers",
    "apply four contact slots and common 604*660 force before the selected ground producer",
    "apply ordered ground correction primary/kind31..37 actions, including direct C0 zero, before ground drive",
    "select ground state scalar from post-expiry boost state",
    "select low-projection branch before high-projection surface/recovery branches",
    "select B08 drift-window before active/recovery and normal surface branches",
    "select surface1 recovery from entry latch/timer gates",
    "calculate either selected surface1 recovery or normal/nonzero/slip ground-drive force/torque components",
    "for recovery, decrement +0x65C then +0xAF8 before submitting the precomputed force/torque to their sinks",
    "if BS trigger is accepted, replace +0x614 with BS vector and set state13/1000ms while preserving +0x620",
    "if MZ trigger is selected, replace C0 with horizontal (authored_target-sub_6F7C50(this))*token_scale, then conditionally set state16/3000ms; an already-active state13..16 suppresses only the state/FX transition",
    "if JM/DJ trigger is selected, add the exact JM basis/world-Z delta or apply the DJ direct-vector replacement, reset +0x648 cooldown to 0.3, then conditionally set state14/15 for 1000ms",
    "on the branch mutually exclusive with contact/trigger handlers (including BS) and rail state2/3, reset AF4/AFC/B08 and apply the B2D3C0 no-contact fallback force/torque",
    "if a JM/DJ hard-contact writer fires, apply the JM additive or DJ replacement C0 write after ground/BS handlers and before common damping",
    "apply common linear/angular damping and optional normalized directional drag to +0x614/+0x620",
    "clear B8C speed-threshold latch if entry velocity length reaches it",
    "apply explicit post-BS later-producer delta, fold +0xBAC, then integrate +0x614/+0x660*dt into velocity +0xC0",
    "integrate +0x620 into angular velocity +0xCC through M9C4 and gyroscopic cross term",
    "if the later wall-collision writer fires, replace C0 after linear integration and before the pose integrator",
    "apply explicit B2FD80 pre-position velocity delta, then integrate +0x90 += +0xC0*dt",
    "integrate orientation q += (q*[0,CC])*dt/2, normalize, rebuild +0x9C, then run the gated four-attempt upright stabilization",
    "after the pose step, if BD7840 observes position.z < -5, apply B16290 as a terminal replacement: preserve x/y and matrix, set z=0, copy +0x90 to +0x70, and clear +0xC0/+0xCC",
    "after the immediate fall clamp, dispatch stage vtable+0x164; C01690/C72C80 mode overrides may replace the state again with the saved checkpoint target, reconstructed course-facing matrix, and zero +0xC0/+0xCC",
    "before checkpoint replacement, apply ordered C016/C72C saved-target initialization, mission-progress snapshots, and gate arm/clear writers",
];

/// Python `composition_policy` result key, verbatim.
pub const COMPOSITION_POLICY: &str = "Compose rail, four-slot contact, ordered ground correction, and the selected mutually exclusive ApplyGroundDriveForces producer in UpdateTick order before authored trigger handlers. BS replaces +0x614; MZ directly replaces C0 before common damping; JM adds its basis/world-Z launch delta while DJ replaces C0 at the same pre-damping trigger phase. A post_linear_pre_pose wall response is later than damping/linear integration and feeds only pose plus next-substep C0. Ground-correction direct C0 writes feed later consumers unless superseded at one of those explicit phases.";

/// Python `static_sources` result key; each entry maps to one or more VAs.
pub const STATIC_SOURCES: &[(&str, &[&str])] = &[
    ("boost_timer_clear_va", &["0xB26101-0xB2611F"]),
    ("ground_scalar_call_va", &["0xB262E6"]),
    ("mz_velocity_write_va", &["0xB2C838-0xB2C843"]),
    ("mz_state16_set_va", &["0xB2C870"]),
    ("jm_dj_velocity_write_va", &["0xB2D014-0xB2D113"]),
    ("jm_dj_state14_15_set_vas", &["0xB2D142", "0xB2D097"]),
    ("surface1_recovery_va", &["0xB2B396-0xB2B7E8"]),
    ("recovery_force_sink_va", &["0xB2C111"]),
    ("recovery_torque_sink_va", &["0xB2C177"]),
    ("recovery_af8_decay_va", &["0xB2BB9E-0xB2BBCE"]),
    ("fall_reset_selector_va", &["0xBD7840"]),
    ("fall_reset_writer_va", &["0xB16290"]),
    ("fall_reset_callsite_va", &["0xBD788C"]),
    ("checkpoint_respawn_mode_vas", &["0xC01690", "0xC72C80"]),
    ("checkpoint_respawn_writer_va", &["0xB16290"]),
];

// ---------------------------------------------------------------------------
// Serde helpers: engine-captured payloads sometimes carry int-valued floats
// (e.g. ground_correction.selected_slot = 1.0). Python is numeric-agnostic;
// accept both JSON ints and integral floats for i64 fields.
// ---------------------------------------------------------------------------

#[cfg(feature = "serde")]
fn de_i64_lenient<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(f64::deserialize(deserializer)? as i64)
}

#[cfg(feature = "serde")]
fn de_opt_i64_lenient<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<f64>::deserialize(deserializer)?.map(|v| v as i64))
}

// ---------------------------------------------------------------------------
// Args: one struct per Python kwargs payload. Fields that have Python
// defaults, or whose key presence the orchestrator inspects, are `Option`.
// ---------------------------------------------------------------------------

/// Python `evaluate_branch_aware_step(**kwargs)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BranchAwareStepArgs {
    // Bound from the frame by `_bind_state`; Python-built control frames
    // omit this key inside the model payload.
    #[cfg_attr(feature = "serde", serde(default, deserialize_with = "de_i64_lenient"))]
    pub elapsed_ms: i64,
    pub boost: BoostArgs,
    #[cfg_attr(feature = "serde", serde(default))]
    pub boost_gauge_producers: Option<CurrentFamilyGaugeProducersArgs>,
    pub recovery: RecoveryArgs,
    pub state18_feedback: Option<State18FeedbackArgs>,
    pub pit_resource: Option<PitResourceArgs>,
    pub bcharge_resource: Option<BchargeResourceArgs>,
    pub accumulator: Option<AccumulatorArgs>,
    pub substep_boundary: Option<SubstepBoundaryArgs>,
    pub rail: Option<RailArgs>,
    pub contact: Option<ContactArgs>,
    pub hard_contact: Option<HardContactArgs>,
    pub soft_contact: Option<SoftContactArgs>,
    pub ground_correction: Option<GroundCorrectionArgs>,
    pub bs_zone: Option<BsZoneArgs>,
    pub mz_magnet: Option<MzMagnetArgs>,
    pub jm_dj_trigger: Option<JmDjArgs>,
    pub post_bs_reset: Option<PostBsResetArgs>,
    pub common_damping: Option<CommonDampingArgs>,
    pub speed_threshold: Option<SpeedThresholdArgs>,
    pub linear_integrator: Option<LinearIntegratorArgs>,
    pub angular_integrator: Option<AngularIntegratorArgs>,
    pub position_integrator: Option<PositionIntegratorArgs>,
    pub orientation_integrator: Option<OrientationIntegratorArgs>,
    pub terminal_fall_reset: Option<FallResetArgs>,
    pub checkpoint_target_selector: Option<CheckpointTargetSelectorArgs>,
    pub terminal_checkpoint_respawn: Option<CheckpointRespawnArgs>,
}

/// Python `boost` payload (`evaluate_vehicle_step` kwargs minus
/// `elapsed_ms`, which the orchestrator injects).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BoostArgs {
    // Bound from engine state (`_bind_state`); Python-built control frames
    // omit these keys.
    #[cfg_attr(feature = "serde", serde(default, deserialize_with = "de_i64_lenient"))]
    pub state: i64,
    #[cfg_attr(feature = "serde", serde(default, deserialize_with = "de_i64_lenient"))]
    pub remaining_ms: i64,
    pub correction_dir: [f64; 3],
    pub direction_gain: f64,
    pub selected_slot: f64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
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
    pub inventory_use: Option<InventoryUseArgs>,
}

/// Python `inventory_use` payload (`evaluate_booster_inventory_use` kwargs
/// minus `current_state`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct InventoryUseArgs {
    pub slots: Vec<InventorySlotArgs>,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub stage_mode_id: i64,
    pub allow_refresh_in_mode72: bool,
    pub positive_a80: f64,
    pub state_setter_environment_allowed: bool,
    pub normal_booster_time: f64,
    pub item_booster_time: f64,
    pub animal_booster_time: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct InventorySlotArgs {
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub item_id: i64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub quantity: i64,
}

impl InventoryUseArgs {
    fn to_input(&self) -> InventoryUseInput {
        InventoryUseInput {
            slots: self
                .slots
                .iter()
                .map(|slot| InventorySlot {
                    item_id: slot.item_id,
                    quantity: slot.quantity,
                })
                .collect(),
            stage_mode_id: self.stage_mode_id,
            allow_refresh_in_mode72: self.allow_refresh_in_mode72,
            positive_a80: self.positive_a80,
            state_setter_environment_allowed: self.state_setter_environment_allowed,
            normal_booster_time: self.normal_booster_time,
            item_booster_time: self.item_booster_time,
            animal_booster_time: self.animal_booster_time,
        }
    }
}

/// Python `state18_feedback` payload (`select_state18_feedback` kwargs).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct State18FeedbackArgs {
    pub stroke_dispatch_selected: bool,
    pub control_lock_1c0: bool,
    pub event_active: bool,
    pub effect_feedback_enabled_258: bool,
}

/// Python `pit_resource` payload (`evaluate_pit_resource_step` kwargs minus
/// `dt`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct PitResourceArgs {
    pub active: bool,
    pub rate: f64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub mode: i64,
    pub accum_before: f64,
}

/// Python `bcharge_resource` payload (`evaluate_bcharge_surface_step`
/// kwargs).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BchargeResourceArgs {
    pub active: bool,
    pub accum_b10: f64,
    pub raw_760: f64,
}

/// Python `recovery` payload: the orchestrator reads the five sub-payloads.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RecoveryArgs {
    #[cfg_attr(feature = "serde", serde(default))]
    pub disabled_b9c: bool,
    pub schedule: Option<RecoveryScheduleArgs>,
    pub low_projection: Option<LowProjectionArgs>,
    pub drift_window: Option<DriftWindowArgs>,
    pub force: Option<Surface1RecoveryForceArgs>,
    pub ground_force: Option<GroundDriveForceArgs>,
}

/// Python `recovery["schedule"]` (`evaluate_recovery_substep` kwargs minus
/// `dt`). All fields are `Option` because the orchestrator inspects key
/// presence of `surface_mode`/`selector_af4` for the contradiction guards.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RecoveryScheduleArgs {
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub surface_mode: Option<i64>,
    pub selector_af4: Option<bool>,
    pub field_af8: Option<f64>,
    pub recovery_timer: Option<f64>,
    pub other_recovery_gate: Option<bool>,
}

/// Python `recovery["low_projection"]` (`evaluate_low_projection_branch`
/// kwargs).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct LowProjectionArgs {
    pub projection_len: f64,
    pub ground_forward_term: f64,
    pub drive_direction: f64,
    pub normalized_velocity_raw: f64,
    pub correction_raw: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub base_lateral_gain: f64,
    pub axis_long: Option<f64>,
    pub axis_lat: Option<f64>,
    pub state_5c8: Option<f64>,
    pub b1c_charge_settlement: Option<bool>,
    pub b10: Option<f64>,
    pub b20: Option<f64>,
    pub b2c: Option<f64>,
    pub b38: Option<f64>,
    pub cap_760: Option<f64>,
    pub numeric_mode: Option<String>,
}

/// Python `recovery["drift_window"]` (`evaluate_drift_window_branch` kwargs
/// minus `dt`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct DriftWindowArgs {
    pub selector_b08: bool,
    pub timer_b0c: f64,
    pub configured_duration_734: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub signed_axis: f64,
    pub scaled_tune_660: f64,
    pub drift_gain_728: f64,
    pub tune_758: f64,
    pub axis_lat: f64,
    pub drive_raw_aa0: f64,
    pub debug_force_scale: Option<f64>,
}

/// Python `surface_tune_defaults` ground profile dict.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct GroundSurfaceProfileArgs {
    pub speed_denominator: f64,
    pub speed_quadratic_gain: f64,
    pub speed_bias: f64,
    pub force_scale: f64,
}

/// Python `surface_tune_defaults` recovery profile dict.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RecoverySurfaceProfileArgs {
    pub longitudinal_gain_offset: f64,
    pub lateral_gain_offset: f64,
    pub force_scale: f64,
    pub forward_target_gain: f64,
}

/// Python `surface_tune_defaults` non-surface1 profile dict.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct NonSurfaceProfileArgs {
    pub longitudinal_gain_offset: f64,
    pub lateral_gain_offset: f64,
    pub force_scale: f64,
}

impl From<GroundSurfaceProfileArgs> for GroundSurfaceProfile {
    fn from(args: GroundSurfaceProfileArgs) -> Self {
        GroundSurfaceProfile {
            speed_denominator: args.speed_denominator,
            speed_quadratic_gain: args.speed_quadratic_gain,
            speed_bias: args.speed_bias,
            force_scale: args.force_scale,
        }
    }
}

impl From<RecoverySurfaceProfileArgs> for RecoverySurfaceProfile {
    fn from(args: RecoverySurfaceProfileArgs) -> Self {
        RecoverySurfaceProfile {
            longitudinal_gain_offset: args.longitudinal_gain_offset,
            lateral_gain_offset: args.lateral_gain_offset,
            force_scale: args.force_scale,
            forward_target_gain: args.forward_target_gain,
        }
    }
}

impl From<NonSurfaceProfileArgs> for NonSurfaceProfile {
    fn from(args: NonSurfaceProfileArgs) -> Self {
        NonSurfaceProfile {
            longitudinal_gain_offset: args.longitudinal_gain_offset,
            lateral_gain_offset: args.lateral_gain_offset,
            force_scale: args.force_scale,
        }
    }
}

/// Python `recovery["force"]` (`evaluate_surface1_recovery_force` kwargs).
/// `selector_af4`/`surface_mode` are `Option` because the orchestrator
/// reconciles them against the entry selector snapshot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct Surface1RecoveryForceArgs {
    pub selector_af4: Option<bool>,
    pub speed_magnitude: f64,
    pub speed_profile_denominator: f64,
    pub blend_ab0: f64,
    pub blend_ab4: f64,
    pub active_ab0_rate: f64,
    pub active_ab4_rate: f64,
    pub timer_ab0_rate: f64,
    pub contact_basis_product: f64,
    pub ground_forward_term: f64,
    pub drive_direction: f64,
    pub normalized_velocity: f64,
    pub longitudinal_correction: f64,
    pub lateral_correction: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub base_lateral_gain: f64,
    pub ground_force_multiplier: f64,
    pub tune_758: f64,
    pub axis_long: f64,
    pub axis_lat: f64,
    pub active_surface: Option<RecoverySurfaceProfileArgs>,
    pub timer_surface: Option<RecoverySurfaceProfileArgs>,
    pub field_af0: Option<f64>,
    pub torque_y_scale: Option<f64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub surface_mode: Option<i64>,
    pub non_surface_profile: Option<NonSurfaceProfileArgs>,
}

/// Python `recovery["ground_force"]` (`evaluate_ground_drive_force_body`
/// kwargs). `surface_mode` is `Option` for the entry-snapshot
/// reconciliation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct GroundDriveForceArgs {
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub surface_mode: Option<i64>,
    pub speed_magnitude: f64,
    pub ground_forward_term: f64,
    pub contact_basis_product: f64,
    pub drive_direction: f64,
    pub normalized_velocity: f64,
    pub longitudinal_correction: f64,
    pub lateral_correction: f64,
    pub mass_scale: f64,
    pub base_longitudinal_gain: f64,
    pub base_lateral_gain: f64,
    pub ground_force_multiplier: f64,
    pub settlement_multiplier: f64,
    pub tune_75c: f64,
    pub axis_long: f64,
    pub axis_lat: f64,
    pub surface_profile: Option<GroundSurfaceProfileArgs>,
    pub torque_y_scale: Option<f64>,
    pub torque_y_gain: Option<f64>,
    pub state_5c8: Option<f64>,
    pub b1c_charge_settlement: Option<bool>,
    pub b10: Option<f64>,
    pub b20: Option<f64>,
    pub b2c: Option<f64>,
    pub b38: Option<f64>,
    pub cap_760: Option<f64>,
    pub numeric_mode: Option<String>,
}

/// Python `accumulator` payload (`apply_ground_accumulator_sink` kwargs
/// minus the orchestrator-supplied force/torque/befores).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct AccumulatorArgs {
    // Bound from engine state (`_bind_state`); Python-built control frames
    // omit these keys.
    #[cfg_attr(feature = "serde", serde(default))]
    pub accumulator_614_before: [f64; 3],
    #[cfg_attr(feature = "serde", serde(default))]
    pub accumulator_620_before: [f64; 3],
    pub matrix_9c: [f64; 9],
    pub numeric_mode: Option<String>,
}

/// Python `substep_boundary` payload.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct SubstepBoundaryArgs {
    pub pending_force_b74: [f64; 3],
    pub pending_torque_b80: [f64; 3],
    pub pending_b90: [f64; 3],
}

/// Python `rail["selector"]` (`select_rail_accumulator_kinds` kwargs).
/// `boost_family_active`/`attached_object_state` are `Option` because the
/// orchestrator reconciles them against the post-timer decoded state.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RailSelectorArgs {
    pub rail_contact: bool,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub rail_state_1cc: i64,
    pub gate_1c0_decoded: bool,
    pub positive_a80: f64,
    pub negative_a80: f64,
    pub boost_family_active: Option<bool>,
    pub boost_special_gate_9b8: bool,
    pub boost_special_gate_6e4: bool,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub attached_object_state: Option<i64>,
    pub abs_ae0: f64,
    pub adc: f64,
}

/// Python `rail` payload: `selector`/`enabled_kinds_source` are popped by
/// the orchestrator; the rest forwards to
/// `evaluate_rail_accumulator_producers`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RailArgs {
    pub selector: Option<RailSelectorArgs>,
    pub enabled_kinds_source: Option<String>,
    pub enabled_kinds: Option<Vec<i64>>,
    pub rail_dir: [f64; 3],
    pub selected_a80: f64,
    pub accel_be0: f64,
    pub obf_6e8: f64,
    pub obf_808: f64,
    pub obf_814: f64,
    pub grip_720: f64,
    pub angular_velocity_cc: [f64; 3],
    pub field_604: [f64; 3],
    pub obf_660: f64,
    pub gravity_be8: f64,
}

/// Python `contact` payload (`evaluate_contact_accumulator_producer` kwargs
/// minus the orchestrator-supplied befores).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct ContactArgs {
    pub elapsed_seconds: f64,
    pub contact_active: [bool; 4],
    pub compression_a54: [f64; 4],
    pub compression_rate_a64: [f64; 4],
    pub contact_vectors_a14: [[f64; 3]; 4],
    pub contact_normal_ac4: [f64; 3],
    pub spring_7c0: f64,
    pub rebound_7c4: f64,
    pub bump_7c8: f64,
    pub lever_x_9f4: [f64; 4],
    pub lever_y_9f8: [f64; 4],
    pub obf_66c: f64,
    pub obf_678: f64,
    pub field_604: [f64; 3],
    pub obf_660: f64,
}

/// Python `hard_contact` payload; `selection_authority`/`scheduler_phase`
/// are popped by the orchestrator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct HardContactArgs {
    pub velocity_c0_before: [f64; 3],
    pub hit_normal: [f64; 3],
    pub basis_ad0: [f64; 3],
    pub restitution: Option<f64>,
    pub selection_authority: Option<String>,
    pub scheduler_phase: Option<String>,
}

/// Python `soft_contact` payload; `selection_authority`/`scheduler_phase`
/// are popped by the orchestrator and `velocity_c0_before` is injected at
/// the pre-pose stage.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct SoftContactArgs {
    pub velocity_c0_before: Option<[f64; 3]>,
    pub velocity_delta: [f64; 3],
    pub selection_authority: Option<String>,
    pub scheduler_phase: Option<String>,
}

/// Python `ground_correction` payload
/// (`evaluate_ground_correction_pipeline` kwargs minus the
/// orchestrator-supplied befores and `dt`).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct GroundCorrectionArgs {
    pub velocity_c0: [f64; 3],
    pub disabled_b9c: bool,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub selected_slot: i64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub opposite_slot: i64,
    pub gate_1c0: bool,
    pub gate_6f3b00: bool,
    pub gate_6f3ae0: bool,
    pub field_adc: f64,
    pub timer_a80_before: f64,
    pub field_ae0: f64,
    pub b8c: f64,
    pub lateral_dot: f64,
    pub correction_dir: [f64; 3],
    pub direction_gain: f64,
    pub state_mult: f64,
    pub flag_af4: bool,
    pub flag_afc: bool,
    pub sub_ad0: [f64; 3],
    pub sub_af0: f64,
    pub field_9f0: f64,
    pub sub_6f4: f64,
    pub sub_bc4: f64,
    pub drive_slot1_a90: f64,
    pub axis_ac4: [f64; 3],
    pub sub_720: f64,
    pub sub_724: f64,
    pub numeric_mode: Option<String>,
}

/// Python `bs_zone` payload; `selection_authority` is popped and
/// `velocity_c0` plus the befores are reconciled by the orchestrator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct BsZoneArgs {
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub digit_x: i64,
    #[cfg_attr(feature = "serde", serde(deserialize_with = "de_i64_lenient"))]
    pub digit_y: i64,
    pub flag_64c: bool,
    pub field_190: [f64; 3],
    pub velocity_c0: [f64; 3],
    pub selection_authority: Option<String>,
}

/// Python `mz_magnet` payload; `selection_authority` is popped and
/// `special_state_family_already_active` is reconciled by the orchestrator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct MzMagnetArgs {
    pub current_reference_xyz: [f64; 3],
    pub authored_target_xyz: [f64; 3],
    pub velocity_scale: f64,
    pub special_state_family_already_active: Option<bool>,
    pub selection_authority: Option<String>,
}

/// Python `jm_dj_trigger` payload; `selection_authority` is popped, the
/// special-family gate is reconciled, and JM `velocity_c0_before` is
/// injected from the pre-damping velocity.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct JmDjArgs {
    pub variant: String,
    pub special_state_family_already_active: Option<bool>,
    pub velocity_c0_before: Option<[f64; 3]>,
    pub jm_basis_190: Option<[f64; 3]>,
    pub jm_horizontal_param: Option<f64>,
    pub jm_basis_param: Option<f64>,
    pub dj_vector: Option<[f64; 3]>,
    pub boost_multiplier_64c: Option<bool>,
    pub selection_authority: Option<String>,
}

/// Python `post_bs_reset["selector"]` (`select_post_bs_reset_constraint`
/// kwargs). `rail_state_1cc` is `Option` because the orchestrator injects
/// the Rail selector snapshot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct PostBsResetSelectorArgs {
    pub current_contact_updated: bool,
    pub contact_trigger_e38: bool,
    pub contact_trigger_e4c: bool,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub rail_state_1cc: Option<i64>,
}

/// Python `post_bs_reset` payload; `selector`/`selection_authority` are
/// popped and the befores are orchestrator-supplied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct PostBsResetArgs {
    pub selector: Option<PostBsResetSelectorArgs>,
    pub selection_authority: Option<String>,
    pub vector_604: Option<[f64; 3]>,
    pub sub_660: Option<f64>,
    pub divisor_e50: Option<f64>,
    pub angular_velocity_cc: Option<[f64; 3]>,
    pub special_orientation_gate: Option<bool>,
    pub field_acc: Option<f64>,
    pub field_ac0: Option<f64>,
    pub field_ad8: Option<f64>,
}

/// Python `common_damping` payload (`evaluate_common_damping` kwargs minus
/// the orchestrator-supplied befores). `velocity_c0` is `Option` because
/// the orchestrator overrides it with the pre-damping direct writer when
/// one fired, and JM inspects key presence.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct CommonDampingArgs {
    pub velocity_c0: Option<[f64; 3]>,
    // Bound from engine state (`_bind_state`); Python-built control frames
    // omit this key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub angular_velocity_cc: [f64; 3],
    pub damping_700: f64,
    pub special_gate: bool,
    pub directional_70c: Option<f64>,
    pub runtime_ba0: Option<f64>,
    pub token_multiplier: Option<f64>,
    pub numeric_mode: Option<String>,
}

/// Python `speed_threshold` payload (`evaluate_speed_threshold_latch`
/// kwargs; `velocity_c0` may be overridden by the pre-damping writer).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct SpeedThresholdArgs {
    pub threshold_b8c: f64,
    pub velocity_c0: Option<[f64; 3]>,
}

/// Python `linear_integrator` payload; `accumulator_614_override` is popped
/// and `dt` plus the befores are orchestrator-supplied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct LinearIntegratorArgs {
    pub velocity_c0_before: Option<[f64; 3]>,
    pub accumulator_614_override: Option<[f64; 3]>,
    pub bac_force: [f64; 3],
    pub sub_660: f64,
    pub post_bs_later_delta: Option<[f64; 3]>,
    pub numeric_mode: Option<String>,
}

/// Python `angular_integrator` payload; `accumulator_620_override` is
/// popped and `dt` is orchestrator-supplied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct AngularIntegratorArgs {
    pub angular_velocity_cc_before: [f64; 3],
    pub accumulator_620_override: Option<[f64; 3]>,
    pub matrix_9c4: [f64; 9],
    pub numeric_mode: Option<String>,
}

/// Python `position_integrator` payload; `dt` and the effective
/// `velocity_c0_before` are orchestrator-supplied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct PositionIntegratorArgs {
    // Bound from engine state (`_bind_state`); Python-built control frames
    // omit this key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub position_90_before: [f64; 3],
    pub velocity_c0_before: Option<[f64; 3]>,
    pub pre_position_velocity_delta: Option<[f64; 3]>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub state_1d0: Option<i64>,
    pub latch_658: Option<bool>,
    pub obf_b44: Option<f64>,
    pub axis_ab8: Option<[f64; 3]>,
    pub axis_a48: Option<[f64; 3]>,
}

/// Python `orientation_integrator` payload; `dt` and the effective
/// `angular_velocity_cc_before` are orchestrator-supplied.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct OrientationIntegratorArgs {
    // Bound from engine state (`_bind_state`); Python-built control frames
    // omit this key.
    #[cfg_attr(feature = "serde", serde(default))]
    pub matrix_9c_before: [f64; 9],
    pub angular_velocity_cc_before: Option<[f64; 3]>,
    pub pre_clear_xy: Option<bool>,
    pub stabilization_disabled_b70: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub rail_state_1cc: Option<i64>,
    pub runtime_float32_exact: Option<bool>,
}

/// Python `terminal_fall_reset` payload; the `*_before` fields are
/// overridden by the post-pose state when the pose step ran.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct FallResetArgs {
    pub position_90_before: Option<[f64; 3]>,
    pub matrix_9c_before: Option<[f64; 9]>,
    pub velocity_c0_before: Option<[f64; 3]>,
    pub angular_velocity_cc_before: Option<[f64; 3]>,
    pub drop_threshold: Option<f64>,
    pub clamp_z: Option<f64>,
}

/// Python `checkpoint_target_selector["events"]` entry
/// (`evaluate_checkpoint_target_selector` kwargs minus the chained
/// mode/target/gate).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct CheckpointTargetEventArgs {
    pub event: String,
    pub current_kart_position: Option<[f64; 3]>,
    pub config_gate: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub mode_state: Option<i64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub mission_index: Option<i64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub mission_count: Option<i64>,
    pub retire_detected: Option<bool>,
    pub reset_suppressed: Option<bool>,
    pub trace_value: Option<f64>,
    pub trace_min: Option<f64>,
    pub trace_max: Option<f64>,
    pub peer_present: Option<bool>,
    pub kart_event_accepts: Option<bool>,
    pub stage_input_blocked: Option<bool>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub tick_ms: Option<i64>,
    #[cfg_attr(
        feature = "serde",
        serde(default, deserialize_with = "de_opt_i64_lenient")
    )]
    pub gate_armed_tick_ms: Option<i64>,
}

/// Python `checkpoint_target_selector` payload.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct CheckpointTargetSelectorArgs {
    pub mode_family: String,
    pub target_before: [f64; 3],
    pub gate_before: bool,
    pub events: Option<Vec<CheckpointTargetEventArgs>>,
}

/// Python `terminal_checkpoint_respawn` payload; `mode_family`,
/// `checkpoint_gate`, `target_position`, and the velocity fields are
/// overridden by the chained selector state / post-pose state when
/// available.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct CheckpointRespawnArgs {
    pub mode_family: Option<String>,
    pub checkpoint_gate: Option<bool>,
    pub sample_result_zero: Option<bool>,
    pub target_position: Option<[f64; 3]>,
    pub course_reference_position: Option<[f64; 3]>,
    pub velocity_c0_before: Option<[f64; 3]>,
    pub angular_velocity_cc_before: Option<[f64; 3]>,
}

// ---------------------------------------------------------------------------
// Result: Python's ordered result dict, as typed fields. The `*_status`
// strings and the outcome enums carry exactly the values Python computes.
// ---------------------------------------------------------------------------

/// Python `surface1_recovery_schedule` value: either the inline suppression
/// dict or the full `evaluate_recovery_substep` result.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryScheduleOutcome {
    /// `{"branch_for_force": False, "suppressed": True, "reason": reason}`.
    Suppressed {
        reason: &'static str,
    },
    Evaluated(RecoverySubstepResult),
}

impl RecoveryScheduleOutcome {
    fn branch_for_force(&self) -> bool {
        match self {
            RecoveryScheduleOutcome::Suppressed { .. } => false,
            RecoveryScheduleOutcome::Evaluated(result) => result.branch_for_force,
        }
    }
}

/// Python `rail_selector` value. The exact-selector path adds
/// `selection_authority="exact_selector"` to the selector result; the
/// replay path is the inline observed-probe-rows dict.
#[derive(Debug, Clone, PartialEq)]
pub enum RailSelectorOutcome {
    Exact(RailAccumulatorKinds),
    Observed { enabled_kinds: Vec<i64> },
}

/// Python `post_bs_reset_selector` value (`selection_authority` added by
/// the orchestrator).
#[derive(Debug, Clone, PartialEq)]
pub enum PostBsResetSelectorOutcome {
    Exact(PostBsResetSelection),
    ObservedProbeRows,
}

/// Python `soft_contact_velocity` value: while the position integrator has
/// not run, the pending-args dict is embedded verbatim
/// (`{"pending_args": ..., "selection_authority": ...}`); after the
/// pre-pose stage it is the evaluated response with
/// `selection_authority`/`scheduler_phase` attached.
#[derive(Debug, Clone, PartialEq)]
pub enum SoftContactOutcome {
    NotSelected,
    Pending {
        velocity_c0_before: Option<[f64; 3]>,
        velocity_delta: [f64; 3],
        selection_authority: String,
    },
    Evaluated {
        result: SoftContactVelocityResponse,
        selection_authority: String,
        scheduler_phase: String,
    },
}

impl SoftContactOutcome {
    pub fn evaluated_velocity(&self) -> Option<[f64; 3]> {
        match self {
            SoftContactOutcome::Evaluated { result, .. } => Some(result.velocity_c0_after),
            _ => None,
        }
    }
}

/// Python `evaluate_branch_aware_step` result dict. Field docs name the
/// corresponding dict key where it is not identical.
#[derive(Debug, Clone, PartialEq)]
pub struct BranchAwareStepResult {
    pub elapsed_ms: i64,
    /// `substep_accumulator_boundary`.
    pub boundary: Option<SubstepAccumulatorBoundaryResult>,
    /// `boost_ground_primary`.
    pub boost: BoosterVehicleStepResult,
    pub state18_feedback_selected: bool,
    pub state18_feedback_status: &'static str,
    pub pit_resource: Option<PitResourceStep>,
    pub bcharge_resource: Option<BchargeSurfaceStep>,
    /// `surface1_recovery_schedule`.
    pub recovery_schedule: RecoveryScheduleOutcome,
    /// `surface1_recovery_force`.
    pub recovery_force: Option<Surface1RecoveryForceResult>,
    pub surface1_recovery_force_status: &'static str,
    /// `ground_drive_force`.
    pub ground_force: Option<GroundDriveForceBodyResult>,
    pub ground_drive_force_status: &'static str,
    /// `low_projection_branch`.
    pub low_projection: Option<LowProjectionBranchResult>,
    /// `drift_window_branch`.
    pub drift_window: Option<DriftWindowBranchResult>,
    pub selected_ground_force_producer: Option<&'static str>,
    /// `ground_accumulator`.
    pub accumulator: Option<GroundAccumulatorSinkResult>,
    pub ground_accumulator_status: &'static str,
    /// `rail_accumulator`.
    pub rail: Option<RailAccumulatorProducersResult>,
    pub rail_selector: Option<RailSelectorOutcome>,
    pub rail_accumulator_status: &'static str,
    /// `contact_accumulator`.
    pub contact: Option<ContactAccumulatorProducerResult>,
    pub contact_accumulator_status: &'static str,
    /// `hard_contact_velocity` (with `selection_authority` and
    /// `scheduler_phase` attached, as Python does).
    pub hard_contact: Option<HardContactVelocityResponse>,
    pub hard_contact_selection_authority: Option<String>,
    pub hard_contact_scheduler_phase: Option<String>,
    /// `soft_contact_velocity`.
    pub soft_contact: SoftContactOutcome,
    pub soft_contact_scheduler_phase: Option<String>,
    /// `ground_correction`.
    pub correction: Option<GroundCorrectionPipelineResult>,
    pub ground_correction_status: &'static str,
    pub bs_zone: Option<BsZoneOverrideResult>,
    pub bs_zone_selection_authority: Option<String>,
    pub bs_zone_status: &'static str,
    pub mz_magnet: Option<MzMagnetTrigger>,
    pub mz_magnet_selection_authority: Option<String>,
    pub mz_magnet_status: &'static str,
    /// `jm_dj_trigger`.
    pub jm_dj: Option<JmDjTrigger>,
    pub jm_dj_selection_authority: Option<String>,
    pub jm_dj_trigger_status: &'static str,
    pub accumulator_614_after_bs: Option<[f64; 3]>,
    pub accumulator_620_after_bs: Option<[f64; 3]>,
    /// `accumulator_614_after_post_bs_reset` — note this reflects the
    /// float32 re-quantized pre-damping value when common damping runs in
    /// float32 mode, exactly like the Python variable reuse.
    pub accumulator_614_after_post_bs_reset: Option<[f64; 3]>,
    pub accumulator_620_after_post_bs_reset: Option<[f64; 3]>,
    /// `common_damping`.
    pub damping: Option<CommonDampingResult>,
    pub common_damping_status: &'static str,
    pub accumulator_614_after_damping: Option<[f64; 3]>,
    pub accumulator_620_after_damping: Option<[f64; 3]>,
    /// `post_bs_reset_constraint` (also `no_contact_fallback_constraint`).
    pub post_bs_reset: Option<PostBsResetConstraintResult>,
    /// `post_bs_reset_selector` (also `no_contact_fallback_selector`).
    pub post_bs_reset_selector: Option<PostBsResetSelectorOutcome>,
    pub post_bs_reset_constraint_status: &'static str,
    /// `speed_threshold_latch`.
    pub speed_threshold: Option<SpeedThresholdLatchResult>,
    /// `linear_integrator`.
    pub linear: Option<LinearVelocityIntegratorResult>,
    pub linear_integrator_status: &'static str,
    /// `angular_integrator`.
    pub angular: Option<AngularVelocityIntegratorResult>,
    pub angular_integrator_status: &'static str,
    /// `position_integrator`.
    pub position: Option<PositionIntegratorResult>,
    pub position_integrator_status: &'static str,
    /// `orientation_integrator`.
    pub orientation: Option<OrientationIntegratorResult>,
    pub orientation_integrator_status: &'static str,
    pub velocity_c0_after_pose_step: Option<[f64; 3]>,
    pub angular_velocity_cc_after_pose_step: Option<[f64; 3]>,
    /// `terminal_fall_reset`.
    pub fall_reset: Option<FallClampStateReplacement>,
    pub terminal_fall_reset_status: &'static str,
    /// `terminal_checkpoint_respawn`.
    pub checkpoint_respawn: Option<CheckpointRespawnStateReplacement>,
    pub terminal_checkpoint_respawn_status: &'static str,
    /// `checkpoint_target_selector_events`.
    pub checkpoint_target_events: Vec<CheckpointTargetSelectorResult>,
    /// `checkpoint_target_selector_final`.
    pub checkpoint_target_final: Option<CheckpointTargetSelectorResult>,
    pub position_90_final: Option<[f64; 3]>,
    pub matrix_9c_final: Option<[f64; 9]>,
    pub velocity_c0_final: Option<[f64; 3]>,
    pub angular_velocity_cc_final: Option<[f64; 3]>,
}

// ---------------------------------------------------------------------------
// The orchestrator, transliterated stage by stage from
// `evaluate_branch_aware_step` in evaluation order.
// ---------------------------------------------------------------------------

fn local_force_torque_low(result: &LowProjectionBranchResult) -> Option<([f64; 3], [f64; 3])> {
    match result {
        LowProjectionBranchResult::Selected(selected) => {
            Some((selected.local_force, selected.local_torque))
        }
        LowProjectionBranchResult::NotSelected { .. } => None,
    }
}

fn local_force_torque_drift(result: &DriftWindowBranchResult) -> Option<([f64; 3], [f64; 3])> {
    match result {
        DriftWindowBranchResult::Selected(selected) => {
            Some((selected.local_force, selected.local_torque))
        }
        DriftWindowBranchResult::NotSelected => None,
    }
}

/// `evaluate_branch_aware_step(**kwargs)` — evaluate the proven selector
/// order without merging unlike coordinate sinks.
///
/// Boost state expiry occurs in UpdateTick before the ground-force
/// selector. Surface-1 recovery selection occurs on entry to
/// ApplyGroundDriveForces and its timer expires only after that call's
/// force/torque submission.
#[allow(clippy::too_many_lines)]
pub fn evaluate_branch_aware_step(
    args: &BranchAwareStepArgs,
) -> Result<BranchAwareStepResult, String> {
    let elapsed_ms = args.elapsed_ms;
    let dt = elapsed_ms as f64 / 1000.0;

    // --- state18 feedback precedes the UpdateTick timer pass. -------------
    let mut boost_args = args.boost.clone();
    let mut state18_selected = false;
    let mut state18_status = "not_requested";
    if let Some(state18) = &args.state18_feedback {
        state18_selected = select_state18_feedback(
            state18.stroke_dispatch_selected,
            state18.control_lock_1c0,
            state18.event_active,
            state18.effect_feedback_enabled_258,
        );
        state18_status = if state18_selected {
            "selected_state18_1000ms"
        } else {
            "predicate_false"
        };
        if state18_selected {
            if boost_args.inventory_use.is_some() {
                return Err(
                    "simultaneous state18 feedback and booster inventory activation has no proven cross-event order"
                        .to_string(),
                );
            }
            // The event handler requests state18/+0x630=1000 before the next
            // UpdateTick timer pass.
            boost_args.state = 18;
            boost_args.remaining_ms = 1000;
        }
    }
    let inventory_input = boost_args
        .inventory_use
        .as_ref()
        .map(InventoryUseArgs::to_input);
    let boost_result = evaluate_vehicle_step(
        boost_args.state,
        boost_args.remaining_ms,
        elapsed_ms,
        boost_args.correction_dir,
        boost_args.direction_gain,
        boost_args.selected_slot,
        boost_args.attached_state,
        boost_args.e8c,
        boost_args.efc,
        boost_args.flag_5bc,
        boost_args.raw_144,
        boost_args.obf_808,
        boost_args.obf_814,
        boost_args.obf_69c,
        boost_args.obf_684,
        boost_args.obf_8ac,
        boost_args.obf_8e8,
        boost_args.obf_94c,
        inventory_input.as_ref(),
    )?;

    let pit_resource_result = match &args.pit_resource {
        Some(pit) => Some(evaluate_pit_resource_step(
            pit.active,
            pit.rate,
            pit.mode,
            pit.accum_before,
            dt,
        )?),
        None => None,
    };

    // --- mutually exclusive ApplyGroundDriveForces producer selection. ----
    let low_projection = match &args.recovery.low_projection {
        _ if args.recovery.disabled_b9c => None,
        Some(low) => Some(evaluate_low_projection_branch(
            low.projection_len,
            low.ground_forward_term,
            low.drive_direction,
            low.normalized_velocity_raw,
            low.correction_raw,
            low.mass_scale,
            low.base_longitudinal_gain,
            low.base_lateral_gain,
            low.axis_long.unwrap_or(0.5),
            low.axis_lat.unwrap_or(0.5),
            low.state_5c8.unwrap_or(0.0),
            low.b1c_charge_settlement.unwrap_or(false),
            low.b10.unwrap_or(0.0),
            low.b20.unwrap_or(0.0),
            low.b2c.unwrap_or(0.0),
            low.b38.unwrap_or(0.0),
            low.cap_760.unwrap_or(0.0),
            low.numeric_mode.as_deref().unwrap_or("float64"),
        )?),
        None => None,
    };
    let low_selected = matches!(
        &low_projection,
        Some(LowProjectionBranchResult::Selected(_))
    );

    let drift_window = match &args.recovery.drift_window {
        _ if args.recovery.disabled_b9c => None,
        Some(drift) if !low_selected => Some(evaluate_drift_window_branch(
            drift.selector_b08,
            drift.timer_b0c,
            dt,
            drift.configured_duration_734,
            drift.mass_scale,
            drift.base_longitudinal_gain,
            drift.signed_axis,
            drift.scaled_tune_660,
            drift.drift_gain_728,
            drift.tune_758,
            drift.axis_lat,
            drift.drive_raw_aa0,
            drift.debug_force_scale.unwrap_or(1.0),
        )),
        _ => None,
    };
    let drift_selected = matches!(&drift_window, Some(DriftWindowBranchResult::Selected(_)));

    let recovery_args = &args.recovery.schedule;
    let recovery_schedule = if args.recovery.disabled_b9c {
        RecoveryScheduleOutcome::Suppressed {
            reason: "disabled_b9c",
        }
    } else if low_selected {
        RecoveryScheduleOutcome::Suppressed {
            reason: "low_projection_precedes_recovery_selector",
        }
    } else if drift_selected {
        RecoveryScheduleOutcome::Suppressed {
            reason: "drift_window_precedes_recovery_selector",
        }
    } else {
        let schedule = recovery_args
            .as_ref()
            .ok_or_else(|| "missing recovery schedule parameters".to_string())?;
        RecoveryScheduleOutcome::Evaluated(evaluate_recovery_substep(
            schedule
                .surface_mode
                .ok_or_else(|| "missing recovery schedule surface_mode".to_string())?,
            schedule
                .selector_af4
                .ok_or_else(|| "missing recovery schedule selector_af4".to_string())?,
            schedule
                .field_af8
                .ok_or_else(|| "missing recovery schedule field_af8".to_string())?,
            schedule
                .recovery_timer
                .ok_or_else(|| "missing recovery schedule recovery_timer".to_string())?,
            schedule
                .other_recovery_gate
                .ok_or_else(|| "missing recovery schedule other_recovery_gate".to_string())?,
            dt,
        ))
    };

    let mut recovery_force = None;
    let mut force_status = if args.recovery.disabled_b9c {
        "disabled_b9c"
    } else if low_selected {
        "suppressed_by_low_projection"
    } else if drift_selected {
        "suppressed_by_drift_window"
    } else {
        "branch_not_selected"
    };
    if !args.recovery.disabled_b9c
        && !low_selected
        && !drift_selected
        && recovery_schedule.branch_for_force()
    {
        match &args.recovery.force {
            None => force_status = "selected_parameters_missing",
            Some(force_payload) => {
                let mut force_args = force_payload.clone();
                if let Some(schedule) = recovery_args {
                    if let (Some(snapshot), Some(supplied)) =
                        (schedule.surface_mode, force_args.surface_mode)
                    {
                        if snapshot != supplied {
                            return Err(
                                "recovery force surface_mode contradicts entry selector snapshot"
                                    .to_string(),
                            );
                        }
                    }
                    if schedule.surface_mode.is_some() {
                        force_args.surface_mode = schedule.surface_mode;
                    }
                    if let (Some(snapshot), Some(supplied)) =
                        (schedule.selector_af4, force_args.selector_af4)
                    {
                        if snapshot != supplied {
                            return Err(
                                "recovery force selector_af4 contradicts entry selector snapshot"
                                    .to_string(),
                            );
                        }
                    }
                    if schedule.selector_af4.is_some() {
                        force_args.selector_af4 = schedule.selector_af4;
                    }
                }
                recovery_force = Some(evaluate_surface1_recovery_force(
                    force_args
                        .selector_af4
                        .ok_or_else(|| "missing recovery force selector_af4".to_string())?,
                    force_args.speed_magnitude,
                    force_args.speed_profile_denominator,
                    force_args.blend_ab0,
                    force_args.blend_ab4,
                    force_args.active_ab0_rate,
                    force_args.active_ab4_rate,
                    force_args.timer_ab0_rate,
                    force_args.contact_basis_product,
                    force_args.ground_forward_term,
                    force_args.drive_direction,
                    force_args.normalized_velocity,
                    force_args.longitudinal_correction,
                    force_args.lateral_correction,
                    force_args.mass_scale,
                    force_args.base_longitudinal_gain,
                    force_args.base_lateral_gain,
                    force_args.ground_force_multiplier,
                    force_args.tune_758,
                    force_args.axis_long,
                    force_args.axis_lat,
                    force_args.active_surface.map(RecoverySurfaceProfile::from),
                    force_args.timer_surface.map(RecoverySurfaceProfile::from),
                    force_args.field_af0.unwrap_or(0.0),
                    force_args.torque_y_scale.unwrap_or(1.0),
                    force_args.surface_mode.unwrap_or(1),
                    force_args.non_surface_profile.map(NonSurfaceProfile::from),
                )?);
                force_status = "evaluated";
            }
        }
    }

    let mut ground_force = None;
    let mut ground_force_status = if args.recovery.disabled_b9c {
        "disabled_b9c"
    } else if low_selected {
        "suppressed_by_low_projection"
    } else if drift_selected {
        "suppressed_by_drift_window"
    } else if recovery_force.is_some() {
        "suppressed_by_surface1_recovery"
    } else {
        "parameters_missing"
    };
    if !args.recovery.disabled_b9c
        && !low_selected
        && !drift_selected
        && recovery_force.is_none()
        && !recovery_schedule.branch_for_force()
    {
        if let Some(ground_payload) = &args.recovery.ground_force {
            let mut ground_args = ground_payload.clone();
            if let Some(schedule) = recovery_args {
                if let (Some(snapshot), Some(supplied)) =
                    (schedule.surface_mode, ground_args.surface_mode)
                {
                    if snapshot != supplied {
                        return Err(
                            "ground force surface_mode contradicts entry selector snapshot"
                                .to_string(),
                        );
                    }
                }
                if schedule.surface_mode.is_some() {
                    ground_args.surface_mode = schedule.surface_mode;
                }
            }
            ground_force = Some(evaluate_ground_drive_force_body(
                ground_args
                    .surface_mode
                    .ok_or_else(|| "missing ground force surface_mode".to_string())?,
                ground_args.speed_magnitude,
                ground_args.ground_forward_term,
                ground_args.contact_basis_product,
                ground_args.drive_direction,
                ground_args.normalized_velocity,
                ground_args.longitudinal_correction,
                ground_args.lateral_correction,
                ground_args.mass_scale,
                ground_args.base_longitudinal_gain,
                ground_args.base_lateral_gain,
                ground_args.ground_force_multiplier,
                ground_args.settlement_multiplier,
                ground_args.tune_75c,
                ground_args.axis_long,
                ground_args.axis_lat,
                ground_args.surface_profile.map(GroundSurfaceProfile::from),
                ground_args.torque_y_scale.unwrap_or(1.0),
                ground_args.torque_y_gain,
                ground_args.state_5c8.unwrap_or(0.0),
                ground_args.b1c_charge_settlement.unwrap_or(false),
                ground_args.b10.unwrap_or(0.0),
                ground_args.b20.unwrap_or(0.0),
                ground_args.b2c.unwrap_or(0.0),
                ground_args.b38.unwrap_or(0.0),
                ground_args.cap_760.unwrap_or(0.0),
                ground_args.numeric_mode.as_deref().unwrap_or("float64"),
            )?);
            ground_force_status = "evaluated";
        }
    } else if force_status == "selected_parameters_missing" {
        ground_force_status = "suppressed_by_selected_active_recovery";
    }

    // UpdateTick calls B24DE0 after the ground/contact/correction family
    // for this substep and clears +0xECC at tick end.
    let bcharge_resource_result = args.bcharge_resource.as_ref().map(|bcharge| {
        evaluate_bcharge_surface_step(bcharge.active, bcharge.accum_b10, bcharge.raw_760)
    });

    let mut selected_force_producer = None;
    let mut selected_force_name = None;
    for (name, producer) in [
        (
            "low_projection",
            if low_selected {
                low_projection.as_ref().and_then(local_force_torque_low)
            } else {
                None
            },
        ),
        (
            "drift_window",
            if drift_selected {
                drift_window.as_ref().and_then(local_force_torque_drift)
            } else {
                None
            },
        ),
        (
            "active_recovery",
            recovery_force
                .as_ref()
                .map(|force| (force.local_force, force.local_torque)),
        ),
        (
            "ground_normal_surface",
            ground_force
                .as_ref()
                .map(|force| (force.local_force, force.local_torque)),
        ),
    ] {
        if producer.is_some() {
            selected_force_name = Some(name);
            selected_force_producer = producer;
            break;
        }
    }

    // --- active accumulator boundary transfer. ----------------------------
    let boundary_result = args.substep_boundary.as_ref().map(|boundary| {
        evaluate_substep_accumulator_boundary(
            boundary.pending_force_b74,
            boundary.pending_torque_b80,
            boundary.pending_b90,
            args.accumulator
                .as_ref()
                .map(|acc| acc.accumulator_614_before),
            args.accumulator
                .as_ref()
                .map(|acc| acc.accumulator_620_before),
        )
    });
    let initial_pair: Option<([f64; 3], [f64; 3])> = boundary_result
        .as_ref()
        .map(|boundary| {
            (
                boundary.accumulator_614_after,
                boundary.accumulator_620_after,
            )
        })
        .or_else(|| {
            args.accumulator
                .as_ref()
                .map(|acc| (acc.accumulator_614_before, acc.accumulator_620_before))
        });

    // --- rail writer producers (before contact/ground). -------------------
    let mut rail_result = None;
    let mut rail_selector_result = None;
    let mut rail_state_snapshot = None;
    let mut rail_status = "not_selected";
    if let Some(rail) = &args.rail {
        if args.accumulator.is_none() {
            rail_status = "accumulator_parameters_missing";
        } else {
            let enabled_kinds: Vec<i64>;
            if let Some(selector) = &rail.selector {
                match rail.enabled_kinds_source.as_deref() {
                    None | Some("exact_selector") => {}
                    _ => {
                        return Err(
                            "rail enabled_kinds_source contradicts exact selector mode".to_string()
                        );
                    }
                }
                rail_state_snapshot = Some(selector.rail_state_1cc);
                let post_timer_boost_family = boost_result.boost_family_for_force;
                if let Some(supplied) = selector.boost_family_active {
                    if supplied != post_timer_boost_family {
                        return Err(
                            "rail boost_family_active contradicts post-timer decoded state"
                                .to_string(),
                        );
                    }
                }
                let attached_snapshot = boost_args.attached_state;
                if let Some(supplied) = selector.attached_object_state {
                    if supplied != attached_snapshot {
                        return Err(
                            "rail attached_object_state contradicts ground-state snapshot"
                                .to_string(),
                        );
                    }
                }
                let selected = select_rail_accumulator_kinds(
                    selector.rail_contact,
                    selector.rail_state_1cc,
                    selector.gate_1c0_decoded,
                    selector.positive_a80,
                    selector.negative_a80,
                    post_timer_boost_family,
                    selector.boost_special_gate_9b8,
                    selector.boost_special_gate_6e4,
                    attached_snapshot,
                    selector.abs_ae0,
                    selector.adc,
                );
                if let Some(supplied) = &rail.enabled_kinds {
                    if *supplied != selected.enabled_kinds {
                        return Err(
                            "rail enabled_kinds contradict exact B29350 selector".to_string()
                        );
                    }
                }
                enabled_kinds = selected.enabled_kinds.clone();
                rail_selector_result = Some(RailSelectorOutcome::Exact(selected));
            } else {
                match rail.enabled_kinds_source.as_deref() {
                    Some("observed_probe_rows") => {}
                    _ => {
                        return Err(
                            "rail enabled_kinds require either raw selector inputs or enabled_kinds_source=observed_probe_rows"
                                .to_string(),
                        );
                    }
                }
                enabled_kinds = rail.enabled_kinds.clone().unwrap_or_default();
                rail_selector_result = Some(RailSelectorOutcome::Observed {
                    enabled_kinds: enabled_kinds.clone(),
                });
            }
            let (initial_force, initial_torque) =
                initial_pair.expect("accumulator presence implies initial pair");
            rail_result = Some(evaluate_rail_accumulator_producers(
                &enabled_kinds,
                initial_force,
                initial_torque,
                rail.rail_dir,
                rail.selected_a80,
                rail.accel_be0,
                rail.obf_6e8,
                rail.obf_808,
                rail.obf_814,
                rail.grip_720,
                rail.angular_velocity_cc,
                rail.field_604,
                rail.obf_660,
                rail.gravity_be8,
            )?);
            rail_status = "evaluated";
        }
    }
    let rail_pair = rail_result
        .as_ref()
        .map(|rail| (rail.accumulator_614_after, rail.accumulator_620_after));

    // --- four-slot contact producer plus the common 604*660 force. --------
    let mut contact_result = None;
    let mut contact_status = "not_selected";
    if let Some(contact) = &args.contact {
        if args.accumulator.is_none() {
            contact_status = "accumulator_parameters_missing";
        } else {
            let (before_force, before_torque) = rail_pair
                .or(initial_pair)
                .expect("accumulator presence implies initial pair");
            contact_result = Some(evaluate_contact_accumulator_producer(
                before_force,
                before_torque,
                contact.elapsed_seconds,
                contact.contact_active,
                contact.compression_a54,
                contact.compression_rate_a64,
                contact.contact_vectors_a14,
                contact.contact_normal_ac4,
                contact.spring_7c0,
                contact.rebound_7c4,
                contact.bump_7c8,
                contact.lever_x_9f4,
                contact.lever_y_9f8,
                contact.obf_66c,
                contact.obf_678,
                contact.field_604,
                contact.obf_660,
            )?);
            contact_status = "evaluated";
        }
    }
    let contact_pair = contact_result
        .as_ref()
        .map(|contact| (contact.accumulator_614_after, contact.accumulator_620_after));

    // --- direct wall writers: hard (eager) and soft (deferred). -----------
    let mut hard_contact_result = None;
    let mut hard_contact_authority = None;
    let mut hard_contact_scheduler_phase = None;
    if let Some(hard) = &args.hard_contact {
        match hard.selection_authority.as_deref() {
            Some("observed_probe_rows") | Some("collision_solver_dispatch") => {}
            _ => {
                return Err(
                    "hard_contact requires selection_authority=observed_probe_rows or collision_solver_dispatch"
                        .to_string(),
                );
            }
        }
        match hard.scheduler_phase.as_deref() {
            Some("pre_common_damping") | Some("post_linear_pre_pose") => {}
            _ => {
                return Err(
                    "hard_contact requires scheduler_phase=pre_common_damping or post_linear_pre_pose"
                        .to_string(),
                );
            }
        }
        hard_contact_authority = hard.selection_authority.clone();
        hard_contact_scheduler_phase = hard.scheduler_phase.clone();
        hard_contact_result = Some(evaluate_hard_contact_velocity_response(
            hard.velocity_c0_before,
            hard.hit_normal,
            hard.basis_ad0,
            hard.restitution.unwrap_or(0.7),
        ));
    }

    let mut soft_contact_outcome = SoftContactOutcome::NotSelected;
    let mut soft_contact_scheduler_phase: Option<String> = None;
    if let Some(soft) = &args.soft_contact {
        if args.hard_contact.is_some() {
            return Err(
                "hard_contact and soft_contact direct writers are mutually exclusive".to_string(),
            );
        }
        match soft.selection_authority.as_deref() {
            Some("observed_probe_rows") | Some("collision_solver_dispatch") => {}
            _ => {
                return Err(
                    "soft_contact requires observed_probe_rows or collision_solver_dispatch authority"
                        .to_string(),
                );
            }
        }
        match soft.scheduler_phase.as_deref() {
            Some("post_linear_pre_pose") => {}
            _ => {
                return Err(
                    "soft_contact currently requires scheduler_phase=post_linear_pre_pose"
                        .to_string(),
                );
            }
        }
        soft_contact_scheduler_phase = soft.scheduler_phase.clone();
        soft_contact_outcome = SoftContactOutcome::Pending {
            velocity_c0_before: soft.velocity_c0_before,
            velocity_delta: soft.velocity_delta,
            selection_authority: soft
                .selection_authority
                .clone()
                .expect("authority validated above"),
        };
    }

    // --- ordered ground correction primary/kind31..37 actions. ------------
    let mut correction_result = None;
    let mut correction_status = "not_selected";
    if let Some(correction) = &args.ground_correction {
        if args.accumulator.is_none() {
            correction_status = "accumulator_parameters_missing";
        } else {
            let (before_force, before_torque) = contact_pair
                .or(rail_pair)
                .or(initial_pair)
                .expect("accumulator presence implies initial pair");
            correction_result = Some(evaluate_ground_correction_pipeline(
                before_force,
                before_torque,
                correction.velocity_c0,
                correction.disabled_b9c,
                correction.selected_slot,
                correction.opposite_slot,
                correction.gate_1c0,
                correction.gate_6f3b00,
                correction.gate_6f3ae0,
                correction.field_adc,
                correction.timer_a80_before,
                dt,
                correction.field_ae0,
                correction.b8c,
                correction.lateral_dot,
                correction.correction_dir,
                correction.direction_gain,
                correction.state_mult,
                correction.flag_af4,
                correction.flag_afc,
                correction.sub_ad0,
                correction.sub_af0,
                correction.field_9f0,
                correction.sub_6f4,
                correction.sub_bc4,
                correction.drive_slot1_a90,
                correction.axis_ac4,
                correction.sub_720,
                correction.sub_724,
                correction.numeric_mode.as_deref().unwrap_or("float64"),
            ));
            correction_status = "evaluated";
        }
    }
    let correction_pair = correction_result.as_ref().map(|correction| {
        (
            correction.accumulator_614_after,
            correction.accumulator_620_after,
        )
    });

    // --- selected ground producer sinks into +0x614/+0x620. ---------------
    let mut accumulator_result = None;
    let mut accumulator_status = "parameters_missing";
    if selected_force_producer.is_none() {
        accumulator_status = "no_evaluated_ground_force_producer";
    } else if let (Some((local_force, local_torque)), Some(accumulator)) =
        (selected_force_producer, args.accumulator.as_ref())
    {
        let (before_force, before_torque) = correction_pair
            .or(contact_pair)
            .or(rail_pair)
            .or(initial_pair)
            .expect("accumulator presence implies initial pair");
        accumulator_result = Some(apply_ground_accumulator_sink(
            local_force,
            local_torque,
            accumulator.matrix_9c,
            before_force,
            before_torque,
            accumulator.numeric_mode.as_deref().unwrap_or("float64"),
        )?);
        accumulator_status = "evaluated";
    }
    let accumulator_pair = accumulator_result
        .as_ref()
        .map(|acc| (acc.accumulator_614_after, acc.accumulator_620_after));

    // --- authored trigger handlers (BS, then MZ, then JM/DJ). -------------
    let latest_pre_bs = accumulator_pair
        .or(correction_pair)
        .or(contact_pair)
        .or(rail_pair);
    let mut bs_zone_result = None;
    let mut bs_zone_authority = None;
    let mut bs_zone_status = "not_selected";
    if let Some(bs) = &args.bs_zone {
        if args.accumulator.is_none() {
            bs_zone_status = "accumulator_parameters_missing";
        } else {
            match bs.selection_authority.as_deref() {
                Some("authored_trigger_dispatch") | Some("observed_probe_rows") => {}
                _ => {
                    return Err(
                        "bs_zone requires selection_authority=authored_trigger_dispatch or observed_probe_rows"
                            .to_string(),
                    );
                }
            }
            let velocity_c0 = correction_result
                .as_ref()
                .map(|correction| correction.velocity_c0_after)
                .unwrap_or(bs.velocity_c0);
            let (before_force, before_torque) = latest_pre_bs
                .or(initial_pair)
                .expect("accumulator presence implies initial pair");
            bs_zone_result = Some(evaluate_bs_zone_override(
                bs.digit_x,
                bs.digit_y,
                bs.flag_64c,
                bs.field_190,
                velocity_c0,
                before_force,
                before_torque,
            )?);
            bs_zone_authority = bs.selection_authority.clone();
            bs_zone_status = "evaluated_replacement_barrier";
        }
    }

    let mut mz_magnet_result = None;
    let mut mz_magnet_authority = None;
    let mut mz_magnet_status = "not_selected";
    if let Some(mz) = &args.mz_magnet {
        match mz.selection_authority.as_deref() {
            Some("authored_trigger_dispatch") | Some("observed_probe_rows") => {}
            _ => {
                return Err(
                    "mz_magnet requires selection_authority=authored_trigger_dispatch or observed_probe_rows"
                        .to_string(),
                );
            }
        }
        if bs_zone_result.is_some() {
            return Err(
                "MZ and BS authored token handlers are mutually exclusive in one substep"
                    .to_string(),
            );
        }
        if hard_contact_scheduler_phase.as_deref() == Some("pre_common_damping") {
            return Err(
                "MZ and JM/DJ authored token velocity writers are mutually exclusive".to_string(),
            );
        }
        let special_active = (13..17).contains(&boost_result.state_for_force);
        if let Some(supplied) = mz.special_state_family_already_active {
            if supplied != special_active {
                return Err(
                    "MZ special-family gate contradicts the post-timer decoded state".to_string(),
                );
            }
        }
        let result = evaluate_mz_magnet_trigger(
            mz.current_reference_xyz,
            mz.authored_target_xyz,
            mz.velocity_scale,
            special_active,
        );
        mz_magnet_status = if result.accepted {
            "evaluated_velocity_and_state16"
        } else {
            "evaluated_velocity_state_gate_rejected"
        };
        mz_magnet_authority = mz.selection_authority.clone();
        mz_magnet_result = Some(result);
    }

    let mut jm_dj_result = None;
    let mut jm_dj_authority = None;
    let mut jm_dj_status = "not_selected";
    if let Some(jm_dj) = &args.jm_dj_trigger {
        match jm_dj.selection_authority.as_deref() {
            Some("authored_trigger_dispatch") | Some("observed_probe_rows") => {}
            _ => {
                return Err(
                    "jm_dj_trigger requires selection_authority=authored_trigger_dispatch or observed_probe_rows"
                        .to_string(),
                );
            }
        }
        if bs_zone_result.is_some() || mz_magnet_result.is_some() {
            return Err(
                "JM/DJ, MZ, and BS authored token handlers are mutually exclusive".to_string(),
            );
        }
        if hard_contact_scheduler_phase.as_deref() == Some("pre_common_damping") {
            return Err(
                "explicit JM/DJ trigger cannot be combined with generic pre_common_damping replay"
                    .to_string(),
            );
        }
        let special_active = (13..17).contains(&boost_result.state_for_force);
        if let Some(supplied) = jm_dj.special_state_family_already_active {
            if supplied != special_active {
                return Err(
                    "JM/DJ special-family gate contradicts the post-timer decoded state"
                        .to_string(),
                );
            }
        }
        let mut velocity_c0_before = jm_dj.velocity_c0_before;
        if jm_dj.variant == "JM" {
            match &args.common_damping {
                Some(damping) if damping.velocity_c0.is_some() => {
                    velocity_c0_before = damping.velocity_c0;
                }
                _ => {
                    return Err("JM requires the current pre-damping velocity_c0".to_string());
                }
            }
        }
        let result = evaluate_jm_dj_trigger(
            &jm_dj.variant,
            special_active,
            velocity_c0_before,
            jm_dj.jm_basis_190,
            jm_dj.jm_horizontal_param,
            jm_dj.jm_basis_param,
            jm_dj.dj_vector,
            jm_dj.boost_multiplier_64c.unwrap_or(false),
        )?;
        jm_dj_status = if result.accepted {
            "evaluated_velocity_and_state"
        } else {
            "evaluated_velocity_state_gate_rejected"
        };
        jm_dj_authority = jm_dj.selection_authority.clone();
        jm_dj_result = Some(result);
    }

    // --- no-contact fallback B2D3C0 on the opposite branch. ---------------
    let post_bs_pair = bs_zone_result
        .as_ref()
        .map(|bs| (bs.accumulator_614_after, bs.accumulator_620_after))
        .or(latest_pre_bs)
        .or(initial_pair);
    let mut post_bs_reset_result = None;
    let mut post_bs_reset_selector_result = None;
    let mut post_bs_reset_status = "not_selected";
    if let Some(post_bs) = &args.post_bs_reset {
        match post_bs_pair {
            None => post_bs_reset_status = "accumulator_parameters_missing",
            Some((before_force, before_torque)) => {
                let selected: bool;
                if let Some(selector) = &post_bs.selector {
                    match post_bs.selection_authority.as_deref() {
                        None | Some("exact_selector") => {}
                        _ => {
                            return Err(
                                "post_bs_reset selection authority contradicts exact selector"
                                    .to_string(),
                            );
                        }
                    }
                    let mut rail_state_1cc = selector.rail_state_1cc;
                    if let Some(snapshot) = rail_state_snapshot {
                        if let Some(supplied) = rail_state_1cc {
                            if supplied != snapshot {
                                return Err(
                                "no-contact fallback rail_state_1cc contradicts Rail selector snapshot"
                                    .to_string(),
                            );
                            }
                        }
                        rail_state_1cc = Some(snapshot);
                    }
                    let selector_result = select_post_bs_reset_constraint(
                        selector.current_contact_updated,
                        selector.contact_trigger_e38,
                        selector.contact_trigger_e4c,
                        rail_state_1cc.ok_or_else(|| {
                            "missing post_bs_reset selector rail_state_1cc".to_string()
                        })?,
                    );
                    selected = selector_result.selected;
                    post_bs_reset_selector_result =
                        Some(PostBsResetSelectorOutcome::Exact(selector_result));
                } else if post_bs.selection_authority.as_deref() == Some("observed_probe_rows") {
                    selected = true;
                    post_bs_reset_selector_result =
                        Some(PostBsResetSelectorOutcome::ObservedProbeRows);
                } else {
                    return Err(
                    "post_bs_reset requires exact selector inputs or selection_authority=observed_probe_rows"
                        .to_string(),
                );
                }
                if selected {
                    if bs_zone_result.is_some() {
                        return Err(
                        "BS trigger and no-contact fallback B2D3C0 are mutually exclusive in one substep"
                            .to_string(),
                    );
                    }
                    if mz_magnet_result.is_some() {
                        return Err(
                        "MZ contact trigger and no-contact fallback B2D3C0 are mutually exclusive in one substep"
                            .to_string(),
                    );
                    }
                    if jm_dj_result.is_some() {
                        return Err(
                        "JM/DJ contact trigger and no-contact fallback B2D3C0 are mutually exclusive in one substep"
                            .to_string(),
                    );
                    }
                    if rail_result
                        .as_ref()
                        .is_some_and(|rail| !rail.events.is_empty())
                    {
                        return Err(
                        "Rail writer events and no-contact fallback B2D3C0 are mutually exclusive in one substep"
                            .to_string(),
                    );
                    }
                    post_bs_reset_result = Some(evaluate_post_bs_reset_constraint(
                        before_force,
                        before_torque,
                        post_bs
                            .vector_604
                            .ok_or_else(|| "missing post_bs_reset vector_604".to_string())?,
                        post_bs
                            .sub_660
                            .ok_or_else(|| "missing post_bs_reset sub_660".to_string())?,
                        post_bs
                            .divisor_e50
                            .ok_or_else(|| "missing post_bs_reset divisor_e50".to_string())?,
                        post_bs.angular_velocity_cc.ok_or_else(|| {
                            "missing post_bs_reset angular_velocity_cc".to_string()
                        })?,
                        post_bs.special_orientation_gate.unwrap_or(false),
                        post_bs.field_acc.unwrap_or(0.0),
                        post_bs.field_ac0.unwrap_or(0.0),
                        post_bs.field_ad8.unwrap_or(0.0),
                    )?);
                    post_bs_reset_status = "evaluated";
                } else {
                    post_bs_reset_status = "predicate_false";
                }
            }
        }
    }
    let post_bs_reset_pair = post_bs_reset_result
        .as_ref()
        .map(|result| (result.accumulator_614_after, result.accumulator_620_after));

    // --- common damping; the pre-damping direct writer wins velocity. -----
    let mut pre_damping_pair = post_bs_reset_pair.or(post_bs_pair);
    let pre_common_velocity: Option<[f64; 3]> =
        if hard_contact_scheduler_phase.as_deref() == Some("pre_common_damping") {
            hard_contact_result
                .as_ref()
                .map(|hard| hard.velocity_c0_after)
        } else {
            jm_dj_result
                .as_ref()
                .map(|jm| jm.velocity_c0_after)
                .or_else(|| mz_magnet_result.as_ref().map(|mz| mz.velocity_c0_after))
                .or_else(|| {
                    correction_result
                        .as_ref()
                        .map(|correction| correction.velocity_c0_after)
                })
        };
    let mut damping_result = None;
    let mut damping_status = "not_selected";
    if let Some(damping) = &args.common_damping {
        match pre_damping_pair {
            None => damping_status = "accumulator_parameters_missing",
            Some((mut before_force, mut before_torque)) => {
                if damping.numeric_mode.as_deref() == Some("float32") {
                    // Every producer commits through a float-vector sink before
                    // B2D5D0 reads +0x614/+0x620.  Preserve that boundary instead
                    // of carrying Python-double intermediates into damping.
                    before_force = [
                        f32q(before_force[0]),
                        f32q(before_force[1]),
                        f32q(before_force[2]),
                    ];
                    before_torque = [
                        f32q(before_torque[0]),
                        f32q(before_torque[1]),
                        f32q(before_torque[2]),
                    ];
                    pre_damping_pair = Some((before_force, before_torque));
                }
                let velocity_c0 = pre_common_velocity
                    .or(damping.velocity_c0)
                    .ok_or_else(|| "missing common damping velocity_c0".to_string())?;
                damping_result = Some(evaluate_common_damping(
                    before_force,
                    before_torque,
                    velocity_c0,
                    damping.angular_velocity_cc,
                    damping.damping_700,
                    damping.special_gate,
                    damping.directional_70c.unwrap_or(0.0),
                    damping.runtime_ba0.unwrap_or(0.0),
                    damping.token_multiplier,
                    damping.numeric_mode.as_deref().unwrap_or("float64"),
                )?);
                damping_status = "evaluated";
            }
        }
    }
    let final_pair = damping_result
        .as_ref()
        .map(|damping| (damping.accumulator_614_after, damping.accumulator_620_after))
        .or(pre_damping_pair);

    let speed_threshold_result = match &args.speed_threshold {
        Some(threshold) => Some(evaluate_speed_threshold_latch(
            threshold.threshold_b8c,
            pre_common_velocity
                .or(threshold.velocity_c0)
                .ok_or_else(|| "missing speed threshold velocity_c0".to_string())?,
        )),
        None => None,
    };

    // --- integrators. ------------------------------------------------------
    let mut linear_result = None;
    let mut linear_status = "parameters_missing";
    if let Some(linear) = &args.linear_integrator {
        match final_pair {
            None => linear_status = "force_accumulator_missing",
            Some((final_force, _)) => {
                let velocity_c0_before = pre_common_velocity
                    .or(linear.velocity_c0_before)
                    .ok_or_else(|| "missing linear integrator velocity_c0_before".to_string())?;
                let accumulator_614_before = linear.accumulator_614_override.unwrap_or(final_force);
                linear_result = Some(evaluate_linear_velocity_integrator(
                    velocity_c0_before,
                    accumulator_614_before,
                    linear.bac_force,
                    linear.sub_660,
                    dt,
                    linear.post_bs_later_delta,
                    linear.numeric_mode.as_deref().unwrap_or("float64"),
                )?);
                linear_status = "evaluated";
            }
        }
    }

    let mut angular_result = None;
    let mut angular_status = "parameters_missing";
    if let Some(angular) = &args.angular_integrator {
        match final_pair {
            None => angular_status = "torque_accumulator_missing",
            Some((_, final_torque)) => {
                let accumulator_620 = angular.accumulator_620_override.unwrap_or(final_torque);
                angular_result = Some(evaluate_angular_velocity_integrator(
                    angular.angular_velocity_cc_before,
                    accumulator_620,
                    angular.matrix_9c4,
                    dt,
                    angular.numeric_mode.as_deref().unwrap_or("float64"),
                )?);
                angular_status = "evaluated";
            }
        }
    }

    let mut position_result = None;
    let mut position_status = "parameters_missing";
    if let Some(position) = &args.position_integrator {
        let velocity_c0_before: Option<[f64; 3]>;
        if soft_contact_scheduler_phase.as_deref() == Some("post_linear_pre_pose") {
            let (pending_velocity_delta, authority) = match &soft_contact_outcome {
                SoftContactOutcome::Pending {
                    velocity_delta,
                    selection_authority,
                    ..
                } => (*velocity_delta, selection_authority.clone()),
                _ => unreachable!("soft phase validated above"),
            };
            let soft_velocity = linear_result
                .as_ref()
                .map(|linear| linear.velocity_c0_after)
                .or(pre_common_velocity)
                .ok_or_else(|| "missing soft contact velocity_c0_before".to_string())?;
            let evaluated =
                evaluate_soft_contact_velocity_response(soft_velocity, pending_velocity_delta);
            velocity_c0_before = Some(evaluated.velocity_c0_after);
            soft_contact_outcome = SoftContactOutcome::Evaluated {
                result: evaluated,
                selection_authority: authority,
                scheduler_phase: soft_contact_scheduler_phase
                    .clone()
                    .expect("phase validated above"),
            };
        } else if hard_contact_scheduler_phase.as_deref() == Some("post_linear_pre_pose") {
            velocity_c0_before = hard_contact_result
                .as_ref()
                .map(|hard| hard.velocity_c0_after);
        } else if let Some(linear) = &linear_result {
            velocity_c0_before = Some(linear.velocity_c0_after);
        } else if pre_common_velocity.is_some() {
            velocity_c0_before = pre_common_velocity;
        } else {
            velocity_c0_before = position.velocity_c0_before;
        }
        match velocity_c0_before {
            None => position_status = "velocity_missing",
            Some(velocity) => {
                position_result = Some(evaluate_position_integrator(
                    position.position_90_before,
                    velocity,
                    dt,
                    position.pre_position_velocity_delta,
                    position.state_1d0.unwrap_or(0),
                    position.latch_658.unwrap_or(false),
                    position.obf_b44.unwrap_or(0.0),
                    position.axis_ab8,
                    position.axis_a48,
                )?);
                position_status = "evaluated";
            }
        }
    }

    let mut orientation_result = None;
    let mut orientation_status = "parameters_missing";
    if let Some(orientation) = &args.orientation_integrator {
        let angular_velocity = angular_result
            .as_ref()
            .map(|angular| angular.angular_velocity_cc_after)
            .or(orientation.angular_velocity_cc_before);
        match angular_velocity {
            None => orientation_status = "angular_velocity_missing",
            Some(omega) => {
                orientation_result = Some(evaluate_orientation_integrator(
                    orientation.matrix_9c_before,
                    omega,
                    dt,
                    orientation.pre_clear_xy.unwrap_or(false),
                    orientation.stabilization_disabled_b70.unwrap_or(false),
                    orientation.rail_state_1cc.unwrap_or(0),
                    orientation.runtime_float32_exact.unwrap_or(false),
                )?);
                orientation_status = "evaluated";
            }
        }
    }

    // --- post-pose projections. --------------------------------------------
    let velocity_after_pose_step = position_result
        .as_ref()
        .map(|position| position.velocity_c0_for_position)
        .or_else(|| soft_contact_outcome.evaluated_velocity())
        .or_else(|| {
            if hard_contact_scheduler_phase.as_deref() == Some("post_linear_pre_pose") {
                hard_contact_result
                    .as_ref()
                    .map(|hard| hard.velocity_c0_after)
            } else {
                None
            }
        })
        .or_else(|| {
            linear_result
                .as_ref()
                .map(|linear| linear.velocity_c0_after)
        })
        .or(pre_common_velocity);
    let angular_velocity_after_pose_step = orientation_result
        .as_ref()
        .map(|orientation| orientation.angular_velocity_cc_after)
        .or_else(|| {
            angular_result
                .as_ref()
                .map(|angular| angular.angular_velocity_cc_after)
        });
    let position_after_pose_step = position_result
        .as_ref()
        .map(|position| position.position_90_after);
    let matrix_after_pose_step = orientation_result
        .as_ref()
        .map(|orientation| orientation.matrix_9c_after);

    // --- terminal fall clamp. ----------------------------------------------
    let mut fall_reset_result = None;
    let mut fall_reset_status = "not_selected";
    if let Some(fall) = &args.terminal_fall_reset {
        let position_90 = position_after_pose_step.or(fall.position_90_before);
        let matrix_9c = matrix_after_pose_step.or(fall.matrix_9c_before);
        let velocity_c0 = velocity_after_pose_step.or(fall.velocity_c0_before);
        let angular_velocity_cc =
            angular_velocity_after_pose_step.or(fall.angular_velocity_cc_before);
        if let (Some(position), Some(matrix), Some(velocity), Some(angular_velocity)) =
            (position_90, matrix_9c, velocity_c0, angular_velocity_cc)
        {
            let result = evaluate_fall_clamp_state_replacement(
                position,
                matrix,
                velocity,
                angular_velocity,
                fall.drop_threshold.unwrap_or(-5.0),
                fall.clamp_z.unwrap_or(0.0),
            );
            fall_reset_status = if result.selected {
                "selected_terminal_replacement"
            } else {
                "predicate_false"
            };
            fall_reset_result = Some(result);
        } else {
            fall_reset_status = "state_parameters_missing";
        }
    }
    let after_fall_position = fall_reset_result
        .as_ref()
        .map(|fall| fall.position_90_after)
        .or(position_after_pose_step);
    let after_fall_matrix = fall_reset_result
        .as_ref()
        .map(|fall| fall.matrix_9c_after)
        .or(matrix_after_pose_step);
    let after_fall_velocity = fall_reset_result
        .as_ref()
        .map(|fall| fall.velocity_c0_after)
        .or(velocity_after_pose_step);
    let after_fall_angular_velocity = fall_reset_result
        .as_ref()
        .map(|fall| fall.angular_velocity_cc_after)
        .or(angular_velocity_after_pose_step);

    // --- ordered checkpoint target writers. --------------------------------
    let mut checkpoint_target_results = Vec::new();
    let mut checkpoint_target_state = None;
    if let Some(selector) = &args.checkpoint_target_selector {
        let mut target = selector.target_before;
        let mut gate = selector.gate_before;
        for event in selector.events.as_deref().unwrap_or(&[]) {
            let result = evaluate_checkpoint_target_selector(
                &selector.mode_family,
                target,
                gate,
                &event.event,
                event.current_kart_position,
                event.config_gate.unwrap_or(false),
                event.mode_state,
                event.mission_index,
                event.mission_count,
                event.retire_detected.unwrap_or(false),
                event.reset_suppressed.unwrap_or(false),
                event.trace_value,
                event.trace_min,
                event.trace_max,
                event.peer_present.unwrap_or(false),
                event.kart_event_accepts.unwrap_or(false),
                event.stage_input_blocked.unwrap_or(false),
                event.tick_ms,
                event.gate_armed_tick_ms,
            )?;
            target = result.target_after;
            gate = result.gate_after;
            checkpoint_target_state = Some(result.clone());
            checkpoint_target_results.push(result);
        }
    }

    // --- terminal checkpoint respawn replacement. --------------------------
    let mut checkpoint_result = None;
    let mut checkpoint_status = "not_selected";
    if let Some(checkpoint) = &args.terminal_checkpoint_respawn {
        let mut mode_family = checkpoint.mode_family.clone();
        let mut checkpoint_gate = checkpoint.checkpoint_gate;
        let mut target_position = checkpoint.target_position;
        if let Some(state) = &checkpoint_target_state {
            mode_family = Some(state.mode_family.clone());
            checkpoint_gate = Some(state.gate_after);
            target_position = Some(state.target_after);
        }
        let velocity_c0 = after_fall_velocity.or(checkpoint.velocity_c0_before);
        let angular_velocity_cc =
            after_fall_angular_velocity.or(checkpoint.angular_velocity_cc_before);
        if let (
            Some(mode_family),
            Some(gate),
            Some(sample_result_zero),
            Some(target),
            Some(course_reference),
            Some(velocity),
            Some(angular_velocity),
        ) = (
            mode_family,
            checkpoint_gate,
            checkpoint.sample_result_zero,
            target_position,
            checkpoint.course_reference_position,
            velocity_c0,
            angular_velocity_cc,
        ) {
            let result = evaluate_checkpoint_respawn_state_replacement(
                &mode_family,
                gate,
                sample_result_zero,
                target,
                course_reference,
                velocity,
                angular_velocity,
            )?;
            checkpoint_status = if result.selected {
                "selected_terminal_replacement"
            } else {
                "predicate_false"
            };
            checkpoint_result = Some(result);
        } else {
            checkpoint_status = "state_parameters_missing";
        }
    }

    let checkpoint_selected = checkpoint_result
        .as_ref()
        .is_some_and(|result| result.selected);
    let final_position_90 = if checkpoint_selected {
        checkpoint_result
            .as_ref()
            .and_then(|result| result.position_90_after)
    } else {
        after_fall_position
    };
    let final_matrix_9c = if checkpoint_selected {
        checkpoint_result
            .as_ref()
            .and_then(|result| result.matrix_9c_after)
    } else {
        after_fall_matrix
    };
    let final_velocity_c0 = if checkpoint_selected {
        checkpoint_result
            .as_ref()
            .map(|result| result.velocity_c0_after)
    } else {
        after_fall_velocity
    };
    let final_angular_velocity_cc = if checkpoint_selected {
        checkpoint_result
            .as_ref()
            .map(|result| result.angular_velocity_cc_after)
    } else {
        after_fall_angular_velocity
    };

    let pre_damping_force = pre_damping_pair.map(|pair| pair.0);
    let pre_damping_torque = pre_damping_pair.map(|pair| pair.1);
    let final_force_accumulator = final_pair.map(|pair| pair.0);
    let final_torque_accumulator = final_pair.map(|pair| pair.1);

    Ok(BranchAwareStepResult {
        elapsed_ms,
        boundary: boundary_result,
        boost: boost_result,
        state18_feedback_selected: state18_selected,
        state18_feedback_status: state18_status,
        pit_resource: pit_resource_result,
        bcharge_resource: bcharge_resource_result,
        recovery_schedule,
        recovery_force,
        surface1_recovery_force_status: force_status,
        ground_force,
        ground_drive_force_status: ground_force_status,
        low_projection,
        drift_window,
        selected_ground_force_producer: selected_force_name,
        accumulator: accumulator_result,
        ground_accumulator_status: accumulator_status,
        rail: rail_result,
        rail_selector: rail_selector_result,
        rail_accumulator_status: rail_status,
        contact: contact_result,
        contact_accumulator_status: contact_status,
        hard_contact: hard_contact_result,
        hard_contact_selection_authority: hard_contact_authority,
        hard_contact_scheduler_phase,
        soft_contact: soft_contact_outcome,
        soft_contact_scheduler_phase,
        correction: correction_result,
        ground_correction_status: correction_status,
        bs_zone: bs_zone_result,
        bs_zone_selection_authority: bs_zone_authority,
        bs_zone_status,
        mz_magnet: mz_magnet_result,
        mz_magnet_selection_authority: mz_magnet_authority,
        mz_magnet_status,
        jm_dj: jm_dj_result,
        jm_dj_selection_authority: jm_dj_authority,
        jm_dj_trigger_status: jm_dj_status,
        accumulator_614_after_bs: post_bs_pair.map(|pair| pair.0),
        accumulator_620_after_bs: post_bs_pair.map(|pair| pair.1),
        accumulator_614_after_post_bs_reset: pre_damping_force,
        accumulator_620_after_post_bs_reset: pre_damping_torque,
        damping: damping_result,
        common_damping_status: damping_status,
        accumulator_614_after_damping: final_force_accumulator,
        accumulator_620_after_damping: final_torque_accumulator,
        post_bs_reset: post_bs_reset_result,
        post_bs_reset_selector: post_bs_reset_selector_result,
        post_bs_reset_constraint_status: post_bs_reset_status,
        speed_threshold: speed_threshold_result,
        linear: linear_result,
        linear_integrator_status: linear_status,
        angular: angular_result,
        angular_integrator_status: angular_status,
        position: position_result,
        position_integrator_status: position_status,
        orientation: orientation_result,
        orientation_integrator_status: orientation_status,
        velocity_c0_after_pose_step: velocity_after_pose_step,
        angular_velocity_cc_after_pose_step: angular_velocity_after_pose_step,
        fall_reset: fall_reset_result,
        terminal_fall_reset_status: fall_reset_status,
        checkpoint_respawn: checkpoint_result,
        terminal_checkpoint_respawn_status: checkpoint_status,
        checkpoint_target_events: checkpoint_target_results,
        checkpoint_target_final: checkpoint_target_state,
        position_90_final: final_position_90,
        matrix_9c_final: final_matrix_9c,
        velocity_c0_final: final_velocity_c0,
        angular_velocity_cc_final: final_angular_velocity_cc,
    })
}
