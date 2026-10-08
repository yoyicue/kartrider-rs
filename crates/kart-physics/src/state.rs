//! Vehicle state (port of `VehicleState` in `scripts/kartrider_vehicle_engine.py`).
//!
//! Authority: `scripts/kartrider_vehicle_engine.py:207-451`. Every cross-substep
//! latch lives here (see docs/kart-physics-rust-handoff-2026-07-28.md §3/§6):
//! rollback/snapshot must carry all of them, so the future `Engine` struct may
//! hold only state + params + prediction history.
//!
//! Type mapping from the Python dataclass:
//! - float vectors -> `[f64; 3]`, matrices -> `[f64; 9]`,
//!   `contact_compression` -> `[f64; 4]` (length checks are type-enforced);
//! - int fields -> `i64`, except semantically-uint32 fields
//!   (`last_network_sequence`, `scheduler_previous_tick_ms` -> `Option<u32>`)
//!   and the monotonic clocks (`time_ms`, `step_index` -> `u64`);
//! - `int | None` / `float | None` / `str | None` / `dict | None` -> `Option<_>`;
//! - `topology` -> `crate::topology::TopologyState`;
//! - dict-typed latches -> dedicated structs (see their doc comments).
//!
//! With the `serde` feature the JSON shape matches `dataclasses.asdict`
//! exactly: snake_case field names, `None` <-> `null`, `MotionDomain` as its
//! persisted string value.

use std::collections::BTreeMap;

use crate::domains::booster_resource::{
    empty_current_manager_slots, validate_current_manager_slots, InventorySlot,
};
use crate::topology::{MotionDomain, TopologyState};

/// `DEFAULT_COLLISION_ARG_THRESHOLD` (kartrider_vehicle_engine.py:85).
pub const DEFAULT_COLLISION_ARG_THRESHOLD: f64 = 0.65;

/// Minimal JSON value for genuinely open-ended world payloads (e.g. the
/// decoded rail selector carried inside `rail_capture_latch`). Numbers are
/// `f64` per the crate numeric discipline; Python `int`/`float` both map to
/// `Number`.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum JsonValue {
    #[default]
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

#[cfg(feature = "serde")]
impl serde::Serialize for JsonValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            JsonValue::Null => serializer.serialize_none(),
            JsonValue::Bool(value) => serializer.serialize_bool(*value),
            JsonValue::Number(value) => serializer.serialize_f64(*value),
            JsonValue::String(value) => serializer.serialize_str(value),
            JsonValue::Array(values) => {
                use serde::ser::SerializeSeq;
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    seq.serialize_element(value)?;
                }
                seq.end()
            }
            JsonValue::Object(entries) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for JsonValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct JsonValueVisitor;

        impl<'de> serde::de::Visitor<'de> for JsonValueVisitor {
            type Value = JsonValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_bool<E>(self, value: bool) -> Result<JsonValue, E> {
                Ok(JsonValue::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<JsonValue, E> {
                Ok(JsonValue::Number(value as f64))
            }

            fn visit_u64<E>(self, value: u64) -> Result<JsonValue, E> {
                Ok(JsonValue::Number(value as f64))
            }

            fn visit_f64<E>(self, value: f64) -> Result<JsonValue, E> {
                Ok(JsonValue::Number(value))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<JsonValue, E> {
                Ok(JsonValue::String(value.to_string()))
            }

            fn visit_string<E>(self, value: String) -> Result<JsonValue, E> {
                Ok(JsonValue::String(value))
            }

            fn visit_none<E>(self) -> Result<JsonValue, E> {
                Ok(JsonValue::Null)
            }

            fn visit_unit<E>(self) -> Result<JsonValue, E> {
                Ok(JsonValue::Null)
            }

            fn visit_some<D: serde::Deserializer<'de>>(
                self,
                deserializer: D,
            ) -> Result<JsonValue, D::Error> {
                serde::Deserialize::deserialize(deserializer)
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<JsonValue, A::Error> {
                let mut values = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(value) = seq.next_element()? {
                    values.push(value);
                }
                Ok(JsonValue::Array(values))
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<JsonValue, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((key, value)) = map.next_entry::<String, JsonValue>()? {
                    entries.insert(key, value);
                }
                Ok(JsonValue::Object(entries))
            }
        }

        deserializer.deserialize_any(JsonValueVisitor)
    }
}

/// `rail_selector_runtime["path_context_id"]`: the retained registered rail
/// path context. Runtime evidence carries both integer context indices
/// (`{"path_context_id": 32}`, see test_motion_topology.py) and string
/// context ids.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum RailPathContextId {
    Index(i64),
    Name(String),
}

/// `rail_selector_runtime` latch (`dict | None` in Python).
///
/// The Python field is an open mapping whose values must be numeric, except
/// `path_context_id` which may be a string or an int
/// (kartrider_vehicle_engine.py:386-394). The engine reads/writes exactly
/// these keys:
/// - `alignment_primary_remaining_ms` / `alignment_stable_remaining_ms`
///   (state1->2 alignment timers, engine :4015-4022);
/// - `path_context_id` (retained registered query context, :3044, :3399,
///   :4026, :4038);
/// - `state6_contact_remaining_ms` (state6 C60 confirmation countdown,
///   :1891, :1965, :3235, :3256);
/// - `state6_contact_candidate_latched` (substep-frequency hit latched for
///   the frame-owned countdown, :1885-1888);
/// - `state6_zero_boundary_release_delay_substeps` (float32 zero-boundary
///   writer phase, :1876, :1930, :1954);
/// - `state6_entry_force_pending_{x,y,z}` / `state6_entry_force_delay_substeps`
///   / `state6_entry_common_tail_delay_substeps` (state6 entry-force staging,
///   :2033-2148, :3278-3287).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RailSelectorRuntime {
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alignment_primary_remaining_ms: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub alignment_stable_remaining_ms: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub path_context_id: Option<RailPathContextId>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_contact_remaining_ms: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_contact_candidate_latched: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_zero_boundary_release_delay_substeps: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_entry_force_pending_x: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_entry_force_pending_y: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_entry_force_pending_z: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_entry_force_delay_substeps: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub state6_entry_common_tail_delay_substeps: Option<f64>,
}

/// `rail_capture_latch` latch (`dict | None` in Python).
///
/// Built by `inspect_autonomous_rail_capture` (engine :3753-3791) and stored
/// verbatim (`copy.deepcopy`, engine :6327, :6835) between the substep-frequency
/// entry-plane sample and the frame-owned selector commit. Keys produced by the
/// engine: `contact_active` (bool), `rail_state` (int, only when active),
/// `selector` (open-ended `world.query_rail_contact` payload, decoded later by
/// `_decode_active_rail_runtime`), `authority` (str).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RailCaptureLatch {
    pub contact_active: bool,
    pub rail_state: Option<i64>,
    /// Open-ended decoded selector payload; `VehicleState.validate` requires
    /// it to be a mapping, which the type enforces.
    pub selector: BTreeMap<String, JsonValue>,
    /// Producer-written keys (`rail_state`, `authority`) are always
    /// serialized so engine-produced latches round-trip; they deserialize as
    /// `None` when absent.
    pub authority: Option<String>,
}

/// `moving_contact_anchor` latch (`dict | None` in Python).
///
/// The seven required fields are the exact key set checked by
/// `VehicleState.validate` (kartrider_vehicle_engine.py:319-332). The producer
/// `acquire_moving_contact_frame` (scripts/evaluate_moving_contact_topology.py:153-175)
/// additionally stores `discarded_normal_velocity`, `object_point_velocity`
/// and `kinematic_contract`; they are kept as optional fields so an
/// engine-produced anchor round-trips without loss.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct MovingContactAnchor {
    pub object_id: String,
    pub local_position: [f64; 3],
    pub local_velocity: [f64; 3],
    pub relative_matrix: [f64; 9],
    pub local_angular_velocity: [f64; 3],
    pub local_contact_normal: [f64; 3],
    pub local_plane_offset: f64,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub discarded_normal_velocity: Option<f64>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub object_point_velocity: Option<[f64; 3]>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub kinematic_contract: Option<String>,
}

/// Rigid-body snapshot inside a network correction transaction. Keys are
/// `RIGID_FIELDS` in scripts/evaluate_network_correction_topology.py:9.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RigidStateSnapshot {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub matrix: [f64; 9],
}

/// `network_correction` latch (`dict | None` in Python).
///
/// Built by `begin_network_correction` in
/// scripts/evaluate_network_correction_topology.py:180-210 (`mode`,
/// `duration_ms`, `elapsed_ms`, `sequence`, `initial_state`, `target_state`,
/// `complete`, `policy_authority`); the engine adds `resume_topology`
/// (`asdict(TopologyState)`, kartrider_vehicle_engine.py:4449). The required
/// key set validated by `VehicleState.validate` is `mode`, `duration_ms`,
/// `elapsed_ms`, `initial_state`, `target_state`, `complete`
/// (kartrider_vehicle_engine.py:352-359); `initial_state`/`target_state` are
/// rigid-body snapshots, not full `VehicleState` copies.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct NetworkCorrectionState {
    /// `"snap"` or `"blend"` (producer-validated).
    pub mode: String,
    pub duration_ms: f64,
    pub elapsed_ms: f64,
    /// Producer-written keys (`sequence`, `policy_authority`,
    /// `resume_topology`) are always serialized so engine-produced
    /// transactions round-trip; they deserialize as `None` when absent.
    pub sequence: Option<u32>,
    pub initial_state: RigidStateSnapshot,
    pub target_state: RigidStateSnapshot,
    pub complete: bool,
    pub policy_authority: Option<String>,
    pub resume_topology: Option<TopologyState>,
}

/// `VehicleState` (kartrider_vehicle_engine.py:207-255).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct VehicleState {
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub matrix: [f64; 9],
    pub force_accumulator: [f64; 3],
    pub torque_accumulator: [f64; 3],
    pub boost_state: i64,
    pub boost_remaining_ms: i64,
    pub booster_inventory_slots: Vec<InventorySlot>,
    pub item_use_key_latched: bool,
    pub banana_effect_remaining_ms: i64,
    pub banana_effect_owner_id: Option<String>,
    pub drive_aa4: f64,
    pub drive_aa8: f64,
    pub drive_selected_slot: f64,
    pub drive_raw_aa0: f64,
    pub drive_sign_aac: bool,
    pub ground_timer_a80: f64,
    pub contact_compression: [f64; 4],
    pub collision_arg_threshold: f64,
    pub collision_e38: bool,
    pub collision_e4c: bool,
    pub soft_selector_af4: bool,
    pub soft_flag_188: bool,
    pub selector_b08: bool,
    pub selector_b5c: bool,
    pub selector_f08: bool,
    pub selector_fec: i64,
    pub selector_ff8: i64,
    pub field_af8: f64,
    pub recovery_timer: f64,
    pub drift_timer_b0c: f64,
    pub drift_field_b00: f64,
    pub drift_snapshot_5d0: f64,
    pub drift_b1c_charge_settlement: bool,
    pub boost_gauge_committed_b10: f64,
    pub boost_gauge_progress_b20: f64,
    pub boost_gauge_pending_b2c: f64,
    pub boost_gauge_history_b38: f64,
    pub drift_blend_ab0: f64,
    pub drift_blend_ab4: f64,
    pub drift_key_latched: bool,
    pub takeoff_separation_pending: bool,
    pub surface_mode_latched: Option<i64>,
    pub surface_selector_elapsed_ms: f64,
    pub checkpoint_mode_family: String,
    pub checkpoint_target: [f64; 3],
    pub checkpoint_gate: bool,
    pub checkpoint_gate_armed_tick_ms: Option<i64>,
    pub checkpoint_marker_active_id: Option<String>,
    pub manual_restart_key_latched: bool,
    pub manual_restart_count: i64,
    pub special_trigger_cooldown_ms: f64,
    pub authored_trigger_contact_id: Option<String>,
    pub rail_selector_runtime: Option<RailSelectorRuntime>,
    pub rail_base_bf8: Option<[f64; 9]>,
    pub rail_constraint_c1c: Option<[f64; 9]>,
    pub rail_orientation_dt_latch: Option<f64>,
    pub rail_capture_latch: Option<RailCaptureLatch>,
    pub moving_contact_anchor: Option<MovingContactAnchor>,
    pub network_correction: Option<NetworkCorrectionState>,
    pub last_network_sequence: Option<u32>,
    pub network_packets_accepted: i64,
    pub network_packets_rejected: i64,
    pub network_packets_lost_estimate: i64,
    pub topology: TopologyState,
    pub time_ms: u64,
    pub step_index: u64,
    pub scheduler_previous_tick_ms: Option<u32>,
}

impl Default for VehicleState {
    fn default() -> Self {
        VehicleState {
            position: [0.0; 3],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            force_accumulator: [0.0; 3],
            torque_accumulator: [0.0; 3],
            boost_state: 0,
            boost_remaining_ms: 0,
            booster_inventory_slots: empty_current_manager_slots(),
            item_use_key_latched: false,
            banana_effect_remaining_ms: 0,
            banana_effect_owner_id: None,
            drive_aa4: 0.0,
            drive_aa8: 0.0,
            drive_selected_slot: 0.0,
            drive_raw_aa0: 0.0,
            drive_sign_aac: false,
            ground_timer_a80: 0.0,
            contact_compression: [0.5; 4],
            collision_arg_threshold: DEFAULT_COLLISION_ARG_THRESHOLD,
            collision_e38: false,
            collision_e4c: false,
            soft_selector_af4: false,
            soft_flag_188: false,
            selector_b08: false,
            selector_b5c: false,
            selector_f08: false,
            selector_fec: 0,
            selector_ff8: 0,
            field_af8: 0.0,
            recovery_timer: 0.0,
            drift_timer_b0c: 0.0,
            drift_field_b00: 0.0,
            drift_snapshot_5d0: 0.0,
            drift_b1c_charge_settlement: false,
            boost_gauge_committed_b10: 0.0,
            boost_gauge_progress_b20: 0.0,
            boost_gauge_pending_b2c: 0.0,
            boost_gauge_history_b38: 0.0,
            drift_blend_ab0: 0.0,
            drift_blend_ab4: 0.0,
            drift_key_latched: false,
            takeoff_separation_pending: false,
            surface_mode_latched: None,
            surface_selector_elapsed_ms: 0.0,
            checkpoint_mode_family: "c016".to_string(),
            checkpoint_target: [0.0; 3],
            checkpoint_gate: false,
            checkpoint_gate_armed_tick_ms: None,
            checkpoint_marker_active_id: None,
            manual_restart_key_latched: false,
            manual_restart_count: 0,
            special_trigger_cooldown_ms: 0.0,
            authored_trigger_contact_id: None,
            rail_selector_runtime: None,
            rail_base_bf8: None,
            rail_constraint_c1c: None,
            rail_orientation_dt_latch: None,
            rail_capture_latch: None,
            moving_contact_anchor: None,
            network_correction: None,
            last_network_sequence: None,
            network_packets_accepted: 0,
            network_packets_rejected: 0,
            network_packets_lost_estimate: 0,
            topology: TopologyState::default(),
            time_ms: 0,
            step_index: 0,
            scheduler_previous_tick_ms: None,
        }
    }
}

impl VehicleState {
    /// Port of `VehicleState.validate` (kartrider_vehicle_engine.py:257-451),
    /// `ValueError` -> `Err(String)` with equivalent messages. Checks whose
    /// Python version only constrains type or container length (vector/matrix
    /// sizes, uint32 ranges, nonnegative `time_ms`/`step_index`, field
    /// presence inside the latch structs) are enforced by the Rust types and
    /// are not repeated here.
    pub fn validate(&self) -> Result<(), String> {
        validate_current_manager_slots(&self.booster_inventory_slots)?;
        if self.banana_effect_remaining_ms < 0 {
            return Err("banana_effect_remaining_ms must be nonnegative".to_string());
        }
        if self.banana_effect_remaining_ms == 0 && self.banana_effect_owner_id.is_some() {
            return Err("banana_effect_owner_id requires an active banana effect".to_string());
        }
        for (name, value) in [
            ("boost_gauge_committed_b10", self.boost_gauge_committed_b10),
            ("boost_gauge_progress_b20", self.boost_gauge_progress_b20),
            ("boost_gauge_pending_b2c", self.boost_gauge_pending_b2c),
            ("boost_gauge_history_b38", self.boost_gauge_history_b38),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and nonnegative"));
            }
        }
        if self.checkpoint_mode_family != "c016" && self.checkpoint_mode_family != "c72c" {
            return Err("checkpoint_mode_family must be c016 or c72c".to_string());
        }
        if self.manual_restart_count < 0 {
            return Err("manual_restart_count must be nonnegative".to_string());
        }
        if self.special_trigger_cooldown_ms < 0.0 {
            return Err("special_trigger_cooldown_ms must be nonnegative".to_string());
        }
        for (name, value) in [
            ("field_af8", self.field_af8),
            ("recovery_timer", self.recovery_timer),
            ("drift_timer_b0c", self.drift_timer_b0c),
        ] {
            if value < 0.0 {
                return Err(format!("{name} must be nonnegative"));
            }
        }
        if !matches!(self.selector_fec, 0..=2) {
            return Err("selector_fec must be 0, 1, or 2".to_string());
        }
        if !matches!(self.selector_ff8, 0..=2) {
            return Err("selector_ff8 must be 0, 1, or 2".to_string());
        }
        for (name, value) in [
            ("network_packets_accepted", self.network_packets_accepted),
            ("network_packets_rejected", self.network_packets_rejected),
            (
                "network_packets_lost_estimate",
                self.network_packets_lost_estimate,
            ),
        ] {
            if value < 0 {
                return Err(format!("{name} must be nonnegative"));
            }
        }
        let domain = self.topology.domain;
        if let Some(anchor) = &self.moving_contact_anchor {
            if !matches!(
                domain,
                MotionDomain::MovingContact | MotionDomain::NetworkCorrection
            ) {
                return Err("moving_contact_anchor requires MOVING-CONTACT ownership \
                     or NET-CORRECTION suspension"
                    .to_string());
            }
            if domain == MotionDomain::MovingContact
                && self.topology.constraint_owner_id.as_deref() != Some(anchor.object_id.as_str())
            {
                return Err("moving contact owner must match anchor object_id".to_string());
            }
        } else if domain == MotionDomain::MovingContact {
            return Err("MOVING-CONTACT ownership requires an anchor".to_string());
        }
        if self.network_correction.is_some() {
            if domain != MotionDomain::NetworkCorrection {
                return Err(
                    "network_correction payload requires NET-CORRECTION ownership".to_string(),
                );
            }
        } else if domain == MotionDomain::NetworkCorrection {
            return Err("NET-CORRECTION ownership requires an active transaction".to_string());
        }
        if domain == MotionDomain::Rail && !matches!(self.topology.rail_state, 1..=6) {
            return Err("RAIL ownership requires active rail_state 1..6".to_string());
        }
        if self.rail_selector_runtime.is_some()
            && !matches!(domain, MotionDomain::Rail | MotionDomain::NetworkCorrection)
        {
            return Err("rail_selector_runtime requires RAIL ownership or \
                 NET-CORRECTION suspension"
                .to_string());
        }
        if self.rail_base_bf8.is_none() != self.rail_constraint_c1c.is_none() {
            return Err("rail_base_bf8 and rail_constraint_c1c must coexist".to_string());
        }
        if self.rail_base_bf8.is_some()
            && !matches!(domain, MotionDomain::Rail | MotionDomain::NetworkCorrection)
        {
            return Err("RAIL orientation state requires RAIL ownership or \
                 NET-CORRECTION suspension"
                .to_string());
        }
        if let Some(dt_latch) = self.rail_orientation_dt_latch {
            if dt_latch <= 0.0 {
                return Err("rail_orientation_dt_latch must be positive".to_string());
            }
            if !matches!(domain, MotionDomain::Rail | MotionDomain::NetworkCorrection) {
                return Err("rail_orientation_dt_latch requires RAIL ownership or \
                     NET-CORRECTION suspension"
                    .to_string());
            }
        }
        if let Some(latch) = &self.rail_capture_latch {
            if !latch.contact_active {
                return Err("rail_capture_latch requires an active contact candidate".to_string());
            }
            // "rail_capture_latch requires a decoded selector": the selector
            // mapping shape is enforced by the Rust type.
        }
        if !matches!(domain, MotionDomain::Rail | MotionDomain::NetworkCorrection)
            && self.topology.rail_state != 0
        {
            return Err("rail_state outside RAIL/NET-CORRECTION is stale".to_string());
        }
        self.topology.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moving_anchor(object_id: &str) -> MovingContactAnchor {
        MovingContactAnchor {
            object_id: object_id.to_string(),
            ..Default::default()
        }
    }

    fn snap_correction() -> NetworkCorrectionState {
        NetworkCorrectionState {
            mode: "snap".to_string(),
            duration_ms: 0.0,
            elapsed_ms: 0.0,
            sequence: None,
            initial_state: RigidStateSnapshot::default(),
            target_state: RigidStateSnapshot::default(),
            complete: false,
            policy_authority: Some("explicit_network_reconciliation_policy".to_string()),
            resume_topology: Some(TopologyState::default()),
        }
    }

    #[test]
    fn default_state_matches_python_dataclass_defaults() {
        let state = VehicleState::default();
        assert_eq!(state.contact_compression, [0.5; 4]);
        assert_eq!(state.collision_arg_threshold, 0.65);
        assert_eq!(state.checkpoint_mode_family, "c016");
        assert_eq!(state.matrix, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        state.validate().unwrap();
    }

    #[test]
    fn rejects_orphaned_constraint_payloads() {
        // Mirror of scripts/test_motion_topology.py
        // test_vehicle_state_rejects_orphaned_constraint_payloads.
        let state = VehicleState {
            topology: TopologyState {
                domain: MotionDomain::MovingContact,
                constraint_owner_id: Some("roadobj:42".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(state.validate().unwrap_err().contains("requires an anchor"));

        let state = VehicleState {
            topology: TopologyState {
                domain: MotionDomain::NetworkCorrection,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(state.validate().unwrap_err().contains("active transaction"));

        let state = VehicleState {
            topology: TopologyState {
                domain: MotionDomain::Rail,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(state.validate().unwrap_err().contains("active rail_state"));

        let state = VehicleState {
            topology: TopologyState {
                domain: MotionDomain::Ground,
                rail_state: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        let error = state.validate().unwrap_err();
        assert!(error.contains("rail_state") && error.contains("stale"));
    }

    #[test]
    fn moving_contact_anchor_domain_contract() {
        let anchor = moving_anchor("roadobj:42");
        // Anchor outside MOVING-CONTACT / NET-CORRECTION is rejected.
        let state = VehicleState {
            moving_contact_anchor: Some(anchor.clone()),
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("moving_contact_anchor requires MOVING-CONTACT ownership \
                 or NET-CORRECTION suspension"
                .to_string())
        );
        // Owner mismatch is rejected.
        let state = VehicleState {
            moving_contact_anchor: Some(anchor.clone()),
            topology: TopologyState {
                domain: MotionDomain::MovingContact,
                constraint_owner_id: Some("roadobj:other".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("moving contact owner must match anchor object_id".to_string())
        );
        // Matching owner validates.
        let state = VehicleState {
            moving_contact_anchor: Some(anchor.clone()),
            topology: TopologyState {
                domain: MotionDomain::MovingContact,
                constraint_owner_id: Some("roadobj:42".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        state.validate().unwrap();
        // NET-CORRECTION suspension tolerates the anchor.
        let state = VehicleState {
            moving_contact_anchor: Some(anchor),
            network_correction: Some(snap_correction()),
            topology: TopologyState {
                domain: MotionDomain::NetworkCorrection,
                return_domain: Some(MotionDomain::MovingContact),
                constraint_owner_id: Some("roadobj:42".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        state.validate().unwrap();
    }

    #[test]
    fn network_correction_domain_contract() {
        let state = VehicleState {
            network_correction: Some(snap_correction()),
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("network_correction payload requires NET-CORRECTION ownership".to_string())
        );
        let state = VehicleState {
            network_correction: Some(snap_correction()),
            topology: TopologyState {
                domain: MotionDomain::NetworkCorrection,
                return_domain: Some(MotionDomain::Ground),
                ..Default::default()
            },
            ..Default::default()
        };
        state.validate().unwrap();
    }

    #[test]
    fn rail_latch_domain_contracts() {
        // rail_selector_runtime requires RAIL / NET-CORRECTION.
        let state = VehicleState {
            rail_selector_runtime: Some(RailSelectorRuntime::default()),
            ..Default::default()
        };
        assert!(state
            .validate()
            .unwrap_err()
            .contains("rail_selector_runtime requires RAIL ownership"));

        // BF8/C1C must coexist.
        let state = VehicleState {
            rail_base_bf8: Some([0.0; 9]),
            topology: TopologyState {
                domain: MotionDomain::Rail,
                rail_state: 3,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("rail_base_bf8 and rail_constraint_c1c must coexist".to_string())
        );

        // Orientation matrices require RAIL / NET-CORRECTION.
        let state = VehicleState {
            rail_base_bf8: Some([0.0; 9]),
            rail_constraint_c1c: Some([0.0; 9]),
            ..Default::default()
        };
        assert!(state
            .validate()
            .unwrap_err()
            .contains("RAIL orientation state requires RAIL ownership"));

        // dt latch must be positive; the sign check precedes the domain check.
        let state = VehicleState {
            rail_orientation_dt_latch: Some(-1.0),
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("rail_orientation_dt_latch must be positive".to_string())
        );
        let state = VehicleState {
            rail_orientation_dt_latch: Some(1.0),
            ..Default::default()
        };
        assert!(state
            .validate()
            .unwrap_err()
            .contains("rail_orientation_dt_latch requires RAIL ownership"));

        // Capture latch requires an active contact candidate.
        let state = VehicleState {
            rail_capture_latch: Some(RailCaptureLatch::default()),
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("rail_capture_latch requires an active contact candidate".to_string())
        );

        // Fully populated RAIL state validates.
        let state = VehicleState {
            rail_selector_runtime: Some(RailSelectorRuntime {
                path_context_id: Some(RailPathContextId::Index(32)),
                state6_contact_remaining_ms: Some(100.00000149011612),
                ..Default::default()
            }),
            rail_base_bf8: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            rail_constraint_c1c: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            rail_orientation_dt_latch: Some(2.0),
            rail_capture_latch: Some(RailCaptureLatch {
                contact_active: true,
                rail_state: Some(3),
                selector: BTreeMap::new(),
                authority: Some("world_query_rail_contact".to_string()),
            }),
            topology: TopologyState {
                domain: MotionDomain::Rail,
                rail_state: 3,
                constraint_owner_id: Some("rail:path".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        state.validate().unwrap();
    }

    #[test]
    fn scalar_invariants() {
        let state = VehicleState {
            checkpoint_mode_family: "bogus".to_string(),
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("checkpoint_mode_family must be c016 or c72c".to_string())
        );
        let state = VehicleState {
            checkpoint_mode_family: "c72c".to_string(),
            ..Default::default()
        };
        state.validate().unwrap();
        let state = VehicleState {
            manual_restart_count: -1,
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("manual_restart_count must be nonnegative".to_string())
        );
        let state = VehicleState {
            special_trigger_cooldown_ms: -0.5,
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("special_trigger_cooldown_ms must be nonnegative".to_string())
        );
        let state = VehicleState {
            network_packets_rejected: -1,
            ..Default::default()
        };
        assert_eq!(
            state.validate(),
            Err("network_packets_rejected must be nonnegative".to_string())
        );
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    use crate::topology::{advance_topology, TopologyObservation};

    #[test]
    fn default_state_json_shape_matches_python_asdict() {
        let json = serde_json::to_value(VehicleState::default()).unwrap();
        let object = json.as_object().unwrap();
        // Every dataclass field is emitted; `None` fields become null.
        assert_eq!(object.len(), 69);
        assert_eq!(json["banana_effect_remaining_ms"], 0);
        assert_eq!(json["banana_effect_owner_id"], serde_json::Value::Null);
        assert_eq!(json["surface_mode_latched"], serde_json::Value::Null);
        assert_eq!(json["rail_selector_runtime"], serde_json::Value::Null);
        assert_eq!(json["network_correction"], serde_json::Value::Null);
        assert_eq!(json["last_network_sequence"], serde_json::Value::Null);
        assert_eq!(json["scheduler_previous_tick_ms"], serde_json::Value::Null);
        assert_eq!(json["checkpoint_mode_family"], "c016");
        assert_eq!(
            json["contact_compression"],
            serde_json::json!([0.5, 0.5, 0.5, 0.5])
        );
        assert_eq!(json["collision_arg_threshold"], 0.65);
        assert_eq!(json["selector_b08"], false);
        assert_eq!(json["drift_timer_b0c"], 0.0);
        assert_eq!(json["drift_key_latched"], false);
        assert_eq!(
            json["booster_inventory_slots"],
            serde_json::json!([
                {"item_id": 65535, "quantity": 0},
                {"item_id": 65535, "quantity": 0},
            ])
        );
        assert_eq!(json["item_use_key_latched"], false);
        assert_eq!(json["boost_gauge_committed_b10"], 0.0);
        assert_eq!(json["boost_gauge_progress_b20"], 0.0);
        assert_eq!(json["boost_gauge_pending_b2c"], 0.0);
        assert_eq!(json["boost_gauge_history_b38"], 0.0);
        assert_eq!(json["topology"]["domain"], "GND");
        assert_eq!(json["topology"]["previous_domain"], serde_json::Value::Null);
        assert_eq!(json["topology"]["entry_reason"], "initialization");
        let round_trip: VehicleState = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, VehicleState::default());
        round_trip.validate().unwrap();
    }

    #[test]
    fn rail_selector_runtime_serializes_only_present_keys() {
        let runtime = RailSelectorRuntime {
            path_context_id: Some(RailPathContextId::Index(32)),
            state6_contact_remaining_ms: Some(100.00000149011612),
            ..Default::default()
        };
        let json = serde_json::to_value(&runtime).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "path_context_id": 32,
                "state6_contact_remaining_ms": 100.00000149011612,
            })
        );
        let round_trip: RailSelectorRuntime = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, runtime);
        // Empty runtime round-trips as `{}`.
        let empty = serde_json::to_value(RailSelectorRuntime::default()).unwrap();
        assert_eq!(empty, serde_json::json!({}));
        // String context ids use the same untagged slot.
        let named: RailSelectorRuntime =
            serde_json::from_value(serde_json::json!({"path_context_id": "asset-context-0"}))
                .unwrap();
        assert_eq!(
            named.path_context_id,
            Some(RailPathContextId::Name("asset-context-0".to_string()))
        );
        assert_eq!(
            serde_json::to_value(&named).unwrap(),
            serde_json::json!({"path_context_id": "asset-context-0"})
        );
    }

    #[test]
    fn network_correction_serializes_producer_keys() {
        let correction = NetworkCorrectionState {
            mode: "snap".to_string(),
            ..Default::default()
        };
        let json = serde_json::to_value(&correction).unwrap();
        // The producer always writes these keys, even when None/null.
        assert_eq!(json["sequence"], serde_json::Value::Null);
        assert_eq!(json["policy_authority"], serde_json::Value::Null);
        assert_eq!(json["resume_topology"], serde_json::Value::Null);
        assert_eq!(json["mode"], "snap");
        assert_eq!(json["complete"], false);
        let round_trip: NetworkCorrectionState = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, correction);
    }

    #[test]
    fn capture_latch_selector_payload_round_trip() {
        let mut selector = BTreeMap::new();
        selector.insert("contact_active".to_string(), JsonValue::Bool(true));
        selector.insert("rail_state".to_string(), JsonValue::Number(1.0));
        selector.insert(
            "path_id".to_string(),
            JsonValue::String("rail:path-9".to_string()),
        );
        selector.insert(
            "runtime".to_string(),
            JsonValue::Object(BTreeMap::from([(
                "rail_dir".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(0.0),
                    JsonValue::Number(1.0),
                    JsonValue::Number(0.0),
                ]),
            )])),
        );
        selector.insert("min_velocity".to_string(), JsonValue::Null);
        let latch = RailCaptureLatch {
            contact_active: true,
            rail_state: Some(1),
            selector,
            authority: Some("world_query_rail_contact".to_string()),
        };
        let json = serde_json::to_string(&latch).unwrap();
        let round_trip: RailCaptureLatch = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, latch);
    }

    #[test]
    fn observation_and_decision_json_shape() {
        let observation = TopologyObservation {
            support_contact_count: 4,
            surface_mode: 2,
            ..Default::default()
        };
        let decision = advance_topology(&TopologyState::default(), &observation).unwrap();
        let json = serde_json::to_value(&decision).unwrap();
        assert_eq!(json["selected_domain"], "GND-SLIP");
        assert_eq!(json["transitioned"], true);
        assert_eq!(json["reason"], "ground_support_active");
        assert_eq!(json["authority"], "engine_owned_topology_selectors");
        assert_eq!(json["state_after"]["domain"], "GND-SLIP");
        assert_eq!(
            json["observation"]["resume_domain"],
            serde_json::Value::Null
        );
        let round_trip: crate::topology::TopologyDecision = serde_json::from_value(json).unwrap();
        assert_eq!(round_trip, decision);
    }
}
