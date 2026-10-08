//! World adapter trait (port of the duck-typed protocol in
//! `scripts/kart_track_world.py`).
//!
//! The engine owns all physics; the world only answers geometric queries.
//! `query_ground` is the single required method (Python raises
//! `ValueError("autonomous topology stepping requires world.query_ground")`
//! when it is missing; the Rust trait makes that absence unrepresentable).
//!
//! Optional capabilities are probed with `hasattr` in Python. In Rust each
//! capability is a trait method with a default `None` implementation, where
//! the OUTER `Option` expresses capability presence and the INNER `Option`
//! expresses Python's `dict | None` return. Capability-absence semantics are
//! contractual (docs/kart-physics-rust-handoff-2026-07-28.md §4): callers
//! must branch on the outer `Option`, never unwrap through a missing
//! capability.
//!
//! Only `query_ground` is consumed by the currently ported engine slices
//! (`engine::query_wheel_ground`); the remaining capabilities are consumed
//! by the not-yet-ported `step_world` / topology slices, so their result
//! structs carry exactly the keys the Python engine reads today.

use std::collections::BTreeMap;

use crate::state::JsonValue;

/// `world-query-ground` row (`parse_world_query_tsv`,
/// scripts/kart_track_world.py:57-77). This is also the per-wheel ground
/// plane shape consumed by `step_raw_ground_correction` (`normal`, `point`,
/// `triangle`, `road_surface`, `node`, `tri_index`).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct WorldGroundResult {
    pub track_id: String,
    pub node: String,
    pub tri_index: i64,
    pub road_surface: String,
    pub surface_id: i64,
    pub road_rail: String,
    pub distance: f64,
    pub point: [f64; 3],
    pub normal: [f64; 3],
    /// Present when the query row carries the authored triangle vertices.
    pub triangle: Option<[[f64; 3]; 3]>,
    pub contact_basis_44: Option<[f64; 3]>,
}

/// One `overlap_obb_all` hit candidate (`parse_world_obb_overlaps_tsv`,
/// scripts/kart_track_world.py:364-401).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct WorldObbOverlap {
    pub track_id: String,
    pub hit: bool,
    pub loop_index: i64,
    pub fraction: f64,
    pub node: String,
    pub tri_index: i64,
    pub road_surface: String,
    pub surface_id: i64,
    pub road_rail: String,
    pub center: [f64; 3],
    pub support_point: [f64; 3],
    pub ccd_normal: [f64; 3],
    pub normal: [f64; 3],
    pub normal_authority: String,
    pub contact_basis_44: Option<[f64; 3]>,
    /// Optional authored boost multiplier flag read by the autonomous rail
    /// JM/DJ trigger query (`candidate.get("boost_multiplier_64c", False)`,
    /// kartrider_vehicle_engine.py:3478-3480). Skipped when false so the
    /// serialized shape matches Python payloads that omit the key.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "std::ops::Not::not"))]
    pub boost_multiplier_64c: bool,
}

/// One decoded ToMovableObject obstacle face overlapping the vehicle OBB.
///
/// This is distinct from [`WorldObbOverlap`]: static road/wall candidates
/// feed the hard/soft wall selector, while an event obstacle carries its
/// sampled surface motion into the GoItemObstacle CONTACT source formula.
/// `normal` preserves the authored triangle winding and `motion` is the
/// forward-difference velocity of the same material point at `time_ms`.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct EventObstacleContact {
    pub event_index: i64,
    pub object_id: String,
    pub node: String,
    pub tri_index: i64,
    pub loop_index: i64,
    pub centroid: [f64; 3],
    pub normal: [f64; 3],
    pub motion: [f64; 3],
    pub obstacle_position: [f64; 3],
}

/// Candidate mapping returned by `query_takeoff_contact` /
/// `query_landing_contact`. The engine reads
/// `candidate.get("selected", candidate.get("hit", True))` and requires
/// `normal` (kartrider_vehicle_engine.py:6447-6460, :6586-6600).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct TakeoffLandingContact {
    pub selected: Option<bool>,
    pub hit: Option<bool>,
    pub normal: Option<[f64; 3]>,
    pub face_id: Option<String>,
    /// Optional provenance tag read by the takeoff validation
    /// (`candidate.get("source")`, kartrider_vehicle_engine.py:6489).
    pub source: Option<String>,
}

/// `match_rail_entry_boundary` result (scripts/kart_track_world.py:198-275):
/// one authored ToRoad entry-rectangle match.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RailEntryBoundaryMatch {
    /// Authored context id; runtime evidence carries both ints and strings.
    pub context_id: JsonValue,
    pub context_index: i64,
    pub road: String,
    pub element: JsonValue,
    pub marker: JsonValue,
    pub signed_plane_distance: f64,
    pub plane_distance: f64,
    pub projected_coordinates: [f64; 2],
    pub path_start: [f64; 3],
    pub tangent: [f64; 3],
}

/// Nearest KAP event-marker candidate (`query_checkpoint_target_candidate`).
/// The engine reads `marker_id` (required) and `marker_ordinal`
/// (kartrider_vehicle_engine.py:1536-1543).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct CheckpointTargetCandidate {
    pub marker_id: String,
    pub marker_ordinal: i64,
    pub position: [f64; 3],
    pub distance: f64,
    pub authority: String,
}

/// Reset-facing reference built from the authored course graph
/// (`query_checkpoint_respawn`). The engine requires
/// `course_reference_position` and `sample_result_zero` and reads
/// `authority` with a `world_query_checkpoint_respawn` default
/// (kartrider_vehicle_engine.py:1732-1750); the remaining keys mirror the
/// world selector payload (scripts/kart_track_world.py:849-870).
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct CheckpointRespawnCandidate {
    pub course_reference_position: [f64; 3],
    pub sample_result_zero: bool,
    pub course_context_id: JsonValue,
    pub course_road: String,
    pub course_segment_index: i64,
    pub course_segment_fraction: f64,
    pub course_projection: f64,
    pub course_distance: f64,
    pub course_tangent: [f64; 3],
    pub authority: Option<String>,
}

/// Accepted GoItemBanana run-over candidate. The world owns the animated
/// item position and the exact captured 2 m gate; the engine owns the timed
/// B9C effect latch after acceptance.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct BananaTrigger {
    pub object_id: String,
    pub event_index: i64,
    pub position: [f64; 3],
    pub distance: f64,
    pub radius: f64,
    pub authority: String,
}

/// Optional-capability presence flags: the Rust expression of Python's
/// `hasattr(world, ...)` probes. Probing by *calling* a method is not
/// acceptable — a recording/replaying world treats every call as a consumed
/// log entry, and several engine branches check presence without calling
/// (e.g. the `takeoff_separation_pending` fallback, handoff §4). Every flag
/// defaults to `false`; an implementor sets the flag for each optional
/// method it actually implements. `query_ground` is required and has no
/// flag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorldCapabilities {
    pub overlap_obb_all: bool,
    /// `query_event_obstacle_contacts(center, matrix, half_extents,
    /// time_ms, delta_ms)` — decoded, P1-enabled ToMovableObject obstacle
    /// faces. This CONTACT-family path is deliberately separate from
    /// `query_moving_contact`.
    pub query_event_obstacle_contacts: bool,
    /// `query_banana_trigger(position, time_ms)` — P1-closed factory
    /// GoItemBanana 2 m run-over gate. This is an item-effect query, not a
    /// CONTACT surface.
    pub query_banana_trigger: bool,
    pub query_takeoff_contact: bool,
    pub query_landing_contact: bool,
    pub query_rail_contact: bool,
    pub query_rail_entry_boundary: bool,
    pub query_moving_contact: bool,
    pub query_checkpoint_target_candidate: bool,
    pub query_checkpoint_respawn: bool,
    /// `sweep_obb_rotating_all(start, end, start_matrix, end_matrix,
    /// half_extents)` — the all-candidates vehicle-volume sweep consumed by
    /// `step_world` (kartrider_vehicle_engine.py:7232-7242).
    pub sweep_obb_rotating_all: bool,
    /// `sweep_obb_rotating(...)` — the single-candidate fallback used when
    /// `sweep_obb_rotating_all` is absent (kartrider_vehicle_engine.py:7244).
    pub sweep_obb_rotating: bool,
    /// `sweep_point(start, end)` — the A5B840 point-segment query consumed by
    /// the state6 B276F0 tail (center segment candidate) and the B2FB00
    /// partial-wheel bridge (kartrider_vehicle_engine.py:1858, :2321).
    pub sweep_point: bool,
    /// Python `getattr(world, "query_at_step_start", False)`: sweep at the
    /// step-start pose instead of the step-end pose.
    pub query_at_step_start: bool,
    /// `rail_query_contexts()` (scripts/kart_track_world.py:800): the
    /// registered RAIL path-context list consumed by
    /// `evaluate_rail_state1_query`.
    pub rail_query_contexts: bool,
    /// `rail_entry_contexts()` (scripts/kart_track_world.py:817): the
    /// authored ToRoad entry-rectangle list backing
    /// `query_rail_entry_boundary`.
    pub rail_entry_contexts: bool,
    /// `query_network_correction(position)` (engine :4484): the already-
    /// accepted authoritative correction packet consumed by
    /// `inspect_autonomous_network_correction` (M3).
    pub query_network_correction: bool,
    /// `query_distinct_water_volume(position)` (engine :5036): the WATER
    /// resume selector consumed by `step_autonomous_network_correction`.
    pub query_distinct_water_volume: bool,
}

/// World adapter protocol. `query_ground` is required; every other method
/// is an optional capability whose default implementation reports absence
/// (`None`), matching Python's `hasattr` probe.
pub trait World {
    /// Capability presence flags (Python `hasattr` probes). The default
    /// reports all-absent; implementors must keep the flags in sync with
    /// the methods they override.
    fn capabilities(&self) -> WorldCapabilities {
        WorldCapabilities::default()
    }

    /// `query_ground(position, up, min_dot=0.65)`.
    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult;

    /// `sweep_obb_rotating_all(sweep_start, sweep_end, start_matrix,
    /// end_matrix, half_extents)` — all OBB sweep hit candidates in authored
    /// loop order (same row shape as [`WorldObbOverlap`]).
    fn sweep_obb_rotating_all(
        &self,
        _start: [f64; 3],
        _end: [f64; 3],
        _start_matrix: [f64; 9],
        _end_matrix: [f64; 9],
        _half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        None
    }

    /// `sweep_obb_rotating(sweep_start, sweep_end, start_matrix, end_matrix,
    /// half_extents)` — single-candidate sweep fallback.
    fn sweep_obb_rotating(
        &self,
        _start: [f64; 3],
        _end: [f64; 3],
        _start_matrix: [f64; 9],
        _end_matrix: [f64; 9],
        _half_extents: [f64; 3],
    ) -> Option<WorldObbOverlap> {
        None
    }

    /// `sweep_point(segment_start, segment_end)` — the A5B840 point-segment
    /// query. Python requires the return to be a mapping
    /// (`ValueError("state6 sweep_point must return a mapping")`), which the
    /// return type enforces; the outer `Option` is capability presence. The
    /// payload is an open mapping: the engine reads `hit` and `road_rail`
    /// and stores a deep copy as evidence.
    fn sweep_point(&self, _start: [f64; 3], _end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        None
    }

    /// `overlap_obb_all(center, matrix, half_extents)` — all OBB overlap hit
    /// candidates. Absence releases `takeoff_separation_pending` immediately
    /// and falls back to wheel-manifold landing (handoff §4).
    fn overlap_obb_all(
        &self,
        _center: [f64; 3],
        _matrix: [f64; 9],
        _half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        None
    }

    /// P1-enabled dynamic event-obstacle faces overlapping the current
    /// vehicle OBB. The outer `Option` is capability presence.
    fn query_event_obstacle_contacts(
        &self,
        _center: [f64; 3],
        _matrix: [f64; 9],
        _half_extents: [f64; 3],
        _time_ms: i64,
        _delta_ms: i64,
    ) -> Option<Vec<EventObstacleContact>> {
        None
    }

    /// Nearest active GoItemBanana whose captured interaction radius contains
    /// the kart position. The engine owns the resulting timed effect latch.
    fn query_banana_trigger(&self, _position: [f64; 3], _time_ms: i64) -> Option<BananaTrigger> {
        None
    }

    /// `query_takeoff_contact(position, velocity, support_planes=..., active_slots=...)`.
    fn query_takeoff_contact(
        &self,
        _position: [f64; 3],
        _velocity: [f64; 3],
        _support_planes: &[WorldGroundResult],
        _active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        None
    }

    /// `query_landing_contact(position, velocity, support_planes=..., active_slots=...)`.
    fn query_landing_contact(
        &self,
        _position: [f64; 3],
        _velocity: [f64; 3],
        _support_planes: &[WorldGroundResult],
        _active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        None
    }

    /// `query_rail_contact(position, previous_rail_state=..., owner_id=...)`.
    /// The selector payload is open-ended: the engine stores it verbatim
    /// (deepcopy) inside `rail_capture_latch["selector"]`, so it is typed as
    /// a JSON mapping rather than a closed struct.
    fn query_rail_contact(
        &self,
        _position: [f64; 3],
        _previous_rail_state: i64,
        _owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        None
    }

    /// `query_rail_entry_boundary(position, road=None, plane_tolerance=0.05)`.
    fn query_rail_entry_boundary(
        &self,
        _position: [f64; 3],
        _road: Option<&str>,
        _plane_tolerance: f64,
    ) -> Option<Option<RailEntryBoundaryMatch>> {
        None
    }

    /// `rail_query_contexts()` (scripts/kart_track_world.py:800): the
    /// registered RAIL path contexts (`sub_A4EA20` input). The payload shape
    /// is asset-driven and consumed by `evaluate_rail_state1_query`, so it
    /// stays an open JSON mapping.
    fn rail_query_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        None
    }

    /// `rail_entry_contexts()` (scripts/kart_track_world.py:817): the
    /// authored ToRoad entry rectangles backing `query_rail_entry_boundary`.
    fn rail_entry_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        None
    }

    /// `query_moving_contact(position, time_ms=..., object_id=None, delta_ms=1,
    /// max_distance=1.5)`. The returned mapping feeds
    /// `acquire_moving_contact_frame` (not yet ported), so it stays an open
    /// JSON mapping for now.
    fn query_moving_contact(
        &self,
        _position: [f64; 3],
        _time_ms: i64,
        _object_id: Option<&str>,
        _delta_ms: i64,
        _max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        None
    }

    /// `query_checkpoint_target_candidate(position, radius=2.0)`.
    fn query_checkpoint_target_candidate(
        &self,
        _position: [f64; 3],
        _radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        None
    }

    /// `query_checkpoint_respawn(position, checkpoint_target=..., checkpoint_gate=...,
    /// mode_family=..., max_course_distance=20.0)`.
    fn query_checkpoint_respawn(
        &self,
        _position: [f64; 3],
        _checkpoint_target: [f64; 3],
        _checkpoint_gate: bool,
        _mode_family: &str,
        _max_course_distance: f64,
    ) -> Option<Option<CheckpointRespawnCandidate>> {
        None
    }

    /// `query_network_correction(position)` (kartrider_vehicle_engine.py
    /// :4484): one already-accepted authoritative correction packet, or
    /// `None` when the world has none pending. The payload is an open
    /// mapping: the engine reads `authoritative_position` /
    /// `authoritative_velocity` / `authoritative_matrix` /
    /// `authoritative_angular_velocity` / `mode` plus optional
    /// `duration_ms`, `sequence`, `correction_active` and
    /// `authoritative_time_ms`, and merges the mapping verbatim into the
    /// inspection candidate. Python requires a mapping-or-None return
    /// (`ValueError("query_network_correction must return a mapping or
    /// None")`), which the return type enforces.
    fn query_network_correction(
        &self,
        _position: [f64; 3],
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        None
    }

    /// `query_distinct_water_volume(position)` (kartrider_vehicle_engine.py
    /// :5036): the WATER resume selector. Python only takes `bool(...)` of
    /// the returned value, so the payload stays an open JSON value.
    fn query_distinct_water_volume(&self, _position: [f64; 3]) -> Option<Option<JsonValue>> {
        None
    }
}
