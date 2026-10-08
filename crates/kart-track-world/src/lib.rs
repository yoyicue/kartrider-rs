//! `kart-track-world`: the track.1s world-query geometry core extracted
//! from `kart-autoprobe`, plus the typed [`world::TrackWorld`] adapter that
//! implements `kart_physics::world::World`.
//!
//! - [`geom`]: pure f32 geometry (triangle/sphere/OBB sweep math).
//! - [`track`]: asset helpers, the virtual-bytes registry (W3: browser assets
//!   arrive as byte slices), and the cached static-track loader.
//! - [`query`]: the pre-derived per-track query core (world-space triangle
//!   soup + uniform-grid candidate index + selection functions) shared by
//!   the CLI renderers and the typed adapter (P2 of the performance
//!   handoff: parse/flatten/expand once per track, select in memory).
//! - [`topology`]: the KAP static-support gap detector shared by the
//!   `kap-topology` renderer and kart-autoprobe's gap-transfer controller.
//! - [`cli`]: byte-exact TSV renderers backing the kart-autoprobe CLI.
//! - [`world`]: typed API consumed by the physics engine. Decoded
//!   `ToMovableObject` rows are also exposed as diagnostic
//!   [`world::EventWorldObject`] metadata with an explicit fail-safe
//!   disposition. Only factory_I02 event 1743 has a P1-closed dynamic
//!   CONTACT query; unadjudicated classes do not enter physics.
//! - [`replay`] (feature `replay`): the shared W2/W3 live-world parity replay
//!   used by both the native `live_world_trace` example and the kart-wasm
//!   `kart_wasm_run_replay` export.

pub mod cli;
pub mod geom;
pub mod query;
#[cfg(feature = "replay")]
pub mod replay;
pub mod topology;
pub mod track;
pub mod world;

/// Everything kart-autoprobe's remaining code re-imports after the
/// extraction.
pub mod prelude {
    pub use crate::cli::{
        is_query_command, run_command, run_kap_markers, run_kap_points, run_kap_topology,
        run_world_banana_trigger, run_world_dynamic_objects, run_world_dynamic_state,
        run_world_event_obstacle_contacts, run_world_event_obstacles, run_world_node_info,
        run_world_node_triangle, run_world_overlap_obb_all, run_world_query,
        run_world_query_dynamic, run_world_query_ground, run_world_query_moving,
        run_world_query_rail, run_world_query_server_session, run_world_rail_contexts,
        run_world_rail_entries, run_world_sweep, run_world_sweep_obb, run_world_sweep_obb_rotating,
        run_world_sweep_obb_rotating_all, run_world_sweep_sphere, FlagCursor,
    };
    pub use crate::geom::*;
    pub use crate::topology::*;
    pub use crate::track::*;
}
