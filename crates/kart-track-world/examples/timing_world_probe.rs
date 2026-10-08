//! timing_world_probe — wrap TrackWorld with per-method timing and drive the
//! engine's first steps (pinpoint the frame-0 spike). Usage:
//! cargo run -p kart-track-world --release --example timing_world_probe -- \
//!   <track.1s> <sequence.json> [<spawn.json>]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

use kart_physics::engine::{KartRiderVehicleEngine, NumericMode};
use kart_physics::params::VehicleParameters;
use kart_physics::state::{JsonValue, VehicleState};
use kart_physics::world::{
    BananaTrigger, CheckpointRespawnCandidate, CheckpointTargetCandidate, EventObstacleContact,
    RailEntryBoundaryMatch, TakeoffLandingContact, World, WorldCapabilities, WorldGroundResult,
    WorldObbOverlap,
};
use kart_track_world::query::load_track_query_index;
use kart_track_world::world::TrackWorld;

struct TimingWorld {
    inner: TrackWorld,
    totals: RefCell<BTreeMap<&'static str, (u64, f64)>>,
}

impl TimingWorld {
    fn time<T>(&self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let mut totals = self.totals.borrow_mut();
        let entry = totals.entry(name).or_default();
        entry.0 += 1;
        entry.1 += t.elapsed().as_secs_f64() * 1000.0;
        out
    }
    fn report(&self) {
        let totals = self.totals.borrow();
        let mut rows: Vec<_> = totals.iter().collect();
        rows.sort_by(|a, b| b.1 .1.total_cmp(&a.1 .1));
        for (name, (count, ms)) in rows {
            eprintln!("  {name}: {count} calls, {ms:.3} ms");
        }
    }
}

macro_rules! timed {
    ($self:ident, $name:literal, $($arg:tt)*) => {
        $self.time($name, || $self.inner.$($arg)*)
    };
}

impl World for TimingWorld {
    fn capabilities(&self) -> WorldCapabilities {
        self.inner.capabilities()
    }
    fn query_ground(&self, position: [f64; 3], up: [f64; 3], min_dot: f64) -> WorldGroundResult {
        timed!(self, "query_ground", query_ground(position, up, min_dot))
    }
    fn sweep_obb_rotating_all(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        timed!(
            self,
            "sweep_obb_rotating_all",
            sweep_obb_rotating_all(start, end, start_matrix, end_matrix, half_extents)
        )
    }
    fn sweep_obb_rotating(
        &self,
        start: [f64; 3],
        end: [f64; 3],
        start_matrix: [f64; 9],
        end_matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<WorldObbOverlap> {
        timed!(
            self,
            "sweep_obb_rotating",
            sweep_obb_rotating(start, end, start_matrix, end_matrix, half_extents)
        )
    }
    fn sweep_point(&self, start: [f64; 3], end: [f64; 3]) -> Option<BTreeMap<String, JsonValue>> {
        timed!(self, "sweep_point", sweep_point(start, end))
    }
    fn overlap_obb_all(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
    ) -> Option<Vec<WorldObbOverlap>> {
        timed!(
            self,
            "overlap_obb_all",
            overlap_obb_all(center, matrix, half_extents)
        )
    }
    fn query_event_obstacle_contacts(
        &self,
        center: [f64; 3],
        matrix: [f64; 9],
        half_extents: [f64; 3],
        time_ms: i64,
        delta_ms: i64,
    ) -> Option<Vec<EventObstacleContact>> {
        timed!(
            self,
            "query_event_obstacle_contacts",
            query_event_obstacle_contacts(center, matrix, half_extents, time_ms, delta_ms)
        )
    }
    fn query_banana_trigger(&self, position: [f64; 3], time_ms: i64) -> Option<BananaTrigger> {
        timed!(
            self,
            "query_banana_trigger",
            query_banana_trigger(position, time_ms)
        )
    }
    fn query_takeoff_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        timed!(
            self,
            "query_takeoff_contact",
            query_takeoff_contact(position, velocity, support_planes, active_slots)
        )
    }
    fn query_landing_contact(
        &self,
        position: [f64; 3],
        velocity: [f64; 3],
        support_planes: &[WorldGroundResult],
        active_slots: &[i64],
    ) -> Option<Option<TakeoffLandingContact>> {
        timed!(
            self,
            "query_landing_contact",
            query_landing_contact(position, velocity, support_planes, active_slots)
        )
    }
    fn query_rail_contact(
        &self,
        position: [f64; 3],
        previous_rail_state: i64,
        owner_id: Option<&str>,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        timed!(
            self,
            "query_rail_contact",
            query_rail_contact(position, previous_rail_state, owner_id)
        )
    }
    fn query_rail_entry_boundary(
        &self,
        position: [f64; 3],
        road: Option<&str>,
        plane_tolerance: f64,
    ) -> Option<Option<RailEntryBoundaryMatch>> {
        timed!(
            self,
            "query_rail_entry_boundary",
            query_rail_entry_boundary(position, road, plane_tolerance)
        )
    }
    fn rail_query_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        timed!(self, "rail_query_contexts", rail_query_contexts())
    }
    fn rail_entry_contexts(&self) -> Option<Vec<BTreeMap<String, JsonValue>>> {
        timed!(self, "rail_entry_contexts", rail_entry_contexts())
    }
    fn query_moving_contact(
        &self,
        position: [f64; 3],
        time_ms: i64,
        object_id: Option<&str>,
        delta_ms: i64,
        max_distance: f64,
    ) -> Option<Option<BTreeMap<String, JsonValue>>> {
        timed!(
            self,
            "query_moving_contact",
            query_moving_contact(position, time_ms, object_id, delta_ms, max_distance)
        )
    }
    fn query_checkpoint_target_candidate(
        &self,
        position: [f64; 3],
        radius: f64,
    ) -> Option<Option<CheckpointTargetCandidate>> {
        timed!(
            self,
            "query_checkpoint_target_candidate",
            query_checkpoint_target_candidate(position, radius)
        )
    }
    fn query_checkpoint_respawn(
        &self,
        position: [f64; 3],
        checkpoint_target: [f64; 3],
        checkpoint_gate: bool,
        mode_family: &str,
        max_course_distance: f64,
    ) -> Option<Option<CheckpointRespawnCandidate>> {
        timed!(
            self,
            "query_checkpoint_respawn",
            query_checkpoint_respawn(
                position,
                checkpoint_target,
                checkpoint_gate,
                mode_family,
                max_course_distance
            )
        )
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let track = args
        .next()
        .expect("usage: timing_world_probe <track.1s> <sequence.json>");
    let sequence = args
        .next()
        .expect("usage: timing_world_probe <track.1s> <sequence.json>");
    let spawn = args.next();
    let seq_text = std::fs::read_to_string(sequence).unwrap();
    let seq: serde_json::Value = serde_json::from_str(&seq_text).unwrap();
    let state_json = if let Some(spawn) = spawn.as_ref() {
        let value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(spawn).unwrap()).unwrap();
        value["initial_state"].clone()
    } else {
        seq["initial_state"].clone()
    };
    let state: VehicleState = serde_json::from_value(state_json).unwrap();
    let params: VehicleParameters = serde_json::from_value(seq["parameters"].clone()).unwrap();
    let frame_ms = seq["frame_ms"].as_u64().unwrap() as u32;
    if spawn.is_some() {
        let index = load_track_query_index(std::path::Path::new(&track)).unwrap();
        let up = [-0.866_025_03_f32, 0.000_368_46, 0.500_000_5];
        let eligible = index
            .road
            .iter()
            .filter(|entry| {
                entry.normal[0] * up[0] + entry.normal[1] * up[1] + entry.normal[2] * up[2] >= 0.65
            })
            .count();
        eprintln!(
            "road triangles={} tilted-normal eligible={eligible}",
            index.road.len()
        );
    }

    for rep in 1..3 {
        let world = TimingWorld {
            inner: TrackWorld::from_path(&track).unwrap(),
            totals: RefCell::new(BTreeMap::new()),
        };
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).expect("engine");
        let frame_count: u32 = if spawn.is_some() { 220 } else { 3 };
        let mut curb_step_ms = Vec::new();
        for frame in 0..frame_count {
            let keys = if spawn.is_some() && frame >= 80 {
                vec!["UP".to_string(), "LEFT".to_string(), "CTRL".to_string()]
            } else {
                vec!["UP".to_string()]
            };
            let t = Instant::now();
            let result = engine.step_raw_frame(
                frame.wrapping_mul(frame_ms),
                &keys,
                &params,
                Some(&world),
                None,
                None,
            );
            let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
            if spawn.is_some() && frame >= 80 {
                curb_step_ms.push(elapsed_ms);
            }
            eprintln!(
                "rep{rep} step {frame}: {:.3} ms ({})",
                elapsed_ms,
                match &result {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("err {e}"),
                }
            );
        }
        if spawn.is_some() {
            curb_step_ms.sort_by(f64::total_cmp);
            let mean = curb_step_ms.iter().sum::<f64>() / curb_step_ms.len() as f64;
            let p95 = curb_step_ms[(curb_step_ms.len() * 95).div_ceil(100) - 1];
            eprintln!("rep{rep} curb frames 80..219: mean={mean:.3} ms p95={p95:.3} ms");
            eprintln!("rep{rep} 220-frame world calls:");
        } else {
            eprintln!("rep{rep} three-frame world calls:");
        }
        world.report();
    }
}
