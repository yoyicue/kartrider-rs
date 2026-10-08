//! lazy_probe — time the per-TrackWorld lazy initializers (diagnostic).
//! Usage: cargo run -p kart-track-world --release --example lazy_probe -- <track.1s>

use std::time::Instant;

use kart_physics::world::World;
use kart_track_world::world::TrackWorld;

fn main() {
    let track = std::env::args()
        .nth(1)
        .expect("usage: lazy_probe <track.1s>");
    for rep in 0..3 {
        let t = Instant::now();
        let world = TrackWorld::from_path(&track).expect("load");
        eprintln!(
            "rep{rep} from_path: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.query_ground([555.0, 695.0, 47.0], [0.0, 0.0, 1.0], 0.65);
        eprintln!(
            "rep{rep} first query_ground: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.rail_query_contexts();
        eprintln!(
            "rep{rep} rail_query_contexts: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.rail_entry_contexts();
        eprintln!(
            "rep{rep} rail_entry_contexts: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.query_rail_contact([555.0, 695.0, 47.0], 0, None);
        eprintln!(
            "rep{rep} query_rail_contact: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.query_moving_contact([555.0, 695.0, 47.0], 0, None, 1, 1.5);
        eprintln!(
            "rep{rep} query_moving_contact: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let t = Instant::now();
        let _ = world.overlap_obb_all([555.0, 695.0, 47.0], m, [0.8, 0.8, 0.7]);
        eprintln!(
            "rep{rep} overlap_obb_all: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.sweep_obb_rotating_all(
            [555.0, 695.0, 47.0],
            [556.0, 695.0, 47.0],
            m,
            m,
            [0.8, 0.8, 0.7],
        );
        eprintln!(
            "rep{rep} sweep_obb_rotating_all: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );

        let t = Instant::now();
        let _ = world.query_checkpoint_target_candidate([555.0, 695.0, 47.0], 2.0);
        eprintln!(
            "rep{rep} checkpoint_target: {:.3} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
}
