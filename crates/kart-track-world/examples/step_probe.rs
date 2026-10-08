//! step_probe — time the engine's first steps directly (diagnostic for the
//! per-session frame-0 spike). Usage:
//! cargo run -p kart-track-world --release --example step_probe -- \
//!   <track.1s> <sequence.json>

use std::time::Instant;

use kart_physics::engine::{KartRiderVehicleEngine, NumericMode};
use kart_physics::params::VehicleParameters;
use kart_physics::state::VehicleState;
use kart_track_world::world::TrackWorld;

fn main() {
    let mut args = std::env::args().skip(1);
    let track = args
        .next()
        .expect("usage: step_probe <track.1s> <sequence.json>");
    let sequence = args
        .next()
        .expect("usage: step_probe <track.1s> <sequence.json>");
    let seq_text = std::fs::read_to_string(sequence).unwrap();
    let seq: serde_json::Value = serde_json::from_str(&seq_text).unwrap();
    let state: VehicleState = serde_json::from_value(seq["initial_state"].clone()).unwrap();
    let params: VehicleParameters = serde_json::from_value(seq["parameters"].clone()).unwrap();
    let frame_ms = seq["frame_ms"].as_u64().unwrap() as u32;

    for rep in 0..3 {
        let world = TrackWorld::from_path(&track).unwrap();
        let mut engine = KartRiderVehicleEngine::new(&state, NumericMode::Float32).expect("engine");
        for frame in 0..3u32 {
            let t = Instant::now();
            let result = engine.step_raw_frame(
                frame.wrapping_mul(frame_ms),
                &["UP".to_string()],
                &params,
                Some(&world),
                None,
                None,
            );
            eprintln!(
                "rep{rep} step {frame}: {:.3} ms ({})",
                t.elapsed().as_secs_f64() * 1000.0,
                match &result {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("err {e}"),
                }
            );
        }
    }
}
