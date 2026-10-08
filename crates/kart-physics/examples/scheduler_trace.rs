//! Scheduler parity trace tool.
//!
//! Reads a JSON spec `{"pairs": [[prev_tick_ms, cur_tick_ms], ...]}` and
//! prints one JSON row per pair with the full schedule result. f32 values are
//! emitted as IEEE-754 bit patterns (`*_bits`) so the Python side can diff
//! bit-exactly.

use std::io::Read;

use kart_physics::scheduler::evaluate_physics_frame_scheduler;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Spec {
    pairs: Vec<[u32; 2]>,
}

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    let spec: Spec = serde_json::from_str(&input).expect("parse spec JSON");
    let rows: Vec<_> = spec
        .pairs
        .iter()
        .map(|[prev, cur]| {
            let s = evaluate_physics_frame_scheduler(*prev, *cur);
            json!({
                "previous_tick_ms": s.previous_tick_ms,
                "current_tick_ms": s.current_tick_ms,
                "raw_frame_delta_ms": s.raw_frame_delta_ms,
                "capped_frame_delta_ms": s.capped_frame_delta_ms,
                "frame_delta_capped": s.frame_delta_capped,
                "substeps_ms": s.substeps_ms,
                "substeps_seconds_bits": s
                    .substeps_seconds_f32
                    .iter()
                    .map(|v| format!("{:016x}", v.to_bits()))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    println!("{}", serde_json::to_string(&rows).unwrap());
}
