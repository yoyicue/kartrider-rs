//! W2 live-world parity replay tool (native shell): reads the case spec from
//! stdin, runs the shared [`kart_track_world::replay::run_live_world_replay`]
//! implementation (the same code the kart-wasm `kart_wasm_run_replay` export
//! executes inside the .wasm), and prints the output JSON document.
//!
//! See `rust/crates/kart-track-world/src/replay.rs` for the full contract
//! (row shape, world-response diff, capability seam, provider seam).

use std::io::Read;

fn main() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("read spec from stdin");
    println!(
        "{}",
        kart_track_world::replay::run_live_world_replay(&input)
    );
}
