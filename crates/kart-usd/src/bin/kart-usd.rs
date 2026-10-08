//! `.1s` → USD ASCII (`.usda`) converter — the oracle-local export format.

use kart_fmt::{kart_object, raw_model};
use std::path::Path;

fn main() {
    // Own the process panic hook so kart-fmt's internally-recovered
    // trailing-MorphTontroller probe stays quiet (the library no longer touches
    // the global hook), then surface the payload of any genuine, unrecovered
    // panic — I/O errors or a real parser desync. See kart-gltf's main.
    std::panic::set_hook(Box::new(|_| {}));
    if let Err(payload) = std::panic::catch_unwind(run) {
        let msg = payload
            .downcast_ref::<String>()
            .map(|s| s.as_str())
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("malformed or unsupported .1s layout");
        eprintln!("error: {msg}");
        std::process::exit(1);
    }
}

fn run() {
    let mut a = std::env::args().skip(1);
    let usage = "usage: kart-usd <model-or-track.1s> <out.usda> [tex_dir]";
    let input = a.next().expect(usage);
    let output = a.next().expect(usage);
    let in_path = Path::new(&input);
    let tex_dir = a
        .next()
        .map(|s| s.into())
        .unwrap_or_else(|| in_path.parent().unwrap_or(Path::new(".")).to_path_buf());
    let data = std::fs::read(&input).expect("read");
    let name = in_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("asset");

    let usda = if raw_model::is_raw_model(&data) {
        let model = raw_model::parse(&data);
        kart_usd::raw_model_usda(&model, name, &tex_dir)
    } else if data.len() >= 2 && data[0] == 0xAA && data[1] == 0x47 {
        match kart_object::parse(&data) {
            Ok(track) => kart_usd::track_usda(&track, name, &tex_dir),
            Err(e) => {
                eprintln!("parse error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("unrecognized .1s format");
        std::process::exit(2);
    };
    std::fs::write(&output, usda).expect("write usda");
    println!("usda -> {output}");
}
