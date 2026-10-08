//! `.1s` → glTF (GLB) converter — the Rust counterpart to C# Kart1sToGltf.
//! E1: raw-magic static geometry. E3: character skins + skeletal animation
//! (sibling `fXX.1s` clips, loaded with `--anims`).

use kart_fmt::loader::{load_char_anims, load_kart_anims};
use kart_fmt::{kart_object, raw_model};
use std::path::Path;

fn main() {
    // kart-fmt's parsers panic on a desynced/truncated stream (the documented
    // crate contract), and the KartObject parser deliberately catch_unwinds an
    // optional trailing-MorphTontroller probe *inside* the library — a recovered
    // event that must stay quiet. Since kart-fmt no longer touches the global
    // panic hook (it's the binary's to own), install a quiet hook here so the
    // recovered probe prints nothing, then surface the panic payload ourselves
    // for genuine, unrecovered failures (I/O errors, real parser desyncs).
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
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let anims_flag = args.iter().any(|a| a == "--anims");
    args.retain(|a| a != "--anims");
    // --events: export only the To* event subtrees (normalized roots), not
    // the scene tree (docs/kart-fmt-to-events-handoff-2026-07-30.md E2).
    let events_flag = args.iter().any(|a| a == "--events");
    args.retain(|a| a != "--events");
    let usage = "usage: kart-gltf <model.1s> <out.glb> [tex_dir...] [--anims] [--events]";
    let input = args.first().cloned().expect(usage);
    let output = args.get(1).cloned().expect(usage);
    let in_path = Path::new(&input);
    let dir = in_path.parent().unwrap_or(Path::new("."));
    // Track exports search every given texture dir in order (matching C#'s
    // repeatable --tex-dir); with no dir the input's parent is the only one.
    // Raw-model exports keep the historical single-dir contract (first dir).
    let tex_dirs: Vec<std::path::PathBuf> = if args.len() > 2 {
        args[2..].iter().map(|s| s.into()).collect()
    } else {
        vec![dir.to_path_buf()]
    };
    let tex_dir = tex_dirs[0].clone();

    let data = std::fs::read(&input).expect("read input");
    let name = in_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");

    if raw_model::is_raw_model(&data) {
        let model = raw_model::parse(&data);
        let is_char = model.asset_type == "character";
        let char_anims = if anims_flag && is_char {
            load_char_anims(dir)
        } else {
            Vec::new()
        };
        let kart_anims = if anims_flag && !is_char {
            load_kart_anims(dir)
        } else {
            Vec::new()
        };
        let g = kart_gltf::scene::convert_raw(&model, name, &tex_dir, &char_anims, &kart_anims);
        std::fs::write(&output, g.into_glb()).expect("write glb");
        println!(
            "raw-{} (bones={}, char_anims={}, kart_anims={}) -> {output}",
            model.asset_type,
            model.bones.len(),
            char_anims.len(),
            kart_anims.len()
        );
    } else if kart_object::is_track(&data) {
        match kart_object::parse(&data) {
            Ok(track) => {
                let g = if events_flag {
                    kart_gltf::scene::convert_track_events(&track, name, &tex_dirs, anims_flag)
                } else {
                    kart_gltf::scene::convert_track(&track, name, &tex_dirs, anims_flag)
                };
                std::fs::write(&output, g.into_glb()).expect("write glb");
                println!("track ({} nodes) -> {output}", track.nodes.len());
            }
            Err(e) => {
                eprintln!("track parse error: {e}");
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("unrecognized .1s format");
        std::process::exit(2);
    }
}
