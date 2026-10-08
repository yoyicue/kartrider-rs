//! FNV-1a checksum of a parsed kart vehicle anim clip: per channel anim_id +
//! every keyframe's time/pos/rot_euler/scale. Matches the C# reference dumper.

use kart_fmt::kart_anim;

fn fnv(h: &mut u64, b: &[u8]) {
    for &x in b {
        *h ^= x as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}
fn ff(h: &mut u64, v: f32) {
    fnv(h, &v.to_le_bytes());
}
fn fi(h: &mut u64, v: i32) {
    fnv(h, &v.to_le_bytes());
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_kart_anim <fXX.1s>");
    let data = std::fs::read(&path).expect("read");
    let nm = std::path::Path::new(&path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("clip");
    let clip = match kart_anim::parse(&data, nm) {
        Some(c) => c,
        None => {
            eprintln!("not a kart anim");
            std::process::exit(1);
        }
    };
    let mut h = 0xcbf29ce484222325u64;
    let mut animated = 0;
    let mut total_kf = 0;
    for c in &clip.channels {
        fi(&mut h, c.anim_id);
        if c.is_animated() {
            animated += 1;
        }
        for kf in &c.keyframes {
            total_kf += 1;
            fi(&mut h, kf.time_ms);
            for v in kf
                .pos
                .iter()
                .chain(kf.rot_euler.iter())
                .chain(kf.scale.iter())
            {
                ff(&mut h, *v);
            }
        }
    }
    println!(
        "channels={} animated={} frames={} kf={} dur={} hash={:016x}",
        clip.channels.len(),
        animated,
        clip.num_frames,
        total_kf,
        clip.duration_ms,
        h
    );
}
