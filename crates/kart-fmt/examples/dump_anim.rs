use kart_fmt::anim_clip;
fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_anim <fXX.1s>");
    let d = std::fs::read(&path).unwrap();
    let clip = anim_clip::parse(&d, "x").expect("not a char anim");
    let mut h: u64 = 0xcbf29ce484222325;
    let fnv = |x: u64, h: &mut u64| {
        *h = (*h ^ x).wrapping_mul(0x100000001b3);
    };
    for t in &clip.tracks {
        fnv(t.bone_index as u32 as u64, &mut h);
        for (tm, v) in &t.pos {
            fnv(*tm as u32 as u64, &mut h);
            for f in v {
                fnv(f.to_bits() as u64, &mut h);
            }
        }
        for (tm, q) in &t.rot {
            fnv(*tm as u32 as u64, &mut h);
            for f in q {
                fnv(f.to_bits() as u64, &mut h);
            }
        }
        for (tm, v) in &t.scale {
            fnv(*tm as u32 as u64, &mut h);
            for f in v {
                fnv(f.to_bits() as u64, &mut h);
            }
        }
    }
    println!(
        "dur={} tracks={} hash={:016x}",
        clip.duration_ms,
        clip.tracks.len(),
        h
    );
}
