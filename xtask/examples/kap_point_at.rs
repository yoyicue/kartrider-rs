use std::path::PathBuf;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let kap_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: kap_point_at <path.kap> <path_s> [path_s ...]")?;
    let s_values: Vec<f32> = args
        .map(|arg| {
            arg.parse::<f32>()
                .map_err(|_| format!("invalid path_s: {arg}"))
        })
        .collect::<Result<_, _>>()?;
    if s_values.is_empty() {
        return Err("usage: kap_point_at <path.kap> <path_s> [path_s ...]".to_string());
    }
    let bytes =
        std::fs::read(&kap_path).map_err(|e| format!("read {}: {e}", kap_path.display()))?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|e| format!("decode {}: {e}", kap_path.display()))?;
    let kap =
        kart_fmt::kap::parse(&clear).ok_or_else(|| format!("parse {}", kap_path.display()))?;
    let segments = build_segments(&kap.points);
    let total = segments
        .last()
        .map(|seg| seg.start_s + seg.len)
        .unwrap_or(0.0);
    println!("path_s\tx\ty\tz\ttotal_length");
    for s in s_values {
        let p = point_at(&segments, total, s).ok_or_else(|| format!("path_s out of range: {s}"))?;
        println!("{s:.6}\t{:.6}\t{:.6}\t{:.6}\t{total:.6}", p[0], p[1], p[2]);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Segment {
    a: [f32; 3],
    b: [f32; 3],
    start_s: f32,
    len: f32,
}

fn build_segments(points: &[[f32; 3]]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut s = 0.0;
    for pair in points.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let len = dist2(a, b).max(0.001);
        out.push(Segment {
            a,
            b,
            start_s: s,
            len,
        });
        s += len;
    }
    out
}

fn point_at(segments: &[Segment], total: f32, mut path_s: f32) -> Option<[f32; 3]> {
    if segments.is_empty() {
        return None;
    }
    if total > 0.0 {
        path_s = path_s.rem_euclid(total);
    }
    for seg in segments {
        if path_s <= seg.start_s + seg.len {
            let t = ((path_s - seg.start_s) / seg.len).clamp(0.0, 1.0);
            return Some([
                seg.a[0] + (seg.b[0] - seg.a[0]) * t,
                seg.a[1] + (seg.b[1] - seg.a[1]) * t,
                seg.a[2] + (seg.b[2] - seg.a[2]) * t,
            ]);
        }
    }
    segments.last().map(|seg| seg.b)
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}
