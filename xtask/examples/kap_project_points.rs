use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let kap_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: kap_project_points <path.kap> <label:x:y:z> [label:x:y:z ...]")?;
    let points = read_kap_points(&kap_path)?;
    if points.len() < 2 {
        return Err("KAP needs at least two points".to_string());
    }
    let segments = build_segments(&points);
    let total = segments
        .last()
        .map(|seg| seg.start_s + seg.len)
        .unwrap_or(0.0);
    println!("kap\tlabel\tpath_s\tprogress\tcross_track\tproj_x\tproj_y\tproj_z\ttotal_length");
    for arg in args {
        let (label, p) = parse_labeled_point(&arg)?;
        let projection = project(&p, &segments);
        let at = point_at(&segments, total, projection.path_s).unwrap_or([0.0; 3]);
        println!(
            "{}\t{}\t{:.6}\t{:.9}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            clean(&kap_path.display().to_string()),
            clean(&label),
            projection.path_s,
            if total > 0.0 {
                projection.path_s / total
            } else {
                0.0
            },
            projection.cross_track,
            at[0],
            at[1],
            at[2],
            total
        );
    }
    Ok(())
}

fn parse_labeled_point(arg: &str) -> Result<(String, [f32; 3]), String> {
    let parts: Vec<_> = arg.split(':').collect();
    if parts.len() != 4 {
        return Err(format!("point must be label:x:y:z, got {arg}"));
    }
    Ok((
        parts[0].to_string(),
        [
            parts[1]
                .parse::<f32>()
                .map_err(|_| format!("invalid x in {arg}"))?,
            parts[2]
                .parse::<f32>()
                .map_err(|_| format!("invalid y in {arg}"))?,
            parts[3]
                .parse::<f32>()
                .map_err(|_| format!("invalid z in {arg}"))?,
        ],
    ))
}

fn read_kap_points(path: &Path) -> Result<Vec<[f32; 3]>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let kap = kart_fmt::kap::parse(&clear).ok_or_else(|| format!("parse {}", path.display()))?;
    Ok(kap.points)
}

#[derive(Clone, Copy)]
struct Projection {
    path_s: f32,
    cross_track: f32,
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

fn project(point: &[f32; 3], segments: &[Segment]) -> Projection {
    let mut best = Projection {
        path_s: 0.0,
        cross_track: f32::INFINITY,
    };
    for seg in segments {
        let projection = project_on_segment(point, seg);
        if projection.cross_track < best.cross_track {
            best = projection;
        }
    }
    best
}

fn project_on_segment(point: &[f32; 3], seg: &Segment) -> Projection {
    let vx = seg.b[0] - seg.a[0];
    let vy = seg.b[1] - seg.a[1];
    let wx = point[0] - seg.a[0];
    let wy = point[1] - seg.a[1];
    let denom = vx * vx + vy * vy;
    let t = if denom > 0.000001 {
        ((wx * vx + wy * vy) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let px = seg.a[0] + vx * t;
    let py = seg.a[1] + vy * t;
    Projection {
        path_s: seg.start_s + seg.len * t,
        cross_track: ((point[0] - px).powi(2) + (point[1] - py).powi(2)).sqrt(),
    }
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

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}
