use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
struct Sample {
    tick_ms: f32,
    pos: [f32; 3],
    vel: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Projection {
    path_s: f32,
    cross_track: f32,
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let pose_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: project_pose_to_kap <pose.tsv> <path.kap>")?;
    let kap_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: project_pose_to_kap <pose.tsv> <path.kap>")?;

    let samples = read_samples(&pose_path)?;
    let points = read_kap_points(&kap_path)?;
    if samples.is_empty() {
        return Err("no usable pose samples".to_string());
    }
    if points.len() < 2 {
        return Err("KAP needs at least two points".to_string());
    }

    let segments = build_segments(&points);
    let total_length = segments
        .last()
        .map(|seg| seg.start_s + seg.len)
        .unwrap_or(0.0)
        .max(0.001);
    let mut cursor_s = None;
    let mut progress_s = 0.0;
    let window_s = 120.0;
    println!(
        "tick_ms\tpath_s\tprogress_s\tcross_track\tspeed\tpos_x\tpos_y\tpos_z\tvel_x\tvel_y\tvel_z"
    );
    for sample in samples {
        let projection = if let Some(cursor) = cursor_s {
            project_near(&sample.pos, &segments, cursor, total_length, window_s)
                .unwrap_or_else(|| project(&sample.pos, &segments))
        } else {
            project(&sample.pos, &segments)
        };
        if let Some(cursor) = cursor_s {
            let delta = forward_delta(cursor, projection.path_s, total_length);
            if delta <= window_s {
                progress_s += delta.max(0.0);
            }
        }
        cursor_s = Some(projection.path_s);
        let speed = norm3(sample.vel);
        println!(
            "{:.3}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            sample.tick_ms,
            projection.path_s,
            progress_s,
            projection.cross_track,
            speed,
            sample.pos[0],
            sample.pos[1],
            sample.pos[2],
            sample.vel[0],
            sample.vel[1],
            sample.vel[2]
        );
    }

    Ok(())
}

fn read_samples(path: &Path) -> Result<Vec<Sample>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty pose file")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let tick_i = col(&cols, "tick_ms")?;
    let x_i = col(&cols, "post_pos_x")?;
    let y_i = col(&cols, "post_pos_y")?;
    let z_i = col(&cols, "post_pos_z")?;
    let vx_i = col(&cols, "vel_x").ok();
    let vy_i = col(&cols, "vel_y").ok();
    let vz_i = col(&cols, "vel_z").ok();
    let mut out = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let Some(tick_ms) = parse_field(&fields, tick_i) else {
            continue;
        };
        let Some(x) = parse_field(&fields, x_i) else {
            continue;
        };
        let Some(y) = parse_field(&fields, y_i) else {
            continue;
        };
        let Some(z) = parse_field(&fields, z_i) else {
            continue;
        };
        let vx = vx_i.and_then(|i| parse_field(&fields, i)).unwrap_or(0.0);
        let vy = vy_i.and_then(|i| parse_field(&fields, i)).unwrap_or(0.0);
        let vz = vz_i.and_then(|i| parse_field(&fields, i)).unwrap_or(0.0);
        out.push(Sample {
            tick_ms,
            pos: [x, y, z],
            vel: [vx, vy, vz],
        });
    }
    Ok(out)
}

fn read_kap_points(path: &Path) -> Result<Vec<[f32; 3]>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let kap = kart_fmt::kap::parse(&clear).ok_or_else(|| format!("parse {}", path.display()))?;
    Ok(kap.points)
}

fn col(cols: &[&str], name: &str) -> Result<usize, String> {
    cols.iter()
        .position(|c| *c == name)
        .ok_or_else(|| format!("missing column {name}"))
}

fn parse_field(fields: &[&str], idx: usize) -> Option<f32> {
    fields.get(idx)?.parse().ok()
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

fn project_near(
    point: &[f32; 3],
    segments: &[Segment],
    cursor_s: f32,
    total_length: f32,
    window_s: f32,
) -> Option<Projection> {
    let mut best = Projection {
        path_s: 0.0,
        cross_track: f32::INFINITY,
    };
    let mut found = false;
    for seg in segments {
        let projection = project_on_segment(point, seg);
        if circular_distance(cursor_s, projection.path_s, total_length) > window_s {
            continue;
        }
        if projection.cross_track < best.cross_track {
            best = projection;
            found = true;
        }
    }
    found.then_some(best)
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

fn circular_distance(a: f32, b: f32, total_length: f32) -> f32 {
    let d = (a - b).abs();
    d.min(total_length - d)
}

fn forward_delta(from: f32, to: f32, total_length: f32) -> f32 {
    if to >= from {
        to - from
    } else {
        to + total_length - from
    }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}
