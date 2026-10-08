use std::cmp::Ordering;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
struct P3 {
    x: f32,
    y: f32,
    z: f32,
}

struct Match {
    track_id: String,
    path: String,
    points: usize,
    avg: f64,
    p50: f64,
    p90: f64,
    max: f64,
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let pose_path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: match_pose_to_kap <probe_pose.tsv> <aiPath-dir>")?;
    let kap_dir = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: match_pose_to_kap <probe_pose.tsv> <aiPath-dir>")?;

    let pose = read_pose_samples(&pose_path)?;
    if pose.is_empty() {
        return Err("no usable pose samples".into());
    }

    let mut matches = Vec::new();
    visit(&kap_dir, &mut |path| {
        if path.extension().and_then(|s| s.to_str()) != Some("kap") {
            return;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        let Ok(clear) = kart_rho::encoded_block::decode_len_prefixed(&bytes) else {
            return;
        };
        let Some(kap) = kart_fmt::kap::parse(&clear) else {
            return;
        };
        if kap.points.is_empty() {
            return;
        }
        let mut ds: Vec<f64> = pose
            .iter()
            .map(|p| nearest_distance(*p, &kap.points))
            .collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let avg = ds.iter().sum::<f64>() / ds.len() as f64;
        let p50 = percentile(&ds, 0.50);
        let p90 = percentile(&ds, 0.90);
        let max = *ds.last().unwrap_or(&0.0);
        matches.push(Match {
            track_id: kap_track_id(path).unwrap_or_default(),
            path: path.display().to_string(),
            points: kap.points.len(),
            avg,
            p50,
            p90,
            max,
        });
    })?;

    matches.sort_by(|a, b| a.avg.partial_cmp(&b.avg).unwrap_or(Ordering::Equal));
    println!("rank\ttrack_id\tavg_dist\tp50_dist\tp90_dist\tmax_dist\tkap_points\tpath");
    for (i, m) in matches.iter().take(40).enumerate() {
        println!(
            "{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}",
            i + 1,
            m.track_id,
            m.avg,
            m.p50,
            m.p90,
            m.max,
            m.points,
            m.path
        );
    }
    Ok(())
}

fn read_pose_samples(path: &Path) -> Result<Vec<P3>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty pose file")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let ix = col(&cols, "post_pos_x")?;
    let iy = col(&cols, "post_pos_y")?;
    let iz = col(&cols, "post_pos_z")?;
    let mut out = Vec::new();
    for (idx, line) in lines.enumerate() {
        if idx % 32 != 0 {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let Some(x) = parse_field(&fields, ix) else {
            continue;
        };
        let Some(y) = parse_field(&fields, iy) else {
            continue;
        };
        let Some(z) = parse_field(&fields, iz) else {
            continue;
        };
        out.push(P3 { x, y, z });
    }
    Ok(out)
}

fn col(cols: &[&str], name: &str) -> Result<usize, String> {
    cols.iter()
        .position(|c| *c == name)
        .ok_or_else(|| format!("missing column {name}"))
}

fn parse_field(fields: &[&str], idx: usize) -> Option<f32> {
    fields.get(idx)?.parse().ok()
}

fn nearest_distance(p: P3, points: &[[f32; 3]]) -> f64 {
    let mut best = f64::INFINITY;
    for q in points {
        let dx = (p.x - q[0]) as f64;
        let dy = (p.y - q[1]) as f64;
        let dz = (p.z - q[2]) as f64;
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < best {
            best = d;
        }
    }
    best
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn visit(dir: &Path, f: &mut impl FnMut(&Path)) -> Result<(), String> {
    for entry in std::fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            visit(&path, f)?;
        } else {
            f(&path);
        }
    }
    Ok(())
}

fn kap_track_id(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let mut parts: Vec<&str> = stem.split('_').collect();
    let last = parts.last().copied()?;
    if last.parse::<u32>().is_ok() {
        parts.pop();
    }
    if parts.last() == Some(&"rvs") {
        parts.pop();
    }
    Some(parts.join("_"))
}
