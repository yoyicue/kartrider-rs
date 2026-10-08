use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

type P3 = [f32; 3];

#[derive(Clone)]
struct TargetFace {
    face_ptr: String,
    metadata_ptr: String,
    surface_id: String,
    rows: usize,
    tri: [P3; 3],
}

#[derive(Default)]
struct MatchStats {
    face_hits: usize,
    best_dist: f32,
    best_node: String,
    best_tri: [P3; 3],
}

#[derive(Clone)]
struct FaceMatch {
    face_ptr: String,
    metadata_ptr: String,
    surface_id: String,
    rows: usize,
    dist: f32,
    track_path: String,
    track_id: String,
    node: String,
    tex: String,
    alpha: String,
    tag: String,
    road_surface: String,
    road_rail: String,
    raw_road_surface: String,
    tri_indices: [u32; 3],
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let probe = PathBuf::from(args.next().ok_or(
        "usage: match_collision_face_to_track <probe_collision_face.tsv> <unpacked-root> [detail-out.tsv]",
    )?);
    let root = PathBuf::from(args.next().ok_or(
        "usage: match_collision_face_to_track <probe_collision_face.tsv> <unpacked-root> [detail-out.tsv]",
    )?);
    let detail_out = args.next().map(PathBuf::from);

    let targets = read_targets(&probe)?;
    if targets.is_empty() {
        return Err("no nonzero target faces in probe_collision_face.tsv".into());
    }

    let mut tracks = Vec::new();
    collect_track_files(&root, &mut tracks);
    tracks.sort();

    let mut best_by_face = BTreeMap::<String, FaceMatch>::new();
    println!("track_path\ttrack_id\tmatched_faces\ttarget_faces\tbest_dist\tbest_node\tbest_face_ptr\tbest_metadata_ptr\tbest_surface_id");
    for track_path in tracks {
        let Ok(bytes) = fs::read(&track_path) else {
            continue;
        };
        let Ok(track) = kart_fmt::kart_object::parse(&bytes) else {
            continue;
        };
        let mut raw_surface_by_node = BTreeMap::<String, String>::new();
        let mut stats = MatchStats {
            best_dist: f32::INFINITY,
            ..Default::default()
        };
        let mut seen = BTreeMap::<String, bool>::new();

        for (node_idx, world) in track.flatten() {
            let node = &track.nodes[node_idx];
            let Some(geom) = &node.geom else { continue };
            for tri_idx in geom_triangles(geom) {
                if tri_idx.iter().any(|&i| i as usize >= geom.vertices.len()) {
                    continue;
                }
                let tri = [
                    xform(&world, geom.vertices[tri_idx[0] as usize]),
                    xform(&world, geom.vertices[tri_idx[1] as usize]),
                    xform(&world, geom.vertices[tri_idx[2] as usize]),
                ];
                for target in &targets {
                    let dist = tri_distance_unordered(tri, target.tri);
                    let should_store = best_by_face
                        .get(&target.face_ptr)
                        .map(|old| dist < old.dist)
                        .unwrap_or(true);
                    if should_store {
                        let raw_road_surface =
                            if let Some(value) = raw_surface_by_node.get(&node.name) {
                                value.clone()
                            } else {
                                let value =
                                    raw_near_road_surface(&bytes, &node.name).unwrap_or_default();
                                raw_surface_by_node.insert(node.name.clone(), value.clone());
                                value
                            };
                        best_by_face.insert(
                            target.face_ptr.clone(),
                            FaceMatch {
                                face_ptr: target.face_ptr.clone(),
                                metadata_ptr: target.metadata_ptr.clone(),
                                surface_id: target.surface_id.clone(),
                                rows: target.rows,
                                dist,
                                track_path: track_path.display().to_string(),
                                track_id: track_id(&track_path),
                                node: node.name.clone(),
                                tex: node.tex.clone().unwrap_or_default(),
                                alpha: alpha_summary(node.alpha),
                                tag: tag_summary(node.extra_tag.as_ref()),
                                road_surface: tag_attr(node.extra_tag.as_ref(), "road", "surface")
                                    .unwrap_or_default(),
                                road_rail: tag_attr(node.extra_tag.as_ref(), "road", "rail")
                                    .unwrap_or_default(),
                                raw_road_surface,
                                tri_indices: tri_idx,
                            },
                        );
                    }
                    if dist < stats.best_dist {
                        stats.best_dist = dist;
                        stats.best_node = node.name.clone();
                        stats.best_tri = tri;
                    }
                    if dist <= 0.05 {
                        seen.insert(target.face_ptr.clone(), true);
                    }
                }
            }
        }

        stats.face_hits = seen.len();
        if stats.face_hits > 0 || stats.best_dist <= 1.0 {
            let best_target = targets
                .iter()
                .min_by(|a, b| {
                    tri_distance_unordered(stats.best_tri, a.tri)
                        .partial_cmp(&tri_distance_unordered(stats.best_tri, b.tri))
                        .unwrap()
                })
                .unwrap();
            println!(
                "{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}",
                track_path.display(),
                track_id(&track_path),
                stats.face_hits,
                targets.len(),
                stats.best_dist,
                clean(&stats.best_node),
                best_target.face_ptr,
                best_target.metadata_ptr,
                best_target.surface_id
            );
        }
    }
    if let Some(path) = detail_out {
        write_detail(&path, best_by_face.into_values())?;
    }
    Ok(())
}

fn read_targets(path: &Path) -> Result<Vec<TargetFace>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty tsv")?;
    let cols: Vec<&str> = header.split('\t').collect();
    let idx = |name: &str| -> Result<usize, String> {
        cols.iter()
            .position(|&c| c == name)
            .ok_or_else(|| format!("missing column {name}"))
    };
    let c_face = idx("face_ptr")?;
    let c_meta = idx("metadata_ptr")?;
    let c_sid = idx("surface_id")?;
    let c = |name: &str| idx(name);
    let vi = [
        [c("v0_x")?, c("v0_y")?, c("v0_z")?],
        [c("v1_x")?, c("v1_y")?, c("v1_z")?],
        [c("v2_x")?, c("v2_y")?, c("v2_z")?],
    ];

    let mut by_face = BTreeMap::<String, TargetFace>::new();
    for line in lines {
        let vals: Vec<&str> = line.split('\t').collect();
        if vals.len() <= c_sid || vals[c_sid] == "0" {
            continue;
        }
        let face_ptr = vals[c_face].to_string();
        by_face
            .entry(face_ptr.clone())
            .and_modify(|target| target.rows += 1)
            .or_insert_with(|| TargetFace {
                face_ptr,
                metadata_ptr: vals[c_meta].to_string(),
                surface_id: vals[c_sid].to_string(),
                rows: 1,
                tri: vi
                    .map(|is| is.map(|i| vals.get(i).and_then(|v| v.parse().ok()).unwrap_or(0.0))),
            });
    }
    Ok(by_face.into_values().collect())
}

fn collect_track_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.is_dir() {
            collect_track_files(&p, out);
        } else if p.file_name().and_then(|s| s.to_str()) == Some("track.1s") {
            out.push(p);
        }
    }
}

fn geom_triangles(g: &kart_fmt::kart_object::Geom) -> Vec<[u32; 3]> {
    if g.strip {
        (0..g.indices.len().saturating_sub(2))
            .filter_map(|k| {
                let (a, b, c) = (g.indices[k], g.indices[k + 1], g.indices[k + 2]);
                if a == b || b == c || a == c {
                    None
                } else if k % 2 == 0 {
                    Some([a, b, c])
                } else {
                    Some([b, a, c])
                }
            })
            .collect()
    } else {
        g.indices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }
}

fn xform(m: &[f32; 16], p: P3) -> P3 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

fn tri_distance_unordered(a: [P3; 3], b: [P3; 3]) -> f32 {
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    perms
        .iter()
        .map(|p| (0..3).map(|i| dist2(a[i], b[p[i]])).sum::<f32>().sqrt())
        .fold(f32::INFINITY, f32::min)
}

fn dist2(a: P3, b: P3) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

fn track_id(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn alpha_summary(alpha: Option<kart_fmt::kart_object::AlphaInfo>) -> String {
    match alpha {
        Some(a) => format!(
            "blend={},alpha_test={},ref={}",
            a.use_blend as u8, a.use_alpha_test as u8, a.alpha_ref
        ),
        None => String::new(),
    }
}

fn write_detail(path: &Path, rows: impl Iterator<Item = FaceMatch>) -> Result<(), String> {
    let mut out = String::from(
        "face_ptr\trows\tmetadata_ptr\tsurface_id\tdist\ttrack_id\ttrack_path\tnode\ttex\talpha\ttag\troad_surface\troad_rail\traw_road_surface\ttri_i0\ttri_i1\ttri_i2\n",
    );
    for r in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.face_ptr,
            r.rows,
            r.metadata_ptr,
            r.surface_id,
            r.dist,
            clean(&r.track_id),
            clean(&r.track_path),
            clean(&r.node),
            clean(&r.tex),
            clean(&r.alpha),
            clean(&r.tag),
            clean(&r.road_surface),
            clean(&r.road_rail),
            clean(&r.raw_road_surface),
            r.tri_indices[0],
            r.tri_indices[1],
            r.tri_indices[2],
        ));
    }
    fs::write(path, out).map_err(|e| format!("write {}: {e}", path.display()))
}

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

fn raw_near_road_surface(bytes: &[u8], node_name: &str) -> Option<String> {
    let node_key = utf16_len_string(node_name);
    let pos = bytes
        .windows(node_key.len())
        .position(|window| window == node_key.as_slice())?;
    let window = &bytes[pos..bytes.len().min(pos + 700)];
    let surface_key = utf16_len_string("surface");
    let surface_pos = window
        .windows(surface_key.len())
        .position(|candidate| candidate == surface_key.as_slice())?
        + surface_key.len();
    read_len_utf16(window, surface_pos)
}

fn utf16_len_string(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + text.len() * 2);
    out.extend_from_slice(&(text.encode_utf16().count() as i32).to_le_bytes());
    for unit in text.encode_utf16() {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out
}

fn read_len_utf16(bytes: &[u8], offset: usize) -> Option<String> {
    if offset + 4 > bytes.len() {
        return None;
    }
    let len = i32::from_le_bytes(bytes[offset..offset + 4].try_into().ok()?);
    if !(0..=80).contains(&len) {
        return None;
    }
    let start = offset + 4;
    let end = start + len as usize * 2;
    if end > bytes.len() {
        return None;
    }
    let units = bytes[start..end]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    String::from_utf16(&units).ok()
}

fn tag_summary(tag: Option<&kart_fmt::kart_object::BinaryXmlTag>) -> String {
    let Some(tag) = tag else { return String::new() };
    if tag.attrs.is_empty() {
        return tag.name.clone();
    }
    let attrs = tag
        .attrs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", tag.name, attrs)
}

fn tag_attr(
    tag: Option<&kart_fmt::kart_object::BinaryXmlTag>,
    tag_name: &str,
    attr_name: &str,
) -> Option<String> {
    let tag = tag?;
    if tag.name == tag_name {
        if let Some((_, value)) = tag.attrs.iter().find(|(key, _)| key == attr_name) {
            return Some(value.clone());
        }
    }
    for child in &tag.children {
        if let Some(value) = tag_attr(Some(child), tag_name, attr_name) {
            return Some(value);
        }
    }
    None
}
