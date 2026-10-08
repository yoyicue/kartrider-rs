use std::env;
use std::fs;
use std::path::{Path, PathBuf};

type P3 = [f32; 3];

#[derive(Default)]
struct Bounds {
    tris: usize,
    vertices: usize,
    min: P3,
    max: P3,
    sum: P3,
}

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let track_path = PathBuf::from(
        args.next()
            .ok_or("usage: list_track_surface_bounds <track.1s> <surface>")?,
    );
    let surface_filter = args
        .next()
        .ok_or("usage: list_track_surface_bounds <track.1s> <surface>")?;

    let bytes =
        fs::read(&track_path).map_err(|err| format!("read {}: {err}", track_path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|err| format!("parse {}: {err}", track_path.display()))?;

    println!(
        "track_path\ttrack_id\tnode\ttex\troad_surface\tsurface_id\ttris\tvertices\tmin_x\tmin_y\tmin_z\tmax_x\tmax_y\tmax_z\tcentroid_x\tcentroid_y\tcentroid_z"
    );
    for (node_idx, world) in track.flatten() {
        let node = &track.nodes[node_idx];
        let Some(surface) = tag_attr(node.extra_tag.as_ref(), "road", "surface") else {
            continue;
        };
        if surface != surface_filter {
            continue;
        }
        let Some(geom) = &node.geom else { continue };
        let mut bounds = Bounds::new();
        for tri_idx in geom_triangles(geom) {
            if tri_idx.iter().any(|&i| i as usize >= geom.vertices.len()) {
                continue;
            }
            bounds.tris += 1;
            for idx in tri_idx {
                bounds.add(xform(&world, geom.vertices[idx as usize]));
            }
        }
        if bounds.vertices == 0 {
            continue;
        }
        let c = bounds.centroid();
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            clean(&track_path.display().to_string()),
            clean(&track_id(&track_path)),
            clean(&node.name),
            clean(node.tex.as_deref().unwrap_or("")),
            clean(&surface),
            surface_id(&surface),
            bounds.tris,
            bounds.vertices,
            bounds.min[0],
            bounds.min[1],
            bounds.min[2],
            bounds.max[0],
            bounds.max[1],
            bounds.max[2],
            c[0],
            c[1],
            c[2],
        );
    }
    Ok(())
}

impl Bounds {
    fn new() -> Self {
        Self {
            tris: 0,
            vertices: 0,
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            sum: [0.0; 3],
        }
    }

    fn add(&mut self, p: P3) {
        self.vertices += 1;
        for (idx, value) in p.into_iter().enumerate() {
            self.min[idx] = self.min[idx].min(value);
            self.max[idx] = self.max[idx].max(value);
            self.sum[idx] += value;
        }
    }

    fn centroid(&self) -> P3 {
        let denom = self.vertices.max(1) as f32;
        [
            self.sum[0] / denom,
            self.sum[1] / denom,
            self.sum[2] / denom,
        ]
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

fn xform(m: &[f32; 16], p: P3) -> P3 {
    [
        m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
        m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
        m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14],
    ]
}

fn track_id(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn surface_id(text: &str) -> u32 {
    let mut a = 0u32;
    let mut b = 0u32;
    for unit in text.encode_utf16() {
        for byte in unit.to_le_bytes() {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
    }
    match (b << 16) | a {
        0 => 1,
        0xffff_ffff => 2,
        value => value,
    }
}

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}
