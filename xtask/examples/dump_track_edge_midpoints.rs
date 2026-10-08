//! Dump transformed mesh-edge midpoints for authored course-path provenance.
//!
//! Usage: dump_track_edge_midpoints <track.1s>

use std::collections::BTreeSet;
use std::path::PathBuf;

fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn transform(matrix: &[f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: dump_track_edge_midpoints <track.1s>")?;
    let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|err| format!("parse {}: {err}", path.display()))?;

    println!("node_index\tnode\tvertex_a\tvertex_b\tx\ty\tz");
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        let Some(geom) = &node.geom else { continue };
        let mut edges = BTreeSet::new();
        let triangles: Vec<[u32; 3]> = if geom.strip {
            geom.indices
                .windows(3)
                .enumerate()
                .filter_map(|(index, values)| {
                    let tri = if index % 2 == 0 {
                        [values[0], values[1], values[2]]
                    } else {
                        [values[1], values[0], values[2]]
                    };
                    (tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2]).then_some(tri)
                })
                .collect()
        } else {
            geom.indices
                .chunks_exact(3)
                .map(|values| [values[0], values[1], values[2]])
                .collect()
        };
        for triangle in triangles {
            for [a, b] in [
                [triangle[0], triangle[1]],
                [triangle[1], triangle[2]],
                [triangle[2], triangle[0]],
            ] {
                if a as usize >= geom.vertices.len() || b as usize >= geom.vertices.len() {
                    continue;
                }
                edges.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        for (a, b) in edges {
            let pa = transform(&world, geom.vertices[a as usize]);
            let pb = transform(&world, geom.vertices[b as usize]);
            println!(
                "{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}",
                node_index,
                clean(&node.name),
                a,
                b,
                (pa[0] + pb[0]) * 0.5,
                (pa[1] + pb[1]) * 0.5,
                (pa[2] + pb[2]) * 0.5
            );
        }
    }
    Ok(())
}
