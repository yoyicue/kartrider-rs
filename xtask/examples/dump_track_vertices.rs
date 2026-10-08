//! Dump every transformed KartObject mesh vertex as stable TSV.
//!
//! Usage: dump_track_vertices <track.1s>

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
        .ok_or("usage: dump_track_vertices <track.1s>")?;
    let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|err| format!("parse {}: {err}", path.display()))?;

    println!("node_index\tnode\tvertex_index\tx\ty\tz");
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        let Some(geom) = &node.geom else { continue };
        for (vertex_index, &local) in geom.vertices.iter().enumerate() {
            let point = transform(&world, local);
            println!(
                "{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}",
                node_index,
                clean(&node.name),
                vertex_index,
                point[0],
                point[1],
                point[2]
            );
        }
    }
    Ok(())
}
