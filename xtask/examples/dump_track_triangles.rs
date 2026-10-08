//! Dump transformed triangle vertices and indices for one named scene node.
//!
//! Usage: dump_track_triangles <track.1s> <node-name>

use std::path::PathBuf;

fn transform(matrix: &[f32; 16], point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ]
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: dump_track_triangles <track.1s> <node-name>")?;
    let wanted = args
        .next()
        .ok_or("usage: dump_track_triangles <track.1s> <node-name>")?;
    let bytes = std::fs::read(&path).map_err(|error| format!("read: {error}"))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    println!("triangle\ti0\ti1\ti2\tx0\ty0\tz0\tx1\ty1\tz1\tx2\ty2\tz2");
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        if node.name != wanted {
            continue;
        }
        let geom = node
            .geom
            .as_ref()
            .ok_or_else(|| format!("{wanted} has no geometry"))?;
        let triangles = if geom.strip {
            geom.indices
                .windows(3)
                .enumerate()
                .filter_map(|(index, values)| {
                    let triangle = if index % 2 == 0 {
                        [values[0], values[1], values[2]]
                    } else {
                        [values[1], values[0], values[2]]
                    };
                    (triangle[0] != triangle[1]
                        && triangle[1] != triangle[2]
                        && triangle[0] != triangle[2])
                        .then_some(triangle)
                })
                .collect::<Vec<_>>()
        } else {
            geom.indices
                .chunks_exact(3)
                .map(|values| [values[0], values[1], values[2]])
                .collect::<Vec<_>>()
        };
        for (index, triangle) in triangles.into_iter().enumerate() {
            let points = triangle.map(|vertex| transform(&world, geom.vertices[vertex as usize]));
            println!(
                "{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}\t{:.9}",
                index,
                triangle[0],
                triangle[1],
                triangle[2],
                points[0][0],
                points[0][1],
                points[0][2],
                points[1][0],
                points[1][1],
                points[1][2],
                points[2][0],
                points[2][1],
                points[2][2]
            );
        }
        return Ok(());
    }
    Err(format!("node not found: {wanted}"))
}
