//! Reconstruct the asset-owned triangle corridor for one named scene node.
//!
//! Usage: dump_track_corridor <track.1s> <node-name> [tolerance]

use serde_json::json;
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
        .ok_or("usage: dump_track_corridor <track.1s> <node-name> [tolerance]")?;
    let wanted = args
        .next()
        .ok_or("usage: dump_track_corridor <track.1s> <node-name> [tolerance]")?;
    let tolerance = args
        .next()
        .map(|value| value.parse::<f32>())
        .transpose()
        .map_err(|error| format!("invalid tolerance: {error}"))?
        .unwrap_or(1.0e-3);
    let bytes = std::fs::read(&path).map_err(|error| format!("read: {error}"))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|error| format!("parse {}: {error}", path.display()))?;
    for (node_index, world) in track.flatten() {
        let node = &track.nodes[node_index];
        if node.name != wanted {
            continue;
        }
        let geom = node
            .geom
            .as_ref()
            .ok_or_else(|| format!("{wanted} has no geometry"))?;
        let corridors = kart_fmt::road_corridor::reconstruct(geom, tolerance)
            .into_iter()
            .map(|corridor| {
                json!({
                    "is_chain": corridor.is_chain,
                    "triangle_count": corridor.triangles.len(),
                    "triangles": corridor.triangles,
                    "portal_count": corridor.portals.len(),
                    "portals": corridor.portals.into_iter().map(|portal| json!({
                        "from_triangle": portal.from_triangle,
                        "to_triangle": portal.to_triangle,
                        "midpoint_world": transform(&world, portal.midpoint),
                        "endpoints_world": portal.endpoints.map(|point| transform(&world, point)),
                    })).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        let result = json!({
            "schema": "kartrider.road_triangle_corridor.v1",
            "track_asset": path,
            "node": wanted,
            "tolerance": tolerance,
            "triangle_count": kart_fmt::road_corridor::triangles(geom).len(),
            "component_count": corridors.len(),
            "corridors": corridors,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    Err(format!("node not found: {wanted}"))
}
