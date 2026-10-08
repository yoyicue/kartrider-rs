//! Discover embedded road navigation-record vectors in a `track.1s`.
//!
//! Usage: dump_track_nav <track.1s>

use serde_json::json;
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: dump_track_nav <track.1s>")?;
    let data = std::fs::read(&path).map_err(|error| format!("read: {error}"))?;
    let vectors = kart_fmt::track_nav::discover_record_vectors(&data);
    let roads = kart_fmt::track_nav::discover_to_roads(&data);
    let result = json!({
        "schema": "kartrider.track_nav_record_vectors.v1",
        "track_asset": path,
        "vector_count": vectors.len(),
        "record_count": vectors.iter().map(|vector| vector.records.len()).sum::<usize>(),
        "to_road_count": roads.len(),
        "to_roads": roads.into_iter().map(|road| json!({
            "offset": road.offset,
            "offset_hex": format!("0x{:X}", road.offset),
            "name": road.name,
            "flags": road.flags,
            "element_count": road.elements.len(),
            "elements": road.elements.into_iter().enumerate().map(|(ordinal, element)| json!({
                "ordinal": ordinal,
                "marker": element.marker,
                "boundary_count": element.boundary.len(),
                "triangle_count": element.triangles.len(),
                "surface": element.surface,
                "surface_triangle_count": element.surface_triangles.len(),
                "record_count": element.records.len(),
                "records": element.records.into_iter().map(|record| json!({
                    "position": record.position,
                    "secondary": record.secondary,
                    "normal_or_side": record.normal_or_side,
                })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "vectors": vectors.into_iter().map(|vector| json!({
            "offset": vector.offset,
            "offset_hex": format!("0x{:X}", vector.offset),
            "record_count": vector.records.len(),
            "records": vector.records.into_iter().map(|record| json!({
                "position": record.position,
                "secondary": record.secondary,
                "normal_or_side": record.normal_or_side,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}
