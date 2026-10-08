//! Dump KartObject scene-node transforms for asset-provenance comparisons.
//!
//! Usage: dump_track_nodes <track.1s>

use std::path::PathBuf;

fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: dump_track_nodes <track.1s>")?;
    let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let track = kart_fmt::kart_object::parse(&bytes)
        .map_err(|err| format!("parse {}: {err}", path.display()))?;

    eprintln!("event_count={}", track.event_count);
    let mut parent_by_index = vec![None; track.nodes.len()];
    for (parent, node) in track.nodes.iter().enumerate() {
        for &child in &node.children {
            parent_by_index[child] = Some(parent);
        }
    }
    println!("index\tparent\tclass_stamp\tname\tx\ty\tz\thas_geom\tchild_count\tvertex_count");
    let world_by_index = track
        .flatten()
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();
    for (index, node) in track.nodes.iter().enumerate() {
        let world = world_by_index.get(&index);
        let position = world
            .map(|matrix| [matrix[12], matrix[13], matrix[14]])
            .unwrap_or(node.position);
        println!(
            "{}\t{}\t0x{:08X}\t{}\t{:.9}\t{:.9}\t{:.9}\t{}\t{}\t{}",
            index,
            parent_by_index[index]
                .map(|parent| parent.to_string())
                .unwrap_or_default(),
            node.class_stamp,
            clean(&node.name),
            position[0],
            position[1],
            position[2],
            node.geom.is_some(),
            node.children.len(),
            node.geom
                .as_ref()
                .map(|geom| geom.vertices.len())
                .unwrap_or(0)
        );
    }
    Ok(())
}
