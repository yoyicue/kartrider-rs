//! track_anim_export — dump the authoritative PRS animation channels of a
//! track.1s as JSON (offline input for kart-three's dynamic-object playback
//! and the D3 phase check; see docs/kart-three-dynamic-objects-handoff-2026-07-30.md).
//!
//! The glTF track animation keeps key-time quaternion projections, which
//! collapse full revolutions between sparse keys (gears with 0 -> 2pi keys
//! interpolate as frozen). The authoritative runtime sampling
//! (`kart_object` `sampled_local`) evaluates the Euler axis curves with
//! `channel_time` wrap semantics, so the demo drives playback from THIS
//! table instead of the glTF animation channels.
//!
//! Usage: track_anim_export <track.1s> > track_anim.json

use std::path::Path;

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("usage: track_anim_export <track.1s>");
    let data = std::fs::read(Path::new(&input)).expect("read track");
    let track = kart_fmt::kart_object::parse(&data).expect("parse track");

    let mut nodes = Vec::new();
    for (index, node) in track.nodes.iter().enumerate() {
        let Some(prs) = &node.prs else { continue };
        if !prs.has_anim() {
            continue;
        }
        let range = |offset: usize| {
            prs.controller_tail
                .map(|tail| [tail[offset], tail[offset + 1]])
        };
        nodes.push(serde_json::json!({
            "index": index,
            "name": node.name,
            "repeat": prs.controller_base.map(|base| base[1] == 0).unwrap_or(false),
            "pos_range": range(0),
            "rot_range": range(2),
            "scale_range": range(4),
            "pos_keys": prs.pos.iter().map(|&(t, v)| serde_json::json!([t, v[0], v[1], v[2]])).collect::<Vec<_>>(),
            "scale_keys": prs.scale.iter().map(|&(t, v)| serde_json::json!([t, v[0], v[1], v[2]])).collect::<Vec<_>>(),
            "rot_quat_keys": prs.rot.iter().map(|&(t, q)| serde_json::json!([t, q[0], q[1], q[2], q[3]])).collect::<Vec<_>>(),
            "rot_axes": prs.rot_axes.as_ref().map(|axes| {
                axes.iter().map(|axis| {
                    serde_json::json!({
                        "kf_type": axis.kf_type,
                        "keys": axis.keys.iter().map(|k| serde_json::json!([k.0, k.1, k.2, k.3])).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>()
            }),
        }));
    }
    let doc = serde_json::json!({
        "schema": "track_anim_channels_v1",
        "source": input,
        "nodes": nodes,
    });
    println!("{}", serde_json::to_string(&doc).expect("encode"));
}
