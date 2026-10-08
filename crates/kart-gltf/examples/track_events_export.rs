//! track_events_export — offline E3 export of the To* event placements and
//! animation channels for kart-three (docs/kart-fmt-to-events-handoff-2026-07-30.md).
//!
//! For every ToDummy/ToMovableObject event root with a typed placement, emit
//! the placement (raw game-space transform + the glb-space instancing matrix
//! from kart_gltf::scene::track_event_placement_glb) and the matching '#'-prefixed
//! Relement subtree root name (the node to instance from the --events glb).
//! Each placement also carries its subtree's authoritative PRS channel data in
//! the exact track_anim.json shape (kart-three/track_anim.js consumes it; the
//! glTF key-time quaternion projection collapses full revolutions, see
//! track_anim_export).
//!
//! Usage: track_events_export <track.1s> > track_events.json

use kart_fmt::kart_object::{PrsAnim, Track};
use serde_json::json;

fn prs_json(index: usize, name: &str, prs: &PrsAnim) -> serde_json::Value {
    let range = |offset: usize| {
        prs.controller_tail
            .map(|tail| [tail[offset], tail[offset + 1]])
    };
    json!({
        "index": index,
        "name": name,
        "repeat": prs.controller_base.map(|base| base[1] == 0).unwrap_or(false),
        "pos_range": range(0),
        "rot_range": range(2),
        "scale_range": range(4),
        "pos_keys": prs.pos.iter().map(|&(t, v)| json!([t, v[0], v[1], v[2]])).collect::<Vec<_>>(),
        "scale_keys": prs.scale.iter().map(|&(t, v)| json!([t, v[0], v[1], v[2]])).collect::<Vec<_>>(),
        "rot_quat_keys": prs.rot.iter().map(|&(t, q)| json!([t, q[0], q[1], q[2], q[3]])).collect::<Vec<_>>(),
        "rot_axes": prs.rot_axes.as_ref().map(|axes| {
            axes.iter().map(|axis| {
                json!({
                    "kf_type": axis.kf_type,
                    "keys": axis.keys.iter().map(|k| json!([k.0, k.1, k.2, k.3])).collect::<Vec<_>>(),
                })
            }).collect::<Vec<_>>()
        }),
    })
}

/// Collect the subtree's PRS-animated nodes (depth-first from the root).
fn subtree_anim_nodes(track: &Track, root: usize, out: &mut Vec<serde_json::Value>) {
    let n = &track.nodes[root];
    if let Some(prs) = &n.prs {
        if prs.has_anim() {
            out.push(prs_json(root, &n.name, prs));
        }
    }
    for &c in &n.children {
        subtree_anim_nodes(track, c, out);
    }
}

/// The event's visual subtree root: the Relement-family event root named
/// `'#' + <event name>` (factory_I02 uses exact or capitalized-name matches:
/// 'mo_dummy01-…' ↔ '#mo_Dummy01-…').
fn find_subtree_root<'a>(track: &'a Track, name: &str) -> Option<&'a str> {
    let wanted = format!("#{name}").to_lowercase();
    track.runtime_event_roots().find_map(|r| {
        let i = r.node?;
        let node_name = track.nodes[i].name.as_str();
        node_name.to_lowercase().eq(&wanted).then_some(node_name)
    })
}

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("usage: track_events_export <track.1s>");
    let data = std::fs::read(&input).expect("read track");
    let track = kart_fmt::kart_object::parse(&data).expect("parse track");

    let mut placements = Vec::new();
    for root in track.runtime_event_roots() {
        let Some(p) = &root.placement else { continue };
        let object_type = p.property.as_ref().and_then(|tag| {
            tag.children
                .iter()
                .find(|c| c.name == "object")
                .and_then(|c| c.attrs.iter().find(|(k, _)| k == "type"))
                .map(|(_, v)| v.clone())
        });
        let node = find_subtree_root(&track, &p.name);
        let mut anim_nodes = Vec::new();
        if let Some(node_name) = node {
            let subtree = track
                .runtime_event_roots()
                .filter_map(|r| r.node)
                .find(|&i| track.nodes[i].name == node_name)
                .expect("subtree root arena index");
            subtree_anim_nodes(&track, subtree, &mut anim_nodes);
        }
        placements.push(json!({
            "event_index": root.index,
            "kind": root.class_name.unwrap_or("?"),
            "name": p.name,
            "object_type": object_type,
            "physics": {
                "enabled": root.index == 1743
                    && p.name == "mo_dummy01-톱니바퀴a",
                "disposition": if root.index == 1743
                    && p.name == "mo_dummy01-톱니바퀴a"
                {
                    "CONTACT_DYNAMIC_OBSTACLE_P1_P5_CLOSED"
                } else {
                    "VISUAL_ONLY_UNADJUDICATED"
                },
            },
            "node": node,
            "matrix": kart_gltf::scene::track_event_placement_glb(
                &p.transform, p.position, p.scale),
            "position": p.position,
            "transform": p.transform,
            "scale": p.scale,
            "anim_nodes": anim_nodes,
        }));
    }

    let doc = json!({
        "schema": "track_events_v1",
        "source": input,
        "event_count": track.event_count,
        "decoded_events": track.events.len(),
        "top_level_roots": track.event_roots.len(),
        "post_event_runtime_roots": track.post_event_roots.len(),
        "phase_2_physics": {
            "as_of": "2026-07-31",
            "verdict": "SCOPED_EVENT_PHYSICS_ENABLED",
            "enabled_event_indices": [1743],
            "authority": [
                "analysis/input_response/to_event_big_gear_adjudication_20260731.json",
                "analysis/input_response/to_event_big_gear_p2_p5_acceptance_20260731.json",
                "analysis/input_response/to_event_spring_pad_adjudication_20260731.json",
                "analysis/input_response/to_event_gate_adjudication_20260731.json",
            ],
        },
        "ponytail": [
            "trigger conditions are NOT evaluated: every decoded event object is permanently visible; event 1743 is scoped to factory_I02's always-present rotating obstacle",
            "only factory_I02 event 1743 is physics-enabled through its P1-closed dynamic CONTACT query; every other event placement remains visual-only and fail-safe",
            "placements come from the event bodies' authored transforms; the '#'-prefixed subtree roots carry the identical transform (factory_I02 verified equal), so the --events glb keeps subtree roots normalized and the page instances them at these matrices",
        ],
        "placements": placements,
    });
    println!("{}", serde_json::to_string(&doc).expect("encode"));
}
