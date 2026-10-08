//! dump_event_tree — print the typed event-section tree of a track.1s (E2 of
//! docs/kart-fmt-to-events-handoff-2026-07-30.md): every top-level event root,
//! its ToDummy/ToMovableObject placement when present, and for Relement
//! subtree roots the recursive node listing (name, local position, geometry
//! size, texture, PRS animation presence).
//!
//! Usage: dump_event_tree <track.1s> [track.1s...]

use kart_fmt::kart_object::{Node, Track};

fn fmt_pos(p: &[f32; 3]) -> String {
    format!("({:.2}, {:.2}, {:.2})", p[0], p[1], p[2])
}

fn walk(track: &Track, i: usize, depth: usize) {
    let n: &Node = &track.nodes[i];
    let geom = n
        .geom
        .as_ref()
        .map(|g| format!("verts={} tris={}", g.vertices.len(), g.indices.len() / 3))
        .unwrap_or_default();
    let tex = n.tex.as_deref().unwrap_or("");
    let prs = n
        .prs
        .as_ref()
        .filter(|p| p.has_anim())
        .map(|p| {
            format!(
                "PRS(pos={} rot={} axes={} scale={})",
                p.pos.len(),
                p.rot.len(),
                p.rot_axes.is_some(),
                p.scale.len()
            )
        })
        .unwrap_or_default();
    let uv = if n.uv_curves.is_empty() {
        String::new()
    } else {
        format!("uv_curves={}", n.uv_curves.len())
    };
    println!(
        "{}{} pos={} scale={} {} {} {} {}",
        "  ".repeat(depth),
        if n.name.is_empty() {
            "<unnamed>"
        } else {
            &n.name
        },
        fmt_pos(&n.position),
        fmt_pos(&n.scale),
        geom,
        tex,
        prs,
        uv
    );
    for &c in &n.children {
        walk(track, c, depth + 1);
    }
}

fn main() {
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read track");
        let track = kart_fmt::kart_object::parse(&data).expect("parse track");
        println!(
            "== {path}: {} declared event roots + {} post-event runtime roots",
            track.event_roots.len(),
            track.post_event_roots.len()
        );
        for root in track.runtime_event_roots() {
            let class = root.class_name.unwrap_or("?");
            if let Some(p) = &root.placement {
                let prop = p
                    .property
                    .as_ref()
                    .map(|t| {
                        t.children
                            .iter()
                            .map(|c| {
                                format!(
                                    "{}:{}",
                                    c.name,
                                    c.attrs
                                        .iter()
                                        .map(|(k, v)| format!("{k}={v}"))
                                        .collect::<Vec<_>>()
                                        .join(",")
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" | ")
                    })
                    .unwrap_or_default();
                println!(
                    "[{}] {} '{}' pos={} scale={} tf={:?} {}",
                    root.index,
                    class,
                    p.name,
                    fmt_pos(&p.position),
                    fmt_pos(&p.scale),
                    p.transform,
                    prop
                );
            } else if let Some(node) = root.node {
                println!("[{}] {} subtree:", root.index, class);
                walk(&track, node, 1);
            } else {
                println!("[{}] {} (opaque)", root.index, class);
            }
        }
    }
}
