//! uv_anim_export — dump TexProperty UV controller curves (with kf_type and
//! tangents) from a track.1s as JSON. This is the GATE AUTHORITY for
//! kart-three's UV animation (docs: kart-three UV 动画保真修正) — the check
//! compares the page's applied offset/repeat against THIS source-decoded
//! data, not against the glb extras it also consumes (self-consistency trap).
//!
//! Usage: uv_anim_export <track.1s> > uv_anim_authority.json

fn main() {
    let input = std::env::args()
        .nth(1)
        .expect("usage: uv_anim_export <track.1s>");
    let data = std::fs::read(&input).expect("read track");
    let track = kart_fmt::kart_object::parse(&data).expect("parse track");

    let mut materials: Vec<serde_json::Value> = Vec::new();
    for node in &track.nodes {
        if node.uv_curves.is_empty() {
            continue;
        }
        let mut curves = serde_json::Map::new();
        for (param, curve) in &node.uv_curves {
            curves.insert(
                param.to_string(),
                serde_json::json!({
                    "kf_type": curve.kf_type,
                    "keys": curve.keys.iter().map(|(t, v, tin, tout)| {
                        serde_json::json!([t, v, tin, tout])
                    }).collect::<Vec<_>>(),
                }),
            );
        }
        materials.push(serde_json::json!({
            "node": node.name,
            "material": node.tex,
            "curves": curves,
        }));
    }
    let doc = serde_json::json!({
        "schema": "uv_anim_authority_v1",
        "source": input,
        "materials": materials,
    });
    println!("{}", serde_json::to_string(&doc).expect("encode"));
}
