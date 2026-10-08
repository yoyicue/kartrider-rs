use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let root = PathBuf::from(
        env::args()
            .nth(1)
            .ok_or("usage: list_track_road_surfaces <track-root-or-unpacked-root>")?,
    );
    let mut tracks = Vec::new();
    collect_track_files(&root, &mut tracks);
    tracks.sort();

    println!("track_id\ttrack_path\tnode\ttex\troad_surface\tsurface_id\troad_rail\ttris");
    for track_path in tracks {
        let Ok(bytes) = fs::read(&track_path) else {
            continue;
        };
        let Ok(track) = kart_fmt::kart_object::parse(&bytes) else {
            continue;
        };
        for node in &track.nodes {
            let Some(surface) = tag_attr(node.extra_tag.as_ref(), "road", "surface") else {
                continue;
            };
            let tris = node.geom.as_ref().map(triangle_count).unwrap_or(0);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                clean(&track_id(&track_path)),
                clean(&track_path.display().to_string()),
                clean(&node.name),
                clean(node.tex.as_deref().unwrap_or("")),
                clean(&surface),
                surface_id(&surface),
                clean(&tag_attr(node.extra_tag.as_ref(), "road", "rail").unwrap_or_default()),
                tris
            );
        }
    }

    Ok(())
}

fn collect_track_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let p = ent.path();
        if p.is_dir() {
            collect_track_files(&p, out);
        } else if p.file_name().and_then(|s| s.to_str()) == Some("track.1s") {
            out.push(p);
        }
    }
}

fn track_id(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

fn triangle_count(geom: &kart_fmt::kart_object::Geom) -> usize {
    if geom.strip {
        geom.indices
            .windows(3)
            .filter(|tri| tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2])
            .count()
    } else {
        geom.indices.len() / 3
    }
}

fn tag_attr(
    tag: Option<&kart_fmt::kart_object::BinaryXmlTag>,
    tag_name: &str,
    attr_name: &str,
) -> Option<String> {
    let tag = tag?;
    if tag.name == tag_name {
        if let Some((_, value)) = tag.attrs.iter().find(|(key, _)| key == attr_name) {
            return Some(value.clone());
        }
    }
    for child in &tag.children {
        if let Some(value) = tag_attr(Some(child), tag_name, attr_name) {
            return Some(value);
        }
    }
    None
}

fn surface_id(text: &str) -> u32 {
    let mut a = 0u32;
    let mut b = 0u32;
    for unit in text.encode_utf16() {
        for byte in unit.to_le_bytes() {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
    }
    match (b << 16) | a {
        0 => 1,
        0xffff_ffff => 2,
        value => value,
    }
}

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}
