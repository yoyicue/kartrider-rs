use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

#[derive(Default)]
struct TokenStats {
    node_count: usize,
    triangle_count: usize,
    track_ids: BTreeSet<String>,
}

fn main() -> Result<(), String> {
    let root = PathBuf::from(
        env::args()
            .nth(1)
            .ok_or("usage: inventory_road_surface_tokens <unpacked-root>")?,
    );
    let mut track_paths = Vec::new();
    collect_track_files(&root, &mut track_paths)?;
    track_paths.sort();
    if track_paths.is_empty() {
        return Err(format!("no track.1s assets found under {}", root.display()));
    }

    let mut tokens: BTreeMap<String, TokenStats> = BTreeMap::new();
    let mut parsed_tracks = 0usize;
    let mut reachable_nodes = 0usize;
    for track_path in &track_paths {
        let bytes = fs::read(track_path)
            .map_err(|error| format!("read {}: {error}", track_path.display()))?;
        let track = kart_fmt::kart_object::parse(&bytes)
            .map_err(|error| format!("parse {}: {error}", track_path.display()))?;
        parsed_tracks += 1;
        let track_id = canonical_track_id(track_path);

        // Track::flatten is the public depth-first Track::walk contract.
        // Indexing track.nodes directly would also count disconnected binary
        // debris that is not reachable from the authored scene root.
        for (node_index, _) in track.flatten() {
            reachable_nodes += 1;
            let node = &track.nodes[node_index];
            let Some(token) = tag_attr(node.extra_tag.as_ref(), "road", "surface") else {
                continue;
            };
            let stats = tokens.entry(token).or_default();
            stats.node_count += 1;
            stats.triangle_count += node.geom.as_ref().map(triangle_count).unwrap_or(0);
            stats.track_ids.insert(track_id.clone());
        }
    }

    let token_rows: Vec<Value> = tokens
        .iter()
        .map(|(token, stats)| token_row(token, stats))
        .collect();
    let df_rows: Vec<Value> = tokens
        .iter()
        .filter_map(|(token, stats)| {
            df_payload(token).map(|payload| {
                let mut row = token_row(token, stats);
                row["df_payload"] = json!(payload);
                row["requires_game_adjudication"] = json!(true);
                row
            })
        })
        .collect();
    let total_surface_nodes: usize = tokens.values().map(|stats| stats.node_count).sum();
    let total_surface_triangles: usize = tokens.values().map(|stats| stats.triangle_count).sum();

    let inventory = json!({
        "schema": "kr_road_surface_token_inventory_v1",
        "date": "2026-07-30",
        "source": {
            "root": root.display().to_string(),
            "parser": "kart_fmt::kart_object::parse",
            "traversal": "Track::flatten depth-first Track::walk",
            "track_assets_discovered": track_paths.len(),
            "track_assets_parsed": parsed_tracks,
            "reachable_nodes_visited": reachable_nodes,
            "parse_failures": [],
        },
        "summary": {
            "distinct_road_surface_tokens": tokens.len(),
            "authored_surface_nodes": total_surface_nodes,
            "authored_surface_triangles": total_surface_triangles,
            "distinct_df_tokens": df_rows.len(),
        },
        "tokens": token_rows,
        "df_family": {
            "grammar": "case-insensitive literal DF followed by one finite f64 payload",
            "tokens": df_rows,
        },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&inventory)
            .map_err(|error| format!("serialize inventory: {error}"))?
    );
    Ok(())
}

fn token_row(token: &str, stats: &TokenStats) -> Value {
    json!({
        "token": token,
        "normalized_token": token.trim().to_lowercase(),
        "family": token_family(token),
        "surface_id": surface_id(token),
        "node_count": stats.node_count,
        "triangle_count": stats.triangle_count,
        "track_count": stats.track_ids.len(),
        "track_ids": stats.track_ids,
    })
}

fn token_family(token: &str) -> &'static str {
    let normalized = token.trim().to_ascii_lowercase();
    if df_payload(token).is_some() {
        "DF"
    } else if normalized.starts_with("jm") {
        "JM"
    } else if normalized.starts_with("dj") {
        "DJ"
    } else if normalized.starts_with("bs") {
        "BS"
    } else if normalized.starts_with("mz") {
        "MZ"
    } else {
        "exact_or_other"
    }
}

fn df_payload(token: &str) -> Option<f64> {
    let normalized = token.trim();
    let prefix = normalized.get(..2)?;
    let encoded_payload = normalized.get(2..)?;
    if encoded_payload.is_empty() || !prefix.eq_ignore_ascii_case("df") {
        return None;
    }
    let payload = encoded_payload.parse::<f64>().ok()?;
    payload.is_finite().then_some(payload)
}

fn collect_track_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries =
        fs::read_dir(dir).map_err(|error| format!("read_dir {}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("read_dir entry {}: {error}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect_track_files(&path, out)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("track.1s") {
            out.push(path);
        }
    }
    Ok(())
}

fn canonical_track_id(path: &Path) -> String {
    let raw = path
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    raw.strip_prefix("track_")
        .unwrap_or(raw)
        .strip_suffix(".rho")
        .unwrap_or_else(|| raw.strip_prefix("track_").unwrap_or(raw))
        .to_string()
}

fn triangle_count(geom: &kart_fmt::kart_object::Geom) -> usize {
    if geom.strip {
        geom.indices
            .windows(3)
            .filter(|triangle| {
                triangle[0] != triangle[1]
                    && triangle[1] != triangle[2]
                    && triangle[0] != triangle[2]
            })
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
            a = (a + u32::from(byte)) % 65_521;
            b = (b + a) % 65_521;
        }
    }
    match (b << 16) | a {
        0 => 1,
        u32::MAX => 2,
        value => value,
    }
}

#[cfg(test)]
mod tests {
    use super::{df_payload, surface_id, token_family};

    #[test]
    fn df_payload_requires_complete_finite_numeric_suffix() {
        assert_eq!(df_payload("DF2.0"), Some(2.0));
        assert_eq!(df_payload("df0.05"), Some(0.05));
        assert_eq!(df_payload(" DF5.0 "), Some(5.0));
        assert_eq!(df_payload("DF"), None);
        assert_eq!(df_payload("DF2.0x"), None);
        assert_eq!(df_payload("DFNaN"), None);
        assert_eq!(df_payload("리셋"), None);
    }

    #[test]
    fn family_and_surface_id_match_factory_df_asset() {
        assert_eq!(token_family("DF2.0"), "DF");
        assert_eq!(surface_id("DF2.0"), 119_275_802);
    }
}
