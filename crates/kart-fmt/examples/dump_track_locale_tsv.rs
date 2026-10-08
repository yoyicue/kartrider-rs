//! Dump the track registry plus a locale overlay as machine-readable TSV.
//!
//! Usage: dump_track_locale_tsv <track@zz.bml> <trackLocale@cc.bml>

use kart_fmt::{bml, track_list};
use std::collections::BTreeMap;

fn tsv_escape(s: &str) -> String {
    s.replace(['\t', '\r', '\n'], " ")
}

fn variant_id(kind: &str, ref_id: &str) -> String {
    format!("{}_{}", ref_id, kind.strip_prefix("track_").unwrap_or(kind))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let reg_path = args
        .next()
        .expect("usage: dump_track_locale_tsv <track@zz.bml> <trackLocale@cc.bml>");
    let locale_path = args
        .next()
        .expect("usage: dump_track_locale_tsv <track@zz.bml> <trackLocale@cc.bml>");

    let registry = std::fs::read(&reg_path).expect("read registry");
    let tl = track_list::parse(&registry).expect("not a <trackList> bml");

    let locale_root =
        bml::parse(&std::fs::read(&locale_path).expect("read locale")).expect("not a locale BML");
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for tag in &locale_root.children {
        if tag.name == "track" {
            if let (Some(id), Some(name)) = (tag.attr("id"), tag.attr("name")) {
                names.insert(id.to_string(), name.to_string());
            }
        } else if let Some(ref_id) = tag.attr("refId") {
            if let Some(name) = tag.attr("name") {
                names.insert(variant_id(&tag.name, ref_id), name.to_string());
            }
        }
    }

    println!(
        "track_id\tdisplay_name\tkind\tbase_id\tgame_type\ttheme\ttexture_themes\tgeometry_folder"
    );
    for track in &tl.tracks {
        println!(
            "{}\t{}\tbase\t{}\t{}\t{}\t{}\t{}",
            tsv_escape(&track.id),
            tsv_escape(names.get(&track.id).map(String::as_str).unwrap_or("")),
            tsv_escape(&track.id),
            tsv_escape(&track.game_type),
            tsv_escape(track.theme()),
            tsv_escape(&track.texture_themes().join("|")),
            tsv_escape(track.geometry_folder()),
        );
    }
    for variant in &tl.variants {
        let id = variant.derived_id();
        let base_name = names.get(&variant.ref_id).map(String::as_str).unwrap_or("");
        let display_name = names.get(&id).map(String::as_str).unwrap_or(base_name);
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t",
            tsv_escape(&id),
            tsv_escape(display_name),
            tsv_escape(&variant.kind),
            tsv_escape(&variant.ref_id),
            tsv_escape(variant.attr("gameType").unwrap_or("")),
            tsv_escape(variant.theme()),
            tsv_escape(variant.attr("texTheme").unwrap_or(variant.theme())),
        );
    }
}
