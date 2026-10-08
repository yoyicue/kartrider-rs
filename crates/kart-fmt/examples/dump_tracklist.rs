//! Parse `track@zz.bml` (the track registry) and report it as BML-confirmed
//! data: counts, theme histogram, and the texTheme/folder overrides that a
//! filename-based graph would miss. Optionally merges a `trackLocale@<cc>.bml`
//! for human names.
//!
//! Usage: dump_tracklist <track@zz.bml> [trackLocale@cc.bml]

use kart_fmt::{bml, track_list};
use std::collections::BTreeMap;

fn main() {
    let mut args = std::env::args().skip(1);
    let reg_path = args
        .next()
        .expect("usage: dump_tracklist <track@zz.bml> [trackLocale@cc.bml]");
    let data = std::fs::read(&reg_path).expect("read registry");
    let tl = track_list::parse(&data).expect("not a <trackList> bml");

    // optional locale name overlay: trackLocale is also a <trackList> of <track id name=...>
    let names: BTreeMap<String, String> = args
        .next()
        .and_then(|p| std::fs::read(p).ok())
        .and_then(|b| bml::parse(&b))
        .map(|root| {
            root.children_named("track")
                .filter_map(|t| Some((t.attr("id")?.to_string(), t.attr("name")?.to_string())))
                .collect()
        })
        .unwrap_or_default();

    let mut by_theme: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_game: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_variant: BTreeMap<&str, usize> = BTreeMap::new();
    let (mut tex_overrides, mut folder_overrides) = (0usize, 0usize);
    for t in &tl.tracks {
        *by_theme.entry(t.theme()).or_default() += 1;
        *by_game.entry(t.game_type.as_str()).or_default() += 1;
        if t.tex_theme.is_some() {
            tex_overrides += 1;
        }
        if t.folder.is_some() {
            folder_overrides += 1;
        }
    }
    for v in &tl.variants {
        *by_variant.entry(v.kind.as_str()).or_default() += 1;
    }

    println!(
        "tracks={}  variants={} {:?}  other={}  themes={}",
        tl.tracks.len(),
        tl.variants.len(),
        by_variant,
        tl.other.len(),
        by_theme.len()
    );
    println!("texTheme overrides={tex_overrides}  folder overrides={folder_overrides}");
    println!("\ngameType:");
    for (g, n) in &by_game {
        println!("  {g:12} {n}");
    }
    println!("\ntexTheme overrides (BML-confirmed, ≠ id-prefix theme):");
    for t in &tl.tracks {
        if t.tex_theme.is_some() {
            let nm = names
                .get(&t.id)
                .map(|s| format!("  «{s}»"))
                .unwrap_or_default();
            println!(
                "  {:16} theme={:10} → textures {:?}{nm}",
                t.id,
                t.theme(),
                t.texture_themes()
            );
        }
    }
    println!("\nfolder reuse:");
    for t in &tl.tracks {
        if let Some(f) = &t.folder {
            println!("  {:16} → geometry folder {f}", t.id);
        }
    }
}
