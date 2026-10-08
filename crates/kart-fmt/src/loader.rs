//! Sibling-clip asset discovery and best-effort animation loading.
//!
//! This is a pure asset-tree concern — find the `fXX.1s` skeletal/rigid clips
//! that live next to a `model.1s`, then parse the usable ones — so it lives in
//! the parser layer where every backend (glTF, USD, the renderer, the batch
//! pipeline) can share one implementation instead of each re-deriving it.
//!
//! Unlike the core `parse` entry points (which panic on a malformed stream — see
//! the crate root contract), the `load_*` helpers here are deliberately
//! best-effort: a single unparseable or animation-free clip is skipped, not
//! fatal, because their job is to gather whatever clips a model actually has.
//! The character loader wraps each parse in `catch_unwind`; install a panic hook
//! at the binary if you want recovered clips to stay quiet.

use crate::anim_clip::{self, AnimClip};
use crate::kart_anim::{self, KartAnimClip};
use std::path::{Path, PathBuf};

/// Collect a model's sibling `fXX.1s` skeletal-anim clips as raw `(bytes, name)`
/// pairs. Character/kart keep them next to `model.1s`; pet / flyingPet keep them
/// in `equipped/` `unequipped/` … subdirs — so when the model's own dir has none,
/// fall back to its immediate subdirs (clip names prefixed by the subdir to avoid
/// `fXX` collisions).
pub fn sibling_clips(dir: &Path) -> Vec<(Vec<u8>, String)> {
    let mut clips = collect_fxx(dir, "");
    if clips.is_empty() {
        let mut subs: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subs.sort();
        for sub in subs {
            let prefix = sub
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            clips.extend(collect_fxx(&sub, &prefix));
        }
    }
    clips
}

fn collect_fxx(dir: &Path, name_prefix: &str) -> Vec<(Vec<u8>, String)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with('f') && s.ends_with(".1s"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
        .into_iter()
        .filter_map(|f| {
            let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("clip");
            let nm = if name_prefix.is_empty() {
                stem.to_string()
            } else {
                format!("{name_prefix}_{stem}")
            };
            std::fs::read(&f).ok().map(|b| (b, nm))
        })
        .collect()
}

/// Load a character/pet model's sibling skeletal clips, keeping those with at
/// least one bone track. Each parse is caught so one bad clip can't abort the set.
pub fn load_char_anims(dir: &Path) -> Vec<AnimClip> {
    sibling_clips(dir)
        .into_iter()
        .filter_map(|(bytes, nm)| {
            std::panic::catch_unwind(|| anim_clip::parse(&bytes, &nm))
                .ok()
                .flatten()
                .filter(|c| !c.tracks.is_empty())
        })
        .collect()
}

/// Load a kart model's sibling rigid-body clips, keeping those that actually
/// animate at least one channel.
pub fn load_kart_anims(dir: &Path) -> Vec<KartAnimClip> {
    sibling_clips(dir)
        .into_iter()
        .filter_map(|(bytes, nm)| {
            kart_anim::parse(&bytes, &nm).filter(|c| c.channels.iter().any(|ch| ch.is_animated()))
        })
        .collect()
}
