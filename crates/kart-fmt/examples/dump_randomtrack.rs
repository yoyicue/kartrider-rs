//! Parse `randomTrack@<cc>.bml` (random-match rotations) and, given the track
//! registry, cross-validate that every referenced track id is a defined track —
//! a referential-integrity check across the two configs.
//!
//! Usage: dump_randomtrack <randomTrack@cc.bml> [track@zz.bml]

use kart_fmt::{random_track, track_list};
use std::collections::{BTreeMap, BTreeSet, HashSet};

fn main() {
    let mut args = std::env::args().skip(1);
    let rt_path = args
        .next()
        .expect("usage: dump_randomtrack <randomTrack@cc.bml> [track@zz.bml]");
    let rl = random_track::parse(&std::fs::read(&rt_path).expect("read"))
        .expect("not a <RandomTrackSetList> bml");

    let unique: BTreeSet<&str> = rl.track_refs().collect();
    let mut by_group: BTreeMap<&str, usize> = BTreeMap::new();
    for s in &rl.sets {
        *by_group
            .entry(s.group.as_deref().unwrap_or("(top)"))
            .or_default() += 1;
    }
    println!(
        "cc={}  sets={}  track refs={} ({} unique)",
        rl.cc,
        rl.sets.len(),
        rl.track_refs().count(),
        unique.len()
    );
    print!("sets by group:");
    for (g, n) in &by_group {
        print!("  {g}={n}");
    }
    println!();

    // Cross-validate against the registry when provided.
    if let Some(reg_path) = args.next() {
        let reg = track_list::parse(&std::fs::read(&reg_path).expect("read registry"))
            .expect("not a <trackList>");
        // full id space = base tracks + variant-derived ids (e.g. abyss_R02_rvs)
        let defined: HashSet<String> = reg.defined_ids();
        let missing: Vec<&str> = unique
            .iter()
            .copied()
            .filter(|id| !defined.contains(*id))
            .collect();
        if missing.is_empty() {
            println!(
                "referential integrity: ✓ all {} referenced tracks are defined in the registry",
                unique.len()
            );
        } else {
            println!(
                "referential integrity: {} referenced track(s) NOT in the registry:",
                missing.len()
            );
            for id in &missing {
                println!("  {id}");
            }
        }
    }
}
