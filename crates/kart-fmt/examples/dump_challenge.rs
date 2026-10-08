//! Parse `challenge.bml` and report the license/challenge progression: counts,
//! type histogram, the DAG roots, and any dangling `require` edges (a broken
//! prerequisite chain). Validates the progression is self-consistent.
//!
//! Usage: dump_challenge <challenge.bml>

use kart_fmt::challenge;
use std::collections::BTreeMap;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: dump_challenge <challenge.bml>");
    let data = std::fs::read(&path).expect("read");
    let cl = challenge::parse(&data).expect("not a <challengeList> bml");

    let mut by_type: BTreeMap<i32, usize> = BTreeMap::new();
    for c in &cl.challenges {
        *by_type.entry(c.kind).or_default() += 1;
    }

    println!(
        "challenges={}  other={}",
        cl.challenges.len(),
        cl.other.len()
    );
    print!("type histogram:");
    for (t, n) in &by_type {
        print!("  type{t}={n}");
    }
    println!();

    let roots: Vec<&str> = cl.roots().map(|c| c.id.as_str()).collect();
    println!("DAG roots (no require): {roots:?}");

    let dangling = cl.dangling_requires();
    if dangling.is_empty() {
        println!("require DAG: self-consistent ✓ (every require resolves to a defined challenge)");
    } else {
        println!("require DAG: {} DANGLING edge(s):", dangling.len());
        for (id, missing) in &dangling {
            println!("  {id} requires {missing} (undefined)");
        }
    }

    println!("\nprogression (id → require, track, time):");
    for c in &cl.challenges {
        println!(
            "  {:6} type={} require={:6} track={:18} time={}",
            c.id,
            c.kind,
            c.require.as_deref().unwrap_or("—"),
            c.track.as_deref().unwrap_or("—"),
            c.time_ms
                .map(|t| t.to_string())
                .unwrap_or_else(|| "—".into())
        );
    }
}
