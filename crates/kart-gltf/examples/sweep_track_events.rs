//! sweep_track_events — E2 robustness sweep for the kart-fmt event-section
//! decode (docs/kart-fmt-to-events-handoff-2026-07-30.md §E1/E2 acceptance:
//! the decoder must not panic on any inventoried track, and decoded event
//! count must equal event_count).
//!
//! Unpacks every given .rho pack IN MEMORY (kart_rho::archive::unpack),
//! parses each KartObject .1s entry with kart_fmt::kart_object::parse, and
//! reports per-entry event statistics plus any parse failure. Malformed
//! inputs may still make the parser panic by crate contract, so each entry
//! is parsed under catch_unwind and counted instead of aborting the sweep.
//!
//! Usage: sweep_track_events <pack.rho> [pack.rho...]

use std::panic::{catch_unwind, AssertUnwindSafe};

fn main() {
    // Silence the parser's optional trailing-morph probe recoveries and let
    // real panics be counted by the catch below.
    std::panic::set_hook(Box::new(|_| {}));
    let mut tracks = 0u32;
    let mut with_events = 0u32;
    let mut mismatched = 0u32;
    let mut parse_failed = 0u32;
    let mut unpack_failed = 0u32;
    let mut total_events = 0u32;
    let mut total_roots = 0u32;
    let mut opaque_roots = 0u32;
    for path in std::env::args().skip(1) {
        let entries = match kart_rho::archive::unpack(std::path::Path::new(&path)) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("{path}: unpack failed: {e}");
                unpack_failed += 1;
                continue;
            }
        };
        for entry in entries {
            if !entry.path.ends_with(".1s") || !kart_fmt::kart_object::is_track(&entry.data) {
                continue;
            }
            tracks += 1;
            let parsed = catch_unwind(AssertUnwindSafe(|| {
                kart_fmt::kart_object::parse(&entry.data)
            }));
            match parsed {
                Ok(Ok(track)) => {
                    if track.event_count > 0 {
                        with_events += 1;
                        total_events += track.events.len() as u32;
                        total_roots += track.event_roots.len() as u32;
                        opaque_roots += track
                            .event_roots
                            .iter()
                            .filter(|r| r.node.is_none() && r.placement.is_none())
                            .count() as u32;
                        if track.event_count != track.events.len() as i32 {
                            mismatched += 1;
                            eprintln!(
                                "{path}/{}: event_count={} decoded={}",
                                entry.path,
                                track.event_count,
                                track.events.len()
                            );
                        }
                    }
                }
                Ok(Err(e)) => {
                    parse_failed += 1;
                    eprintln!("{path}/{}: parse error: {e}", entry.path);
                }
                Err(_) => {
                    parse_failed += 1;
                    eprintln!("{path}/{}: PANIC during parse", entry.path);
                }
            }
        }
    }
    println!(
        "tracks={tracks} with_events={with_events} events={total_events} \
         roots={total_roots} opaque_roots={opaque_roots} \
         count_mismatch={mismatched} parse_failed={parse_failed} unpack_failed={unpack_failed}"
    );
    if mismatched > 0 || parse_failed > 0 || unpack_failed > 0 {
        std::process::exit(1);
    }
}
