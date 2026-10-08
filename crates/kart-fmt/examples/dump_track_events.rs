//! dump_track_events — dump the decoded To* trigger event table of a
//! track.1s as TSV (E1 of docs/kart-fmt-to-events-handoff-2026-07-30.md).
//!
//! Columns: index, stamp, class, name, body_len, and for ToRoad events the
//! element count. Usage: dump_track_events <track.1s> [track.1s...]

fn main() {
    let mut rc = 0;
    println!("index\tstamp\tclass\tname\tbody_len\ttoroad_elements");
    for path in std::env::args().skip(1) {
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("error: read {path}: {e}");
                rc = 1;
                continue;
            }
        };
        let track = match kart_fmt::kart_object::parse(&data) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: parse {path}: {e}");
                rc = 1;
                continue;
            }
        };
        eprintln!(
            "{path}: event_count={} decoded={}",
            track.event_count,
            track.events.len()
        );
        if track.event_count != track.events.len() as i32 {
            eprintln!("warn: decoded count differs from event_count");
            rc = 1;
        }
        for ev in &track.events {
            println!(
                "{}\t0x{:08x}\t{}\t{}\t{}\t{}",
                ev.index,
                ev.stamp,
                ev.class_name.unwrap_or("?"),
                ev.name.as_deref().unwrap_or(""),
                ev.body.len(),
                ev.to_road.as_ref().map(|r| r.elements.len()).unwrap_or(0),
            );
        }
    }
    std::process::exit(rc);
}
