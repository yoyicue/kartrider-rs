//! Decode a KAP AI path into a stable TSV for provenance comparisons.
//!
//! Usage: dump_kap_points <path.kap>

use std::path::PathBuf;

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: dump_kap_points <path.kap>")?;
    let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let clear = kart_rho::encoded_block::decode_len_prefixed(&bytes)
        .map_err(|err| format!("decode {}: {err}", path.display()))?;
    let kap = kart_fmt::kap::parse(&clear).ok_or_else(|| format!("parse {}", path.display()))?;

    println!("index\tx\ty\tz\tmarker");
    for (index, point) in kap.points.iter().enumerate() {
        println!(
            "{}\t{:.9}\t{:.9}\t{:.9}\t{}",
            index,
            point[0],
            point[1],
            point[2],
            kap.markers.contains(&(index as u16))
        );
    }
    Ok(())
}
