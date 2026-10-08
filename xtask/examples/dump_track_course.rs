//! Dump the authoritative embedded course graph from a track `.1s`.
//!
//! Usage: dump_track_course <track.1s> [forward|reverse]

use kart_fmt::bml::Tag;
use std::path::PathBuf;

fn walk(tag: &Tag, path: &str, out: &mut Vec<(String, String, String, String)>) {
    let here = if path.is_empty() {
        tag.name.clone()
    } else {
        format!("{path}/{}", tag.name)
    };
    if tag.name == "road" {
        out.push((
            here.clone(),
            tag.attr("name").unwrap_or("").to_string(),
            tag.attr("start").unwrap_or("").to_string(),
            tag.attr("end").unwrap_or("").to_string(),
        ));
    }
    for (index, child) in tag.children.iter().enumerate() {
        walk(child, &format!("{here}[{index}]"), out);
    }
}

fn clean(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: dump_track_course <track.1s> [forward|reverse]")?;
    let reverse = match args.next().as_deref() {
        None | Some("forward") => false,
        Some("reverse") => true,
        Some(other) => return Err(format!("invalid direction: {other}")),
    };
    let bytes = std::fs::read(&path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let property = kart_fmt::track_course::find_embedded_property(&bytes)
        .ok_or_else(|| format!("no embedded property/course in {}", path.display()))?;
    let course = property
        .course(reverse)
        .ok_or_else(|| format!("no requested course direction in {}", path.display()))?;
    let mut roads = Vec::new();
    walk(course, "", &mut roads);

    eprintln!(
        "property_offset=0x{:X} property_len={} direction={} road_count={}",
        property.offset,
        property.encoded_len,
        if reverse { "reverse" } else { "forward" },
        roads.len()
    );
    println!("path\tname\tstart\tend");
    for (path, name, start, end) in roads {
        println!(
            "{}\t{}\t{}\t{}",
            clean(&path),
            clean(&name),
            clean(&start),
            clean(&end)
        );
    }
    Ok(())
}
