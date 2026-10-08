use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or(
        "usage: find_track_surface_id_candidates <track-root-or-unpacked-root> <surface-id> [--all-1s]",
    )?);
    let target = parse_u32(&args.next().ok_or("missing surface-id")?)?;
    let all_1s = args.any(|arg| arg == "--all-1s");
    let mut tracks = Vec::new();
    collect_scene_files(&root, all_1s, &mut tracks);
    tracks.sort();
    let scene_files = tracks.len();

    println!("track_id\ttrack_path\tnode\tsource\tvalue\tsurface_id");
    let mut scanned_values = 0usize;
    let mut hits = 0usize;
    let mut parsed_files = 0usize;
    let mut parse_failures = 0usize;
    for track_path in tracks {
        let Ok(bytes) = fs::read(&track_path) else {
            continue;
        };
        let target_bytes = target.to_le_bytes();
        for (offset, window) in bytes.windows(4).enumerate() {
            if window == target_bytes {
                hits += 1;
                println!(
                    "{}\t{}\t\traw_u32_le\t0x{:X}\t{}",
                    clean(&track_id(&track_path)),
                    clean(&track_path.display().to_string()),
                    offset,
                    target
                );
            }
        }
        let track = match kart_fmt::kart_object::parse(&bytes) {
            Ok(track) => track,
            Err(_) => {
                parse_failures += 1;
                continue;
            }
        };
        parsed_files += 1;
        for node in &track.nodes {
            scan_value(
                &track_path,
                &node.name,
                "node.name",
                &node.name,
                target,
                &mut scanned_values,
                &mut hits,
            );
            if let Some(tex) = node.tex.as_deref() {
                scan_value(
                    &track_path,
                    &node.name,
                    "node.tex",
                    tex,
                    target,
                    &mut scanned_values,
                    &mut hits,
                );
            }
            if let Some(tag) = node.extra_tag.as_ref() {
                scan_tag(
                    &track_path,
                    &node.name,
                    tag,
                    "tag",
                    target,
                    &mut scanned_values,
                    &mut hits,
                );
            }
        }
    }
    eprintln!(
        "target={target} scene_files={} parsed_files={parsed_files} parse_failures={parse_failures} scanned_values={scanned_values} hits={hits}",
        scene_files
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scan_tag(
    track_path: &Path,
    node: &str,
    tag: &kart_fmt::kart_object::BinaryXmlTag,
    prefix: &str,
    target: u32,
    scanned: &mut usize,
    hits: &mut usize,
) {
    scan_value(
        track_path,
        node,
        &format!("{prefix}.name"),
        &tag.name,
        target,
        scanned,
        hits,
    );
    for (key, value) in &tag.attrs {
        scan_value(
            track_path,
            node,
            &format!("{prefix}.attr_key"),
            key,
            target,
            scanned,
            hits,
        );
        scan_value(
            track_path,
            node,
            &format!("{prefix}.attr.{key}"),
            value,
            target,
            scanned,
            hits,
        );
    }
    for child in &tag.children {
        scan_tag(
            track_path,
            node,
            child,
            &format!("{prefix}.{}", child.name),
            target,
            scanned,
            hits,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_value(
    track_path: &Path,
    node: &str,
    source: &str,
    value: &str,
    target: u32,
    scanned: &mut usize,
    hits: &mut usize,
) {
    if value.is_empty() {
        return;
    }
    *scanned += 1;
    let id = surface_id(value);
    if id != target {
        return;
    }
    *hits += 1;
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}",
        clean(&track_id(track_path)),
        clean(&track_path.display().to_string()),
        clean(node),
        clean(source),
        clean(value),
        id
    );
}

fn collect_scene_files(dir: &Path, all_1s: bool, out: &mut Vec<PathBuf>) {
    if dir.is_file() {
        if is_selected_scene(dir, all_1s) {
            out.push(dir.to_path_buf());
        }
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else { return };
    for ent in rd.flatten() {
        let path = ent.path();
        if path.is_dir() {
            collect_scene_files(&path, all_1s, out);
        } else if is_selected_scene(&path, all_1s) {
            out.push(path);
        }
    }
}

fn is_selected_scene(path: &Path, all_1s: bool) -> bool {
    if all_1s {
        path.extension().and_then(|ext| ext.to_str()) == Some("1s")
    } else {
        path.file_name().and_then(|name| name.to_str()) == Some("track.1s")
    }
}

fn parse_u32(value: &str) -> Result<u32, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).map_err(|err| err.to_string())
    } else {
        value.parse::<u32>().map_err(|err| err.to_string())
    }
}

fn track_id(path: &Path) -> String {
    path.parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string()
}

fn surface_id(text: &str) -> u32 {
    let mut a = 0u32;
    let mut b = 0u32;
    for unit in text.encode_utf16() {
        for byte in unit.to_le_bytes() {
            a = (a + u32::from(byte)) % 65521;
            b = (b + a) % 65521;
        }
    }
    match (b << 16) | a {
        0 => 1,
        u32::MAX => 2,
        value => value,
    }
}

fn clean(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}
