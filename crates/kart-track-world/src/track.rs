//! Track asset helpers: tag access, surface ids, TSV-safe text, KAP path
//! identity, and the cached static-track loader.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub fn flatten_tag(tag: &kart_fmt::kart_object::BinaryXmlTag, path: &str, out: &mut Vec<String>) {
    let current = if path.is_empty() {
        tag.name.clone()
    } else {
        format!("{path}/{}", tag.name)
    };
    let attrs = tag
        .attrs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(",");
    out.push(format!("{}[{}]={}", current, attrs, tag.text));
    for child in &tag.children {
        flatten_tag(child, &current, out);
    }
}

pub fn track_id(path: &Path) -> String {
    path.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

pub fn tag_attr(
    tag: Option<&kart_fmt::kart_object::BinaryXmlTag>,
    tag_name: &str,
    attr_name: &str,
) -> Option<String> {
    let tag = tag?;
    if tag.name == tag_name {
        if let Some((_, value)) = tag.attrs.iter().find(|(key, _)| key == attr_name) {
            return Some(value.clone());
        }
    }
    for child in &tag.children {
        if let Some(value) = tag_attr(Some(child), tag_name, attr_name) {
            return Some(value);
        }
    }
    None
}

pub fn has_tag(tag: Option<&kart_fmt::kart_object::BinaryXmlTag>, tag_name: &str) -> bool {
    let Some(tag) = tag else { return false };
    tag.name == tag_name
        || tag
            .children
            .iter()
            .any(|child| has_tag(Some(child), tag_name))
}

pub fn surface_id(text: &str) -> u32 {
    let mut a = 0u32;
    let mut b = 0u32;
    for unit in text.encode_utf16() {
        for byte in unit.to_le_bytes() {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
    }
    match (b << 16) | a {
        0 => 1,
        0xffff_ffff => 2,
        value => value,
    }
}

pub fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

pub fn kap_track_id(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let mut parts: Vec<&str> = stem.split('_').collect();
    let last = parts.last().copied()?;
    if last.parse::<usize>().is_ok() {
        parts.pop();
    }
    if parts.last() == Some(&"rvs") {
        parts.pop();
    }
    Some(parts.join("_"))
}

pub fn kap_slot(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;
    let last = stem.split('_').next_back()?;
    last.parse::<usize>().ok()
}

static STATIC_TRACK_CACHE: OnceLock<Mutex<BTreeMap<PathBuf, Arc<kart_fmt::kart_object::Track>>>> =
    OnceLock::new();

/// Virtual-bytes registry (W3): maps a path string to caller-supplied asset
/// bytes. The browser/demo path has no filesystem — the fetched track.1s
/// ArrayBuffer is registered under the original track path and every asset
/// read below consults this registry first, so the whole TSV render/parse
/// pipeline (and its bit-exact `{:.9}` round-trip) runs unchanged on bytes.
static VIRTUAL_BYTES: OnceLock<Mutex<BTreeMap<PathBuf, Arc<Vec<u8>>>>> = OnceLock::new();

/// Registers `bytes` as the asset content for `path`, shadowing the
/// filesystem on the next [`read_bytes`] call for that exact path.
pub fn register_virtual_bytes(path: &Path, bytes: &[u8]) -> Result<(), String> {
    {
        let mut registry = VIRTUAL_BYTES
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .map_err(|_| "virtual bytes registry lock poisoned".to_string())?;
        if let Some(existing) = registry.get(path) {
            if existing.as_slice() == bytes {
                // Identical re-registration (e.g. a fresh kart-wasm session
                // over the same fetched ArrayBuffer): the parse and derived
                // query index are still valid, keep them.
                return Ok(());
            }
        }
        registry.insert(path.to_path_buf(), Arc::new(bytes.to_vec()));
    }
    // A re-registered path may carry different bytes; drop any stale parse so
    // the next load_static_track re-parses from the new content.
    invalidate_static_track_cache(path)
}

/// Removes a previously registered virtual-bytes entry (test hygiene).
pub fn unregister_virtual_bytes(path: &Path) -> Result<(), String> {
    VIRTUAL_BYTES
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .map_err(|_| "virtual bytes registry lock poisoned".to_string())?
        .remove(path);
    // The path now resolves to the filesystem again; drop any parse cached
    // from the virtual bytes.
    invalidate_static_track_cache(path)
}

fn invalidate_static_track_cache(path: &Path) -> Result<(), String> {
    {
        let mut cache = STATIC_TRACK_CACHE
            .get_or_init(|| Mutex::new(BTreeMap::new()))
            .lock()
            .map_err(|_| "static track cache lock poisoned".to_string())?;
        cache.remove(path);
        cache.remove(&static_track_cache_key(path));
    }
    // The derived query index is keyed identically and must not survive a
    // byte swap either (P2: it holds the parse's flattened triangle soup).
    crate::query::invalidate_query_index_cache(path);
    // Same for the process-level rail context/entry caches (P4).
    crate::world::adapter::invalidate_rail_caches(path);
    Ok(())
}

/// The cache key shared by the static-track parse cache and the derived
/// query-index cache: a virtually-registered path keeps its exact
/// (unresolvable) spelling — canonicalize would either fail (wasm) or escape
/// the registry onto the real filesystem — everything else canonicalizes.
pub fn static_track_cache_key(track_path: &Path) -> PathBuf {
    if has_virtual_bytes(track_path) {
        track_path.to_path_buf()
    } else {
        track_path
            .canonicalize()
            .unwrap_or_else(|_| track_path.to_path_buf())
    }
}

/// Reads an asset's bytes: the virtual registry first, then the filesystem.
/// Error text keeps the historical `read <path>: <err>` shape.
pub fn read_bytes(path: &Path) -> Result<Arc<Vec<u8>>, String> {
    if let Some(bytes) = VIRTUAL_BYTES
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .map_err(|_| "virtual bytes registry lock poisoned".to_string())?
        .get(path)
        .cloned()
    {
        return Ok(bytes);
    }
    fs::read(path)
        .map(Arc::new)
        .map_err(|err| format!("read {}: {err}", path.display()))
}

fn has_virtual_bytes(path: &Path) -> bool {
    VIRTUAL_BYTES
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .map(|registry| registry.contains_key(path))
        .unwrap_or(false)
}

pub fn load_static_track(track_path: &Path) -> Result<Arc<kart_fmt::kart_object::Track>, String> {
    let key = static_track_cache_key(track_path);
    let cache = STATIC_TRACK_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Some(track) = cache
        .lock()
        .map_err(|_| "static track cache lock poisoned".to_string())?
        .get(&key)
        .cloned()
    {
        return Ok(track);
    }
    let bytes = read_bytes(&key)?;
    let track = Arc::new(
        kart_fmt::kart_object::parse(&bytes)
            .map_err(|err| format!("parse {}: {err}", key.display()))?,
    );
    cache
        .lock()
        .map_err(|_| "static track cache lock poisoned".to_string())?
        .insert(key, Arc::clone(&track));
    Ok(track)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative)
    }

    #[test]
    fn reregistering_virtual_bytes_invalidates_cached_parse() {
        let factory = repo_path("analysis/rail/factory_I02.track.1s");
        let desert = repo_path("analysis/motion_topology/desert_I08.track.1s");
        if !factory.is_file() || !desert.is_file() {
            eprintln!("skipping: track assets not present");
            return;
        }
        let virtual_path = Path::new("virtual://cache-invalidation-test.track.1s");
        let factory_bytes = fs::read(&factory).unwrap();
        let desert_bytes = fs::read(&desert).unwrap();

        register_virtual_bytes(virtual_path, &factory_bytes).unwrap();
        let first = load_static_track(virtual_path).unwrap();
        // Re-register different bytes under the same path: the cached parse
        // must not leak into the next load.
        register_virtual_bytes(virtual_path, &desert_bytes).unwrap();
        let second = load_static_track(virtual_path).unwrap();
        assert!(
            !Arc::ptr_eq(&first, &second),
            "re-register must drop the cached parse"
        );

        unregister_virtual_bytes(virtual_path).unwrap();
    }
}

#[cfg(test)]
mod identical_registration_tests {
    use super::*;

    fn repo_path(relative: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .join(relative)
    }

    #[test]
    fn identical_reregistration_keeps_cached_parse() {
        let factory = repo_path("analysis/rail/factory_I02.track.1s");
        if !factory.is_file() {
            eprintln!("skipping: track asset not present");
            return;
        }
        let virtual_path = Path::new("virtual://identical-reregister.track.1s");
        let bytes = fs::read(&factory).unwrap();
        register_virtual_bytes(virtual_path, &bytes).unwrap();
        let first = load_static_track(virtual_path).unwrap();
        // Same bytes again: no invalidation, same cached Arc.
        register_virtual_bytes(virtual_path, &bytes).unwrap();
        let second = load_static_track(virtual_path).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        unregister_virtual_bytes(virtual_path).unwrap();
    }
}
