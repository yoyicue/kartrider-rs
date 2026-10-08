//! Minimal GLB structural reader: pulls the geometry counts that are stable
//! across serializers (mesh/material/primitive counts, total deduplicated
//! vertices, total triangles) so a C#-produced GLB and a Rust-produced GLB can be
//! compared without depending on byte layout or chunk ordering.

use std::collections::BTreeMap;

/// Serializer-independent geometry summary of a GLB.
pub struct GlbStats {
    pub meshes: usize,
    pub materials: usize,
    pub primitives: usize,
    pub verts: usize,
    pub tris: usize,
}

impl GlbStats {
    /// The compared metrics — counts that are identical regardless of serializer.
    /// `verts` is deliberately excluded: SharpGLTF (C#) deduplicates vertices that
    /// share POSITION/NORMAL/UV while this workspace's writer keeps them per-corner;
    /// both are valid glTF, so vertex counts legitimately differ. Triangle count is
    /// stable because both apply the same degenerate-triangle cull.
    pub fn line(&self) -> String {
        format!(
            "meshes={} materials={} primitives={} tris={}",
            self.meshes, self.materials, self.primitives, self.tris
        )
    }

    /// Informational vertex count (not compared).
    pub fn verts_note(&self) -> String {
        format!("verts={}", self.verts)
    }
}

/// Parse a GLB's JSON chunk and tally geometry. Returns None if not a GLB.
pub fn stats(glb: &[u8]) -> Option<GlbStats> {
    if glb.len() < 20 || &glb[0..4] != b"glTF" {
        return None;
    }
    let jlen = u32::from_le_bytes(glb[12..16].try_into().ok()?) as usize;
    let json = &glb[20..20 + jlen];
    let v: serde_json::Value = serde_json::from_slice(json).ok()?;

    let accessors = v.get("accessors")?.as_array()?;
    let count = |i: usize| -> usize {
        accessors
            .get(i)
            .and_then(|a| a.get("count"))
            .and_then(|c| c.as_u64())
            .unwrap_or(0) as usize
    };

    let meshes = v
        .get("meshes")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let materials = v
        .get("materials")
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let (mut prims, mut verts, mut tris) = (0usize, 0usize, 0usize);
    for m in &meshes {
        if let Some(ps) = m.get("primitives").and_then(|p| p.as_array()) {
            for p in ps {
                prims += 1;
                if let Some(pos) = p
                    .get("attributes")
                    .and_then(|a| a.get("POSITION"))
                    .and_then(|i| i.as_u64())
                {
                    verts += count(pos as usize);
                }
                match p.get("indices").and_then(|i| i.as_u64()) {
                    Some(idx) => tris += count(idx as usize) / 3,
                    None => {
                        // non-indexed: triangles = POSITION count / 3
                        if let Some(pos) = p
                            .get("attributes")
                            .and_then(|a| a.get("POSITION"))
                            .and_then(|i| i.as_u64())
                        {
                            tris += count(pos as usize) / 3;
                        }
                    }
                }
            }
        }
    }
    Some(GlbStats {
        meshes: meshes.len(),
        materials,
        primitives: prims,
        verts,
        tris,
    })
}

/// Parse a `"k=v k=v"` stat line into an ordered map for diffing.
pub fn parse_kv(line: &str) -> BTreeMap<String, String> {
    line.split_whitespace()
        .filter_map(|t| t.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}
