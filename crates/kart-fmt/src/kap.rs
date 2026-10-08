//! KAP — KartRider **AI / track path** (`track_common.rho/aiPath/*.kap`).
//!
//! The on-disk file is a `KREncodedBlock` (decode it first with
//! `kart_rho::encoded_block::decode_len_prefixed` — the same wrapper as
//! `aaa.pk`/`KartRider.pin`). This module parses the **decoded** inner payload,
//! which is a header, a vec3 waypoint path, and a list of u16 path markers.
//!
//! Format reverse-engineered and validated byte-exact across all 2179 modern
//! files (see `docs/kap-deferred.md`):
//!
//! ```text
//! [0x00] magic   80 04 25 1c
//! [0x04] u8[4]   per-file field (checksum/seed? — kept verbatim, not interpreted)
//! [0x08] u8      kind/lane? (observed {0,1,2,3,6})
//! [0x09] magic2  37 02 46 07
//! [0x0d] u32     point_count
//! [0x11] point_count × vec3<f32>   the AI path (world-space)
//!        u32     marker_count
//!        marker_count × u16        indices into the path (event/checkpoint markers)
//! ```
//!
//! Like [`crate::bml`], this is bounds-checked and fallible (config data we probe
//! loosely) rather than panic-on-malformed.

/// Header magic at offset `0x00`.
pub const MAGIC: [u8; 4] = [0x80, 0x04, 0x25, 0x1c];
/// Second magic at offset `0x09`.
pub const MAGIC2: [u8; 4] = [0x37, 0x02, 0x46, 0x07];

/// A parsed KAP AI path.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Kap {
    /// 4 bytes at `0x04`: a per-track 32-bit id/hash (constant across a track's
    /// `_N` path files, independent of path content; not adler/crc/djb2/fnv of the
    /// id). Exact derivation unidentified — kept verbatim for round-trip.
    pub field4: [u8; 4],
    /// Byte at `0x08`: a per-track category / AI-profile index (observed
    /// `{0,1,2,3,6}`; correlates with `gameType` but not a clean function of it).
    pub kind: u8,
    /// AI path waypoints in the track's world space.
    pub points: Vec<[f32; 3]>,
    /// Indices into [`points`](Kap::points) marking events/checkpoints (each is
    /// `< points.len()` in every observed file).
    pub markers: Vec<u16>,
}

impl Kap {
    /// True if every marker indexes a real waypoint (`< points.len()`).
    pub fn markers_in_range(&self) -> bool {
        let n = self.points.len();
        self.markers.iter().all(|&m| (m as usize) < n)
    }

    /// Axis-aligned bounding box of the path, or `None` if empty.
    pub fn bbox(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut it = self.points.iter();
        let first = *it.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in self.points.iter() {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        Some((lo, hi))
    }

    /// Re-serialize to the decoded-payload wire format (inverse of [`parse`]).
    /// Reproduces the original decoded bytes for proving round-trip identity.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(17 + self.points.len() * 12 + 4 + self.markers.len() * 2);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&self.field4);
        out.push(self.kind);
        out.extend_from_slice(&MAGIC2);
        out.extend_from_slice(&(self.points.len() as u32).to_le_bytes());
        for p in &self.points {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.markers.len() as u32).to_le_bytes());
        for m in &self.markers {
            out.extend_from_slice(&m.to_le_bytes());
        }
        out
    }
}

/// Parse the **decoded** KAP payload (after `KREncodedBlock` unwrapping). Returns
/// `None` if the magics don't match or the buffer is truncated / has trailing
/// bytes (the length is exact: `0x11 + n*12 + 4 + m*2`).
pub fn parse(clear: &[u8]) -> Option<Kap> {
    if clear.len() < 17 || clear[0..4] != MAGIC || clear[9..13] != MAGIC2 {
        return None;
    }
    let field4 = clear[4..8].try_into().ok()?;
    let kind = clear[8];
    let point_count = u32::from_le_bytes(clear[13..17].try_into().ok()?) as usize;

    let pts_end = 17usize.checked_add(point_count.checked_mul(12)?)?;
    let marker_off = pts_end.checked_add(4)?;
    if marker_off > clear.len() {
        return None;
    }
    let mut points = Vec::with_capacity(point_count);
    let mut o = 17;
    for _ in 0..point_count {
        points.push([
            f32_le(clear, o)?,
            f32_le(clear, o + 4)?,
            f32_le(clear, o + 8)?,
        ]);
        o += 12;
    }

    let marker_count = u32::from_le_bytes(clear[pts_end..pts_end + 4].try_into().ok()?) as usize;
    let end = marker_off.checked_add(marker_count.checked_mul(2)?)?;
    if end != clear.len() {
        return None; // trailing/truncated → not a well-formed KAP payload
    }
    let mut markers = Vec::with_capacity(marker_count);
    let mut o = marker_off;
    for _ in 0..marker_count {
        markers.push(u16::from_le_bytes([clear[o], clear[o + 1]]));
        o += 2;
    }

    Some(Kap {
        field4,
        kind,
        points,
        markers,
    })
}

fn f32_le(d: &[u8], o: usize) -> Option<f32> {
    Some(f32::from_le_bytes(d.get(o..o + 4)?.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(points: &[[f32; 3]], markers: &[u16]) -> Vec<u8> {
        Kap {
            field4: [1, 2, 3, 4],
            kind: 2,
            points: points.to_vec(),
            markers: markers.to_vec(),
        }
        .to_bytes()
    }

    #[test]
    fn roundtrip_path_and_markers() {
        let pts = [
            [643.41, 704.10, 558.72],
            [643.41, 704.12, 558.72],
            [643.50, 705.00, 558.72],
        ];
        let bytes = build(&pts, &[1]);
        let k = parse(&bytes).expect("parse");
        assert_eq!(k.field4, [1, 2, 3, 4]);
        assert_eq!(k.kind, 2);
        assert_eq!(k.points, pts);
        assert_eq!(k.markers, vec![1]);
        assert!(k.markers_in_range());
        assert_eq!(k.to_bytes(), bytes); // byte round-trip
        let (lo, hi) = k.bbox().unwrap();
        assert_eq!(lo, [643.41, 704.10, 558.72]);
        assert_eq!(hi, [643.50, 705.00, 558.72]);
    }

    #[test]
    fn rejects_bad_magic_truncated_and_trailing() {
        let good = build(&[[0.0, 0.0, 0.0]], &[]);
        assert!(parse(&good).is_some());
        let mut bad_magic = good.clone();
        bad_magic[0] = 0;
        assert!(parse(&bad_magic).is_none());
        assert!(parse(&good[..good.len() - 1]).is_none()); // truncated marker_count
        let mut trailing = good.clone();
        trailing.push(0xff);
        assert!(parse(&trailing).is_none()); // exact-length check
        assert!(parse(&[]).is_none());
    }
}
