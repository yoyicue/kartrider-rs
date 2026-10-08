//! Clean-room parser for KartRider skeletal animation clips (`fXX.1s`): a 20-byte
//! header then one PRSTontroller per bone carrying position/rotation/scale
//! keyframes (rotation stored W-first). The same per-bone structure is shared by
//! the character (`MAGIC_CHAR_ANIM`), pet, and boss anim families — they differ
//! only in the container magic and a pet-only track-count field — so we detect
//! by structure (a PRSTontroller right after the header) rather than by magic,
//! matching the oracle's parse_pet_animation. See docs/kartrider-1s-format.md.

const MAGIC_CHAR_ANIM: &[u8] = &[0xAA, 0x47, 0xB7, 0x04, 0xBB, 0x1D];
const OBJ_NEW: u16 = 0x47AA;
const FIELD_NEW: u16 = 0x27AA;
const STAMP_PRS: u32 = 0x224B052A;

pub type Vec3 = [f32; 3];
pub type Quat = [f32; 4]; // xyzw

#[derive(Default)]
pub struct AnimClip {
    pub name: String,
    pub duration_ms: u32,
    pub tracks: Vec<BoneTrack>,
}

#[derive(Default)]
pub struct BoneTrack {
    pub bone_index: i32,
    pub pos: Vec<(i32, Vec3)>,
    pub rot: Vec<(i32, Quat)>,
    pub scale: Vec<(i32, Vec3)>,
}

pub fn is_char_anim(d: &[u8]) -> bool {
    d.len() >= 6 && &d[..6] == MAGIC_CHAR_ANIM
}

/// True if this is a skeletal anim clip of *any* family (character / pet / boss):
/// `AA 47` magic with a PRSTontroller object right after the 20-byte header
/// (char/boss) or after the pet-only track-count field at offset 24.
pub fn is_anim(d: &[u8]) -> bool {
    if d.len() < 26 || d[0] != 0xAA || d[1] != 0x47 {
        return false;
    }
    let prs_at = |o: usize| {
        d.len() >= o + 6
            && d[o] == 0xAA
            && d[o + 1] == 0x47
            && u32::from_le_bytes([d[o + 2], d[o + 3], d[o + 4], d[o + 5]]) == STAMP_PRS
    };
    prs_at(20) || prs_at(24)
}

pub fn parse(data: &[u8], name: &str) -> Option<AnimClip> {
    if !is_anim(data) {
        return None;
    }
    let mut r = Reader::new(data);
    r.skip(6); // magic
    r.skip(6); // padding
    r.u32(); // time_start
    let duration = r.u32();
    let mut clip = AnimClip {
        name: name.to_string(),
        duration_ms: duration,
        tracks: Vec::new(),
    };

    // Pet anims carry a track-count i32 between the header and the first object.
    if r.remaining() >= 2 && r.peek_u16() != OBJ_NEW {
        r.i32();
    }

    // Character/pet clips are a flat sequence of PRSTontroller objects (one per
    // bone) terminated by a different-class object — stop at the first non-PRS
    // stamp. (Boss clips use a deeper KartObject-nested layout with per-bone
    // class stamps and are intentionally left partial; their models use classes
    // no tool implements anyway.)
    while r.remaining() >= 8 && r.peek_u16() == OBJ_NEW {
        r.u16(); // object marker
        let stamp = r.u32();
        let obj_index = r.i16();
        if stamp != STAMP_PRS {
            break;
        }
        let mut track = decode_prs(&mut r);
        track.bone_index = obj_index as i32 - 1;
        clip.tracks.push(track);
    }
    Some(clip)
}

fn decode_prs(r: &mut Reader) -> BoneTrack {
    let mut t = BoneTrack::default();
    // tontroller base: i32 + i32 + read(16) + i32 start + i32 end
    r.i32();
    r.i32();
    r.skip(16);
    r.i32();
    r.i32();
    if r.u8() == 1 {
        read_field(r, |r| read_vec3_keys(r, &mut t.pos));
    }
    if r.u8() == 1 {
        read_field(r, |r| read_rot_keys(r, &mut t.rot));
    }
    if r.u8() == 1 {
        read_field(r, |r| read_vec3_keys(r, &mut t.scale));
    }
    r.skip(24);
    t
}

fn read_field(r: &mut Reader, body: impl FnOnce(&mut Reader)) {
    let token = r.u16();
    assert_eq!(
        token, FIELD_NEW,
        "unexpected field marker (back-refs unsupported)"
    );
    r.i16(); // field index
    body(r);
}

fn read_vec3_keys(r: &mut Reader, keys: &mut Vec<(i32, Vec3)>) {
    let kf_type = r.i32();
    let count = r.i32();
    for _ in 0..count {
        let t = r.i32();
        let v = [r.f32(), r.f32(), r.f32()];
        keys.push((t, v));
        match kf_type {
            0 => r.skip(24), // spline: 6 extra floats
            2 => r.skip(12), // packed: 3 extra i32
            1 | 3 => {}
            _ => panic!("unsupported vec3 kf_type {kf_type}"),
        }
    }
}

fn read_rot_keys(r: &mut Reader, keys: &mut Vec<(i32, Quat)>) {
    let rot_type = r.i32();
    let count = r.i32();
    match rot_type {
        0 | 1 | 3 => {
            // direct quaternion keyframes (stored W-first)
            for _ in 0..count {
                let t = r.i32();
                let (w, x, y, z) = (r.f32(), r.f32(), r.f32(), r.f32());
                keys.push((t, [x, y, z, w]));
            }
        }
        4 => {
            // ThreeAxis euler (boss anims): header + 3 float axes, sampled to
            // quaternions at the union of the axes' keyframe times (q = qZ·qY·qX).
            read_three_axis_rot(r, keys);
        }
        _ => {} // unknown rotation form: leave rotation empty (don't desync hard)
    }
}

fn read_three_axis_rot(r: &mut Reader, keys: &mut Vec<(i32, Quat)>) {
    use crate::keyframe::{eval_float_axis, read_float_axis};
    r.skip(4 + 16); // time(i32) + 4 floats header
    let x = read_float_axis(r.d, &mut r.p);
    let y = read_float_axis(r.d, &mut r.p);
    let z = read_float_axis(r.d, &mut r.p);
    let mut times = std::collections::BTreeSet::new();
    for ax in [&x, &y, &z] {
        for k in &ax.keys {
            times.insert(k.0);
        }
    }
    for t in times {
        let euler = [
            eval_float_axis(&x, t as f64) as f32,
            eval_float_axis(&y, t as f64) as f32,
            eval_float_axis(&z, t as f64) as f32,
        ];
        keys.push((t, crate::kart_anim::euler_to_quat(euler)));
    }
}

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Reader<'a> {
    fn new(d: &'a [u8]) -> Self {
        Reader { d, p: 0 }
    }
    fn remaining(&self) -> usize {
        self.d.len() - self.p
    }
    fn skip(&mut self, n: usize) {
        self.p += n;
    }
    fn u8(&mut self) -> u8 {
        let v = self.d[self.p];
        self.p += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        v
    }
    fn peek_u16(&self) -> u16 {
        u16::from_le_bytes([self.d[self.p], self.d[self.p + 1]])
    }
    fn i16(&mut self) -> i16 {
        let v = i16::from_le_bytes([self.d[self.p], self.d[self.p + 1]]);
        self.p += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
    fn f32(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.d[self.p..self.p + 4].try_into().unwrap());
        self.p += 4;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Little builder for a synthetic MAGIC_CHAR_ANIM clip.
    #[derive(Default)]
    struct B(Vec<u8>);
    impl B {
        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        fn u16(mut self, v: u16) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn i32(mut self, v: i32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn f32(mut self, v: f32) -> Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn zeros(mut self, n: usize) -> Self {
            self.0.extend(std::iter::repeat_n(0u8, n));
            self
        }
    }

    /// One bone with a single linear position key (1,2,3) and rotation key
    /// (w=1 → identity quaternion); no scale.
    fn one_bone_clip() -> Vec<u8> {
        B(MAGIC_CHAR_ANIM.to_vec())
            .zeros(6) // padding
            .i32(0) // time_start
            .i32(1000) // duration_ms
            // --- bone object ---
            .u16(OBJ_NEW)
            .i32(STAMP_PRS as i32)
            .u16(1) // obj_index (i16) → bone_index 0
            .zeros(32) // tontroller base
            // position present
            .u8(1)
            .u16(FIELD_NEW)
            .u16(0) // field index
            .i32(1) // kf_type Linear
            .i32(1) // count
            .i32(0)
            .f32(1.0)
            .f32(2.0)
            .f32(3.0)
            // rotation present
            .u8(1)
            .u16(FIELD_NEW)
            .u16(0)
            .i32(1) // rot_type (quaternion)
            .i32(1) // count
            .i32(0)
            .f32(1.0) // w (stored W-first)
            .f32(0.0)
            .f32(0.0)
            .f32(0.0)
            // scale absent
            .u8(0)
            .zeros(24) // PRS tail
            .0
    }

    #[test]
    fn parses_one_bone() {
        let d = one_bone_clip();
        assert!(is_anim(&d));
        let clip = parse(&d, "f00").expect("parse");
        assert_eq!(clip.name, "f00");
        assert_eq!(clip.duration_ms, 1000);
        assert_eq!(clip.tracks.len(), 1);
        let t = &clip.tracks[0];
        assert_eq!(t.bone_index, 0); // obj_index 1 − 1
        assert_eq!(t.pos, vec![(0, [1.0, 2.0, 3.0])]);
        assert_eq!(t.rot, vec![(0, [0.0, 0.0, 0.0, 1.0])]); // xyzw from W-first wire
        assert!(t.scale.is_empty());
    }

    #[test]
    fn rejects_non_anim() {
        assert!(!is_anim(b"not an anim clip....."));
        assert!(parse(b"\xAA\x47not-prs-after-header____________", "x").is_none());
    }
}
